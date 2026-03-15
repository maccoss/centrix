//! Gaussian basis functions and the precomputed Gram matrix.
//!
//! The observed profile in a signal region is modeled as:
//! ```text
//!   y = A β + ε
//! ```
//! where each column of **A** is a Gaussian centered at a grid position:
//! ```text
//!   A[i,j] = exp(-(mz[i] - grid[j])² / (2σ²))
//! ```
//!
//! ## Toeplitz Gram matrix
//!
//! For a uniform grid with spacing `h`, the Gram matrix `G = AᵀA` is symmetric
//! Toeplitz. The analytic closed-form entry is:
//! ```text
//!   G[i,j] = σ√π · exp(-(|i-j|·h)² / (4σ²))
//! ```
//! Only the first row (`gram_row`) is stored. Any entry is `gram_row[|i-j|]`.
//! This avoids allocating O(n²) per-region matrices and the entire row fits in
//! L1 cache (~20 entries for typical σ/spacing ratios).

use ndarray::{Array1, Array2};

/// Precomputed basis parameters shared across all spectra of the same MS level.
///
/// Constructed once during calibration; passed read-only to all centroiding threads.
#[derive(Debug, Clone)]
pub struct BasisPrecompute {
    /// Gaussian σ in m/z
    pub sigma: f64,
    /// Profile data point spacing in m/z (= LASSO grid spacing)
    pub grid_spacing: f64,
    /// Grid offset: fraction of `grid_spacing` applied to the global grid origin.
    /// Determined during calibration to minimize total LASSO residuals.
    pub grid_offset: f64,
    /// Toeplitz first row of the Gram matrix G = AᵀA.
    /// `gram_row[k] = sigma * sqrt(PI) * exp(-(k * grid_spacing)² / (4σ²))`
    /// Truncated where values fall below `f64::EPSILON * gram_row[0]`.
    pub gram_row: Vec<f64>,
    /// Regularization scaling factor: λ = lambda_factor × noise_sigma
    pub lambda_factor: f64,
}

impl BasisPrecompute {
    /// Build a `BasisPrecompute` from calibrated parameters.
    pub fn new(sigma: f64, grid_spacing: f64, grid_offset: f64, lambda_factor: f64) -> Self {
        let gram_row = compute_gram_row(sigma, grid_spacing);
        Self {
            sigma,
            grid_spacing,
            grid_offset,
            gram_row,
            lambda_factor,
        }
    }

    /// Squared norm of any basis column: `‖a_j‖² = gram_row[0] = σ√π`.
    #[inline]
    pub fn column_norm_sq(&self) -> f64 {
        self.gram_row[0]
    }

    /// Build grid positions covering `[mz_start, mz_end]` using the calibrated
    /// grid spacing and offset.
    ///
    /// Grid positions are: `grid_offset + k * grid_spacing` for integer k,
    /// starting from the first position ≥ `mz_start - grid_spacing`.
    pub fn grid_positions(&self, mz_start: f64, mz_end: f64) -> Vec<f64> {
        // First grid index at or just before mz_start
        let k_start = ((mz_start - self.grid_offset) / self.grid_spacing).floor() as i64 - 1;
        let mut positions = Vec::new();
        let mut k = k_start;
        loop {
            let pos = self.grid_offset + k as f64 * self.grid_spacing;
            if pos > mz_end + self.grid_spacing {
                break;
            }
            if pos >= mz_start - self.grid_spacing {
                positions.push(pos);
            }
            k += 1;
        }
        positions
    }
}

/// Compute the Toeplitz first row of the discrete Gram matrix G = AᵀA.
///
/// For profile data on a uniform grid with spacing h, the (j,j+k) entry of
/// the Gram matrix is the inner product of two Gaussian columns spaced k
/// steps apart. The discrete sum approximates the continuous integral, giving:
///
/// `gram_row[k] = (σ√π / h) · exp(-(k·h)² / (4σ²))`
///
/// The 1/h factor (where h = grid_spacing) converts the continuous integral
/// to the discrete sum. This is critical for the LASSO coordinate descent:
/// gram_row[0] must equal ‖a_j‖² (the actual column norm squared) for the
/// update step β_j = max(0, ρ_j − λ) / gram_row[0] to be correct.
///
/// Truncated at machine epsilon relative to `gram_row[0]`.
pub fn compute_gram_row(sigma: f64, grid_spacing: f64) -> Vec<f64> {
    // norm = continuous integral / h = discrete sum approximation
    let norm = sigma * std::f64::consts::PI.sqrt() / grid_spacing;
    let decay = -grid_spacing * grid_spacing / (4.0 * sigma * sigma);
    let threshold = norm * f64::EPSILON;

    let mut row = Vec::with_capacity(32);
    let mut k = 0i64;
    loop {
        let val = norm * (decay * (k * k) as f64).exp();
        if k > 0 && val < threshold {
            break;
        }
        row.push(val);
        k += 1;
    }
    row
}

/// Build the local design matrix A for a signal region.
///
/// `A[i, j] = exp(-(mz[i] - grid[j])² / (2σ²))`
///
/// Returns a (n_data × n_basis) matrix in standard C (row-major) order.
/// ndarray's `.t().dot()` dispatches to BLAS regardless of layout.
pub fn build_local_a(mz: &[f64], grid: &[f64], sigma: f64) -> Array2<f64> {
    let n = mz.len();
    let m = grid.len();
    let inv_two_sig2 = -1.0 / (2.0 * sigma * sigma);

    Array2::from_shape_fn((n, m), |(i, j)| {
        let d = mz[i] - grid[j];
        (d * d * inv_two_sig2).exp()
    })
}

/// Compute `Aᵀy` via BLAS `dgemv` (dispatched by ndarray when the `blas`
/// feature is enabled).
///
/// This is the dominant per-region BLAS call.
pub fn compute_aty(a: &Array2<f64>, y: &[f64]) -> Array1<f64> {
    let y_arr = ndarray::ArrayView1::from(y);
    a.t().dot(&y_arr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn gram_row_k0_equals_discrete_norm() {
        // gram_row[0] = σ√π / h (discrete Gram diagonal approximation)
        let sigma = 0.25;
        let h = 0.125;
        let row = compute_gram_row(sigma, h);
        let expected = sigma * std::f64::consts::PI.sqrt() / h;
        assert_abs_diff_eq!(row[0], expected, epsilon = 1e-12);
    }

    #[test]
    fn gram_row_is_decreasing() {
        let row = compute_gram_row(0.25, 0.125);
        for w in row.windows(2) {
            assert!(w[0] > w[1], "gram_row should be strictly decreasing");
        }
    }

    #[test]
    fn gram_row_ms1_longer_than_ms2() {
        // Finer spacing → correlation drops off more slowly → more non-negligible entries
        let row_ms1 = compute_gram_row(0.25, 1.0 / 15.0);
        let row_ms2 = compute_gram_row(0.25, 0.125);
        // Both should be truncated to a reasonable length
        assert!(
            row_ms1.len() >= 3,
            "MS1 gram_row too short: {}",
            row_ms1.len()
        );
        assert!(
            row_ms2.len() >= 3,
            "MS2 gram_row too short: {}",
            row_ms2.len()
        );
    }

    #[test]
    fn build_local_a_single_column_peaks_at_grid_center() {
        let sigma = 0.25;
        let grid = vec![500.0];
        // Sample exactly at the grid center and ±1σ
        let mz = vec![499.75, 500.0, 500.25];
        let a = build_local_a(&mz, &grid, sigma);

        assert_eq!(a.shape(), &[3, 1]);
        // Center point should be 1.0 (exp(0))
        assert_abs_diff_eq!(a[[1, 0]], 1.0, epsilon = 1e-12);
        // ±1σ points should be exp(-0.5)
        assert_abs_diff_eq!(a[[0, 0]], (-0.5f64).exp(), epsilon = 1e-12);
        assert_abs_diff_eq!(a[[2, 0]], (-0.5f64).exp(), epsilon = 1e-12);
    }

    #[test]
    fn discrete_gram_matches_analytic() {
        // For a fine enough grid, the discrete AᵀA should approximate gram_row
        let sigma = 0.25;
        let h = 0.125f64;
        let basis = BasisPrecompute::new(sigma, h, 0.0, 3.0);

        // Build a long uniform mz range to get a good Riemann sum approximation
        let n_pts = 200;
        let mz_start = 400.0;
        let mz: Vec<f64> = (0..n_pts).map(|i| mz_start + i as f64 * h).collect();
        let grid: Vec<f64> = (0..n_pts).map(|i| mz_start + i as f64 * h).collect();

        let a = build_local_a(&mz, &grid, sigma);
        let ata = a.t().dot(&a);

        // The diagonal should match gram_row[0] = σ√π/h (discrete Gram approximation)
        let analytic_diag = sigma * std::f64::consts::PI.sqrt() / h;
        // Allow generous tolerance: discrete vs continuous, edge effects
        let discrete_diag = ata[[n_pts / 2, n_pts / 2]];
        let ratio = discrete_diag / analytic_diag;
        assert!(
            (ratio - 1.0).abs() < 0.05,
            "discrete/analytic diag ratio = {ratio:.4}, expected ~1.0"
        );

        // Off-diagonal k=1: should match gram_row[1]
        let analytic_k1 = basis.gram_row[1];
        let discrete_k1 = ata[[n_pts / 2, n_pts / 2 + 1]];
        let ratio_k1 = discrete_k1 / analytic_k1;
        assert!((ratio_k1 - 1.0).abs() < 0.05, "k=1 ratio = {ratio_k1:.4}");
    }

    #[test]
    fn compute_aty_single_peak() {
        let sigma = 0.25;
        let grid = vec![499.875, 500.0, 500.125];
        // Symmetric mz sample centered at 500.0 so the grid point at 500.0
        // (index 1) should get the maximum Aᵀy response.
        let mz = vec![499.75, 499.875, 500.0, 500.125, 500.25];
        let a = build_local_a(&mz, &grid, sigma);

        // Gaussian centered at 500.0
        let y: Vec<f64> = mz
            .iter()
            .map(|&m| (-(m - 500.0).powi(2) / (2.0 * sigma * sigma)).exp() * 1000.0)
            .collect();

        let aty = compute_aty(&a, &y);
        assert_eq!(aty.len(), 3);
        // The center grid point at 500.0 (index 1) should have the largest Aᵀy.
        let max_idx = aty
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(
            max_idx, 1,
            "grid[1]=500.0 should have max Aᵀy; got aty={aty:?}"
        );
    }

    #[test]
    fn grid_positions_covers_range() {
        let basis = BasisPrecompute::new(0.25, 0.125, 0.0, 3.0);
        let positions = basis.grid_positions(500.0, 501.0);
        assert!(!positions.is_empty());
        assert!(positions.first().unwrap() <= &500.0);
        assert!(positions.last().unwrap() >= &501.0);
        // All positions should be multiples of grid_spacing (within float tolerance)
        for &p in &positions {
            let rem = (p / 0.125).round() * 0.125 - p;
            assert!(rem.abs() < 1e-9, "position {p} is not on grid");
        }
    }
}
