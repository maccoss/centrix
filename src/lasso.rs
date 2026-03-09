//! Non-negative LASSO solver via coordinate descent with active-set acceleration.
//!
//! Solves: minimize ½||y - Aβ||² + λ||β||₁  subject to β ≥ 0
//!
//! ## Algorithm
//!
//! Coordinate descent: for each variable j, the optimal update is the soft-threshold:
//! ```text
//!   β_j = max(0, ρ_j − λ) / G[j,j]
//! ```
//! where:
//! ```text
//!   ρ_j = (Aᵀy)[j] − Σ_{k≠j} G[j,k] β_k
//!        = (Aᵀy)[j] − (Σ_k G[j,k] β_k) + G[j,j] β_j
//! ```
//! G is the Gram matrix AᵀA. Since G is Toeplitz, G[j,k] = gram_row[|j−k|].
//!
//! ## Active-set acceleration
//!
//! Only variables satisfying the KKT optimality conditions are updated:
//! - Variable j is **active** if β_j > 0 or (Aᵀy)[j] > λ (could become non-zero).
//! - An inactive variable satisfying KKT (ρ_j ≤ λ) never needs updating.
//!
//! Every `CHECK_FREQ` inner iterations, the full variable set is swept to detect
//! newly activatable variables (KKT violations that opened up as neighbors changed).
//!
//! ## Warm start
//!
//! If `warm_start` is provided (e.g., the Pass-1 solution), the active set and β
//! are initialized from it, and the solver typically converges in < 10 iterations.

/// Number of inner iterations between full-sweep KKT checks.
const CHECK_FREQ: usize = 10;

/// Input to the non-negative LASSO solver.
pub struct LassoInput<'a> {
    /// Aᵀy: cross-covariance of design matrix and signal, length n_basis.
    pub aty: &'a [f64],
    /// Toeplitz first row of the Gram matrix G = AᵀA, from `basis::compute_gram_row`.
    /// `gram_row[k] = G[j, j+k]` for all j (Toeplitz structure).
    pub gram_row: &'a [f64],
    /// Regularization parameter λ. Peaks with Aᵀy[j] ≤ λ are suppressed.
    pub lambda: f64,
    /// Optional warm-start β from a previous solve (e.g., Pass 1 solution).
    pub warm_start: Option<&'a [f64]>,
    /// Maximum coordinate descent iterations.
    pub max_iter: usize,
    /// Convergence tolerance: stop when max|β_new − β_old| < tol.
    pub tol: f64,
}

/// Output from the non-negative LASSO solver.
#[derive(Debug, Clone)]
pub struct LassoOutput {
    /// Solution coefficients β ≥ 0, length n_basis.
    pub beta: Vec<f64>,
    /// Number of coordinate descent iterations performed.
    pub n_iter: u32,
    /// Whether the solver converged within `max_iter`.
    pub converged: bool,
}

impl LassoOutput {
    /// Indices of non-zero β entries (the detected peaks).
    pub fn nonzero_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.beta
            .iter()
            .enumerate()
            .filter(|(_, &v)| v > 0.0)
            .map(|(i, _)| i)
    }

    /// Number of non-zero coefficients (detected peaks in this region).
    pub fn n_nonzero(&self) -> usize {
        self.beta.iter().filter(|&&v| v > 0.0).count()
    }
}

/// Solve the non-negative LASSO using coordinate descent with active-set acceleration.
pub fn solve_nonneg_lasso(input: &LassoInput<'_>) -> LassoOutput {
    let n = input.aty.len();
    if n == 0 {
        return LassoOutput {
            beta: Vec::new(),
            n_iter: 0,
            converged: true,
        };
    }

    let g0 = input.gram_row[0]; // G[j,j] — same for all j (Toeplitz diagonal)
    let lambda = input.lambda;

    // Initialize β from warm start or zeros
    let mut beta = match input.warm_start {
        Some(ws) => {
            debug_assert_eq!(ws.len(), n, "warm start length must match aty");
            ws.to_vec()
        }
        None => vec![0.0f64; n],
    };

    // Boolean membership array for O(1) active-set lookup
    let mut is_active = vec![false; n];

    // Build initial active set: indices where β > 0 OR Aᵀy[j] > λ
    let mut active: Vec<usize> = Vec::new();
    for j in 0..n {
        if beta[j] > 0.0 || input.aty[j] > lambda {
            is_active[j] = true;
            active.push(j);
        }
    }

    let mut converged = false;
    let mut n_iter = 0u32;

    'outer: for iter in 0..input.max_iter {
        n_iter = iter as u32 + 1;

        // Full KKT sweep every CHECK_FREQ iterations to find newly activatable variables.
        // Uses O(n) but the O(1) membership check avoids redundant recomputation.
        if iter % CHECK_FREQ == 0 {
            for (j, active_flag) in is_active.iter_mut().enumerate() {
                if !*active_flag {
                    let rho = compute_rho(j, &beta, input.aty, input.gram_row);
                    if rho > lambda {
                        *active_flag = true;
                        active.push(j);
                    }
                }
            }
        }

        // Coordinate descent sweep over active set
        let mut max_delta = 0.0f64;
        let mut new_active = Vec::with_capacity(active.len());

        for &j in &active {
            let rho = compute_rho(j, &beta, input.aty, input.gram_row);
            let beta_new = (rho - lambda).max(0.0) / g0;
            let delta = (beta_new - beta[j]).abs();
            if delta > max_delta {
                max_delta = delta;
            }
            beta[j] = beta_new;

            // Keep variable active if β > 0 or it could become non-zero
            if beta_new > 0.0 || rho > lambda {
                new_active.push(j);
            } else {
                is_active[j] = false;
            }
        }
        active = new_active;

        if max_delta < input.tol {
            converged = true;
            break 'outer;
        }
    }

    // Enforce non-negativity (guards against tiny floating-point negatives)
    for v in &mut beta {
        if *v < 0.0 {
            *v = 0.0;
        }
    }

    LassoOutput {
        beta,
        n_iter,
        converged,
    }
}

/// Compute ρ_j = (Aᵀy)[j] − Σ_{k≠j} G[j,k] β_k
///             = aty[j] − (Σ_k gram_row[|j−k|] β_k) + gram_row[0] β_j
#[inline]
fn compute_rho(j: usize, beta: &[f64], aty: &[f64], gram_row: &[f64]) -> f64 {
    let max_dist = gram_row.len(); // gram_row is truncated where it becomes negligible
    let n = beta.len();

    let lo = j.saturating_sub(max_dist - 1);
    let hi = (j + max_dist).min(n);
    let dot: f64 = beta[lo..hi]
        .iter()
        .enumerate()
        .map(|(rel, &b)| gram_row[(lo + rel).abs_diff(j)] * b)
        .sum();
    aty[j] - dot + gram_row[0] * beta[j]
}

// ── Sub-grid centroid refinement ──────────────────────────────────────────────

/// Refine centroid positions using a parabola fit on ln(β) at neighboring grid points.
///
/// For each non-zero β_j, fits `ln(β) = a*(grid - center)² + ...` using β_{j-1},
/// β_j, β_{j+1} (if both neighbors are also non-zero) to find the sub-grid
/// peak center. Returns `(refined_mz, intensity)` pairs for each detected centroid.
///
/// Falls back to the grid position if neighbors are zero or the parabola opens upward.
pub fn refine_subgrid(beta: &[f64], grid_positions: &[f64]) -> Vec<(f64, f64)> {
    debug_assert_eq!(beta.len(), grid_positions.len());
    let n = beta.len();
    let mut centroids = Vec::new();

    let mut j = 0;
    while j < n {
        if beta[j] <= 0.0 {
            j += 1;
            continue;
        }

        // Try parabola refinement using neighbors
        let refined_mz = if j > 0 && j + 1 < n && beta[j - 1] > 0.0 && beta[j + 1] > 0.0 {
            let ly0 = beta[j - 1].ln();
            let ly1 = beta[j].ln();
            let ly2 = beta[j + 1].ln();
            let h = grid_positions[j] - grid_positions[j - 1];
            if h > 1e-9 {
                let a = (ly0 - 2.0 * ly1 + ly2) / (2.0 * h * h);
                if a < 0.0 {
                    let b = (ly2 - ly0) / (2.0 * h);
                    let offset = -b / (2.0 * a);
                    // Only accept if the offset is within ±0.5 grid spacings
                    if offset.abs() <= 0.5 * h {
                        grid_positions[j] + offset
                    } else {
                        grid_positions[j]
                    }
                } else {
                    grid_positions[j]
                }
            } else {
                grid_positions[j]
            }
        } else {
            grid_positions[j]
        };

        // Intensity is the β coefficient (proportional to peak area)
        centroids.push((refined_mz, beta[j]));
        j += 1;
    }

    centroids
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::{build_local_a, compute_aty, compute_gram_row};
    use approx::assert_abs_diff_eq;

    /// Solve the LASSO from a synthetic Gaussian profile directly.
    /// Returns (beta, grid_positions).
    fn solve_from_profile(
        mz_region: &[f64],
        intensity: &[f64],
        sigma: f64,
        lambda: f64,
        warm_start: Option<&[f64]>,
    ) -> (LassoOutput, Vec<f64>) {
        let grid = mz_region.to_vec(); // grid = data positions for square system
        let a = build_local_a(mz_region, &grid, sigma);
        let aty = compute_aty(&a, intensity);
        let gram_row = compute_gram_row(sigma, mz_region[1] - mz_region[0]);

        let input = LassoInput {
            aty: aty.as_slice().unwrap(),
            gram_row: &gram_row,
            lambda,
            warm_start,
            // High-correlation MS2 problems (G[k=1]/G[k=0] ≈ 0.97) may need many iterations
            max_iter: 10000,
            tol: 1e-6,
        };
        let output = solve_nonneg_lasso(&input);
        (output, grid)
    }

    #[test]
    fn single_gaussian_produces_one_nonzero() {
        let sigma = 0.340f64;
        let spacing = 0.125f64;
        let center = 500.0f64;
        let amplitude = 10000.0f64;

        // Build a profile region around the peak
        let mz: Vec<f64> = (0..32).map(|i| 498.0 + i as f64 * spacing).collect();
        let y: Vec<f64> = mz
            .iter()
            .map(|&m| amplitude * (-(m - center).powi(2) / (2.0 * sigma.powi(2))).exp())
            .collect();

        // λ = 3 × noise ≈ 3 × 0 for a clean signal; set low enough to detect
        let lambda = amplitude * 0.01;
        let (output, _) = solve_from_profile(&mz, &y, sigma, lambda, None);

        assert!(output.converged, "solver should converge");
        assert!(
            output.n_nonzero() >= 1,
            "should find at least one peak, got {} non-zero",
            output.n_nonzero()
        );
        assert!(
            output.n_nonzero() <= 3,
            "should not over-split; got {} peaks",
            output.n_nonzero()
        );

        // The non-zero coefficient should be near the true peak center
        let centroids = refine_subgrid(&output.beta, &mz);
        let peak_mz = centroids
            .iter()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|&(m, _)| m)
            .unwrap();
        assert!(
            (peak_mz - center).abs() < spacing * 1.5,
            "peak at {peak_mz:.4} should be within 1.5 grid spacings of true center {center}"
        );
    }

    #[test]
    fn two_peaks_separated_by_2sigma_are_resolved() {
        let sigma = 0.340f64;
        let spacing = 0.125f64;
        let c1 = 500.0f64;
        let c2 = 500.75f64; // 2.2σ apart (>2σ separation)
        let amplitude = 10000.0f64;

        let mz: Vec<f64> = (0..48).map(|i| 497.0 + i as f64 * spacing).collect();
        let y: Vec<f64> = mz
            .iter()
            .map(|&m| {
                amplitude * (-(m - c1).powi(2) / (2.0 * sigma.powi(2))).exp()
                    + amplitude * (-(m - c2).powi(2) / (2.0 * sigma.powi(2))).exp()
            })
            .collect();

        let lambda = amplitude * 0.005;
        let (output, _) = solve_from_profile(&mz, &y, sigma, lambda, None);

        assert!(output.converged);
        assert!(
            output.n_nonzero() >= 2,
            "should resolve 2 peaks separated by 2.2σ; got {} non-zero at lambda={lambda}",
            output.n_nonzero()
        );
    }

    #[test]
    fn pure_noise_gives_all_zeros_at_high_lambda() {
        let sigma = 0.340f64;
        let spacing = 0.125f64;

        // Signal well below lambda
        let mz: Vec<f64> = (0..24).map(|i| 499.0 + i as f64 * spacing).collect();
        let y: Vec<f64> = vec![1.0f64; mz.len()]; // flat, low-intensity "noise"
        let lambda = 1000.0; // λ >> signal level

        let (output, _) = solve_from_profile(&mz, &y, sigma, lambda, None);

        assert_eq!(
            output.n_nonzero(),
            0,
            "high λ should suppress all coefficients"
        );
    }

    #[test]
    fn beta_is_always_nonnegative() {
        let sigma = 0.340f64;
        let spacing = 0.125f64;
        let mz: Vec<f64> = (0..32).map(|i| 498.0 + i as f64 * spacing).collect();
        // Adversarial input: oscillating intensities
        let y: Vec<f64> = mz
            .iter()
            .enumerate()
            .map(|(i, _)| {
                if i % 2 == 0 {
                    1000.0
                } else {
                    -100.0_f64.max(0.0)
                }
            })
            .collect();

        let (output, _) = solve_from_profile(&mz, &y, sigma, 1.0, None);

        for (j, &b) in output.beta.iter().enumerate() {
            assert!(b >= 0.0, "β[{j}] = {b} < 0 — non-negativity violated");
        }
    }

    #[test]
    fn warm_start_converges_faster() {
        let sigma = 0.340f64;
        let spacing = 0.125f64;
        let center = 500.375f64;
        let amplitude = 5000.0f64;

        let mz: Vec<f64> = (0..32).map(|i| 498.0 + i as f64 * spacing).collect();
        let y: Vec<f64> = mz
            .iter()
            .map(|&m| amplitude * (-(m - center).powi(2) / (2.0 * sigma.powi(2))).exp())
            .collect();

        let lambda = amplitude * 0.01;

        // Cold start
        let (cold, _) = solve_from_profile(&mz, &y, sigma, lambda, None);

        // Warm start from cold solution with slightly different λ
        let lambda2 = lambda * 0.9;
        let (warm, _) = solve_from_profile(&mz, &y, sigma, lambda2, Some(&cold.beta));

        // Warm start should converge in fewer iterations
        assert!(
            warm.n_iter <= cold.n_iter,
            "warm start should not need more iterations than cold: warm={} cold={}",
            warm.n_iter,
            cold.n_iter
        );
    }

    #[test]
    fn refine_subgrid_improves_centering() {
        // Three-point parabola with peak between grid points
        let grid = vec![499.875, 500.0, 500.125, 500.25, 500.375];
        // Peak at 500.05 (between 500.0 and 500.125 grid points)
        let sigma = 0.340f64;
        let true_center = 500.05f64;
        let beta: Vec<f64> = grid
            .iter()
            .map(|&g| (-(g - true_center).powi(2) / (2.0 * sigma.powi(2))).exp() * 1000.0)
            .collect();

        let centroids = refine_subgrid(&beta, &grid);
        assert!(!centroids.is_empty());

        let best_mz = centroids
            .iter()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|&(m, _)| m)
            .unwrap();

        // Refined position should be closer to true center than the nearest grid point
        let nearest_grid = 500.0f64; // 500.0 is closest grid point to 500.05
        assert!(
            (best_mz - true_center).abs() < (nearest_grid - true_center).abs(),
            "refined {best_mz:.4} should be closer to true center {true_center} \
             than grid point {nearest_grid}"
        );
    }

    #[test]
    fn compute_rho_at_zero_beta_equals_aty() {
        let aty = vec![100.0, 200.0, 150.0, 80.0];
        let beta = vec![0.0f64; 4];
        let gram_row = vec![4.821, 4.662, 4.192, 3.537]; // realistic MS2 values

        // When β=0, ρ_j = aty[j] − 0 + G[j,j]×0 = aty[j]
        for j in 0..4 {
            let rho = compute_rho(j, &beta, &aty, &gram_row);
            assert_abs_diff_eq!(rho, aty[j], epsilon = 1e-10);
        }
    }
}
