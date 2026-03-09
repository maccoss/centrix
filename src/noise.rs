//! Noise estimation for profile mass spectra.
//!
//! ## Two-pass noise model
//!
//! **Pass 0 (rough):** Estimate baseline and noise sigma from the raw intensity
//! distribution using order statistics. Baseline ≈ 10th percentile; noise sigma
//! ≈ (75th − 25th percentile) / 1.349 (robust σ via IQR).
//!
//! **Pass 1→2 refinement:** After LASSO fitting, collect residuals (y − Aβ)
//! from fitted regions and gap intensities between regions. Build a smooth
//! per-m/z noise model using overlapping windows. This refined model is used
//! in Pass 2 to set per-region λ adaptively.

use crate::signal::SignalRegion;

// ── Noise profile ─────────────────────────────────────────────────────────────

/// Noise and baseline model for a single spectrum.
#[derive(Debug, Clone)]
pub struct NoiseProfile {
    /// Rough baseline estimate (intensity floor)
    pub baseline: f64,
    /// Rough noise sigma (standard deviation of noise distribution)
    pub noise_sigma: f64,
    /// Whether the refined model has been computed from LASSO residuals
    pub is_refined: bool,
    /// Refined noise model: (mz_center, noise_sigma) pairs, sorted by mz
    refined_pts: Vec<(f64, f64)>,
}

impl NoiseProfile {
    pub fn rough(baseline: f64, noise_sigma: f64) -> Self {
        Self {
            baseline,
            noise_sigma,
            is_refined: false,
            refined_pts: Vec::new(),
        }
    }

    /// Noise sigma at a given m/z (linear interpolation between refined points,
    /// falls back to rough estimate if not refined or outside range).
    pub fn noise_at(&self, mz: f64) -> f64 {
        if !self.is_refined || self.refined_pts.is_empty() {
            return self.noise_sigma;
        }
        let pts = &self.refined_pts;
        if mz <= pts.first().unwrap().0 {
            return pts.first().unwrap().1;
        }
        if mz >= pts.last().unwrap().0 {
            return pts.last().unwrap().1;
        }
        // Binary search for the interval
        let idx = pts.partition_point(|&(m, _)| m < mz);
        let (m0, n0) = pts[idx - 1];
        let (m1, n1) = pts[idx];
        let t = (mz - m0) / (m1 - m0);
        n0 + t * (n1 - n0)
    }

    /// Baseline at a given m/z (currently constant; could be refined similarly).
    pub fn baseline_at(&self, _mz: f64) -> f64 {
        self.baseline
    }

    /// λ = lambda_factor × noise_sigma(mz) for the LASSO.
    pub fn lambda_at(&self, mz: f64, lambda_factor: f64) -> f64 {
        lambda_factor * self.noise_at(mz)
    }
}

// ── Rough noise estimate ──────────────────────────────────────────────────────

/// Estimate baseline and noise sigma from the raw intensity distribution.
///
/// - **Baseline** ≈ 10th percentile (the floor of the distribution)
/// - **Noise sigma** ≈ (75th − 25th percentile) / 1.349  (robust IQR-based estimate)
///
/// This is fast (one sort) and avoids being pulled up by signal peaks.
pub fn rough_noise_estimate(intensity: &[f32]) -> NoiseProfile {
    if intensity.is_empty() {
        return NoiseProfile::rough(0.0, 1.0);
    }

    let mut sorted: Vec<f64> = intensity.iter().map(|&v| v as f64).collect();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len();

    let p10 = sorted[n / 10].max(0.0);
    let p25 = sorted[n / 4];
    let p75 = sorted[n * 3 / 4];

    // IQR-based σ estimate: IQR / 1.349 ≈ σ for a normal distribution
    let noise_sigma = ((p75 - p25) / 1.349).max(p10 * 0.1).max(1.0);

    NoiseProfile::rough(p10, noise_sigma)
}

// ── Residual sample ───────────────────────────────────────────────────────────

/// Residual data from a single LASSO-fitted region, used for noise refinement.
pub struct ResidualSample {
    /// Center m/z of the region
    pub mz_center: f64,
    /// Residuals (y − Aβ) for each data point in the region
    pub residuals: Vec<f64>,
}

impl ResidualSample {
    /// RMS of the residuals (robust noise estimate for this region).
    pub fn rms(&self) -> f64 {
        if self.residuals.is_empty() {
            return 0.0;
        }
        let ss: f64 = self.residuals.iter().map(|r| r * r).sum();
        (ss / self.residuals.len() as f64).sqrt()
    }
}

// ── Noise refinement ──────────────────────────────────────────────────────────

/// Refine the noise model from LASSO residuals and gap intensities.
///
/// Builds overlapping m/z windows of width `window_da` stepped by `step_da`.
/// In each window, collects:
/// - Residuals from LASSO-fitted regions (y − Aβ)
/// - Raw intensities of gap points (between signal regions, below threshold)
///
/// The noise sigma for each window = RMS of all collected values.
/// Returns a refined `NoiseProfile` with linear interpolation.
pub fn refine_from_residuals(
    mz: &[f64],
    intensity: &[f32],
    regions: &[SignalRegion],
    residuals: &[ResidualSample],
    rough: &NoiseProfile,
    window_mz: f64,
    step_mz: f64,
) -> NoiseProfile {
    if mz.is_empty() || residuals.is_empty() {
        return rough.clone();
    }

    let mz_min = mz.first().copied().unwrap_or(0.0);
    let mz_max = mz.last().copied().unwrap_or(0.0);

    // Build gap mask: true for points NOT in any signal region
    let n = mz.len();
    let mut in_region = vec![false; n];
    for r in regions {
        for flag in in_region[r.start_idx..=r.end_idx.min(n - 1)].iter_mut() {
            *flag = true;
        }
    }

    // Sweep windows
    let mut noise_pts: Vec<(f64, f64)> = Vec::new();
    let mut center = mz_min + window_mz / 2.0;
    while center <= mz_max - window_mz / 2.0 {
        let lo = center - window_mz / 2.0;
        let hi = center + window_mz / 2.0;

        // Collect gap intensities in this window
        let mut samples: Vec<f64> = mz
            .iter()
            .zip(intensity.iter())
            .zip(in_region.iter())
            .filter(|((&m, _), &in_r)| !in_r && m >= lo && m <= hi)
            .map(|((_, &v), _)| v as f64)
            .collect();

        // Collect LASSO residuals whose region center falls in this window
        for rs in residuals {
            if rs.mz_center >= lo && rs.mz_center <= hi {
                samples.extend_from_slice(&rs.residuals);
            }
        }

        if samples.len() >= 5 {
            // RMS of all samples (includes both gap noise and fitting residuals)
            let rms: f64 = {
                let ss: f64 = samples.iter().map(|v| v * v).sum();
                (ss / samples.len() as f64).sqrt()
            };
            if rms > 0.0 {
                noise_pts.push((center, rms.max(rough.noise_sigma * 0.1)));
            }
        }

        center += step_mz;
    }

    if noise_pts.is_empty() {
        return rough.clone();
    }

    // Apply light smoothing: replace each point with median of itself and neighbors
    let smoothed = smooth_noise_pts(&noise_pts);

    NoiseProfile {
        baseline: rough.baseline,
        noise_sigma: rough.noise_sigma,
        is_refined: true,
        refined_pts: smoothed,
    }
}

/// Light 3-point median smoothing of (mz, noise) pairs.
fn smooth_noise_pts(pts: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let n = pts.len();
    if n < 3 {
        return pts.to_vec();
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let lo = i.saturating_sub(1);
        let hi = (i + 1).min(n - 1);
        let mut vals: Vec<f64> = pts[lo..=hi].iter().map(|&(_, v)| v).collect();
        vals.sort_by(|a, b| a.total_cmp(b));
        let median = vals[vals.len() / 2];
        out.push((pts[i].0, median));
    }
    out
}

/// Check if the refined λ differs enough from the rough λ to warrant a Pass-2 re-fit.
/// Returns true if |λ_refined − λ_rough| / λ_rough > threshold.
pub fn needs_refit(rough_lambda: f64, refined_lambda: f64, threshold: f64) -> bool {
    if rough_lambda <= 0.0 {
        return false;
    }
    (refined_lambda - rough_lambda).abs() / rough_lambda > threshold
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn rough_noise_iqr_recovery() {
        // Build a dataset with known IQR: n equally-spaced values from 0 to 1000.
        // p25=250, p75=750 → IQR=500 → expected sigma = 500/1.349 ≈ 370.6
        let n = 10000usize;
        let vals: Vec<f32> = (0..n)
            .map(|i| (i as f64 * 1000.0 / n as f64) as f32)
            .collect();
        let profile = rough_noise_estimate(&vals);
        // IQR-based sigma for uniform[0,1000]: IQR=500, expected ~370.
        // Tolerance: ±20%
        assert!(
            profile.noise_sigma > 280.0 && profile.noise_sigma < 450.0,
            "noise_sigma={:.1} expected ~370 (IQR/1.349 for uniform[0,1000])",
            profile.noise_sigma
        );
    }

    #[test]
    fn rough_noise_all_zeros() {
        let intensity = vec![0.0f32; 100];
        let p = rough_noise_estimate(&intensity);
        assert!(p.noise_sigma > 0.0, "noise_sigma must be positive");
    }

    #[test]
    fn noise_at_returns_rough_when_not_refined() {
        let p = NoiseProfile::rough(10.0, 50.0);
        assert_abs_diff_eq!(p.noise_at(500.0), 50.0, epsilon = 1e-9);
        assert_abs_diff_eq!(p.noise_at(1000.0), 50.0, epsilon = 1e-9);
    }

    #[test]
    fn noise_at_interpolates_refined() {
        let p = NoiseProfile {
            baseline: 5.0,
            noise_sigma: 30.0,
            is_refined: true,
            refined_pts: vec![(300.0, 20.0), (500.0, 40.0), (700.0, 30.0)],
        };
        // At endpoints
        assert_abs_diff_eq!(p.noise_at(300.0), 20.0, epsilon = 1e-9);
        assert_abs_diff_eq!(p.noise_at(700.0), 30.0, epsilon = 1e-9);
        // Interpolated midpoint 300→500: expect 30.0
        assert_abs_diff_eq!(p.noise_at(400.0), 30.0, epsilon = 1e-9);
    }

    #[test]
    fn needs_refit_detects_large_change() {
        assert!(
            needs_refit(100.0, 125.0, 0.20),
            "25% change should trigger refit"
        );
        assert!(
            !needs_refit(100.0, 115.0, 0.20),
            "15% change should not trigger refit"
        );
        assert!(
            !needs_refit(0.0, 100.0, 0.20),
            "zero rough lambda → no refit"
        );
    }
}
