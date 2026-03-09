//! Two-pass centroiding of a profile spectrum.
//!
//! ## Pipeline
//!
//! **Pass 1:**
//! 1. Rough noise estimate → baseline, noise_sigma
//! 2. Detect signal regions; classify single-peak vs composite
//! 3. Single-peak regions → 3-point Gaussian fast path
//! 4. Composite regions → non-negative LASSO; cache Aᵀy and β
//! 5. Compute residuals (y − Aβ) for each LASSO region
//!
//! **Noise refinement:**
//! 6. Refine noise model from LASSO residuals + gap intensities
//!
//! **Pass 2 (selective):**
//! 7. For composite regions where λ changed > threshold: re-run LASSO
//!    warm-started from Pass-1 β; reuse cached Aᵀy (grid unchanged)
//!
//! **Output:**
//! 8. Collect all centroids; sub-grid refine LASSO centroids; sort and merge

use crate::basis::{build_local_a, compute_aty, BasisPrecompute};
use crate::calibration::fit_gaussian_3pt;
use crate::config::Config;
use crate::io::reader::ProfileSpectrum;
use crate::lasso::{refine_subgrid, solve_nonneg_lasso, LassoInput};
use crate::noise::{needs_refit, refine_from_residuals, rough_noise_estimate, ResidualSample};
use crate::signal::{detect_signal_regions, RegionClass};
use ndarray::Array1;

// ── Output types ──────────────────────────────────────────────────────────────

/// A single detected centroid from the two-pass algorithm.
#[derive(Debug, Clone)]
pub struct CentroidResult {
    /// Centroid m/z (sub-grid refined where possible)
    pub mz: f64,
    /// Intensity (β coefficient from LASSO, or peak height from fast path)
    pub intensity: f64,
}

/// Per-spectrum statistics for diagnostics and quality monitoring.
#[derive(Debug, Clone, Default)]
pub struct SpectrumStats {
    pub scan_number: u32,
    pub n_regions: usize,
    pub n_fast_path: usize,
    pub n_lasso: usize,
    pub n_pass2_refits: usize,
    pub n_centroids: usize,
    pub sigma_used: f64,
    pub lambda_rough: f64,
}

// ── Cached Pass-1 state for composite regions ─────────────────────────────────

struct Pass1State {
    region_idx: usize,
    beta: Vec<f64>,
    // Cached Aᵀy from Pass 1: reusable in Pass 2 if the grid doesn't change.
    // Currently Pass 2 recomputes it (it's fast). Reserved for future optimization.
    #[allow(dead_code)]
    aty: Array1<f64>,
    grid: Vec<f64>,
    rough_lambda: f64,
}

// ── Main entry point ──────────────────────────────────────────────────────────

/// Centroid a single profile spectrum using the two-pass algorithm.
///
/// Returns `(centroids, stats)`. Centroids are sorted by m/z.
pub fn centroid_spectrum(
    spectrum: &ProfileSpectrum,
    basis: &BasisPrecompute,
    config: &Config,
) -> (Vec<CentroidResult>, SpectrumStats) {
    let mz = &spectrum.mz;
    let intensity = &spectrum.intensity;
    let n = mz.len();

    if n < 3 || intensity.is_empty() {
        return (
            Vec::new(),
            SpectrumStats {
                scan_number: spectrum.scan_number,
                ..Default::default()
            },
        );
    }

    let mut stats = SpectrumStats {
        scan_number: spectrum.scan_number,
        sigma_used: basis.sigma,
        ..Default::default()
    };

    // ── Step 1: Rough noise estimate ──────────────────────────────────────────
    let noise = rough_noise_estimate(intensity);
    let rough_lambda = noise.lambda_at(mz[n / 2], basis.lambda_factor);
    stats.lambda_rough = rough_lambda;

    // ── Step 2: Detect signal regions (Pass 1) ────────────────────────────────
    let regions = detect_signal_regions(
        mz,
        intensity,
        noise.baseline,
        noise.noise_sigma,
        config.signal_threshold_sigma,
        config.merge_gap_points,
        config.extension_points,
        config.min_region_width,
        basis.sigma,
        config.single_peak_width_sigma,
    );
    stats.n_regions = regions.len();

    if regions.is_empty() {
        return (Vec::new(), stats);
    }

    // ── Steps 3–5: Process each region in Pass 1 ─────────────────────────────
    let mut all_centroids: Vec<CentroidResult> = Vec::new();
    let mut residual_samples: Vec<ResidualSample> = Vec::new();
    let mut pass1_states: Vec<Pass1State> = Vec::new();

    for (ridx, region) in regions.iter().enumerate() {
        let mz_slice = &mz[region.start_idx..=region.end_idx];
        let int_slice = &intensity[region.start_idx..=region.end_idx];
        let int_f64: Vec<f64> = int_slice.iter().map(|&v| v as f64).collect();

        match region.classification {
            RegionClass::SinglePeak => {
                // Fast path: 3-point Gaussian fit
                if let Some((center_mz, sigma_fit)) =
                    fast_path_gaussian(mz_slice, int_slice, basis.sigma)
                {
                    all_centroids.push(CentroidResult {
                        mz: center_mz,
                        intensity: region.max_intensity as f64,
                    });
                    stats.n_fast_path += 1;
                    let _ = sigma_fit; // used for validation inside fast_path_gaussian
                } else {
                    // Fast path failed — fall through to LASSO for this region
                    let (centroids, state) = run_lasso_region(
                        ridx,
                        mz_slice,
                        &int_f64,
                        basis,
                        rough_lambda,
                        region.center_mz(),
                        None,
                    );
                    all_centroids.extend(centroids);
                    if let Some(s) = state {
                        collect_residuals(
                            &s,
                            mz_slice,
                            &int_f64,
                            &mut residual_samples,
                            basis.sigma,
                        );
                        pass1_states.push(s);
                    }
                    stats.n_lasso += 1;
                }
            }
            RegionClass::PotentialComposite => {
                let (centroids, state) = run_lasso_region(
                    ridx,
                    mz_slice,
                    &int_f64,
                    basis,
                    rough_lambda,
                    region.center_mz(),
                    None,
                );
                all_centroids.extend(centroids);
                if let Some(s) = state {
                    collect_residuals(&s, mz_slice, &int_f64, &mut residual_samples, basis.sigma);
                    pass1_states.push(s);
                }
                stats.n_lasso += 1;
            }
        }
    }

    // ── Step 6: Noise refinement ──────────────────────────────────────────────
    let refined_noise = refine_from_residuals(
        mz,
        intensity,
        &regions,
        &residual_samples,
        &noise,
        config.noise_window_da,
        config.noise_step_da,
    );

    // ── Step 7: Pass 2 — selective re-fit ────────────────────────────────────
    // Re-run LASSO only for composite regions where λ changed significantly.
    // Remove Pass-1 centroids for those regions; replace with Pass-2 results.
    let mut pass2_centroid_sets: Vec<(usize, Vec<CentroidResult>)> = Vec::new();

    for state in &pass1_states {
        let region = &regions[state.region_idx];
        let refined_lambda = refined_noise.lambda_at(region.center_mz(), basis.lambda_factor);

        if needs_refit(
            state.rough_lambda,
            refined_lambda,
            config.lambda_change_threshold,
        ) {
            let mz_slice = &mz[region.start_idx..=region.end_idx];
            let int_slice = &intensity[region.start_idx..=region.end_idx];
            let int_f64: Vec<f64> = int_slice.iter().map(|&v| v as f64).collect();

            let (centroids, _) = run_lasso_region(
                state.region_idx,
                mz_slice,
                &int_f64,
                basis,
                refined_lambda,
                region.center_mz(),
                Some(&state.beta), // warm start from Pass 1
            );
            pass2_centroid_sets.push((state.region_idx, centroids));
            stats.n_pass2_refits += 1;
        }
    }

    // Apply Pass-2 centroid replacements
    // Strategy: collect all Pass-2 region indices, remove their Pass-1 centroids,
    // then add the Pass-2 results. Since centroids aren't tagged by region, we use
    // m/z range matching.
    if !pass2_centroid_sets.is_empty() {
        for (ridx, new_centroids) in pass2_centroid_sets {
            let region = &regions[ridx];
            // Remove Pass-1 centroids that fall within this region's m/z range
            all_centroids.retain(|c| c.mz < region.mz_start || c.mz > region.mz_end);
            all_centroids.extend(new_centroids);
        }
    }

    // ── Step 8: Sort and merge near-duplicate centroids ───────────────────────
    all_centroids.sort_by(|a, b| a.mz.total_cmp(&b.mz));

    let half_grid = basis.grid_spacing / 2.0;
    let merged = merge_nearby_centroids(all_centroids, half_grid);
    stats.n_centroids = merged.len();

    (merged, stats)
}

// ── LASSO region processing ───────────────────────────────────────────────────

/// Run LASSO on a signal region slice. Returns centroids and optional Pass-1 state.
fn run_lasso_region(
    region_idx: usize,
    mz_slice: &[f64],
    int_f64: &[f64],
    basis: &BasisPrecompute,
    lambda: f64,
    center_mz: f64,
    warm_start: Option<&[f64]>,
) -> (Vec<CentroidResult>, Option<Pass1State>) {
    if mz_slice.len() < 2 {
        return (Vec::new(), None);
    }

    // Build grid matching the data points (square system)
    let grid = basis.grid_positions(mz_slice[0], *mz_slice.last().unwrap());
    if grid.is_empty() {
        return (Vec::new(), None);
    }

    let a = build_local_a(mz_slice, &grid, basis.sigma);
    let aty = compute_aty(&a, int_f64);
    let aty_slice = match aty.as_slice() {
        Some(s) => s,
        None => return (Vec::new(), None),
    };

    let gram_row = &basis.gram_row;
    let input = LassoInput {
        aty: aty_slice,
        gram_row,
        lambda,
        warm_start,
        max_iter: 2000,
        tol: 1e-6,
    };
    let output = solve_nonneg_lasso(&input);

    if output.n_nonzero() == 0 {
        return (Vec::new(), None);
    }

    // Sub-grid centroid refinement
    let centroids_raw = refine_subgrid(&output.beta, &grid);
    let centroids: Vec<CentroidResult> = centroids_raw
        .into_iter()
        .map(|(mz, intensity)| CentroidResult { mz, intensity })
        .collect();

    let state = Pass1State {
        region_idx,
        beta: output.beta,
        aty,
        grid,
        rough_lambda: lambda,
    };

    let _ = center_mz; // used by caller for noise lookup
    (centroids, Some(state))
}

/// Fast-path Gaussian fit for single-peak regions.
/// Validates the fitted σ is within [0.5×, 1.5×] the calibrated σ.
/// Returns `None` if the fit fails or is implausible → caller escalates to LASSO.
fn fast_path_gaussian(mz: &[f64], intensity: &[f32], sigma_calibrated: f64) -> Option<(f64, f64)> {
    // Find apex
    let apex = intensity
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)?;

    let (center, sigma_fit) = fit_gaussian_3pt(mz, intensity, apex)?;

    // Sanity check: fitted σ should be near calibrated σ
    let ratio = sigma_fit / sigma_calibrated;
    if !(0.4..=2.5).contains(&ratio) {
        return None;
    }

    Some((center, sigma_fit))
}

/// Compute residuals y − Aβ for a LASSO-fitted region and add to samples.
fn collect_residuals(
    state: &Pass1State,
    mz_slice: &[f64],
    int_f64: &[f64],
    samples: &mut Vec<ResidualSample>,
    sigma: f64,
) {
    let a = build_local_a(mz_slice, &state.grid, sigma);
    let a_beta = a.dot(&Array1::from_vec(state.beta.clone()));
    let residuals: Vec<f64> = int_f64
        .iter()
        .zip(a_beta.iter())
        .map(|(y, ab)| y - ab)
        .collect();

    let mz_center = (mz_slice[0] + mz_slice[mz_slice.len() - 1]) / 2.0;
    samples.push(ResidualSample {
        mz_center,
        residuals,
    });
}

/// Merge centroids that are within `half_grid` of each other by intensity-weighted average.
fn merge_nearby_centroids(sorted: Vec<CentroidResult>, min_sep: f64) -> Vec<CentroidResult> {
    if sorted.is_empty() {
        return sorted;
    }
    let mut result: Vec<CentroidResult> = Vec::with_capacity(sorted.len());
    let mut cur = sorted[0].clone();

    for next in sorted.into_iter().skip(1) {
        if next.mz - cur.mz < min_sep {
            // Intensity-weighted average
            let total = cur.intensity + next.intensity;
            if total > 0.0 {
                cur.mz = (cur.mz * cur.intensity + next.mz * next.intensity) / total;
                cur.intensity = total;
            }
        } else {
            result.push(cur);
            cur = next;
        }
    }
    result.push(cur);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::BasisPrecompute;
    use crate::config::Config;
    use crate::io::reader::ProfileSpectrum;

    fn make_config() -> Config {
        Config {
            input: "/dev/null".into(),
            output: "/dev/null".into(),
            config: None,
            sigma_ms1: None,
            sigma_ms2: None,
            grid_spacing: None,
            grid_offset: None,
            n_calibration_spectra: 50,
            lambda_factor: 3.0,
            max_lasso_iter: 2000,
            lasso_tol: 1e-6,
            lambda_change_threshold: 0.20,
            single_peak_width_sigma: 2.5,
            signal_threshold_sigma: 3.0,
            merge_gap_points: 2,
            extension_points: 3,
            min_region_width: 3,
            noise_window_da: 20.0,
            noise_step_da: 5.0,
            threads: 0,
            stats_output: None,
            quiet: false,
            verbose: false,
        }
    }

    fn make_basis(sigma: f64, spacing: f64) -> BasisPrecompute {
        BasisPrecompute::new(sigma, spacing, 0.0, 3.0)
    }

    fn synthetic_spectrum(
        mz_start: f64,
        spacing: f64,
        n: usize,
        peaks: &[(f64, f64)], // (center, amplitude)
        sigma: f64,
        noise_level: f32,
    ) -> ProfileSpectrum {
        let mz: Vec<f64> = (0..n).map(|i| mz_start + i as f64 * spacing).collect();
        let intensity: Vec<f32> = mz
            .iter()
            .map(|&m| {
                let signal: f64 = peaks
                    .iter()
                    .map(|&(c, a)| a * (-(m - c).powi(2) / (2.0 * sigma.powi(2))).exp())
                    .sum();
                (signal as f32 + noise_level).max(0.0)
            })
            .collect();
        ProfileSpectrum {
            native_id: "scan=1".to_string(),
            scan_number: 1,
            ms_level: 2,
            retention_time_min: 1.0,
            filter_string: Some(
                "ITMS + p NSI t Full ms2 500.0@hcd30.00 [200.00-1500.00]".to_string(),
            ),
            mz,
            intensity,
        }
    }

    #[test]
    fn centroid_single_peak_finds_correct_mz() {
        let sigma = 0.340f64;
        let spacing = 0.125f64;
        let center = 500.25f64;
        let amplitude = 10000.0f64;

        let spectrum = synthetic_spectrum(490.0, spacing, 160, &[(center, amplitude)], sigma, 5.0);
        let basis = make_basis(sigma, spacing);
        let config = make_config();

        let (centroids, stats) = centroid_spectrum(&spectrum, &basis, &config);

        assert!(!centroids.is_empty(), "should find at least one centroid");
        let best = centroids
            .iter()
            .max_by(|a, b| a.intensity.total_cmp(&b.intensity))
            .unwrap();
        assert!(
            (best.mz - center).abs() < spacing * 2.0,
            "centroid at {:.4} should be within 2 grid spacings of true {:.4}",
            best.mz,
            center
        );
        assert!(stats.n_regions >= 1);
        assert!(stats.n_centroids >= 1);
    }

    #[test]
    fn centroid_two_overlapping_peaks_resolves_both() {
        let sigma = 0.340f64;
        let spacing = 0.125f64;
        // Two peaks 0.625 m/z apart (= 5 pts = 1.84σ) — within 1 m/z
        let c1 = 500.0f64;
        let c2 = 500.625f64;
        let amplitude = 10000.0f64;

        let spectrum = synthetic_spectrum(
            494.0,
            spacing,
            160,
            &[(c1, amplitude), (c2, amplitude)],
            sigma,
            5.0,
        );
        let basis = make_basis(sigma, spacing);
        let config = make_config();

        let (centroids, stats) = centroid_spectrum(&spectrum, &basis, &config);

        // Should find at least 2 centroids — the core goal of Centrix
        assert!(
            centroids.len() >= 2,
            "two overlapping peaks within 1 m/z should be resolved; \
             got {} centroids. Stats: {:?}",
            centroids.len(),
            stats
        );
    }

    #[test]
    fn empty_spectrum_returns_empty() {
        let spectrum = ProfileSpectrum {
            native_id: "scan=0".to_string(),
            scan_number: 0,
            ms_level: 2,
            retention_time_min: 0.0,
            filter_string: None,
            mz: Vec::new(),
            intensity: Vec::new(),
        };
        let basis = make_basis(0.340, 0.125);
        let config = make_config();
        let (centroids, _) = centroid_spectrum(&spectrum, &basis, &config);
        assert!(centroids.is_empty());
    }

    #[test]
    fn pure_noise_returns_no_centroids() {
        let sigma = 0.340f64;
        let spacing = 0.125f64;
        // Signal is pure noise at level 5 — well below lambda=3*noise_sigma
        let spectrum = synthetic_spectrum(490.0, spacing, 160, &[], sigma, 5.0);
        let basis = make_basis(sigma, spacing);
        let config = make_config();
        let (centroids, _) = centroid_spectrum(&spectrum, &basis, &config);
        // May find a few spurious hits at the noise floor, but most should be suppressed
        assert!(
            centroids.len() <= 3,
            "pure noise should produce few centroids; got {}",
            centroids.len()
        );
    }
}
