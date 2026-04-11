//! Two-pass centroiding of a profile spectrum.
//!
//! ## Pipeline
//!
//! **Pass 1:**
//! 1. Rough noise estimate → baseline, noise_sigma
//! 2. Detect signal regions
//! 3. All regions → non-negative LASSO; cache Aᵀy and β
//! 4. Compute residuals (y − Aβ) for each region
//!
//! **Noise refinement:**
//! 5. Refine noise model from LASSO residuals + gap intensities
//!
//! **Pass 2 (selective):**
//! 6. For regions where λ changed > threshold: re-run LASSO
//!    warm-started from Pass-1 β; reuse cached Aᵀy (grid unchanged)
//!
//! **Output:**
//! 7. Collect all centroids; sub-grid refine LASSO centroids; sort and merge

use crate::basis::{build_local_a, compute_aty, BasisPrecompute};
use crate::config::Config;
use crate::io::reader::ProfileSpectrum;
use crate::lasso::{refine_subgrid, solve_nonneg_lasso, LassoInput};
use crate::noise::{needs_refit, refine_from_residuals, rough_noise_estimate, ResidualSample};
use crate::signal::detect_signal_regions;
use ndarray::Array1;

// ── Output types ──────────────────────────────────────────────────────────────

/// A single detected centroid from the two-pass algorithm.
#[derive(Debug, Clone)]
pub struct CentroidResult {
    /// Centroid m/z (sub-grid refined where possible)
    pub mz: f64,
    /// Integrated area (discrete sum convention): β × σ × √(2π) / h
    pub intensity: f64,
}

/// Per-spectrum statistics for diagnostics and quality monitoring.
#[derive(Debug, Clone, Default)]
pub struct SpectrumStats {
    pub scan_number: u32,
    pub n_regions: usize,
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
    /// Cached Aᵀy from Pass 1: reused in Pass 2 (grid doesn't change).
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
    );
    stats.n_regions = regions.len();

    if regions.is_empty() {
        return (Vec::new(), stats);
    }

    // ── Steps 3–4: Process each region in Pass 1 via LASSO ─────────────────────
    let mut all_centroids: Vec<CentroidResult> = Vec::new();
    let mut residual_samples: Vec<ResidualSample> = Vec::new();
    let mut pass1_states: Vec<Pass1State> = Vec::new();

    for (ridx, region) in regions.iter().enumerate() {
        let mz_slice = &mz[region.start_idx..=region.end_idx];
        let int_slice = &intensity[region.start_idx..=region.end_idx];
        let int_f64: Vec<f64> = int_slice.iter().map(|&v| v as f64).collect();

        let (centroids, state, residual) = run_lasso_region(
            ridx,
            mz_slice,
            &int_f64,
            basis,
            rough_lambda,
            region.center_mz(),
            None,
        );
        all_centroids.extend(centroids);
        if let Some(r) = residual {
            residual_samples.push(r);
        }
        if let Some(s) = state {
            pass1_states.push(s);
        }
        stats.n_lasso += 1;
    }

    // ── Step 5: Noise refinement ──────────────────────────────────────────────
    let refined_noise = refine_from_residuals(
        mz,
        intensity,
        &regions,
        &residual_samples,
        &noise,
        config.noise_window_da,
        config.noise_step_da,
    );

    // ── Step 6: Pass 2 — selective re-fit ────────────────────────────────────
    // Re-run LASSO only for regions where λ changed significantly.
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
            // Reuse cached Aᵀy from Pass 1 — grid doesn't change, only λ does
            let aty_slice = match state.aty.as_slice() {
                Some(s) => s,
                None => continue,
            };

            let input = LassoInput {
                aty: aty_slice,
                gram_row: &basis.gram_row,
                lambda: refined_lambda,
                warm_start: Some(&state.beta),
                max_iter: 2000,
                tol: 1e-6,
            };
            let output = solve_nonneg_lasso(&input);

            if output.n_nonzero() > 0 {
                let area_scale = basis.sigma * std::f64::consts::TAU.sqrt() / basis.grid_spacing;
                let centroids_raw = refine_subgrid(&output.beta, &state.grid);
                let centroids: Vec<CentroidResult> = centroids_raw
                    .into_iter()
                    .map(|(mz, amplitude)| CentroidResult {
                        mz,
                        intensity: amplitude * area_scale,
                    })
                    .collect();
                pass2_centroid_sets.push((state.region_idx, centroids));
            } else {
                pass2_centroid_sets.push((state.region_idx, Vec::new()));
            }
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

    // ── Step 7: Sort and merge near-duplicate centroids ───────────────────────
    // Merge centroids closer than σ — two Gaussians separated by less than σ
    // are physically unresolvable at 2 pts/σ sampling (MS2). The previous
    // threshold of grid_spacing/2 was too permissive and allowed the LASSO to
    // split a single peak into two adjacent grid-point coefficients that
    // survived as spurious close doublets.
    all_centroids.sort_by(|a, b| a.mz.total_cmp(&b.mz));

    let min_sep = config.min_centroid_separation.unwrap_or(basis.sigma);
    let merged = merge_nearby_centroids(all_centroids, min_sep);
    stats.n_centroids = merged.len();

    (merged, stats)
}

// ── LASSO region processing ───────────────────────────────────────────────────

/// Run LASSO on a signal region slice. Returns centroids, optional Pass-1 state,
/// and optional residual sample (computed using the A matrix before it's dropped).
fn run_lasso_region(
    region_idx: usize,
    mz_slice: &[f64],
    int_f64: &[f64],
    basis: &BasisPrecompute,
    lambda: f64,
    center_mz: f64,
    warm_start: Option<&[f64]>,
) -> (
    Vec<CentroidResult>,
    Option<Pass1State>,
    Option<ResidualSample>,
) {
    if mz_slice.len() < 2 {
        return (Vec::new(), None, None);
    }

    // Build grid matching the data points (square system)
    let grid = basis.grid_positions(mz_slice[0], *mz_slice.last().unwrap());
    if grid.is_empty() {
        return (Vec::new(), None, None);
    }

    let a = build_local_a(mz_slice, &grid, basis.sigma);
    let aty = compute_aty(&a, int_f64);
    let aty_slice = match aty.as_slice() {
        Some(s) => s,
        None => return (Vec::new(), None, None),
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
        return (Vec::new(), None, None);
    }

    // Compute residuals using A before it's dropped (avoids redundant rebuild)
    let a_beta = a.dot(&Array1::from_vec(output.beta.clone()));
    let residuals: Vec<f64> = int_f64
        .iter()
        .zip(a_beta.iter())
        .map(|(y, ab)| y - ab)
        .collect();
    let mz_center = (mz_slice[0] + mz_slice[mz_slice.len() - 1]) / 2.0;
    let residual_sample = ResidualSample {
        mz_center,
        residuals,
    };

    // Sub-grid centroid refinement; convert amplitude → integrated area
    // Discrete sum convention: β × σ√(2π) / h, matching Thermo centroider output
    let area_scale = basis.sigma * std::f64::consts::TAU.sqrt() / basis.grid_spacing;
    let centroids_raw = refine_subgrid(&output.beta, &grid);
    let centroids: Vec<CentroidResult> = centroids_raw
        .into_iter()
        .map(|(mz, amplitude)| CentroidResult {
            mz,
            intensity: amplitude * area_scale,
        })
        .collect();

    let state = Pass1State {
        region_idx,
        beta: output.beta,
        aty,
        grid,
        rough_lambda: lambda,
    };

    let _ = center_mz; // used by caller for noise lookup
    (centroids, Some(state), Some(residual_sample))
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
            input: vec!["/dev/null".to_string()],
            output: None,
            config: None,
            scan_rate: None,
            sigma: None,
            sigma_ms1: None,
            sigma_ms2: None,
            grid_spacing: None,
            grid_offset: None,
            n_calibration_spectra: 50,
            lambda_factor: 3.0,
            max_lasso_iter: 2000,
            lasso_tol: 1e-6,
            lambda_change_threshold: 0.20,
            signal_threshold_sigma: 3.0,
            merge_gap_points: 2,
            extension_points: 3,
            min_region_width: 3,
            noise_window_da: 20.0,
            noise_step_da: 5.0,
            min_centroid_separation: None,
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
    #[test]
    fn close_doublet_merged_into_single_centroid() {
        // A peak centered between two grid points should produce ONE centroid,
        // not a doublet at adjacent grid positions.
        let sigma = 0.340f64;
        let spacing = 0.125f64;
        // Place peak at a grid midpoint to maximize doublet splitting
        let center = 500.0625f64;
        let amplitude = 10000.0f64;

        let spectrum = synthetic_spectrum(494.0, spacing, 160, &[(center, amplitude)], sigma, 5.0);
        let basis = make_basis(sigma, spacing);
        let config = make_config();

        let (centroids, _) = centroid_spectrum(&spectrum, &basis, &config);

        // Find centroids near the true center
        let nearby: Vec<_> = centroids
            .iter()
            .filter(|c| (c.mz - center).abs() < 1.0)
            .collect();

        assert_eq!(
            nearby.len(),
            1,
            "single peak at grid midpoint should produce 1 centroid, not {}; \
             positions: {:?}",
            nearby.len(),
            nearby.iter().map(|c| c.mz).collect::<Vec<_>>()
        );
        // Merged centroid should be close to the true center
        assert!(
            (nearby[0].mz - center).abs() < spacing,
            "merged centroid at {:.4} should be within one grid spacing of {:.4}",
            nearby[0].mz,
            center
        );
    }

    #[test]
    fn well_separated_peaks_preserved() {
        // Two peaks separated by > σ should remain as two distinct centroids.
        let sigma = 0.340f64;
        let spacing = 0.125f64;
        let c1 = 500.0f64;
        let c2 = 500.625f64; // 0.625 Th apart = 1.84σ
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

        let (centroids, _) = centroid_spectrum(&spectrum, &basis, &config);

        let near_c1: Vec<_> = centroids
            .iter()
            .filter(|c| (c.mz - c1).abs() < 0.2)
            .collect();
        let near_c2: Vec<_> = centroids
            .iter()
            .filter(|c| (c.mz - c2).abs() < 0.2)
            .collect();

        assert!(
            !near_c1.is_empty() && !near_c2.is_empty(),
            "two peaks 1.84σ apart should both be preserved; centroids: {:?}",
            centroids.iter().map(|c| c.mz).collect::<Vec<_>>()
        );
    }
}
