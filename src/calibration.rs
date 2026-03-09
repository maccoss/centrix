//! Auto-calibration of Gaussian σ and profile grid spacing.
//!
//! ## σ calibration via CWT
//!
//! For a linear ion trap at a fixed scan rate, σ is **constant across the entire
//! m/z range**. The Stellar has 4 discrete scan rates:
//!
//! | Scan rate  | FWHM (m/z) | σ (m/z) |
//! |------------|------------|---------|
//! | 33 kTh/s  | 0.5        | 0.212   |
//! | 67 kTh/s  | 0.6        | 0.255   |
//! | 125 kTh/s | 0.8        | 0.340   |
//! | 200 kTh/s | 2.0        | 0.849   |
//!
//! MS1 and MS2 may use different scan rates and are calibrated independently.
//!
//! CWT with a Mexican Hat wavelet sweeps σ_pts values; the scale that maximizes
//! CWT response at each peak gives a direct, noise-robust σ estimate.
//! Unlike 3-point fitting, the wavelet rejects noise spikes and works without
//! requiring peak isolation.
//!
//! ## Grid spacing detection
//!
//! The Stellar writes data on a perfectly uniform firmware-fixed grid.
//! Empirically: MS2 = 1/8 m/z, MS1 = 1/15 m/z.

use crate::cwt::{
    cwt_scalogram, cwt_sigma_estimates, infer_scan_rate, relative_min_response, SIGMA_FALLBACK_MZ,
    SIGMA_PTS_SWEEP,
};
use crate::io::reader::ProfileSpectrum;
use crate::Result;

// ── Calibration result ────────────────────────────────────────────────────────

/// Result of the auto-calibration pass over the first N spectra.
#[derive(Debug, Clone)]
pub struct CalibrationResult {
    /// Gaussian σ in m/z for MS1 spectra
    pub sigma_ms1: f64,
    /// Gaussian σ in m/z for MS2 spectra
    pub sigma_ms2: f64,
    /// Profile grid spacing in m/z for MS1 (typically 1/15 ≈ 0.0667 m/z)
    pub grid_spacing_ms1: f64,
    /// Profile grid spacing in m/z for MS2 (typically 1/8 = 0.125 m/z)
    pub grid_spacing_ms2: f64,
    /// Number of peaks used for MS1 σ estimate
    pub n_peaks_ms1: usize,
    /// Number of peaks used for MS2 σ estimate
    pub n_peaks_ms2: usize,
}

impl CalibrationResult {
    pub fn sigma_for_level(&self, ms_level: u8) -> f64 {
        if ms_level == 1 {
            self.sigma_ms1
        } else {
            self.sigma_ms2
        }
    }

    pub fn grid_spacing_for_level(&self, ms_level: u8) -> f64 {
        if ms_level == 1 {
            self.grid_spacing_ms1
        } else {
            self.grid_spacing_ms2
        }
    }
}

// ── Main calibration entry point ──────────────────────────────────────────────

// ── Filter string scan rate parsing ──────────────────────────────────────────

/// Extract the Stellar scan rate code from a Thermo filter string.
///
/// The filter string format is: `"ITMS + p NSI {code} Full ms[2] ..."`
/// where code is one of: `n` (33 kTh/s), `r` (67 kTh/s), `t` (125 kTh/s),
/// `u` (200 kTh/s).
///
/// Returns the corresponding σ (m/z) if recognized, `None` otherwise.
pub fn sigma_from_filter_string(filter: &str) -> Option<f64> {
    // The scan rate letter appears after "NSI " in the filter string
    let after_nsi = filter.find("NSI ")?.checked_add(4)?;
    let code = filter[after_nsi..].chars().next()?;
    match code {
        'n' => Some(0.212), // 33 kTh/s, FWHM 0.5 m/z
        'r' => Some(0.255), // 67 kTh/s, FWHM 0.6 m/z
        't' => Some(0.340), // 125 kTh/s, FWHM 0.8 m/z
        'u' => Some(0.849), // 200 kTh/s, FWHM 2.0 m/z
        _ => None,
    }
}

/// Run the calibration pass over the provided spectra.
pub fn calibrate(
    spectra: &[ProfileSpectrum],
    sigma_override_ms1: Option<f64>,
    sigma_override_ms2: Option<f64>,
    grid_spacing_override: Option<f64>,
) -> Result<CalibrationResult> {
    let ms1: Vec<&ProfileSpectrum> = spectra.iter().filter(|s| s.ms_level == 1).collect();
    let ms2: Vec<&ProfileSpectrum> = spectra.iter().filter(|s| s.ms_level == 2).collect();

    let grid_spacing_ms1 = grid_spacing_override.unwrap_or_else(|| {
        ms1.first()
            .map(|s| detect_grid_spacing(s))
            .unwrap_or(1.0 / 15.0)
    });
    let grid_spacing_ms2 = grid_spacing_override
        .unwrap_or_else(|| ms2.first().map(|s| detect_grid_spacing(s)).unwrap_or(0.125));

    let (sigma_ms1, n_peaks_ms1) = if let Some(s) = sigma_override_ms1 {
        (s, 0)
    } else if ms1.is_empty() {
        log::warn!("No MS1 spectra for σ calibration; using fallback {SIGMA_FALLBACK_MZ:.3} m/z");
        (SIGMA_FALLBACK_MZ, 0)
    } else if let Some(s) = sigma_from_filter_strings(&ms1) {
        log::info!("MS1 σ from filter string: {s:.4} m/z");
        (s, ms1.len())
    } else {
        calibrate_sigma_cwt(&ms1, grid_spacing_ms1)
    };

    let (sigma_ms2, n_peaks_ms2) = if let Some(s) = sigma_override_ms2 {
        (s, 0)
    } else if ms2.is_empty() {
        log::warn!("No MS2 spectra for σ calibration; using fallback {SIGMA_FALLBACK_MZ:.3} m/z");
        (SIGMA_FALLBACK_MZ, 0)
    } else if let Some(s) = sigma_from_filter_strings(&ms2) {
        log::info!("MS2 σ from filter string: {s:.4} m/z");
        (s, ms2.len())
    } else {
        calibrate_sigma_cwt(&ms2, grid_spacing_ms2)
    };

    Ok(CalibrationResult {
        sigma_ms1,
        sigma_ms2,
        grid_spacing_ms1,
        grid_spacing_ms2,
        n_peaks_ms1,
        n_peaks_ms2,
    })
}

// ── Grid spacing detection ────────────────────────────────────────────────────

/// Detect the profile grid spacing from the median consecutive m/z difference.
pub fn detect_grid_spacing(spectrum: &ProfileSpectrum) -> f64 {
    let mz = &spectrum.mz;
    if mz.len() < 2 {
        return 0.125;
    }
    let mut diffs: Vec<f64> = mz.windows(2).map(|w| w[1] - w[0]).collect();
    diffs.sort_by(|a, b| a.total_cmp(b));
    let median = diffs[diffs.len() / 2];
    let q10 = diffs[diffs.len() / 10];
    let q90 = diffs[diffs.len() * 9 / 10];
    if (q90 - q10) / median > 0.1 {
        log::warn!(
            "Profile m/z spacing not uniform (10th={q10:.5}, 90th={q90:.5}); \
             grid spacing estimate may be unreliable"
        );
    }
    median
}

// ── Filter string σ lookup ────────────────────────────────────────────────────

/// Try to determine σ from the filter strings of a set of spectra.
///
/// Checks the first 10 spectra with a filter string. If all agree on the same
/// scan rate code, returns that σ. Returns `None` if no filter strings are
/// present or if scan rate codes are inconsistent.
fn sigma_from_filter_strings(spectra: &[&ProfileSpectrum]) -> Option<f64> {
    let mut seen: Option<f64> = None;
    let mut checked = 0;
    for s in spectra.iter().take(10) {
        if let Some(ref fs) = s.filter_string {
            if let Some(sigma) = sigma_from_filter_string(fs) {
                match seen {
                    None => seen = Some(sigma),
                    Some(prev) if (prev - sigma).abs() < 1e-6 => {}
                    _ => {
                        log::warn!(
                            "Inconsistent scan rate codes in filter strings; \
                             falling back to CWT calibration"
                        );
                        return None;
                    }
                }
                checked += 1;
            }
        }
    }
    if checked > 0 {
        seen
    } else {
        None
    }
}

// ── CWT-based σ calibration ───────────────────────────────────────────────────

fn calibrate_sigma_cwt(spectra: &[&ProfileSpectrum], grid_spacing_mz: f64) -> (f64, usize) {
    let isolation_pts = (1.0 / grid_spacing_mz).round() as usize;
    let mut all_sigma_mz: Vec<f64> = Vec::new();

    for spectrum in spectra {
        let intensity: Vec<f64> = spectrum.intensity.iter().map(|&v| v as f64).collect();
        if intensity.is_empty() {
            continue;
        }
        let scalogram = cwt_scalogram(&intensity, SIGMA_PTS_SWEEP);
        let min_resp = relative_min_response(&scalogram, 0.1);
        if min_resp <= 0.0 {
            continue;
        }
        let pts_estimates =
            cwt_sigma_estimates(&scalogram, SIGMA_PTS_SWEEP, min_resp, isolation_pts);
        for s_pts in pts_estimates {
            all_sigma_mz.push(s_pts * grid_spacing_mz);
        }
    }

    if all_sigma_mz.len() < 5 {
        let (rate, fwhm, sigma) = infer_scan_rate(SIGMA_FALLBACK_MZ);
        log::warn!(
            "Too few CWT peaks ({}) for σ calibration; using fallback \
             σ={:.3} m/z ({:.0} kTh/s, FWHM {:.2} m/z)",
            all_sigma_mz.len(),
            sigma,
            rate,
            fwhm,
        );
        return (SIGMA_FALLBACK_MZ, all_sigma_mz.len());
    }

    let n_raw = all_sigma_mz.len();
    log::info!("CWT σ calibration: {n_raw} peak estimates collected");

    let mut sorted = all_sigma_mz.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let median = sorted[sorted.len() / 2];
    let mut abs_devs: Vec<f64> = sorted.iter().map(|&s| (s - median).abs()).collect();
    abs_devs.sort_by(|a, b| a.total_cmp(b));
    let mad = abs_devs[abs_devs.len() / 2];
    let cutoff = 2.0 * mad.max(0.01 * median);

    let filtered: Vec<f64> = all_sigma_mz
        .iter()
        .copied()
        .filter(|&s| (s - median).abs() <= cutoff)
        .collect();

    log::info!(
        "CWT σ: {}/{n_raw} kept after 2×MAD (median={:.4} m/z, MAD={:.4})",
        filtered.len(),
        median,
        mad,
    );

    let sigma_final = filtered.iter().sum::<f64>() / filtered.len() as f64;
    let (rate, fwhm, _) = infer_scan_rate(sigma_final);
    log::info!("σ = {sigma_final:.4} m/z → {rate:.0} kTh/s (FWHM {fwhm:.2} m/z)");

    (sigma_final, filtered.len())
}

// ── 3-point Gaussian fit (Phase 4 fast path) ─────────────────────────────────

/// Fit a Gaussian to a peak apex using the 3-point log-space parabola trick.
///
/// Returns `(center_mz, sigma_mz)` or `None` if the fit fails.
pub fn fit_gaussian_3pt(mz: &[f64], intensity: &[f32], apex: usize) -> Option<(f64, f64)> {
    let n = mz.len();
    if apex == 0 || apex + 1 >= n {
        return None;
    }
    let (x0, y0) = (mz[apex - 1], intensity[apex - 1] as f64);
    let (x1, y1) = (mz[apex], intensity[apex] as f64);
    let (x2, y2) = (mz[apex + 1], intensity[apex + 1] as f64);
    if y0 <= 0.0 || y1 <= 0.0 || y2 <= 0.0 {
        return None;
    }
    let h = ((x1 - x0) + (x2 - x1)) / 2.0;
    if h < 1e-9 {
        return None;
    }
    let a = (y0.ln() - 2.0 * y1.ln() + y2.ln()) / (2.0 * h * h);
    let b = (y2.ln() - y0.ln()) / (2.0 * h);
    if a >= 0.0 {
        return None;
    }
    let sigma = (-1.0 / (2.0 * a)).sqrt();
    let center = x1 - b / (2.0 * a);
    if (center - x1).abs() > 2.0 * h {
        return None;
    }
    Some((center, sigma))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn filter_string_parses_all_scan_rates() {
        assert_abs_diff_eq!(
            sigma_from_filter_string("ITMS + p NSI n Full ms [350.00-1250.00]").unwrap(),
            0.212,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            sigma_from_filter_string("ITMS + p NSI r Full ms [350.00-1250.00]").unwrap(),
            0.255,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            sigma_from_filter_string("ITMS + p NSI t Full ms2 400.93@hcd30.00 [200.00-1500.00]")
                .unwrap(),
            0.340,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            sigma_from_filter_string("ITMS + p NSI u Full ms [350.00-1250.00]").unwrap(),
            0.849,
            epsilon = 1e-6
        );
        assert!(sigma_from_filter_string("unrecognized filter string").is_none());
    }

    fn make_spectrum(
        mz_start: f64,
        spacing: f64,
        n: usize,
        centers: &[(f64, f64)],
        sigma: f64,
        ms_level: u8,
    ) -> ProfileSpectrum {
        let mz: Vec<f64> = (0..n).map(|i| mz_start + i as f64 * spacing).collect();
        let intensity: Vec<f32> = mz
            .iter()
            .map(|&m| {
                centers
                    .iter()
                    .map(|&(c, a)| a * (-(m - c).powi(2) / (2.0 * sigma.powi(2))).exp())
                    .sum::<f64>() as f32
            })
            .collect();
        ProfileSpectrum {
            native_id: "scan=1".to_string(),
            scan_number: 0,
            ms_level,
            filter_string: None,
            retention_time_min: 0.0,
            mz,
            intensity,
        }
    }

    #[test]
    fn detect_grid_spacing_ms2() {
        let s = make_spectrum(200.0, 0.125, 800, &[(250.0, 1000.0)], 0.34, 2);
        assert_abs_diff_eq!(detect_grid_spacing(&s), 0.125, epsilon = 1e-6);
    }

    #[test]
    fn detect_grid_spacing_ms1() {
        let s = make_spectrum(350.0, 1.0 / 15.0, 1000, &[(600.0, 1000.0)], 0.34, 1);
        assert_abs_diff_eq!(detect_grid_spacing(&s), 1.0 / 15.0, epsilon = 1e-9);
    }

    #[test]
    fn cwt_calibration_recovers_sigma_125_khz() {
        let sigma_true = 0.340f64;
        let spacing = 0.125f64;
        let spectra: Vec<ProfileSpectrum> = [300.0, 400.0, 500.0, 600.0, 700.0f64]
            .iter()
            .map(|&c| make_spectrum(c - 40.0, spacing, 640, &[(c, 50000.0)], sigma_true, 2))
            .collect();
        let refs: Vec<&ProfileSpectrum> = spectra.iter().collect();
        let (sigma, n) = calibrate_sigma_cwt(&refs, spacing);
        assert!(n >= 3, "should use at least 3 peaks, got {n}");
        let ratio = sigma / sigma_true;
        assert!(
            ratio > 0.6 && ratio < 1.4,
            "σ={sigma:.4} vs true={sigma_true:.4} (ratio={ratio:.2})"
        );
    }

    #[test]
    fn fit_gaussian_3pt_recovers_sigma() {
        let sigma_true = 0.34f64;
        let spacing = 0.125f64;
        let mz: Vec<f64> = (0..20).map(|i| 498.0 + i as f64 * spacing).collect();
        let center = 500.0f64;
        let intensity: Vec<f32> = mz
            .iter()
            .map(|&m| (10000.0 * (-(m - center).powi(2) / (2.0 * sigma_true.powi(2))).exp()) as f32)
            .collect();
        let apex = intensity
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap();
        let (c, s) = fit_gaussian_3pt(&mz, &intensity, apex).unwrap();
        assert_abs_diff_eq!(c, center, epsilon = 0.02);
        assert_abs_diff_eq!(s, sigma_true, epsilon = 0.03);
    }
}
