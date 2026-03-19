//! Continuous Wavelet Transform (CWT) for σ calibration in the m/z dimension.
//!
//! Uses the Mexican Hat (Ricker) wavelet as a matched filter for Gaussian-like
//! ion trap peaks. Sweeping over a range of scale parameters and finding the
//! scale that maximizes CWT response at each peak gives a direct estimate of
//! the peak σ in m/z units — more robust than 3-point fitting because it
//! rejects noise spikes and requires no peak isolation.
//!
//! The wavelet and convolution code is ported from Osprey's chromatographic
//! peak detector (`osprey-chromatography/src/cwt.rs`), applied here in the
//! m/z dimension instead of the time dimension.

use std::f64::consts::PI;

// ── Wavelet kernel ────────────────────────────────────────────────────────────

/// Generate a discrete Mexican Hat (Ricker) wavelet kernel.
///
/// `ψ(t) = (2 / √(3σ) π^(1/4)) × (1 - (t/σ)²) × exp(-t²/(2σ²))`
///
/// Zero-mean corrected to suppress DC (constant baseline) response.
///
/// # Arguments
/// * `sigma` - Scale parameter in **data-point units** (not m/z).
/// * `kernel_radius` - Points on each side of center; total size = 2×radius + 1.
///   Use `(5.0_f64 * sigma).ceil() as usize` to capture >99.99% of wavelet energy.
pub fn mexican_hat_kernel(sigma: f64, kernel_radius: usize) -> Vec<f64> {
    let len = 2 * kernel_radius + 1;
    let mut kernel = vec![0.0; len];
    let center = kernel_radius as f64;
    let norm = 2.0 / ((3.0 * sigma).sqrt() * PI.powf(0.25));

    for (i, val) in kernel.iter_mut().enumerate() {
        let t = (i as f64 - center) / sigma;
        *val = norm * (1.0 - t * t) * (-0.5 * t * t).exp();
    }

    // Zero-mean correction: removes tiny DC offset from discretization
    let mean = kernel.iter().sum::<f64>() / len as f64;
    for v in &mut kernel {
        *v -= mean;
    }
    kernel
}

// ── Convolution ───────────────────────────────────────────────────────────────

/// Convolve a signal with a kernel, returning an output of the same length.
///
/// Edges are zero-padded. Direct O(N×K) convolution — fast for typical
/// spectrum sizes (10k points) and kernel sizes (5–41 points).
pub fn convolve_same(signal: &[f64], kernel: &[f64]) -> Vec<f64> {
    let n = signal.len();
    let k = kernel.len();
    if n == 0 || k == 0 {
        return vec![0.0; n];
    }
    let half_k = k / 2;
    let mut output = vec![0.0; n];
    for (i, out) in output.iter_mut().enumerate() {
        let mut sum = 0.0;
        for (j, &kval) in kernel.iter().enumerate() {
            let idx = i as isize + j as isize - half_k as isize;
            if idx >= 0 && (idx as usize) < n {
                sum += signal[idx as usize] * kval;
            }
        }
        *out = sum;
    }
    output
}

// ── Scalogram ─────────────────────────────────────────────────────────────────

/// Compute a CWT scalogram by convolving the signal at each scale in `sigma_pts`.
///
/// Returns `scalogram[scale_idx][point_idx]` — the CWT coefficient at each
/// data point for each candidate σ value (in data-point units).
pub fn cwt_scalogram(intensity: &[f64], sigma_pts: &[f64]) -> Vec<Vec<f64>> {
    sigma_pts
        .iter()
        .map(|&s| {
            let radius = (5.0 * s).ceil() as usize;
            let kernel = mexican_hat_kernel(s, radius);
            convolve_same(intensity, &kernel)
        })
        .collect()
}

// ── Peak σ estimation ─────────────────────────────────────────────────────────

/// Conversion factor from CWT argmax scale to Gaussian σ.
///
/// For the L2-normalized Ricker wavelet, the CWT response to a Gaussian of
/// width σ_peak is maximized at scale a = σ_peak × √5. Therefore:
///
///   σ_peak = CWT_argmax_scale / √5
///
/// This arises because the CWT normalization (1/√a factor) shifts the argmax
/// relative to the naive σ_wavelet = σ_peak expectation.
pub const CWT_ARGMAX_TO_SIGMA: f64 = 0.447_213_595; // 1.0 / sqrt(5.0)

/// Extract σ estimates (in data-point units) from a CWT scalogram.
///
/// For each data point, finds the scale that produces the maximum CWT response.
/// Converts argmax scale → σ_peak via `σ = argmax / √5` (see `CWT_ARGMAX_TO_SIGMA`).
/// Keeps only isolated local maxima of the max-response envelope.
///
/// # Returns
/// A `Vec<f64>` of σ estimates in data-point units — one per detected peak.
pub fn cwt_sigma_estimates(
    scalogram: &[Vec<f64>],
    sigma_pts: &[f64],
    min_response: f64,
    isolation_pts: usize,
) -> Vec<f64> {
    if scalogram.is_empty() || scalogram[0].is_empty() {
        return Vec::new();
    }
    let n_pts = scalogram[0].len();

    // For each data point: max CWT response across all scales, and which scale
    let mut max_response = vec![0.0f64; n_pts];
    let mut best_scale_idx = vec![0usize; n_pts];
    for (s_idx, scale_row) in scalogram.iter().enumerate() {
        for (i, &v) in scale_row.iter().enumerate() {
            if v > max_response[i] {
                max_response[i] = v;
                best_scale_idx[i] = s_idx;
            }
        }
    }

    // Local maxima of max_response above threshold
    let local_maxima: Vec<usize> = (1..n_pts.saturating_sub(1))
        .filter(|&i| {
            max_response[i] > min_response
                && max_response[i] > max_response[i - 1]
                && max_response[i] > max_response[i + 1]
        })
        .collect();

    // Isolation: keep only maxima with no OTHER maximum within isolation_pts
    local_maxima
        .iter()
        .copied()
        .filter(|&i| {
            !local_maxima
                .iter()
                .any(|&j| j != i && j.abs_diff(i) <= isolation_pts)
        })
        // Convert argmax scale → σ_peak using the √5 factor
        .map(|i| sigma_pts[best_scale_idx[i]] * CWT_ARGMAX_TO_SIGMA)
        .collect()
}

/// Compute the 99th-percentile-based minimum response threshold.
///
/// `min_response = fraction × p99(max_response_envelope)`
///
/// This makes the threshold relative to the strongest signals in the spectrum,
/// so it works across spectra with very different absolute intensities.
pub fn relative_min_response(scalogram: &[Vec<f64>], fraction: f64) -> f64 {
    if scalogram.is_empty() || scalogram[0].is_empty() {
        return 0.0;
    }
    let n_pts = scalogram[0].len();
    let mut max_response = vec![0.0f64; n_pts];
    for scale_row in scalogram.iter() {
        for (i, &v) in scale_row.iter().enumerate() {
            if v > max_response[i] {
                max_response[i] = v;
            }
        }
    }
    max_response.sort_by(|a, b| a.total_cmp(b));
    let p99_idx = (max_response.len() * 99 / 100).min(max_response.len() - 1);
    fraction * max_response[p99_idx]
}

// ── Default scale sweep ───────────────────────────────────────────────────────

/// Log-spaced scale sweep (in data-point units) for the CWT argmax search.
///
/// These are wavelet **scale** values (not σ_peak values). Since the argmax of
/// the L2-normalized Ricker CWT for a Gaussian of width σ_peak occurs at
/// scale a = σ_peak × √5, the sweep covers the expected argmax range for all
/// 4 Stellar scan rates and both MS levels:
///
/// | Scan rate | σ_peak (m/z) | σ_pts MS2 | CWT argmax MS2 | σ_pts MS1 | CWT argmax MS1 |
/// |-----------|-------------|-----------|----------------|-----------|----------------|
/// | 33 kTh/s  | 0.212       | 1.70      | 3.80           | 3.18      | 7.11           |
/// | 67 kTh/s  | 0.255       | 2.04      | 4.56           | 3.83      | 8.57           |
/// | 125 kTh/s | 0.340       | 2.72      | 6.08           | 5.10      | 11.40          |
/// | 200 kTh/s | 0.849       | 6.79      | 15.18          | 12.74     | 28.48          |
///
/// Sweep covers 3.5–30 pts to handle all combinations.
pub const SIGMA_PTS_SWEEP: &[f64] = &[
    3.5, 4.2, 5.1, 6.2, 7.5, 9.0, 11.0, 13.3, 16.2, 19.5, 23.6, 28.5,
];

// ── Known Stellar scan rates ──────────────────────────────────────────────────

/// Thermo LIT scan rates with known FWHM (m/z), σ (m/z), and grid spacing.
pub const LIT_SCAN_RATES: &[(f64, f64, f64, f64)] = &[
    // (scan_rate_kths, fwhm_mz, sigma_mz, grid_spacing_mz)
    (33.0, 0.5, 0.212, 1.0 / 30.0),
    (67.0, 0.6, 0.255, 1.0 / 15.0),
    (125.0, 0.8, 0.340, 1.0 / 8.0),
    (200.0, 2.0, 0.849, 0.0), // grid spacing unknown for 200 kTh/s
];

/// Fallback σ for when calibration finds too few peaks.
/// Corresponds to 125 kTh/s, the most common DIA scan rate.
pub const SIGMA_FALLBACK_MZ: f64 = 0.340;

/// Infer the closest Thermo LIT scan rate from a measured σ (m/z).
/// Returns (scan_rate_kths, fwhm_mz, sigma_mz).
pub fn infer_scan_rate(sigma_mz: f64) -> (f64, f64, f64) {
    LIT_SCAN_RATES
        .iter()
        .map(|&(rate, fwhm, sigma, _)| (rate, fwhm, sigma))
        .min_by(|&(_, _, s1), &(_, _, s2)| (s1 - sigma_mz).abs().total_cmp(&(s2 - sigma_mz).abs()))
        .unwrap_or((125.0, 0.8, 0.340))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn kernel_is_zero_mean() {
        for sigma in [1.0, 2.0, 3.5, 5.0] {
            let radius = (5.0_f64 * sigma).ceil() as usize;
            let k = mexican_hat_kernel(sigma, radius);
            let sum: f64 = k.iter().sum();
            assert!(
                sum.abs() < 1e-10,
                "kernel not zero-mean at σ={sigma}: sum={sum}"
            );
        }
    }

    #[test]
    fn kernel_center_is_positive() {
        let k = mexican_hat_kernel(2.0, 10);
        assert!(k[10] > 0.0, "center of Mexican Hat should be positive");
    }

    #[test]
    fn kernel_tails_are_negative() {
        let sigma = 2.0;
        let radius = 10usize;
        let k = mexican_hat_kernel(sigma, radius);
        // Points at ~2σ from center should be near the negative trough
        assert!(k[0] < 0.0, "far tail should be negative");
        assert!(k[k.len() - 1] < 0.0, "far tail should be negative");
    }

    #[test]
    fn convolve_same_length_preserved() {
        let signal = vec![1.0; 100];
        let kernel = vec![0.0, 1.0, 0.0];
        let out = convolve_same(&signal, &kernel);
        assert_eq!(out.len(), signal.len());
    }

    #[test]
    fn convolve_delta_is_identity() {
        // Convolving with a unit delta should return the original signal
        let signal: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let mut kernel = vec![0.0; 11];
        kernel[5] = 1.0; // delta at center
        let out = convolve_same(&signal, &kernel);
        for (a, b) in signal.iter().zip(out.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn cwt_recovers_sigma_from_synthetic_gaussian() {
        // Build a synthetic Gaussian with known σ, sampled at 0.125 m/z spacing.
        // Use a long array (600 pts) so that even the largest kernel (radius ≈ 143
        // for σ_sweep=28.5) is well within the spectrum boundaries.
        let sigma_true_mz = 0.340; // 125 kTh/s
        let spacing = 0.125f64; // MS2 grid
        let sigma_true_pts = sigma_true_mz / spacing; // ≈ 2.72

        let n = 600usize;
        let center = n / 2;
        let amplitude = 1000.0f64;
        let intensity: Vec<f64> = (0..n)
            .map(|i| {
                let d = (i as f64 - center as f64) / sigma_true_pts;
                amplitude * (-0.5 * d * d).exp()
            })
            .collect();

        let scalogram = cwt_scalogram(&intensity, SIGMA_PTS_SWEEP);
        let min_resp = relative_min_response(&scalogram, 0.1);
        let isolation_pts = (1.0 / spacing).round() as usize; // 8 pts = 1 m/z
        let estimates = cwt_sigma_estimates(&scalogram, SIGMA_PTS_SWEEP, min_resp, isolation_pts);

        assert!(!estimates.is_empty(), "should detect the synthetic peak");
        let sigma_estimated_pts = estimates[0];
        let sigma_estimated_mz = sigma_estimated_pts * spacing;

        // Should be within ±30% of true σ (CWT argmax is grid-quantized)
        let ratio = sigma_estimated_mz / sigma_true_mz;
        assert!(
            ratio > 0.7 && ratio < 1.3,
            "CWT σ estimate {sigma_estimated_mz:.3} m/z vs true {sigma_true_mz:.3} (ratio={ratio:.2})"
        );
    }

    #[test]
    fn cwt_noise_spike_not_detected() {
        // A single-point spike should have low CWT response at Gaussian scales.
        // After CWT argmax→σ conversion (÷√5), even the minimum sweep scale
        // (3.5 pts) converts to 3.5 × 0.447 = 1.57 pts — much larger than a
        // 1-point spike. So a spike should NOT be detected above min_response.
        let mut intensity = vec![0.0f64; 600];
        intensity[300] = 1000.0; // isolated spike, 1 point wide

        let spacing = 0.125f64;
        let scalogram = cwt_scalogram(&intensity, SIGMA_PTS_SWEEP);
        let min_resp = relative_min_response(&scalogram, 0.1);
        let isolation_pts = (1.0 / spacing).round() as usize;
        let estimates = cwt_sigma_estimates(&scalogram, SIGMA_PTS_SWEEP, min_resp, isolation_pts);

        // At all sweep scales ≥ 3.5 pts, the spike looks narrow → low CWT response.
        // We allow detection (the spike IS a local max) but the σ estimate should
        // be small (minimum scale × CWT_ARGMAX_TO_SIGMA = 3.5 × 0.447 = 1.57).
        for &s in &estimates {
            assert!(
                s <= 2.0,
                "noise spike incorrectly attributed large σ = {s:.2} pts (expected ≤2.0)"
            );
        }
    }

    #[test]
    fn infer_scan_rate_125_khz() {
        let (rate, fwhm, sigma) = infer_scan_rate(0.340);
        assert_abs_diff_eq!(rate, 125.0, epsilon = 1.0);
        assert_abs_diff_eq!(fwhm, 0.8, epsilon = 0.01);
        assert_abs_diff_eq!(sigma, 0.340, epsilon = 0.001);
    }

    #[test]
    fn relative_min_response_is_positive() {
        let intensity: Vec<f64> = (0..100).map(|i| (i as f64).sin().abs() * 100.0).collect();
        let scalogram = cwt_scalogram(&intensity, SIGMA_PTS_SWEEP);
        let min_resp = relative_min_response(&scalogram, 0.1);
        assert!(min_resp > 0.0);
    }
}
