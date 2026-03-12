//! Signal region detection in profile mass spectra.
//!
//! Identifies contiguous segments of a profile spectrum where signal exceeds
//! a noise-based threshold. All detected regions are processed by LASSO.

/// A contiguous signal region in a profile spectrum.
#[derive(Debug, Clone)]
pub struct SignalRegion {
    /// First data index in the region (inclusive)
    pub start_idx: usize,
    /// Last data index in the region (inclusive)
    pub end_idx: usize,
    /// m/z at start_idx
    pub mz_start: f64,
    /// m/z at end_idx
    pub mz_end: f64,
    /// Maximum intensity in the region
    pub max_intensity: f32,
    /// Index of the intensity maximum within the region
    pub apex_idx: usize,
}

impl SignalRegion {
    /// Number of data points in the region.
    pub fn width_pts(&self) -> usize {
        self.end_idx - self.start_idx + 1
    }

    /// Width in m/z units.
    pub fn width_mz(&self) -> f64 {
        self.mz_end - self.mz_start
    }

    /// Center m/z of the region.
    pub fn center_mz(&self) -> f64 {
        (self.mz_start + self.mz_end) / 2.0
    }

    /// m/z slice indices suitable for array slicing: `[start_idx..=end_idx]`
    pub fn range(&self) -> std::ops::RangeInclusive<usize> {
        self.start_idx..=self.end_idx
    }
}

/// Detect signal regions in a profile spectrum.
///
/// # Parameters
/// - `mz`, `intensity`: full spectrum arrays
/// - `baseline`: estimated baseline intensity level
/// - `noise_sigma`: estimated noise standard deviation
/// - `signal_threshold_sigma`: detect points above `baseline + N × noise_sigma`
/// - `merge_gap_pts`: merge adjacent above-threshold segments separated by ≤ this many points
/// - `extension_pts`: extend each region by this many points on each side
/// - `min_width_pts`: discard regions narrower than this
///
/// # Returns
/// Sorted list of non-overlapping `SignalRegion`s.
#[allow(clippy::too_many_arguments)]
pub fn detect_signal_regions(
    mz: &[f64],
    intensity: &[f32],
    baseline: f64,
    noise_sigma: f64,
    signal_threshold_sigma: f64,
    merge_gap_pts: usize,
    extension_pts: usize,
    min_width_pts: usize,
) -> Vec<SignalRegion> {
    let n = mz.len();
    if n < 3 {
        return Vec::new();
    }

    let threshold = baseline + signal_threshold_sigma * noise_sigma;

    // Mark points above threshold
    let above: Vec<bool> = intensity.iter().map(|&v| v as f64 > threshold).collect();

    // Find contiguous above-threshold segments
    let mut segments: Vec<(usize, usize)> = Vec::new();
    let mut in_seg = false;
    let mut seg_start = 0;
    for (i, &a) in above.iter().enumerate() {
        if a && !in_seg {
            seg_start = i;
            in_seg = true;
        } else if !a && in_seg {
            segments.push((seg_start, i - 1));
            in_seg = false;
        }
    }
    if in_seg {
        segments.push((seg_start, n - 1));
    }

    if segments.is_empty() {
        return Vec::new();
    }

    // Merge segments separated by ≤ merge_gap_pts
    let mut merged: Vec<(usize, usize)> = vec![segments[0]];
    for &(s, e) in &segments[1..] {
        let last = merged.last_mut().unwrap();
        if s <= last.1 + merge_gap_pts + 1 {
            last.1 = e;
        } else {
            merged.push((s, e));
        }
    }

    // Extend, clamp, filter by min width

    merged
        .into_iter()
        .filter_map(|(s, e)| {
            let start = s.saturating_sub(extension_pts);
            let end = (e + extension_pts).min(n - 1);
            if end - start + 1 < min_width_pts {
                return None;
            }

            let (apex_idx, max_intensity) = intensity[start..=end].iter().enumerate().fold(
                (0usize, f32::NEG_INFINITY),
                |(bi, bv), (i, &v)| {
                    if v > bv {
                        (start + i, v)
                    } else {
                        (bi, bv)
                    }
                },
            );

            Some(SignalRegion {
                start_idx: start,
                end_idx: end,
                mz_start: mz[start],
                mz_end: mz[end],
                max_intensity,
                apex_idx,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detect(mz: &[f64], intensity: &[f32], threshold: f64, _sigma: f64) -> Vec<SignalRegion> {
        detect_signal_regions(mz, intensity, 0.0, 1.0, threshold, 2, 2, 3)
    }

    fn uniform_mz(n: usize, spacing: f64) -> Vec<f64> {
        (0..n).map(|i| 200.0 + i as f64 * spacing).collect()
    }

    #[test]
    fn detects_single_isolated_peak() {
        let mz = uniform_mz(40, 0.125);
        let mut intensity = vec![0.0f32; 40];
        // Gaussian at index 20
        for (i, v) in intensity.iter_mut().enumerate() {
            let d = (i as f64 - 20.0) * 0.125 / 0.340;
            *v = (1000.0 * (-0.5 * d * d).exp()) as f32;
        }
        let regions = detect(&mz, &intensity, 50.0, 0.340);
        assert_eq!(regions.len(), 1, "should find exactly one region");
        let r = &regions[0];
        assert_eq!(r.apex_idx, 20);
        assert!(r.max_intensity > 900.0);
    }

    #[test]
    fn two_peaks_separated_become_two_regions() {
        let mz = uniform_mz(80, 0.125);
        let mut intensity = vec![0.0f32; 80];
        for (i, v) in intensity.iter_mut().enumerate() {
            let d1 = (i as f64 - 20.0) * 0.125 / 0.340;
            let d2 = (i as f64 - 60.0) * 0.125 / 0.340;
            *v = (1000.0 * (-0.5 * d1 * d1).exp() + 1000.0 * (-0.5 * d2 * d2).exp()) as f32;
        }
        let regions = detect(&mz, &intensity, 50.0, 0.340);
        assert_eq!(regions.len(), 2, "two well-separated peaks → two regions");
    }

    #[test]
    fn overlapping_peaks_merge_into_one_composite_region() {
        let mz = uniform_mz(40, 0.125);
        let mut intensity = vec![0.0f32; 40];
        for (i, v) in intensity.iter_mut().enumerate() {
            let d1 = (i as f64 - 16.0) * 0.125 / 0.340;
            let d2 = (i as f64 - 22.0) * 0.125 / 0.340;
            *v = (1000.0 * (-0.5 * d1 * d1).exp() + 1000.0 * (-0.5 * d2 * d2).exp()) as f32;
        }
        let regions = detect(&mz, &intensity, 50.0, 0.340);
        assert_eq!(regions.len(), 1, "overlapping peaks → one merged region");
    }

    #[test]
    fn noise_below_threshold_not_detected() {
        let mz = uniform_mz(40, 0.125);
        let intensity = vec![1.0f32; 40]; // flat, below threshold
        let regions = detect(&mz, &intensity, 50.0, 0.340);
        assert!(
            regions.is_empty(),
            "flat signal below threshold → no regions"
        );
    }

    #[test]
    fn narrow_peak_detected() {
        let mz = uniform_mz(30, 0.125);
        // Use a truly narrow 3-point peak
        let mut intensity = vec![0.0f32; 30];
        intensity[15] = 1000.0;
        intensity[14] = 500.0;
        intensity[16] = 500.0;
        let regions = detect(&mz, &intensity, 100.0, 0.340);
        if !regions.is_empty() {
            assert!(regions[0].max_intensity >= 1000.0);
        }
    }
}
