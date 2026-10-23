use crate::{CentrixError, Result};
use mzdata::io::mzml::MzMLReader;
use mzdata::prelude::*;
use mzdata::spectrum::RefPeakDataLevel;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

/// A single profile-mode spectrum extracted from mzML.
#[derive(Debug, Clone)]
pub struct ProfileSpectrum {
    /// Native spectrum identifier from the mzML file (e.g. `controllerType=0 ... scan=1234`)
    pub native_id: String,
    /// 0-based scan index
    pub scan_number: u32,
    pub ms_level: u8,
    /// Retention time in minutes
    pub retention_time_min: f64,
    /// Thermo filter string, e.g. `"ITMS + p NSI t Full ms2 400.93@hcd30.00 [200.00-1500.00]"`.
    /// Contains the scan rate letter: n=33kTh/s, r=67kTh/s, t=125kTh/s, u=200kTh/s.
    pub filter_string: Option<String>,
    /// m/z array (profile sampling positions)
    pub mz: Vec<f64>,
    /// Intensity array (profile intensities)
    pub intensity: Vec<f32>,
}

/// Streaming reader for profile-mode mzML files.
///
/// Iterates over all spectra (MS1 and MS2) in file order, returning
/// `ProfileSpectrum` for each, including spectra with no data points, which
/// come back with empty arrays. Emits an error and stops if a centroid-mode
/// spectrum is encountered, since Centrix requires raw profile data.
///
/// Empty spectra must not be skipped: the passthrough writer rewrites each
/// `<spectrum>` of the input with the result at the same position, so a
/// skipped spectrum would shift every later result onto the spectrum before it.
pub struct ProfileReader {
    path: PathBuf,
    inner: MzMLReader<BufReader<File>>,
}

impl ProfileReader {
    /// Open a profile-mode mzML file for reading.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path).map_err(CentrixError::Io)?;
        let inner = MzMLReader::new(BufReader::new(file));
        Ok(Self { path, inner })
    }

    /// Path to the source file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Total number of spectra, if available from mzML metadata.
    pub fn spectrum_count_hint(&self) -> Option<u64> {
        self.inner.spectrum_count_hint()
    }
}

impl Iterator for ProfileReader {
    type Item = Result<ProfileSpectrum>;

    fn next(&mut self) -> Option<Self::Item> {
        let mz_spectrum = self.inner.next()?;
        Some(convert_spectrum(mz_spectrum, &self.path))
    }
}

/// Convert an mzdata `MultiLayerSpectrum` to a `ProfileSpectrum`.
///
/// Spectra with no usable data (no arrays, unreadable arrays, or no points)
/// come back with empty `mz` and `intensity`.
/// Returns `Err` if the spectrum is centroid-mode (Centrix requires profile).
fn convert_spectrum(
    mz_spectrum: mzdata::spectrum::MultiLayerSpectrum,
    path: &Path,
) -> Result<ProfileSpectrum> {
    let desc = mz_spectrum.description();

    let native_id = desc.id.clone();
    let scan_number = desc.index as u32;
    let ms_level = desc.ms_level;
    let retention_time_min = desc
        .acquisition
        .first_scan()
        .map_or(0.0, |scan| scan.start_time);
    let filter_string = desc
        .acquisition
        .first_scan()
        .and_then(|scan| scan.filter_string())
        .map(|s| s.to_string());

    let peaks = mz_spectrum.peaks();
    let (mz, intensity) = match peaks {
        RefPeakDataLevel::Missing => (Vec::new(), Vec::new()),

        RefPeakDataLevel::RawData(arrays) => match (arrays.mzs(), arrays.intensities()) {
            (Ok(mz), Ok(intensity)) => (
                mz.iter().copied().collect(),
                intensity.iter().copied().collect(),
            ),
            _ => {
                log::warn!("Unreadable data arrays in spectrum {native_id}; writing it empty");
                (Vec::new(), Vec::new())
            }
        },

        RefPeakDataLevel::Centroid(_) | RefPeakDataLevel::Deconvoluted(_) => {
            return Err(CentrixError::NotProfileMode {
                path: path.display().to_string(),
            });
        }
    };

    Ok(ProfileSpectrum {
        native_id,
        scan_number,
        ms_level,
        retention_time_min,
        filter_string,
        mz,
        intensity,
    })
}

/// Load the first `n` non-empty spectra from a profile mzML file.
///
/// Used during calibration to auto-detect σ and grid spacing, which reads the
/// grid spacing from the first spectrum of each MS level, so empty spectra are
/// left out.
pub fn load_first_n<P: AsRef<Path>>(path: P, n: usize) -> Result<Vec<ProfileSpectrum>> {
    ProfileReader::open(path)?
        .filter(|r| r.as_ref().map_or(true, |s| !s.mz.is_empty()))
        .take(n)
        .collect::<Result<Vec<_>>>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_spectrum_fields() {
        let s = ProfileSpectrum {
            native_id: "scan=1".to_string(),
            scan_number: 0,
            ms_level: 2,
            retention_time_min: 1.5,
            filter_string: Some(
                "ITMS + p NSI t Full ms2 400.93@hcd30.00 [200.00-1500.00]".to_string(),
            ),
            mz: vec![100.0, 100.1, 100.2],
            intensity: vec![0.0, 1000.0, 0.0],
        };
        assert_eq!(s.ms_level, 2);
        assert_eq!(s.mz.len(), s.intensity.len());
    }
}
