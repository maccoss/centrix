use clap::Parser;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

mod defaults {
    pub const N_CALIBRATION_SPECTRA: usize = 50;
    pub const LAMBDA_FACTOR: f64 = 3.0;
    pub const MAX_LASSO_ITER: usize = 1000;
    pub const LASSO_TOL: f64 = 1e-6;
    pub const LAMBDA_CHANGE_THRESHOLD: f64 = 0.20;
    pub const SIGNAL_THRESHOLD_SIGMA: f64 = 3.0;
    pub const MERGE_GAP_POINTS: usize = 2;
    pub const EXTENSION_POINTS: usize = 3;
    pub const MIN_REGION_WIDTH: usize = 3;
    pub const NOISE_WINDOW_DA: f64 = 20.0;
    pub const NOISE_STEP_DA: f64 = 5.0;

    pub fn n_calibration_spectra() -> usize {
        N_CALIBRATION_SPECTRA
    }
    pub fn lambda_factor() -> f64 {
        LAMBDA_FACTOR
    }
    pub fn max_lasso_iter() -> usize {
        MAX_LASSO_ITER
    }
    pub fn lasso_tol() -> f64 {
        LASSO_TOL
    }
    pub fn lambda_change_threshold() -> f64 {
        LAMBDA_CHANGE_THRESHOLD
    }
    pub fn signal_threshold_sigma() -> f64 {
        SIGNAL_THRESHOLD_SIGMA
    }
    pub fn merge_gap_points() -> usize {
        MERGE_GAP_POINTS
    }
    pub fn extension_points() -> usize {
        EXTENSION_POINTS
    }
    pub fn min_region_width() -> usize {
        MIN_REGION_WIDTH
    }
    pub fn noise_window_da() -> f64 {
        NOISE_WINDOW_DA
    }
    pub fn noise_step_da() -> f64 {
        NOISE_STEP_DA
    }
}

/// Post-acquisition centroiding for Thermo Stellar linear ion trap DIA data.
///
/// Deconvolutes overlapping signals within 1 m/z using non-negative LASSO
/// regression against Gaussian basis functions.
#[derive(Debug, Clone, Parser, Serialize, Deserialize)]
#[command(version, about, long_about = None)]
pub struct Config {
    /// Input profile-mode mzML file(s) or glob patterns (e.g. *.mzML)
    #[arg(short, long, required = true, num_args = 1..)]
    pub input: Vec<String>,

    /// Output directory (default: same directory as each input file).
    /// Output files are named <stem>.centrix.mzML.
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// YAML config file (CLI args override)
    #[arg(long)]
    #[serde(skip)]
    pub config: Option<PathBuf>,

    // --- Calibration overrides ---
    /// Override Gaussian σ for both MS1 and MS2 (Th). Overridden by
    /// --sigma-ms1 / --sigma-ms2 if also specified.
    #[arg(long)]
    #[serde(default)]
    pub sigma: Option<f64>,

    /// Override Gaussian σ for MS1 (Th). Auto-calibrated if not set.
    #[arg(long)]
    #[serde(default)]
    pub sigma_ms1: Option<f64>,

    /// Override Gaussian σ for MS2 (Th). Auto-calibrated if not set.
    #[arg(long)]
    #[serde(default)]
    pub sigma_ms2: Option<f64>,

    /// Override grid spacing (Da). Auto-detected from profile point density if not set.
    #[arg(long)]
    #[serde(default)]
    pub grid_spacing: Option<f64>,

    /// Override grid offset (fraction of grid_spacing, 0.0-1.0).
    #[arg(long)]
    #[serde(default)]
    pub grid_offset: Option<f64>,

    /// Number of spectra used for auto-calibration
    #[arg(long, default_value_t = defaults::N_CALIBRATION_SPECTRA)]
    #[serde(default = "defaults::n_calibration_spectra")]
    pub n_calibration_spectra: usize,

    // --- Algorithm tuning ---
    /// LASSO λ = lambda_factor × noise_sigma
    #[arg(long, default_value_t = defaults::LAMBDA_FACTOR)]
    #[serde(default = "defaults::lambda_factor")]
    pub lambda_factor: f64,

    /// Maximum LASSO coordinate descent iterations
    #[arg(long, default_value_t = defaults::MAX_LASSO_ITER)]
    #[serde(default = "defaults::max_lasso_iter")]
    pub max_lasso_iter: usize,

    /// LASSO convergence tolerance
    #[arg(long, default_value_t = defaults::LASSO_TOL)]
    #[serde(default = "defaults::lasso_tol")]
    pub lasso_tol: f64,

    /// Re-run LASSO in Pass 2 if λ changed by more than this fraction
    #[arg(long, default_value_t = defaults::LAMBDA_CHANGE_THRESHOLD)]
    #[serde(default = "defaults::lambda_change_threshold")]
    pub lambda_change_threshold: f64,

    /// Signal detection threshold in units of noise σ
    #[arg(long, default_value_t = defaults::SIGNAL_THRESHOLD_SIGMA)]
    #[serde(default = "defaults::signal_threshold_sigma")]
    pub signal_threshold_sigma: f64,

    /// Merge adjacent signal regions separated by ≤ this many points
    #[arg(long, default_value_t = defaults::MERGE_GAP_POINTS)]
    #[serde(default = "defaults::merge_gap_points")]
    pub merge_gap_points: usize,

    /// Extend signal regions by this many points on each side
    #[arg(long, default_value_t = defaults::EXTENSION_POINTS)]
    #[serde(default = "defaults::extension_points")]
    pub extension_points: usize,

    /// Minimum region width in data points to process
    #[arg(long, default_value_t = defaults::MIN_REGION_WIDTH)]
    #[serde(default = "defaults::min_region_width")]
    pub min_region_width: usize,

    /// Noise estimation window width (Da)
    #[arg(long, default_value_t = defaults::NOISE_WINDOW_DA)]
    #[serde(default = "defaults::noise_window_da")]
    pub noise_window_da: f64,

    /// Noise estimation window step (Da)
    #[arg(long, default_value_t = defaults::NOISE_STEP_DA)]
    #[serde(default = "defaults::noise_step_da")]
    pub noise_step_da: f64,

    /// Minimum separation (Da) between output centroids. Centroids closer than
    /// this are merged by intensity-weighted averaging. Default: σ (calibrated
    /// per MS level). Two Gaussians separated by less than σ are physically
    /// unresolvable at the Stellar's sampling density.
    #[arg(long)]
    #[serde(default)]
    pub min_centroid_separation: Option<f64>,

    // --- Performance ---
    /// Number of threads (0 = use all available)
    #[arg(long, default_value_t = 0)]
    #[serde(default)]
    pub threads: usize,

    // --- Output ---
    /// Write per-spectrum stats to this TSV file
    #[arg(long)]
    #[serde(default)]
    pub stats_output: Option<PathBuf>,

    /// Suppress progress output
    #[arg(long)]
    #[serde(default)]
    pub quiet: bool,

    /// Enable verbose debug logging
    #[arg(long)]
    #[serde(default)]
    pub verbose: bool,
}

impl Config {
    /// Load a YAML config file and merge with CLI args (CLI wins).
    pub fn load_and_merge(mut self) -> crate::Result<Self> {
        if let Some(ref path) = self.config.clone() {
            let file = std::fs::File::open(path).map_err(crate::CentrixError::Io)?;
            let file_config: Config = serde_yaml::from_reader(file)
                .map_err(|e| crate::CentrixError::Config(e.to_string()))?;

            // CLI options override file config for Option fields only if not set
            if self.sigma.is_none() {
                self.sigma = file_config.sigma;
            }
            if self.sigma_ms1.is_none() {
                self.sigma_ms1 = file_config.sigma_ms1;
            }
            if self.sigma_ms2.is_none() {
                self.sigma_ms2 = file_config.sigma_ms2;
            }
            if self.grid_spacing.is_none() {
                self.grid_spacing = file_config.grid_spacing;
            }
            if self.grid_offset.is_none() {
                self.grid_offset = file_config.grid_offset;
            }
            if self.stats_output.is_none() {
                self.stats_output = file_config.stats_output;
            }
        }
        Ok(self)
    }
}
