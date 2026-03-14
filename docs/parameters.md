# Centrix Parameters Reference

All parameters can be set via CLI flags or a YAML config file. CLI flags take
priority over config file values. Optional parameters (`sigma_ms1`, `sigma_ms2`,
etc.) use auto-detection when not specified.

## Usage

```bash
# CLI flags
centrix run -i input.mzML --lambda-factor 1.5

# Multiple files / glob patterns
centrix run -i '*.mzML' -o results/ --lambda-factor 1.5

# YAML config file (CLI args override file values)
centrix run -i input.mzML --config params.yaml
```

Example YAML config:
```yaml
lambda_factor: 1.5
signal_threshold_sigma: 2.0
noise_window_da: 20.0
```

---

## I/O Parameters

| Flag | YAML Key | Default | Description |
|------|----------|---------|-------------|
| `-i, --input` | `input` | *required* | Input profile-mode mzML file(s) or glob patterns. Accepts multiple values. |
| `-o, --output` | `output` | same as input | Output directory. Output files are named `<stem>.centrix.mzML`. If omitted, output is placed next to each input file. |
| `--config` | — | none | Path to YAML config file |
| `--stats-output` | `stats_output` | none | Path for per-spectrum TSV statistics |

---

## Calibration Parameters

These control how Centrix determines the instrument's peak shape (σ) and sampling
grid.

| Flag | YAML Key | Default | Description |
|------|----------|---------|-------------|
| `--sigma-ms1` | `sigma_ms1` | auto | Gaussian σ for MS1 in Da. Overrides auto-calibration. |
| `--sigma-ms2` | `sigma_ms2` | auto | Gaussian σ for MS2 in Da. Overrides auto-calibration. |
| `--grid-spacing` | `grid_spacing` | auto | Profile grid spacing in Da. Auto-detected from median point spacing. |
| `--grid-offset` | `grid_offset` | auto | Grid phase offset as fraction of grid_spacing (0.0–1.0). |
| `--n-calibration-spectra` | `n_calibration_spectra` | 50 | Number of spectra to load for auto-calibration. |

### Auto-calibration priority

When σ is not specified by the user, Centrix attempts (in order):
1. Parse scan rate from Thermo filter strings → known σ values
2. CWT-based σ estimation from profile peak shapes

Grid spacing is always detected from the data unless overridden.

---

## Algorithm Tuning Parameters

### Detection Sensitivity

These parameters control which signals are detected as peaks and how much
regularization is applied. **Lowering these values produces more centroids.**

| Flag | YAML Key | Default | Range | Description |
|------|----------|---------|-------|-------------|
| `--lambda-factor` | `lambda_factor` | **3.0** | 0.5–10.0 | LASSO penalty: λ = factor × noise_σ. **Primary knob for centroid count.** Lower → more peaks survive. |
| `--signal-threshold-sigma` | `signal_threshold_sigma` | **3.0** | 1.0–10.0 | Signal detection threshold in noise σ units above baseline. Lower → fainter peaks detected. |

**Recommended starting points for more centroids:**
```bash
# Moderate increase in sensitivity
--lambda-factor 1.5

# More aggressive (closer to Thermo centroider count)
--lambda-factor 1.5 --signal-threshold-sigma 2.0

# Very aggressive (may include noise peaks)
--lambda-factor 1.0 --signal-threshold-sigma 1.5
```

### Region Detection

These control how signal regions are identified and classified before centroiding.

| Flag | YAML Key | Default | Description |
|------|----------|---------|-------------|
| `--merge-gap-points` | `merge_gap_points` | 2 | Data points. Merge adjacent above-threshold segments separated by ≤ this many points. |
| `--extension-points` | `extension_points` | 3 | Data points. Extend each signal region by this many points on each side to capture tails. |
| `--min-region-width` | `min_region_width` | 3 | Data points. Discard signal regions narrower than this. |

### LASSO Solver

These control the convergence behavior of the coordinate descent solver. Rarely
need adjustment.

| Flag | YAML Key | Default | Description |
|------|----------|---------|-------------|
| `--max-lasso-iter` | `max_lasso_iter` | 1000 | Maximum iterations per LASSO solve. |
| `--lasso-tol` | `lasso_tol` | 1e-6 | Convergence tolerance: max |β_new − β_old| < tol. |
| `--lambda-change-threshold` | `lambda_change_threshold` | 0.20 | Fractional λ change (after noise refinement) that triggers a Pass 2 re-solve. |

### Noise Estimation

These control the sliding-window noise refinement after Pass 1.

| Flag | YAML Key | Default | Description |
|------|----------|---------|-------------|
| `--noise-window-da` | `noise_window_da` | 20.0 | Width of noise estimation window in Da. |
| `--noise-step-da` | `noise_step_da` | 5.0 | Step size between noise estimation windows in Da. |

---

## Runtime Parameters

| Flag | YAML Key | Default | Description |
|------|----------|---------|-------------|
| `--threads` | `threads` | 0 (all) | Number of threads for parallel centroiding. 0 = use all available cores. |
| `--quiet` | `quiet` | false | Suppress progress bar output. |
| `--verbose` | `verbose` | false | Enable debug-level logging. |

---

## Parameter Interactions

### λ and signal_threshold_sigma

These are the two main sensitivity controls, but they operate at different stages:

- `signal_threshold_sigma` acts during **region detection** — it determines which
  stretches of the profile spectrum are considered signal vs. noise. Lowering this
  adds more regions (potentially very faint ones) for centroiding to process.

- `lambda_factor` acts during **LASSO solving** — it controls how aggressively
  small coefficients are zeroed out within detected regions. Lowering this allows
  weaker peaks to survive the sparsity penalty.

Lowering `lambda_factor` alone is usually the most effective way to increase
centroid count, since more regions can produce more centroids from their existing
signals. Lowering `signal_threshold_sigma` additionally captures fainter regions
near the noise floor.

### LASSO for all regions

All signal regions are processed by LASSO regardless of width. There is no
fast-path shortcut — even narrow regions that appear to contain a single peak
are solved by LASSO, since the solver naturally converges to a single non-zero
coefficient in one or two iterations for genuinely isolated peaks.

### lambda_change_threshold

Controls Pass 2 behavior. At 0.20 (default), only regions where the refined noise
model changed λ by >20% are re-solved. Lowering this (e.g., to 0.05) forces more
regions through Pass 2, potentially improving centroid positions in regions where
noise was poorly estimated in Pass 1. The cost is additional LASSO solves, but
warm-starting keeps this fast.

---

## Known Thermo Stellar Scan Rates

These are the known peak widths for the Stellar's four scan rate modes. The
filter string parsing uses the one-letter code after "NSI " in the Thermo filter
string.

| Rate (kTh/s) | FWHM (Da) | σ (Da) | Filter Code |
|-------------|-----------|--------|-------------|
| 33          | 0.5       | 0.212  | `n`         |
| 67          | 0.6       | 0.255  | `r`         |
| 125         | 0.8       | 0.340  | `t`         |
| 200         | 2.0       | 0.849  | `u`         |
