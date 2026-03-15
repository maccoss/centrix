# Centrix Algorithm

Centrix performs post-acquisition centroiding of Thermo Stellar linear ion trap
profile-mode data using **non-negative LASSO regression** against Gaussian basis
functions. It is designed to deconvolute two or more signals within 1 m/z of each
other — a case where the Stellar's onboard centroider collapses overlapping
signals into a single incorrect centroid.

## Target Instrument

Centrix is designed for the **Thermo Stellar** radial ejection linear ion trap.
Profile peaks from this instrument are well-modeled by Gaussians with full-width
at half-maximum (FWHM) of 0.5–2.0 Da, depending on scan rate.

| Scan Rate | FWHM (Da) | σ (Da) | Filter Code | Typical Use |
|-----------|-----------|--------|-------------|-------------|
| 33 kTh/s  | 0.5       | 0.212  | `n`         | High-res MS1 |
| 67 kTh/s  | 0.6       | 0.255  | `r`         | Standard MS1 |
| 125 kTh/s | 0.8       | 0.340  | `t`         | Standard DIA MS2 |
| 200 kTh/s | 2.0       | 0.849  | `u`         | Fast scan MS2 |

Profile grid spacing is exact and fixed by firmware:

| MS Level | Spacing          | Points/Th | Points/σ (σ≈0.25) | Typical Points/Scan |
|----------|------------------|-----------|--------------------|--------------------|
| MS1      | 1/15 ≈ 0.0667 Th | 15        | ~3.75              | 13,500             |
| MS2      | 1/8  = 0.125 Th  | 8         | ~2.00              | 10,400             |

## Pipeline Overview

The centroiding pipeline processes each spectrum through a **two-pass** algorithm
preceded by a calibration step.

```
  ┌──────────────────────────────────────────────────────────┐
  │                    CALIBRATION PASS                       │
  │  Load first N spectra → detect grid spacing → estimate σ │
  └──────────────┬───────────────────────────────────────────┘
                 │
  ┌──────────────▼───────────────────────────────────────────┐
  │                    PER-SPECTRUM PIPELINE                   │
  │                                                           │
  │  ┌─────────────────────┐                                  │
  │  │  1. Rough Noise Est  │  IQR-based σ from sorted I      │
  │  └──────────┬──────────┘                                  │
  │             ▼                                             │
  │  ┌─────────────────────┐                                  │
  │  │  2. Region Detection │  Threshold → segment → merge    │
  │  └──────────┬──────────┘                                  │
  │             ▼                                             │
  │  ┌─────────────────────────────────────────────┐          │
  │  │  3. Pass 1: LASSO all regions                │          │
  │  │     Build basis → solve → cache β, Aᵀy       │          │
  │  └──────────┬──────────────────────────────────┘          │
  │             ▼                                             │
  │  ┌─────────────────────┐                                  │
  │  │  4. Noise Refinement │  Residuals + gaps → smooth σ(m/z)│
  │  └──────────┬──────────┘                                  │
  │             ▼                                             │
  │  ┌─────────────────────────────────────────────┐          │
  │  │  5. Pass 2: Selective Re-fit                 │          │
  │  │     Only regions where λ changed > 20%       │          │
  │  │     Warm-started from Pass 1 β               │          │
  │  └──────────┬──────────────────────────────────┘          │
  │             ▼                                             │
  │  ┌─────────────────────┐                                  │
  │  │  6. Sort & Merge     │  Deduplicate close centroids    │
  │  └─────────────────────┘                                  │
  └───────────────────────────────────────────────────────────┘
```

## Step 1: Calibration

Before processing spectra, Centrix loads the first N spectra (default 50) to
auto-detect instrument parameters.

### Grid Spacing Detection

The median of consecutive m/z differences gives the profile grid spacing. This is
deterministic firmware output — the spacing is exactly 1/15 Th for MS1 and 1/8 Th
for MS2.

### Peak Width (σ) Estimation

σ is determined by priority cascade:

1. **User override** (`--sigma-ms1`, `--sigma-ms2`) — highest priority
2. **Filter string parsing** — reads the scan rate code (`n`, `r`, `t`, `u`) from
   Thermo filter strings (checks first 10 spectra for consistency) and maps to
   known σ values
3. **CWT calibration** — if filter strings are unavailable, a Mexican Hat wavelet
   scalogram is computed at 12 log-spaced scales. The scale maximizing CWT response
   for each peak is converted to σ using the factor 1/√5 (from Ricker wavelet
   theory). Outliers are filtered by 2×MAD before taking the mean.

## Step 2: Rough Noise Estimation

A fast order-statistics approach estimates the noise floor per spectrum:

- **Baseline**: 10th percentile of sorted intensities
- **Noise σ**: interquartile range (IQR) / 1.349 — a robust estimator of
  Gaussian σ  (equivalent to the normal distribution's IQR/σ ratio)
- **Floor**: max(10% of baseline, 1.0) prevents zero-noise pathologies

This is computationally cheap (one sort) and provides the initial λ for Pass 1.

## Step 3: Signal Region Detection

Signal regions are contiguous stretches of profile data above the noise threshold:

1. **Threshold**: intensity > baseline + `signal_threshold_sigma` × noise_sigma
   (default: 3σ above baseline)
2. **Segment**: find contiguous runs of above-threshold points
3. **Merge**: join segments separated by ≤ `merge_gap_points` (default 2) to
   handle valleys between peaks
4. **Extend**: expand each region by `extension_points` (default 3) on each side
   to capture tails
5. **Filter**: discard regions narrower than `min_region_width` (default 3 points)

All detected regions are processed by LASSO — there is no fast-path shortcut.
Even narrow regions that appear to contain a single peak are solved by LASSO,
since two overlapping signals <1 Da apart (the primary use case) can appear
as a single merged peak at the profile level.

## Step 4: Pass 1 — Centroiding

### LASSO

Each signal region's profile data is modeled as a sum of Gaussians:

```
y(m/z) = Σⱼ βⱼ · exp(-(m/z - gⱼ)² / (2σ²)) + ε
```

where `gⱼ` are grid positions spaced at the profile grid spacing (not the basis
grid — they align to the data).

This is expressed in matrix form as **y = Aβ + ε**, where A is the design matrix
of Gaussian basis functions.

#### Design Matrix Construction

The design matrix A has dimensions (n_data × n_basis):
```
A[i,j] = exp(-(mz[i] - grid[j])² / (2σ²))
```

#### The Gram Matrix Is Toeplitz

Because the grid is uniform, the Gram matrix G = AᵀA has **Toeplitz structure**:
entry (i,j) depends only on |i-j|. Only the first row needs to be stored:

```
gram_row[k] = (σ√π / h) · exp(-(k·h)² / (4σ²))
```

where h is the grid spacing. This enables O(bandwidth) per-variable updates in the
LASSO solver instead of O(n²), and the entire row (~6–15 entries) fits in L1 cache.

#### Non-Negative LASSO Solver

The solver minimizes:

```
½‖y - Aβ‖² + λ‖β‖₁   subject to β ≥ 0
```

using **coordinate descent** with active-set acceleration.

The update for variable j is:
```
ρⱼ = (Aᵀy)[j] - Σ_{k≠j} G[j,k]·βₖ

βⱼ = max(0, ρⱼ - λ) / G[j,j]
```

The ρ computation exploits Toeplitz structure: `G[j,k] = gram_row[|j-k|]`, and
only loops over the non-zero bandwidth of the gram_row.

**Active-set acceleration**: only variables with β > 0 or with gradient suggesting
potential activation are updated each iteration. Every 10 iterations, a full sweep
checks all variables for KKT violations. The active set is typically 2–5 variables,
keeping per-iteration cost very low.

**BLAS usage**: `Aᵀy` is computed via ndarray's `.t().dot()` (dispatched to BLAS
dgemv). The coordinate descent inner loop uses gram_row scalars directly — BLAS
overhead would exceed computation cost for the small active set.

#### Regularization Parameter λ

λ controls the trade-off between data fidelity and sparsity:

```
λ = lambda_factor × noise_sigma
```

Higher λ → fewer centroids (more peaks zeroed out). Lower λ → more centroids
(weaker peaks survive). The default `lambda_factor` of 3.0 is conservative; values
of 1.5–2.0 produce more centroids closer to the Thermo centroider count.

#### Sub-Grid Refinement

After LASSO converges, each non-zero coefficient with two non-zero neighbors
undergoes sub-grid refinement:

1. Fit a log-parabola through ln(β_{j-1}), ln(β_j), ln(β_{j+1})
2. Peak center = refined m/z position (clamped to ±0.5 grid spacings)

This recovers sub-grid precision from the discrete basis coefficients.

#### Intensity as Integrated Gaussian Area

The final reported intensity for each centroid is the **discrete sum** of the
fitted Gaussian across profile data points, not the raw LASSO coefficient:

```
intensity = β × σ × √(2π) / h
```

where h is the profile grid spacing. This is equivalent to summing the fitted
Gaussian values at each data point, matching the convention used by the Thermo
onboard centroider.

### Pass 1 Output

After Pass 1, the system has:
- Centroids from all regions (preliminary — may be refined in Pass 2)
- Cached β, Aᵀy, and λ per region
- LASSO residuals (y - Aβ) for noise refinement

## Step 5: Noise Refinement

The rough noise estimate from Step 2 is refined using Pass 1 results:

1. Sweep overlapping windows of `noise_window_da` (default 20 Da) stepped by
   `noise_step_da` (default 5 Da) across the m/z range
2. In each window, collect:
   - Raw intensities from **gap points** (not in any signal region)
   - **LASSO residuals** from fitted regions (y - Aβ)
3. Compute RMS of all collected values → noise σ for that window center
4. Apply 3-point median smoothing across windows
5. Linear interpolation between window centers gives a continuous noise model σ(m/z)

This produces a spatially-varying noise estimate that accounts for the actual fit
quality from Pass 1, not just the raw intensity distribution.

## Step 6: Pass 2 — Selective Re-fit

For each LASSO region from Pass 1, the refined noise model gives a new λ. If the
fractional change exceeds `lambda_change_threshold` (default 20%):

```
|λ_refined - λ_rough| / λ_rough > 0.20
```

the LASSO is re-run for that region, **warm-started** from the Pass 1 β. Warm
starting typically converges in <10 iterations (vs ~50–200 cold-start).

The Aᵀy vector is cached from Pass 1 and reused directly (the grid doesn't change
between passes, only λ changes). This avoids rebuilding the design matrix A and
recomputing the matrix-vector product. Pass 2 centroids replace Pass 1 centroids
for the affected m/z range.

Regions where λ didn't change significantly retain their Pass 1 centroids
unmodified.

## Step 7: Sort and Merge

All centroids (from Pass 1 regions not refit and Pass 2 refit regions) are
collected, sorted by m/z, and merged.

### Minimum separation (σ-based merging)

Centroids closer than a minimum separation threshold are merged via
intensity-weighted m/z averaging (summing their intensities). The default
threshold is **σ** — the calibrated Gaussian peak width for the MS level.

This is necessary because the LASSO basis grid is placed at the ADC data point
positions (0.125 Th for MS2, ~0.067 Th for MS1). When a true peak center falls
between two grid points, the LASSO solver may assign non-zero coefficients to
both adjacent grid positions, producing a spurious "close doublet" — two
centroids separated by one grid spacing that actually represent a single ion.

Two Gaussians with the same σ separated by less than σ are physically
indistinguishable at the Stellar's sampling density (~2 data points per σ for
MS2). The intensity-weighted merge naturally recovers the correct centroid
position:

$$m/z_{\text{merged}} = \frac{m/z_1 \cdot I_1 + m/z_2 \cdot I_2}{I_1 + I_2}, \quad I_{\text{merged}} = I_1 + I_2$$

The threshold is configurable via `--min-centroid-separation <DA>` for
non-standard instruments or analytes. Setting it to 0 disables merging entirely.

## Parallelism

Centrix uses **batched rayon parallelism**:

1. Spectra are read sequentially in batches of 256
2. Each batch is centroided in parallel via `rayon::par_iter()` — each spectrum
   is independent (no shared mutable state)
3. Results are written sequentially in file order (mzML requires ordered spectra)

The `BasisPrecompute` structures (one per MS level) are read-only and shared
across all threads. Each thread operates on its own spectrum data with no
synchronization overhead.

### BLAS Threading (Critical)

Centrix uses BLAS only for the Aᵀy computation (`a.t().dot(y)`, dispatched as
`dgemv`). The design matrices are tiny: 8×8 for MS2 and up to ~22×22 for MS1.

OpenBLAS (and MKL) spawn internal thread pools sized to the number of CPU cores
by default. For these tiny matrices, the thread synchronization overhead is
catastrophic — on a 10-core machine, `sys` time inflated from 6 seconds to 150+
minutes (>1500× overhead), roughly doubling wall-clock time.

Centrix forces BLAS to single-threaded mode via FFI calls at startup
(`openblas_set_num_threads(1)` or `mkl_set_num_threads(1)`). All parallelism is
handled by Rayon at the spectrum level, where the work units are large enough to
amortize scheduling overhead. BLAS must never compete with its own internal
thread pool.
