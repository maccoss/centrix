# Ion Trap Profile Centroiding via Sparse Signal Decomposition

## Software Specification — Phase 1

**Project:** Post-acquisition centroiding for linear ion trap profile data
**Authors:** MacCoss Lab, Department of Genome Sciences, University of Washington
**Target Instrument:** Thermo Stellar (radial ejection linear ion trap)
**Language:** Rust
**Related Projects:** Osprey (mzML reading), MARS (mzML writing, mass recalibration)

---

## 1. Problem Statement

The Thermo Stellar mass spectrometer performs onboard centroiding during data acquisition. When two or more ions are not resolved in the m/z dimension — common at unit resolution where isotope envelopes from different peptides routinely overlap — the instrument reports a single centroid at the intensity-weighted center of mass of the composite signal. This centroid is incorrect for all contributing ions, degrading mass accuracy and reducing peptide identifications in downstream analysis.

This tool replaces the instrument's centroiding with a sparse signal decomposition approach. Given profile-mode spectral data, it decomposes the observed signal into a sparse set of individual ion contributions using non-negative LASSO regression against a precomputed dictionary of Gaussian basis functions. Each non-zero coefficient in the solution corresponds to a centroid in the output.

### Why This Matters for DIA on the Stellar

In DIA experiments, MS2 spectra are inherently chimeric — fragments from multiple co-isolated precursors contribute signal to the same spectrum. At unit resolution (~0.5-0.7 FWHM), product ions from different peptides frequently overlap within a single nominal mass unit. Correct deconvolution of these composite peaks is essential for both peptide-centric search tools (Osprey) and targeted quantification (Skyline/PRISM).

---

## 2. Algorithm Overview

### Core Formulation

The observed profile spectrum in a signal region is modeled as a linear combination of Gaussian basis functions:

```
y = Aβ + ε

where:
  y    = observed profile intensities (n_points × 1)
  A    = design matrix of Gaussian basis functions (n_points × n_basis)
  β    = coefficients to solve for (n_basis × 1), β ≥ 0
  ε    = noise
```

Each column of **A** is a Gaussian centered at a grid point, evaluated at the profile data's m/z sampling positions. Gaussian basis functions are the correct choice for the linear ion trap: unlike Orbitrap peaks (which have Lorentzian tails from FT truncation of a time-domain transient), ion trap resonance ejection produces peaks whose temporal width is governed by the number of RF cycles at the stability boundary, yielding a near-Gaussian profile. This also means the closed-form Gram matrix entries (see §3.2) are exact rather than approximate.

```
A[i,j] = exp(-(mz_data[i] - mz_grid[j])² / (2σ²))
```

We solve for β using non-negative LASSO:

```
minimize  ½||y - Aβ||² + λ||β||₁
subject to  β ≥ 0
```

The L1 penalty enforces sparsity: most coefficients are driven to exactly zero, and the non-zero entries identify discrete ion signals. Each non-zero β_j produces an output centroid at `mz_grid[j]` with intensity proportional to β_j.

### Pipeline

```
Profile mzML (input)
  │
  ├─ Step 1: Auto-calibrate σ (per MS level) and grid spacing from first N spectra
  │           → σ from isolated peak fitting, grid spacing from profile point density
  │
  ├─ Step 2: Evaluate grid offsets via multi-grid LASSO on representative regions
  │           → Select offset that minimizes total residual (per MS level)
  │
  ├─ Step 3: Precompute Gram matrix template (AᵀA) and basis template
  │           → Toeplitz structure, one per distinct σ value
  │
  ├─ Step 4: For each spectrum (two-pass iterative):
  │   │
  │   ├─ Pass 1 (rough):
  │   │   ├─ 4a. Rough noise estimate (10th percentile or running minimum)
  │   │   ├─ 4b. Signal region detection using rough threshold
  │   │   ├─ 4c. Classify regions: single peak vs. potential composite
  │   │   ├─ 4d. Single peaks → 3-point Gaussian fit (fast path)
  │   │   ├─ 4e. Composites → non-negative LASSO decomposition
  │   │   └─ 4f. Compute residuals (y - Aβ) across all fitted regions
  │   │
  │   ├─ Noise refinement:
  │   │   ├─ 4g. Estimate true noise width from residual σ across fitted regions
  │   │   ├─ 4h. Estimate true baseline from residual mean + gap intensities
  │   │   └─ 4i. Recalculate λ from refined noise estimate
  │   │
  │   ├─ Pass 2 (refined, only where needed):
  │   │   ├─ 4j. Re-detect signal regions with refined threshold
  │   │   ├─ 4k. Re-run LASSO only on new/changed regions (warm-start)
  │   │   └─ 4l. Collect final centroids: (m/z, intensity) pairs
  │   │
  │
  └─ Step 5: Write centroided mzML (passthrough writer)

Centroided mzML (output)
```

---

## 3. Detailed Algorithm Specification

### 3.1 Automatic σ Calibration

The Gaussian width (σ) is constant across the m/z range for a linear ion trap — the peak width is set by the number of RF cycles an ion experiences during resonance ejection, which depends on the scan rate but not on m/z. Different experiments with different ejection rates will have different σ values.

**Procedure:**

1. Read the first `N_CALIBRATION_SPECTRA` spectra (default: 50) from the mzML file.
2. **Stratify by MS level.** MS1 and MS2 scans may use different scan rates on the Stellar, producing different peak widths. Collect isolated peaks separately for MS1 and MS2 spectra. If the instrument uses multiple MS2 scan rates (e.g., different rates for narrow vs. wide isolation windows), further stratify by scan rate if that information is available in the filter string or CV terms.
3. For each spectrum, compute a rough noise estimate using the minimum-statistics approach (§3.3.1). Identify **isolated peaks**: local maxima with intensity > `3 × rough_noise` above the 10th percentile baseline, with no other such signal within ±1.0 Da.
4. For each isolated peak, fit a single Gaussian to the profile data within ±0.5 Da of the apex using least-squares (3 parameters: amplitude, center, σ).
5. Collect all fitted σ values per stratum. Reject outliers beyond 2× MAD from the median within each stratum.
6. Report `σ_calibrated` per MS level (and per scan rate if stratified further).
7. **Validation**: Within each stratum, plot σ vs. m/z to verify the constant-σ assumption. If a statistically significant trend exists (Spearman |ρ| > 0.3, p < 0.01), fit a linear model σ(m/z) and use that instead. Log a warning.
8. **Compare strata**: Log the calibrated σ for MS1 vs. MS2. If they differ by more than 10%, use separate σ values and precompute separate Gram matrix templates for each. If they are within 10%, use a single σ for simplicity.
9. Store σ per MS level in the output mzML metadata as `userParam` entries.

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_calibration_spectra` | 50 | Number of spectra to use for σ calibration |
| `isolation_radius_da` | 1.0 | Minimum distance to nearest signal for a peak to be "isolated" |
| `min_isolated_peaks` | 20 | Minimum number of isolated peaks required for reliable σ estimate |

**Fallback:** If fewer than `min_isolated_peaks` are found, use a default σ based on instrument settings. The user can also specify σ directly via CLI to bypass auto-calibration.

---

### 3.2 Precomputed Gram Matrix Template

Because the basis functions are uniformly-spaced identical Gaussians at fixed σ, the Gram matrix **G** = **AᵀA** has **Toeplitz structure**: element G[i,j] depends only on |i-j|.

For two Gaussians separated by `Δ = |i-j| × grid_spacing`:

```
G[i,j] = σ√π × exp(-Δ² / (4σ²))
```

(This is the inner product of two Gaussians of equal width, which has a closed-form solution.)

**Precomputation:**

1. Compute the unique Toeplitz row: `g[k] = σ√π × exp(-(k × grid_spacing)² / (4σ²))` for k = 0, 1, ..., `K_max` where `K_max` is the number of grid points beyond which `g[k]` < machine epsilon. The `grid_spacing` is auto-detected from the profile data (§3.7).
2. Store this vector once. For any signal region of size `n_basis`, the local Gram matrix is constructed by indexing into this precomputed row.

**Precomputed Aᵀ template:** Similarly, the columns of **A** are translations of the same Gaussian. A single "basis template" vector is computed once at the calibrated σ and evaluated at sub-grid resolution. For each signal region, the actual **A** matrix is constructed by shifting and sampling this template.

**BLAS integration:** The **Aᵀy** product (the only per-spectrum, per-region computation that depends on the data) is computed using BLAS `dgemv` (matrix-vector multiply). For coordinate descent iterations, individual column dot products are extracted from the precomputed Gram matrix — no BLAS needed for the inner loop since it operates on cached scalars.

---

### 3.3 Iterative Baseline and Noise Estimation

**Problem:** Accurate noise estimation is essential for setting the LASSO regularization parameter λ and for distinguishing real centroids from artifacts. Traditional approaches fail for dense DIA MS2 spectra on a unit-resolution instrument like the Stellar.

At FTMS resolution (10⁵), peaks are ~0.01 Da wide, so even in a crowded spectrum, >75% of data points sit on the baseline. The THRASH algorithm (Horn et al., JASMS 2000) exploits this: the most frequent intensity value in a local window is the baseline. But at unit resolution on the Stellar, peaks are ~0.5-0.7 Da wide at the base. With ~50 co-isolated peptides generating ~1000 fragments across a 200-1500 m/z range, signal regions can cover the majority of the m/z space. The baseline is no longer the most frequent intensity; the THRASH assumption is inverted.

Valley-based approaches (fitting a curve through local minima between peaks) also fail because adjacent peak tails overlap, elevating valley intensities well above the true baseline.

**Solution: let the LASSO residuals measure the noise.** After the LASSO decomposes a signal region into Gaussian basis functions, the residual `r = y - Aβ` should contain only noise plus model mismatch. The standard deviation of those residuals is a direct measurement of the local noise, and the mean of the residuals estimates the baseline offset — regardless of peak density.

#### 3.3.1 Pass 0: Rough Initial Noise Estimate

The LASSO needs an initial λ to get started. This estimate does not need to be accurate — it only needs to be within a factor of ~2-3 of the true noise so that dominant peaks are correctly identified in the first pass.

**Minimum-statistics approach:**

1. Slide a window of width `rough_noise_window_mz` (default: 10.0 Da) across the profile spectrum.
2. In each window, compute the minimum intensity value.
3. Track the running minimum across windows, then apply a bias correction factor: the expected minimum of N samples from a Gaussian is below the mean by ~√(2 ln N) × σ, so `rough_noise_estimate ≈ global_minimum × correction_factor`.
4. As a secondary estimate, compute the 10th percentile of all intensity values in the spectrum. Take the larger of the two estimates to avoid underestimating noise in spectra with very low baselines.

**Rough λ:**

```
λ_rough = lambda_factor × rough_noise_estimate × column_norm
```

where `lambda_factor` is the user-configurable detection threshold (default: 3.0). This λ will be too high for some faint signals and too low for some noise — both are corrected in pass 2.

#### 3.3.2 Pass 1: Initial LASSO Decomposition

Run the full centroiding pipeline (signal region detection, fast-path/LASSO classification, fitting) using `λ_rough`. This captures the dominant peaks reliably. Some faint signals near the threshold may be missed, and a few spurious centroids may appear in noisy regions — both are acceptable for the purpose of noise estimation.

**Signal region detection for pass 1:**

1. Subtract the rough baseline (10th percentile of intensities) from the profile data.
2. Mark all data points with baseline-subtracted intensity > `signal_threshold × rough_noise_estimate` (default: `signal_threshold = 2.0` — deliberately permissive to include marginal regions that will be evaluated).
3. Merge marks separated by fewer than `merge_gap` data points (default: 2) into contiguous signal regions.
4. Extend each region by `extension_points` (default: 5) in each direction to capture peak tails.
5. Discard regions narrower than `min_region_width` data points (default: 3).

#### 3.3.3 Noise Refinement from LASSO Residuals

After pass 1 fitting, compute the noise model from two complementary sources:

**Source 1: Residuals within fitted regions.**

For each signal region where LASSO was applied (not fast-path regions, which have too few points):

1. Compute the residual vector: `r = y - Aβ` (observed profile minus fitted model).
2. Compute the local residual standard deviation: `σ_residual = std(r)`.
3. Compute the local residual mean: `μ_residual = mean(r)`. A non-zero mean indicates a baseline offset — the data sits above or below the model's implicit zero baseline.

**Source 2: Gap intensities between signal regions.**

1. Collect all data points that fall outside any signal region (the "gaps"). In very dense spectra, there may be few such points; in sparse spectra, they dominate.
2. Compute the mean and standard deviation of gap intensities.
3. The gap mean is a direct estimate of the baseline, and the gap standard deviation is a direct estimate of the noise.

**Combining the estimates:**

1. If sufficient gap points exist (> 100 points): use the gap-based estimates as primary, with residual-based estimates as a consistency check. If they disagree by more than a factor of 2, log a warning.
2. If few gap points exist (< 100): use the residual-based estimates as primary. The residual mean provides the baseline offset; the residual σ provides the noise width.
3. Build a smooth noise model across the m/z range by computing estimates in overlapping windows (default: 20 Da wide, 5 Da step) and fitting a LOESS curve or low-order polynomial through them. This captures any m/z-dependent variation in baseline or noise.

**Refined λ:**

```
λ_refined = lambda_factor × noise_σ_refined × column_norm
```

where `noise_σ_refined` is the local noise standard deviation from the refined model, evaluated at the center of each signal region.

#### 3.3.4 Pass 2: Refined LASSO (Only Where Needed)

Re-run centroiding only on regions where the refined noise model differs significantly from the rough estimate:

1. **New signal regions:** Re-detect signal regions using the refined baseline and noise estimates. Any new regions (below the rough threshold but above the refined threshold) are processed for the first time.
2. **Removed regions:** Regions where pass 1 produced centroids but the refined λ is higher (rough estimate was too low) are re-evaluated. Centroids whose coefficient β_j < λ_refined / G[j,j] are removed.
3. **Unchanged regions:** Regions where λ_rough and λ_refined differ by less than 20% keep their pass 1 results unchanged.
4. **Changed regions:** Regions where λ_rough and λ_refined differ by more than 20% are re-run with the refined λ, warm-started from the pass 1 solution for fast convergence.

**Warm-starting:** The pass 1 LASSO solution is stored per-region. For re-run regions, initialize the coordinate descent with the pass 1 β vector. Since most coefficients won't change, the warm-started solver converges in a few iterations rather than starting from scratch.

**Convergence:** A third pass is almost never needed. The noise estimate from pass 1 residuals is already quite accurate because the dominant peaks (which are insensitive to λ) are correctly fitted. The refinement primarily affects faint signals near the detection threshold.

#### 3.3.5 Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rough_noise_window_mz` | 10.0 | Window width for minimum-statistics initial noise estimate |
| `signal_threshold` | 2.0 | Initial signal detection threshold in rough noise units (pass 1) |
| `lambda_factor` | 3.0 | Detection threshold in noise σ units (both passes) |
| `noise_window_mz` | 20.0 | Window width for refined noise model estimation |
| `noise_step_mz` | 5.0 | Step size for refined noise model windows |
| `min_gap_points` | 100 | Minimum gap data points for gap-based noise estimation |
| `lambda_change_threshold` | 0.20 | Fractional change in λ that triggers pass 2 re-fitting |
| `merge_gap` | 2 | Maximum gap (data points) to merge adjacent signal marks |
| `extension_points` | 5 | Data points to extend signal regions on each side |
| `min_region_width` | 3 | Minimum signal region width (data points) |

**Output per spectrum:** A `NoiseProfile` containing the baseline and noise σ as smooth functions of m/z, plus per-region λ values.

**Output:** A list of `SignalRegion { start_idx, end_idx, mz_start, mz_end, max_intensity }` for each pass.

---

### 3.4 Single-Peak Fast Path

**Purpose:** For signal regions containing a single well-separated peak, a 3-point Gaussian fit is faster and equally accurate as LASSO. This handles the majority (~70-80%) of signal regions.

**Classification rule:** A signal region is classified as a single peak if its width at half-maximum (measured in m/z) is less than `single_peak_threshold × σ_calibrated` (default: 2.5).

**3-point Gaussian fit (identical to the Orbitrap centroider):**

1. Find the apex (maximum intensity point) in the region.
2. Take the apex and its two neighbors (3 points total).
3. Fit ln(intensity) = a × mz² + b × mz + c (parabola in log space = Gaussian).
4. Centroid m/z = -b / (2a).
5. Centroid intensity = exp(c - b²/(4a)).

**Validation:** If the fitted σ from the 3-point fit deviates more than 50% from σ_calibrated, reclassify as composite and send to LASSO path.

---

### 3.5 Non-Negative LASSO Decomposition

**Purpose:** Decompose composite signal regions into individual ion contributions.

**Setup for each signal region:**

1. Define the basis grid: m/z positions from `mz_start` to `mz_end` at `grid_spacing`, aligned to `grid_offset` (see §3.7). Specifically, grid positions are `grid_offset + j × grid_spacing` for integer j values that fall within [mz_start, mz_end]. The grid spacing is set to match the profile data point spacing, and the offset is optimized for peptide mass defect alignment. Placing basis functions more densely than the data sampling creates an underdetermined system where the LASSO cannot discriminate between adjacent basis positions.
2. Construct the local **A** matrix from the precomputed basis template, evaluated at the region's data m/z values. Size: `n_data_points × n_basis`.
3. Compute **Aᵀy** via BLAS `dgemv`.
4. Extract the local Gram matrix from the precomputed Toeplitz row.

**Non-negative LASSO coordinate descent:**

```
Initialize: β = 0 (all zeros)
Precompute: Aᵀy, G = AᵀA (from Toeplitz template)

Repeat until convergence or max_iter:
    For j = 1 to n_basis:
        # Compute the partial residual gradient
        rho_j = Aᵀy[j] - Σ_{k≠j} G[j,k] × β[k]
        
        # Soft-thresholding with non-negativity
        β[j] = max(0, rho_j - λ) / G[j,j]
    
    Check convergence: max|β_new - β_old| < tolerance

Report non-zero entries of β as centroids.
```

**Active set acceleration:** After the first full pass, only iterate over variables with β > 0 or gradient > λ. Variables at zero with gradient ≤ λ cannot become active and are skipped.

**Regularization parameter λ:**

λ is set by the iterative noise estimation procedure (§3.3). In pass 1, a rough estimate is used:

```
λ_rough = lambda_factor × rough_noise_estimate × column_norm
```

In pass 2, the refined estimate from LASSO residuals replaces this:

```
λ_refined = lambda_factor × noise_σ_refined(mz) × column_norm
```

where `noise_σ_refined(mz)` is the local noise standard deviation from the smooth noise model, evaluated at the center of the signal region. The `lambda_factor` (default: 3.0) controls the detection threshold and has an interpretable meaning: `lambda_factor ≈ 3` corresponds to a ~3σ detection limit. The column norms of **A** are precomputed (they're all equal for identical Gaussians, simplifying `column_norm` to a single scalar).

**Sub-grid centroid refinement:** After LASSO identifies non-zero coefficients, refine each centroid's m/z position by fitting a parabola to ln(β) at the non-zero grid point and its neighbors (if they are also non-zero) or by performing a local Newton step. This provides sub-grid precision beyond the 0.02 Da grid spacing.

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `grid_spacing_da` | auto | Spacing between basis function centers; auto-detected from profile data point spacing (see §3.7). Can be overridden via CLI. |
| `max_lasso_iter` | 500 | Maximum coordinate descent iterations |
| `convergence_tol` | 1e-6 | Convergence tolerance on max |Δβ| |
| `single_peak_threshold` | 2.5 | Width/σ ratio below which single-peak fast path is used |

---

### 3.6 Window Overlap and Boundary Handling

The full m/z range is not processed as one LASSO problem. Instead, signal regions are processed independently. For adjacent or very wide signal regions:

1. If two signal regions are separated by fewer than `3σ` in m/z, merge them into a single region before LASSO.
2. For very wide signal regions (> 5 Da), split into overlapping windows of 5 Da with 0.5 Da overlap. Reconcile duplicate centroids in overlap zones by keeping the one with higher coefficient (i.e., better fit in its window).

---

### 3.7 Grid Spacing and Offset Auto-Detection

#### Grid Spacing

The LASSO basis function grid spacing must not be finer than the profile data point spacing. If basis functions are placed more densely than the ADC sampling, multiple basis functions fall between adjacent data points, creating an underdetermined system where the solver cannot distinguish which basis position the signal belongs to. This produces arbitrary coefficient assignments among neighboring grid points rather than a clean sparse solution.

**Spacing procedure (during calibration, alongside σ estimation):**

1. From the first `N_CALIBRATION_SPECTRA` spectra, compute the median spacing between consecutive profile data points: `Δmz_data = median(mz[i+1] - mz[i])` for all consecutive pairs.
2. Verify that the spacing is approximately uniform. On the Stellar, the RF ramp produces a monotonic mapping from time to m/z, but the ADC samples at uniform time intervals, which maps to approximately uniform m/z spacing (unlike TOF instruments where spacing grows with √m/z). If spacing varies by more than 20% across the mass range, log a warning and use the minimum spacing observed.
3. Set `grid_spacing = Δmz_data`. This places exactly one basis function per data point, which is the natural choice — each basis function is supported by the data point nearest its center plus its neighbors within the Gaussian's width.
4. If the user provides `--grid-spacing` via CLI, validate that it is ≥ `Δmz_data` and warn if it is finer.
5. Log the auto-detected grid spacing and number of data points per σ (= σ / grid_spacing). This latter number indicates how well the peak shape is sampled — if it's less than ~3, the profile data may be too coarsely sampled for reliable Gaussian fitting, and the tool should warn.

**Impact on the Toeplitz Gram matrix:** The grid spacing determines the separation `Δ` in the Gram matrix formula `g[k] = σ√π × exp(-(k × grid_spacing)² / (4σ²))`. A coarser grid spacing means the off-diagonal elements of the Gram matrix decay faster (adjacent basis functions overlap less), which actually makes the LASSO problem better conditioned and faster to solve.

#### Grid Offset

The grid origin matters as much as the spacing. If the grid is placed at m/z values 500.00, 500.02, 500.04... but real ion signals cluster at 500.01, 500.03, 500.05 due to peptide mass defect patterns, every signal falls at the worst-case midpoint between two basis functions. The LASSO must split the coefficient across two adjacent grid points to represent the signal, which fights against the sparsity assumption and degrades both centroid accuracy and the clean one-coefficient-per-ion ideal.

This is the same insight behind Comet's `fragment_bin_offset` parameter (typically set to ~0.4 × bin_width), where the XCorr bin grid is shifted so that bin centers align with where peptide b/y fragment ions most commonly fall. Amino acid residue masses have characteristic mass defects that cause fragment m/z values to cluster at specific fractional positions rather than being uniformly distributed. The "forbidden zones" in peptide fragment m/z space — regions where no peptide fragment can land given the elemental composition constraints of amino acids — mean that a naive grid placement wastes basis functions on impossible positions while leaving real signal positions poorly represented.

**Offset procedure — multi-grid evaluation:**

Rather than estimating the optimal offset from the fractional mass distribution of calibration peaks (which requires many isolated peaks), we evaluate multiple offset grids directly and select the one that best fits the data. This is analogous to how Comet's `fragment_bin_offset` of ~0.4 was determined empirically — but here we do it per-file rather than relying on a fixed default.

1. Define `N_OFFSETS` candidate offsets uniformly spanning one grid spacing interval: `offset[k] = k × grid_spacing / N_OFFSETS` for k = 0, 1, ..., N_OFFSETS-1. Default `N_OFFSETS = 3` (offsets at 0, 1/3, and 2/3 of grid spacing).
2. During the σ calibration pass, select a representative set of signal regions from the calibration spectra (e.g., 50-100 regions with good S/N, mixture of single and composite).
3. For each candidate offset, run the LASSO decomposition on each representative region and record the total residual: `RSS[k] = Σ_regions ||y - Aβ||²`.
4. Select the offset with the lowest total residual: `grid_offset = offset[argmin(RSS)]`.
5. The final grid positions are: `mz_grid[j] = grid_offset + j × grid_spacing`.

The cost of this step is `N_OFFSETS × N_regions` LASSO solves during calibration — with 3 offsets and 100 regions, this is ~300 small LASSO problems, adding well under a second to the calibration phase. The Gram matrix template only needs to be computed once (it is offset-independent), and only the **Aᵀy** product changes between offsets since the data points are evaluated against shifted basis positions.

**User override and defaults:** The grid offset is exposed as a CLI parameter (`--grid-offset`). Once a characteristic offset has been determined across multiple Stellar DIA runs, this can be set as a default value in the code — similar to how Comet's 0.4 offset became the community standard after empirical characterization. Users can still override it for non-standard samples (e.g., non-peptide analytes with different mass defect distributions).

**Stratification by MS level:** The multi-grid evaluation is run separately for MS1 and MS2 spectra, since precursor ions and fragment ions have different fractional mass distributions. Store the offset per MS level in the `BasisPrecompute` struct.

**Impact:** A well-chosen offset ensures that the most common signal positions land on or very near a grid point, where the LASSO recovers them with a single non-zero coefficient. Signals at less common fractional masses still fall between grid points but are recovered via the sub-grid refinement step (§3.5). At runtime, only the winning offset grid is used — there is no per-spectrum cost for the multi-grid evaluation.

**Impact on the Toeplitz Gram matrix:** The grid offset does not change the Gram matrix structure. The Toeplitz property depends only on the uniform spacing between basis functions, not their absolute positions. The same precomputed Gram row is valid regardless of offset.

**Impact on sub-grid refinement (§3.5):** When the grid is well-aligned with common signal positions, fewer centroids require sub-grid refinement, and those that do are closer to a grid point, improving the parabolic interpolation accuracy. With a poorly aligned grid, sub-grid refinement bears the full burden of centroid precision and is more susceptible to noise.

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_grid_offsets` | 3 | Number of candidate offsets to evaluate during calibration |
| `n_offset_eval_regions` | 100 | Number of signal regions used for offset evaluation |

---

### 3.8 Output Centroid Assembly

For each spectrum, collect all centroids from both the fast path and LASSO path:

1. Sort centroids by m/z.
2. Merge centroids within `grid_spacing / 2` of each other (shouldn't happen if grid is consistent, but safety check).
3. Compute intensity: for fast-path centroids, use the fitted Gaussian amplitude. For LASSO centroids, use β_j × peak_of_basis_function (i.e., β_j × 1.0 since basis Gaussians have unit peak height).
4. Optionally: report a quality flag per centroid (`single_peak` vs `deconvolved`) in a userParam array.

---

## 4. mzML I/O

### 4.1 Reading Profile mzML (from Osprey patterns)

Use the `mzdata` crate (same as Osprey's `osprey-io/mzml/` module) for mzML parsing in Rust:

- Parse indexed mzML with random access via the offset index
- Decode binary data arrays: base64 → zlib decompress → f64/f32 arrays
- Extract per-spectrum metadata: ms level, scan time, TIC, filter string, precursor info
- Handle both MS1 and MS2 spectra
- Detect profile vs. centroid mode from CV terms (`MS:1000128` = profile, `MS:1000127` = centroid)
- **Critical:** Validate that input data is profile mode. If centroid mode is detected, emit error and exit.

**nativeID preservation:** Preserve the Thermo nativeID format `controllerType=0 controllerNumber=1 scan=NNNN` with CV term `MS:1000768` exactly as in the input file.

### 4.2 Writing Centroided mzML (from MARS patterns)

Use the **passthrough writer** pattern developed for MARS. This preserves the original file structure byte-for-byte and only modifies the binary data arrays and associated metadata:

**What changes:**
- m/z binary data arrays: replaced with centroided m/z values (sorted, f64)
- Intensity binary data arrays: replaced with centroided intensities (f32)
- `encodedLength` attribute: updated to match new base64 string lengths
- `defaultArrayLength` attribute on each `<spectrum>`: updated to centroid count
- Spectrum representation CV term: changed from `MS:1000128` (profile) to `MS:1000127` (centroid)
- `<indexList>`: regenerated (byte offsets change when binary data changes)
- `<indexListOffset>`: recalculated
- `<fileChecksum>`: recalculated SHA-1 hash

**What is preserved (byte-for-byte):**
- All XML namespaces and schema locations
- Thermo nativeID format and source file references
- Instrument configuration (model, serial number, components)
- All spectrum CV parameters (base peak, TIC, filter string, scan time, precursor info)
- All chromatograms (TIC, pump pressure, etc.)
- `cvRef="MS"` (not changed to `"PSI-MS"`)
- Attribute ordering within elements

**Additional metadata written:**
- `<dataProcessing>` entry documenting the centroiding step, including:
  - Software name and version
  - Processing method: "peak picking" (`MS:1000035`)
  - σ_calibrated value (as `userParam`)
  - λ_factor value (as `userParam`)
  - Number of spectra processed
  - Timestamp

**Binary encoding rules (from MARS/Osprey patterns):**
- m/z arrays: 64-bit float, zlib compressed
- Intensity arrays: 32-bit float, zlib compressed
- Always match the compression settings of the input file
- `encodedLength` = character count of base64 string (not byte count of raw data)

**Rust implementation:** The passthrough writer in Rust uses `quick-xml` for streaming XML parsing/writing. Unlike the Python lxml approach used in MARS, the Rust version can stream through the file without loading it entirely into memory, which is critical for large DIA files (multi-GB).

```
Streaming passthrough writer architecture:

Input mzML → quick-xml reader (streaming)
  │
  ├─ Non-spectrum elements → pass through unchanged
  │
  ├─ <spectrum> elements:
  │   ├─ Parse metadata (id, ms level, etc.)
  │   ├─ Decode profile binary data
  │   ├─ Run centroiding algorithm
  │   ├─ Encode centroid arrays as binary
  │   ├─ Update encodedLength, defaultArrayLength, representation CV term
  │   └─ Write modified spectrum
  │
  ├─ After all spectra: buffer byte offsets for index
  │
  └─ Write regenerated <indexList>, <indexListOffset>, <fileChecksum>
```

---

## 5. Rust Crate Architecture

```
centroid/
├── Cargo.toml
├── src/
│   ├── main.rs                    # CLI entry point (clap)
│   ├── lib.rs                     # Public API
│   │
│   ├── calibration.rs             # σ auto-calibration (per MS level)
│   │   ├── find_isolated_peaks()
│   │   ├── fit_single_gaussian()
│   │   ├── calibrate_sigma()      # Returns σ per MS level stratum
│   │   ├── compare_strata()       # Decides if MS1/MS2 need separate σ
│   │   ├── detect_grid_spacing()  # Auto-detect from profile point density
│   │   └── evaluate_grid_offsets()# Multi-grid LASSO evaluation for optimal offset
│   │
│   ├── noise.rs                   # Iterative baseline/noise estimation (§3.3)
│   │   ├── NoiseProfile           # Per-spectrum baseline + noise σ curves
│   │   ├── rough_noise_estimate() # Pass 0: minimum-statistics / 10th percentile
│   │   ├── refine_from_residuals()# Compute noise model from LASSO residuals
│   │   ├── gap_noise_estimate()   # Noise from inter-region gap intensities
│   │   ├── combine_estimates()    # Merge residual + gap estimates, fit smooth curve
│   │   └── needs_refit()          # Check if λ changed enough to warrant pass 2
│   │
│   ├── basis.rs                   # Basis function precomputation
│   │   ├── GramTemplate           # Toeplitz Gram matrix (precomputed row)
│   │   ├── BasisTemplate          # Single Gaussian template at sub-grid resolution
│   │   ├── precompute_gram_row()
│   │   └── build_local_gram()     # Extract sub-matrix for a signal region
│   │
│   ├── signal.rs                  # Signal region detection
│   │   ├── SignalRegion
│   │   ├── detect_signal_regions()# Uses NoiseProfile for threshold
│   │   └── merge_adjacent_regions()
│   │
│   ├── centroid.rs                # Core centroiding logic (two-pass)
│   │   ├── CentroidResult { mz: f64, intensity: f32, quality: Quality }
│   │   ├── centroid_spectrum()    # Orchestrates pass 1, noise refinement, pass 2
│   │   ├── centroid_pass()        # Single pass: region detection → classify → fit
│   │   ├── fast_path_gaussian()   # 3-point Gaussian fit
│   │   ├── lasso_decompose()      # Non-negative LASSO coordinate descent
│   │   ├── refine_subgrid()       # Sub-grid centroid position refinement
│   │   └── collect_residuals()    # Gather residuals from fitted regions for noise est.
│   │
│   ├── lasso.rs                   # LASSO solver (BLAS-accelerated)
│   │   ├── NonNegativeLasso
│   │   ├── solve()                # Coordinate descent with active set
│   │   ├── compute_aty()          # BLAS dgemv wrapper
│   │   └── soft_threshold_nn()
│   │
│   ├── io/
│   │   ├── mod.rs
│   │   ├── reader.rs              # Profile mzML reader (mzdata crate)
│   │   │   ├── ProfileMzmlReader
│   │   │   ├── validate_profile_mode()
│   │   │   └── iter_spectra()     # Streaming spectrum iterator
│   │   │
│   │   └── writer.rs              # Centroided mzML writer (passthrough pattern)
│   │       ├── PassthroughWriter
│   │       ├── write_centroided_spectrum()
│   │       ├── update_binary_array()
│   │       ├── regenerate_index()
│   │       └── compute_sha1()
│   │
│   └── config.rs                  # Configuration and CLI args
│       ├── Config (serde + clap)
│       └── defaults
│
├── benches/
│   ├── lasso_benchmark.rs         # LASSO solver performance
│   ├── centroid_benchmark.rs      # Full per-spectrum timing
│   └── io_benchmark.rs            # mzML read/write throughput
│
└── tests/
    ├── calibration_tests.rs
    ├── lasso_tests.rs
    ├── centroid_tests.rs
    └── roundtrip_tests.rs         # Profile → centroid → validate vs Thermo
```

---

## 6. Key Data Structures

```rust
/// Auto-calibrated instrument parameters
pub struct InstrumentProfile {
    /// Gaussian σ per MS level (MS1 and MS2 may differ due to scan rate)
    pub sigma_ms1: SigmaModel,
    pub sigma_ms2: SigmaModel,
    /// Estimated noise characteristics
    pub noise_model: NoiseModel,
}

pub enum SigmaModel {
    Constant(f64),                    // σ in Da
    Linear { slope: f64, intercept: f64 }, // σ(mz) = slope * mz + intercept
}

pub struct NoiseModel {
    /// Rough initial noise estimate (from minimum statistics / 10th percentile)
    pub rough_noise: f64,
    /// Refined noise σ as a function of m/z (from LASSO residuals, §3.3)
    /// Stored as (mz_center, noise_sigma) pairs for interpolation
    pub refined_noise_curve: Vec<(f64, f64)>,
    /// Refined baseline as a function of m/z
    /// Stored as (mz_center, baseline) pairs for interpolation
    pub refined_baseline_curve: Vec<(f64, f64)>,
    /// Whether pass 2 refinement has been computed
    pub is_refined: bool,
}

/// Precomputed basis function templates (one per distinct σ value)
pub struct BasisPrecompute {
    /// Toeplitz row for Gram matrix: g[k] = σ√π × exp(-(k×spacing)²/(4σ²))
    pub gram_row: Vec<f64>,
    /// Grid spacing in Da (auto-detected from profile data, see §3.7)
    pub grid_spacing: f64,
    /// Grid offset in Da (optimized for peptide mass defect alignment, see §3.7)
    pub grid_offset: f64,
    /// Calibrated σ
    pub sigma: f64,
    /// Norm of a single basis column (constant for identical Gaussians)
    pub column_norm: f64,
    /// λ_factor (user-configurable detection threshold; actual λ is computed
    /// per-spectrum using λ = lambda_factor × noise_σ(mz) × column_norm, see §3.3)
    pub lambda_factor: f64,
}

/// A contiguous region of signal in a profile spectrum
pub struct SignalRegion {
    pub start_idx: usize,             // Index into profile data arrays
    pub end_idx: usize,
    pub mz_start: f64,
    pub mz_end: f64,
    pub max_intensity: f64,
    pub width_da: f64,                // mz_end - mz_start
    pub classification: RegionClass,
}

pub enum RegionClass {
    SinglePeak,                        // Width < threshold × σ
    PotentialComposite,                // Width ≥ threshold × σ → LASSO
}

/// Result of centroiding a single signal region
pub struct CentroidResult {
    pub mz: f64,
    pub intensity: f32,
    pub quality: CentroidQuality,
}

pub enum CentroidQuality {
    SinglePeakFit,                    // From 3-point Gaussian
    LassoDeconvolved {                // From LASSO decomposition
        n_components: u8,             // Total components in this region
        coefficient: f64,             // Raw LASSO coefficient
    },
}

/// Per-spectrum centroiding statistics (for logging/QC)
pub struct SpectrumStats {
    pub scan_number: u32,
    pub n_signal_regions_pass1: usize,
    pub n_signal_regions_pass2: usize,  // Regions re-fitted in pass 2
    pub n_single_peaks: usize,
    pub n_composite_regions: usize,
    pub n_centroids_total: usize,
    pub rough_noise_estimate: f64,
    pub refined_noise_median: f64,      // Median of refined noise σ across m/z
    pub refined_baseline_median: f64,   // Median of refined baseline across m/z
    pub processing_time_us: u64,
}
```

---

## 7. BLAS Integration

**Crate:** `ndarray` with `ndarray-linalg` (OpenBLAS or Intel MKL backend).

BLAS is used for two operations:

### 7.1 Aᵀy computation (`dgemv`)

For each signal region, compute the correlation of the data with each basis function:

```rust
use ndarray::Array1;
use ndarray_linalg::blas::Gemv;

fn compute_aty(
    basis_matrix: &Array2<f64>,  // n_data × n_basis (constructed from template)
    y: &Array1<f64>,             // n_data (observed profile intensities)
) -> Array1<f64> {
    basis_matrix.t().dot(y)      // Uses BLAS dgemv internally
}
```

This is O(n_data × n_basis) and is the dominant per-region cost.

### 7.2 Batch spectrum processing (`dgemm`)

When processing multiple signal regions from the same spectrum, regions of similar size can be batched into a single matrix multiply:

```rust
// If multiple regions have the same n_basis (common for similar-width regions),
// stack their y vectors into a matrix Y and compute AᵀY via dgemm
fn compute_aty_batch(
    basis_matrix: &Array2<f64>,  // n_data × n_basis (shared for same-size regions)
    y_batch: &Array2<f64>,       // n_data × n_regions
) -> Array2<f64> {
    basis_matrix.t().dot(y_batch) // Uses BLAS dgemm internally
}
```

### 7.3 What doesn't need BLAS

The coordinate descent inner loop operates on precomputed scalars (Gram matrix entries) and updates one variable at a time. BLAS overhead (function call, memory setup) would exceed the cost of the actual computation for these scalar operations. The active set is typically 2-5 variables, so the inner loop is ~10-50 multiply-adds per iteration.

---

## 8. Parallelization Strategy

**Crate:** `rayon` for data parallelism.

### Level 1: Across spectra

Spectra are independent and can be processed in parallel. The `BasisPrecompute` structs (one per MS level if σ differs) are shared (read-only) across all threads. The two-pass noise refinement is entirely contained within each spectrum's processing — pass 1 and pass 2 are sequential within a spectrum, but spectra are fully parallel with each other.

```rust
use rayon::prelude::*;

let results: Vec<Vec<CentroidResult>> = spectra
    .par_iter()
    .map(|spectrum| {
        let precompute = match spectrum.ms_level {
            1 => &precompute_ms1,
            _ => &precompute_ms2,
        };
        centroid_spectrum(spectrum, precompute, &config)
    })
    .collect();
```

### Level 2: Across signal regions within a spectrum

For spectra with many signal regions, regions can be processed in parallel within a single spectrum. However, the overhead of spawning parallel tasks may exceed the savings for small regions. Use a threshold: parallelize within-spectrum only if `n_signal_regions > 20`.

### Memory considerations

Each thread needs a small working buffer for the local **A** matrix and coordinate descent state. Pre-allocate per-thread buffers to avoid allocation in the hot loop. Additionally, pass 1 LASSO solutions (the β vectors per signal region) must be retained until pass 2 completes, to enable warm-starting. For a spectrum with ~100 signal regions and ~30 basis functions per region, this is ~24 KB per spectrum — negligible.

```rust
use rayon::ThreadPoolBuilder;
use std::cell::RefCell;

thread_local! {
    static WORK_BUFFER: RefCell<WorkBuffer> = RefCell::new(WorkBuffer::new(MAX_REGION_SIZE));
}
```

---

## 9. Performance Targets

### Per-spectrum budget

For a DIA experiment at ~30 Hz scan rate (33 ms per spectrum). The number of basis functions per signal region depends on the profile point density, which will be determined empirically. These estimates assume a typical region of ~30 basis functions:

| Step | Estimated time | Notes |
|------|---------------|-------|
| Rough noise estimate | < 0.05 ms | Sliding minimum + percentile |
| Pass 1: signal region detection | < 0.1 ms | Linear sweep, O(n_profile_points) |
| Pass 1: fast-path centroids (~80% of regions) | < 0.2 ms | Analytical, no iteration |
| Pass 1: LASSO decomposition (~20% of regions) | < 1.0 ms | ~10 regions × ~30 basis × 50 iterations |
| Pass 1: Aᵀy via BLAS | < 0.3 ms | Dominant cost for LASSO regions |
| Noise refinement from residuals | < 0.1 ms | Std/mean over residual vectors |
| Pass 2: re-fit changed regions (~10-20% of regions) | < 0.3 ms | Warm-started, few iterations |
| Output assembly | < 0.05 ms | Sort + merge |
| **Total per spectrum** | **< 3 ms** | **Single thread** |

The two-pass approach adds ~50% overhead vs. single-pass (primarily from the pass 2 re-fits), but pass 2 is cheap because it re-fits only regions where λ changed significantly, and warm-starting means the LASSO converges in a few iterations rather than starting from scratch.

With 16-thread parallelism and ~30 spectra/second acquisition: processing should run at ~10× real-time (i.e., 30 minutes of data processed in ~3 minutes).

### Throughput targets

| Metric | Target |
|--------|--------|
| Single-spectrum latency (1 thread) | < 5 ms |
| Throughput (16 threads) | > 3,000 spectra/sec |
| 30-min DIA run (~54,000 spectra) | < 5 minutes total |
| Memory usage | < 2 GB (streaming I/O) |
| Output file size | ~30-50% of profile input (centroid data is smaller) |

---

## 10. CLI Interface

```
centroid [OPTIONS] <INPUT_MZML> -o <OUTPUT_MZML>

Arguments:
  <INPUT_MZML>           Input profile-mode mzML file

Options:
  -o, --output <PATH>    Output centroided mzML file [required]
  
Calibration:
  --sigma-ms1 <DA>       Override auto-calibrated σ for MS1 (Da)
  --sigma-ms2 <DA>       Override auto-calibrated σ for MS2 (Da)
  --sigma <DA>           Override auto-calibrated σ for all levels (Da)
  --n-cal-spectra <N>    Number of spectra for σ calibration [default: 50]
  
Algorithm:
  --grid-spacing <DA>    Basis function grid spacing [default: auto from profile data]
  --grid-offset <DA>     Basis function grid offset [default: auto via multi-grid evaluation]
  --n-grid-offsets <N>   Number of candidate offsets to evaluate [default: 3]
  --lambda-factor <F>    Detection threshold in noise σ units [default: 3.0]
  --max-iter <N>         Maximum LASSO iterations [default: 500]
  --single-peak-threshold <F>  Width/σ ratio for fast path [default: 2.5]
  --single-pass          Skip pass 2 noise refinement (faster, less accurate on faint signals)
  --noise-window <DA>    Window width for refined noise model [default: 20.0]
  
Performance:
  --threads <N>          Number of threads [default: auto]
  
Output:
  --stats <PATH>         Write per-spectrum statistics to TSV
  --verbose              Print per-spectrum progress
  --quiet                Suppress all output except errors
```

---

## 11. Testing Strategy

### 11.1 Unit Tests

**σ calibration:**
- Synthetic profile spectra with known σ → verify recovery within 5%
- Different σ for MS1 vs MS2 → verify stratification produces separate estimates
- Same σ for MS1 vs MS2 (within 10%) → verify they are merged into a single estimate
- Insufficient isolated peaks in one stratum → verify fallback to default or user-provided σ

**Grid spacing and offset auto-detection:**
- Uniform profile spacing → verify exact spacing recovery
- Slightly non-uniform spacing (ADC jitter) → verify median is reported
- User override finer than data spacing → verify warning is emitted
- Very coarse data (< 3 points per σ) → verify warning about undersampled peaks
- Synthetic peaks at known fractional masses → verify multi-grid selects offset closest to true positions
- Synthetic peaks at uniform fractional positions → verify all offsets produce similar RSS (no strong preference)
- Different fractional mass distributions for MS1 vs MS2 → verify separate offsets selected
- User override via CLI → verify override is used without multi-grid evaluation
- N_OFFSETS=1 → verify degenerates to offset=0 (no optimization)

**Iterative noise estimation:**
- Sparse spectrum (few peaks, mostly baseline) → rough estimate and refined estimate should agree closely
- Dense spectrum (wall-to-wall signal, ~1000 fragments) → rough estimate may be off, but residual-based refinement should recover true noise σ within 20%
- Known synthetic noise (Gaussian noise added to known signal) → verify refined noise σ matches known σ
- Varying baseline across m/z → verify smooth noise model captures the trend
- Zero gap points (100% signal coverage) → verify residual-only estimation works
- Abundant gap points → verify gap-based and residual-based estimates agree
- Pass 2 warm-start convergence → verify pass 2 LASSO converges in <10 iterations from pass 1 solution
- --single-pass mode → verify pass 2 is skipped and output uses rough λ

**LASSO solver:**
- Single Gaussian in basis → should recover one non-zero coefficient at correct position
- Two Gaussians separated by 2σ → should recover both
- Two Gaussians separated by 0.5σ → should attempt decomposition (may merge to one depending on λ)
- Pure noise → should return empty (all zeros)
- Non-negativity → no negative coefficients under any input

**Signal region detection:**
- Empty spectrum → no regions
- Single isolated peak → one region, classified as SinglePeak
- Two peaks separated by 3σ → two regions
- Two peaks separated by 0.5σ → one region, classified as PotentialComposite

### 11.2 Integration Tests

**Round-trip test:**
- Read profile mzML → centroid → write mzML → read centroided mzML → verify
- All spectrum metadata preserved (scan number, RT, precursor info, TIC)
- nativeID format preserved exactly
- Index is valid (DIA-NN can read the output)
- SHA-1 checksum is valid

**Compatibility tests:**
- Output mzML loads in Skyline
- Output mzML loads in DIA-NN
- Output mzML loads in SeeMS
- Output mzML loads in Osprey

### 11.3 Validation Tests

**Ground truth comparison:**
- Acquire the same sample on the Stellar in profile mode and on an Orbitrap (Astral or Exploris) for ground truth
- Apply centroiding to Stellar profile data
- Compare recovered centroid lists against Orbitrap-resolved ion lists
- Metrics: recall (fraction of Orbitrap ions recovered), precision (fraction of centroid tool outputs matching real ions), m/z accuracy of recovered centroids

**Thermo centroid comparison:**
- Acquire the same data in both profile and centroid mode on the Stellar
- Compare this tool's centroids vs. Thermo's centroids
- Specifically: find cases where Thermo reports one centroid but this tool reports two or more → validate which is correct using the Orbitrap ground truth

**Known mixture test:**
- Synthetic peptide mixture with known fragments at closely-spaced m/z values
- Verify that composite peaks are correctly decomposed

---

## 12. Phase 2 Integration Points (Future)

This specification is for Phase 1 (centroiding only). The following integration points are designed into the architecture for Phase 2:

**MARS mass recalibration:** After centroiding, MARS can be applied to the centroided m/z values. The centroided mzML output from this tool is a valid input for the existing MARS Python pipeline. In Phase 2, the MARS model (currently XGBoost in Python) can optionally be integrated into the Rust binary using XGBoost C bindings (`xgboost-rs` crate) or replaced with a Ridge regression model.

**Osprey integration:** The centroided mzML output is a standard input for Osprey. No changes to Osprey are required.

**Averagine-informed centroiding (Phase 2b):** The architecture supports adding isotope-envelope-aware basis functions as an alternative to the uniform Gaussian grid. This would replace the basis construction in `basis.rs` while the LASSO solver and I/O remain unchanged.

---

## 13. Dependencies

```toml
[dependencies]
# mzML I/O
mzdata = "0.30"           # mzML reading (same as Osprey)
quick-xml = "0.31"        # Streaming XML for passthrough writer
base64 = "0.22"           # Binary encoding
flate2 = "1.0"            # zlib compression

# Linear algebra (BLAS)
ndarray = "0.15"
ndarray-linalg = { version = "0.16", features = ["openblas-system"] }

# Parallelism
rayon = "1.8"

# CLI
clap = { version = "4", features = ["derive"] }

# Configuration
serde = { version = "1", features = ["derive"] }
serde_yaml = "0.9"

# Logging
log = "0.4"
env_logger = "0.10"

# Checksums
sha1 = "0.10"

[dev-dependencies]
criterion = "0.5"          # Benchmarks
approx = "0.5"             # Float comparison in tests
tempfile = "3"             # Temp files for I/O tests
```

---

## 14. Open Questions

1. **Profile data sampling density on the Stellar.** The ADC sampling rate during the RF ramp determines the m/z spacing of profile data points, which in turn sets the LASSO grid spacing (§3.7). If the profile is very coarsely sampled (e.g., fewer than 3-4 points per peak σ), the Gaussian fitting in both the σ calibration and the fast-path centroider may be unreliable, and the LASSO basis functions would be poorly constrained. **Action required:** Collect profile-mode data from the Stellar at typical DIA scan rates to determine the point density and validate that peaks are adequately sampled. This is the gating prerequisite before implementation.

2. **Very high ion density regions.** In MS1 spectra with dense isotope envelopes from multiply-charged precursors, the number of overlapping signals in a region could be large (>10). The LASSO formulation handles this in principle, but convergence may be slow. May need to increase `max_lasso_iter` or use a warm-start strategy for MS1.

3. **Interaction with MARS features.** MARS currently uses features derived from the centroid list (local ion density in m/z bins). If centroiding changes (splitting composites into multiple centroids), the MARS feature distributions will change. The MARS model may need retraining on centroided-then-recentroided data.
