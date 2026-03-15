# CLAUDE.md — Centrix Development Context

## Project Overview

Centrix is a post-acquisition centroiding tool for Thermo Stellar linear ion trap
profile-mode mzML data. It replaces onboard centroiding using non-negative LASSO
regression with Gaussian basis functions.

**Primary goal:** Deconvolute two or more signals within 1 m/z of each other. The
Stellar's onboard centroider places one centroid per ~1 Da interval and collapses
overlapping signals into a single incorrect centroid. Centrix solves this using a
sub-Da Gaussian basis grid and non-negative LASSO.

Target instrument: Thermo Stellar (radial ejection linear ion trap, ~0.5-0.7 Da FWHM).
Language: Rust 2021 edition, single-crate lib/bin split (not a workspace).
Spec: centroiding_spec.md

## Build Commands

```bash
# Development build (OpenBLAS default)
cargo build

# Release build
cargo build --release

# Run on a file
cargo run --release -- input.mzML -o output.mzML

# Run tests
cargo test

# Run benchmarks
cargo bench
```

## CI Requirements (run before every commit)

```bash
cargo fmt
cargo clippy --all-targets -- -D warnings
cargo test
```

## Module Layout

```
src/
├── main.rs         CLI entry point (clap)
├── lib.rs          Public API, pub mod declarations
├── config.rs       Config struct (clap + serde), defaults
├── error.rs        CentrixError (thiserror), Result<T> alias
├── calibration.rs  σ auto-calibration, grid spacing/offset detection
├── noise.rs        Rough noise estimate, Pass 1→2 refinement
├── basis.rs        GramTemplate (Toeplitz row), BasisPrecompute
├── signal.rs       SignalRegion detection and merging
├── centroid.rs     Two-pass orchestration: centroid_spectrum()
├── lasso.rs        Non-negative LASSO coordinate descent solver
└── io/
    ├── mod.rs
    ├── reader.rs   Profile mzML reader (mzdata crate)
    └── writer.rs   Passthrough centroided mzML writer (quick-xml)
```

## Key Architectural Decisions

### Gram Matrix Is Toeplitz
Only the first row is stored as `gram_row: Vec<f64>`. Entry (i,j) = `gram_row[|i-j|]`.
No O(n²) per-region allocation. The entire row fits in L1 cache (~20 entries for
typical σ/grid-spacing ratios).

### BLAS Used Only for Aᵀy
`a.t().dot(y)` (ndarray dispatches to BLAS dgemv) is the only BLAS call in the hot
path. The coordinate descent inner loop uses gram_row scalars directly — BLAS overhead
would exceed computation cost for the 2-5 variable active set typical in these spectra.

### BLAS Must Be Single-Threaded (CRITICAL)
OpenBLAS (and MKL) spawn their own internal thread pools by default, sized to the
number of CPU cores. For Centrix this is catastrophic: every dgemv call on a tiny
8×8 to 22×22 matrix triggers thread synchronization that costs orders of magnitude
more than the actual computation. On a 10-core machine this inflated `sys` time
from 6s to 150+ minutes (>1500× overhead).

**Fix:** `main.rs` calls `openblas_set_num_threads(1)` (or `mkl_set_num_threads(1)`
for the MKL feature) via FFI at startup. Rayon handles spectrum-level parallelism;
BLAS must not compete with its own thread pool.

**DO NOT REMOVE THIS.** If you see poor performance or massive `sys` time, check
that the BLAS single-threading call is still present in `main.rs`.

### Two-Pass Design
- Pass 1: rough noise → detect regions → LASSO all regions → compute residuals inline
- Residuals are computed using the A matrix before it's dropped (avoids redundant rebuild)
- Noise refinement: LASSO residuals + gap intensities → smooth noise model per m/z
- Pass 2: re-run LASSO only on regions where λ changed >20%, warm-started from Pass 1 β
- Aᵀy is cached and reused in Pass 2 (grid doesn't change between passes, only λ changes)

### Single-Peak Fast Path Threshold Is Conservative
A region up to ~2σ wide could be two overlapping signals <1 Da apart — the key use
case. Default threshold errs toward escalating to LASSO rather than the fast path.

### Passthrough Writer
quick-xml streaming mode: pass all XML events byte-for-byte except spectrum binary
arrays. `ByteCountingWriter<W>` wrapper tracks byte offsets for index regeneration.
Regenerates `<indexList>`, `<indexListOffset>`, and SHA-1 `<fileChecksum>` at end.

### Compile Flags
`.cargo/config.toml` sets `-C target-cpu=native` for better auto-vectorization
of exp() calls in the basis matrix construction.

## Dependencies (match Osprey versions exactly)

```toml
mzdata = "0.63"
quick-xml = "0.37"
base64 = "0.22"
flate2 = "1.0"
sha1 = "0.10"
ndarray = { version = "0.16", features = ["rayon", "blas"] }
blas-src = { version = "0.10", default-features = false, features = ["openblas"] }
openblas-src = { version = "0.10", default-features = false, features = ["system", "cblas"] }
rayon = "1.10"
clap = { version = "4.5", features = ["derive", "env"] }
serde = { version = "1.0", features = ["derive"] }
serde_yaml = "0.9"
thiserror = "2.0"
anyhow = "1.0"
log = "0.4"
env_logger = "0.11"
indicatif = "0.17"
chrono = "0.4"
sysinfo = "0.33"
```

Dev dependencies: `criterion = "0.5"`, `approx = "0.5"`, `tempfile = "3.14"`

## Performance Targets

- < 3 ms per spectrum (single thread)
- > 3000 spectra/sec (16 threads)
- < 2 GB memory (streaming + chunked I/O)

## Code Style

- `///` doc comments on all `pub` items; `//` inline for non-obvious logic
- `log::info!` for pipeline steps, `log::debug!` for per-spectrum detail, `log::warn!` for recoverable issues
- `thiserror` `CentrixError` enum + `anyhow` in `main.rs`
- Iterators and functional style preferred
- `cargo fmt` defaults — no manual width overrides

## Empirical Data Findings (from real Stellar mzML files)

Profile grid spacing is **exact and fixed by firmware** — perfectly uniform within each scan:

| MS level | Spacing           | Points/Th | Points/σ (σ≈0.25 Th) | Pts/scan | m/z range |
|----------|-------------------|-----------|----------------------|----------|-----------|
| MS1      | 1/15 = 0.06667 Th | 15        | ~3.75                | 13,500   | 350–1250  |
| MS2      | 1/8  = 0.125 Th   | 8         | 2.00                 | 10,400   | 200–1500  |

MS2 has only **2 data points per σ** — the 3-point fast-path Gaussian fit is
marginal at this density; LASSO is preferred for MS2. Grid auto-detection recovers
these exact spacings from the data point density.

Thermo centroider output for comparison:

- MS1: ~1167 centroids/scan, median gap ~0.76 Th
- MS2: ~70–80 centroids/scan, median gap ~2 Th (confirms ~1 centroid/nominal mass unit)

File `GPF-001.mzML`: 69,993 spectra (693 MS1 + 69,300 MS2).

Example data locations:

- `example-data/thermo-profile/` — input profile mzML (Centrix input)
- `example-data/thermo-centroid/` — Thermo onboard centroiding (baseline comparison)

## BLAS Prerequisite

OpenBLAS system headers required on Linux/WSL:

```bash
sudo apt install libopenblas-dev
```

For Windows/Intel HPC, build with:

```bash
cargo build --release --no-default-features --features mkl
```
