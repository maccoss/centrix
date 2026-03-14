# Centrix

Post-acquisition centroiding for Thermo Stellar linear ion trap profile-mode DIA
data. Centrix replaces onboard centroiding using **non-negative LASSO regression**
with Gaussian basis functions, deconvoluting two or more overlapping signals
within 1 m/z of each other that the Stellar's onboard centroider collapses into
a single incorrect centroid.

## Why Centrix?

The Thermo Stellar's onboard centroider places approximately one centroid per
~1 Da interval. When two signals fall within 1 Da of each other — common in
DIA experiments — they are collapsed into a single centroid at an incorrect m/z
and intensity. Centrix solves this by modeling the profile signal as a sparse sum
of Gaussians and solving for the individual peak contributions via LASSO.

## How It Works

Each spectrum is processed through a two-pass algorithm:

1. **Rough noise estimation** — IQR-based noise floor from sorted intensities
2. **Signal region detection** — contiguous above-threshold stretches of the
   profile data
3. **Pass 1: LASSO centroiding** — every signal region is modeled as `y = Aβ + ε`
   where A is a Gaussian basis matrix. Non-negative LASSO with coordinate descent
   finds the sparse set of peak coefficients β. The Toeplitz structure of the Gram
   matrix (uniform grid) means only one row needs to be stored, keeping the solver
   in L1 cache.
4. **Noise refinement** — LASSO residuals + gap intensities give a spatially-varying
   noise model
5. **Pass 2: Selective re-fit** — regions where the refined λ changed >20% are
   re-solved, warm-started from Pass 1 β
6. **Sub-grid refinement** — log-parabola interpolation of LASSO coefficients
   recovers sub-grid m/z precision

See [docs/algorithm.md](docs/algorithm.md) for full algorithmic details.

## Quick Start

### Prerequisites

OpenBLAS headers (Linux/WSL):
```bash
sudo apt install libopenblas-dev
```

### Build

```bash
cargo build --release
```

### Run

```bash
# Single file — output is profile.centrix.mzML in the same directory
centrix run -i profile.mzML

# Multiple files via glob
centrix run -i '*.mzML'

# Explicit output directory
centrix run -i '*.mzML' -o results/
```

### Tune sensitivity

The default parameters are conservative (fewer, high-confidence centroids).
To increase centroid count:

```bash
# Moderate increase
centrix run -i profile.mzML --lambda-factor 1.5

# More aggressive (closer to Thermo centroider count)
centrix run -i profile.mzML \
  --lambda-factor 1.5 --signal-threshold-sigma 2.0
```

See [docs/parameters.md](docs/parameters.md) for all parameters.

## Subcommands

| Command | Purpose |
|---------|---------|
| `run` | Full pipeline: centroid a profile mzML file |
| `inspect` | Print summary statistics for an mzML file |
| `calibrate` | Run calibration only and report detected σ, grid spacing |
| `centroid-test` | Centroid N spectra, print stats (no output file) |

See [docs/usage.md](docs/usage.md) for detailed usage.

## Target Instrument

Centrix is designed for the **Thermo Stellar** radial ejection linear ion trap.
Profile peaks are Gaussian with FWHM 0.5–2.0 Da depending on scan rate:

| Scan Rate | FWHM | σ (Da) | Filter Code |
|-----------|------|--------|-------------|
| 33 kTh/s  | 0.5  | 0.212  | `n`         |
| 67 kTh/s  | 0.6  | 0.255  | `r`         |
| 125 kTh/s | 0.8  | 0.340  | `t`         |
| 200 kTh/s | 2.0  | 0.849  | `u`         |

Profile grid spacing is firmware-fixed: 1/15 Th (MS1), 1/8 Th (MS2).

## Output

Centrix produces standard **centroided mzML** that is compatible with any mzML
reader (ProteoWizard, DIA-NN, etc.). Output files are automatically named
`<stem>.centrix.mzML` and placed next to the input (or in the `-o` directory).

The output is a passthrough copy of the input with only the spectrum binary
arrays replaced:

- Profile → centroid CV term swap
- m/z: 64-bit float, zlib compressed
- Intensity: 32-bit float, zlib compressed (integrated Gaussian area)
- Index and SHA-1 checksum regenerated

Centroid intensities are the **integrated area** of each fitted Gaussian
(β × σ × √(2π)), matching the convention used by the Thermo onboard centroider.

All non-spectrum content (metadata, chromatograms, etc.) is preserved
byte-for-byte. See [docs/io-format.md](docs/io-format.md) for format details.

## Performance

- Batched rayon parallelism (256 spectra per batch)
- Toeplitz Gram matrix: O(bandwidth) per coordinate descent step, ~6–15 entries
  in L1 cache
- Active-set LASSO: typical active set of 2–5 variables
- BLAS only for Aᵀy (dgemv); coordinate descent uses scalars directly
- Streaming I/O: constant memory regardless of file size

## Building with MKL

For Intel MKL instead of OpenBLAS:

```bash
cargo build --release --no-default-features --features mkl
```

## Documentation

- [Algorithm](docs/algorithm.md) — full pipeline and mathematical details
- [Parameters](docs/parameters.md) — all parameters with defaults and tuning guidance
- [I/O Format](docs/io-format.md) — input requirements, output format, writer architecture
- [Usage](docs/usage.md) — subcommands, workflows, batch processing

## License

MIT — see [LICENSE](LICENSE).
