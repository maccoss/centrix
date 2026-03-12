# Centrix Usage Guide

## Installation

### Prerequisites

OpenBLAS system headers are required on Linux/WSL:

```bash
sudo apt install libopenblas-dev
```

### Build

```bash
cargo build --release
```

The binary is at `target/release/centrix`.

For Intel MKL (Windows/HPC):
```bash
cargo build --release --no-default-features --features mkl
```

---

## Subcommands

Centrix has four subcommands:

| Command | Purpose |
|---------|---------|
| `run` | Full pipeline: centroid a profile mzML file |
| `inspect` | Print summary statistics for an mzML file |
| `calibrate` | Run calibration only and report parameters |
| `centroid-test` | Centroid N spectra and print per-spectrum stats (no output file) |

---

## Full Pipeline

```bash
centrix run -i input.mzML -o output.mzML
```

This runs the complete pipeline: calibration → two-pass centroiding → mzML output.

### With custom parameters

```bash
centrix run \
  -i input.mzML \
  -o output.mzML \
  --lambda-factor 1.5 \
  --signal-threshold-sigma 2.0
```

### With a YAML config file

```bash
centrix run -i input.mzML -o output.mzML --config params.yaml
```

### Controlling thread count

```bash
# Use 8 threads
centrix run -i input.mzML -o output.mzML --threads 8

# Use all available cores (default)
centrix run -i input.mzML -o output.mzML --threads 0
```

### Verbose logging

```bash
RUST_LOG=debug centrix run -i input.mzML -o output.mzML
```

Or for info-level:
```bash
RUST_LOG=info centrix run -i input.mzML -o output.mzML
```

---

## Inspect a File

Print per-scan summary statistics for profile or centroided mzML files:

```bash
centrix inspect --input file.mzML
```

Optionally limit the number of spectra:
```bash
centrix inspect --input file.mzML --n 500
```

Output includes: scan number, MS level, data mode (profile/centroid), number of
data points, m/z range, max intensity, and median spacing.

This is useful for:
- Verifying a file is profile mode before running Centrix
- Comparing point counts between profile input and centroided output
- Checking that the output was written correctly

---

## Calibration Only

Run the calibration step and print detected parameters without centroiding:

```bash
centrix calibrate --input file.mzML
```

Output includes:
- Detected σ for MS1 and MS2 (FWHM and σ in Da)
- Profile grid spacing per MS level
- Sampling density (points per σ)
- Gram matrix properties (neighbor correlation)
- Matched Stellar scan rate

This is useful for verifying that auto-calibration produces reasonable values
before running the full pipeline.

---

## Centroid Test

Centroid a limited number of spectra and print per-spectrum statistics without
writing an output file:

```bash
centrix centroid-test --input file.mzML --n 200
```

Output is a table with columns: scan number, MS level, region count,
LASSO count, Pass 2 refit count, centroid count, and processing time.

This is useful for:
- Quick performance profiling
- Evaluating parameter choices before committing to a full run
- Verifying centroid counts are in a reasonable range

---

## Typical Workflows

### Compare Centrix output with Thermo centroiding

```bash
# Run Centrix on profile data
centrix run \
  -i data/profile.mzML \
  -o data/profile.centrix.mzML

# Open both in SeeMS or another viewer
# - data/profile.mzML (profile input)
# - data/profile.centrix.mzML (Centrix output)
# - data/centroid.mzML (Thermo onboard centroiding, if available)
```

### Tune sensitivity

```bash
# 1. Quick test with default parameters
centrix centroid-test --input data/profile.mzML --n 200

# 2. Check centroid counts — if too few, lower lambda_factor
centrix centroid-test --input data/profile.mzML --n 200 \
  -- --lambda-factor 1.5

# 3. Full run with tuned parameters
centrix run \
  -i data/profile.mzML \
  -o data/profile.centrix.mzML \
  --lambda-factor 1.5 \
  --signal-threshold-sigma 2.0
```

### Verify calibration

```bash
# Check what Centrix detects
centrix calibrate --input data/profile.mzML

# If auto-calibration is wrong, override
centrix run \
  -i data/profile.mzML \
  -o data/profile.centrix.mzML \
  --sigma-ms1 0.255 \
  --sigma-ms2 0.340
```

### Batch processing

```bash
for f in data/profile/*.mzML; do
  out="data/centroided/$(basename "$f" .mzML).centrix.mzML"
  centrix run -i "$f" -o "$out" --lambda-factor 1.5
done
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `RUST_LOG` | Log level: `error`, `warn`, `info`, `debug`, `trace` |
| `RAYON_NUM_THREADS` | Alternative way to set thread count (overridden by `--threads`) |

---

## Performance Notes

- The pipeline is I/O-bound on fast machines and compute-bound through WSL or slow
  storage
- Placing input files on native filesystem (not cross-filesystem mounts like
  WSL's `/mnt/`) significantly improves throughput
- Release builds (`--release`) are ~10–20× faster than debug builds
- The progress bar shows total throughput (spectra/sec) across all threads
