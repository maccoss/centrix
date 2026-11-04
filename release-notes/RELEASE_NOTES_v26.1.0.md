# Centrix v26.1.0

The first release of Centrix in C# on .NET 10: it reads Thermo `.raw` files directly (and mzML),
writes centroided mzML through ProteoWizard, and demultiplexes staggered
Stellar DIA as it centroids. It installs on Windows and Linux with nothing else to install.
Centrix is now licensed under the Apache License 2.0 (the Rust releases were MIT).

## New features

- **Reads Thermo `.raw` files directly**, through ProteoWizard's C# libraries; there is no need to
  convert to profile mzML first. Thermo's library is downloaded on first use. mzML (and ProteoWizard's
  other open formats) read as before; other vendors' formats are not read.
- **Packages for Windows and Linux** that carry their own .NET runtime: a Windows installer
  (`Centrix-Setup-26.1.0.exe`, per-user or for everyone, optionally on the PATH), a Windows zip, and a
  Linux tarball with `install.sh`.
- **Staggered DIA demultiplexing** (50% overlap and SuperOverlap) solved jointly with centroiding, as in
  Osprey; on by default when the windows are staggered (`--demux`).
- **The peak width follows the scan rate.** On a linear ion trap the peak sigma is 1.1 profile point
  spacings (0.22 Th at the Stellar's 200 kTh/s MS2, 0.073 Th at its 67 kTh/s MS1), read from the point
  spacing; elsewhere it is measured from isolated peaks. `centrix calibrate` reports the measured width
  beside the one used, and `--sigma` / `--sigma-ms1` override it.
- **Routing** (`--centroid auto`): Centrix centroids ion trap profile spectra; other analyzers' spectra
  (a Fusion's Orbitrap MS1) get the vendor's centroids, and centroided spectra pass through.
  `--centroid vendor` writes the vendor's centroids throughout, for comparison.
- **Stellar zoom-scan label removed.** The Stellar performs no zoom scans, but its `.raw` files store
  its 200 kTh/s scan rate in the zoom flag, so its spectra came out labelled as zoom scans. Centrix
  removes the label (the zoom-scan term and the "Z" in the filter string); `--keep-zoom-label` keeps it.
- **Adaptive RT spectra are left out by default.** The Stellar's Adaptive RT alignment scans (MS2
  wider than `--adaptive-rt-width`, 10 Th) are not for searching; `--adaptive-rt include` writes them,
  centroided, outside demultiplexing.
- A faster exact solver for the lasso (`--solver active-set`, about 7 times faster); the default
  coordinate descent matches Rust Centrix.
- Every spectrum is written, empty ones included, in its place (DIA-NN needs them); output goes to
  `<name>.partial` and is renamed when complete; file names can be wildcards.

## Changes to centroids

- Given the same peak sigma, the centroids are Rust Centrix's: they match it on 30 real Stellar
  spectra (`TestCentrixRustGolden`). The default sigma now comes from the profile point spacing (above).
  In a search of three Stellar HeLa runs with Osprey, Centrix's mzML gave 36,842 precursors at 1% FDR
  against 24,132 from the vendor's centroids (docs/results.md).
