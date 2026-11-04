# CLAUDE.md — Centrix Development Context

## Project Overview

Centrix centroids, and demultiplexes, linear ion trap profile data (Thermo Stellar, the ion trap of a
Fusion/Eclipse). It reads any file ProteoWizard reads (Thermo `.raw` directly) and writes centroided
mzML through ProteoWizard's C# libraries (pwiz-sharp).

**Primary goal:** separate two or more signals within 1 m/z of each other, which the vendor's
one-centroid-per-maximum centroider merges, by a non-negative lasso on a Gaussian basis, one
candidate per profile point. **Second goal:** demultiplex staggered Stellar DIA (50% overlap,
SuperOverlap) jointly with centroiding.

Language: C# on .NET 10. The Rust implementation is history (tag `rust-final`).

## Build Commands

```bash
./build.sh --test --i-agree-to-the-vendor-licenses        # Linux / WSL
pwsh ./build.ps1 -RunTests -IAgreeToVendorLicenses        # Windows
pwsh ./package.ps1 -IAgreeToVendorLicenses [-Setup]       # dist/ zip, tar.gz, Setup.exe
```

By hand (after `scripts/get-pwiz.ps1`):

```bash
dotnet build Centrix.sln -c Release -p:Platform=x64 -p:IAgreeToVendorLicenses=true
dotnet test Centrix.sln -c Release -p:Platform=x64 -p:IAgreeToVendorLicenses=true --no-build
```

`-p:IAgreeToVendorLicenses=true` must be a global property (command line), or the pwiz vendor
projects compile without vendor support. Warnings are errors. Run the tests before every commit.

`pwsh scripts/smoke-test.ps1 -Centrix <exe>` converts ProteoWizard's Thermo test `.raw` and checks the
mzML (CI runs it on Windows and Linux, and against the installed Linux package).

## CI and releases

`.github/workflows/ci.yml`: build+test (Windows, Ubuntu, no vendor), Thermo smoke (vendor build, both
OSes), package (main and `v*` tags: zip, tarball, Inno Setup installer), install the Linux package on a
clean runner and smoke it, release (`v*` tag must equal `v<Version>` from `Directory.Build.props`).
To release: bump `<Version>`, commit, tag `v<version>`, push the tag.

## Layout

```
src/Centrix.Core/      The centroider (CentrixCentroider, NonNegativeLasso, CentrixBasis, NoiseProfile,
                       SignalRegion) and the peak-sigma calibration (CentrixCalibration). No dependencies.
src/Centrix.Demux/     The staggered joint demultiplexer (StaggeredDemuxPipeline, JointDemultiplexer,
                       JointApportioner, DemuxSchemeDetector, ...). Depends on Centrix.Core only.
src/Centrix/           The `centrix` executable: CentrixOptions (CLI), Conversion (one file),
                       CentrixSpectrumList (parallel read-ahead centroiding), SpectrumRouting (who
                       centroids which spectrum), StaggeredDemuxSpectrumList + PwizDemuxSource (pwiz glue).
tests/Centrix.Tests/   MSTest; real Stellar fixtures and goldens under Data/.
pwiz/                  The pinned ProteoWizard tree (gitignored; scripts/get-pwiz.*, pwiz.lock).
installer/Setup.iss    The Windows installer (Inno Setup 6).
docs/                  usage, algorithm, demultiplexing, building; history/ holds the Rust spec.
scripts/               Python mzML validators and notebooks (unchanged from the Rust era).
```

## Key Decisions

### ProteoWizard is pinned source, not a package
pwiz-sharp has no NuGet packages. `pwiz.lock` pins a commit; `scripts/get-pwiz.*` makes a shallow,
blobless, sparse clone into `./pwiz` (`scripts/pwiz-sparse-paths.txt`). `PWIZ_ROOT` overrides it with
another checkout. Bump the pin and the tests together.

### Core and Demux stay free of file formats
They take spectra in and hand centroids back, so they can be synced with Osprey, where the same
classes live as `Osprey.Centroid` / `Osprey.Demux` (ProteoWizard/pwiz, `pwiz_tools/Osprey`), and so
any reader can feed them. Only `src/Centrix` knows ProteoWizard.

### Fidelity goldens
- `CentrixTest.TestCentrixRustGolden`: the port matches Rust Centrix's centroids on 30 real Stellar
  spectra. Never regenerate it from C#.
- `StellarDemuxFixtureTest`: the joint solve matches Osprey's golden (Osprey's reader kept fill times
  as float32; the test rounds them likewise). Rebless only with `CENTRIX_REBLESS_DEMUX_FIXTURE=1` and
  a reason in the commit.
- Raise `CentrixCentroider.ALGORITHM_VERSION` with any change that moves a centroid.

### The peak sigma follows the point spacing
On a linear ion trap, sigma = 1.1 point spacings (0.22 Th at the Stellar's 200 kTh/s MS2), from a
sweep searched with entrapment; elsewhere measured from isolated peaks. Not from filter strings: the
Stellar's MS2 carry "Z" (its Adaptive RT annotation) instead of a scan-rate letter.

### Calibrate measures, run does not
`centrix calibrate` samples up to 3,000 spectra and reports each level's measured sigma (isolated-peak
FWHM lower quartile / 2.3548) beside the fitted one (`MakePlan(spectra, measure: true)`,
`CentrixSigmaCalibrator.MeasuredSigma`). `run` stops sampling once the point spacing settles the sigma.
The 1.1-spacing rule is validated by searches only at 200 kTh/s (ratio fitted/measured 0.81 there);
the Stellar's 67 kTh/s MS1 is fitted at 0.42 of its measured width; no real 125 or 33 kTh/s data yet.

### Routing (`--centroid auto`)
Centrix for an ion trap analyzer's profile spectra; the vendor's centroids (pwiz CWT without a
vendor) for other analyzers (a Fusion's Orbitrap MS1); centroided spectra pass through.

### Threading
A Thermo `.raw` serves one request at a time: all reads of the inner list happen on one read-ahead
thread under a lock; centroiding a batch runs on every thread. The output never depends on the
thread count. The demultiplexer reads its source synchronously on the caller's thread.

### Demultiplexed outputs are independent objects
`StaggeredDemuxSpectrumList` reads an acquired spectrum's metadata once for all its bins, and gives
each output its own terms, precursors and arrays (`Detached`): `ParamContainer.Set` mutates a
`CVParam` in place, and the read-ahead holds several siblings at once.

### Writer
`MSDataFile.Write` lists `msd.DataProcessings`, not the spectrum list's: `Conversion` adds Centrix's
processing record there (its own id, `Centrix_processing`). Output goes to `<name>.partial`, renamed
when complete. Every spectrum is written, empty ones too (DIA-NN needs them).

### Vendor SDKs in packages
Compiled in, stripped from packages by `pwiz/build/vendor-sdk-pins.json`'s prefixes, fetched on
first use by `VendorSdkLoader` (registered in `Program.Main`). `-BundleVendorSdks` keeps them, for
offline machines only.

## Code Style

Follows the Osprey code it came from: file headers with the AI-assistance line, `_camelCase` private
fields, `///` docs on public members, comments that say why, `CultureInfo.InvariantCulture` for every
number formatted or parsed, `Array.Sort` only where ties cannot matter (stable `OrderBy` otherwise).

## Empirical Data Findings

Profile grid spacing is fixed by firmware per scan rate, uniform within a scan:

| Scan rate | Spacing | Centrix sigma (1.1 spacings) |
|---|---|---|
| 33 kTh/s | 1/30 Th | 0.037 Th |
| 67 kTh/s (Stellar MS1) | 1/15 Th | 0.073 Th |
| 125 kTh/s | 1/8 Th | 0.138 Th |
| 200 kTh/s (Stellar MS2) | 1/5 Th | 0.220 Th |

ProteoWizard's Thermo reader test file (an LTQ Velos run from 2009, the only `.raw` in the pwiz
clone, used only to test `.raw` reading) samples full scans every 1/12 Th (not in the table: sigma
measured) and data-dependent zoom scans every 0.02 Th; one sigma per MS level serves both.

### The Stellar's zoom-scan label and Adaptive RT
The Stellar's firmware writes its ordinary DIA MS2 as zoom scans (filter "Z", ProteoWizard's MS:1000497),
a bug reported to Thermo. On Stellar runs `StellarZoomLabelFix` (applied in `Conversion.Open`) removes the
term and the filter's "Z" from MS2 no wider than `--adaptive-rt-width` (10 Th) and records it;
`--keep-zoom-label` disables it. Wider MS2 are the Adaptive RT alignment spectra: they keep the label,
are recognized by width (never by "Z"), and are excluded by default (`SpectrumListFilter` in
`Conversion.ExcludeAdaptiveRt`); `--adaptive-rt include` keeps them, centroided, outside demux.

Stellar intensities are rates (ions/s); ions = intensity x fill time (s) / 0.98 (the ion trap response).
Thermo onboard centroids: about one per nominal mass; a Centrix centroid's intensity is its peak's
area over the spacing, about 3.4x a vendor centroid's height.
