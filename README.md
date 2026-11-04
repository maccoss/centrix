# Centrix

[![CI](https://github.com/maccoss/centrix/actions/workflows/ci.yml/badge.svg)](https://github.com/maccoss/centrix/actions/workflows/ci.yml)

Centroiding, and demultiplexing, for linear ion trap profile data: the Thermo Stellar, and the ion
trap of a Fusion or an Eclipse. Centrix reads a run straight from the instrument's `.raw` file (or
any file ProteoWizard reads) and writes a centroided mzML that any search engine reads.

**Why.** A linear ion trap resolves only to about 1 m/z: a fragment's profile peak is 0.4 to 0.8 Th
wide at half maximum. The vendor's centroider places one centroid per local maximum, so two
fragments less than about 1 Th apart come out as one centroid, at an m/z that belongs to neither.
Centrix fits each region of the profile with Gaussian peaks of the instrument's width, one candidate
per profile point, by a non-negative lasso, and keeps the few that explain the signal. Signals
closer than 1 Th come out separately.

Three Stellar HeLa runs (2 Th DIA, profile), searched by Osprey with a CarafeSharp library
fine-tuned on the middle run, at 1% FDR ([docs/results.md](docs/results.md)):

| | Thermo's centroids (.raw) | Centrix |
|---|---|---|
| Precursors | 24,132 | **36,842** (1.53x) |
| Peptides | 21,638 | **33,017** |
| Protein groups | 3,516 | **4,534** |
| Fragment m/z error (SD) | 0.39 m/z | **0.17 m/z** |

Both searched with Osprey's fragment calibration starting from ±1.0 m/z. At Osprey's ±0.5 m/z default
the vendor centroids fare worse still (19,265 precursors, against 37,532 for Centrix): they scatter
wider than the window the calibration starts from.

**Staggered DIA.** On Stellar runs acquired with overlapping windows (50% overlap, SuperOverlap),
Centrix demultiplexes and centroids in one solve: each acquired profile is split among the narrow
bins its window covers by the peaks fitted to every window at once, and each bin's share is
centroided. In Osprey searches, 4 Th staggered windows demultiplexed this way recovered most of what
2 Th windows give, at a 4 Th acquisition's cycle time ([docs/results.md](docs/results.md#earlier-searches-osprey-proteowizardpwiz4710)).

## Install

Download a package from the [releases page](https://github.com/maccoss/centrix/releases); neither
needs .NET installed.

- **Windows**: run `Centrix-Setup-<version>.exe`. It installs for you alone (no administrator rights)
  or for everyone, and can put `centrix` on the PATH. Or unzip `centrix-<version>-win-x64.zip`
  anywhere and run `centrix.exe` from it.
- **Linux** (x64):

  ```bash
  tar -xzf centrix-<version>-linux-x64.tar.gz
  centrix-<version>-linux-x64/install.sh            # into ~/.local; --prefix /opt for elsewhere
  centrix --version                                 # ~/.local/bin must be on the PATH
  ```

  `install.sh --uninstall` removes it. The unpacked folder also runs as it is.

The vendor libraries that read `.raw` and the other vendor formats are not in the packages, since
their licenses do not allow redistributing them. The first time Centrix reads a vendor file it
downloads that vendor's library, pinned and hash-checked, from ProteoWizard into a per-user cache,
as ProteoWizard's own tools do. Using them means agreeing to their licenses (`EULA.*.txt` beside
`centrix`). A machine without internet access needs a package built with `-BundleVendorSdks`
([docs/building.md](docs/building.md)).

## Use

```bash
# Centroid a Stellar run, demultiplexing it if its windows overlap: writes run.centrix.mzML beside it
centrix run.raw

# Several files into one directory
centrix -i *.raw -o centroided/

# The peak sigmas and demultiplexing a run would get, and the peak widths it measures, writing nothing
centrix calibrate run.raw
```

By default (`--centroid auto`) Centrix centroids the profile spectra of an ion trap and leaves the
rest to the vendor's centroider: on a Fusion, the ion trap's MS2 by Centrix and the Orbitrap's MS1 by
Thermo's. Every option is in [docs/usage.md](docs/usage.md); `centrix --help` lists them.

**What it decides from the data.** The peak width Centrix fits with comes from the run: on a linear
ion trap, 1.1 profile point spacings, and the point spacing identifies the scan rate (0.22 Th at
200 kTh/s, 0.138 Th at 125 kTh/s, 0.073 Th at 67 kTh/s). That rule was set by searches at 200 kTh/s
only; at other rates `centrix calibrate` shows the width the run's peaks measure beside it, and
`--sigma` overrides it ([docs/algorithm.md](docs/algorithm.md#the-peak-sigma)). Staggered windows are
demultiplexed when they are detected. On a Stellar, the zoom-scan label its firmware puts on DIA
spectra is removed, and its Adaptive RT alignment scans are left out ([docs/usage.md](docs/usage.md#the-stellars-zoom-scan-label)).

Centrix's command line is Rust Centrix's (`centrix run -i ... -o ...`, the same lasso options and
`<stem>.centrix.mzML` outputs), with additions: vendor files read directly, the peak sigma from the
run's point spacing, the exact active-set solver (`--solver active-set`), and demultiplexing.

## Documentation

- [Results](docs/results.md): searches of Centrix's output against the vendor's centroids.
- [Usage](docs/usage.md): the command line, its options and outputs.
- [Algorithm](docs/algorithm.md): the centroider, the peak sigma, the solvers, fidelity to Rust Centrix.
- [Demultiplexing](docs/demultiplexing.md): the staggered joint solve, ions from fill times, Adaptive RT spectra.
- [Building](docs/building.md): the pinned ProteoWizard tree, vendor support, tests, CI, packages and releases.

## History

Centrix began as a Rust program (tagged `rust-final`). It was ported to C# in ProteoWizard's Osprey
(`Osprey.Centroid`, `Osprey.Demux`, ProteoWizard/pwiz#4710), where it was held to the Rust
implementation's centroids on real Stellar spectra, gained the active-set solver and the peak-sigma
rule, and was joined to the staggered demultiplexer. This repository is that C# code as a tool of its
own, reading and writing through ProteoWizard (pwiz-sharp, .NET 10).

## License

MIT, see [LICENSE](LICENSE). The C# sources carry Apache 2.0 headers from their ProteoWizard
origin; ProteoWizard itself is Apache 2.0, and each vendor library has its own license.
