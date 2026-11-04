# Building, testing and packaging

Centrix is a .NET 10 solution (`Centrix.sln`) that builds on Windows and Linux with the .NET 10 SDK.

| Project | What it is |
|---|---|
| `src/Centrix.Core` | The centroider and its peak-sigma calibration; depends on nothing |
| `src/Centrix.Demux` | The staggered joint demultiplexer; depends on Centrix.Core only |
| `src/Centrix` | The `centrix` executable: ProteoWizard reading and writing, the command line |
| `tests/Centrix.Tests` | MSTest tests, with real Stellar fixtures under `Data/` |

## ProteoWizard

The executable reads and writes through ProteoWizard's C# libraries (pwiz-sharp), which are not
published as packages; it builds them from source. `pwiz.lock` pins the ProteoWizard commit, and
`scripts/get-pwiz.ps1` (or `scripts/get-pwiz.sh`) puts that commit in `./pwiz` as a shallow,
blobless, sparse clone of the parts Centrix builds from (`scripts/pwiz-sparse-paths.txt`): about
300 MB, a minute to fetch. The build scripts run it for you. To build against another checkout,
set `PWIZ_ROOT` to it (for example a full ProteoWizard clone you are changing). `./pwiz` is ignored
by git.

To move to a newer ProteoWizard: change the commit in `pwiz.lock`, rerun the build with tests, and
commit the two together.

## Vendor support

Reading `.raw` and the other vendor formats needs the vendors' libraries, which ProteoWizard keeps in
encrypted archives that a build opens only with `-p:IAgreeToVendorLicenses=true`, agreeing to their
licenses (`pwiz/pwiz/data/vendor_readers/*/EULA.*.txt`). Without it every reader still recognizes its
files, and reading one says vendor support is not enabled; open formats read either way. The
property has to be global (`-p:` on the command line, or the scripts' switches), so the ProteoWizard
projects see it too.

Thermo's library is managed code and reads `.raw` on Windows and Linux. Bruker's and Waters' have
Linux builds; the other vendors' are Windows only.

## Build and test

```bash
./build.sh --test --i-agree-to-the-vendor-licenses
```

```powershell
./build.ps1 -RunTests -IAgreeToVendorLicenses
```

or by hand, after fetching ProteoWizard:

```bash
dotnet build Centrix.sln -c Release -p:Platform=x64 -p:IAgreeToVendorLicenses=true
dotnet test Centrix.sln -c Release -p:Platform=x64 -p:IAgreeToVendorLicenses=true --no-build
```

The executable is `src/Centrix/bin/x64/Release/net10.0/centrix[.exe]`. The build treats warnings
as errors.

The tests (27, about 30 s):

| Test class | What it holds |
|---|---|
| `CentrixTest` | Rust Centrix's own unit tests, ported; signals 0.6 Th apart resolved and isolated peaks centroided at their m/z and area at 67, 125 and 200 kTh/s; the active-set solver against the optimum; the peak-sigma rule; the port against Rust Centrix's centroids of 30 real Stellar spectra |
| `DemuxTest` | NNLS against brute force, the portable exp and log bit for bit, the RT interpolants, scheme detection |
| `StaggeredJointDemuxTest` | a simulated Stellar staggered profile run: recovery of m/z and rates, fill-time and thread-count invariance |
| `StellarDemuxFixtureTest` | the joint solve on real SuperOverlap spectra against Osprey's golden |
| `CliTest` | the command line and its exit codes; whole files through `centrix`, read back and checked against the goldens; batch and thread invariance; the routing of spectra; the sigma from a 125 kTh/s run with no instrument metadata; the Stellar's zoom-scan label and Adaptive RT spectra |

A golden changes only with a reason: `StellarDemuxFixtureTest` rewrites its golden when
`CENTRIX_REBLESS_DEMUX_FIXTURE=1` and then fails, so the diff is reviewed and the commit says why.
The Rust golden is never regenerated from the C# code ([Data/Centrix/README.md](../tests/Centrix.Tests/Data/Centrix/README.md)).

## Packages

```powershell
./package.ps1 -IAgreeToVendorLicenses            # dist/centrix-<v>-win-x64.zip and -linux-x64.tar.gz
./package.ps1 -IAgreeToVendorLicenses -Setup     # and dist/Centrix-Setup-<v>.exe (Inno Setup 6)
```

`package.ps1` runs on PowerShell 7 on either OS and builds both packages from either. Each is a
self-contained publish (it carries its own .NET runtime) in one versioned folder:
- **win-x64**: a zip, and with `-Setup` an installer (`installer/Setup.iss`): for the current user
  (no administrator rights) or for everyone, an optional PATH entry, uninstall from Apps & features.
  It needs Inno Setup 6 (`ISCC.exe` on the PATH or in its default folder).
- **linux-x64**: a tar.gz whose `centrix`, `7zz` and `install.sh` are executable, written by .NET's tar
  writer so it is the same from Windows or Linux. `install.sh [--prefix <dir>]` copies the package to
  `<prefix>/lib/centrix-<version>` and links `<prefix>/bin/centrix` to it (default `~/.local`);
  `--uninstall` removes both.

The vendor libraries are compiled in but left out of the packages, as the ProteoWizard installers
leave them out, by ProteoWizard's own table of their file names (`pwiz/build/vendor-sdk-pins.json`);
the script refuses to package one that slips through. The first time a package reads a vendor file,
ProteoWizard's `VendorSdkLoader` downloads that vendor's archive from GitHub, pinned to the
ProteoWizard commit in `pwiz.lock` and hash-checked, into a per-user cache (`%LOCALAPPDATA%\ProteoWizard\vendor`
on Windows, `~/.local/share/ProteoWizard/vendor` on Linux). `-BundleVendorSdks` keeps the libraries
in the package instead, for machines with no internet access; that package is for your own use, not
for redistribution. The vendors' EULAs are in every package built with vendor support.

The version is `<Version>` in `Directory.Build.props`: CalVer `YY.feature.patch`, bumped only at release
time ([release-notes/README.md](../release-notes/README.md)).

## Continuous integration

`.github/workflows/ci.yml`, on GitHub Actions:

| Job | When | What |
|---|---|---|
| Build and test | every push and pull request | build and run the tests on Windows and Ubuntu, without vendor support (the tests read mzML) |
| Read a Thermo .raw | every push and pull request | a vendor build on Windows and Ubuntu runs `scripts/smoke-test.ps1`: ProteoWizard's Thermo test file (an LTQ Velos run in the pwiz tree) calibrated and converted, its mzML checked (85 spectra, none left profile, Centrix's processing record, a complete index) |
| Package | pushes to `main`, `v*` tags, and by hand (Run workflow) | `package.ps1 -IAgreeToVendorLicenses -Setup` on Windows (Inno Setup from Chocolatey if the runner lacks it); the zip, tarball and installer as the run's `packages` artifact |
| Install the Linux package | after Package | unpack the tarball on a clean Ubuntu runner, `install.sh`, and run the smoke test with the installed `centrix`, which fetches the Thermo SDK on first use |
| Release | `v*` tags | check the tag against `<Version>` (before packaging) and publish a GitHub release with the packages, its description `release-notes/RELEASE_NOTES_<tag>.md` |

The pinned ProteoWizard tree is cached between runs, keyed on `pwiz.lock` and the sparse paths. The
vendor jobs pass `-p:IAgreeToVendorLicenses=true`; the workflow's header records that this is the
repository owner's agreement to the vendor licenses.

## Releasing

1. Set `<Version>` in `Directory.Build.props`, and commit.
2. Tag the commit `v<version>` and push the tag.
3. CI builds, tests, packages, installs the Linux package on a clean runner, and publishes the release
   with the three packages and generated notes.

## Code conventions

The C# follows the code it was ported from (ProteoWizard's Osprey): `_camelCase` private fields,
XML doc comments on public members, comments that say why, `CultureInfo.InvariantCulture` for every
number written or parsed. Keep `Centrix.Core` and `Centrix.Demux` free of file formats and of
ProteoWizard, so any reader can feed them, and so changes can be carried to and from Osprey, where
the same classes live as `Osprey.Centroid` and `Osprey.Demux`.
