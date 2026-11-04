# Using Centrix

```
centrix [run|calibrate] -i <input>... [-o <output>] [options]
```

`run` (the default, and optional) centroids each input into an mzML; `calibrate` reports the peak
sigmas and the demultiplexing scheme each input would get, and writes nothing.

## Inputs and outputs

Anything ProteoWizard reads: Thermo `.raw`, mzML (gzipped too), mzXML, mz5, mzMLb, MGF, and on Windows
the other vendors' formats (Bruker `.d`, SCIEX `.wiff`/`.wiff2`, Waters, Agilent, Shimadzu; Waters
from a package only with its license key in `MASSLYNX_LICENSE_KEY`, [building](building.md#packages)). Inputs
follow `-i` or stand alone; several may follow one `-i`.

Each input's output is `<stem>.centrix.mzML`, beside the input or in the `-o` directory; with a
single input, `-o file.mzML` names it. An output that exists is skipped unless `--overwrite`. Each is
written as `<name>.partial` and renamed when complete, so an interrupted run leaves no file that
looks finished.

The mzML is indexed, its m/z and times 64-bit and its intensities 32-bit, zlib-compressed
(`--no-zlib`: uncompressed). Every spectrum of the input is written, in order, including any left
empty by centroiding (DIA-NN needs them all). Its software list names Centrix, and its processing
record names the centroider's settings (sigma, grid, lambda, solver, algorithm version).

A centroid's intensity is the sum of its peak's profile samples (the Gaussian's area over the point
spacing), about 3.4 times the peak height a Stellar vendor centroid reports, so absolute
intensities differ between the two.

## What happens to each spectrum

`--centroid` decides who centroids a profile spectrum:

| Mode | A linear ion trap's profile | Any other analyzer's profile | Already centroided |
|---|---|---|---|
| `auto` (default) | Centrix | the vendor's centroids (pwiz's wavelet picker for a file with none) | as read |
| `centrix` | Centrix | Centrix | as read |
| `vendor` | the vendor's centroids | the vendor's centroids | as read |

A spectrum is an ion trap's when its instrument configuration names an ion trap analyzer; when its
file names none, when its points are spaced as a linear ion trap scan rate spaces them. `vendor` is
the baseline to compare Centrix with: the same file written the same way, centroided by Thermo.

## The Stellar's zoom-scan label

The Stellar cannot perform a zoom scan, but its `.raw` files mark every spectrum it scans at
200 kTh/s, its fastest rate, as one: the format has room for only so many scan options, and Thermo
stored that rate in the zoom flag. Its DIA MS2 and its Adaptive RT alignment spectra alike have "Z" in
their filter strings (`ITMS + p NSI Z ms2 ...`), and ProteoWizard reports each as a zoom scan
(MS:1000497). On a run whose instrument model is a Stellar, Centrix takes the label off every
spectrum: the term goes, the filter string loses its "Z" (`ITMS + p NSI ms2 ...`), and the processing
record names the correction. The log says how many were relabelled. `--keep-zoom-label` writes them
as the file has them.

The Adaptive RT alignment scans are spectra for the instrument's own retention time alignment, not
for a search, 20 or 50 m/z wide where its DIA windows are never wider than 10. Since both were
labelled alike, they are told apart by width: an MS2 wider than `--adaptive-rt-width` is one. By default
(`--adaptive-rt exclude`) they are left out of the output; `--adaptive-rt include` writes them,
centroided like the rest and passed through any demultiplexing unchanged. The log says how many were
left out. Runs from other instruments are not touched by either.

## Options

| Option | Default | Meaning |
|---|---|---|
| `-i, --input <file>...` | | Input files |
| `-o, --output <path>` | beside each input | Output directory, or an `.mzML` for one input |
| `--overwrite` | | Replace existing outputs |
| `--filter <spec>` | | A ProteoWizard spectrum filter in msconvert's syntax (repeatable), applied after reading |
| `--no-zlib` | | Uncompressed binary arrays |
| `--centroid auto\|centrix\|vendor` | `auto` | See above |
| `--sigma <Th>` | from the run | The peak sigma of every MS level |
| `--sigma-ms1 <Th>`, `--sigma-ms2 <Th>` | from the run | One level's sigma, over `--sigma` |
| `--solver descent\|active-set` | `descent` | Rust Centrix's coordinate descent, or the exact optimum ([algorithm](algorithm.md#solvers)) |
| `--demux auto\|off\|staggered` | `auto` | Demultiplex staggered windows ([demultiplexing](demultiplexing.md)) |
| `--no-ions` | | Weigh the demultiplexing by intensity, not by ions from fill times |
| `--adaptive-rt exclude\|include` | `exclude` | A Stellar's Adaptive RT alignment spectra: left out, or written (centroided, not demultiplexed) |
| `--adaptive-rt-width <Th>` | 10 | A Stellar MS2 isolated wider than this is an Adaptive RT spectrum; on any run, MS2 this wide are never demultiplexed |
| `--keep-zoom-label` | | Keep the zoom-scan label (term and "Z") on the Stellar's 200 kTh/s spectra |
| `-j, --threads <n>` | every core | The output does not depend on it |
| `-q, --quiet`, `-v, --verbose` | | Errors only; more detail (the demultiplexer's batches, stack traces) |
| `--version`, `-h, --help` | | |

The lasso's settings, as Rust Centrix names them (defaults are Rust Centrix's):

| Option | Default | Meaning |
|---|---|---|
| `--lambda-factor` | 3 | The lasso penalty, in noise sigmas |
| `--signal-threshold-sigma` | 3 | Signal regions: points above baseline by this many noise sigmas |
| `--merge-gap-points` | 2 | Runs of signal this many points apart or fewer form one region |
| `--extension-points` | 3 | Points added to each side of a region |
| `--min-region-width` | 3 | Regions narrower than this many points are dropped |
| `--noise-window-da`, `--noise-step-da` | 20, 5 | The windows the noise is refined in, Th |
| `--lambda-change-threshold` | 0.2 | Regions whose penalty the refined noise moves by more than this are solved again |
| `--max-lasso-iter`, `--lasso-tol` | 2000, 1e-6 | Coordinate descent's sweep cap and absolute tolerance |
| `--min-centroid-separation` | the sigma | Centroids closer than this (Th) merge |
| `--grid-spacing` | each spectrum's | The profile point spacing, Th |
| `--grid-offset-th` | 0 | The m/z of basis position 0, Th (Rust's `--grid-offset` was a fraction of the spacing) |
| `--min-noise-sigma` | 1 | The least noise sigma the penalty is set from |

Gone with Rust Centrix's calibration: `--scan-rate` and `--n-calibration-spectra` (the sigma comes
from the point spacing; see [the peak sigma](algorithm.md#the-peak-sigma)), and `--config`.

## Calibrate

`centrix calibrate <file>` writes nothing. It settles each level's peak sigma as a conversion would,
detects the window scheme, and also measures the peak width of each level from the isolated peaks of
up to 3,000 spectra spread across the run, beside the sigma Centrix fits:

```
Centrix MS1 peak sigma 0.073 Th (67 kTh/s scan rate, from its point spacing); MS2 0.220 Th (200 kTh/s scan rate, from its point spacing)
MS2 windows: 251 into 251 bins, 1-fold (non-overlapping)
  MS1: its isolated peaks measure sigma 0.176 Th (FWHM 0.41 Th, lower quartile of 141 peaks); Centrix fits 0.073 Th, 0.42 of it
  MS2: its isolated peaks measure sigma 0.270 Th (FWHM 0.64 Th, lower quartile of 20,650 peaks); Centrix fits 0.220 Th, 0.81 of it
```

(a Stellar 2 Th run). The rule that sets the sigma from the point spacing was validated by searches
at 200 kTh/s, where the ratio is 0.81; at another scan rate this is the check that the rule suits the
data ([algorithm](algorithm.md#the-peak-sigma)).

## The log

On the 16-spectrum Stellar slice the tests use (`tests/Centrix.Tests/Data/Centrix`):

```
centrix 26.1.0, 32 threads
Opened stellar-2th-profile-slice.mzML: 16 spectra in 0 s
Centrix MS1 peak sigma 0.073 Th (67 kTh/s scan rate, from its point spacing); MS2 0.220 Th (200 kTh/s scan rate, from its point spacing)
MS2 windows: 14 into 14 bins, 1-fold (non-overlapping)
Wrote stellar-2th-profile-slice.centrix.mzML in 0 s
  16 spectra centroided by Centrix (4,698 centroids), 0 by the vendor or pwiz, 0 centroided already
```

A sigma line that says "isolated peaks measured" means the run's point spacing is no known linear
ion trap scan rate's, and the sigma was measured from its peaks instead (an older ion trap, or a
zoom scan); "no profile spectra for Centrix at this level" means Centrix centroids none of that
level's spectra (a Fusion's Orbitrap MS1).

Exit status: 0 when every input converted, 1 when any failed (the others are still converted), 2 for
a command-line mistake.

## Examples

```bash
# Stellar staggered DIA: detected and demultiplexed with no options
centrix 20261009_Stellar_4Th-SuperOverlap_TR02.raw

# Compare with Thermo's centroids of the same run
centrix run.raw -o vendor/ --centroid vendor

# A fixed sigma and the exact solver
centrix run.raw --sigma-ms2 0.22 --solver active-set

# A slice of a run, for a test fixture
centrix run.raw --filter "scanTime [600,720]" --filter "mzWindow [450,750]" -o slice.mzML
```
