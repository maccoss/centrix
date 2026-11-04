# The Centrix algorithm

A linear ion trap (the Thermo Stellar, the ion trap of a Fusion or an Eclipse) has unit
resolution: a fragment's profile peak is 0.4 to 0.8 Th wide at half maximum. The vendor's
centroider places one centroid per local maximum, so two fragments less than about 1 Th apart come
out as one centroid at an m/z that belongs to neither. Centrix fits each region of the profile with
Gaussian peaks of the instrument's width, one candidate per profile point, by a non-negative lasso,
and keeps the few that explain the signal.

The code is in `src/Centrix.Core`:

| File | Role |
|---|---|
| `CentrixCentroider.cs` | the two passes, the merge, a basis per grid spacing |
| `CentrixParams.cs` | the settings (Rust Centrix's defaults) and the solver choice |
| `CentrixBasis.cs` | the Gaussian basis, its Gram row, the basis positions |
| `NonNegativeLasso.cs` | coordinate descent, the active-set solve, the sub-grid refinement |
| `NoiseProfile.cs`, `SignalRegion.cs` | the noise estimates and the signal regions |
| `CentrixCalibration.cs` | scan rates and a run's peak sigma |

## Per spectrum

1. **Rough noise.** The baseline is the 10th percentile of all intensities; the noise sigma the
   interquartile range / 1.349, at least a tenth of the baseline and at least 1
   (`NoiseProfile.Rough`). The floor of 1 assumes an instrument's intensity units, an ion trap's
   rates: feed it intensities, not ions.
2. **Signal regions.** Runs of points above baseline + 3 noise sigmas, runs at most 2 points apart
   joined, each extended by 3 points a side, regions under 3 points dropped (`SignalRegion.Detect`).
3. **Pass 1.** Each region is fitted on basis positions every grid spacing, one beyond each end:
   `A[i, j] = exp(-(mz_i - grid_j)^2 / (2 sigma^2))`, the non-negative lasso
   `min 1/2 |y - A b|^2 + lambda |b|_1, b >= 0` with lambda = 3 rough noise sigmas. The Gram matrix
   is the analytic Toeplitz one of an infinite grid (`CentrixBasis.GramRow`): only its first row is
   kept, entry (i, j) being `row[|i - j|]`, a dozen or so values that sit in L1 cache.
4. **Noise refinement.** In 20 Th windows stepped 5 Th, the root mean square of pass 1's residuals
   and of the points outside every region, smoothed by a 3-point median (`NoiseProfile.Refine`).
5. **Pass 2.** Regions whose lambda the refined noise changes by more than 20% are solved again,
   from their pass-1 solution, with their pass-1 A'y (the grid has not changed, only lambda); their
   centroids replace pass 1's.
6. **Centroids.** Every coefficient above zero is a centroid, moved between grid points to the
   vertex of the parabola through the logarithms of it and its neighbors when both are above zero.
   Its intensity is the coefficient times `sigma sqrt(2 pi) / spacing`, the sum of its peak's
   samples. Centroids closer than sigma merge, intensity-weighted.

The grid spacing is each spectrum's median point gap, and a basis is kept per spacing, so a run
whose levels (or scan types) are sampled differently is fitted on each one's own grid.

In the executable (`src/Centrix`), spectra are read in file order on one thread, 256 at a time, and
each batch is centroided on every thread while the next is read: a Thermo `.raw` serves one request
at a time, and centroiding costs several times the read. Each spectrum is centroided on its own, so
the output does not depend on the thread count or the batch size
(`CliTest.TestCentrixSpectrumListInvariance`).

## The peak sigma

**On a linear ion trap the sigma is 1.1 point spacings** (`CentrixCalibration.SIGMA_PER_SPACING`).
The firmware fixes the profile point spacing per scan rate, so the spacing identifies the rate
(`LinearIonTrapScanRate.FromSpacing`), and a level's first profile spectrum of 100 points or more
settles its sigma.

| Scan rate | Point spacing | Sigma | Broadest peak (FWHM) | Measured on the Stellar (FWHM lower quartile) |
|---|---|---|---|---|
| 33 kTh/s | 1/30 Th | 0.037 Th | 0.5 Th | |
| 67 kTh/s | 1/15 Th | 0.073 Th | 0.6 Th | MS1: 0.40 Th |
| 125 kTh/s | 1/8 Th | 0.138 Th | 0.8 Th | |
| 200 kTh/s | 1/5 Th | 0.220 Th | 2.0 Th | MS2: 0.63 to 0.72 Th |

**Why a little over one point spacing.** Centrix is there to split a peak wider than one species,
and splitting a little too much costs little. The rule comes from one scan rate: the Stellar's 2 Th
profile runs (three HeLa runs, MS2 at 200 kTh/s), centroided at fixed MS2 sigmas (MS1 0.2 Th in
every arm) and searched by Osprey with a library fine-tuned on these runs and carrying natural
entrapment:

| MS2 sigma | Experiment precursors at q 1% | Entrapment FDP (combined) |
|---|---|---|
| 0.12 Th | 33,725 | 0.27% |
| 0.15 Th | 34,604 | 0.34% |
| 0.18 Th | 34,621 | 0.32% |
| 0.20 Th (1 spacing) | 35,077 | 0.34% |
| 0.21 Th | 35,849 | 0.51% |
| **0.22 Th (1.1 spacings, the default)** | **36,001** | 0.53% |
| 0.24 Th | 35,680 | 0.52% |
| 0.27 Th | 35,116 | 0.63% |
| 0.30 Th | 34,330 | 0.51% |
| 0.33 Th | 32,526 | 0.53% |
| 0.36 Th | 32,297 | 0.47% |

The same three runs acquired in centroid mode give 47,405 (0.49%); why profile + Centrix stays
below the instrument's own centroids is open. Identifications plateau from 0.21 to 0.24 Th (1.05 to
1.2 spacings) and fall on both sides. At one spacing and below, the search's calibrated MS2 window
narrows from 0.48 Th to 0.31-0.35 Th and identifications drop; by 0.18 Th the centroids per MS2
spectrum have risen from about 260 to over 400. A wide sigma merges a fragment with its M+1
isotope into one centroid between them, which mass calibration reads as an offset and a wider
window (0.48 Th at sigma 0.21, 1.04 at 0.33). Identifications alone cannot place the rule finely,
since unit-resolution scoring barely penalizes splitting.

**Not a measured width.** A linear ion trap's peak width depends on the ions in the peak. The
Stellar's MS2 widths' lower quartile is 0.71 to 0.74 Th at every m/z from 200 to 1200, but by ions
in the peak it rises from 0.41 Th under 3 ions to 0.9 Th over 300, with no plateau: few ions arrive
as short pulses, many spread by space charge and merge with unresolved neighbors. A measured sigma is
therefore set by which peaks are counted (0.27 to 0.33 Th over a whole 2 Th run, depending), and all
are wider than the identifications' optimum.

**Other scan rates are untested by searches.** The rule rests on 200 kTh/s MS2 alone. What is known
of the others:
- *67 kTh/s* (the Stellar's MS1, the only real data at another rate): the rule gives 0.073 Th where
  the peaks measure 0.176 Th (`centrix calibrate` on TR02, 141 isolated peaks), 0.42 of it. On real
  spectra that splits little: 4 MS1 spectra of TR01 give a median of 462 centroids at 0.073 Th, 250 at
  0.13 Th and 171 at 0.17 Th, against Thermo's 422. For Osprey's search it did not matter (MS1 at
  0.073 Th gives exactly the 36,001 precursors of MS1 at 0.2 Th, unit-resolution scoring having no
  MS1 features); for MS1 quantification it is untested.
- *Simulated Gaussian peaks* (`TestCentrixSeparatesCloseSignals`, and a probe at four true widths):
  fitted at their own sigma, every scan rate gives one centroid per isolated peak at its m/z (within
  0.025 Th) and resolves every pair 0.6 Th apart. Fitted narrower than they are, clean Gaussians split:
  at the rule's sigma, peaks of sigma 0.21 Th give 1.0 centroids each at 200 kTh/s, 4.2 at 125 kTh/s and
  10 at 67 kTh/s. Real ion trap peaks split far less than that (above), having a narrow core and broad
  wings rather than a Gaussian's shape.
- *125 and 33 kTh/s*: no real data yet. At 125 kTh/s the rule gives 0.138 Th. Run `centrix calibrate`
  on such a run: it prints each level's measured sigma beside the one Centrix fits, and their ratio
  (0.81 for the Stellar's validated MS2). A ratio far below that is a reason to try `--sigma` near the
  measured width, and searches at both are what will decide whether the optimum follows the spacing.

**Elsewhere it is measured** (a spacing no scan rate in the table has), for each MS level apart:
1. In up to 3,000 spectra spread evenly over the run, the isolated intense peaks: local maxima
   above the spectrum's 99th percentile whose intensity falls without rising again, or reaching
   zero, to half the apex on both sides, over points with none missing between them; each one's
   full width at half maximum, the apex from the log-parabola through its top three points.
2. The sigma is the widths' lower quartile / 2.3548; with under 20 peaks, 0.34 Th.

The narrow side, because a wider peak is most often two species the instrument did not resolve; a
full width at half maximum rather than a fitted Gaussian, because these peaks have a narrow core and
broad wings and a fit's sigma grows with the points it is given (0.31 Th from three points, 0.61 from
eleven, the last what Rust Centrix's wavelet calibration reported); spikes of a few ions, which fall
to zero before half their apex, left out.

ProteoWizard's own Thermo reader test file (`Reader_Thermo_Test.data`, an LTQ Velos run from 2009,
used here only to test reading `.raw`) samples its full scans every 1/12 Th, no rate in the table,
so its sigma is measured (0.116 Th MS1, 0.086 Th MS2). Its data-dependent zoom scans (0.02 Th
points) share the MS1 sigma: one sigma serves each MS level, which suits the Stellar but not a
method mixing scan types within a level.

## The Stellar's filter strings: "Z" and Adaptive RT

Rust Centrix read the scan rate from a letter in the filter string (n, r, t, u). The Stellar's MS1
carry one ("ITMS + p NSI r Full ms", 67 kTh/s), but its MS2 carry "Z" instead
("ITMS + p NSI Z ms2 480.4683@hcd20.00 [200.0000-1200.0000]"). In a Thermo filter "Z" is a zoom scan,
and ProteoWizard reports one (MS:1000497), but the Stellar cannot perform a zoom scan: the `.raw`
format has room for only so many scan options, and Thermo stored the Stellar's fastest rate,
200 kTh/s, in the zoom flag. Every spectrum scanned at that rate carries it, DIA and Adaptive RT
alike: every DIA MS2 of the 2026-10-09 window-width HeLa runs (in the 2 Th TR01 run, 138,803 MS2 of
its 139,356 spectra; the other 553 are MS1). So:
- the scan rate comes from the point spacing, never from the filter;
- no spectrum is dropped for carrying "Z";
- on a Stellar run, every spectrum comes out without the label, the term and the filter's "Z" both
  removed and the correction named in the processing record (`StellarZoomLabelFix`;
  `--keep-zoom-label` writes them as read);
- Adaptive RT spectra, labelled like the DIA ones, are recognized by their width (20 or 50 m/z, where
  Stellar DIA never exceeds 10) and left out of the output by default ([usage](usage.md#the-stellars-zoom-scan-label)).

ProteoWizard's Thermo reader also looks, for every MS2, for a preceding zoom MS1 to take a triple-play
precursor charge from; on these runs it finds none and walks back to the run's start each time, which
makes the last spectra about 0.2 ms slower to read than the first (some 15 s over a 24-minute run).

## Solvers

The basis peaks of neighboring grid points correlate strongly (0.91 at the Stellar's MS2 spacing,
1.65 points per sigma at sigma 0.33), so coordinate descent converges slowly: on 6,000 Stellar MS2
spectra it averaged 347 sweeps per region and stopped at its 2,000-sweep cap in 6% of regions,
which then keep an unconverged solution.

`--solver active-set` (`NonNegativeLasso.SolveActiveSet`) reaches the exact optimum of the same
problem by Lawson and Hanson's active-set method: the few nonzero coefficients are solved directly
by Cholesky, stepping back to feasibility when a solve leaves one at or below zero, falling back to
coordinate descent if a factorization fails. On the same 6,000 spectra it took 18-31 s against
64-169 s for coordinate descent. It moves 8.5% of centroids by more than 1e-4 Th and changes the
count in 23% of spectra (net +0.13%): the capped solves, and through pass 1's residuals the refined
noise. Against truth the two are equal to 0.001 ([demultiplexing](demultiplexing.md#validation)).
Coordinate descent stays the default, so the output stays comparable with Rust Centrix's.

## Fidelity to Rust Centrix

The port follows Rust Centrix at 39370f1 (tag `rust-final`) with the coordinate-descent solver.
`CentrixTest.TestCentrixRustGolden` holds it there: 30 real Stellar profile spectra and the 4,836
centroids Rust Centrix gives them (`tests/Centrix.Tests/Data/Centrix`, whose README says how the
golden was made). Every spectrum must have the same number of centroids, each within 1e-3 Th and 1%;
the port matches to 1e-11 Th, and a 1.7% change to the lasso penalty fails the test.
`CliTest.TestCentrixConversionMatchesRust` holds the whole command (ProteoWizard's reading and
writing, the parallel batches) to the same golden.

Over all 6,000 spectra of the benchmark slice the port and Rust give the same 832,332 centroids;
0.02% differ by more than 1e-4, by up to 4e-4 Th and 0.4%. Deliberate differences:
- Rust summed A'y with BLAS, the port in index order, so a solve stopped by the 1e-6 absolute
  tolerance can stop a sweep apart.
- The port detects the grid spacing per spectrum (a basis per spacing); Rust took the first
  spectrum's of each MS level.
- Rust's sigma calibration (filter-string scan rate, wavelet fit) is replaced by the rule above.
- Rust's own mzML reader and byte-for-byte passthrough writer are replaced by ProteoWizard's: the
  output is a ProteoWizard mzML, every spectrum's metadata as ProteoWizard reads it, and `.raw`
  files are read directly.

The centroider computes its exponentials and logarithms with .NET's `Math`, which calls the
operating system's C library, so Windows and Linux can differ in the last bit: on ProteoWizard's
Thermo test file 6 of 85 spectra differ, by at most 2e-13 Th, with the same centroids. The demultiplexer uses
`PortableMath` throughout, and its output is identical on both.

**Changing the output.** A change that moves a centroid raises `CentrixCentroider.ALGORITHM_VERSION`,
and must still pass the Rust golden, or be one Rust Centrix made too with the golden regenerated
from it (the C# port is never used to make it).
