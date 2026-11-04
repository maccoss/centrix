# Demultiplexing staggered Stellar DIA

The Thermo Stellar's staggered methods acquire each offset window set in its own pass over the
precursor range: `4Th-50pctOverlap` (k=2, 2 Th bins) and `4Th-SuperOverlap` (k=4, 1 Th bins) in the
2026-10-09 window-width HeLa set, about 1.2 s a pass mid-run. Every narrow bin is seen in every
pass, through a different window. Centrix demultiplexes such a run and centroids it in one solve,
from the profile: each MS2 spectrum becomes one centroided spectrum per narrow bin its window
covers, under that bin's isolation window, with an id ending `demux=<n>` (its place among its
window's bins), as msconvert's demultiplex filter writes them. MS1 spectra are centroided by Centrix
as in any run.

| Thermo Stellar, HeLa, three runs per method | Experiment precursors / peptides |
|---|---|
| 4 Th windows, profile + Centrix, no demultiplexing | 20,137 / 18,766 |
| **4 Th 50% overlap, joint solve + Centrix (2 Th bins)** | **25,464 / 23,504 (+26%)** |
| **4 Th SuperOverlap, joint solve + Centrix (1 Th bins)** | **26,468 / 24,537 (+31%)** |
| 2 Th windows, profile + Centrix | 27,673 / 25,591 |

Searched by Osprey with a Carafe library at 1% precursor q (ProteoWizard/pwiz#4710), at a Centrix
sigma of 0.33 Th, before the 1.1-spacing rule; the joint solve at that rule is not yet searched,
though it scores at least as well on the truth benchmark below. Staggered 4 Th windows recover most
of what 2 Th windows give, at a 4 Th acquisition's cycle time.

## Using it

`--demux auto` (the default) demultiplexes a run when its first 5,000 MS2 spectra's windows form a
stagger (`DemuxSchemeDetector`: some narrow bin covered by more than one window, beyond the thin
margins ordinary DIA windows share) and Centrix centroids its MS2 (a linear ion trap's profile: the
joint solve fits peaks on a linear grid). `--demux off` only centroids; `--demux staggered` insists,
and fails if either condition does not hold. Overlapping windows on centroided or non-ion-trap data
are centroided and not demultiplexed, with a warning: msconvert's demultiplex filter and Osprey's
`--demux auto` handle those.

```
centrix run_4Th-SuperOverlap.raw
...
MS2 windows: 8 into 11 bins, 4-fold (overlapping)
Scheme: overlapping, 8 windows into 11 bins, 4-fold
Demultiplexing: 800 spectra out, indexed in 1 s
Profile grid: step 0.2000 Th; peak sigma 0.220 Th = 1.10 samples
```

(the SuperOverlap test slice, `tests/Centrix.Tests/Data/Demux`).

## The joint solve

`StaggeredDemuxPlan` with `JointDemultiplexer` on a `LinearGrid` (`src/Centrix.Demux`):

1. **Passes.** A window below the one before starts a new pass; each pass is scaled to ions at the
   median fill time of its spectra.
2. **Pseudo-sweeps.** For each pass of a block, every window covering the unit's bins: the pass's own
   acquisitions as measured, the others interpolated (makima, grid point by grid point, in rates) to
   the pass's time, so every window is seen at once.
3. **The fit.** Each bin's Gaussian peaks on the 0.2 Th grid, at the MS2 peak sigma
   ([algorithm](algorithm.md#the-peak-sigma): 1.1 point spacings, 0.22 Th), Poisson-weighted,
   with a z-scaled lasso.
4. **The split.** Each acquired spectrum's measured profile is split among the bins its window covers
   by those bins' fitted peaks at each grid point (equally where none was fitted). The fit decides the
   split; the measurements are what is written.
5. **Centroiding.** Each bin's share in each pass is centroided by Centrix, in the pass's intensity
   units.
6. **Layout.** Each acquired MS2 becomes one spectrum per bin of its window, with that bin's centroids
   from the spectrum's own pass, at the spectrum's intensity scale.

The fit runs with the staggered settings (`JointDemuxParams.UseStaggeredDefaults`): each pass of a
unit starts from the previous pass's solve carried to its own ion scale, the first descent stops at
a relative decrease of 1e-3, and the active set stops growing after a round whose additions promise
at most 1e-4 of the objective. On a 2-minute SuperOverlap slice these took 21% less time and came
closer to the converged solve (0.88% of its intensity unmatched, against 1.02%); on three whole runs
the detections were level.

Blocks of cycles are read in order, a batch at a time, and solved in parallel, the largest solves
first, while the next batch is read.

## Ions from the fill time

An ion trap's intensities are rates: a spectrum's ions are its intensity x its fill (injection)
time over the analyzer's ion response alpha, I = alpha x N / t (`IonResponse`; for an ion trap 0.98,
the median of the Stellar's 0.97, the Ascend's 0.98 and the Fusion Lumos's 1.09, from Table 1 of Hsu
et al., *J. Proteome Res.* 2025, 24, 5742). The analyzer is read from each spectrum's instrument
configuration, the fill time from its scan. Spectra are combined as rates, the only unit that
compares across fill times, and written in the instrument's units; the ions set only the Poisson
weights and thresholds. Weighing every observation by its own ions instead of one scale matters: the
per-channel staggered solve once combined spectra of different fill times unconverted, and at
100-fold fill times that moved its output by up to 37% of the largest rate (now 0.04%,
`StaggeredJointDemuxTest.TestStaggeredChannelFillTimeInvariance`). `--no-ions` weighs by intensity
alone.

## Adaptive RT spectra

The Stellar stores its Adaptive RT alignment spectra among its DIA spectra, both scanned at
200 kTh/s and so both labelled zoom scans, the flag its `.raw` files store that rate in (Centrix takes
the label off, [usage](usage.md#the-stellars-zoom-scan-label)), so the label identifies nothing. Adaptive RT spectra are 20 or 50 m/z wide and Stellar DIA windows never exceed 10 m/z, so
on a Stellar run MS2 spectra wider than `--adaptive-rt-width` (10 Th) are taken for them and, by
default, left out of the output before anything else happens ([usage](usage.md#the-stellars-zoom-scan-label)).
With `--adaptive-rt include` they are kept out of the scheme instead and passed through, centroided
by Centrix like MS1: left in, a wide window's edges would join the boundaries the scheme's bins are
cut at. On other instruments' runs the same width keeps wide MS2 out of the scheme, and nothing is
left out.

## Validation

**Against Osprey.** `StellarDemuxFixtureTest` runs the joint solve on 200 real SuperOverlap profile
spectra (8 windows of 4 Th in four passes, 11 bins of 1 Th) and holds its 800 output spectra to the
golden Osprey.Demux wrote for them, peak counts exactly and summed intensities to 1e-9. Osprey's
reader keeps fill times as 32-bit floats, and the golden carries that: the solve's choice of peaks
is sensitive enough that full-precision fill times move the first bin's intensity by 0.1% and the
next bin's peak count by two. Centrix uses full precision, as `Osprey.DemuxTool` did for the searches
above. `CliTest.TestCentrixDemuxConversion` runs the same slice through the whole command and checks
what it writes against the solve.

**Against truth.** A 2 Th run's narrow-window spectra summed into k-fold staggered windows (each
window's ions added through its own fill time), demultiplexed, and compared with the narrow spectra
centroided the same way: cosine on a 0.05 Th grid blurred by 0.3 Th over 6,000 spectra (median,
ion-weighted median, 10th percentile; pwiz-ai `scripts/Osprey/Demux/eval_remux.py`). It is
pessimistic on sampling: a summed window returns every k x 2.4 s, against 4.7 s for the real
SuperOverlap's, and is twice as wide.

| k | No demultiplexing (median) | Makima split + Centrix | Joint split + Centrix (sigma 0.33) |
|---|---|---|---|
| 2 | 0.735 | 0.979 / 0.983 / 0.877 | 0.982 / 0.985 / 0.926 |
| 3 | 0.507 | 0.950 / 0.961 / 0.706 | 0.959 / 0.967 / 0.798 |
| 4 | 0.389 | 0.917 / 0.934 / 0.603 | 0.930 / 0.942 / 0.684 |

At the 1.1-spacing sigma (0.22 Th) the joint solve scores 0.985 at k=2 and 0.939 at k=4 (median),
against 0.982 and 0.930 at 0.33 Th. Writing the fitted peaks instead of splitting the measurements
scores 0.944 at k=2: the fit carries the errors of the interpolated windows, and Centrix centroids
the split measurements more stably.

**Determinism.** Blocks are independent and each writes only the output it owns; the source is read
from one thread; the output is the same at any thread count. Every exp and log in
`Centrix.Demux` comes from `PortableMath`, built from IEEE arithmetic alone rather than the operating
system's C library (Windows and Linux disagree in the last bit for about 0.5% of arguments, which an
iterative solve turns into visible differences), and the joint solve's peak loops reduce in one fixed
order. The SuperOverlap slice demultiplexes to the same 800 spectra on Windows and Linux.

## Speed

The joint solve is the slow part: a whole SuperOverlap run took 3,741-3,958 s on 16 threads under
load in Osprey.DemuxTool.

## Not done here

- **Other demultiplexers.** Orbitrap and Astral staggered data, SCIEX ZT Scan and Thermo MSX are
  demultiplexed by msconvert, by Osprey's `--demux auto`, and by `Osprey.DemuxTool`; this solve is
  for linear ion trap profile data only. The engine it shares with them (`ScanningDemultiplexer`,
  `JointDemultiplexer`) is kept whole, so the code can follow Osprey's.
- **Per-acquisition ion response.** One alpha serves each analyzer.
