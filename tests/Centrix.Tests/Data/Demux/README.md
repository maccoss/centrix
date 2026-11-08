# Demultiplexing test fixture: Thermo Stellar SuperOverlap profile spectra

Used by `StellarDemuxFixtureTest` (the joint solve against Osprey's golden) and
`CliTest.TestCentrixDemuxConversion` (the whole `centrix` command). What the solve is:
`docs/demultiplexing.md`.

| File | What it is |
|---|---|
| `stellar-superoverlap-profile-slice.mzML` | Thermo Stellar SuperOverlap, profile: 200 MS2 spectra, 8 windows of 4 Th in four passes 1 Th apart (centers 600.52-607.53 m/z), 10.0-12.0 min, fragments 450-750 m/z on the 0.2 Th grid, zero points present, with fill times and the ion trap analyzer |
| `stellar-superoverlap-joint-demux.golden.tsv` | The staggered joint solve, split and Centrix (`Osprey.DemuxTool --scheme staggered --profile --joint --ions`): 800 spectra in 11 bins of 1 Th. One line per output spectrum: its index, the index of the acquired spectrum it came from, its bin, its peak count, summed intensity and intensity-weighted mean m/z |
| `stellar-superoverlap-joint-demux-sigma038.golden.tsv` | The same at MS2 sigma 0.38 Th, which measures fragment m/z best (`--peak-sigma-th 0.38`) |

Source run: `20261009_Stellar_WWopt_4Th-SuperOverlap_400-900mz_24min_profile_TR02.raw` (Thermo Stellar, HeLa,
MacCoss lab). Cut with `Osprey.DemuxTool`, which read the `.raw` through ProteoWizard as Centrix does and wrote the
spectra as read:

```
Osprey.DemuxTool --in 20261009_Stellar_WWopt_4Th-SuperOverlap_..._profile_TR02.raw --out stellar-superoverlap-profile-slice.mzML
    --scheme none --profile --filter "scanTime [600,720]"
    --filter "mzPrecursors [600.523,601.523,602.524,603.524,604.525,605.525,606.526,607.526] target=isolated"
    --filter "mzWindow [450,750]"
```

Centrix always centroids what it writes, so a new profile fixture is cut with msconvert: the same filters, no peak
picking.

The goldens were written by Osprey's `DemuxFixtureTest.TestDemuxStellarFixture` (ProteoWizard/pwiz#4710, commit
057028daba at the default MS2 sigma of three point spacings, dff91a4b69 at 0.38 Th), whose reader kept each fill
time as a 32-bit float; the test rounds them the same way. They are rewritten only by
`CENTRIX_REBLESS_DEMUX_FIXTURE=1`, which fails the test so the diff is reviewed.
