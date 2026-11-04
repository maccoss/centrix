# Results: Centrix against vendor centroiding

## Thermo Stellar 2 Th DIA, searched by Osprey (2026-11-03)

Three HeLa runs (`20261009_Stellar_WWopt_2Th_400-900mz_24min_profile_TR01`-`TR03`: 2 Th windows over
400-900 m/z, 24-minute gradient, profile mode; MS1 at 67 kTh/s, MS2 at 200 kTh/s), searched two ways
with otherwise identical settings:
- **Vendor**: the `.raw` files read by Osprey directly, so with Thermo's centroids.
- **Centrix**: the same runs converted by `centrix` 2.0.0 with its defaults (MS1 sigma 0.073 Th, MS2
  0.22 Th, both from the point spacing; the zoom scan term ProteoWizard reads from the Stellar's "Z" removed).

Osprey 26.1.1.307 (ProteoWizard master 4ec653c0d4), unit resolution. Each arm predicted its library
from the human FASTA (`uniprot_human_march2026_yeastENO1_contam_ADpeps.fasta`, 20,668 proteins, about
4.4 million target precursors at 400-900 m/z) with CarafeSharp on the GPU (AlphaPeptDeep MS2,
Chronologer RT; `--ms-instrument Stellar --activation beam-CID --analyzer LIT --nce 20`), searched its
own TR02 with it, fine-tuned both models on that search (`--cor 0.8 --n-ion-min 2 --c-ion-min 2
--valid`), predicted again and searched all three runs. Counts are at 1% experiment-level FDR
(target-decoy; no entrapment was run).

| | Vendor, ±0.5 m/z | Vendor, ±1.0 m/z | Centrix, ±0.5 m/z | Centrix, ±1.0 m/z |
|---|---|---|---|---|
| Precursors | 19,265 | 24,132 | **37,532** | 36,842 |
| Peptides | 17,175 | 21,638 | **33,601** | 33,017 |
| Protein groups | 2,077 | 3,516 | **4,503** | 4,534 |
| First-pass precursors per run (1% run FDR) | 9,162-11,673 | 14,422-15,975 | 24,863-25,414 | 24,749-25,252 |

The ±0.5/±1.0 is the fragment tolerance Osprey starts calibrating from (its unit-resolution default
is ±0.5 m/z; `--fragment-tolerance 1.0` widens it). The ±1.0 searches reused the libraries the ±0.5
runs had fine-tuned, so only their searches changed.

**Fragment m/z calibration**, per run (the window searched is 3 SD, after the mean is subtracted):

| | Vendor, ±0.5 start | Vendor, ±1.0 start | Centrix, either start |
|---|---|---|---|
| Mean offset | +0.08 m/z | 0.00 m/z | 0.00 m/z |
| SD | 0.28 m/z | 0.39 m/z | 0.16-0.17 m/z |
| Window searched | ±0.85 m/z | ±1.18 m/z | ±0.49-0.52 m/z |

Thermo's centroids of these spectra scatter by about 0.4 m/z (SD). Starting from ±0.5, the
calibration saw only the errors inside ±0.5, which spread across all of it (a flat spread over ±0.5
has an SD of 0.29), so it under-measured them, found few confident peptides to calibrate from (6-16
in its first 100,000-target sample), and skipped the cross-run RT refinement. Centrix's centroids
scatter by 0.17 m/z with no offset, inside either window, so the start does not matter for them.

**RT calibration** was the same in both arms: R² 0.9991, MAD 0.10 min, a ±0.5 min window (the floor;
±0.42-0.48 computed), refined across the runs to a residual SD of 0.057 min wherever the first pass
gave the cross-run consensus enough peptides (Centrix always; the vendor arm only from ±1.0).

**Fine-tuning** on TR02: the tuning search found 9,713 target precursors to train on from vendor
centroids and 24,545 from Centrix. MS2 cosine to the held-out spectra rose from 0.909 to 0.958
(vendor) and 0.916 to 0.949 (Centrix); Chronologer's R² from 0.997 to 0.999 in both.

Run times on a 32-thread workstation with an RTX 4060 Ti: 80 min (vendor) and 41 min (Centrix) for the
whole predict-tune-search pipeline; converting the three runs with `centrix` took 467 s.

The outputs are in `D:\Stellar-Profile_Testdata\osprey-{vendor,centrix}[-tol1]\` on the lab
workstation (each with its log, predicted library and tuned model).

## Earlier searches (Osprey, ProteoWizard/pwiz#4710)

The same 2 Th runs and the lab's staggered sets, searched with a Carafe library at a Centrix sigma of
0.33 Th, before the 1.1-spacing rule:

| Thermo Stellar, HeLa, three runs per method | Experiment precursors / peptides |
|---|---|
| 2 Th windows, vendor centroids | 19,483 / 17,992 |
| 2 Th windows, Centrix | 27,673 / 25,591 |
| 4 Th windows, Centrix | 20,137 / 18,766 |
| 4 Th, 50% overlap, demultiplexed + Centrix (2 Th bins) | 25,464 / 23,504 |
| 4 Th, SuperOverlap, demultiplexed + Centrix (1 Th bins) | 26,468 / 24,537 |

The peak-sigma sweep that set the 1.1-spacing rule, and the same runs acquired in centroid mode
(47,405 precursors), are in [algorithm.md](algorithm.md#the-peak-sigma).
