# Centrix v26.2.0

A feature release: a linear ion trap's MS2 is centroided at three point spacings, which identified 14% to 16% more
precursors in Stellar searches than 26.1's 1.1 spacings.

## Changes to centroids

- **MS2 is fitted at three point spacings on a linear ion trap, not 1.1** (0.6 Th at the Stellar's 200 kTh/s, was
  0.22 Th). Searched at matched entrapment FDP, the Stellar's 2 Th and 50% overlap runs identify 14% to 16% more
  precursors at 0.6 Th than at 0.22 Th, and identifications plateau from 0.6 to 0.85 Th in SuperOverlap runs too.
  MS1 stays at 1.1 spacings (0.073 Th at 67 kTh/s), which measures precursor isotopes best of the sigmas tried.
  Files centroided before and after this change differ.
- **Detecting peptides and measuring them want different sigmas.** About two spacings (`--sigma-ms2 0.38` on a
  Stellar) measures fragment m/z best, keeping each fragment's M+1 isotope apart; three spacings merges most of
  them. See docs/algorithm.md.

## Breaking changes

- The default MS2 sigma changed, and with it the default `--min-centroid-separation`, which is the sigma: on a
  Stellar, MS2 centroids closer than 0.6 Th now merge (0.22 Th before). To keep 26.1's MS2 sigma, pass
  `--sigma-ms2 0.22` on a Stellar; on another linear ion trap, 0.37 times the MS2 sigma `centrix calibrate` reports.
