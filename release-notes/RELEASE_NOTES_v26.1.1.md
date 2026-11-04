# Centrix v26.1.1

A patch release: `--centroid vendor` writes Thermo's own centroids again on Stellar files and under filters.

## Bug fixes

- `--centroid vendor` now writes Thermo's own centroids, the same as msconvert's `peakPicking vendor`, on Stellar
  files and under any `--filter`. In 26.1.0 the vendor's peak picker was put over Centrix's own wrappers (the Stellar
  zoom-label fix, the Adaptive RT filter, the filters), could not see the Thermo reader beneath them, and fell back to
  ProteoWizard's wavelet picker (CWT) without saying so. On a Stellar run (1,519 spectra checked) the output now
  matches msconvert's Thermo centroids spectrum for spectrum. In `auto` mode, the profile spectra Centrix does not
  centroid (a Fusion or Eclipse's Orbitrap MS1) get Thermo's centroids under filters too, and when the ion trap MS2
  around them are demultiplexed.
