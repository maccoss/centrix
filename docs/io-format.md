# Centrix I/O and File Format

## Input Requirements

Centrix accepts **profile-mode mzML** files from the Thermo Stellar linear ion
trap. The input must be:

- **mzML format** (1.1.0 or later)
- **Profile mode** (MS:1000128 "profile spectrum" CV term) — centroid-mode input
  is rejected with an error
- **Indexed mzML** (with `<indexedmzML>` wrapper) — recommended for best
  compatibility, though the reader can handle unindexed files

Both MS1 and MS2 spectra are processed. The reader streams spectra in file order
without loading the entire file into memory.

### Expected File Structure

Centrix reads from standard Thermo-exported mzML with this structure:

```xml
<?xml version="1.0" encoding="utf-8"?>
<indexedmzML ...>
  <mzML ...>
    <fileDescription>...</fileDescription>
    <referenceableParamGroupList>...</referenceableParamGroupList>
    <softwareList>...</softwareList>
    <instrumentConfigurationList>...</instrumentConfigurationList>
    <dataProcessingList>...</dataProcessingList>
    <run ...>
      <spectrumList count="69993" ...>
        <spectrum index="0" id="..." defaultArrayLength="13500">
          ...
          <cvParam ... accession="MS:1000128" name="profile spectrum"/>
          ...
          <binaryDataArrayList count="2">
            <binaryDataArray encodedLength="...">
              <cvParam ... accession="MS:1000514" name="m/z array"/>
              <cvParam ... accession="MS:1000523" name="64-bit float"/>
              <cvParam ... accession="MS:1000574" name="zlib compression"/>
              <binary>...</binary>
            </binaryDataArray>
            <binaryDataArray encodedLength="...">
              <cvParam ... accession="MS:1000515" name="intensity array"/>
              <cvParam ... accession="MS:1000523" name="64-bit float"/>
              <cvParam ... accession="MS:1000574" name="zlib compression"/>
              <binary>...</binary>
            </binaryDataArray>
          </binaryDataArrayList>
        </spectrum>
        ...
      </spectrumList>
      <chromatogramList ...>
        ...
      </chromatogramList>
    </run>
  </mzML>
  <indexList count="2">
    <index name="spectrum">
      <offset idRef="...">byte_offset</offset>
      ...
    </index>
    <index name="chromatogram">
      <offset idRef="...">byte_offset</offset>
      ...
    </index>
  </indexList>
  <indexListOffset>byte_offset</indexListOffset>
  <fileChecksum>sha1_hex</fileChecksum>
</indexedmzML>
```

### Metadata Extracted from Input

- **Native ID** (`id` attribute) — preserved in output and index
- **MS level** — determines which σ and basis to use
- **Retention time** — from scan start time CV param
- **Filter string** — from scan acquisition metadata; used for σ auto-calibration
- **m/z and intensity arrays** — decoded from base64/zlib binary data

---

## Output Format

Centrix produces a **centroided mzML** file that is a modified copy of the input.
All metadata, non-spectrum content, and chromatograms are preserved byte-for-byte
from the input. Only spectrum binary data is replaced.

### What Changes in the Output

For each spectrum:

| Element | Input (Profile) | Output (Centroided) |
|---------|----------------|---------------------|
| `defaultArrayLength` | e.g., 13500 | e.g., 474 |
| Spectrum type CV | MS:1000128 "profile spectrum" | MS:1000127 "centroid spectrum" |
| m/z binary array | 64-bit float, zlib | 64-bit float, zlib |
| intensity binary array | 64-bit float, zlib | **32-bit float**, zlib |
| `encodedLength` | original | recomputed |

**Intensity precision change**: the output intensity array is 32-bit float (4
bytes/value) instead of the input's 64-bit float. This is scientifically
appropriate — the centroid intensities represent integrated Gaussian area
(β × σ × √(2π)), not raw detector counts, and 32-bit float provides ~7
significant digits, more than sufficient. This matches the convention used by
the Thermo onboard centroider.

### What Is Preserved Unchanged

- All XML elements outside `<spectrum>` tags: file metadata, instrument
  configuration, data processing, software lists
- All spectrum metadata: scan times, precursor info, scan event details, user
  params, CV params (except the profile→centroid swap)
- Chromatograms: TIC, pump pressure, etc. — passed through byte-for-byte
- Element ordering and nesting structure

### Index and Checksum Regeneration

Because spectrum binary data changes size, byte offsets shift throughout the file.
Centrix regenerates:

1. **`<indexList>`** — new byte offsets for every spectrum and chromatogram
2. **`<indexListOffset>`** — byte position of the new index list
3. **`<fileChecksum>`** — SHA-1 hash of the entire output up to and including the
   `<fileChecksum>` opening tag (per the mzML specification)

The original index and checksum from the input are discarded.

---

## Passthrough Writer Architecture

The writer uses **quick-xml's streaming event-based parser/writer**. This approach
has several advantages:

- **Low memory**: processes one XML event at a time, no DOM tree in memory
- **Byte-faithful**: non-spectrum content is passed through without
  re-serialization, preserving exact formatting, whitespace, and attribute ordering
- **Single-pass**: reads input and writes output simultaneously

### Implementation Details

```
  Input File                           Output File
     │                                      ▲
     ▼                                      │
  quick-xml                             quick-xml
   Reader    ─── XML Event Stream ───    Writer
                       │
              ┌────────┴────────┐
              │  Is this a       │
              │  <spectrum>?     │
              └────┬───────┬────┘
                   │       │
                  YES      NO
                   │       │
              ┌────▼───┐   └──► pass through unchanged
              │ Rewrite │
              │ binary  │
              │ arrays  │
              └─────────┘
```

A `HashCountWriter` wraps the output file, simultaneously:
- Tracking the cumulative byte position (for index offsets)
- Computing a rolling SHA-1 hash (for the file checksum)

This avoids a second pass over the output file for checksum computation.

---

## Binary Data Encoding

### Encoding Pipeline

Centroid data is encoded for the mzML binary arrays:

```
CentroidResult.mz  → f64 little-endian bytes → zlib compress → base64 encode
CentroidResult.intensity → cast to f32 → f32 LE bytes → zlib → base64
```

### Decoding the Output

Standard mzML tools (ProteoWizard, mzdata, pyteomics, etc.) can read the output
directly. The CV terms are standard:

| Array | CV Accession | Encoding |
|-------|-------------|----------|
| m/z | MS:1000514 | 64-bit float (MS:1000523), zlib (MS:1000574) |
| intensity | MS:1000515 | 32-bit float (MS:1000521), zlib (MS:1000574) |

---

## Streaming and Memory Usage

Centrix processes files in streaming fashion:

- The **reader** iterates over spectra one at a time (via mzdata's streaming
  MzMLReader)
- The **writer** emits XML events as they are produced, with BufWriter for
  efficient disk I/O
- Batches of 256 spectra are held in memory for parallel centroiding

Peak memory usage is approximately:
- ~256 × (profile spectrum size) for the batch buffer
- ~2× the largest spectrum for per-thread working buffers
- Constant overhead for the Gram matrix, basis precompute, XML parser state

For a typical Stellar file (13,500 points/MS1, 10,400 points/MS2), this is well
under 1 GB.

---

## File Size

Centroided output is significantly smaller than profile input because centroids are
much sparser than the full profile. Typical compression ratios:

| Input | Output | Ratio |
|-------|--------|-------|
| 6.1 GB (profile, 70K spectra) | ~700 MB (centroided) | ~9:1 |

The exact ratio depends on the number of centroids produced (controlled by
`lambda_factor` and `signal_threshold_sigma`).
