// Regression test: an empty spectrum must not shift the results written after it.
//
// The passthrough writer rewrites each <spectrum> of the input with the result
// at the same position. When the reader skipped spectra with no data points,
// every later spectrum was written with the next spectrum's centroids.

use base64::Engine;
use centrix::Config;
use clap::Parser;
use flate2::write::ZlibEncoder;
use flate2::Compression;
use mzdata::io::mzml::MzMLReader;
use std::fs::File;
use std::io::{BufReader, Write};

const SPACING: f64 = 0.2;
const SIGMA: f64 = 0.33;

/// A profile spectrum's points: a Gaussian peak at `center` on a 0.2 Th grid.
fn profile(center: f64) -> (Vec<f64>, Vec<f32>) {
    let mz: Vec<f64> = (0..101)
        .map(|i| center - 10.0 + i as f64 * SPACING)
        .collect();
    let intensity = mz
        .iter()
        .map(|m| (10_000.0 * (-0.5 * ((m - center) / SIGMA).powi(2)).exp()) as f32)
        .collect();
    (mz, intensity)
}

/// Little-endian values, zlib-compressed and base64-encoded, as msconvert writes them.
fn encode<T: Copy>(values: &[T], to_le: impl Fn(T) -> Vec<u8>) -> String {
    let bytes: Vec<u8> = values.iter().flat_map(|&v| to_le(v)).collect();
    let mut zlib = ZlibEncoder::new(Vec::new(), Compression::default());
    zlib.write_all(&bytes).unwrap();
    base64::engine::general_purpose::STANDARD.encode(zlib.finish().unwrap())
}

fn spectrum_xml(index: usize, mz: &[f64], intensity: &[f32]) -> String {
    let mz_b64 = encode(mz, |v: f64| v.to_le_bytes().to_vec());
    let int_b64 = encode(intensity, |v: f32| v.to_le_bytes().to_vec());
    format!(
        r#"      <spectrum index="{index}" id="scan={scan}" defaultArrayLength="{n}">
        <cvParam cvRef="MS" accession="MS:1000580" name="MSn spectrum" value=""/>
        <cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="2"/>
        <cvParam cvRef="MS" accession="MS:1000128" name="profile spectrum" value=""/>
        <scanList count="1">
          <cvParam cvRef="MS" accession="MS:1000795" name="no combination" value=""/>
          <scan>
            <cvParam cvRef="MS" accession="MS:1000016" name="scan start time" value="{rt}" unitCvRef="UO" unitAccession="UO:0000031" unitName="minute"/>
          </scan>
        </scanList>
        <binaryDataArrayList count="2">
          <binaryDataArray encodedLength="{mz_len}">
            <cvParam cvRef="MS" accession="MS:1000523" name="64-bit float" value=""/>
            <cvParam cvRef="MS" accession="MS:1000574" name="zlib compression" value=""/>
            <cvParam cvRef="MS" accession="MS:1000514" name="m/z array" value="" unitCvRef="MS" unitAccession="MS:1000040" unitName="m/z"/>
            <binary>{mz_b64}</binary>
          </binaryDataArray>
          <binaryDataArray encodedLength="{int_len}">
            <cvParam cvRef="MS" accession="MS:1000521" name="32-bit float" value=""/>
            <cvParam cvRef="MS" accession="MS:1000574" name="zlib compression" value=""/>
            <cvParam cvRef="MS" accession="MS:1000515" name="intensity array" value="" unitCvRef="MS" unitAccession="MS:1000131" unitName="number of detector counts"/>
            <binary>{int_b64}</binary>
          </binaryDataArray>
        </binaryDataArrayList>
      </spectrum>
"#,
        scan = index + 1,
        n = mz.len(),
        rt = 1.0 + index as f64 * 0.01,
        mz_len = mz_b64.len(),
        int_len = int_b64.len(),
    )
}

/// A profile mzML of MS2 spectra, each a peak at the given m/z or, for `None`,
/// no data points at all.
fn write_mzml(path: &std::path::Path, peaks: &[Option<f64>]) {
    let spectra: String = peaks
        .iter()
        .enumerate()
        .map(|(i, peak)| match peak {
            Some(center) => {
                let (mz, intensity) = profile(*center);
                spectrum_xml(i, &mz, &intensity)
            }
            None => spectrum_xml(i, &[], &[]),
        })
        .collect();
    let xml = format!(
        r#"<?xml version="1.0" encoding="utf-8"?>
<mzML xmlns="http://psi.hupo.org/ms/mzml" version="1.1.0">
  <cvList count="2">
    <cv id="MS" fullName="Proteomics Standards Initiative Mass Spectrometry Ontology" version="4.1.0" URI="https://raw.githubusercontent.com/HUPO-PSI/psi-ms-CV/master/psi-ms.obo"/>
    <cv id="UO" fullName="Unit Ontology" version="09:04:2014" URI="https://raw.githubusercontent.com/bio-ontology-research-group/unit-ontology/master/unit.obo"/>
  </cvList>
  <fileDescription>
    <fileContent>
      <cvParam cvRef="MS" accession="MS:1000580" name="MSn spectrum" value=""/>
    </fileContent>
  </fileDescription>
  <softwareList count="1">
    <software id="test" version="1">
      <cvParam cvRef="MS" accession="MS:1000799" name="custom unreleased software tool" value="test"/>
    </software>
  </softwareList>
  <instrumentConfigurationList count="1">
    <instrumentConfiguration id="IC1">
      <cvParam cvRef="MS" accession="MS:1000031" name="instrument model" value=""/>
    </instrumentConfiguration>
  </instrumentConfigurationList>
  <dataProcessingList count="1">
    <dataProcessing id="dp">
      <processingMethod order="0" softwareRef="test">
        <cvParam cvRef="MS" accession="MS:1000544" name="Conversion to mzML" value=""/>
      </processingMethod>
    </dataProcessing>
  </dataProcessingList>
  <run id="test" defaultInstrumentConfigurationRef="IC1">
    <spectrumList count="{count}" defaultDataProcessingRef="dp">
{spectra}    </spectrumList>
  </run>
</mzML>
"#,
        count = peaks.len(),
    );
    std::fs::write(path, xml).unwrap();
}

#[test]
fn empty_spectrum_does_not_shift_later_results() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("empty.mzML");
    let peaks = [Some(300.0), None, Some(500.0), Some(700.0)];
    write_mzml(&input, &peaks);

    let config = Config::try_parse_from([
        "centrix",
        "-i",
        input.to_str().unwrap(),
        "-o",
        dir.path().to_str().unwrap(),
        "--sigma",
        "0.33",
        "--grid-spacing",
        "0.2",
        "--quiet",
    ])
    .unwrap();
    centrix::run(&config).unwrap();

    let output = File::open(dir.path().join("empty.centrix.mzML")).unwrap();
    let spectra: Vec<_> = MzMLReader::new(BufReader::new(output)).collect();
    assert_eq!(spectra.len(), peaks.len());
    for (spectrum, peak) in spectra.iter().zip(peaks) {
        let mzs: Vec<f64> = spectrum
            .arrays
            .as_ref()
            .map(|arrays| arrays.mzs().unwrap().to_vec())
            .unwrap_or_default();
        match peak {
            // Each spectrum's centroids are its own peak's, none from its neighbors.
            Some(center) => {
                assert!(!mzs.is_empty(), "no centroids for the peak at {center}");
                assert!(
                    mzs.iter().all(|m| (m - center).abs() < 1.0),
                    "centroids {mzs:?} written for the peak at {center}"
                );
            }
            None => assert!(
                mzs.is_empty(),
                "centroids {mzs:?} written to the empty spectrum"
            ),
        }
    }
}
