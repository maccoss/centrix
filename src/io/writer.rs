//! Passthrough mzML writer.
//!
//! Streams the input mzML through quick-xml, passing all XML events unchanged
//! except spectrum binary data arrays, which are replaced with centroided data.
//! Regenerates `<indexList>`, `<indexListOffset>`, and SHA-1 `<fileChecksum>`.

use crate::centroid::CentroidResult;
use crate::CentrixError;
use crate::Result;
use base64::engine::general_purpose::STANDARD as B64;
use base64::Engine;
use flate2::write::ZlibEncoder;
use flate2::Compression;
use quick_xml::events::{BytesEnd, BytesStart, BytesText, Event};
use quick_xml::Reader;
use quick_xml::Writer;
use sha1::{Digest, Sha1};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

// ── CV accessions ─────────────────────────────────────────────────────────────

const CV_PROFILE: &str = "MS:1000128";
const CV_CENTROID: &str = "MS:1000127";
const CV_MZ_ARRAY: &str = "MS:1000514";
const CV_INTENSITY_ARRAY: &str = "MS:1000515";
const CV_64BIT: &str = "MS:1000523";
const CV_32BIT: &str = "MS:1000521";

// ── Helpers ───────────────────────────────────────────────────────────────────

fn xml_err<E: std::fmt::Display>(e: E) -> CentrixError {
    CentrixError::Xml(e.to_string())
}

fn hex_encode(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        write!(s, "{b:02x}").unwrap();
    }
    s
}

fn encode_mz_array(centroids: &[CentroidResult]) -> String {
    let raw: Vec<u8> = centroids.iter().flat_map(|c| c.mz.to_le_bytes()).collect();
    let mut enc = ZlibEncoder::new(Vec::new(), Compression::default());
    enc.write_all(&raw).expect("zlib mz");
    B64.encode(enc.finish().expect("zlib finish mz"))
}

fn encode_intensity_array(centroids: &[CentroidResult]) -> String {
    let raw: Vec<u8> = centroids
        .iter()
        .flat_map(|c| (c.intensity as f32).to_le_bytes())
        .collect();
    let mut enc = ZlibEncoder::new(Vec::new(), Compression::default());
    enc.write_all(&raw).expect("zlib int");
    B64.encode(enc.finish().expect("zlib finish int"))
}

fn extract_attr(elem: &BytesStart<'_>, key: &[u8]) -> Option<String> {
    elem.attributes()
        .filter_map(|a| a.ok())
        .find(|a| a.key.as_ref() == key)
        .map(|a| String::from_utf8_lossy(&a.value).to_string())
}

fn has_accession(elem: &BytesStart<'_>, acc: &str) -> bool {
    extract_attr(elem, b"accession").is_some_and(|a| a == acc)
}

// ── SHA-1 + byte-counting writer ──────────────────────────────────────────────

/// Wraps a `Write`, tracking cumulative bytes and computing a running SHA-1.
struct HashCountWriter<W: Write> {
    inner: W,
    hasher: Sha1,
    bytes_written: u64,
    hashing: bool,
}

impl<W: Write> HashCountWriter<W> {
    fn new(inner: W) -> Self {
        Self {
            inner,
            hasher: Sha1::new(),
            bytes_written: 0,
            hashing: true,
        }
    }

    /// Cumulative bytes written (correct even with buffering).
    fn position(&self) -> u64 {
        self.bytes_written
    }

    /// Stop updating the hash (call before writing fileChecksum content).
    fn stop_hashing(&mut self) {
        self.hashing = false;
    }

    /// Clone the hasher state and finalize to hex — does not consume self.
    fn digest_hex(&self) -> String {
        hex_encode(&self.hasher.clone().finalize())
    }
}

impl<W: Write> Write for HashCountWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = self.inner.write(buf)?;
        if self.hashing {
            self.hasher.update(&buf[..n]);
        }
        self.bytes_written += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

// ── Owned cvParam data (for buffering inside binaryDataArray) ─────────────────

#[derive(Clone, Copy, PartialEq)]
enum ArrayKind {
    Mz,
    Intensity,
    Other,
}

struct CvParamOwned {
    accession: String,
    attrs: Vec<(String, String)>,
}

struct IndexEntry {
    id: String,
    offset: u64,
}

// ── PassthroughWriter ─────────────────────────────────────────────────────────

/// Streaming passthrough writer that replaces spectrum binary data arrays with
/// centroided data while preserving all other mzML content.
///
/// Call [`write_spectrum`] once per spectrum in file order, then [`finish`].
pub struct PassthroughWriter {
    reader: Reader<BufReader<std::fs::File>>,
    writer: Writer<HashCountWriter<BufWriter<std::fs::File>>>,
    buf: Vec<u8>,
    spectrum_offsets: Vec<IndexEntry>,
    chromatogram_offsets: Vec<IndexEntry>,
}

impl PassthroughWriter {
    /// Open an input mzML for reading and create the output file.
    pub fn new(input: &Path, output: &Path) -> Result<Self> {
        let src = std::fs::File::open(input).map_err(CentrixError::Io)?;
        let dst = std::fs::File::create(output).map_err(CentrixError::Io)?;

        let mut reader = Reader::from_reader(BufReader::new(src));
        reader.config_mut().trim_text_end = false;

        let writer = Writer::new(HashCountWriter::new(BufWriter::new(dst)));

        Ok(Self {
            reader,
            writer,
            buf: Vec::with_capacity(64 * 1024),
            spectrum_offsets: Vec::new(),
            chromatogram_offsets: Vec::new(),
        })
    }

    /// Write the next spectrum with the given centroided data.
    ///
    /// Pumps XML events forward — passing through non-spectrum content — until
    /// one `<spectrum>` has been fully rewritten, then returns.
    pub fn write_spectrum(&mut self, centroids: &[CentroidResult]) -> Result<()> {
        let mz_b64 = encode_mz_array(centroids);
        let int_b64 = encode_intensity_array(centroids);
        let n_centroids = centroids.len();

        // Move buf out of self to avoid borrow-checker conflict between
        // self.reader (needs &mut buf) and self.writer / self.spectrum_offsets.
        let mut buf = std::mem::take(&mut self.buf);

        loop {
            buf.clear();
            let event = self.reader.read_event_into(&mut buf).map_err(xml_err)?;

            // Spectrum start — rewrite and process body
            if let Event::Start(ref e) = event {
                if e.name().as_ref() == b"spectrum" {
                    let offset = self.writer.get_ref().position();
                    let id = extract_attr(e, b"id").unwrap_or_default();
                    self.spectrum_offsets.push(IndexEntry { id, offset });

                    let mut new_start = BytesStart::new("spectrum");
                    for attr in e.attributes().filter_map(|a| a.ok()) {
                        if attr.key.as_ref() == b"defaultArrayLength" {
                            new_start.push_attribute((
                                "defaultArrayLength",
                                n_centroids.to_string().as_str(),
                            ));
                        } else {
                            new_start.push_attribute(attr);
                        }
                    }
                    self.writer
                        .write_event(Event::Start(new_start))
                        .map_err(xml_err)?;

                    self.process_spectrum_body(&mut buf, &mz_b64, &int_b64)?;
                    self.buf = buf;
                    return Ok(());
                }
            }

            // Chromatogram start — record offset, then pass through
            if let Event::Start(ref e) = event {
                if e.name().as_ref() == b"chromatogram" {
                    let offset = self.writer.get_ref().position();
                    let id = extract_attr(e, b"id").unwrap_or_default();
                    self.chromatogram_offsets.push(IndexEntry { id, offset });
                }
            }

            // EOF before all spectra written
            if matches!(event, Event::Eof) {
                self.buf = buf;
                return Err(CentrixError::Xml(
                    "unexpected EOF before all spectra were written".into(),
                ));
            }

            // indexList encountered too early
            if let Event::Start(ref e) = event {
                if e.name().as_ref() == b"indexList" {
                    self.buf = buf;
                    return Err(CentrixError::Xml(
                        "reached indexList before all spectra were written".into(),
                    ));
                }
            }

            // Default: pass through
            self.writer.write_event(event).map_err(xml_err)?;
        }
    }

    /// Process events inside `<spectrum>…</spectrum>`, rewriting binary arrays.
    fn process_spectrum_body(
        &mut self,
        buf: &mut Vec<u8>,
        mz_b64: &str,
        int_b64: &str,
    ) -> Result<()> {
        loop {
            buf.clear();
            let event = self.reader.read_event_into(buf).map_err(xml_err)?;

            // </spectrum> — done
            if let Event::End(ref e) = event {
                if e.name().as_ref() == b"spectrum" {
                    self.writer.write_event(event).map_err(xml_err)?;
                    return Ok(());
                }
            }

            // Replace profile → centroid CV term
            if let Event::Empty(ref e) = event {
                if e.name().as_ref() == b"cvParam" && has_accession(e, CV_PROFILE) {
                    let mut cv = BytesStart::new("cvParam");
                    for attr in e.attributes().filter_map(|a| a.ok()) {
                        if attr.key.as_ref() == b"accession" {
                            cv.push_attribute(("accession", CV_CENTROID));
                        } else if attr.key.as_ref() == b"name" {
                            cv.push_attribute(("name", "centroid spectrum"));
                        } else {
                            cv.push_attribute(attr);
                        }
                    }
                    self.writer.write_event(Event::Empty(cv)).map_err(xml_err)?;
                    continue;
                }
            }

            // <binaryDataArray> — clone start tag, then rewrite
            if let Event::Start(ref e) = event {
                if e.name().as_ref() == b"binaryDataArray" {
                    let start_owned = e.to_owned();
                    self.rewrite_binary_data_array(&start_owned, mz_b64, int_b64)?;
                    continue;
                }
            }

            if matches!(event, Event::Eof) {
                return Err(CentrixError::Xml("unexpected EOF inside spectrum".into()));
            }

            self.writer.write_event(event).map_err(xml_err)?;
        }
    }

    /// Rewrite one `<binaryDataArray>` element with centroided binary data.
    fn rewrite_binary_data_array(
        &mut self,
        start_elem: &BytesStart<'_>,
        mz_b64: &str,
        int_b64: &str,
    ) -> Result<()> {
        let mut buf = Vec::new();

        // Phase 1: scan cvParams inside binaryDataArray to determine kind.
        // Stop when we reach <binary>.
        let mut kind = ArrayKind::Other;
        let mut cv_params: Vec<CvParamOwned> = Vec::new();
        let mut ws_before_binary = String::new();

        loop {
            buf.clear();
            let event = self.reader.read_event_into(&mut buf).map_err(xml_err)?;
            match event {
                Event::Start(ref e) | Event::Empty(ref e) if e.name().as_ref() == b"binary" => {
                    break;
                }
                Event::Empty(ref e) if e.name().as_ref() == b"cvParam" => {
                    let acc = extract_attr(e, b"accession").unwrap_or_default();
                    if acc == CV_MZ_ARRAY {
                        kind = ArrayKind::Mz;
                    } else if acc == CV_INTENSITY_ARRAY {
                        kind = ArrayKind::Intensity;
                    }
                    let attrs: Vec<(String, String)> = e
                        .attributes()
                        .filter_map(|a| a.ok())
                        .map(|a| {
                            (
                                String::from_utf8_lossy(a.key.as_ref()).to_string(),
                                String::from_utf8_lossy(&a.value).to_string(),
                            )
                        })
                        .collect();
                    cv_params.push(CvParamOwned {
                        accession: acc,
                        attrs,
                    });
                }
                Event::Text(ref t) => {
                    ws_before_binary = String::from_utf8_lossy(t.as_ref()).to_string();
                }
                Event::Eof => {
                    return Err(CentrixError::Xml(
                        "unexpected EOF inside binaryDataArray".into(),
                    ));
                }
                _ => {}
            }
        }

        // Phase 2: write the rewritten binaryDataArray.
        let b64_data = match kind {
            ArrayKind::Mz => mz_b64,
            ArrayKind::Intensity => int_b64,
            ArrayKind::Other => {
                // Unknown array type — skip entirely (shouldn't happen in standard mzML)
                log::warn!("Unknown binaryDataArray type — skipping");
                self.skip_past_end(&mut buf, b"binaryDataArray")?;
                return Ok(());
            }
        };

        // <binaryDataArray encodedLength="N" ...>
        let mut new_start = BytesStart::new("binaryDataArray");
        new_start.push_attribute(("encodedLength", b64_data.len().to_string().as_str()));
        for attr in start_elem.attributes().filter_map(|a| a.ok()) {
            if attr.key.as_ref() != b"encodedLength" {
                new_start.push_attribute(attr);
            }
        }
        self.writer
            .write_event(Event::Start(new_start))
            .map_err(xml_err)?;

        // cvParams (rewrite 64-bit → 32-bit for intensity)
        for cv in &cv_params {
            self.writer
                .write_event(Event::Text(BytesText::new("\n              ")))
                .map_err(xml_err)?;
            let mut elem = BytesStart::new("cvParam");
            if kind == ArrayKind::Intensity && cv.accession == CV_64BIT {
                for (k, v) in &cv.attrs {
                    if k == "accession" {
                        elem.push_attribute(("accession", CV_32BIT));
                    } else if k == "name" {
                        elem.push_attribute(("name", "32-bit float"));
                    } else {
                        elem.push_attribute((k.as_str(), v.as_str()));
                    }
                }
            } else {
                for (k, v) in &cv.attrs {
                    elem.push_attribute((k.as_str(), v.as_str()));
                }
            }
            self.writer
                .write_event(Event::Empty(elem))
                .map_err(xml_err)?;
        }

        // <binary>base64data</binary>
        if !ws_before_binary.is_empty() {
            self.writer
                .write_event(Event::Text(BytesText::new(&ws_before_binary)))
                .map_err(xml_err)?;
        }
        self.writer
            .write_event(Event::Start(BytesStart::new("binary")))
            .map_err(xml_err)?;
        self.writer
            .write_event(Event::Text(BytesText::new(b64_data)))
            .map_err(xml_err)?;
        self.writer
            .write_event(Event::End(BytesEnd::new("binary")))
            .map_err(xml_err)?;

        // Skip original binary content + </binaryDataArray> in input
        self.skip_past_end(&mut buf, b"binaryDataArray")?;

        // Write closing tag
        self.writer
            .write_event(Event::Text(BytesText::new("\n            ")))
            .map_err(xml_err)?;
        self.writer
            .write_event(Event::End(BytesEnd::new("binaryDataArray")))
            .map_err(xml_err)?;

        Ok(())
    }

    /// Finish writing: drain remaining XML (chromatograms, `</run>`, `</mzML>`),
    /// then regenerate `<indexList>`, `<indexListOffset>`, and `<fileChecksum>`.
    pub fn finish(mut self) -> Result<()> {
        let mut buf = std::mem::take(&mut self.buf);
        let mut found_index = false;

        // Drain events through </mzML>; skip the original index tail.
        loop {
            buf.clear();
            let event = self.reader.read_event_into(&mut buf).map_err(xml_err)?;

            if matches!(event, Event::Eof) {
                break;
            }

            // <indexList> — skip original index and everything after
            if let Event::Start(ref e) = event {
                if e.name().as_ref() == b"indexList" {
                    found_index = true;
                    self.skip_past_end(&mut buf, b"indexList")?;
                    self.skip_to_eof(&mut buf)?;
                    break;
                }
            }

            // Track chromatogram offsets
            if let Event::Start(ref e) = event {
                if e.name().as_ref() == b"chromatogram" {
                    let offset = self.writer.get_ref().position();
                    let id = extract_attr(e, b"id").unwrap_or_default();
                    self.chromatogram_offsets.push(IndexEntry { id, offset });
                }
            }

            self.writer.write_event(event).map_err(xml_err)?;
        }

        if !found_index {
            log::warn!("Input mzML had no <indexList> — output will also be unindexed");
            self.writer.get_mut().flush().map_err(CentrixError::Io)?;
            return Ok(());
        }

        // ── Regenerate index ──────────────────────────────────────────────

        // The whitespace before <indexList> was already passed through from the
        // original file by the drain loop. Record position for indexListOffset.
        let index_list_offset = self.writer.get_ref().position();
        self.write_index_list()?;

        // indexListOffset
        self.write_text_element("\n  ", "indexListOffset", &index_list_offset.to_string())?;

        // fileChecksum — opening tag IS part of the hash
        self.writer
            .write_event(Event::Text(BytesText::new("\n  ")))
            .map_err(xml_err)?;
        self.writer
            .write_event(Event::Start(BytesStart::new("fileChecksum")))
            .map_err(xml_err)?;

        self.writer.get_mut().stop_hashing();
        let sha1_hex = self.writer.get_ref().digest_hex();

        self.writer
            .write_event(Event::Text(BytesText::new(&sha1_hex)))
            .map_err(xml_err)?;
        self.writer
            .write_event(Event::End(BytesEnd::new("fileChecksum")))
            .map_err(xml_err)?;

        // </indexedmzML>
        self.writer
            .write_event(Event::Text(BytesText::new("\n")))
            .map_err(xml_err)?;
        self.writer
            .write_event(Event::End(BytesEnd::new("indexedmzML")))
            .map_err(xml_err)?;
        self.writer
            .write_event(Event::Text(BytesText::new("\n")))
            .map_err(xml_err)?;

        self.writer.get_mut().flush().map_err(CentrixError::Io)?;

        let total = self.writer.get_ref().position();
        log::info!("Wrote {total} bytes to output");
        Ok(())
    }

    // ── Private helpers ───────────────────────────────────────────────────

    fn write_index_list(&mut self) -> Result<()> {
        let n_indices = if self.chromatogram_offsets.is_empty() {
            1
        } else {
            2
        };
        let mut il = BytesStart::new("indexList");
        il.push_attribute(("count", n_indices.to_string().as_str()));
        self.writer.write_event(Event::Start(il)).map_err(xml_err)?;

        // Spectrum index — collect to release borrow on self
        let spec_entries: Vec<(String, u64)> = self
            .spectrum_offsets
            .iter()
            .map(|e| (e.id.clone(), e.offset))
            .collect();
        self.write_offset_index("spectrum", &spec_entries)?;

        if !self.chromatogram_offsets.is_empty() {
            let chrom_entries: Vec<(String, u64)> = self
                .chromatogram_offsets
                .iter()
                .map(|e| (e.id.clone(), e.offset))
                .collect();
            self.write_offset_index("chromatogram", &chrom_entries)?;
        }

        self.writer
            .write_event(Event::Text(BytesText::new("\n  ")))
            .map_err(xml_err)?;
        self.writer
            .write_event(Event::End(BytesEnd::new("indexList")))
            .map_err(xml_err)?;
        Ok(())
    }

    fn write_offset_index(&mut self, name: &str, entries: &[(String, u64)]) -> Result<()> {
        let mut idx = BytesStart::new("index");
        idx.push_attribute(("name", name));
        self.writer
            .write_event(Event::Text(BytesText::new("\n    ")))
            .map_err(xml_err)?;
        self.writer
            .write_event(Event::Start(idx))
            .map_err(xml_err)?;

        for (id, offset) in entries {
            let mut oe = BytesStart::new("offset");
            oe.push_attribute(("idRef", id.as_str()));
            self.writer
                .write_event(Event::Text(BytesText::new("\n      ")))
                .map_err(xml_err)?;
            self.writer.write_event(Event::Start(oe)).map_err(xml_err)?;
            self.writer
                .write_event(Event::Text(BytesText::new(&offset.to_string())))
                .map_err(xml_err)?;
            self.writer
                .write_event(Event::End(BytesEnd::new("offset")))
                .map_err(xml_err)?;
        }

        self.writer
            .write_event(Event::Text(BytesText::new("\n    ")))
            .map_err(xml_err)?;
        self.writer
            .write_event(Event::End(BytesEnd::new("index")))
            .map_err(xml_err)?;
        Ok(())
    }

    fn write_text_element(&mut self, prefix: &str, tag: &str, text: &str) -> Result<()> {
        self.writer
            .write_event(Event::Text(BytesText::new(prefix)))
            .map_err(xml_err)?;
        self.writer
            .write_event(Event::Start(BytesStart::new(tag)))
            .map_err(xml_err)?;
        self.writer
            .write_event(Event::Text(BytesText::new(text)))
            .map_err(xml_err)?;
        self.writer
            .write_event(Event::End(BytesEnd::new(tag)))
            .map_err(xml_err)?;
        Ok(())
    }

    /// Skip events in the reader until past the end tag of `name`.
    fn skip_past_end(&mut self, buf: &mut Vec<u8>, name: &[u8]) -> Result<()> {
        let mut depth = 1i32;
        loop {
            buf.clear();
            let event = self.reader.read_event_into(buf).map_err(xml_err)?;
            match event {
                Event::Start(ref e) if e.name().as_ref() == name => depth += 1,
                Event::End(ref e) if e.name().as_ref() == name => {
                    depth -= 1;
                    if depth == 0 {
                        return Ok(());
                    }
                }
                Event::Eof => {
                    return Err(CentrixError::Xml(format!(
                        "unexpected EOF skipping <{}>",
                        String::from_utf8_lossy(name)
                    )));
                }
                _ => {}
            }
        }
    }

    /// Skip all remaining events to EOF.
    fn skip_to_eof(&mut self, buf: &mut Vec<u8>) -> Result<()> {
        loop {
            buf.clear();
            let event = self.reader.read_event_into(buf).map_err(xml_err)?;
            if matches!(event, Event::Eof) {
                return Ok(());
            }
        }
    }
}

// ── Phase 1 identity passthrough (kept for testing) ───────────────────────────

/// Byte-for-byte identity copy of input to output (no XML modification).
pub fn passthrough_identity(input: &Path, output: &Path) -> Result<()> {
    let mut src = std::fs::File::open(input).map_err(CentrixError::Io)?;
    let mut dst = BufWriter::new(std::fs::File::create(output).map_err(CentrixError::Io)?);

    let mut buf = vec![0u8; 64 * 1024];
    let mut total = 0u64;
    loop {
        let n = src.read(&mut buf).map_err(CentrixError::Io)?;
        if n == 0 {
            break;
        }
        dst.write_all(&buf[..n]).map_err(CentrixError::Io)?;
        total += n as u64;
    }

    log::debug!("Passthrough: wrote {total} bytes to {}", output.display());
    Ok(())
}
