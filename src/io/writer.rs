use crate::Result;
use std::io::{self, Read, Write};
use std::path::Path;

/// Tracks bytes written through an inner `Write`, enabling byte-offset recording
/// for mzML index regeneration.
pub struct ByteCountingWriter<W: Write> {
    inner: W,
    pub bytes_written: u64,
}

impl<W: Write> ByteCountingWriter<W> {
    pub fn new(inner: W) -> Self {
        Self {
            inner,
            bytes_written: 0,
        }
    }
}

impl<W: Write> Write for ByteCountingWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.bytes_written += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

/// Phase 1 identity passthrough: copies input mzML to output byte-for-byte.
/// This validates that the I/O pipeline is wired correctly before centroiding
/// logic modifies the binary arrays.
pub fn passthrough_identity(input: &Path, output: &Path) -> Result<()> {
    let mut src = std::fs::File::open(input).map_err(crate::CentrixError::Io)?;
    let dst_file = std::fs::File::create(output).map_err(crate::CentrixError::Io)?;
    let mut dst = ByteCountingWriter::new(io::BufWriter::new(dst_file));

    let mut buf = vec![0u8; 64 * 1024];
    loop {
        let n = src.read(&mut buf).map_err(crate::CentrixError::Io)?;
        if n == 0 {
            break;
        }
        dst.write_all(&buf[..n]).map_err(crate::CentrixError::Io)?;
    }

    log::debug!(
        "Passthrough: wrote {} bytes to {}",
        dst.bytes_written,
        output.display()
    );
    Ok(())
}
