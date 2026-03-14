pub mod reader;
pub mod writer;

pub use writer::PassthroughWriter;

use crate::Result;

/// Identity passthrough: read profile mzML, write it back unchanged.
/// Used in Phase 1 to validate I/O infrastructure before centroiding is added.
pub fn passthrough(input: &std::path::Path, output: &std::path::Path) -> Result<()> {
    writer::passthrough_identity(input, output)
}
