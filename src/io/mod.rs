pub mod reader;
pub mod writer;

use crate::{Config, Result};

/// Identity passthrough: read profile mzML, write it back unchanged.
/// Used in Phase 1 to validate I/O infrastructure before centroiding is added.
pub fn passthrough(config: &Config) -> Result<()> {
    writer::passthrough_identity(&config.input, &config.output)
}
