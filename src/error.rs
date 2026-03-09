use thiserror::Error;

pub type Result<T> = std::result::Result<T, CentrixError>;

#[derive(Debug, Error)]
pub enum CentrixError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("mzML error: {0}")]
    Mzml(String),

    #[error("Input file '{path}' is already centroid-mode (MS:1000127). Centrix requires profile-mode data (MS:1000128).")]
    NotProfileMode { path: String },

    #[error("Calibration error: {0}")]
    Calibration(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("XML error: {0}")]
    Xml(String),
}
