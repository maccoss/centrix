use anyhow::Context;
use centrix::Config;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

// Force BLAS to single-threaded mode via FFI. Centrix uses rayon for
// spectrum-level parallelism; BLAS operations are on tiny matrices (8×8 to 22×22)
// where the BLAS internal thread-pool overhead far exceeds the computation cost.
#[cfg(feature = "openblas")]
extern "C" {
    fn openblas_set_num_threads(num_threads: std::ffi::c_int);
}

#[cfg(feature = "mkl")]
extern "C" {
    fn mkl_set_num_threads(num_threads: std::ffi::c_int);
}

#[derive(Parser)]
#[command(
    name = "centrix",
    version,
    about = "Post-acquisition centroiding for Thermo Stellar DIA data"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Centroid a profile mzML file (full pipeline)
    Run(Box<Config>),

    /// Inspect a profile or centroid mzML file: print per-scan summary
    Inspect {
        /// Input mzML file
        input: PathBuf,
        /// Number of spectra to read
        #[arg(short, long, default_value_t = 200)]
        n: usize,
    },

    /// Run calibration on a profile mzML and report σ, grid spacing, Gram matrix
    Calibrate {
        /// Input profile mzML file
        input: PathBuf,
        /// Number of spectra to use for calibration
        #[arg(short, long, default_value_t = 50)]
        n: usize,
    },

    /// Centroid N spectra and print per-spectrum stats (no output file written)
    CentroidTest {
        /// Input profile mzML file
        input: PathBuf,
        /// Number of spectra to process
        #[arg(short, long, default_value_t = 200)]
        n: usize,
        /// Number of calibration spectra
        #[arg(long, default_value_t = 50)]
        n_cal: usize,
    },
}

fn main() -> anyhow::Result<()> {
    // Force BLAS to single-threaded mode at runtime.
    #[cfg(feature = "openblas")]
    unsafe {
        openblas_set_num_threads(1);
    }
    #[cfg(feature = "mkl")]
    unsafe {
        mkl_set_num_threads(1);
    }

    let cli = Cli::parse();

    match cli.command {
        Command::Inspect { input, n } => {
            env_logger::Builder::new()
                .filter_level(log::LevelFilter::Warn)
                .init();
            centrix::inspect(&input, n).context("Inspect failed")?;
        }

        Command::Calibrate { input, n } => {
            env_logger::Builder::new()
                .filter_level(log::LevelFilter::Info)
                .init();
            centrix::run_calibration(&input, n).context("Calibration failed")?;
        }

        Command::CentroidTest { input, n, n_cal } => {
            env_logger::Builder::new()
                .filter_level(log::LevelFilter::Info)
                .init();
            centrix::run_centroid_test(&input, n, n_cal).context("CentroidTest failed")?;
        }

        Command::Run(config) => {
            let config = (*config).load_and_merge()?;

            let log_level = if config.verbose {
                log::LevelFilter::Debug
            } else if config.quiet {
                log::LevelFilter::Warn
            } else {
                log::LevelFilter::Info
            };
            env_logger::Builder::new().filter_level(log_level).init();

            if config.threads > 0 {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(config.threads)
                    .build_global()
                    .context("Failed to configure thread pool")?;
            }

            centrix::run(&config).context("Centroiding pipeline failed")?;
        }
    }
    Ok(())
}
