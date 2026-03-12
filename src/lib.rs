pub mod basis;
pub mod calibration;
pub mod centroid;
pub mod config;
pub mod cwt;
pub mod error;
pub mod io;
pub mod lasso;
pub mod noise;
pub mod signal;

pub use config::Config;
pub use error::{CentrixError, Result};

/// Run the centroiding pipeline with the given configuration.
pub fn run(config: &Config) -> Result<()> {
    use indicatif::{ProgressBar, ProgressStyle};
    use rayon::prelude::*;

    log::info!(
        "centrix {}: {} -> {}",
        env!("CARGO_PKG_VERSION"),
        config.input.display(),
        config.output.display()
    );

    // Calibration pass
    log::info!(
        "Loading {} spectra for calibration...",
        config.n_calibration_spectra
    );
    let cal_spectra = io::reader::load_first_n(&config.input, config.n_calibration_spectra)?;

    let cal = calibration::calibrate(
        &cal_spectra,
        config.sigma_ms1,
        config.sigma_ms2,
        config.grid_spacing,
    )?;

    log::info!(
        "Calibration complete: σ_MS1={:.4} m/z, σ_MS2={:.4} m/z, grid_MS1={:.6} m/z, grid_MS2={:.6} m/z",
        cal.sigma_ms1,
        cal.sigma_ms2,
        cal.grid_spacing_ms1,
        cal.grid_spacing_ms2,
    );

    // Centroid + write output using batched rayon parallelism
    let basis_ms1 = basis::BasisPrecompute::new(
        cal.sigma_ms1,
        cal.grid_spacing_ms1,
        0.0,
        config.lambda_factor,
    );
    let basis_ms2 = basis::BasisPrecompute::new(
        cal.sigma_ms2,
        cal.grid_spacing_ms2,
        0.0,
        config.lambda_factor,
    );

    let mut writer = io::PassthroughWriter::new(&config.input, &config.output)?;
    let reader = io::reader::ProfileReader::open(&config.input)?;

    // Progress bar
    let total = reader.spectrum_count_hint().unwrap_or(0);
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} spectra ({per_sec}, ETA {eta})",
        )
        .unwrap()
        .progress_chars("█▉▊▋▌▍▎▏  "),
    );

    const BATCH_SIZE: usize = 256;
    let mut batch: Vec<io::reader::ProfileSpectrum> = Vec::with_capacity(BATCH_SIZE);
    let mut n_centroids_total = 0usize;

    for result in reader {
        batch.push(result?);
        if batch.len() < BATCH_SIZE {
            continue;
        }

        // Centroid batch in parallel, preserving order
        let results: Vec<_> = batch
            .par_iter()
            .map(|spectrum| {
                let basis = if spectrum.ms_level == 1 {
                    &basis_ms1
                } else {
                    &basis_ms2
                };
                centroid::centroid_spectrum(spectrum, basis, config)
            })
            .collect();

        // Write results in file order (single-threaded I/O)
        for (centroids, _stats) in &results {
            writer.write_spectrum(centroids)?;
            n_centroids_total += centroids.len();
        }

        pb.inc(batch.len() as u64);
        batch.clear();
    }

    // Flush remaining partial batch
    if !batch.is_empty() {
        let results: Vec<_> = batch
            .par_iter()
            .map(|spectrum| {
                let basis = if spectrum.ms_level == 1 {
                    &basis_ms1
                } else {
                    &basis_ms2
                };
                centroid::centroid_spectrum(spectrum, basis, config)
            })
            .collect();

        for (centroids, _stats) in &results {
            writer.write_spectrum(centroids)?;
            n_centroids_total += centroids.len();
        }

        pb.inc(batch.len() as u64);
    }

    pb.finish_with_message("centroiding complete");
    let n_processed = pb.position();
    log::info!("Centroiding complete: {n_processed} spectra, {n_centroids_total} total centroids");

    // Finish writing: drain chromatograms, regenerate index/checksum
    writer.finish()?;

    log::info!("Done.");
    Ok(())
}

/// Inspect a profile or centroid mzML file: print summary statistics.
pub fn inspect(path: &std::path::Path, n: usize) -> Result<()> {
    use mzdata::io::mzml::MzMLReader;
    use mzdata::prelude::*;
    use mzdata::spectrum::RefPeakDataLevel;
    use std::fs::File;
    use std::io::BufReader;

    let file = File::open(path).map_err(CentrixError::Io)?;
    let reader = MzMLReader::new(BufReader::new(file));

    let mut count = 0usize;
    let mut ms1_count = 0usize;
    let mut ms2_count = 0usize;
    let mut is_centroid = false;

    for mz_spectrum in reader.take(n) {
        let desc = mz_spectrum.description();
        let ms_level = desc.ms_level;
        let rt = desc
            .acquisition
            .first_scan()
            .map_or(0.0, |scan| scan.start_time);
        let scan_number = desc.index;

        let peaks = mz_spectrum.peaks();
        let (n_pts, mz_first, mz_last, max_int, median_spacing, mode) = match peaks {
            RefPeakDataLevel::Missing => (0, 0.0, 0.0, 0.0f32, 0.0, "missing"),
            RefPeakDataLevel::RawData(arrays) => {
                let mzs: Vec<f64> = arrays
                    .mzs()
                    .map(|c| c.iter().copied().collect())
                    .unwrap_or_default();
                let ints: Vec<f32> = arrays
                    .intensities()
                    .map(|c| c.iter().copied().collect())
                    .unwrap_or_default();
                let max_int = ints.iter().copied().fold(0f32, f32::max);
                let mz_first = mzs.first().copied().unwrap_or(0.0);
                let mz_last = mzs.last().copied().unwrap_or(0.0);
                let mut spacings: Vec<f64> = mzs.windows(2).map(|w| w[1] - w[0]).collect();
                spacings.sort_by(|a, b| a.total_cmp(b));
                let med = spacings.get(spacings.len() / 2).copied().unwrap_or(0.0);
                (mzs.len(), mz_first, mz_last, max_int, med, "profile")
            }
            RefPeakDataLevel::Centroid(peaks) => {
                is_centroid = true;
                let mzs: Vec<f64> = peaks.iter().map(|p| p.mz).collect();
                let ints: Vec<f32> = peaks.iter().map(|p| p.intensity).collect();
                let max_int = ints.iter().copied().fold(0f32, f32::max);
                let mz_first = mzs.first().copied().unwrap_or(0.0);
                let mz_last = mzs.last().copied().unwrap_or(0.0);
                let mut spacings: Vec<f64> = mzs.windows(2).map(|w| w[1] - w[0]).collect();
                spacings.sort_by(|a, b| a.total_cmp(b));
                let med = spacings.get(spacings.len() / 2).copied().unwrap_or(0.0);
                (mzs.len(), mz_first, mz_last, max_int, med, "centroid")
            }
            RefPeakDataLevel::Deconvoluted(_) => (0, 0.0, 0.0, 0.0f32, 0.0, "deconv"),
        };

        count += 1;
        if ms_level == 1 {
            ms1_count += 1;
        } else {
            ms2_count += 1;
        }

        if count <= 5 {
            println!(
                "  scan={:<6} MS{}  RT={:.3}min  {:<8}  pts={:>6}  m/z=[{:.4}..{:.4}]  max_int={:>10.0}  median_gap={:.6} m/z",
                scan_number, ms_level, rt, mode, n_pts, mz_first, mz_last, max_int, median_spacing
            );
        }
    }

    let file_type = if is_centroid { "centroid" } else { "profile" };
    println!(
        "\n{} [{} mode]: {} spectra ({} MS1, {} MS2)",
        path.display(),
        file_type,
        count,
        ms1_count,
        ms2_count
    );
    Ok(())
}

/// Run the calibration pass and print results. Useful for validating σ and
/// grid spacing auto-detection before running the full pipeline.
pub fn run_calibration(path: &std::path::Path, n: usize) -> Result<()> {
    use basis::BasisPrecompute;

    println!("Loading {} spectra from {}...", n, path.display());
    let spectra = io::reader::load_first_n(path, n)?;

    let ms1 = spectra.iter().filter(|s| s.ms_level == 1).count();
    let ms2 = spectra.iter().filter(|s| s.ms_level == 2).count();
    println!("  Loaded: {} MS1, {} MS2", ms1, ms2);

    let cal = calibration::calibrate(&spectra, None, None, None)?;

    let (ms1_rate, _ms1_fwhm, _) = cwt::infer_scan_rate(cal.sigma_ms1);
    let (ms2_rate, _ms2_fwhm, _) = cwt::infer_scan_rate(cal.sigma_ms2);

    println!("\n── Calibration Results ─────────────────────────────────────");
    println!(
        "  MS1  σ = {:.4} m/z  FWHM = {:.3} m/z  grid = {:.6} m/z  ({} peaks)  → {:.0} kTh/s",
        cal.sigma_ms1,
        cal.sigma_ms1 * 2.355,
        cal.grid_spacing_ms1,
        cal.n_peaks_ms1,
        ms1_rate,
    );
    println!(
        "  MS2  σ = {:.4} m/z  FWHM = {:.3} m/z  grid = {:.6} m/z  ({} peaks)  → {:.0} kTh/s",
        cal.sigma_ms2,
        cal.sigma_ms2 * 2.355,
        cal.grid_spacing_ms2,
        cal.n_peaks_ms2,
        ms2_rate,
    );
    println!("  (Stellar scan rates: 33 kTh/s→FWHM 0.5, 67→0.6, 125→0.8, 200→2.0 m/z)");

    // Sampling density
    let pts_ms1 = cal.sigma_ms1 / cal.grid_spacing_ms1;
    let pts_ms2 = cal.sigma_ms2 / cal.grid_spacing_ms2;
    println!("\n── Sampling Density ─────────────────────────────────────────");
    println!("  MS1  {pts_ms1:.2} data points per σ");
    println!("  MS2  {pts_ms2:.2} data points per σ");
    if pts_ms2 < 2.5 {
        println!("  NOTE: MS2 sparse sampling (<2.5 pts/σ); LASSO preferred over fast-path");
    }

    // Gram matrix (basis overlap)
    let basis_ms1 = BasisPrecompute::new(cal.sigma_ms1, cal.grid_spacing_ms1, 0.0, 3.0);
    let basis_ms2 = BasisPrecompute::new(cal.sigma_ms2, cal.grid_spacing_ms2, 0.0, 3.0);
    println!("\n── Gram Matrix (basis column overlap) ───────────────────────");
    let r_ms1 = basis_ms1.gram_row.get(1).copied().unwrap_or(0.0) / basis_ms1.gram_row[0];
    let r_ms2 = basis_ms2.gram_row.get(1).copied().unwrap_or(0.0) / basis_ms2.gram_row[0];
    println!(
        "  MS1  row length={:2}  G[k=0]={:.4}  G[k=1]/G[k=0]={r_ms1:.4}  (neighbor correlation)",
        basis_ms1.gram_row.len(),
        basis_ms1.gram_row[0],
    );
    println!(
        "  MS2  row length={:2}  G[k=0]={:.4}  G[k=1]/G[k=0]={r_ms2:.4}  (neighbor correlation)",
        basis_ms2.gram_row.len(),
        basis_ms2.gram_row[0],
    );

    Ok(())
}

/// Centroid N spectra and print per-spectrum stats. No output file written.
/// Use this to validate centroiding quality and performance on real data.
pub fn run_centroid_test(path: &std::path::Path, n: usize, n_cal: usize) -> Result<()> {
    use std::time::Instant;

    println!("Loading {n_cal} spectra for calibration...");
    let cal_spectra = io::reader::load_first_n(path, n_cal)?;
    let cal = calibration::calibrate(&cal_spectra, None, None, None)?;

    let sigma_ms1 = cal.sigma_ms1;
    let sigma_ms2 = cal.sigma_ms2;
    println!(
        "Calibration: MS1 σ={sigma_ms1:.4} m/z ({:.0} kTh/s), MS2 σ={sigma_ms2:.4} m/z ({:.0} kTh/s)",
        cwt::infer_scan_rate(sigma_ms1).0,
        cwt::infer_scan_rate(sigma_ms2).0,
    );

    let basis_ms1 = basis::BasisPrecompute::new(sigma_ms1, cal.grid_spacing_ms1, 0.0, 3.0);
    let basis_ms2 = basis::BasisPrecompute::new(sigma_ms2, cal.grid_spacing_ms2, 0.0, 3.0);

    let config = Config {
        input: path.to_path_buf(),
        output: path.to_path_buf(),
        config: None,
        sigma_ms1: None,
        sigma_ms2: None,
        grid_spacing: None,
        grid_offset: None,
        n_calibration_spectra: n_cal,
        lambda_factor: 3.0,
        max_lasso_iter: 2000,
        lasso_tol: 1e-6,
        lambda_change_threshold: 0.20,
        signal_threshold_sigma: 3.0,
        merge_gap_points: 2,
        extension_points: 3,
        min_region_width: 3,
        noise_window_da: 20.0,
        noise_step_da: 5.0,
        threads: 0,
        stats_output: None,
        quiet: false,
        verbose: false,
    };

    println!(
        "\n{:<8} {:<5} {:<8} {:<8} {:<8} {:<10}  ms",
        "scan", "MS", "regions", "lasso", "pass2", "centroids"
    );
    println!("{}", "─".repeat(68));

    let reader = io::reader::ProfileReader::open(path)?;
    let mut total_centroids = 0usize;
    let mut total_ms = 0.0f64;
    let mut count = 0usize;

    for result in reader.take(n) {
        let spectrum = result?;
        let basis = if spectrum.ms_level == 1 {
            &basis_ms1
        } else {
            &basis_ms2
        };
        let t0 = Instant::now();
        let (centroids, stats) = centroid::centroid_spectrum(&spectrum, basis, &config);
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        total_centroids += centroids.len();
        total_ms += elapsed_ms;
        count += 1;

        if count <= 20 || count % 100 == 0 {
            println!(
                "{:<8} MS{:<3} {:<8} {:<8} {:<8} {:<10}  {:.3}",
                spectrum.scan_number,
                spectrum.ms_level,
                stats.n_regions,
                stats.n_lasso,
                stats.n_pass2_refits,
                stats.n_centroids,
                elapsed_ms
            );
        }
    }

    println!("{}", "─".repeat(68));
    println!(
        "Total: {count} spectra, {total_centroids} centroids, avg {:.2} ms/spectrum",
        total_ms / count.max(1) as f64
    );
    Ok(())
}
