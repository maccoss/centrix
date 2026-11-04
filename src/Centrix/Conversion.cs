/*
 * Original author: Michael MacCoss <maccoss .at. uw.edu>,
 *                  MacCoss Lab, Department of Genome Sciences, UW
 * AI assistance: Claude Code (Claude Opus 5.5) <noreply .at. anthropic.com>
 *
 * Copyright 2026 University of Washington - Seattle, WA
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Centrix.Demux;
using Pwiz.Analysis;
using Pwiz.Analysis.Filters;
using Pwiz.Analysis.PeakPicking;
using Pwiz.Data.Common.Cv;
using Pwiz.Data.Common.Params;
using Pwiz.Data.MsData;
using Pwiz.Data.MsData.Encoding;
using Pwiz.Data.MsData.Instruments;
using Pwiz.Data.MsData.Readers;
using Pwiz.Data.MsData.Spectra;
using IsolationWindow = Centrix.IsolationWindow;

namespace Centrix.Cli
{
    /// <summary>
    /// One input file through Centrix: opened by ProteoWizard without vendor peak picking, its peak sigmas settled,
    /// its MS2 windows checked for a stagger, demultiplexed if staggered, centroided, and written as mzML.
    /// </summary>
    internal sealed class Conversion
    {
        /// <summary>MS2 spectra whose windows are checked for a stagger, from the start of the run.</summary>
        private const int SCHEME_SPECTRA = 5000;

        /// <summary>Spectra of a level sampled before a level none of whose spectra Centrix centroids stops being looked for.</summary>
        private const int LEVEL_GIVE_UP = 50;

        /// <summary>pwiz's wavelet picker as msconvert's <c>peakPicking cwt</c> sets it, for profile spectra no vendor centroids.</summary>
        private static IPeakDetector WaveletPicker => new CwtPeakDetector(1.0, 0, 0.1);

        private readonly CentrixOptions _options;
        private readonly Log _log;
        private StellarZoomLabelFix _zoomLabelFix;   // the last opened file's, when it is a Stellar's
        private ISpectrumList _read;                 // the last opened file's spectra as its reader gives them

        public Conversion(CentrixOptions options, Log log)
        {
            _options = options;
            _log = log;
        }

        /// <summary>The settings a run was centroided with, for the calibrate command and the tests.</summary>
        public sealed class Plan
        {
            public double SigmaMs1 { get; set; }
            public double SigmaMs2 { get; set; }
            public string SigmaMs1Source { get; set; }
            public string SigmaMs2Source { get; set; }
            public DemuxScheme Scheme { get; set; }
            public double MeasuredSigmaMs1 { get; set; } = double.NaN;
            public double MeasuredSigmaMs2 { get; set; } = double.NaN;
            public int MeasuredPeaksMs1 { get; set; }
            public int MeasuredPeaksMs2 { get; set; }
            public bool Demultiplex { get; set; }
            public string DemuxReason { get; set; }
        }

        /// <summary>Converts one input; throws on failure.</summary>
        public void Run(string input, string output)
        {
            var clock = Stopwatch.StartNew();
            bool vendor = _options.Centroid == CentroidMode.vendor;
            using var msd = Open(input, vendor);
            var spectra = msd.Run.SpectrumList;
            _log.Info($"Opened {input}: {spectra.Count:N0} spectra in {clock.Elapsed.TotalSeconds:F0} s");

            ISpectrumList final;
            CentrixSpectrumList centrix = null;
            StaggeredDemuxSpectrumList stepped = null;
            var software = AddSoftware(msd);
            if (vendor)
            {
                final = spectra;
                _log.Info(_read is IVendorCentroidingSpectrumList reader
                    ? $"Centroiding: the vendor's centroids ({reader.VendorCentroidName})"
                    : "Centroiding: pwiz's wavelet picker (the file carries no vendor centroids)");
            }
            else
            {
                var plan = MakePlan(spectra);
                var source = spectra;
                if (plan.Demultiplex)
                {
                    var demux = DemuxOptions(plan);
                    stepped = new StaggeredDemuxSpectrumList(spectra, demux, _log.Writer);
                    source = stepped;
                    _log.Info($"Demultiplexing: {stepped.Count:N0} spectra out, indexed in {clock.Elapsed.TotalSeconds:F0} s");
                }
                // The profile spectra Centrix is not for (a Fusion's Orbitrap MS1, demultiplexed or not): the vendor's
                // centroids, from a second list read through the same reader, built only if one is asked for.
                var read = _read;
                var fallback = new Lazy<ISpectrumList>(() => Fallback(msd, read, spectra));
                var ms1 = Centroider(plan.SigmaMs1);
                var ms2 = Centroider(plan.SigmaMs2);
                centrix = new CentrixSpectrumList(source, fallback, ms1, ms2, _options.Centroid, _options.EffectiveThreads, software);
                final = centrix;
            }
            msd.Run.SpectrumList = final;
            // The writer lists the file's processing records, not its spectrum list's: add Centrix's, as msconvert adds
            // its filters' (replacing one of the same id, the input's that a vendor peak picker copies).
            if (final.DataProcessing is { } processing)
            {
                int existing = msd.DataProcessings.FindIndex(d => d.Id == processing.Id);
                if (existing >= 0)
                    msd.DataProcessings[existing] = processing;
                else
                    msd.DataProcessings.Add(processing);
            }

            Write(msd, output);
            _log.Info($"Wrote {output} in {clock.Elapsed.TotalSeconds:F0} s");
            if (_zoomLabelFix != null && _zoomLabelFix.Relabelled > 0)
            {
                _log.Info($"  zoom scan label removed from {_zoomLabelFix.Relabelled:N0} Stellar spectra: the Stellar stores " +
                    "its 200 kTh/s rate as the zoom flag (--keep-zoom-label keeps it)");
            }
            if (centrix != null)
            {
                // Demultiplexed spectra come out of the joint solve centroided, and pass through Centrix's list.
                string centroided = stepped != null ? "demultiplexed and centroided by the joint solve" : "centroided already";
                _log.Info($"  {centrix.CentrixSpectra:N0} spectra centroided by Centrix ({centrix.CentroidsOut:N0} centroids), " +
                    $"{centrix.FallbackSpectra:N0} by the vendor or pwiz, {centrix.PassedThrough:N0} {centroided}");
            }
            if (stepped != null)
                _log.Info($"  demultiplexing: {stepped.IonsIn:E3} ions split among the bins");
        }

        /// <summary>Settles one input's sigmas and scheme, and logs them, writing nothing.</summary>
        public Plan Calibrate(string input)
        {
            using var msd = Open(input);
            _log.Info($"{input}: {msd.Run.SpectrumList.Count:N0} spectra");
            var plan = MakePlan(msd.Run.SpectrumList, true);
            foreach (int level in new[] { 1, 2 })
            {
                double measured = level == 1 ? plan.MeasuredSigmaMs1 : plan.MeasuredSigmaMs2;
                int peaks = level == 1 ? plan.MeasuredPeaksMs1 : plan.MeasuredPeaksMs2;
                double sigma = level == 1 ? plan.SigmaMs1 : plan.SigmaMs2;
                _log.Info(double.IsNaN(measured)
                    ? $"  MS{level}: too few isolated peaks to measure a width ({peaks:N0})"
                    : $"  MS{level}: its isolated peaks measure sigma {measured:F3} Th (FWHM {measured * CentrixCalibration.FWHM_PER_SIGMA:F2} Th, " +
                      $"lower quartile of {peaks:N0} peaks); Centrix fits {sigma:F3} Th, {sigma / measured:F2} of it");
            }
            return plan;
        }

        /// <summary>
        /// The input opened by ProteoWizard: a Thermo .raw or an open format, with no vendor peak picking, so Centrix sees
        /// the profile, or, for <see cref="CentroidMode.vendor"/>, with the vendor's. Centrix's wrappers (<see cref="Wrap"/>)
        /// go on top.
        /// </summary>
        private MSData Open(string input, bool vendorCentroids = false)
        {
            var msd = new MSData();
            Readers().Read(input, msd, new ReaderConfig());
            _read = msd.Run.SpectrumList;
            msd.Run.SpectrumList = Wrap(msd, vendorCentroids ? VendorPicker(_read) : _read, true);
            return msd;
        }

        /// <summary>
        /// The vendor's peak picking, pwiz's wavelet picker where the reader has none, over every MS level, as msconvert's
        /// <c>peakPicking vendor msLevel=1-</c>. It must wrap the reader's own list: the picker takes the vendor's
        /// centroids only from a list that is the reader's, so a filter beneath it would turn it into the wavelet picker.
        /// </summary>
        internal static ISpectrumList VendorPicker(ISpectrumList read) =>
            new SpectrumList_PeakPicker(read, WaveletPicker, true, "1-");

        /// <summary>
        /// Centrix's wrappers over a list read from the file: the user's filters, then on a Stellar's the zoom label
        /// removed (<see cref="StellarZoomLabelFix"/>) and the Adaptive RT spectra left out.
        /// </summary>
        /// <param name="msd">The file the list was read from.</param>
        /// <param name="spectra">The list as read, or the vendor's peak picker over it.</param>
        /// <param name="primary">The list the output is made from: it logs, and keeps the zoom-label fix's count.</param>
        internal ISpectrumList Wrap(MSData msd, ISpectrumList spectra, bool primary)
        {
            if (_options.Filters.Count > 0)
            {
                spectra = SpectrumListFactory.Wrap(spectra, _options.Filters, msd);
                if (primary)
                    _log.Info($"Filters: {string.Join("; ", _options.Filters)}");
            }
            bool stellar = SpectrumRouting.IsStellar(msd);
            if (primary)
                _zoomLabelFix = null;
            if (stellar && !_options.KeepZoomLabel)
            {
                var fix = new StellarZoomLabelFix(spectra);
                if (primary)
                    _zoomLabelFix = fix;
                spectra = fix;
            }
            if (stellar && _options.AdaptiveRt == AdaptiveRtMode.exclude)
                spectra = ExcludeAdaptiveRt(spectra, primary);
            return spectra;
        }

        /// <summary>
        /// Centroids for the profile spectra Centrix is not for, spectrum for spectrum with <paramref name="wrapped"/>: the
        /// vendor's, through the vendor's peak picker on the reader's own list under the same wrappers; pwiz's wavelet
        /// picker over <paramref name="wrapped"/> when the file has no vendor centroids, or when the wrappers do not give
        /// the same spectra twice. A demultiplexed list passes such spectra through under their own ids, which
        /// <see cref="CentrixSpectrumList"/> finds them by.
        /// </summary>
        /// <param name="msd">The file.</param>
        /// <param name="read">Its spectra as its reader gives them.</param>
        /// <param name="wrapped"><paramref name="read"/> under <see cref="Wrap"/>, before any demultiplexing.</param>
        internal ISpectrumList Fallback(MSData msd, ISpectrumList read, ISpectrumList wrapped)
        {
            if (read is IVendorCentroidingSpectrumList)
            {
                var vendor = Wrap(msd, VendorPicker(read), false);
                if (vendor.Count == wrapped.Count)
                    return vendor;
                _log.Warn($"The vendor's centroids give {vendor.Count:N0} spectra under the filters, not {wrapped.Count:N0}: " +
                    "pwiz's wavelet picker centroids the spectra Centrix is not for");
            }
            return new SpectrumList_PeakPicker(wrapped, WaveletPicker, true, "1-");
        }

        /// <summary>
        /// A Stellar's Adaptive RT alignment spectra (<see cref="SpectrumRouting.IsWiderThan"/>) left out of the run: the
        /// instrument's own retention time alignment scans, not data for a search. Walks every spectrum's metadata once.
        /// </summary>
        internal void ExcludeAdaptiveRt(MSData msd) => msd.Run.SpectrumList = ExcludeAdaptiveRt(msd.Run.SpectrumList, true);

        private ISpectrumList ExcludeAdaptiveRt(ISpectrumList spectra, bool log)
        {
            var kept = new SpectrumListFilter(spectra, new NotWiderThan(_options.AdaptiveRtWidthTh));
            int excluded = spectra.Count - kept.Count;
            if (excluded == 0)
                return spectra;
            if (log)
            {
                _log.Info($"Adaptive RT: {excluded:N0} spectra isolated wider than {_options.AdaptiveRtWidthTh:G} Th left out " +
                    "(--adaptive-rt include keeps them)");
            }
            return kept;
        }

        /// <summary>ProteoWizard's open formats (mzML among them) and Thermo's reader, the only vendor's.</summary>
        internal static ReaderList Readers() => Pwiz.Vendor.Thermo.ThermoReaderRegistration.CreateDefaultWithThermo();

        /// <summary>
        /// Each level's peak sigma, given or from the run (<see cref="CentrixSigmaCalibrator"/>: a linear ion trap's from
        /// its point spacing, any other's measured from spectra spread across the run), and whether to demultiplex.
        /// </summary>
        /// <param name="spectra">The run's spectra.</param>
        /// <param name="measure">
        /// Also measure each level's peak width from every sampled spectrum (<see cref="CentrixSigmaCalibrator.MeasuredSigma"/>),
        /// for the calibrate command, rather than stopping once the point spacing settles the sigma.
        /// </param>
        internal Plan MakePlan(ISpectrumList spectra, bool measure = false)
        {
            var plan = new Plan { SigmaMs1 = _options.SigmaMs1, SigmaMs2 = _options.SigmaMs2 };
            bool needMs1 = measure || !(plan.SigmaMs1 > 0), needMs2 = measure || !(plan.SigmaMs2 > 0);
            var calibrator = new CentrixSigmaCalibrator();
            var seen = new int[3];
            var centrixSpectra = new int[3];
            bool Done(int slot, bool need) =>
                !need || (calibrator.IsSettled(slot) && !measure) || (seen[slot] >= LEVEL_GIVE_UP && centrixSpectra[slot] == 0);
            if (needMs1 || needMs2)
            {
                foreach (int i in CentrixSigmaCalibrator.SampleIndices(spectra.Count))
                {
                    if (Done(1, needMs1) && Done(2, needMs2))
                        break;
                    var spectrum = spectra.GetSpectrum(i, true);
                    int level = spectrum.Params.CvParamValueOrDefault(CVID.MS_ms_level, 0);
                    if (level < 1)
                        continue;
                    int slot = level == 1 ? 1 : 2;
                    seen[slot]++;
                    if (SpectrumRouting.Route(spectrum, _options.Centroid) != SpectrumRoute.Centrix)
                        continue;
                    var mz = spectrum.GetMZArray()?.Data;
                    var intensity = spectrum.GetIntensityArray()?.Data;
                    if (mz == null || intensity == null)
                        continue;
                    centrixSpectra[slot]++;
                    calibrator.Add(level, mz, intensity);
                }
            }
            plan.MeasuredSigmaMs1 = calibrator.MeasuredSigma(1);
            plan.MeasuredSigmaMs2 = calibrator.MeasuredSigma(2);
            plan.MeasuredPeaksMs1 = calibrator.PeakCount(1);
            plan.MeasuredPeaksMs2 = calibrator.PeakCount(2);
            (plan.SigmaMs1, plan.SigmaMs1Source) = SigmaOf(1, plan.SigmaMs1, calibrator, centrixSpectra[1]);
            (plan.SigmaMs2, plan.SigmaMs2Source) = SigmaOf(2, plan.SigmaMs2, calibrator, centrixSpectra[2]);
            _log.Info($"Centrix MS1 peak sigma {plan.SigmaMs1:F3} Th ({plan.SigmaMs1Source}); " +
                $"MS2 {plan.SigmaMs2:F3} Th ({plan.SigmaMs2Source})");

            if (_options.Demux != DemuxMode.off)
                DecideDemux(spectra, plan, centrixSpectra[2] > 0);
            return plan;
        }

        private static (double Sigma, string Source) SigmaOf(int level, double given, CentrixSigmaCalibrator calibrator, int spectra)
        {
            if (given > 0)
                return (given, "given");
            if (spectra == 0)
                return (CentrixCalibration.FALLBACK_SIGMA_TH, "no profile spectra for Centrix at this level");
            return (calibrator.Sigma(level), calibrator.Describe(level));
        }

        /// <summary>
        /// Whether to demultiplex: the scheme of the first MS2 spectra's windows (but those wider than
        /// <see cref="CentrixOptions.AdaptiveRtWidthTh"/>), staggered, and Centrix centroiding the MS2 (a linear ion
        /// trap's profile: the joint solve fits a linear grid's profile).
        /// </summary>
        private void DecideDemux(ISpectrumList spectra, Plan plan, bool centrixMs2)
        {
            var windows = new List<IsolationWindow>();
            for (int i = 0; i < spectra.Count && windows.Count < SCHEME_SPECTRA; i++)
            {
                var spectrum = spectra.GetSpectrum(i, false);
                if (spectrum.Params.CvParamValueOrDefault(CVID.MS_ms_level, 0) < 2 || spectrum.Precursors.Count == 0)
                    continue;
                var window = spectrum.Precursors[0].IsolationWindow;
                var w = new IsolationWindow(window.CvParamValueOrDefault(CVID.MS_isolation_window_target_m_z, 0.0),
                    window.CvParamValueOrDefault(CVID.MS_isolation_window_lower_offset, 0.0),
                    window.CvParamValueOrDefault(CVID.MS_isolation_window_upper_offset, 0.0));
                if (w.Width > 0 && (_options.AdaptiveRtWidthTh <= 0 || w.Width <= _options.AdaptiveRtWidthTh))
                    windows.Add(w);
            }
            plan.Scheme = windows.Count > 0 ? DemuxSchemeDetector.Detect(windows) : null;
            bool staggered = plan.Scheme != null && plan.Scheme.Kind == DemuxSchemeKind.overlapping;
            if (plan.Scheme != null)
            {
                _log.Info($"MS2 windows: {plan.Scheme.Windows.Count} into {plan.Scheme.Bins.Count} bins, " +
                    $"{plan.Scheme.OverlapFactor}-fold ({plan.Scheme.Kind.ToString().Replace('_', '-')})");
            }
            if (_options.Demux == DemuxMode.staggered && !staggered)
                throw new InvalidOperationException("--demux staggered: the MS2 windows do not overlap.");
            if (_options.Demux == DemuxMode.staggered && !centrixMs2)
                throw new InvalidOperationException("--demux staggered: the MS2 spectra are not linear ion trap profile spectra.");
            plan.Demultiplex = staggered && centrixMs2;
            plan.DemuxReason = !staggered ? "the windows do not overlap"
                : !centrixMs2 ? "the MS2 spectra are not a linear ion trap's profile"
                : "staggered windows";
            if (staggered && !centrixMs2)
                _log.Warn("The MS2 windows overlap, but the spectra are not a linear ion trap's profile, which Centrix's " +
                    "joint solve needs: they are centroided, not demultiplexed (msconvert's demultiplex filter or Osprey's " +
                    "--demux handle centroided staggered data).");
        }

        /// <summary>The staggered joint solve, as the Stellar staggered searches ran it.</summary>
        private ScanningDemuxOptions DemuxOptions(Plan plan)
        {
            var options = new ScanningDemuxOptions
            {
                Threads = _options.EffectiveThreads,
                Joint = true,
                IonsFromFillTime = _options.Ions,
                // Larger blocks solve less context per bin kept; 16 bins suit the staggered joint solve.
                GroupBins = 16,
                LinearPeakSigmaTh = plan.SigmaMs2,
                MaxWindowWidthTh = _options.AdaptiveRtWidthTh,
                Centrix = _options.Parameters.Copy(),
            };
            options.JointParameters.UseStaggeredDefaults();
            return options;
        }

        private CentrixCentroider Centroider(double sigma)
        {
            var parameters = _options.Parameters.Copy();
            parameters.SigmaTh = sigma;
            return new CentrixCentroider(parameters);
        }

        /// <summary>Centrix in the file's software list, for its processing method to name.</summary>
        private static Software AddSoftware(MSData msd)
        {
            var software = new Software("Centrix", new CVParam(CVID.MS_custom_unreleased_software_tool, "Centrix"),
                CentrixVersion.Current);
            msd.Software.Add(software);
            return software;
        }

        /// <summary>Keeps every spectrum but those isolated wider than a width.</summary>
        private sealed class NotWiderThan : ISpectrumPredicate
        {
            private readonly double _widthTh;

            public NotWiderThan(double widthTh)
            {
                _widthTh = widthTh;
            }

            public DetailLevel SuggestedDetailLevel => DetailLevel.FullMetadata;

            public PredicateDecision Accept(SpectrumIdentity identity) => PredicateDecision.NeedSpectrum;

            public bool Accept(Spectrum spectrum) => !SpectrumRouting.IsWiderThan(spectrum, _widthTh);

            public string Describe() => $"isolation width at most {_widthTh:G} Th";
        }

        /// <summary>
        /// Writes indexed mzML: m/z and time in 64 bits, intensities in 32, zlib unless turned off; through a partial file,
        /// so an interrupted run leaves nothing that looks finished.
        /// </summary>
        private void Write(MSData msd, string output)
        {
            var config = new WriteConfig { Format = WriteFormat.Mzml };
            config.EncoderConfig.Precision = BinaryPrecision.Bits32;
            config.EncoderConfig.PrecisionOverrides[CVID.MS_m_z_array] = BinaryPrecision.Bits64;
            config.EncoderConfig.PrecisionOverrides[CVID.MS_time_array] = BinaryPrecision.Bits64;
            config.EncoderConfig.Compression = _options.Zlib ? BinaryCompression.Zlib : BinaryCompression.None;
            string directory = Path.GetDirectoryName(Path.GetFullPath(output));
            if (!string.IsNullOrEmpty(directory))
                Directory.CreateDirectory(directory);
            string partial = output + ".partial";
            try
            {
                MSDataFile.Write(msd, partial, config);
                File.Move(partial, output, true);
            }
            finally
            {
                if (File.Exists(partial))
                    File.Delete(partial);
            }
        }
    }
}
