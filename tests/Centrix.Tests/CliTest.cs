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
using System.IO;
using System.Linq;
using Centrix.Cli;
using Centrix.Demux;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Pwiz.Analysis;
using Pwiz.Analysis.PeakPicking;
using Pwiz.Data.Common.Cv;
using Pwiz.Data.MsData.Instruments;

namespace Centrix.Tests
{
    /// <summary>
    /// The centrix command line end to end: its options, its output names, and whole files read by ProteoWizard,
    /// centroided (and demultiplexed) and written as mzML, read back and held to the goldens.
    /// </summary>
    [TestClass]
    public class CliTest
    {
        private const string PROFILE_SLICE = "stellar-2th-profile-slice.mzML";

        [TestMethod]
        public void TestCentrixOptions()
        {
            // Rust Centrix's command line still parses: the run subcommand, -i with several files, -o, its lasso options.
            var options = CentrixOptions.Parse(new[] { "run", "-i", "a.raw", "b.raw", "-o", "out", "--lambda-factor", "2.5",
                "--max-lasso-iter", "500", "--threads", "3", "--sigma", "0.3" });
            Assert.AreEqual(CentrixCommand.run, options.Command);
            CollectionAssert.AreEqual(new[] { "a.raw", "b.raw" }, options.Inputs);
            Assert.AreEqual(2.5, options.Parameters.LambdaFactor);
            Assert.AreEqual(500, options.Parameters.MaxIterations);
            Assert.AreEqual(3, options.EffectiveThreads);
            Assert.AreEqual(0.3, options.SigmaMs2);
            Assert.AreEqual(0.3, options.SigmaMs1);
            var levels = CentrixOptions.Parse(new[] { "a", "--sigma-ms1", "0.1", "--sigma", "0.3" });
            Assert.AreEqual(0.1, levels.SigmaMs1);
            Assert.AreEqual(0.3, levels.SigmaMs2);
            Assert.AreEqual(Path.Combine("out", "a.centrix.mzML"), options.OutputPathFor("a.raw"));

            // The defaults: auto centroiding and demultiplexing, ions from fill times, descent, every core.
            options = CentrixOptions.Parse(new[] { "run.raw" });
            Assert.AreEqual(CentroidMode.auto, options.Centroid);
            Assert.AreEqual(DemuxMode.auto, options.Demux);
            Assert.IsTrue(options.Ions);
            Assert.AreEqual(AdaptiveRtMode.exclude, options.AdaptiveRt);
            Assert.AreEqual(10, options.AdaptiveRtWidthTh);
            Assert.AreEqual(CentrixSolver.coordinate_descent, options.Parameters.Solver);
            Assert.AreEqual(Environment.ProcessorCount, options.EffectiveThreads);
            Assert.AreEqual(Path.Combine(Path.GetDirectoryName(Path.GetFullPath("run.raw")), "run.centrix.mzML"),
                options.OutputPathFor("run.raw"));

            // One input and an .mzML output names the file; --name=value; the solver and modes by name.
            options = CentrixOptions.Parse(new[] { "calibrate", "x.mzML.gz", "--output=y.mzML", "--solver", "active-set",
                "--centroid", "centrix", "--demux=off", "--adaptive-rt", "include", "--adaptive-rt-width", "12" });
            Assert.AreEqual(CentrixCommand.calibrate, options.Command);
            Assert.AreEqual("y.mzML", options.OutputPathFor("x.mzML.gz"));
            Assert.AreEqual(CentrixSolver.active_set, options.Parameters.Solver);
            Assert.AreEqual(CentroidMode.centrix, options.Centroid);
            Assert.AreEqual(DemuxMode.off, options.Demux);
            Assert.AreEqual(AdaptiveRtMode.include, options.AdaptiveRt);
            Assert.AreEqual(12, options.AdaptiveRtWidthTh);
            Assert.AreEqual(Path.Combine("d", "x.centrix.mzML"),
                CentrixOptions.Parse(new[] { "x.mzML.gz", "-o", "d" }).OutputPathFor("x.mzML.gz"));

            // Every lasso option sets the setting it names.
            var defaults = new CentrixParams();
            foreach (string option in CentrixOptions.ParameterOptionNames)
            {
                var parameters = CentrixOptions.Parse(new[] { "a", option, "17" }).Parameters;
                bool changed = typeof(CentrixParams).GetProperties().Any(p => p.CanWrite &&
                    !Equals(p.GetValue(parameters), p.GetValue(defaults)));
                Assert.IsTrue(changed, option);
            }

            // Mistakes are reported, not guessed at.
            foreach (var bad in new[]
                     {
                         Array.Empty<string>(), new[] { "a", "--sigma", "-1" }, new[] { "a", "--sigma" },
                         new[] { "a", "--centroid", "cwt" }, new[] { "a", "--merge-gap-points", "2.5" }, new[] { "a", "--bogus" },
                         new[] { "a", "--solver", "lbfgs" }, new[] { "a", "--adaptive-rt", "drop" },
                     })
            {
                Assert.ThrowsException<ArgumentException>(() => CentrixOptions.Parse(bad), string.Join(" ", bad));
            }
            Assert.IsTrue(CentrixOptions.Parse(new[] { "--help" }).ShowHelp);
        }

        [TestMethod]
        public void TestCentrixConversionMatchesRust()
        {
            // The real Stellar profile slice through the whole command: read, centroided in parallel batches, written,
            // read back: Rust Centrix's centroids, every spectrum marked centroided, Centrix in the processing record.
            string input = TestData.Path("Centrix", PROFILE_SLICE);
            var golden = CentrixTest.ReadGolden(TestData.Path("Centrix", "stellar-centrix-rust.golden.tsv"));
            string directory = TestData.TempDirectory();
            try
            {
                var options = CentrixOptions.Parse(new[] { input, "-o", directory, "--sigma-ms1", "0.2", "--sigma", "0.33", "-q" });
                string output = options.OutputPathFor(input);
                new Conversion(options, new Log(true, false)).Run(input, output);
                Assert.IsTrue(File.Exists(output));
                Assert.IsFalse(File.Exists(output + ".partial"));

                using var msd = TestData.Open(output);
                Assert.IsTrue(msd.Software.Any(s => s.Id == "Centrix"));
                var spectra = msd.Run.SpectrumList;
                Assert.AreEqual(16, spectra.Count);
                var ordinal = new Dictionary<int, int>();
                for (int i = 0; i < spectra.Count; i++)
                {
                    var spectrum = spectra.GetSpectrum(i, true);
                    Assert.IsFalse(SpectrumRouting.IsProfile(spectrum), $"spectrum {i} is still profile");
                    int level = spectrum.Params.CvParamValueOrDefault(CVID.MS_ms_level, 0);
                    ordinal.TryGetValue(level, out int n);
                    ordinal[level] = n + 1;
                    var expected = golden[(PROFILE_SLICE, level, n)];
                    var mz = spectrum.GetMZArray().Data;
                    var intensity = spectrum.GetIntensityArray().Data;
                    Assert.AreEqual(expected.Count, mz.Count, $"spectrum {i} centroids");
                    for (int k = 0; k < mz.Count; k++)
                    {
                        Assert.AreEqual(expected[k].Mz, mz[k], 1e-3, $"spectrum {i} centroid {k}");
                        Assert.AreEqual(expected[k].Intensity, intensity[k], 0.01 * expected[k].Intensity, $"spectrum {i} centroid {k}");
                    }
                }
                Assert.AreEqual(2, ordinal[1]);
                Assert.AreEqual(14, ordinal[2]);
                var processing = spectra.GetSpectrum(0).DataProcessing ?? spectra.DataProcessing;
                Assert.IsNotNull(processing);
                Assert.IsTrue(processing.ProcessingMethods.Any(m => m.UserParams.Any(u => u.Name == "Centrix centroiding")));
            }
            finally
            {
                Directory.Delete(directory, true);
            }
        }

        [TestMethod]
        public void TestCentrixSpectrumListInvariance()
        {
            // The same centroids whatever the batch size and thread count, read in order or not.
            string input = TestData.Path("Centrix", PROFILE_SLICE);
            var ms1 = new CentrixCentroider(new CentrixParams { SigmaTh = 0.2 });
            var ms2 = new CentrixCentroider(new CentrixParams { SigmaTh = 0.33 });
            List<double[]> Centroids(int threads, int batch, bool reverse)
            {
                var spectra = TestData.Open(input).Run.SpectrumList;
                var list = new CentrixSpectrumList(spectra, null, ms1, ms2, CentroidMode.auto, threads, new Software("Centrix"), batch);
                var order = Enumerable.Range(0, list.Count);
                var result = new double[list.Count][];
                foreach (int i in reverse ? order.Reverse() : order)
                    result[i] = list.GetSpectrum(i, true).GetMZArray().Data.ToArray();
                Assert.AreEqual(list.Count, list.CentrixSpectra);
                return result.ToList();
            }
            var reference = Centroids(1, 1, false);
            foreach (var (threads, batch, reverse) in new[] { (8, 3, false), (4, 256, false), (2, 5, true) })
            {
                var other = Centroids(threads, batch, reverse);
                for (int i = 0; i < reference.Count; i++)
                    CollectionAssert.AreEqual(reference[i], other[i], $"spectrum {i}, {threads} threads, batches of {batch}");
            }

            // The vendor mode sends every profile spectrum to the fallback; a centroided spectrum passes through.
            var raw = TestData.Open(input).Run.SpectrumList;
            var cwt = new SpectrumList_PeakPicker(raw, new CwtPeakDetector(1.0, 0, 0.1), false, "1-");
            var vendor = new CentrixSpectrumList(raw, cwt, ms1, ms2, CentroidMode.vendor, 2, new Software("Centrix"), 4);
            for (int i = 0; i < vendor.Count; i++)
                Assert.IsFalse(SpectrumRouting.IsProfile(vendor.GetSpectrum(i, true)));
            Assert.AreEqual(0, vendor.CentrixSpectra);
            Assert.AreEqual(vendor.Count, vendor.FallbackSpectra);
            var once = new CentrixSpectrumList(TestData.Open(input).Run.SpectrumList, null, ms1, ms2, CentroidMode.auto, 2,
                new Software("Centrix"));
            var twice = new CentrixSpectrumList(once, null, ms1, ms2, CentroidMode.auto, 2, new Software("Centrix"));
            for (int i = 0; i < twice.Count; i++)
                twice.GetSpectrum(i, true);
            Assert.AreEqual(twice.Count, twice.PassedThrough);
        }

        [TestMethod]
        public void TestCentrixRouting()
        {
            // The Stellar's spectra name a linear ion trap; the routing follows the analyzer, and a centroid passes.
            var spectra = TestData.Open(TestData.Path("Centrix", PROFILE_SLICE)).Run.SpectrumList;
            var spectrum = spectra.GetSpectrum(0, true);
            Assert.IsTrue(SpectrumRouting.IsProfile(spectrum));
            Assert.IsTrue(SpectrumRouting.IsLinearIonTrap(spectrum));
            Assert.AreEqual(SpectrumRoute.Centrix, SpectrumRouting.Route(spectrum, CentroidMode.auto));
            Assert.AreEqual(SpectrumRoute.Fallback, SpectrumRouting.Route(spectrum, CentroidMode.vendor));

            // An Orbitrap's profile goes to the fallback unless Centrix is asked for everything.
            var configuration = spectrum.ScanList.Scans[0].InstrumentConfiguration;
            var analyzer = configuration.ComponentList.Last(c => c.Type == ComponentType.Analyzer);
            var ionTrap = analyzer.CVParams.ToList();
            analyzer.CVParams.Clear();
            analyzer.Set(CVID.MS_orbitrap);
            Assert.IsFalse(SpectrumRouting.IsLinearIonTrap(spectrum));
            Assert.AreEqual(SpectrumRoute.Fallback, SpectrumRouting.Route(spectrum, CentroidMode.auto));
            Assert.AreEqual(SpectrumRoute.Centrix, SpectrumRouting.Route(spectrum, CentroidMode.centrix));
            analyzer.CVParams.Clear();
            analyzer.CVParams.AddRange(ionTrap);

            // With no analyzer named, the point spacing decides: a 0.2 Th grid is a linear ion trap's.
            spectrum.ScanList.Scans[0].InstrumentConfiguration = null;
            Assert.IsTrue(SpectrumRouting.IsLinearIonTrap(spectrum));
            SpectrumRouting.MarkCentroid(spectrum);
            Assert.AreEqual(SpectrumRoute.PassThrough, SpectrumRouting.Route(spectrum, CentroidMode.auto));
        }

        [TestMethod]
        public void TestCentrixAdaptiveRt()
        {
            // The Stellar slice is a Stellar's; two of its MS2 widened to 50 m/z, as Adaptive RT spectra are, are left
            // out by default and kept with --adaptive-rt include; its 2 m/z DIA windows are never taken for them.
            string input = TestData.Path("Centrix", PROFILE_SLICE);
            using var msd = TestData.Open(input);
            Assert.IsTrue(SpectrumRouting.IsStellar(msd));
            var simple = new Pwiz.Data.MsData.Spectra.SpectrumListSimple();
            for (int i = 0; i < msd.Run.SpectrumList.Count; i++)
                simple.Spectra.Add(msd.Run.SpectrumList.GetSpectrum(i, true));
            Assert.IsFalse(simple.Spectra.Any(s => SpectrumRouting.IsWiderThan(s, 10)));
            var widened = simple.Spectra.Where(s => s.Params.CvParamValueOrDefault(CVID.MS_ms_level, 0) == 2).Take(2).ToList();
            foreach (var spectrum in widened)
            {
                var window = spectrum.Precursors[0].IsolationWindow;
                window.Set(CVID.MS_isolation_window_lower_offset, 25.0, CVID.MS_m_z);
                window.Set(CVID.MS_isolation_window_upper_offset, 25.0, CVID.MS_m_z);
                Assert.IsTrue(SpectrumRouting.IsWiderThan(spectrum, 10));
                Assert.IsFalse(SpectrumRouting.IsWiderThan(spectrum, 0));
            }
            msd.Run.SpectrumList = simple;
            new Conversion(CentrixOptions.Parse(new[] { input }), new Log(true, false)).ExcludeAdaptiveRt(msd);
            var kept = msd.Run.SpectrumList;
            Assert.AreEqual(simple.Count - 2, kept.Count);
            var ids = Enumerable.Range(0, kept.Count).Select(i => kept.SpectrumIdentity(i)).ToList();
            CollectionAssert.AreEqual(Enumerable.Range(0, kept.Count).ToList(), ids.Select(id => id.Index).ToList());
            foreach (var spectrum in widened)
                Assert.IsFalse(ids.Any(id => id.Id == spectrum.Id), spectrum.Id);

            // Included, they are written, centroided by Centrix like the MS2 around them.
            var included = CentrixOptions.Parse(new[] { input, "--adaptive-rt", "include" });
            Assert.AreEqual(AdaptiveRtMode.include, included.AdaptiveRt);
            var ms1 = new CentrixCentroider(new CentrixParams { SigmaTh = 0.073 });
            var ms2 = new CentrixCentroider(new CentrixParams { SigmaTh = 0.22 });
            var all = new CentrixSpectrumList(simple, null, ms1, ms2, CentroidMode.auto, 2, new Software("Centrix"));
            Assert.AreEqual(simple.Count, all.Count);
            for (int i = 0; i < all.Count; i++)
                Assert.IsFalse(SpectrumRouting.IsProfile(all.GetSpectrum(i, true)));
        }

        [TestMethod]
        public void TestStellarZoomLabel()
        {
            // The Stellar writes its DIA MS2 as zoom scans: the slice's 14 MS2 carry the term and the "Z". Through the
            // command they come out without either, their MS1 untouched, the correction in the processing record; with
            // --keep-zoom-label as read.
            string input = TestData.Path("Centrix", PROFILE_SLICE);
            using (var read = TestData.Open(input))
            {
                var ms2 = Enumerable.Range(0, read.Run.SpectrumList.Count).Select(i => read.Run.SpectrumList.GetSpectrum(i))
                    .Where(s => s.Params.CvParamValueOrDefault(CVID.MS_ms_level, 0) == 2).ToList();
                Assert.AreEqual(14, ms2.Count);
                Assert.IsTrue(ms2.All(s => s.Params.HasCVParam(CVID.MS_zoom_scan) && FilterOf(s).Contains(" Z ms2 ")));

                // An Adaptive RT spectrum, wider than any DIA window, keeps its label.
                var wide = ms2[0];
                wide.Precursors[0].IsolationWindow.Set(CVID.MS_isolation_window_lower_offset, 25.0, CVID.MS_m_z);
                wide.Precursors[0].IsolationWindow.Set(CVID.MS_isolation_window_upper_offset, 25.0, CVID.MS_m_z);
                Assert.IsFalse(StellarZoomLabelFix.Relabel(wide, 10));
                Assert.IsTrue(wide.Params.HasCVParam(CVID.MS_zoom_scan));
                Assert.IsTrue(StellarZoomLabelFix.Relabel(ms2[1], 10));
                Assert.IsFalse(StellarZoomLabelFix.Relabel(ms2[1], 10), "a second pass finds nothing to remove");
                StringAssert.StartsWith(FilterOf(ms2[1]), "ITMS + p NSI ms2 ");
            }

            string directory = TestData.TempDirectory();
            try
            {
                foreach (bool keep in new[] { false, true })
                {
                    var args = new List<string> { input, "-o", Path.Combine(directory, keep ? "kept" : "fixed"), "-q" };
                    if (keep)
                        args.Add("--keep-zoom-label");
                    var options = CentrixOptions.Parse(args);
                    string output = options.OutputPathFor(input);
                    new Conversion(options, new Log(true, false)).Run(input, output);
                    using var written = TestData.Open(output);
                    var spectra = written.Run.SpectrumList;
                    int labelled = 0, ms1 = 0;
                    for (int i = 0; i < spectra.Count; i++)
                    {
                        var spectrum = spectra.GetSpectrum(i);
                        if (spectrum.Params.CvParamValueOrDefault(CVID.MS_ms_level, 0) == 1)
                        {
                            ms1++;
                            StringAssert.StartsWith(FilterOf(spectrum), "ITMS + p NSI r Full ms ");
                            continue;
                        }
                        bool zoom = spectrum.Params.HasCVParam(CVID.MS_zoom_scan);
                        Assert.AreEqual(zoom, FilterOf(spectrum).Contains(" Z ms2 "), FilterOf(spectrum));
                        if (zoom)
                            labelled++;
                    }
                    Assert.AreEqual(2, ms1);
                    Assert.AreEqual(keep ? 14 : 0, labelled);
                    bool recorded = written.DataProcessings.SelectMany(d => d.ProcessingMethods)
                        .Any(m => m.UserParams.Any(u => u.Name.Contains("zoom-scan label removed")));
                    Assert.AreEqual(!keep, recorded);
                }
            }
            finally
            {
                Directory.Delete(directory, true);
            }
        }

        private static string FilterOf(Pwiz.Data.MsData.Spectra.Spectrum spectrum)
        {
            return spectrum.ScanList.Scans[0].CvParamValueOrDefault(CVID.MS_filter_string, string.Empty);
        }

        [TestMethod]
        public void TestCentrixSigmaFromScanRate()
        {
            // A run at 125 kTh/s whose file names no instrument (an mzML stripped of it): its profile spectra are a
            // linear ion trap's by their 1/8 Th point spacing, Centrix centroids them, and the sigma follows the spacing,
            // 1.1 spacings, whatever the peaks' width. The calibrate command also measures that width, for comparing.
            var random = new Random(17);
            const double spacing = 0.125, peakSigma = 0.21;
            var spectra = new Pwiz.Data.MsData.Spectra.SpectrumListSimple();
            for (int k = 0; k < 40; k++)
            {
                var mz = Enumerable.Range(0, 8000).Select(i => 200 + i * spacing).ToArray();
                var centers = Enumerable.Range(0, 30).Select(p => 210 + 32 * p + random.NextDouble()).ToArray();
                var y = mz.Select(m => centers.Sum(c => 4000 * Math.Exp(-0.5 * Math.Pow((m - c) / peakSigma, 2))) +
                    10 * random.NextDouble()).ToArray();
                var spectrum = new Pwiz.Data.MsData.Spectra.Spectrum { Index = k, Id = "scan=" + (k + 1) };
                spectrum.Params.Set(CVID.MS_ms_level, 2);
                spectrum.Params.Set(CVID.MS_profile_spectrum);
                spectrum.SetMZIntensityArrays(mz, y, CVID.MS_number_of_detector_counts);
                spectra.Spectra.Add(spectrum);
            }
            Assert.IsTrue(SpectrumRouting.IsLinearIonTrap(spectra.GetSpectrum(0, true)));
            var conversion = new Conversion(CentrixOptions.Parse(new[] { "x", "--demux", "off" }), new Log(true, false));
            var plan = conversion.MakePlan(spectra);
            Assert.AreEqual(CentrixCalibration.SIGMA_PER_SPACING * spacing, plan.SigmaMs2, 1e-12);
            StringAssert.Contains(plan.SigmaMs2Source, "125 kTh/s");

            var measured = conversion.MakePlan(spectra, true);
            Assert.AreEqual(plan.SigmaMs2, measured.SigmaMs2, "measuring never changes the sigma used");
            Assert.AreEqual(peakSigma, measured.MeasuredSigmaMs2, 0.1 * peakSigma, "the peaks' width, measured");
            Assert.IsTrue(measured.MeasuredPeaksMs2 >= CentrixCalibration.MIN_PEAKS);
            Assert.IsTrue(double.IsNaN(measured.MeasuredSigmaMs1), "no MS1 to measure");
        }

        [TestMethod]
        public void TestCentrixExitCodes()
        {
            // 0 for success and for help, 1 when an input fails (the others still converted), 2 for a command-line
            // mistake; an output that exists is skipped, not replaced, without --overwrite.
            string input = TestData.Path("Centrix", PROFILE_SLICE);
            string directory = TestData.TempDirectory();
            try
            {
                Assert.AreEqual(2, Program.Main(new[] { input, "--bogus" }));
                Assert.AreEqual(2, Program.Main(Array.Empty<string>()));
                Assert.AreEqual(0, Program.Main(new[] { "--help" }));
                Assert.AreEqual(0, Program.Main(new[] { "--version" }));
                string missing = Path.Combine(directory, "missing.raw");
                Assert.AreEqual(1, Program.Main(new[] { missing, "-o", directory, "-q" }));

                Assert.AreEqual(0, Program.Main(new[] { input, "-o", directory, "-q" }));
                string output = Path.Combine(directory, "stellar-2th-profile-slice.centrix.mzML");
                Assert.IsTrue(File.Exists(output));
                File.WriteAllText(output, "not replaced");
                Assert.AreEqual(0, Program.Main(new[] { input, "-o", directory, "-q" }));
                Assert.AreEqual("not replaced", File.ReadAllText(output));
                Assert.AreEqual(0, Program.Main(new[] { input, "-o", directory, "-q", "--overwrite" }));
                Assert.AreNotEqual("not replaced", File.ReadAllText(output));

                // One bad input among good ones: the good one is still written.
                File.Delete(output);
                Assert.AreEqual(1, Program.Main(new[] { "-i", missing, input, "-o", directory, "-q" }));
                Assert.IsTrue(File.Exists(output));

                // calibrate writes nothing.
                var before = Directory.GetFiles(directory).Length;
                Assert.AreEqual(0, Program.Main(new[] { "calibrate", input, "-q" }));
                Assert.AreEqual(before, Directory.GetFiles(directory).Length);
            }
            finally
            {
                Directory.Delete(directory, true);
            }
        }

        [TestMethod]
        public void TestCentrixDemuxConversion()
        {
            // The Stellar SuperOverlap slice through the whole command with --demux auto: detected as a 4-fold stagger,
            // every MS2 written as its four 1 Th bins, each the joint solve's centroids of its share, as the pipeline
            // gives them read directly (to the 32-bit intensities the mzML stores).
            string input = TestData.Path("Demux", StellarDemuxFixtureTest.STELLAR_SLICE);
            var plan = TestData.Plan(input, CentrixOptions.Parse(new[] { input }));
            Assert.IsTrue(plan.Demultiplex, plan.DemuxReason);
            Assert.AreEqual(4, plan.Scheme.OverlapFactor);
            Assert.AreEqual(CentrixCalibration.SIGMA_PER_SPACING * 0.2, plan.SigmaMs2, 1e-12);
            Assert.IsFalse(TestData.Plan(input, CentrixOptions.Parse(new[] { input, "--demux", "off" })).Demultiplex);

            var source = new PwizDemuxSource(TestData.Open(input).Run.SpectrumList);
            var direct = new ScanningDemuxOptions { Joint = true, IonsFromFillTime = true, Threads = 4, GroupBins = 16 };
            direct.JointParameters.UseStaggeredDefaults();
            var pipeline = new StaggeredDemuxPipeline(source, direct, TextWriter.Null);

            string directory = TestData.TempDirectory();
            try
            {
                var options = CentrixOptions.Parse(new[] { input, "-o", directory, "-q", "-j", "3" });
                string output = options.OutputPathFor(input);
                new Conversion(options, new Log(true, false)).Run(input, output);
                using var written = TestData.Open(output);
                var spectra = written.Run.SpectrumList;
                Assert.AreEqual(pipeline.OutputCount, spectra.Count);
                Assert.AreEqual(800, spectra.Count);
                for (int o = 0; o < spectra.Count; o++)
                {
                    var spectrum = spectra.GetSpectrum(o, true);
                    Assert.IsFalse(SpectrumRouting.IsProfile(spectrum));
                    StringAssert.EndsWith(spectrum.Id, " demux=" + pipeline.PartOf(o));
                    var window = spectrum.Precursors[0].IsolationWindow;
                    Assert.AreEqual(pipeline.BinOf(o).Center,
                        window.CvParamValueOrDefault(CVID.MS_isolation_window_target_m_z, 0.0), 1e-9);
                    var (mz, ions) = pipeline.Peaks(o);
                    double perIon = pipeline.IntensityPerIon(o);
                    var writtenMz = spectrum.GetMZArray().Data;
                    var writtenIntensity = spectrum.GetIntensityArray().Data;
                    Assert.AreEqual(mz.Length, writtenMz.Count, $"spectrum {o} peaks");
                    for (int k = 0; k < mz.Length; k++)
                    {
                        Assert.AreEqual(mz[k], writtenMz[k], 1e-9);
                        Assert.AreEqual(ions[k] * perIon, writtenIntensity[k], 1e-6 * ions[k] * perIon);
                    }
                }
            }
            finally
            {
                Directory.Delete(directory, true);
            }
        }
    }
}
