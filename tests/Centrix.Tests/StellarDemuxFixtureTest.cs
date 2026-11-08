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
using System.Globalization;
using System.IO;
using System.Linq;
using Centrix.Cli;
using Centrix.Demux;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Pwiz.Data.Common.Cv;

namespace Centrix.Tests
{
    /// <summary>
    /// The staggered joint solve on real Thermo Stellar SuperOverlap profile spectra (<c>Data/Demux/README.md</c>):
    /// 4 Th windows in four passes 1 Th apart, each acquired spectrum split among the four 1 Th bins it covers, each
    /// share centroided by Centrix, every spectrum's ions counted from its fill time and the ion trap's ion response.
    /// The golden was made by the solve in Osprey.Demux (ProteoWizard/pwiz#4710) reading the same spectra, so this holds
    /// the port, and ProteoWizard's reading of the mzML, to it. Osprey's reader keeps each fill time as a 32-bit float,
    /// which the golden carries (<see cref="FillTimeAsFloat"/>): the solve's choices of peaks are that sensitive, a
    /// double's fill times moving the first bin's intensity by 0.1% and the next bin's peak count by two.
    /// <para>Regenerating the golden: set <see cref="REBLESS_VARIABLE"/> to 1 and run the test, which rewrites it and
    /// fails; review the diff and say in the commit why the output moved.</para>
    /// </summary>
    [TestClass]
    public class StellarDemuxFixtureTest
    {
        private const string FOLDER = "Demux";
        private const string REBLESS_VARIABLE = "CENTRIX_REBLESS_DEMUX_FIXTURE";
        internal const string STELLAR_SLICE = "stellar-superoverlap-profile-slice.mzML";
        internal const string STELLAR_GOLDEN = "stellar-superoverlap-joint-demux.golden.tsv";
        internal const string STELLAR_MEASURE_GOLDEN = "stellar-superoverlap-joint-demux-sigma038.golden.tsv";

        /// <summary>The summed intensities and mean m/z of a golden are compared to this relative tolerance.</summary>
        private const double RELATIVE_TOLERANCE = 1e-9;

        /// <summary>
        /// The joint solve twice: at the default sigma, which detects the most peptides (three point spacings, 0.6 Th),
        /// and at 0.38 Th, which measures fragment m/z best and keeps most M+1 isotopes apart (<c>--sigma-ms2 0.38</c>).
        /// </summary>
        [TestMethod]
        public void TestDemuxStellarFixture()
        {
            CheckStellarJoint(0, STELLAR_GOLDEN);
            CheckStellarJoint(0.38, STELLAR_MEASURE_GOLDEN);
        }

        /// <summary>The slice through the staggered joint solve at one profile peak sigma, Th (0: the run's).</summary>
        private static void CheckStellarJoint(double peakSigmaTh, string golden)
        {
            var msd = TestData.Open(TestData.Path(FOLDER, STELLAR_SLICE));
            var spectra = msd.Run.SpectrumList;
            Assert.AreEqual(200, spectra.Count);
            var source = new PwizDemuxSource(spectra);
            for (int i = 0; i < spectra.Count; i++)
            {
                var info = source.Describe(i);
                Assert.AreEqual(2, info.MsLevel);
                Assert.IsTrue(info.InjectionTime > 0, $"spectrum {i} fill time");
                Assert.AreEqual(IonResponse.Of(MassAnalyzerKind.ion_trap), info.IonResponse, $"spectrum {i} ion response");
            }

            var options = new ScanningDemuxOptions
            {
                Joint = true, IonsFromFillTime = true, Threads = 4, LinearPeakSigmaTh = peakSigmaTh
            };
            options.JointParameters.UseStaggeredDefaults();
            var pipeline = new StaggeredDemuxPipeline(new FillTimeAsFloat(source), options, TextWriter.Null);
            Assert.AreEqual(DemuxSchemeKind.overlapping, pipeline.Scheme.Kind);
            Assert.AreEqual(4, pipeline.Scheme.OverlapFactor);
            Assert.AreEqual(8, pipeline.Scheme.Windows.Count);
            Assert.AreEqual(11, pipeline.Scheme.Bins.Count);
            Assert.AreEqual(4 * spectra.Count, pipeline.OutputCount);
            var lines = new List<string>();
            for (int o = 0; o < pipeline.OutputCount; o++)
            {
                var bin = pipeline.BinOf(o);
                var (mz, ions) = pipeline.Peaks(o);
                double perIon = pipeline.IntensityPerIon(o);
                // The golden numbers each acquired spectrum by its place in the slice, as Osprey's reader did.
                lines.Add(Line(o, pipeline.BaseIndex(o).ToString(CultureInfo.InvariantCulture), bin.LowerBound,
                    bin.UpperBound, mz, ions.Select(v => v * perIon).ToArray()));
            }
            CheckGolden(golden, lines, RELATIVE_TOLERANCE);
        }

        /// <summary>A source whose fill times are rounded to 32-bit floats, as Osprey's reader keeps them.</summary>
        private sealed class FillTimeAsFloat : IDemuxSource
        {
            private readonly IDemuxSource _source;

            public FillTimeAsFloat(IDemuxSource source)
            {
                _source = source;
            }

            public int Count => _source.Count;

            public string NativeId(int index) => _source.NativeId(index);

            public DemuxSpectrumInfo Describe(int index)
            {
                var info = _source.Describe(index);
                return new DemuxSpectrumInfo(info.MsLevel, info.HasWindow, info.Window, info.Time, (float)info.InjectionTime,
                    info.IonResponse);
            }

            public void Read(int index, out IReadOnlyList<double> mz, out IReadOnlyList<double> intensity) =>
                _source.Read(index, out mz, out intensity);

            public void Release(int index) => _source.Release(index);

            public void Forget(int index) => _source.Forget(index);
        }

        /// <summary>
        /// One output spectrum: its index, source scan, window, peak count, summed intensity and intensity-weighted
        /// mean m/z.
        /// </summary>
        internal static string Line(int output, string scan, double low, double high, IReadOnlyList<double> mz,
            IReadOnlyList<double> intensity)
        {
            var ic = CultureInfo.InvariantCulture;
            double sum = 0, weighted = 0;
            for (int i = 0; i < mz.Count; i++)
            {
                sum += intensity[i];
                weighted += intensity[i] * mz[i];
            }
            return string.Join("\t", output.ToString(ic), scan, low.ToString("F4", ic), high.ToString("F4", ic),
                mz.Count.ToString(ic), sum.ToString("R", ic), (sum > 0 ? weighted / sum : 0).ToString("R", ic));
        }

        /// <summary>
        /// The output spectra against the committed golden: index, scan, window and peak count exactly, summed intensity
        /// and mean m/z to a relative tolerance. With the rebless variable set, rewrites the golden and fails, so a
        /// changed output is never accepted silently.
        /// </summary>
        internal static void CheckGolden(string name, IReadOnlyList<string> lines, double relativeTolerance)
        {
            string path = TestData.Path(FOLDER, name);
            var all = new List<string> { string.Join("\t", "output", "scan", "window_lower", "window_upper", "peaks",
                "intensity", "mean_mz") };
            all.AddRange(lines);
            if (Environment.GetEnvironmentVariable(REBLESS_VARIABLE) == "1")
            {
                // The copy in the source tree, not the one beside the tests.
                string source = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "Data", FOLDER, name));
                File.WriteAllLines(source, all);
                Assert.Fail("Reblessed {0} ({1} was set); review the diff and commit it.", source, REBLESS_VARIABLE);
            }
            Assert.IsTrue(File.Exists(path), path);
            var expected = File.ReadAllLines(path).Where(l => l.Length > 0).ToArray();
            Assert.AreEqual(expected.Length, all.Count, "output spectra");
            for (int i = 1; i < all.Count; i++)
            {
                var want = expected[i].Split('\t');
                var have = all[i].Split('\t');
                for (int f = 0; f < 5; f++)
                    Assert.AreEqual(want[f], have[f], $"{name} line {i} field {f}");
                for (int f = 5; f < 7; f++)
                {
                    double wantValue = double.Parse(want[f], CultureInfo.InvariantCulture);
                    double haveValue = double.Parse(have[f], CultureInfo.InvariantCulture);
                    Assert.AreEqual(wantValue, haveValue, Math.Abs(wantValue) * relativeTolerance, $"{name} line {i} field {f}");
                }
            }
        }
    }
}
