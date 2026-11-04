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
using System.Linq;
using Centrix.Demux;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Centrix.Tests
{
    /// <summary>
    /// Tests for the demultiplexer's parts against exact truth: the NNLS solver, the portable math, the RT interpolants
    /// and the scheme detection.
    /// </summary>
    [TestClass]
    public class DemuxTest
    {
        private const double BIN_WIDTH = 4.0;
        private const double WINDOW_WIDTH = 8.0;
        private const double RANGE_LOW = 500.0;
        private const double RANGE_HIGH = 540.0;
        private const double SCAN_MINUTES = 0.1 / 60.0;

        /// <summary>
        /// NNLS: exact and constrained cases, then agreement with an independent brute-force
        /// active-set enumeration over random small systems (spec gate G1.3).
        /// </summary>
        [TestMethod]
        public void TestDemuxNnlsSolver()
        {
            // Full rank, feasible: the unconstrained path returns the exact solution.
            var identityLike = new double[,] { { 1, 1, 0 }, { 0, 1, 1 }, { 0, 0, 1 } };
            var solver = new NnlsSolver(identityLike);
            Assert.IsTrue(solver.IsFullColumnRank);
            var x = new double[3];
            var path = solver.Solve(new[] { 3.0, 5.0, 2.0 }, x, new NnlsSolver.Workspace(3));
            Assert.AreEqual(NnlsPath.unconstrained, path);
            AssertArrayNear(new[] { 0.0, 3.0, 2.0 }, x, 1e-12);

            // The constraint binds: the unconstrained answer has a negative component.
            path = solver.Solve(new[] { 1.0, 5.0, 2.0 }, x, new NnlsSolver.Workspace(3));
            Assert.AreEqual(NnlsPath.active_set, path);
            Assert.IsTrue(x.All(v => v >= 0));

            // All-zero right-hand side.
            path = solver.Solve(new double[3], x, new NnlsSolver.Workspace(3));
            Assert.AreEqual(NnlsPath.zero, path);
            AssertArrayNear(new double[3], x, 0);

            // Underdetermined (the covered-bins block shape): no pseudo-inverse, still solvable.
            var ladder = new double[,] { { 1, 1, 0 }, { 0, 1, 1 } };
            var ladderSolver = new NnlsSolver(ladder);
            Assert.IsFalse(ladderSolver.IsFullColumnRank);
            ladderSolver.Solve(new[] { 2.0, 0.0 }, x, new NnlsSolver.Workspace(3));
            // Only the first bin can explain signal in window 1 and none in window 2.
            AssertArrayNear(new[] { 2.0, 0.0, 0.0 }, x, 1e-12);

            CompareWithBruteForce(3000, 12345);
        }

        /// <summary>
        /// PortableMath: the same bits on every operating system, pinned here so the Linux and Windows builds both check
        /// them (two of the arguments are ones where the C libraries' exp and log disagree), and within a few units in the
        /// last place of Math.Exp and Math.Log.
        /// </summary>
        [TestMethod]
        public void TestDemuxPortableMath()
        {
            var expected = new (string Function, double X, ulong Bits)[]
            {
                (@"exp", -0.5, 0x3FE368B2FC6F960AUL),
                (@"exp", -4.710385710268998, 0x3F826F46B55AC868UL),
                (@"exp", 1.0, 0x4005BF0A8B14576AUL),
                (@"exp", -41.40463204448576, 0x3C333C6F85FDB7A6UL),
                (@"exp", -700.0, 0x00D14F2B0FB9307FUL),
                (@"exp", -745.0, 0x0000000000000001UL),
                (@"exp", 709.0, 0x7FDD422D2BE5DC9BUL),
                (@"log", 2.0, 0x3FE62E42FEFA39EFUL),
                (@"log", 1.8675521641060904, 0x3FE3FCF50E27F9EAUL),
                (@"log", 600.5234, 0x401997594E079217UL),
                (@"log", 0.75, 0xBFD269621134DB92UL),
                (@"log", 1e-310, 0xC0864E69394D9508UL),
                (@"log", 1e300, 0x4085963447F87FB5UL),
            };
            foreach (var (function, x, bits) in expected)
            {
                double value = function == @"exp" ? PortableMath.Exp(x) : PortableMath.Log(x);
                Assert.AreEqual(bits, (ulong)BitConverter.DoubleToInt64Bits(value), string.Format(@"{0}({1:R})", function, x));
            }

            Assert.AreEqual(1.0, PortableMath.Exp(0));
            Assert.AreEqual(0.0, PortableMath.Exp(-800));
            Assert.AreEqual(double.PositiveInfinity, PortableMath.Exp(710));
            Assert.IsTrue(double.IsNaN(PortableMath.Exp(double.NaN)));
            Assert.AreEqual(0.0, PortableMath.Log(1));
            Assert.AreEqual(double.NegativeInfinity, PortableMath.Log(0));
            Assert.AreEqual(double.PositiveInfinity, PortableMath.Log(double.PositiveInfinity));
            Assert.IsTrue(double.IsNaN(PortableMath.Log(-1)));

            var random = new Random(4711);
            for (int i = 0; i < 100000; i++)
            {
                double x = 1400 * random.NextDouble() - 700;
                Assert.IsTrue(UlpDistance(PortableMath.Exp(x), Math.Exp(x)) <= 2, string.Format(@"exp({0:R})", x));
                double y = Math.ScaleB(1 + random.NextDouble(), random.Next(-1070, 1020));
                double log = Math.Log(y);
                Assert.IsTrue(Math.Abs(PortableMath.Log(y) - log) <= 4e-16 * Math.Max(1, Math.Abs(log)), string.Format(@"log({0:R})", y));
            }
        }

        /// <summary>
        /// The RT interpolants: exactness where they must be exact, holding at the ends, and the
        /// apex-bias ordering behind the choice of makima as the default.
        /// </summary>
        [TestMethod]
        public void TestDemuxInterpolation()
        {
            var methods = new[]
            {
                RtInterpolation.makima, RtInterpolation.pchip,
                RtInterpolation.natural_three_point, RtInterpolation.linear,
            };
            var times = new[] { 1.0, 2.0, 3.5, 4.0, 5.0, 6.5 };
            var linearValues = times.Select(t => 3 * t + 2).ToArray();
            foreach (var method in methods)
            {
                // Linear data is reproduced exactly between the samples.
                foreach (double t in new[] { 1.5, 2.7, 3.9, 4.4, 6.0 })
                    Assert.AreEqual(3 * t + 2, RtInterpolator.Interpolate(method, times, linearValues, times.Length, t), 1e-9);
                // Nodes are reproduced exactly, and the ends are held, not extrapolated.
                Assert.AreEqual(linearValues[2], RtInterpolator.Interpolate(method, times, linearValues, times.Length, times[2]), 1e-12);
                Assert.AreEqual(linearValues[0], RtInterpolator.Interpolate(method, times, linearValues, times.Length, 0.0));
                Assert.AreEqual(linearValues[5], RtInterpolator.Interpolate(method, times, linearValues, times.Length, 9.0));
            }

            // PCHIP never overshoots a step.
            var step = new[] { 0.0, 0.0, 0.0, 10.0, 10.0, 10.0 };
            for (double t = 1.0; t <= 6.5; t += 0.05)
            {
                double v = RtInterpolator.Interpolate(RtInterpolation.pchip, times, step, times.Length, t);
                Assert.IsTrue(v >= -1e-12 && v <= 10 + 1e-12);
            }

            // Apex bias on a Gaussian sampled 4 times per FWHM, evaluated half a cycle off the
            // samples: makima beats PCHIP and the pwiz 3-point natural spline.
            double makima = MeanApexError(RtInterpolation.makima, 4.0);
            double pchip = MeanApexError(RtInterpolation.pchip, 4.0);
            double natural = MeanApexError(RtInterpolation.natural_three_point, 4.0);
            double linear = MeanApexError(RtInterpolation.linear, 4.0);
            Assert.IsTrue(makima < pchip, string.Format(@"makima {0} vs pchip {1}", makima, pchip));
            Assert.IsTrue(makima < natural, string.Format(@"makima {0} vs natural {1}", makima, natural));
            Assert.IsTrue(pchip < linear, string.Format(@"pchip {0} vs linear {1}", pchip, linear));
        }

        /// <summary>
        /// Scheme detection: k=2 and k=3 staggers, boundary jitter, variable widths, a
        /// non-overlapping scheme, and a window that first appears late in the run.
        /// </summary>
        [TestMethod]
        public void TestDemuxSchemeDetection()
        {
            // k=2 staggered: set A tiles 500-540, set B is offset by half a window and extends
            // one half-window past each end, so the two edge bins are covered once.
            var staggered = StaggeredCycle(RANGE_LOW, RANGE_HIGH, WINDOW_WIDTH, 2);
            var scheme = DemuxSchemeDetector.Detect(Repeat(staggered, 3));
            Assert.AreEqual(DemuxSchemeKind.overlapping, scheme.Kind);
            Assert.AreEqual(2, scheme.OverlapFactor);
            Assert.AreEqual(11, scheme.Windows.Count);
            Assert.AreEqual(12, scheme.Bins.Count);
            Assert.AreEqual(496.0, scheme.Bins[0].LowerBound, 1e-9);
            Assert.AreEqual(544.0, scheme.Bins[11].UpperBound, 1e-9);
            foreach (var bin in scheme.Bins)
                Assert.AreEqual(BIN_WIDTH, bin.Width, 1e-9);
            foreach (var window in scheme.Windows)
                Assert.AreEqual(2, window.BinCount);
            // Window indices follow m/z, whatever the acquisition order.
            Assert.AreEqual(3 * staggered.Count, scheme.WindowOfSpectrum.Length);
            Assert.AreEqual(496.0, scheme.Windows[scheme.WindowOfSpectrum[5]].LowerBound, 1e-9);

            // Reported edges jittered by 10 mTh give the same bins.
            var jittered = Repeat(staggered, 3).Select((w, i) =>
                new IsolationWindow(w.Center + ((i % 3) - 1) * 0.01, w.LowerOffset, w.UpperOffset)).ToList();
            var jitterScheme = DemuxSchemeDetector.Detect(jittered);
            Assert.AreEqual(12, jitterScheme.Bins.Count);
            Assert.AreEqual(2, jitterScheme.OverlapFactor);
            Assert.AreEqual(11, jitterScheme.Windows.Count);

            // k=3: 12 Th windows stepped by 4 Th.
            var k3 = StaggeredCycle(RANGE_LOW, RANGE_HIGH, 12.0, 3);
            var k3Scheme = DemuxSchemeDetector.Detect(Repeat(k3, 2));
            Assert.AreEqual(3, k3Scheme.OverlapFactor);
            Assert.IsTrue(k3Scheme.Bins.All(b => Math.Abs(b.Width - BIN_WIDTH) < 1e-9));

            // Variable widths (spec G4.3): no zero or negative width bins.
            var variable = new List<IsolationWindow>();
            var edges = new[] { 400.0, 406.0, 414.0, 424.0, 436.0 };
            for (int i = 0; i + 1 < edges.Length; i++)
                variable.Add(Window(edges[i], edges[i + 1]));
            for (int i = 0; i + 1 < edges.Length; i++)
            {
                double half = (edges[i + 1] - edges[i]) / 2;
                variable.Add(Window(edges[i] + half, edges[i + 1] + half));
            }
            var variableScheme = DemuxSchemeDetector.Detect(Repeat(variable, 2));
            Assert.AreEqual(2, variableScheme.OverlapFactor);
            Assert.IsTrue(variableScheme.Bins.All(b => b.Width > DemuxSchemeDetector.DEFAULT_MINIMUM_BIN_WIDTH));

            // Non-overlapping: nothing to do.
            var plain = new List<IsolationWindow>();
            for (double lo = RANGE_LOW; lo < RANGE_HIGH; lo += WINDOW_WIDTH)
                plain.Add(Window(lo, lo + WINDOW_WIDTH));
            Assert.AreEqual(DemuxSchemeKind.non_overlapping, DemuxSchemeDetector.Detect(Repeat(plain, 2)).Kind);

            // Ordinary DIA whose neighbors overlap by a 1 Th margin is not a stagger.
            var margins = new List<IsolationWindow>();
            for (double lo = 400; lo < 500; lo += 11)
                margins.Add(Window(lo, lo + 12));
            var marginScheme = DemuxSchemeDetector.Detect(Repeat(margins, 2));
            Assert.AreEqual(DemuxSchemeKind.non_overlapping, marginScheme.Kind);
            Assert.AreEqual(1, marginScheme.OverlapFactor);
            Assert.AreEqual(2, marginScheme.MaxCoverage);

            // A window seen only after several cycles is still a window.
            var late = Repeat(plain, 4);
            late.Add(Window(RANGE_HIGH, RANGE_HIGH + WINDOW_WIDTH));
            Assert.AreEqual(plain.Count + 1, DemuxSchemeDetector.Detect(late).Windows.Count);
        }

        private static void CompareWithBruteForce(int systems, int seed)
        {
            var random = new Random(seed);
            var workspace = new NnlsSolver.Workspace(8);
            for (int s = 0; s < systems; s++)
            {
                int rows = random.Next(2, 10);
                int columns = random.Next(1, 8);
                bool mask = random.Next(2) == 0;
                var a = new double[rows, columns];
                for (int r = 0; r < rows; r++)
                {
                    for (int c = 0; c < columns; c++)
                        a[r, c] = mask ? random.Next(2) : random.NextDouble();
                }
                var b = new double[rows];
                for (int r = 0; r < rows; r++)
                    b[r] = random.NextDouble() * 2 - 0.5;

                var solver = new NnlsSolver(a);
                var x = new double[columns];
                var path = solver.Solve(b, x, workspace);
                Assert.AreNotEqual(NnlsPath.iteration_cap, path);
                Assert.IsTrue(x.All(v => v >= 0));
                double objective = Objective(a, b, x);
                double best = BruteForceObjective(a, b);
                Assert.AreEqual(best, objective, 1e-9 * Math.Max(1, best),
                    string.Format(@"system {0}: {1} x {2}", s, rows, columns));

                // The normal-equations path with the incremental factor, from x = 0 and warm-started
                // from a feasible point near the answer, reaches the same optimum.
                var (ata, atb) = NormalEquations(a, b);
                var cold = new double[columns];
                Assert.AreNotEqual(NnlsPath.iteration_cap, NnlsSolver.SolveNormal(ata, atb, columns, cold, workspace));
                Assert.AreEqual(best, Objective(a, b, cold), 1e-9 * Math.Max(1, best), @"cold, system " + s);
                var start = x.Select(v => random.Next(3) == 0 ? 0 : v + random.NextDouble() * 0.1).ToArray();
                var warm = new double[columns];
                Assert.AreNotEqual(NnlsPath.iteration_cap,
                    NnlsSolver.SolveNormal(ata, atb, columns, warm, workspace, 0, start));
                Assert.AreEqual(best, Objective(a, b, warm), 1e-9 * Math.Max(1, best), @"warm, system " + s);
            }
        }

        private static (double[] Ata, double[] Atb) NormalEquations(double[,] a, double[] b)
        {
            int rows = a.GetLength(0), columns = a.GetLength(1);
            var ata = new double[columns * columns];
            var atb = new double[columns];
            for (int i = 0; i < columns; i++)
            {
                for (int r = 0; r < rows; r++)
                    atb[i] += a[r, i] * b[r];
                for (int j = 0; j < columns; j++)
                {
                    for (int r = 0; r < rows; r++)
                        ata[i * columns + j] += a[r, i] * a[r, j];
                }
            }
            return (ata, atb);
        }

        /// <summary>
        /// The NNLS optimum by exhaustion: the best non-negative unconstrained least-squares
        /// solution over every column subset, solved by Gaussian elimination with partial
        /// pivoting. Deliberately shares no code with <see cref="NnlsSolver"/>.
        /// </summary>
        private static double BruteForceObjective(double[,] a, double[] b)
        {
            int rows = a.GetLength(0), columns = a.GetLength(1);
            double best = Objective(a, b, new double[columns]);
            for (int subset = 1; subset < 1 << columns; subset++)
            {
                var index = Enumerable.Range(0, columns).Where(c => (subset & (1 << c)) != 0).ToArray();
                int p = index.Length;
                var m = new double[p, p + 1];
                for (int i = 0; i < p; i++)
                {
                    for (int j = 0; j < p; j++)
                    {
                        for (int r = 0; r < rows; r++)
                            m[i, j] += a[r, index[i]] * a[r, index[j]];
                    }
                    for (int r = 0; r < rows; r++)
                        m[i, p] += a[r, index[i]] * b[r];
                }
                var z = SolveGaussian(m, p);
                if (z == null || z.Any(v => v < 0))
                    continue;
                var x = new double[columns];
                for (int i = 0; i < p; i++)
                    x[index[i]] = z[i];
                best = Math.Min(best, Objective(a, b, x));
            }
            return best;
        }

        private static double[] SolveGaussian(double[,] m, int p)
        {
            for (int col = 0; col < p; col++)
            {
                int pivot = col;
                for (int r = col + 1; r < p; r++)
                {
                    if (Math.Abs(m[r, col]) > Math.Abs(m[pivot, col]))
                        pivot = r;
                }
                if (Math.Abs(m[pivot, col]) < 1e-9)
                    return null;
                for (int j = 0; j <= p; j++)
                {
                    double tmp = m[col, j];
                    m[col, j] = m[pivot, j];
                    m[pivot, j] = tmp;
                }
                for (int r = 0; r < p; r++)
                {
                    if (r == col)
                        continue;
                    double f = m[r, col] / m[col, col];
                    for (int j = col; j <= p; j++)
                        m[r, j] -= f * m[col, j];
                }
            }
            var z = new double[p];
            for (int i = 0; i < p; i++)
                z[i] = m[i, p] / m[i, i];
            return z;
        }

        private static double Objective(double[,] a, double[] b, double[] x)
        {
            double sum = 0;
            for (int r = 0; r < a.GetLength(0); r++)
            {
                double residual = -b[r];
                for (int c = 0; c < a.GetLength(1); c++)
                    residual += a[r, c] * x[c];
                sum += residual * residual;
            }
            return sum;
        }

        /// <summary>
        /// Mean absolute error at the apex region, as a fraction of peak height, for a Gaussian
        /// sampled <paramref name="pointsPerFwhm"/> times per FWHM and evaluated at the
        /// half-cycle offsets, over evenly spaced apex phases.
        /// </summary>
        private static double MeanApexError(RtInterpolation method, double pointsPerFwhm)
        {
            const double fwhm = 1.0;
            double sigma = fwhm / 2.3548;
            double cycle = fwhm / pointsPerFwhm;
            int perSide = RtInterpolator.SamplesPerSide(method);
            double total = 0;
            int count = 0;
            for (int phase = 0; phase < 20; phase++)
            {
                double apex = phase * cycle / 20.0;
                // The sample interval containing the apex, target at its midpoint.
                double target = Math.Floor(apex / cycle) * cycle + cycle / 2;
                var times = new double[2 * perSide];
                var values = new double[2 * perSide];
                for (int s = 0; s < 2 * perSide; s++)
                {
                    times[s] = target - cycle / 2 + (s - perSide + 1) * cycle;
                    values[s] = Math.Exp(-0.5 * Math.Pow((times[s] - apex) / sigma, 2));
                }
                double truth = Math.Exp(-0.5 * Math.Pow((target - apex) / sigma, 2));
                total += Math.Abs(RtInterpolator.Interpolate(method, times, values, times.Length, target) - truth);
                count++;
            }
            return total / count;
        }

        private static List<IsolationWindow> StaggeredCycle(double low, double high, double width, int k)
        {
            // Set j is offset by j*width/k and shifted down so that every bin in [low, high]
            // is covered k times; windows beyond the range edges supply the edge coverage.
            var cycle = new List<IsolationWindow>();
            double step = width / k;
            for (int set = 0; set < k; set++)
            {
                for (double lo = low - set * step; lo < high; lo += width)
                {
                    if (lo + width <= low)
                        continue;
                    cycle.Add(Window(lo, lo + width));
                }
            }
            return cycle;
        }

        private static List<IsolationWindow> Repeat(List<IsolationWindow> cycle, int times)
        {
            var result = new List<IsolationWindow>();
            for (int i = 0; i < times; i++)
                result.AddRange(cycle);
            return result;
        }

        private static IsolationWindow Window(double low, double high)
        {
            return IsolationWindow.Symmetric((low + high) / 2, (high - low) / 2);
        }

        private static long UlpDistance(double a, double b)
        {
            return Math.Abs(BitConverter.DoubleToInt64Bits(a) - BitConverter.DoubleToInt64Bits(b));
        }

        private static void AssertArrayNear(double[] expected, double[] actual, double tolerance)
        {
            Assert.AreEqual(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
                Assert.AreEqual(expected[i], actual[i], tolerance);
        }
    }
}
