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
using Centrix.Cli;
using Pwiz.Data.Common.Cv;
using Pwiz.Data.MsData;
using Pwiz.Data.MsData.Readers;

namespace Centrix.Tests
{
    /// <summary>The committed fixtures (<c>Data/</c>, copied beside the tests) and how the tests read them.</summary>
    internal static class TestData
    {
        public static string Path(params string[] parts)
        {
            var all = new List<string> { AppContext.BaseDirectory, "Data" };
            all.AddRange(parts);
            return System.IO.Path.Combine(all.ToArray());
        }

        /// <summary>A file's spectra as ProteoWizard reads them, numbered within their MS level from 0.</summary>
        public static List<(int Level, int Ordinal, double[] Mz, double[] Intensity)> ProfileSpectra(string path)
        {
            using var msd = Open(path);
            var spectra = new List<(int, int, double[], double[])>();
            var ordinal = new Dictionary<int, int>();
            for (int i = 0; i < msd.Run.SpectrumList.Count; i++)
            {
                var spectrum = msd.Run.SpectrumList.GetSpectrum(i, true);
                int level = spectrum.Params.CvParamValueOrDefault(CVID.MS_ms_level, 0);
                ordinal.TryGetValue(level, out int n);
                ordinal[level] = n + 1;
                spectra.Add((level, n, spectrum.GetMZArray().Data.ToArray(), spectrum.GetIntensityArray().Data.ToArray()));
            }
            return spectra;
        }

        public static MSData Open(string path)
        {
            var msd = new MSData();
            Conversion.Readers().Read(path, msd, new ReaderConfig());
            return msd;
        }

        /// <summary>What centrix would settle for a file: its sigmas and whether to demultiplex.</summary>
        public static Conversion.Plan Plan(string path, CentrixOptions options)
        {
            using var msd = Open(path);
            return new Conversion(options, new Log(true, false)).MakePlan(msd.Run.SpectrumList);
        }

        /// <summary>A fresh directory under the system's temporary folder, for a test's output.</summary>
        public static string TempDirectory()
        {
            string directory = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "centrix-test-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(directory);
            return directory;
        }
    }
}
