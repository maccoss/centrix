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
using System.IO;

namespace Centrix.Cli
{
    /// <summary>The centrix command line.</summary>
    internal static class Program
    {
        internal static int Main(string[] args)
        {
            CentrixOptions options;
            try
            {
                options = CentrixOptions.Parse(args);
            }
            catch (ArgumentException e)
            {
                Console.Error.WriteLine(e.Message);
                Console.Error.WriteLine("Run centrix --help for the options.");
                return 2;
            }
            if (options.ShowHelp)
            {
                Console.Out.Write(CentrixOptions.USAGE);
                return 0;
            }
            if (options.ShowVersion)
            {
                Console.Out.WriteLine("centrix " + CentrixVersion.Current);
                return 0;
            }

            // Vendor SDKs a package leaves out (their licenses forbid redistributing them) are fetched on first use,
            // pinned and hash-checked, into a per-user cache, as ProteoWizard's own tools fetch them; a build that has
            // them beside the executable uses those.
            Pwiz.Vendor.Common.VendorSdkLoader.RegisterAssemblyResolver();
            var log = new Log(options.Quiet, options.Verbose);
            log.Info($"centrix {CentrixVersion.Current}, {options.EffectiveThreads} threads");
            var conversion = new Conversion(options, log);
            int failed = 0;
            foreach (string input in options.Inputs)
            {
                try
                {
                    if (!File.Exists(input) && !Directory.Exists(input))
                        throw new FileNotFoundException($"{input} does not exist.");
                    if (options.Command == CentrixCommand.calibrate)
                    {
                        conversion.Calibrate(input);
                        continue;
                    }
                    string output = options.OutputPathFor(input);
                    if (File.Exists(output) && !options.Overwrite)
                    {
                        log.Warn($"{output} exists; skipped (--overwrite replaces it).");
                        continue;
                    }
                    conversion.Run(input, output);
                }
                catch (Exception e)
                {
                    failed++;
                    log.Error($"{input}: {Describe(e)}");
                    if (options.Verbose)
                        Console.Error.WriteLine(e);
                }
            }
            return failed == 0 ? 0 : 1;
        }

        private static string Describe(Exception e)
        {
            while (e is AggregateException { InnerExceptions.Count: 1 } aggregate)
                e = aggregate.InnerExceptions[0];
            return e.InnerException != null && e is Pwiz.Util.Misc.EnumerationException
                ? e.InnerException.Message
                : e.Message;
        }
    }
}
