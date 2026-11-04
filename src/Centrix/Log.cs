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
    /// <summary>The console log: progress to standard output, warnings and errors to standard error.</summary>
    internal sealed class Log
    {
        private readonly bool _quiet;
        private readonly bool _verbose;

        public Log(bool quiet, bool verbose)
        {
            _quiet = quiet;
            _verbose = verbose;
            Writer = quiet ? TextWriter.Null : verbose ? Console.Out : new DetailWriter();
        }

        /// <summary>Where the demultiplexer's per-batch lines go: standard output when verbose, nowhere otherwise.</summary>
        public TextWriter Writer { get; }

        public void Info(string message)
        {
            if (!_quiet)
                Console.Out.WriteLine(message);
        }

        public void Detail(string message)
        {
            if (_verbose && !_quiet)
                Console.Out.WriteLine(message);
        }

        public void Warn(string message)
        {
            Console.Error.WriteLine("Warning: " + message);
        }

        public void Error(string message)
        {
            Console.Error.WriteLine("Error: " + message);
        }

        /// <summary>Keeps the demultiplexer's scheme line, which says what it found, and drops its per-batch timings.</summary>
        private sealed class DetailWriter : TextWriter
        {
            public override System.Text.Encoding Encoding => Console.Out.Encoding;

            public override void Write(char value)
            {
            }

            public override void WriteLine(string value)
            {
                if (value != null && !value.StartsWith(" ", StringComparison.Ordinal))
                    Console.Out.WriteLine(value);
            }
        }
    }
}
