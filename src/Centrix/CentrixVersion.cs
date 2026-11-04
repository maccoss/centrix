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

using System.Reflection;

namespace Centrix.Cli
{
    /// <summary>
    /// Centrix's version as it is released, CalVer <c>YY.feature.patch</c> (<c>26.1.0</c>, never <c>26.1.0.0</c>):
    /// the informational version, MSBuild's verbatim copy of <c>&lt;Version&gt;</c>, without any "+&lt;commit&gt;"
    /// build metadata. The assembly version is padded to four parts, so it is only the fallback, its padding dropped.
    /// </summary>
    internal static class CentrixVersion
    {
        public static string Current { get; } = Resolve();

        private static string Resolve()
        {
            var assembly = typeof(CentrixVersion).Assembly;
            string informational = assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion;
            if (!string.IsNullOrWhiteSpace(informational))
            {
                int plus = informational.IndexOf('+');
                string trimmed = (plus >= 0 ? informational.Substring(0, plus) : informational).Trim();
                if (trimmed.Length > 0)
                    return trimmed;
            }
            var version = assembly.GetName().Version;
            return version == null ? "unknown" : version.Major + "." + version.Minor + "." + System.Math.Max(version.Build, 0);
        }
    }
}
