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

namespace Centrix.Cli
{
    /// <summary>What the command line asks for.</summary>
    public enum CentrixCommand
    {
        /// <summary>Centroid (and demultiplex) each input into an mzML.</summary>
        run,

        /// <summary>Report each input's peak sigmas and demultiplexing scheme, writing nothing.</summary>
        calibrate,
    }

    /// <summary>Whether staggered windows are demultiplexed (<c>--demux</c>).</summary>
    public enum DemuxMode
    {
        /// <summary>When the MS2 windows overlap and the MS2 is a linear ion trap's profile. The default.</summary>
        auto,

        /// <summary>Never: every spectrum is only centroided.</summary>
        off,

        /// <summary>Always: the run must be a staggered acquisition of linear ion trap profile spectra.</summary>
        staggered,
    }

    /// <summary>What happens to a Thermo Stellar's Adaptive RT spectra (<c>--adaptive-rt</c>).</summary>
    public enum AdaptiveRtMode
    {
        /// <summary>Left out of the output. The default: they are the instrument's retention time alignment scans, not data.</summary>
        exclude,

        /// <summary>Written, centroided like the rest, and passed through the demultiplexer untouched.</summary>
        include,
    }

    /// <summary>
    /// The command line, parsed. The lasso's settings keep Rust Centrix's option names, so its scripts still run; the
    /// options Rust Centrix had for its own calibration (<c>--scan-rate</c>, <c>--n-calibration-spectra</c>,
    /// <c>--config</c>) are gone with it.
    /// </summary>
    public sealed class CentrixOptions
    {
        /// <summary>Kebab-case options that set a <see cref="CentrixParams"/> property, with their Rust Centrix names.</summary>
        private static readonly Dictionary<string, string> PARAMETER_OPTIONS = new Dictionary<string, string>
        {
            { "--lambda-factor", nameof(CentrixParams.LambdaFactor) },
            { "--max-lasso-iter", nameof(CentrixParams.MaxIterations) },
            { "--lasso-tol", nameof(CentrixParams.Tolerance) },
            { "--lambda-change-threshold", nameof(CentrixParams.LambdaChangeThreshold) },
            { "--signal-threshold-sigma", nameof(CentrixParams.SignalThresholdSigma) },
            { "--merge-gap-points", nameof(CentrixParams.MergeGapPoints) },
            { "--extension-points", nameof(CentrixParams.ExtensionPoints) },
            { "--min-region-width", nameof(CentrixParams.MinRegionWidth) },
            { "--noise-window-da", nameof(CentrixParams.NoiseWindowTh) },
            { "--noise-step-da", nameof(CentrixParams.NoiseStepTh) },
            { "--min-centroid-separation", nameof(CentrixParams.MinCentroidSeparationTh) },
            { "--grid-spacing", nameof(CentrixParams.GridSpacingTh) },
            // Rust Centrix's --grid-offset was a fraction of the spacing; this one is in Th, so it has its own name.
            { "--grid-offset-th", nameof(CentrixParams.GridOffsetTh) },
            { "--min-noise-sigma", nameof(CentrixParams.MinNoiseSigma) },
        };

        public const string USAGE = @"Usage: centrix [run|calibrate] -i <input>... [-o <output>] [options]

Centroids linear ion trap profile spectra (Thermo Stellar, the ion trap of a Fusion or an Eclipse) by a non-negative
lasso on Gaussian peaks, which separates signals less than 1 m/z apart; demultiplexes staggered (overlapping-window)
Stellar DIA in the same solve. Reads Thermo .raw and mzML (and ProteoWizard's other open formats) and writes mzML.

Input and output:
  -i, --input <file>...        Input files (also accepted without -i). Several may follow one -i.
  -o, --output <path>          An output directory, or for one input an output .mzML. Default: beside each
                               input, as <stem>.centrix.mzML.
      --overwrite              Replace output files that exist (otherwise they are skipped).
      --filter <spec>          A ProteoWizard spectrum filter, as msconvert's --filter takes it (repeatable),
                               applied after reading and before centroiding.
      --no-zlib                Write the binary arrays uncompressed.

Centroiding:
      --centroid <mode>        auto (default): Centrix for ion trap profile spectra, the vendor's centroids for
                               other analyzers' (a Fusion's Orbitrap MS1). centrix: Centrix for every profile
                               spectrum. vendor: the vendor's centroids only, the baseline to compare with.
      --sigma <Th>             The peak sigma of every MS level. Default: from the run. On a linear ion trap,
                               MS2 at three profile point spacings (0.6 Th at the Stellar's 200 kTh/s), the
                               sigma that detects the most peptides, and MS1 at 1.1 (0.073 Th at its
                               67 kTh/s), which measures precursor isotopes best of the sigmas tried;
                               measured from the run's isolated peaks elsewhere. About two spacings
                               (--sigma-ms2 0.38 on a Stellar) measures fragment m/z best, resolving M+1
                               isotopes.
      --sigma-ms1, --sigma-ms2 <Th>
                               One level's peak sigma, over --sigma.
      --solver <solver>        descent (default): Rust Centrix's coordinate descent. active-set: the exact optimum
                               of the same lasso, several times faster.
      --lambda-factor <x>      The lasso penalty, in noise sigmas (3).
      --signal-threshold-sigma, --merge-gap-points, --extension-points, --min-region-width,
      --noise-window-da, --noise-step-da, --lambda-change-threshold, --max-lasso-iter, --lasso-tol,
      --min-centroid-separation, --grid-spacing, --grid-offset-th, --min-noise-sigma <value>
                               The rest of the lasso's settings, as Rust Centrix names them.

Demultiplexing:
      --demux <mode>           auto (default): demultiplex when the MS2 windows overlap (50% overlap, SuperOverlap)
                               and are a linear ion trap's profile. off: centroid only. staggered: always.
      --no-ions                Weigh the demultiplexing by intensity alone rather than by ions counted from each
                               spectrum's fill time.

Adaptive RT:
      --adaptive-rt <mode>     exclude (default): leave a Stellar's Adaptive RT alignment spectra out of the
                               output. include: write them, centroided, outside any demultiplexing.
      --adaptive-rt-width <Th> A Stellar MS2 isolated wider than this is an Adaptive RT spectrum (10: they are
                               20 or 50 m/z wide, and Stellar DIA windows never exceed 10).
      --keep-zoom-label        Keep the zoom-scan label a Stellar's 200 kTh/s spectra carry (the term and the ""Z""
                               in the filter string). By default both are removed: the Stellar performs no zoom
                               scans, and stores that scan rate in the zoom flag.

General:
  -j, --threads <n>            Threads (0, the default: every core). The output does not depend on it.
  -q, --quiet                  Errors only.
  -v, --verbose                More detail.
      --version                The version.
  -h, --help                   This help.
";

        public CentrixCommand Command { get; set; } = CentrixCommand.run;
        public List<string> Inputs { get; } = new List<string>();
        public string Output { get; set; }
        public bool Overwrite { get; set; }
        public List<string> Filters { get; } = new List<string>();
        public bool Zlib { get; set; } = true;
        public CentroidMode Centroid { get; set; } = CentroidMode.auto;
        public double SigmaMs1 { get; set; }
        public double SigmaMs2 { get; set; }
        public CentrixParams Parameters { get; } = new CentrixParams();
        public DemuxMode Demux { get; set; } = DemuxMode.auto;
        public bool Ions { get; set; } = true;
        public AdaptiveRtMode AdaptiveRt { get; set; } = AdaptiveRtMode.exclude;

        /// <summary>Keep the zoom-scan label on a Stellar's 200 kTh/s spectra, term and "Z" (<see cref="StellarZoomLabelFix"/>).</summary>
        public bool KeepZoomLabel { get; set; }

        /// <summary>
        /// MS2 isolated wider than this (Th) are Adaptive RT spectra on a Stellar, and on any run are passed through,
        /// not demultiplexed. 0: no spectrum is.
        /// </summary>
        public double AdaptiveRtWidthTh { get; set; } = 10;
        public int Threads { get; set; }
        public bool Quiet { get; set; }
        public bool Verbose { get; set; }
        public bool ShowHelp { get; set; }
        public bool ShowVersion { get; set; }

        /// <summary>Threads to use: the option, or every core.</summary>
        public int EffectiveThreads => Threads > 0 ? Threads : Environment.ProcessorCount;

        /// <summary>Parses a command line; throws <see cref="ArgumentException"/> with a message for the user.</summary>
        public static CentrixOptions Parse(IReadOnlyList<string> args)
        {
            var options = new CentrixOptions();
            double sigmaBoth = 0;
            int i = 0;
            if (args.Count > 0 && (args[0] == "run" || args[0] == "calibrate"))
            {
                options.Command = Enum.Parse<CentrixCommand>(args[0]);
                i = 1;
            }
            for (; i < args.Count; i++)
            {
                string arg = args[i];
                string value = null;
                // --name=value as well as --name value.
                int equals = arg.StartsWith("--", StringComparison.Ordinal) ? arg.IndexOf('=') : -1;
                if (equals > 0)
                {
                    value = arg.Substring(equals + 1);
                    arg = arg.Substring(0, equals);
                }
                string Value()
                {
                    if (value != null)
                        return value;
                    if (i + 1 >= args.Count)
                        throw new ArgumentException($"{arg} needs a value.");
                    return args[++i];
                }

                if (!arg.StartsWith("-", StringComparison.Ordinal) || arg == "-")
                {
                    options.Inputs.Add(arg);
                    continue;
                }
                if (PARAMETER_OPTIONS.TryGetValue(arg, out string property))
                {
                    SetParameter(options.Parameters, property, Value(), arg);
                    continue;
                }
                switch (arg)
                {
                    case "-i":
                    case "--input":
                        // Every argument up to the next option is an input.
                        options.Inputs.Add(Value());
                        while (i + 1 < args.Count && !args[i + 1].StartsWith("-", StringComparison.Ordinal))
                            options.Inputs.Add(args[++i]);
                        break;
                    case "-o":
                    case "--output":
                        options.Output = Value();
                        break;
                    case "--overwrite":
                        options.Overwrite = true;
                        break;
                    case "--filter":
                        options.Filters.Add(Value());
                        break;
                    case "--no-zlib":
                        options.Zlib = false;
                        break;
                    case "--centroid":
                        options.Centroid = ParseEnum<CentroidMode>(Value(), arg);
                        break;
                    case "--sigma":
                        sigmaBoth = ParsePositive(Value(), arg);
                        break;
                    case "--sigma-ms2":
                        options.SigmaMs2 = ParsePositive(Value(), arg);
                        break;
                    case "--sigma-ms1":
                        options.SigmaMs1 = ParsePositive(Value(), arg);
                        break;
                    case "--solver":
                        string solver = Value();
                        options.Parameters.Solver = solver switch
                        {
                            "descent" or "coordinate-descent" or "coordinate_descent" => CentrixSolver.coordinate_descent,
                            "active-set" or "active_set" => CentrixSolver.active_set,
                            _ => throw new ArgumentException($"--solver is descent or active-set, not {solver}."),
                        };
                        break;
                    case "--demux":
                        options.Demux = ParseEnum<DemuxMode>(Value(), arg);
                        break;
                    case "--no-ions":
                        options.Ions = false;
                        break;
                    case "--adaptive-rt":
                        options.AdaptiveRt = ParseEnum<AdaptiveRtMode>(Value(), arg);
                        break;
                    case "--keep-zoom-label":
                        options.KeepZoomLabel = true;
                        break;
                    case "--adaptive-rt-width":
                        options.AdaptiveRtWidthTh = ParseDouble(Value(), arg);
                        break;
                    case "-j":
                    case "--threads":
                        options.Threads = (int)ParseDouble(Value(), arg);
                        break;
                    case "-q":
                    case "--quiet":
                        options.Quiet = true;
                        break;
                    case "-v":
                    case "--verbose":
                        options.Verbose = true;
                        break;
                    case "--version":
                        options.ShowVersion = true;
                        break;
                    case "-h":
                    case "--help":
                        options.ShowHelp = true;
                        break;
                    default:
                        throw new ArgumentException($"Unknown option {arg}.");
                }
            }
            // A lasso setting the centroider cannot run with (a noise step of 0 would never end), reported by its option.
            try
            {
                options.Parameters.Validate();
            }
            catch (ArgumentException e)
            {
                string message = e.Message;
                foreach (var (option, property) in PARAMETER_OPTIONS)
                    message = message.Replace(property + " ", option + " ");
                throw new ArgumentException(message, e);
            }
            // --sigma is every level's, as in Rust Centrix; a level's own option wins, wherever it is given.
            if (!(options.SigmaMs1 > 0))
                options.SigmaMs1 = sigmaBoth;
            if (!(options.SigmaMs2 > 0))
                options.SigmaMs2 = sigmaBoth;
            if (!options.ShowHelp && !options.ShowVersion && options.Inputs.Count == 0)
                throw new ArgumentException("No input files.");
            return options;
        }

        /// <summary>Whether -o names an .mzML file rather than a directory.</summary>
        public bool NamesOutputFile => Output != null && Output.EndsWith(".mzML", StringComparison.OrdinalIgnoreCase);

        /// <summary>
        /// Replaces each input whose file name holds a wildcard (* or ?) with the files and directories it matches, in
        /// name order (a pattern may match directories); returns the patterns that matched nothing. Other inputs are kept as
        /// given. Only the last part of a path may hold one.
        /// </summary>
        public List<string> ExpandInputs()
        {
            var unmatched = new List<string>();
            var expanded = new List<string>();
            foreach (string input in Inputs)
            {
                string name = Path.GetFileName(input);
                if (name.IndexOfAny(new[] { '*', '?' }) < 0)
                {
                    expanded.Add(input);
                    continue;
                }
                string directory = Path.GetDirectoryName(input);
                string searched = string.IsNullOrEmpty(directory) ? "." : directory;
                var matches = Directory.Exists(searched)
                    ? Directory.EnumerateFileSystemEntries(searched, name)
                        .Select(m => string.IsNullOrEmpty(directory) ? Path.GetFileName(m) : m)
                        .OrderBy(m => m, StringComparer.Ordinal).ToList()
                    : new List<string>();
                if (matches.Count == 0)
                    unmatched.Add(input);
                expanded.AddRange(matches);
            }
            Inputs.Clear();
            Inputs.AddRange(expanded);
            return unmatched;
        }

        /// <summary>
        /// The output path for an input: the -o file for one input given a path ending in .mzML, else
        /// <c>&lt;stem&gt;.centrix.mzML</c> in the -o directory or beside the input.
        /// </summary>
        public string OutputPathFor(string input)
        {
            if (Inputs.Count == 1 && NamesOutputFile)
                return Output;
            string directory = Output ?? Path.GetDirectoryName(Path.GetFullPath(input));
            return Path.Combine(directory, Stem(input) + ".centrix.mzML");
        }

        /// <summary>A file's name without its extensions (run.mzML.gz and a run.d directory both give run).</summary>
        private static string Stem(string path)
        {
            string name = Path.GetFileName(Path.TrimEndingDirectorySeparator(path));
            foreach (string compression in new[] { ".gz", ".bz2", ".zst" })
            {
                if (name.EndsWith(compression, StringComparison.OrdinalIgnoreCase))
                    name = name.Substring(0, name.Length - compression.Length);
            }
            return Path.GetFileNameWithoutExtension(name);
        }

        private static void SetParameter(CentrixParams parameters, string property, string text, string option)
        {
            var info = typeof(CentrixParams).GetProperty(property);
            double number = ParseDouble(text, option);
            if (info.PropertyType == typeof(int))
            {
                if (number != Math.Floor(number))
                    throw new ArgumentException($"{option} takes a whole number, not {text}.");
                info.SetValue(parameters, (int)number);
            }
            else
            {
                info.SetValue(parameters, number);
            }
        }

        private static T ParseEnum<T>(string text, string option) where T : struct, Enum
        {
            if (Enum.TryParse(text.Replace('-', '_'), true, out T result) && Enum.IsDefined(result))
                return result;
            throw new ArgumentException($"{option} is one of {string.Join(", ", Enum.GetNames<T>())}, not {text}.");
        }

        private static double ParseDouble(string text, string option)
        {
            if (double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out double value) && !double.IsNaN(value))
                return value;
            throw new ArgumentException($"{option} takes a number, not {text}.");
        }

        private static double ParsePositive(string text, string option)
        {
            double value = ParseDouble(text, option);
            if (!(value > 0) || double.IsInfinity(value))
                throw new ArgumentException($"{option} must be a positive number of Th, not {text}.");
            return value;
        }

        /// <summary>Every option this parser knows, for the tests.</summary>
        internal static IEnumerable<string> ParameterOptionNames => PARAMETER_OPTIONS.Keys.OrderBy(k => k, StringComparer.Ordinal);
    }
}
