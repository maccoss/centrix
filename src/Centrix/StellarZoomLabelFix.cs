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

using System.Linq;
using System.Text.RegularExpressions;
using System.Threading;
using Pwiz.Analysis;
using Pwiz.Data.Common.Cv;
using Pwiz.Data.Common.Params;
using Pwiz.Data.MsData.Processing;
using Pwiz.Data.MsData.Spectra;

namespace Centrix.Cli
{
    /// <summary>
    /// Takes the zoom-scan label off a Thermo Stellar's spectra. The Stellar cannot perform a zoom scan, but it marks
    /// every spectrum it scans at 200 kTh/s, its fastest rate, with "Z" in its filter string: the .raw format has room
    /// for only so many scan options, and Thermo stored that rate in the zoom flag. Its DIA MS2
    /// ("ITMS + p NSI Z ms2 411.4369@hcd20.00 [200.0000-1200.0000]") and its Adaptive RT alignment spectra carry it alike,
    /// and ProteoWizard reports each as a zoom scan (MS:1000497). On a Stellar run both go: the term, and the filter's
    /// "Z" ("ITMS + p NSI ms2 ..."), each from a referenceable parameter group too (detached, its other terms kept). The
    /// scan rate is the point spacing's to say, not the filter's; the processing record names the correction.
    /// </summary>
    internal sealed class StellarZoomLabelFix : SpectrumListWrapper
    {
        private static readonly Regex ZOOM_IN_FILTER = new Regex(@" Z (?=ms\d*\b)");

        private readonly DataProcessing _dataProcessing;
        private readonly int[] _relabelledAt;   // 1 for each spectrum relabelled, so one read many times counts once
        private long _relabelled;

        /// <param name="inner">A Stellar run's spectra.</param>
        public StellarZoomLabelFix(ISpectrumList inner)
            : base(inner)
        {
            _relabelledAt = new int[inner.Count];
            var innerProcessing = inner.DataProcessing;
            _dataProcessing = new DataProcessing(innerProcessing?.Id ?? "pwiz_Reader_conversion");
            if (innerProcessing != null)
                _dataProcessing.ProcessingMethods.AddRange(innerProcessing.ProcessingMethods);
            var method = new ProcessingMethod
            {
                Order = _dataProcessing.ProcessingMethods.Count,
                Software = _dataProcessing.ProcessingMethods.FirstOrDefault()?.Software,
            };
            method.Set(CVID.MS_data_transformation);
            method.UserParams.Add(new UserParam("Centrix: zoom scan label removed from Stellar spectra",
                "the Stellar performs no zoom scans; its .raw files store its 200 kTh/s scan rate as the zoom flag (Z)"));
            _dataProcessing.ProcessingMethods.Add(method);
        }

        public override DataProcessing DataProcessing => _dataProcessing;

        /// <summary>Spectra relabelled so far, each counted once.</summary>
        public long Relabelled => Interlocked.Read(ref _relabelled);

        public override Spectrum GetSpectrum(int index, bool getBinaryData = false)
        {
            var spectrum = Inner.GetSpectrum(index, getBinaryData);
            Tally(index, Relabel(spectrum));
            return spectrum;
        }

        public override Spectrum GetSpectrum(int index, DetailLevel detailLevel)
        {
            var spectrum = Inner.GetSpectrum(index, detailLevel);
            Tally(index, Relabel(spectrum));
            return spectrum;
        }

        private void Tally(int index, bool relabelled)
        {
            if (relabelled && index >= 0 && index < _relabelledAt.Length && Interlocked.Exchange(ref _relabelledAt[index], 1) == 0)
                Interlocked.Increment(ref _relabelled);
        }

        /// <summary>
        /// Removes a spectrum's zoom-scan term and its filter strings' "Z", from referenceable parameter groups too
        /// (detached, their other terms kept: <see cref="SpectrumRouting.DetachGroupsHolding"/>); true when either was there.
        /// </summary>
        internal static bool Relabel(Spectrum spectrum)
        {
            var terms = spectrum.Params;
            bool removed = false;
            if (terms.HasCVParam(CVID.MS_zoom_scan))
            {
                SpectrumRouting.DetachGroupsHolding(terms, CVID.MS_zoom_scan);
                removed = terms.CVParams.RemoveAll(p => p.Cvid == CVID.MS_zoom_scan) > 0;
            }
            foreach (var scan in spectrum.ScanList.Scans)
            {
                // The filter string may come from a referenceable group the scan shares with others.
                var filter = scan.CvParam(CVID.MS_filter_string);
                if (filter.Cvid != CVID.MS_filter_string || !ZOOM_IN_FILTER.IsMatch(filter.Value))
                    continue;
                SpectrumRouting.DetachGroupsHolding(scan, CVID.MS_filter_string);
                int own = scan.CVParams.FindIndex(p => p.Cvid == CVID.MS_filter_string);
                // A new term rather than the old one changed: the reader may share it with the spectrum it came from.
                scan.CVParams[own] = new CVParam(CVID.MS_filter_string, ZOOM_IN_FILTER.Replace(scan.CVParams[own].Value, " "));
                removed = true;
            }
            return removed;
        }
    }
}
