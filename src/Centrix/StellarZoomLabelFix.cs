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
using System.Threading;
using Pwiz.Analysis;
using Pwiz.Data.Common.Cv;
using Pwiz.Data.Common.Params;
using Pwiz.Data.MsData.Processing;
using Pwiz.Data.MsData.Spectra;

namespace Centrix.Cli
{
    /// <summary>
    /// Takes ProteoWizard's zoom-scan term off a Thermo Stellar's spectra. The Stellar marks every spectrum it scans at
    /// 200 kTh/s with "Z" in its filter string (Thermo's scan-rate code for that rate, where the other rates have n, r
    /// and t): its DIA MS2 ("ITMS + p NSI Z ms2 411.4369@hcd20.00 [200.0000-1200.0000]") and its Adaptive RT alignment
    /// spectra alike. ProteoWizard reads a Thermo filter's "Z" as a zoom scan and reports one (MS:1000497), but the
    /// Stellar has no zoom scans: 200 kTh/s is its fastest scan, not a slow narrow one. On a Stellar run the term is
    /// removed, from a referenceable parameter group too (detached, its other terms kept), wherever it appears; the
    /// filter string keeps its "Z" as Thermo recorded it, since it says the scan rate. The processing record says so.
    /// </summary>
    internal sealed class StellarZoomLabelFix : SpectrumListWrapper
    {
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
            method.UserParams.Add(new UserParam("Centrix: zoom scan term removed from Stellar spectra",
                "the filter strings' Z is the Stellar's 200 kTh/s scan-rate code, not a zoom scan; the filter strings are as recorded"));
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
        /// Removes a spectrum's zoom-scan term, from a referenceable parameter group too (detached, its other terms kept:
        /// <see cref="SpectrumRouting.DetachGroupsHolding"/>); true when there was one. The filter string is left alone.
        /// </summary>
        internal static bool Relabel(Spectrum spectrum)
        {
            var terms = spectrum.Params;
            if (!terms.HasCVParam(CVID.MS_zoom_scan))
                return false;
            SpectrumRouting.DetachGroupsHolding(terms, CVID.MS_zoom_scan);
            return terms.CVParams.RemoveAll(p => p.Cvid == CVID.MS_zoom_scan) > 0;
        }
    }
}
