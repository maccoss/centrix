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

using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using Pwiz.Analysis;
using Pwiz.Data.Common.Cv;
using Pwiz.Data.Common.Params;
using Pwiz.Data.MsData.Spectra;
using Centrix.Demux;

namespace Centrix.Cli
{
    /// <summary>
    /// A spectrum list that demultiplexes a stepped staggered acquisition (Orbitrap, Astral) on
    /// <see cref="StaggeredDemuxPipeline"/>: each MS2 spectrum becomes one spectrum per narrow bin
    /// its window covers, under that bin's window, as msconvert's demultiplex filter writes them;
    /// MS1 spectra pass through.
    /// </summary>
    internal sealed class StaggeredDemuxSpectrumList : SpectrumListWrapper
    {
        private readonly StaggeredDemuxPipeline _pipeline;
        private readonly bool _centroided; // the demultiplexed spectra are centroids, whatever the input was
        private long _sourceTicks, _peaksTicks; // with the solve profile: the acquired spectra read, and the peaks got
        // The acquired spectrum the last demultiplexed output was built on, as read (metadata only), and its index.
        private Spectrum _acquired;
        private int _acquiredIndex = -1;
        private CVID? _intensityUnits;   // the acquired spectra's intensity unit, once read

        public StaggeredDemuxSpectrumList(ISpectrumList inner, ScanningDemuxOptions options, TextWriter log)
            : base(inner)
        {
            _pipeline = new StaggeredDemuxPipeline(new PwizDemuxSource(inner), options, log);
            // The joint solve writes centroids from the profile: its fitted peaks, or Centrix's of each bin's share.
            // Its shares written as profile (--joint-profile) stay profile, and the other solves' output follows
            // their input.
            _centroided = options.Joint && (!options.JointApportion || options.JointCentroid);
        }

        public long Channels
        {
            get { return _pipeline.Channels; }
        }

        public long ChannelsSolved
        {
            get { return _pipeline.ChannelsSolved; }
        }

        public double IonsIn
        {
            get { return _pipeline.IonsIn; }
        }

        public double IonsPassedThrough
        {
            get { return _pipeline.IonsPassedThrough; }
        }

        public double IonsUnexplained
        {
            get { return _pipeline.IonsUnexplained; }
        }

        public override int Count
        {
            get { return _pipeline.OutputCount; }
        }

        public override SpectrumIdentity SpectrumIdentity(int index)
        {
            var identity = Inner.SpectrumIdentity(_pipeline.BaseIndex(index));
            return new SpectrumIdentity
            {
                Index = index,
                Id = OutputId(identity.Id, index),
                SpotId = identity.SpotId,
                SourceFilePosition = identity.SourceFilePosition,
            };
        }

        /// <summary>With the solve profile: seconds spent reading each output spectrum's acquired spectrum from the source.</summary>
        public double SourceSeconds
        {
            get { return _sourceTicks / (double)Stopwatch.Frequency; }
        }

        /// <summary>
        /// With the solve profile: seconds spent getting each output spectrum's peaks, the batches' solves waited for and
        /// laid out included.
        /// </summary>
        public double PeaksSeconds
        {
            get { return _peaksTicks / (double)Stopwatch.Frequency; }
        }

        /// <summary>
        /// An output spectrum. The demultiplexed outputs of one acquired spectrum come one after another, and when they are
        /// centroids its metadata is read once for them all (reading it costs more than the rest of writing a spectrum):
        /// each is a copy of it with its own parameters, precursors and peak arrays (<see cref="Detached"/>), so a caller
        /// can hold several at once, as Centrix's read-ahead does.
        /// Profile outputs are read each on its own: a peak picker after this list marks the one it centroids, which
        /// in a shared copy would mark its siblings centroided before they were picked.
        /// </summary>
        public override Spectrum GetSpectrum(int index, bool getBinaryData = false)
        {
            bool passThrough = _pipeline.IsPassThrough(index);
            int baseIndex = _pipeline.BaseIndex(index);
            long t0 = JointDemuxProfile.Enabled ? Stopwatch.GetTimestamp() : 0;
            Spectrum spectrum;
            if (passThrough || !_centroided)
            {
                // A demultiplexed spectrum's peaks are replaced, so only a pass-through reads the acquired peaks.
                spectrum = Inner.GetSpectrum(baseIndex, getBinaryData && passThrough);
            }
            else
            {
                if (baseIndex != _acquiredIndex)
                {
                    _acquired = Inner.GetSpectrum(baseIndex);
                    _acquiredIndex = baseIndex;
                }
                spectrum = Detached(_acquired);
            }
            if (JointDemuxProfile.Enabled)
                _sourceTicks += Stopwatch.GetTimestamp() - t0;
            spectrum.Index = index;
            if (passThrough)
                return spectrum;
            spectrum.Id = OutputId(spectrum.Id, index);
            var demuxBin = _pipeline.BinOf(index);
            var window = spectrum.Precursors[0].IsolationWindow;
            window.Set(CVID.MS_isolation_window_target_m_z, demuxBin.Center, CVID.MS_m_z);
            window.Set(CVID.MS_isolation_window_lower_offset, demuxBin.Width / 2, CVID.MS_m_z);
            window.Set(CVID.MS_isolation_window_upper_offset, demuxBin.Width / 2, CVID.MS_m_z);
            // The selected ion follows the window to the bin's center, and its intensity, which was the whole window's,
            // is zeroed, as pwiz's demultiplex filter writes them.
            foreach (var ion in spectrum.Precursors[0].SelectedIons)
            {
                ion.Set(CVID.MS_selected_ion_m_z, demuxBin.Center, CVID.MS_m_z);
                if (ion.HasCVParam(CVID.MS_peak_intensity))
                    ion.Set(CVID.MS_peak_intensity, 0.0, ion.CvParam(CVID.MS_peak_intensity).Units);
            }
            if (_centroided)
                SpectrumRouting.MarkCentroid(spectrum);
            if (!getBinaryData)
            {
                // The acquired spectrum's TIC and base peak are not this bin's, and without its peaks there is nothing to
                // give in their place.
                RemovePeakSummary(spectrum);
                return spectrum;
            }
            long t1 = JointDemuxProfile.Enabled ? Stopwatch.GetTimestamp() : 0;
            var (mz, ions) = _pipeline.Peaks(index);
            if (JointDemuxProfile.Enabled)
                _peaksTicks += Stopwatch.GetTimestamp() - t1;
            var counts = new double[ions.Length];
            double perIon = _pipeline.IntensityPerIon(index);
            for (int i = 0; i < ions.Length; i++)
                counts[i] = ions[i] * perIon;
            var units = IntensityUnits(baseIndex);
            spectrum.SetMZIntensityArrays(mz, counts, units);
            SetPeakSummary(spectrum, mz, counts, units);
            return spectrum;
        }

        /// <summary>
        /// The unit of the acquired spectra's intensities, which the outputs keep (they are converted back to the
        /// acquired spectrum's scale): read once, from the first demultiplexed spectrum's acquired peaks, as the metadata
        /// read carries no arrays. Detector counts when the file names none.
        /// </summary>
        private CVID IntensityUnits(int baseIndex)
        {
            if (_intensityUnits == null)
            {
                var array = Inner.GetSpectrum(baseIndex, true).GetIntensityArray();
                var units = array?.CvParam(CVID.MS_intensity_array).Units ?? CVID.CVID_Unknown;
                _intensityUnits = units != CVID.CVID_Unknown ? units : CVID.MS_number_of_detector_counts;
            }
            return _intensityUnits.Value;
        }

        /// <summary>Terms that summarize a spectrum's peaks, which an output gets from its own peaks or not at all.</summary>
        private static readonly CVID[] PEAK_SUMMARY =
        {
            CVID.MS_total_ion_current, CVID.MS_base_peak_m_z, CVID.MS_base_peak_intensity,
            CVID.MS_lowest_observed_m_z, CVID.MS_highest_observed_m_z,
        };

        private static void RemovePeakSummary(Spectrum spectrum)
        {
            foreach (var cvid in PEAK_SUMMARY)
            {
                SpectrumRouting.DetachGroupsHolding(spectrum.Params, cvid);
                spectrum.Params.CVParams.RemoveAll(p => p.Cvid == cvid);
            }
        }

        /// <summary>An output's TIC, base peak and observed m/z range, from its own peaks; with no peaks, a TIC of 0.</summary>
        private static void SetPeakSummary(Spectrum spectrum, double[] mz, double[] intensity, CVID units)
        {
            RemovePeakSummary(spectrum);
            var terms = spectrum.Params;
            double total = 0;
            int basePeak = -1;
            for (int i = 0; i < intensity.Length; i++)
            {
                total += intensity[i];
                if (basePeak < 0 || intensity[i] > intensity[basePeak])
                    basePeak = i;
            }
            terms.Set(CVID.MS_total_ion_current, total, units);
            if (basePeak < 0)
                return;
            terms.Set(CVID.MS_base_peak_m_z, mz[basePeak], CVID.MS_m_z);
            terms.Set(CVID.MS_base_peak_intensity, intensity[basePeak], units);
            terms.Set(CVID.MS_lowest_observed_m_z, mz.Min(), CVID.MS_m_z);
            terms.Set(CVID.MS_highest_observed_m_z, mz.Max(), CVID.MS_m_z);
        }

        /// <summary>
        /// A copy of an acquired spectrum's metadata that shares nothing an output rewrites: its own terms, precursors
        /// (with their isolation windows and selected ions) and no peak arrays. What no output changes (the scan list,
        /// the activation) is shared.
        /// </summary>
        private static Spectrum Detached(Spectrum acquired)
        {
            var copy = new Spectrum
            {
                Index = acquired.Index,
                Id = acquired.Id,
                SpotId = acquired.SpotId,
                SourceFilePosition = acquired.SourceFilePosition,
                DefaultArrayLength = acquired.DefaultArrayLength,
                DataProcessing = acquired.DataProcessing,
                SourceFile = acquired.SourceFile,
                ScanList = acquired.ScanList,
            };
            CopyParams(acquired.Params, copy.Params);
            foreach (var precursor in acquired.Precursors)
            {
                var detached = new Precursor
                {
                    SourceFile = precursor.SourceFile,
                    ExternalSpectrumId = precursor.ExternalSpectrumId,
                    SpectrumId = precursor.SpectrumId,
                    Activation = precursor.Activation,
                };
                CopyParams(precursor, detached);
                CopyParams(precursor.IsolationWindow, detached.IsolationWindow);
                foreach (var ion in precursor.SelectedIons)
                {
                    var own = new SelectedIon();
                    CopyParams(ion, own);
                    detached.SelectedIons.Add(own);
                }
                copy.Precursors.Add(detached);
            }
            copy.Products.AddRange(acquired.Products);
            return copy;
        }

        /// <summary>
        /// A container's terms copied, each a new object (<see cref="ParamContainer.Set(CVID, string, CVID)"/> changes a
        /// term in place); its parameter groups shared, as the document defines them once.
        /// </summary>
        private static void CopyParams(ParamContainer from, ParamContainer to)
        {
            to.CVParams.AddRange(from.CVParams.Select(p => new CVParam(p.Cvid, p.Value, p.Units)));
            to.UserParams.AddRange(from.UserParams.Select(p => new UserParam(p.Name, p.Value, p.Type, p.Units)));
            to.ParamGroups.AddRange(from.ParamGroups);
        }

        private string OutputId(string id, int index)
        {
            if (_pipeline.IsPassThrough(index))
                return id;
            return id + @" demux=" + _pipeline.PartOf(index).ToString(CultureInfo.InvariantCulture);
        }
    }
}
