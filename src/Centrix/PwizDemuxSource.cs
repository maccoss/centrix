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

using System.Collections.Generic;
using System.Linq;
using Pwiz.Data.Common.Cv;
using Pwiz.Data.Common.Params;
using Pwiz.Data.MsData.Instruments;
using Pwiz.Data.MsData.Spectra;
using Centrix.Demux;
using IsolationWindow = Centrix.IsolationWindow;

namespace Centrix.Cli
{
    /// <summary>
    /// A pwiz spectrum list as a demultiplexing source that keeps nothing it reads: for detecting the scheme, and
    /// for the descriptions a spectrum list built on the pipeline passes through. Requests are served one at a time,
    /// whatever the pipeline's read threads: a Thermo .raw shares one raw-file accessor and an unsynchronized precursor
    /// cache among its readers, and an mzML is read from one stream.
    /// </summary>
    internal sealed class PwizDemuxSource : IDemuxSource
    {
        private readonly ISpectrumList _spectra;
        private readonly object _readLock = new object();
        private readonly Dictionary<string, double> _ionResponse = new Dictionary<string, double>();   // by configuration id

        public PwizDemuxSource(ISpectrumList spectra)
        {
            _spectra = spectra;
        }

        public int Count
        {
            get
            {
                lock (_readLock)
                {
                    return _spectra.Count;
                }
            }
        }

        public string NativeId(int index)
        {
            lock (_readLock)
            {
                return _spectra.SpectrumIdentity(index).Id;
            }
        }

        public DemuxSpectrumInfo Describe(int index)
        {
            Spectrum spectrum;
            double ionResponse;
            lock (_readLock)
            {
                spectrum = _spectra.GetSpectrum(index);
                ionResponse = IonResponseOf(spectrum);
            }
            int level = spectrum.Params.CvParamValueOrDefault(CVID.MS_ms_level, 0);
            var scan = spectrum.ScanList.Scans.Count > 0 ? spectrum.ScanList.Scans[0] : null;
            // The demultiplexer wants minutes and milliseconds, whatever units the file gives them in.
            double time = scan != null ? InUnits(scan.CvParam(CVID.MS_scan_start_time), 0.0, MINUTE_UNITS) : 0.0;
            double injectionTime = scan != null
                ? InUnits(scan.CvParam(CVID.MS_ion_injection_time), double.NaN, MILLISECOND_UNITS)
                : double.NaN;
            if (spectrum.Precursors.Count == 0)
                return new DemuxSpectrumInfo(level, false, default, time, injectionTime, ionResponse);
            var window = spectrum.Precursors[0].IsolationWindow;
            return new DemuxSpectrumInfo(level, true, new IsolationWindow(
                window.CvParamValueOrDefault(CVID.MS_isolation_window_target_m_z, 0.0),
                window.CvParamValueOrDefault(CVID.MS_isolation_window_lower_offset, 0.0),
                window.CvParamValueOrDefault(CVID.MS_isolation_window_upper_offset, 0.0)), time, injectionTime, ionResponse);
        }

        public void Read(int index, out IReadOnlyList<double> mz, out IReadOnlyList<double> intensity)
        {
            Spectrum spectrum;
            lock (_readLock)
            {
                spectrum = _spectra.GetSpectrum(index, true);
            }
            mz = spectrum.GetMZArray()?.Data;
            intensity = spectrum.GetIntensityArray()?.Data;
        }

        public void Release(int index)
        {
        }

        public void Forget(int index)
        {
        }

        /// <summary>
        /// The ion response (alpha) of the analyzer that acquired a spectrum, from its scan's instrument configuration's
        /// analyzer terms (<see cref="IonResponse"/>); NaN when the configuration names none known. Kept by
        /// configuration, under the read lock.
        /// </summary>
        private double IonResponseOf(Spectrum spectrum)
        {
            var configuration = spectrum.ScanList.Scans.Count > 0 ? spectrum.ScanList.Scans[0].InstrumentConfiguration : null;
            if (configuration == null)
                return double.NaN;
            if (!_ionResponse.TryGetValue(configuration.Id, out double alpha))
            {
                // The last analyzer is the one that measured the spectrum, as SpectrumRouting reads it (an ion trap
                // configuration may list an Orbitrap before it, whose response is ten times an ion trap's).
                var analyzer = configuration.ComponentList.LastOrDefault(c => c.Type == ComponentType.Analyzer);
                alpha = IonResponse.Of(IonResponse.KindOf(analyzer?.CvParamChild(CVID.MS_mass_analyzer_type).Name));
                _ionResponse[configuration.Id] = alpha;
            }
            return alpha;
        }

        /// <summary>Scale factors to minutes by unit; a time with no unit is taken to be in minutes, as pwiz writes it.</summary>
        private static readonly Dictionary<CVID, double> MINUTE_UNITS = new Dictionary<CVID, double>
        {
            { CVID.UO_minute, 1 },
            { CVID.UO_second, 1.0 / 60 },
            { CVID.UO_millisecond, 1.0 / 60000 },
            { CVID.CVID_Unknown, 1 },
        };

        /// <summary>Scale factors to milliseconds by unit; a fill time with no unit is taken to be in milliseconds.</summary>
        private static readonly Dictionary<CVID, double> MILLISECOND_UNITS = new Dictionary<CVID, double>
        {
            { CVID.UO_millisecond, 1 },
            { CVID.UO_second, 1000 },
            { CVID.UO_microsecond, 0.001 },
            { CVID.UO_minute, 60000 },
            { CVID.CVID_Unknown, 1 },
        };

        /// <summary>A term's value in the unit the factors convert to; the default when the term is absent or its unit unknown.</summary>
        internal static double InUnits(CVParam term, double absent, IReadOnlyDictionary<CVID, double> factors)
        {
            if (term == null || term.Cvid == CVID.CVID_Unknown || !factors.TryGetValue(term.Units, out double factor))
                return absent;
            return term.ValueAs<double>() * factor;
        }
    }
}
