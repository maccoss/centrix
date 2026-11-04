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
using Pwiz.Data.Common.Cv;
using Pwiz.Data.MsData;
using Pwiz.Data.MsData.Instruments;
using Pwiz.Data.MsData.Spectra;

namespace Centrix.Cli
{
    /// <summary>Who centroids the profile spectra (<c>--centroid</c>).</summary>
    public enum CentroidMode
    {
        /// <summary>
        /// Centrix for a linear ion trap's profile spectra, the vendor's centroids (pwiz's wavelet picker for a file
        /// with none) for any other analyzer's: on a Fusion, the ion trap's MS2 by Centrix, the Orbitrap's MS1 by the
        /// vendor. The default.
        /// </summary>
        auto,

        /// <summary>Centrix for every profile spectrum, whatever its analyzer, as Osprey's <c>--centroid centrix</c>.</summary>
        centrix,

        /// <summary>The vendor's centroids only (pwiz's wavelet picker for a file with none): the baseline to compare with.</summary>
        vendor,
    }

    /// <summary>What happens to one spectrum on its way to the output.</summary>
    internal enum SpectrumRoute
    {
        /// <summary>Written as read: already centroided, or not a spectrum of peaks.</summary>
        PassThrough,

        /// <summary>Centroided by Centrix.</summary>
        Centrix,

        /// <summary>Centroided by the fallback (the vendor's centroids, or pwiz's wavelet picker).</summary>
        Fallback,
    }

    /// <summary>
    /// Decides each spectrum's <see cref="SpectrumRoute"/>. Centrix fits Gaussians on a uniform m/z grid, which is what a
    /// linear ion trap's profile is (its firmware fixes the point spacing per scan rate); an Orbitrap's or a TOF's
    /// profile is not uniform in m/z, and their vendors' centroiders are good at their peaks.
    /// </summary>
    internal static class SpectrumRouting
    {
        public static SpectrumRoute Route(Spectrum spectrum, CentroidMode mode)
        {
            if (!IsProfile(spectrum))
                return SpectrumRoute.PassThrough;
            switch (mode)
            {
                case CentroidMode.centrix:
                    return SpectrumRoute.Centrix;
                case CentroidMode.vendor:
                    return SpectrumRoute.Fallback;
                default:
                    return IsLinearIonTrap(spectrum) ? SpectrumRoute.Centrix : SpectrumRoute.Fallback;
            }
        }

        /// <summary>Whether a spectrum is profile: anything not marked centroid, as pwiz's peak picker reads it.</summary>
        public static bool IsProfile(Spectrum spectrum)
        {
            return spectrum.Params.CvParamChild(CVID.MS_spectrum_representation).Cvid != CVID.MS_centroid_spectrum;
        }

        /// <summary>
        /// Whether an ion trap measured a spectrum: its scan's instrument configuration has an ion trap analyzer. A
        /// spectrum whose file names no analyzer for it (an mzML written without instrument metadata) is judged by its
        /// points instead: spaced as a linear ion trap scan rate spaces them.
        /// </summary>
        public static bool IsLinearIonTrap(Spectrum spectrum)
        {
            var configuration = spectrum.ScanList.Scans.Count > 0 ? spectrum.ScanList.Scans[0].InstrumentConfiguration : null;
            var analyzers = configuration?.ComponentList.Where(c => c.Type == ComponentType.Analyzer)
                .Select(c => c.CvParamChild(CVID.MS_mass_analyzer_type).Cvid)
                .Where(cvid => cvid != CVID.CVID_Unknown)
                .ToList();
            if (analyzers != null && analyzers.Count > 0)
            {
                // The last analyzer is the one that scanned (a Fusion's configuration lists the quadrupole first).
                return CvLookup.CvIsA(analyzers[analyzers.Count - 1], CVID.MS_ion_trap);
            }
            var mz = spectrum.GetMZArray()?.Data;
            return mz != null && mz.Count >= 3 && LinearIonTrapScanRate.FromSpacing(CentrixCentroider.DetectGridSpacing(mz)) != null;
        }

        /// <summary>Whether a run is a Thermo Stellar's: an instrument configuration names it as the model.</summary>
        public static bool IsStellar(MSData msd)
        {
            return msd.InstrumentConfigurations.Any(c => c.CvParamChild(CVID.MS_instrument_model).Cvid == CVID.MS_Stellar);
        }

        /// <summary>
        /// Whether a spectrum is wider than any DIA window: an MS2 or above whose isolation window spans more than
        /// <paramref name="widthTh"/>. On a Stellar these are its Adaptive RT alignment spectra (20 or 50 m/z wide;
        /// its DIA windows never exceed 10). Their filter strings carry "Z", but so does every DIA MS2 of the
        /// Stellar's, so the width, not the annotation, tells them apart. 0 or less: none is.
        /// </summary>
        public static bool IsWiderThan(Spectrum spectrum, double widthTh)
        {
            if (!(widthTh > 0) || spectrum.Params.CvParamValueOrDefault(CVID.MS_ms_level, 0) < 2 || spectrum.Precursors.Count == 0)
                return false;
            var window = spectrum.Precursors[0].IsolationWindow;
            double width = window.CvParamValueOrDefault(CVID.MS_isolation_window_lower_offset, 0.0) +
                window.CvParamValueOrDefault(CVID.MS_isolation_window_upper_offset, 0.0);
            return width > widthTh;
        }

        /// <summary>Marks a spectrum centroided, its profile term removed, from a shared parameter group too.</summary>
        public static void MarkCentroid(Spectrum spectrum)
        {
            var terms = spectrum.Params;
            for (int i = terms.CVParams.Count - 1; i >= 0; i--)
            {
                if (terms.CVParams[i].Cvid == CVID.MS_profile_spectrum)
                    terms.CVParams.RemoveAt(i);
            }
            foreach (var group in terms.ParamGroups.ToList())
            {
                if (group.CVParams.All(p => p.Cvid != CVID.MS_profile_spectrum))
                    continue;
                // The term comes from a shared group: copy the group's other terms onto the spectrum.
                foreach (var p in group.CVParams.Where(p => p.Cvid != CVID.MS_profile_spectrum && !terms.HasCVParam(p.Cvid)))
                    terms.CVParams.Add(p);
                terms.UserParams.AddRange(group.UserParams);
                terms.ParamGroups.Remove(group);
            }
            terms.Set(CVID.MS_centroid_spectrum);
        }
    }
}
