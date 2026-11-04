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
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Centrix;

namespace Centrix.Demux
{
    /// <summary>Settings for <see cref="StaggeredDemuxPipeline"/> and the solves it runs.</summary>
    public sealed class ScanningDemuxOptions
    {
        public ScanningLayout Layout { get; set; } = new ScanningLayout(ScanningLayoutKind.centered, 5);
        public ScanningDemuxParams Parameters { get; set; } = new ScanningDemuxParams();
        public int Threads { get; set; } = Environment.ProcessorCount;
        public int FirstCycle { get; set; }
        public int LastCycle { get; set; } = int.MaxValue;
        public double MinMz { get; set; }
        public double MaxMz { get; set; } = double.MaxValue;

        /// <summary>The acquired spectra of the selected bins, zeros dropped: the control arm.</summary>
        public bool Raw { get; set; }

        /// <summary>Detector counts per ion (about 100 on a ZenoTOF 8600).</summary>
        public double CountsPerIon { get; set; } = 100;

        /// <summary>
        /// Staggered data whose ions follow each spectrum's fill time (Thermo: an ion trap, an Astral, an Orbitrap):
        /// a spectrum's ions are its intensity x its injection (fill) time in seconds over its analyzer's ion response
        /// (<see cref="Demux.IonResponse"/>, the spectrum's or <see cref="DefaultIonResponse"/>). Spectra are combined
        /// as rates, the only unit that compares across fill times, while each observation's own ions set the Poisson
        /// weights and the channel thresholds. Output stays in the instrument's units. Spectra without an injection
        /// time keep <see cref="CountsPerIon"/>.
        /// </summary>
        public bool IonsFromFillTime { get; set; }

        /// <summary>
        /// With <see cref="IonsFromFillTime"/>, the ion response of spectra whose analyzer the source does not name:
        /// 1, intensities in ions per second (an ion trap's or an Astral's, near enough).
        /// </summary>
        public double DefaultIonResponse { get; set; } = 1;

        /// <summary>
        /// Staggered profile data on a linear grid (an ion trap's), with <see cref="Joint"/>: the profile peak's sigma,
        /// in Th, which the joint solve's Gaussian and Centrix take. 0, the default, takes it from the run's MS2
        /// (<see cref="CentrixCalibration"/>: 1.1 point spacings on a linear ion trap, 0.22 Th on a Thermo Stellar at
        /// 200 kTh/s).
        /// </summary>
        public double LinearPeakSigmaTh { get; set; }

        /// <summary>
        /// The joint solve: split the measured profile among the bins (staggered) or source positions (ZT Scan) by the
        /// fitted peaks of each, then centroid each share (<see cref="JointApportioner"/>), rather than writing the fitted
        /// peaks themselves. Staggered, the fit carries the errors of the interpolated windows. ZT Scan, each position's
        /// share in a sweep is summed over the encoded bins that saw it and centroided by Centrix on the TOF grid
        /// (GridCentroider) in place of the fit's own runs of coefficients.
        /// </summary>
        public bool JointApportion { get; set; } = true;

        /// <summary>
        /// ZT Scan, with <see cref="JointApportion"/>: Centrix centroids each position's fitted profile (its peaks,
        /// B * beta) rather than its split measurement, which isolates the centroider from the measurement's noise.
        /// </summary>
        public bool JointCentrixFitted { get; set; }

        /// <summary>
        /// With <see cref="JointApportion"/>: centroid each bin's share with Centrix (<see cref="Centrix"/>); false
        /// writes the shares as profile, for a centroider of its own.
        /// </summary>
        public bool JointCentroid { get; set; } = true;

        /// <summary>
        /// With <see cref="JointCentroid"/>: the settings of the Centrix lasso that centroids each bin's share. Its peak
        /// sigma is <see cref="LinearPeakSigmaTh"/> and its grid spacing the profile's, whatever is set here.
        /// </summary>
        public CentrixParams Centrix { get; set; } = new CentrixParams();

        /// <summary>
        /// Staggered data, for measuring how sampling matters: when above 1, only every this many cycles' MS2 spectra
        /// are demultiplexed and written (a cycle being one pass of every window set); the others are dropped.
        /// </summary>
        public int ThinCycles { get; set; } = 1;

        /// <summary>
        /// Staggered data: fit each fragment channel's elution with chromatographic peaks of this shape to apportion
        /// each observation (<see cref="ElutionDemultiplexer"/>), instead of interpolating each window's own series.
        /// Null interpolates.
        /// </summary>
        public ElutionDemuxParams Elution { get; set; }

        /// <summary>
        /// Demultiplex and centroid profile spectra in one solve (<see cref="JointDemultiplexer"/>) instead of
        /// demultiplexing centroids channel by channel. Needs profile input on one TOF grid.
        /// </summary>
        public bool Joint { get; set; }

        /// <summary>Settings for <see cref="Joint"/>.</summary>
        public JointDemuxParams JointParameters { get; set; } = new JointDemuxParams();

        /// <summary>
        /// Staggered data: MS2 spectra isolated wider than this (Th) are passed through rather than demultiplexed, and
        /// are no part of the scheme. A Thermo Stellar stores its Adaptive RT spectra (20 or 50 m/z wide) among its DIA
        /// spectra, whose windows never exceed 10 m/z. 0 demultiplexes every MS2 spectrum.
        /// </summary>
        public double MaxWindowWidthTh { get; set; }

        /// <summary>Sweeps whose output one block owns.</summary>
        public int BlockCycles { get; set; } = 12;

        /// <summary>Sweeps read beyond a block on each side, so its edge sweeps see their neighbors.</summary>
        public int CyclePad { get; set; } = 4;

        /// <summary>Encoded bins whose output one block owns.</summary>
        public int GroupBins { get; set; } = 16;

        /// <summary>Source positions solved beyond a block's own bins on each side.</summary>
        public int ContextBins { get; set; } = 10;

        /// <summary>Encoded bins a precursor's transmission reaches on each side.</summary>
        public int ReachBins { get; set; } = 9;

        /// <summary>
        /// Threads reading one sweep's spectra from the source, for a source that serves concurrent requests:
        /// the vendor readers do (SCIEX .wiff2: 4x on four threads, peaks identical). 1 reads serially.
        /// </summary>
        public int ReadThreads { get; set; } = 1;

        /// <summary>
        /// Demultiplexed peaks of neighbouring positions closer than this are summed into one in an output
        /// spectrum. The channel solve gives one fragment nearly the same m/z in every position; the joint solve
        /// centroids each position on its own, so the same fragment's centroids can lie a grid sample apart.
        /// </summary>
        public double MergePpm { get; set; } = ScanningLayout.MERGE_PPM;

        /// <summary>
        /// For the joint solve: neighbouring positions' centroids closer than this many TOF peak sigmas (at their
        /// m/z, <see cref="JointDemuxParams.SigmaAt"/>) are summed into one, in place of <see cref="MergePpm"/>;
        /// 0 uses <see cref="MergePpm"/>. Two Gaussians closer than sigma are not resolved (Centrix's merge); on
        /// the ZT Scan slice two sigmas left about 60% of one sigma's close doublets and found 3-4% more peptides.
        /// </summary>
        public double JointMergeSigmas { get; set; } = 2;

        /// <summary>
        /// Centroid MS1 with the joint solve (one bin, one position) instead of passing it through: needs profile
        /// MS1, which is on a TOF grid of its own, and takes each spectrum's own counts per ion (SCIEX MS1
        /// intensities are rates over an accumulation time that varies from spectrum to spectrum).
        /// </summary>
        public bool JointMs1 { get; set; }

        /// <summary>
        /// Fit the joint solve's MS2 with the TOF peak kernels measured from the file's first batch of sweeps
        /// (TofPeakShape) instead of the Gaussian of <see cref="JointDemuxParams.PeakSigmaSamples"/>.
        /// MS1, with <see cref="JointMs1"/>, is always fitted with its own measured kernels.
        /// </summary>
        public bool MeasuredPeakShape { get; set; }
    }
}
