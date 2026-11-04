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
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Pwiz.Analysis;
using Pwiz.Data.Common.Cv;
using Pwiz.Data.Common.Params;
using Pwiz.Data.MsData.Instruments;
using Pwiz.Data.MsData.Processing;
using Pwiz.Data.MsData.Spectra;

namespace Centrix.Cli
{
    /// <summary>
    /// A spectrum list whose profile spectra are centroided: by Centrix where <see cref="SpectrumRouting"/> says the
    /// spectrum is a linear ion trap's (or always, with <see cref="CentroidMode.centrix"/>), by a fallback list (the
    /// vendor's centroids, or pwiz's wavelet picker) otherwise. Spectra already centroided pass through.
    /// <para>Read in order by a writer, the spectra are read ahead on one thread, a batch at a time, and each batch's
    /// Centrix spectra are centroided in parallel while the next is read: reading stays serial (a Thermo .raw serves one
    /// request at a time), and Centrix, several times the cost of the read, uses every thread. Each spectrum is
    /// centroided on its own, so the output does not depend on the thread count or the batch size. Any other access
    /// is served one spectrum at a time.</para>
    /// </summary>
    internal sealed class CentrixSpectrumList : SpectrumListWrapper
    {
        /// <summary>Spectra read and centroided together.</summary>
        public const int DEFAULT_BATCH_SIZE = 256;

        /// <summary>Batches read ahead of the one the writer is taking spectra from.</summary>
        private const int BATCHES_AHEAD = 2;

        private readonly Lazy<ISpectrumList> _fallback;
        private Dictionary<string, int> _fallbackById;   // made when the fallback's indices are not the inner list's
        private readonly CentrixCentroider[] _centroiders;
        private readonly CentroidMode _mode;
        private readonly int _threads;
        private readonly int _batchSize;
        private readonly object _readLock = new object();
        private readonly DataProcessing _dataProcessing;
        private readonly CancellationTokenSource _cancel = new CancellationTokenSource();

        // The read-ahead: batches in index order, each a task that completes when its spectra are centroided.
        private BlockingCollection<Task<Batch>> _ahead;
        private Task _reader;
        private Batch _current;
        private int _lastIndex = -1;

        // Counts, for the log.
        private long _centrixSpectra, _fallbackSpectra, _passedThrough, _centroidsOut;

        /// <param name="inner">The profile spectra, as read (no vendor peak picking).</param>
        /// <param name="fallback">
        /// The same spectra centroided another way, for the profile spectra Centrix is not for: index for index with
        /// <paramref name="inner"/>, or holding them under the same ids (as a list <paramref name="inner"/> demultiplexes
        /// does); made the first time a spectrum needs it (the vendor's centroids are a second list over the
        /// reader, not worth building for a run Centrix centroids all of). Null, or a null list, leaves them as read. It is
        /// read under the same lock as <paramref name="inner"/>, since a Thermo .raw serves one request at a time.
        /// </param>
        /// <param name="ms1">The MS1 centroider.</param>
        /// <param name="ms2">The centroider of MS2 and above.</param>
        /// <param name="mode">Which profile spectra Centrix centroids.</param>
        /// <param name="threads">Threads centroiding a batch.</param>
        /// <param name="software">The software the processing method names (Centrix's).</param>
        /// <param name="batchSize">Spectra read and centroided together.</param>
        public CentrixSpectrumList(ISpectrumList inner, Lazy<ISpectrumList> fallback, CentrixCentroider ms1,
            CentrixCentroider ms2, CentroidMode mode, int threads, Software software, int batchSize = DEFAULT_BATCH_SIZE)
            : base(inner)
        {
            _fallback = fallback;
            _centroiders = new[] { null, ms1, ms2 };
            _mode = mode;
            _threads = Math.Max(1, threads);
            _batchSize = Math.Max(1, batchSize);
            _dataProcessing = BuildDataProcessing(inner.DataProcessing, software, ms1, ms2);
        }

        public override DataProcessing DataProcessing => _dataProcessing;

        public long CentrixSpectra => Interlocked.Read(ref _centrixSpectra);
        public long FallbackSpectra => Interlocked.Read(ref _fallbackSpectra);
        public long PassedThrough => Interlocked.Read(ref _passedThrough);
        public long CentroidsOut => Interlocked.Read(ref _centroidsOut);

        public override SpectrumIdentity SpectrumIdentity(int index)
        {
            lock (_readLock)
            {
                return Inner.SpectrumIdentity(index);
            }
        }

        public override Spectrum GetSpectrum(int index, bool getBinaryData = false)
        {
            int last = _lastIndex;
            _lastIndex = index;
            if (_current != null && index >= _current.Start && index < _current.End)
                return _current.Spectra[index - _current.Start];
            // The next batch, in order: from the read-ahead, which starts when the spectra are asked for in order from
            // the first one asked for.
            bool next = _ahead != null ? index == _current.End : index == last + 1;
            if (next && index < Count)
            {
                if (_ahead == null)
                    StartReadAhead(index);
                _current = _ahead.Take().GetAwaiter().GetResult();
                return _current.Spectra[index - _current.Start];
            }
            // Out of order: one spectrum on its own.
            return Process(Read(index));
        }

        protected override void DisposeCore()
        {
            _cancel.Cancel();
            try
            {
                _reader?.Wait();
            }
            catch (AggregateException)
            {
                // Cancelled or failed; the writer has its result already.
            }
            _cancel.Dispose();
            base.DisposeCore();
        }

        /// <summary>
        /// Reads every batch from <paramref name="from"/> on, in order, on its own thread, queueing each as a task
        /// that centroids it in parallel; at most <see cref="BATCHES_AHEAD"/> wait in the queue.
        /// </summary>
        private void StartReadAhead(int from)
        {
            _ahead = new BlockingCollection<Task<Batch>>(BATCHES_AHEAD);
            var token = _cancel.Token;
            _reader = Task.Factory.StartNew(() =>
            {
                try
                {
                    for (int first = from; first < Count && !token.IsCancellationRequested; first += _batchSize)
                    {
                        // Its own copy: a for loop's variable is one variable, which the task would see move on.
                        int start = first, end = Math.Min(Count, first + _batchSize);
                        Task<Batch> batch;
                        try
                        {
                            var read = Enumerable.Range(start, end - start).Select(Read).ToArray();
                            batch = Task.Run(() => Centroid(start, read), token);
                        }
                        catch (Exception e)
                        {
                            // Handed to the writer when it reaches this batch.
                            batch = Task.FromException<Batch>(e);
                        }
                        _ahead.Add(batch, token);
                        if (batch.IsFaulted)
                            break;
                    }
                }
                catch (OperationCanceledException)
                {
                }
                finally
                {
                    _ahead.CompleteAdding();
                }
            }, token, TaskCreationOptions.LongRunning, TaskScheduler.Default);
        }

        private Batch Centroid(int first, ReadSpectrum[] read)
        {
            var spectra = new Spectrum[read.Length];
            Parallel.For(0, read.Length, new ParallelOptions { MaxDegreeOfParallelism = _threads },
                k => spectra[k] = Process(read[k]));
            return new Batch(first, spectra);
        }

        /// <summary>
        /// One spectrum as the writer will take it: read with its peaks, routed, and the fallback's centroids read now,
        /// under the read lock, so only Centrix is left for the parallel part.
        /// </summary>
        private ReadSpectrum Read(int index)
        {
            lock (_readLock)
            {
                var spectrum = Inner.GetSpectrum(index, true);
                var route = SpectrumRouting.Route(spectrum, _mode);
                var fallback = route == SpectrumRoute.Fallback ? _fallback?.Value : null;
                if (fallback == null)
                    return new ReadSpectrum(spectrum, route, false);
                var centroided = fallback.GetSpectrum(FallbackIndex(fallback, index, spectrum.Id), true);
                if (centroided.Id != spectrum.Id)
                {
                    throw new InvalidOperationException($"The fallback centroids spectrum {index} as '{centroided.Id}', " +
                        $"not '{spectrum.Id}'");
                }
                centroided.Index = index;
                return new ReadSpectrum(centroided, SpectrumRoute.PassThrough, true);
            }
        }

        /// <summary>
        /// Where a spectrum is in the fallback: at its own index, or, when the spectra Centrix reads are not the
        /// fallback's index for index (a demultiplexed list passes its MS1 through at new indices, under their ids), by
        /// its id. Called under the read lock.
        /// </summary>
        private int FallbackIndex(ISpectrumList fallback, int index, string id)
        {
            if (_fallbackById == null && index < fallback.Count && fallback.SpectrumIdentity(index).Id == id)
                return index;
            if (_fallbackById == null)
            {
                _fallbackById = new Dictionary<string, int>(fallback.Count, StringComparer.Ordinal);
                for (int i = 0; i < fallback.Count; i++)
                    _fallbackById.TryAdd(fallback.SpectrumIdentity(i).Id, i);
            }
            if (!_fallbackById.TryGetValue(id, out int found))
                throw new InvalidOperationException($"The fallback has no spectrum '{id}'");
            return found;
        }

        private Spectrum Process(ReadSpectrum read)
        {
            var spectrum = read.Spectrum;
            if (read.Route != SpectrumRoute.Centrix)
            {
                if (read.FromFallback)
                    Interlocked.Increment(ref _fallbackSpectra);
                else
                    Interlocked.Increment(ref _passedThrough);
                return spectrum;
            }
            int level = spectrum.Params.CvParamValueOrDefault(CVID.MS_ms_level, 0);
            var centroider = _centroiders[level <= 1 ? 1 : 2];
            var mzArray = spectrum.GetMZArray();
            var intensityArray = spectrum.GetIntensityArray();
            var mz = new List<double>();
            var intensity = new List<double>();
            if (mzArray != null && intensityArray != null)
                centroider.Centroid(mzArray.Data, intensityArray.Data, mz, intensity);

            // The peak arrays replaced in place, keeping their units and encoding; arrays that ran point for point
            // with the profile no longer mean anything.
            int profilePoints = mzArray?.Data.Count ?? 0;
            for (int i = spectrum.BinaryDataArrays.Count - 1; i >= 0; i--)
            {
                var array = spectrum.BinaryDataArrays[i];
                if (array != mzArray && array != intensityArray && array.Data.Count == profilePoints)
                    spectrum.BinaryDataArrays.RemoveAt(i);
            }
            if (mzArray != null && intensityArray != null)
            {
                mzArray.Data.Clear();
                mzArray.Data.AddRange(mz);
                intensityArray.Data.Clear();
                intensityArray.Data.AddRange(intensity);
            }
            spectrum.DefaultArrayLength = mz.Count;
            SpectrumRouting.MarkCentroid(spectrum);
            spectrum.DataProcessing = _dataProcessing;
            Interlocked.Increment(ref _centrixSpectra);
            Interlocked.Add(ref _centroidsOut, mz.Count);
            return spectrum;
        }

        /// <summary>The input's processing with Centrix's peak picking added, naming the solver and each level's sigma.</summary>
        private static DataProcessing BuildDataProcessing(DataProcessing inner, Software software,
            CentrixCentroider ms1, CentrixCentroider ms2)
        {
            // An id of its own: the input's processing is still in the file's list (an mzML's), and a reader resolves
            // a reference to the first of two with one id.
            var dataProcessing = new DataProcessing("Centrix_processing");
            if (inner != null)
                dataProcessing.ProcessingMethods.AddRange(inner.ProcessingMethods);
            var method = new ProcessingMethod { Order = dataProcessing.ProcessingMethods.Count, Software = software };
            method.Set(CVID.MS_peak_picking);
            method.UserParams.Add(new UserParam("Centrix centroiding", ms2.Describe()));
            if (ms1 != ms2)
                method.UserParams.Add(new UserParam("Centrix MS1", ms1.Describe()));
            dataProcessing.ProcessingMethods.Add(method);
            return dataProcessing;
        }

        private readonly struct ReadSpectrum
        {
            public ReadSpectrum(Spectrum spectrum, SpectrumRoute route, bool fromFallback)
            {
                Spectrum = spectrum;
                Route = route;
                FromFallback = fromFallback;
            }

            public Spectrum Spectrum { get; }
            public SpectrumRoute Route { get; }
            public bool FromFallback { get; }
        }

        private sealed class Batch
        {
            public Batch(int start, Spectrum[] spectra)
            {
                Start = start;
                Spectra = spectra;
            }

            public int Start { get; }
            public int End => Start + Spectra.Length;
            public Spectrum[] Spectra { get; }
        }
    }
}
