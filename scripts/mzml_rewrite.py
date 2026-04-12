#!/usr/bin/env python3
"""Rewrite a broken mzML file by parsing and re-emitting it cleanly.

When MSConvert produces structurally damaged centroid mzML files (empty
spectra, inconsistent indices, missing scans), surgical patching is fragile.
This script reads the broken file with pyteomics and writes a fresh mzML
using psims, which handles all the structural bookkeeping (indices, counts,
checksums, metadata sections, references).

By default, spectra with zero peaks are dropped. The output is guaranteed
structurally correct because psims emits a fully-populated mzML.

Usage:
    python mzml_rewrite.py input.mzML -o output.mzML
    python mzml_rewrite.py *.mzML -o /output/dir/
"""

import argparse
import logging
import mmap
import re
import sys
from pathlib import Path

import numpy as np
from psims.mzml import MzMLWriter
from pyteomics import mzml as pyteomics_mzml

log = logging.getLogger(__name__)


def _find_spectrum_block(data: bytes, offset: int) -> tuple[int, int] | None:
    """Find the `<spectrum>` block containing a given byte offset.

    Searches backwards for the nearest `<spectrum ` and forwards for the
    matching `</spectrum>` end tag.

    Args:
        data: File contents.
        offset: Byte offset inside a spectrum.

    Returns:
        Tuple of (start_byte, end_byte_exclusive) or None if not found.
    """
    # Search backwards for <spectrum
    search_start = max(0, offset - 200_000)
    chunk = data[search_start:offset]
    last = chunk.rfind(b"<spectrum ")
    if last == -1:
        return None
    block_start = search_start + last

    # Search forwards for </spectrum>
    end_tag = data.find(b"</spectrum>", offset)
    if end_tag == -1:
        return None
    block_end = end_tag + len(b"</spectrum>")

    return (block_start, block_end)


def strip_bad_spectra(
    input_path: Path,
    output_path: Path,
    null_threshold: int = 64,
) -> dict:
    """Create a cleaned mzML with corrupt or empty spectra removed.

    Strips any `<spectrum>` block that contains null-byte corruption, and
    optionally empty spectra (defaultArrayLength="0"). The output is NOT
    guaranteed to have valid indices or counts, but should be well-formed
    XML that downstream parsers (like pyteomics) can stream-read.

    Args:
        input_path: Broken mzML file.
        output_path: Output file with bad spectra excised.
        null_threshold: Minimum consecutive null bytes to treat as corruption.

    Returns:
        Stats dict: {'null_regions', 'corrupt_spectra', 'empty_spectra',
        'total_removed'}.
    """
    log.info("Pre-cleaning %s", input_path.name)
    with open(input_path, "rb") as f:
        data = bytearray(f.read())

    stats = {
        "null_regions": 0,
        "corrupt_spectra": 0,
        "empty_spectra": 0,
        "total_removed": 0,
    }

    # Find spectrum blocks to remove
    ranges_to_remove: list[tuple[int, int]] = []

    # 1. Null-byte corrupted spectrum blocks
    needle = b"\x00" * null_threshold
    pos = 0
    seen_null_regions: list[tuple[int, int]] = []
    while True:
        idx = data.find(needle, pos)
        if idx == -1:
            break
        end = idx + null_threshold
        while end < len(data) and data[end] == 0:
            end += 1
        seen_null_regions.append((idx, end))
        pos = end

    stats["null_regions"] = len(seen_null_regions)
    for null_start, null_end in seen_null_regions:
        block = _find_spectrum_block(bytes(data), null_start)
        if block is None:
            log.warning(
                "  null region at %d not inside a spectrum block", null_start
            )
            continue
        ranges_to_remove.append(block)
        stats["corrupt_spectra"] += 1

    # 2. Empty spectra (defaultArrayLength="0")
    empty_pattern = re.compile(rb'<spectrum [^>]*defaultArrayLength="0"[^>]*>')
    for m in empty_pattern.finditer(data):
        end_tag = data.find(b"</spectrum>", m.start())
        if end_tag == -1:
            continue
        block_start = m.start()
        block_end = end_tag + len(b"</spectrum>")
        ranges_to_remove.append((block_start, block_end))
        stats["empty_spectra"] += 1

    # Deduplicate and merge overlapping ranges
    ranges_to_remove.sort()
    merged: list[tuple[int, int]] = []
    for start, end in ranges_to_remove:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))

    stats["total_removed"] = len(merged)
    log.info(
        "  %d null regions, %d corrupt spectra, %d empty spectra (%d unique blocks)",
        stats["null_regions"],
        stats["corrupt_spectra"],
        stats["empty_spectra"],
        stats["total_removed"],
    )

    # Delete in reverse order
    for start, end in reversed(merged):
        # Also consume leading whitespace
        s = start
        while s > 0 and data[s - 1 : s] in (b" ", b"\t", b"\n"):
            s -= 1
        del data[s:end]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(bytes(data))
    log.info("  wrote cleaned intermediate: %s", output_path.name)
    return stats


def _fix_index_list_offset(path: Path) -> bool:
    """Correct `<indexListOffset>` to point at `<indexList` exactly.

    psims emits indexListOffset pointing at the whitespace before <indexList>
    rather than at the tag itself. Strict parsers (including DIA-NN) can
    reject this. This function finds the actual <indexList position and
    patches the offset value in place.

    Args:
        path: Output mzML file to fix.

    Returns:
        True if a fix was applied, False if already correct or not needed.
    """
    with open(path, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_WRITE)

        ilo_match = re.search(rb"<indexListOffset>(\d+)</indexListOffset>", mm)
        if ilo_match is None:
            mm.close()
            return False

        declared = int(ilo_match.group(1))
        actual = mm.find(b"<indexList ")
        if actual == -1:
            mm.close()
            return False

        if declared == actual:
            mm.close()
            return False

        new_str = str(actual).encode()
        old_str = ilo_match.group(1)
        if len(new_str) != len(old_str):
            # Pad or truncate to maintain byte count
            if len(new_str) < len(old_str):
                new_str = new_str + b" " * (len(old_str) - len(new_str))
            else:
                log.warning(
                    "indexListOffset digit count grew (%d -> %d); "
                    "cannot fix in place",
                    len(old_str),
                    len(new_str),
                )
                mm.close()
                return False

        mm[ilo_match.start(1) : ilo_match.end(1)] = new_str
        mm.flush()
        mm.close()

    log.info("  corrected indexListOffset: %d -> %d", declared, actual)
    return True


def _extract_precursor(spectrum: dict) -> dict | None:
    """Extract precursor information from a pyteomics spectrum.

    Args:
        spectrum: Pyteomics spectrum dict.

    Returns:
        Dict suitable for psims write_spectrum precursor_information arg, or
        None if no precursor information is present.
    """
    plist = spectrum.get("precursorList")
    if not plist:
        return None
    precursors = plist.get("precursor", [])
    if not precursors:
        return None
    p = precursors[0]

    sel_ions = p.get("selectedIonList", {}).get("selectedIon", [{}])
    ion = sel_ions[0] if sel_ions else {}
    iso = p.get("isolationWindow", {})
    activation = p.get("activation", {})

    target = iso.get(
        "isolation window target m/z", ion.get("selected ion m/z", 0.0)
    )
    lower = iso.get("isolation window lower offset", 0.0)
    upper = iso.get("isolation window upper offset", 0.0)

    info = {
        "mz": float(ion.get("selected ion m/z", target)),
        "intensity": float(ion.get("peak intensity", 0.0)),
        "isolation_window_args": {
            "target": float(target),
            "lower": float(lower),
            "upper": float(upper),
        },
    }
    if "charge state" in ion:
        info["charge"] = int(ion["charge state"])

    # Activation params (e.g. CID, HCD)
    activation_params = []
    for key in (
        "collision-induced dissociation",
        "higher energy collision-induced dissociation",
        "beam-type collision-induced dissociation",
        "electron transfer dissociation",
    ):
        if key in activation:
            activation_params.append(key)
    if "collision energy" in activation:
        activation_params.append(
            {"name": "collision energy", "value": activation["collision energy"]}
        )
    info["activation"] = activation_params

    return info


def _extract_scan_info(spectrum: dict) -> tuple[float | None, list]:
    """Extract scan start time and scan-level params from a pyteomics spectrum.

    Args:
        spectrum: Pyteomics spectrum dict.

    Returns:
        Tuple of (scan_start_time in minutes, list of scan params).
    """
    scan_list = spectrum.get("scanList", {}).get("scan", [{}])
    scan = scan_list[0] if scan_list else {}
    rt = scan.get("scan start time")

    scan_params = []
    if "filter string" in scan:
        scan_params.append({"name": "filter string", "value": scan["filter string"]})
    if "ion injection time" in scan:
        scan_params.append(
            {"name": "ion injection time", "value": scan["ion injection time"]}
        )
    if "preset scan configuration" in scan:
        scan_params.append(
            {
                "name": "preset scan configuration",
                "value": scan["preset scan configuration"],
            }
        )
    return rt, scan_params


def _extract_scan_windows(spectrum: dict) -> list:
    """Extract scan window list from a pyteomics spectrum.

    Args:
        spectrum: Pyteomics spectrum dict.

    Returns:
        List of (lower, upper) m/z tuples for psims scan_window_list arg.
    """
    scan_list = spectrum.get("scanList", {}).get("scan", [{}])
    scan = scan_list[0] if scan_list else {}
    windows = scan.get("scanWindowList", {}).get("scanWindow", [])
    result = []
    for w in windows:
        lower = w.get("scan window lower limit")
        upper = w.get("scan window upper limit")
        if lower is not None and upper is not None:
            result.append((float(lower), float(upper)))
    return result


def rewrite_mzml(
    input_path: Path,
    output_path: Path,
    drop_empty: bool = True,
    pre_clean: bool = True,
) -> dict:
    """Read an mzML file and write it back fresh.

    If the input has null-byte corruption, performs a pre-cleaning pass
    that strips corrupted spectrum blocks into a temp file before running
    the pyteomics/psims rewrite.

    Args:
        input_path: Source mzML file (possibly broken).
        output_path: Destination for the rewritten mzML.
        drop_empty: If True, skip spectra with zero peaks (default).
        pre_clean: If True, auto-strip corrupt spectra before rewrite.

    Returns:
        Stats dict.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Input:  %s", input_path.name)
    log.info("Output: %s", output_path.name)

    stats: dict = {
        "total_read": 0,
        "written": 0,
        "skipped_empty": 0,
        "pre_cleaned": False,
    }

    # Check for null-byte corruption and optionally pre-clean
    read_from = input_path
    temp_cleaned = None
    if pre_clean:
        with open(input_path, "rb") as f:
            needle = b"\x00" * 64
            chunk_size = 16 * 1024 * 1024
            has_nulls = False
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                if needle in chunk:
                    has_nulls = True
                    break

        if has_nulls:
            import tempfile

            tmp = tempfile.NamedTemporaryFile(
                suffix=".mzML", prefix="mzml_clean_", delete=False
            )
            tmp.close()
            temp_cleaned = Path(tmp.name)
            strip_bad_spectra(input_path, temp_cleaned)
            read_from = temp_cleaned
            stats["pre_cleaned"] = True

    try:
        # Counting pass
        log.info("Counting spectra...")
        n_total = 0
        n_empty = 0
        with pyteomics_mzml.read(str(read_from), use_index=False) as r:
            for s in r:
                n_total += 1
                if len(s.get("m/z array", [])) == 0:
                    n_empty += 1
        n_to_write = n_total - n_empty if drop_empty else n_total
        log.info(
            "Found %d spectra (%d empty); will write %d",
            n_total,
            n_empty,
            n_to_write,
        )
        stats["n_total"] = n_total
        stats["n_empty"] = n_empty

        with MzMLWriter(str(output_path), close=True) as writer:
            writer.controlled_vocabularies()

            writer.file_description(
                file_contents=[
                    "MS1 spectrum",
                    "MSn spectrum",
                    "centroid spectrum",
                ],
                source_files=[
                    {
                        "id": "RAW1",
                        "name": input_path.stem + ".raw",
                        "location": "file:///",
                        "params": [
                            "Thermo nativeID format",
                            "Thermo RAW format",
                        ],
                    }
                ],
            )

            writer.software_list(
                [
                    {
                        "id": "psims-rewriter",
                        "version": "1.0",
                        "params": ["python-psims"],
                    }
                ]
            )

            writer.instrument_configuration_list(
                [
                    {
                        "id": "IC1",
                        "component_list": [],
                        "params": ["instrument model"],
                    }
                ]
            )

            writer.data_processing_list(
                [
                    {
                        "id": "rewriter_processing",
                        "processing_methods": [
                            {
                                "order": 1,
                                "software_reference": "psims-rewriter",
                                "params": ["file format conversion"],
                            }
                        ],
                    }
                ]
            )

            with writer.run(
                id=input_path.stem,
                instrument_configuration="IC1",
            ):
                with writer.spectrum_list(count=n_to_write):
                    with pyteomics_mzml.read(
                        str(read_from), use_index=False
                    ) as reader:
                        for spectrum in reader:
                            stats["total_read"] += 1

                            mz_array = spectrum.get("m/z array")
                            int_array = spectrum.get("intensity array")

                            if mz_array is None or len(mz_array) == 0:
                                if drop_empty:
                                    stats["skipped_empty"] += 1
                                    continue
                                mz_array = np.array([], dtype=np.float64)
                                int_array = np.array([], dtype=np.float32)

                            ms_level = int(spectrum.get("ms level", 1))
                            scan_id = spectrum.get(
                                "id",
                                f"controllerType=0 controllerNumber=1 scan={stats['written']+1}",
                            )

                            rt, scan_params = _extract_scan_info(spectrum)
                            windows = _extract_scan_windows(spectrum)
                            precursor_info = (
                                _extract_precursor(spectrum) if ms_level >= 2 else None
                            )

                            spec_params = [
                                {"ms level": ms_level},
                                "MS1 spectrum" if ms_level == 1 else "MSn spectrum",
                                "centroid spectrum",
                                "positive scan",
                            ]

                            try:
                                writer.write_spectrum(
                                    mz_array=np.asarray(mz_array, dtype=np.float64),
                                    intensity_array=np.asarray(
                                        int_array, dtype=np.float32
                                    ),
                                    id=scan_id,
                                    centroided=True,
                                    scan_start_time=rt,
                                    params=spec_params,
                                    scan_params=scan_params,
                                    scan_window_list=windows if windows else None,
                                    precursor_information=precursor_info,
                                    instrument_configuration_id="IC1",
                                )
                            except Exception as e:
                                log.error(
                                    "Failed to write spectrum %s: %s", scan_id, e
                                )
                                raise

                            stats["written"] += 1
                            if stats["written"] % 5000 == 0:
                                log.info(
                                    "  written %d / %d (skipped %d empty)",
                                    stats["written"],
                                    n_to_write,
                                    stats["skipped_empty"],
                                )

        _fix_index_list_offset(output_path)

        log.info(
            "Done: %d read, %d written, %d skipped empty",
            stats["total_read"],
            stats["written"],
            stats["skipped_empty"],
        )
        return stats
    finally:
        if temp_cleaned is not None and temp_cleaned.exists():
            temp_cleaned.unlink()


def main() -> int:
    """Entry point.

    Returns:
        Exit code.
    """
    parser = argparse.ArgumentParser(
        description="Rewrite broken mzML files by parsing and re-emitting them"
    )
    parser.add_argument("input", nargs="+", help="Input mzML file(s)")
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output file (single input) or output directory (multiple inputs)",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Keep spectra with zero peaks (default: drop them)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    inputs = [Path(p) for p in args.input]
    output = Path(args.output)

    if len(inputs) > 1 or output.is_dir() or output.exists() and output.is_dir():
        output.mkdir(parents=True, exist_ok=True)
        for inp in inputs:
            out = output / inp.name
            rewrite_mzml(inp, out, drop_empty=not args.keep_empty)
    else:
        rewrite_mzml(inputs[0], output, drop_empty=not args.keep_empty)

    return 0


if __name__ == "__main__":
    sys.exit(main())
