#!/usr/bin/env python3
"""Verify and repair mzML files.

Detects and repairs two classes of mzML corruption:

1. Null-byte corruption: Network filesystems (SMB/NFS) can silently zero
   out aligned sectors (~4 KB) during large writes. Repair requires the
   original profile source mzML to reconstruct the missing XML.

2. Spectrum count mismatch: MSConvert skips spectra that Thermo's centroider
   fails on, but still writes the original (too-high) count in the
   spectrumList attribute. This causes some parsers (e.g. DIA-NN) to fail.
   No source file needed for this repair.

Works on both Centrix-generated and Thermo/MSConvert centroid mzML files.

Usage:
    # Verify all mzML files in a directory
    python mzml_verify_repair.py verify /path/to/*.mzML

    # Repair count mismatches (no source needed)
    python mzml_verify_repair.py repair /path/to/centroid/*.mzML

    # Repair null-byte corruption (needs profile source)
    python mzml_verify_repair.py repair \\
        --source-dir /path/to/profile/ \\
        /path/to/centroid/*.mzML
"""

import argparse
import logging
import mmap
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# Consecutive null bytes above this threshold indicate corruption, not
# legitimate data (base64-encoded binary arrays never contain null runs
# this long).
NULL_THRESHOLD = 512


@dataclass
class CorruptRegion:
    """A contiguous run of null bytes in a file."""

    offset: int
    length: int


@dataclass
class CountMismatch:
    """A mismatch between declared and actual spectrum/index counts."""

    declared_count: int
    actual_spectra: int
    index_entries: int


def find_null_regions(
    path: Path, threshold: int = NULL_THRESHOLD
) -> list[CorruptRegion]:
    """Scan a file for contiguous runs of null bytes.

    Args:
        path: File to scan.
        threshold: Minimum run length to report.

    Returns:
        List of corrupt regions found.
    """
    regions: list[CorruptRegion] = []
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        size = len(mm)
        needle = b"\x00" * threshold
        pos = 0
        while True:
            idx = mm.find(needle, pos)
            if idx == -1:
                break
            # Extend to find full extent of null run
            end = idx + threshold
            while end < size and mm[end] == 0:
                end += 1
            regions.append(CorruptRegion(offset=idx, length=end - idx))
            pos = end
        mm.close()
    return regions


def check_spectrum_counts(path: Path) -> CountMismatch | None:
    """Check if the declared spectrumList count matches actual spectra and index.

    MSConvert can skip spectra that Thermo's centroider fails on, but still
    writes the original (too-high) count in the spectrumList attribute. This
    causes some parsers (e.g. DIA-NN) to fail.

    Args:
        path: Path to the mzML file.

    Returns:
        A CountMismatch if counts don't agree, or None if everything matches.
    """
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        # Find declared count in <spectrumList count="N">
        m = re.search(rb'<spectrumList\s+count="(\d+)"', mm)
        if m is None:
            mm.close()
            return None
        declared = int(m.group(1))

        # Count actual <spectrum elements (fast: just count occurrences)
        actual = 0
        pos = 0
        while True:
            pos = mm.find(b"<spectrum ", pos)
            if pos == -1:
                break
            actual += 1
            pos += 10

        # Count index entries for spectra
        index_entries = 0
        # Find the spectrum index section
        idx_start = mm.find(b'<index name="spectrum">')
        if idx_start != -1:
            idx_end = mm.find(b"</index>", idx_start)
            if idx_end != -1:
                chunk = mm[idx_start:idx_end]
                index_entries = chunk.count(b"<offset ")

        mm.close()

        if declared == actual and declared == index_entries:
            return None

        return CountMismatch(
            declared_count=declared,
            actual_spectra=actual,
            index_entries=index_entries,
        )


def verify_file(path: Path) -> tuple[list[CorruptRegion], CountMismatch | None]:
    """Verify a single mzML file for corruption and structural issues.

    Checks for:
    - Null-byte corruption (SMB/NFS write errors)
    - spectrumList count mismatches (MSConvert skipped spectra)

    Args:
        path: Path to the mzML file.

    Returns:
        Tuple of (null-byte regions, count mismatch or None).
    """
    has_issues = False

    regions = find_null_regions(path)
    if regions:
        log.warning(
            "%s: CORRUPT - %d null-byte region(s) found",
            path.name,
            len(regions),
        )
        for r in regions:
            log.warning(
                "  offset %s: %s null bytes",
                f"{r.offset:,}",
                f"{r.length:,}",
            )
        has_issues = True

    mismatch = check_spectrum_counts(path)
    if mismatch is not None:
        log.warning(
            "%s: COUNT MISMATCH - spectrumList declares %d, "
            "file contains %d spectra, index has %d entries",
            path.name,
            mismatch.declared_count,
            mismatch.actual_spectra,
            mismatch.index_entries,
        )
        has_issues = True

    if not has_issues:
        log.info("%s: OK", path.name)

    return regions, mismatch


def find_source_file(
    target: Path, source_dir: Path
) -> Path | None:
    """Find the corresponding source mzML for a centroided file.

    Strips common suffixes like '.centrix' to match back to the original
    profile filename.

    Args:
        target: Path to the corrupted file.
        source_dir: Directory containing original profile mzML files.

    Returns:
        Path to the matching source file, or None if not found.
    """
    stem = target.stem  # e.g. "file.centrix" from "file.centrix.mzML"
    # Strip known centroid suffixes
    for suffix in [".centrix", ".centroid", ".peaks"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    candidates = [
        source_dir / f"{stem}.mzML",
        source_dir / f"{stem}.mzml",
        source_dir / target.name,
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def extract_context(
    mm: mmap.mmap, offset: int, length: int, context_bytes: int = 8192
) -> tuple[bytes, bytes]:
    """Extract surrounding context around a null-byte region.

    Args:
        mm: Memory-mapped file.
        offset: Start of null region.
        length: Length of null region.
        context_bytes: How many bytes of context to grab on each side.

    Returns:
        Tuple of (before_context, after_context) byte strings.
    """
    start = max(0, offset - context_bytes)
    end = min(len(mm), offset + length + context_bytes)
    before = mm[start:offset]
    after = mm[offset + length : end]
    return before, after


def _find_next_xml_tag(data: bytes, start: int = 0) -> int | None:
    """Find the byte offset of the next XML tag (< followed by a letter or /).

    Args:
        data: Byte string to search.
        start: Offset to start searching from.

    Returns:
        Offset of the '<' character, or None if not found.
    """
    pos = start
    while pos < len(data):
        idx = data.find(b"<", pos)
        if idx == -1:
            return None
        # Verify it looks like an XML tag: <letter or </
        if idx + 1 < len(data):
            next_byte = data[idx + 1 : idx + 2]
            if next_byte.isalpha() or next_byte == b"/" or next_byte == b"!":
                return idx
        pos = idx + 1
    return None


def _find_spectrum_id(mm: mmap.mmap, offset: int) -> str | None:
    """Find the spectrum ID containing the given byte offset.

    Searches backwards from offset for the nearest '<spectrum ' tag and
    extracts its id attribute.

    Args:
        mm: Memory-mapped file.
        offset: Byte offset within the spectrum.

    Returns:
        The spectrum id string, or None if not found.
    """
    search_start = max(0, offset - 100_000)
    chunk = mm[search_start:offset]
    last = chunk.rfind(b"<spectrum ")
    if last == -1:
        return None
    abs_pos = search_start + last
    tag_end = mm.find(b">", abs_pos)
    if tag_end == -1:
        return None
    tag = mm[abs_pos : tag_end + 1].decode("utf-8", errors="replace")
    m = re.search(r'id="([^"]+)"', tag)
    return m.group(1) if m else None


def find_matching_region_in_source(
    target_mm: mmap.mmap,
    source_mm: mmap.mmap,
    region: "CorruptRegion",
    context_bytes: int = 8192,
) -> bytes | None:
    """Find replacement bytes for a null-byte region using the source file.

    The null region typically spans from XML passthrough content into base64
    binary data. The XML structure is identical between source and target
    (passthrough), but binary arrays differ. This function:
    1. Finds the spectrum containing the corruption.
    2. Locates the same spectrum in the source file.
    3. Matches before/after context relative to that spectrum to extract
       the correct replacement bytes.

    Args:
        target_mm: Memory-mapped corrupt file.
        source_mm: Memory-mapped source file.
        region: The corrupt region to repair.

    Returns:
        The replacement bytes from the source, or None if no match found.
    """
    offset = region.offset
    null_length = region.length

    before_ctx = target_mm[max(0, offset - context_bytes) : offset]
    after_ctx = target_mm[offset + null_length : offset + null_length + context_bytes]

    # Step 1: Find the spectrum ID in the target file
    spectrum_id = _find_spectrum_id(target_mm, offset)
    if spectrum_id is None:
        log.debug("Could not find spectrum ID for offset %d", offset)
        return None

    log.debug("Corruption is in spectrum: %s", spectrum_id)

    # Step 2: Find the same spectrum in the source file
    spectrum_tag = f'id="{spectrum_id}"'.encode()
    source_spectrum_pos = source_mm.find(spectrum_tag)
    if source_spectrum_pos == -1:
        log.debug("Spectrum %s not found in source", spectrum_id)
        return None

    # Step 3: Use before-context to find exact position within this spectrum.
    # The before-context is XML passthrough content, identical in both files.
    # Search in a window around the source spectrum position.
    for anchor_size in [200, 100, 60]:
        anchor_size = min(anchor_size, len(before_ctx))
        anchor = before_ctx[-anchor_size:]
        # Search within a reasonable range of the source spectrum
        search_start = max(0, source_spectrum_pos - 1000)
        search_end = min(len(source_mm), source_spectrum_pos + 200_000)
        source_pos = source_mm.find(anchor, search_start, search_end)
        if source_pos != -1:
            break
    else:
        log.debug("Before-context anchor not found near spectrum in source")
        return None

    source_start = source_pos + anchor_size

    # Step 4: Try direct after-context match (works when null region is
    # entirely within XML passthrough, not spanning into binary data)
    for after_anchor_size in [200, 100]:
        after_anchor_size = min(after_anchor_size, len(after_ctx))
        after_anchor = after_ctx[:after_anchor_size]
        search_end = min(len(source_mm), source_start + null_length * 3)
        after_pos = source_mm.find(after_anchor, source_start, search_end)
        if after_pos != -1:
            log.debug("Direct after-context match at source offset %d", after_pos)
            return source_mm[source_start:after_pos]

    # Step 5: After-context starts with base64 data (different between files).
    # The null region spans from XML passthrough into binary array content.
    # Strategy: read the source XML forward from source_start until we hit
    # <binary> (the boundary between XML structure and base64 data). That
    # gives us the missing XML. The lost base64 data is unrecoverable, so
    # pad with valid base64 to maintain file offsets.

    # Find the next <binary> tag in the source after our position
    search_end = min(len(source_mm), source_start + null_length * 5)
    binary_tag_pos = source_mm.find(b"<binary>", source_start, search_end)

    if binary_tag_pos == -1:
        # Try </binary> (we might be past the opening tag in source)
        binary_tag_pos = source_mm.find(b"</binary>", source_start, search_end)
        if binary_tag_pos == -1:
            log.debug("No <binary> tag found near source position")
            return None

    source_xml = source_mm[source_start:binary_tag_pos]

    # Now figure out where in the target the base64 data resumes.
    # The after-context may start with surviving base64 data, then XML.
    # Find <binary> or </binary> in the after-context to determine how
    # much base64 padding we need.
    tag_offset = _find_next_xml_tag(after_ctx)
    if tag_offset is None:
        # Entire after-context is base64; pad the full gap
        tag_offset = 0

    # The replacement must be exactly null_length bytes.
    # It consists of: source_xml (reconstructed XML) + <binary> tag +
    # base64 padding for the lost binary data.
    binary_tag = b"<binary>"
    xml_part = source_xml + binary_tag
    gap = null_length - len(xml_part)

    if gap < 0:
        # The XML portion is longer than the null region. This can happen
        # if the source file has more XML content (e.g. extra processing
        # methods). Truncate and warn.
        log.warning(
            "Source XML region (%d bytes) exceeds null region (%d bytes); "
            "byte offsets will shift. Consider re-running centrix.",
            len(xml_part),
            null_length,
        )
        return xml_part

    # Fill lost base64 data with 'A' characters (decodes to zero bytes,
    # valid base64). This produces zero-intensity entries for the affected
    # spectrum, which is better than a parse error.
    padding = b"A" * gap
    log.debug(
        "Replacing %d null bytes: %d bytes XML + %d bytes base64 padding",
        null_length,
        len(xml_part),
        gap,
    )
    return xml_part + padding


def fix_spectrum_count(data: bytearray) -> bool:
    """Fix spectrumList count attribute to match actual spectrum count.

    MSConvert sometimes declares more spectra than it actually writes
    (e.g. when Thermo's centroider fails on some scans). This corrects
    the count= attribute to match reality.

    Args:
        data: Mutable file contents.

    Returns:
        True if a fix was applied, False if counts already matched.
    """
    # Find the spectrumList tag
    m = re.search(rb'(<spectrumList\s+count=")(\d+)(")', data)
    if m is None:
        return False

    declared = int(m.group(2))

    # Count actual <spectrum elements
    actual = data.count(b"<spectrum ")

    if declared == actual:
        return False

    log.info(
        "  fixing spectrumList count: %d -> %d (%d spectra were skipped)",
        declared,
        actual,
        declared - actual,
    )

    # Replace the count value. The new count string may be shorter
    # (e.g. 96455 -> 96407), so we pad with spaces to keep byte offsets stable.
    old_count = m.group(2)
    new_count = str(actual).encode()
    # Pad to same length to preserve byte offsets for the index
    if len(new_count) < len(old_count):
        new_count = new_count + b" " * (len(old_count) - len(new_count))

    start = m.start(2)
    end = m.end(2)
    data[start:end] = new_count
    return True


def repair_file(
    target: Path,
    source: Path | None = None,
    output: Path | None = None,
    dry_run: bool = False,
) -> bool:
    """Repair an mzML file: fix null-byte corruption and/or count mismatches.

    Args:
        target: Path to the corrupted file.
        source: Path to the original source mzML. Required for null-byte
            repair, not needed for count-only fixes.
        output: Output path. If None, repairs in-place (with backup).
        dry_run: If True, only report what would be done.

    Returns:
        True if repair succeeded, False otherwise.
    """
    regions = find_null_regions(target)
    mismatch = check_spectrum_counts(target)

    if not regions and mismatch is None:
        log.info("%s: no issues found, nothing to repair", target.name)
        return True

    needs_null_repair = len(regions) > 0
    needs_count_fix = mismatch is not None

    if needs_null_repair:
        if source is None:
            log.error(
                "%s: has null-byte corruption but no --source-dir provided",
                target.name,
            )
            return False
        log.info(
            "%s: %d corrupt region(s) to repair using %s",
            target.name,
            len(regions),
            source.name,
        )

    if needs_count_fix:
        log.info(
            "%s: spectrumList count=%d but file has %d spectra "
            "(index has %d entries)",
            target.name,
            mismatch.declared_count,
            mismatch.actual_spectra,
            mismatch.index_entries,
        )

    # Build null-byte replacement plan if needed
    replacements: list[tuple[int, int, bytes]] = []
    if needs_null_repair:
        with open(target, "rb") as tf, open(source, "rb") as sf:
            target_mm = mmap.mmap(tf.fileno(), 0, access=mmap.ACCESS_READ)
            source_mm = mmap.mmap(sf.fileno(), 0, access=mmap.ACCESS_READ)

            for region in regions:
                replacement = find_matching_region_in_source(
                    target_mm, source_mm, region
                )
                if replacement is None:
                    log.error(
                        "  offset %s: could not find matching region in source",
                        f"{region.offset:,}",
                    )
                    target_mm.close()
                    source_mm.close()
                    return False

                log.info(
                    "  offset %s: replacing %s null bytes with %s bytes from source",
                    f"{region.offset:,}",
                    f"{region.length:,}",
                    f"{len(replacement):,}",
                )
                replacements.append((region.offset, region.length, replacement))

            target_mm.close()
            source_mm.close()

    if dry_run:
        log.info("  (dry run, no changes written)")
        return True

    # Apply repairs
    out_path = output or target
    if output is None:
        backup = target.with_suffix(target.suffix + ".bak")
        log.info("  backing up to %s", backup.name)
        shutil.copy2(target, backup)

    data = bytearray(target.read_bytes())

    # Apply null-byte replacements in reverse order so offsets stay valid
    for offset, length, replacement in reversed(replacements):
        if len(replacement) == length:
            data[offset : offset + length] = replacement
        else:
            data[offset : offset + length] = replacement
            log.warning(
                "  replacement length differs (%d vs %d), "
                "index offsets in the output will be invalid. "
                "Consider re-running centrix on this file instead.",
                len(replacement),
                length,
            )

    # Fix spectrum count mismatch
    if needs_count_fix:
        fix_spectrum_count(data)

    out_path.write_bytes(bytes(data))
    log.info("  repaired: %s", out_path.name)

    # Post-repair verification
    post_regions = find_null_regions(out_path)
    if post_regions:
        log.error("  post-repair verification FAILED, null-byte corruption remains")
        return False

    post_mismatch = check_spectrum_counts(out_path)
    if post_mismatch is not None:
        log.warning(
            "  post-repair: count still mismatched (declared=%d, actual=%d, index=%d)",
            post_mismatch.declared_count,
            post_mismatch.actual_spectra,
            post_mismatch.index_entries,
        )

    log.info("  post-repair verification passed")
    return True


def cmd_verify(args: argparse.Namespace) -> int:
    """Run the verify subcommand.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 = all clean, 1 = issues found).
    """
    any_issues = False
    for pattern in args.input:
        paths = sorted(Path(".").glob(pattern)) if "*" in pattern else [Path(pattern)]
        for path in paths:
            if not path.exists():
                log.error("File not found: %s", path)
                any_issues = True
                continue
            regions, mismatch = verify_file(path)
            if regions or mismatch is not None:
                any_issues = True
    return 1 if any_issues else 0


def cmd_repair(args: argparse.Namespace) -> int:
    """Run the repair subcommand.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 = all repaired, 1 = some failures).
    """
    source_dir = Path(args.source_dir) if args.source_dir else None
    if source_dir and not source_dir.is_dir():
        log.error("Source directory not found: %s", source_dir)
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    any_failed = False
    for pattern in args.input:
        paths = sorted(Path(".").glob(pattern)) if "*" in pattern else [Path(pattern)]
        for path in paths:
            if not path.exists():
                log.error("File not found: %s", path)
                any_failed = True
                continue

            # Find source file if source_dir is provided
            source = None
            if source_dir:
                source = find_source_file(path, source_dir)
                if source is None:
                    # Only an error if the file actually has null-byte corruption
                    regions = find_null_regions(path)
                    if regions:
                        log.error(
                            "No matching source file found for %s in %s "
                            "(needed for null-byte repair)",
                            path.name,
                            source_dir,
                        )
                        any_failed = True
                        continue

            out = output_dir / path.name if output_dir else None
            ok = repair_file(path, source, output=out, dry_run=args.dry_run)
            if not ok:
                any_failed = True

    return 1 if any_failed else 0


def main() -> int:
    """Entry point for the mzML verify/repair script.

    Returns:
        Exit code.
    """
    parser = argparse.ArgumentParser(
        description="Verify and repair mzML files for null-byte corruption"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # verify
    p_verify = subparsers.add_parser(
        "verify", help="Check mzML files for null-byte corruption"
    )
    p_verify.add_argument(
        "input", nargs="+", help="mzML file(s) or glob patterns"
    )

    # repair
    p_repair = subparsers.add_parser(
        "repair",
        help="Repair corrupted mzML files using source profiles as reference",
    )
    p_repair.add_argument(
        "input", nargs="+", help="Corrupted mzML file(s) or glob patterns"
    )
    p_repair.add_argument(
        "--source-dir",
        help="Directory containing original source mzML files "
        "(required for null-byte repair, not needed for count-only fixes)",
    )
    p_repair.add_argument(
        "--output-dir",
        help="Output directory for repaired files (default: repair in-place with .bak backup)",
    )
    p_repair.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without writing",
    )

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(levelname)s: %(message)s"
    )

    if args.command == "verify":
        return cmd_verify(args)
    elif args.command == "repair":
        return cmd_repair(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
