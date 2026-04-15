#!/usr/bin/env python3
"""Validate mzML files against HUPO-PSI conformance rules.

This script layers on top of mzml_verify_repair.py and adds semantic checks
inspired by the HUPO-PSI mzML validator:

- XSD schema validation (against bundled mzML 1.1 schemas)
- Required element presence (fileDescription, softwareList, etc.)
- Required cvParam presence per element type (e.g. spectrum MUST have a
  descendant of MS:1000559 "spectrum type" and MS:1000525 "spectrum
  representation")
- Reuses byte-level corruption checks from mzml_verify_repair

Strictness levels:
    minimal  - byte-level checks only (null bytes, count mismatch,
               empty spectra, index integrity)
    standard - minimal + XSD schema + required elements + key cvParams
    strict   - standard + full per-element CV mapping rules for every
               element listed in the HUPO-PSI ms-mapping.xml

The validator prints findings grouped by severity (INFO / WARNING / ERROR)
and returns a non-zero exit code if any ERROR is emitted.

Usage:
    python mzml_validate.py file.mzML
    python mzml_validate.py --level strict *.mzML
    python mzml_validate.py --level minimal --quiet *.mzML
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any

from lxml import etree

# Import byte-level checks from the sibling script
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from mzml_verify_repair import (  # noqa: E402
    check_spectrum_counts,
    find_empty_spectra,
    find_null_regions,
)

log = logging.getLogger(__name__)

DATA_DIR = SCRIPT_DIR / "mzml_validator_data"
CV_DESCENDANTS_PATH = DATA_DIR / "cv_descendants.json"

# Namespace used in mzML files
MZML_NS = "http://psi.hupo.org/ms/mzml"
NS = {"m": MZML_NS}

# Map from mzML version to bare XSD filename (for non-indexed files)
XSD_BY_VERSION = {
    "1.1.0": "mzML1.1.0.xsd",
    "1.1.1": "mzML1.1.1.xsd",
}
# Map from mzML version to indexed XSD filename (for <indexedmzML> wrapper)
INDEXED_XSD_BY_VERSION = {
    "1.1.0": "mzML1.1.0_idx.xsd",
    "1.1.1": "mzML1.1.1_idx.xsd",
}
DEFAULT_XSD = "mzML1.1.1.xsd"
DEFAULT_INDEXED_XSD = "mzML1.1.1_idx.xsd"


class Severity(IntEnum):
    """Finding severity levels."""

    INFO = 0
    WARNING = 1
    ERROR = 2

    def __str__(self) -> str:
        return self.name


@dataclass
class Finding:
    """A single validation finding."""

    severity: Severity
    check: str
    message: str
    location: str = ""


@dataclass
class ValidationReport:
    """Collected findings for one file."""

    path: Path
    findings: list[Finding] = field(default_factory=list)

    def add(
        self,
        severity: Severity,
        check: str,
        message: str,
        location: str = "",
    ) -> None:
        """Append a finding."""
        self.findings.append(Finding(severity, check, message, location))

    @property
    def max_severity(self) -> Severity:
        """Return the highest severity found (default INFO if empty)."""
        if not self.findings:
            return Severity.INFO
        return max(f.severity for f in self.findings)

    def count(self, severity: Severity) -> int:
        """Count findings at a given severity level."""
        return sum(1 for f in self.findings if f.severity == severity)


# ── CV descendants loader ─────────────────────────────────────────────────────


def load_cv_descendants() -> dict[str, Any]:
    """Load the bundled CV descendants lookup table.

    Returns:
        Dict with keys 'parents' and 'term_names' (see build_cv_subset.py).
    """
    if not CV_DESCENDANTS_PATH.exists():
        raise FileNotFoundError(
            f"CV descendants table not found at {CV_DESCENDANTS_PATH}. "
            "Did you run the build_cv_subset.py helper?"
        )
    return json.loads(CV_DESCENDANTS_PATH.read_text())


# ── Byte-level (minimal) checks ───────────────────────────────────────────────


def check_byte_level(path: Path, report: ValidationReport) -> None:
    """Run null-byte, count-mismatch, and empty-spectra checks.

    Args:
        path: mzML file to check.
        report: Report to append findings to.
    """
    regions = find_null_regions(path)
    for r in regions:
        report.add(
            Severity.ERROR,
            "null-byte-corruption",
            f"{r.length:,} consecutive null bytes at offset {r.offset:,} "
            "(likely SMB/NFS write corruption)",
        )

    mismatch = check_spectrum_counts(path)
    if mismatch is not None:
        report.add(
            Severity.ERROR,
            "spectrum-count-mismatch",
            f"spectrumList declares {mismatch.declared_count} but file has "
            f"{mismatch.actual_spectra} spectra ({mismatch.index_entries} index entries)",
        )

    empties = find_empty_spectra(path)
    if empties:
        report.add(
            Severity.WARNING,
            "empty-spectra",
            f"{len(empties)} spectra with defaultArrayLength=\"0\" "
            "(tolerated by DIA-NN if TIC count matches, but may crash other tools)",
        )


# ── Schema (XSD) validation ───────────────────────────────────────────────────


def _detect_mzml_version(path: Path) -> str:
    """Peek at the first few KB to find the mzML version attribute.

    Args:
        path: mzML file to inspect.

    Returns:
        Version string ('1.1.0', '1.1.1', ...) or the default if not found.
    """
    with open(path, "rb") as f:
        head = f.read(16 * 1024)
    import re

    m = re.search(rb'<mzML[^>]*version="([^"]+)"', head)
    if m:
        return m.group(1).decode()
    return "1.1.1"


def _detect_indexed_wrapper(path: Path) -> bool:
    """Check whether the file uses the <indexedmzML> wrapper.

    Args:
        path: mzML file.

    Returns:
        True if the file starts with <indexedmzML>, False otherwise.
    """
    with open(path, "rb") as f:
        head = f.read(4096)
    return b"<indexedmzML" in head


def check_schema(path: Path, report: ValidationReport) -> None:
    """Validate the file against the bundled mzML XSD schema.

    Args:
        path: mzML file.
        report: Report to append findings to.
    """
    version = _detect_mzml_version(path)
    indexed = _detect_indexed_wrapper(path)
    if indexed:
        xsd_name = INDEXED_XSD_BY_VERSION.get(version, DEFAULT_INDEXED_XSD)
    else:
        xsd_name = XSD_BY_VERSION.get(version, DEFAULT_XSD)
    xsd_path = DATA_DIR / xsd_name
    if not xsd_path.exists():
        report.add(
            Severity.WARNING,
            "schema-unavailable",
            f"No bundled XSD for mzML version {version}; skipping schema check",
        )
        return

    log.debug("Validating against %s", xsd_name)
    try:
        xsd_doc = etree.parse(str(xsd_path))
        schema = etree.XMLSchema(xsd_doc)
    except etree.XMLSchemaParseError as e:
        report.add(
            Severity.ERROR,
            "schema-load",
            f"Failed to load schema {xsd_name}: {e}",
        )
        return

    # Parse the input file; capture well-formedness errors too
    try:
        doc = etree.parse(str(path))
    except etree.XMLSyntaxError as e:
        report.add(
            Severity.ERROR,
            "xml-syntax",
            f"XML is not well-formed: {e}",
        )
        return

    # schema.validate populates schema.error_log
    if schema.validate(doc):
        report.add(
            Severity.INFO,
            "schema",
            f"Conforms to {xsd_name}",
        )
    else:
        # Limit to first 20 errors to keep output manageable
        errors = list(schema.error_log)
        report.add(
            Severity.ERROR,
            "schema",
            f"{len(errors)} schema violation(s) against {xsd_name}",
        )
        for err in errors[:20]:
            report.add(
                Severity.ERROR,
                "schema-detail",
                err.message,
                f"line {err.line}",
            )
        if len(errors) > 20:
            report.add(
                Severity.INFO,
                "schema",
                f"... and {len(errors) - 20} more schema errors (suppressed)",
            )


# ── Required element presence ────────────────────────────────────────────────

REQUIRED_ELEMENTS = [
    ("m:mzML", "root mzML element"),
    ("m:mzML/m:cvList", "cvList"),
    ("m:mzML/m:fileDescription", "fileDescription"),
    ("m:mzML/m:fileDescription/m:fileContent", "fileContent"),
    ("m:mzML/m:softwareList", "softwareList"),
    ("m:mzML/m:instrumentConfigurationList", "instrumentConfigurationList"),
    ("m:mzML/m:dataProcessingList", "dataProcessingList"),
    ("m:mzML/m:run", "run"),
    ("m:mzML/m:run/m:spectrumList", "spectrumList"),
]


def check_required_elements(
    doc: etree._ElementTree, report: ValidationReport
) -> None:
    """Check that all structurally required elements are present.

    Args:
        doc: Parsed mzML document.
        report: Report to append findings to.
    """
    root = doc.getroot()

    # The doc root may be <indexedmzML> wrapping <mzML>, or just <mzML>
    if root.tag == f"{{{MZML_NS}}}indexedmzML":
        base_xpath = "m:mzML"
    else:
        base_xpath = "."

    for xpath, label in REQUIRED_ELEMENTS:
        # Rewrite xpath to be relative to the chosen root
        if xpath == "m:mzML":
            full = base_xpath
        else:
            full = xpath.replace("m:mzML", base_xpath, 1)
        found = root.xpath(full, namespaces=NS)
        if not found:
            report.add(
                Severity.ERROR,
                "missing-element",
                f"Required element <{label}> not found",
                full,
            )


# ── Required cvParam presence per element (standard & strict) ─────────────────

# Rules from the HUPO-PSI ms-mapping.xml file. Each rule is:
#   (xpath, [(parent_accession, label, severity), ...])
# Where parent_accession is a CV term whose descendant must appear as a
# cvParam accession on the matched element.
#
# STANDARD rules: the most important MUST rules for DIA-NN / common tools
STANDARD_RULES: list[tuple[str, list[tuple[str, str]]]] = [
    # element XPath, [(parent, label), ...]
    (
        "m:fileDescription/m:fileContent",
        [
            ("MS:1000524", "data file content"),
        ],
    ),
    (
        "m:instrumentConfigurationList/m:instrumentConfiguration",
        [
            ("MS:1000031", "instrument model"),
        ],
    ),
    (
        "m:softwareList/m:software",
        [
            ("MS:1000531", "software"),
        ],
    ),
    (
        "m:dataProcessingList/m:dataProcessing/m:processingMethod",
        [
            ("MS:1000452", "data transformation"),
        ],
    ),
    (
        "m:run/m:spectrumList/m:spectrum",
        [
            ("MS:1000559", "spectrum type"),
            ("MS:1000525", "spectrum representation"),
        ],
    ),
    (
        "m:run/m:spectrumList/m:spectrum/m:binaryDataArrayList/m:binaryDataArray",
        [
            ("MS:1000513", "binary data array"),
            ("MS:1000518", "binary data type"),
            ("MS:1000572", "binary data compression type"),
        ],
    ),
]

# STRICT rules: everything from the HUPO-PSI CV mapping file
STRICT_RULES: list[tuple[str, list[tuple[str, str]]]] = STANDARD_RULES + [
    (
        "m:instrumentConfigurationList/m:instrumentConfiguration/m:componentList/m:source",
        [
            ("MS:1000008", "ionization type"),
        ],
    ),
    (
        "m:instrumentConfigurationList/m:instrumentConfiguration/m:componentList/m:analyzer",
        [
            ("MS:1000443", "mass analyzer type"),
        ],
    ),
    (
        "m:instrumentConfigurationList/m:instrumentConfiguration/m:componentList/m:detector",
        [
            ("MS:1000026", "detector type"),
        ],
    ),
    (
        "m:run/m:spectrumList/m:spectrum/m:scanList",
        [
            ("MS:1000570", "spectra combination"),
        ],
    ),
    (
        "m:run/m:spectrumList/m:spectrum/m:scanList/m:scan/m:scanWindowList/m:scanWindow",
        [
            ("MS:1000500", "scan window upper limit"),
            ("MS:1000501", "scan window lower limit"),
        ],
    ),
    (
        "m:run/m:spectrumList/m:spectrum/m:precursorList/m:precursor/m:selectedIonList/m:selectedIon",
        [
            ("MS:1000455", "ion selection attribute"),
        ],
    ),
    (
        "m:run/m:spectrumList/m:spectrum/m:precursorList/m:precursor/m:activation",
        [
            ("MS:1000044", "dissociation method"),
        ],
    ),
    (
        "m:run/m:chromatogramList/m:chromatogram",
        [
            ("MS:1000626", "chromatogram type"),
        ],
    ),
    (
        "m:run/m:chromatogramList/m:chromatogram/m:binaryDataArrayList/m:binaryDataArray",
        [
            ("MS:1000513", "binary data array"),
            ("MS:1000518", "binary data type"),
            ("MS:1000572", "binary data compression type"),
        ],
    ),
]


def _build_param_group_table(
    doc: etree._ElementTree,
) -> dict[str, set[str]]:
    """Build a map of referenceableParamGroup id -> set of cvParam accessions.

    Args:
        doc: Parsed mzML document.

    Returns:
        Dict mapping group id to set of accessions in that group.
    """
    root = doc.getroot()
    table: dict[str, set[str]] = {}
    for group in root.iter(f"{{{MZML_NS}}}referenceableParamGroup"):
        gid = group.get("id")
        if gid is None:
            continue
        accs = {
            p.get("accession")
            for p in group.findall(f"{{{MZML_NS}}}cvParam")
            if p.get("accession")
        }
        table[gid] = accs
    return table


def _collect_cvparam_accessions(
    element: etree._Element,
    param_groups: dict[str, set[str]] | None = None,
) -> set[str]:
    """Collect cvParam accessions for an element, resolving param group refs.

    Args:
        element: XML element.
        param_groups: Pre-built param group table (optional). If None, only
            direct cvParam children are counted.

    Returns:
        Set of accession strings for direct cvParam children plus any from
        referenced parameter groups.
    """
    accs: set[str] = set()
    for child in element.findall(f"{{{MZML_NS}}}cvParam"):
        acc = child.get("accession")
        if acc:
            accs.add(acc)
    if param_groups is not None:
        for ref in element.findall(f"{{{MZML_NS}}}referenceableParamGroupRef"):
            gid = ref.get("ref")
            if gid and gid in param_groups:
                accs.update(param_groups[gid])
    return accs


def check_cv_rules(
    doc: etree._ElementTree,
    rules: list[tuple[str, list[tuple[str, str]]]],
    cv_data: dict[str, Any],
    report: ValidationReport,
    sample_limit: int = 3,
) -> None:
    """Check that required cvParam ancestors are present on matched elements.

    Args:
        doc: Parsed mzML document.
        rules: List of (xpath, [(parent_accession, label), ...]) tuples.
        cv_data: Loaded CV descendants data.
        report: Report to append findings to.
        sample_limit: When a rule fails on many elements, only report the
            first N as examples and then a summary count.
    """
    root = doc.getroot()
    if root.tag == f"{{{MZML_NS}}}indexedmzML":
        base = "m:mzML"
    else:
        base = "."

    parents = cv_data["parents"]
    param_groups = _build_param_group_table(doc)

    for xpath_rel, required_parents in rules:
        full_xpath = f"{base}/{xpath_rel}"
        elements = root.xpath(full_xpath, namespaces=NS)
        if not elements:
            # No elements matched — not necessarily wrong, but log as INFO
            log.debug("No elements matched %s", full_xpath)
            continue

        for parent_acc, label in required_parents:
            if parent_acc not in parents:
                log.debug("Parent %s not in CV table, skipping", parent_acc)
                continue
            allowed = set(parents[parent_acc]["descendants"])

            failures: list[str] = []
            for elem in elements:
                accs = _collect_cvparam_accessions(elem, param_groups)
                if not accs & allowed:
                    # Build a useful location identifier
                    idref = elem.get("id") or elem.get("index") or ""
                    failures.append(str(idref) if idref else elem.tag)

            if failures:
                loc_summary = ", ".join(failures[:sample_limit])
                if len(failures) > sample_limit:
                    loc_summary += f" (and {len(failures) - sample_limit} more)"
                report.add(
                    Severity.WARNING,
                    "missing-cvparam",
                    f"<{xpath_rel.split('/')[-1].replace('m:', '')}> "
                    f"missing required '{label}' ({parent_acc}) "
                    f"in {len(failures)}/{len(elements)} elements",
                    loc_summary,
                )


# ── Main validation driver ────────────────────────────────────────────────────


def validate_file(
    path: Path,
    level: str,
    cv_data: dict[str, Any] | None = None,
) -> ValidationReport:
    """Validate a single mzML file at the requested strictness level.

    Args:
        path: mzML file to validate.
        level: One of 'minimal', 'standard', 'strict'.
        cv_data: Pre-loaded CV data (optional; loaded on demand otherwise).

    Returns:
        A ValidationReport with all findings.
    """
    report = ValidationReport(path=path)

    if not path.exists():
        report.add(Severity.ERROR, "file", f"File not found: {path}")
        return report

    # 1. Byte-level checks (all levels)
    check_byte_level(path, report)

    # For standard and strict, we need to parse the XML document
    if level in ("standard", "strict"):
        # Schema check
        check_schema(path, report)

        # Parse for element/CV checks (even if schema failed, try anyway)
        try:
            doc = etree.parse(str(path))
        except etree.XMLSyntaxError as e:
            report.add(
                Severity.ERROR,
                "xml-parse",
                f"Could not parse XML for semantic checks: {e}",
            )
            return report

        # Required elements
        check_required_elements(doc, report)

        # CV rules
        if cv_data is None:
            cv_data = load_cv_descendants()
        rules = STANDARD_RULES if level == "standard" else STRICT_RULES
        check_cv_rules(doc, rules, cv_data, report)

    return report


# ── Output formatting ────────────────────────────────────────────────────────


# ANSI color codes (disabled when stdout is not a TTY or when --no-color set)
_COLORS = {
    Severity.INFO: "\033[36m",  # cyan
    Severity.WARNING: "\033[33m",  # yellow
    Severity.ERROR: "\033[31m",  # red
}
_RESET = "\033[0m"


def print_report(
    report: ValidationReport, use_color: bool = True, quiet: bool = False
) -> None:
    """Print a validation report to stdout.

    Args:
        report: The report to print.
        use_color: Whether to use ANSI colors.
        quiet: If True, suppress INFO findings.
    """
    header = f"=== {report.path.name} ==="
    print(header)
    if not report.findings:
        print("  No findings.")
        return

    printed = 0
    for f in report.findings:
        if quiet and f.severity == Severity.INFO:
            continue
        prefix = f"{_COLORS[f.severity]}{f.severity.name:7s}{_RESET}" if use_color else f"{f.severity.name:7s}"
        loc = f" [{f.location}]" if f.location else ""
        print(f"  {prefix} {f.check:25s} {f.message}{loc}")
        printed += 1

    # Summary
    n_err = report.count(Severity.ERROR)
    n_warn = report.count(Severity.WARNING)
    n_info = report.count(Severity.INFO)
    print(
        f"  Summary: {n_err} error(s), {n_warn} warning(s), {n_info} info"
    )


def main() -> int:
    """Entry point for the mzML validator.

    Returns:
        Exit code (0 if no errors, 1 if any error was found).
    """
    parser = argparse.ArgumentParser(
        description="Validate mzML files against HUPO-PSI conformance rules"
    )
    parser.add_argument(
        "input", nargs="+", help="mzML file(s) or glob patterns"
    )
    parser.add_argument(
        "--level",
        choices=["minimal", "standard", "strict"],
        default="standard",
        help="Validation strictness: minimal (byte-level only), "
        "standard (schema + core rules), strict (full CV mapping rules)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress INFO findings in the output",
    )
    parser.add_argument(
        "--no-color", action="store_true", help="Disable ANSI colors"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    use_color = sys.stdout.isatty() and not args.no_color

    # Load CV data once
    try:
        cv_data = load_cv_descendants() if args.level in ("standard", "strict") else None
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    # Expand globs
    paths: list[Path] = []
    for pattern in args.input:
        if "*" in pattern or "?" in pattern or "[" in pattern:
            paths.extend(sorted(Path(".").glob(pattern)))
        else:
            paths.append(Path(pattern))

    if not paths:
        print("No input files.", file=sys.stderr)
        return 2

    any_error = False
    for path in paths:
        report = validate_file(path, args.level, cv_data)
        print_report(report, use_color=use_color, quiet=args.quiet)
        if report.max_severity >= Severity.ERROR:
            any_error = True
        print()

    return 1 if any_error else 0


if __name__ == "__main__":
    sys.exit(main())
