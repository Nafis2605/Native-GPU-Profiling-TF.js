"""
Nsight Compute CSV parser for native-tfjs-bench.

Parses the CSV metric dump produced by ``ncu --csv --quiet`` and converts
it into structured Python objects.

ncu CSV format (stdout from ``ncu --csv``)
------------------------------------------
Each row describes one metric for one kernel launch::

    "ID","Process ID","Process Name","Host Name","Kernel Name","Kernel Time",
    "Context","Stream","Section Name","Metric Name","Metric Unit","Metric Value"

    "1","1234","python.exe","HOST","ampere_sgemm_128x64_nt",
    "01/01/2024 00:00:00 UTC","1","7",
    "GPU Speed Of Light Throughput",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed","%","72.34"

Multiple rows share the same "ID" value for the same kernel launch.  The
parser groups these into one NcuKernelMetrics per (ID, Kernel Name) pair.

Handling unsupported metrics
-----------------------------
Metrics not supported on the current GPU architecture appear as "N/A" or
as an empty string in the "Metric Value" column.  The parser records these
as ``NcuMetricValue(value=None, unit="", raw_value="N/A")`` so callers can
distinguish "not measured" from "measured as zero".

Input sources
-------------
NcuParser accepts either:
  * A Path to a file containing the ncu stdout (saved by
    NcuRunResult.save_subprocess_logs).
  * A raw string containing the CSV text (e.g. NcuRunResult.stdout).
"""

from __future__ import annotations

import csv
import io
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from benchmark.profilers.ncu_runner import MetricCategory, categorize_metric

logger = logging.getLogger(__name__)

# Column names expected in the ncu CSV header (case-insensitive match)
_COL_ID            = "id"
_COL_KERNEL_NAME   = "kernel name"
_COL_KERNEL_TIME   = "kernel time"
_COL_CONTEXT       = "context"
_COL_STREAM        = "stream"
_COL_SECTION_NAME  = "section name"
_COL_METRIC_NAME   = "metric name"
_COL_METRIC_UNIT   = "metric unit"
_COL_METRIC_VALUE  = "metric value"

# Values treated as "metric not available on this GPU"
_NA_STRINGS: frozenset[str] = frozenset({"n/a", "na", "", "-", "not available"})


# ── NcuMetricValue ────────────────────────────────────────────────────────────

@dataclass
class NcuMetricValue:
    """
    Parsed value of one ncu metric for one kernel launch.

    Attributes
    ----------
    value : float | None
        Numeric value, or None when the metric is unsupported (N/A) or
        could not be parsed as a float.
    unit : str
        Unit string from the CSV, e.g. `"%"`, `"nsecond"`, `"inst"`.
    raw_value : str
        Original unmodified string from the CSV (for diagnostics).
    """

    value: Optional[float]
    unit: str
    raw_value: str

    def is_available(self) -> bool:
        """Return True when the metric was measured (value is not None)."""
        return self.value is not None


# ── NcuKernelMetrics ──────────────────────────────────────────────────────────

@dataclass
class NcuKernelMetrics:
    """
    All collected metrics for one kernel launch.

    Attributes
    ----------
    launch_id : int
        1-based kernel launch counter from ncu's "ID" column.
    kernel_name : str
        Demangled kernel function name.
    stream_id : str
        CUDA stream ID string.
    context_id : str
        CUDA context ID string.
    kernel_time_str : str
        Human-readable timestamp from the "Kernel Time" column.
    metrics : dict[str, NcuMetricValue]
        Mapping of metric_name → NcuMetricValue.
    """

    launch_id: int
    kernel_name: str
    stream_id: str = ""
    context_id: str = ""
    kernel_time_str: str = ""
    metrics: dict[str, NcuMetricValue] = field(default_factory=dict)

    def get_metric(self, name: str) -> Optional[NcuMetricValue]:
        """Return the NcuMetricValue for *name*, or None if not recorded."""
        return self.metrics.get(name)

    def get_value(self, name: str) -> Optional[float]:
        """Return the numeric value for *name*, or None if absent / N/A."""
        mv = self.metrics.get(name)
        return mv.value if mv is not None else None

    def metrics_by_category(self) -> dict[str, dict[str, Optional[float]]]:
        """
        Return metrics grouped by MetricCategory.

        Returns a dict of category → {metric_name: value_or_None}.
        """
        groups: dict[str, dict[str, Optional[float]]] = {}
        for m_name, m_val in self.metrics.items():
            cat = categorize_metric(m_name)
            groups.setdefault(cat, {})[m_name] = m_val.value
        return groups

    def to_dict(self) -> dict:
        """Convert to plain dict; safe for JSON serialisation."""
        return {
            "launch_id": self.launch_id,
            "kernel_name": self.kernel_name,
            "stream_id": self.stream_id,
            "context_id": self.context_id,
            "kernel_time_str": self.kernel_time_str,
            "metrics": {
                k: {
                    "value": v.value,
                    "unit": v.unit,
                    "raw_value": v.raw_value,
                }
                for k, v in self.metrics.items()
            },
        }


# ── NcuProfilingResult ────────────────────────────────────────────────────────

@dataclass
class NcuProfilingResult:
    """
    Top-level result from parsing one ncu CSV output.

    Attributes
    ----------
    report_path : Path | None
        Path to the source ``.ncu-rep`` binary report (informational).
    profiler_tool : str
        Always ``"ncu"``.
    profiler_mode : str
        Always ``RunMode.PROFILE_NCU`` (``"profile_ncu"``).
    ncu_version : str
        Version string extracted from the profiler run metadata.
    kernels : list[NcuKernelMetrics]
        One entry per profiled kernel launch, in the order ncu captures them.
    parsed_ok : bool
        True when parsing completed without fatal errors.  Individual metric
        parse failures set warnings but do not flip this to False.
    parse_warnings : list[str]
        Non-fatal issues encountered during parsing.
    parse_timestamp_utc : str
        ISO-8601 UTC timestamp of when NcuParser.parse() was called.
    """

    report_path: Optional[Path]
    profiler_tool: str = "ncu"
    profiler_mode: str = "profile_ncu"
    ncu_version: str = ""
    kernels: list[NcuKernelMetrics] = field(default_factory=list)
    parsed_ok: bool = False
    parse_warnings: list[str] = field(default_factory=list)
    parse_timestamp_utc: str = ""

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def kernel_names(self) -> list[str]:
        """Return a deduplicated list of observed kernel names (insertion order)."""
        seen: dict[str, None] = {}
        for k in self.kernels:
            seen[k.kernel_name] = None
        return list(seen)

    def kernels_for_name(self, name_fragment: str) -> list[NcuKernelMetrics]:
        """Return all launches whose kernel name contains *name_fragment*."""
        return [k for k in self.kernels if name_fragment in k.kernel_name]

    def metrics_by_category_summary(
        self,
    ) -> dict[str, dict[str, list[Optional[float]]]]:
        """
        Aggregate metric values across all kernel launches, grouped by category.

        Returns
        -------
        dict[category, dict[metric_name, list[values_or_None]]]
            For each metric, one value per kernel launch.
        """
        summary: dict[str, dict[str, list[Optional[float]]]] = {}
        for kernel in self.kernels:
            for cat, metrics in kernel.metrics_by_category().items():
                cat_dict = summary.setdefault(cat, {})
                for m_name, m_val in metrics.items():
                    cat_dict.setdefault(m_name, []).append(m_val)
        return summary

    def to_dict(self) -> dict:
        """Convert to plain dict; safe for JSON serialisation."""
        return {
            "report_path": str(self.report_path) if self.report_path else None,
            "profiler_tool": self.profiler_tool,
            "profiler_mode": self.profiler_mode,
            "ncu_version": self.ncu_version,
            "kernel_count": len(self.kernels),
            "kernel_names": self.kernel_names(),
            "kernels": [k.to_dict() for k in self.kernels],
            "parsed_ok": self.parsed_ok,
            "parse_warnings": self.parse_warnings,
            "parse_timestamp_utc": self.parse_timestamp_utc,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialise to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ── NcuParser ─────────────────────────────────────────────────────────────────

class NcuParser:
    """
    Parse the CSV metric dump from ``ncu --csv --quiet``.

    Primary interface
    -----------------
    ::

        parser = NcuParser()

        # From a file (saved ncu stdout log)
        result = parser.parse(Path("results/ncu_profile_stdout.log"))

        # From an in-memory string (NcuRunResult.stdout)
        result = parser.parse_text(run_result.stdout, report_path=report_path)

    The returned NcuProfilingResult contains one NcuKernelMetrics per kernel
    launch spotted in the CSV.  Metrics not present in the CSV for a given
    kernel are simply absent from that kernel's .metrics dict.
    """

    def parse(
        self,
        source: Path,
        ncu_version: str = "",
    ) -> NcuProfilingResult:
        """
        Parse ncu CSV output from a log file.

        Parameters
        ----------
        source : Path
            Path to the file containing the ncu stdout CSV dump.
        ncu_version : str
            Version string (optional; embedded in the result for traceability).

        Returns
        -------
        NcuProfilingResult
            Parsed result.  ``parsed_ok`` is False when the file cannot be
            read or contains no valid CSV rows.
        """
        warnings: list[str] = []
        ts = datetime.now(tz=timezone.utc).isoformat()

        try:
            text = source.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return NcuProfilingResult(
                report_path=None,
                ncu_version=ncu_version,
                parsed_ok=False,
                parse_warnings=[f"Cannot read file {source}: {exc}"],
                parse_timestamp_utc=ts,
            )

        return self._parse_impl(
            csv_text=text,
            report_path=source,
            ncu_version=ncu_version,
            warnings=warnings,
            ts=ts,
        )

    def parse_text(
        self,
        csv_text: str,
        report_path: Optional[Path] = None,
        ncu_version: str = "",
    ) -> NcuProfilingResult:
        """
        Parse ncu CSV output from an in-memory string.

        Parameters
        ----------
        csv_text : str
            Text content of the ncu stdout CSV dump.
        report_path : Path | None
            Path to the associated ``.ncu-rep`` report (informational only).
        ncu_version : str
            Version string to embed in the result.

        Returns
        -------
        NcuProfilingResult
            Parsed result.
        """
        ts = datetime.now(tz=timezone.utc).isoformat()
        return self._parse_impl(
            csv_text=csv_text,
            report_path=report_path,
            ncu_version=ncu_version,
            warnings=[],
            ts=ts,
        )

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _parse_impl(
        self,
        csv_text: str,
        report_path: Optional[Path],
        ncu_version: str,
        warnings: list[str],
        ts: str,
    ) -> NcuProfilingResult:
        """Core parsing logic shared by parse() and parse_text()."""
        if not csv_text or not csv_text.strip():
            return NcuProfilingResult(
                report_path=report_path,
                ncu_version=ncu_version,
                parsed_ok=False,
                parse_warnings=["Empty CSV input — ncu may not have produced any output"],
                parse_timestamp_utc=ts,
            )

        rows, col_map, hdr_warnings = self._read_csv(csv_text)
        warnings.extend(hdr_warnings)

        if not col_map:
            return NcuProfilingResult(
                report_path=report_path,
                ncu_version=ncu_version,
                parsed_ok=False,
                parse_warnings=warnings + ["Could not locate CSV header row"],
                parse_timestamp_utc=ts,
            )

        if not rows:
            warnings.append("CSV header found but no data rows present")
            return NcuProfilingResult(
                report_path=report_path,
                ncu_version=ncu_version,
                parsed_ok=False,
                parse_warnings=warnings,
                parse_timestamp_utc=ts,
            )

        kernels, row_warnings = self._build_kernel_map(rows, col_map)
        warnings.extend(row_warnings)

        logger.info(
            "ncu CSV parsed: %d kernel launch(es), %d parse warnings",
            len(kernels),
            len(warnings),
        )

        return NcuProfilingResult(
            report_path=report_path,
            ncu_version=ncu_version,
            kernels=list(kernels.values()),
            parsed_ok=True,
            parse_warnings=warnings,
            parse_timestamp_utc=ts,
        )

    def _read_csv(
        self, text: str
    ) -> tuple[list[list[str]], dict[str, int], list[str]]:
        """
        Find the CSV header and return (data_rows, col_index_map, warnings).

        ncu sometimes emits non-CSV lines before the header (e.g. progress
        messages that survive ``--quiet``).  We scan for the first line that
        contains all expected column keywords.

        Returns
        -------
        rows : list[list[str]]
            Parsed data rows (not including the header).
        col_map : dict[str, int]
            Mapping of lowercase column name → 0-based column index.
        warnings : list[str]
        """
        warnings: list[str] = []
        lines = text.splitlines()

        header_idx: Optional[int] = None
        required_cols = {_COL_METRIC_NAME, _COL_METRIC_VALUE, _COL_KERNEL_NAME}

        for i, line in enumerate(lines):
            line_lower = line.lower()
            if all(col in line_lower for col in required_cols):
                header_idx = i
                break

        if header_idx is None:
            return [], {}, warnings

        # Parse from header row onward using csv.reader
        csv_block = "\n".join(lines[header_idx:])
        reader = csv.reader(io.StringIO(csv_block))

        try:
            raw_header = next(reader)
        except StopIteration:
            return [], {}, ["CSV block is empty after header line"]

        col_map = {col.strip().lower(): idx for idx, col in enumerate(raw_header)}

        if not required_cols.issubset(col_map):
            missing = required_cols - set(col_map)
            warnings.append(f"CSV header missing expected columns: {missing}")
            return [], col_map, warnings

        rows = list(reader)
        # Filter out blank / comment lines (artifacts of --quiet residue)
        rows = [r for r in rows if r and r[0].strip() and r[0].strip()[:2] != "=="]

        return rows, col_map, warnings

    def _build_kernel_map(
        self,
        rows: list[list[str]],
        col_map: dict[str, int],
    ) -> tuple[dict[tuple, NcuKernelMetrics], list[str]]:
        """
        Group CSV rows by (launch_id, kernel_name) and build NcuKernelMetrics.

        Returns
        -------
        kernel_map : dict[(launch_id, kernel_name), NcuKernelMetrics]
        warnings : list[str]
        """
        warnings: list[str] = []
        kernel_map: dict[tuple, NcuKernelMetrics] = {}

        def _get(row: list[str], col_name: str, default: str = "") -> str:
            idx = col_map.get(col_name)
            if idx is None or idx >= len(row):
                return default
            return row[idx].strip()

        for row_num, row in enumerate(rows, start=1):
            if not row:
                continue

            raw_id      = _get(row, _COL_ID, "0")
            kernel_name = _get(row, _COL_KERNEL_NAME)
            stream_id   = _get(row, _COL_STREAM)
            context_id  = _get(row, _COL_CONTEXT)
            kern_time   = _get(row, _COL_KERNEL_TIME)
            metric_name = _get(row, _COL_METRIC_NAME)
            metric_unit = _get(row, _COL_METRIC_UNIT)
            metric_raw  = _get(row, _COL_METRIC_VALUE)

            if not kernel_name or not metric_name:
                # Rows without a kernel name or metric name are noise
                continue

            # Parse launch ID (ncu uses double-quoted integers)
            try:
                launch_id = int(raw_id)
            except ValueError:
                launch_id = 0

            key = (launch_id, kernel_name)
            if key not in kernel_map:
                kernel_map[key] = NcuKernelMetrics(
                    launch_id=launch_id,
                    kernel_name=kernel_name,
                    stream_id=stream_id,
                    context_id=context_id,
                    kernel_time_str=kern_time,
                )

            kernel_map[key].metrics[metric_name] = NcuMetricValue(
                value=self._parse_float(metric_raw, metric_name, row_num, warnings),
                unit=metric_unit,
                raw_value=metric_raw,
            )

        return kernel_map, warnings

    @staticmethod
    def _parse_float(
        raw: str,
        metric_name: str,
        row_num: int,
        warnings: list[str],
    ) -> Optional[float]:
        """
        Parse *raw* as a float.

        Returns None for N/A values and logs a warning for unexpected
        parse failures.
        """
        stripped = raw.strip()
        if stripped.lower() in _NA_STRINGS:
            return None
        # ncu may use comma as thousand separator in some locales
        cleaned = stripped.replace(",", "")
        try:
            return float(cleaned)
        except ValueError:
            warnings.append(
                f"Row {row_num}: cannot parse metric {metric_name!r} "
                f"value {raw!r} as float"
            )
            return None
