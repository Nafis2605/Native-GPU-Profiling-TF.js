"""
Nsight Systems report parser for native-tfjs-bench.

Parses an ``.nsys-rep`` report into a structured :class:`NsysSummary` by:

  1. Running ``nsys stats`` for each supported report type and parsing the
     CSV output into per-category dataclasses.
  2. Optionally exporting to SQLite via ``nsys export --type sqlite`` and
     querying the timeline for GPU-side duration bounds.

Report types used
-----------------
  gpukernsum      GPU kernel execution summary (kernel name, time, count)
  cudaapisum      CUDA API call summary (API name, time, call count)
  cuda_memcpy_sum CUDA memory transfer summary (direction, bytes, time)

The column names emitted by ``nsys stats`` vary between Nsight Systems versions
(the tool changed its CSV schema multiple times between 2022 and 2025).  The
parser normalises column names defensively and records a warning for any column
it cannot map.

SQLite timeline query
---------------------
When the SQLite export succeeds, the parser queries for:
  * The total GPU-side timeline duration (max(end) − min(start) across all
    kernel events).
  * The number of distinct GPU streams used.

Absence of any individual data source is handled gracefully: the corresponding
field in :class:`NsysSummary` is set to None and a warning is appended to
``parse_warnings``.

Windows notes
-------------
* ``nsys stats`` requires the same nsys executable used for profiling.
* ``nsys export`` may fail if the report was written by a different nsys version
  than the one running the parse.  The parser proceeds without the SQLite file
  in that case and records a warning.
* All subprocesses suppress the Windows console window.
"""

from __future__ import annotations

import csv
import datetime
import io
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Windows: suppress console window
_SUBPROCESS_FLAGS: dict = (
    {"creationflags": subprocess.CREATE_NO_WINDOW}
    if os.name == "nt"
    else {}
)

# Maximum kernel rows to include in the summary (by total GPU time descending)
_MAX_KERNEL_ROWS = 30

# Maximum API-summary rows to include
_MAX_API_ROWS = 40

# CUDA sync APIs used to compute aggregate synchronization overhead
_SYNC_API_NAMES: frozenset[str] = frozenset(
    {
        "cudaDeviceSynchronize",
        "cudaStreamSynchronize",
        "cudaEventSynchronize",
        "cuStreamSynchronize",
        "cuCtxSynchronize",
    }
)

# nsys stats report type identifiers: (nsys_name, fallback_name)
# Some builds call it "cuda_memcpy_sum", others "cuda_memcpy_tm_sum"
_REPORT_TYPES: list[tuple[str, Optional[str]]] = [
    ("gpukernsum", None),
    ("cudaapisum", None),
    ("cuda_memcpy_sum", "cuda_memcpy_tm_sum"),
]

# Tables tried in order when querying the SQLite for kernel timeline
_KERNEL_TABLES: list[str] = [
    "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL",
    "CUPTI_ACTIVITY_KIND_KERNEL",
    "CONCURRENTKERNELINFO",
    "KERNELINFO",
]


# ── Per-row dataclasses ───────────────────────────────────────────────────────

@dataclass
class NsysKernelStat:
    """
    Summary row for a single CUDA kernel from ``nsys stats --report gpukernsum``.

    Fields are populated directly from the CSV; absent columns are None.
    """

    name: str
    time_pct: Optional[float]        # % of total GPU time
    total_time_ns: Optional[int]     # total accumulated kernel time (ns)
    instances: Optional[int]         # launch count
    avg_ns: Optional[float]          # average per-launch time (ns)
    med_ns: Optional[float]          # median per-launch time (ns)
    min_ns: Optional[int]            # minimum per-launch time (ns)
    max_ns: Optional[int]            # maximum per-launch time (ns)
    stddev_ns: Optional[float]       # standard deviation (ns)


@dataclass
class NsysApiStat:
    """
    Summary row for a single CUDA API from ``nsys stats --report cudaapisum``.
    """

    name: str
    time_pct: Optional[float]
    total_time_ns: Optional[int]
    num_calls: Optional[int]
    avg_ns: Optional[float]
    med_ns: Optional[float]
    min_ns: Optional[int]
    max_ns: Optional[int]
    stddev_ns: Optional[float]


@dataclass
class NsysMemcpyStat:
    """
    Summary row for a memory-transfer type from
    ``nsys stats --report cuda_memcpy_sum``.
    """

    operation: str                   # e.g. "HtoD", "DtoH", "DtoD"
    time_pct: Optional[float]
    total_time_ns: Optional[int]
    count: Optional[int]
    avg_ns: Optional[float]
    min_ns: Optional[int]
    max_ns: Optional[int]
    stddev_ns: Optional[float]


# ── Top-level summary dataclass ───────────────────────────────────────────────

@dataclass
class NsysSummary:
    """
    Parsed Nsight Systems report summary for one profile run.

    All numeric fields are None when the underlying data source was unavailable
    (nsys not installed, report corrupted, or nsys version incompatibility).

    Timing conventions
    ------------------
    All ``*_ms`` fields are in **milliseconds**.  The raw nsys CSV reports
    nanoseconds; the parser converts on ingestion.

    Fields
    ------
    report_path : str
        Absolute path to the source ``.nsys-rep`` file.
    gpu_timeline_duration_ms : float | None
        Total GPU-side timeline span: ``(max_kernel_end − min_kernel_start)``
        in milliseconds, from the SQLite export.  None when SQLite export
        is unavailable.
    total_kernel_time_ms : float | None
        Sum of all kernel ``Total Time`` values from ``gpukernsum``.
        This is accumulated time (all kernels × all launches), not elapsed
        wall time.  For sequential execution without overlap it approximates
        the GPU compute time; for concurrent execution it overcounts.
    kernel_count : int
        Number of distinct kernel names appearing in ``gpukernsum``.
    total_kernel_launch_count : int
        Sum of all per-kernel instance counts from ``gpukernsum``.
    top_kernels : list[NsysKernelStat]
        Up to :data:`_MAX_KERNEL_ROWS` rows, sorted by ``total_time_ns``
        descending.
    cuda_api_top : list[NsysApiStat]
        CUDA API call summary, up to :data:`_MAX_API_ROWS` rows, sorted by
        ``total_time_ns`` descending.
    total_api_time_ms : float | None
        Sum of all CUDA API ``Total Time`` values from ``cudaapisum``.
    sync_call_count : int
        Total ``num_calls`` across all recognised synchronisation API names.
    sync_total_time_ms : float | None
        Total time (ms) spent in synchronisation API calls.
    memcpy_rows : list[NsysMemcpyStat]
        All memory-transfer rows from ``cuda_memcpy_sum``.
    total_memcpy_time_ms : float | None
        Sum of memory-transfer ``Total Time`` values.
    gpu_stream_count : int | None
        Number of distinct GPU streams active during the session
        (from SQLite query).  None when SQLite is unavailable.
    nsys_version : str | None
        Version string reported by ``nsys --version``.
    parsed_ok : bool
        False when a fatal error prevented parsing.  Individual missing
        subsections set warnings but do not flip this flag.
    parse_warnings : list[str]
        Non-fatal issues encountered during parsing.
    parse_timestamp_utc : str
        ISO-8601 UTC timestamp when this summary was generated.
    """

    report_path: str = ""

    # Timeline (from SQLite)
    gpu_timeline_duration_ms: Optional[float] = None
    gpu_stream_count: Optional[int] = None

    # Kernel stats (from gpukernsum)
    total_kernel_time_ms: Optional[float] = None
    kernel_count: int = 0
    total_kernel_launch_count: int = 0
    top_kernels: list[NsysKernelStat] = field(default_factory=list)

    # CUDA API stats (from cudaapisum)
    total_api_time_ms: Optional[float] = None
    cuda_api_top: list[NsysApiStat] = field(default_factory=list)

    # Synchronisation derived from API summary
    sync_call_count: int = 0
    sync_total_time_ms: Optional[float] = None

    # Memory copy stats (from cuda_memcpy_sum)
    total_memcpy_time_ms: Optional[float] = None
    memcpy_rows: list[NsysMemcpyStat] = field(default_factory=list)

    # Metadata
    nsys_version: Optional[str] = None
    parsed_ok: bool = True
    parse_warnings: list[str] = field(default_factory=list)
    parse_timestamp_utc: str = field(
        default_factory=lambda: datetime.datetime.utcnow().isoformat() + "Z"
    )

    def to_dict(self) -> dict:
        """Convert to plain dict (safe for JSON serialisation)."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialise to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ── Parser ────────────────────────────────────────────────────────────────────

class NsysParser:
    """
    Parse an Nsight Systems ``.nsys-rep`` report into an :class:`NsysSummary`.

    Parameters
    ----------
    nsys_binary : str | Path | None
        Path to the ``nsys`` executable.  When None the parser searches PATH
        and standard Windows install directories using the same strategy as
        :class:`NsysRunner`.
    stats_timeout_s : int
        Maximum seconds allowed per ``nsys stats`` invocation.  Default 120.
    export_timeout_s : int
        Maximum seconds allowed for ``nsys export --type sqlite``.  Default 180.
    """

    def __init__(
        self,
        nsys_binary: Optional[str | Path] = None,
        stats_timeout_s: int = 120,
        export_timeout_s: int = 180,
    ) -> None:
        self._binary: Optional[Path] = (
            Path(nsys_binary) if nsys_binary is not None else None
        )
        self._stats_timeout = stats_timeout_s
        self._export_timeout = export_timeout_s
        self._resolved: Optional[Path] = None  # cached binary path

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def parse(self, report_path: Path) -> NsysSummary:
        """
        Parse *report_path* and return an :class:`NsysSummary`.

        Always returns a summary object; sets ``parsed_ok=False`` only on
        fatal errors (binary absent, report not found).  Non-fatal issues
        (missing subsections, empty CSV) are recorded in ``parse_warnings``.

        Parameters
        ----------
        report_path : Path
            Path to the ``.nsys-rep`` file to parse.

        Returns
        -------
        NsysSummary
            Populated summary, possibly with some optional fields set to None.
        """
        summary = NsysSummary(report_path=str(report_path.resolve()))
        warnings: list[str] = []

        # --- validate report file ------------------------------------------
        if not report_path.exists():
            summary.parsed_ok = False
            summary.parse_warnings = [
                f"Report file not found: {report_path}"
            ]
            logger.error("nsys report not found: %s", report_path)
            return summary

        # --- locate nsys binary -------------------------------------------
        binary = self._get_binary()
        if binary is None:
            msg = (
                "nsys binary not found; cannot run nsys stats or export. "
                "Install Nsight Systems or set NSYS_EXE."
            )
            warnings.append(msg)
            logger.warning(msg)
            # Partial summary: no data, but not a fatal error for the schema
            summary.parsed_ok = False
            summary.parse_warnings = warnings
            return summary

        summary.nsys_version = self._get_version(binary)
        logger.info(
            "Parsing nsys report | binary=%s | version=%s | report=%s",
            binary,
            summary.nsys_version or "unknown",
            report_path,
        )

        # --- gather stats CSVs -------------------------------------------
        kern_csv = self._run_stats(binary, report_path, "gpukernsum", warnings)
        api_csv = self._run_stats_with_fallback(
            binary, report_path, "cudaapisum", None, warnings
        )
        memcpy_csv = self._run_stats_with_fallback(
            binary, report_path, "cuda_memcpy_sum", "cuda_memcpy_tm_sum", warnings
        )

        # --- parse kernels -----------------------------------------------
        if kern_csv:
            try:
                kernels, total_ns, launch_count = self._parse_kernel_csv(kern_csv)
                summary.top_kernels = kernels[:_MAX_KERNEL_ROWS]
                summary.kernel_count = len(kernels)
                summary.total_kernel_launch_count = launch_count
                if total_ns is not None:
                    summary.total_kernel_time_ms = total_ns / 1_000_000.0
            except Exception as exc:
                warnings.append(f"gpukernsum parse error: {exc}")
                logger.debug("gpukernsum parse error", exc_info=True)
        else:
            warnings.append("gpukernsum CSV not available or empty.")

        # --- parse CUDA API summary --------------------------------------
        if api_csv:
            try:
                api_rows, total_ns = self._parse_api_csv(api_csv)
                summary.cuda_api_top = sorted(
                    api_rows,
                    key=lambda r: r.total_time_ns or 0,
                    reverse=True,
                )[:_MAX_API_ROWS]
                if total_ns is not None:
                    summary.total_api_time_ms = total_ns / 1_000_000.0
                # Derive sync metrics
                sync_rows = [
                    r for r in api_rows if r.name in _SYNC_API_NAMES
                ]
                summary.sync_call_count = sum(
                    (r.num_calls or 0) for r in sync_rows
                )
                sync_ns = sum(
                    (r.total_time_ns or 0) for r in sync_rows
                )
                if sync_rows:
                    summary.sync_total_time_ms = sync_ns / 1_000_000.0
            except Exception as exc:
                warnings.append(f"cudaapisum parse error: {exc}")
                logger.debug("cudaapisum parse error", exc_info=True)
        else:
            warnings.append("cudaapisum CSV not available or empty.")

        # --- parse memcpy -------------------------------------------------
        if memcpy_csv:
            try:
                memcpy_rows, total_ns = self._parse_memcpy_csv(memcpy_csv)
                summary.memcpy_rows = memcpy_rows
                if total_ns is not None:
                    summary.total_memcpy_time_ms = total_ns / 1_000_000.0
            except Exception as exc:
                warnings.append(f"cuda_memcpy parse error: {exc}")
                logger.debug("cuda_memcpy parse error", exc_info=True)
        else:
            warnings.append(
                "cuda_memcpy_sum/cuda_memcpy_tm_sum CSV not available or empty."
            )

        # --- SQLite timeline query ----------------------------------------
        sqlite_path = self._export_sqlite(binary, report_path, warnings)
        if sqlite_path is not None and sqlite_path.exists():
            try:
                duration_ms, stream_count = self._query_sqlite(sqlite_path)
                summary.gpu_timeline_duration_ms = duration_ms
                summary.gpu_stream_count = stream_count
            except Exception as exc:
                warnings.append(f"SQLite timeline query error: {exc}")
                logger.debug("SQLite query error", exc_info=True)

        summary.parse_warnings = warnings
        logger.info(
            "nsys parse complete | kernels=%d | api_rows=%d | warnings=%d",
            summary.kernel_count,
            len(summary.cuda_api_top),
            len(warnings),
        )
        return summary

    # ------------------------------------------------------------------
    # nsys subprocess helpers
    # ------------------------------------------------------------------

    def _get_binary(self) -> Optional[Path]:
        """Return the nsys binary, using the cached value after the first call."""
        if self._resolved is not None:
            return self._resolved
        if self._binary is not None:
            if self._binary.is_file():
                self._resolved = self._binary
                return self._resolved
            else:
                logger.warning(
                    "Explicit nsys_binary %r not found; falling back to auto-detect.",
                    str(self._binary),
                )

        # Mirror NsysRunner._find_nsys_binary() without importing it to avoid
        # circular imports (parser ↔ runner would be circular if runner imports
        # parser; keep parser standalone).
        env_path = os.environ.get("NSYS_EXE")
        if env_path:
            p = Path(env_path)
            if p.is_file():
                self._resolved = p
                return p

        if os.name == "nt":
            prog_files = Path(os.environ.get("PROGRAMFILES", r"C:\Program Files"))
            candidates = sorted(
                prog_files.glob(
                    "NVIDIA Corporation/Nsight Systems*/target-windows-x64/nsys.exe"
                ),
                reverse=True,
            )
            if candidates:
                self._resolved = candidates[0]
                return self._resolved
            candidates_host = sorted(
                prog_files.glob(
                    "NVIDIA Corporation/Nsight Systems*/host-windows-x64/nsys.exe"
                ),
                reverse=True,
            )
            if candidates_host:
                self._resolved = candidates_host[0]
                return self._resolved

        which_result = shutil.which("nsys")
        if which_result:
            self._resolved = Path(which_result)
            return self._resolved

        return None

    def _get_version(self, binary: Path) -> Optional[str]:
        try:
            r = subprocess.run(
                [str(binary), "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                **_SUBPROCESS_FLAGS,
            )
            for line in r.stdout.splitlines():
                if line.strip():
                    return line.strip()
        except Exception:
            pass
        return None

    def _run_stats(
        self,
        binary: Path,
        report: Path,
        report_type: str,
        warnings: list[str],
    ) -> Optional[str]:
        """
        Run ``nsys stats --report <report_type> --format csv`` and return stdout.

        Returns None on any failure (and appends to *warnings*).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write to a temp file; some nsys versions ignore --output -
            out_stem = Path(tmpdir) / "stats_out"
            cmd = [
                str(binary),
                "stats",
                "--report", report_type,
                "--format", "csv",
                "--output", str(out_stem),
                "--force-export=true",  # re-export even if SQLite present
                str(report),
            ]
            logger.debug("nsys stats command: %s", " ".join(cmd))
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self._stats_timeout,
                    **_SUBPROCESS_FLAGS,
                )
            except subprocess.TimeoutExpired:
                warnings.append(
                    f"nsys stats --report {report_type} timed out after "
                    f"{self._stats_timeout} s."
                )
                return None
            except OSError as exc:
                warnings.append(f"nsys stats launch failed: {exc}")
                return None

            if result.returncode != 0:
                warnings.append(
                    f"nsys stats --report {report_type} exited {result.returncode}. "
                    f"stderr: {result.stderr[:200]}"
                )
                return None

            # nsys writes <stem>_<report_type>_<hash>.csv
            csv_candidates = list(Path(tmpdir).glob(f"*{report_type}*.csv"))
            if csv_candidates:
                return csv_candidates[0].read_text(encoding="utf-8", errors="replace")

            # Fallback: some builds write to stdout
            if result.stdout.strip():
                return result.stdout

            warnings.append(
                f"nsys stats --report {report_type}: no CSV output found."
            )
            return None

    def _run_stats_with_fallback(
        self,
        binary: Path,
        report: Path,
        primary: str,
        fallback: Optional[str],
        warnings: list[str],
    ) -> Optional[str]:
        """Try *primary* report type; if it fails try *fallback*."""
        result = self._run_stats(binary, report, primary, [])
        if result:
            return result
        if fallback:
            logger.debug(
                "Primary report type %r returned nothing; trying fallback %r",
                primary,
                fallback,
            )
            return self._run_stats(binary, report, fallback, warnings)
        warnings.append(f"nsys stats --report {primary}: no data returned.")
        return None

    def _export_sqlite(
        self,
        binary: Path,
        report: Path,
        warnings: list[str],
    ) -> Optional[Path]:
        """
        Export the report to SQLite via ``nsys export --type sqlite``.

        The SQLite file is written adjacent to the ``.nsys-rep`` with the
        same stem and a ``.sqlite`` extension.

        Returns the Path to the SQLite file, or None on failure.
        """
        sqlite_path = report.with_suffix(".sqlite")
        if sqlite_path.exists():
            logger.debug("SQLite already exists, reusing: %s", sqlite_path)
            return sqlite_path

        cmd = [
            str(binary),
            "export",
            "--type", "sqlite",
            "--output", str(sqlite_path),
            "--force-overwrite",
            str(report),
        ]
        logger.debug("nsys export command: %s", " ".join(cmd))
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._export_timeout,
                **_SUBPROCESS_FLAGS,
            )
        except subprocess.TimeoutExpired:
            warnings.append(
                f"nsys export timed out after {self._export_timeout} s."
            )
            return None
        except OSError as exc:
            warnings.append(f"nsys export launch failed: {exc}")
            return None

        if result.returncode != 0:
            warnings.append(
                f"nsys export exited {result.returncode}. "
                f"stderr: {result.stderr[:200]}. "
                "SQLite timeline metrics will be unavailable."
            )
            return None

        if sqlite_path.exists():
            logger.debug("SQLite export successful: %s", sqlite_path)
            return sqlite_path

        warnings.append(
            f"nsys export exited 0 but SQLite file not found at {sqlite_path}."
        )
        return None

    # ------------------------------------------------------------------
    # SQLite query
    # ------------------------------------------------------------------

    def _query_sqlite(
        self, sqlite_path: Path
    ) -> tuple[Optional[float], Optional[int]]:
        """
        Query the SQLite database for GPU timeline duration and stream count.

        Returns
        -------
        (duration_ms, stream_count) : tuple
            duration_ms  : float | None — timeline span in milliseconds
            stream_count : int | None   — number of distinct CUDA streams
        """
        try:
            import sqlite3
        except ImportError:
            return None, None

        duration_ms: Optional[float] = None
        stream_count: Optional[int] = None

        with sqlite3.connect(str(sqlite_path)) as conn:
            cursor = conn.cursor()

            # List all tables for debugging unknown schemas
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = {row[0] for row in cursor.fetchall()}
            logger.debug("SQLite tables: %s", sorted(tables))

            # Try each known kernel table for timeline duration
            for table in _KERNEL_TABLES:
                if table not in tables:
                    continue
                try:
                    cursor.execute(
                        f"SELECT MIN(start), MAX(end) FROM {table}"  # noqa: S608
                    )
                    row = cursor.fetchone()
                    if row and row[0] is not None and row[1] is not None:
                        duration_ms = (row[1] - row[0]) / 1_000_000.0
                        logger.debug(
                            "Timeline from %s: %.2f ms", table, duration_ms
                        )
                        break
                except sqlite3.OperationalError:
                    # Table exists but column names differ
                    continue

            # Stream count: try CUDA_EVENTS or the kernel table
            for table in _KERNEL_TABLES:
                if table not in tables:
                    continue
                # Check if a 'streamId' or 'stream' column exists
                try:
                    cursor.execute(f"PRAGMA table_info({table})")  # noqa: S608
                    col_names = {row[1].lower() for row in cursor.fetchall()}
                    stream_col = None
                    for candidate in ("streamid", "stream", "stream_id"):
                        if candidate in col_names:
                            stream_col = candidate
                            break
                    if stream_col:
                        cursor.execute(
                            f"SELECT COUNT(DISTINCT {stream_col}) FROM {table}"  # noqa: S608
                        )
                        sc_row = cursor.fetchone()
                        if sc_row and sc_row[0] is not None:
                            stream_count = int(sc_row[0])
                            logger.debug(
                                "Stream count from %s.%s: %d",
                                table,
                                stream_col,
                                stream_count,
                            )
                            break
                except sqlite3.OperationalError:
                    continue

        return duration_ms, stream_count

    # ------------------------------------------------------------------
    # CSV parsers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_col(name: str) -> str:
        """
        Normalise a CSV column header for fuzzy matching.

        Strips quotes, lowercases, and replaces spaces / punctuation with
        underscores so columns such as "Total Time (ns)" and "total_time_ns"
        both map to "total_time_ns".
        """
        name = name.strip().strip('"').lower()
        # Remove parenthesised unit suffixes: "(ns)", "(%)", "(mb)" → ""
        name = re.sub(r"\s*\([^)]*\)", "", name)
        # Replace runs of whitespace and punctuation with underscore
        name = re.sub(r"[\s\-./]+", "_", name)
        name = re.sub(r"[^a-z0-9_]", "", name)
        return name.strip("_")

    @staticmethod
    def _find_col(
        normalised_headers: list[str],
        *candidates: str,
    ) -> Optional[int]:
        """
        Return the index of the first *candidates* that appears in
        *normalised_headers*, or None.
        """
        for candidate in candidates:
            try:
                return normalised_headers.index(candidate)
            except ValueError:
                pass
        return None

    @classmethod
    def _read_csv_rows(cls, csv_text: str) -> tuple[list[str], list[list[str]]]:
        """
        Parse CSV text, skipping non-table preamble lines.

        nsys stats output occasionally prepends status lines like
        "Generating report..." before the CSV data.  We detect the header
        by looking for the first line that contains 'time' or 'name'
        (case-insensitive).

        Returns
        -------
        (header, rows) : tuple
            header : list[str]   normalised column names
            rows   : list[list[str]]  data rows (may be empty)
        """
        lines = csv_text.splitlines()

        # Find the CSV header line: must have at least 2 comma-separated parts
        # and contain a recognisable column keyword
        header_idx = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split(",")
            if len(parts) >= 2:
                lower = stripped.lower()
                if any(k in lower for k in ("time", "name", "instances", "calls")):
                    header_idx = i
                    break

        if header_idx is None:
            return [], []

        header_raw = lines[header_idx].strip()
        data_text = "\n".join(lines[header_idx:])
        reader = csv.reader(io.StringIO(data_text))
        all_rows = list(reader)
        if not all_rows:
            return [], []

        header = [cls._normalise_col(h) for h in all_rows[0]]
        data_rows = [row for row in all_rows[1:] if any(c.strip() for c in row)]
        return header, data_rows

    @classmethod
    def _safe_int(cls, value: str) -> Optional[int]:
        try:
            return int(float(value.strip().replace(",", "")))
        except (ValueError, AttributeError):
            return None

    @classmethod
    def _safe_float(cls, value: str) -> Optional[float]:
        try:
            return float(value.strip().replace(",", ""))
        except (ValueError, AttributeError):
            return None

    def _parse_kernel_csv(
        self, csv_text: str
    ) -> tuple[list[NsysKernelStat], Optional[int], int]:
        """
        Parse ``gpukernsum`` CSV.

        Returns
        -------
        (kernels, total_time_ns, total_launch_count)
            kernels            : sorted by total_time_ns descending
            total_time_ns      : sum of all kernel total_time_ns values
            total_launch_count : sum of all instance counts
        """
        header, rows = self._read_csv_rows(csv_text)
        if not header or not rows:
            return [], None, 0

        h = header  # shorthand
        i_pct = self._find_col(h, "time_pct", "time")
        i_total = self._find_col(h, "total_time_ns", "total_time", "total")
        i_inst = self._find_col(h, "instances", "num_calls", "count")
        i_avg = self._find_col(h, "avg_ns", "avg")
        i_med = self._find_col(h, "med_ns", "med", "median")
        i_min = self._find_col(h, "min_ns", "min")
        i_max = self._find_col(h, "max_ns", "max")
        i_std = self._find_col(h, "stddev_ns", "stddev", "std_dev")
        i_name = self._find_col(h, "name", "kernel_name", "function_name")

        def _get(row: list[str], idx: Optional[int]) -> str:
            return row[idx].strip() if idx is not None and idx < len(row) else ""

        kernels: list[NsysKernelStat] = []
        sum_ns = 0
        sum_launches = 0

        for row in rows:
            if not row:
                continue
            ttl = _get(row, i_total)
            ttl_int = self._safe_int(ttl)
            if ttl_int is not None:
                sum_ns += ttl_int
            inst = _get(row, i_inst)
            inst_int = self._safe_int(inst)
            if inst_int is not None:
                sum_launches += inst_int

            kernels.append(
                NsysKernelStat(
                    name=_get(row, i_name) or "<unknown>",
                    time_pct=self._safe_float(_get(row, i_pct)),
                    total_time_ns=ttl_int,
                    instances=inst_int,
                    avg_ns=self._safe_float(_get(row, i_avg)),
                    med_ns=self._safe_float(_get(row, i_med)),
                    min_ns=self._safe_int(_get(row, i_min)),
                    max_ns=self._safe_int(_get(row, i_max)),
                    stddev_ns=self._safe_float(_get(row, i_std)),
                )
            )

        kernels.sort(key=lambda k: k.total_time_ns or 0, reverse=True)
        return kernels, sum_ns if sum_ns else None, sum_launches

    def _parse_api_csv(
        self, csv_text: str
    ) -> tuple[list[NsysApiStat], Optional[int]]:
        """
        Parse ``cudaapisum`` CSV.

        Returns (api_rows, total_time_ns).
        """
        header, rows = self._read_csv_rows(csv_text)
        if not header or not rows:
            return [], None

        h = header
        i_pct = self._find_col(h, "time_pct", "time")
        i_total = self._find_col(h, "total_time_ns", "total_time", "total")
        i_calls = self._find_col(h, "num_calls", "calls", "count", "instances")
        i_avg = self._find_col(h, "avg_ns", "avg")
        i_med = self._find_col(h, "med_ns", "med", "median")
        i_min = self._find_col(h, "min_ns", "min")
        i_max = self._find_col(h, "max_ns", "max")
        i_std = self._find_col(h, "stddev_ns", "stddev", "std_dev")
        i_name = self._find_col(h, "name", "api_name", "function")

        def _get(row: list[str], idx: Optional[int]) -> str:
            return row[idx].strip() if idx is not None and idx < len(row) else ""

        api_rows: list[NsysApiStat] = []
        sum_ns = 0

        for row in rows:
            if not row:
                continue
            ttl = _get(row, i_total)
            ttl_int = self._safe_int(ttl)
            if ttl_int is not None:
                sum_ns += ttl_int

            api_rows.append(
                NsysApiStat(
                    name=_get(row, i_name) or "<unknown>",
                    time_pct=self._safe_float(_get(row, i_pct)),
                    total_time_ns=ttl_int,
                    num_calls=self._safe_int(_get(row, i_calls)),
                    avg_ns=self._safe_float(_get(row, i_avg)),
                    med_ns=self._safe_float(_get(row, i_med)),
                    min_ns=self._safe_int(_get(row, i_min)),
                    max_ns=self._safe_int(_get(row, i_max)),
                    stddev_ns=self._safe_float(_get(row, i_std)),
                )
            )

        return api_rows, sum_ns if sum_ns else None

    def _parse_memcpy_csv(
        self, csv_text: str
    ) -> tuple[list[NsysMemcpyStat], Optional[int]]:
        """
        Parse ``cuda_memcpy_sum`` / ``cuda_memcpy_tm_sum`` CSV.

        Returns (memcpy_rows, total_time_ns).
        """
        header, rows = self._read_csv_rows(csv_text)
        if not header or not rows:
            return [], None

        h = header
        i_pct = self._find_col(h, "time_pct", "time")
        i_total = self._find_col(h, "total_time_ns", "total_time", "total")
        i_count = self._find_col(h, "count", "num_calls", "instances")
        i_avg = self._find_col(h, "avg_ns", "avg")
        i_min = self._find_col(h, "min_ns", "min")
        i_max = self._find_col(h, "max_ns", "max")
        i_std = self._find_col(h, "stddev_ns", "stddev", "std_dev")
        # The operation column may be called "operations", "operation", "kind"
        i_op = self._find_col(h, "operations", "operation", "kind", "name")

        def _get(row: list[str], idx: Optional[int]) -> str:
            return row[idx].strip() if idx is not None and idx < len(row) else ""

        memcpy_rows: list[NsysMemcpyStat] = []
        sum_ns = 0

        for row in rows:
            if not row:
                continue
            ttl = _get(row, i_total)
            ttl_int = self._safe_int(ttl)
            if ttl_int is not None:
                sum_ns += ttl_int

            memcpy_rows.append(
                NsysMemcpyStat(
                    operation=_get(row, i_op) or "<unknown>",
                    time_pct=self._safe_float(_get(row, i_pct)),
                    total_time_ns=ttl_int,
                    count=self._safe_int(_get(row, i_count)),
                    avg_ns=self._safe_float(_get(row, i_avg)),
                    min_ns=self._safe_int(_get(row, i_min)),
                    max_ns=self._safe_int(_get(row, i_max)),
                    stddev_ns=self._safe_float(_get(row, i_std)),
                )
            )

        return memcpy_rows, sum_ns if sum_ns else None
