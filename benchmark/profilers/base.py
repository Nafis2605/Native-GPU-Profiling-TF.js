"""
Profiler abstraction layer for native-tfjs-bench.

Defines:
  RunMode      – String constants for benchmark vs profiler execution modes.
  ProfilerBase – Abstract base class that all profiler backends must implement.

Design notes
------------
RunMode is implemented as a plain class with string class constants rather than
an enum.  This keeps values serialisable to JSON without special handling and
avoids importing enum in downstream files that process result dicts.

ProfilerBase is intentionally minimal: it only mandates the interface contract,
not any buffering or subprocess logic (that lives in each concrete subclass).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ── RunMode ───────────────────────────────────────────────────────────────────

class RunMode:
    """
    Execution mode constants.

    CLEAN_BENCHMARK
        Normal benchmark execution.  No profiler is attached.  nvidia-smi
        polling is the only permitted third-party instrumentation.  All
        latency numbers produced in this mode are valid for reporting.

    PROFILE_NSYS
        Nsight Systems profiling execution.  The benchmark subprocess is
        launched under ``nsys profile``.  Wall-clock and CUDA-event timings
        produced in this mode are biased by ~5–15 % trace-buffer overhead and
        MUST NOT be reported as benchmark latency.  nvidia-smi polling is
        suppressed to avoid compounding overhead.
    """

    CLEAN_BENCHMARK: str = "clean_benchmark"
    PROFILE_NSYS: str = "profile_nsys"
    PROFILE_NCU: str = "profile_ncu"

    _VALID: frozenset[str] = frozenset({CLEAN_BENCHMARK, PROFILE_NSYS, PROFILE_NCU})

    @classmethod
    def validate(cls, value: str) -> str:
        """
        Return *value* unchanged if it is a known RunMode constant.

        Raises
        ------
        ValueError
            If *value* is not a recognised mode string.
        """
        if value not in cls._VALID:
            raise ValueError(
                f"Unknown RunMode {value!r}. "
                f"Valid values: {sorted(cls._VALID)}"
            )
        return value

    @classmethod
    def is_profiling(cls, value: str) -> bool:
        """Return True when *value* indicates any profiler-attached run."""
        return value in (cls.PROFILE_NSYS, cls.PROFILE_NCU)


# ── ProfilerBase ──────────────────────────────────────────────────────────────

class ProfilerBase(ABC):
    """
    Abstract base class for profiler backends.

    Each Backend is responsible for:
      1. Detecting whether its underlying tool is installed (``is_available``).
      2. Wrapping a target command so it runs under the profiler
         (``build_profiled_command``).
      3. Collecting artefact metadata after a run (``collect_artifact_metadata``).

    Concrete subclasses (e.g. NsysRunner) handle subprocess management and
    output file naming internally.  This base class does **not** manage
    subprocess lifecycle — it only defines the interface contract.

    Parameters
    ----------
    binary_path : str | Path | None
        Explicit path to the profiler executable.  When None the subclass
        searches standard installation directories and PATH.
    """

    def __init__(self, binary_path: Optional[str | Path] = None) -> None:
        self._binary_path: Optional[Path] = (
            Path(binary_path) if binary_path is not None else None
        )
        self._resolved_binary: Optional[Path] = None  # cached after first find

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @staticmethod
    @abstractmethod
    def is_available() -> bool:
        """
        Return True if the profiler binary is discoverable on this system.

        Must not raise; degrade gracefully to False for missing tools.
        """

    @abstractmethod
    def find_binary(self) -> Optional[Path]:
        """
        Locate the profiler executable.

        Returns
        -------
        Path
            Absolute path to the binary if found.
        None
            When the tool is not installed or the path is invalid.
        """

    @abstractmethod
    def get_version(self) -> Optional[str]:
        """
        Query and return the profiler version string.

        Returns None when the binary cannot be found or the version query
        fails.  Never raises.
        """

    @abstractmethod
    def build_profiled_command(
        self,
        target_command: list[str],
        output_path: Path,
        extra_flags: tuple[str, ...] = (),
    ) -> list[str]:
        """
        Return a new command list that runs *target_command* under the profiler.

        Parameters
        ----------
        target_command : list[str]
            The original command to profile (e.g.
            ``["python", "runner.py", "--model-id", "6"]``).
        output_path : Path
            Where the profiler should write its report.  The concrete class
            determines the exact file extension.
        extra_flags : tuple[str, ...]
            Additional profiler flags to append.  Callers may use this to
            override defaults without subclassing.

        Returns
        -------
        list[str]
            The full command including the profiler binary and flags, with
            *target_command* appended at the end.
        """

    @abstractmethod
    def collect_artifact_metadata(self, output_path: Path) -> dict[str, str]:
        """
        Inspect artefact files produced after a profiler run and return a
        metadata dictionary suitable for ``TrialResult.profiler_artifact_paths``.

        Parameters
        ----------
        output_path : Path
            Base path passed to ``build_profiled_command``.

        Returns
        -------
        dict[str, str]
            Mapping of logical artefact role to absolute file path string,
            e.g. ``{"nsys_report": "/path/to/model.nsys-rep",
                     "nsys_sqlite": "/path/to/model.sqlite"}``.
            Keys and values that do not correspond to existing files should
            be omitted so callers can iterate the dict and open every path.
        """

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def get_resolved_binary(self) -> Optional[Path]:
        """
        Return the resolved binary path, using a cache after the first call.

        Subclasses should call this instead of ``find_binary`` in hot paths.
        """
        if self._resolved_binary is None:
            self._resolved_binary = self.find_binary()
        return self._resolved_binary

    def _log_unavailable(self, tool_name: str) -> None:
        """Emit a consistent warning when the tool binary is not found."""
        logger.warning(
            "%s binary not found. "
            "Install %s and ensure it is on PATH or provide the path via "
            "the binary_path constructor argument. "
            "Profiling is disabled for this run.",
            tool_name,
            tool_name,
        )
