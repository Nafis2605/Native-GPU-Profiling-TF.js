"""
Nsight Compute CLI runner for native-tfjs-bench.

Wraps a target benchmark command in an ``ncu`` (Nsight Compute CLI) invocation,
captures the resulting ``.ncu-rep`` report and the CSV metric dump written to
stdout, and records profiling metadata.

Design constraints
------------------
* Windows-native: uses ``subprocess.CREATE_NO_WINDOW`` and avoids shell=True.
* ncu profiles a configurable window of kernel launches (launch_skip +
  launch_count) rather than the entire process, keeping overhead tractable.
* ``ncu --csv`` writes metric output to stdout; stderr carries ncu progress
  messages.  Both are captured separately.
* The runner does NOT parse metric CSV — that is NcuParser's responsibility.
* Robust to ncu being absent: all public methods degrade gracefully.

Typical ncu installation paths on Windows
------------------------------------------
  C:\\Program Files\\NVIDIA Corporation\\Nsight Compute <year>.<x>.<y>\\ncu.exe

Binary search order
-------------------
  1. ``NCU_EXE`` environment variable (explicit override).
  2. ``%PROGRAMFILES%\\NVIDIA Corporation\\Nsight Compute*\\ncu.exe`` glob
     (sorted descending → newest install wins).
  3. ``shutil.which("ncu")`` (PATH fallback).

Profiler overhead note
----------------------
ncu replays every profiled kernel launch multiple times to collect hardware
performance counters.  Expect 10×–100× slowdown per profiled kernel depending
on the metric set.  ``wall_time_s`` in NcuRunResult reflects this inflated
execution time and MUST NOT be compared against clean-benchmark TrialResult
latency values.
"""

from __future__ import annotations

import datetime
import glob as _glob
import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from benchmark.profilers.base import ProfilerBase

logger = logging.getLogger(__name__)

# Windows: suppress console window for all child processes
_SUBPROCESS_FLAGS: dict = (
    {"creationflags": subprocess.CREATE_NO_WINDOW}
    if os.name == "nt"
    else {}
)

# Windows glob fragment for ncu.exe.  Sorted descending so the newest
# Nsight Compute release is tried first.
_NCU_WIN_GLOB = "NVIDIA Corporation/Nsight Compute*/ncu.exe"


# ── MetricCategory ────────────────────────────────────────────────────────────

class MetricCategory:
    """
    String constants identifying groups of Nsight Compute metrics.

    Used to annotate every metric in NcuKernelMetrics and to populate
    the metric_category field of NcuKernelResult in result_schema.py.
    """

    KERNEL_DURATION   = "kernel_duration"
    OCCUPANCY         = "occupancy"
    MEMORY_THROUGHPUT = "memory_throughput"
    CACHE_BEHAVIOR    = "cache_behavior"
    SM_EFFICIENCY     = "sm_efficiency"
    TENSOR_CORE       = "tensor_core"
    WARP_SCHEDULER    = "warp_scheduler"
    UNCLASSIFIED      = "unclassified"

    _ALL: frozenset[str] = frozenset({
        "kernel_duration", "occupancy", "memory_throughput",
        "cache_behavior", "sm_efficiency", "tensor_core",
        "warp_scheduler", "unclassified",
    })


# ── Metric set definitions ────────────────────────────────────────────────────

# Each tuple contains raw ncu metric identifiers.  These are consistent across
# Volta (SM 7.0) and later.  Older metrics that were renamed keep their modern
# names here; ncu degrades gracefully to "N/A" for unsupported metrics.

KERNEL_DURATION_METRICS: tuple[str, ...] = (
    "gpu__time_duration.sum",        # kernel execution duration (nanoseconds)
    "sm__cycles_elapsed.avg",        # average SM clock cycles
)

OCCUPANCY_METRICS: tuple[str, ...] = (
    "sm__warps_active.avg.pct_of_peak_sustained_active",  # achieved occupancy
    "launch__occupancy_theoretical_pct",                   # theoretical occupancy
    "launch__waves_per_multiprocessor",                    # waves per SM
    "launch__registers_per_thread",                        # register pressure
    "launch__shared_mem_per_block_static",                 # static shared memory
)

MEMORY_THROUGHPUT_METRICS: tuple[str, ...] = (
    "dram__bytes_read.sum.pct_of_peak_sustained_elapsed",   # DRAM read BW%
    "dram__bytes_write.sum.pct_of_peak_sustained_elapsed",  # DRAM write BW%
    "lts__t_bytes.sum.pct_of_peak_sustained_elapsed",       # L2 cache BW%
    "l1tex__t_bytes.sum.pct_of_peak_sustained_elapsed",     # L1/TEX BW%
)

CACHE_METRICS: tuple[str, ...] = (
    "l1tex__t_sector_hit_rate.pct",               # L1 hit rate
    "lts__t_sector_hit_rate.pct",                 # L2 hit rate
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",  # global load requests
    "lts__t_requests.sum",                         # L2 total requests
)

SM_EFFICIENCY_METRICS: tuple[str, ...] = (
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",          # SM utilisation
    "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active",  # ALU pipe
    "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",  # FMA pipe
    "sm__inst_executed.sum",                                         # instructions
)

TENSOR_CORE_METRICS: tuple[str, ...] = (
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active",
)

WARP_SCHEDULER_METRICS: tuple[str, ...] = (
    "scheduler__warps_eligible.avg.per_cycle_active",  # eligible warps/cycle
    "scheduler__warps_issued.avg.per_cycle_active",    # issued warps/cycle
)

# Combined default metric set — order determines CSV column order in output.
DEFAULT_METRICS: tuple[str, ...] = (
    KERNEL_DURATION_METRICS
    + OCCUPANCY_METRICS
    + MEMORY_THROUGHPUT_METRICS
    + CACHE_METRICS
    + SM_EFFICIENCY_METRICS
    + TENSOR_CORE_METRICS
    + WARP_SCHEDULER_METRICS
)

# Lookup: metric name substring → MetricCategory.  Checked in insertion order;
# the first matching fragment wins.  Longer/more-specific fragments first.
_METRIC_CATEGORY_MAP: dict[str, str] = {
    # Kernel duration
    "gpu__time_duration":            MetricCategory.KERNEL_DURATION,
    "sm__cycles_elapsed":            MetricCategory.KERNEL_DURATION,
    # Occupancy
    "sm__warps_active":              MetricCategory.OCCUPANCY,
    "launch__occupancy":             MetricCategory.OCCUPANCY,
    "launch__waves":                 MetricCategory.OCCUPANCY,
    "launch__registers":             MetricCategory.OCCUPANCY,
    "launch__shared_mem":            MetricCategory.OCCUPANCY,
    # Memory throughput
    "dram__bytes":                   MetricCategory.MEMORY_THROUGHPUT,
    "lts__t_bytes":                  MetricCategory.MEMORY_THROUGHPUT,
    "l1tex__t_bytes":                MetricCategory.MEMORY_THROUGHPUT,
    "gpu__dram_throughput":          MetricCategory.MEMORY_THROUGHPUT,
    # Cache behavior — check before SM so "hit_rate" doesn't fall to sm_efficiency
    "l1tex__t_sector_hit":           MetricCategory.CACHE_BEHAVIOR,
    "lts__t_sector_hit":             MetricCategory.CACHE_BEHAVIOR,
    "l1tex__t_requests":             MetricCategory.CACHE_BEHAVIOR,
    "lts__t_requests":               MetricCategory.CACHE_BEHAVIOR,
    # Tensor core — check before general pipe_cycles
    "sm__pipe_tensor_op_hmma":       MetricCategory.TENSOR_CORE,
    "sm__pipe_tensor":               MetricCategory.TENSOR_CORE,
    # SM efficiency
    "sm__throughput":                MetricCategory.SM_EFFICIENCY,
    "sm__pipe_alu":                  MetricCategory.SM_EFFICIENCY,
    "sm__pipe_fma":                  MetricCategory.SM_EFFICIENCY,
    "sm__pipe_fp16":                 MetricCategory.SM_EFFICIENCY,
    "sm__inst_executed":             MetricCategory.SM_EFFICIENCY,
    # Warp / scheduler
    "scheduler__warps":              MetricCategory.WARP_SCHEDULER,
}


def categorize_metric(metric_name: str) -> str:
    """
    Classify *metric_name* into a MetricCategory constant.

    Iterates ``_METRIC_CATEGORY_MAP`` in insertion order and returns the
    category for the first matching substring.  Returns UNCLASSIFIED when
    no fragment matches.
    """
    for fragment, category in _METRIC_CATEGORY_MAP.items():
        if fragment in metric_name:
            return category
    return MetricCategory.UNCLASSIFIED


# ── NcuRunConfig ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class NcuRunConfig:
    """
    Immutable configuration for one Nsight Compute profile run.

    Parameters
    ----------
    output_dir : Path
        Directory that will receive the ``.ncu-rep`` report and log files.
    report_name : str
        Base filename **without** extension.  ncu appends ``.ncu-rep``.
    metrics : tuple[str, ...]
        ncu metric identifiers to collect.  Defaults to DEFAULT_METRICS.
        Unsupported metrics on the current GPU produce "N/A" in the output.
    launch_skip : int
        Number of kernel launches to skip before profiling begins.  Use this
        to skip CUDA driver / framework initialisation kernels so the profiling
        window lands on representative steady-state inference kernels.
    launch_count : int
        Number of consecutive kernel launches to profile.  A value of 5–20
        typically captures all unique kernels in a single forward pass.
        Keep this small — ncu replay multiplies measured time by ~10×–100×.
    kernel_regex : str | None
        Optional Python-compatible regex passed to ``ncu --kernel-regex``.
        When set, ncu only profiles kernel launches whose demangled name
        matches the pattern.  None = profile all observed kernel launches.
    target_processes : str
        ``"all"`` profiles the application and all its spawned subprocesses.
        ``"application-only"`` profiles only the top-level process.
    extra_flags : tuple[str, ...]
        Additional flags appended verbatim to the ncu command line.
    timeout_s : int
        Per-run timeout in seconds.  ncu overhead is high; default is 1 hour.
    """

    output_dir: Path
    report_name: str = "ncu_profile"
    metrics: tuple[str, ...] = DEFAULT_METRICS
    launch_skip: int = 0
    launch_count: int = 10
    kernel_regex: Optional[str] = None
    target_processes: str = "all"
    extra_flags: tuple[str, ...] = ()
    timeout_s: int = 3600

    def report_stem(self) -> Path:
        """Return output_dir / report_name (no extension)."""
        return self.output_dir / self.report_name

    def expected_report_path(self) -> Path:
        """Return the path where ncu will write the .ncu-rep binary report."""
        return self.report_stem().with_suffix(".ncu-rep")


# ── NcuRunResult ──────────────────────────────────────────────────────────────

@dataclass
class NcuRunResult:
    """
    Result of one NcuRunner.run() call.

    Attributes
    ----------
    report_path : Path | None
        Path to the ``.ncu-rep`` binary report if it was produced; None otherwise.
    ncu_return_code : int
        Exit code of the ncu process itself.  0 = ncu ran without error.
    target_return_code : int
        Exit code of the profiled target process.  Best-effort: ncu propagates
        the wrapped process's exit code as its own when profiling succeeds.
    wall_time_s : float
        Elapsed wall-clock seconds from ncu launch to process exit, including
        all hardware-counter replay overhead.  MUST NOT be used for latency
        benchmarking.
    profiler_metadata : dict
        Internal metadata (binary path, version, config, timestamp).
    stdout : str
        Captured stdout from ncu.  Contains the CSV metric dump when
        ``--csv`` is passed (which NcuRunner always does).
    stderr : str
        Captured stderr from ncu.  Contains ncu progress, warnings, and errors.
    success : bool
        True iff ncu exited without timeout AND the .ncu-rep file was produced.
    failure_reason : str | None
        Human-readable reason for failure; None when success is True.
    """

    report_path: Optional[Path]
    ncu_return_code: int
    target_return_code: int
    wall_time_s: float
    profiler_metadata: dict
    stdout: str = ""
    stderr: str = ""
    success: bool = False
    failure_reason: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to plain dict; safe for JSON serialisation."""
        return {
            "report_path": str(self.report_path) if self.report_path else None,
            "ncu_return_code": self.ncu_return_code,
            "target_return_code": self.target_return_code,
            "wall_time_s": self.wall_time_s,
            "profiler_metadata": self.profiler_metadata,
            "success": self.success,
            "failure_reason": self.failure_reason,
        }

    def save_subprocess_logs(self, directory: Path) -> None:
        """
        Write captured stdout and stderr to log files inside *directory*.

        stdout (CSV metric dump) → ``ncu_profile_stdout.log``
        stderr (ncu progress)    → ``ncu_profile_stderr.log``
        """
        directory.mkdir(parents=True, exist_ok=True)
        if self.stdout:
            (directory / "ncu_profile_stdout.log").write_text(
                self.stdout, encoding="utf-8"
            )
        if self.stderr:
            (directory / "ncu_profile_stderr.log").write_text(
                self.stderr, encoding="utf-8"
            )


# ── NcuRunner ─────────────────────────────────────────────────────────────────

class NcuRunner(ProfilerBase):
    """
    Nsight Compute CLI runner.

    Launches a target command wrapped in ``ncu``, captures the CSV metric
    dump from stdout, saves the binary ``.ncu-rep`` report, and returns a
    NcuRunResult with all artefact paths and profiling metadata.

    Typical usage
    -------------
    ::

        runner = NcuRunner()
        if not NcuRunner.is_available():
            raise SystemExit("ncu not found")

        config = NcuRunConfig(
            output_dir=Path("results/ncu"),
            launch_skip=5,
            launch_count=10,
        )
        result = runner.run(
            ["python", "scripts/run_one_model.py", "--model-id", "3"],
            config,
        )
        print(result.success, result.report_path)

    Profiler overhead
    -----------------
    ncu replays each profiled kernel launch multiple times (hardware-counter
    replay) to gather all requested metrics simultaneously.  Expect
    10×–100× slowdown relative to an un-profiled run.  ``wall_time_s`` in
    the result reflects this overhead and is meaningless for latency analysis.
    """

    # ------------------------------------------------------------------
    # Static availability check
    # ------------------------------------------------------------------

    @staticmethod
    def is_available() -> bool:
        """Return True if ncu is discoverable on this system.  Never raises."""
        try:
            return NcuRunner._find_ncu_binary() is not None
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Binary discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _find_ncu_binary() -> Optional[Path]:
        """Search for the ncu executable in known locations.  Returns None if absent."""
        # 1. Explicit environment override
        env_val = os.environ.get("NCU_EXE")
        if env_val:
            p = Path(env_val)
            if p.is_file():
                return p

        # 2. Standard Windows Program Files glob (newest install first)
        if os.name == "nt":
            prog_files = os.environ.get("PROGRAMFILES", r"C:\Program Files")
            pattern = str(Path(prog_files) / _NCU_WIN_GLOB)
            for candidate in sorted(_glob.glob(pattern), reverse=True):
                cp = Path(candidate)
                if cp.is_file():
                    return cp

        # 3. PATH fallback
        found = shutil.which("ncu")
        if found:
            return Path(found)

        return None

    def find_binary(self) -> Optional[Path]:
        if self._binary_path and self._binary_path.is_file():
            return self._binary_path
        return self._find_ncu_binary()

    def get_version(self) -> Optional[str]:
        """
        Query ``ncu --version`` and return the version string.

        Returns None if the binary is not found or the query fails.  Never raises.
        """
        binary = self.get_resolved_binary()
        if binary is None:
            return None
        try:
            proc = subprocess.run(
                [str(binary), "--version"],
                capture_output=True,
                text=True,
                timeout=15,
                **_SUBPROCESS_FLAGS,
            )
            for line in proc.stdout.splitlines():
                stripped = line.strip()
                if stripped and ("Nsight Compute" in stripped or "NVIDIA" in stripped):
                    return stripped
            # Fallback: return first non-empty line
            for line in proc.stdout.splitlines():
                stripped = line.strip()
                if stripped:
                    return stripped
        except Exception as exc:
            logger.debug("ncu --version failed: %s", exc)
        return None

    # ------------------------------------------------------------------
    # Command construction
    # ------------------------------------------------------------------

    def build_profiled_command(
        self,
        target_command: list[str],
        output_path: Path,
        extra_flags: tuple[str, ...] = (),
    ) -> list[str]:
        """
        Implement ProfilerBase interface using a default NcuRunConfig.

        For full control use build_ncu_command() with an explicit NcuRunConfig.
        """
        config = NcuRunConfig(
            output_dir=output_path.parent,
            report_name=output_path.stem,
            extra_flags=extra_flags,
        )
        return self.build_ncu_command(target_command, config)

    def build_ncu_command(
        self,
        target_command: list[str],
        config: NcuRunConfig,
    ) -> list[str]:
        """
        Build the full ncu invocation as a ``list[str]``.

        Flags generated
        ---------------
        ``--output <stem>``         Save binary .ncu-rep report.
        ``--csv``                   Write metric CSV to stdout.
        ``--quiet``                 Suppress decorative headers from stdout
                                    so the CSV parser sees a clean stream.
        ``--force-overwrite``       Do not fail if .ncu-rep already exists.
        ``--metrics <list>``        Comma-joined metric identifiers.
        ``--launch-skip <N>``       Skip first N kernel launches.
        ``--launch-count <N>``      Profile exactly N kernel launches.
        ``--kernel-name-base function`` + ``--kernel-regex <re>``
                                    Filter by kernel name when kernel_regex is set.
        ``--target-processes <val>``  "all" or "application-only".
        ``[extra_flags]``           Any additional caller-supplied flags.
        ``-- <target_command>``     Profiled target.
        """
        binary = self.get_resolved_binary()
        if binary is None:
            raise FileNotFoundError(
                "ncu binary not found. "
                "Install NVIDIA Nsight Compute or set NCU_EXE to its path."
            )

        config.output_dir.mkdir(parents=True, exist_ok=True)

        cmd: list[str] = [str(binary)]

        # Binary report output
        cmd += ["--output", str(config.report_stem())]

        # Metric CSV to stdout + clean output
        cmd += ["--csv", "--quiet", "--force-overwrite"]

        # Metric selection
        if config.metrics:
            cmd += ["--metrics", ",".join(config.metrics)]

        # Kernel launch window
        if config.launch_skip > 0:
            cmd += ["--launch-skip", str(config.launch_skip)]
        if config.launch_count > 0:
            cmd += ["--launch-count", str(config.launch_count)]

        # Kernel name filter
        if config.kernel_regex:
            cmd += ["--kernel-name-base", "function"]
            cmd += ["--kernel-regex", config.kernel_regex]

        # Target process scope
        cmd += ["--target-processes", config.target_processes]

        # Extra user-supplied flags
        cmd += list(config.extra_flags)

        # Separator + target command
        cmd += ["--"]
        cmd += target_command

        return cmd

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(
        self,
        target_command: list[str],
        config: NcuRunConfig,
    ) -> NcuRunResult:
        """
        Profile *target_command* under ncu and return a NcuRunResult.

        The CSV metric dump is captured from ncu's stdout.  ncu progress
        messages arrive on stderr.  Both are stored in the result for
        downstream parsing and logging.

        Overhead note
        -------------
        ncu replays every profiled kernel launch multiple times to collect
        hardware-counter data.  ``wall_time_s`` in the result is 10×–100×
        the normal kernel execution time.  It must not be compared with
        clean-benchmark TrialResult latency values.

        Parameters
        ----------
        target_command : list[str]
            The command to profile, e.g.
            ``["python", "scripts/run_one_model.py", "--model-id", "3"]``.
        config : NcuRunConfig
            Profiling parameters (metric set, launch window, output directory).

        Returns
        -------
        NcuRunResult
            All captured metadata, stdout CSV, stderr, and paths.
        """
        binary = self.get_resolved_binary()
        if binary is None:
            self._log_unavailable("ncu")
            return NcuRunResult(
                report_path=None,
                ncu_return_code=-1,
                target_return_code=-1,
                wall_time_s=0.0,
                profiler_metadata={},
                success=False,
                failure_reason="ncu binary not found",
            )

        ncu_cmd = self.build_ncu_command(target_command, config)
        logger.info("ncu command: %s", " ".join(ncu_cmd))
        logger.warning(
            "Nsight Compute profiling is active.  "
            "Wall-clock times will be 10×–100× slower than normal due to "
            "hardware-counter replay overhead.  "
            "Do NOT use any timing values from this run for latency reporting.  "
            "launch_skip=%d  launch_count=%d  timeout=%ds",
            config.launch_skip,
            config.launch_count,
            config.timeout_s,
        )

        t_start = time.perf_counter()
        stdout_text = ""
        stderr_text = ""
        ncu_rc = -1
        timed_out = False

        try:
            proc = subprocess.Popen(
                ncu_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                **_SUBPROCESS_FLAGS,
            )
            try:
                stdout_text, stderr_text = proc.communicate(timeout=config.timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                logger.error(
                    "ncu timed out after %d s — killing process tree",
                    config.timeout_s,
                )
                proc.kill()
                stdout_text, stderr_text = proc.communicate()

            ncu_rc = proc.returncode

        except Exception as exc:
            wall_time_s = time.perf_counter() - t_start
            logger.error("Failed to launch ncu: %s", exc)
            return NcuRunResult(
                report_path=None,
                ncu_return_code=-1,
                target_return_code=-1,
                wall_time_s=wall_time_s,
                profiler_metadata=self._build_metadata(config, None),
                stdout=stdout_text,
                stderr=stderr_text,
                success=False,
                failure_reason=f"Launch error: {exc}",
            )

        wall_time_s = time.perf_counter() - t_start

        report_path = config.expected_report_path()
        report_exists = report_path.is_file()
        success = (not timed_out) and report_exists

        failure_reason: Optional[str] = None
        if timed_out:
            failure_reason = f"Timeout after {config.timeout_s} s"
        elif not report_exists:
            failure_reason = (
                f"Report not produced at {report_path}"
                + (f"; ncu exit code {ncu_rc}" if ncu_rc != 0 else "")
            )

        if not success:
            logger.error("ncu run failed: %s", failure_reason)
        else:
            logger.info(
                "ncu run succeeded in %.1f s — report: %s",
                wall_time_s,
                report_path,
            )
            if ncu_rc != 0:
                logger.warning(
                    "ncu exit code %d (non-zero may indicate profiled process "
                    "failed — check ncu_profile_stderr.log for details)",
                    ncu_rc,
                )

        metadata = self._build_metadata(config, report_path if report_exists else None)
        metadata["wall_time_s"] = wall_time_s
        metadata["ncu_return_code"] = ncu_rc
        metadata["timed_out"] = timed_out

        return NcuRunResult(
            report_path=report_path if report_exists else None,
            ncu_return_code=ncu_rc,
            target_return_code=ncu_rc,  # ncu propagates wrapped process rc
            wall_time_s=wall_time_s,
            profiler_metadata=metadata,
            stdout=stdout_text,
            stderr=stderr_text,
            success=success,
            failure_reason=failure_reason,
        )

    # ------------------------------------------------------------------
    # Artifact metadata
    # ------------------------------------------------------------------

    def collect_artifact_metadata(self, output_path: Path) -> dict[str, str]:
        """
        Return artefact paths suitable for ``TrialResult.profiler_artifact_paths``.

        Checks for the ``.ncu-rep`` report adjacent to *output_path*.
        """
        result: dict[str, str] = {}
        rep = output_path.with_suffix(".ncu-rep")
        if rep.is_file():
            result["ncu_report"] = str(rep)
        return result

    def _build_metadata(
        self,
        config: NcuRunConfig,
        report_path: Optional[Path],
    ) -> dict:
        return {
            "profiler": "ncu",
            "ncu_binary": str(self.get_resolved_binary() or ""),
            "ncu_version": self.get_version() or "unknown",
            "report_path": str(report_path) if report_path else None,
            "report_name": config.report_name,
            "metrics_requested": list(config.metrics),
            "launch_skip": config.launch_skip,
            "launch_count": config.launch_count,
            "kernel_regex": config.kernel_regex,
            "target_processes": config.target_processes,
            "timestamp_utc": datetime.datetime.now(
                tz=datetime.timezone.utc
            ).isoformat(),
        }
