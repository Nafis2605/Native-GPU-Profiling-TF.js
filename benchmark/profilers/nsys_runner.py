"""
Nsight Systems CLI runner for native-tfjs-bench.

Wraps a target benchmark command in an ``nsys profile`` invocation,
captures the resulting ``.nsys-rep`` report, and records profiling metadata.

Design constraints
------------------
* Windows-native: uses ``subprocess.CREATE_NO_WINDOW`` and avoids shell=True.
* The runner does NOT parse the report — that is NsysParser's responsibility.
* The runner does NOT interpret timing numbers from the profiled process;
  callers must understand that any latency observed in a profiled run is
  biased by nsys overhead and is invalid for reporting.
* Robust to nsys being absent: all public methods degrade gracefully.

Typical nsys default installation paths on Windows
---------------------------------------------------
  C:\\Program Files\\NVIDIA Corporation\\Nsight Systems <year>.<x>\\
      target-windows-x64\\nsys.exe
  C:\\Program Files\\NVIDIA Corporation\\Nsight Systems <year>.<x>\\
      host-windows-x64\\nsys.exe   (same binary via PATH)

The runner searches the ``NSYS_EXE`` environment variable first, then
standard program-files directories, then falls back to a plain PATH lookup.
"""

from __future__ import annotations

import datetime
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

# Standard Windows install directory glob fragment.  Evaluated lazily.
_NSYS_WIN_GLOB = "NVIDIA Corporation/Nsight Systems*/target-windows-x64/nsys.exe"

# Sync API names whose combined count/time we treat as "sync overhead"
_SYNC_API_NAMES: frozenset[str] = frozenset(
    {
        "cudaDeviceSynchronize",
        "cudaStreamSynchronize",
        "cudaEventSynchronize",
        "cuStreamSynchronize",
        "cuCtxSynchronize",
    }
)


# ── Configuration dataclass ───────────────────────────────────────────────────

@dataclass(frozen=True)
class NsysRunConfig:
    """
    Immutable configuration for one Nsight Systems profile run.

    Parameters
    ----------
    output_dir : Path
        Directory that will receive the ``.nsys-rep`` (and exported) files.
    report_name : str
        Base filename **without** extension.  nsys appends ``.nsys-rep``
        automatically.
    trace : str
        Comma-separated list of trace domains passed to ``--trace``.
        Default: ``"cuda,nvtx,osrt"`` — captures CUDA API, NVTX ranges, and
        Windows OS-runtime wait primitives.
        Add ``"cudnn,cublas"`` if those libs are available in your nsys build.
    sample : str
        CPU sampling mode for ``--sample``.  Default ``"none"`` disables CPU
        sampling to reduce overhead during GPU-focused benchmarks.
    cpuctxsw : str
        Context-switch tracing mode for ``--cpuctxsw``.  Default ``"none"``.
    capture_range : str | None
        Optional ``--capture-range`` argument (e.g. ``"cudaProfilerApi"``).
        When None the entire subprocess duration is captured.
    extra_flags : tuple[str, ...]
        Additional raw flags appended verbatim after the standard flags.
    timeout_s : int
        Maximum wall-clock seconds allowed for the profiling run.
        Includes nsys startup + target execution + report write.
        Default: 1800 (30 minutes), which accommodates large models with many
        kernels and slow report serialisation on rotating storage.
    """

    output_dir: Path
    report_name: str = "nsys_profile"
    trace: str = "cuda,nvtx,osrt"
    sample: str = "none"
    cpuctxsw: str = "none"
    capture_range: Optional[str] = None
    extra_flags: tuple[str, ...] = ()
    timeout_s: int = 1800

    def report_stem(self) -> Path:
        """
        Return the full path stem that nsys will use as ``--output``.

        nsys appends ``.nsys-rep`` to this path automatically.
        """
        return self.output_dir / self.report_name

    def expected_report_path(self) -> Path:
        """Return the expected ``.nsys-rep`` path after a successful run."""
        return self.output_dir / f"{self.report_name}.nsys-rep"


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class NsysRunResult:
    """
    Outcome of one ``nsys profile`` invocation.

    Fields
    ------
    report_path : Path | None
        Absolute path to the ``.nsys-rep`` file created by nsys.
        None when nsys was unavailable, the run timed out, or nsys itself
        crashed before writing the file.
    nsys_return_code : int
        Exit code of the ``nsys profile`` process.  0 = success.
    target_return_code : int | None
        Exit code of the *target* subprocess (the benchmark script we
        wrapped).  None when the target's exit code cannot be determined
        (e.g. due to a timeout kill).
    wall_time_s : float
        Elapsed wall-clock seconds from ``nsys profile`` start to finish,
        including report serialisation.
    profiler_metadata : dict
        nsys version, command line, trace domains, and environment snapshot.
    stdout : str
        Captured stdout from the ``nsys profile`` process.
    stderr : str
        Captured stderr from the ``nsys profile`` process.
    success : bool
        True when nsys reported exit code 0 **and** the expected report file
        exists on disk.
    failure_reason : str
        Human-readable explanation when ``success`` is False.
    """

    report_path: Optional[Path]
    nsys_return_code: int
    target_return_code: Optional[int]
    wall_time_s: float
    profiler_metadata: dict = field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    success: bool = False
    failure_reason: str = ""

    def to_dict(self) -> dict:
        """Serialise to a plain dict suitable for JSON output."""
        d = {
            "report_path": str(self.report_path) if self.report_path else None,
            "nsys_return_code": self.nsys_return_code,
            "target_return_code": self.target_return_code,
            "wall_time_s": round(self.wall_time_s, 4),
            "profiler_metadata": self.profiler_metadata,
            "success": self.success,
            "failure_reason": self.failure_reason,
        }
        # stdout/stderr excluded by default (can be large); callers write
        # them to separate log files via save_subprocess_logs().
        return d

    def save_subprocess_logs(self, directory: Path) -> dict[str, Path]:
        """
        Write stdout and stderr to text files in *directory*.

        Returns
        -------
        dict[str, Path]
            ``{"stdout": <path>, "stderr": <path>}`` for files that were
            written (skipped when the corresponding string is empty).
        """
        directory.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}
        if self.stdout:
            p = directory / "nsys_profile_stdout.log"
            p.write_text(self.stdout, encoding="utf-8")
            written["stdout"] = p
        if self.stderr:
            p = directory / "nsys_profile_stderr.log"
            p.write_text(self.stderr, encoding="utf-8")
            written["stderr"] = p
        return written


# ── NsysRunner ────────────────────────────────────────────────────────────────

class NsysRunner(ProfilerBase):
    """
    Nsight Systems CLI runner.

    Wraps a target command with ``nsys profile`` and returns an
    :class:`NsysRunResult` describing what happened.

    Instantiation
    -------------
    Pass ``binary_path`` to force a specific nsys executable.  When omitted
    the runner searches, in order:

    1. The ``NSYS_EXE`` environment variable.
    2. Standard Windows Nsight Systems installation directories under
       ``%PROGRAMFILES%``.
    3. The system PATH (``shutil.which("nsys")``).

    Parameters
    ----------
    binary_path : str | Path | None
        Explicit path to ``nsys.exe``.  When None, auto-detection is used.
    """

    # ------------------------------------------------------------------
    # Construction and binary detection
    # ------------------------------------------------------------------

    @staticmethod
    def is_available() -> bool:
        """
        Return True if an nsys binary can be located on this system.

        Checks ``NSYS_EXE`` env var, Windows program-files directories, and
        PATH.  Does not verify that the binary actually runs.
        """
        return NsysRunner._find_nsys_binary() is not None

    @staticmethod
    def _find_nsys_binary() -> Optional[Path]:
        """
        Locate the nsys binary without instantiating NsysRunner.

        Search order (Windows-first):
          1. NSYS_EXE environment variable
          2. %PROGRAMFILES%\\NVIDIA Corporation\\Nsight Systems *\\...nsys.exe
          3. shutil.which("nsys")
        """
        # 1. Explicit env override
        env_path = os.environ.get("NSYS_EXE")
        if env_path:
            p = Path(env_path)
            if p.is_file():
                logger.debug("nsys binary from NSYS_EXE: %s", p)
                return p
            else:
                logger.warning(
                    "NSYS_EXE is set to %r but the file does not exist. "
                    "Falling back to auto-detection.",
                    env_path,
                )

        # 2. Standard Windows install directories
        if os.name == "nt":
            prog_files = Path(
                os.environ.get("PROGRAMFILES", r"C:\Program Files")
            )
            nvidia_dir = prog_files / "NVIDIA Corporation"
            if nvidia_dir.is_dir():
                # Find the most recently installed Nsight Systems version
                # by sorting directory names (versions sort lexicographically).
                candidates: list[Path] = sorted(
                    nvidia_dir.glob("Nsight Systems*/target-windows-x64/nsys.exe"),
                    reverse=True,
                )
                if candidates:
                    logger.debug("nsys binary from program-files: %s", candidates[0])
                    return candidates[0]

                # Also try host-windows-x64 (older installs)
                candidates_host: list[Path] = sorted(
                    nvidia_dir.glob("Nsight Systems*/host-windows-x64/nsys.exe"),
                    reverse=True,
                )
                if candidates_host:
                    logger.debug(
                        "nsys binary from program-files (host dir): %s",
                        candidates_host[0],
                    )
                    return candidates_host[0]

        # 3. PATH lookup
        which_result = shutil.which("nsys")
        if which_result:
            logger.debug("nsys binary from PATH: %s", which_result)
            return Path(which_result)

        return None

    def find_binary(self) -> Optional[Path]:
        """
        Return the resolved nsys binary path.

        Respects the ``binary_path`` constructor argument; falls back to
        :meth:`_find_nsys_binary`.
        """
        if self._binary_path is not None:
            if self._binary_path.is_file():
                return self._binary_path
            logger.warning(
                "Explicit binary_path %r is not a file; falling back to "
                "auto-detection.",
                str(self._binary_path),
            )

        return self._find_nsys_binary()

    def get_version(self) -> Optional[str]:
        """
        Query the nsys version string.

        Runs ``nsys --version`` and returns the first non-empty output line.
        Returns None on any error.
        """
        binary = self.get_resolved_binary()
        if binary is None:
            return None
        try:
            result = subprocess.run(
                [str(binary), "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                **_SUBPROCESS_FLAGS,
            )
            for line in result.stdout.splitlines():
                line = line.strip()
                if line:
                    return line
            # version sometimes in stderr for older builds
            for line in result.stderr.splitlines():
                line = line.strip()
                if line and "version" in line.lower():
                    return line
        except Exception as exc:
            logger.debug("nsys --version failed: %s", exc)
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
        Build an ``nsys profile`` command wrapping *target_command*.

        This low-level method uses ``--trace=cuda,nvtx,osrt`` and sane
        :class:`NsysRunConfig` defaults.  For full control use
        :meth:`build_nsys_command` with an explicit :class:`NsysRunConfig`.

        Parameters
        ----------
        target_command : list[str]
            Command to profile (executable + args).
        output_path : Path
            Path stem for ``--output`` (nsys appends ``.nsys-rep``).
        extra_flags : tuple[str, ...]
            Extra nsys flags inserted before the target command.

        Returns
        -------
        list[str]
            Full command starting with the nsys binary.

        Raises
        ------
        FileNotFoundError
            If nsys cannot be located.
        """
        binary = self.get_resolved_binary()
        if binary is None:
            raise FileNotFoundError(
                "nsys binary not found. Install Nsight Systems or set NSYS_EXE."
            )
        config = NsysRunConfig(
            output_dir=output_path.parent,
            report_name=output_path.name,
            extra_flags=extra_flags,
        )
        return self.build_nsys_command(target_command, config)

    def build_nsys_command(
        self,
        target_command: list[str],
        config: NsysRunConfig,
    ) -> list[str]:
        """
        Build the full ``nsys profile`` command from an explicit
        :class:`NsysRunConfig`.

        Parameters
        ----------
        target_command : list[str]
            Command to profile.
        config : NsysRunConfig
            Profiling configuration (trace domains, output path, flags).

        Returns
        -------
        list[str]
            Full command; empty list if nsys binary is unavailable.
        """
        binary = self.get_resolved_binary()
        if binary is None:
            self._log_unavailable("nsys")
            return []

        config.output_dir.mkdir(parents=True, exist_ok=True)

        cmd: list[str] = [
            str(binary),
            "profile",
            f"--trace={config.trace}",
            f"--output={config.report_stem()}",
            "--force-overwrite=true",
            f"--sample={config.sample}",
            f"--cpuctxsw={config.cpuctxsw}",
        ]

        if config.capture_range:
            cmd.append(f"--capture-range={config.capture_range}")

        cmd.extend(config.extra_flags)

        # separator: everything after '--' is the target command
        cmd.append("--")
        cmd.extend(target_command)

        return cmd

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(
        self,
        target_command: list[str],
        config: NsysRunConfig,
    ) -> NsysRunResult:
        """
        Launch *target_command* under ``nsys profile`` and return results.

        This method blocks until the subprocess exits or ``config.timeout_s``
        elapses.  On timeout the subprocess tree is killed before returning.

        On Windows, ``nsys profile`` spawns its own child process for the
        target.  ``proc.kill()`` terminates the nsys parent; the target child
        should also be cleaned up by Windows job-object semantics, but callers
        should verify orphan processes in long test suites.

        Parameters
        ----------
        target_command : list[str]
            The benchmark command to profile.
        config : NsysRunConfig
            Profiling configuration.

        Returns
        -------
        NsysRunResult
            Always returns a result; never raises.  Check ``result.success``
            and ``result.failure_reason`` for error details.
        """
        binary = self.get_resolved_binary()
        if binary is None:
            self._log_unavailable("nsys")
            return NsysRunResult(
                report_path=None,
                nsys_return_code=-1,
                target_return_code=None,
                wall_time_s=0.0,
                success=False,
                failure_reason=(
                    "nsys binary not found. Install Nsight Systems or set NSYS_EXE."
                ),
            )

        nsys_version = self.get_version()
        full_cmd = self.build_nsys_command(target_command, config)
        if not full_cmd:
            return NsysRunResult(
                report_path=None,
                nsys_return_code=-1,
                target_return_code=None,
                wall_time_s=0.0,
                success=False,
                failure_reason="Failed to build nsys command (binary missing?).",
            )

        metadata = self._build_metadata(
            nsys_version=nsys_version,
            full_cmd=full_cmd,
            config=config,
        )

        logger.info(
            "Launching nsys profile run | report_stem=%s | timeout=%ds",
            config.report_stem(),
            config.timeout_s,
        )
        logger.debug("nsys command: %s", " ".join(full_cmd))

        start_t = time.perf_counter()
        proc: Optional[subprocess.Popen] = None  # type: ignore[type-arg]
        stdout_text = ""
        stderr_text = ""
        nsys_rc = -1
        timed_out = False

        try:
            proc = subprocess.Popen(
                full_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                **_SUBPROCESS_FLAGS,
            )
            try:
                stdout_text, stderr_text = proc.communicate(
                    timeout=config.timeout_s
                )
                nsys_rc = proc.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
                logger.warning(
                    "nsys profile timed out after %d s — killing process",
                    config.timeout_s,
                )
                try:
                    proc.kill()
                except Exception:
                    pass
                try:
                    stdout_text, stderr_text = proc.communicate(timeout=10)
                except Exception:
                    stdout_text = stdout_text or ""
                    stderr_text = stderr_text or ""
                nsys_rc = proc.returncode if proc.returncode is not None else -1

        except OSError as exc:
            wall_time_s = time.perf_counter() - start_t
            msg = f"Failed to launch nsys: {exc}"
            logger.error(msg)
            return NsysRunResult(
                report_path=None,
                nsys_return_code=-1,
                target_return_code=None,
                wall_time_s=wall_time_s,
                profiler_metadata=metadata,
                stdout=stdout_text,
                stderr=stderr_text,
                success=False,
                failure_reason=msg,
            )
        finally:
            if proc is not None and proc.poll() is None:
                try:
                    proc.kill()
                except Exception:
                    pass

        wall_time_s = time.perf_counter() - start_t
        expected_report = config.expected_report_path()

        # --- check for report file ------------------------------------------
        if timed_out:
            failure_reason = (
                f"nsys profile timed out after {config.timeout_s} s. "
                "Report file may be incomplete or absent."
            )
            report_path = expected_report if expected_report.exists() else None
            success = False
        elif nsys_rc != 0:
            failure_reason = (
                f"nsys profile exited with code {nsys_rc}. "
                "Check nsys_profile_stderr.log for details."
            )
            report_path = expected_report if expected_report.exists() else None
            success = False
            logger.warning(
                "nsys profile failed (rc=%d) | stderr snippet: %s",
                nsys_rc,
                stderr_text[:400] if stderr_text else "<empty>",
            )
        elif not expected_report.exists():
            failure_reason = (
                f"nsys profile exited 0 but expected report not found at "
                f"{expected_report}. "
                "Check output permissions and available disk space."
            )
            report_path = None
            success = False
            logger.warning(failure_reason)
        else:
            report_path = expected_report
            failure_reason = ""
            success = True
            logger.info(
                "nsys profile completed | report=%s | wall=%.1f s",
                report_path,
                wall_time_s,
            )

        return NsysRunResult(
            report_path=report_path,
            nsys_return_code=nsys_rc,
            target_return_code=None,  # nsys wraps the target; return code is nsys's
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
        Scan the output directory for artefact files produced by nsys.

        Returns paths for ``.nsys-rep`` and, if present, ``.sqlite`` (from a
        prior :class:`NsysParser` export run).

        Parameters
        ----------
        output_path : Path
            The base path stem used in the original :class:`NsysRunConfig`
            (i.e. ``config.report_stem()``).
        """
        artifacts: dict[str, str] = {}

        rep = Path(str(output_path) + ".nsys-rep")
        if rep.exists():
            artifacts["nsys_report"] = str(rep.resolve())

        sqlite = Path(str(output_path) + ".sqlite")
        if sqlite.exists():
            artifacts["nsys_sqlite"] = str(sqlite.resolve())

        return artifacts

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_metadata(
        nsys_version: Optional[str],
        full_cmd: list[str],
        config: NsysRunConfig,
    ) -> dict:
        return {
            "nsys_version": nsys_version or "unknown",
            "trace_domains": config.trace,
            "sample_mode": config.sample,
            "cpuctxsw_mode": config.cpuctxsw,
            "capture_range": config.capture_range,
            "extra_flags": list(config.extra_flags),
            "full_command": full_cmd,
            "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        }
