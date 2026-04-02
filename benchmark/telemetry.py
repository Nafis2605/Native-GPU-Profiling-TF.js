"""
GPU telemetry for native-tfjs-bench.

Primary backend: nvidia-smi subprocess (1-second intervals)
------------------------------------------------------------
nvidia-smi is spawned as a long-running background process with --loop=1,
which outputs one CSV line per second to stdout. A daemon thread reads and
parses those lines. This approach mirrors the paper's telemetry style and
requires no extra Python packages beyond the stdlib.

Command used:
    nvidia-smi \\
        --query-gpu=utilization.gpu,memory.used,power.draw \\
        --format=csv,noheader,nounits \\
        --id=<device_index> \\
        --loop=1

Sample output line:  45, 3124, 125.50
                     ^    ^     ^
                     util mem   power_mW(no, it's W with nounits)

power.draw may return "[Not Supported]" on consumer GPUs — handled gracefully.

Secondary backend: pynvml (optional, higher frequency)
------------------------------------------------------
If pynvml is installed, NvmlTelemetry provides configurable-Hz polling
(default 5 Hz). Used when sub-second granularity is needed.

Memory bandwidth interface
--------------------------
Phase 1 returns supported=False via NullMemoryBandwidthCollector.
The CuptiMemoryBandwidthCollector stub shows the integration point for
Phase 2 CUPTI / Nsight Systems work.

Failure handling
----------------
All methods degrade gracefully:
  - nvidia-smi not on PATH → start() returns False; summary has nvml_available=False
  - power sensor not supported → power fields are None
  - parse errors on individual lines → that sample is skipped
  - subprocess dies early → thread exits cleanly; stop() returns what was collected
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from benchmark.result_schema import RunMode

logger = logging.getLogger(__name__)

# Windows: suppress console window when spawning nvidia-smi subprocess
_SUBPROCESS_FLAGS: dict = (
    {"creationflags": subprocess.CREATE_NO_WINDOW}
    if os.name == "nt"
    else {}
)

# ── Data containers ───────────────────────────────────────────────────────────

@dataclass
class TelemetrySample:
    """One point-in-time GPU snapshot from a single poll."""
    timestamp_s: float
    gpu_util_pct: Optional[float] = None     # 0–100
    gpu_mem_used_mb: Optional[float] = None  # MiB
    power_w: Optional[float] = None          # Watts; None if unsupported


@dataclass
class TelemetrySummary:
    """
    Aggregated GPU telemetry across an entire measurement window.

    None values indicate the sensor was unavailable for the duration.
    Always check collection_method and nvml_available before trusting
    numeric fields.
    """
    # GPU compute utilisation
    gpu_util_avg_pct: Optional[float] = None
    gpu_util_peak_pct: Optional[float] = None
    # VRAM consumption (MiB)
    gpu_mem_avg_mb: Optional[float] = None
    gpu_mem_peak_mb: Optional[float] = None
    # Power draw (Watts)
    power_avg_w: Optional[float] = None
    power_peak_w: Optional[float] = None
    # Time-integrated energy (Joules) via trapezoidal integration of power samples
    energy_j: Optional[float] = None
    # Diagnostics
    num_samples: int = 0
    # "nvidia_smi" | "nvml" | "unavailable"
    collection_method: str = "unavailable"
    # True when a live GPU was successfully polled
    nvml_available: bool = False
    power_supported: bool = False
    # RunMode active when telemetry was collected (informational)
    run_mode: str = RunMode.CLEAN_BENCHMARK


# ── NvidiaSmiTelemetry ────────────────────────────────────────────────────────

class NvidiaSmiTelemetry:
    """
    GPU telemetry via a long-running nvidia-smi subprocess at 1-second intervals.

    Design rationale
    ----------------
    nvidia-smi --loop=1 writes one CSV line per second to stdout. This matches
    the paper's 1-second telemetry sampling style exactly and avoids requiring
    pynvml. A daemon thread reads stdout line-by-line without blocking the
    benchmark loop.

    Parameters
    ----------
    device_index : int
        CUDA device index to query (0-based, corresponds to nvidia-smi GPU ID).
    """

    def __init__(self, device_index: int = 0, run_mode: str = RunMode.CLEAN_BENCHMARK) -> None:
        self._device_index = device_index
        self._run_mode = run_mode
        self._samples: list[TelemetrySample] = []
        self._proc: Optional[subprocess.Popen] = None  # type: ignore[type-arg]
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._power_supported: Optional[bool] = None  # determined at first parse

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def is_available() -> bool:
        """Return True if the nvidia-smi binary is found on PATH."""
        return shutil.which("nvidia-smi") is not None

    def start(self) -> bool:
        """
        Start the nvidia-smi subprocess and background reader thread.

        Returns True on successful launch; False if nvidia-smi is not found,
        the process fails to start, or run_mode is PROFILE_NSYS (suppressed
        to avoid compounding nsys trace overhead with smi IPC overhead).
        """
        if RunMode.is_profiling(self._run_mode):
            logger.info(
                "nvidia-smi telemetry suppressed in run_mode=%r. "
                "GPU metrics will not be collected during Nsight Systems runs.",
                self._run_mode,
            )
            return False

        if not self.is_available():
            logger.warning(
                "nvidia-smi not found on PATH — GPU telemetry unavailable. "
                "Ensure NVIDIA drivers are installed and nvidia-smi is in PATH."
            )
            return False

        self._samples.clear()
        self._stop_event.clear()

        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,power.draw",
            "--format=csv,noheader,nounits",
            f"--id={self._device_index}",
            "--loop=1",
        ]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,          # line-buffered for low latency
                **_SUBPROCESS_FLAGS,
            )
        except OSError as exc:
            logger.error("Failed to start nvidia-smi: %s", exc)
            return False

        self._thread = threading.Thread(
            target=self._reader_loop,
            daemon=True,
            name="nvidia_smi_telemetry",
        )
        self._thread.start()
        logger.debug("nvidia-smi telemetry started (device_index=%d)", self._device_index)
        return True

    def stop(self) -> TelemetrySummary:
        """
        Stop polling and return aggregated statistics.

        Safe to call even when start() was not called or returned False.
        Terminates the nvidia-smi subprocess and joins the reader thread.
        """
        self._stop_event.set()

        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5.0)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None

        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

        logger.debug(
            "nvidia-smi telemetry stopped (%d samples collected)", len(self._samples)
        )
        return self._summarize()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _reader_loop(self) -> None:
        """Background thread: read stdout lines from nvidia-smi until stopped."""
        if self._proc is None or self._proc.stdout is None:
            return
        try:
            for line in self._proc.stdout:
                if self._stop_event.is_set():
                    break
                sample = self._parse_line(line.strip())
                if sample is not None:
                    self._samples.append(sample)
        except ValueError:
            # stdout closed mid-read (process was killed)
            pass

    def _parse_line(self, line: str) -> Optional[TelemetrySample]:
        """
        Parse one CSV line from nvidia-smi.

        Expected format (--format=csv,noheader,nounits):
            "<util_pct>, <mem_mib>, <power_w>"
        e.g.:
            "45, 3124, 125.50"
            "45, 3124, [Not Supported]"

        Returns None if the line is empty or entirely unparseable.
        """
        if not line:
            return None

        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            logger.debug("Skipping malformed nvidia-smi line: %r", line)
            return None

        sample = TelemetrySample(timestamp_s=time.perf_counter())

        # GPU utilisation (%)
        sample.gpu_util_pct = _try_float(parts[0])

        # Memory used (MiB)
        sample.gpu_mem_used_mb = _try_float(parts[1])

        # Power draw (W) — may be "[Not Supported]" on consumer GPUs
        power_raw = parts[2]
        if "[Not Supported]" in power_raw or "N/A" in power_raw:
            if self._power_supported is None:
                self._power_supported = False
                logger.debug(
                    "Power telemetry not supported on this GPU (device_index=%d). "
                    "power_avg_w and energy_j will be None.",
                    self._device_index,
                )
            sample.power_w = None
        else:
            power_val = _try_float(power_raw)
            if power_val is not None and self._power_supported is None:
                self._power_supported = True
            sample.power_w = power_val

        return sample

    def _summarize(self) -> TelemetrySummary:
        """Aggregate all collected samples into a TelemetrySummary."""
        summary = TelemetrySummary(
            num_samples=len(self._samples),
            collection_method="nvidia_smi" if self._samples else "unavailable",
            nvml_available=len(self._samples) > 0,
            power_supported=self._power_supported is True,
            run_mode=self._run_mode,
        )
        if not self._samples:
            return summary

        util_vals = [s.gpu_util_pct for s in self._samples if s.gpu_util_pct is not None]
        mem_vals = [s.gpu_mem_used_mb for s in self._samples if s.gpu_mem_used_mb is not None]
        power_vals = [s.power_w for s in self._samples if s.power_w is not None]

        if util_vals:
            summary.gpu_util_avg_pct = sum(util_vals) / len(util_vals)
            summary.gpu_util_peak_pct = max(util_vals)

        if mem_vals:
            summary.gpu_mem_avg_mb = sum(mem_vals) / len(mem_vals)
            summary.gpu_mem_peak_mb = max(mem_vals)

        if power_vals:
            summary.power_avg_w = sum(power_vals) / len(power_vals)
            summary.power_peak_w = max(power_vals)
            # Trapezoidal integration: ∫ P(t) dt ≈ Σ (P_i + P_{i+1})/2 × Δt
            # Uses samples with valid power readings only.
            valid = [s for s in self._samples if s.power_w is not None]
            energy = 0.0
            for i in range(len(valid) - 1):
                dt = valid[i + 1].timestamp_s - valid[i].timestamp_s
                avg_p = (valid[i].power_w + valid[i + 1].power_w) / 2.0  # type: ignore[operator]
                energy += avg_p * dt
            summary.energy_j = energy if energy > 0 else None

        return summary


# ── NvmlTelemetry ─────────────────────────────────────────────────────────────

class NvmlTelemetry:
    """
    Optional higher-frequency GPU telemetry via pynvml (NVML Python bindings).

    Use this when sub-second granularity is needed. Falls back gracefully
    if pynvml is not installed. At 5 Hz polling, NVML typically provides
    ~200 ms temporal resolution instead of the 1-second nvidia-smi interval.

    Parameters
    ----------
    device_index : int  CUDA device to monitor.
    poll_hz      : float  Polling frequency. Values above ~10 Hz are rarely
                          useful due to NVML counter update latency.
    """

    _nvml_available: Optional[bool] = None  # class-level init cache

    def __init__(self, device_index: int = 0, poll_hz: float = 5.0, run_mode: str = RunMode.CLEAN_BENCHMARK) -> None:
        self._device_index = device_index
        self._run_mode = run_mode
        self._poll_interval_s = 1.0 / max(poll_hz, 0.1)
        self._samples: list[TelemetrySample] = []
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._handle: Optional[object] = None

    @classmethod
    def is_available(cls) -> bool:
        """Return True if pynvml is installed and nvmlInit() succeeds."""
        if cls._nvml_available is None:
            try:
                import pynvml
                pynvml.nvmlInit()
                cls._nvml_available = True
            except Exception:
                cls._nvml_available = False
        return cls._nvml_available  # type: ignore[return-value]

    def start(self) -> bool:
        """Start background polling. Returns False if pynvml is unavailable or
        run_mode is PROFILE_NSYS (suppressed to avoid compounding overhead)."""
        if RunMode.is_profiling(self._run_mode):
            logger.info(
                "NVML telemetry suppressed in run_mode=%r.",
                self._run_mode,
            )
            return False

        self._samples.clear()
        self._stop_event.clear()

        if not self._init_handle():
            return False

        self._thread = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name="nvml_telemetry",
        )
        self._thread.start()
        return True

    def stop(self) -> TelemetrySummary:
        """Stop polling and return aggregated statistics."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        return self._summarize()

    def _init_handle(self) -> bool:
        if not self.is_available():
            return False
        try:
            import pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self._device_index)
            return True
        except Exception:
            return False

    def _poll_once(self) -> TelemetrySample:
        sample = TelemetrySample(timestamp_s=time.perf_counter())
        if self._handle is None:
            return sample
        try:
            import pynvml
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            sample.gpu_util_pct = float(util.gpu)
        except Exception:
            pass
        try:
            import pynvml
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            sample.gpu_mem_used_mb = mem.used / (1024 ** 2)
        except Exception:
            pass
        try:
            import pynvml
            # nvmlDeviceGetPowerUsage returns milliwatts → convert to W
            sample.power_w = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1_000.0
        except Exception:
            pass
        return sample

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            self._samples.append(self._poll_once())
            self._stop_event.wait(timeout=self._poll_interval_s)

    def _summarize(self) -> TelemetrySummary:
        summary = TelemetrySummary(
            num_samples=len(self._samples),
            collection_method="nvml" if self._samples else "unavailable",
            nvml_available=len(self._samples) > 0,
            run_mode=self._run_mode,
        )
        if not self._samples:
            return summary

        util_vals = [s.gpu_util_pct for s in self._samples if s.gpu_util_pct is not None]
        mem_vals = [s.gpu_mem_used_mb for s in self._samples if s.gpu_mem_used_mb is not None]
        power_vals = [s.power_w for s in self._samples if s.power_w is not None]

        if util_vals:
            summary.gpu_util_avg_pct = sum(util_vals) / len(util_vals)
            summary.gpu_util_peak_pct = max(util_vals)
        if mem_vals:
            summary.gpu_mem_avg_mb = sum(mem_vals) / len(mem_vals)
            summary.gpu_mem_peak_mb = max(mem_vals)
        if power_vals:
            summary.power_avg_w = sum(power_vals) / len(power_vals)
            summary.power_peak_w = max(power_vals)
            summary.power_supported = True
            valid = [s for s in self._samples if s.power_w is not None]
            energy = 0.0
            for i in range(len(valid) - 1):
                dt = valid[i + 1].timestamp_s - valid[i].timestamp_s
                avg_p = (valid[i].power_w + valid[i + 1].power_w) / 2.0  # type: ignore[operator]
                energy += avg_p * dt
            summary.energy_j = energy if energy > 0 else None

        return summary


# ── Factory ───────────────────────────────────────────────────────────────────

def create_telemetry(
    device_index: int = 0,
    prefer: str = "nvidia_smi",
    nvml_poll_hz: float = 5.0,
    run_mode: str = RunMode.CLEAN_BENCHMARK,
) -> "NvidiaSmiTelemetry | NvmlTelemetry | _NullTelemetry":
    """
    Return the best available telemetry collector.

    Parameters
    ----------
    device_index : int   CUDA device to monitor.
    prefer       : str   "nvidia_smi" (default, 1 Hz, no deps) or "nvml"
                         (needs pynvml, configurable Hz).
    nvml_poll_hz : float Only used when prefer="nvml".
    run_mode     : str   RunMode constant.  When PROFILE_NSYS, the returned
                         collector's start() will be a no-op to avoid
                         compounding overhead with the nsys tracer.

    Returns
    -------
    NvidiaSmiTelemetry | NvmlTelemetry | _NullTelemetry
        Use the returned object's start() / stop() interface regardless of type.
    """
    if prefer == "nvml" and NvmlTelemetry.is_available():
        logger.debug("Using NvmlTelemetry at %.1f Hz", nvml_poll_hz)
        return NvmlTelemetry(device_index=device_index, poll_hz=nvml_poll_hz, run_mode=run_mode)

    if NvidiaSmiTelemetry.is_available():
        logger.debug("Using NvidiaSmiTelemetry at 1 Hz")
        return NvidiaSmiTelemetry(device_index=device_index, run_mode=run_mode)

    if NvmlTelemetry.is_available():
        logger.debug("nvidia-smi not found; falling back to NvmlTelemetry")
        return NvmlTelemetry(device_index=device_index, poll_hz=nvml_poll_hz, run_mode=run_mode)

    logger.warning(
        "No GPU telemetry backend available. "
        "Install pynvml or ensure nvidia-smi is on PATH for GPU metrics."
    )
    return _NullTelemetry()


class _NullTelemetry:
    """No-op telemetry backend used when no GPU toolkit is available."""

    def start(self) -> bool:
        return False

    def stop(self) -> TelemetrySummary:
        return TelemetrySummary(collection_method="unavailable", nvml_available=False)


# ── Memory bandwidth interface ────────────────────────────────────────────────

@dataclass
class MemoryBandwidthResult:
    """
    Result of a memory bandwidth measurement pass.

    Fields
    ------
    supported   : Whether any measurement was collected.
    gbps        : Effective memory bandwidth in gigabytes per second.
                  None when supported=False.
    backend     : Which collector produced this result.
    notes       : Human-readable explanation of the result or limitation.
    """
    supported: bool = False
    gbps: Optional[float] = None
    backend: str = "null"
    notes: str = ""


class MemoryBandwidthCollector(ABC):
    """
    Abstract interface for memory bandwidth measurement.

    Implementations must follow the start()/stop() pattern used by
    NvidiaSmiTelemetry to allow uniform handling in runner.py.

    Phase 1: use NullMemoryBandwidthCollector (always supported=False).
    Phase 2: implement CuptiMemoryBandwidthCollector (see stub below).
    """

    @abstractmethod
    def start(self) -> None:
        """Begin measurement window."""
        ...

    @abstractmethod
    def stop(self) -> MemoryBandwidthResult:
        """End measurement window and return result."""
        ...


class NullMemoryBandwidthCollector(MemoryBandwidthCollector):
    """
    Phase 1 placeholder — always returns supported=False.

    Used by runner.py as the default. Replace with CuptiMemoryBandwidthCollector
    when CUPTI or Nsight integration is implemented in Phase 2.
    """

    def start(self) -> None:
        pass  # intentional no-op

    def stop(self) -> MemoryBandwidthResult:
        return MemoryBandwidthResult(
            supported=False,
            gbps=None,
            backend="null",
            notes=(
                "Memory bandwidth collection deferred to Phase 2. "
                "Planned integration: CuptiMemoryBandwidthCollector via "
                "NVIDIA CUPTI activity API or Nsight Systems offline profiling."
            ),
        )


class CuptiMemoryBandwidthCollector(MemoryBandwidthCollector):
    """
    Planned Phase 2 implementation using NVIDIA CUPTI or Nsight Systems.

    Integration roadmap
    -------------------
    Option A — CUPTI activity API (in-process):
      1. Use ctypes to load libcupti and call:
           cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY)
           cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL)
      2. Register an activity buffer callback.
      3. Post-process: sum bytes_transferred, divide by kernel_time_s.

    Option B — Nsight Systems offline (subprocess):
      1. Wrap the benchmark run with:
           nsys profile --stats=true --output=report.nsys-rep <benchmark>
      2. Parse the SQLite .nsys-rep output for MemoryBandwidthReport table.

    Option C — nvprof / ncu (NVIDIA Nsight Compute):
      1. ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,... <benchmark>
      2. Parse stdout or JSON output for bandwidth metrics.

    Set self._started = True after start() succeeds so stop() can validate.
    """

    def __init__(self) -> None:
        self._started = False

    def start(self) -> None:
        raise NotImplementedError(
            "CuptiMemoryBandwidthCollector is not yet implemented. "
            "Use NullMemoryBandwidthCollector for Phase 1. "
            "See Phase 2 implementation roadmap in the docstring."
        )

    def stop(self) -> MemoryBandwidthResult:
        raise NotImplementedError(
            "CuptiMemoryBandwidthCollector is not yet implemented."
        )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _try_float(s: str) -> Optional[float]:
    """
    Convert a string to float, returning None on any failure.

    Handles the "[Not Supported]", "N/A", and empty-string cases that
    nvidia-smi may emit for unsupported sensors.
    """
    try:
        return float(s)
    except (ValueError, TypeError):
        return None
