"""
Timing utilities for native-tfjs-bench.

Two low-level timers
--------------------
  WallClockTimer  – Host-side high-resolution timer (time.perf_counter).
                    On Windows this wraps QueryPerformanceCounter, giving
                    sub-microsecond resolution with no external dependencies.

  CudaEventTimer  – GPU-side kernel timer via torch.cuda.Event.
                    Measures only what the GPU hardware counters report for
                    kernel execution, excluding host launch overhead and PCIe
                    transfer time. Falls back to WallClockTimer when CUDA is
                    unavailable. Never raises an ImportError.

CUDA synchronisation strategy
------------------------------
Both timers are always nested with WallClockTimer *outside*:

    with WallClockTimer() as wall:       # host clock starts
        with CudaEventTimer() as cuda:   # GPU start event recorded
            model.infer(inputs)          # async kernel launch
        # CudaEventTimer.__exit__:
        #   records GPU end event
        #   calls torch.cuda.synchronize()  ← GPU finishes *inside* wall
        #   reads hardware elapsed time
    # WallClockTimer.__exit__ runs now → includes full GPU round-trip

    wall.elapsed_ms  = true end-to-end host latency (includes sync wait)
    cuda.elapsed_ms  = pure GPU kernel execution time (hardware counter)
    wall − cuda      ≈ host overhead + PCIe + sync latency

This ensures wall_clock_ms is the latency a caller observes, not just the
time to *submit* work to the GPU.

Higher-level helpers
--------------------
  TimingPhases  – Named dataclass holding all per-trial timing slices.
  TrialTimer    – Context-manager orchestrator that populates TimingPhases
                  incrementally as each phase completes.
  measure_inference – One-shot function returning (wall_ms, kernel_ms).
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Generator, Optional

# ── Module-level CUDA availability cache ─────────────────────────────────────
# Queried once and cached to avoid repeated torch imports inside hot loops.

_CUDA_AVAILABLE: Optional[bool] = None


def _cuda_available() -> bool:
    """
    Return True if a CUDA-capable GPU is accessible via PyTorch.

    Caches the result after the first call. Thread-safe for reads; the small
    risk of a double-check on the first call is harmless (same result).
    """
    global _CUDA_AVAILABLE
    if _CUDA_AVAILABLE is None:
        try:
            import torch
            _CUDA_AVAILABLE = torch.cuda.is_available()
        except ImportError:
            _CUDA_AVAILABLE = False
    return _CUDA_AVAILABLE


# ── Named phase container ─────────────────────────────────────────────────────

@dataclass
class TimingPhases:
    """
    Named wall-clock measurements for the distinct phases of a single trial.

    Populated incrementally by TrialTimer as each phase completes.
    All values are milliseconds unless the field note says otherwise.

    Fields
    ------
    model_load_ms        Wall-clock from process start to model weights
                         resident on GPU (includes driver init, disk I/O,
                         CUDA graph optimisation).
    first_inference_ms   Wall-clock for the very first inference call
                         (first warm-up iteration). Captures cold-start
                         dispatch latency before pipelines are warm.
    warmup_total_ms      Total wall-clock across all warm-up iterations.
    kernel_timer_backend Which backend supplied kernel_ms in the measured
                         phase: "cuda_events" or "wall_clock_fallback".
    """
    model_load_ms: float = 0.0
    first_inference_ms: float = 0.0
    warmup_total_ms: float = 0.0
    kernel_timer_backend: str = "undetermined"


# ── WallClockTimer ────────────────────────────────────────────────────────────

class WallClockTimer:
    """
    High-resolution host-side timer with no CUDA dependency.

    time.perf_counter() on Windows wraps QueryPerformanceCounter
    (≈ 100 ns resolution). Safe to use without a GPU or torch installed.

    Context manager or manual start/stop are both supported:

        with WallClockTimer() as t:
            do_work()
        print(t.elapsed_ms)    # ms since __enter__

        t = WallClockTimer()
        t.start()
        do_work()
        ms = t.stop()          # also stored in t.elapsed_ms
    """

    __slots__ = ("_start", "_end", "elapsed_ms")

    def __init__(self) -> None:
        self._start: float = 0.0
        self._end: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "WallClockTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        self._end = time.perf_counter()
        self.elapsed_ms = (self._end - self._start) * 1_000.0

    def start(self) -> None:
        """Begin timing. Resets any previous measurement."""
        self._start = time.perf_counter()

    def stop(self) -> float:
        """
        Stop timing and return elapsed milliseconds.
        Also stored in self.elapsed_ms for later access.
        """
        self._end = time.perf_counter()
        self.elapsed_ms = (self._end - self._start) * 1_000.0
        return self.elapsed_ms


# ── CudaEventTimer ────────────────────────────────────────────────────────────

class CudaEventTimer:
    """
    GPU-side kernel execution timer via torch.cuda.Event.

    Records cudaEventRecord() before and after the timed code block, then
    calls torch.cuda.synchronize() to flush the device and read the hardware
    elapsed-time counter via cudaEventElapsedTime().

    Always nest this INSIDE a WallClockTimer so synchronize() is still within
    the wall-clock window (see module docstring for rationale).

    Fallback: if CUDA is unavailable the timer silently falls back to
    WallClockTimer. Check is_cuda_backed to see which path was taken.
    """

    # Class-level cache — shared across all instances
    _cls_cuda_available: Optional[bool] = None

    def __init__(self) -> None:
        self.elapsed_ms: float = 0.0
        self._fallback: Optional[WallClockTimer] = None
        self._start_event: Any = None
        self._end_event: Any = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _has_cuda(cls) -> bool:
        if cls._cls_cuda_available is None:
            cls._cls_cuda_available = _cuda_available()
        return cls._cls_cuda_available

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "CudaEventTimer":
        if self._has_cuda():
            import torch
            # enable_timing=True is required; without it elapsed_time() raises
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._fallback = None
            self._start_event.record()
        else:
            self._fallback = WallClockTimer()
            self._fallback.start()
        return self

    def __exit__(self, *_: object) -> None:
        if self._has_cuda() and self._start_event is not None:
            import torch
            self._end_event.record()
            # Synchronize BEFORE reading elapsed_time. This blocks the host
            # until the GPU has processed all preceding ops, including the
            # end_event record. The enclosing WallClockTimer stops AFTER
            # us, so it correctly captures this wait.
            torch.cuda.synchronize()
            self.elapsed_ms = self._start_event.elapsed_time(self._end_event)
        elif self._fallback is not None:
            self.elapsed_ms = self._fallback.stop()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_cuda_backed(self) -> bool:
        """True when CUDA Events drove this measurement (not the wall fallback)."""
        return self._has_cuda() and self._fallback is None


# ── measure_inference ─────────────────────────────────────────────────────────

def measure_inference(
    fn: Callable,
    *args: Any,
    cuda: bool = True,
    **kwargs: Any,
) -> tuple[float, float]:
    """
    Time a single call to fn(*args, **kwargs).

    Parameters
    ----------
    fn:      Callable to invoke once (typically model.infer).
    cuda:    If True and CUDA is available, also measure GPU kernel time via
             CudaEventTimer. If False, kernel_ms == wall_clock_ms.
    *args, **kwargs: Forwarded verbatim to fn.

    Returns
    -------
    (wall_clock_ms, kernel_ms) : tuple[float, float]
        wall_clock_ms – end-to-end host latency, including GPU synchronize()
                        when CUDA is in use. This is what a caller perceives.
        kernel_ms     – pure GPU kernel time from hardware counters.
                        Equals wall_clock_ms when CUDA events are not used.

    Note: the nesting order is intentional.
          wall wraps cuda so synchronize() runs before wall stops.
    """
    use_cuda = cuda and _cuda_available()

    wall = WallClockTimer()
    kernel: CudaEventTimer | WallClockTimer = (
        CudaEventTimer() if use_cuda else WallClockTimer()
    )

    with wall:          # host clock starts
        with kernel:    # GPU event (or fallback wall) starts
            fn(*args, **kwargs)
        # kernel.__exit__: record end event, synchronize, compute elapsed_ms

    # wall.__exit__: host clock stops (after synchronize is done)
    return wall.elapsed_ms, kernel.elapsed_ms


# ── TrialTimer ────────────────────────────────────────────────────────────────

class TrialTimer:
    """
    Orchestrate and record all timing phases for a single benchmark trial.

    Populates a TimingPhases instance incrementally as each phase completes.
    Used by runner.py to keep timing logic out of the execution loop.

    Usage
    -----
        timer = TrialTimer(use_cuda=True)

        with timer.phase_model_load():
            model.load(device="cuda")
        # timer.phases.model_load_ms is now set

        with timer.phase_warmup():
            wall, k = timer.measure_first_inference(model.infer, inputs)
            for _ in range(warmup_iters - 1):
                model.infer(inputs)           # remaining warmups, not timed
        # timer.phases.warmup_total_ms and first_inference_ms are set

        wall_times, kernel_times = [], []
        for _ in range(measured_iters):
            w, k = timer.measure_iteration(model.infer, inputs)
            wall_times.append(w)
            kernel_times.append(k)
    """

    def __init__(self, use_cuda: bool = True) -> None:
        self._use_cuda: bool = use_cuda and _cuda_available()
        self.phases = TimingPhases(
            kernel_timer_backend=(
                "cuda_events" if self._use_cuda else "wall_clock_fallback"
            )
        )

    # ------------------------------------------------------------------
    # Phase context managers
    # ------------------------------------------------------------------

    @contextmanager
    def phase_model_load(self) -> Generator[WallClockTimer, None, None]:
        """
        Measure wall-clock time for model loading.

        Yields the active WallClockTimer in case the caller wants to read
        intermediate progress (rarely needed). On exit, populates
        self.phases.model_load_ms.
        """
        t = WallClockTimer()
        t.start()
        try:
            yield t
        finally:
            self.phases.model_load_ms = t.stop()

    @contextmanager
    def phase_warmup(self) -> Generator[None, None, None]:
        """
        Measure total wall-clock time across all warm-up iterations.

        On exit, populates self.phases.warmup_total_ms.
        Use measure_first_inference() for the first iteration to also
        capture self.phases.first_inference_ms.
        """
        t = WallClockTimer()
        t.start()
        try:
            yield
        finally:
            self.phases.warmup_total_ms = t.stop()

    # ------------------------------------------------------------------
    # Per-iteration measurement
    # ------------------------------------------------------------------

    def measure_first_inference(
        self, fn: Callable, *args: Any, **kwargs: Any
    ) -> tuple[float, float]:
        """
        Measure and record the first warm-up inference call.

        Captures the cold-start dispatch latency before CUDA pipelines are
        fully warmed up. Populates self.phases.first_inference_ms.

        Returns (wall_clock_ms, kernel_ms). Call once during phase_warmup.
        """
        wall_ms, kernel_ms = measure_inference(
            fn, *args, cuda=self._use_cuda, **kwargs
        )
        self.phases.first_inference_ms = wall_ms
        return wall_ms, kernel_ms

    def measure_iteration(
        self, fn: Callable, *args: Any, **kwargs: Any
    ) -> tuple[float, float]:
        """
        Measure one iteration in the measured (post-warmup) phase.

        Returns (wall_clock_ms, kernel_ms). Accumulate in caller lists.
        """
        return measure_inference(fn, *args, cuda=self._use_cuda, **kwargs)
