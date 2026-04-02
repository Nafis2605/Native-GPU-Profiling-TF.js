"""
Self-test for native-tfjs-bench timing and telemetry infrastructure.

Validates the measurement layer using a fake workload (time.sleep) that
requires no ML framework, no GPU, and no model weights. It is intentionally
lightweight: if this passes, the timing and telemetry primitives are wired
correctly and the benchmark runner can trust the numbers it produces.

Run from the project root:
    python -m benchmark.self_test           # summary output
    python -m benchmark.self_test --verbose # detailed per-test output

Exit codes
----------
    0  All tests passed.
    1  One or more tests failed.

Tests
-----
1. timing_wall_clock   – WallClockTimer correctly measures a known-duration sleep.
2. timing_cuda_event   – CudaEventTimer: CUDA path exercised when available, or
                         fallback/wall_clock path exercised otherwise.
3. timing_measure_inference – measure_inference() returns wall >= kernel.
4. timing_trial_timer  – TrialTimer populates all phases correctly.
5. telemetry_smi       – NvidiaSmiTelemetry collects ≥ 2 samples in 3 s (skipped
                         if nvidia-smi is not on PATH).
6. telemetry_null_bw   – NullMemoryBandwidthCollector returns supported=False.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# ── Fake workload ─────────────────────────────────────────────────────────────

_FAKE_WORKLOAD_S = 0.005  # 5 ms target latency


def _fake_workload(duration_s: float = _FAKE_WORKLOAD_S) -> None:
    """Simulate a fixed-duration inference call via time.sleep."""
    time.sleep(duration_s)


# ── Test result container ─────────────────────────────────────────────────────

@dataclass
class _TestResult:
    name: str
    passed: bool
    skipped: bool = False
    details: str = ""
    error: Optional[str] = None


# ── Individual tests ──────────────────────────────────────────────────────────

def _test_wall_clock(verbose: bool) -> _TestResult:
    """WallClockTimer measures a known-duration sleep within tolerance."""
    from benchmark.timing import WallClockTimer

    target_ms = _FAKE_WORKLOAD_S * 1_000.0  # 5.0 ms
    tolerance_ms = 4.0                       # Windows timer resolution is ~1–2 ms

    # Context manager path
    with WallClockTimer() as t:
        _fake_workload()
    cm_elapsed = t.elapsed_ms

    # Manual start/stop path
    t2 = WallClockTimer()
    t2.start()
    _fake_workload()
    manual_elapsed = t2.stop()

    low = target_ms
    high = target_ms + tolerance_ms

    cm_ok = low <= cm_elapsed <= (target_ms + tolerance_ms * 3)
    manual_ok = manual_elapsed >= low

    passed = cm_ok and manual_ok
    details = (
        f"context-manager: {cm_elapsed:.2f} ms  (expected ≥ {low:.1f} ms)  "
        f"manual: {manual_elapsed:.2f} ms  passed={passed}"
    )
    return _TestResult(name="timing_wall_clock", passed=passed, details=details)


def _test_cuda_event(verbose: bool) -> _TestResult:
    """CudaEventTimer: exercises CUDA path or confirms graceful fallback."""
    from benchmark.timing import CudaEventTimer, WallClockTimer, _cuda_available

    with WallClockTimer() as wall:
        with CudaEventTimer() as cuda:
            _fake_workload()

    # kernel time must be non-negative and ≤ wall time
    if cuda.elapsed_ms < 0:
        return _TestResult(
            name="timing_cuda_event",
            passed=False,
            details=f"elapsed_ms={cuda.elapsed_ms:.4f} ms (negative, unexpected)",
        )
    if cuda.elapsed_ms > wall.elapsed_ms + 1.0:
        return _TestResult(
            name="timing_cuda_event",
            passed=False,
            details=(
                f"cuda.elapsed_ms ({cuda.elapsed_ms:.2f} ms) > "
                f"wall.elapsed_ms ({wall.elapsed_ms:.2f} ms) + 1 ms"
            ),
        )

    cuda_avail = _cuda_available()
    if cuda_avail and not cuda.is_cuda_backed:
        return _TestResult(
            name="timing_cuda_event",
            passed=False,
            details="CUDA is available but CudaEventTimer fell back to wall clock",
        )
    if not cuda_avail and cuda.is_cuda_backed:
        return _TestResult(
            name="timing_cuda_event",
            passed=False,
            details="CUDA unavailable but CudaEventTimer claims cuda-backed",
        )

    backend = "cuda_events" if cuda.is_cuda_backed else "wall_clock_fallback"
    details = (
        f"backend={backend}  wall={wall.elapsed_ms:.2f} ms  "
        f"kernel={cuda.elapsed_ms:.2f} ms"
    )
    return _TestResult(name="timing_cuda_event", passed=True, details=details)


def _test_measure_inference(verbose: bool) -> _TestResult:
    """measure_inference() returns wall_ms >= kernel_ms for a fake workload."""
    from benchmark.timing import measure_inference

    results = []
    for _ in range(5):
        wall_ms, kernel_ms = measure_inference(_fake_workload, cuda=True)
        results.append((wall_ms, kernel_ms))

    failures = [
        (w, k) for w, k in results
        if not (w >= 0 and k >= 0 and w >= k - 0.5)  # 0.5 ms float tolerance
    ]
    if failures:
        return _TestResult(
            name="timing_measure_inference",
            passed=False,
            details=f"wall < kernel on {len(failures)}/5 samples: {failures}",
        )

    mean_wall = sum(w for w, _ in results) / len(results)
    mean_kernel = sum(k for _, k in results) / len(results)
    details = f"mean wall={mean_wall:.2f} ms  mean kernel={mean_kernel:.2f} ms  (5 samples)"
    return _TestResult(
        name="timing_measure_inference", passed=True, details=details
    )


def _test_trial_timer(verbose: bool) -> _TestResult:
    """TrialTimer populates all phase fields from fake workloads."""
    from benchmark.timing import TrialTimer

    timer = TrialTimer(use_cuda=True)

    with timer.phase_model_load():
        _fake_workload(duration_s=0.010)  # 10 ms simulated load

    with timer.phase_warmup():
        timer.measure_first_inference(_fake_workload)
        for _ in range(4):
            _fake_workload()

    iters_wall, iters_kernel = [], []
    for _ in range(10):
        w, k = timer.measure_iteration(_fake_workload)
        iters_wall.append(w)
        iters_kernel.append(k)

    errors = []
    if not (timer.phases.model_load_ms >= 8.0):
        errors.append(
            f"model_load_ms={timer.phases.model_load_ms:.2f} ms, expected ≥ 8 ms"
        )
    if not (timer.phases.first_inference_ms >= _FAKE_WORKLOAD_S * 800):
        errors.append(
            f"first_inference_ms={timer.phases.first_inference_ms:.2f} ms, expected ≥ 4 ms"
        )
    if not (timer.phases.warmup_total_ms >= timer.phases.first_inference_ms):
        errors.append(
            f"warmup_total_ms ({timer.phases.warmup_total_ms:.2f}) < "
            f"first_inference_ms ({timer.phases.first_inference_ms:.2f})"
        )
    if len(iters_wall) != 10:
        errors.append(f"expected 10 iteration timings, got {len(iters_wall)}")
    if timer.phases.kernel_timer_backend not in ("cuda_events", "wall_clock_fallback"):
        errors.append(
            f"unexpected kernel_timer_backend: {timer.phases.kernel_timer_backend!r}"
        )

    if errors:
        return _TestResult(
            name="timing_trial_timer",
            passed=False,
            details="; ".join(errors),
        )

    mean_wall = sum(iters_wall) / len(iters_wall)
    details = (
        f"model_load={timer.phases.model_load_ms:.1f} ms  "
        f"first_inference={timer.phases.first_inference_ms:.1f} ms  "
        f"warmup_total={timer.phases.warmup_total_ms:.1f} ms  "
        f"mean_iter_wall={mean_wall:.2f} ms  "
        f"backend={timer.phases.kernel_timer_backend}"
    )
    return _TestResult(name="timing_trial_timer", passed=True, details=details)


def _test_telemetry_smi(verbose: bool) -> _TestResult:
    """
    NvidiaSmiTelemetry collects ≥ 2 samples during a 3-second window.

    SKIPPED when nvidia-smi is not on PATH (graceful degradation).
    """
    from benchmark.telemetry import NvidiaSmiTelemetry

    if not NvidiaSmiTelemetry.is_available():
        return _TestResult(
            name="telemetry_smi",
            passed=True,
            skipped=True,
            details="nvidia-smi not found on PATH; telemetry test skipped",
        )

    telem = NvidiaSmiTelemetry(device_index=0)
    started = telem.start()

    if not started:
        return _TestResult(
            name="telemetry_smi",
            passed=False,
            details="NvidiaSmiTelemetry.start() returned False despite nvidia-smi being on PATH",
        )

    # Run a fake workload for ~3 seconds to collect ≥ 2 one-second samples
    _fake_workload(duration_s=3.1)

    summary = telem.stop()

    errors = []
    if summary.num_samples < 2:
        errors.append(
            f"Expected ≥ 2 samples after 3 s, got {summary.num_samples}"
        )
    if summary.collection_method != "nvidia_smi":
        errors.append(
            f"Expected collection_method='nvidia_smi', got {summary.collection_method!r}"
        )
    if not summary.nvml_available:
        errors.append("nvml_available=False despite samples being collected")

    if errors:
        return _TestResult(
            name="telemetry_smi",
            passed=False,
            details=(
                f"num_samples={summary.num_samples}  "
                f"method={summary.collection_method!r}  "
                f"gpu_util_avg={summary.gpu_util_avg_pct}  "
                f"errors: {'; '.join(errors)}"
            ),
        )

    details = (
        f"num_samples={summary.num_samples}  "
        f"gpu_util_avg={summary.gpu_util_avg_pct:.1f}%  "
        f"mem_avg={summary.gpu_mem_avg_mb:.0f} MiB  "
        f"power={'N/A' if summary.power_avg_w is None else f'{summary.power_avg_w:.1f} W'}  "
        f"power_supported={summary.power_supported}"
    )
    return _TestResult(name="telemetry_smi", passed=True, details=details)


def _test_null_memory_bandwidth(verbose: bool) -> _TestResult:
    """NullMemoryBandwidthCollector always returns supported=False."""
    from benchmark.telemetry import NullMemoryBandwidthCollector

    collector = NullMemoryBandwidthCollector()
    collector.start()
    result = collector.stop()

    if result.supported:
        return _TestResult(
            name="telemetry_null_bw",
            passed=False,
            details=f"Expected supported=False, got supported={result.supported}",
        )
    if result.gbps is not None:
        return _TestResult(
            name="telemetry_null_bw",
            passed=False,
            details=f"Expected gbps=None, got gbps={result.gbps}",
        )

    details = f"supported=False  backend={result.backend!r}  notes snippet: {result.notes[:60]!r}"
    return _TestResult(name="telemetry_null_bw", passed=True, details=details)


# ── Test runner ───────────────────────────────────────────────────────────────

_ALL_TESTS: list[Callable[[bool], _TestResult]] = [
    _test_wall_clock,
    _test_cuda_event,
    _test_measure_inference,
    _test_trial_timer,
    _test_telemetry_smi,
    _test_null_memory_bandwidth,
]


def run_full_self_test(verbose: bool = False) -> bool:
    """
    Execute all self-tests and print a pass/fail summary.

    Returns True when every non-skipped test passes; False otherwise.
    """
    print("\nnative-tfjs-bench — timing & telemetry self-test")
    print("=" * 56)

    results: list[_TestResult] = []
    for test_fn in _ALL_TESTS:
        name = test_fn.__name__.lstrip("_test_")
        sys.stdout.write(f"  {name:<34}")
        sys.stdout.flush()
        try:
            result = test_fn(verbose)
        except Exception as exc:
            result = _TestResult(
                name=name,
                passed=False,
                error=f"{type(exc).__name__}: {exc}",
            )

        if result.skipped:
            status_str = "SKIP"
        elif result.passed:
            status_str = "PASS"
        else:
            status_str = "FAIL"

        print(status_str)

        if verbose or not result.passed:
            if result.details:
                print(f"         {result.details}")
            if result.error:
                print(f"         ERROR: {result.error}")

        results.append(result)

    # Summary
    n_pass = sum(1 for r in results if r.passed and not r.skipped)
    n_fail = sum(1 for r in results if not r.passed and not r.skipped)
    n_skip = sum(1 for r in results if r.skipped)
    total_run = len(results) - n_skip

    print("=" * 56)
    print(f"  Results: {n_pass}/{total_run} passed", end="")
    if n_skip:
        print(f"  ({n_skip} skipped)", end="")
    print()

    if n_fail:
        print("  FAILED tests:")
        for r in results:
            if not r.passed and not r.skipped:
                print(f"    - {r.name}: {r.details or r.error or '(no details)'}")

    overall = n_fail == 0
    print(f"  Overall: {'PASS' if overall else 'FAIL'}\n")
    return overall


# ── CLI entry-point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run native-tfjs-bench timing and telemetry self-tests."
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed output for every test, not just failures.",
    )
    args = parser.parse_args()
    passed = run_full_self_test(verbose=args.verbose)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
