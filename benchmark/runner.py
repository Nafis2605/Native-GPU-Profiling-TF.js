"""
Single-trial benchmark runner for native-tfjs-bench.

This module is the execution core. It runs inside a fresh subprocess
(spawned by trial_manager.py) and owns the complete trial lifecycle
for one model:

  Phase 1 - Model loading       (timed: model_load_ms)
  Phase 2 - 1st-call measurement (exactly one timed inference; records
                                  first_call_ms; captures CUDA JIT
                                  compilation and cuDNN autotuning;
                                  followed by explicit GPU sync barrier)
  Phase 3 - Measurement         (1,024 full inference runs; wall + kernel
                                  timers active; GPU telemetry scoped here)
  Phase 4 - Cleanup & output    (CSV + JSON emitted to output_dir)

1st-call protocol
-----------------
Exactly one timed inference is run immediately after model loading.  This
call bears the one-time cost of CUDA JIT compilation and cuDNN algorithm
selection.  It is stored as first_call_ms and is NOT counted in the measured
inference statistics.  After this single call, torch.cuda.synchronize() +
empty_cache() drain any residual GPU state so no JIT kernel overlaps the
first measured iteration.

Non-critical failures are handled locally:
  - NotImplementedError on load -> STATUS_UNSUPPORTED (not a crash)
  - RuntimeError on load        -> STATUS_FAILED, skip model
  - Per-iteration exception     -> mark iteration failed; abort after 5 consecutive

The subprocess writes a result JSON file that trial_manager.py reads back
to collect the TrialResult without shared memory.
"""

from __future__ import annotations

import csv
import gc
import logging
import time
import traceback
from pathlib import Path
from typing import Optional

from benchmark.models.base import BaseModel
from benchmark.models.registry import get_model
from benchmark.result_schema import (
    STATUS_FAILED,
    STATUS_SUCCESS,
    STATUS_UNSUPPORTED,
    RunMode,
    TrialResult,
    write_trial_result_csv,
    write_trial_result_json,
)
from benchmark.telemetry import NvidiaSmiTelemetry
from benchmark.timing import WallClockTimer, measure_inference
from benchmark.utils import compute_statistics, set_global_seed

logger = logging.getLogger(__name__)

# How many consecutive iteration failures trigger an early abort
_CONSECUTIVE_FAILURE_LIMIT = 5


# ── Cache and memory clearing utilities ──────────────────────────────────────

def _clear_gpu_cache():
    """Clear GPU cache and synchronize to ensure no residual state."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("GPU cache cleared and synchronized")
    except Exception as e:
        logger.debug(f"Could not clear GPU cache: {e}")


def _clear_memory():
    """Clear CPU memory via garbage collection."""
    try:
        gc.collect()
        logger.debug("CPU memory cleared via garbage collection")
    except Exception as e:
        logger.debug(f"Could not clear CPU memory: {e}")


# ── Public entry point ────────────────────────────────────────────────────────

def run_trial(
    model_id: int,
    trial_id: int,
    output_dir: Path,
    measured_iterations: int = 1024,
    random_seed: int = 12345,
    device: str = "cuda",
    nvml_poll_hz: float = 5.0,
    run_mode: str = RunMode.CLEAN_BENCHMARK,
) -> TrialResult:
    """
    Execute one complete benchmark trial for a single model.

    This function is designed to be called once per fresh subprocess.
    It emits per-iteration CSV and trial-summary JSON to output_dir,
    then returns the TrialResult.

    Args:
        model_id:            Registry ID of the model to benchmark (1-10).
        trial_id:            0-based trial index (0-4).
        output_dir:          Directory for output artefacts (created if absent).
        measured_iterations: Number of timed forward passes (default 1024).
        random_seed:         Seed for deterministic input generation.
        device:              CUDA device string, e.g. "cuda" or "cuda:0".
        nvml_poll_hz:        NVML polling frequency during measured phase.
        run_mode:            RunMode constant controlling profiler behaviour.
                             PROFILE_NSYS and PROFILE_NCU suppress telemetry
                             and emit an overhead warning.  Timing values
                             produced in any non-CLEAN_BENCHMARK mode must NOT
                             be used for latency reporting.

    Returns:
        TrialResult populated with all captured metrics.
    """
    set_global_seed(random_seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    if RunMode.is_profiling(run_mode):
        logger.warning(
            "run_trial called with run_mode=%r. "
            "Profiler overhead inflates all wall-clock and kernel timings. "
            "Values produced in this run MUST NOT be used for latency reporting. "
            "Telemetry polling (nvidia-smi / NVML) is suppressed.",
            run_mode,
        )

    device_name, cuda_version, driver_version = _query_device_info()

    # ── Instantiate model (registry lookup only, no loading yet) ──────────
    try:
        model = get_model(model_id)
    except KeyError as exc:
        logger.error("Unknown model_id=%d: %s", model_id, exc)
        return _error_result(
            model_id=model_id, trial_id=trial_id, status=STATUS_FAILED,
            error=str(exc), device_name=device_name,
            cuda_version=cuda_version, driver_version=driver_version,
        )

    meta = model.get_metadata()
    logger.info(
        "Trial %d | Model %d (%s) | Framework: %s",
        trial_id, model_id, meta["model_name"], meta["native_framework"],
    )

    # ── Phase 1: Model loading ────────────────────────────────────────────
    model_load_ms = 0.0
    try:
        with WallClockTimer() as t:
            model.load(device=device)
        model_load_ms = t.elapsed_ms
        logger.info("  Load complete in %.2f ms", model_load_ms)

        # Clear GPU and CPU cache after model loading
        _clear_gpu_cache()
        _clear_memory()
        time.sleep(0.5)  # Brief pause for system stabilization

    except NotImplementedError as exc:
        logger.warning("  Model %d not yet implemented: %s", model_id, exc)
        result = _model_result(model, trial_id, device_name, cuda_version, driver_version)
        result.status = STATUS_UNSUPPORTED
        result.error_message = str(exc)
        _emit_outputs(result, output_dir)
        return result

    except Exception as exc:
        logger.error("  Model %d load failed: %s", model_id, exc)
        logger.debug(traceback.format_exc())
        result = _model_result(model, trial_id, device_name, cuda_version, driver_version)
        result.model_load_ms = model_load_ms
        result.status = STATUS_FAILED
        result.error_message = f"Load failed: {exc}"
        _emit_outputs(result, output_dir)
        return result

    # ── Pre-generate inputs (before measuring — avoids I/O jitter) ────────
    inputs = model.generate_input(seed=random_seed)

    # =========================================================================
    # Phase 2: 1st-Call Measurement -- exactly one timed inference
    # =========================================================================
    # Run one timed inference immediately after model loading and input
    # generation.  This captures CUDA JIT compilation, cuDNN algorithm
    # selection, and any first-use memory allocation overhead.
    # The result is stored as first_call_ms and is never mixed into the
    # steady-state inference statistics collected in Phase 3.
    # =========================================================================
    first_call_ms = 0.0
    logger.info("  [Phase 2] 1st-call: single timed inference (first-use measurement)")
    try:
        wall_ms, _kernel_ms = measure_inference(model.infer, inputs, cuda=True)
        first_call_ms = wall_ms
        logger.info("  [Phase 2] 1st-call complete: %.2f ms", first_call_ms)
    except Exception as exc:
        logger.error("  Model %d 1st-call failed: %s", model_id, exc)
        result = _model_result(model, trial_id, device_name, cuda_version, driver_version)
        result.model_load_ms = model_load_ms
        result.status = STATUS_FAILED
        result.error_message = f"1st-call failed: {exc}"
        _emit_outputs(result, output_dir)
        model.cleanup()
        return result

    # -- Phase 2 / Phase 3 barrier -------------------------------------------
    # Drain all GPU work from the 1st-call before the measured loop begins.
    # Guarantees no JIT or initialization kernel overlaps the first measured
    # iteration, and no 1st-call state contaminates Phase 3 timing.
    _clear_gpu_cache()
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            _torch.cuda.synchronize()
    except Exception:
        pass

    # =========================================================================
    # Phase 3: Measurement — 1,024 independent full inference runs
    # =========================================================================
    # The measured loop starts only after all warmup runs have completed and
    # the GPU has been fully synchronised.  Each iteration uses the identical
    # measure_inference call: wall clock wraps a CUDA Event timer, model.infer
    # is called, torch.cuda.synchronize() drains the device, and both wall_ms
    # and kernel_ms are appended to wall_times / kernel_times.  Warmup times
    # are never written to these lists.
    #
    # Telemetry (nvidia-smi) starts here — scoped to Phase 3 only — so GPU
    # utilisation, memory, power and energy reflect the measured phase alone.
    # =========================================================================
    telemetry = NvidiaSmiTelemetry(device_index=0, run_mode=run_mode)
    telemetry.start()
    # Brief pause so the nvidia-smi subprocess emits at least one sample
    # before the first measured iteration starts.
    time.sleep(1.1)

    logger.info(
        "  [Phase 3] Measurement: running %d full inference passes (timed)",
        measured_iterations,
    )

    wall_times: list[float] = []
    kernel_times: list[float] = []
    consecutive_failures = 0

    for i in range(measured_iterations):
        try:
            wall_ms, kernel_ms = measure_inference(model.infer, inputs, cuda=True)
            wall_times.append(wall_ms)
            kernel_times.append(kernel_ms)
            consecutive_failures = 0
        except Exception as exc:
            logger.warning("  Iteration %d failed: %s", i, exc)
            consecutive_failures += 1
            if consecutive_failures >= _CONSECUTIVE_FAILURE_LIMIT:
                logger.error(
                    "  %d consecutive failures at iter %d — aborting measured phase",
                    _CONSECUTIVE_FAILURE_LIMIT, i,
                )
                break

    telem = telemetry.stop()
    logger.info(
        "  [Phase 3] Measurement complete: %d/%d iters recorded",
        len(wall_times), measured_iterations,
    )

    # -- Aggregate statistics ------------------------------------------------
    wall_stats = compute_statistics(wall_times)
    kernel_stats = compute_statistics(kernel_times)

    _write_iteration_csv(
        output_dir=output_dir,
        model_id=model_id,
        model_name=meta["model_name"],
        wall_times=wall_times,
        kernel_times=kernel_times,
    )

    # ── Assemble TrialResult ───────────────────────────────────────────────
    result = _model_result(model, trial_id, device_name, cuda_version, driver_version)

    result.run_mode = run_mode
    result.model_load_ms = model_load_ms
    result.first_call_ms = first_call_ms

    result.measured_iterations = len(wall_times)

    result.mean_inference_ms = wall_stats["mean"]
    result.std_inference_ms = wall_stats["std"]
    result.p50_inference_ms = wall_stats["p50"]
    result.p95_inference_ms = wall_stats["p95"]
    result.p99_inference_ms = wall_stats["p99"]
    result.min_inference_ms = wall_stats["min"]
    result.max_inference_ms = wall_stats["max"]
    result.mean_end_to_end_ms = wall_stats["mean"]

    result.mean_kernel_ms = kernel_stats["mean"]
    result.std_kernel_ms = kernel_stats["std"]
    result.p95_kernel_ms = kernel_stats["p95"]

    result.gpu_util_avg_pct = telem.gpu_util_avg_pct
    result.gpu_util_peak_pct = telem.gpu_util_peak_pct
    result.gpu_mem_avg_mb = telem.gpu_mem_avg_mb
    result.gpu_mem_peak_mb = telem.gpu_mem_peak_mb
    result.power_avg_w = telem.power_avg_w
    result.power_peak_w = telem.power_peak_w
    result.energy_j = telem.energy_j
    result.mem_bandwidth_supported = False  # Deferred to Phase 2

    completed = len(wall_times)
    if completed == measured_iterations:
        result.status = STATUS_SUCCESS
    else:
        result.status = STATUS_FAILED
        result.error_message = (
            f"Only {completed}/{measured_iterations} iterations completed"
        )

    model.cleanup()
    _emit_outputs(result, output_dir)

    logger.info(
        "  Done | 1st-call=%.2f ms | mean=%.2f ms | p95=%.2f ms | kernel=%.2f ms | status=%s",
        result.first_call_ms, result.mean_inference_ms, result.p95_inference_ms,
        result.mean_kernel_ms, result.status,
    )
    return result


# ── Internal helpers ──────────────────────────────────────────────────────────

def _query_device_info() -> tuple[str, str, str]:
    """Return (device_name, cuda_version, driver_version). Never raises."""
    device_name = "unknown"
    cuda_version = "unknown"
    driver_version = "unknown"

    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda or "unknown"
    except ImportError:
        pass

    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        raw = pynvml.nvmlDeviceGetDriverVersion(handle)
        driver_version = raw.decode() if isinstance(raw, bytes) else str(raw)
    except Exception:
        pass

    return device_name, cuda_version, driver_version


def _model_result(
    model: BaseModel,
    trial_id: int,
    device_name: str,
    cuda_version: str,
    driver_version: str,
) -> TrialResult:
    """Build a baseline TrialResult from model metadata and device info."""
    m = model.get_metadata()
    return TrialResult(
        model_id=m["model_id"],
        model_name=m["model_name"],
        paper_arch=m["paper_arch"],
        native_framework=m["native_framework"],
        native_model_name=m["native_model_name"],
        exactness_status=m["exactness_status"],
        device_name=device_name,
        cuda_version=cuda_version,
        driver_version=driver_version,
        trial_id=trial_id,
        process_fresh_start=True,
    )


def _error_result(
    model_id: int,
    trial_id: int,
    status: str,
    error: str,
    device_name: str,
    cuda_version: str,
    driver_version: str,
) -> TrialResult:
    """Build a minimal TrialResult for registry-level errors."""
    return TrialResult(
        model_id=model_id,
        model_name=f"model_{model_id}",
        paper_arch="",
        native_framework="",
        native_model_name="",
        exactness_status="unknown",
        device_name=device_name,
        cuda_version=cuda_version,
        driver_version=driver_version,
        trial_id=trial_id,
        process_fresh_start=True,
        status=status,
        error_message=error,
    )


def _write_iteration_csv(
    output_dir: Path,
    model_id: int,
    model_name: str,
    wall_times: list[float],
    kernel_times: list[float],
) -> None:
    """Write per-iteration timing data to a CSV file inside output_dir."""
    csv_path = output_dir / f"model_{model_id:03d}_{model_name}_iterations.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["iteration_id", "wall_clock_ms", "kernel_time_ms"]
        )
        writer.writeheader()
        for i, (wall, kernel) in enumerate(zip(wall_times, kernel_times), start=1):
            writer.writerow({
                "iteration_id": i,
                "wall_clock_ms": round(wall, 4),
                "kernel_time_ms": round(kernel, 4),
            })


def _emit_outputs(result: TrialResult, output_dir: Path) -> None:
    """Write TrialResult as JSON (per-trial) and append row to aggregate CSV."""
    json_path = (
        output_dir / f"model_{result.model_id:03d}_{result.model_name}_result.json"
    )
    write_trial_result_json(result, json_path)

    # Aggregate CSV lives one level up (trial root), collecting all models
    agg_csv = output_dir.parent.parent / "all_results.csv"
    write_trial_result_csv(result, agg_csv)
