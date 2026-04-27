"""
Entry script: run the full benchmark suite.

Equivalent to: native-bench run [OPTIONS]

Usage
-----
    python scripts/run_all.py
    python scripts/run_all.py --config configs/experiment_manifest.yaml
    python scripts/run_all.py --model-ids 1,6 --output-dir benchmark/output
    python scripts/run_all.py --help

After the main benchmark completes, this script runs an in-process
measurement pass that produces a trial-level benchmark_results.csv.

CSV schema
----------
    model, trial_id, first_call_ms, inference_mean_ms, inference_std_ms,
    inference_iters

Metric definitions
------------------
first_call_ms
    The wall-clock time for iteration 0 only — the very first inference
    call after model load.  Captures CUDA JIT compilation, cuDNN algorithm
    selection, and first-use memory allocation.  One value per trial;
    σ is computed *across* trials during aggregation, not within a trial.

inference_mean_ms / inference_std_ms
    Steady-state repeated inference latency measured after first_call_ms
    and an optional un-timed stabilisation phase.  CUDA sync barriers
    flank each timed iteration so wall-clock reflects completed GPU work.

Trial structure
---------------
Each trial reloads the model from scratch and runs three phases:
  A) Single timed inference  → first_call_ms
  B) Un-timed stabilisation  → not reported
  C) 1,024 timed iterations  → inference statistics
GPU cache is cleared and a 3-second cooldown separates consecutive trials.

NOTE: Trial isolation is in-process (model reload + GPU cache clear).
For full subprocess-level isolation, use the CLI pipeline directly:
    python -m benchmark.cli run
which spawns a fresh process per trial (runner.py).
"""

import csv
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

# Ensure the project root is importable when running as a plain script
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from benchmark.cli import main  # noqa: E402
from benchmark.models.registry import MODEL_REGISTRY  # noqa: E402

# Models enabled for benchmarking (IDs 1-6, 9-10; 7-8 are disabled stubs)
_ENABLED_MODEL_IDS = [1, 2, 3, 4, 5, 6, 9, 10]

# Number of complete fresh-load trials per model.
# Each trial contributes one first_call_ms value; σ across trials is the
# reported 1st-call variability.
_NUM_TRIALS = 3

# Un-timed stabilisation iterations run between first_call and the measured
# inference loop.  Not reported; purpose is to reach GPU steady state.
_STABILIZATION_ITERS = 9

_MEASURED_ITERS = 1024
_SEED = 12345
_OUTPUT_CSV = _ROOT / "benchmark_results.csv"
_CSV_COLUMNS = [
    "model",
    "trial_id",
    "first_call_ms",
    "inference_mean_ms",
    "inference_std_ms",
    "inference_iters",
]


# ── CUDA helpers ──────────────────────────────────────────────────────────────

def _cuda_sync() -> None:
    """
    Synchronise the CUDA device if available.

    Called before stopping the timer after each timed inference so that
    time.perf_counter() measures wall-clock for *completed* GPU work, not
    just kernel-submission latency.
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _clear_gpu_cache() -> None:
    """Empty the CUDA allocator cache and synchronise. Used between trials."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


# ── Core measurement function ─────────────────────────────────────────────────

def measure_model(
    run_inference_fn: Callable[[], object],
    stabilization_iters: int = _STABILIZATION_ITERS,
    measured_iters: int = _MEASURED_ITERS,
) -> dict:
    """
    Measure one trial of a single model's 1st-call and steady-state inference.

    Protocol
    --------
    Phase A — 1st-call (single timed inference):
        GPU is synchronised before the timer starts.  One forward pass is
        executed.  GPU is synchronised again before the timer stops.
        The elapsed time is recorded as first_call_ms.  This is the only
        iteration whose cost includes CUDA JIT compilation, cuDNN algorithm
        selection, and first-use memory allocation.

    Phase B — Un-timed stabilisation:
        ``stabilization_iters`` forward passes are executed without timing.
        These drain any residual JIT/autotuning state and bring the GPU to
        a thermal steady state.  The results are NOT reported.

    Phase C — Measured inference:
        ``measured_iters`` forward passes, each wrapped in a GPU sync so
        wall-clock reflects completed execution.  Produces
        inference_mean_ms and inference_std_ms.

    Parameters
    ----------
    run_inference_fn:
        Zero-argument callable that executes one forward pass.
    stabilization_iters:
        Number of un-timed stabilisation calls between Phase A and Phase C.
    measured_iters:
        Number of timed steady-state iterations.

    Returns
    -------
    dict with keys:
        first_call_ms, inference_mean_ms, inference_std_ms, inference_iters
    """
    # ── Phase A: 1st-call ─────────────────────────────────────────────────────
    _cuda_sync()                    # ensure GPU is idle before timing starts
    t0 = time.perf_counter()
    run_inference_fn()
    _cuda_sync()                    # wait for GPU to complete before stopping
    t1 = time.perf_counter()
    first_call_ms = (t1 - t0) * 1_000.0

    # ── Phase B: Un-timed stabilisation ──────────────────────────────────────
    for _ in range(stabilization_iters):
        run_inference_fn()
    _cuda_sync()                    # flush all pending work before timed loop

    # ── Phase C: Measured inference ───────────────────────────────────────────
    inference_times_ms: list[float] = []
    for _ in range(measured_iters):
        t0 = time.perf_counter()
        run_inference_fn()
        _cuda_sync()                # wait for GPU before stopping timer
        t1 = time.perf_counter()
        inference_times_ms.append((t1 - t0) * 1_000.0)

    inference_arr = np.array(inference_times_ms, dtype=np.float64)

    return {
        "first_call_ms": first_call_ms,
        "inference_mean_ms": float(np.mean(inference_arr)),
        "inference_std_ms": float(np.std(inference_arr, ddof=1)),
        "inference_iters": len(inference_arr),
    }


# ── Multi-trial pass ──────────────────────────────────────────────────────────

def _run_latency_pass(device: str = "cuda", num_trials: int = _NUM_TRIALS) -> list[dict]:
    """
    For each enabled model run ``num_trials`` independent measurement trials.

    Each trial reloads the model from scratch to reset CUDA JIT and cuDNN
    state as completely as possible within a single process.  Between trials
    the GPU cache is cleared and a 3-second cooldown separates consecutive
    trials to allow GPU thermal settling and driver cleanup.

    Each trial row contains its own trial_id so cross-trial statistics
    (mean ± σ of first_call_ms) can be computed during aggregation.

    Returns a flat list of result dicts (one per model per trial).
    """
    results: list[dict] = []

    for model_id in _ENABLED_MODEL_IDS:
        if model_id not in MODEL_REGISTRY:
            print(f"[WARN] Model ID {model_id} not in registry — skipping.")
            continue

        # Instantiate once just to read the model name, then discard
        _probe = MODEL_REGISTRY[model_id]()
        model_name = getattr(_probe, "paper_model_name", str(model_id))
        del _probe

        print(f"\n  → Model {model_id}: {model_name}  ({num_trials} trial(s))", flush=True)

        for trial_idx in range(num_trials):
            trial_id = trial_idx + 1          # 1-based trial numbering

            # Fresh model instance each trial — resets any cached state
            model = MODEL_REGISTRY[model_id]()
            try:
                model.load_model(device=device)
                dummy_input = model.make_dummy_input(seed=_SEED)

                metrics = measure_model(
                    run_inference_fn=lambda: model.run_inference(dummy_input),
                    stabilization_iters=_STABILIZATION_ITERS,
                    measured_iters=_MEASURED_ITERS,
                )

                results.append({
                    "model": model_name,
                    "trial_id": trial_id,
                    **metrics,
                })

                print(
                    f"     trial {trial_id}:  1st-call {metrics['first_call_ms']:.3f} ms | "
                    f"inference {metrics['inference_mean_ms']:.3f} ± "
                    f"{metrics['inference_std_ms']:.3f} ms"
                )

            except Exception as exc:
                print(
                    f"  [ERROR] Model {model_id} ({model_name}) "
                    f"trial {trial_id} failed: {exc}"
                )
            finally:
                try:
                    model.cleanup()
                except Exception:
                    pass
                _clear_gpu_cache()
                time.sleep(3.0)     # inter-trial cooldown

    return results


# ── CSV writer ────────────────────────────────────────────────────────────────

def _save_benchmark_results(rows: list[dict], output_path: Path) -> None:
    """Write rows to benchmark_results.csv, overwriting any existing file."""
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "model":              row["model"],
                "trial_id":           row["trial_id"],
                "first_call_ms":      f"{row['first_call_ms']:.6f}",
                "inference_mean_ms":  f"{row['inference_mean_ms']:.6f}",
                "inference_std_ms":   f"{row['inference_std_ms']:.6f}",
                "inference_iters":    row["inference_iters"],
            })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Step 1: Run the full subprocess-based benchmark (existing pipeline) ──
    main(["run", *sys.argv[1:]], standalone_mode=False)

    # ── Step 2: In-process latency pass ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("LATENCY MEASUREMENT PASS (time.perf_counter + CUDA sync)")
    print(f"  Trials per model       : {_NUM_TRIALS}")
    print(f"  Stabilisation iters    : {_STABILIZATION_ITERS}  (un-timed, not reported)")
    print(f"  Measured iters / trial : {_MEASURED_ITERS}")
    print("=" * 70)

    rows = _run_latency_pass(device="cuda")

    if not rows:
        print("\n[ERROR] No models produced results — benchmark_results.csv not written.")
        sys.exit(1)

    _save_benchmark_results(rows, _OUTPUT_CSV)

    print(f"\n✓ Saved benchmark_results.csv → {_OUTPUT_CSV}")
    print(f"  {len(rows)} row(s) ({_NUM_TRIALS} trials × {len(_ENABLED_MODEL_IDS)} models)")
    print(f"  Columns: {', '.join(_CSV_COLUMNS)}")
