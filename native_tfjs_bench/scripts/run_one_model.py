"""
Benchmark entry point for native-tfjs-bench.

Two invocation modes
--------------------

**Orchestrator mode** (user-facing)
    Run all configured trials for one model in fresh subprocesses, print a
    summary, and save aggregated JSON + CSV to the output directory.

        python scripts/run_one_model.py --model mobilenetv3 --device cuda
        python scripts/run_one_model.py --model mobilenetv3 --trials 5 --output-dir results/

    Options:
        --model       Model name, e.g. ``mobilenetv3``.
        --device      CUDA device string (default: ``cuda``).
        --trials      Number of trial subprocesses (default: 5).
        --warmup      Warm-up iterations per trial (default: 10).
        --iterations  Measured iterations per trial (default: 1024).
        --seed        Base random seed (default: 12345).
        --output-dir  Root directory for output artefacts (default: benchmark/output).
        --log-level   Logging verbosity (default: INFO).

**Subprocess mode** (called by trial_manager.py — do not use directly)
    Run a single trial and write a JSON result file.

        python scripts/run_one_model.py \\
            --model-id 6 --trial-id 0 \\
            --output-dir benchmark/output/trial_000/model_006 \\
            --warmup 10 --iterations 1024 --seed 12345 --device cuda

Exit codes
----------
    0  Completed successfully (or with status skipped / unsupported)
    1  Trial failed or an unhandled exception occurred
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Ensure the project root is importable when run as a subprocess
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.models.registry import get_model_id_by_name  # noqa: E402
from benchmark.result_schema import STATUS_SUCCESS           # noqa: E402
from benchmark.runner import run_trial                       # noqa: E402
from benchmark.trial_manager import run_model_trials         # noqa: E402
from benchmark.utils import configure_logging                # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark a single model through all trials.  "
            "Pass --model <name> for the user-facing orchestrator mode, or "
            "--model-id + --trial-id for the subprocess mode used by trial_manager."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Orchestrator mode (user-facing) ──────────────────────────────────────
    parser.add_argument(
        "--model", type=str, default=None, metavar="NAME",
        help=(
            "Model name, e.g. 'mobilenetv3'.  When provided without "
            "--trial-id, runs all --trials trials in fresh subprocesses "
            "and prints an aggregated summary."
        ),
    )
    parser.add_argument(
        "--trials", type=int, default=5,
        help="Number of trials to run in orchestrator mode (default: 5).",
    )

    # ── Subprocess mode (called by trial_manager) ─────────────────────────────
    parser.add_argument(
        "--model-id", type=int, default=None,
        help="Model ID from the registry (1–10). Required in subprocess mode.",
    )
    parser.add_argument(
        "--trial-id", type=int, default=None,
        help="0-based trial index. Required in subprocess mode.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help=(
            "Root output directory.  In orchestrator mode defaults to "
            "'benchmark/output'.  Required in subprocess mode."
        ),
    )
    parser.add_argument(
        "--warmup", type=int, default=10,
        help="Number of warm-up iterations per trial (default: 10).",
    )
    parser.add_argument(
        "--iterations", type=int, default=1024,
        help="Number of measured inference iterations per trial (default: 1024).",
    )
    parser.add_argument(
        "--seed", type=int, default=12345,
        help="Base random seed for reproducible input generation (default: 12345).",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="CUDA device string, e.g. 'cuda' or 'cuda:0' (default: cuda).",
    )
    parser.add_argument(
        "--log-level",
        type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    parser.add_argument(
        "--run-mode",
        type=str, default="clean_benchmark",
        choices=["clean_benchmark", "profile_nsys", "profile_ncu"],
        help=(
            "Execution mode: clean_benchmark (default), profile_nsys, or "
            "profile_ncu.  Profiler modes suppress telemetry and log an "
            "overhead warning; timing values are invalid for reporting."
        ),
    )
    return parser.parse_args()


# ── Mode dispatch ─────────────────────────────────────────────────────────────

def main() -> int:
    args = _parse_args()

    # ── Orchestrator mode: --model <name> without --trial-id ─────────────────
    if args.model is not None and args.trial_id is None:
        return _run_orchestrator(args)

    # ── Subprocess mode: --model-id + --trial-id (called by trial_manager) ───
    if args.model_id is not None and args.trial_id is not None:
        return _run_subprocess_trial(args)

    # ── Resolve --model → --model-id (orchestrator with explicit trials arg) ──
    # (e.g. someone passes --model mobilenetv3 --trial-id 0 for one-off debug)
    if args.model is not None and args.trial_id is not None:
        try:
            args.model_id = get_model_id_by_name(args.model)
        except KeyError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1
        return _run_subprocess_trial(args)

    print(
        "[ERROR] Provide either --model <name> (orchestrator) "
        "or both --model-id and --trial-id (subprocess mode).",
        file=sys.stderr,
    )
    return 1


# ── Orchestrator mode implementation ─────────────────────────────────────────

def _run_orchestrator(args: argparse.Namespace) -> int:
    """
    Run *args.trials* fresh-subprocess trials for the named model.

    Each trial is executed in a fresh Python subprocess by trial_manager so
    that GPU state and CUDA contexts are fully isolated between trials,
    matching the paper methodology.

    Outputs
    -------
    benchmark/output/<model_name>/
        trial_000/model_006/<model>_result.json   per-trial JSON
        trial_000/model_006/<model>_iterations.csv per-iter timings
        ...
        aggregated_result.json                    cross-trial summary
        aggregated_result.csv                     flat CSV of all trials
    """
    output_dir = Path(args.output_dir) if args.output_dir else Path("benchmark/output")
    configure_logging(level=args.log_level)
    logger = logging.getLogger("run_one_model")

    # Resolve model name → ID
    try:
        model_id = get_model_id_by_name(args.model)
    except KeyError as exc:
        logger.error("%s", exc)
        return 1

    logger.info("=" * 60)
    logger.info(
        "[BENCHMARK RUN]  model=%s  id=%d  device=%s  trials=%d  "
        "warmup=%d  iterations=%d  seed=%d",
        args.model, model_id, args.device, args.trials,
        args.warmup, args.iterations, args.seed,
    )
    logger.info("=" * 60)

    # Delegate subprocess orchestration to trial_manager
    try:
        trial_results = run_model_trials(
            model_id=model_id,
            output_base_dir=output_dir,
            num_trials=args.trials,
            warmup_iterations=args.warmup,
            measured_iterations=args.iterations,
            random_seed=args.seed,
            device=args.device,
            timeout=600,
        )
    except Exception:
        logger.exception("Unexpected error during trial orchestration")
        return 1

    # ── Aggregate and print summary ───────────────────────────────────────────
    _print_summary(trial_results, model_name=args.model, logger=logger)

    # ── Save aggregated JSON ──────────────────────────────────────────────────
    model_dir = output_dir / f"model_{model_id:03d}_aggregated"
    model_dir.mkdir(parents=True, exist_ok=True)
    _save_aggregated(trial_results, model_dir, model_id=model_id, logger=logger)

    failed = sum(1 for r in trial_results if r.status not in ("success", "skipped", "unsupported"))
    return 0 if failed == 0 else 1


def _print_summary(
    results: list,
    model_name: str,
    logger: logging.Logger,
) -> None:
    """Print a human-readable cross-trial summary table to the logger."""
    successes = [r for r in results if r.status == STATUS_SUCCESS]

    logger.info("")
    logger.info("=" * 68)
    logger.info("  Results: %s  (%d/%d trials succeeded)", model_name, len(successes), len(results))
    logger.info("=" * 68)
    logger.info(
        "  %-7s  %-9s  %-9s  %-9s  %-9s  %-9s  %s",
        "Trial", "Status", "Mean ms", "Std ms", "P50 ms", "P95 ms", "Kernel ms",
    )
    logger.info("  " + "-" * 65)
    for r in results:
        logger.info(
            "  %-7d  %-9s  %-9.3f  %-9.3f  %-9.3f  %-9.3f  %.3f",
            r.trial_id, r.status,
            r.mean_inference_ms, r.std_inference_ms,
            r.p50_inference_ms, r.p95_inference_ms,
            r.mean_kernel_ms,
        )

    if successes:
        logger.info("  " + "-" * 65)

        def _avg(attr: str) -> float:
            vals = [getattr(r, attr) for r in successes]
            return sum(vals) / len(vals)

        logger.info("  %-7s  %-9s  %-9.3f  %-9.3f  %-9.3f  %-9.3f  %.3f",
                    "MEAN", "", _avg("mean_inference_ms"), _avg("std_inference_ms"),
                    _avg("p50_inference_ms"), _avg("p95_inference_ms"), _avg("mean_kernel_ms"))

        # Telemetry summary (None-safe)
        def _avg_opt(attr: str) -> str:
            vals = [v for r in successes if (v := getattr(r, attr)) is not None]
            return f"{sum(vals)/len(vals):.1f}" if vals else "N/A"

        logger.info("")
        logger.info("  GPU util avg/peak : %s%% / %s%%",
                    _avg_opt("gpu_util_avg_pct"), _avg_opt("gpu_util_peak_pct"))
        logger.info("  GPU mem  avg/peak : %s MiB / %s MiB",
                    _avg_opt("gpu_mem_avg_mb"), _avg_opt("gpu_mem_peak_mb"))
        logger.info("  Power    avg/peak : %s W / %s W",
                    _avg_opt("power_avg_w"), _avg_opt("power_peak_w"))
        logger.info("  Energy (avg/trial): %s J", _avg_opt("energy_j"))

    logger.info("=" * 68)


def _save_aggregated(
    results: list,
    output_dir: Path,
    model_id: int,
    logger: logging.Logger,
) -> None:
    """Save aggregated JSON and CSV of all trial results."""
    import csv as _csv

    # ── JSON ─────────────────────────────────────────────────────────────────
    agg_json_path = output_dir / "aggregated_result.json"
    agg_data = {
        "model_id": model_id,
        "total_trials": len(results),
        "successful_trials": sum(1 for r in results if r.status == STATUS_SUCCESS),
        "trials": [r.to_dict() for r in results],
    }
    try:
        agg_json_path.write_text(
            json.dumps(agg_data, indent=2, default=str), encoding="utf-8"
        )
        logger.info("Aggregated JSON saved to %s", agg_json_path)
    except OSError as exc:
        logger.warning("Could not write aggregated JSON: %s", exc)

    # ── CSV ──────────────────────────────────────────────────────────────────
    agg_csv_path = output_dir / "aggregated_result.csv"
    if not results:
        return
    fieldnames = list(results[0].to_dict().keys())
    try:
        with agg_csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = _csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(r.to_dict())
        logger.info("Aggregated CSV  saved to %s", agg_csv_path)
    except OSError as exc:
        logger.warning("Could not write aggregated CSV: %s", exc)


# ── Subprocess mode implementation ────────────────────────────────────────────

def _run_subprocess_trial(args: argparse.Namespace) -> int:
    """
    Execute a single trial in this process (called by trial_manager as subprocess).

    Writes per-iteration CSV and trial-summary JSON to args.output_dir, then
    exits with code 0 on success or code 1 on unhandled exception.
    """
    if args.output_dir is None:
        print("[ERROR] --output-dir is required in subprocess mode.", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"trial_{args.trial_id:03d}_model_{args.model_id:03d}.log"
    configure_logging(level=args.log_level, log_file=log_file)

    logger = logging.getLogger("run_one_model")
    logger.info(
        "Process start | pid=%d | model_id=%d | trial_id=%d",
        os.getpid(), args.model_id, args.trial_id,
    )

    try:
        result = run_trial(
            model_id=args.model_id,
            trial_id=args.trial_id,
            output_dir=output_dir,
            warmup_iterations=args.warmup,
            measured_iterations=args.iterations,
            random_seed=args.seed,
            device=args.device,
            run_mode=args.run_mode,
        )
    except Exception:
        logger.exception("Unhandled exception in run_trial — subprocess exiting with code 1")
        return 1

    logger.info("Process exit | status=%s", result.status)
    return 0 if result.status in ("success", "skipped", "unsupported") else 1


if __name__ == "__main__":
    sys.exit(main())

