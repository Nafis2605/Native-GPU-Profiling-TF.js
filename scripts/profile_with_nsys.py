"""
profile_with_nsys.py
====================
Run a single model benchmark under Nsight Systems and optionally parse the
resulting .nsys-rep report into a structured JSON summary.

Usage
-----
    python scripts/profile_with_nsys.py \\
        --model-id 5 \\
        --output-dir results/nsys \\
        [--warmup 1] \\
        [--iterations 50] \\
        [--seed 42] \\
        [--device 0] \\
        [--trials 1] \\
        [--trace cuda,nvtx,osrt] \\
        [--timeout 1800] \\
        [--no-parse] \\
        [--log-level INFO]

Exit codes
----------
0  — success (nsys ran and, unless --no-parse, report was parsed)
1  — nsys not available / hard error before launch
2  — nsys launched but the profiled process exited non-zero
3  — nsys ran but the .nsys-rep file was not produced
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve project root so the script works regardless of CWD
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from benchmark.profilers import (  # noqa: E402
    NsysParser,
    NsysRunConfig,
    NsysRunner,
    RunMode,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_LOG_FMT = "%(asctime)s  %(levelname)-8s  %(name)s: %(message)s"
logger = logging.getLogger("profile_with_nsys")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Profile a single model benchmark run with Nsight Systems.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    p.add_argument(
        "--model-id",
        type=int,
        required=True,
        metavar="N",
        help="Numeric model ID to profile (passed to run_one_model.py).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        metavar="DIR",
        help="Root directory for profiling artefacts.",
    )

    # Benchmark knobs
    p.add_argument(
        "--warmup",
        type=int,
        default=1,
        metavar="N",
        help="Number of warmup iterations (default: 1).",
    )
    p.add_argument(
        "--iterations",
        type=int,
        default=50,
        metavar="N",
        help="Number of measured iterations (default: 50).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="N",
        help="Random seed for reproducible inputs.",
    )
    p.add_argument(
        "--device",
        type=int,
        default=0,
        metavar="N",
        help="CUDA device index (default: 0).",
    )

    # Profile knobs
    p.add_argument(
        "--trials",
        type=int,
        default=1,
        metavar="N",
        help="Number of independent profiling passes (default: 1).",
    )
    p.add_argument(
        "--trace",
        default="cuda,nvtx,osrt",
        metavar="APIS",
        help="Comma-separated nsys trace APIs (default: cuda,nvtx,osrt).",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=1800,
        metavar="SECS",
        help="Per-trial timeout in seconds (default: 1800).",
    )
    p.add_argument(
        "--no-parse",
        action="store_true",
        help="Skip parsing the .nsys-rep report to JSON.",
    )

    # Verbosity
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )

    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _build_target_command(
    args: argparse.Namespace,
    trial_output_dir: Path,
    trial_id: int,
) -> list[str]:
    """Build the command that nsys will wrap."""
    run_one = str(_PROJECT_ROOT / "scripts" / "run_one_model.py")
    cmd = [
        sys.executable,
        run_one,
        "--model-id", str(args.model_id),
        "--trial-id", str(trial_id),
        "--output-dir", str(trial_output_dir),
        "--warmup", str(args.warmup),
        "--iterations", str(args.iterations),
        "--device", str(args.device),
        "--run-mode", RunMode.PROFILE_NSYS,
    ]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    return cmd


def _print_summary(
    model_id: int,
    trial: int,
    run_result_path: Path,
    summary_path: Path | None,
    artifacts: dict[str, str],
) -> None:
    """Print a human-readable summary table to stdout."""
    width = 70
    print()
    print("=" * width)
    print(f"  Nsight Systems profile — model {model_id:03d}  trial {trial}")
    print("=" * width)

    if run_result_path.is_file():
        data: dict = json.loads(run_result_path.read_text(encoding="utf-8"))
        success = data.get("success", False)
        wall = data.get("wall_time_s")
        rc_target = data.get("target_return_code")
        rc_nsys = data.get("nsys_return_code")
        fail_reason = data.get("failure_reason", "")

        status = "OK" if success else "FAILED"
        print(f"  Status          : {status}")
        if wall is not None:
            print(f"  Wall time       : {wall:.2f} s")
        if rc_target is not None:
            print(f"  Target exit code: {rc_target}")
        if rc_nsys is not None:
            print(f"  Nsys exit code  : {rc_nsys}")
        if fail_reason:
            print(f"  Failure reason  : {fail_reason}")

    if summary_path is not None and summary_path.is_file():
        sumdata: dict = json.loads(summary_path.read_text(encoding="utf-8"))
        gpu_ms = sumdata.get("gpu_timeline_duration_ms")
        kern_ms = sumdata.get("total_kernel_time_ms")
        api_ms = sumdata.get("total_api_time_ms")
        memcpy_ms = sumdata.get("total_memcpy_time_ms")
        kern_count = sumdata.get("kernel_count")
        syncs = sumdata.get("sync_call_count")
        parsed_ok = sumdata.get("parsed_ok", False)
        warnings = sumdata.get("parse_warnings", [])

        print()
        print("  --- Profile summary ---")
        if parsed_ok:
            if gpu_ms is not None:
                print(f"  GPU timeline    : {gpu_ms:.2f} ms")
            if kern_ms is not None:
                print(f"  Kernel time     : {kern_ms:.2f} ms")
            if api_ms is not None:
                print(f"  CUDA API time   : {api_ms:.2f} ms")
            if memcpy_ms is not None:
                print(f"  Memcpy time     : {memcpy_ms:.2f} ms")
            if kern_count is not None:
                print(f"  Kernel count    : {kern_count}")
            if syncs is not None:
                print(f"  Sync calls      : {syncs}")

            top_kernels: list[dict] = sumdata.get("top_kernels", [])
            if top_kernels:
                print()
                print("  Top kernels by total time:")
                for k in top_kernels[:5]:
                    name = k.get("name", "")[:55]
                    pct = k.get("time_pct", 0.0)
                    print(f"    {pct:5.1f}%  {name}")
        else:
            print("  (report parse failed or not attempted)")

        if warnings:
            print()
            print("  Parse warnings:")
            for w in warnings:
                print(f"    - {w}")

    print()
    print("  Artefacts:")
    for key, path in artifacts.items():
        label = key.replace("_", " ").capitalize()
        print(f"    {label:<22}: {path}")

    print("=" * width)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(format=_LOG_FMT, level=getattr(logging, args.log_level))

    # ------------------------------------------------------------------ #
    # 1. Check that nsys is available                                     #
    # ------------------------------------------------------------------ #
    if not NsysRunner.is_available():
        logger.error(
            "Nsight Systems (nsys) is not available on this machine.\n"
            "  Install NVIDIA Nsight Systems and ensure 'nsys' is on PATH,\n"
            "  or set the NSYS_EXE environment variable to its full path.\n"
            "  Download: https://developer.nvidia.com/nsight-systems"
        )
        return 1

    runner = NsysRunner()
    nsys_version = runner.get_version()
    logger.info("Found nsys: %s  (version: %s)", runner.get_resolved_binary(), nsys_version)

    # ------------------------------------------------------------------ #
    # 2. Prepare output root                                              #
    # ------------------------------------------------------------------ #
    ts = _timestamp()
    profile_root = args.output_dir / f"model_{args.model_id:03d}_nsys_{ts}"
    profile_root.mkdir(parents=True, exist_ok=True)
    logger.info("Profile output root: %s", profile_root)

    # ------------------------------------------------------------------ #
    # 3. Run profiling trials                                             #
    # ------------------------------------------------------------------ #
    overall_exit = 0

    for trial_idx in range(args.trials):
        trial_label = f"trial_{trial_idx:02d}"
        trial_dir = profile_root / trial_label
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Sub-directory that run_one_model.py will write its TrialResult JSON into.
        bench_out_dir = trial_dir / "bench_output"
        bench_out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "--- Trial %d / %d  (model %d) ---",
            trial_idx + 1, args.trials, args.model_id,
        )

        # ---- Build commands ------------------------------------------ #
        target_cmd = _build_target_command(args, bench_out_dir, trial_idx)
        config = NsysRunConfig(
            output_dir=trial_dir,
            report_name=f"model_{args.model_id:03d}_profile",
            trace=args.trace,
            timeout_s=args.timeout,
        )

        logger.debug("Target command: %s", " ".join(target_cmd))
        logger.debug("Expected report: %s", config.expected_report_path())

        # ---- Profile ------------------------------------------------- #
        run_result = runner.run(target_cmd, config)

        # ---- Persist run result JSON ---------------------------------- #
        run_result_path = trial_dir / "nsys_run_result.json"
        run_result_json = json.dumps(run_result.to_dict(), indent=2, default=str)
        run_result_path.write_text(run_result_json, encoding="utf-8")
        logger.info("Run result saved: %s", run_result_path)

        # ---- Persist subprocess logs ---------------------------------- #
        run_result.save_subprocess_logs(trial_dir)

        # ---- Exit-code diagnostics ------------------------------------ #
        if not run_result.success:
            logger.error("Profiling failed: %s", run_result.failure_reason)
            if run_result.target_return_code not in (None, 0):
                logger.error(
                    "Benchmarked process exited with code %d; "
                    "check %s for details.",
                    run_result.target_return_code,
                    trial_dir / "nsys_profile_stderr.log",
                )
                overall_exit = 2
            else:
                overall_exit = 3

        # ---- Parse report -------------------------------------------- #
        summary_path: Path | None = None

        if args.no_parse:
            logger.info("--no-parse specified; skipping report parsing.")
        elif run_result.report_path is None or not run_result.report_path.is_file():
            logger.warning(
                "Report file not found at expected location (%s); "
                "skipping parse step.",
                config.expected_report_path(),
            )
        else:
            logger.info("Parsing report: %s", run_result.report_path)
            nsys_summary = NsysParser().parse(run_result.report_path)

            if nsys_summary.parsed_ok:
                logger.info(
                    "Parse OK — GPU timeline %.2f ms, %d kernel(s)",
                    nsys_summary.gpu_timeline_duration_ms or 0.0,
                    nsys_summary.kernel_count or 0,
                )
            else:
                logger.warning(
                    "Report parse completed with warnings: %s",
                    "; ".join(nsys_summary.parse_warnings),
                )

            summary_path = trial_dir / "nsys_summary.json"
            summary_path.write_text(nsys_summary.to_json(), encoding="utf-8")
            logger.info("Summary saved: %s", summary_path)

        # ---- Collect artifact paths dict ----------------------------- #
        artifacts: dict[str, str] = {}
        if run_result.report_path and run_result.report_path.is_file():
            artifacts["nsys_report"] = str(run_result.report_path)
        sqlite_path = trial_dir / f"model_{args.model_id:03d}_profile.sqlite"
        if sqlite_path.is_file():
            artifacts["nsys_sqlite"] = str(sqlite_path)
        if summary_path and summary_path.is_file():
            artifacts["nsys_summary"] = str(summary_path)
        stdout_log = trial_dir / "nsys_profile_stdout.log"
        stderr_log = trial_dir / "nsys_profile_stderr.log"
        if stdout_log.is_file():
            artifacts["nsys_stdout"] = str(stdout_log)
        if stderr_log.is_file():
            artifacts["nsys_stderr"] = str(stderr_log)
        artifacts["run_result_json"] = str(run_result_path)

        # ---- Save profiler_metadata.json ----------------------------- #
        meta_path = trial_dir / "profiler_metadata.json"
        meta = {
            "run_mode": RunMode.PROFILE_NSYS,
            "model_id": args.model_id,
            "trial_index": trial_idx,
            "nsys_version": nsys_version,
            "nsys_binary": str(runner.get_resolved_binary()),
            "profile_timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
            "artifact_paths": artifacts,
            "nsys_run_config": {
                "output_dir": str(config.output_dir),
                "report_name": config.report_name,
                "trace": config.trace,
                "sample": config.sample,
                "cpuctxsw": config.cpuctxsw,
                "capture_range": config.capture_range,
                "extra_flags": list(config.extra_flags),
                "timeout_s": config.timeout_s,
            },
        }
        meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        logger.info("Metadata saved: %s", meta_path)

        # ---- Human-readable summary ---------------------------------- #
        _print_summary(
            model_id=args.model_id,
            trial=trial_idx,
            run_result_path=run_result_path,
            summary_path=summary_path,
            artifacts=artifacts,
        )

    logger.info("Profile session complete. Root: %s", profile_root)
    return overall_exit


if __name__ == "__main__":
    sys.exit(main())
