"""
profile_with_ncu.py
===================
Run a single model benchmark under Nsight Compute (ncu) and collect
detailed kernel-level GPU metrics.  Optionally parse the CSV metric dump
into a structured JSON summary.

Usage — real model
------------------
    python scripts/profile_with_ncu.py \\
        --model-id 3 \\
        --output-dir results/ncu \\
        [--warmup 2] \\
        [--iterations 20] \\
        [--launch-skip 5] \\
        [--launch-count 10] \\
        [--kernel-regex "gemm|conv"] \\
        [--timeout 3600] \\
        [--no-parse] \\
        [--log-level INFO]

Usage — dummy CUDA workload (no model artefacts required)
----------------------------------------------------------
    python scripts/profile_with_ncu.py \\
        --use-dummy \\
        --output-dir results/ncu_dummy \\
        [--launch-skip 5] \\
        [--launch-count 20]

Exit codes
----------
0  — success
1  — ncu not available / hard error before launch
2  — ncu ran but the profiled process exited non-zero
3  — ncu ran but the .ncu-rep report was not produced

Profiler overhead note
----------------------
ncu replays every profiled kernel launch multiple times to collect hardware
performance counters.  Expect 10×–100× slowdown per profiled kernel.
wall_time_s in the result reflects this overhead.  ALL timing values
produced under ncu MUST NOT be used for latency benchmarking.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Resolve project root so the script works regardless of CWD
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from benchmark.profilers import (  # noqa: E402
    MetricCategory,
    NcuParser,
    NcuRunConfig,
    NcuRunner,
    NcuProfilingResult,
    RunMode,
)
from benchmark.result_schema import NcuKernelResult  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_LOG_FMT = "%(asctime)s  %(levelname)-8s  %(name)s: %(message)s"
logger = logging.getLogger("profile_with_ncu")

# ---------------------------------------------------------------------------
# Dummy CUDA workload (for --use-dummy)
# ---------------------------------------------------------------------------
# Written to a tempfile so Windows subprocess quoting is not a concern.
_DUMMY_WORKLOAD_CODE = """\
\"\"\"
Synthetic CUDA workload for Nsight Compute pipeline validation.
Performs repeated 1024x1024 SGEMM to exercise cuBLAS GEMM kernels.
NOT a benchmark model — latency values are meaningless.
\"\"\"
import sys
import torch


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        return 1

    device = torch.device("cuda")
    dim = 1024

    # Allocate matrices
    a = torch.randn(dim, dim, device=device, dtype=torch.float32)
    b = torch.randn(dim, dim, device=device, dtype=torch.float32)
    torch.cuda.synchronize()

    # Repeated SGEMM — ncu will profile a window of these launches
    n_iters = 50
    for _ in range(n_iters):
        c = torch.mm(a, b)
        torch.cuda.synchronize()

    print(f"dummy_workload: completed {n_iters} SGEMM iterations "
          f"on {torch.cuda.get_device_name(device)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Profile a benchmark model or dummy CUDA workload with Nsight Compute.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Target selection (mutually exclusive)
    target_grp = p.add_mutually_exclusive_group(required=True)
    target_grp.add_argument(
        "--model-id",
        type=int,
        metavar="N",
        help="Numeric model ID to profile (passed to run_one_model.py).",
    )
    target_grp.add_argument(
        "--use-dummy",
        action="store_true",
        help=(
            "Use the built-in DummyCudaModel (1024×1024 SGEMM) instead of "
            "a real model.  No model artefacts required.  Useful for testing "
            "the Nsight Compute integration end-to-end."
        ),
    )

    # Required
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        metavar="DIR",
        help="Root directory for profiling artefacts.",
    )

    # Benchmark knobs (only used with --model-id)
    p.add_argument(
        "--warmup",
        type=int,
        default=2,
        metavar="N",
        help="Warm-up iterations for the benchmarked model (default: 2).",
    )
    p.add_argument(
        "--iterations",
        type=int,
        default=20,
        metavar="N",
        help=(
            "Measured iterations for the benchmarked model (default: 20). "
            "Keep low — ncu overhead inflates execution time significantly."
        ),
    )
    p.add_argument("--seed",   type=int, default=None, metavar="N",
                   help="Random seed for reproducible inputs.")
    p.add_argument("--device", type=int, default=0,    metavar="N",
                   help="CUDA device index (default: 0).")

    # ncu knobs
    p.add_argument(
        "--launch-skip",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Kernel launches to skip before profiling begins. "
            "Defaults to the model's ProfilingHint.launch_skip (or 0)."
        ),
    )
    p.add_argument(
        "--launch-count",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Kernel launches to profile after the skip window. "
            "Defaults to the model's ProfilingHint.launch_count (or 10)."
        ),
    )
    p.add_argument(
        "--kernel-regex",
        default=None,
        metavar="PATTERN",
        help=(
            "Optional regex to restrict profiling to matching kernel names "
            "(passed to ncu --kernel-regex).  "
            "Defaults to the model's ProfilingHint.representative_kernel_regex."
        ),
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=3600,
        metavar="SECS",
        help="Per-run timeout in seconds (default: 3600).",
    )
    p.add_argument(
        "--no-parse",
        action="store_true",
        help="Skip parsing the ncu stdout CSV to NcuProfilingResult JSON.",
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


def _write_dummy_script(tmp_dir: Path) -> Path:
    """Write the dummy CUDA workload to a temp file and return its path."""
    script = tmp_dir / "_dummy_cuda_workload.py"
    script.write_text(_DUMMY_WORKLOAD_CODE, encoding="utf-8")
    return script


def _get_profiling_hint(args: argparse.Namespace) -> Optional[object]:
    """
    Return the ProfilingHint for the selected target, or None.

    Instantiates the model class without loading weights (just to read the hint).
    """
    if args.use_dummy:
        try:
            from benchmark.models.base import DummyCudaModel
            return DummyCudaModel().get_profiling_hint()
        except Exception as exc:
            logger.debug("Could not get DummyCudaModel hint: %s", exc)
            return None

    try:
        from benchmark.models.registry import get_model
        model = get_model(args.model_id)
        hint = model.get_profiling_hint()
        return hint
    except Exception as exc:
        logger.debug("Could not get ProfilingHint for model %d: %s", args.model_id, exc)
        return None


def _resolve_ncu_config(
    args: argparse.Namespace,
    output_dir: Path,
    hint: Optional[object],
    model_label: str,
) -> NcuRunConfig:
    """Build NcuRunConfig, applying ProfilingHint defaults for unset args."""
    # Cascade: CLI arg > hint > hard default
    launch_skip  = args.launch_skip
    launch_count = args.launch_count
    kernel_regex = args.kernel_regex

    if hint is not None:
        if launch_skip  is None: launch_skip  = getattr(hint, "launch_skip",  0)
        if launch_count is None: launch_count = getattr(hint, "launch_count", 10)
        if kernel_regex is None:
            kernel_regex = getattr(hint, "representative_kernel_regex", None)
        hint_notes = getattr(hint, "notes", "")
        if hint_notes:
            logger.info("ProfilingHint notes: %s", hint_notes)
    else:
        if launch_skip  is None: launch_skip  = 0
        if launch_count is None: launch_count = 10

    logger.info(
        "ncu config: launch_skip=%d  launch_count=%d  kernel_regex=%r",
        launch_skip, launch_count, kernel_regex,
    )

    return NcuRunConfig(
        output_dir=output_dir,
        report_name=f"{model_label}_ncu_profile",
        launch_skip=launch_skip,
        launch_count=launch_count,
        kernel_regex=kernel_regex,
        timeout_s=args.timeout,
    )


def _build_model_target_command(args: argparse.Namespace, bench_out_dir: Path) -> list[str]:
    """Build the target command for --model-id mode."""
    run_one = str(_PROJECT_ROOT / "scripts" / "run_one_model.py")
    cmd = [
        sys.executable,
        run_one,
        "--model-id", str(args.model_id),
        "--trial-id", "0",
        "--output-dir", str(bench_out_dir),
        "--warmup", str(args.warmup),
        "--iterations", str(args.iterations),
        "--device", f"cuda:{args.device}",
        "--run-mode", RunMode.PROFILE_NCU,
    ]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    return cmd


def _build_dummy_target_command(tmp_script: Path) -> list[str]:
    """Build the target command for --use-dummy mode."""
    return [sys.executable, str(tmp_script)]


def _save_ncu_kernel_results(
    profiling_result: NcuProfilingResult,
    output_dir: Path,
    raw_report_path: Optional[Path],
    ncu_version: str,
) -> Path:
    """
    Write per-category NcuKernelResult JSON files for all profiled kernels.

    Each file contains the metrics for one kernel launch grouped by
    MetricCategory, satisfying the result schema spec:
      profiler_mode, profiler_tool, raw_report_path,
      parsed_metric_dict, kernel_name, metric_category.

    Returns the directory where the files were written.
    """
    kernel_results_dir = output_dir / "kernel_results"
    kernel_results_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []

    for kernel in profiling_result.kernels:
        by_cat = kernel.metrics_by_category()
        for category, metrics_dict in by_cat.items():
            record = NcuKernelResult(
                profiler_mode=RunMode.PROFILE_NCU,
                profiler_tool="ncu",
                raw_report_path=str(raw_report_path) if raw_report_path else "",
                kernel_name=kernel.kernel_name,
                metric_category=category,
                parsed_metric_dict={
                    k: v for k, v in metrics_dict.items() if v is not None
                },
                ncu_version=ncu_version,
                kernel_launch_id=kernel.launch_id,
                stream_id=kernel.stream_id,
                context_id=kernel.context_id,
            )
            all_records.append(record.to_dict())

    # Write all records to a single JSON array file
    records_path = kernel_results_dir / "ncu_kernel_results.json"
    records_path.write_text(
        json.dumps(all_records, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("Kernel results written: %s (%d records)", records_path, len(all_records))
    return kernel_results_dir


def _print_summary(
    model_label: str,
    run_result_path: Path,
    summary_path: Optional[Path],
    artifacts: dict[str, str],
) -> None:
    """Print a human-readable summary to stdout."""
    width = 72
    print()
    print("=" * width)
    print(f"  Nsight Compute profile — {model_label}")
    print("=" * width)

    if run_result_path.is_file():
        data: dict = json.loads(run_result_path.read_text(encoding="utf-8"))
        success     = data.get("success", False)
        wall        = data.get("wall_time_s")
        ncu_rc      = data.get("ncu_return_code")
        fail_reason = data.get("failure_reason", "")

        print(f"  Status           : {'OK' if success else 'FAILED'}")
        if wall is not None:
            print(f"  Wall time (w/ overhead): {wall:.1f} s  [NOT for benchmarking]")
        if ncu_rc is not None:
            print(f"  ncu exit code    : {ncu_rc}")
        if fail_reason:
            print(f"  Failure reason   : {fail_reason}")

    if summary_path is not None and summary_path.is_file():
        sumdata: dict = json.loads(summary_path.read_text(encoding="utf-8"))
        kernel_count = sumdata.get("kernel_count", 0)
        parsed_ok    = sumdata.get("parsed_ok", False)
        kernel_names = sumdata.get("kernel_names", [])
        warnings     = sumdata.get("parse_warnings", [])

        print()
        print("  --- Metric summary ---")
        if parsed_ok:
            print(f"  Kernel launches  : {kernel_count}")
            if kernel_names:
                print("  Kernels observed :")
                for name in kernel_names[:8]:
                    print(f"    {name[:65]}")
                if len(kernel_names) > 8:
                    print(f"    ... and {len(kernel_names) - 8} more")

            # Print a compact metric table for the first kernel
            kernels = sumdata.get("kernels", [])
            if kernels:
                first = kernels[0]
                metrics: dict = first.get("metrics", {})
                if metrics:
                    _print_metric_table(first["kernel_name"], metrics)
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
        print(f"    {label:<24}: {path}")
    print("=" * width)
    print()


def _print_metric_table(kernel_name: str, metrics: dict) -> None:
    """Print a compact table of metrics for one kernel, grouped by category."""
    print(f"\n  First kernel: {kernel_name[:60]}")
    print(f"  {'Metric':<55} {'Value':>10}  {'Unit':<10}")
    print(f"  {'-'*55} {'-'*10}  {'-'*10}")

    from benchmark.profilers import categorize_metric, MetricCategory

    # Print by category order
    cat_order = [
        MetricCategory.KERNEL_DURATION,
        MetricCategory.OCCUPANCY,
        MetricCategory.MEMORY_THROUGHPUT,
        MetricCategory.CACHE_BEHAVIOR,
        MetricCategory.SM_EFFICIENCY,
        MetricCategory.TENSOR_CORE,
        MetricCategory.WARP_SCHEDULER,
        MetricCategory.UNCLASSIFIED,
    ]
    by_cat: dict[str, list] = {}
    for m_name, m_data in metrics.items():
        cat = categorize_metric(m_name)
        by_cat.setdefault(cat, []).append((m_name, m_data))

    for cat in cat_order:
        rows = by_cat.get(cat, [])
        if not rows:
            continue
        print(f"\n  [{cat}]")
        for m_name, m_data in rows:
            val  = m_data.get("value")
            unit = m_data.get("unit", "")
            val_str = f"{val:10.3f}" if isinstance(val, (int, float)) else "       N/A"
            print(f"  {m_name:<55} {val_str}  {unit:<10}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = _build_parser()
    args   = parser.parse_args()

    logging.basicConfig(format=_LOG_FMT, level=getattr(logging, args.log_level))

    # ------------------------------------------------------------------ #
    # 1. Check ncu availability                                           #
    # ------------------------------------------------------------------ #
    if not NcuRunner.is_available():
        logger.error(
            "Nsight Compute (ncu) is not available on this machine.\n"
            "  Install NVIDIA Nsight Compute and ensure 'ncu' is on PATH,\n"
            "  or set the NCU_EXE environment variable to its full path.\n"
            "  Download: https://developer.nvidia.com/nsight-compute"
        )
        return 1

    runner      = NcuRunner()
    ncu_version = runner.get_version() or "unknown"
    logger.info("Found ncu: %s  (version: %s)", runner.get_resolved_binary(), ncu_version)

    # ------------------------------------------------------------------ #
    # 2. Identify target and resolve profiling hint                       #
    # ------------------------------------------------------------------ #
    model_label = "dummy" if args.use_dummy else f"model_{args.model_id:03d}"
    hint        = _get_profiling_hint(args)

    # ------------------------------------------------------------------ #
    # 3. Prepare output root                                              #
    # ------------------------------------------------------------------ #
    ts           = _timestamp()
    profile_root = args.output_dir / f"{model_label}_ncu_{ts}"
    profile_root.mkdir(parents=True, exist_ok=True)
    logger.info("Profile output root: %s", profile_root)

    # ------------------------------------------------------------------ #
    # 4. Build target command                                             #
    # ------------------------------------------------------------------ #
    tmp_dir_ctx: Optional[tempfile.TemporaryDirectory] = None
    bench_out_dir = profile_root / "bench_output"
    bench_out_dir.mkdir(parents=True, exist_ok=True)

    if args.use_dummy:
        tmp_dir_ctx = tempfile.TemporaryDirectory(prefix="ncu_dummy_")
        tmp_script  = _write_dummy_script(Path(tmp_dir_ctx.name))
        target_cmd  = _build_dummy_target_command(tmp_script)
        logger.info("Dummy workload script: %s", tmp_script)
    else:
        target_cmd = _build_model_target_command(args, bench_out_dir)

    # ------------------------------------------------------------------ #
    # 5. Build ncu config                                                 #
    # ------------------------------------------------------------------ #
    config = _resolve_ncu_config(args, profile_root, hint, model_label)
    logger.debug("Target command: %s", " ".join(target_cmd))
    logger.debug("Expected report: %s", config.expected_report_path())

    # ------------------------------------------------------------------ #
    # 6. Run ncu                                                          #
    # ------------------------------------------------------------------ #
    run_result = runner.run(target_cmd, config)

    # ------------------------------------------------------------------ #
    # 7. Persist run result and logs                                      #
    # ------------------------------------------------------------------ #
    run_result_path = profile_root / "ncu_run_result.json"
    run_result_path.write_text(
        json.dumps(run_result.to_dict(), indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("Run result saved: %s", run_result_path)

    run_result.save_subprocess_logs(profile_root)

    # ------------------------------------------------------------------ #
    # 8. Determine overall exit code from run result                     #
    # ------------------------------------------------------------------ #
    overall_exit = 0
    if not run_result.success:
        logger.error("ncu run failed: %s", run_result.failure_reason)
        overall_exit = 3 if run_result.ncu_return_code == 0 else 2

    # ------------------------------------------------------------------ #
    # 9. Parse ncu CSV metric dump                                       #
    # ------------------------------------------------------------------ #
    summary_path:   Optional[Path] = None
    kernel_results_dir: Optional[Path] = None
    profiling_result: Optional[NcuProfilingResult] = None

    if args.no_parse:
        logger.info("--no-parse: skipping CSV parsing.")
    elif not run_result.stdout or not run_result.stdout.strip():
        logger.warning(
            "ncu stdout is empty — no CSV data to parse.  "
            "This may indicate ncu ran in a mode that does not write CSV "
            "to stdout, or --csv was not honoured.  "
            "Check ncu_profile_stderr.log for details."
        )
    else:
        logger.info("Parsing ncu CSV metric dump (%d bytes)", len(run_result.stdout))
        profiling_result = NcuParser().parse_text(
            run_result.stdout,
            report_path=run_result.report_path,
            ncu_version=ncu_version,
        )

        if profiling_result.parsed_ok:
            logger.info(
                "Parse OK — %d kernel launch(es), %d unique kernel names",
                len(profiling_result.kernels),
                len(profiling_result.kernel_names()),
            )
        else:
            logger.warning(
                "Parse completed with issues: %s",
                "; ".join(profiling_result.parse_warnings),
            )

        # Summary JSON
        summary_path = profile_root / "ncu_profile_summary.json"
        summary_path.write_text(profiling_result.to_json(), encoding="utf-8")
        logger.info("Summary saved: %s", summary_path)

        # Per-kernel NcuKernelResult records (structured schema output)
        if profiling_result.kernels:
            kernel_results_dir = _save_ncu_kernel_results(
                profiling_result,
                profile_root,
                run_result.report_path,
                ncu_version,
            )

    # ------------------------------------------------------------------ #
    # 10. Clean up temp directory                                        #
    # ------------------------------------------------------------------ #
    if tmp_dir_ctx is not None:
        tmp_dir_ctx.cleanup()

    # ------------------------------------------------------------------ #
    # 11. Collect artifact dict                                          #
    # ------------------------------------------------------------------ #
    artifacts: dict[str, str] = {}
    if run_result.report_path and run_result.report_path.is_file():
        artifacts["ncu_report"] = str(run_result.report_path)
    if summary_path and summary_path.is_file():
        artifacts["ncu_summary"] = str(summary_path)
    if kernel_results_dir and kernel_results_dir.is_dir():
        artifacts["ncu_kernel_results"] = str(kernel_results_dir / "ncu_kernel_results.json")
    stdout_log = profile_root / "ncu_profile_stdout.log"
    stderr_log = profile_root / "ncu_profile_stderr.log"
    if stdout_log.is_file():
        artifacts["ncu_stdout"] = str(stdout_log)
    if stderr_log.is_file():
        artifacts["ncu_stderr"] = str(stderr_log)
    artifacts["run_result_json"] = str(run_result_path)

    # ------------------------------------------------------------------ #
    # 12. Save profiler_metadata.json                                    #
    # ------------------------------------------------------------------ #
    meta_path = profile_root / "profiler_metadata.json"
    meta: dict = {
        "run_mode":               RunMode.PROFILE_NCU,
        "profiler_tool":          "ncu",
        "target":                 model_label,
        "ncu_version":            ncu_version,
        "ncu_binary":             str(runner.get_resolved_binary()),
        "profile_timestamp_utc":  datetime.now(tz=timezone.utc).isoformat(),
        "artifact_paths":         artifacts,
        "overhead_warning": (
            "ncu uses hardware-counter replay: every profiled kernel launch "
            "is replayed 10×–100× times.  wall_time_s in ncu_run_result.json "
            "reflects this inflated execution time and is NOT valid for "
            "latency benchmarking."
        ),
        "ncu_run_config": {
            "output_dir":       str(config.output_dir),
            "report_name":      config.report_name,
            "launch_skip":      config.launch_skip,
            "launch_count":     config.launch_count,
            "kernel_regex":     config.kernel_regex,
            "target_processes": config.target_processes,
            "metrics_count":    len(config.metrics),
            "timeout_s":        config.timeout_s,
        },
    }
    if hint is not None:
        meta["profiling_hint"] = {
            "launch_skip":                 getattr(hint, "launch_skip",  None),
            "launch_count":                getattr(hint, "launch_count", None),
            "representative_kernel_regex": getattr(hint, "representative_kernel_regex", None),
            "notes":                       getattr(hint, "notes", ""),
        }
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    logger.info("Metadata saved: %s", meta_path)

    # ------------------------------------------------------------------ #
    # 13. Print human-readable summary                                   #
    # ------------------------------------------------------------------ #
    _print_summary(
        model_label=model_label,
        run_result_path=run_result_path,
        summary_path=summary_path,
        artifacts=artifacts,
    )

    logger.info("Profile session complete. Root: %s", profile_root)
    return overall_exit


if __name__ == "__main__":
    sys.exit(main())
