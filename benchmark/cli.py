"""
Command-line interface for native-tfjs-bench.

Registered entry point: ``native-bench`` (see pyproject.toml).

Commands
--------
run          Run the full benchmark suite (all enabled models, all trials).
run-model    Run benchmark for a single model by ID.
validate-env Check CUDA, NVML, ONNX Runtime, and MediaPipe availability.
list-models  Print the model registry with metadata.

Usage examples
--------------
    # After `pip install -e .`
    native-bench validate-env
    native-bench list-models
    native-bench run --config configs/experiment_manifest.yaml
    native-bench run --model-ids 1,6,10
    native-bench run-model --model-id 6

    # Without install
    python -m benchmark.cli run
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
import yaml

from benchmark.env_check import check_environment, print_env_report
from benchmark.models.registry import get_enabled_model_ids, list_models
from benchmark.trial_manager import (
    ExperimentConfig,
    ExperimentMode,
    ProfilerOptions,
    run_all_model_trials,
    run_experiment,
    run_model_trials,
)
from benchmark.utils import configure_logging

logger = logging.getLogger(__name__)


# ── Root group ────────────────────────────────────────────────────────────────

@click.group()
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    show_default=True,
    help="Logging verbosity.",
)
@click.option(
    "--log-file",
    default=None,
    type=click.Path(dir_okay=False, writable=True),
    help="Optional path to write log output (in addition to stdout).",
)
@click.pass_context
def main(ctx: click.Context, log_level: str, log_file: str | None) -> None:
    """native-tfjs-bench: Native Windows CUDA inference benchmark."""
    ctx.ensure_object(dict)
    ctx.obj["log_level"] = log_level
    configure_logging(
        level=log_level,
        log_file=Path(log_file) if log_file else None,
    )


# ── run ───────────────────────────────────────────────────────────────────────

@main.command("run")
@click.option(
    "--config",
    default="configs/experiment_manifest.yaml",
    type=click.Path(exists=True, dir_okay=False),
    show_default=True,
    help="Path to experiment manifest YAML.",
)
@click.option(
    "--output-dir",
    default="benchmark/output",
    type=click.Path(file_okay=False),
    show_default=True,
    help="Root directory for benchmark output artefacts.",
)
@click.option(
    "--device",
    default="cuda",
    show_default=True,
    help="CUDA device string, e.g. 'cuda' or 'cuda:0'.",
)
@click.option(
    "--model-ids",
    default=None,
    help=(
        "Comma-separated model IDs to run, e.g. '1,3,6'. "
        "Defaults to all models enabled in the manifest."
    ),
)
@click.option(
    "--skip-env-check",
    is_flag=True,
    default=False,
    help="Skip the pre-flight environment check (not recommended).",
)
@click.option(
    "--mode",
    default="clean",
    type=click.Choice(["clean", "nsys", "ncu", "hybrid"], case_sensitive=False),
    show_default=True,
    help=(
        "Experiment mode. clean=publishable benchmark only; "
        "nsys/ncu=diagnostic profiler pass after clean; "
        "hybrid=clean then both profilers."
    ),
)
@click.option(
    "--profile-trials",
    default=None,
    help="Comma-separated 0-based trial indices to profile, e.g. '0,1'. Defaults to trial 0.",
)
@click.option(
    "--profile-iterations",
    default=20,
    type=int,
    show_default=True,
    help="Measured iterations for each profiler re-run (keep smaller than clean run).",
)
@click.option(
    "--profile-models",
    default=None,
    help="Comma-separated model IDs to profile. Defaults to all selected models.",
)
@click.option(
    "--keep-raw-profiler-artifacts",
    is_flag=True,
    default=False,
    help="Retain binary .nsys-rep / .ncu-rep files alongside parsed JSON.",
)
@click.option(
    "--fail-on-missing-profiler",
    is_flag=True,
    default=False,
    help="Abort the run if nsys / ncu binary is not found (default: warn and continue).",
)
def run_all(
    config: str,
    output_dir: str,
    device: str,
    model_ids: str | None,
    skip_env_check: bool,
    mode: str,
    profile_trials: str | None,
    profile_iterations: int,
    profile_models: str | None,
    keep_raw_profiler_artifacts: bool,
    fail_on_missing_profiler: bool,
) -> None:
    """Run the complete benchmark suite (all models, all trials)."""
    manifest = _load_manifest(config)
    trial_cfg = manifest.get("trial_config", {})
    profiler_cfg = manifest.get("profiler_config", {})

    # Determine which model IDs to run
    all_manifest_models = manifest.get("models", [])
    enabled_ids = get_enabled_model_ids(all_manifest_models)

    if model_ids:
        requested = {int(x.strip()) for x in model_ids.split(",")}
        enabled_ids = [mid for mid in enabled_ids if mid in requested]

    if not enabled_ids:
        click.echo(
            "[ERROR] No models selected. "
            "Check --model-ids or set enabled: true in the manifest."
        )
        sys.exit(1)

    # Resolve profiler options — CLI flags override manifest defaults
    resolved_mode = mode or profiler_cfg.get("mode", "clean")

    parsed_profile_trials: list[int] | None = None
    if profile_trials:
        parsed_profile_trials = [int(x.strip()) for x in profile_trials.split(",")]
    elif profiler_cfg.get("profile_trials") is not None:
        parsed_profile_trials = list(profiler_cfg["profile_trials"])

    parsed_profile_models: list[int] | None = None
    if profile_models:
        parsed_profile_models = [int(x.strip()) for x in profile_models.split(",")]
    elif profiler_cfg.get("profile_models") is not None:
        parsed_profile_models = list(profiler_cfg["profile_models"])

    opts = ProfilerOptions(
        profile_trials=parsed_profile_trials,
        profile_models=parsed_profile_models,
        profile_iterations=profile_iterations or profiler_cfg.get("profile_iterations", 20),
        profile_warmup=profiler_cfg.get("profile_warmup", 2),
        nsys_trace=profiler_cfg.get("nsys_trace", "cuda,nvtx,osrt"),
        ncu_launch_skip=profiler_cfg.get("ncu_launch_skip", 0),
        ncu_launch_count=profiler_cfg.get("ncu_launch_count", 10),
        ncu_kernel_regex=profiler_cfg.get("ncu_kernel_regex"),
        keep_raw_artifacts=keep_raw_profiler_artifacts or profiler_cfg.get("keep_raw_artifacts", False),
        fail_on_missing_profiler=fail_on_missing_profiler or profiler_cfg.get("fail_on_missing_profiler", False),
        profiler_timeout_s=profiler_cfg.get("profiler_timeout_seconds", 3600),
    )

    exp_config = ExperimentConfig(
        mode=resolved_mode,
        num_trials=trial_cfg.get("num_trials", 5),
        warmup_iterations=trial_cfg.get("warmup_iterations", 10),
        measured_iterations=trial_cfg.get("measured_iterations", 1024),
        random_seed=trial_cfg.get("random_seed", 12345),
        device=device,
        timeout_s=trial_cfg.get("timeout_seconds_per_trial", 600),
        profiler_opts=opts,
    )

    is_diagnostic = ExperimentMode.includes_nsys(resolved_mode) or ExperimentMode.includes_ncu(resolved_mode)
    run_label = "[DIAGNOSTIC PROFILE RUN]" if is_diagnostic else "[BENCHMARK RUN]"

    click.echo(f"{run_label}  mode={resolved_mode}")
    click.echo(f"Models    : {enabled_ids}")
    click.echo(f"Trials    : {exp_config.num_trials} per model")
    click.echo(f"Warm-up   : {exp_config.warmup_iterations} iterations")
    click.echo(f"Measured  : {exp_config.measured_iterations} iterations")
    click.echo(f"Output    : {output_dir}")

    # Environment pre-flight
    if not skip_env_check:
        env = check_environment()
        if not env.all_critical_passed:
            click.echo(
                "[ERROR] Environment check failed. "
                "Run 'native-bench validate-env' for details."
            )
            sys.exit(1)
        click.echo(f"  GPU: {env.gpu_name}  CUDA: {env.cuda_version}\n")

    all_results = run_experiment(
        model_ids=enabled_ids,
        output_dir=Path(output_dir),
        config=exp_config,
    )

    _print_run_summary(all_results, mode=resolved_mode)


# ── run-model ─────────────────────────────────────────────────────────────────

@main.command("run-model")
@click.option(
    "--model-id", required=True, type=int, help="Model ID to benchmark (1–10)."
)
@click.option(
    "--config",
    default="configs/experiment_manifest.yaml",
    type=click.Path(exists=True, dir_okay=False),
    show_default=True,
)
@click.option(
    "--output-dir", default="benchmark/output", show_default=True,
    type=click.Path(file_okay=False),
)
@click.option("--device", default="cuda", show_default=True)
def run_model(model_id: int, config: str, output_dir: str, device: str) -> None:
    """Run benchmark for a single model (all configured trials)."""
    manifest = _load_manifest(config)
    trial_cfg = manifest.get("trial_config", {})

    results = run_model_trials(
        model_id=model_id,
        output_base_dir=Path(output_dir),
        num_trials=trial_cfg.get("num_trials", 5),
        warmup_iterations=trial_cfg.get("warmup_iterations", 10),
        measured_iterations=trial_cfg.get("measured_iterations", 1024),
        random_seed=trial_cfg.get("random_seed", 12345),
        device=device,
        timeout=trial_cfg.get("timeout_seconds_per_trial", 600),
    )

    click.echo(f"\nResults for model_id={model_id}:")
    click.echo(f"  {'Trial':<7} {'Status':<15} {'Mean ms':<12} {'P95 ms':<12} {'Kernel ms'}")
    click.echo("  " + "-" * 60)
    for r in results:
        click.echo(
            f"  {r.trial_id:<7} {r.status:<15} "
            f"{r.mean_inference_ms:<12.2f} {r.p95_inference_ms:<12.2f} "
            f"{r.mean_kernel_ms:.2f}"
        )


# ── validate-env ──────────────────────────────────────────────────────────────

@main.command("validate-env")
@click.option(
    "--device-index", default=0, type=int, show_default=True,
    help="CUDA device index to probe.",
)
def validate_env(device_index: int) -> None:
    """Check CUDA availability, framework versions, and telemetry support."""
    report = check_environment(device_index=device_index)
    print_env_report(report)
    sys.exit(0 if report.all_critical_passed else 1)


# ── list-models ───────────────────────────────────────────────────────────────

@main.command("list-models")
@click.option(
    "--format", "fmt",
    default="table",
    type=click.Choice(["table", "json"]),
    show_default=True,
    help="Output format.",
)
def list_models_cmd(fmt: str) -> None:
    """Print all registered models with their metadata."""
    models = list_models()
    if fmt == "json":
        click.echo(json.dumps(models, indent=2))
    else:
        _print_model_table(models)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_manifest(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _print_run_summary(all_results: dict, mode: str = "clean") -> None:
    """
    Print a benchmark summary table.

    *all_results* may be either:
    - ``dict[int, list[TrialResult]]`` — legacy shape from run_all_model_trials
    - ``dict[int, dict]`` — new shape from run_experiment with keys
      ``clean_results``, ``nsys_results``, ``ncu_results``
    """
    click.echo("\n" + "=" * 75)
    is_diagnostic = mode in ("nsys", "ncu", "hybrid")
    label = "[DIAGNOSTIC PROFILE RUN]" if is_diagnostic else "[BENCHMARK RUN]"
    click.echo(f"  {label}  Benchmark Summary  (mode={mode})")
    click.echo("=" * 75)
    click.echo(
        f"  {'ID':<4} {'Model':<28} {'OK/Total':<10} "
        f"{'Mean ms':<10} {'P95 ms':<10} {'Kernel ms':<12} Status"
    )
    click.echo("  " + "-" * 72)

    for model_id, data in sorted(all_results.items()):
        # Support both result shapes
        if isinstance(data, dict):
            results = data.get("clean_results", [])
        else:
            results = data

        successes = [r for r in results if r.status == "success"]
        name = results[0].model_name if results else f"model_{model_id}"
        ok_total = f"{len(successes)}/{len(results)}"

        if successes:
            mean_ms = sum(r.mean_inference_ms for r in successes) / len(successes)
            p95_ms = sum(r.p95_inference_ms for r in successes) / len(successes)
            kern_ms = sum(r.mean_kernel_ms for r in successes) / len(successes)
            status = "success"
        else:
            mean_ms = p95_ms = kern_ms = 0.0
            status = results[0].status if results else "no_results"

        click.echo(
            f"  {model_id:<4} {name:<28} {ok_total:<10} "
            f"{mean_ms:<10.2f} {p95_ms:<10.2f} {kern_ms:<12.2f} {status}"
        )

        # Show profiler pass summary when diagnostic data is present
        if isinstance(data, dict):
            for prof_key in ("nsys_results", "ncu_results"):
                prof_list = data.get(prof_key, [])
                if prof_list:
                    ok_p = sum(1 for p in prof_list if p.get("status") == "ok")
                    profiler_name = prof_key.replace("_results", "")
                    click.echo(
                        f"  {'':4} {'':28}   {profiler_name:<10} "
                        f"{ok_p}/{len(prof_list)} passes OK"
                    )

    click.echo("=" * 75)


def _print_model_table(models: list[dict]) -> None:
    click.echo(f"\n  {'ID':<5} {'Name':<28} {'Framework':<20} {'Exactness'}")
    click.echo("  " + "-" * 80)
    for m in models:
        click.echo(
            f"  {m['model_id']:<5} {m['model_name']:<28} "
            f"{m['native_framework']:<20} {m['exactness_status']}"
        )
