"""
Trial manager for native-tfjs-bench.

Orchestrates multiple independent trials for one or all models.
Each trial is executed in a fresh subprocess to satisfy the cold-start
requirement from the specification: the model is loaded from scratch on
every trial, with no GPU kernel or memory state carried over between trials.

Inter-process communication is file-based:
  - The subprocess (scripts/run_one_model.py) writes a JSON result file.
  - This process reads the file back and deserialises it into a TrialResult.
  - No shared memory, pipes, or sockets are used.

Failure isolation:
  - A subprocess timeout is caught and recorded as STATUS_FAILED.
  - A missing or corrupt result JSON is caught and recorded as STATUS_FAILED.
  - Either event does not stop the remaining trials or models.

Windows safety
--------------
subprocess.run(..., timeout=…) raises TimeoutExpired but does NOT kill the
child process on Windows, leaving it orphaned. This module uses Popen +
communicate(timeout=…) so we can call proc.kill() before draining I/O,
guaranteeing the child is gone before the next trial starts.

Subprocess diagnostics
----------------------
stdout and stderr from each subprocess are captured (PIPE) and written to
<trial_dir>/subprocess.log for post-hoc debugging. They are also emitted at DEBUG
level so `logging.basicConfig(level=logging.DEBUG)` shows them inline.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from benchmark.result_schema import STATUS_FAILED, RunMode, TrialResult

logger = logging.getLogger(__name__)

# Windows: suppress console window when spawning trial subprocesses
_SUBPROCESS_FLAGS: dict = (
    {"creationflags": subprocess.CREATE_NO_WINDOW}
    if os.name == "nt"
    else {}
)

# ── Defaults (overridden by experiment_manifest.yaml) ────────────────────────
DEFAULT_NUM_TRIALS = 5
DEFAULT_WARMUP_ITERS = 10
DEFAULT_MEASURED_ITERS = 1024
DEFAULT_SEED = 12345
DEFAULT_TIMEOUT_S = 600  # 10 minutes per trial subprocess


# ── ExperimentMode ───────────────────────────────────────────────────────────────────

class ExperimentMode:
    """
    Top-level mode constants for a benchmark experiment run.

    CLEAN
        Publishable latency benchmark.  No heavy profiler attached.  nvidia-smi /
        NVML polling is the only permitted instrumentation.  All latency numbers
        produced by CLEAN runs are valid for reporting and cross-comparison.

    NSYS
        Diagnostic profiling with Nsight Systems.  After all clean trials finish,
        selected trials are re-run under ``nsys profile`` to capture GPU timelines.
        GPU telemetry is suppressed for profiled trials to avoid compounding
        overhead.  Clean latency numbers are unaffected.

    NCU
        Diagnostic profiling with Nsight Compute.  After all clean trials finish,
        selected trials are re-run under ``ncu`` to collect kernel-level hardware
        counters.  ncu replay inflates execution time 10×–100×; all timing values
        from NCU runs are diagnostic only.

    HYBRID
        Full diagnostic suite: "clean" phase runs first to produce publishable
        latency results, then both Nsight Systems and Nsight Compute passes run
        for the selected models / trial indices.
    """

    CLEAN  = "clean"
    NSYS   = "nsys"
    NCU    = "ncu"
    HYBRID = "hybrid"

    _VALID: frozenset[str] = frozenset({"clean", "nsys", "ncu", "hybrid"})

    @classmethod
    def validate(cls, value: str) -> str:
        if value not in cls._VALID:
            raise ValueError(
                f"Unknown ExperimentMode {value!r}. "
                f"Valid values: {sorted(cls._VALID)}"
            )
        return value

    @classmethod
    def includes_clean(cls, value: str) -> bool:
        """True when publishable clean-timing trials should run."""
        return True  # all modes run clean trials first

    @classmethod
    def includes_nsys(cls, value: str) -> bool:
        return value in (cls.NSYS, cls.HYBRID)

    @classmethod
    def includes_ncu(cls, value: str) -> bool:
        return value in (cls.NCU, cls.HYBRID)


# ── ProfilerOptions ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ProfilerOptions:
    """
    Immutable profiling parameters controlling the diagnostic profiler passes.

    Fields
    ------
    profile_trials : list[int] | None
        0-based trial indices to profile within each model run.  None means
        profile only trial 0 (the first trial).
    profile_models : list[int] | None
        Model IDs to include in profiling passes.  None means all selected
        models receive a profiling pass.
    profile_iterations : int
        Measured iterations for profiled subprocesses.  Keep smaller than
        the clean-run value to limit profiler overhead time.
    profile_warmup : int
        Warm-up iterations for profiled subprocesses.
    nsys_trace : str
        Comma-separated nsys trace APIs (e.g. "cuda,nvtx,osrt").
    ncu_launch_skip : int
        Kernel launches to skip before ncu starts profiling.
    ncu_launch_count : int
        Number of consecutive kernel launches to profile per ncu run.
    ncu_kernel_regex : str | None
        Optional regex to restrict ncu profiling to matching kernel names.
    keep_raw_artifacts : bool
        When True, binary .nsys-rep and .ncu-rep files are kept alongside
        the parsed JSON summaries.  When False, only parsed JSON is retained.
    fail_on_missing_profiler : bool
        When True, a missing nsys / ncu binary aborts the experiment.
        When False (default), the profiling pass is skipped with a WARNING
        and the clean latency results are unaffected.
    profiler_timeout_s : int
        Per-profiling-run timeout in seconds.  ncu overhead is high; the
        default is 1 hour.
    """
    profile_trials:             Optional[list[int]] = None
    profile_models:             Optional[list[int]] = None
    profile_iterations:         int = 20
    profile_warmup:             int = 2
    nsys_trace:                 str = "cuda,nvtx,osrt"
    ncu_launch_skip:            int = 0
    ncu_launch_count:           int = 10
    ncu_kernel_regex:           Optional[str] = None
    keep_raw_artifacts:         bool = False
    fail_on_missing_profiler:   bool = False
    profiler_timeout_s:         int = 3600

    def effective_profile_trials(self, num_trials: int) -> list[int]:
        """Return the actual trial indices to profile, clamped to [0, num_trials)."""
        if self.profile_trials is None:
            return [0]
        return [t for t in self.profile_trials if 0 <= t < num_trials]

    def should_profile_model(self, model_id: int) -> bool:
        """Return True if this model should receive a diagnostic profiling pass."""
        return self.profile_models is None or model_id in self.profile_models


# ── ExperimentConfig ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ExperimentConfig:
    """
    Full configuration for one experiment run.

    Constructed in cli.py from the YAML manifest + CLI overrides and forwarded
    to run_experiment().  Serialisable to dict for logging and manifest output.
    """
    mode:                 str             = ExperimentMode.CLEAN
    num_trials:           int             = DEFAULT_NUM_TRIALS
    warmup_iterations:    int             = DEFAULT_WARMUP_ITERS
    measured_iterations:  int             = DEFAULT_MEASURED_ITERS
    random_seed:          int             = DEFAULT_SEED
    device:               str             = "cuda"
    timeout_s:            int             = DEFAULT_TIMEOUT_S
    profiler_opts:        ProfilerOptions = field(default_factory=ProfilerOptions)

    def to_dict(self) -> dict:
        return {
            "mode":                self.mode,
            "num_trials":          self.num_trials,
            "warmup_iterations":   self.warmup_iterations,
            "measured_iterations": self.measured_iterations,
            "random_seed":         self.random_seed,
            "device":              self.device,
            "timeout_s":           self.timeout_s,
            "profiler_opts": {
                "profile_trials":           self.profiler_opts.profile_trials,
                "profile_models":           self.profiler_opts.profile_models,
                "profile_iterations":       self.profiler_opts.profile_iterations,
                "profile_warmup":           self.profiler_opts.profile_warmup,
                "nsys_trace":               self.profiler_opts.nsys_trace,
                "ncu_launch_skip":          self.profiler_opts.ncu_launch_skip,
                "ncu_launch_count":         self.profiler_opts.ncu_launch_count,
                "ncu_kernel_regex":         self.profiler_opts.ncu_kernel_regex,
                "keep_raw_artifacts":       self.profiler_opts.keep_raw_artifacts,
                "fail_on_missing_profiler": self.profiler_opts.fail_on_missing_profiler,
                "profiler_timeout_s":       self.profiler_opts.profiler_timeout_s,
            },
        }


# ── TrialConfig ───────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TrialConfig:
    """
    Immutable configuration for one trial execution.

    Passed from the parent process to the subprocess via CLI flags.
    Using a dataclass avoids long argument lists and makes it easy to
    serialise / log the full config alongside results.
    """
    model_id: int
    trial_id: int
    output_base_dir: Path
    warmup_iterations: int = DEFAULT_WARMUP_ITERS
    measured_iterations: int = DEFAULT_MEASURED_ITERS
    random_seed: int = DEFAULT_SEED
    device: str = "cuda"
    timeout_s: int = DEFAULT_TIMEOUT_S

    def trial_dir(self) -> Path:
        """Canonical output directory for this specific trial."""
        return (
            self.output_base_dir
            / f"trial_{self.trial_id:03d}"
            / f"model_{self.model_id:03d}"
        )


# ── Public API ────────────────────────────────────────────────────────────────

def run_model_trials(
    model_id: int,
    output_base_dir: Path,
    num_trials: int = DEFAULT_NUM_TRIALS,
    warmup_iterations: int = DEFAULT_WARMUP_ITERS,
    measured_iterations: int = DEFAULT_MEASURED_ITERS,
    random_seed: int = DEFAULT_SEED,
    device: str = "cuda",
    timeout: int = DEFAULT_TIMEOUT_S,
) -> list[TrialResult]:
    """
    Run all trials for a single model in independent fresh subprocesses.

    One subprocess is launched per trial; each loads the model from scratch.
    Failed subprocesses are recorded as STATUS_FAILED and do not stop the
    remaining trials.

    Args:
        model_id:            Registry model ID (1–10).
        output_base_dir:     Root directory for all output artefacts.
        num_trials:          Number of independent trials (default 5).
        warmup_iterations:   Warm-up iterations per trial (default 10).
        measured_iterations: Measured iterations per trial (default 1024).
        random_seed:         Seed forwarded to each subprocess.
        device:              CUDA device string.
        timeout:             Max seconds before a subprocess is declared failed.

    Returns:
        List[TrialResult] of length num_trials (may contain failed entries).
    """
    logger.info("Starting %d trial(s) for model_id=%d", num_trials, model_id)
    results: list[TrialResult] = []

    for trial_id in range(num_trials):
        logger.info(
            "— Trial %d / %d  (model_id=%d) —", trial_id + 1, num_trials, model_id
        )
        cfg = TrialConfig(
            model_id=model_id,
            trial_id=trial_id,
            output_base_dir=output_base_dir,
            warmup_iterations=warmup_iterations,
            measured_iterations=measured_iterations,
            random_seed=random_seed,
            device=device,
            timeout_s=timeout,
        )
        result = _run_subprocess_trial(cfg)
        results.append(result)
        logger.info(
            "  Trial %d complete | model_id=%d | status=%s | mean=%.2f ms",
            trial_id, model_id, result.status, result.mean_inference_ms,
        )

    return results


def run_all_model_trials(
    model_ids: list[int],
    output_base_dir: Path,
    num_trials: int = DEFAULT_NUM_TRIALS,
    warmup_iterations: int = DEFAULT_WARMUP_ITERS,
    measured_iterations: int = DEFAULT_MEASURED_ITERS,
    random_seed: int = DEFAULT_SEED,
    device: str = "cuda",
    timeout: int = DEFAULT_TIMEOUT_S,
) -> dict[int, list[TrialResult]]:
    """
    Run trials for all specified models, one model at a time.

    All trials for model N complete before model N+1 begins.
    An unexpected exception for one model is caught and recorded; remaining
    models continue uninterrupted.

    Args:
        model_ids:  List of model IDs to benchmark.
        ...         (see run_model_trials for remaining args)

    Returns:
        Dict mapping model_id → list[TrialResult].
    """
    all_results: dict[int, list[TrialResult]] = {}

    for model_id in model_ids:
        logger.info("=" * 60)
        logger.info(
            "Benchmarking model_id=%d  (%d trial(s))", model_id, num_trials
        )
        try:
            results = run_model_trials(
                model_id=model_id,
                output_base_dir=output_base_dir,
                num_trials=num_trials,
                warmup_iterations=warmup_iterations,
                measured_iterations=measured_iterations,
                random_seed=random_seed,
                device=device,
                timeout=timeout,
            )
        except Exception as exc:
            logger.error("Unexpected error for model_id=%d: %s", model_id, exc)
            results = []

        all_results[model_id] = results

    logger.info("=" * 60)
    logger.info("All models complete.")
    return all_results


# ── run_experiment: profiler-aware top-level orchestrator ────────────────────

def run_experiment(
    model_ids: list[int],
    output_dir: Path,
    config: ExperimentConfig,
) -> dict[int, dict[str, list]]:
    """
    Run a complete experiment according to *config.mode*.

    Execution order
    ---------------
    1. **Clean phase** (always) — all trials for all models run with no
       heavy profiler attached so latency numbers are publication-quality.
    2. **Nsys phase** (modes: nsys, hybrid) — for each model selected by
       ``config.profiler_opts.should_profile_model()``, the nominated trial
       indices are re-run under ``nsys profile`` via a subprocess.
    3. **Ncu phase** (modes: ncu, hybrid) — same model / trial selection,
       re-run under ``ncu``.

    Phases 2 and 3 never overwrite the clean-phase result files; they write
    to a sibling ``profiling/`` sub-directory inside each trial directory.

    Returns
    -------
    dict[model_id, {"clean_results": list[TrialResult],
                    "nsys_results":  list[dict],
                    "ncu_results":   list[dict]}]
    """
    ExperimentMode.validate(config.mode)
    output_dir = Path(output_dir)

    is_diagnostic = ExperimentMode.includes_nsys(config.mode) or ExperimentMode.includes_ncu(config.mode)
    run_label = "[DIAGNOSTIC PROFILE RUN]" if is_diagnostic else "[BENCHMARK RUN]"

    logger.info("=" * 70)
    logger.info("%s  mode=%s  models=%s", run_label, config.mode, model_ids)
    logger.info("=" * 70)

    results: dict[int, dict[str, list]] = {}

    # ── Phase 1: clean benchmark trials ──────────────────────────────────────
    logger.info("--- Phase 1: clean latency benchmark ---")
    clean_all = run_all_model_trials(
        model_ids=model_ids,
        output_base_dir=output_dir,
        num_trials=config.num_trials,
        warmup_iterations=config.warmup_iterations,
        measured_iterations=config.measured_iterations,
        random_seed=config.random_seed,
        device=config.device,
        timeout=config.timeout_s,
    )
    for mid in model_ids:
        results[mid] = {
            "clean_results": clean_all.get(mid, []),
            "nsys_results": [],
            "ncu_results": [],
        }

    opts = config.profiler_opts
    trial_indices = opts.effective_profile_trials(config.num_trials)

    # ── Phase 2: Nsight Systems diagnostic pass ───────────────────────────────
    if ExperimentMode.includes_nsys(config.mode):
        logger.info("--- Phase 2: Nsight Systems diagnostic pass ---")
        for mid in model_ids:
            if not opts.should_profile_model(mid):
                logger.debug("Skipping nsys pass for model_id=%d (not in profile_models)", mid)
                continue
            nsys_out = _run_profiler_pass_nsys(
                model_id=mid,
                trial_indices=trial_indices,
                output_dir=output_dir,
                config=config,
            )
            results[mid]["nsys_results"] = nsys_out

    # ── Phase 3: Nsight Compute diagnostic pass ───────────────────────────────
    if ExperimentMode.includes_ncu(config.mode):
        logger.info("--- Phase 3: Nsight Compute diagnostic pass ---")
        for mid in model_ids:
            if not opts.should_profile_model(mid):
                logger.debug("Skipping ncu pass for model_id=%d (not in profile_models)", mid)
                continue
            ncu_out = _run_profiler_pass_ncu(
                model_id=mid,
                trial_indices=trial_indices,
                output_dir=output_dir,
                config=config,
            )
            results[mid]["ncu_results"] = ncu_out

    logger.info("=" * 70)
    logger.info("%s complete.", run_label)
    return results


# ── Internal: profiler pass helpers ──────────────────────────────────────────

def _run_profiler_pass_nsys(
    model_id: int,
    trial_indices: list[int],
    output_dir: Path,
    config: ExperimentConfig,
) -> list[dict]:
    """
    Invoke scripts/profile_with_nsys.py as a subprocess for each requested
    trial index of *model_id*.  Returns a list of status dicts (one per trial).
    """
    opts = config.profiler_opts
    scripts_dir = Path(__file__).parent.parent / "scripts"
    script = scripts_dir / "profile_with_nsys.py"

    pass_results = []
    for trial_id in trial_indices:
        trial_dir = output_dir / f"trial_{trial_id:03d}" / f"model_{model_id:03d}"
        profiling_dir = trial_dir / "profiling" / "nsys"
        profiling_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, str(script),
            "--model-id",   str(model_id),
            "--output-dir", str(profiling_dir),
            "--warmup",     str(opts.profile_warmup),
            "--iterations", str(opts.profile_iterations),
            "--trace",      opts.nsys_trace,
        ]
        if not opts.keep_raw_artifacts:
            cmd.append("--no-raw")

        logger.info(
            "[DIAGNOSTIC PROFILE RUN] nsys  model_id=%d  trial_id=%d",
            model_id, trial_id,
        )
        status = _run_profiler_subprocess(
            model_id=model_id,
            trial_id=trial_id,
            cmd=cmd,
            output_dir=profiling_dir,
            timeout_s=opts.profiler_timeout_s,
            profiler="nsys",
            fail_on_missing=opts.fail_on_missing_profiler,
        )
        pass_results.append(status)
    return pass_results


def _run_profiler_pass_ncu(
    model_id: int,
    trial_indices: list[int],
    output_dir: Path,
    config: ExperimentConfig,
) -> list[dict]:
    """
    Invoke scripts/profile_with_ncu.py as a subprocess for each requested
    trial index of *model_id*.  Returns a list of status dicts (one per trial).

    ncu replay overhead inflates execution time by 10×–100×.  Timing values
    produced by ncu runs must never be mixed into clean latency statistics.
    """
    opts = config.profiler_opts
    scripts_dir = Path(__file__).parent.parent / "scripts"
    script = scripts_dir / "profile_with_ncu.py"

    pass_results = []
    for trial_id in trial_indices:
        trial_dir = output_dir / f"trial_{trial_id:03d}" / f"model_{model_id:03d}"
        profiling_dir = trial_dir / "profiling" / "ncu"
        profiling_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, str(script),
            "--model-id",    str(model_id),
            "--output-dir",  str(profiling_dir),
            "--warmup",      str(opts.profile_warmup),
            "--iterations",  str(opts.profile_iterations),
            "--launch-skip", str(opts.ncu_launch_skip),
            "--launch-count",str(opts.ncu_launch_count),
        ]
        if opts.ncu_kernel_regex:
            cmd.extend(["--kernel-regex", opts.ncu_kernel_regex])
        if not opts.keep_raw_artifacts:
            cmd.append("--no-raw")

        logger.info(
            "[DIAGNOSTIC PROFILE RUN] ncu  model_id=%d  trial_id=%d",
            model_id, trial_id,
        )
        status = _run_profiler_subprocess(
            model_id=model_id,
            trial_id=trial_id,
            cmd=cmd,
            output_dir=profiling_dir,
            timeout_s=opts.profiler_timeout_s,
            profiler="ncu",
            fail_on_missing=opts.fail_on_missing_profiler,
        )
        pass_results.append(status)
    return pass_results


def _run_profiler_subprocess(
    model_id: int,
    trial_id: int,
    cmd: list[str],
    output_dir: Path,
    timeout_s: int,
    profiler: str,
    fail_on_missing: bool,
) -> dict:
    """
    Generic profiler subprocess launcher shared by nsys and ncu helpers.

    Returns a status dict with keys: profiler, model_id, trial_id, status,
    output_dir, error.
    """
    base: dict = {
        "profiler": profiler,
        "model_id": model_id,
        "trial_id": trial_id,
        "output_dir": str(output_dir),
        "status": "ok",
        "error": None,
    }

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            **_SUBPROCESS_FLAGS,
        )
        try:
            stdout_text, stderr_text = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                stdout_text, stderr_text = proc.communicate(timeout=10.0)
            except subprocess.TimeoutExpired:
                stdout_text, stderr_text = "", ""
            msg = f"{profiler} profiler timed out after {timeout_s}s"
            logger.error(msg + "  model_id=%d trial_id=%d", model_id, trial_id)
            base["status"] = "timeout"
            base["error"] = msg
            return base

        if proc.returncode != 0:
            msg = f"{profiler} exited with code {proc.returncode}"
            logger.warning(
                "%s  model_id=%d trial_id=%d stderr: %s",
                msg, model_id, trial_id, stderr_text[:400],
            )
            base["status"] = "failed"
            base["error"] = msg
        else:
            logger.debug(
                "%s pass OK  model_id=%d trial_id=%d", profiler, model_id, trial_id
            )

    except FileNotFoundError as exc:
        msg = f"{profiler} script not found: {exc}"
        if fail_on_missing:
            raise RuntimeError(msg) from exc
        logger.warning("Skipping %s pass: %s", profiler, msg)
        base["status"] = "missing"
        base["error"] = msg
    except OSError as exc:
        msg = f"Failed to launch {profiler} subprocess: {exc}"
        logger.error(msg + "  model_id=%d trial_id=%d", model_id, trial_id)
        base["status"] = "failed"
        base["error"] = msg

    return base


# ── Internal: subprocess launch ───────────────────────────────────────────────

def _run_subprocess_trial(cfg: TrialConfig) -> TrialResult:
    """
    Spawn a fresh Python process for one trial and collect its result JSON.

    The subprocess runs scripts/run_one_model.py, which calls runner.run_trial()
    and writes a JSON file at cfg.trial_dir()/model_<id>_result.json.
    This function reads that file back and deserialises it into a TrialResult.

    Process lifecycle (Windows-safe)
    ---------------------------------
    1. Popen with stdout=PIPE, stderr=PIPE to capture all subprocess output.
    2. communicate(timeout=cfg.timeout_s) blocks until the process exits.
    3. If TimeoutExpired is raised:
         a. proc.kill() terminates the child immediately (required on Windows
            where terminate() is also kill() but kill() is unambiguous).
         b. proc.communicate() drains pipes so the OS can release them.
    4. stdout + stderr are written to <trial_dir>/subprocess.log and emitted
       at DEBUG level for inline inspection during development.

    Returns a STATUS_FAILED TrialResult on any subprocess- or I/O-level error.
    """
    trial_dir = cfg.trial_dir()
    trial_dir.mkdir(parents=True, exist_ok=True)

    # Exact result file path — run_one_model.py must write here:
    result_json_exact = trial_dir / f"model_{cfg.model_id:03d}_result.json"

    scripts_dir = Path(__file__).parent.parent / "scripts"
    runner_script = scripts_dir / "run_one_model.py"

    cmd = [
        sys.executable,
        str(runner_script),
        "--model-id",   str(cfg.model_id),
        "--trial-id",   str(cfg.trial_id),
        "--output-dir", str(trial_dir),
        "--warmup",     str(cfg.warmup_iterations),
        "--iterations", str(cfg.measured_iterations),
        "--seed",       str(cfg.random_seed),
        "--device",     cfg.device,
        "--run-mode",   RunMode.CLEAN_BENCHMARK,
    ]

    logger.debug("Subprocess cmd: %s", " ".join(cmd))
    t0 = time.perf_counter()

    proc: Optional[subprocess.Popen] = None  # type: ignore[type-arg]
    stdout_text = ""
    stderr_text = ""
    timed_out = False

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            **_SUBPROCESS_FLAGS,
        )

        try:
            stdout_text, stderr_text = proc.communicate(timeout=cfg.timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            # On Windows, TimeoutExpired does NOT kill the child. Kill explicitly.
            proc.kill()
            # Drain pipes so the OS can release them; ignore further output.
            try:
                stdout_text, stderr_text = proc.communicate(timeout=10.0)
            except subprocess.TimeoutExpired:
                stdout_text, stderr_text = "", ""

        elapsed_s = time.perf_counter() - t0

        # Persist subprocess I/O for post-hoc debugging
        _write_subprocess_log(
            trial_dir=trial_dir,
            cmd=cmd,
            returncode=proc.returncode if not timed_out else -1,
            elapsed_s=elapsed_s,
            stdout=stdout_text,
            stderr=stderr_text,
            timed_out=timed_out,
        )

        # Emit subprocess output at DEBUG level for inline development use
        if stdout_text:
            logger.debug(
                "Subprocess stdout (model_id=%d trial_id=%d):\n%s",
                cfg.model_id, cfg.trial_id, stdout_text.rstrip(),
            )
        if stderr_text:
            logger.debug(
                "Subprocess stderr (model_id=%d trial_id=%d):\n%s",
                cfg.model_id, cfg.trial_id, stderr_text.rstrip(),
            )

        if timed_out:
            logger.error(
                "Subprocess timed out after %ds for model_id=%d trial_id=%d",
                cfg.timeout_s, cfg.model_id, cfg.trial_id,
            )
            return _subprocess_fail(
                cfg.model_id, cfg.trial_id,
                f"Subprocess timed out after {cfg.timeout_s}s",
            )

        if proc.returncode != 0:
            logger.warning(
                "Subprocess exited with code %d for model_id=%d trial_id=%d",
                proc.returncode, cfg.model_id, cfg.trial_id,
            )

    except OSError as exc:
        logger.error(
            "Failed to spawn subprocess for model_id=%d trial_id=%d: %s",
            cfg.model_id, cfg.trial_id, exc,
        )
        return _subprocess_fail(
            cfg.model_id, cfg.trial_id, f"Subprocess spawn failed: {exc}"
        )

    # ── Deserialise result JSON written by the subprocess ────────────────────
    # Primary: exact name matching the spec.
    # Fallback glob: tolerate minor variation introduced by helper scripts.
    candidates: list[Path] = []
    if result_json_exact.exists():
        candidates = [result_json_exact]
    else:
        candidates = list(trial_dir.glob(f"model_{cfg.model_id:03d}_*result*.json"))

    if not candidates:
        logger.error(
            "No result JSON found in %s for model_id=%d trial_id=%d",
            trial_dir, cfg.model_id, cfg.trial_id,
        )
        return _subprocess_fail(
            cfg.model_id, cfg.trial_id,
            "Subprocess did not produce a result file",
        )

    try:
        raw = json.loads(candidates[0].read_text(encoding="utf-8"))
        return TrialResult.from_dict(raw)
    except Exception as exc:
        logger.error(
            "Failed to parse result JSON %s: %s", candidates[0], exc
        )
        return _subprocess_fail(
            cfg.model_id, cfg.trial_id,
            f"Result JSON parse error: {exc}",
        )


def _write_subprocess_log(
    trial_dir: Path,
    cmd: list[str],
    returncode: int,
    elapsed_s: float,
    stdout: str,
    stderr: str,
    timed_out: bool,
) -> None:
    """
    Write subprocess stdout + stderr to <trial_dir>/subprocess.log.

    This file is the primary debugging aid when a trial fails silently.
    It is always written, even on success, so the full stdout (which may
    contain progress information) is always available.
    """
    log_path = trial_dir / "subprocess.log"
    lines = [
        f"Command : {' '.join(cmd)}",
        f"Exit code: {returncode}",
        f"Elapsed  : {elapsed_s:.2f}s",
        f"Timed out: {timed_out}",
        "",
        "--- STDOUT ---",
        stdout or "(empty)",
        "",
        "--- STDERR ---",
        stderr or "(empty)",
    ]
    try:
        log_path.write_text("\n".join(lines), encoding="utf-8")
    except OSError:
        pass  # non-critical; log failure silently


def _subprocess_fail(model_id: int, trial_id: int, error: str) -> TrialResult:
    """Build a minimal STATUS_FAILED TrialResult for subprocess-level errors."""
    return TrialResult(
        model_id=model_id,
        model_name=f"model_{model_id}",
        paper_arch="",
        native_framework="",
        native_model_name="",
        exactness_status="unknown",
        device_name="unknown",
        cuda_version="unknown",
        driver_version="unknown",
        trial_id=trial_id,
        process_fresh_start=True,
        status=STATUS_FAILED,
        error_message=error,
    )
