"""
Result schema for native-tfjs-bench.

TrialResult is the single source of truth for one (model_id, trial_id) pair.
It captures all metrics from the spec:

  - Model identity & mapping metadata
  - Hardware & software environment fingerprint
  - Timing: model load, warm-up, measured-phase aggregates
  - GPU telemetry: utilization, memory, power, energy
  - Outcome: success / skipped / failed / unsupported

AggregatedResult summarises cross-trial statistics after all 5 trials
complete (computed in post-processing, not inside the trial subprocess).
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Optional

# ── Schema version ─────────────────────────────────────────────────────────
# 1.1 — Added RunMode, run_mode and profiler_artifact_paths to TrialResult.
# 1.2 — Added PROFILE_NCU to RunMode; added NcuKernelResult dataclass.
SCHEMA_VERSION = "1.2"


# ── RunMode ─────────────────────────────────────────────────────────────────

class RunMode:
    """
    Execution mode constants for a benchmark trial.

    CLEAN_BENCHMARK
        Normal latency benchmark.  No profiler attached.  nvidia-smi polling
        is the only permitted third-party instrumentation.  All latency
        numbers produced in this mode are valid for reporting.

    PROFILE_NSYS
        The benchmark subprocess runs under ``nsys profile``.  Wall-clock
        and CUDA-event timings are biased by ~5–15 % overhead from nsys
        trace buffering.  Latency values in this mode MUST NOT be compared
        against paper numbers or clean-benchmark results.  nvidia-smi
        polling is suppressed.
    """

    CLEAN_BENCHMARK: str = "clean_benchmark"
    PROFILE_NSYS: str = "profile_nsys"
    PROFILE_NCU: str = "profile_ncu"

    _VALID: frozenset = frozenset({"clean_benchmark", "profile_nsys", "profile_ncu"})

    @classmethod
    def validate(cls, value: str) -> str:
        """Return *value* if recognised; raise ValueError otherwise."""
        if value not in cls._VALID:
            raise ValueError(
                f"Unknown RunMode {value!r}. Valid values: {sorted(cls._VALID)}"
            )
        return value

    @classmethod
    def is_profiling(cls, value: str) -> bool:
        """Return True when *value* indicates any profiler-attached run."""
        return value in (cls.PROFILE_NSYS, cls.PROFILE_NCU)

# ── Status constants ────────────────────────────────────────────────────────
STATUS_SUCCESS = "success"
STATUS_SKIPPED = "skipped"
STATUS_FAILED = "failed"
STATUS_UNSUPPORTED = "unsupported"

# ── Exactness constants (aligned with spec) ─────────────────────────────────
EXACTNESS_EXACT = "exact_official_equivalent"
EXACTNESS_NEAR = "near_equivalent_substitute"
EXACTNESS_PROXY = "proxy_reference_implementation"
EXACTNESS_UNSUPPORTED = "unsupported_no_equivalent"


@dataclass
class TrialResult:
    """
    Complete result record for one model trial.

    One TrialResult is emitted per (model_id, trial_id) pair and written to
    both a per-trial JSON file and the shared all_results.csv.

    Nullable fields (Optional[float]) are set to None when the underlying
    hardware or framework does not support that measurement.
    """

    # ── Model identity ──────────────────────────────────────────────────────
    model_id: int
    model_name: str
    paper_arch: str
    native_framework: str
    native_model_name: str
    exactness_status: str

    # ── Hardware / environment fingerprint ─────────────────────────────────
    device_name: str
    cuda_version: str
    driver_version: str
    cudnn_version: str = ""
    framework_version: str = ""

    # ── Trial identity ──────────────────────────────────────────────────────
    trial_id: int = 0
    # Always True in this suite; recorded for schema completeness
    process_fresh_start: bool = True

    # ── Timing: loading & warm-up (milliseconds) ───────────────────────────
    # Wall-clock time from process start to model ready (GPU weights loaded)
    model_load_ms: float = 0.0
    # Wall-clock time for the very first inference call (after load)
    first_inference_ms: float = 0.0
    # Wall-clock time for all 10 warm-up iterations combined
    warmup_total_ms: float = 0.0

    # ── Timing: measured phase (milliseconds) ──────────────────────────────
    measured_iterations: int = 1024

    # End-to-end (host submission → result retrieval)
    mean_inference_ms: float = 0.0
    std_inference_ms: float = 0.0
    p50_inference_ms: float = 0.0
    p95_inference_ms: float = 0.0
    p99_inference_ms: float = 0.0
    min_inference_ms: float = 0.0
    max_inference_ms: float = 0.0

    # GPU-side kernel time (CUDA Events; wall-clock fallback if unavailable)
    mean_kernel_ms: float = 0.0
    std_kernel_ms: float = 0.0
    p95_kernel_ms: float = 0.0

    # Alias kept for spec compatibility (same as mean_inference_ms here)
    mean_end_to_end_ms: float = 0.0

    # ── GPU telemetry (None = not available on this hardware/driver) ────────
    gpu_util_avg_pct: Optional[float] = None
    gpu_util_peak_pct: Optional[float] = None
    gpu_mem_avg_mb: Optional[float] = None
    gpu_mem_peak_mb: Optional[float] = None
    power_avg_w: Optional[float] = None
    power_peak_w: Optional[float] = None
    # Calculated: mean(power_w) × total_kernel_time_s
    energy_j: Optional[float] = None

    # ── Memory bandwidth (deferred to Phase 2) ─────────────────────────────
    mem_bandwidth_gbps: Optional[float] = None
    mem_bandwidth_supported: bool = False

    # ── Outcome ─────────────────────────────────────────────────────────────
    status: str = STATUS_SUCCESS
    error_message: Optional[str] = None

    # ── Run mode and profiler artefacts ─────────────────────────────────────
    # One of RunMode.CLEAN_BENCHMARK or RunMode.PROFILE_NSYS.
    # Aggregation scripts must exclude PROFILE_NSYS rows from latency stats.
    run_mode: str = RunMode.CLEAN_BENCHMARK

    # Mapping of artefact role → absolute file path string.
    # Populated by profile_with_nsys.py / profile_with_ncu.py; empty in clean runs.
    # Keys used by the Nsight Systems layer:
    #   "nsys_report"   — .nsys-rep binary report
    #   "nsys_sqlite"   — exported .sqlite from nsys export
    #   "nsys_summary"  — parsed NsysSummary JSON
    #   "nsys_stdout"   — captured nsys profile stdout log
    #   "nsys_stderr"   — captured nsys profile stderr log
    # Keys used by the Nsight Compute layer:
    #   "ncu_report"    — .ncu-rep binary report
    #   "ncu_summary"   — parsed NcuProfilingResult JSON
    #   "ncu_stdout"    — captured ncu stdout (CSV metric dump)
    #   "ncu_stderr"    — captured ncu stderr log
    profiler_artifact_paths: dict = field(default_factory=dict)

    # ── Schema metadata ─────────────────────────────────────────────────────
    schema_version: str = SCHEMA_VERSION

    # ── Serialization ───────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Convert to plain dict; safe for JSON serialisation."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialise to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, d: dict) -> "TrialResult":
        """
        Deserialise from a plain dict.
        Unknown keys are silently dropped for forward-compatibility.
        """
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def is_valid(self) -> bool:
        """Minimal sanity check: required identity fields are non-empty."""
        return bool(self.model_name and self.device_name and self.status)


@dataclass
class AggregatedResult:
    """
    Cross-trial aggregate for one model.

    Computed in post-processing after all trials for a model complete.
    Not produced inside trial subprocesses.
    """

    model_id: int
    model_name: str
    exactness_status: str
    num_trials: int
    num_successful_trials: int

    # Latency aggregates across trials
    mean_inference_ms_mean: float = 0.0
    mean_inference_ms_std: float = 0.0
    p95_inference_ms_mean: float = 0.0

    mean_kernel_ms_mean: float = 0.0

    # Power / energy across trials
    power_avg_w_mean: Optional[float] = None
    energy_j_mean: Optional[float] = None

    # Memory
    gpu_mem_peak_mb_max: Optional[float] = None

    status: str = STATUS_SUCCESS
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ── I/O helpers ─────────────────────────────────────────────────────────────

def write_trial_result_json(result: TrialResult, path: Path) -> None:
    """Write a TrialResult to a JSON file (overwrites if exists)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.to_json(), encoding="utf-8")


def write_trial_result_csv(result: TrialResult, path: Path) -> None:
    """
    Append a single TrialResult row to an aggregate CSV file.
    Writes the header row if the file does not yet exist.
    """
    row = result.to_dict()
    write_header = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_trial_results_csv(path: Path) -> list[TrialResult]:
    """Load all TrialResult rows from an aggregate CSV."""
    results: list[TrialResult] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            results.append(TrialResult.from_dict(row))
    return results


# ── NcuKernelResult ──────────────────────────────────────────────────────────

@dataclass
class NcuKernelResult:
    """
    Kernel-level metric record from one Nsight Compute profiling run.

    One NcuKernelResult is produced per (kernel_launch_id, metric_category)
    grouping, allowing downstream analysis to filter by kernel name or
    category independently.

    Fields satisfy the spec requirement for:
      profiler_mode, profiler_tool, raw_report_path,
      parsed_metric_dict, kernel_name, metric_category.

    Notes
    -----
    Wall-clock and kernel-duration values captured while ncu is active
    are inflated 10×–100× by hardware-counter replay and MUST NOT be
    compared against TrialResult latency numbers from clean runs.
    """

    # Required spec fields
    profiler_mode: str      # RunMode.PROFILE_NCU
    profiler_tool: str      # "ncu"
    raw_report_path: str    # absolute path to .ncu-rep binary report
    kernel_name: str        # demangled kernel function name
    metric_category: str    # MetricCategory constant

    # metric_name → numeric value (None = unsupported on this GPU)
    parsed_metric_dict: dict = field(default_factory=dict)

    # Optional context fields
    ncu_version: str = ""
    kernel_launch_id: int = 0   # 1-based kernel launch counter from ncu
    stream_id: str = ""
    context_id: str = ""

    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict:
        """Convert to plain dict; safe for JSON serialisation."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialise to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
