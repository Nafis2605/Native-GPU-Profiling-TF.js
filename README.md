# native-tfjs-bench

Native Windows + NVIDIA CUDA benchmark suite extending the browser-based TensorFlow.js
benchmarking paper. Measures inference performance for 10 pretrained TF.js model equivalents
on a CUDA GPU with full telemetry (latency, kernel time, GPU utilization, memory, power, energy).

## Requirements

- Windows 10 / 11
- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.8+ (12.0+ recommended), driver 450+
- Python 3.10+

## Installation

```powershell
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install project and core dependencies
pip install -e .

# Install ONNX Runtime GPU backend (optional but needed for most models)
pip install onnxruntime-gpu

# Full install including optional backends
pip install -r requirements.txt
```

## Quick Start

```powershell
# 1. Verify environment
python scripts/validate_env.py

# 2. Run full benchmark (all 10 models, 5 trials each)
python scripts/run_all.py

# 3. Run a single model by ID
python scripts/run_one_model.py --model-id 6 --trial-id 0

# 4. CLI (after pip install -e .)
native-bench validate-env
native-bench list-models
native-bench run --config configs/experiment_manifest.yaml
native-bench run-model --model-id 1

# 5. Run with Nsight Systems diagnostic pass on trial 0 for all models
native-bench run --mode nsys --profile-trials 0

# 6. Run with Nsight Compute on models 1 and 6 only
native-bench run --mode ncu --profile-models 1,6 --profile-trials 0,1

# 7. Full hybrid run: clean benchmark + both profilers, keep raw artifacts
native-bench run --mode hybrid --keep-raw-profiler-artifacts
```

## Experiment Modes

| Mode     | Description                                                        |
|----------|--------------------------------------------------------------------|
| `clean`  | **Publishable benchmark.** No profiler overhead. All latency numbers are valid for reporting. |
| `nsys`   | Clean trials first, then Nsight Systems diagnostic pass for selected models/trials. |
| `ncu`    | Clean trials first, then Nsight Compute diagnostic pass (kernel-level counters). |
| `hybrid` | Clean trials first, then **both** nsys and ncu diagnostic passes.  |

**Key design invariant:** the clean latency phase always runs first with no heavy profiler
attached. Profiler re-runs are isolated in `profiling/nsys/` and `profiling/ncu/`
sub-directories — they can never overwrite or contaminate clean latency results.

> **Warning:** ncu replay inflates execution time 10×–100×. Timing values from ncu
> runs are **diagnostic only** and must never be mixed into clean latency statistics.

### Profiling CLI Options

| Flag | Default | Description |
|---|---|---|
| `--mode` | `clean` | Experiment mode (clean / nsys / ncu / hybrid) |
| `--profile-trials` | `0` | Comma-separated trial indices to profile |
| `--profile-iterations` | `20` | Measured iterations for profiler re-runs |
| `--profile-models` | all | Comma-separated model IDs to profile |
| `--keep-raw-profiler-artifacts` | off | Retain binary `.nsys-rep` / `.ncu-rep` files |
| `--fail-on-missing-profiler` | off | Abort if nsys/ncu binary not found (default: warn and skip) |

## Project Structure

```
native_tfjs_bench/
├── README.md
├── requirements.txt
├── pyproject.toml
│
├── configs/
│   ├── experiment_manifest.yaml   # Model list, trial settings, profiler config
│   └── device_config.yaml         # GPU settings, NVML, output paths
│
├── docs/
│   ├── native_tfjs_pretrained_spec.md   # Specification document
│   └── nsight_measurement_plan.md       # Profiling measurement plan
│
├── benchmark/                     # Core package
│   ├── __init__.py
│   ├── cli.py                     # Click CLI (native-bench entrypoint)
│   ├── runner.py                  # In-process trial execution
│   ├── trial_manager.py           # Subprocess orchestration + ExperimentMode/Config
│   ├── timing.py                  # WallClockTimer + CudaEventTimer
│   ├── telemetry.py               # NVML GPU telemetry (util, mem, power)
│   ├── result_schema.py           # TrialResult, RunMode, NcuKernelResult
│   ├── env_check.py               # Environment validation
│   ├── utils.py                   # Logging, seeding, statistics helpers
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py                # BaseModel + ProfilingHint
│   │   └── registry.py            # MODEL_REGISTRY dict (all 10 models)
│   │
│   ├── profilers/
│   │   ├── __init__.py
│   │   ├── base.py                # BaseProfiler abstract class
│   │   ├── nsys_runner.py         # NsysRunner — subprocess nsys profile
│   │   ├── nsys_parser.py         # Parse nsys SQLite / JSON exports
│   │   ├── ncu_runner.py          # NcuRunner — subprocess ncu profiling
│   │   └── ncu_parser.py          # Parse ncu CSV exports
│   │
│   └── output/                    # Benchmark results (gitignored)
│       └── .gitkeep
│
└── scripts/
    ├── run_all.py                  # Run full suite
    ├── run_one_model.py            # Fresh-process trial entry (called by trial_manager)
    ├── validate_env.py             # Standalone environment check
    ├── profile_with_nsys.py        # Nsight Systems profiling entry point
    └── profile_with_ncu.py         # Nsight Compute profiling entry point
```

## Trial Structure

Each model is benchmarked in **5 independent trials**. Each trial runs in a **fresh subprocess**
to ensure cold-start model loading, matching the paper's warm-up semantics.

Per trial:
1. Fresh process start → model loaded from disk
2. **10 warm-up iterations** (discarded, not measured)
3. **1024 measured iterations** (full per-iteration timing + telemetry)
4. Results written to CSV + JSON

## Output Files

```
benchmark/output/
├── trial_000/
│   └── model_001/
│       ├── model_001_selfie_segmentation_iterations.csv   # Per-iteration timings
│       ├── model_001_selfie_segmentation_result.json      # Trial summary
│       ├── subprocess.log
│       └── profiling/                                     # Only present in nsys/ncu/hybrid modes
│           ├── nsys/
│           │   ├── nsys_profile.nsys-rep                  # Raw trace (--keep-raw-profiler-artifacts)
│           │   └── nsys_summary.json                      # Parsed kernel timeline
│           └── ncu/
│               ├── ncu_report.ncu-rep                     # Raw report (--keep-raw-profiler-artifacts)
│               └── ncu_metrics.json                       # Parsed hardware counters
├── ...
└── all_results.csv                                        # All trials, all models
```

## Model Coverage

| ID | Model | Exactness |
|---|---|---|
| 1  | Selfie Segmentation       | near_equivalent_substitute |
| 2  | Hand Pose 3D              | exact_official_equivalent  |
| 3  | Speech Command Recognizer | near_equivalent_substitute |
| 4  | COCO-SSD                  | near_equivalent_substitute |
| 5  | MobileBERT                | near_equivalent_substitute |
| 6  | MobileNetV3               | near_equivalent_substitute |
| 7  | AR PortraitDepth          | exact_official_equivalent  |
| 8  | BodyPix                   | near_equivalent_substitute |
| 9  | PoseNet                   | exact_official_equivalent  |
| 10 | DeepLabV3                 | near_equivalent_substitute |

## Reproducibility

- Fixed random seed (default: `12345`) for all input generation
- Fresh subprocess per trial — no shared GPU state between trials
- All input tensors pre-generated before the measured phase (no I/O jitter)
- Hardware fingerprint (GPU name, CUDA version, driver) recorded in every result
- See `docs/native_tfjs_pretrained_spec.md` for the full specification

## Implementation Status

Model implementations are stubs (Phase 1B infrastructure only). See `benchmark/models/registry.py`
and the specification checklist for Phase 1C model conversion tasks.
