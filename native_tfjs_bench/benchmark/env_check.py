"""
Environment validation for native-tfjs-bench.

Checks all runtime dependencies required for the benchmark:
  - CUDA availability and version (critical)
  - PyTorch version and CUDA backend
  - cuDNN version
  - ONNX Runtime GPU provider
  - TensorRT Python bindings
  - MediaPipe
  - pynvml (GPU telemetry)
  - NVIDIA driver version via NVML

Call check_environment() before launching any benchmark run.
Use print_env_report() to display results in a human-readable format.

Exit behaviour in scripts/validate_env.py:
  sys.exit(0)  if all_critical_passed is True
  sys.exit(1)  otherwise
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EnvReport:
    """
    Aggregated environment check results.

    all_critical_passed is True only when CUDA is available and no
    critical errors were recorded. Non-critical issues appear in warnings.
    """

    # Python
    python_version: str = ""

    # PyTorch / CUDA
    cuda_available: bool = False
    cuda_version: str = "unavailable"
    cudnn_version: str = "unavailable"
    torch_version: str = "unavailable"
    gpu_name: str = "unavailable"
    gpu_total_memory_gb: float = 0.0
    # VRAM in GiB (same as gpu_total_memory_gb, kept for explicit schema mapping)
    gpu_vram_gb: float = 0.0
    compute_capability: str = "unavailable"   # e.g. "8.9" for Ada Lovelace
    driver_version: str = "unavailable"

    # nvidia-smi binary (critical for telemetry)
    nvidia_smi_available: bool = False
    nvidia_smi_path: str = "not found"
    nvidia_smi_version: str = "unavailable"

    # ONNX Runtime (optional but needed for most models)
    onnxruntime_available: bool = False
    onnxruntime_version: str = "unavailable"
    onnxruntime_providers: list[str] = field(default_factory=list)

    # TensorRT (optional)
    tensorrt_available: bool = False
    tensorrt_version: str = "unavailable"

    # MediaPipe (needed for models 2, 7, 9)
    mediapipe_available: bool = False
    mediapipe_version: str = "unavailable"

    # pynvml (needed for GPU telemetry)
    nvml_available: bool = False
    pynvml_version: str = "unavailable"

    # Outcome
    all_critical_passed: bool = False
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def check_environment(device_index: int = 0) -> EnvReport:
    """
    Run all environment checks and return a populated EnvReport.

    Args:
        device_index: CUDA device index to probe (default 0).

    Returns:
        EnvReport with version information and pass/fail status.
    """
    report = EnvReport()
    report.python_version = sys.version

    _check_torch(report, device_index)
    _check_nvidia_smi(report)
    _check_onnxruntime(report)
    _check_tensorrt(report)
    _check_mediapipe(report)
    _check_nvml(report)
    _evaluate_critical(report)

    return report


# ── Individual checks ────────────────────────────────────────────────────────

def _check_torch(report: EnvReport, device_index: int) -> None:
    try:
        import torch
        report.torch_version = torch.__version__
        if torch.cuda.is_available():
            report.cuda_available = True
            report.cuda_version = torch.version.cuda or "unknown"
            report.gpu_name = torch.cuda.get_device_name(device_index)
            props = torch.cuda.get_device_properties(device_index)
            mem_gib = round(props.total_memory / (1024 ** 3), 2)
            report.gpu_total_memory_gb = mem_gib
            report.gpu_vram_gb = mem_gib
            report.compute_capability = f"{props.major}.{props.minor}"
            try:
                report.cudnn_version = str(torch.backends.cudnn.version())
            except Exception:
                report.cudnn_version = "unknown"
        else:
            report.errors.append(
                "torch.cuda.is_available() is False. "
                "Ensure a CUDA-enabled PyTorch wheel is installed."
            )
    except ImportError:
        report.errors.append(
            "PyTorch is not installed. "
            "Install via: pip install torch --index-url https://download.pytorch.org/whl/cu121"
        )


def _check_nvidia_smi(report: EnvReport) -> None:
    """
    Detect whether the nvidia-smi binary is on PATH and record its version.

    nvidia-smi is the primary GPU telemetry backend (NvidiaSmiTelemetry uses
    it at 1-second intervals). Its absence is a WARNING — benchmarks can still
    run but GPU utilisation, power, and energy fields will all be None.

    Detection strategy:
      1. shutil.which("nvidia-smi") for fast binary lookup.
      2. Run nvidia-smi -L (list GPUs) for a smoke test and version string.
    """
    smi_path = shutil.which("nvidia-smi")
    if smi_path is None:
        report.nvidia_smi_available = False
        report.nvidia_smi_path = "not found"
        report.warnings.append(
            "nvidia-smi is not on PATH. GPU utilisation, power draw, and energy "
            "telemetry will be unavailable. Ensure NVIDIA drivers are installed "
            "and that nvidia-smi is accessible (typical path on Windows: "
            r"C:\Windows\System32\nvidia-smi.exe)."
        )
        return

    report.nvidia_smi_path = smi_path

    # Smoke-test: nvidia-smi --version outputs a version line without touching
    # the GPU, so it works even when CUDA is not available to PyTorch.
    try:
        result = subprocess.run(
            [smi_path, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            report.nvidia_smi_available = True
            # Extract version from first line of output like:
            #   NVIDIA-SMI 535.86.10  Driver Version: 535.86.10  CUDA Version: 12.2
            first_line = result.stdout.strip().splitlines()[0] if result.stdout else ""
            report.nvidia_smi_version = first_line[:80] if first_line else "unknown"
        else:
            report.nvidia_smi_available = False
            report.warnings.append(
                f"nvidia-smi found at {smi_path} but exited with code "
                f"{result.returncode}. Telemetry may not function correctly."
            )
    except subprocess.TimeoutExpired:
        report.nvidia_smi_available = False
        report.warnings.append(
            f"nvidia-smi at {smi_path} timed out during version check."
        )
    except OSError as exc:
        report.nvidia_smi_available = False
        report.warnings.append(
            f"nvidia-smi found at {smi_path} but could not be executed: {exc}"
        )


def _check_onnxruntime(report: EnvReport) -> None:
    try:
        import onnxruntime as ort
        report.onnxruntime_available = True
        report.onnxruntime_version = ort.__version__
        report.onnxruntime_providers = ort.get_available_providers()
        if "CUDAExecutionProvider" not in report.onnxruntime_providers:
            report.warnings.append(
                "onnxruntime CUDAExecutionProvider is not available. "
                "Install onnxruntime-gpu (not onnxruntime) to enable GPU inference "
                "for ONNX models."
            )
    except ImportError:
        report.warnings.append(
            "onnxruntime is not installed (required for 7 of 10 models). "
            "Install via: pip install onnxruntime-gpu"
        )


def _check_tensorrt(report: EnvReport) -> None:
    try:
        import tensorrt as trt
        report.tensorrt_available = True
        report.tensorrt_version = trt.__version__
    except ImportError:
        report.warnings.append(
            "TensorRT Python bindings are not installed (optional backend). "
            "Install from NVIDIA developer toolkit if TensorRT-optimised engines are needed."
        )


def _check_mediapipe(report: EnvReport) -> None:
    try:
        import mediapipe as mp
        report.mediapipe_available = True
        report.mediapipe_version = mp.__version__
    except ImportError:
        report.warnings.append(
            "mediapipe is not installed (required for models 2=HandPose, "
            "7=PortraitDepth, 9=PoseNet). Install via: pip install mediapipe"
        )


def _check_nvml(report: EnvReport) -> None:
    try:
        import pynvml
        pynvml.nvmlInit()
        report.nvml_available = True
        report.pynvml_version = getattr(pynvml, "__version__", "unknown")
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            raw = pynvml.nvmlDeviceGetDriverVersion(handle)
            report.driver_version = (
                raw.decode() if isinstance(raw, bytes) else str(raw)
            )
        except Exception:
            report.warnings.append("NVML initialised but driver version query failed.")
    except Exception:
        report.warnings.append(
            "pynvml is not available or NVML init failed. "
            "GPU utilization, memory, and power telemetry will be absent. "
            "Install via: pip install pynvml"
        )


def _evaluate_critical(report: EnvReport) -> None:
    """
    Set all_critical_passed.
    Critical requirement: CUDA must be available (torch + GPU).
    All other checks produce warnings, not failures.
    """
    report.all_critical_passed = report.cuda_available and len(report.errors) == 0


# ── Display ──────────────────────────────────────────────────────────────────

def print_env_report(report: EnvReport) -> None:
    """Print a concise environment summary to stdout."""
    SEP = "=" * 60
    print(f"\n{SEP}")
    print("  native-tfjs-bench  —  Environment Report")
    print(SEP)
    print(f"  Python             : {report.python_version.split()[0]}")
    print(f"  PyTorch            : {report.torch_version}")
    print(f"  CUDA available     : {report.cuda_available}")
    if report.cuda_available:
        print(f"  CUDA version       : {report.cuda_version}")
        print(f"  cuDNN version      : {report.cudnn_version}")
        print(f"  GPU                : {report.gpu_name}")
        print(f"  Compute capability : {report.compute_capability}")
        print(f"  VRAM               : {report.gpu_vram_gb:.1f} GiB")
        print(f"  Driver             : {report.driver_version}")
    smi_status = "OK" if report.nvidia_smi_available else "NOT FOUND"
    print(f"  nvidia-smi         : {smi_status}  ({report.nvidia_smi_path})")
    if report.nvidia_smi_available:
        print(f"    version          : {report.nvidia_smi_version}")
    print(f"  ONNX Runtime       : {report.onnxruntime_version}"
          f"  ({'OK' if report.onnxruntime_available else 'NOT INSTALLED'})")
    if report.onnxruntime_available and report.onnxruntime_providers:
        cuda_ok = "CUDAExecutionProvider" in report.onnxruntime_providers
        print(f"    CUDA provider    : {'available' if cuda_ok else 'MISSING — install onnxruntime-gpu'}")
    print(f"  TensorRT           : {report.tensorrt_version}"
          f"  ({'OK' if report.tensorrt_available else 'NOT INSTALLED'})")
    print(f"  MediaPipe          : {report.mediapipe_version}"
          f"  ({'OK' if report.mediapipe_available else 'NOT INSTALLED'})")
    print(f"  pynvml             : {report.pynvml_version}"
          f"  ({'OK' if report.nvml_available else 'NOT INSTALLED'})")

    if report.warnings:
        print(f"\n  WARNINGS ({len(report.warnings)}):")
        for w in report.warnings:
            # Wrap long warning text for readability
            print(f"    [WARN]  {w}")
    if report.errors:
        print(f"\n  ERRORS ({len(report.errors)}):")
        for e in report.errors:
            print(f"    [ERROR] {e}")

    status = "PASS" if report.all_critical_passed else "FAIL"
    print(f"\n  Overall status : {status}")
    print(SEP + "\n")
