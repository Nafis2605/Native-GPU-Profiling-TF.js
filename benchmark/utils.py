"""
Shared utilities for native-tfjs-bench.

Provides:
  configure_logging  – Root logger setup (console + optional file handler)
  set_global_seed    – Deterministic seeding across Python / NumPy / PyTorch
  generate_random_input – Reproducible synthetic tensor generation
  compute_statistics – Percentile / std aggregation over timing lists
  resolve_output_dir – Canonical output path builder
  get_project_root   – Path to the repository root

All utilities are pure functions with no side effects on benchmark state.
"""

from __future__ import annotations

import logging
import math
import random
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np


# ── Logging ──────────────────────────────────────────────────────────────────

def configure_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """
    Configure the root logger with a console handler and an optional file handler.

    Args:
        level:    Log level string: "DEBUG" | "INFO" | "WARNING" | "ERROR".
        log_file: If given, also write to this file (created with parents).

    Returns:
        The root logger after configuration.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(log_file), encoding="utf-8"))

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("native_tfjs_bench")


# ── Seeding ───────────────────────────────────────────────────────────────────

def set_global_seed(seed: int = 12345) -> None:
    """
    Set all random seeds for deterministic input generation.

    Covers Python built-in random, NumPy, and (if installed) PyTorch CPU
    and CUDA RNGs. Call once per trial subprocess before generating inputs.

    Args:
        seed: Integer seed value (default matches spec: 12345).
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ── Input generation ─────────────────────────────────────────────────────────

def generate_random_input(
    shape: list[int] | tuple[int, ...],
    dtype: str = "float32",
    seed: int = 12345,
    framework: str = "numpy",
) -> Any:
    """
    Generate a deterministic synthetic input tensor.

    Do NOT read from disk inside this function; the result is held in memory
    before the measured phase begins to avoid I/O confounding timings.

    Args:
        shape:     Tensor dimensions, e.g. [1, 3, 224, 224].
        dtype:     NumPy dtype name: "float32" | "float16" | "int64" | "uint8".
        seed:      Random seed for reproducibility across trials.
        framework: "numpy" returns an ndarray.
                   "torch"  returns a torch.Tensor.

    Returns:
        numpy ndarray or torch.Tensor depending on the framework arg.

    Raises:
        ImportError: If framework="torch" but torch is not installed.
    """
    rng = np.random.default_rng(seed=seed)
    np_dtype = np.dtype(dtype)

    if np_dtype.kind == "f":
        # Float: uniform [0, 1)
        arr = rng.random(shape).astype(np_dtype)
    elif np_dtype.kind == "i":
        # Signed int: uniform [0, 1000)
        arr = rng.integers(0, 1_000, size=shape, dtype=np_dtype)
    elif np_dtype.kind == "u":
        # Unsigned int: uniform [0, 256)
        arr = rng.integers(0, 256, size=shape, dtype=np_dtype)
    else:
        arr = rng.random(shape).astype(np_dtype)

    if framework == "torch":
        import torch
        return torch.from_numpy(arr)
    return arr


# ── Statistics ────────────────────────────────────────────────────────────────

def compute_statistics(values: list[float]) -> dict[str, float]:
    """
    Compute summary statistics for a list of numeric values.

    Used to aggregate per-iteration timing / metric lists into the fields
    required by TrialResult (mean, std, p50, p95, p99, min, max).

    Args:
        values: List of float measurements (e.g. per-iteration wall_clock_ms).

    Returns:
        Dict with keys: mean, std, min, max, p50, p95, p99.
        All values are 0.0 if the input list is empty.
    """
    empty = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
             "p50": 0.0, "p95": 0.0, "p99": 0.0}
    if not values:
        return empty

    n = len(values)
    sorted_vals = sorted(values)

    def _percentile(p: float) -> float:
        # Linear interpolation percentile
        if n == 1:
            return sorted_vals[0]
        idx = (p / 100.0) * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac

    mean = sum(values) / n
    # Bessel-corrected sample variance
    variance = sum((x - mean) ** 2 for x in values) / max(n - 1, 1)

    return {
        "mean": mean,
        "std": math.sqrt(variance),
        "min": sorted_vals[0],
        "max": sorted_vals[-1],
        "p50": _percentile(50),
        "p95": _percentile(95),
        "p99": _percentile(99),
    }


# ── Path helpers ──────────────────────────────────────────────────────────────

def get_project_root() -> Path:
    """Return the repository root (parent of the benchmark/ package)."""
    return Path(__file__).parent.parent


def resolve_output_dir(
    base_dir: str | Path,
    trial_id: int,
    model_id: int,
) -> Path:
    """
    Return the canonical output directory for a (trial_id, model_id) pair.

    Directory pattern: <base_dir>/trial_<NNN>/model_<NNN>/

    Args:
        base_dir: Root output directory (from device_config.yaml).
        trial_id: 0-based trial index.
        model_id: Model ID from registry.

    Returns:
        Path object (not yet created; call .mkdir(parents=True, exist_ok=True) yourself).
    """
    return Path(base_dir) / f"trial_{trial_id:03d}" / f"model_{model_id:03d}"
