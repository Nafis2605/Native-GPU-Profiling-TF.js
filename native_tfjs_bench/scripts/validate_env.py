"""
Standalone environment validation script.

Checks CUDA availability, GPU info, framework versions (PyTorch, ONNX Runtime,
TensorRT, MediaPipe), and NVML telemetry support. Prints a human-readable
report and exits with code 0 (pass) or 1 (fail).

Usage
-----
    python scripts/validate_env.py
    python scripts/validate_env.py --device-index 0
    python scripts/validate_env.py --json        # machine-readable output

Equivalent to: native-bench validate-env
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is importable when run as a plain script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.env_check import check_environment, print_env_report  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the native-tfjs-bench runtime environment."
    )
    parser.add_argument(
        "--device-index", type=int, default=0,
        help="CUDA device index to probe (default: 0)."
    )
    parser.add_argument(
        "--json", action="store_true", dest="as_json",
        help="Emit machine-readable JSON report instead of human-readable text."
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = check_environment(device_index=args.device_index)

    if args.as_json:
        # Emit dataclass as JSON for CI / downstream tooling
        from dataclasses import asdict
        print(json.dumps(asdict(report), indent=2))
    else:
        print_env_report(report)

    return 0 if report.all_critical_passed else 1


if __name__ == "__main__":
    sys.exit(main())
