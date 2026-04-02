"""
Entry script: run the full benchmark suite.

Equivalent to: native-bench run [OPTIONS]

Usage
-----
    python scripts/run_all.py
    python scripts/run_all.py --config configs/experiment_manifest.yaml
    python scripts/run_all.py --model-ids 1,6 --output-dir benchmark/output
    python scripts/run_all.py --help
"""

import sys
from pathlib import Path

# Ensure the project root is importable when running as a plain script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.cli import main  # noqa: E402

if __name__ == "__main__":
    # Inject the "run" subcommand so sys.argv only needs to carry options
    main(["run", *sys.argv[1:]], standalone_mode=True)
