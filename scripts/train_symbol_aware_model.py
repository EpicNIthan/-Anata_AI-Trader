from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compatibility wrapper. Symbol-aware training is now built into scripts/train_best_model.py by default."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    args, remaining = parser.parse_known_args()

    train_script = _repo_root() / "scripts" / "train_best_model.py"
    command = [sys.executable, str(train_script), "--dataset", str(args.dataset), *remaining]
    print("Symbol-aware training is now built into train_best_model.py; forwarding command...", flush=True)
    raise SystemExit(subprocess.run(command, check=False).returncode)


if __name__ == "__main__":
    main()
