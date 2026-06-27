from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _load_pandas():
    try:
        import pandas as pd
    except Exception as exc:
        raise SystemExit("Install local training dependencies first: pip install -r requirements-local-training.txt") from exc
    return pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_symbol_features(dataset: Path, output_dir: Path) -> Path:
    sys.path.insert(0, str(_repo_root()))
    from app.ai.symbol_identity import SYMBOL_FEATURE_COLUMNS, symbol_identity_values

    pd = _load_pandas()
    frame = pd.read_csv(dataset)
    if "symbol" not in frame.columns:
        raise SystemExit("Dataset is missing symbol column; cannot build symbol-aware features.")
    symbol_features = [symbol_identity_values(symbol) for symbol in frame["symbol"]]
    symbol_frame = pd.DataFrame(symbol_features, index=frame.index)
    for column in SYMBOL_FEATURE_COLUMNS:
        frame[column] = symbol_frame[column].astype(float)

    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"symbol_aware_{dataset.stem}_{stamp}.csv.gz"
    frame.to_csv(output_path, index=False, compression="gzip")

    report_path = output_dir / f"symbol_aware_report_{stamp}.json"
    report = {
        "status": "ok",
        "source_dataset": str(dataset),
        "symbol_aware_dataset": str(output_path),
        "rows": int(len(frame)),
        "symbols": sorted(frame["symbol"].dropna().astype(str).str.upper().unique().tolist()),
        "symbol_feature_columns": SYMBOL_FEATURE_COLUMNS,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add symbol identity columns to a processed dataset, then run scripts/train_best_model.py."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--symbol-output-dir", type=Path, default=Path("datasets/processed_symbol_aware"))
    args, remaining = parser.parse_known_args()

    symbol_dataset = _add_symbol_features(args.dataset, args.symbol_output_dir)
    train_script = _repo_root() / "scripts" / "train_best_model.py"
    command = [sys.executable, str(train_script), "--dataset", str(symbol_dataset), *remaining]
    raise SystemExit(subprocess.run(command, check=False).returncode)


if __name__ == "__main__":
    main()
