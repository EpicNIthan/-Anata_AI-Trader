from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path

from prepare_training_data import _process


def _extract_daily_zips(input_dir: Path, extract_root: Path) -> list[str]:
    zip_paths = sorted(input_dir.glob("raw_*.zip")) or sorted(input_dir.glob("*.zip"))
    if not zip_paths:
        raise SystemExit(f"No raw day ZIP files found in {input_dir}. Expected files like raw_2026-06-27.zip")

    extracted: list[str] = []
    for zip_path in zip_paths:
        target = extract_root / zip_path.stem
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(target)
        extracted.append(str(zip_path))
    return extracted


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a folder of raw_YYYY-MM-DD.zip files into one training-ready dataset.")
    parser.add_argument("--input-dir", type=Path, default=Path("datasets/raw_days"), help="Folder containing raw_YYYY-MM-DD.zip files.")
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/processed"))
    parser.add_argument("--news-lookback-hours", type=float, default=6.0)
    parser.add_argument("--fee-rate", type=float, default=0.0004)
    parser.add_argument(
        "--news-converter",
        choices=["smart", "finbert", "cryptobert", "rule-based"],
        default="smart",
        help="Use smart/finbert/cryptobert on your PC. rule-based is only the weak fallback.",
    )
    args = parser.parse_args()

    if not args.input_dir.exists() or not args.input_dir.is_dir():
        raise SystemExit(f"Input folder not found: {args.input_dir}")

    with tempfile.TemporaryDirectory(prefix="anata_raw_days_") as temp_dir:
        temp_root = Path(temp_dir)
        extracted = _extract_daily_zips(args.input_dir, temp_root)
        result = _process(temp_root, args.output_dir, args.news_lookback_hours, args.fee_rate, args.news_converter)
        result["raw_day_files_used"] = extracted
        result["raw_day_file_count"] = len(extracted)
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
