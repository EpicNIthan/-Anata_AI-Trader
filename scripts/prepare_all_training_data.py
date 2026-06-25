from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from prepare_training_data import _process


def _extract_archives(input_path: Path, target_root: Path) -> dict[str, object]:
    archives: list[Path] = []
    raw_folders: list[Path] = []
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        archives = [input_path]
    elif input_path.is_dir():
        archives = sorted(input_path.rglob("*.zip"))
        raw_folders = [path for path in input_path.rglob("manifest.json") if path.parent not in {archive.parent for archive in archives}]
    else:
        raise SystemExit(f"Input path not found: {input_path}")

    extracted = []
    for index, archive_path in enumerate(archives, start=1):
        destination = target_root / f"archive_{index:04d}_{archive_path.stem}"
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(destination)
        extracted.append(str(archive_path))

    # If the input is already an extracted raw folder tree, process it directly by
    # copying nothing. _process reads recursively, so we can pass input_path later.
    return {
        "archives_found": len(archives),
        "archives_extracted": extracted,
        "raw_manifest_folders_found": len(raw_folders),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare one training dataset from many raw daily ZIP files or raw folders.")
    parser.add_argument("--input", type=Path, required=True, help="Folder containing raw_*.zip files, a single ZIP, or an extracted raw folder.")
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/processed"))
    parser.add_argument("--news-lookback-hours", type=float, default=6.0)
    parser.add_argument("--fee-rate", type=float, default=0.0004)
    parser.add_argument(
        "--news-converter",
        choices=["smart", "finbert", "cryptobert", "rule-based"],
        default="smart",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="anata_all_raw_") as temp_dir:
        temp_root = Path(temp_dir)
        info = _extract_archives(args.input, temp_root)
        root_to_process = temp_root if info["archives_found"] else args.input
        result = _process(root_to_process, args.output_dir, args.news_lookback_hours, args.fee_rate, args.news_converter)
        result["multi_archive_input"] = {
            "input": str(args.input),
            **info,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
