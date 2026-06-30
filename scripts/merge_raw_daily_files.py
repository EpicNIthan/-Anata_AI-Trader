from __future__ import annotations

import argparse
import gzip
import json
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def _zip_folder(folder: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(folder))


def _read_json_file(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _merge_gzip_text_files(files: list[Path], output: Path, *, csv_mode: bool = False) -> None:
    seen: set[str] = set()
    wrote_header = False
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wt", encoding="utf-8", newline="") as writer:
        for source in files:
            if not source.exists():
                continue
            with gzip.open(source, "rt", encoding="utf-8", errors="replace") as reader:
                for line_number, line in enumerate(reader):
                    key = line.rstrip("\r\n")
                    if not key:
                        continue
                    if csv_mode and line_number == 0:
                        if wrote_header:
                            continue
                        wrote_header = True
                    if key in seen:
                        continue
                    seen.add(key)
                    writer.write(line if line.endswith("\n") else f"{line}\n")


def _merge_manifest_files(files: list[Path], output: Path, input_zips: list[Path]) -> None:
    manifests = [_read_json_file(path) for path in files if path.exists()]
    base = next((manifest for manifest in reversed(manifests) if manifest), {})
    merged = dict(base)
    merged["manual_raw_daily_merge"] = {
        "merged_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "input_files": [str(path) for path in input_zips],
        "input_manifests": manifests,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(merged, indent=2, sort_keys=True), encoding="utf-8")


def merge_raw_daily_zips(inputs: list[Path], output: Path, *, overwrite: bool) -> dict:
    missing = [str(path) for path in inputs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Input file not found: {', '.join(missing)}")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}. Use --overwrite to replace it.")

    with tempfile.TemporaryDirectory(prefix="anata_manual_raw_merge_") as temp_dir:
        temp_root = Path(temp_dir)
        extracted_roots: list[Path] = []
        for index, input_zip in enumerate(inputs, start=1):
            extract_root = temp_root / f"input_{index}"
            extract_root.mkdir(parents=True)
            with zipfile.ZipFile(input_zip) as archive:
                archive.extractall(extract_root)
            extracted_roots.append(extract_root)

        merged_root = temp_root / "merged"
        merged_root.mkdir(parents=True)

        relative_paths: set[Path] = set()
        for root in extracted_roots:
            relative_paths.update(path.relative_to(root) for path in root.rglob("*") if path.is_file())

        for relative_path in sorted(relative_paths):
            source_files = [root / relative_path for root in extracted_roots if (root / relative_path).exists()]
            output_file = merged_root / relative_path
            if relative_path.name == "manifest.json":
                _merge_manifest_files(source_files, output_file, inputs)
            elif relative_path.name.endswith(".jsonl.gz"):
                _merge_gzip_text_files(source_files, output_file)
            elif relative_path.name.endswith(".csv.gz"):
                _merge_gzip_text_files(source_files, output_file, csv_mode=True)
            else:
                source = max(source_files, key=lambda path: path.stat().st_size)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, output_file)

        temp_output = temp_root / output.name
        _zip_folder(merged_root, temp_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(temp_output), output)

    return {
        "status": "ok",
        "output": str(output),
        "output_size_bytes": output.stat().st_size,
        "input_files": [str(path) for path in inputs],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge two or more raw_YYYY-MM-DD ZIP files without leaving extra files.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Input daily raw ZIP files, for example part1 and part2.")
    parser.add_argument("--output", type=Path, required=True, help="Merged output ZIP path.")
    parser.add_argument("--overwrite", action="store_true", help="Replace --output if it already exists.")
    args = parser.parse_args()

    result = merge_raw_daily_zips(args.inputs, args.output, overwrite=args.overwrite)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
