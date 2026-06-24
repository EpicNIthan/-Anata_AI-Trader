from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Package a trained local model for Railway upload.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    metadata_path = args.metadata or args.model.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["model_file"] = args.model.name
    output = args.output or Path(f"model_package_{metadata.get('version') or args.model.stem}.zip")
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(args.model, args.model.name)
        archive.writestr("metadata.json", json.dumps(metadata, indent=2))
    print(json.dumps({"package": str(output), "model": args.model.name, "metadata": "metadata.json"}, indent=2))


if __name__ == "__main__":
    main()
