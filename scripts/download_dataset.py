from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from urllib import request


def _api(url: str, token: str, path: str, *, method: str = "GET", body: dict | None = None) -> bytes:
    data = json.dumps(body or {}).encode("utf-8") if body is not None else None
    req = request.Request(
        url.rstrip("/") + path,
        data=data,
        method=method,
        headers={
            "x-admin-token": token,
            "Content-Type": "application/json",
        },
    )
    with request.urlopen(req, timeout=120) as response:
        return response.read()


def _row_count(path: Path) -> int | None:
    try:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            return sum(1 for _ in reader)
    except Exception:
        return None


def _label_count(path: Path) -> int | None:
    try:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return sum(1 for row in reader if row.get("target_trade_quality_score") not in (None, ""))
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Download an exported dataset from Railway.")
    parser.add_argument("--url", required=True, help="Railway app URL, for example https://your-app.up.railway.app")
    parser.add_argument("--token", required=True, help="ADMIN_TOKEN")
    parser.add_argument("--dataset-id", default=None, help="Existing dataset id/filename. If omitted, a new export is created.")
    parser.add_argument("--output", type=Path, default=Path("datasets/latest.csv.gz"))
    args = parser.parse_args()

    dataset_id = args.dataset_id
    if not dataset_id:
        payload = json.loads(_api(args.url, args.token, "/api/training/export", method="POST", body={"use_all_data": True}))
        dataset_id = payload["dataset_id"]

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    data = _api(args.url, args.token, f"/api/training/download/{Path(dataset_id).name}")
    output.write_bytes(data)
    rows = _row_count(output)
    labeled_rows = _label_count(output)
    result = {
        "downloaded_path": str(output),
        "file_size_bytes": output.stat().st_size,
        "row_count": rows,
        "labeled_target_trade_quality_score_rows": labeled_rows,
        "source_dataset_id": Path(dataset_id).name,
    }
    print(json.dumps(result, indent=2))
    if rows and labeled_rows == 0:
        print("WARNING: Dataset has features but no labels. Run /api/training/build-labels or wait for future candles.")


if __name__ == "__main__":
    main()
