from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from urllib import request


def _api(url: str, token: str, path: str, *, method: str = "GET", body: dict | None = None, timeout: int = 600) -> bytes:
    data = json.dumps(body or {}).encode("utf-8") if body is not None else None
    req = request.Request(
        url.rstrip("/") + path,
        data=data,
        method=method,
        headers={"x-admin-token": token, "Content-Type": "application/json"},
    )
    with request.urlopen(req, timeout=timeout) as response:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Download raw news from Railway for local heavy sentiment processing.")
    parser.add_argument("--url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--output", type=Path, default=Path("datasets/latest_raw_news.csv.gz"))
    parser.add_argument("--since-date", default=None)
    parser.add_argument("--use-all-data", action="store_true")
    parser.add_argument("--provider", default=None, help="Optional rss, gdelt, or newsapi filter.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()

    dataset_id = args.dataset_id
    if not dataset_id:
        payload = json.loads(
            _api(
                args.url,
                args.token,
                "/api/news/export-raw",
                method="POST",
                body={
                    "since_date": args.since_date,
                    "use_all_data": args.use_all_data,
                    "provider": args.provider,
                    "limit": args.limit,
                },
                timeout=args.timeout,
            )
        )
        dataset_id = payload["dataset_id"]
    if args.export_only:
        print(Path(dataset_id).name)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = _api(args.url, args.token, f"/api/news/download/{Path(dataset_id).name}", timeout=args.timeout)
    args.output.write_bytes(data)
    print(
        json.dumps(
            {
                "downloaded_path": str(args.output),
                "file_size_bytes": args.output.stat().st_size,
                "row_count": _row_count(args.output),
                "source_dataset_id": Path(dataset_id).name,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
