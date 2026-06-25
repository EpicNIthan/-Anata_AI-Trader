from __future__ import annotations

import argparse
import csv
import gzip
import json
import socket
import sys
from pathlib import Path
from urllib import error, request


def _api(
    url: str,
    token: str,
    path: str,
    *,
    method: str = "GET",
    body: dict | None = None,
    timeout: int = 600,
) -> bytes:
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
    with request.urlopen(req, timeout=timeout) as response:
        return response.read()


def _is_timeout(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return True
    if isinstance(exc, error.URLError):
        reason = getattr(exc, "reason", None)
        return isinstance(reason, (TimeoutError, socket.timeout)) or "timed out" in str(reason).lower()
    return "timed out" in str(exc).lower()


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
    parser.add_argument("--timeout", type=int, default=600, help="HTTP timeout in seconds for export and download.")
    parser.add_argument("--since-date", default=None, help="Only export rows at or after this date, for example 2026-06-24.")
    parser.add_argument("--use-all-data", action="store_true", help="Export all available data. If set, since-date is ignored by the server.")
    parser.add_argument("--no-auto-build-labels", action="store_true", help="Disable label building during export.")
    parser.add_argument("--export-only", action="store_true", help="Create or reuse a dataset id and print it without downloading.")
    args = parser.parse_args()

    dataset_id = args.dataset_id
    if not dataset_id:
        export_body = {
            "use_all_data": args.use_all_data,
            "since_date": args.since_date,
            "auto_build_labels": False if args.no_auto_build_labels else None,
        }
        try:
            payload = json.loads(
                _api(
                    args.url,
                    args.token,
                    "/api/training/export",
                    method="POST",
                    body=export_body,
                    timeout=args.timeout,
                )
            )
        except Exception as exc:
            if _is_timeout(exc):
                print("Export timed out. Build labels first, compact DB, or export a smaller since_date range.", file=sys.stderr)
                raise SystemExit(2) from exc
            raise
        dataset_id = payload["dataset_id"]

    if args.export_only:
        print(Path(dataset_id).name)
        return

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = _api(args.url, args.token, f"/api/training/download/{Path(dataset_id).name}", timeout=args.timeout)
    except Exception as exc:
        if _is_timeout(exc):
            print("Download timed out. Retry with --dataset-id and a higher --timeout.", file=sys.stderr)
            raise SystemExit(2) from exc
        raise
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
