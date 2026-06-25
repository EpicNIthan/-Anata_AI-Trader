from __future__ import annotations

import argparse
import csv
import gzip
import json
import socket
from datetime import datetime, timezone
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
        headers={"x-admin-token": token, "Content-Type": "application/json"},
    )
    with request.urlopen(req, timeout=timeout) as response:
        return response.read()


def _json_api(url: str, token: str, path: str, *, method: str = "GET", body: dict | None = None, timeout: int = 600) -> dict:
    return json.loads(_api(url, token, path, method=method, body=body, timeout=timeout).decode("utf-8"))


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


def _download(url: str, token: str, api_path: str, output: Path, *, timeout: int) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_api(url, token, api_path, timeout=timeout))
    return {
        "path": str(output),
        "file_size_bytes": output.stat().st_size,
        "row_count": _row_count(output),
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Railway data into a dated laptop folder, then optionally compact Railway after a verified download."
    )
    parser.add_argument("--url", required=True, help="Railway app URL, for example https://anataai-trader-production.up.railway.app")
    parser.add_argument("--token", required=True, help="ADMIN_TOKEN")
    parser.add_argument("--output-root", type=Path, default=Path("local_data"))
    parser.add_argument("--since-date", default=None, help="Only export rows from this date, for example 2026-06-24.")
    parser.add_argument("--use-all-data", action="store_true")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--no-auto-build-labels", action="store_true")
    parser.add_argument("--skip-raw-news", action="store_true")
    parser.add_argument("--clear-railway-after-download", action="store_true")
    parser.add_argument(
        "--railway-keep-days",
        type=int,
        default=7,
        help="When clearing Railway, keep only this many recent days of candles/features/experiences/news.",
    )
    args = parser.parse_args()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "url": args.url,
        "since_date": args.since_date,
        "use_all_data": args.use_all_data,
        "files": {},
        "railway_cleanup": None,
    }

    try:
        report_before = _json_api(args.url, args.token, "/api/data/collection-report?include_storage=true", timeout=args.timeout)
        _write_json(run_dir / "collection_report_before.json", report_before)
        manifest["collection_report_before"] = str(run_dir / "collection_report_before.json")

        export_body = {
            "use_all_data": args.use_all_data,
            "since_date": args.since_date,
            "auto_build_labels": False if args.no_auto_build_labels else None,
        }
        dataset_export = _json_api(args.url, args.token, "/api/training/export", method="POST", body=export_body, timeout=args.timeout)
        dataset_id = Path(dataset_export["dataset_id"]).name
        dataset_path = run_dir / "training_dataset.csv.gz"
        manifest["training_export"] = dataset_export
        manifest["files"]["training_dataset"] = _download(
            args.url,
            args.token,
            f"/api/training/download/{dataset_id}",
            dataset_path,
            timeout=args.timeout,
        )

        if not args.skip_raw_news:
            raw_news_export = _json_api(
                args.url,
                args.token,
                "/api/news/export-raw",
                method="POST",
                body={"use_all_data": args.use_all_data, "since_date": args.since_date},
                timeout=args.timeout,
            )
            raw_news_id = Path(raw_news_export["dataset_id"]).name
            raw_news_path = run_dir / "raw_news.csv.gz"
            manifest["raw_news_export"] = raw_news_export
            manifest["files"]["raw_news"] = _download(
                args.url,
                args.token,
                f"/api/news/download/{raw_news_id}",
                raw_news_path,
                timeout=args.timeout,
            )

        downloaded_files = [item for item in manifest["files"].values() if item.get("file_size_bytes", 0) > 0]
        if args.clear_railway_after_download and downloaded_files:
            cleanup_body = {"factory_mode": True, "keep_recent_days": max(args.railway_keep_days, 1)}
            manifest["railway_cleanup"] = _json_api(
                args.url,
                args.token,
                "/api/db/compact",
                method="POST",
                body=cleanup_body,
                timeout=args.timeout,
            )
            report_after = _json_api(args.url, args.token, "/api/data/collection-report?include_storage=true", timeout=args.timeout)
            _write_json(run_dir / "collection_report_after_cleanup.json", report_after)
            manifest["collection_report_after_cleanup"] = str(run_dir / "collection_report_after_cleanup.json")

        _write_json(run_dir / "manifest.json", manifest)
        _write_json(args.output_root / "latest_manifest.json", manifest)
        print(json.dumps({"status": "ok", "run_dir": str(run_dir), "manifest": str(run_dir / "manifest.json"), **manifest["files"]}, indent=2))
        if args.clear_railway_after_download:
            print("Railway cleanup ran only after local files were written.")
    except Exception as exc:
        _write_json(run_dir / "manifest_failed.json", manifest | {"error": str(exc), "error_type": type(exc).__name__})
        if _is_timeout(exc):
            print("Sync timed out. Retry with a smaller --since-date range or higher --timeout.")
        raise


if __name__ == "__main__":
    main()
