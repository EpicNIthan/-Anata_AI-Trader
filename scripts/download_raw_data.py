from __future__ import annotations

import argparse
import json
import socket
import tempfile
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib import error, request


def _api(
    url: str,
    token: str,
    path: str,
    *,
    method: str = "GET",
    body: dict | None = None,
    timeout: int = 900,
) -> bytes:
    data = json.dumps(body or {}).encode("utf-8") if body is not None else None
    req = request.Request(
        url.rstrip("/") + path,
        data=data,
        method=method,
        headers={"x-admin-token": token, "Content-Type": "application/json"},
    )
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return response.read()
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1200].strip()
        message = f"Railway returned HTTP {exc.code} {exc.reason} for {path}"
        if detail:
            message = f"{message}: {detail}"
        raise RuntimeError(message) from exc


def _is_timeout(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return True
    if isinstance(exc, error.URLError):
        reason = getattr(exc, "reason", None)
        return isinstance(reason, (TimeoutError, socket.timeout)) or "timed out" in str(reason).lower()
    return "timed out" in str(exc).lower()


def _json_api(url: str, token: str, path: str, *, method: str = "GET", body: dict | None = None, timeout: int = 900) -> dict:
    return json.loads(_api(url, token, path, method=method, body=body, timeout=timeout).decode("utf-8"))


def _finished_cutoff_iso(hours: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).replace(microsecond=0).isoformat()


def _compact_options(args: argparse.Namespace) -> dict:
    if args.finished_only and args.use_all_data:
        raise SystemExit("Do not combine --finished-only with --use-all-data. finished-only protects the under-24h live data.")
    until_date = args.until_date
    use_all_data = args.use_all_data
    if args.finished_only:
        until_date = until_date or _finished_cutoff_iso(args.finished_older_than_hours)
        use_all_data = False
    if args.daily_files:
        use_all_data = True if not args.date and not args.since_date and not args.until_date else use_all_data
    payload = {
        "date": args.date,
        "since_date": args.since_date,
        "until_date": until_date,
        "use_all_data": use_all_data,
        "news_only": args.news_only,
        "include_market": args.include_market,
        "include_news": args.include_news,
        "include_external": args.include_external,
        "include_experience": args.include_experience,
        "include_models": args.include_models,
        "finished_only": args.finished_only,
        "finished_older_than_hours": args.finished_older_than_hours if args.finished_only else None,
        "daily_files": args.daily_files,
    }
    if args.symbols:
        payload["symbols"] = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
    return {key: value for key, value in payload.items() if value is not None}


def _zip_manifest(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        manifest_names = [name for name in archive.namelist() if name.endswith("manifest.json") or name.endswith("daily_manifest.json")]
        if not manifest_names:
            raise RuntimeError("Downloaded ZIP does not contain manifest.json; cleanup was not attempted.")
        manifest_name = sorted(manifest_names, key=lambda name: (not name.endswith("daily_manifest.json"), name))[0]
        with archive.open(manifest_name) as handle:
            return json.loads(handle.read().decode("utf-8"))


def _split_daily_archive(archive_path: Path, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    with tempfile.TemporaryDirectory(prefix="anata_daily_raw_") as temp_dir:
        temp_root = Path(temp_dir)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(temp_root)
        root = temp_root / "daily_raw_data"
        if not root.exists():
            raise RuntimeError("Daily archive did not contain daily_raw_data folder.")
        for day_folder in sorted(path for path in root.iterdir() if path.is_dir()):
            output_path = output_dir / f"raw_{day_folder.name}.zip"
            with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
                for path in sorted(day_folder.rglob("*")):
                    if path.is_file():
                        archive.write(path, path.relative_to(day_folder))
            written.append(str(output_path))
    return {"daily_file_count": len(written), "daily_files": written}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download raw Anata trading data from Railway.")
    parser.add_argument("--url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/raw_days"))
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--date", default=None)
    parser.add_argument("--since-date", default=None)
    parser.add_argument("--until-date", default=None)
    parser.add_argument("--use-all-data", action="store_true")
    parser.add_argument("--daily-files", action="store_true", help="Download all selected data split into one local raw_YYYY-MM-DD.zip per day, including current under-24h day.")
    parser.add_argument("--keep-daily-bundle", action="store_true", help="Keep the temporary local raw_daily_bundle_*.zip after daily files are written.")
    parser.add_argument(
        "--finished-only",
        action="store_true",
        help="Export only data older than --finished-older-than-hours, so current under-24h DB rows stay untouched.",
    )
    parser.add_argument(
        "--finished-older-than-hours",
        type=float,
        default=24.0,
        help="Cutoff for --finished-only. Default: older than 24 hours.",
    )
    parser.add_argument("--news-only", action="store_true")
    parser.add_argument("--include-market", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-news", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-external", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-experience", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-models", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--symbols", default=None, help="Comma-separated symbols, for example BTCUSDT,ETHUSDT.")
    parser.add_argument(
        "--cleanup-after-download",
        action="store_true",
        help="After the ZIP is written and manifest is verified, delete Railway export files/finished_data for this range.",
    )
    parser.add_argument(
        "--delete-railway-db-rows",
        action="store_true",
        help="With --cleanup-after-download, also delete matching raw DB rows from Railway after successful PC download.",
    )
    parser.add_argument(
        "--delete-all-finished-data",
        action="store_true",
        help="With --cleanup-after-download, delete all finished_data folders on Railway. Does not change the protected finished-only DB cutoff.",
    )
    parser.add_argument("--keep-railway-archive", action="store_true", help="Do not delete the temporary raw export ZIP from Railway.")
    parser.add_argument("--keep-railway-finished-data", action="store_true", help="Do not delete finished_data folders from Railway.")
    args = parser.parse_args()

    try:
        export_options = _compact_options(args)
        export_path = "/api/raw-data/export-daily" if args.daily_files else "/api/raw-data/export"
        export = _json_api(
            args.url,
            args.token,
            export_path,
            method="POST",
            body=export_options,
            timeout=args.timeout,
        )
        archive_id = export["archive_id"]
        if args.output is None and args.finished_only:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            output = Path("datasets") / "raw_finished" / f"raw_finished_until_{stamp}.zip"
        elif args.output is None and args.daily_files:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            output = args.output_dir / f"raw_daily_bundle_{stamp}.zip"
        else:
            output = args.output or Path("datasets") / archive_id
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(_api(args.url, args.token, f"/api/raw-data/download/{archive_id}", timeout=args.timeout))
        local_manifest = _zip_manifest(output)
        manifest = export.get("manifest") or local_manifest
        split_result = _split_daily_archive(output, args.output_dir) if args.daily_files else None
        bundle_removed = False
        if args.daily_files and not args.keep_daily_bundle and split_result and output.exists():
            output.unlink()
            bundle_removed = True
        result = {
            "status": "ok",
            "output": None if bundle_removed else str(output),
            "bundle_removed_after_split": bundle_removed,
            "file_size_bytes": 0 if bundle_removed else output.stat().st_size,
            "archive_id": archive_id,
            "row_counts": manifest.get("row_counts", manifest.get("total_row_counts", {})),
            "file_sizes": manifest.get("file_sizes", {}),
            "time_range": manifest.get("time_range", {}),
            "symbols": manifest.get("symbols", []),
            "warnings": manifest.get("warnings", []),
            "daily_split": split_result,
            "cleanup": None,
        }
        if args.cleanup_after_download:
            cleanup_options = dict(export_options)
            if args.daily_files and args.delete_railway_db_rows:
                cleanup_options["until_date"] = _finished_cutoff_iso(args.finished_older_than_hours)
                cleanup_options["use_all_data"] = False
            cleanup_body = {
                **cleanup_options,
                "archive_id": archive_id,
                "delete_archive": not args.keep_railway_archive,
                "delete_finished_data": not args.keep_railway_finished_data,
                "delete_all_finished_data": args.delete_all_finished_data,
                "delete_db_rows": args.delete_railway_db_rows,
                "local_manifest_verified": True,
                "local_file_size_bytes": sum(Path(path).stat().st_size for path in (split_result or {}).get("daily_files", [])) if split_result else output.stat().st_size,
            }
            result["cleanup"] = _json_api(
                args.url,
                args.token,
                "/api/raw-data/cleanup-downloaded",
                method="POST",
                body=cleanup_body,
                timeout=args.timeout,
            )
        print(json.dumps(result, indent=2))
    except Exception as exc:
        if _is_timeout(exc):
            print("Raw data export timed out. Retry with --date YYYY-MM-DD, --news-only, --finished-only, --daily-files, or a smaller --since-date range.")
        elif "HTTP 502" in str(exc) or "Bad Gateway" in str(exc):
            print("Railway returned 502 while exporting raw data. Wait for redeploy/restart, then retry a smaller range.")
        else:
            print(f"Raw data download failed: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
