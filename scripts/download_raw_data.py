from __future__ import annotations

import argparse
import gzip
import json
import shutil
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


def _finished_days_from_manifest(manifest: dict) -> list[str]:
    days = manifest.get("finished_days")
    if isinstance(days, list):
        return sorted({str(day)[:10] for day in days if str(day).strip()})

    day_manifests = manifest.get("day_manifests")
    if isinstance(day_manifests, dict):
        finished: list[str] = []
        for day, day_manifest in day_manifests.items():
            if not isinstance(day_manifest, dict):
                continue
            if day_manifest.get("is_finished") is True or day_manifest.get("status") == "finished":
                finished.append(str(day)[:10])
        return sorted({day for day in finished if day})

    statuses = manifest.get("day_statuses")
    if isinstance(statuses, dict):
        return sorted({str(day)[:10] for day, status in statuses.items() if status == "finished"})
    return []


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _unique_sibling(path: Path, label: str) -> Path:
    stamp = _timestamp()
    candidate = path.with_name(f"{path.stem}_{label}_{stamp}{path.suffix}")
    counter = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{label}_{stamp}_{counter}{path.suffix}")
        counter += 1
    return candidate


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


def _merge_gzip_text_files(existing: Path, incoming: Path, output: Path, *, csv_mode: bool = False) -> None:
    seen: set[str] = set()
    wrote_header = False
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wt", encoding="utf-8", newline="") as writer:
        for source in (existing, incoming):
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


def _merge_daily_manifest(existing_manifest: Path, incoming_manifest: Path, output: Path, existing_zip: Path) -> None:
    existing = _read_json_file(existing_manifest) if existing_manifest.exists() else {}
    incoming = _read_json_file(incoming_manifest) if incoming_manifest.exists() else {}
    merged = dict(incoming or existing)
    merged["local_daily_file_merge"] = {
        "merged_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "existing_backup_file": str(existing_zip),
        "reason": "Existing local raw day ZIP was merged with a new download to avoid overwriting partial local data.",
        "existing_manifest": existing,
        "incoming_manifest": incoming,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(merged, indent=2, sort_keys=True), encoding="utf-8")


def _merge_daily_zip(existing_zip: Path, incoming_day_folder: Path, output_path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="anata_merge_raw_day_") as temp_dir:
        temp_root = Path(temp_dir)
        existing_root = temp_root / "existing"
        merged_root = temp_root / "merged"
        existing_root.mkdir(parents=True)
        merged_root.mkdir(parents=True)
        with zipfile.ZipFile(existing_zip) as archive:
            archive.extractall(existing_root)

        relative_paths = {
            path.relative_to(existing_root)
            for path in existing_root.rglob("*")
            if path.is_file()
        }
        relative_paths.update(
            path.relative_to(incoming_day_folder)
            for path in incoming_day_folder.rglob("*")
            if path.is_file()
        )

        for relative_path in sorted(relative_paths):
            existing_file = existing_root / relative_path
            incoming_file = incoming_day_folder / relative_path
            output_file = merged_root / relative_path
            output_file.parent.mkdir(parents=True, exist_ok=True)

            if relative_path.name == "manifest.json":
                _merge_daily_manifest(existing_file, incoming_file, output_file, existing_zip)
            elif existing_file.exists() and incoming_file.exists() and relative_path.name.endswith(".jsonl.gz"):
                _merge_gzip_text_files(existing_file, incoming_file, output_file)
            elif existing_file.exists() and incoming_file.exists() and relative_path.name.endswith(".csv.gz"):
                _merge_gzip_text_files(existing_file, incoming_file, output_file, csv_mode=True)
            else:
                if existing_file.exists() and incoming_file.exists():
                    source = incoming_file if incoming_file.stat().st_size >= existing_file.stat().st_size else existing_file
                else:
                    source = incoming_file if incoming_file.exists() else existing_file
                shutil.copy2(source, output_file)

        _zip_folder(merged_root, output_path)


def _write_daily_zip(
    day_folder: Path,
    output_path: Path,
    *,
    merge_existing: bool,
    overwrite: bool,
) -> dict[str, object]:
    if not output_path.exists():
        _zip_folder(day_folder, output_path)
        return {"path": str(output_path), "action": "written", "size_bytes": output_path.stat().st_size}

    if overwrite:
        previous_size = output_path.stat().st_size
        _zip_folder(day_folder, output_path)
        return {
            "path": str(output_path),
            "action": "overwritten",
            "previous_size_bytes": previous_size,
            "size_bytes": output_path.stat().st_size,
        }

    if merge_existing:
        backup_path = _unique_sibling(output_path, "before_merge")
        previous_size = output_path.stat().st_size
        output_path.replace(backup_path)
        try:
            _merge_daily_zip(backup_path, day_folder, output_path)
        except Exception:
            if output_path.exists():
                output_path.unlink()
            backup_path.replace(output_path)
            raise
        return {
            "path": str(output_path),
            "action": "merged_existing",
            "backup_path": str(backup_path),
            "previous_size_bytes": previous_size,
            "size_bytes": output_path.stat().st_size,
        }

    duplicate_path = _unique_sibling(output_path, "redownload")
    _zip_folder(day_folder, duplicate_path)
    return {
        "path": str(output_path),
        "action": "kept_existing_saved_new_as_duplicate",
        "duplicate_path": str(duplicate_path),
        "size_bytes": output_path.stat().st_size,
        "duplicate_size_bytes": duplicate_path.stat().st_size,
    }


def _split_daily_archive(
    archive_path: Path,
    output_dir: Path,
    *,
    merge_existing: bool = True,
    overwrite: bool = False,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="anata_daily_raw_") as temp_dir:
        temp_root = Path(temp_dir)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(temp_root)
        root = temp_root / "daily_raw_data"
        if not root.exists():
            raise RuntimeError("Daily archive did not contain daily_raw_data folder.")
        for day_folder in sorted(path for path in root.iterdir() if path.is_dir()):
            output_path = output_dir / f"raw_{day_folder.name}.zip"
            entry = _write_daily_zip(
                day_folder,
                output_path,
                merge_existing=merge_existing,
                overwrite=overwrite,
            )
            entry["day"] = day_folder.name
            entries.append(entry)
    return {
        "daily_file_count": len(entries),
        "daily_files": [str(entry["path"]) for entry in entries],
        "daily_file_entries": entries,
    }


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
        "--merge-existing-daily-files",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When a local raw_YYYY-MM-DD.zip already exists, merge it with the new download instead of overwriting it. Default: enabled.",
    )
    parser.add_argument(
        "--overwrite-daily-files",
        action="store_true",
        help="Replace existing local raw_YYYY-MM-DD.zip files. Not recommended unless you intentionally want a fresh server snapshot.",
    )
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
        split_result = (
            _split_daily_archive(
                output,
                args.output_dir,
                merge_existing=args.merge_existing_daily_files,
                overwrite=args.overwrite_daily_files,
            )
            if args.daily_files
            else None
        )
        finished_days = _finished_days_from_manifest(manifest) if args.daily_files else []
        unfinished_days = manifest.get("unfinished_days", []) if isinstance(manifest.get("unfinished_days"), list) else []
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
            "finished_days": finished_days,
            "unfinished_days": unfinished_days,
            "daily_split": split_result,
            "cleanup": None,
        }
        if args.cleanup_after_download:
            cleanup_options = dict(export_options)
            if args.daily_files and args.delete_railway_db_rows:
                for key in ("daily_files", "date", "since_date", "until_date", "use_all_data"):
                    cleanup_options.pop(key, None)
                cleanup_options["delete_finished_days"] = finished_days
                cleanup_options["protect_unfinished_days"] = True
                if not finished_days:
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
