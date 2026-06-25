from __future__ import annotations

import argparse
import json
import socket
import zipfile
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
        try:
            body_text = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body_text = ""
        detail = body_text[:1200].strip()
        message = f"Railway returned HTTP {exc.code} {exc.reason} for {path}"
        if detail:
            message = f"{message}: {detail}"
        raise RuntimeError(message) from exc


def _json_api(url: str, token: str, path: str, *, method: str = "GET", body: dict | None = None, timeout: int = 900) -> dict:
    return json.loads(_api(url, token, path, method=method, body=body, timeout=timeout).decode("utf-8"))


def _is_timeout(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return True
    if isinstance(exc, error.URLError):
        reason = getattr(exc, "reason", None)
        return isinstance(reason, (TimeoutError, socket.timeout)) or "timed out" in str(reason).lower()
    return "timed out" in str(exc).lower()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _download_bundle(url: str, token: str, bundle: dict, output_root: Path, *, timeout: int, extract: bool) -> dict:
    bundle_id = Path(bundle["bundle_id"]).name
    day_dir = output_root / str(bundle.get("day") or bundle_id.replace(".zip", ""))
    day_dir.mkdir(parents=True, exist_ok=True)
    zip_path = day_dir / bundle_id
    zip_path.write_bytes(_api(url, token, f"/api/data/bundles/download/{bundle_id}", timeout=timeout))
    result = {
        "bundle_id": bundle_id,
        "status": bundle.get("status"),
        "day": bundle.get("day"),
        "zip_path": str(zip_path),
        "size_bytes": zip_path.stat().st_size,
        "extracted_dir": None,
    }
    if extract:
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(day_dir)
        result["extracted_dir"] = str(day_dir)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Railway daily data bundles and optionally delete finished Railway data.")
    parser.add_argument("--url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("local_data/daily_bundles"))
    parser.add_argument("--since-date", default=None)
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--all-data", action="store_true", help="Download every available day. This is also the default when --days is omitted.")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--no-unfinished", action="store_true", help="Do not include today's unfinished bundle.")
    parser.add_argument("--no-extract", action="store_true", help="Keep zip files only.")
    parser.add_argument("--delete-finished-from-railway", action="store_true")
    parser.add_argument(
        "--preset",
        choices=["training", "all"],
        default="training",
        help="training = only useful model-training tables. all = every supported DB table snapshot.",
    )
    parser.add_argument(
        "--tables",
        default=None,
        help="Optional comma-separated tables. Overrides --preset.",
    )
    parser.add_argument("--compact-first", action="store_true", help="Run Railway compact cleanup before building bundles.")
    args = parser.parse_args()

    manifest: dict = {"files": [], "cleanup": None}
    try:
        if args.compact_first:
            manifest["compact_first"] = _json_api(args.url, args.token, "/api/db/compact", method="POST", body={}, timeout=args.timeout)
        selected_tables = args.tables or args.preset
        build = _json_api(
            args.url,
            args.token,
            "/api/data/bundles/build",
            method="POST",
            body={
                "since_date": args.since_date,
                "days": None if args.all_data else args.days,
                "include_unfinished": not args.no_unfinished,
                "tables": selected_tables,
            },
            timeout=args.timeout,
        )
        manifest["build"] = build
        bundles = build.get("bundles") or []
        args.output_root.mkdir(parents=True, exist_ok=True)
        for bundle in bundles:
            manifest["files"].append(
                _download_bundle(args.url, args.token, bundle, args.output_root, timeout=args.timeout, extract=not args.no_extract)
            )
        if args.delete_finished_from_railway:
            finished_downloaded = [item for item in manifest["files"] if item.get("status") == "finished" and item.get("size_bytes", 0) > 0]
            if finished_downloaded:
                manifest["cleanup"] = _json_api(
                    args.url,
                    args.token,
                    "/api/data/bundles/cleanup-finished",
                    method="POST",
                    timeout=args.timeout,
                )
            else:
                manifest["cleanup"] = {"status": "skipped", "message": "No finished bundles were downloaded."}
        _write_json(args.output_root / "latest_manifest.json", manifest)
        print(json.dumps({"status": "ok", "output_root": str(args.output_root), "bundles_downloaded": len(manifest["files"]), "cleanup": manifest["cleanup"]}, indent=2))
    except Exception as exc:
        _write_json(args.output_root / "latest_manifest_failed.json", manifest | {"error": str(exc), "error_type": type(exc).__name__})
        if _is_timeout(exc):
            print("Bundle sync timed out. Retry with --days 2 or a higher --timeout.")
        elif "HTTP 502" in str(exc) or "Bad Gateway" in str(exc):
            print("Railway returned 502 while building bundles.")
            print("Most likely the old deployment ran out of memory or Railway is still redeploying.")
            print("Wait for the latest deployment, then retry. If it still fails, use --days 1 --compact-first.")
        else:
            print(f"Bundle sync failed: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
