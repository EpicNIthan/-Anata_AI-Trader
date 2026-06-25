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
    with request.urlopen(req, timeout=timeout) as response:
        return response.read()


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
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--no-unfinished", action="store_true", help="Do not include today's unfinished bundle.")
    parser.add_argument("--no-extract", action="store_true", help="Keep zip files only.")
    parser.add_argument("--delete-finished-from-railway", action="store_true")
    args = parser.parse_args()

    manifest: dict = {"files": [], "cleanup": None}
    try:
        build = _json_api(
            args.url,
            args.token,
            "/api/data/bundles/build",
            method="POST",
            body={"since_date": args.since_date, "days": args.days, "include_unfinished": not args.no_unfinished},
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
        raise


if __name__ == "__main__":
    main()
