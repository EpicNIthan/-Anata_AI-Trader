from __future__ import annotations

import argparse
import json
import socket
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


def _compact_options(args: argparse.Namespace) -> dict:
    payload = {
        "date": args.date,
        "since_date": args.since_date,
        "until_date": args.until_date,
        "use_all_data": args.use_all_data,
        "news_only": args.news_only,
        "include_market": args.include_market,
        "include_news": args.include_news,
        "include_external": args.include_external,
        "include_experience": args.include_experience,
        "include_models": args.include_models,
    }
    if args.symbols:
        payload["symbols"] = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
    return {key: value for key, value in payload.items() if value is not None}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download raw Anata trading data from Railway.")
    parser.add_argument("--url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--date", default=None)
    parser.add_argument("--since-date", default=None)
    parser.add_argument("--until-date", default=None)
    parser.add_argument("--use-all-data", action="store_true")
    parser.add_argument("--news-only", action="store_true")
    parser.add_argument("--include-market", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-news", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-external", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-experience", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-models", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--symbols", default=None, help="Comma-separated symbols, for example BTCUSDT,ETHUSDT.")
    args = parser.parse_args()

    try:
        export = _json_api(
            args.url,
            args.token,
            "/api/raw-data/export",
            method="POST",
            body=_compact_options(args),
            timeout=args.timeout,
        )
        archive_id = export["archive_id"]
        output = args.output or Path("datasets") / archive_id
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(_api(args.url, args.token, f"/api/raw-data/download/{archive_id}", timeout=args.timeout))
        manifest = export.get("manifest") or {}
        result = {
            "status": "ok",
            "output": str(output),
            "file_size_bytes": output.stat().st_size,
            "archive_id": archive_id,
            "row_counts": manifest.get("row_counts", {}),
            "file_sizes": manifest.get("file_sizes", {}),
            "time_range": manifest.get("time_range", {}),
            "symbols": manifest.get("symbols", []),
            "warnings": manifest.get("warnings", []),
        }
        print(json.dumps(result, indent=2))
    except Exception as exc:
        if _is_timeout(exc):
            print("Raw data export timed out. Retry with --date YYYY-MM-DD, --news-only, or a smaller --since-date range.")
        elif "HTTP 502" in str(exc) or "Bad Gateway" in str(exc):
            print("Railway returned 502 while exporting raw data. Wait for redeploy/restart, then retry a smaller range.")
        else:
            print(f"Raw data download failed: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

