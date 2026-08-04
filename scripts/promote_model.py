"""Explicitly promote one already-uploaded registry candidate through the V2 API."""

from __future__ import annotations

import argparse
import json
from urllib import error, request


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manually promote one uploaded model; this never uploads or auto-selects a candidate."
    )
    parser.add_argument("--url", required=True, help="Application base URL.")
    parser.add_argument("--token", required=True, help="Administrative API token.")
    parser.add_argument("--model-id", required=True, type=int, help="Exact model_versions integer ID.")
    parser.add_argument("--family", required=True, help="Exact registered model family.")
    parser.add_argument(
        "--symbol-scope",
        default="*",
        help="Explicit symbol or '*' (the intelligence news-student family requires '*').",
    )
    parser.add_argument("--reason", required=True, help="Operator audit reason.")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Required acknowledgement that this writes a manual champion assignment.",
    )
    args = parser.parse_args()
    if not args.confirm:
        raise SystemExit("Refusing promotion without --confirm")
    payload = json.dumps(
        {
            "model_family": args.family,
            "symbol_scope": args.symbol_scope,
            "reason": args.reason,
            "confirm": True,
        }
    ).encode("utf-8")
    endpoint = args.url.rstrip("/") + f"/api/v2/models/{args.model_id}/promote"
    api_request = request.Request(
        endpoint,
        data=payload,
        method="POST",
        headers={
            "x-admin-token": args.token,
            "Content-Type": "application/json",
        },
    )
    try:
        with request.urlopen(api_request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Promotion failed with HTTP {exc.code}: {detail}") from exc
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
