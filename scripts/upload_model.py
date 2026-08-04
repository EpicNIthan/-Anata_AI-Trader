from __future__ import annotations

import argparse
import json
import mimetypes
import uuid
from pathlib import Path
from typing import Any, Callable
from urllib import request
from urllib.parse import urlsplit


def _multipart_body(field_name: str, path: Path, boundary: str) -> bytes:
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field_name}"; filename="{path.name}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("utf-8")
    footer = f"\r\n--{boundary}--\r\n".encode("utf-8")
    return header + path.read_bytes() + footer


def upload_package(
    *,
    url: str,
    token: str,
    package: Path,
    timeout_seconds: float = 120.0,
    opener: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Upload one checksummed package as a candidate; never activate it."""

    if not token:
        raise ValueError("admin token is required")
    if not package.is_file() or package.suffix.lower() != ".zip":
        raise ValueError("package must be an existing .zip file")
    parsed_url = urlsplit(url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("url must be an absolute HTTP(S) base URL")
    if parsed_url.username or parsed_url.password or parsed_url.query or parsed_url.fragment:
        raise ValueError("url must not contain credentials, a query, or a fragment")
    boundary = f"----anata-{uuid.uuid4().hex}"
    data = _multipart_body("file", package, boundary)
    req = request.Request(
        url.rstrip("/") + "/api/models/upload",
        data=data,
        method="POST",
        headers={
            "x-admin-token": token,
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    active_opener = opener or request.urlopen
    with active_opener(req, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("upload response must contain a JSON object")
    model = payload.get("model") if isinstance(payload.get("model"), dict) else {}
    lifecycle = str(model.get("lifecycle_state") or "").upper()
    if lifecycle and lifecycle != "TRAINED":
        raise ValueError("upload endpoint returned a non-TRAINED lifecycle")
    response_status = str(payload.get("status") or "").lower()
    model_status = str(model.get("status") or "").lower()
    if response_status and response_status not in {"candidate", "trained"}:
        raise ValueError("upload endpoint returned a non-candidate status")
    if model_status and model_status not in {"candidate", "trained"}:
        raise ValueError("upload endpoint returned an active model")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a packaged model to Railway as a candidate.")
    parser.add_argument("--url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--package", type=Path, required=True)
    args = parser.parse_args()

    print(
        json.dumps(
            upload_package(url=args.url, token=args.token, package=args.package),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
