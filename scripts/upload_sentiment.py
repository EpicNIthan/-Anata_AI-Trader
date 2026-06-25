from __future__ import annotations

import argparse
import json
import mimetypes
import uuid
from pathlib import Path
from urllib import request


def _multipart_body(field_name: str, path: Path, boundary: str) -> bytes:
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field_name}"; filename="{path.name}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("utf-8")
    footer = f"\r\n--{boundary}--\r\n".encode("utf-8")
    return header + path.read_bytes() + footer


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload locally-scored news sentiment back to Railway.")
    parser.add_argument("--url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--file", type=Path, default=Path("datasets/latest_news_sentiment.jsonl.gz"))
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    boundary = f"----anata-sentiment-{uuid.uuid4().hex}"
    data = _multipart_body("file", args.file, boundary)
    req = request.Request(
        args.url.rstrip("/") + "/api/sentiment/import",
        data=data,
        method="POST",
        headers={
            "x-admin-token": args.token,
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    with request.urlopen(req, timeout=args.timeout) as response:
        print(json.dumps(json.loads(response.read().decode("utf-8")), indent=2))


if __name__ == "__main__":
    main()
