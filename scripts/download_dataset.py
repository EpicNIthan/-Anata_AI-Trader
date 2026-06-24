from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib import request


def _api(url: str, token: str, path: str, *, method: str = "GET", body: dict | None = None) -> bytes:
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
    with request.urlopen(req, timeout=120) as response:
        return response.read()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download an exported dataset from Railway.")
    parser.add_argument("--url", required=True, help="Railway app URL, for example https://your-app.up.railway.app")
    parser.add_argument("--token", required=True, help="ADMIN_TOKEN")
    parser.add_argument("--dataset-id", default=None, help="Existing dataset id/filename. If omitted, a new export is created.")
    parser.add_argument("--out-dir", type=Path, default=Path("datasets"))
    args = parser.parse_args()

    dataset_id = args.dataset_id
    if not dataset_id:
        payload = json.loads(_api(args.url, args.token, "/api/training/export", method="POST", body={"use_all_data": True}))
        dataset_id = payload["dataset_id"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = _api(args.url, args.token, f"/api/training/download/{Path(dataset_id).name}")
    output = args.out_dir / Path(dataset_id).name
    output.write_bytes(data)
    latest = args.out_dir / "latest.csv.gz"
    latest.write_bytes(data)
    print(json.dumps({"downloaded": str(output), "latest": str(latest), "bytes": len(data)}, indent=2))


if __name__ == "__main__":
    main()
