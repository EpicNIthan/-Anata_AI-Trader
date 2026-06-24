from __future__ import annotations

import argparse
import json
from urllib import request


def main() -> None:
    parser = argparse.ArgumentParser(description="Activate a candidate model on Railway after checking metrics.")
    parser.add_argument("--url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--model-id", required=True)
    args = parser.parse_args()

    data = json.dumps({"model_id": args.model_id}).encode("utf-8")
    req = request.Request(
        args.url.rstrip("/") + "/api/models/activate",
        data=data,
        method="POST",
        headers={
            "x-admin-token": args.token,
            "Content-Type": "application/json",
        },
    )
    with request.urlopen(req, timeout=120) as response:
        print(json.dumps(json.loads(response.read().decode("utf-8")), indent=2))


if __name__ == "__main__":
    main()
