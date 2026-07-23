"""Launch one explicit Anata V2 process role.

Every role hosts the same lightweight health/API surface while the application
lifespan starts only the background services assigned to ``WORKER_ROLE``.  The
trader role is the fake-money paper trader; this launcher has no live broker mode.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

VALID_ROLES = ("web", "collector", "paper-trader", "enrichment", "all")


def normalize_role(value: str) -> str:
    role = value.strip().lower().replace("_", "-")
    if role not in VALID_ROLES:
        raise argparse.ArgumentTypeError(f"role must be one of: {', '.join(VALID_ROLES)}")
    return role


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch a paper-only Anata process role.")
    parser.add_argument(
        "--role",
        type=normalize_role,
        default=normalize_role(os.getenv("WORKER_ROLE", "all")),
        choices=VALID_ROLES,
    )
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    parser.add_argument("--log-level", choices=("critical", "error", "warning", "info", "debug", "trace"), default="info")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not (1 <= args.port <= 65535):
        raise SystemExit("--port must be between 1 and 65535")

    # app.config creates its frozen settings object on import. Set the role before
    # uvicorn imports app.main so process responsibilities cannot drift at runtime.
    os.environ["WORKER_ROLE"] = args.role
    os.environ["ANATA_EXECUTION_BOUNDARY"] = "paper-only"

    import uvicorn

    print(f"Starting Anata role={args.role} boundary=paper-only on {args.host}:{args.port}", flush=True)
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        workers=1,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
