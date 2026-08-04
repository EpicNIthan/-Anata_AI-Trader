"""Run one bounded Anata V2 paper-decision cycle from a trusted shell."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one paper-only V2 decision cycle.")
    parser.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated symbols. Defaults to AUTO_TRADER_SYMBOLS.",
    )
    parser.add_argument(
        "--paper-account-id",
        default=None,
        help="Champion account or an active registered sandbox account.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from app.config import settings
    from app.db.session import SessionLocal, create_db_and_tables
    from app.pipeline.service import V2PipelineService

    if not settings.is_paper_mode:
        print(json.dumps({"status": "rejected", "reason": "TRADING_MODE must be paper"}, indent=2))
        return 2
    symbols = [
        value.strip().upper()
        for value in (args.symbols.split(",") if args.symbols else settings.auto_trader_symbols)
        if value.strip()
    ]
    if not symbols:
        print(json.dumps({"status": "error", "reason": "no symbols configured"}, indent=2))
        return 2
    account_id = args.paper_account_id or settings.v2_champion_account_id
    create_db_and_tables()
    results: list[dict[str, object]] = []
    with SessionLocal() as session:
        service = V2PipelineService(session)
        for symbol in symbols:
            try:
                results.append(service.run_symbol(symbol, account_id=account_id).as_dict())
            except Exception as exc:
                session.rollback()
                results.append(
                    {
                        "symbol": symbol,
                        "status": "ERROR",
                        "message": f"{type(exc).__name__}: {exc}",
                    }
                )
    print(
        json.dumps(
            {
                "status": "completed",
                "paper_only": True,
                "paper_account_id": account_id,
                "cycles": results,
            },
            indent=2,
            default=str,
        )
    )
    return 0 if all(item.get("status") != "ERROR" for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
