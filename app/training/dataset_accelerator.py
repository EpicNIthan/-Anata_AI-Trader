from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import settings
from app.features.schema import CURRENT_FEATURE_SCHEMA_VERSION
from app.training.backfill_candles import backfill_historical_candles
from app.training.build_training_labels import build_training_labels
from app.training.export_dataset import export_dataset
from app.training.replay_experiences import replay_experiences


async def build_accelerated_dataset(
    *,
    symbols: list[str] | None = None,
    interval: str = "1m",
    days: int = 14,
    max_rows_per_symbol: int = 5000,
    lookback: int = 60,
    stride: int = 5,
    replay_limit: int = 20_000,
    backfill: bool = True,
    mock: bool = False,
    export: bool = True,
) -> dict[str, Any]:
    normalized_symbols = [symbol.upper() for symbol in (symbols or settings.binance_symbols)]
    started = datetime.now(timezone.utc)
    backfill_result = None
    if backfill:
        backfill_result = await backfill_historical_candles(
            symbols=normalized_symbols,
            interval=interval,
            days=days,
            max_rows_per_symbol=max_rows_per_symbol,
            mock=mock,
        )
    labels_result = build_training_labels(
        symbols=normalized_symbols,
        interval=interval,
        lookback=lookback,
        stride=stride,
        max_rows_per_symbol=max_rows_per_symbol,
    )
    replay_result = replay_experiences(
        symbols=normalized_symbols,
        actions=["BUY", "HOLD"],
        limit=replay_limit,
        schema_version=CURRENT_FEATURE_SCHEMA_VERSION,
    )
    export_path = None
    if export:
        output_path = Path("datasets") / f"accelerated_features_{started.strftime('%Y%m%d_%H%M%S')}.csv"
        export_path = export_dataset(
            output_path,
            feature_schema_version=CURRENT_FEATURE_SCHEMA_VERSION,
            use_all_data=True,
        )
    return {
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "symbols": normalized_symbols,
        "interval": interval,
        "days": days,
        "max_rows_per_symbol": max_rows_per_symbol,
        "lookback": lookback,
        "stride": stride,
        "mock": mock,
        "backfill": backfill_result,
        "labels": labels_result,
        "replay": replay_result,
        "exported_path": str(export_path) if export_path else None,
        "next_step": "Train with: python -m app.training.train_price_model --dataset <exported_path>",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast dataset builder: backfill candles, label rows, replay experiences, export CSV.")
    parser.add_argument("--symbols", default=",".join(settings.binance_symbols))
    parser.add_argument("--interval", default=settings.paper_trade_timeframe)
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--max-rows-per-symbol", type=int, default=5000)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--replay-limit", type=int, default=20_000)
    parser.add_argument("--no-backfill", action="store_true")
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    result = asyncio.run(
        build_accelerated_dataset(
            symbols=[item.strip().upper() for item in args.symbols.split(",") if item.strip()],
            interval=args.interval,
            days=args.days,
            max_rows_per_symbol=args.max_rows_per_symbol,
            lookback=args.lookback,
            stride=args.stride,
            replay_limit=args.replay_limit,
            backfill=not args.no_backfill,
            mock=args.mock,
            export=not args.no_export,
        )
    )
    print(result)


if __name__ == "__main__":
    main()
