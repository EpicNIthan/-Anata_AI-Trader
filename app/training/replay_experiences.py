from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import ExperienceRecord, TrainingFeature
from app.db.session import SessionLocal, create_db_and_tables
from app.features.schema import CURRENT_FEATURE_SCHEMA_VERSION


def _reward_for_action(action: str, values: dict[str, Any]) -> float:
    future_return = float(values.get("target_future_return_15m") or 0.0)
    max_drawdown = float(values.get("target_max_drawdown_1h") or 0.0)
    fee_penalty = settings.paper_fee_rate * 2.0
    if action == "BUY":
        return future_return - fee_penalty + min(max_drawdown, 0.0) * 0.25
    if action == "HOLD":
        return -abs(future_return) * 0.15
    return 0.0


def _result_for_action(action: str, values: dict[str, Any]) -> dict[str, Any]:
    future_return = float(values.get("target_future_return_15m") or 0.0)
    return {
        "status": "REPLAYED",
        "message": f"Offline replay {action}; no live paper order was created.",
        "future_return_15m": future_return,
        "future_return_1h": values.get("target_future_return_1h"),
        "max_upside_1h": values.get("target_max_upside_1h"),
        "max_drawdown_1h": values.get("target_max_drawdown_1h"),
        "stop_loss_hit_first": bool(values.get("target_stop_loss_hit_first")),
        "take_profit_hit_first": bool(values.get("target_take_profit_hit_first")),
    }


def _existing_replay(session: Session, symbol: str, action: str, created_at: datetime) -> bool:
    return (
        session.scalar(
            select(ExperienceRecord.id)
            .where(
                ExperienceRecord.symbol == symbol,
                ExperienceRecord.action == action,
                ExperienceRecord.created_at == created_at,
            )
            .limit(1)
        )
        is not None
    )


def replay_experiences(
    *,
    symbols: list[str] | None = None,
    actions: list[str] | None = None,
    limit: int = 10_000,
    schema_version: str = CURRENT_FEATURE_SCHEMA_VERSION,
) -> dict[str, Any]:
    create_db_and_tables()
    normalized_symbols = [symbol.upper() for symbol in symbols] if symbols else None
    actions = [action.upper() for action in (actions or ["BUY", "HOLD"])]
    created = 0
    per_action = {action: 0 for action in actions}
    with SessionLocal() as session:
        query = select(TrainingFeature).where(
            TrainingFeature.schema_version == schema_version,
            TrainingFeature.source_name == "historical_replay_builder",
        )
        if normalized_symbols:
            query = query.where(TrainingFeature.symbol.in_(normalized_symbols))
        features = list(session.scalars(query.order_by(TrainingFeature.as_of).limit(limit)))
        for feature in features:
            values = dict(feature.feature_values or (feature.payload or {}).get("values", {}))
            for action in actions:
                if _existing_replay(session, feature.symbol, action, feature.as_of):
                    continue
                reward = _reward_for_action(action, values)
                session.add(
                    ExperienceRecord(
                        feature_id=None,
                        symbol=feature.symbol,
                        feature_schema_version=feature.schema_version,
                        market_state={
                            "source": "historical_replay",
                            "as_of": feature.as_of.isoformat(),
                            "last_close": values.get("last_close"),
                        },
                        news_state={"source": "historical_replay", "available": False},
                        feature_payload=feature.payload,
                        action=action,
                        confidence=0.5,
                        result=_result_for_action(action, values),
                        reward=reward,
                        raw_payload={
                            "source": "historical_replay",
                            "training_feature_id": feature.id,
                            "label_values": {key: value for key, value in values.items() if key.startswith("target_")},
                        },
                        created_at=feature.as_of,
                    )
                )
                created += 1
                per_action[action] += 1
        session.commit()
    return {
        "source": "historical_replay",
        "schema_version": schema_version,
        "symbols": normalized_symbols or "all",
        "actions": actions,
        "features_seen": len(features),
        "experiences_created": created,
        "per_action": per_action,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create offline replay experience rows from training features.")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--actions", default="BUY,HOLD")
    parser.add_argument("--limit", type=int, default=10_000)
    args = parser.parse_args()
    symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()] or None
    actions = [item.strip().upper() for item in args.actions.split(",") if item.strip()]
    print(replay_experiences(symbols=symbols, actions=actions, limit=args.limit))


if __name__ == "__main__":
    main()
