from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from app.ai.experience_buffer import record_experience
from app.db.models import AiDecision
from app.db.session import get_session
from app.security import require_admin
from app.trading.paper_engine import PaperEngine

router = APIRouter(prefix="/api", tags=["signals"])


class SignalRequest(BaseModel):
    symbol: str = Field(min_length=3, max_length=32)
    action: Literal["BUY", "SELL", "HOLD", "CLOSE", "buy", "sell", "hold", "close"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str | None = None
    stop_loss: float | None = Field(default=None, gt=0)
    take_profit: float | None = Field(default=None, gt=0)
    price: float | None = Field(default=None, gt=0)
    quantity: float | None = Field(default=None, gt=0)
    notional: float | None = Field(default=None, gt=0)
    source: str = "external-signal"

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("action")
    @classmethod
    def normalize_action(cls, value: str) -> str:
        return value.upper()


@router.post("/signal")
def receive_signal(
    payload: SignalRequest,
    session: Session = Depends(get_session),
    _: None = Depends(require_admin),
) -> dict[str, object]:
    engine = PaperEngine(session)
    result = engine.execute_signal(
        symbol=payload.symbol,
        action=payload.action,
        confidence=payload.confidence,
        reason=payload.reason,
        stop_loss=payload.stop_loss,
        take_profit=payload.take_profit,
        price=payload.price,
        quantity=payload.quantity,
        notional=payload.notional,
    )
    execution = {
        "status": result.status,
        "message": result.message,
        "trade_id": result.trade_id,
        "balance": result.balance,
        "equity": result.equity,
    }
    decision = AiDecision(
        symbol=payload.symbol,
        strategy_name=payload.source,
        source_name=payload.source,
        action=payload.action,
        confidence=payload.confidence,
        reason=payload.reason,
        stop_loss=payload.stop_loss,
        take_profit=payload.take_profit,
        execution_status=result.status,
        execution_message=result.message,
        trade_id=result.trade_id,
        raw=payload.model_dump(),
        result=execution,
    )
    session.add(decision)
    session.flush()
    record_experience(session, decision=decision, feature=None, execution_result=execution)
    session.commit()
    session.refresh(decision)

    return {
        "decision_id": decision.id,
        **execution,
    }
