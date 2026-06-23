from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import (
    AccountEquity,
    AiDecision,
    Candle,
    ExperienceRecord,
    ModelVersion,
    NewsArticle,
    NewsSentiment,
    PaperTrade,
    Position,
)
from app.db.session import get_session
from app.services.collector_status import market_snapshot, news_snapshot
from app.trading.paper_engine import PaperEngine

router = APIRouter(tags=["dashboard"])
templates = Jinja2Templates(directory="app/dashboard/templates")


def _dt(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:,.{digits}f}"


@router.get("/", include_in_schema=False)
def index() -> RedirectResponse:
    return RedirectResponse(url="/dashboard")


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request, session: Session = Depends(get_session)) -> HTMLResponse:
    account = PaperEngine(session).snapshot()
    open_positions = list(session.scalars(select(Position).where(Position.status == "OPEN").order_by(desc(Position.opened_at))))
    trades = list(session.scalars(select(PaperTrade).order_by(desc(PaperTrade.created_at)).limit(25)))
    latest_decision = session.scalar(select(AiDecision).order_by(desc(AiDecision.created_at)).limit(1))
    latest_model = session.scalar(select(ModelVersion).order_by(desc(ModelVersion.created_at)).limit(1))

    equity_rows = list(session.scalars(select(AccountEquity).order_by(desc(AccountEquity.timestamp)).limit(100)))
    equity_rows.reverse()
    equity_points = [{"time": row.timestamp.strftime("%H:%M:%S"), "equity": row.equity} for row in equity_rows]

    closed_trades = list(session.scalars(select(PaperTrade).where(PaperTrade.action == "SELL").order_by(desc(PaperTrade.created_at)).limit(100)))
    wins = len([trade for trade in closed_trades if trade.realized_pnl > 0])
    win_rate = wins / len(closed_trades) if closed_trades else 0.0

    news_rows = session.execute(
        select(NewsSentiment, NewsArticle)
        .join(NewsArticle, NewsArticle.id == NewsSentiment.article_id)
        .order_by(desc(NewsSentiment.created_at))
        .limit(8)
    ).all()
    latest_news = [
        {
            "title": article.title,
            "source": article.source,
            "url": article.url,
            "sentiment": sentiment.sentiment_score,
            "risk": sentiment.risk_score,
            "symbols": ", ".join(sentiment.affected_symbols or []),
        }
        for sentiment, article in news_rows
    ]

    trade_count = session.scalar(select(func.count(PaperTrade.id))) or 0
    pnl = account["equity"] - settings.paper_start_balance
    collector_status = getattr(request.app.state, "worker_manager", None)
    collectors = collector_status.snapshot() if collector_status else {}
    market_status = market_snapshot(session, collectors.get("market", {}))
    news_status = news_snapshot(session, collectors.get("news", {}))
    auto_trader_service = getattr(request.app.state, "auto_trader", None)
    auto_trader = auto_trader_service.status() if auto_trader_service else {}
    latest_auto_decision = session.scalar(
        select(AiDecision)
        .where(AiDecision.source_name == "auto-trader-v1")
        .order_by(desc(AiDecision.created_at))
        .limit(1)
    )
    data_counts = {
        "candles": session.scalar(select(func.count(Candle.id))) or 0,
        "news": session.scalar(select(func.count(NewsArticle.id))) or 0,
        "experiences": session.scalar(select(func.count(ExperienceRecord.id))) or 0,
    }

    context: dict[str, Any] = {
        "request": request,
        "account": account,
        "pnl": pnl,
        "win_rate": win_rate,
        "positions": open_positions,
        "trades": trades,
        "trade_count": trade_count,
        "latest_news": latest_news,
        "latest_decision": latest_decision,
        "latest_model": latest_model,
        "collectors": collectors,
        "market_status": market_status,
        "news_status": news_status,
        "auto_trader": auto_trader,
        "latest_auto_decision": latest_auto_decision,
        "data_counts": data_counts,
        "equity_points_json": json.dumps(equity_points),
        "fmt": _fmt,
        "dt": _dt,
        "mode": settings.trading_mode,
    }
    return templates.TemplateResponse(request, "dashboard.html", context)
