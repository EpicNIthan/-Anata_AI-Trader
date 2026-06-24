from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Candle(Base):
    __tablename__ = "candles"
    __table_args__ = (
        UniqueConstraint("exchange", "symbol", "interval", "open_time", name="uq_candle_identity"),
        Index("ix_candles_symbol_time", "symbol", "open_time"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    exchange: Mapped[str] = mapped_column(String(32), default="binance", index=True)
    source_name: Mapped[str | None] = mapped_column(String(128), default="binance_kline", nullable=True, index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    interval: Mapped[str] = mapped_column(String(16), default="1m")
    open_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    close_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float, default=0.0)
    quote_volume: Mapped[float] = mapped_column(Float, default=0.0)
    trades: Mapped[int] = mapped_column(Integer, default=0)
    is_closed: Mapped[bool] = mapped_column(default=False)
    raw: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    raw_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)


class MarketTick(Base):
    __tablename__ = "market_ticks"
    __table_args__ = (Index("ix_market_ticks_symbol_time", "symbol", "event_time"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    exchange: Mapped[str] = mapped_column(String(32), default="binance", index=True)
    source_name: Mapped[str | None] = mapped_column(String(128), default="binance_tick", nullable=True, index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    event_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    price: Mapped[float] = mapped_column(Float)
    quantity: Mapped[float] = mapped_column(Float, default=0.0)
    raw: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    raw_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class NewsArticle(Base):
    __tablename__ = "news_articles"
    __table_args__ = (
        UniqueConstraint("source", "url", name="uq_news_source_url"),
        Index("ix_news_articles_published", "published_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String(128), index=True)
    source_name: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    title: Mapped[str] = mapped_column(Text)
    url: Mapped[str] = mapped_column(Text)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    raw_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    sentiment: Mapped["NewsSentiment | None"] = relationship(back_populates="article")


class NewsSentiment(Base):
    __tablename__ = "news_sentiment"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    article_id: Mapped[int] = mapped_column(ForeignKey("news_articles.id"), index=True)
    sentiment_score: Mapped[float] = mapped_column(Float, default=0.0)
    risk_score: Mapped[float] = mapped_column(Float, default=0.0)
    topics: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    affected_symbols: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    model_name: Mapped[str] = mapped_column(String(128), default="placeholder-v1")
    sentiment_label: Mapped[str | None] = mapped_column(String(32), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    source_name: Mapped[str | None] = mapped_column(String(128), default="placeholder-v1", nullable=True, index=True)
    raw_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    article: Mapped[NewsArticle] = relationship(back_populates="sentiment")


class Feature(Base):
    __tablename__ = "features"
    __table_args__ = (Index("ix_features_symbol_time", "symbol", "as_of"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    schema_version: Mapped[str] = mapped_column(String(64), default="price-news-v1", index=True)
    source_name: Mapped[str | None] = mapped_column(String(128), default="feature_builder", nullable=True, index=True)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    price_change: Mapped[float] = mapped_column(Float, default=0.0)
    volume_change: Mapped[float] = mapped_column(Float, default=0.0)
    volatility: Mapped[float] = mapped_column(Float, default=0.0)
    trend: Mapped[str] = mapped_column(String(32), default="sideways")
    sentiment_score: Mapped[float] = mapped_column(Float, default=0.0)
    risk_score: Mapped[float] = mapped_column(Float, default=0.0)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    raw_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class PaperTrade(Base):
    __tablename__ = "paper_trades"
    __table_args__ = (Index("ix_paper_trades_symbol_time", "symbol", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    action: Mapped[str] = mapped_column(String(16))
    side: Mapped[str] = mapped_column(String(16), default="LONG")
    quantity: Mapped[float] = mapped_column(Float, default=0.0)
    price: Mapped[float] = mapped_column(Float, default=0.0)
    notional: Mapped[float] = mapped_column(Float, default=0.0)
    fee: Mapped[float] = mapped_column(Float, default=0.0)
    realized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    balance_after: Mapped[float] = mapped_column(Float, default=0.0)
    equity_after: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(32), default="FILLED")
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class Position(Base):
    __tablename__ = "positions"
    __table_args__ = (Index("ix_positions_symbol_status", "symbol", "status"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    side: Mapped[str] = mapped_column(String(16), default="LONG")
    quantity: Mapped[float] = mapped_column(Float, default=0.0)
    entry_price: Mapped[float] = mapped_column(Float, default=0.0)
    current_price: Mapped[float] = mapped_column(Float, default=0.0)
    stop_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    take_profit: Mapped[float | None] = mapped_column(Float, nullable=True)
    realized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    unrealized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(32), default="OPEN", index=True)
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    name: Mapped[str] = mapped_column(String(128), index=True)
    version: Mapped[str] = mapped_column(String(64), index=True)
    feature_schema_version: Mapped[str] = mapped_column(String(64), default="price-news-v1", index=True)
    feature_columns: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    path: Mapped[str] = mapped_column(Text)
    parent_model_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    checkpoint_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="trained")
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    raw_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_name: Mapped[str] = mapped_column(String(128), index=True)
    dataset_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_version_id: Mapped[int | None] = mapped_column(ForeignKey("model_versions.id"), nullable=True)
    feature_schema_version: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    from_checkpoint_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    since_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    use_all_data: Mapped[bool] = mapped_column(default=False)
    status: Mapped[str] = mapped_column(String(32), default="created")
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class AiDecision(Base):
    __tablename__ = "ai_decisions"
    __table_args__ = (Index("ix_ai_decisions_symbol_time", "symbol", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    strategy_name: Mapped[str] = mapped_column(String(128), default="rule-based-v1")
    source_name: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    feature_id: Mapped[int | None] = mapped_column(ForeignKey("features.id"), nullable=True)
    model_version_id: Mapped[int | None] = mapped_column(ForeignKey("model_versions.id"), nullable=True)
    feature_schema_version: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    action: Mapped[str] = mapped_column(String(16))
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    stop_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    take_profit: Mapped[float | None] = mapped_column(Float, nullable=True)
    execution_status: Mapped[str] = mapped_column(String(32), default="PENDING")
    execution_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    trade_id: Mapped[int | None] = mapped_column(ForeignKey("paper_trades.id"), nullable=True)
    market_state: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    news_state: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    reward: Mapped[float | None] = mapped_column(Float, nullable=True)
    raw: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    raw_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class ExternalDataEvent(Base):
    __tablename__ = "external_data_events"
    __table_args__ = (
        Index("ix_external_data_source_type_time", "source_name", "data_type", "event_time"),
        Index("ix_external_data_symbol_time", "symbol", "event_time"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_name: Mapped[str] = mapped_column(String(128), index=True)
    data_type: Mapped[str] = mapped_column(String(64), index=True)
    symbol: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    event_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    numeric_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    raw_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class ExperienceRecord(Base):
    __tablename__ = "experience_buffer"
    __table_args__ = (
        Index("ix_experience_symbol_time", "symbol", "created_at"),
        Index("ix_experience_schema_time", "feature_schema_version", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ai_decision_id: Mapped[int | None] = mapped_column(ForeignKey("ai_decisions.id"), nullable=True, index=True)
    feature_id: Mapped[int | None] = mapped_column(ForeignKey("features.id"), nullable=True, index=True)
    model_version_id: Mapped[int | None] = mapped_column(ForeignKey("model_versions.id"), nullable=True, index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    feature_schema_version: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    market_state: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    news_state: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    feature_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    action: Mapped[str] = mapped_column(String(16))
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    reward: Mapped[float] = mapped_column(Float, default=0.0)
    raw_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class AccountEquity(Base):
    __tablename__ = "account_equity"
    __table_args__ = (Index("ix_account_equity_time", "timestamp"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    cash_balance: Mapped[float] = mapped_column(Float, default=0.0)
    equity: Mapped[float] = mapped_column(Float, default=0.0)
    realized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    unrealized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    drawdown: Mapped[float] = mapped_column(Float, default=0.0)
    raw: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
