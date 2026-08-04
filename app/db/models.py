from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, JSON, LargeBinary, String, Text, UniqueConstraint
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


class LiveCandleUpdate(Base):
    __tablename__ = "live_candle_updates"
    __table_args__ = (
        UniqueConstraint("exchange", "symbol", "interval", "open_time", name="uq_live_candle_update_identity"),
        Index("ix_live_candle_updates_symbol_time", "symbol", "open_time"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    exchange: Mapped[str] = mapped_column(String(32), default="binance", index=True)
    source_name: Mapped[str | None] = mapped_column(String(128), default="binance_kline_live", nullable=True, index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    interval: Mapped[str] = mapped_column(String(16), default="1m")
    event_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    open_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    close_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float, default=0.0)
    quote_volume: Mapped[float] = mapped_column(Float, default=0.0)
    trades: Mapped[int] = mapped_column(Integer, default=0)
    update_count: Mapped[int] = mapped_column(Integer, default=1)
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
    event_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    received_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    processed_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    available_to_model_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
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
    event_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    received_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    processed_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    available_to_model_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    price_change: Mapped[float] = mapped_column(Float, default=0.0)
    volume_change: Mapped[float] = mapped_column(Float, default=0.0)
    volatility: Mapped[float] = mapped_column(Float, default=0.0)
    trend: Mapped[str] = mapped_column(String(32), default="sideways")
    sentiment_score: Mapped[float] = mapped_column(Float, default=0.0)
    risk_score: Mapped[float] = mapped_column(Float, default=0.0)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    raw_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class TrainingFeature(Base):
    __tablename__ = "training_features"
    __table_args__ = (
        UniqueConstraint("source_feature_id", name="uq_training_features_source_feature_id"),
        Index("ix_training_features_symbol_time", "symbol", "as_of"),
        Index("ix_training_features_schema_time", "schema_version", "as_of"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_feature_id: Mapped[int | None] = mapped_column(ForeignKey("features.id"), nullable=True, index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    schema_version: Mapped[str] = mapped_column(String(64), default="price-news-v3", index=True)
    source_name: Mapped[str | None] = mapped_column(String(128), default="feature_builder", nullable=True, index=True)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    feature_values: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class PaperTrade(Base):
    __tablename__ = "paper_trades"
    __table_args__ = (Index("ix_paper_trades_symbol_time", "symbol", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    paper_account_id: Mapped[str] = mapped_column(String(128), default="champion", index=True)
    risk_decision_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    simulated_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    decision_trace_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
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
    paper_account_id: Mapped[str] = mapped_column(String(128), default="champion", index=True)
    side: Mapped[str] = mapped_column(String(16), default="LONG")
    quantity: Mapped[float] = mapped_column(Float, default=0.0)
    entry_price: Mapped[float] = mapped_column(Float, default=0.0)
    current_price: Mapped[float] = mapped_column(Float, default=0.0)
    notional: Mapped[float] = mapped_column(Float, default=0.0)
    margin_used: Mapped[float] = mapped_column(Float, default=0.0)
    leverage: Mapped[float] = mapped_column(Float, default=1.0)
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
    model_family: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    lifecycle_state: Mapped[str] = mapped_column(String(32), default="TRAINED", index=True)
    health_status: Mapped[str] = mapped_column(String(32), default="HEALTHY", index=True)
    artifact_checksum: Mapped[str | None] = mapped_column(String(128), nullable=True)
    preprocessing_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    training_dataset_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    training_start_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    training_end_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    forecast_horizon_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    package_manifest: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    promotion_history: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    suspension_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    retirement_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    raw_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class ModelArtifactBlob(Base):
    """Immutable durable bytes for one registered model version.

    ``ModelVersion.path`` remains useful as an origin/audit hint, while runtime roles
    can reconstruct the exact artifact from this database row when their container
    does not share the uploader's filesystem.
    """

    __tablename__ = "model_artifact_blobs"
    __table_args__ = (
        UniqueConstraint("model_version_id", name="uq_model_artifact_blobs_model_version"),
        Index("ix_model_artifact_blobs_sha256", "sha256"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_version_id: Mapped[int] = mapped_column(
        ForeignKey("model_versions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    media_type: Mapped[str] = mapped_column(String(128), default="application/octet-stream")
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
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
    paper_account_id: Mapped[str] = mapped_column(String(128), default="champion", index=True)
    cash_balance: Mapped[float] = mapped_column(Float, default=0.0)
    equity: Mapped[float] = mapped_column(Float, default=0.0)
    realized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    unrealized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    drawdown: Mapped[float] = mapped_column(Float, default=0.0)
    raw: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


# ---------------------------------------------------------------------------
# V2 quantitative pipeline ledger.  These tables intentionally coexist with
# the legacy operational tables above so existing Railway deployments and paper
# history remain readable while new decisions receive stage-level traceability.
# ---------------------------------------------------------------------------


class ModelPredictionRecord(Base):
    __tablename__ = "model_predictions"
    __table_args__ = (
        UniqueConstraint("prediction_id", name="uq_model_predictions_prediction_id"),
        Index("ix_model_predictions_symbol_time", "symbol", "generated_at"),
        Index("ix_model_predictions_model_time", "model_id", "generated_at"),
        Index("ix_model_predictions_trace_time", "decision_trace_id", "generated_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    prediction_id: Mapped[str] = mapped_column(String(64), index=True)
    decision_trace_id: Mapped[str] = mapped_column(String(64), index=True)
    model_version_id: Mapped[int | None] = mapped_column(ForeignKey("model_versions.id"), nullable=True, index=True)
    model_id: Mapped[str] = mapped_column(String(128), index=True)
    model_version: Mapped[str] = mapped_column(String(64), index=True)
    model_family: Mapped[str] = mapped_column(String(128), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    valid_from: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    forecast_horizon_seconds: Mapped[int] = mapped_column(Integer)
    expected_return: Mapped[float] = mapped_column(Float)
    expected_volatility: Mapped[float] = mapped_column(Float, default=0.0)
    probability_up: Mapped[float] = mapped_column(Float, default=0.5)
    probability_down: Mapped[float] = mapped_column(Float, default=0.5)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    calibration_score: Mapped[float] = mapped_column(Float, default=0.5)
    uncertainty: Mapped[float] = mapped_column(Float, default=0.5)
    regime: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    feature_schema_version: Mapped[str] = mapped_column(String(64), index=True)
    feature_snapshot_id: Mapped[str] = mapped_column(String(64), index=True)
    feature_id: Mapped[int | None] = mapped_column(ForeignKey("features.id"), nullable=True, index=True)
    data_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    external_context_available: Mapped[bool] = mapped_column(default=False, index=True)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class TradingSignalRecord(Base):
    __tablename__ = "trading_signals"
    __table_args__ = (
        UniqueConstraint("signal_id", name="uq_trading_signals_signal_id"),
        Index("ix_trading_signals_symbol_time", "symbol", "generated_at"),
        Index("ix_trading_signals_prediction", "prediction_id"),
        Index("ix_trading_signals_lifecycle", "lifecycle_status", "health_status"),
        Index("ix_trading_signals_trace_time", "decision_trace_id", "generated_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    signal_id: Mapped[str] = mapped_column(String(64), index=True)
    prediction_id: Mapped[str] = mapped_column(String(64), ForeignKey("model_predictions.prediction_id"), index=True)
    decision_trace_id: Mapped[str] = mapped_column(String(64), index=True)
    signal_family: Mapped[str] = mapped_column(String(128), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    valid_until: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    direction: Mapped[str] = mapped_column(String(16), index=True)
    strength: Mapped[float] = mapped_column(Float, default=0.0)
    expected_return: Mapped[float] = mapped_column(Float, default=0.0)
    expected_cost: Mapped[float] = mapped_column(Float, default=0.0)
    net_expected_return: Mapped[float] = mapped_column(Float, default=0.0)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    uncertainty: Mapped[float] = mapped_column(Float, default=0.5)
    regime: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    liquidity_score: Mapped[float] = mapped_column(Float, default=0.5)
    health_status: Mapped[str] = mapped_column(String(32), default="HEALTHY", index=True)
    lifecycle_status: Mapped[str] = mapped_column(String(32), default="PAPER", index=True)
    reason_codes: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class SignalOutcome(Base):
    __tablename__ = "signal_outcomes"
    __table_args__ = (
        Index("ix_signal_outcomes_signal_time", "signal_id", "observed_at"),
        Index("ix_signal_outcomes_symbol_time", "symbol", "observed_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    signal_id: Mapped[str] = mapped_column(String(64), ForeignKey("trading_signals.signal_id"), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    horizon_seconds: Mapped[int] = mapped_column(Integer, default=0)
    realized_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    realized_cost: Mapped[float | None] = mapped_column(Float, nullable=True)
    net_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    directional_hit: Mapped[bool | None] = mapped_column(nullable=True)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class EnsembleDecisionRecord(Base):
    __tablename__ = "ensemble_decisions"
    __table_args__ = (
        UniqueConstraint("ensemble_decision_id", name="uq_ensemble_decisions_id"),
        Index("ix_ensemble_decisions_symbol_time", "symbol", "generated_at"),
        Index("ix_ensemble_decisions_trace_time", "decision_trace_id", "generated_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ensemble_decision_id: Mapped[str] = mapped_column(String(64), index=True)
    decision_trace_id: Mapped[str] = mapped_column(String(64), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    valid_until: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    combined_expected_return: Mapped[float] = mapped_column(Float, default=0.0)
    combined_expected_volatility: Mapped[float] = mapped_column(Float, default=0.0)
    combined_uncertainty: Mapped[float] = mapped_column(Float, default=0.5)
    combined_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    current_regime: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    supporting_signals: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    conflicting_signals: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    signal_weights: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    correlation_penalty: Mapped[float] = mapped_column(Float, default=0.0)
    transaction_cost_penalty: Mapped[float] = mapped_column(Float, default=0.0)
    regime_penalty: Mapped[float] = mapped_column(Float, default=0.0)
    external_context_adjustment: Mapped[float] = mapped_column(Float, default=0.0)
    decision_status: Mapped[str] = mapped_column(String(32), default="NEUTRAL", index=True)
    reason_codes: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class EnsembleSignalWeight(Base):
    __tablename__ = "ensemble_signal_weights"
    __table_args__ = (
        UniqueConstraint("ensemble_decision_id", "signal_id", name="uq_ensemble_signal_weight"),
        Index("ix_ensemble_signal_weights_signal", "signal_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ensemble_decision_id: Mapped[str] = mapped_column(String(64), ForeignKey("ensemble_decisions.ensemble_decision_id"), index=True)
    signal_id: Mapped[str] = mapped_column(String(64), ForeignKey("trading_signals.signal_id"), index=True)
    weight: Mapped[float] = mapped_column(Float, default=0.0)
    exclusion_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class PortfolioTargetRecord(Base):
    __tablename__ = "portfolio_targets"
    __table_args__ = (
        UniqueConstraint("portfolio_target_id", name="uq_portfolio_targets_id"),
        Index("ix_portfolio_targets_symbol_time", "symbol", "created_at"),
        Index("ix_portfolio_targets_trace_time", "decision_trace_id", "created_at"),
        Index("ix_portfolio_targets_account_time", "paper_account_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    portfolio_target_id: Mapped[str] = mapped_column(String(64), index=True)
    decision_trace_id: Mapped[str] = mapped_column(String(64), index=True)
    paper_account_id: Mapped[str] = mapped_column(String(128), default="champion", index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    current_exposure: Mapped[float] = mapped_column(Float, default=0.0)
    requested_target_exposure: Mapped[float] = mapped_column(Float, default=0.0)
    requested_delta: Mapped[float] = mapped_column(Float, default=0.0)
    expected_return: Mapped[float] = mapped_column(Float, default=0.0)
    expected_risk: Mapped[float] = mapped_column(Float, default=0.0)
    risk_contribution: Mapped[float] = mapped_column(Float, default=0.0)
    urgency: Mapped[float] = mapped_column(Float, default=0.0)
    source_ensemble_decision_id: Mapped[str] = mapped_column(String(64), ForeignKey("ensemble_decisions.ensemble_decision_id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class RiskDecisionRecord(Base):
    __tablename__ = "risk_decisions"
    __table_args__ = (
        UniqueConstraint("risk_decision_id", name="uq_risk_decisions_id"),
        Index("ix_risk_decisions_target", "portfolio_target_id"),
        Index("ix_risk_decisions_trace_time", "decision_trace_id", "created_at"),
        Index("ix_risk_decisions_account_time", "paper_account_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    risk_decision_id: Mapped[str] = mapped_column(String(64), index=True)
    decision_trace_id: Mapped[str] = mapped_column(String(64), index=True)
    portfolio_target_id: Mapped[str] = mapped_column(String(64), ForeignKey("portfolio_targets.portfolio_target_id"), index=True)
    paper_account_id: Mapped[str] = mapped_column(String(128), default="champion", index=True)
    approved: Mapped[bool] = mapped_column(default=False, index=True)
    requested_exposure: Mapped[float] = mapped_column(Float, default=0.0)
    approved_exposure: Mapped[float] = mapped_column(Float, default=0.0)
    requested_leverage: Mapped[float] = mapped_column(Float, default=0.0)
    approved_leverage: Mapped[float] = mapped_column(Float, default=0.0)
    triggered_limits: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    rejection_reasons: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    configuration_version: Mapped[str] = mapped_column(String(128), default="v2")
    kill_switch_state: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class SimulatedOrderRecord(Base):
    __tablename__ = "simulated_orders"
    __table_args__ = (
        UniqueConstraint("order_id", name="uq_simulated_orders_order_id"),
        UniqueConstraint("client_order_id", name="uq_simulated_orders_client_order_id"),
        Index("ix_simulated_orders_account_symbol_state", "paper_account_id", "symbol", "state"),
        Index("ix_simulated_orders_trace_time", "decision_trace_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_id: Mapped[str] = mapped_column(String(64), index=True)
    decision_trace_id: Mapped[str] = mapped_column(String(64), index=True)
    risk_decision_id: Mapped[str] = mapped_column(String(64), ForeignKey("risk_decisions.risk_decision_id"), index=True)
    portfolio_target_id: Mapped[str] = mapped_column(String(64), ForeignKey("portfolio_targets.portfolio_target_id"), index=True)
    paper_account_id: Mapped[str] = mapped_column(String(128), default="champion", index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    side: Mapped[str] = mapped_column(String(16), index=True)
    order_type: Mapped[str] = mapped_column(String(16), default="MARKET")
    requested_quantity: Mapped[float] = mapped_column(Float, default=0.0)
    requested_notional: Mapped[float] = mapped_column(Float, default=0.0)
    limit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    state: Mapped[str] = mapped_column(String(32), default="CREATED", index=True)
    client_order_id: Mapped[str] = mapped_column(String(128), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class SimulatedFillRecord(Base):
    __tablename__ = "simulated_fills"
    __table_args__ = (
        UniqueConstraint("fill_id", name="uq_simulated_fills_fill_id"),
        Index("ix_simulated_fills_order_time", "order_id", "filled_at"),
        Index("ix_simulated_fills_account_symbol", "paper_account_id", "symbol"),
        Index("ix_simulated_fills_trace_time", "decision_trace_id", "filled_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    fill_id: Mapped[str] = mapped_column(String(64), index=True)
    order_id: Mapped[str] = mapped_column(String(64), ForeignKey("simulated_orders.order_id"), index=True)
    decision_trace_id: Mapped[str] = mapped_column(String(64), index=True)
    paper_account_id: Mapped[str] = mapped_column(String(128), default="champion", index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    side: Mapped[str] = mapped_column(String(16), index=True)
    quantity: Mapped[float] = mapped_column(Float, default=0.0)
    price: Mapped[float] = mapped_column(Float, default=0.0)
    notional: Mapped[float] = mapped_column(Float, default=0.0)
    fee: Mapped[float] = mapped_column(Float, default=0.0)
    slippage: Mapped[float] = mapped_column(Float, default=0.0)
    funding: Mapped[float] = mapped_column(Float, default=0.0)
    filled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class StrategyCandidate(Base):
    __tablename__ = "strategy_candidates"
    __table_args__ = (
        UniqueConstraint("candidate_id", name="uq_strategy_candidates_candidate_id"),
        Index("ix_strategy_candidates_lifecycle", "lifecycle_state", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    candidate_id: Mapped[str] = mapped_column(String(64), index=True)
    model_family: Mapped[str] = mapped_column(String(128), index=True)
    feature_families: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    target_name: Mapped[str] = mapped_column(String(128))
    forecast_horizon_seconds: Mapped[int] = mapped_column(Integer, default=300)
    hyperparameters: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    regime_filter: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    signal_threshold: Mapped[float] = mapped_column(Float, default=0.0)
    confidence_policy: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    cost_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    portfolio_policy: Mapped[str | None] = mapped_column(String(128), nullable=True)
    exit_policy_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    training_window: Mapped[str | None] = mapped_column(String(64), nullable=True)
    validation_window: Mapped[str | None] = mapped_column(String(64), nullable=True)
    random_seed: Mapped[int] = mapped_column(Integer, default=0)
    hypothesis: Mapped[str | None] = mapped_column(Text, nullable=True)
    lifecycle_state: Mapped[str] = mapped_column(String(32), default="RESEARCH", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class CandidateEvaluation(Base):
    __tablename__ = "candidate_evaluations"
    __table_args__ = (
        Index("ix_candidate_evaluations_candidate_time", "candidate_id", "created_at"),
        Index("ix_candidate_evaluations_status", "status", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    candidate_id: Mapped[str] = mapped_column(String(64), ForeignKey("strategy_candidates.candidate_id"), index=True)
    experiment_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(32), default="PENDING")
    technically_compatible: Mapped[bool] = mapped_column(default=False)
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)


class ChampionAssignment(Base):
    __tablename__ = "champion_assignments"
    __table_args__ = (
        Index("ix_champion_assignments_scope_time", "model_family", "symbol_scope", "active_from"),
        Index("ix_champion_assignments_active", "active_to"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"), index=True)
    model_family: Mapped[str] = mapped_column(String(128), index=True)
    symbol_scope: Mapped[str] = mapped_column(String(64), default="*")
    active_from: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    active_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    assigned_by: Mapped[str] = mapped_column(String(128), default="manual")
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class PromotionDecision(Base):
    __tablename__ = "promotion_decisions"
    __table_args__ = (Index("ix_promotion_decisions_model_time", "model_version_id", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"), index=True)
    previous_model_version_id: Mapped[int | None] = mapped_column(ForeignKey("model_versions.id"), nullable=True, index=True)
    action: Mapped[str] = mapped_column(String(32), index=True)
    approved: Mapped[bool] = mapped_column(default=False)
    decided_by: Mapped[str] = mapped_column(String(128), default="manual")
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)


class ShadowPrediction(Base):
    __tablename__ = "shadow_predictions"
    __table_args__ = (
        Index("ix_shadow_predictions_model_symbol_time", "model_version_id", "symbol", "generated_at"),
        Index("ix_shadow_predictions_trace", "decision_trace_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"), index=True)
    prediction_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    decision_trace_id: Mapped[str] = mapped_column(String(64), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class PaperSandboxAccount(Base):
    __tablename__ = "paper_sandbox_accounts"
    __table_args__ = (UniqueConstraint("account_id", name="uq_paper_sandbox_accounts_account_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[str] = mapped_column(String(128), index=True)
    name: Mapped[str] = mapped_column(String(128))
    model_version_id: Mapped[int | None] = mapped_column(ForeignKey("model_versions.id"), nullable=True, index=True)
    starting_balance: Mapped[float] = mapped_column(Float, default=0.0)
    max_exposure_pct: Mapped[float] = mapped_column(Float, default=0.05)
    active: Mapped[bool] = mapped_column(default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class StructuredNewsEvent(Base):
    __tablename__ = "structured_news_events"
    __table_args__ = (
        Index("ix_structured_news_events_symbol_time", "primary_symbol", "available_to_model_time"),
        Index("ix_structured_news_events_article", "article_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    article_id: Mapped[int | None] = mapped_column(ForeignKey("news_articles.id"), nullable=True, index=True)
    primary_symbol: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    event_type: Mapped[str] = mapped_column(String(64), index=True)
    affected_assets: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    affected_entities: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    direction: Mapped[str | None] = mapped_column(String(16), nullable=True)
    sentiment: Mapped[float | None] = mapped_column(Float, nullable=True)
    severity: Mapped[float | None] = mapped_column(Float, nullable=True)
    importance: Mapped[float | None] = mapped_column(Float, nullable=True)
    novelty: Mapped[float | None] = mapped_column(Float, nullable=True)
    time_horizon: Mapped[str | None] = mapped_column(String(32), nullable=True)
    factual_claims: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    source_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    provider: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    validation_status: Mapped[str] = mapped_column(String(32), default="VALID", index=True)
    published_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    received_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    processed_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    available_to_model_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class ExternalAIRequest(Base):
    __tablename__ = "external_ai_requests"
    __table_args__ = (
        Index("ix_external_ai_requests_hash_prompt", "content_hash", "prompt_version"),
        Index("ix_external_ai_requests_provider_time", "provider", "requested_at"),
        Index("ix_external_ai_requests_status_time", "status", "requested_at"),
        Index("ix_external_ai_requests_symbol_time", "symbol", "requested_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    provider: Mapped[str] = mapped_column(String(128), index=True)
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    content_hash: Mapped[str] = mapped_column(String(128), index=True)
    prompt_version: Mapped[str] = mapped_column(String(128), default="v1", index=True)
    requested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="PENDING", index=True)
    token_usage: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    estimated_cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    error_category: Mapped[str | None] = mapped_column(String(64), nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    cache_hit: Mapped[bool] = mapped_column(default=False)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class ModelHealthSnapshot(Base):
    __tablename__ = "model_health_snapshots"
    __table_args__ = (Index("ix_model_health_model_time", "model_version_id", "observed_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"), index=True)
    health_status: Mapped[str] = mapped_column(String(32), default="HEALTHY", index=True)
    rolling_information_coefficient: Mapped[float | None] = mapped_column(Float, nullable=True)
    rolling_net_expectancy: Mapped[float | None] = mapped_column(Float, nullable=True)
    calibration_error: Mapped[float | None] = mapped_column(Float, nullable=True)
    prediction_drift: Mapped[float | None] = mapped_column(Float, nullable=True)
    feature_drift: Mapped[float | None] = mapped_column(Float, nullable=True)
    ood_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    missing_feature_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    live_shadow_divergence: Mapped[float | None] = mapped_column(Float, nullable=True)
    transaction_cost_increase: Mapped[float | None] = mapped_column(Float, nullable=True)
    signal_correlation_increase: Mapped[float | None] = mapped_column(Float, nullable=True)
    regime_dependence: Mapped[float | None] = mapped_column(Float, nullable=True)
    capacity_decline: Mapped[float | None] = mapped_column(Float, nullable=True)
    consecutive_errors: Mapped[int] = mapped_column(Integer, default=0)
    recommended_weight_multiplier: Mapped[float] = mapped_column(Float, default=1.0)
    recommended_action: Mapped[str] = mapped_column(String(64), default="NORMAL_WEIGHT")
    reason_codes: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class SignalHealthSnapshot(Base):
    __tablename__ = "signal_health_snapshots"
    __table_args__ = (
        Index("ix_signal_health_signal_time", "signal_family", "observed_at"),
        Index("ix_signal_health_symbol_family_time", "symbol", "signal_family", "observed_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    signal_family: Mapped[str] = mapped_column(String(128), index=True)
    symbol: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    health_status: Mapped[str] = mapped_column(String(32), default="HEALTHY", index=True)
    rolling_information_coefficient: Mapped[float | None] = mapped_column(Float, nullable=True)
    rolling_net_expectancy: Mapped[float | None] = mapped_column(Float, nullable=True)
    calibration_error: Mapped[float | None] = mapped_column(Float, nullable=True)
    prediction_drift: Mapped[float | None] = mapped_column(Float, nullable=True)
    feature_drift: Mapped[float | None] = mapped_column(Float, nullable=True)
    ood_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    missing_feature_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    live_shadow_divergence: Mapped[float | None] = mapped_column(Float, nullable=True)
    transaction_cost_increase: Mapped[float | None] = mapped_column(Float, nullable=True)
    correlation_increase: Mapped[float | None] = mapped_column(Float, nullable=True)
    regime_dependence: Mapped[float | None] = mapped_column(Float, nullable=True)
    capacity_decline: Mapped[float | None] = mapped_column(Float, nullable=True)
    consecutive_errors: Mapped[int] = mapped_column(Integer, default=0)
    recommended_weight_multiplier: Mapped[float] = mapped_column(Float, default=1.0)
    recommended_action: Mapped[str] = mapped_column(String(64), default="NORMAL_WEIGHT")
    reason_codes: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class ExperimentRun(Base):
    __tablename__ = "experiment_runs"
    __table_args__ = (UniqueConstraint("experiment_id", name="uq_experiment_runs_experiment_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    experiment_id: Mapped[str] = mapped_column(String(64), index=True)
    code_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    configuration: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    dataset_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    feature_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    random_seed: Mapped[int] = mapped_column(Integer, default=0)
    train_period: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    validation_period: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    test_period: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    artifacts: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="CREATED", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class DecisionTimelineEvent(Base):
    __tablename__ = "decision_timeline_events"
    __table_args__ = (Index("ix_decision_timeline_trace_time", "decision_trace_id", "occurred_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    decision_trace_id: Mapped[str] = mapped_column(String(64), index=True)
    stage: Mapped[str] = mapped_column(String(64), index=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    status: Mapped[str] = mapped_column(String(32), default="RECORDED")
    reason_codes: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class RiskControlState(Base):
    __tablename__ = "risk_control_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    enabled: Mapped[bool] = mapped_column(default=False, index=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_by: Mapped[str] = mapped_column(String(128), default="system")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)
