from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from app.strategies.regime_pullback_v1 import (
    CandleBar,
    DecisionAction,
    DerivativesSnapshot,
    FeatureSnapshot,
    PaperLedger,
    PositionState,
    ReasonCode,
    Regime,
    RiskContext,
    StrategyConfig,
    build_future_label,
    calculate_quantity,
    classify_regime,
    evaluate_entry,
    evaluate_exit,
    gross_pnl,
    latest_completed_close,
    paper_fill_price,
    point_in_time_bars,
    round_down_quantity,
)

UTC = timezone.utc
T0 = datetime(2026, 8, 6, 0, 0, tzinfo=UTC)


def feature(regime: Regime = Regime.LONG, **changes) -> FeatureSnapshot:
    long = regime != Regime.SHORT
    base = FeatureSnapshot(
        symbol="BTCUSDT",
        decision_time=T0 + timedelta(minutes=16),
        candle_close_time=T0 + timedelta(minutes=15),
        feature_time=T0 + timedelta(minutes=15, seconds=2),
        market_data_time=T0 + timedelta(minutes=15),
        source_timestamps={"decision_15m": (T0 + timedelta(minutes=15)).isoformat()},
        data_fresh=True,
        data_complete=True,
        candle_complete=True,
        missing_intervals=False,
        close=101.0 if long else 99.0,
        high=102.0 if long else 101.0,
        low=99.0 if long else 98.0,
        volume=120.0,
        previous_high=100.5 if long else 101.0,
        previous_low=99.0 if long else 99.5,
        previous_close=100.0,
        ema20=100.0,
        current_low=99.0 if long else 98.0,
        current_high=102.0 if long else 101.0,
        previous_low_15m=99.5 if long else 98.5,
        previous_high_15m=100.5 if long else 100.5,
        rsi14=52.0 if long else 48.0,
        previous_rsi14=49.0 if long else 51.0,
        atr15m=1.0,
        median_volume20=100.0,
        regime=regime,
        ema50_1h=110.0 if long else 90.0,
        ema200_1h=100.0,
        ema50_1h_three_bars_ago=109.0 if long else 91.0,
        adx14_1h=30.0,
        close_1h=111.0 if long else 89.0,
        bid=100.98 if long else 98.98,
        ask=101.02 if long else 99.02,
        spread_bps=4.0,
        spread_estimated=False,
        spread_method="measured_bid_ask",
        derivatives=DerivativesSnapshot(),
    )
    return replace(base, **changes)


def context(**changes) -> RiskContext:
    base = RiskContext(
        paper_equity=Decimal("10000"),
        paper_cash=Decimal("10000"),
        starting_day_equity=Decimal("10000"),
    )
    return replace(base, **changes)


def position(side: str = "LONG", **changes) -> PositionState:
    long = side == "LONG"
    base = PositionState(
        position_id="p1",
        symbol="BTCUSDT",
        side=side,
        quantity=Decimal("1"),
        entry_price=Decimal("100"),
        initial_stop_distance=Decimal("1.5"),
        stop_price=Decimal("98.5") if long else Decimal("101.5"),
        target_price=Decimal("102.5") if long else Decimal("97.5"),
        atr_at_entry=Decimal("1"),
        opened_at=T0,
        entry_fee=Decimal("0.04"),
    )
    for key, value in changes.items():
        setattr(base, key, value)
    return base


def bar(open_minute: int, close: float = 100.0, *, complete: bool = True) -> CandleBar:
    start = T0 + timedelta(minutes=open_minute)
    return CandleBar(start, start + timedelta(minutes=15), close, close + 1, close - 1, close, 100, complete=complete)


# 1
def test_completed_candle_only_decisions():
    now = T0 + timedelta(minutes=29, seconds=59)
    assert latest_completed_close(now, 15) == T0 + timedelta(minutes=15)


# 2
def test_no_look_ahead_leakage():
    bars = [bar(0), bar(15), bar(30)]
    assert [b.open_time for b in point_in_time_bars(bars, T0 + timedelta(minutes=30))] == [T0, T0 + timedelta(minutes=15)]


# 3
def test_long_regime_classification():
    assert classify_regime(close_1h=110, ema50_1h=105, ema200_1h=100, ema50_three_bars_ago=104, adx14_1h=20) == Regime.LONG


# 4
def test_short_regime_classification():
    assert classify_regime(close_1h=90, ema50_1h=95, ema200_1h=100, ema50_three_bars_ago=96, adx14_1h=30) == Regime.SHORT


# 5
def test_neutral_regime_classification():
    assert classify_regime(close_1h=110, ema50_1h=105, ema200_1h=100, ema50_three_bars_ago=104, adx14_1h=19.9) == Regime.NEUTRAL


# 6
def test_valid_long_entry():
    result = evaluate_entry(feature(Regime.LONG), context())
    assert result.action == DecisionAction.ENTER_LONG and result.risk_approved


# 7
def test_valid_short_entry():
    result = evaluate_entry(feature(Regime.SHORT), context())
    assert result.action == DecisionAction.ENTER_SHORT and result.risk_approved


# 8
def test_low_confidence_rejection():
    result = evaluate_entry(feature(), context(), StrategyConfig(min_confidence=0.99))
    assert ReasonCode.CONFIDENCE_TOO_LOW in result.reason_codes


# 9
def test_stale_data_rejection():
    result = evaluate_entry(feature(data_fresh=False), context())
    assert ReasonCode.DATA_STALE in result.reason_codes


# 10
def test_wide_spread_rejection():
    result = evaluate_entry(feature(spread_bps=8.1), context())
    assert ReasonCode.SPREAD_TOO_WIDE in result.reason_codes


# 11
def test_atr_filter_rejection():
    low = evaluate_entry(feature(atr15m=0.1), context())
    high = evaluate_entry(feature(atr15m=4.0), context())
    assert ReasonCode.ATR_TOO_LOW in low.reason_codes
    assert ReasonCode.ATR_TOO_HIGH in high.reason_codes


# 12
def test_one_position_per_symbol():
    result = evaluate_entry(feature(), context(symbol_has_position=True))
    assert ReasonCode.POSITION_ALREADY_OPEN in result.reason_codes


# 13
def test_maximum_two_portfolio_positions():
    result = evaluate_entry(feature(), context(open_positions=2))
    assert ReasonCode.MAX_OPEN_POSITIONS in result.reason_codes


# 14
def test_maximum_portfolio_exposure():
    result = evaluate_entry(feature(), context(current_total_open_notional=Decimal("2000")))
    assert ReasonCode.MAX_EXPOSURE in result.reason_codes


# 15
def test_risk_based_quantity_calculation():
    q = calculate_quantity(entry_price=Decimal("100"), atr_value=Decimal("1"), context=context(), config=StrategyConfig())
    assert q.risk_budget == Decimal("25.0000")
    assert q.requested_quantity == Decimal("25.0000") / Decimal("1.5")
    assert q.approved_quantity <= Decimal("10")


# 16
def test_quantity_rounds_down():
    assert round_down_quantity(Decimal("1.239"), Decimal("0.01")) == Decimal("1.23")


# 17
def test_1x_leverage_enforcement():
    result = evaluate_entry(feature(), context(leverage_requested=Decimal("2")))
    assert ReasonCode.PAPER_ONLY_REQUIRED in result.reason_codes


# 18
def test_cooldown_enforcement():
    result = evaluate_entry(feature(), context(cooldown_active=True))
    assert ReasonCode.COOLDOWN_ACTIVE in result.reason_codes


# 19
def test_per_symbol_daily_trade_limit():
    result = evaluate_entry(feature(), context(symbol_trades_today=3))
    assert ReasonCode.DAILY_TRADE_LIMIT in result.reason_codes


# 20
def test_portfolio_daily_trade_limit():
    result = evaluate_entry(feature(), context(portfolio_trades_today=8))
    assert ReasonCode.PORTFOLIO_DAILY_TRADE_LIMIT in result.reason_codes


# 21
def test_daily_loss_circuit_breaker():
    result = evaluate_entry(feature(), context(day_net_pnl=Decimal("-100")))
    assert ReasonCode.DAILY_LOSS_LIMIT in result.reason_codes


# 22
def test_consecutive_loss_circuit_breaker():
    result = evaluate_entry(feature(), context(consecutive_losses=3))
    assert ReasonCode.CONSECUTIVE_LOSS_LIMIT in result.reason_codes


# 23
def test_no_averaging_down():
    ledger = PaperLedger()
    ledger.open_position(symbol="BTCUSDT", side="LONG", quantity=Decimal("1"), fill_price=Decimal("100"), order_id="o1", fill_id="f1", position_id="p1", stop_price=Decimal("98.5"), target_price=Decimal("102.5"), atr_value=Decimal("1"))
    with pytest.raises(ValueError, match=ReasonCode.POSITION_ALREADY_OPEN.value):
        ledger.open_position(symbol="BTCUSDT", side="LONG", quantity=Decimal("1"), fill_price=Decimal("99"), order_id="o2", fill_id="f2", position_id="p2", stop_price=Decimal("97.5"), target_price=Decimal("101.5"), atr_value=Decimal("1"))


# 24
def test_stop_loss():
    result = evaluate_exit(position(), feature(low=98.4, high=100.5, close=99.0))
    assert result.should_exit and result.reason_code == ReasonCode.STOP_LOSS


# 25
def test_take_profit():
    result = evaluate_exit(position(), feature(low=99.5, high=102.6, close=102.0))
    assert result.should_exit and result.reason_code == ReasonCode.TAKE_PROFIT


# 26
def test_break_even_movement():
    result = evaluate_exit(position(), feature(low=99.8, high=101.6, close=101.2))
    assert not result.should_exit and result.break_even_active and result.updated_stop_price > Decimal("100")


# 27
def test_trailing_stop():
    result = evaluate_exit(position(), feature(low=99.8, high=102.3, close=102.0))
    assert not result.should_exit and result.trailing_active and result.updated_stop_price >= Decimal("101")


# 28
def test_time_exit():
    result = evaluate_exit(position(bars_held=16), feature(low=99.8, high=100.2, close=100.0))
    assert result.should_exit and result.reason_code == ReasonCode.TIME_EXIT


# 29
def test_regime_flip_exit():
    result = evaluate_exit(position(), feature(Regime.SHORT, low=99.8, high=100.2, close=100.0))
    assert result.should_exit and result.reason_code == ReasonCode.REGIME_FLIP


# 30
def test_correct_long_pnl():
    assert gross_pnl("LONG", Decimal("100"), Decimal("110"), Decimal("2")) == Decimal("20")


# 31
def test_correct_short_pnl():
    assert gross_pnl("SHORT", Decimal("100"), Decimal("90"), Decimal("2")) == Decimal("20")


# 32
def test_fees_applied_on_both_sides():
    ledger = PaperLedger()
    opened = ledger.open_position(symbol="BTCUSDT", side="LONG", quantity=Decimal("1"), fill_price=Decimal("100"), order_id="o1", fill_id="f1", position_id="p1", stop_price=Decimal("98.5"), target_price=Decimal("102.5"), atr_value=Decimal("1"))
    closed = ledger.close_position(symbol="BTCUSDT", exit_price=Decimal("101"), order_id="o2", fill_id="f2", close_event_id="c1")
    assert ledger.total_fees == opened.fee + closed.fee


# 33
def test_slippage_is_side_aware():
    config = StrategyConfig(slippage_rate_per_side=Decimal("0.001"))
    assert paper_fill_price(Decimal("100"), "BUY", config) == Decimal("100.100")
    assert paper_fill_price(Decimal("100"), "SELL", config) == Decimal("99.900")


# 34
def test_opening_trades_do_not_create_realized_pnl():
    ledger = PaperLedger()
    trade = ledger.open_position(symbol="BTCUSDT", side="LONG", quantity=Decimal("1"), fill_price=Decimal("100"), order_id="o1", fill_id="f1", position_id="p1", stop_price=Decimal("98.5"), target_price=Decimal("102.5"), atr_value=Decimal("1"))
    assert trade.realized_pnl == 0 and ledger.realized_trading_pnl == 0 and trade.fee > 0


# 35
def test_duplicate_decisions_are_prevented():
    result = evaluate_entry(feature(), context(duplicate_decision=True))
    assert ReasonCode.DUPLICATE_DECISION in result.reason_codes


# 36
def test_duplicate_fills_are_prevented():
    ledger = PaperLedger()
    ledger.open_position(symbol="BTCUSDT", side="LONG", quantity=Decimal("1"), fill_price=Decimal("100"), order_id="o1", fill_id="f1", position_id="p1", stop_price=Decimal("98.5"), target_price=Decimal("102.5"), atr_value=Decimal("1"))
    with pytest.raises(ValueError, match=ReasonCode.DUPLICATE_FILL.value):
        ledger.close_position(symbol="BTCUSDT", exit_price=Decimal("101"), order_id="o2", fill_id="f1", close_event_id="c1")


# 37
def test_restart_recovery():
    ledger = PaperLedger()
    ledger.open_position(symbol="BTCUSDT", side="SHORT", quantity=Decimal("2"), fill_price=Decimal("100"), order_id="o1", fill_id="f1", position_id="p1", stop_price=Decimal("101.5"), target_price=Decimal("97.5"), atr_value=Decimal("1"))
    restored = PaperLedger.restore(ledger.snapshot())
    assert restored.positions["BTCUSDT"].side == "SHORT" and restored.orders == {"o1"}


# 38
def test_missing_optional_derivatives_data():
    result = evaluate_entry(feature(derivatives=DerivativesSnapshot()), context())
    assert result.risk_approved and result.confidence_components.derivatives_confirmation == 0.0


# 39
def test_rejected_shadow_decision_logging():
    result = evaluate_entry(feature(), context(cooldown_active=True))
    assert result.strategy_conditions_passed and result.shadow_decision and not result.risk_approved


# 40
def test_future_labels_unavailable_before_horizon():
    decision = T0 + timedelta(minutes=15)
    future = [CandleBar(decision, decision + timedelta(minutes=15), 100, 101, 99, 100.5, 10)]
    label = build_future_label(decision_close=decision, decision_price=100, direction="LONG", future_bars=future, horizon_minutes=60, stop_price=98.5, target_price=102.5, estimated_round_trip_cost_rate=0.0012)
    assert not label.available and not label.data_complete
