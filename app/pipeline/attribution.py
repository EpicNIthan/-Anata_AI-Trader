"""Evidence-based paper-PnL attribution for the Anata V2 ledger.

The module is intentionally read-only.  It assembles a bounded view over the paper
ledger and its V2 trace records; it never writes inferred attribution back to the
database.  Recorded values are distinguished from estimates and unavailable
counterfactuals.  Every selected PnL event is returned in ``trades`` even when its
legacy lineage is incomplete, and the unexplained residual keeps the decomposition
additive without manufacturing precision.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.db.models import (
    EnsembleDecisionRecord,
    EnsembleSignalWeight,
    Feature,
    ModelPredictionRecord,
    PaperTrade,
    PortfolioTargetRecord,
    RiskDecisionRecord,
    SimulatedFillRecord,
    SimulatedOrderRecord,
    TradingSignalRecord,
)
from app.features.schema import values_from_feature


ATTRIBUTION_METHOD_VERSION = "paper-pnl-attribution-v2"
MAX_ATTRIBUTION_ROWS = 10_000
SUPPORTED_TIME_PERIODS = frozenset({"hour", "day", "week", "month"})

_COMPONENT_KEYS = (
    "alpha_contribution",
    "ensemble_contribution",
    "position_sizing_contribution",
    "broad_market_exposure",
    "execution_contribution",
    "fees",
    "slippage",
    "funding",
)


def _finite_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _finite(value: Any) -> float:
    parsed = _finite_or_none(value)
    return parsed if parsed is not None else 0.0


def _utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _period_key(value: datetime | None, period: str) -> str:
    if value is None:
        return "unknown"
    at = _utc(value) or value.replace(tzinfo=timezone.utc)
    if period == "hour":
        return at.strftime("%Y-%m-%dT%H:00:00Z")
    if period == "week":
        monday = at.date() - timedelta(days=at.weekday())
        return f"{monday.isoformat()}/week"
    if period == "month":
        return at.strftime("%Y-%m")
    return at.strftime("%Y-%m-%d")


def _payload(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text(value: Any, *, maximum: int = 200) -> str | None:
    if value in (None, ""):
        return None
    return str(value)[:maximum]


def _explicit_attribution(trade: PaperTrade) -> dict[str, Any]:
    raw = _payload(trade.raw_payload)
    attribution = raw.get("attribution")
    return attribution if isinstance(attribution, dict) else {}


def _explicit_component(
    attribution: Mapping[str, Any],
    name: str,
) -> tuple[float | None, str | None]:
    if name not in attribution:
        return None, None
    value = _finite_or_none(attribution.get(name))
    methods = attribution.get("methods")
    method = methods.get(name) if isinstance(methods, dict) else None
    return value, _text(method) or "persisted_explicit_attribution"


def _component_detail(
    value: float | None,
    *,
    method: str,
    evidence: Iterable[str] = (),
    estimated_value: float | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    return {
        "value": value,
        # Numeric compatibility/reconciliation uses zero only as an unassigned
        # amount. ``value`` remains null when the counterfactual is unavailable.
        "reconciliation_value": value if value is not None else 0.0,
        "method": method,
        "coverage": "available" if value is not None else "unavailable",
        "estimated_value": estimated_value,
        "evidence": list(evidence),
        "note": note,
    }


def _provider_from_prediction(
    prediction: ModelPredictionRecord,
    feature: Feature | None,
) -> str | None:
    payload = _payload(prediction.payload)
    candidates: list[Any] = [
        payload.get("external_ai_provider"),
        _payload(payload.get("external_context")).get("external_ai_provider"),
    ]
    if feature is not None:
        try:
            candidates.append(values_from_feature(feature).get("external_ai_provider"))
        except (KeyError, TypeError, ValueError):
            # A historic/unknown schema is evidence of missing provider lineage, not
            # permission to infer a provider from a nearby request timestamp.
            pass
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        provider = _text(candidate, maximum=128)
        if provider:
            return provider
    return None


def _portfolio_bucket(target: PortfolioTargetRecord | None) -> str:
    if target is None:
        return "unattributed"
    exposure = _finite_or_none(target.requested_target_exposure)
    if exposure is None:
        return "unattributed"
    absolute = abs(exposure)
    if absolute <= 1e-12:
        return "flat"
    if absolute <= 0.05:
        return "0-5pct"
    if absolute <= 0.10:
        return "5-10pct"
    if absolute <= 0.25:
        return "10-25pct"
    if absolute <= 0.50:
        return "25-50pct"
    return "over-50pct"


def _risk_resize(risk: RiskDecisionRecord | None) -> tuple[str, float | None]:
    if risk is None:
        return "unattributed", None
    requested = _finite_or_none(risk.requested_exposure)
    approved = _finite_or_none(risk.approved_exposure)
    if requested is None or approved is None:
        return "unattributed", None
    tolerance = 1e-12
    ratio = None if abs(requested) <= tolerance else approved / requested
    if math.isclose(requested, approved, rel_tol=1e-9, abs_tol=tolerance):
        return "unchanged", ratio
    if requested * approved < 0:
        return "direction_changed", ratio
    if abs(approved) < abs(requested):
        return "reduced", ratio
    if abs(approved) > abs(requested):
        return "increased", ratio
    return "changed", ratio


def _query_rows(session: Session, model: Any, identifiers: set[Any], column: Any) -> list[Any]:
    if not identifiers:
        return []
    # Stay below conservative SQLite bind-variable limits even when the API's
    # bounded trade window fans out to several model signals per decision.
    values = list(identifiers)
    rows: list[Any] = []
    for offset in range(0, len(values), 500):
        rows.extend(session.scalars(select(model).where(column.in_(values[offset : offset + 500]))))
    return rows


def _load_evidence(session: Session, trades: list[PaperTrade]) -> dict[str, Any]:
    """Load all linked evidence in bounded set queries rather than per-trade queries."""
    order_ids = {row.simulated_order_id for row in trades if row.simulated_order_id}
    orders = _query_rows(session, SimulatedOrderRecord, order_ids, SimulatedOrderRecord.order_id)
    orders_by_id = {row.order_id: row for row in orders}

    risk_ids = {row.risk_decision_id for row in trades if row.risk_decision_id}
    risk_ids.update(row.risk_decision_id for row in orders if row.risk_decision_id)
    risks = _query_rows(session, RiskDecisionRecord, risk_ids, RiskDecisionRecord.risk_decision_id)
    risks_by_id = {row.risk_decision_id: row for row in risks}

    target_ids = {row.portfolio_target_id for row in orders if row.portfolio_target_id}
    target_ids.update(row.portfolio_target_id for row in risks if row.portfolio_target_id)
    targets = _query_rows(session, PortfolioTargetRecord, target_ids, PortfolioTargetRecord.portfolio_target_id)
    targets_by_id = {row.portfolio_target_id: row for row in targets}

    trace_ids = {row.decision_trace_id for row in trades if row.decision_trace_id}
    trace_ids.update(row.decision_trace_id for row in orders if row.decision_trace_id)
    ensemble_ids = {row.source_ensemble_decision_id for row in targets if row.source_ensemble_decision_id}
    ensemble_rows = _query_rows(
        session,
        EnsembleDecisionRecord,
        ensemble_ids,
        EnsembleDecisionRecord.ensemble_decision_id,
    )
    ensemble_rows.extend(
        _query_rows(
            session,
            EnsembleDecisionRecord,
            trace_ids,
            EnsembleDecisionRecord.decision_trace_id,
        )
    )
    # Direct-ID and trace lookups commonly return the same record.
    ensembles = list({row.id: row for row in ensemble_rows}.values())
    ensembles_by_id = {row.ensemble_decision_id: row for row in ensembles}
    ensembles_by_trace: dict[str, EnsembleDecisionRecord] = {}
    for row in sorted(ensembles, key=lambda item: (_utc(item.generated_at) or datetime.min.replace(tzinfo=timezone.utc), item.id), reverse=True):
        ensembles_by_trace.setdefault(row.decision_trace_id, row)

    all_ensemble_ids = set(ensembles_by_id)
    weights = _query_rows(
        session,
        EnsembleSignalWeight,
        all_ensemble_ids,
        EnsembleSignalWeight.ensemble_decision_id,
    )
    weights_by_ensemble: dict[str, list[EnsembleSignalWeight]] = defaultdict(list)
    for row in weights:
        weights_by_ensemble[row.ensemble_decision_id].append(row)
    for rows in weights_by_ensemble.values():
        rows.sort(key=lambda item: (-_finite(item.weight), item.signal_id))

    signal_ids = {row.signal_id for row in weights if row.signal_id}
    signals = _query_rows(session, TradingSignalRecord, signal_ids, TradingSignalRecord.signal_id)
    signals_by_id = {row.signal_id: row for row in signals}
    prediction_ids = {row.prediction_id for row in signals if row.prediction_id}
    predictions = _query_rows(
        session,
        ModelPredictionRecord,
        prediction_ids,
        ModelPredictionRecord.prediction_id,
    )
    predictions_by_id = {row.prediction_id: row for row in predictions}
    feature_ids = {row.feature_id for row in predictions if row.feature_id is not None}
    features = _query_rows(session, Feature, feature_ids, Feature.id)
    features_by_id = {row.id: row for row in features}

    fills = _query_rows(session, SimulatedFillRecord, order_ids, SimulatedFillRecord.order_id)
    fills_by_order: dict[str, list[SimulatedFillRecord]] = defaultdict(list)
    for row in fills:
        fills_by_order[row.order_id].append(row)
    for rows in fills_by_order.values():
        rows.sort(key=lambda item: (_utc(item.filled_at) or datetime.min.replace(tzinfo=timezone.utc), item.id))

    return {
        "orders": orders_by_id,
        "risks": risks_by_id,
        "targets": targets_by_id,
        "ensembles": ensembles_by_id,
        "ensembles_by_trace": ensembles_by_trace,
        "weights": weights_by_ensemble,
        "signals": signals_by_id,
        "predictions": predictions_by_id,
        "features": features_by_id,
        "fills": fills_by_order,
    }


def _trade_links(trade: PaperTrade, evidence: Mapping[str, Any]) -> dict[str, Any]:
    order = evidence["orders"].get(trade.simulated_order_id)
    risk_id = trade.risk_decision_id or (order.risk_decision_id if order else None)
    risk = evidence["risks"].get(risk_id)
    target_id = (order.portfolio_target_id if order else None) or (risk.portfolio_target_id if risk else None)
    target = evidence["targets"].get(target_id)
    trace_id = trade.decision_trace_id or (order.decision_trace_id if order else None) or (
        risk.decision_trace_id if risk else None
    )
    ensemble = evidence["ensembles"].get(target.source_ensemble_decision_id) if target else None
    if ensemble is None and trace_id:
        ensemble = evidence["ensembles_by_trace"].get(trace_id)
    fills = list(evidence["fills"].get(trade.simulated_order_id, ()))
    weights = list(evidence["weights"].get(ensemble.ensemble_decision_id, ())) if ensemble else []
    return {
        "order": order,
        "risk": risk,
        "target": target,
        "trace_id": trace_id,
        "ensemble": ensemble,
        "fills": fills,
        "weights": weights,
    }


def _funding_cash_flow(fills: list[SimulatedFillRecord]) -> tuple[float | None, float | None, list[str]]:
    """Return (booked cash flow, unbooked estimate, evidence labels)."""
    booked_values: list[float] = []
    estimates: list[float] = []
    evidence: list[str] = []
    for fill in fills:
        payload = _payload(fill.payload)
        cash_flow = _finite_or_none(payload.get("funding_cash_flow"))
        if cash_flow is not None:
            booked_values.append(cash_flow)
            evidence.append(f"simulated_fill:{fill.fill_id}:funding_cash_flow")
            continue
        funding = _finite_or_none(fill.funding)
        if funding is not None:
            # Legacy fills persisted only an unsigned estimate. New fills carry the
            # funding_cash_flow branch above and are posted to paper cash/PnL.
            estimates.append(-abs(funding))
            evidence.append(f"simulated_fill:{fill.fill_id}:unbooked_funding_estimate")
    return (sum(booked_values) if booked_values else None, sum(estimates) if estimates else None, evidence)


def _trade_decomposition(
    trade: PaperTrade,
    fills: list[SimulatedFillRecord],
) -> tuple[dict[str, float], dict[str, dict[str, Any]], dict[str, Any]]:
    total_value = _finite_or_none(trade.realized_pnl)
    total = total_value if total_value is not None else 0.0
    raw = _payload(trade.raw_payload)
    attribution = _explicit_attribution(trade)
    details: dict[str, dict[str, Any]] = {}

    # Fees are a booked PaperTrade cash-flow field, not a fill-derived estimate.
    fee_value = _finite_or_none(trade.fee)
    fee_component = -fee_value if fee_value is not None else None
    details["fees"] = _component_detail(
        fee_component,
        method="recorded_paper_trade_fee" if fee_component is not None else "unavailable",
        evidence=[f"paper_trade:{trade.id}:fee"] if fee_component is not None else [],
    )

    explicit_slippage, slippage_method = _explicit_component(attribution, "slippage")
    slippage_cost = sum(
        abs(_finite(row.slippage)) * abs(_finite(row.notional))
        for row in fills
        if _finite_or_none(row.slippage) is not None and _finite_or_none(row.notional) is not None
    )
    if explicit_slippage is not None:
        slippage_component = explicit_slippage
        details["slippage"] = _component_detail(
            slippage_component,
            method=slippage_method or "persisted_explicit_attribution",
            evidence=[f"paper_trade:{trade.id}:raw_payload.attribution.slippage"],
        )
    elif fills:
        slippage_component = -slippage_cost
        details["slippage"] = _component_detail(
            slippage_component,
            method="estimated_from_fill_mid_price_deviation",
            evidence=[f"simulated_fill:{row.fill_id}:slippage*notional" for row in fills],
            note="The simulator stores absolute mid-price deviation; this is an execution-cost estimate embedded in fill price.",
        )
    else:
        slippage_component = None
        details["slippage"] = _component_detail(
            None,
            method="unavailable_no_linked_simulated_fill",
        )

    explicit_funding, funding_method = _explicit_component(attribution, "funding")
    booked_funding, estimated_funding, funding_evidence = _funding_cash_flow(fills)
    if explicit_funding is not None:
        funding_component = explicit_funding
        details["funding"] = _component_detail(
            funding_component,
            method=funding_method or "persisted_explicit_attribution",
            evidence=[f"paper_trade:{trade.id}:raw_payload.attribution.funding"],
        )
    elif booked_funding is not None:
        funding_component = booked_funding
        details["funding"] = _component_detail(
            funding_component,
            method="recorded_fill_funding_cash_flow",
            evidence=funding_evidence,
        )
    else:
        funding_component = None
        details["funding"] = _component_detail(
            None,
            method="unbooked_estimate_only" if estimated_funding is not None else "unavailable",
            estimated_value=estimated_funding,
            evidence=funding_evidence,
            note="A simulated funding estimate is not treated as booked PnL unless an explicit cash flow is recorded.",
        )

    counterfactual_names = (
        "ensemble_contribution",
        "position_sizing_contribution",
        "broad_market_exposure",
        "execution_contribution",
    )
    any_explicit_counterfactual = False
    for name in counterfactual_names:
        value, method = _explicit_component(attribution, name)
        any_explicit_counterfactual = any_explicit_counterfactual or value is not None
        details[name] = _component_detail(
            value,
            method=method or "unavailable_no_recorded_counterfactual",
            evidence=[f"paper_trade:{trade.id}:raw_payload.attribution.{name}"] if value is not None else [],
            note=None if value is not None else "Unassigned to the reconciliation and retained in unexplained residual.",
        )

    explicit_alpha, alpha_method = _explicit_component(attribution, "alpha_contribution")
    gross = _finite_or_none(raw.get("gross_pnl"))
    if explicit_alpha is not None:
        alpha = explicit_alpha
        details["alpha_contribution"] = _component_detail(
            alpha,
            method=alpha_method or "persisted_explicit_attribution",
            evidence=[f"paper_trade:{trade.id}:raw_payload.attribution.alpha_contribution"],
        )
    elif gross is not None and not any_explicit_counterfactual:
        # Gross PnL uses the simulated fill price, so remove the stored adverse
        # slippage component to form the documented counterfactual alpha estimate.
        alpha = gross - (slippage_component or 0.0)
        details["alpha_contribution"] = _component_detail(
            alpha,
            method="recorded_gross_pnl_adjusted_for_embedded_slippage",
            evidence=[f"paper_trade:{trade.id}:raw_payload.gross_pnl"],
            note="This is a fill-price counterfactual estimate, not causal model alpha.",
        )
    else:
        alpha = None
        method = "unavailable_partial_explicit_decomposition" if any_explicit_counterfactual else "unavailable_no_recorded_gross_pnl"
        details["alpha_contribution"] = _component_detail(
            None,
            method=method,
            note="Unassigned to the reconciliation and retained in unexplained residual.",
        )

    components = {name: _finite(details[name]["reconciliation_value"]) for name in _COMPONENT_KEYS}
    residual = total - sum(components.values())
    components["unexplained_residual"] = residual
    details["unexplained_residual"] = _component_detail(
        residual,
        method="arithmetic_reconciliation_residual",
        evidence=[f"paper_trade:{trade.id}:realized_pnl"],
        note="Contains unavailable counterfactuals, incomplete legacy lineage, and any ledger/model mismatch.",
    )
    reconciliation = {
        "total_paper_pnl": total,
        "component_sum": sum(components.values()),
        "difference": total - sum(components.values()),
        "reconciled": math.isclose(total, sum(components.values()), rel_tol=1e-9, abs_tol=1e-9),
        "total_method": "recorded_paper_trade_realized_pnl" if total_value is not None else "invalid_or_unavailable_treated_as_zero",
    }
    return components, details, reconciliation


def _intent(trade: PaperTrade) -> tuple[str, str]:
    raw = _payload(trade.raw_payload)
    intent = _text(raw.get("intent"), maximum=32)
    if intent:
        normalized = intent.lower()
        if normalized == "close":
            return "closed", "recorded_raw_payload_intent"
        if normalized in {"open", "increase"}:
            return "open_or_increase", "recorded_raw_payload_intent"
        return normalized, "recorded_raw_payload_intent"
    if str(trade.action or "").upper() == "CLOSE":
        return "closed", "recorded_action"
    gross = _finite_or_none(raw.get("gross_pnl"))
    if gross is not None and abs(gross) > 1e-12:
        return "closed_inferred", "nonzero_recorded_gross_pnl"
    return "unknown_legacy", "unavailable"


def _coverage_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "decision_trace",
        "model_signal_lineage",
        "ensemble_weighting",
        "portfolio_sizing",
        "risk_resizing",
        "execution_fill",
        "fees",
        "funding_status",
        "external_ai_status",
    )
    total = len(records)
    absolute_pnl = sum(abs(_finite(row["total_paper_pnl"])) for row in records)
    output: dict[str, Any] = {}
    for field in fields:
        covered = sum(1 for row in records if row["coverage_flags"].get(field, False))
        covered_pnl = sum(
            abs(_finite(row["total_paper_pnl"]))
            for row in records
            if row["coverage_flags"].get(field, False)
        )
        output[field] = {
            "covered_trade_count": covered,
            "trade_count": total,
            "trade_ratio": covered / total if total else None,
            "absolute_pnl_ratio": covered_pnl / absolute_pnl if absolute_pnl else None,
        }
    return output


def paper_pnl_attribution(
    session: Session,
    *,
    symbol: str | None = None,
    account_id: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    limit: int = 2_000,
    time_period: str = "day",
) -> dict[str, Any]:
    """Return bounded, reconciled, trace-based paper-PnL attribution.

    ``components`` remains a numeric compatibility view.  Read
    ``component_metadata`` before interpreting a zero: unavailable components have a
    null ``value`` and a zero ``reconciliation_value`` so they are never asserted to
    be genuine zero contribution.
    """
    normalized_period = str(time_period or "day").lower()
    if normalized_period not in SUPPORTED_TIME_PERIODS:
        raise ValueError(f"time_period must be one of: {', '.join(sorted(SUPPORTED_TIME_PERIODS))}")
    safe_limit = min(max(int(limit), 1), MAX_ATTRIBUTION_ROWS)
    start_utc = _utc(start)
    end_utc = _utc(end)
    if start_utc and end_utc and end_utc < start_utc:
        raise ValueError("end must be greater than or equal to start")

    statement = select(PaperTrade).order_by(desc(PaperTrade.created_at), desc(PaperTrade.id)).limit(safe_limit)
    if symbol:
        statement = statement.where(PaperTrade.symbol == symbol.upper())
    if account_id:
        statement = statement.where(PaperTrade.paper_account_id == account_id)
    if start_utc:
        statement = statement.where(PaperTrade.created_at >= start_utc)
    if end_utc:
        statement = statement.where(PaperTrade.created_at <= end_utc)
    trades = list(session.scalars(statement))
    evidence = _load_evidence(session, trades)

    aggregate_components = {name: 0.0 for name in (*_COMPONENT_KEYS, "unexplained_residual")}
    aggregate_component_methods: dict[str, set[str]] = defaultdict(set)
    aggregate_component_available: dict[str, int] = defaultdict(int)
    aggregate_estimates: dict[str, float] = defaultdict(float)
    by_model: dict[str, float] = defaultdict(float)
    by_signal: dict[str, float] = defaultdict(float)
    by_family: dict[str, float] = defaultdict(float)
    by_symbol: dict[str, float] = defaultdict(float)
    by_regime: dict[str, float] = defaultdict(float)
    by_external: dict[str, float] = defaultdict(float)
    by_provider: dict[str, float] = defaultdict(float)
    by_ensemble: dict[str, float] = defaultdict(float)
    by_portfolio_sizing: dict[str, float] = defaultdict(float)
    by_risk_resizing: dict[str, float] = defaultdict(float)
    by_period: dict[str, float] = defaultdict(float)
    ensemble_profiles: dict[str, dict[str, Any]] = {}
    portfolio_evidence: dict[str, dict[str, Any]] = {}
    risk_evidence: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []

    for trade in trades:
        links = _trade_links(trade, evidence)
        order = links["order"]
        risk = links["risk"]
        target = links["target"]
        ensemble = links["ensemble"]
        fills = links["fills"]
        weights = links["weights"]
        components, component_details, reconciliation = _trade_decomposition(trade, fills)
        total = reconciliation["total_paper_pnl"]
        for name, value in components.items():
            aggregate_components[name] += value
            aggregate_component_methods[name].add(component_details[name]["method"])
            if component_details[name]["value"] is not None:
                aggregate_component_available[name] += 1
            estimate = _finite_or_none(component_details[name].get("estimated_value"))
            if estimate is not None:
                aggregate_estimates[name] += estimate

        intent, intent_method = _intent(trade)
        regime = (ensemble.current_regime if ensemble else None) or "unknown"
        ensemble_id = ensemble.ensemble_decision_id if ensemble else "unattributed"
        sizing_bucket = _portfolio_bucket(target)
        resize_category, resize_ratio = _risk_resize(risk)
        period_key = _period_key(trade.created_at, normalized_period)
        by_symbol[trade.symbol] += total
        by_regime[regime] += total
        by_ensemble[ensemble_id] += total
        by_portfolio_sizing[sizing_bucket] += total
        by_risk_resizing[resize_category] += total
        by_period[period_key] += total

        alpha = components["alpha_contribution"]
        weighted_lineage: list[dict[str, Any]] = []
        allocated_alpha = 0.0
        provider_complete = True
        prediction_count = 0
        for weight_row in weights:
            weight = _finite_or_none(weight_row.weight)
            signal = evidence["signals"].get(weight_row.signal_id)
            prediction = evidence["predictions"].get(signal.prediction_id) if signal else None
            feature = evidence["features"].get(prediction.feature_id) if prediction and prediction.feature_id else None
            provider = _provider_from_prediction(prediction, feature) if prediction else None
            contribution = alpha * weight if weight is not None else 0.0
            if weight is not None:
                allocated_alpha += contribution
                by_signal[weight_row.signal_id] += contribution
            family = signal.signal_family if signal else "unattributed"
            if weight is not None:
                by_family[family] += contribution
            model_key = None
            availability = "unknown"
            if prediction is not None:
                prediction_count += 1
                model_key = f"{prediction.model_id}:{prediction.model_version}"
                by_model[model_key] += contribution
                availability = "available" if prediction.external_context_available else "unavailable"
                by_external[availability] += contribution
                provider_key = provider if prediction.external_context_available and provider else (
                    "available_provider_unrecorded" if prediction.external_context_available else "unavailable"
                )
                by_provider[provider_key] += contribution
                if prediction.external_context_available and not provider:
                    provider_complete = False
            else:
                by_external["unknown"] += contribution
                by_provider["unknown"] += contribution
                provider_complete = False
            weighted_lineage.append(
                {
                    "signal_id": weight_row.signal_id,
                    "signal_family": family if signal else None,
                    "prediction_id": prediction.prediction_id if prediction else None,
                    "model": model_key,
                    "weight": weight,
                    "exclusion_reason": weight_row.exclusion_reason,
                    "allocated_alpha_contribution": contribution if weight is not None else None,
                    "external_ai_availability": availability,
                    "external_ai_provider": provider,
                }
            )

        if ensemble is not None:
            ensemble_profiles.setdefault(
                ensemble.ensemble_decision_id,
                {
                    "decision_trace_id": ensemble.decision_trace_id,
                    "regime": ensemble.current_regime,
                    "combined_expected_return": _finite_or_none(ensemble.combined_expected_return),
                    "correlation_penalty": _finite_or_none(ensemble.correlation_penalty),
                    "transaction_cost_penalty": _finite_or_none(ensemble.transaction_cost_penalty),
                    "regime_penalty": _finite_or_none(ensemble.regime_penalty),
                    "external_context_adjustment": _finite_or_none(ensemble.external_context_adjustment),
                    "signal_weights": {
                        row.signal_id: _finite_or_none(row.weight)
                        for row in weights
                    },
                },
            )
        if target is not None:
            portfolio_evidence.setdefault(
                target.portfolio_target_id,
                {
                    "sizing_bucket": sizing_bucket,
                    "current_exposure": _finite_or_none(target.current_exposure),
                    "requested_target_exposure": _finite_or_none(target.requested_target_exposure),
                    "requested_delta": _finite_or_none(target.requested_delta),
                    "expected_risk": _finite_or_none(target.expected_risk),
                    "risk_contribution": _finite_or_none(target.risk_contribution),
                    "source_ensemble_decision_id": target.source_ensemble_decision_id,
                },
            )
        if risk is not None:
            risk_evidence.setdefault(
                risk.risk_decision_id,
                {
                    "resize_category": resize_category,
                    "resize_ratio": resize_ratio,
                    "requested_exposure": _finite_or_none(risk.requested_exposure),
                    "approved_exposure": _finite_or_none(risk.approved_exposure),
                    "requested_leverage": _finite_or_none(risk.requested_leverage),
                    "approved_leverage": _finite_or_none(risk.approved_leverage),
                    "triggered_limits": list(risk.triggered_limits or []),
                    "configuration_version": risk.configuration_version,
                },
            )

        model_signal_lineage = bool(weights) and prediction_count == len(weights)
        funding_status = component_details["funding"]["method"] != "unavailable"
        coverage_flags = {
            "decision_trace": bool(links["trace_id"]),
            "model_signal_lineage": model_signal_lineage,
            "ensemble_weighting": ensemble is not None,
            "portfolio_sizing": target is not None,
            "risk_resizing": risk is not None,
            "execution_fill": bool(fills),
            "fees": component_details["fees"]["value"] is not None,
            "funding_status": funding_status,
            "external_ai_status": model_signal_lineage and provider_complete,
        }
        missing = [name for name, covered in coverage_flags.items() if not covered]
        if all(coverage_flags[name] for name in ("decision_trace", "model_signal_lineage", "ensemble_weighting", "portfolio_sizing", "risk_resizing")):
            lineage_status = "complete_v2"
        elif links["trace_id"]:
            lineage_status = "partial_v2"
        else:
            lineage_status = "legacy_untraced"

        records.append(
            {
                "paper_trade_id": trade.id,
                "paper_account_id": trade.paper_account_id,
                "decision_trace_id": links["trace_id"],
                "risk_decision_id": risk.risk_decision_id if risk else trade.risk_decision_id,
                "portfolio_target_id": target.portfolio_target_id if target else None,
                "ensemble_decision_id": ensemble.ensemble_decision_id if ensemble else None,
                "simulated_order_id": order.order_id if order else trade.simulated_order_id,
                "simulated_fill_ids": [row.fill_id for row in fills],
                "symbol": trade.symbol,
                "regime": regime,
                "time": (_utc(trade.created_at) or trade.created_at).isoformat() if trade.created_at else None,
                "time_period": period_key,
                "pnl_event_kind": intent,
                "pnl_event_kind_method": intent_method,
                "is_closed_pnl": intent in {"closed", "closed_inferred"},
                "action": trade.action,
                "side": trade.side,
                "total_paper_pnl": total,
                "ensemble_weighting": weighted_lineage,
                "ensemble_weight_sum": sum(_finite(row.weight) for row in weights),
                "allocated_alpha_contribution": allocated_alpha,
                "unallocated_alpha_contribution": alpha - allocated_alpha,
                "portfolio_sizing": portfolio_evidence.get(target.portfolio_target_id) if target else None,
                "risk_resizing": risk_evidence.get(risk.risk_decision_id) if risk else None,
                "execution": {
                    "fee": components["fees"],
                    "slippage": components["slippage"],
                    "funding": components["funding"],
                    "funding_estimate_not_booked": component_details["funding"].get("estimated_value"),
                },
                "decomposition": components,
                "component_metadata": component_details,
                "reconciliation": reconciliation,
                "lineage_status": lineage_status,
                "coverage_flags": coverage_flags,
                "missing_evidence": missing,
            }
        )

    total = sum(_finite(row.realized_pnl) for row in trades)
    component_sum = sum(aggregate_components.values())
    closed_records = [row for row in records if row["is_closed_pnl"]]
    component_metadata = {
        name: {
            "value": aggregate_components[name] if aggregate_component_available[name] else None,
            "reconciliation_value": aggregate_components[name],
            "available_trade_count": aggregate_component_available[name],
            "trade_count": len(records),
            "coverage_ratio": aggregate_component_available[name] / len(records) if records else None,
            "methods": sorted(aggregate_component_methods[name]),
            "estimated_value_not_booked": aggregate_estimates.get(name),
        }
        for name in (*_COMPONENT_KEYS, "unexplained_residual")
    }
    return {
        "paper_only": True,
        "method_version": ATTRIBUTION_METHOD_VERSION,
        "selection": {
            "symbol": symbol.upper() if symbol else None,
            "account_id": account_id,
            "start": start_utc.isoformat() if start_utc else None,
            "end": end_utc.isoformat() if end_utc else None,
            "limit": safe_limit,
            "time_period": normalized_period,
            "order": "newest_first",
            "bounded": True,
        },
        "trade_count": len(trades),
        "closed_trade_count": len(closed_records),
        "closed_paper_pnl": sum(_finite(row["total_paper_pnl"]) for row in closed_records),
        "total_paper_pnl": total,
        "components": aggregate_components,
        "component_metadata": component_metadata,
        "reconciliation": {
            "total_paper_pnl": total,
            "component_sum": component_sum,
            "difference": total - component_sum,
            "reconciled": math.isclose(total, component_sum, rel_tol=1e-9, abs_tol=1e-8),
        },
        "by_model": dict(sorted(by_model.items())),
        "by_signal": dict(sorted(by_signal.items())),
        "by_signal_family": dict(sorted(by_family.items())),
        "by_ensemble_weighting": dict(sorted(by_ensemble.items())),
        "ensemble_weighting_evidence": ensemble_profiles,
        "by_portfolio_sizing": dict(sorted(by_portfolio_sizing.items())),
        "portfolio_sizing_evidence": portfolio_evidence,
        "by_risk_resizing": dict(sorted(by_risk_resizing.items())),
        "risk_resizing_evidence": risk_evidence,
        "by_symbol": dict(sorted(by_symbol.items())),
        "by_regime": dict(sorted(by_regime.items())),
        "by_external_ai_availability": dict(sorted(by_external.items())),
        "by_external_ai_provider": dict(sorted(by_provider.items())),
        "by_time_period": dict(sorted(by_period.items())),
        "allocation_reconciliation": {
            "alpha_reconciliation_value": aggregate_components["alpha_contribution"],
            "signal_allocated_alpha": sum(by_signal.values()),
            "signal_unallocated_alpha": aggregate_components["alpha_contribution"] - sum(by_signal.values()),
            "model_allocated_alpha": sum(by_model.values()),
            "model_unallocated_alpha": aggregate_components["alpha_contribution"] - sum(by_model.values()),
            "external_ai_status_allocated_alpha": sum(by_external.values()),
            "external_ai_status_unallocated_alpha": aggregate_components["alpha_contribution"] - sum(by_external.values()),
            "method": "persisted_ensemble_weights_without_renormalizing_missing_lineage",
        },
        "coverage": _coverage_summary(records),
        "trades": records,
        "limitations": [
            "Model/signal/provider values are persisted-weight allocations of the alpha estimate, not causal PnL claims.",
            "Ensemble, sizing, broad-market, and general execution counterfactuals remain null unless explicitly recorded.",
            "New simulator fills book funding cash flow; legacy fills without funding_cash_flow remain estimates reported separately.",
            "Legacy or incomplete traces remain visible and reconcile through unexplained residual.",
        ],
    }


__all__ = [
    "ATTRIBUTION_METHOD_VERSION",
    "MAX_ATTRIBUTION_ROWS",
    "SUPPORTED_TIME_PERIODS",
    "paper_pnl_attribution",
]
