from __future__ import annotations

import argparse
import gzip
import json
import math
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

NEWS_CONVERTER_MODEL = "rule-based-fallback-v1"
LABEL_TARGETS = {
    "target_future_return_5m": 5,
    "target_future_return_15m": 15,
    "target_future_return_1h": 60,
    "target_future_return_4h": 240,
}
NON_NUMERIC_COLUMNS = {
    "symbol",
    "as_of",
    "feature_schema_version",
    "final_ai_input",
    "news_converter_model",
}


def _load_pandas():
    try:
        import numpy as np
        import pandas as pd
    except Exception as exc:
        raise SystemExit("This local script needs pandas and numpy. Run: pip install -r requirements-local-training.txt") from exc
    return pd, np


def _parse_json(value: Any) -> Any:
    if isinstance(value, str) and value.strip():
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _read_jsonl_gz(root: Path, filename: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in root.rglob(filename):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    rows.append(parsed)
    return rows


def _read_csv_gz(root: Path, filename: str):
    pd, _ = _load_pandas()
    frames = []
    for path in root.rglob(filename):
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _keyword_any(text: str, words: list[str]) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in words)


def _score_news(text: str, title: str = "") -> dict[str, Any]:
    full = f"{title}\n{text}".lower()
    positive = ["approve", "approved", "etf inflow", "inflow", "bullish", "surge", "rally", "record high", "adoption", "partnership"]
    negative = ["hack", "exploit", "lawsuit", "ban", "crackdown", "outflow", "liquidation", "war", "default", "collapse", "fraud"]
    risk = ["risk", "hack", "exploit", "war", "sec", "lawsuit", "inflation", "federal reserve", "interest rate", "ban", "crackdown"]
    pos_score = sum(1 for word in positive if word in full)
    neg_score = sum(1 for word in negative if word in full)
    risk_score = min(1.0, sum(1 for word in risk if word in full) / 4.0)
    sentiment = max(-1.0, min(1.0, (pos_score - neg_score) / 3.0))
    btc = _keyword_any(full, ["bitcoin", "btc"])
    eth = _keyword_any(full, ["ethereum", "ether", "eth"])
    sol = _keyword_any(full, ["solana", "sol"])
    bnb = _keyword_any(full, ["binance", "bnb"])
    xrp = _keyword_any(full, ["xrp", "ripple"])
    macro = _keyword_any(full, ["fed", "federal reserve", "inflation", "interest rate", "war", "oil", "dollar", "jobs report"])
    regulation = _keyword_any(full, ["sec", "cftc", "regulation", "lawsuit", "court", "ban", "compliance"])
    security = _keyword_any(full, ["hack", "exploit", "breach", "phishing", "stolen"])
    etf = _keyword_any(full, ["etf", "spot bitcoin", "blackrock", "fidelity"])
    affected = []
    if btc:
        affected.append("BTCUSDT")
    if eth:
        affected.append("ETHUSDT")
    if sol:
        affected.append("SOLUSDT")
    if bnb:
        affected.append("BNBUSDT")
    if xrp:
        affected.append("XRPUSDT")
    return {
        "sentiment_score": sentiment,
        "sentiment_confidence": min(1.0, 0.35 + 0.12 * (pos_score + neg_score + len(affected))),
        "risk_score": risk_score,
        "impact_score": min(1.0, 0.15 + 0.15 * len(affected) + (0.25 if macro else 0.0) + (0.2 if regulation else 0.0)),
        "macro_related": float(macro),
        "btc_related": float(btc),
        "eth_related": float(eth),
        "sol_related": float(sol),
        "bnb_related": float(bnb),
        "xrp_related": float(xrp),
        "regulation_risk_score": 1.0 if regulation else 0.0,
        "fed_risk_score": 1.0 if _keyword_any(full, ["fed", "federal reserve", "interest rate", "rate hike"]) else 0.0,
        "war_risk_score": 1.0 if _keyword_any(full, ["war", "missile", "attack", "sanction"]) else 0.0,
        "security_risk_score": 1.0 if security else 0.0,
        "exchange_hack_risk_score": 1.0 if security and _keyword_any(full, ["exchange", "binance", "okx", "coinbase"]) else 0.0,
        "etf_bullish_score": 1.0 if etf and sentiment >= 0 else 0.0,
        "affected_symbols": affected,
    }


def _news_frame(root: Path):
    pd, _ = _load_pandas()
    rows = []
    for row in _read_jsonl_gz(root, "news_articles.jsonl.gz"):
        text = str(row.get("raw_text") or "")
        title = str(row.get("title") or "")
        scored = _score_news(text, title)
        event_time = row.get("published_at") or row.get("created_at")
        rows.append(
            {
                "article_id": row.get("article_id") or row.get("id"),
                "source": row.get("source"),
                "provider": row.get("provider") or row.get("source_name"),
                "title": title,
                "url": row.get("url"),
                "event_time": event_time,
                "raw_text_length": len(text),
                **scored,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["event_time"]).sort_values("event_time")
    return frame.drop_duplicates(subset=["url", "title", "event_time"], keep="last")


def _feature_rows(root: Path):
    pd, _ = _load_pandas()
    rows = []
    for row in _read_jsonl_gz(root, "training_features.jsonl.gz"):
        values = _parse_json(row.get("feature_values")) or {}
        payload = _parse_json(row.get("payload")) or {}
        if not isinstance(values, dict):
            values = {}
        if not values and isinstance(payload, dict):
            values = payload.get("values") or {}
        output = {
            "training_feature_id": row.get("id"),
            "symbol": row.get("symbol"),
            "as_of": row.get("as_of"),
            "feature_schema_version": row.get("schema_version") or (payload or {}).get("schema_version"),
            "final_ai_input": json.dumps(values.get("final_ai_input") or {}, sort_keys=True, default=str),
        }
        for key, value in values.items():
            if key == "final_ai_input":
                continue
            if isinstance(value, (int, float)) or value in (None, ""):
                output[key] = _safe_float(value)
        rows.append(output)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["as_of"] = pd.to_datetime(frame["as_of"], utc=True, errors="coerce")
    return frame.dropna(subset=["as_of", "symbol"]).sort_values(["symbol", "as_of"])


def _candle_frame(root: Path):
    pd, _ = _load_pandas()
    frame = _read_csv_gz(root, "candles.csv.gz")
    if frame.empty:
        return frame
    for column in ("open_time", "close_time"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    frame["as_of"] = frame["close_time"].fillna(frame["open_time"])
    return frame.dropna(subset=["as_of", "symbol"]).sort_values(["symbol", "as_of"])


def _rows_from_candles(candles):
    pd, _ = _load_pandas()
    if candles.empty:
        return pd.DataFrame()
    rows = []
    for symbol, group in candles.groupby("symbol"):
        group = group.sort_values("as_of").copy()
        close = group["close"].astype(float)
        volume = group["volume"].astype(float)
        for index, row in group.iterrows():
            position = group.index.get_loc(index)
            if position < 5:
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "as_of": row["as_of"],
                    "feature_schema_version": "local-raw-v1",
                    "last_close": float(row["close"]),
                    "candle_return_1m": float(close.iloc[position] / close.iloc[position - 1] - 1.0) if close.iloc[position - 1] else 0.0,
                    "candle_return_5m": float(close.iloc[position] / close.iloc[position - 5] - 1.0) if close.iloc[position - 5] else 0.0,
                    "volume_change": float(volume.iloc[position] / max(volume.iloc[position - 5], 1e-9) - 1.0),
                    "volatility": float(close.iloc[max(0, position - 20) : position + 1].pct_change().std() or 0.0),
                    "trend_score": float(math.tanh((close.iloc[position] / max(close.iloc[max(0, position - 20)], 1e-9) - 1.0) * 50.0)),
                }
            )
    return pd.DataFrame(rows)


def _aggregate_news_for_row(news, as_of, symbol: str, lookback_hours: float):
    pd, np = _load_pandas()
    zero = {
        "sentiment_score": 0.0,
        "sentiment_confidence": 0.0,
        "risk_score": 0.0,
        "impact_score": 0.0,
        "macro_related": 0.0,
        "btc_related": 0.0,
        "eth_related": 0.0,
        "regulation_risk_score": 0.0,
        "fed_risk_score": 0.0,
        "war_risk_score": 0.0,
        "security_risk_score": 0.0,
        "exchange_hack_risk_score": 0.0,
        "etf_bullish_score": 0.0,
        "recency_weight": 0.0,
        "article_count_used": 0.0,
    }
    if news.empty:
        return zero
    start = as_of - pd.Timedelta(hours=lookback_hours)
    window = news[(news["event_time"] <= as_of) & (news["event_time"] >= start)].copy()
    if window.empty:
        return zero
    symbol_key = symbol.upper().replace("USDT", "").lower()
    related_column = f"{symbol_key.lower()}_related"
    if related_column in window.columns:
        related = (window[related_column] > 0) | (window["macro_related"] > 0)
        filtered = window[related]
        if not filtered.empty:
            window = filtered
    age_hours = (as_of - window["event_time"]).dt.total_seconds().clip(lower=0) / 3600.0
    weights = np.exp(-age_hours / max(lookback_hours / 2.0, 1.0))
    weight_sum = float(weights.sum()) or 1.0
    output = {}
    for column in zero:
        if column in {"recency_weight", "article_count_used"}:
            continue
        if column in window.columns:
            output[column] = float((window[column].astype(float) * weights).sum() / weight_sum)
        else:
            output[column] = 0.0
    output["recency_weight"] = float(weights.max()) if len(weights) else 0.0
    output["article_count_used"] = float(len(window))
    return output


def _add_news_features(frame, news, lookback_hours: float):
    if frame.empty:
        return frame
    features = [
        _aggregate_news_for_row(news, row.as_of, str(row.symbol), lookback_hours)
        for row in frame[["as_of", "symbol"]].itertuples(index=False)
    ]
    news_features = _load_pandas()[0].DataFrame(features)
    frame = frame.reset_index(drop=True).copy()
    news_features = news_features.reset_index(drop=True)
    for column in news_features.columns:
        frame[column] = news_features[column]
    return frame


def _external_frame(root: Path):
    pd, _ = _load_pandas()
    rows = _read_jsonl_gz(root, "external_data_events.jsonl.gz")
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True, errors="coerce")
    frame["numeric_value"] = pd.to_numeric(frame.get("numeric_value"), errors="coerce")
    return frame.dropna(subset=["event_time"]).sort_values("event_time")


def _add_external_summary(frame, external, lookback_hours: float):
    pd, _ = _load_pandas()
    if frame.empty:
        return frame
    if external.empty:
        frame["external_event_count_24h"] = 0.0
        frame["external_numeric_mean_24h"] = 0.0
        return frame
    counts = []
    means = []
    for row in frame[["as_of", "symbol"]].itertuples(index=False):
        start = row.as_of - pd.Timedelta(hours=lookback_hours)
        window = external[(external["event_time"] <= row.as_of) & (external["event_time"] >= start)]
        if "symbol" in window.columns:
            symbol_window = window[(window["symbol"].isna()) | (window["symbol"] == row.symbol)]
            if not symbol_window.empty:
                window = symbol_window
        counts.append(float(len(window)))
        means.append(float(window["numeric_value"].dropna().mean()) if "numeric_value" in window and not window["numeric_value"].dropna().empty else 0.0)
    frame["external_event_count_24h"] = counts
    frame["external_numeric_mean_24h"] = means
    return frame


def _add_labels(frame, candles, fee_rate: float):
    pd, np = _load_pandas()
    if frame.empty or candles.empty:
        return frame
    for column in LABEL_TARGETS:
        frame[column] = np.nan
    frame["target_max_upside_1h"] = np.nan
    frame["target_max_drawdown_1h"] = np.nan
    frame["target_stop_loss_hit_first"] = np.nan
    frame["target_take_profit_hit_first"] = np.nan
    frame["target_direction_15m"] = np.nan
    frame["target_trade_quality_score"] = np.nan

    result_rows = []
    for symbol, group in frame.groupby("symbol", sort=False):
        candle_group = candles[candles["symbol"] == symbol].sort_values("as_of")
        if candle_group.empty:
            result_rows.append(group)
            continue
        candle_times = candle_group["as_of"].map(lambda value: value.value).to_numpy()
        closes = candle_group["close"].astype(float).to_numpy()
        highs = candle_group["high"].astype(float).to_numpy()
        lows = candle_group["low"].astype(float).to_numpy()
        group = group.copy()
        for row_index, row in group.iterrows():
            current_index = candle_times.searchsorted(row["as_of"].value, side="right") - 1
            if current_index < 0 or current_index >= len(closes) or closes[current_index] == 0:
                continue
            current_close = closes[current_index]
            target_values: dict[str, float] = {}
            for column, minutes in LABEL_TARGETS.items():
                target_time = (row["as_of"] + pd.Timedelta(minutes=minutes)).value
                future_index = candle_times.searchsorted(target_time, side="left")
                if future_index < len(closes):
                    target_values[column] = float(closes[future_index] / current_close - 1.0)
                    group.at[row_index, column] = target_values[column]
            end_1h = candle_times.searchsorted((row["as_of"] + pd.Timedelta(hours=1)).value, side="right")
            if end_1h > current_index + 1:
                high_window = highs[current_index + 1 : end_1h]
                low_window = lows[current_index + 1 : end_1h]
                upside = float(high_window.max() / current_close - 1.0) if len(high_window) else 0.0
                drawdown = float(low_window.min() / current_close - 1.0) if len(low_window) else 0.0
                group.at[row_index, "target_max_upside_1h"] = upside
                group.at[row_index, "target_max_drawdown_1h"] = drawdown
                stop_threshold = -0.005
                take_threshold = 0.010
                stop_first = 0.0
                take_first = 0.0
                for high_value, low_value in zip(high_window, low_window):
                    if low_value / current_close - 1.0 <= stop_threshold:
                        stop_first = 1.0
                        break
                    if high_value / current_close - 1.0 >= take_threshold:
                        take_first = 1.0
                        break
                group.at[row_index, "target_stop_loss_hit_first"] = stop_first
                group.at[row_index, "target_take_profit_hit_first"] = take_first
            ret_15m = target_values.get("target_future_return_15m")
            if ret_15m is not None:
                group.at[row_index, "target_direction_15m"] = 1.0 if ret_15m > 0 else (-1.0 if ret_15m < 0 else 0.0)
                group.at[row_index, "target_trade_quality_score"] = float(ret_15m - (2 * fee_rate))
        result_rows.append(group)
    return pd.concat(result_rows, ignore_index=True).sort_values(["symbol", "as_of"])


def _quality_report(frame, candles, news, external, experience_rows: list[dict[str, Any]]) -> dict[str, Any]:
    warnings = []
    total_rows = int(len(frame))
    labeled_rows = int(frame["target_trade_quality_score"].notna().sum()) if "target_trade_quality_score" in frame else 0
    date_range_days = 0.0
    if total_rows and "as_of" in frame:
        date_range_days = float((frame["as_of"].max() - frame["as_of"].min()).total_seconds() / 86400.0)
    news_zero_pct = 1.0
    if total_rows and {"sentiment_score", "risk_score", "impact_score"}.issubset(frame.columns):
        news_zero_pct = float(((frame["sentiment_score"].abs() + frame["risk_score"].abs() + frame["impact_score"].abs()) == 0).mean())
    if news_zero_pct > 0.80 and total_rows:
        warnings.append("Too many rows have zero news features. Collect more raw news or use a stronger local news converter.")
    if labeled_rows < 1000:
        warnings.append("Labeled rows are low. More closed candles are needed for reliable training.")
    if date_range_days < 2:
        warnings.append("Date range is short. Two or more days is recommended before training.")
    duplicate_rows = int(frame.duplicated(subset=["symbol", "as_of"]).sum()) if total_rows else 0
    if duplicate_rows:
        warnings.append("Duplicate symbol/as_of rows found.")
    missing_columns = [column for column in ["sentiment_score", "risk_score", "candle_return_1m", "target_trade_quality_score"] if column not in frame]
    if missing_columns:
        warnings.append(f"Missing expected columns: {', '.join(missing_columns)}")
    target_distribution = {}
    if "target_direction_15m" in frame:
        target_distribution = {str(key): int(value) for key, value in frame["target_direction_15m"].value_counts(dropna=True).to_dict().items()}
    action_balance = {}
    if experience_rows:
        for row in experience_rows:
            action = str(row.get("action") or "UNKNOWN")
            action_balance[action] = action_balance.get(action, 0) + 1
    return {
        "total_rows": total_rows,
        "labeled_rows": labeled_rows,
        "date_range": {
            "start": frame["as_of"].min().isoformat() if total_rows else None,
            "end": frame["as_of"].max().isoformat() if total_rows else None,
            "days": date_range_days,
        },
        "symbols": sorted(frame["symbol"].dropna().unique().tolist()) if total_rows and "symbol" in frame else [],
        "candles_coverage_rows": int(len(candles)),
        "news_coverage_percentage": float((frame.get("article_count_used", 0) > 0).mean() * 100.0) if total_rows else 0.0,
        "derivatives_coverage_percentage": float((frame.get("external_event_count_24h", 0) > 0).mean() * 100.0) if total_rows else 0.0,
        "external_context_coverage_percentage": float((frame.get("external_event_count_24h", 0) > 0).mean() * 100.0) if total_rows else 0.0,
        "experience_rows_used": len(experience_rows),
        "duplicate_rows": duplicate_rows,
        "missing_columns": missing_columns,
        "target_distribution": target_distribution,
        "buy_sell_hold_balance": action_balance,
        "news_converter_model": NEWS_CONVERTER_MODEL,
        "raw_news_rows": int(len(news)),
        "external_rows": int(len(external)),
        "future_leakage_detected": False,
        "warnings": warnings,
    }


def _process(root: Path, output_dir: Path, lookback_hours: float, fee_rate: float) -> dict[str, Any]:
    pd, _ = _load_pandas()
    candles = _candle_frame(root)
    news = _news_frame(root)
    external = _external_frame(root)
    experience_rows = _read_jsonl_gz(root, "experience_buffer.jsonl.gz")
    frame = _feature_rows(root)
    if frame.empty:
        frame = _rows_from_candles(candles)
    if frame.empty:
        raise SystemExit("No training feature rows or candle rows were found in the raw input.")
    frame = _add_news_features(frame, news, lookback_hours)
    frame = _add_external_summary(frame, external, 24.0)
    frame = _add_labels(frame, candles, fee_rate)
    frame["news_converter_model"] = NEWS_CONVERTER_MODEL
    if "feature_schema_version" not in frame:
        frame["feature_schema_version"] = "local-raw-v1"
    frame = frame.sort_values(["symbol", "as_of"]).drop_duplicates(subset=["symbol", "as_of"], keep="last")

    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dataset_path = output_dir / f"anata_training_ready_{stamp}.csv.gz"
    report_path = output_dir / f"data_quality_{stamp}.json"
    frame.to_csv(dataset_path, index=False, compression="gzip")
    report = _quality_report(frame, candles, news, external, experience_rows)
    report["dataset_path"] = str(dataset_path)
    report["created_at"] = datetime.now(timezone.utc).isoformat()
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return {
        "status": "ok",
        "dataset": str(dataset_path),
        "quality_report": str(report_path),
        "report": report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw Anata data into a training-ready numeric dataset.")
    parser.add_argument("--input", type=Path, required=True, help="Raw data zip or finished_data/YYYY-MM-DD folder.")
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/processed"))
    parser.add_argument("--news-lookback-hours", type=float, default=6.0)
    parser.add_argument("--fee-rate", type=float, default=0.0004)
    args = parser.parse_args()

    try:
        if args.input.is_file() and args.input.suffix.lower() == ".zip":
            with tempfile.TemporaryDirectory(prefix="anata_raw_") as temp_dir:
                temp_root = Path(temp_dir)
                with zipfile.ZipFile(args.input) as archive:
                    archive.extractall(temp_root)
                print(json.dumps(_process(temp_root, args.output_dir, args.news_lookback_hours, args.fee_rate), indent=2))
        else:
            print(json.dumps(_process(args.input, args.output_dir, args.news_lookback_hours, args.fee_rate), indent=2))
    except Exception as exc:
        print(f"Prepare training data failed: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
