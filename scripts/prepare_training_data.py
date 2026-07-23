from __future__ import annotations

import argparse
import gzip
import json
import math
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
CONVERTER_MODEL_NAMES = {
    "smart": "pc-news-converter-v1-finbert-cryptobert",
    "finbert": "pc-news-converter-v1-finbert",
    "cryptobert": "pc-news-converter-v1-cryptobert",
    "rule-based": "rule-based-fallback-v1",
}
TECHNICAL_FEATURE_COLUMNS = [
    "rsi_14",
    "macd_pct",
    "macd_signal_pct",
    "macd_histogram_pct",
    "sma_20_distance_pct",
    "ema_20_distance_pct",
    "bollinger_width_pct",
    "bollinger_position",
    "atr_14_pct",
    "vwap_20_distance_pct",
    "adx_14",
]
TIME_CONTEXT_FEATURE_COLUMNS = [
    "time_hour_utc_sin",
    "time_hour_utc_cos",
    "time_day_of_week_sin",
    "time_day_of_week_cos",
    "time_is_weekend",
    "session_asia",
    "session_london",
    "session_new_york",
]
CANDLE_FEATURE_COLUMNS = [
    "last_close",
    "candle_return_1m",
    "candle_return_5m",
    "volume_change",
    "volatility",
    "trend_score",
    *TECHNICAL_FEATURE_COLUMNS,
    *TIME_CONTEXT_FEATURE_COLUMNS,
]
_PIPELINES: dict[str, Any] = {}
RAW_FILES_USED = {
    "candles.csv.gz",
    "news_articles.jsonl.gz",
    "external_data_events.jsonl.gz",
    "experience_buffer.jsonl.gz",
    "training_features.jsonl.gz",
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


def _iter_jsonl_gz(root: Path, filename: str):
    for path in root.rglob(filename):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    yield parsed


def _read_jsonl_gz(root: Path, filename: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for parsed in _iter_jsonl_gz(root, filename):
        rows.append(parsed)
    return rows


def _experience_summary(root: Path) -> dict[str, Any]:
    count = 0
    action_balance: dict[str, int] = {}
    for row in _iter_jsonl_gz(root, "experience_buffer.jsonl.gz"):
        count += 1
        action = str(row.get("action") or "UNKNOWN")
        action_balance[action] = action_balance.get(action, 0) + 1
    return {"count": count, "action_balance": action_balance}


def _read_csv_gz(root: Path, filename: str):
    pd, _ = _load_pandas()
    frames = []
    for path in root.rglob(filename):
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _extract_raw_zips(input_path: Path, temp_root: Path) -> Path:
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        target = temp_root / input_path.stem
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(input_path) as archive:
            _extract_needed_raw_files(archive, target)
        return temp_root

    zip_paths = sorted(path for path in input_path.rglob("*.zip") if path.is_file()) if input_path.is_dir() else []
    if not zip_paths:
        return input_path

    for zip_path in zip_paths:
        relative = zip_path.relative_to(input_path).with_suffix("")
        target = temp_root / relative
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as archive:
            _extract_needed_raw_files(archive, target)
    return temp_root


def _extract_needed_raw_files(archive: zipfile.ZipFile, target: Path) -> None:
    for member in archive.infolist():
        if Path(member.filename).name in RAW_FILES_USED:
            archive.extract(member, target)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _datetime_ns(values) -> Any:
    pd, _ = _load_pandas()
    timestamps = pd.Series(pd.to_datetime(values, utc=True, errors="coerce"))
    return timestamps.dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]").astype("int64").to_numpy()


def _keyword_any(text: str, words: list[str]) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in words)


def _local_model_device() -> int:
    try:
        import torch

        return 0 if torch.cuda.is_available() else -1
    except Exception:
        return -1


def _hf_pipeline(name: str, model: str):
    if name in _PIPELINES:
        return _PIPELINES[name]
    try:
        from transformers import pipeline
    except Exception as exc:
        raise SystemExit(
            "Smart news conversion needs transformers + torch on your PC. Run: "
            "pip install transformers torch sentencepiece. "
            "Use --news-converter rule-based only for the weak fallback."
        ) from exc
    _PIPELINES[name] = pipeline("text-classification", model=model, tokenizer=model, device=_local_model_device())
    return _PIPELINES[name]


def _classification_scores(result: Any) -> dict[str, float]:
    if isinstance(result, dict):
        result = [result]
    scores: dict[str, float] = {}
    for item in result or []:
        label = str(item.get("label", "")).lower()
        scores[label] = float(item.get("score", 0.0) or 0.0)
    return scores


def _score_finbert(text: str) -> dict[str, float]:
    classifier = _hf_pipeline("finbert", "ProsusAI/finbert")
    result = classifier(text, truncation=True, max_length=512, top_k=None)
    scores = _classification_scores(result)
    positive = max(scores.get("positive", 0.0), scores.get("label_2", 0.0))
    negative = max(scores.get("negative", 0.0), scores.get("label_0", 0.0))
    neutral = max(scores.get("neutral", 0.0), scores.get("label_1", 0.0))
    return {
        "finbert_positive_score": positive,
        "finbert_negative_score": negative,
        "finbert_neutral_score": neutral,
        "sentiment_score": max(-1.0, min(1.0, positive - negative)),
        "sentiment_confidence": max(positive, negative, neutral),
    }


def _score_cryptobert(text: str) -> dict[str, float]:
    classifier = _hf_pipeline("cryptobert", "ElKulako/cryptobert")
    result = classifier(text, truncation=True, max_length=512, top_k=None)
    scores = _classification_scores(result)
    bullish = max(scores.get("bullish", 0.0), scores.get("positive", 0.0), scores.get("label_2", 0.0))
    bearish = max(scores.get("bearish", 0.0), scores.get("negative", 0.0), scores.get("label_0", 0.0))
    neutral = max(scores.get("neutral", 0.0), scores.get("label_1", 0.0))
    return {
        "crypto_bullish_score": bullish,
        "crypto_bearish_score": bearish,
        "crypto_neutral_score": neutral,
        "crypto_sentiment_score": max(-1.0, min(1.0, bullish - bearish)),
        "crypto_sentiment_confidence": max(bullish, bearish, neutral),
    }


def _topic_scores(full: str) -> dict[str, Any]:
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
        "etf_bullish_score": 1.0 if etf else 0.0,
        "affected_symbols": affected,
    }


def _score_news_rule_based(text: str, title: str = "") -> dict[str, Any]:
    full = f"{title}\n{text}".lower()
    positive = ["approve", "approved", "etf inflow", "inflow", "bullish", "surge", "rally", "record high", "adoption", "partnership"]
    negative = ["hack", "exploit", "lawsuit", "ban", "crackdown", "outflow", "liquidation", "war", "default", "collapse", "fraud"]
    risk = ["risk", "hack", "exploit", "war", "sec", "lawsuit", "inflation", "federal reserve", "interest rate", "ban", "crackdown"]
    pos_score = sum(1 for word in positive if word in full)
    neg_score = sum(1 for word in negative if word in full)
    risk_score = min(1.0, sum(1 for word in risk if word in full) / 4.0)
    sentiment = max(-1.0, min(1.0, (pos_score - neg_score) / 3.0))
    topics = _topic_scores(full)
    return {
        "sentiment_score": sentiment,
        "sentiment_confidence": min(1.0, 0.35 + 0.12 * (pos_score + neg_score + len(topics["affected_symbols"]))),
        "risk_score": risk_score,
        "impact_score": min(1.0, 0.15 + 0.15 * len(topics["affected_symbols"]) + (0.25 if topics["macro_related"] else 0.0) + (0.2 if topics["regulation_risk_score"] else 0.0)),
        **topics,
    }


def _score_news(text: str, title: str = "", converter: str = "smart") -> dict[str, Any]:
    full = f"{title}\n{text}".lower()[:4000]
    if converter == "rule-based":
        return _score_news_rule_based(text, title)
    topics = _topic_scores(full)
    if converter == "finbert":
        scored = _score_finbert(full)
        return {
            **scored,
            "risk_score": scored["finbert_negative_score"],
            "impact_score": scored["sentiment_confidence"],
            **topics,
        }
    if converter == "cryptobert":
        scored = _score_cryptobert(full)
        return {
            "sentiment_score": scored["crypto_sentiment_score"],
            "sentiment_confidence": scored["crypto_sentiment_confidence"],
            "risk_score": scored["crypto_bearish_score"],
            "impact_score": scored["crypto_sentiment_confidence"],
            **topics,
            **scored,
        }
    finbert = _score_finbert(full)
    crypto = _score_cryptobert(full)
    sentiment_score = (finbert["sentiment_score"] * 0.55) + (crypto["crypto_sentiment_score"] * 0.45)
    risk_score = max(finbert["finbert_negative_score"], crypto["crypto_bearish_score"], topics["regulation_risk_score"], topics["security_risk_score"], topics["war_risk_score"])
    return {
        "sentiment_score": max(-1.0, min(1.0, sentiment_score)),
        "sentiment_confidence": max(finbert["sentiment_confidence"], crypto["crypto_sentiment_confidence"]),
        "risk_score": min(1.0, risk_score),
        "impact_score": max(abs(sentiment_score), finbert["sentiment_confidence"] * 0.5, crypto["crypto_sentiment_confidence"] * 0.5),
        **topics,
        **finbert,
        **crypto,
    }


def _news_frame(root: Path, converter: str):
    pd, _ = _load_pandas()
    rows = []
    for row in _read_jsonl_gz(root, "news_articles.jsonl.gz"):
        text = str(row.get("raw_text") or "")
        title = str(row.get("title") or "")
        scored = _score_news(text, title, converter=converter)
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
    frame = _candle_feature_frame(candles)
    if frame.empty:
        return frame
    frame["position"] = frame.groupby("symbol").cumcount()
    frame = frame[frame["position"] >= 5].drop(columns=["position"])
    frame["feature_schema_version"] = "local-raw-v2"
    return frame


def _time_context_frame(frame):
    pd, np = _load_pandas()
    if frame.empty or "as_of" not in frame:
        return frame
    output = frame.copy()
    timestamps = pd.to_datetime(output["as_of"], utc=True, errors="coerce")
    hour = timestamps.dt.hour.fillna(0).astype(float) + timestamps.dt.minute.fillna(0).astype(float) / 60.0
    day = timestamps.dt.dayofweek.fillna(0).astype(float)
    hour_angle = (hour / 24.0) * (math.pi * 2.0)
    day_angle = (day / 7.0) * (math.pi * 2.0)
    output["time_hour_utc_sin"] = np.sin(hour_angle)
    output["time_hour_utc_cos"] = np.cos(hour_angle)
    output["time_day_of_week_sin"] = np.sin(day_angle)
    output["time_day_of_week_cos"] = np.cos(day_angle)
    output["time_is_weekend"] = (day >= 5).astype(float)
    output["session_asia"] = ((hour >= 0) & (hour < 8)).astype(float)
    output["session_london"] = ((hour >= 7) & (hour < 16)).astype(float)
    output["session_new_york"] = ((hour >= 13) & (hour < 22)).astype(float)
    return output


def _candle_feature_frame(candles):
    pd, np = _load_pandas()
    if candles.empty:
        return pd.DataFrame()
    frames = []
    for symbol, group in candles.groupby("symbol"):
        group = group.sort_values("as_of").copy()
        close = group["close"].astype(float)
        high = group["high"].astype(float)
        low = group["low"].astype(float)
        volume = group["volume"].astype(float)
        safe_close = close.replace(0, np.nan)
        returns = close.pct_change().replace([np.inf, -np.inf], np.nan)
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_20 = close.ewm(span=20, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        sma_20 = close.rolling(20, min_periods=1).mean()
        bollinger_std = close.rolling(20, min_periods=2).std(ddof=0).fillna(0.0)
        bollinger_upper = sma_20 + bollinger_std * 2.0
        bollinger_lower = sma_20 - bollinger_std * 2.0
        bollinger_range = (bollinger_upper - bollinger_lower).replace(0, np.nan)
        typical_price = (high + low + close) / 3.0
        vwap_denominator = volume.rolling(20, min_periods=1).sum().replace(0, np.nan)
        vwap_20 = (typical_price * volume).rolling(20, min_periods=1).sum() / vwap_denominator
        previous_close = close.shift(1)
        true_range = pd.concat(
            [
                high - low,
                (high - previous_close).abs(),
                (low - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1).fillna(0.0)
        atr_14 = true_range.rolling(14, min_periods=1).mean()
        delta = close.diff()
        average_gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
        average_loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
        rs = average_gain / average_loss.replace(0, np.nan)
        rsi = (1.0 - (1.0 / (1.0 + rs))).fillna(0.5).clip(0.0, 1.0)
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=group.index)
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=group.index)
        tr_sum = true_range.rolling(14, min_periods=1).sum().replace(0, np.nan)
        plus_di = 100.0 * plus_dm.rolling(14, min_periods=1).sum() / tr_sum
        minus_di = 100.0 * minus_dm.rolling(14, min_periods=1).sum() / tr_sum
        dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0.0)
        adx = dx.rolling(14, min_periods=1).mean().clip(0.0, 1.0)
        output = pd.DataFrame(
            {
                "symbol": symbol,
                "as_of": group["as_of"],
                "feature_schema_version": "local-raw-v2",
                "last_close": close,
                "candle_return_1m": returns.fillna(0.0),
                "candle_return_5m": close.pct_change(5).replace([np.inf, -np.inf], np.nan).fillna(0.0),
                "volume_change": volume.pct_change(5).replace([np.inf, -np.inf], np.nan).fillna(0.0),
                "volatility": returns.rolling(20, min_periods=2).std().fillna(0.0),
                "trend_score": np.tanh((close / close.shift(20).fillna(close.iloc[0]).replace(0, np.nan) - 1.0).fillna(0.0) * 50.0),
                "rsi_14": rsi,
                "macd_pct": (macd / safe_close).fillna(0.0),
                "macd_signal_pct": (macd_signal / safe_close).fillna(0.0),
                "macd_histogram_pct": ((macd - macd_signal) / safe_close).fillna(0.0),
                "sma_20_distance_pct": (close / sma_20.replace(0, np.nan) - 1.0).fillna(0.0),
                "ema_20_distance_pct": (close / ema_20.replace(0, np.nan) - 1.0).fillna(0.0),
                "bollinger_width_pct": ((bollinger_upper - bollinger_lower) / sma_20.replace(0, np.nan)).fillna(0.0),
                "bollinger_position": ((close - bollinger_lower) / bollinger_range).fillna(0.5).clip(0.0, 1.0),
                "atr_14_pct": (atr_14 / safe_close).fillna(0.0),
                "vwap_20_distance_pct": (close / vwap_20.replace(0, np.nan) - 1.0).fillna(0.0),
                "adx_14": adx,
            }
        )
        frames.append(_time_context_frame(output))
    return pd.concat(frames, ignore_index=True).sort_values(["symbol", "as_of"]) if frames else pd.DataFrame()


def _add_candle_features(frame, candles):
    pd, _ = _load_pandas()
    if frame.empty:
        return frame
    if candles.empty:
        return _time_context_frame(frame)
    source = _candle_feature_frame(candles)
    if source.empty:
        return _time_context_frame(frame)
    merged_rows = []
    for symbol, group in frame.groupby("symbol", sort=False):
        source_group = source[source["symbol"] == symbol].sort_values("as_of")
        group = group.sort_values("as_of").copy()
        if source_group.empty:
            merged_rows.append(_time_context_frame(group))
            continue
        merged = pd.merge_asof(
            group,
            source_group[["as_of", *CANDLE_FEATURE_COLUMNS]].sort_values("as_of"),
            on="as_of",
            direction="backward",
            suffixes=("", "_candle"),
        )
        for column in CANDLE_FEATURE_COLUMNS:
            candle_column = f"{column}_candle"
            if candle_column in merged:
                if column in group.columns:
                    merged[column] = merged[column].where(merged[column].notna(), merged[candle_column])
                else:
                    merged[column] = merged[candle_column]
                merged = merged.drop(columns=[candle_column])
        merged_rows.append(_time_context_frame(merged))
    return pd.concat(merged_rows, ignore_index=True).sort_values(["symbol", "as_of"]) if merged_rows else _time_context_frame(frame)


def _zero_news_features() -> dict[str, float]:
    return {
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


def _aggregate_news_window(window, as_of, symbol: str, lookback_hours: float, np) -> dict[str, float]:
    zero = _zero_news_features()
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
    pd, np = _load_pandas()
    if frame.empty:
        return frame
    if news.empty:
        features = [_zero_news_features() for _ in range(len(frame))]
    else:
        news = news.sort_values("event_time").reset_index(drop=True)
        event_ns = _datetime_ns(news["event_time"])
        lookback_ns = int(pd.Timedelta(hours=lookback_hours).value)
        value_columns = [column for column in _zero_news_features() if column not in {"recency_weight", "article_count_used"}]
        value_arrays = {
            column: pd.to_numeric(news[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            for column in value_columns
            if column in news.columns
        }
        related_arrays = {
            column: pd.to_numeric(news[column], errors="coerce").fillna(0.0).to_numpy(dtype=float) > 0
            for column in news.columns
            if column.endswith("_related")
        }
        macro_related = related_arrays.get("macro_related", np.zeros(len(news), dtype=bool))
        features = []
        for row in frame[["as_of", "symbol"]].itertuples(index=False):
            as_of_ns = row.as_of.value
            start_index = int(np.searchsorted(event_ns, as_of_ns - lookback_ns, side="left"))
            end_index = int(np.searchsorted(event_ns, as_of_ns, side="right"))
            output = _zero_news_features()
            if end_index > start_index:
                indices = np.arange(start_index, end_index)
                symbol_key = str(row.symbol).upper().replace("USDT", "").lower()
                related = related_arrays.get(f"{symbol_key}_related")
                if related is not None:
                    filtered = indices[related[indices] | macro_related[indices]]
                    if len(filtered):
                        indices = filtered
                age_hours = np.maximum((as_of_ns - event_ns[indices]) / 3_600_000_000_000.0, 0.0)
                weights = np.exp(-age_hours / max(lookback_hours / 2.0, 1.0))
                weight_sum = float(weights.sum()) or 1.0
                for column in value_columns:
                    values = value_arrays.get(column)
                    output[column] = float((values[indices] * weights).sum() / weight_sum) if values is not None else 0.0
                output["recency_weight"] = float(weights.max()) if len(weights) else 0.0
                output["article_count_used"] = float(len(indices))
            features.append(output)
    news_features = pd.DataFrame(features)
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
    pd, np = _load_pandas()
    if frame.empty:
        return frame
    if external.empty:
        frame["external_event_count_24h"] = 0.0
        frame["external_numeric_mean_24h"] = 0.0
        return frame
    external = external.sort_values("event_time").reset_index(drop=True)
    lookback_ns = int(pd.Timedelta(hours=lookback_hours).value)

    def build_arrays(rows):
        event_ns = _datetime_ns(rows["event_time"])
        numeric_values = pd.to_numeric(rows.get("numeric_value"), errors="coerce").to_numpy(dtype=float)
        valid_numeric = (~np.isnan(numeric_values)).astype(float)
        return (
            event_ns,
            np.concatenate([[0.0], np.nan_to_num(numeric_values, nan=0.0).cumsum()]),
            np.concatenate([[0.0], valid_numeric.cumsum()]),
        )

    def summarize(arrays, as_of_ns: int) -> tuple[float, float, float]:
        event_ns, numeric_cumsum, valid_cumsum = arrays
        start_index = int(np.searchsorted(event_ns, as_of_ns - lookback_ns, side="left"))
        end_index = int(np.searchsorted(event_ns, as_of_ns, side="right"))
        count = float(end_index - start_index)
        valid_count = float(valid_cumsum[end_index] - valid_cumsum[start_index])
        total = float(numeric_cumsum[end_index] - numeric_cumsum[start_index])
        return count, total, valid_count

    if "symbol" in external.columns:
        global_rows = external[external["symbol"].isna()]
        symbol_rows = external[external["symbol"].notna()]
    else:
        global_rows = external
        symbol_rows = external.iloc[0:0]
    global_arrays = build_arrays(global_rows) if not global_rows.empty else None
    symbol_arrays = {str(symbol): build_arrays(group) for symbol, group in symbol_rows.groupby("symbol", sort=False)}

    counts = []
    means = []
    for row in frame[["as_of", "symbol"]].itertuples(index=False):
        as_of_ns = row.as_of.value
        count = total = valid_count = 0.0
        if global_arrays is not None:
            count, total, valid_count = summarize(global_arrays, as_of_ns)
        arrays = symbol_arrays.get(str(row.symbol))
        if arrays is not None:
            symbol_count, symbol_total, symbol_valid_count = summarize(arrays, as_of_ns)
            count += symbol_count
            total += symbol_total
            valid_count += symbol_valid_count
        counts.append(count)
        means.append(float(total / valid_count) if valid_count else 0.0)
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


def _quality_report(
    frame,
    *,
    candles_rows: int,
    news_rows: int,
    external_rows: int,
    experience_summary: dict[str, Any],
    converter_model: str,
) -> dict[str, Any]:
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
    action_balance = experience_summary.get("action_balance", {}) if isinstance(experience_summary, dict) else {}
    return {
        "total_rows": total_rows,
        "labeled_rows": labeled_rows,
        "date_range": {
            "start": frame["as_of"].min().isoformat() if total_rows else None,
            "end": frame["as_of"].max().isoformat() if total_rows else None,
            "days": date_range_days,
        },
        "symbols": sorted(frame["symbol"].dropna().unique().tolist()) if total_rows and "symbol" in frame else [],
        "candles_coverage_rows": int(candles_rows),
        "news_coverage_percentage": float((frame.get("article_count_used", 0) > 0).mean() * 100.0) if total_rows else 0.0,
        "derivatives_coverage_percentage": float((frame.get("external_event_count_24h", 0) > 0).mean() * 100.0) if total_rows else 0.0,
        "external_context_coverage_percentage": float((frame.get("external_event_count_24h", 0) > 0).mean() * 100.0) if total_rows else 0.0,
        "experience_rows_used": int((experience_summary or {}).get("count", 0)),
        "duplicate_rows": duplicate_rows,
        "missing_columns": missing_columns,
        "target_distribution": target_distribution,
        "buy_sell_hold_balance": action_balance,
        "news_converter_model": converter_model,
        "raw_news_rows": int(news_rows),
        "external_rows": int(external_rows),
        "future_leakage_detected": False,
        "warnings": warnings,
    }


def _process(root: Path, output_dir: Path, lookback_hours: float, fee_rate: float, news_converter: str) -> dict[str, Any]:
    pd, _ = _load_pandas()
    converter_model = CONVERTER_MODEL_NAMES[news_converter]
    candles = _candle_frame(root)
    news = _news_frame(root, news_converter)
    external = _external_frame(root)
    experience_summary = _experience_summary(root)
    frame = _feature_rows(root)
    if frame.empty:
        frame = _rows_from_candles(candles)
    if frame.empty:
        raise SystemExit("No training feature rows or candle rows were found in the raw input.")
    frame = _add_candle_features(frame, candles)
    frame = _add_news_features(frame, news, lookback_hours)
    frame = _add_external_summary(frame, external, 24.0)
    frame = _add_labels(frame, candles, fee_rate)
    frame["news_converter_model"] = converter_model
    if "feature_schema_version" not in frame:
        frame["feature_schema_version"] = "local-raw-v1"
    frame = frame.sort_values(["symbol", "as_of"]).drop_duplicates(subset=["symbol", "as_of"], keep="last")

    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dataset_path = output_dir / f"anata_training_ready_{stamp}.csv.gz"
    report_path = output_dir / f"data_quality_{stamp}.json"
    frame.to_csv(dataset_path, index=False, compression="gzip")
    report = _quality_report(
        frame,
        candles_rows=len(candles),
        news_rows=len(news),
        external_rows=len(external),
        experience_summary=experience_summary,
        converter_model=converter_model,
    )
    report["dataset_path"] = str(dataset_path)
    report["created_at"] = datetime.now(timezone.utc).isoformat()
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return {
        "status": "ok",
        "dataset": str(dataset_path),
        "quality_report": str(report_path),
        "report": report,
    }


def _raw_zip_paths(input_path: Path) -> list[Path]:
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        return [input_path]
    if not input_path.is_dir():
        return []
    return sorted(path for path in input_path.rglob("*.zip") if path.is_file())


def _chunks(values: list[Path], size: int):
    size = max(1, size)
    for index in range(0, len(values), size):
        yield values[index : index + size]


def _combine_action_balances(reports: list[dict[str, Any]]) -> dict[str, int]:
    combined: dict[str, int] = {}
    for report in reports:
        for action, count in (report.get("buy_sell_hold_balance") or {}).items():
            combined[str(action)] = combined.get(str(action), 0) + int(count or 0)
    return combined


def _process_daily_batches(
    input_path: Path,
    output_dir: Path,
    lookback_hours: float,
    fee_rate: float,
    news_converter: str,
    batch_size: int,
    keep_batch_files: bool,
) -> dict[str, Any] | None:
    zip_paths = _raw_zip_paths(input_path)
    if len(zip_paths) <= 1:
        return None

    pd, _ = _load_pandas()
    converter_model = CONVERTER_MODEL_NAMES[news_converter]
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    batch_dir = output_dir / f"_daily_batches_{stamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    dataset_paths: list[Path] = []
    batch_reports: list[dict[str, Any]] = []
    batch_inputs: list[list[str]] = []
    for batch_number, batch in enumerate(_chunks(zip_paths, batch_size), start=1):
        with tempfile.TemporaryDirectory(prefix=f"anata_batch_{batch_number}_") as temp_dir:
            temp_root = Path(temp_dir)
            for zip_path in batch:
                _extract_raw_zips(zip_path, temp_root)
            result = _process(temp_root, batch_dir, lookback_hours, fee_rate, news_converter)
        dataset_paths.append(Path(result["dataset"]))
        batch_reports.append(result["report"])
        batch_inputs.append([str(path) for path in batch])
        print(f"processed batch {batch_number}: {', '.join(path.name for path in batch)}", flush=True)

    frames = []
    for dataset_path in dataset_paths:
        frames.append(pd.read_csv(dataset_path))
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if frame.empty:
        raise SystemExit("Daily batches finished, but no processed rows were produced.")
    frame["as_of"] = pd.to_datetime(frame["as_of"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["as_of", "symbol"]).sort_values(["symbol", "as_of"]).drop_duplicates(subset=["symbol", "as_of"], keep="last")
    frame["news_converter_model"] = converter_model

    dataset_path = output_dir / f"anata_training_ready_{stamp}_batched.csv.gz"
    report_path = output_dir / f"data_quality_{stamp}_batched.json"
    frame.to_csv(dataset_path, index=False, compression="gzip")

    experience_summary = {
        "count": sum(int(report.get("experience_rows_used", 0) or 0) for report in batch_reports),
        "action_balance": _combine_action_balances(batch_reports),
    }
    report = _quality_report(
        frame,
        candles_rows=sum(int(report.get("candles_coverage_rows", 0) or 0) for report in batch_reports),
        news_rows=sum(int(report.get("raw_news_rows", 0) or 0) for report in batch_reports),
        external_rows=sum(int(report.get("external_rows", 0) or 0) for report in batch_reports),
        experience_summary=experience_summary,
        converter_model=converter_model,
    )
    report["dataset_path"] = str(dataset_path)
    report["created_at"] = datetime.now(timezone.utc).isoformat()
    report["batch_mode"] = "daily_zip_batches"
    report["batch_size"] = int(batch_size)
    report["batch_count"] = len(batch_reports)
    report["batch_inputs"] = batch_inputs
    if keep_batch_files:
        report["batch_reports"] = batch_reports
    else:
        report["batch_reports"] = [
            {key: value for key, value in batch_report.items() if key not in {"dataset_path", "created_at"}}
            for batch_report in batch_reports
        ]
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    if not keep_batch_files:
        shutil.rmtree(batch_dir, ignore_errors=True)

    return {
        "status": "ok",
        "dataset": str(dataset_path),
        "quality_report": str(report_path),
        "report": report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw Anata data into a training-ready numeric dataset.")
    parser.add_argument("--input", type=Path, required=True, help="Raw data zip, raw_days folder, or finished_data/YYYY-MM-DD folder.")
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/processed"))
    parser.add_argument("--news-lookback-hours", type=float, default=6.0)
    parser.add_argument("--fee-rate", type=float, default=0.0004)
    parser.add_argument(
        "--batch-daily-files",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When --input is a folder of daily ZIPs, process small ZIP batches and merge the final dataset. Default: enabled.",
    )
    parser.add_argument(
        "--daily-batch-size",
        type=int,
        default=1,
        help="Number of daily ZIPs to process at once in --batch-daily-files mode. Use 1 for lowest RAM, 2-3 for faster runs on stronger PCs.",
    )
    parser.add_argument(
        "--keep-batch-files",
        action="store_true",
        help="Keep intermediate processed batch CSVs for debugging. The final merged dataset is always kept.",
    )
    parser.add_argument(
        "--news-converter",
        choices=["smart", "finbert", "cryptobert", "rule-based"],
        default="smart",
        help="Use smart/finbert/cryptobert on your PC. rule-based is only the weak fallback.",
    )
    args = parser.parse_args()

    try:
        if args.batch_daily_files:
            batch_result = _process_daily_batches(
                args.input,
                args.output_dir,
                args.news_lookback_hours,
                args.fee_rate,
                args.news_converter,
                args.daily_batch_size,
                args.keep_batch_files,
            )
            if batch_result is not None:
                print(json.dumps(batch_result, indent=2, default=str))
                return
        if args.input.is_file() and args.input.suffix.lower() != ".zip":
            print(json.dumps(_process(args.input, args.output_dir, args.news_lookback_hours, args.fee_rate, args.news_converter), indent=2))
            return
        with tempfile.TemporaryDirectory(prefix="anata_raw_") as temp_dir:
            temp_root = Path(temp_dir)
            process_root = _extract_raw_zips(args.input, temp_root)
            print(json.dumps(_process(process_root, args.output_dir, args.news_lookback_hours, args.fee_rate, args.news_converter), indent=2))
    except Exception as exc:
        print(f"Prepare training data failed: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
