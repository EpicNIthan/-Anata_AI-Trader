from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import io
import json
import math
import os
import statistics
import sys
import time
import zipfile
from collections import defaultdict
from datetime import date, datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib import error, parse, request

DEFAULT_SYMBOL_COINS = {
    "BTCUSDT": "bitcoin",
    "ETHUSDT": "ethereum",
    "SOLUSDT": "solana",
    "BNBUSDT": "binancecoin",
    "XRPUSDT": "ripple",
    "ADAUSDT": "cardano",
    "DOGEUSDT": "dogecoin",
    "AVAXUSDT": "avalanche-2",
    "LINKUSDT": "chainlink",
    "LTCUSDT": "litecoin",
}

CANDLE_FIELDS = [
    "symbol",
    "timeframe",
    "open_time",
    "close_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "source",
    "market_cap",
    "total_volume_24h",
    "sample_interval_minutes",
]

PUBLIC_BASE_URL = "https://api.coingecko.com/api/v3"
PRO_BASE_URL = "https://pro-api.coingecko.com/api/v3"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_date(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _start_of_day(value: date) -> datetime:
    return datetime.combine(value, dt_time.min, tzinfo=timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def _chunks(start: datetime, end: datetime, chunk_days: int) -> list[tuple[datetime, datetime]]:
    chunks: list[tuple[datetime, datetime]] = []
    cursor = start
    step = timedelta(days=chunk_days)
    while cursor < end:
        chunk_end = min(cursor + step, end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end
    return chunks


def _request_json(url: str, headers: dict[str, str], max_retries: int) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        req = request.Request(url, headers=headers)
        try:
            with request.urlopen(req, timeout=60) as response:
                payload = response.read().decode("utf-8")
                return json.loads(payload)
        except error.HTTPError as exc:
            last_error = exc
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            if exc.code == 429:
                retry_after = exc.headers.get("Retry-After")
                wait_seconds = float(retry_after) if retry_after and retry_after.isdigit() else min(120.0, 10.0 * attempt)
                print(f"Rate limited by CoinGecko. Waiting {wait_seconds:.0f}s before retry {attempt}/{max_retries}...", flush=True)
                time.sleep(wait_seconds)
                continue
            if exc.code in {500, 502, 503, 504}:
                wait_seconds = min(60.0, 5.0 * attempt)
                print(f"CoinGecko HTTP {exc.code}. Waiting {wait_seconds:.0f}s before retry {attempt}/{max_retries}...", flush=True)
                time.sleep(wait_seconds)
                continue
            raise RuntimeError(f"CoinGecko returned HTTP {exc.code}: {detail}") from exc
        except (TimeoutError, error.URLError, json.JSONDecodeError) as exc:
            last_error = exc
            wait_seconds = min(60.0, 5.0 * attempt)
            print(f"Request failed ({exc}). Waiting {wait_seconds:.0f}s before retry {attempt}/{max_retries}...", flush=True)
            time.sleep(wait_seconds)
    raise RuntimeError(f"CoinGecko request failed after {max_retries} retries: {last_error}")


def _market_chart_range(
    base_url: str,
    coin_id: str,
    vs_currency: str,
    start: datetime,
    end: datetime,
    headers: dict[str, str],
    max_retries: int,
) -> dict[str, Any]:
    query = parse.urlencode(
        {
            "vs_currency": vs_currency,
            "from": int(start.timestamp()),
            "to": int(end.timestamp()),
            "precision": "full",
        }
    )
    url = f"{base_url.rstrip('/')}/coins/{parse.quote(coin_id)}/market_chart/range?{query}"
    return _request_json(url, headers, max_retries)


def _nearest_value(points: list[tuple[int, float]], timestamp_ms: int, max_delta_ms: int) -> float | None:
    if not points:
        return None
    timestamps = [item[0] for item in points]
    index = bisect.bisect_left(timestamps, timestamp_ms)
    best: tuple[int, float] | None = None
    for candidate_index in (index - 1, index):
        if 0 <= candidate_index < len(points):
            candidate = points[candidate_index]
            if best is None or abs(candidate[0] - timestamp_ms) < abs(best[0] - timestamp_ms):
                best = candidate
    if best is None or abs(best[0] - timestamp_ms) > max_delta_ms:
        return None
    return best[1]


def _median_interval_ms(timestamps: list[int]) -> int:
    if len(timestamps) < 2:
        return 60 * 60 * 1000
    diffs = [b - a for a, b in zip(timestamps, timestamps[1:]) if b > a]
    if not diffs:
        return 60 * 60 * 1000
    return int(statistics.median(diffs))


def _timeframe_name(interval_minutes: float) -> str:
    if interval_minutes <= 7:
        return "5m"
    if interval_minutes <= 90:
        return "1h"
    if interval_minutes <= 360:
        return "4h"
    return "1d"


def _download_coin(
    symbol: str,
    coin_id: str,
    base_url: str,
    vs_currency: str,
    start: datetime,
    end: datetime,
    chunk_days: int,
    sleep_seconds: float,
    headers: dict[str, str],
    max_retries: int,
) -> list[dict[str, Any]]:
    price_by_ts: dict[int, float] = {}
    market_cap_by_ts: dict[int, float] = {}
    volume_by_ts: dict[int, float] = {}
    ranges = _chunks(start, end, chunk_days)
    for index, (chunk_start, chunk_end) in enumerate(ranges, start=1):
        print(f"{symbol}: fetching {coin_id} chunk {index}/{len(ranges)} {_iso(chunk_start)} -> {_iso(chunk_end)}", flush=True)
        payload = _market_chart_range(base_url, coin_id, vs_currency, chunk_start, chunk_end, headers, max_retries)
        for timestamp, price in payload.get("prices", []) or []:
            price_by_ts[int(timestamp)] = _safe_float(price)
        for timestamp, market_cap in payload.get("market_caps", []) or []:
            market_cap_by_ts[int(timestamp)] = _safe_float(market_cap)
        for timestamp, volume in payload.get("total_volumes", []) or []:
            volume_by_ts[int(timestamp)] = _safe_float(volume)
        if index < len(ranges) and sleep_seconds > 0:
            time.sleep(sleep_seconds)

    timestamps = sorted(price_by_ts)
    market_cap_points = sorted(market_cap_by_ts.items())
    volume_points = sorted(volume_by_ts.items())
    interval_ms = _median_interval_ms(timestamps)
    nearest_window_ms = max(interval_ms * 2, 3 * 60 * 60 * 1000)
    records: list[dict[str, Any]] = []
    closes: list[float] = []
    volumes: list[float] = []
    for position, timestamp_ms in enumerate(timestamps):
        close = price_by_ts[timestamp_ms]
        previous_close = closes[-1] if closes else close
        market_cap = _nearest_value(market_cap_points, timestamp_ms, nearest_window_ms) or 0.0
        total_volume = _nearest_value(volume_points, timestamp_ms, nearest_window_ms) or 0.0
        closes.append(close)
        volumes.append(total_volume)
        recent_returns = []
        for before, after in zip(closes[-21:-1], closes[-20:]):
            if before:
                recent_returns.append(after / before - 1.0)
        volatility = statistics.pstdev(recent_returns) if len(recent_returns) > 1 else 0.0
        return_1 = close / previous_close - 1.0 if previous_close else 0.0
        return_5 = close / closes[-6] - 1.0 if len(closes) >= 6 and closes[-6] else 0.0
        trend_base = closes[-21] if len(closes) >= 21 else closes[0]
        trend_score = math.tanh((close / trend_base - 1.0) * 50.0) if trend_base else 0.0
        volume_base = volumes[-6] if len(volumes) >= 6 and volumes[-6] else max(total_volume, 1e-9)
        volume_change = total_volume / max(volume_base, 1e-9) - 1.0
        close_time = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)
        sample_interval_minutes = max(1.0, interval_ms / 60000.0)
        open_time = close_time - timedelta(milliseconds=interval_ms)
        records.append(
            {
                "id": f"coingecko-{symbol.lower()}-{timestamp_ms}",
                "symbol": symbol,
                "coin_id": coin_id,
                "as_of": _iso(close_time),
                "open_time": _iso(open_time),
                "close_time": _iso(close_time),
                "timeframe": _timeframe_name(sample_interval_minutes),
                "open": previous_close,
                "high": max(previous_close, close),
                "low": min(previous_close, close),
                "close": close,
                "volume": total_volume,
                "market_cap": market_cap,
                "total_volume_24h": total_volume,
                "sample_interval_minutes": sample_interval_minutes,
                "candle_return_1m": return_1,
                "candle_return_5m": return_5,
                "volume_change": volume_change,
                "volatility": volatility,
                "trend_score": trend_score,
                "position": position,
            }
        )
    return records


def _gzip_csv(rows: list[dict[str, Any]], fieldnames: list[str]) -> bytes:
    text_buffer = io.StringIO()
    writer = csv.DictWriter(text_buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return gzip.compress(text_buffer.getvalue().encode("utf-8"), compresslevel=6)


def _gzip_jsonl(rows: list[dict[str, Any]]) -> bytes:
    payload = "".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in rows)
    return gzip.compress(payload.encode("utf-8"), compresslevel=6)


def _daily_outputs(records: list[dict[str, Any]], output_dir: Path, overwrite: bool, run_manifest: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["as_of"])[:10]].append(record)

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    skipped: list[str] = []
    for day, day_records in sorted(grouped.items()):
        output_path = output_dir / f"raw_{day}.zip"
        if output_path.exists() and not overwrite:
            skipped.append(str(output_path))
            continue
        candles = [
            {
                "symbol": row["symbol"],
                "timeframe": row["timeframe"],
                "open_time": row["open_time"],
                "close_time": row["close_time"],
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
                "source": "coingecko_market_chart_range",
                "market_cap": row["market_cap"],
                "total_volume_24h": row["total_volume_24h"],
                "sample_interval_minutes": row["sample_interval_minutes"],
            }
            for row in day_records
        ]
        features = []
        external_events = []
        for row in day_records:
            feature_values = {
                "last_close": row["close"],
                "candle_return_1m": row["candle_return_1m"],
                "candle_return_5m": row["candle_return_5m"],
                "volume_change": row["volume_change"],
                "volatility": row["volatility"],
                "trend_score": row["trend_score"],
                "coingecko_market_cap_usd": row["market_cap"],
                "coingecko_total_volume_24h": row["total_volume_24h"],
                "coingecko_sample_interval_minutes": row["sample_interval_minutes"],
                "final_ai_input": {
                    "source": "coingecko_history",
                    "coin_id": row["coin_id"],
                    "timeframe": row["timeframe"],
                },
            }
            features.append(
                {
                    "id": row["id"],
                    "symbol": row["symbol"],
                    "as_of": row["as_of"],
                    "schema_version": "coingecko-history-v1",
                    "feature_values": feature_values,
                    "payload": {"source": "coingecko", "coin_id": row["coin_id"]},
                }
            )
            external_events.append(
                {
                    "event_id": f"{row['id']}-market-cap",
                    "source": "coingecko",
                    "event_type": "historical_market_cap_usd",
                    "symbol": row["symbol"],
                    "event_time": row["as_of"],
                    "numeric_value": row["market_cap"],
                    "payload": {"coin_id": row["coin_id"]},
                }
            )
            external_events.append(
                {
                    "event_id": f"{row['id']}-volume-24h",
                    "source": "coingecko",
                    "event_type": "historical_total_volume_24h",
                    "symbol": row["symbol"],
                    "event_time": row["as_of"],
                    "numeric_value": row["total_volume_24h"],
                    "payload": {"coin_id": row["coin_id"]},
                }
            )
        day_manifest = {
            **run_manifest,
            "day": day,
            "record_count": len(day_records),
            "symbols": sorted({row["symbol"] for row in day_records}),
            "min_as_of": min(row["as_of"] for row in day_records),
            "max_as_of": max(row["as_of"] for row in day_records),
            "files": ["candles.csv.gz", "training_features.jsonl.gz", "external_data_events.jsonl.gz", "manifest.json"],
        }
        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            archive.writestr("candles.csv.gz", _gzip_csv(candles, CANDLE_FIELDS))
            archive.writestr("training_features.jsonl.gz", _gzip_jsonl(features))
            archive.writestr("external_data_events.jsonl.gz", _gzip_jsonl(external_events))
            archive.writestr("manifest.json", json.dumps(day_manifest, indent=2, sort_keys=True))
        written.append(str(output_path))
    return {
        "written_count": len(written),
        "skipped_count": len(skipped),
        "written_files": written,
        "skipped_existing_files": skipped,
    }


def _symbol_map(symbols: str) -> dict[str, str]:
    selected: dict[str, str] = {}
    for raw in symbols.split(","):
        item = raw.strip()
        if not item:
            continue
        if ":" in item:
            symbol, coin_id = item.split(":", 1)
            selected[symbol.strip().upper()] = coin_id.strip()
        else:
            symbol = item.upper()
            if symbol not in DEFAULT_SYMBOL_COINS:
                raise SystemExit(f"Unknown symbol {symbol}. Use SYMBOL:coingecko_coin_id, for example BTCUSDT:bitcoin")
            selected[symbol] = DEFAULT_SYMBOL_COINS[symbol]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect historical free CoinGecko market data into Anata raw daily files.")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOL_COINS), help="Comma symbols or SYMBOL:coin_id pairs.")
    parser.add_argument("--days", type=int, default=365, help="Days to backfill when --start-date is not provided. Public CoinGecko historical range is normally limited to the past 365 days.")
    parser.add_argument("--start-date", default=None, help="UTC start date/datetime, for example 2025-06-01 or 2025-06-01T00:00:00Z.")
    parser.add_argument("--end-date", default=None, help="UTC end date/datetime. Default: now.")
    parser.add_argument("--vs-currency", default="usd")
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/raw_days"))
    parser.add_argument("--chunk-days", type=int, default=80, help="Keep <=90 to receive hourly CoinGecko granularity instead of daily.")
    parser.add_argument("--sleep-seconds", type=float, default=4.0, help="Pause between CoinGecko requests to avoid rate limits.")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--api-key", default=os.getenv("COINGECKO_API_KEY") or os.getenv("COINGECKO_DEMO_API_KEY"))
    parser.add_argument("--pro", action="store_true", help="Use pro-api.coingecko.com and x-cg-pro-api-key header.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing datasets/raw_days/raw_YYYY-MM-DD.zip files.")
    args = parser.parse_args()

    if args.chunk_days <= 0:
        raise SystemExit("--chunk-days must be positive.")
    if args.chunk_days > 90:
        print("WARNING: chunk-days > 90 usually returns daily data from CoinGecko, not hourly data.", file=sys.stderr)
    if args.days > 365 and not args.api_key:
        print("WARNING: CoinGecko public/demo historical range is normally limited to the past 365 days.", file=sys.stderr)

    end = _parse_date(args.end_date) if args.end_date else _utc_now()
    if args.start_date:
        start = _parse_date(args.start_date)
    else:
        start = end - timedelta(days=args.days)
    if start >= end:
        raise SystemExit("Start date must be earlier than end date.")

    symbol_map = _symbol_map(args.symbols)
    base_url = PRO_BASE_URL if args.pro else PUBLIC_BASE_URL
    headers = {"User-Agent": "AnataAITrader-CoinGeckoHistory/1.0"}
    if args.api_key:
        headers["x-cg-pro-api-key" if args.pro else "x-cg-demo-api-key"] = args.api_key

    all_records: list[dict[str, Any]] = []
    for index, (symbol, coin_id) in enumerate(symbol_map.items(), start=1):
        print(f"Symbol {index}/{len(symbol_map)}: {symbol} -> {coin_id}", flush=True)
        all_records.extend(
            _download_coin(
                symbol=symbol,
                coin_id=coin_id,
                base_url=base_url,
                vs_currency=args.vs_currency,
                start=start,
                end=end,
                chunk_days=args.chunk_days,
                sleep_seconds=args.sleep_seconds,
                headers=headers,
                max_retries=args.max_retries,
            )
        )
        if index < len(symbol_map) and args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    run_manifest = {
        "source": "coingecko_market_chart_range",
        "collector": "historical_collectors/coingecko_history/collect_coingecko_history.py",
        "created_at": _iso(_utc_now()),
        "start": _iso(start),
        "end": _iso(end),
        "vs_currency": args.vs_currency,
        "chunk_days": args.chunk_days,
        "symbols": symbol_map,
        "notes": [
            "CoinGecko market_chart/range returns price, market cap, and total volume points, not exchange-native 1m candles.",
            "Default chunk-days is 80 to keep free API responses hourly instead of daily for long backfills.",
            "volume uses CoinGecko total volume at timestamp; it is not Binance candle volume.",
        ],
    }
    result = _daily_outputs(all_records, args.output_dir, args.overwrite, run_manifest)
    sample_intervals = [row["sample_interval_minutes"] for row in all_records if row.get("sample_interval_minutes")]
    summary = {
        "status": "ok",
        "records": len(all_records),
        "symbols": sorted(symbol_map),
        "date_range": {"start": _iso(start), "end": _iso(end)},
        "output_dir": str(args.output_dir),
        "median_sample_interval_minutes": statistics.median(sample_intervals) if sample_intervals else None,
        **result,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
