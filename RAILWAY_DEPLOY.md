# Railway Deploy Checklist

This app is Railway-ready through `Dockerfile` and `railway.json`.

## Required Railway Variables

Set these in Railway project variables:

```env
TRADING_MODE=paper
BINANCE_SYMBOLS=BTCUSDT,ETHUSDT
BINANCE_INTERVAL=1m
BINANCE_REST_BASE_URL=https://data-api.binance.vision
BINANCE_WS_BASE_URL=wss://data-stream.binance.vision
STORE_LIVE_CANDLE_UPDATES=true
PAPER_TRADE_TIMEFRAME=1m
NEWS_API_KEY=
NEWS_PROVIDER=rss,gdelt,newsapi
RSS_NEWS_ENABLED=true
RSS_FEEDS=https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml,https://cointelegraph.com/rss,https://decrypt.co/feed
RSS_REQUEST_USER_AGENT=AnataAITrader/1.0 RSS reader
GDELT_ENABLED=true
GDELT_POLL_INTERVAL_SECONDS=900
GDELT_MAX_RECORDS=20
NEWSAPI_ENABLED=false
NEWS_POLL_INTERVAL_SECONDS=120
ENABLE_MARKET_COLLECTOR=true
ENABLE_NEWS_COLLECTOR=true
AUTO_TRADER_ENABLED=false
AUTO_TRADER_INTERVAL_SECONDS=60
AUTO_TRADER_SYMBOLS=BTCUSDT,ETHUSDT
PAPER_START_BALANCE=10000
```

Add a Railway PostgreSQL service. Railway should provide `DATABASE_URL`; the app accepts Railway's `postgresql://...` URL and converts it for `psycopg`.

## Start Command

Railway uses:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

## After Deploy

Open:

- `/health`
- `/dashboard`
- `/api/market/status`
- `/api/news/status`

If Binance is blocked locally but reachable from Railway, market backfill and websocket messages should begin there.

## Safety

The app is paper-only. Do not add exchange API keys or live-order credentials yet.
