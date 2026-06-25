(function () {
  const config = window.DASHBOARD_CONFIG || {};
  const state = {
    symbol: config.defaultSymbol || "BTCUSDT",
    timeframe: config.defaultTimeframe || "1m",
    chart: null,
    candleSeries: null,
    volumeSeries: null,
    smaSeries: null,
    lastCandlesKey: "",
    lastSeriesKey: "",
    forceFit: true,
    chartRequestId: 0,
    resizeObserver: null,
  };
  const featureFields = [
    ["sentiment_score", "sentiment_score"],
    ["sentiment_confidence", "sentiment_confidence"],
    ["risk_score", "risk_score"],
    ["impact_score", "impact_score"],
    ["recency_weight", "recency_weight"],
    ["btc_related", "btc_related"],
    ["eth_related", "eth_related"],
    ["macro_related", "macro_related"],
    ["candle_return_1m", "candle_return_1m"],
    ["candle_return_5m", "candle_return_5m"],
    ["volatility", "volatility"],
    ["volume_change", "volume_change"],
    ["trend_score", "trend_score"],
    ["crowd_long_account_pct", "crowd_long_account_pct"],
    ["crowd_short_account_pct", "crowd_short_account_pct"],
    ["crowd_long_short_ratio", "crowd_long_short_ratio"],
    ["top_trader_long_account_pct", "top_trader_long_account_pct"],
    ["top_trader_position_long_pct", "top_trader_position_long_pct"],
    ["taker_buy_pressure", "taker_buy_pressure"],
    ["taker_buy_sell_ratio", "taker_buy_sell_ratio"],
    ["open_interest_value", "open_interest_value"],
    ["open_interest_change", "open_interest_change"],
    ["funding_rate", "funding_rate"],
    ["trader_crowd_score", "trader_crowd_score"],
    ["crowd_risk_score", "crowd_risk_score"],
    ["derivatives_recency_weight", "derivatives_recency_weight"],
    ["fear_greed_value", "fear_greed_value"],
    ["fear_greed_change_1d", "fear_greed_change_1d"],
    ["fear_greed_change_24h", "fear_greed_change_24h"],
    ["fear_greed_classification", "fear_greed_classification"],
    ["total_market_cap_usd", "total_market_cap_usd"],
    ["market_cap_change_24h", "market_cap_change_24h"],
    ["global_market_cap_change_24h", "global_market_cap_change_24h"],
    ["total_volume_usd", "total_volume_usd"],
    ["total_volume_change_24h", "total_volume_change_24h"],
    ["btc_dominance", "btc_dominance"],
    ["btc_dominance_change", "btc_dominance_change"],
    ["btc_dominance_change_24h", "btc_dominance_change_24h"],
    ["eth_dominance", "eth_dominance"],
    ["liquidation_long_usd_1m", "liquidation_long_usd_1m"],
    ["liquidation_short_usd_1m", "liquidation_short_usd_1m"],
    ["liquidation_long_usd_5m", "liquidation_long_usd_5m"],
    ["liquidation_short_usd_5m", "liquidation_short_usd_5m"],
    ["liquidation_total_usd_5m", "liquidation_total_usd_5m"],
    ["liquidation_imbalance_5m", "liquidation_imbalance_5m"],
    ["liquidation_spike_score", "liquidation_spike_score"],
    ["usdt_deviation", "usdt_deviation"],
    ["usdc_deviation", "usdc_deviation"],
    ["usdt_price_deviation", "usdt_price_deviation"],
    ["usdc_price_deviation", "usdc_price_deviation"],
    ["stablecoin_depeg_risk", "stablecoin_depeg_risk"],
    ["stablecoin_supply_change_1d", "stablecoin_supply_change_1d"],
    ["stablecoin_supply_change_24h", "stablecoin_supply_change_24h"],
    ["macro_risk_score", "macro_risk_score"],
    ["regulation_risk_score", "regulation_risk_score"],
    ["fed_risk_score", "fed_risk_score"],
    ["war_risk_score", "war_risk_score"],
    ["exchange_hack_risk_score", "exchange_hack_risk_score"],
    ["etf_positive_score", "etf_positive_score"],
    ["security_risk_score", "security_risk_score"],
    ["etf_bullish_score", "etf_bullish_score"],
    ["world_risk_score", "world_risk_score"],
    ["market_regime_score", "market_regime_score"],
  ];

  const $ = (id) => document.getElementById(id);
  const money = (value, digits = 2) => Number.isFinite(Number(value)) ? `$${Number(value).toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits })}` : "-";
  const number = (value, digits = 2) => Number.isFinite(Number(value)) ? Number(value).toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits }) : "-";
  const pct = (value, digits = 2) => Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(digits)}%` : "-";
  const mb = (value) => Number.isFinite(Number(value)) ? `${Number(value).toFixed(2)} MB` : "-";
  const when = (value) => value ? new Date(value).toLocaleString() : "-";
  const cls = (value) => Number(value) >= 0 ? "positive" : "negative";
  const featureValue = (value) => Number.isFinite(Number(value)) ? Number(value).toFixed(6) : escapeHtml(value ?? "-");
  const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  }[char]));

  async function api(path, options = {}) {
    const response = await fetch(path, { credentials: "same-origin", ...options });
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    return response.json();
  }

  function setText(id, value) {
    const element = $(id);
    if (element) element.textContent = value;
  }

  function setClass(id, className) {
    const element = $(id);
    if (!element) return;
    element.classList.remove("positive", "negative", "warning");
    if (className) element.classList.add(className);
  }

  function updateBook(price) {
    const mark = Number(price);
    if (!Number.isFinite(mark)) return;
    setText("bookAsk", money(mark * 1.0002, 2));
    setText("bookMid", money(mark, 2));
    setText("bookBid", money(mark * 0.9998, 2));
  }

  function setupChart() {
    const element = $("chart");
    if (!element || state.chart || !window.LightweightCharts) return;
    const width = Math.max(element.clientWidth || 0, 320);
    const height = Math.max(element.clientHeight || 0, 360);
    state.chart = LightweightCharts.createChart(element, {
      width,
      height,
      layout: { background: { color: "#0c1015" }, textColor: "#8793a0" },
      grid: { vertLines: { color: "#151b22" }, horzLines: { color: "#151b22" } },
      rightPriceScale: { borderColor: "#252c35" },
      timeScale: { borderColor: "#252c35", timeVisible: true },
      crosshair: { mode: LightweightCharts.CrosshairMode?.Normal || 0 },
    });
    state.candleSeries = state.chart.addCandlestickSeries({
      upColor: "#0ecb81",
      downColor: "#f6465d",
      borderUpColor: "#0ecb81",
      borderDownColor: "#f6465d",
      wickUpColor: "#0ecb81",
      wickDownColor: "#f6465d",
    });
    state.volumeSeries = state.chart.addHistogramSeries({
      priceFormat: { type: "volume" },
      priceScaleId: "",
      scaleMargins: { top: 0.78, bottom: 0 },
    });
    state.smaSeries = state.chart.addLineSeries({ color: "#f0b90b", lineWidth: 1 });
    if (window.ResizeObserver) {
      state.resizeObserver = new ResizeObserver((entries) => {
        const box = entries[0]?.contentRect;
        if (box && state.chart) state.chart.resize(Math.floor(box.width), Math.floor(box.height));
      });
      state.resizeObserver.observe(element);
    }
  }

  function sma(candles, period = 20) {
    return candles.map((candle, index) => {
      if (index + 1 < period) return null;
      const slice = candles.slice(index + 1 - period, index + 1);
      const average = slice.reduce((sum, item) => sum + Number(item.close), 0) / period;
      return { time: candle.time, value: average };
    }).filter(Boolean);
  }

  async function refreshChart() {
    setupChart();
    const requestId = ++state.chartRequestId;
    const requestedSymbol = state.symbol;
    const requestedTimeframe = state.timeframe;
    const seriesKey = `${requestedSymbol}:${requestedTimeframe}`;
    try {
      const candles = await api(`/api/market/candles?symbol=${encodeURIComponent(requestedSymbol)}&timeframe=${encodeURIComponent(requestedTimeframe)}&limit=300`);
      if (requestId !== state.chartRequestId || requestedSymbol !== state.symbol || requestedTimeframe !== state.timeframe) return;

      if (!candles.length) {
        if (state.lastSeriesKey !== seriesKey) {
          state.candleSeries?.setData([]);
          state.volumeSeries?.setData([]);
          state.smaSeries?.setData([]);
        }
        state.lastCandlesKey = "";
        state.lastSeriesKey = seriesKey;
        state.forceFit = true;
        setText("chartStatus", `No ${requestedTimeframe} candles for ${requestedSymbol}`);
        setClass("chartStatus", "warning");
        return;
      }

      const candleData = candles.filter((item) => item.time !== null && item.time !== undefined).map((item) => ({
          time: item.time,
          open: Number(item.open),
          high: Number(item.high),
          low: Number(item.low),
          close: Number(item.close),
        })).filter((item) => Number.isFinite(item.open) && Number.isFinite(item.high) && Number.isFinite(item.low) && Number.isFinite(item.close));
      const volumeData = candles.filter((item) => item.time !== null && item.time !== undefined).map((item) => ({
          time: item.time,
          value: Number(item.volume || 0),
          color: Number(item.close) >= Number(item.open) ? "rgba(14, 203, 129, 0.35)" : "rgba(246, 70, 93, 0.35)",
        }));
      if (!candleData.length) {
        setText("chartStatus", `Invalid candle data for ${requestedSymbol}`);
        setClass("chartStatus", "warning");
        return;
      }

      const lastRaw = candles.at(-1);
      const key = `${seriesKey}:${candles.length}:${candles[0]?.open_time || ""}:${lastRaw?.open_time || ""}:${lastRaw?.close || ""}:${lastRaw?.volume || ""}`;
      const shouldRefit = state.forceFit || state.lastSeriesKey !== seriesKey;
      const visibleRange = !shouldRefit && state.chart ? state.chart.timeScale().getVisibleLogicalRange() : null;
      if (key !== state.lastCandlesKey || shouldRefit) {
        if (state.candleSeries) state.candleSeries.setData(candleData);
        if (state.volumeSeries) state.volumeSeries.setData(volumeData);
        const smaData = sma(candleData);
        if (state.smaSeries) state.smaSeries.setData(smaData);
        if (state.chart) {
          if (shouldRefit) state.chart.timeScale().fitContent();
          else if (visibleRange) state.chart.timeScale().setVisibleLogicalRange(visibleRange);
        }
        state.lastCandlesKey = key;
        state.lastSeriesKey = seriesKey;
        state.forceFit = false;
        const last = candleData.at(-1);
        const first = candleData[0];
        const change = first && first.close ? (Number(last.close) - Number(first.close)) / Number(first.close) : 0;
        setText("latestPrice", money(last.close, 2));
        setText("priceChange", pct(change, 2));
        setClass("priceChange", cls(change));
        setText(`railPrice${requestedSymbol}`, money(last.close, 2));
        setText(`railChange${requestedSymbol}`, pct(change, 2));
        setClass(`railChange${requestedSymbol}`, cls(change));
        updateBook(last.close);
        setText("volumeMetric", number(lastRaw?.volume, 2));
        setText("smaMetric", smaData.length ? money(smaData.at(-1).value, 2) : "-");
        const source = lastRaw?.source_name || "";
        const suffix = source === "aggregated_from_1m" ? " · built from 1m" : (source === "1m_live_fallback" ? " · 1m live fallback" : "");
        setText("chartStatus", `${candles.length} ${requestedTimeframe} candles${suffix}`);
        setClass("chartStatus", source === "1m_live_fallback" ? "warning" : "");
      }
    } catch (error) {
      setText("chartStatus", `Chart error: ${error.message}`);
      setClass("chartStatus", "warning");
    }
  }

  async function refreshSummary() {
    try {
      const data = await api("/api/dashboard/summary");
      setText("marketStatus", data.market?.stale ? "Stale" : (data.collectors?.market?.running ? "Running" : "Ready"));
      setText("newsStatus", data.collectors?.news?.running ? "Running" : "Ready");
      setText("autoStatus", data.auto_trader?.running ? "Running" : "Stopped");
      setText("modelStatus", data.model?.version || "untrained");
      setText("sentimentModel", data.sentiment_model?.active_model || "-");
      setText("lastUpdate", new Date(data.server_time).toLocaleTimeString());

      setText("cashMetric", money(data.account?.cash_balance));
      setText("equityMetric", money(data.account?.equity));
      const pnl = Number(data.account?.equity || 0) - Number(data.paper_start_balance || 10000);
      setText("pnlMetric", money(pnl));
      setClass("pnlMetric", cls(pnl));
      setText("candleMetric", number(data.counts?.candles, 0));
      setText("newsMetric", number(data.counts?.news, 0));
      setText("experienceMetric", number(data.counts?.experiences, 0));
      const coverage = data.coverage || {};
      setText("coverageCandles", number(coverage.candles, 0));
      setText("coverageNews", number(coverage.news, 0));
      setText("coverageSentiment", number(coverage.sentiment, 0));
      setText("coverageDerivatives", number(coverage.derivatives, 0));
      setText("coverageLiquidations", number(coverage.liquidations, 0));
      setText("coverageFearGreed", number(coverage.fear_greed, 0));
      setText("coverageGlobalMarket", number(coverage.global_market, 0));
      setText("coverageStablecoinRisk", number(coverage.stablecoin_risk, 0));
      setText("coverageMacroRisk", number(coverage.macro_risk, 0));
      setText("coverageLabeledRows", `${number(coverage.labeled_rows, 0)} / ${pct(coverage.label_coverage_pct || 0, 1)}`);
      setClass("coverageLabeledRows", Number(coverage.labeled_rows || 0) > 0 ? "positive" : "warning");

      setText("featureCount", number(data.counts?.features, 0));
      setText("trainingCandleCount", number(data.counts?.candles, 0));
      setText("trainingNewsCount", number(data.counts?.news, 0));
      setText("trainingExperienceCount", number(data.counts?.experiences, 0));
      setText("trainingModelVersion", data.model?.version || "untrained");
      setText("trainingModelStatus", data.model?.status || "missing");
      setText("candidateModelCount", number(data.model?.candidate_count, 0));
      setText("featureSchema", data.model?.feature_schema_version || "-");
      setText("modelFallbackReason", data.auto_trader?.model_fallback_reason || "-");
      const trainingWorker = data.training?.worker || {};
      if (trainingWorker.running) {
        setText("lastTrainingRun", `running since ${when(trainingWorker.started_at)}`);
      } else if (trainingWorker.status && trainingWorker.status !== "idle") {
        setText("lastTrainingRun", `${trainingWorker.status} ${trainingWorker.finished_at ? when(trainingWorker.finished_at) : ""}`);
      } else {
        setText("lastTrainingRun", data.training?.last_run_at ? when(data.training.last_run_at) : "-");
      }

      setText("marketDiagnostics", JSON.stringify(data.market, null, 2));
      setText("newsDiagnostics", JSON.stringify(data.news?.providers || data.news, null, 2));
      setText("derivativesDiagnostics", JSON.stringify(data.derivatives || {}, null, 2));
      setText("externalDiagnostics", JSON.stringify(data.external || {}, null, 2));
      setText("autoDiagnostics", JSON.stringify(data.auto_trader, null, 2));
      const trading = data.trading || {};
      setText("strategyTradeCount", number(trading.strategy_trades, 0));
      setText("explorationTradeCount", number(trading.exploration_trades, 0));
      setText("skippedTradeCount", number(trading.skipped_trades, 0));
      const lastAction = trading.last_strategy_action || {};
      setText("lastStrategyAction", lastAction.action ? `${lastAction.action} · ${lastAction.status || "-"}` : "-");
      setText("tradeWarning", trading.latest_warning || "");
      setText("lastStrategyReason", lastAction.reason ? `Reason: ${lastAction.reason}` : "");
      setText("holdReasons", (trading.hold_reasons || []).map((item) => `${when(item.time)} ${item.symbol} ${item.action}/${item.status}: ${item.reason || "-"}`).join("\n"));
      setText("tradeFlowMode", data.auto_trader?.exploration_enabled ? `explore ${pct(data.auto_trader?.exploration_rate || 0, 1)}` : "strategy");
    } catch (error) {
      setText("marketStatus", "Error");
      setClass("marketStatus", "warning");
    }
  }

  async function refreshPositions() {
    const body = $("positionsBody");
    if (!body) return;
    const rows = await api("/api/positions");
    body.innerHTML = rows.length ? rows.filter((row) => row.status === "OPEN").map((row) => `
      <tr>
        <td>${escapeHtml(row.symbol)}</td>
        <td>${escapeHtml(row.side)}</td>
        <td>${number(row.leverage, 1)}x</td>
        <td>${money(row.margin_used)}</td>
        <td>${number(row.quantity, 6)}</td>
        <td>${money(row.entry_price)}</td>
        <td>${money(row.current_price)}</td>
        <td class="${cls(row.unrealized_pnl)}">${money(row.unrealized_pnl)}</td>
        <td class="${cls(row.unrealized_pnl_pct)}">${pct(row.unrealized_pnl_pct)}</td>
        <td>${money(row.notional)}</td>
        <td>${row.stop_loss ? money(row.stop_loss) : "-"}</td>
        <td>${row.take_profit ? money(row.take_profit) : "-"}</td>
        <td>${when(row.opened_at)}</td>
        <td><button data-close-position="${escapeHtml(row.symbol)}">Close</button></td>
      </tr>`).join("") : `<tr><td colspan="14" class="empty">No open positions</td></tr>`;
    body.querySelectorAll("[data-close-position]").forEach((button) => {
      button.addEventListener("click", () => sendSignal({ symbol: button.dataset.closePosition, action: "CLOSE", confidence: 0.9, source: "dashboard-close" }));
    });
  }

  async function refreshTrades() {
    const rows = await api("/api/trades?limit=50");
    $("tradesBody").innerHTML = rows.map((row) => `
      <tr><td>${when(row.created_at)}</td><td>${escapeHtml(row.symbol)}</td><td>${escapeHtml(row.action)}</td><td>${escapeHtml(row.side || "-")}</td><td>${number(row.quantity, 6)}</td><td>${money(row.price)}</td><td>${money(row.fee, 4)}</td><td class="${cls(row.realized_pnl)}">${money(row.realized_pnl)}</td><td>${escapeHtml(row.reason || "-")}</td></tr>
    `).join("") || `<tr><td colspan="9" class="empty">No trades</td></tr>`;
  }

  async function refreshDecisions() {
    const rows = await api("/api/ai-decisions?limit=50");
    $("decisionsBody").innerHTML = rows.map((row) => `
      <tr><td>${when(row.time)}</td><td>${escapeHtml(row.symbol)}</td><td>${escapeHtml(row.action)}</td><td>${escapeHtml(row.decision_source || "-")}</td><td>${escapeHtml(row.execution_status || "-")}</td><td>${pct(row.confidence, 1)}</td><td>${number(row.sentiment_score, 3)}</td><td>${number(row.risk_score, 3)}</td><td>${escapeHtml(row.strategy)}</td><td>${number(row.reward, 4)}</td><td>${escapeHtml(row.reason || "-")}</td></tr>
    `).join("") || `<tr><td colspan="11" class="empty">No decisions</td></tr>`;
  }

  async function refreshNews() {
    const provider = $("newsProviderFilter")?.value || "";
    const rows = await api(`/api/news/latest?limit=50${provider ? `&provider=${encodeURIComponent(provider)}` : ""}`);
    $("newsBody").innerHTML = rows.map((row) => `
      <tr><td>${when(row.published_at)}</td><td>${escapeHtml(row.source)}</td><td>${escapeHtml(row.provider)}</td><td>${escapeHtml(row.title)}</td><td>${escapeHtml((row.affected_symbols || []).join(", "))}</td><td class="${cls(row.sentiment_score)}">${number(row.sentiment_score, 2)}</td><td>${number(row.risk_score, 2)}</td><td><a href="${escapeHtml(row.url)}" target="_blank" rel="noreferrer">Open</a></td></tr>
    `).join("") || `<tr><td colspan="8" class="empty">No news</td></tr>`;
  }

  async function refreshSentiment() {
    const rows = await api("/api/sentiment/latest?limit=50");
    $("sentimentBody").innerHTML = rows.map((row) => `
      <tr><td>${when(row.time)}</td><td>${escapeHtml(row.model_name || "-")}</td><td>${escapeHtml(row.label || "-")}</td><td>${pct(row.confidence, 1)}</td><td class="${cls(row.sentiment_score)}">${number(row.sentiment_score, 2)}</td><td>${number(row.risk_score, 2)}</td><td>${escapeHtml((row.topics || []).join(", "))}</td><td>${escapeHtml(row.title)}</td></tr>
    `).join("") || `<tr><td colspan="8" class="empty">No sentiment</td></tr>`;
  }

  async function refreshFeatures() {
    const body = $("featureVectorBody");
    if (!body) return;
    try {
      const data = await api(`/api/features/latest?symbol=${encodeURIComponent(state.symbol)}`);
      const vector = data.vector || {};
      setText("featureInspectorTitle", `AI Feature Inspector · ${data.symbol || state.symbol}`);
      setText("featureInspectorStatus", `${data.schema_version || "-"} · ${data.persisted ? "stored" : "preview"}`);
      setClass("featureInspectorStatus", "");
      body.innerHTML = featureFields.map(([key, label]) => `
        <tr><td>${escapeHtml(label)}</td><td>${featureValue(vector[key])}</td></tr>
      `).join("");

      const finalInput = data.final_ai_input || {};
      const json = $("featureJson");
      if (json) json.textContent = JSON.stringify(finalInput, null, 2);

      const context = data.news_context || [];
      const newsBox = $("featureNewsContext");
      if (newsBox) {
        newsBox.innerHTML = context.length ? context.map((item) => `
          <div class="news-context-item">
            <strong>${escapeHtml(item.title || "-")}</strong>
            <p>${escapeHtml(item.text || "")}</p>
            <span>${escapeHtml(item.provider || "-")} / ${escapeHtml(item.source || "-")} / ${when(item.published_at)}</span>
            <span>sentiment ${featureValue(item.sentiment_score)} · confidence ${featureValue(item.sentiment_confidence)} · risk ${featureValue(item.risk_score)}</span>
          </div>
        `).join("") : `<div class="empty">No recent news context used for this symbol</div>`;
      }

      const derivativesContext = data.derivatives_context || [];
      const derivativesBox = $("featureDerivativesContext");
      if (derivativesBox) {
        const externalContext = data.external_context || [];
        const mergedContext = [...derivativesContext, ...externalContext];
        derivativesBox.innerHTML = mergedContext.length ? mergedContext.map((item) => `
          <div class="news-context-item">
            <strong>${escapeHtml(item.data_type || "-")}</strong>
            <p>${escapeHtml(JSON.stringify(item.payload || {}))}</p>
            <span>${escapeHtml(item.source_name || "-")} / ${when(item.event_time)}</span>
            <span>value ${featureValue(item.numeric_value)}</span>
          </div>
        `).join("") : `<div class="empty">No trader-flow context collected for this symbol</div>`;
      }
    } catch (error) {
      setText("featureInspectorStatus", `Error: ${error.message}`);
      setClass("featureInspectorStatus", "warning");
    }
  }

  async function refreshDbDiagnostics() {
    const output = $("dbDiagnostics");
    if (!output) return;
    try {
      const data = await api("/api/db/diagnostics");
      const lifecycle = await api("/api/db/lifecycle/status");
      output.textContent = JSON.stringify({ ...data, lifecycle }, null, 2);
    } catch (error) {
      output.textContent = `DB diagnostics error: ${error.message}`;
    }
  }

  async function refreshStorageDiagnostics() {
    const body = $("dbStorageBody");
    if (!body) return;
    try {
      const data = await api("/api/db/storage");
      const lifecycle = await api("/api/db/lifecycle/status");
      const total = data.database_total || {};
      const largest = (data.top_largest_tables || [])[0] || {};
      setText("dbTotalSize", total.mb !== null && total.mb !== undefined ? mb(total.mb) : "row counts only");
      setText("dbMonthlyCost", Number.isFinite(Number(data.estimated_monthly_storage_cost_usd)) ? `$${Number(data.estimated_monthly_storage_cost_usd).toFixed(4)}` : "-");
      setText("dbLargestTable", largest.table ? `${largest.table} (${number(largest.rows, 0)} rows)` : "-");
      setText("dbLastCleanup", lifecycle.last_run_at ? when(lifecycle.last_run_at) : "-");
      const retention = lifecycle.retention || {};
      setText("dbRetentionSummary", `live ${retention.live_update_retention_hours ?? "-"}h / raw external ${retention.raw_external_event_retention_days ?? "-"}d / candles ${retention.keep_closed_candles_days ?? retention.closed_candle_retention_days ?? "-"}d`);
      setText("dbStorageWarning", total.warning || "");
      const rows = data.rows_by_table || [];
      body.innerHTML = rows.length ? rows
        .slice()
        .sort((a, b) => Number(b.bytes || b.raw_payload_estimate?.bytes || 0) - Number(a.bytes || a.raw_payload_estimate?.bytes || 0))
        .map((row) => `
          <tr>
            <td>${escapeHtml(row.table)}</td>
            <td>${number(row.rows, 0)}</td>
            <td>${row.mb !== null && row.mb !== undefined ? number(row.mb, 3) : "-"}</td>
            <td>${number(row.raw_payload_estimate?.mb || 0, 3)}</td>
            <td>${when(row.oldest)}</td>
            <td>${when(row.newest)}</td>
          </tr>
        `).join("") : `<tr><td colspan="6" class="empty">No tables found</td></tr>`;
    } catch (error) {
      body.innerHTML = `<tr><td colspan="6" class="empty">Storage diagnostics failed: ${escapeHtml(error.message)}</td></tr>`;
    }
  }

  async function refreshModels() {
    const body = $("modelsBody");
    if (!body) return;
    try {
      const rows = await api("/api/models?limit=50");
      body.innerHTML = rows.length ? rows.map((row) => {
        const metrics = row.metrics || {};
        const metricText = Object.entries(metrics).slice(0, 5).map(([key, value]) => {
          const safeValue = typeof value === "object" ? JSON.stringify(value) : value;
          return `${key}: ${Number.isFinite(Number(safeValue)) ? number(safeValue, 4) : escapeHtml(safeValue)}`;
        }).join(" / ");
        const action = row.status === "candidate"
          ? `<button data-activate-model="${escapeHtml(row.model_id)}" class="primary">Activate</button>`
          : "-";
        return `
          <tr>
            <td>${escapeHtml(row.status || "-")}</td>
            <td>${escapeHtml(row.model_id || "-")}</td>
            <td>${escapeHtml(row.version || "-")}</td>
            <td>${escapeHtml(row.model_type || row.name || "-")}</td>
            <td>${escapeHtml(row.feature_schema_version || "-")}</td>
            <td>${metricText || "-"}</td>
            <td>${when(row.created_at)}</td>
            <td>${action}</td>
          </tr>`;
      }).join("") : `<tr><td colspan="8" class="empty">No uploaded models yet</td></tr>`;
      body.querySelectorAll("[data-activate-model]").forEach((button) => {
        button.addEventListener("click", () => activateModel(button.dataset.activateModel));
      });
    } catch (error) {
      body.innerHTML = `<tr><td colspan="8" class="empty">Model load failed: ${escapeHtml(error.message)}</td></tr>`;
    }
  }

  async function refreshLabelStatus() {
    try {
      const data = await api("/api/training/label-status");
      const total = Number(data.total_training_feature_rows || data.total_training_features || 0);
      const labeled = Number(data.rows_with_target_trade_quality_score || 0);
      setText("labelCoverage", `${number(labeled, 0)} / ${number(total, 0)} (${pct(data.label_coverage_pct || 0, 1)})`);
      setText("labelFeatureRows", number(data.total_feature_rows, 0));
      setText("labelLabeledRows", number(labeled, 0));
      setText("labelUnlabeledRows", number(data.unlabeled_rows, 0));
      setText("latestLabeledAt", when(data.latest_labeled_as_of));
      setText("trainingTarget", data.selected_training_target || "-");
      setText("labelReadiness", data.label_readiness || "NOT READY");
      setClass("labelReadiness", data.label_readiness === "OK" ? "positive" : "warning");
      setText("labelWarning", data.warning || (labeled === 0 && total > 0 ? "Label coverage is 0. Build labels or wait for future closed candles." : ""));
    } catch (error) {
      setText("labelReadiness", `Error: ${error.message}`);
      setClass("labelReadiness", "warning");
    }
  }

  async function activateModel(modelId) {
    const output = $("modelActionResult") || $("exportResult");
    if (output) output.textContent = `Activating ${modelId}...`;
    try {
      const data = await api("/api/models/activate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_id: modelId }),
      });
      if (output) output.textContent = `Activated ${data.model?.model_id || modelId}`;
      await refreshModels();
      await refreshSummary();
    } catch (error) {
      if (output) output.textContent = `Activate failed: ${error.message}`;
    }
  }

  async function refreshHeavy() {
    try {
      await Promise.all([refreshPositions(), refreshTrades(), refreshDecisions(), refreshNews(), refreshSentiment(), refreshFeatures(), refreshDbDiagnostics(), refreshModels(), refreshLabelStatus()]);
    } catch (error) {
      console.warn("table refresh failed", error);
    }
  }

  async function collectorAction(worker, action) {
    await api(`/api/collectors/${worker}/${action}`, { method: "POST" });
    await refreshSummary();
  }

  async function autoTraderAction(action) {
    await api(`/api/auto-trader/${action}`, { method: "POST" });
    await refreshSummary();
  }

  async function sendSignal(payload) {
    const response = await api("/api/signal", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    setText("signalResult", `${response.status}: ${response.message}`);
    await refreshHeavy();
    await refreshSummary();
  }

  function submitSignal(event) {
    event.preventDefault();
    const formData = new FormData(event.currentTarget);
    const payload = Object.fromEntries(formData.entries());
    ["confidence", "notional", "stop_loss", "take_profit"].forEach((key) => {
      if (payload[key] !== "") payload[key] = Number(payload[key]);
      else delete payload[key];
    });
    payload.source = "dashboard-manual";
    sendSignal(payload).catch((error) => setText("signalResult", `Error: ${error.message}`));
  }

  async function exportDataset(outputId = "exportResult") {
    const output = $(outputId) || $("exportResult") || $("dbActionResult");
    if (output) output.textContent = "Exporting...";
    try {
      const data = await api("/api/training/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ use_all_data: true }),
      });
      if (output) output.textContent = JSON.stringify(data, null, 2);
    } catch (error) {
      if (output) output.textContent = `Export failed: ${error.message}`;
    }
  }

  async function buildTrainingDataset() {
    const output = $("exportResult") || $("dbActionResult");
    if (output) output.textContent = "Building accelerated dataset. This can take a little while...";
    try {
      const data = await api("/api/training/build-dataset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          days: 14,
          max_rows_per_symbol: 5000,
          lookback: 60,
          stride: 5,
          replay_limit: 20000,
          backfill: true,
          export: true,
        }),
      });
      if (output) output.textContent = JSON.stringify(data, null, 2);
      await refreshSummary();
      await refreshFeatures();
      await refreshDbDiagnostics();
    } catch (error) {
      if (output) output.textContent = `Build dataset failed: ${error.message}`;
    }
  }

  async function buildLabels() {
    const output = $("exportResult") || $("dbActionResult");
    if (output) output.textContent = "Building labels from future closed candles...";
    try {
      const data = await api("/api/training/build-labels", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      if (output) output.textContent = JSON.stringify(data, null, 2);
      await refreshLabelStatus();
      await refreshSummary();
    } catch (error) {
      if (output) output.textContent = `Build labels failed: ${error.message}`;
    }
  }

  async function trainModel() {
    const output = $("exportResult") || $("dbActionResult");
    if (output) output.textContent = "Training model from compact features. This can take a little while...";
    try {
      const data = await api("/api/training/train-model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          build_dataset: true,
          days: 14,
          max_rows_per_symbol: 5000,
          lookback: 60,
          stride: 5,
          replay_limit: 20000,
          backfill: true,
          use_all_data: true,
        }),
      });
      if (output) output.textContent = data.status === "started" ? "Training started in background. You can keep using the dashboard." : JSON.stringify(data, null, 2);
      await refreshSummary();
      await refreshFeatures();
      await refreshDbDiagnostics();
    } catch (error) {
      if (output) output.textContent = `Train model failed: ${error.message}`;
    }
  }

  async function runCleanup() {
    const output = $("dbActionResult");
    if (output) output.textContent = "Cleaning...";
    try {
      const data = await api("/api/db/cleanup", { method: "POST" });
      if (output) output.textContent = `Cleanup done: ${JSON.stringify(data.last_cleanup?.deleted || {})}`;
      await refreshDbDiagnostics();
      await refreshStorageDiagnostics();
      await refreshSummary();
    } catch (error) {
      if (output) output.textContent = `Cleanup failed: ${error.message}`;
    }
  }

  async function compactDatabase(archiveBeforeDelete = false) {
    const output = $("dbStorageActionResult") || $("dbActionResult");
    if (output) output.textContent = archiveBeforeDelete ? "Archiving and compacting..." : "Compacting...";
    try {
      const data = await api("/api/db/compact", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ archive_before_delete: archiveBeforeDelete }),
      });
      const cleanup = data.last_cleanup || {};
      if (output) output.textContent = `Compact done. Deleted ${JSON.stringify(cleanup.deleted || {})}; compacted ${JSON.stringify(cleanup.compacted || {})}`;
      await refreshDbDiagnostics();
      await refreshStorageDiagnostics();
      await refreshSummary();
    } catch (error) {
      if (output) output.textContent = `Compact failed: ${error.message}`;
    }
  }

  async function archiveData() {
    const output = $("dbActionResult");
    if (output) output.textContent = "Archiving...";
    try {
      const data = await api("/api/db/archive", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tables: ["candles", "training_features", "experience_buffer"], delete_after_archive: false }),
      });
      if (output) output.textContent = `Archive done: ${(data.exports || []).map((item) => item.path).join(", ") || "no rows"}`;
      await refreshDbDiagnostics();
      await refreshStorageDiagnostics();
    } catch (error) {
      if (output) output.textContent = `Archive failed: ${error.message}`;
    }
  }

  async function reprocessSentiment() {
    const output = $("sentimentActionResult");
    if (output) output.textContent = "Reprocessing...";
    try {
      const data = await api("/api/sentiment/reprocess", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ limit: 200, reset_model: true }),
      });
      const model = data.active_model || {};
      if (output) output.textContent = `${data.processed} articles · ${model.active_model || "-"}${model.hf_last_error ? ` · ${model.hf_last_error}` : ""}`;
      await refreshSentiment();
      await refreshFeatures();
      await refreshSummary();
    } catch (error) {
      if (output) output.textContent = `Reprocess failed: ${error.message}`;
    }
  }

  function wireEvents() {
    function requestChartReload(message) {
      state.forceFit = true;
      state.lastCandlesKey = "";
      state.lastSeriesKey = "";
      setText("chartStatus", message);
      setClass("chartStatus", "");
    }

    $("symbolSelect")?.addEventListener("change", (event) => {
      state.symbol = event.target.value;
      $("signalSymbol").value = state.symbol;
      document.querySelectorAll("[data-market-symbol]").forEach((item) => item.classList.toggle("active", item.dataset.marketSymbol === state.symbol));
      requestChartReload(`Loading ${state.symbol} ${state.timeframe}`);
      refreshChart();
      refreshFeatures();
    });
    document.querySelectorAll("[data-market-symbol]").forEach((button) => {
      button.addEventListener("click", () => {
        state.symbol = button.dataset.marketSymbol;
        $("symbolSelect").value = state.symbol;
        $("signalSymbol").value = state.symbol;
        document.querySelectorAll("[data-market-symbol]").forEach((item) => item.classList.toggle("active", item === button));
        requestChartReload(`Loading ${state.symbol} ${state.timeframe}`);
        refreshChart();
        refreshFeatures();
      });
    });
    $("newsProviderFilter")?.addEventListener("change", refreshNews);
    $("signalForm")?.addEventListener("submit", submitSignal);
    $("buildDatasetButton")?.addEventListener("click", buildTrainingDataset);
    $("buildLabelsButton")?.addEventListener("click", buildLabels);
    $("trainModelButton")?.addEventListener("click", trainModel);
    $("exportDatasetButton")?.addEventListener("click", () => exportDataset("exportResult"));
    $("exportLabeledDatasetButton")?.addEventListener("click", () => exportDataset("exportResult"));
    $("refreshModelsButton")?.addEventListener("click", refreshModels);
    $("exportDatasetDiagnosticsButton")?.addEventListener("click", () => exportDataset("dbActionResult"));
    $("runCleanupButton")?.addEventListener("click", runCleanup);
    $("compactDbButton")?.addEventListener("click", () => compactDatabase(false));
    $("compactArchiveDbButton")?.addEventListener("click", () => compactDatabase(true));
    $("archiveDataButton")?.addEventListener("click", archiveData);
    $("reprocessSentimentButton")?.addEventListener("click", reprocessSentiment);
    document.querySelectorAll("[data-worker]").forEach((button) => button.addEventListener("click", () => collectorAction(button.dataset.worker, button.dataset.action)));
    document.querySelectorAll("[data-auto-trader]").forEach((button) => button.addEventListener("click", () => autoTraderAction(button.dataset.autoTrader)));
    document.querySelectorAll("[data-tab]").forEach((button) => {
      button.addEventListener("click", () => {
        document.querySelectorAll("[data-tab]").forEach((item) => item.classList.toggle("active", item === button));
        document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.toggle("active", panel.id === `tab-${button.dataset.tab}`));
        if (button.dataset.tab === "storage") refreshStorageDiagnostics();
      });
    });
    document.querySelectorAll("[data-timeframe]").forEach((button) => {
      button.addEventListener("click", () => {
        document.querySelectorAll("[data-timeframe]").forEach((item) => item.classList.toggle("active", item === button));
        state.timeframe = button.dataset.timeframe;
        requestChartReload(`Loading ${state.symbol} ${state.timeframe}`);
        refreshChart();
      });
    });
    window.addEventListener("resize", () => {
      const element = $("chart");
      if (state.chart && element) state.chart.resize(element.clientWidth, element.clientHeight);
    });
  }

  async function tickFast() {
    await Promise.all([refreshSummary(), refreshChart()]);
  }

  function start() {
    wireEvents();
    setTimeout(setupChart, 300);
    tickFast();
    refreshHeavy();
    refreshStorageDiagnostics();
    setInterval(tickFast, 1000);
    setInterval(refreshHeavy, 6000);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
