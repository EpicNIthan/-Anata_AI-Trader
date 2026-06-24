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
  };

  const $ = (id) => document.getElementById(id);
  const money = (value, digits = 2) => Number.isFinite(Number(value)) ? `$${Number(value).toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits })}` : "-";
  const number = (value, digits = 2) => Number.isFinite(Number(value)) ? Number(value).toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits }) : "-";
  const pct = (value, digits = 2) => Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(digits)}%` : "-";
  const when = (value) => value ? new Date(value).toLocaleString() : "-";
  const cls = (value) => Number(value) >= 0 ? "positive" : "negative";
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
    state.chart = LightweightCharts.createChart(element, {
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
    try {
      const candles = await api(`/api/market/candles?symbol=${encodeURIComponent(state.symbol)}&timeframe=${encodeURIComponent(state.timeframe)}&limit=300`);
      const key = `${state.symbol}:${state.timeframe}:${candles.length}:${candles.at(-1)?.open_time || ""}:${candles.at(-1)?.close || ""}`;
      if (key !== state.lastCandlesKey && candles.length) {
        const candleData = candles.filter((item) => item.time).map((item) => ({
          time: item.time,
          open: Number(item.open),
          high: Number(item.high),
          low: Number(item.low),
          close: Number(item.close),
        }));
        const volumeData = candles.filter((item) => item.time).map((item) => ({
          time: item.time,
          value: Number(item.volume || 0),
          color: Number(item.close) >= Number(item.open) ? "rgba(14, 203, 129, 0.35)" : "rgba(246, 70, 93, 0.35)",
        }));
        if (state.candleSeries) state.candleSeries.setData(candleData);
        if (state.volumeSeries) state.volumeSeries.setData(volumeData);
        const smaData = sma(candleData);
        if (state.smaSeries) state.smaSeries.setData(smaData);
        if (state.chart) state.chart.timeScale().fitContent();
        state.lastCandlesKey = key;
        const last = candles.at(-1);
        const first = candles[0];
        const change = first && first.close ? (Number(last.close) - Number(first.close)) / Number(first.close) : 0;
        setText("latestPrice", money(last.close, 2));
        setText("priceChange", pct(change, 2));
        setClass("priceChange", cls(change));
        setText(`railPrice${state.symbol}`, money(last.close, 2));
        setText(`railChange${state.symbol}`, pct(change, 2));
        setClass(`railChange${state.symbol}`, cls(change));
        updateBook(last.close);
        setText("volumeMetric", number(last.volume, 2));
        setText("smaMetric", smaData.length ? money(smaData.at(-1).value, 2) : "-");
        setText("chartStatus", `${candles.length} candles`);
      } else if (!candles.length) {
        setText("chartStatus", `No ${state.timeframe} candles for ${state.symbol}`);
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

      setText("featureCount", number(data.counts?.features, 0));
      setText("trainingCandleCount", number(data.counts?.candles, 0));
      setText("trainingNewsCount", number(data.counts?.news, 0));
      setText("trainingExperienceCount", number(data.counts?.experiences, 0));
      setText("trainingModelVersion", data.model?.version || "untrained");
      setText("featureSchema", data.model?.feature_schema_version || "-");
      setText("lastTrainingRun", data.training?.last_run_at ? when(data.training.last_run_at) : "-");

      setText("marketDiagnostics", JSON.stringify(data.market, null, 2));
      setText("newsDiagnostics", JSON.stringify(data.news?.providers || data.news, null, 2));
      setText("autoDiagnostics", JSON.stringify(data.auto_trader, null, 2));
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
      </tr>`).join("") : `<tr><td colspan="12" class="empty">No open positions</td></tr>`;
    body.querySelectorAll("[data-close-position]").forEach((button) => {
      button.addEventListener("click", () => sendSignal({ symbol: button.dataset.closePosition, action: "CLOSE", confidence: 0.9, source: "dashboard-close" }));
    });
  }

  async function refreshTrades() {
    const rows = await api("/api/trades?limit=50");
    $("tradesBody").innerHTML = rows.map((row) => `
      <tr><td>${when(row.created_at)}</td><td>${escapeHtml(row.symbol)}</td><td>${escapeHtml(row.action)}</td><td>${number(row.quantity, 6)}</td><td>${money(row.price)}</td><td>${money(row.fee, 4)}</td><td class="${cls(row.realized_pnl)}">${money(row.realized_pnl)}</td><td>${escapeHtml(row.reason || "-")}</td></tr>
    `).join("") || `<tr><td colspan="8" class="empty">No trades</td></tr>`;
  }

  async function refreshDecisions() {
    const rows = await api("/api/ai-decisions?limit=50");
    $("decisionsBody").innerHTML = rows.map((row) => `
      <tr><td>${when(row.time)}</td><td>${escapeHtml(row.symbol)}</td><td>${escapeHtml(row.action)}</td><td>${pct(row.confidence, 1)}</td><td>${number(row.sentiment_score, 3)}</td><td>${number(row.risk_score, 3)}</td><td>${escapeHtml(row.strategy)}</td><td>${number(row.reward, 4)}</td><td>${escapeHtml(row.reason || "-")}</td></tr>
    `).join("") || `<tr><td colspan="9" class="empty">No decisions</td></tr>`;
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

  async function refreshHeavy() {
    try {
      await Promise.all([refreshPositions(), refreshTrades(), refreshDecisions(), refreshNews(), refreshSentiment()]);
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

  async function exportDataset() {
    const output = $("exportResult");
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

  function wireEvents() {
    $("symbolSelect")?.addEventListener("change", (event) => {
      state.symbol = event.target.value;
      $("signalSymbol").value = state.symbol;
      document.querySelectorAll("[data-market-symbol]").forEach((item) => item.classList.toggle("active", item.dataset.marketSymbol === state.symbol));
      state.lastCandlesKey = "";
      refreshChart();
    });
    document.querySelectorAll("[data-market-symbol]").forEach((button) => {
      button.addEventListener("click", () => {
        state.symbol = button.dataset.marketSymbol;
        $("symbolSelect").value = state.symbol;
        $("signalSymbol").value = state.symbol;
        document.querySelectorAll("[data-market-symbol]").forEach((item) => item.classList.toggle("active", item === button));
        state.lastCandlesKey = "";
        refreshChart();
      });
    });
    $("newsProviderFilter")?.addEventListener("change", refreshNews);
    $("signalForm")?.addEventListener("submit", submitSignal);
    $("exportDatasetButton")?.addEventListener("click", exportDataset);
    document.querySelectorAll("[data-worker]").forEach((button) => button.addEventListener("click", () => collectorAction(button.dataset.worker, button.dataset.action)));
    document.querySelectorAll("[data-auto-trader]").forEach((button) => button.addEventListener("click", () => autoTraderAction(button.dataset.autoTrader)));
    document.querySelectorAll("[data-tab]").forEach((button) => {
      button.addEventListener("click", () => {
        document.querySelectorAll("[data-tab]").forEach((item) => item.classList.toggle("active", item === button));
        document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.toggle("active", panel.id === `tab-${button.dataset.tab}`));
      });
    });
    document.querySelectorAll("[data-timeframe]").forEach((button) => {
      button.addEventListener("click", () => {
        document.querySelectorAll("[data-timeframe]").forEach((item) => item.classList.toggle("active", item === button));
        state.timeframe = button.dataset.timeframe;
        state.lastCandlesKey = "";
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
    setInterval(tickFast, 1000);
    setInterval(refreshHeavy, 6000);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
