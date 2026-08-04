(function () {
  "use strict";

  const config = window.VISION_CONFIG || {};
  const defaultLimit = Math.max(Number(config.defaultLimit || 250), 1);
  const state = {
    symbol: config.defaultSymbol || "BTCUSDT",
    range: "24h",
    chart: null,
    candleSeries: null,
    volumeSeries: null,
    forecastSeries: null,
    upperSeries: null,
    lowerSeries: null,
    priceLines: [],
    candles: [],
    refreshing: false,
    selectedTraceId: null,
    lastState: null,
  };

  const byId = (id) => document.getElementById(id);
  const finite = (value) => Number.isFinite(Number(value));
  const number = (value, digits) => finite(value)
    ? Number(value).toLocaleString(undefined, { minimumFractionDigits: digits ?? 2, maximumFractionDigits: digits ?? 2 })
    : "-";
  const money = (value, digits) => finite(value)
    ? "$" + Number(value).toLocaleString(undefined, { minimumFractionDigits: digits ?? 2, maximumFractionDigits: digits ?? 2 })
    : "-";
  const percent = (value, digits) => finite(value) ? (Number(value) * 100).toFixed(digits ?? 2) + "%" : "-";
  const timeText = (value) => value ? new Date(value).toLocaleString() : "-";
  const html = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  })[char]);

  function setText(id, value) {
    const element = byId(id);
    if (element) element.textContent = value ?? "-";
  }

  function notice(message, tone) {
    const element = byId("visionNotice");
    if (!element) return;
    element.textContent = message;
    element.classList.remove("is-warning", "is-error");
    if (tone) element.classList.add(tone);
  }

  function queryPath(name, params) {
    const query = new URLSearchParams();
    Object.entries(params || {}).forEach(([key, value]) => {
      if (value !== null && value !== undefined && value !== "") query.set(key, String(value));
    });
    const suffix = query.toString();
    return (config.apiBase || "/api/vision") + "/" + name + (suffix ? "?" + suffix : "");
  }

  async function request(name, params) {
    const response = await fetch(queryPath(name, params), { credentials: "same-origin" });
    if (!response.ok) {
      let detail = "";
      try {
        const payload = await response.json();
        detail = payload.detail ? ": " + payload.detail : "";
      } catch (_) {
        // Some proxies return HTML error pages.
      }
      throw new Error(response.status + " " + response.statusText + detail);
    }
    return response.json();
  }

  function rangeParams() {
    const span = {
      "6h": 6 * 60 * 60 * 1000,
      "24h": 24 * 60 * 60 * 1000,
      "7d": 7 * 24 * 60 * 60 * 1000,
    }[state.range];
    if (!span) return {};
    const end = new Date();
    return { start: new Date(end.getTime() - span).toISOString(), end: end.toISOString() };
  }

  function age(seconds) {
    if (!finite(seconds)) return "No timestamp";
    const value = Math.max(Number(seconds), 0);
    if (value < 60) return Math.round(value) + "s old";
    if (value < 3600) return Math.round(value / 60) + "m old";
    if (value < 86400) return (value / 3600).toFixed(1) + "h old";
    return (value / 86400).toFixed(1) + "d old";
  }

  function chartReady() {
    return Boolean(window.LightweightCharts && window.LightweightCharts.createChart);
  }

  function setupChart() {
    const element = byId("visionChart");
    if (!element || state.chart) return Boolean(state.chart);
    if (!chartReady()) return false;
    const charts = window.LightweightCharts;
    state.chart = charts.createChart(element, {
      width: Math.max(element.clientWidth || 0, 320),
      height: Math.max(element.clientHeight || 0, 350),
      layout: { background: { color: "transparent" }, textColor: "#8da1b3" },
      grid: { vertLines: { color: "#16232e" }, horzLines: { color: "#16232e" } },
      rightPriceScale: { borderColor: "#23313e" },
      timeScale: { borderColor: "#23313e", timeVisible: true, secondsVisible: false },
      crosshair: { mode: charts.CrosshairMode?.Normal || 0 },
    });
    state.candleSeries = state.chart.addCandlestickSeries({
      upColor: "#36d399",
      downColor: "#fb7185",
      borderUpColor: "#36d399",
      borderDownColor: "#fb7185",
      wickUpColor: "#36d399",
      wickDownColor: "#fb7185",
    });
    state.volumeSeries = state.chart.addHistogramSeries({
      priceFormat: { type: "volume" },
      priceScaleId: "",
      scaleMargins: { top: 0.78, bottom: 0 },
    });
    const dashed = charts.LineStyle?.Dashed ?? 2;
    const dotted = charts.LineStyle?.Dotted ?? 1;
    state.forecastSeries = state.chart.addLineSeries({
      color: "#54d4e7", lineWidth: 2, lineStyle: dashed, lastValueVisible: false, priceLineVisible: false,
    });
    state.upperSeries = state.chart.addLineSeries({
      color: "rgba(84, 212, 231, 0.42)", lineWidth: 1, lineStyle: dotted, lastValueVisible: false, priceLineVisible: false,
    });
    state.lowerSeries = state.chart.addLineSeries({
      color: "rgba(84, 212, 231, 0.42)", lineWidth: 1, lineStyle: dotted, lastValueVisible: false, priceLineVisible: false,
    });
    if (window.ResizeObserver) {
      new ResizeObserver((entries) => {
        const box = entries[0]?.contentRect;
        if (box && state.chart) state.chart.resize(Math.floor(box.width), Math.floor(box.height));
      }).observe(element);
    }
    return true;
  }

  function eventTime(epoch) {
    const value = Number(epoch);
    if (!Number.isFinite(value) || !state.candles.length) return null;
    let nearest = state.candles[0].time;
    for (const candle of state.candles) {
      if (candle.time > value) break;
      nearest = candle.time;
    }
    return nearest;
  }

  function forecastData(predictions) {
    const result = { line: [], upper: [], lower: [] };
    if (!state.candles.length || !Array.isArray(predictions)) return result;
    const candidates = predictions.filter((item) => finite(item.expected_return) && (item.valid_from || item.generated_at));
    if (!candidates.length) return result;
    candidates.sort((a, b) => String(a.generated_at || a.valid_from).localeCompare(String(b.generated_at || b.valid_from)));
    const prediction = candidates[candidates.length - 1];
    const originEpoch = Math.floor(new Date(prediction.valid_from || prediction.generated_at).getTime() / 1000);
    const origin = eventTime(originEpoch);
    const candle = [...state.candles].reverse().find((item) => item.time <= origin);
    const horizon = Number(prediction.forecast_horizon_seconds);
    const expiryEpoch = prediction.expires_at
      ? Math.floor(new Date(prediction.expires_at).getTime() / 1000)
      : (Number.isFinite(horizon) && horizon > 0 ? originEpoch + horizon : null);
    const expiry = expiryEpoch ? eventTime(expiryEpoch) : null;
    if (!origin || !expiry || expiry <= origin || !candle) return result;
    const base = Number(candle.close);
    const expected = Number(prediction.expected_return);
    result.line = [{ time: origin, value: base }, { time: expiry, value: base * (1 + expected) }];
    if (finite(prediction.uncertainty)) {
      const uncertainty = Math.abs(Number(prediction.uncertainty));
      result.upper = [{ time: origin, value: base }, { time: expiry, value: base * (1 + expected + uncertainty) }];
      result.lower = [{ time: origin, value: base }, { time: expiry, value: base * (1 + expected - uncertainty) }];
    }
    return result;
  }

  function eventMarker(epoch, options) {
    const time = eventTime(epoch);
    return time ? Object.assign({ time }, options) : null;
  }

  function markers(overlays) {
    if (!state.candleSeries || !overlays) return;
    const items = [];
    (overlays.trades || []).forEach((trade) => {
      const exit = trade.kind === "exit";
      items.push(eventMarker(trade.epoch, {
        position: exit ? "aboveBar" : "belowBar",
        color: exit ? "#fb7185" : "#36d399",
        shape: exit ? "arrowDown" : "arrowUp",
        text: (exit ? "EXIT " : "ENTRY ") + (trade.side || trade.action || ""),
      }));
    });
    (overlays.fills || []).forEach((fill) => {
      const fillEpoch = fill.filled_at ? Math.floor(new Date(fill.filled_at).getTime() / 1000) : null;
      const sell = String(fill.side || "").toUpperCase().includes("SELL");
      items.push(eventMarker(fillEpoch, {
        position: sell ? "aboveBar" : "belowBar",
        color: sell ? "#fb7185" : "#36d399",
        shape: sell ? "arrowDown" : "arrowUp",
        text: "FILL " + (fill.side || ""),
      }));
    });
    (overlays.portfolio_targets || []).forEach((target) => {
      const targetEpoch = target.created_at ? Math.floor(new Date(target.created_at).getTime() / 1000) : null;
      items.push(eventMarker(targetEpoch, {
        position: "belowBar",
        color: "#54d4e7",
        shape: "square",
        text: "TARGET REQUEST " + number(target.requested_target_exposure, 4),
      }));
    });
    (overlays.risk_decisions || []).forEach((risk) => {
      const riskEpoch = risk.created_at ? Math.floor(new Date(risk.created_at).getTime() / 1000) : null;
      const approved = risk.approved === true;
      items.push(eventMarker(riskEpoch, {
        position: "aboveBar",
        color: approved ? "#36d399" : "#fb7185",
        shape: "square",
        text: approved
          ? "RISK APPROVED " + number(risk.approved_exposure, 4)
          : "RISK REJECTED",
      }));
    });
    (overlays.news_events || []).forEach((event) => {
      items.push(eventMarker(event.epoch, {
        position: "aboveBar", color: "#f8ca5b", shape: "circle", text: event.event_type || event.title || "NEWS",
      }));
    });
    (overlays.liquidation_events || []).forEach((event) => {
      items.push(eventMarker(event.epoch, {
        position: "belowBar", color: "#fb923c", shape: "square", text: event.data_type || "LIQUIDATION",
      }));
    });
    (overlays.model_disagreements || []).forEach((event) => {
      items.push(eventMarker(event.epoch, {
        position: "aboveBar", color: "#a78bfa", shape: "circle", text: "MODEL CONFLICT",
      }));
    });
    const clean = items.filter(Boolean).sort((a, b) => a.time - b.time).slice(-120);
    if (typeof state.candleSeries.setMarkers === "function") state.candleSeries.setMarkers(clean);
  }

  function positionLines(current) {
    if (!state.candleSeries) return;
    state.priceLines.forEach((line) => {
      try { state.candleSeries.removePriceLine(line); } catch (_) { /* chart reset */ }
    });
    state.priceLines = [];
    const position = current?.position;
    if (!position || !finite(position.entry_price) || typeof state.candleSeries.createPriceLine !== "function") return;
    [[position.entry_price, "#54d4e7", "Entry"], [position.stop_loss, "#fb7185", "Stop"], [position.take_profit, "#36d399", "Take"]]
      .forEach((line) => {
        if (!finite(line[0])) return;
        state.priceLines.push(state.candleSeries.createPriceLine({
          price: Number(line[0]), color: line[1], lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: line[2],
        }));
      });
  }

  function regime(current) {
    const backdrop = byId("regimeBackdrop");
    if (!backdrop) return;
    const source = current?.ensemble?.current_regime || current?.legacy?.feature_snapshot?.trend || "";
    const name = String(source).trim().toLowerCase().replace(/[^a-z0-9]+/g, "_");
    backdrop.className = name ? "regime-" + name : "";
  }

  function renderChart(payload, overlays, current) {
    state.candles = (payload?.candles || [])
      .filter((item) => finite(item.time) && finite(item.open) && finite(item.high) && finite(item.low) && finite(item.close))
      .map((item) => ({
        time: Number(item.time), open: Number(item.open), high: Number(item.high), low: Number(item.low), close: Number(item.close), volume: Number(item.volume || 0),
      }));
    setText("chartTitle", state.symbol + " " + (payload?.timeframe || "1m") + " · stored candles");
    setText("chartSource", payload?.source || "stored data");
    setText("chartFreshness", payload?.latest_at ? age(payload.age_seconds) : "No stored candles");
    if (!setupChart()) {
      setText("chartNote", "Chart library is loading; stored data will appear shortly.");
      setTimeout(() => renderChart(payload, overlays, current), 300);
      return;
    }
    state.candleSeries.setData(state.candles.map((item) => ({
      time: item.time, open: item.open, high: item.high, low: item.low, close: item.close,
    })));
    state.volumeSeries.setData(state.candles.map((item) => ({
      time: item.time, value: item.volume, color: item.close >= item.open ? "rgba(54, 211, 153, 0.30)" : "rgba(251, 113, 133, 0.30)",
    })));
    const forecast = forecastData(overlays?.predictions || []);
    state.forecastSeries.setData(forecast.line);
    state.upperSeries.setData(forecast.upper);
    state.lowerSeries.setData(forecast.lower);
    markers(overlays);
    positionLines(current);
    regime(current);
    if (state.candles.length) {
      state.chart.timeScale().fitContent();
      setText("chartNote", forecast.line.length
        ? "Forecast line and dotted band use the latest recorded expected return and uncertainty only."
        : "No recorded forecast is available in this window. Candle and event overlays remain stored data.");
    } else {
      setText("chartNote", state.range === "all"
        ? "No stored candles are available for this symbol and timeframe."
        : "No stored candles match this time window. Select Stored history to inspect retained data.");
    }
  }

  function renderState(payload) {
    state.lastState = payload;
    const ensemble = payload?.ensemble || {};
    const target = payload?.portfolio_target || {};
    const risk = payload?.risk_decision || {};
    const position = payload?.position || {};
    const champion = payload?.champion || {};
    const external = payload?.external_ai || {};
    const feature = payload?.legacy?.feature_snapshot || {};
    setText("stateSymbol", payload?.symbol || state.symbol);
    setText("stateChampion", champion.model_id
      ? champion.model_id + (champion.model_version ? " · " + champion.model_version : "")
      : (champion.status || "-"));
    setText("stateRegime", ensemble.current_regime || feature.trend || "-");
    setText("stateExpectedReturn", percent(ensemble.combined_expected_return));
    setText("stateConfidence", percent(ensemble.combined_confidence) + " / " + percent(ensemble.combined_uncertainty));
    setText("stateCost", percent(ensemble.transaction_cost_penalty));
    setText("stateExposure", number(target.requested_target_exposure, 4) + " / " + number(risk.approved_exposure, 4));
    setText("statePosition", position.side ? position.side + " " + number(position.quantity, 6) : "No recorded open position");
    setText("statePnl", money(position.unrealized_pnl));
    const externalProviders = Array.isArray(external.providers) && external.providers.length
      ? external.providers.join(", ")
      : external.provider;
    const externalFallback = external.source === "latest_symbol_request_fallback" ? " · latest fallback" : "";
    setText("stateExternal", external.status === "not_recorded"
      ? "Not recorded"
      : (external.status || "-") + (externalProviders ? " · " + externalProviders : "") + externalFallback);
    setText("decisionSource", payload?.source || "-");
    setText("factFeature", feature.id ? (feature.schema_version || "feature") + " · " + timeText(feature.as_of) : "Not recorded");
    setText("factTarget", target.portfolio_target_id ? number(target.requested_target_exposure, 4) + " requested" : "Not recorded");
    setText("factRisk", risk.risk_decision_id ? (risk.approved ? "Approved " + number(risk.approved_exposure, 4) : "Rejected / reduced") : "Not recorded");
    const localNews = payload?.local_news_model || {};
    setText("factNewsModel", (localNews.version || "Not recorded")
      + (localNews.source === "latest_global_news_sentiment_fallback" ? " · latest fallback" : ""));
    setText("factFreshness", payload?.data_status?.latest_candle_at
      ? age(payload.data_status.candle_age_seconds) + (payload.data_status.stale ? " · stale" : "")
      : "No candle timestamp");
    const list = byId("reasonCodes");
    const codes = Array.isArray(payload?.reason_codes) ? payload.reason_codes.filter(Boolean) : [];
    if (list) {
      list.innerHTML = codes.length
        ? codes.map((item) => "<li>" + html(typeof item === "string" ? item : JSON.stringify(item)) + "</li>").join("")
        : "<li>No structured reason codes are recorded for the current state.</li>";
    }
    const lineageNotes = [];
    if (payload?.source === "legacy") lineageNotes.push("Legacy records are partial evidence. Missing V2 stages are not inferred.");
    if (external.source === "latest_symbol_request_fallback") {
      lineageNotes.push("External AI is the latest symbol request fallback and is not linked to the displayed decision.");
    } else if (external.status === "not_recorded") {
      lineageNotes.push("External AI has no recorded evidence for this state.");
    }
    if (localNews.source === "latest_global_news_sentiment_fallback") {
      lineageNotes.push("The local news version is a latest global fallback and is not linked to the displayed decision.");
    }
    setText("decisionNote", lineageNotes.join(" "));
  }

  function tone(value) {
    const text = String(value || "").toUpperCase();
    if (text.includes("BULL") || text.includes("BUY") || text.includes("HEALTHY") || text.includes("APPROV")) return "positive";
    if (text.includes("BEAR") || text.includes("SELL") || text.includes("REJECT") || text.includes("DEGRADED")) return "negative";
    if (text.includes("WATCH") || text.includes("NEUTRAL") || text.includes("SUSPEND")) return "warning";
    return "";
  }

  function renderModels(payload) {
    const rows = payload?.models || [];
    const body = byId("modelsBody");
    if (!body) return;
    body.innerHTML = rows.length ? rows.map((row) => (
      "<tr><td>" + html(row.model_name || "-") + "</td>" +
      "<td>" + html(row.model_family || "-") + "</td>" +
      "<td class='" + tone(row.view) + "'>" + html(row.view || "-") + "</td>" +
      "<td>" + percent(row.expected_return) + "</td>" +
      "<td>" + percent(row.confidence) + "</td>" +
      "<td>" + percent(row.uncertainty) + "</td>" +
      "<td>" + (finite(row.weight) ? percent(row.weight) : "-") + "</td>" +
      "<td class='" + tone(row.health) + "'>" + html(row.health || "-") + "</td>" +
      "<td>" + html(row.lifecycle || "-") + "</td>" +
      "<td>" + timeText(row.last_prediction_time) + "</td></tr>"
    )).join("") : "<tr><td colspan='10' class='empty'>" + html(payload?.message || "No recorded model evidence.") + "</td></tr>";
    const ensemble = payload?.ensemble;
    setText("modelsStatus", ensemble
      ? (ensemble.supporting_signals || []).length + " supporting · " + (ensemble.conflicting_signals || []).length + " conflicting"
      : "No ensemble record");
  }

  function renderHistory(payload) {
    const metrics = payload?.metrics || {};
    setText("metricTrades", number(metrics.trade_count, 0));
    setText("metricLedgerEvents", number(metrics.ledger_event_count, 0));
    setText("metricWins", number(metrics.win_count, 0) + " / " + number(metrics.loss_count, 0));
    setText("metricWinRate", percent(metrics.win_rate));
    setText("metricPnl", money(metrics.closed_paper_pnl));
    setText("metricLedgerPnl", money(metrics.ledger_total_paper_pnl));
    setText("metricProfitFactor", number(metrics.profit_factor));
    setText("metricFees", money(metrics.fees, 4));
    setText("metricDrawdown", money(metrics.maximum_drawdown));
    setText("metricCosts", metrics.slippage === null && metrics.funding === null
      ? "Not recorded"
      : money(metrics.slippage || 0, 4) + " / " + money(metrics.funding || 0, 4));
    const ledger = payload?.paper_ledger_events || payload?.legacy_trades || [];
    const traced = ledger.filter((item) => item.source === "v2-paper-ledger").length;
    const legacy = ledger.filter((item) => item.source === "legacy").length;
    setText("historyStatus", ledger.length + " paper-ledger events (" + traced + " traced V2 / " + legacy
      + " legacy) · " + (payload?.simulated_fills || []).length + " simulated fills");
    setText("historyNote", payload?.availability?.note || "");
  }

  function renderDecisions(payload) {
    const rows = payload?.decisions || [];
    const list = byId("decisionList");
    if (!list) return;
    list.innerHTML = rows.length ? rows.map((row) => {
      const trace = row.trace_id || row.id;
      const active = state.selectedTraceId === trace ? " active" : "";
      return "<li><button type='button' data-trace-id='" + html(trace) + "' class='" + active + "'>" +
        html(row.stage || row.action || "Decision") +
        "<small>" + html(row.status || "-") + " · " + timeText(row.time) + "</small></button></li>";
    }).join("") : "<li class='empty'>No recorded decisions in this window.</li>";
    list.querySelectorAll("[data-trace-id]").forEach((button) => {
      button.addEventListener("click", () => replay(button.dataset.traceId));
    });
    if (!state.selectedTraceId && rows[0]?.trace_id) replay(rows[0].trace_id);
  }

  async function replay(traceId) {
    if (!traceId) return;
    state.selectedTraceId = traceId;
    const timeline = byId("replayTimeline");
    if (timeline) timeline.innerHTML = "<li>Loading recorded timeline…</li>";
    try {
      const payload = await request("replay/" + encodeURIComponent(traceId));
      const events = payload.events || [];
      if (timeline) {
        timeline.innerHTML = events.length ? events.map((event) => {
          const reasons = (event.reason_codes || []).map((item) => typeof item === "string" ? item : JSON.stringify(item)).join(" · ");
          return "<li>" + html(event.stage || "recorded event") +
            "<small>" + timeText(event.time) + " · " + html(event.status || "recorded") + "</small>" +
            (reasons ? "<small>" + html(reasons) + "</small>" : "") + "</li>";
        }).join("") : "<li>No recorded timeline events exist for this trace.</li>";
      }
      document.querySelectorAll("[data-trace-id]").forEach((button) => {
        button.classList.toggle("active", button.dataset.traceId === traceId);
      });
    } catch (error) {
      if (timeline) timeline.innerHTML = "<li>Replay unavailable: " + html(error.message) + "</li>";
    }
  }

  function researchRow(item) {
    const name = item?.model_id || item?.model_family || item?.name || item?.account_id || item?.candidate_id || item?.id || "record";
    const status = item?.health_status || item?.lifecycle_status || "-";
    return "<li>" + html(name) + "<small>" + html(status) + (item?.created_at ? " · " + timeText(item.created_at) : "") + "</small></li>";
  }

  function renderResearch(payload) {
    const mapping = {
      researchChampions: payload?.champion_assignments,
      researchCandidates: payload?.strategy_candidates,
      researchModelHealth: payload?.model_health,
      researchSignalHealth: payload?.signal_health,
      researchPromotions: payload?.promotion_decisions,
      researchSandboxes: payload?.sandbox_accounts,
    };
    Object.entries(mapping).forEach(([id, rows]) => {
      const element = byId(id);
      if (element) element.innerHTML = Array.isArray(rows) && rows.length
        ? rows.map(researchRow).join("")
        : "<li class='muted'>No recorded rows.</li>";
    });
    setText("researchStatus", payload?.availability?.v2_registry_tables_present ? "Recorded V2 registry data" : "No V2 registry rows");
  }

  function populateSymbols(payload) {
    const select = byId("visionSymbol");
    const symbols = payload?.symbols || [];
    if (!select || !symbols.length) return;
    const selected = state.symbol;
    select.innerHTML = "";
    symbols.forEach((item) => {
      const option = document.createElement("option");
      option.value = item.symbol;
      option.textContent = item.stale && item.latest_candle_at ? item.symbol + " · stale" : item.symbol;
      option.selected = item.symbol === selected;
      select.appendChild(option);
    });
    if (![...select.options].some((option) => option.value === selected)) state.symbol = select.value || config.defaultSymbol || "BTCUSDT";
  }

  async function refreshSymbols() {
    try {
      populateSymbols(await request("symbols"));
    } catch (error) {
      notice("Symbol list unavailable: " + error.message, "is-warning");
    }
  }

  async function refresh() {
    if (state.refreshing) return;
    state.refreshing = true;
    const windowValues = rangeParams();
    const base = Object.assign({ symbol: state.symbol }, windowValues);
    notice("Refreshing recorded evidence for " + state.symbol + "…");
    try {
      const results = await Promise.allSettled([
        request("chart", Object.assign({}, base, { timeframe: "1m", limit: defaultLimit })),
        request("overlays", Object.assign({}, base, { limit: defaultLimit })),
        request("state", { symbol: state.symbol }),
        request("models", { symbol: state.symbol, limit: defaultLimit }),
        request("history", Object.assign({}, base, { limit: defaultLimit })),
        request("decisions", Object.assign({}, base, { limit: defaultLimit })),
        request("research", { symbol: state.symbol, limit: defaultLimit }),
      ]);
      const values = results.map((result) => result.status === "fulfilled" ? result.value : null);
      const failed = results.filter((result) => result.status === "rejected");
      const chart = values[0];
      const overlays = values[1];
      const current = values[2];
      if (current) renderState(current);
      if (chart) renderChart(chart, overlays || {}, current || state.lastState);
      if (values[3]) renderModels(values[3]);
      if (values[4]) renderHistory(values[4]);
      if (values[5]) renderDecisions(values[5]);
      if (values[6]) renderResearch(values[6]);
      if (failed.length) {
        notice(failed.length + " Vision request" + (failed.length === 1 ? "" : "s") + " failed; available sections were rendered.", "is-warning");
      } else if (current?.data_status?.stale) {
        notice("Rendered recorded evidence for " + state.symbol + ". Market data is stale (" + age(current.data_status.candle_age_seconds) + ").", "is-warning");
      } else {
        notice("Rendered recorded evidence for " + state.symbol + ".");
      }
    } catch (error) {
      notice("Vision refresh failed: " + error.message, "is-error");
    } finally {
      state.refreshing = false;
    }
  }

  function bind() {
    byId("visionSymbol")?.addEventListener("change", (event) => {
      state.symbol = event.target.value;
      state.selectedTraceId = null;
      refresh();
    });
    byId("visionRange")?.addEventListener("change", (event) => {
      state.range = event.target.value;
      state.selectedTraceId = null;
      refresh();
    });
    byId("visionRefresh")?.addEventListener("click", refresh);
    window.addEventListener("resize", () => {
      const element = byId("visionChart");
      if (state.chart && element) state.chart.resize(element.clientWidth, element.clientHeight);
    });
  }

  function start() {
    bind();
    refreshSymbols().finally(refresh);
    const interval = Math.max(Number(config.refreshSeconds || 15), 5) * 1000;
    setInterval(() => {
      if (document.visibilityState === "visible") refresh();
    }, interval);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
