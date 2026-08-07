(function () {
  const HISTORY_LIMITS = {
    "1s": 600,
    "1m": 1440,
    "5m": 1000,
    "15m": 672,
    "1h": 168,
  };
  const TIMEFRAME_SECONDS = {
    "1s": 1,
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
  };

  const bridge = window.ANATA_CHART_BRIDGE = window.ANATA_CHART_BRIDGE || {
    chart: null,
    candleSeries: null,
    firstTime: null,
    lastTime: null,
    priceLines: [],
    hooksInstalled: false,
    refreshTimer: null,
  };

  const nativeFetch = window.fetch.bind(window);
  bridge.nativeFetch = nativeFetch;

  function currentTimeframe() {
    return document.querySelector("[data-timeframe].active")?.dataset.timeframe
      || window.DASHBOARD_CONFIG?.defaultTimeframe
      || "1m";
  }

  function currentSymbol() {
    return document.getElementById("symbolSelect")?.value
      || window.DASHBOARD_CONFIG?.defaultSymbol
      || "BTCUSDT";
  }

  function localUrl(url) {
    return url.origin === window.location.origin ? `${url.pathname}${url.search}${url.hash}` : url.toString();
  }

  window.fetch = function patchedFetch(input, init) {
    try {
      const rawUrl = typeof input === "string" ? input : input?.url;
      if (rawUrl) {
        const url = new URL(rawUrl, window.location.origin);
        if (url.origin === window.location.origin && url.pathname === "/api/market/candles") {
          const timeframe = (url.searchParams.get("timeframe") || "1m").toLowerCase();
          url.pathname = "/api/chart/candles";
          url.searchParams.set("limit", String(HISTORY_LIMITS[timeframe] || 1000));
          if (typeof input === "string") return nativeFetch(localUrl(url), init);
          return nativeFetch(new Request(url.toString(), input), init);
        }
      }
    } catch (error) {
      console.warn("Anata chart history rewrite skipped", error);
    }
    return nativeFetch(input, init);
  };

  function ensureStatus() {
    let node = document.getElementById("signalOverlayStatus");
    if (node) return node;
    const footer = document.querySelector(".chart-footer");
    if (!footer) return null;
    node = document.createElement("span");
    node.id = "signalOverlayStatus";
    node.textContent = "Signals loading";
    footer.appendChild(node);
    return node;
  }

  function setStatus(text, warning) {
    const node = ensureStatus();
    if (!node) return;
    node.textContent = text;
    node.classList.toggle("warning", Boolean(warning));
  }

  function fmtPrice(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "-";
    return number >= 1000 ? number.toLocaleString(undefined, { maximumFractionDigits: 2 }) : number.toLocaleString(undefined, { maximumFractionDigits: 6 });
  }

  function fmtPnl(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "-";
    return `${number >= 0 ? "+" : ""}$${number.toFixed(2)}`;
  }

  function unix(value) {
    const ms = Date.parse(value || "");
    return Number.isFinite(ms) ? Math.floor(ms / 1000) : null;
  }

  function snapToTimeframe(value, timeframe) {
    const seconds = unix(value);
    if (seconds === null) return null;
    const step = TIMEFRAME_SECONDS[timeframe] || 60;
    return Math.floor(seconds / step) * step;
  }

  function inVisibleData(time) {
    if (!Number.isFinite(Number(time))) return false;
    if (bridge.firstTime !== null && Number(time) < Number(bridge.firstTime)) return false;
    if (bridge.lastTime !== null && Number(time) > Number(bridge.lastTime)) return false;
    return true;
  }

  function clearPriceLines() {
    if (!bridge.candleSeries) return;
    bridge.priceLines.forEach((line) => {
      try { bridge.candleSeries.removePriceLine(line); } catch (_) { }
    });
    bridge.priceLines = [];
  }

  function addPriceLine(price, title, color, lineStyle) {
    const value = Number(price);
    if (!bridge.candleSeries || !Number.isFinite(value) || value <= 0) return;
    const line = bridge.candleSeries.createPriceLine({
      price: value,
      color,
      lineWidth: 1,
      lineStyle,
      axisLabelVisible: true,
      title,
    });
    bridge.priceLines.push(line);
  }

  function positionMarkers(position, timeframe) {
    const markers = [];
    const entryTime = snapToTimeframe(position.opened_at, timeframe);
    const longSide = String(position.side || "LONG").toUpperCase() !== "SHORT";
    if (entryTime !== null && inVisibleData(entryTime)) {
      const details = [
        `${longSide ? "BUY" : "SELL"} @ ${fmtPrice(position.entry_price)}`,
        position.stop_loss ? `SL ${fmtPrice(position.stop_loss)}` : null,
        position.take_profit ? `TP ${fmtPrice(position.take_profit)}` : null,
      ].filter(Boolean).join(" | ");
      markers.push({
        time: entryTime,
        position: longSide ? "belowBar" : "aboveBar",
        color: longSide ? "#0ecb81" : "#f6465d",
        shape: longSide ? "arrowUp" : "arrowDown",
        text: details,
      });
    }

    if (position.closed_at) {
      const exitTime = snapToTimeframe(position.closed_at, timeframe);
      if (exitTime !== null && inVisibleData(exitTime)) {
        markers.push({
          time: exitTime,
          position: longSide ? "aboveBar" : "belowBar",
          color: Number(position.realized_pnl || 0) >= 0 ? "#0ecb81" : "#f6465d",
          shape: "circle",
          text: `EXIT @ ${fmtPrice(position.current_price)} | ${fmtPnl(position.realized_pnl)}`,
        });
      }
    }
    return markers;
  }

  async function refreshSignals() {
    if (!bridge.candleSeries) return;
    const symbol = currentSymbol();
    const timeframe = currentTimeframe();
    try {
      const response = await nativeFetch(`/api/chart/signals?symbol=${encodeURIComponent(symbol)}&limit=250`, { credentials: "same-origin" });
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      const data = await response.json();
      if (symbol !== currentSymbol() || timeframe !== currentTimeframe()) return;

      const markers = (data.positions || [])
        .flatMap((position) => positionMarkers(position, timeframe))
        .sort((a, b) => Number(a.time) - Number(b.time));
      bridge.candleSeries.setMarkers(markers);

      clearPriceLines();
      const openPosition = (data.positions || []).find((position) => String(position.status).toUpperCase() === "OPEN");
      if (openPosition) {
        const solid = window.LightweightCharts?.LineStyle?.Solid ?? 0;
        const dashed = window.LightweightCharts?.LineStyle?.Dashed ?? 2;
        addPriceLine(openPosition.entry_price, `ENTRY ${fmtPrice(openPosition.entry_price)}`, "#f0b90b", solid);
        addPriceLine(openPosition.stop_loss, `SL ${fmtPrice(openPosition.stop_loss)}`, "#f6465d", dashed);
        addPriceLine(openPosition.take_profit, `TP ${fmtPrice(openPosition.take_profit)}`, "#0ecb81", dashed);
      }

      const history = HISTORY_LIMITS[timeframe] || 1000;
      const historyText = timeframe === "1m" ? "24h" : `${history} ${timeframe} bars`;
      setStatus(`Signals ON · ${historyText} · ${markers.length} marker${markers.length === 1 ? "" : "s"}`, false);
    } catch (error) {
      setStatus(`Signals error: ${error.message}`, true);
    }
  }

  function wrapCandleSeries(series) {
    if (!series || series.__anataWrapped) return series;
    series.__anataWrapped = true;
    const originalSetData = series.setData.bind(series);
    series.setData = function (data) {
      const rows = Array.isArray(data) ? data : [];
      bridge.firstTime = rows.length ? rows[0].time : null;
      bridge.lastTime = rows.length ? rows[rows.length - 1].time : null;
      const result = originalSetData(data);
      window.setTimeout(refreshSignals, 0);
      return result;
    };
    bridge.candleSeries = series;
    return series;
  }

  function installChartHooks() {
    if (bridge.hooksInstalled) return true;
    const library = window.LightweightCharts;
    if (!library?.createChart) return false;
    const originalCreateChart = library.createChart.bind(library);
    library.createChart = function (...args) {
      const chart = originalCreateChart(...args);
      bridge.chart = chart;
      const originalAddCandlestickSeries = chart.addCandlestickSeries.bind(chart);
      chart.addCandlestickSeries = function (...seriesArgs) {
        return wrapCandleSeries(originalAddCandlestickSeries(...seriesArgs));
      };
      return chart;
    };
    bridge.hooksInstalled = true;
    return true;
  }

  function wireOverlayEvents() {
    ensureStatus();
    document.getElementById("symbolSelect")?.addEventListener("change", () => window.setTimeout(refreshSignals, 250));
    document.querySelectorAll("[data-market-symbol]").forEach((button) => {
      button.addEventListener("click", () => window.setTimeout(refreshSignals, 250));
    });
    document.querySelectorAll("[data-timeframe]").forEach((button) => {
      button.addEventListener("click", () => window.setTimeout(refreshSignals, 500));
    });
    if (!bridge.refreshTimer) bridge.refreshTimer = window.setInterval(refreshSignals, 15000);
  }

  installChartHooks();
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => {
      installChartHooks();
      wireOverlayEvents();
      window.setTimeout(refreshSignals, 1500);
    });
  } else {
    installChartHooks();
    wireOverlayEvents();
    window.setTimeout(refreshSignals, 1500);
  }
})();
