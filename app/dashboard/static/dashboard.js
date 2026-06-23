(function () {
  const canvas = document.getElementById("equityChart");
  const points = window.EQUITY_POINTS || [];

  function drawChart() {
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const scale = window.devicePixelRatio || 1;
    canvas.width = Math.max(320, Math.floor(rect.width * scale));
    canvas.height = Math.floor(220 * scale);
    const ctx = canvas.getContext("2d");
    ctx.scale(scale, scale);
    const width = canvas.width / scale;
    const height = canvas.height / scale;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#fbfcfd";
    ctx.fillRect(0, 0, width, height);
    ctx.strokeStyle = "#d8e0e6";
    ctx.lineWidth = 1;
    for (let i = 0; i < 5; i += 1) {
      const y = 18 + (i * (height - 36)) / 4;
      ctx.beginPath();
      ctx.moveTo(14, y);
      ctx.lineTo(width - 14, y);
      ctx.stroke();
    }
    if (points.length < 2) {
      ctx.fillStyle = "#64727d";
      ctx.font = "14px system-ui";
      ctx.fillText("Waiting for equity history", 18, 34);
      return;
    }
    const values = points.map((point) => Number(point.equity));
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = Math.max(max - min, 1);
    const xFor = (idx) => 18 + (idx * (width - 36)) / (points.length - 1);
    const yFor = (value) => height - 18 - ((value - min) * (height - 36)) / range;

    ctx.strokeStyle = "#2864d9";
    ctx.lineWidth = 2;
    ctx.beginPath();
    points.forEach((point, idx) => {
      const x = xFor(idx);
      const y = yFor(Number(point.equity));
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    ctx.fillStyle = "#172026";
    ctx.font = "12px system-ui";
    ctx.fillText(`High ${max.toFixed(2)}`, 18, 20);
    ctx.fillText(`Low ${min.toFixed(2)}`, 18, height - 8);
  }

  async function collectorAction(worker, action) {
    const message = document.getElementById("collectorMessage");
    if (message) message.textContent = `${action} ${worker}...`;
    const response = await fetch(`/api/collectors/${worker}/${action}`, { method: "POST" });
    const data = await response.json();
    if (message) message.textContent = data.last_error || `${worker} ${data.running ? "running" : "stopped"}`;
  }

  async function autoTraderAction(action) {
    const message = document.getElementById("autoTraderMessage");
    if (message) message.textContent = `${action} auto trader...`;
    const response = await fetch(`/api/auto-trader/${action}`, { method: "POST" });
    const data = await response.json();
    if (message) message.textContent = data.last_error || (data.running ? "Running" : "Stopped");
  }

  async function submitSignal(event) {
    event.preventDefault();
    const form = event.currentTarget;
    const output = document.getElementById("signalResult");
    const formData = new FormData(form);
    const payload = Object.fromEntries(formData.entries());
    ["confidence", "price", "notional"].forEach((key) => {
      if (payload[key] !== "") payload[key] = Number(payload[key]);
      else delete payload[key];
    });
    payload.source = "dashboard-manual";
    if (output) output.textContent = "Sending...";
    const response = await fetch("/api/signal", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (output) output.textContent = `${data.status}: ${data.message}`;
  }

  document.querySelectorAll("[data-worker]").forEach((button) => {
    button.addEventListener("click", () => collectorAction(button.dataset.worker, button.dataset.action));
  });
  document.querySelectorAll("[data-auto-trader]").forEach((button) => {
    button.addEventListener("click", () => autoTraderAction(button.dataset.autoTrader));
  });
  const signalForm = document.getElementById("signalForm");
  if (signalForm) signalForm.addEventListener("submit", submitSignal);
  window.addEventListener("resize", drawChart);
  drawChart();
})();
