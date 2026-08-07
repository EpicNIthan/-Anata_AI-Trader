(function () {
  function api(path, options = {}) {
    return fetch(path, { credentials: "same-origin", ...options }).then(async (response) => {
      if (!response.ok) {
        let detail = `${response.status} ${response.statusText}`;
        try {
          const body = await response.json();
          detail = body.detail || detail;
        } catch (_) {}
        throw new Error(detail);
      }
      return response.json();
    });
  }

  function simplifyStorage() {
    const panel = document.getElementById("tab-storage");
    if (!panel || panel.dataset.handoffReady === "1") return;
    panel.dataset.handoffReady = "1";

    const tools = panel.querySelector(".panel-tools");
    if (!tools) return;
    tools.innerHTML = `
      <button id="handoffDownloadButton" class="primary">Download All Dataset</button>
      <button id="handoffDeleteButton" class="secondary">Delete Downloaded Dataset</button>
      <output id="dbStorageActionResult"></output>
    `;

    const oldReport = document.getElementById("collectionReportBox");
    if (oldReport) oldReport.style.display = "none";
    const bundlesBody = document.getElementById("dailyBundlesBody");
    if (bundlesBody?.closest("table")) bundlesBody.closest("table").style.display = "none";

    const downloadButton = document.getElementById("handoffDownloadButton");
    const deleteButton = document.getElementById("handoffDeleteButton");
    const output = document.getElementById("dbStorageActionResult");

    downloadButton?.addEventListener("click", async () => {
      downloadButton.disabled = true;
      if (output) output.textContent = "Preparing one complete ZIP. Keep this page open...";
      try {
        const data = await api("/api/data/handoff/prepare", { method: "POST" });
        if (data.status === "empty") {
          if (output) output.textContent = data.message || "No finished data yet.";
          return;
        }
        if (output) {
          output.textContent = `ZIP ready: ${data.archive_id} · ${data.start || "-"} → ${data.end || "-"}`;
        }
        const anchor = document.createElement("a");
        anchor.href = data.download_url;
        anchor.download = data.archive_id;
        document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();
      } catch (error) {
        if (output) output.textContent = `Download preparation failed: ${error.message}`;
      } finally {
        downloadButton.disabled = false;
      }
    });

    deleteButton?.addEventListener("click", async () => {
      const ok = window.confirm(
        "Delete only the dataset range from the ZIP you just downloaded? " +
        "The bot's small strategy-history cache and operational state will be kept."
      );
      if (!ok) return;
      deleteButton.disabled = true;
      if (output) output.textContent = "Deleting downloaded rows and rebuilding the bot history cache...";
      try {
        const data = await api("/api/data/handoff/delete-latest", { method: "POST" });
        const deleted = Object.values(data.deleted_rows || {}).reduce((sum, value) => sum + Number(value || 0), 0);
        const ready = Object.values(data.strategy_history?.symbols || {}).every((item) => item.status === "ready");
        if (output) {
          output.textContent = `Deleted ${deleted.toLocaleString()} downloaded rows. Bot history: ${ready ? "READY" : "building/partial"}. Next file starts ${data.next_archive_start || "-"}.`;
        }
      } catch (error) {
        if (output) output.textContent = `Delete failed: ${error.message}`;
      } finally {
        deleteButton.disabled = false;
      }
    });

    function rewriteWarning() {
      const warning = document.getElementById("dbStorageWarning");
      if (warning && warning.textContent.trim()) {
        warning.textContent = "Database is large. Download the complete dataset ZIP first, then delete only that downloaded range.";
      }
    }
    rewriteWarning();
    setInterval(rewriteWarning, 3000);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", simplifyStorage);
  } else {
    simplifyStorage();
  }
})();
