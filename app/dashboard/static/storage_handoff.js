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

    // dashboard.js still updates this old warning every refresh. Hide it so the
    // user sees one stable handoff status instead of two messages blinking.
    const oldWarning = document.getElementById("dbStorageWarning");
    if (oldWarning) {
      oldWarning.textContent = "";
      oldWarning.style.display = "none";
    }

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
    let pollTimer = null;

    async function refreshDownloadState() {
      try {
        const status = await api("/api/data/handoff/status");
        if (status.download_completed_at && status.download_ready_to_delete) {
          if (output) {
            output.textContent = `Download completed: ${status.latest_archive_id}. The downloaded range is now safe to delete.`;
          }
          if (pollTimer) {
            clearInterval(pollTimer);
            pollTimer = null;
          }
        }
      } catch (_) {}
    }

    downloadButton?.addEventListener("click", () => {
      if (output) {
        output.textContent = "Preparing and downloading the ZIP in one request. Keep this page open until the browser finishes the download...";
      }
      const anchor = document.createElement("a");
      anchor.href = `/api/data/handoff/download-latest?ts=${Date.now()}`;
      anchor.download = "anata_dataset.zip";
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();

      if (pollTimer) clearInterval(pollTimer);
      pollTimer = setInterval(refreshDownloadState, 3000);
      setTimeout(refreshDownloadState, 1500);
    });

    deleteButton?.addEventListener("click", async () => {
      const ok = window.confirm(
        "Delete only the dataset range whose ZIP finished downloading? " +
        "The paper bot's required strategy history and operational state will be kept."
      );
      if (!ok) return;

      deleteButton.disabled = true;
      if (output) output.textContent = "Checking completed download, then deleting only that downloaded range...";
      try {
        const data = await api("/api/data/handoff/delete-latest", { method: "POST" });
        const deleted = Object.values(data.deleted_rows || {}).reduce((sum, value) => sum + Number(value || 0), 0);
        const statuses = Object.values(data.strategy_history?.symbols || {});
        const ready = statuses.length > 0 && statuses.every((item) => item.status === "ready");
        if (output) {
          output.textContent = `Deleted ${deleted.toLocaleString()} downloaded rows. Bot history: ${ready ? "READY" : "building/partial"}. Next dataset starts ${data.next_archive_start || "-"}.`;
        }
      } catch (error) {
        if (output) output.textContent = `Delete blocked: ${error.message}`;
      } finally {
        deleteButton.disabled = false;
      }
    });

    refreshDownloadState();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", simplifyStorage);
  } else {
    simplifyStorage();
  }
})();
