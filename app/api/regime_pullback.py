from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy import desc, select

from app.db.session import SessionLocal
from app.security import require_admin
from app.services.regime_label_builder import RegimeLabelBuilder
from app.services.regime_pullback_service import ACCOUNT_ID
from app.strategies.regime_models import RegimeDecisionRecord, RegimePaperAccount
from app.strategies.regime_pullback_v1 import STRATEGY_NAME, STRATEGY_VERSION

router = APIRouter(tags=["regime-pullback"])


class ConfirmRequest(BaseModel):
    confirm: bool = False


def _service(request: Request):
    service = getattr(request.app.state, "auto_trader", None)
    if service is None:
        raise HTTPException(status_code=503, detail="strategy service is unavailable")
    return service


@router.get("/api/regime-pullback/status")
def status(request: Request, _: None = Depends(require_admin)) -> dict[str, Any]:
    return _service(request).status()


@router.post("/api/regime-pullback/run-once")
def run_once(request: Request, _: None = Depends(require_admin)) -> dict[str, Any]:
    return _service(request).run_once()


@router.get("/api/regime-pullback/decisions")
def decisions(request: Request, limit: int = 100, _: None = Depends(require_admin)) -> dict[str, Any]:
    limit = max(1, min(limit, 1000))
    service = _service(request)
    with SessionLocal() as session:
        rows = session.scalars(
            select(RegimeDecisionRecord).order_by(desc(RegimeDecisionRecord.candle_close_time)).limit(limit)
        ).all()
        return {"strategy": STRATEGY_NAME, "version": STRATEGY_VERSION, "items": [service._decision_summary(row) for row in rows]}


@router.post("/api/regime-pullback/admin-shutdown")
def admin_shutdown(body: ConfirmRequest, request: Request, _: None = Depends(require_admin)) -> dict[str, Any]:
    if not body.confirm:
        raise HTTPException(status_code=400, detail="confirm=true is required")
    with SessionLocal() as session:
        account = session.get(RegimePaperAccount, ACCOUNT_ID)
        if account is None:
            account = _service(request)._get_or_create_account(session)
        account.administrative_shutdown = True
        session.commit()
    return {"strategy": STRATEGY_NAME, "administrative_shutdown": True, "paper_only": True}


@router.post("/api/regime-pullback/resume")
def resume(body: ConfirmRequest, request: Request, _: None = Depends(require_admin)) -> dict[str, Any]:
    if not body.confirm:
        raise HTTPException(status_code=400, detail="confirm=true is required")
    with SessionLocal() as session:
        account = session.get(RegimePaperAccount, ACCOUNT_ID)
        if account is None:
            account = _service(request)._get_or_create_account(session)
        account.administrative_shutdown = False
        session.commit()
    return {"strategy": STRATEGY_NAME, "administrative_shutdown": False, "paper_only": True}


@router.post("/api/regime-pullback/reset-paper-account")
def reset_paper_account(body: ConfirmRequest, request: Request, _: None = Depends(require_admin)) -> dict[str, Any]:
    if not body.confirm:
        raise HTTPException(status_code=400, detail="confirm=true is required")
    with SessionLocal() as session:
        result = _service(request).reset_paper_account(session)
        session.commit()
        return result


@router.post("/api/regime-pullback/build-labels")
def build_labels(_: None = Depends(require_admin)) -> dict[str, Any]:
    with SessionLocal() as session:
        return RegimeLabelBuilder(session).run()


DASHBOARD_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Regime Pullback V1</title>
<style>
body{font-family:system-ui;background:#0b1020;color:#e9eefc;margin:0;padding:24px}h1{margin:0}.sub{color:#9fb0d0;margin:6px 0 20px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}.card{background:#151d33;border:1px solid #2c385b;border-radius:12px;padding:16px}.ok{color:#78e6a6}.warn{color:#ffc56e}.bad{color:#ff8c8c}table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;border-bottom:1px solid #2c385b;padding:8px;vertical-align:top}code{white-space:pre-wrap}.pill{display:inline-block;padding:3px 8px;border-radius:999px;background:#263455;margin:2px}button{background:#3f65e8;color:white;border:0;border-radius:8px;padding:8px 12px;cursor:pointer}button.danger{background:#b84242}</style></head>
<body><h1>regime_pullback_v1</h1><div class="sub">Conservative, explainable, paper-only data-collection strategy. No live exchange execution.</div>
<div class="grid" id="cards"></div><div class="card" style="margin-top:14px"><h2>Latest decisions</h2><table><thead><tr><th>Close</th><th>Symbol</th><th>Regime</th><th>Action</th><th>Confidence</th><th>Reasons</th><th>Data / spread</th></tr></thead><tbody id="rows"></tbody></table></div>
<script>
async function load(){const s=await fetch('/api/regime-pullback/status').then(r=>r.json());const d=await fetch('/api/regime-pullback/decisions?limit=100').then(r=>r.json());
const a=s.paper_account||{}, daily=s.daily_risk||{};document.getElementById('cards').innerHTML=`
<div class="card"><b>Status</b><p class="${s.paper_only?'ok':'bad'}">${s.paper_only?'PAPER ONLY':'UNSAFE'}</p><p>Version: ${s.strategy_version}</p><p>Last candle: ${s.latest_processed_decision_candle||'-'}</p><p>Worker lock: ${s.lock_acquired?'owned':'not owned'}</p></div>
<div class="card"><b>Account</b><p>Equity: ${a.equity||'-'}</p><p>Cash: ${a.cash||'-'}</p><p>Realized trading PnL: ${a.realized_trading_pnl||'-'}</p><p>Unrealized PnL: ${a.unrealized_pnl||'-'}</p><p>Total fees: ${a.total_fees||'-'}</p></div>
<div class="card"><b>Risk</b><p>Daily loss usage: ${daily.loss_limit_usage_pct||'0'}%</p><p>Trades today: ${daily.new_trades||0}</p><p>Consecutive losses: ${daily.consecutive_losses||0}</p><p>Circuit breaker: ${daily.circuit_breaker?'ACTIVE':'clear'}</p></div>
<div class="card"><b>Controls</b><p>Disable the bot while collectors keep running by setting AUTO_TRADER_ENABLED=false.</p><button onclick="runOnce()">Run once</button> <button class="danger" onclick="shutdown()">Paper shutdown</button></div>`;
document.getElementById('rows').innerHTML=(d.items||[]).map(x=>`<tr><td>${x.candle_close_time}</td><td>${x.symbol}</td><td>${x.regime}</td><td>${x.action}${x.shadow_decision?' (shadow)':''}</td><td>${Number(x.confidence).toFixed(3)}</td><td>${(x.reason_codes||[]).map(r=>`<span class="pill">${r}</span>`).join('')}</td><td>${x.data_fresh?'fresh':'stale'} / ${x.spread_bps||'-'} bps ${x.spread_estimated?'estimated':'measured'}</td></tr>`).join('');}
async function runOnce(){await fetch('/api/regime-pullback/run-once',{method:'POST'});load()}async function shutdown(){if(confirm('Close positions safely on the next completed decision cycle?')){await fetch('/api/regime-pullback/admin-shutdown',{method:'POST',headers:{'content-type':'application/json'},body:'{"confirm":true}'});load()}}load();setInterval(load,30000);
</script></body></html>"""


@router.get("/regime-dashboard", response_class=HTMLResponse)
def dashboard(_: None = Depends(require_admin)) -> HTMLResponse:
    return HTMLResponse(DASHBOARD_HTML)
