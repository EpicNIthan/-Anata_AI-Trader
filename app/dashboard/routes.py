from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.config import settings
from app.security import ADMIN_COOKIE_NAME, admin_token_query_is_valid, require_admin

router = APIRouter(tags=["dashboard"])
templates = Jinja2Templates(directory="app/dashboard/templates")


def _dt(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _fmt(value: object | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


@router.get("/", include_in_schema=False)
def index() -> RedirectResponse:
    return RedirectResponse(url="/dashboard")


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard(
    request: Request,
    _: None = Depends(require_admin),
) -> HTMLResponse:
    dashboard_symbols = list(dict.fromkeys([*settings.binance_symbols, *settings.auto_trader_symbols, *settings.derivatives_symbols]))

    context: dict[str, Any] = {
        "request": request,
        "dashboard_symbols": dashboard_symbols,
        "default_symbol": dashboard_symbols[0] if dashboard_symbols else "BTCUSDT",
        "fmt": _fmt,
        "dt": _dt,
        "mode": settings.trading_mode,
    }
    response = templates.TemplateResponse(request, "dashboard.html", context)
    if admin_token_query_is_valid(request):
        response.set_cookie(
            ADMIN_COOKIE_NAME,
            request.query_params.get("admin_token") or request.query_params.get("token") or "",
            httponly=True,
            secure=request.url.scheme == "https",
            samesite="lax",
        )
    return response
