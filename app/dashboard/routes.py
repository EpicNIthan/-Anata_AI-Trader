from __future__ import annotations

from datetime import datetime
from typing import Any
from urllib.parse import parse_qs

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.config import settings
from app.security import ADMIN_COOKIE_NAME, admin_token_is_valid, require_admin

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


def _dashboard_symbols() -> list[str]:
    """Return the configured symbol universe used by dashboard pages."""
    return list(
        dict.fromkeys(
            [
                *settings.binance_symbols,
                *settings.auto_trader_symbols,
                *settings.derivatives_symbols,
            ]
        )
    )


def _template_response(request: Request, template_name: str, context: dict[str, Any]) -> HTMLResponse:
    """Render an authenticated dashboard page without accepting URL credentials."""

    response = templates.TemplateResponse(request, template_name, context)
    if template_name != "dashboard.html":
        return response

    # Load the chart extension before dashboard.js. It expands historical candle
    # requests and decorates the existing paper-only chart with entry/exit/SL/TP.
    body = response.body.decode("utf-8")
    dashboard_script = '<script src="/static/dashboard.js?v=trade-terminal-v19"></script>'
    extension_scripts = (
        '<script src="/static/chart_extension.js?v=regime-chart-v1"></script>\n    '
        + dashboard_script
    )
    if dashboard_script in body:
        body = body.replace(dashboard_script, extension_scripts, 1)
    return HTMLResponse(content=body, status_code=response.status_code, headers=dict(response.headers))


def _safe_dashboard_next(value: str | None) -> str:
    candidate = str(value or "/vision")
    if not candidate.startswith("/") or candidate.startswith("//"):
        return "/vision"
    return candidate


@router.get("/admin/login", response_class=HTMLResponse, include_in_schema=False)
def admin_login(request: Request, next: str = "/vision") -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "admin_login.html",
        {"request": request, "next_path": _safe_dashboard_next(next)},
    )


@router.post("/admin/session", include_in_schema=False)
async def create_admin_session(request: Request) -> RedirectResponse:
    """Exchange a form-body token for an HTTP-only cookie; never read secrets from URLs."""

    fields = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
    token = (fields.get("admin_token") or [""])[0]
    next_path = _safe_dashboard_next((fields.get("next") or ["/vision"])[0])
    if not admin_token_is_valid(token):
        return RedirectResponse(url=f"/admin/login?next={next_path}", status_code=303)
    response = RedirectResponse(url=next_path, status_code=303)
    response.set_cookie(
        ADMIN_COOKIE_NAME,
        token,
        httponly=True,
        secure=request.url.scheme == "https",
        samesite="strict",
    )
    return response


@router.post("/admin/logout", include_in_schema=False)
def clear_admin_session() -> RedirectResponse:
    response = RedirectResponse(url="/admin/login", status_code=303)
    response.delete_cookie(ADMIN_COOKIE_NAME)
    return response


@router.get("/", include_in_schema=False)
def index() -> RedirectResponse:
    return RedirectResponse(url="/dashboard")


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard(
    request: Request,
    _: None = Depends(require_admin),
) -> HTMLResponse:
    dashboard_symbols = _dashboard_symbols()

    context: dict[str, Any] = {
        "request": request,
        "dashboard_symbols": dashboard_symbols,
        "default_symbol": dashboard_symbols[0] if dashboard_symbols else "BTCUSDT",
        "fmt": _fmt,
        "dt": _dt,
        "mode": settings.trading_mode,
    }
    return _template_response(request, "dashboard.html", context)


@router.get("/vision", response_class=HTMLResponse)
@router.get("/dashboard/vision", response_class=HTMLResponse, include_in_schema=False)
def vision(
    request: Request,
    _: None = Depends(require_admin),
) -> HTMLResponse:
    """Render the read-only AI Vision page without changing the admin dashboard."""
    dashboard_symbols = _dashboard_symbols()
    context: dict[str, Any] = {
        "request": request,
        "dashboard_symbols": dashboard_symbols,
        "default_symbol": dashboard_symbols[0] if dashboard_symbols else "BTCUSDT",
        "mode": settings.trading_mode,
        "vision_refresh_seconds": max(int(settings.vision_refresh_seconds), 5),
        "vision_default_limit": max(int(settings.vision_default_limit), 1),
    }
    return _template_response(request, "vision.html", context)
