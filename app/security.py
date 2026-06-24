from __future__ import annotations

import base64
import secrets

from fastapi import HTTPException, Request, status

from app.config import settings


ADMIN_COOKIE_NAME = "anata_admin_token"


def auth_enabled() -> bool:
    return bool(settings.admin_token or (settings.dashboard_username and settings.dashboard_password))


def admin_token_query_is_valid(request: Request) -> bool:
    token = request.query_params.get("admin_token") or request.query_params.get("token")
    return _token_valid(token)


def require_admin(request: Request) -> None:
    if not auth_enabled():
        return
    if _request_is_authorized(request):
        return
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Basic"},
    )


def _request_is_authorized(request: Request) -> bool:
    authorization = request.headers.get("authorization") or ""
    if authorization.lower().startswith("bearer "):
        return _token_valid(authorization.split(" ", 1)[1].strip())
    if authorization.lower().startswith("basic "):
        return _basic_valid(authorization.split(" ", 1)[1].strip())
    return (
        _token_valid(request.headers.get("x-admin-token"))
        or _token_valid(request.cookies.get(ADMIN_COOKIE_NAME))
        or admin_token_query_is_valid(request)
    )


def _token_valid(value: str | None) -> bool:
    return bool(value and settings.admin_token and secrets.compare_digest(value, settings.admin_token))


def _basic_valid(encoded: str) -> bool:
    if not settings.dashboard_username or not settings.dashboard_password:
        return False
    try:
        decoded = base64.b64decode(encoded).decode("utf-8")
    except Exception:
        return False
    username, separator, password = decoded.partition(":")
    if not separator:
        return False
    return secrets.compare_digest(username, settings.dashboard_username) and secrets.compare_digest(
        password,
        settings.dashboard_password,
    )
