from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from app.dashboard.routes import create_admin_session
from app.security import require_admin


def _request(*, query: bytes = b"", headers: list[tuple[bytes, bytes]] | None = None) -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "scheme": "https",
            "path": "/vision",
            "query_string": query,
            "headers": headers or [],
            "server": ("testserver", 443),
            "client": ("127.0.0.1", 1234),
        }
    )


def test_admin_secret_is_rejected_in_query_but_accepted_in_header() -> None:
    credentials = SimpleNamespace(
        admin_token="private-token",
        dashboard_username=None,
        dashboard_password=None,
    )
    with patch("app.security.settings", credentials):
        with pytest.raises(HTTPException) as error:
            require_admin(_request(query=b"admin_token=private-token"))
        assert error.value.status_code == 401

        require_admin(_request(headers=[(b"x-admin-token", b"private-token")]))


def test_admin_routes_fail_closed_when_authentication_is_not_configured() -> None:
    credentials = SimpleNamespace(
        admin_token=None,
        dashboard_username=None,
        dashboard_password=None,
    )
    with patch("app.security.settings", credentials), pytest.raises(HTTPException) as error:
        require_admin(_request())
    assert error.value.status_code == 503


def test_admin_form_exchanges_body_token_for_http_only_cookie_without_url_leak() -> None:
    body = b"admin_token=private-token&next=%2Fvision"

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "scheme": "https",
            "path": "/admin/session",
            "query_string": b"",
            "headers": [(b"content-type", b"application/x-www-form-urlencoded")],
            "server": ("testserver", 443),
            "client": ("127.0.0.1", 1234),
        },
        receive,
    )
    credentials = SimpleNamespace(
        admin_token="private-token",
        dashboard_username=None,
        dashboard_password=None,
    )
    with patch("app.security.settings", credentials):
        response = asyncio.run(create_admin_session(request))

    assert response.status_code == 303
    assert response.headers["location"] == "/vision"
    assert "private-token" not in response.headers["location"]
    cookie = response.headers["set-cookie"].lower()
    assert "httponly" in cookie
    assert "samesite=strict" in cookie
    assert "secure" in cookie
