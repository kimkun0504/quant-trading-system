"""RFC 7807 problem details (``application/problem+json``).

Every error body has ``type``, ``title``, ``status`` and a **string** ``detail`` (the
UI shows ``detail`` verbatim); validation errors add ``errors: [{loc, msg, type}]``.
"""

from __future__ import annotations

from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

PROBLEM_JSON = "application/problem+json"
TYPE_BASE = "https://qts.local/problems/"


class Problem(Exception):
    def __init__(
        self,
        status: int,
        detail: str,
        *,
        type_: str | None = None,
        title: str | None = None,
        errors: list[dict[str, Any]] | None = None,
        headers: dict[str, str] | None = None,
    ):
        super().__init__(detail)
        self.status = status
        self.detail = detail
        self.type = type_
        self.title = title
        self.errors = errors
        self.headers = headers


def problem_response(
    status: int,
    detail: str,
    *,
    type_: str | None = None,
    title: str | None = None,
    errors: list[dict[str, Any]] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    try:
        phrase = HTTPStatus(status).phrase
    except ValueError:
        phrase = "Error"
    body: dict[str, Any] = {
        "type": f"{TYPE_BASE}{type_}" if type_ else "about:blank",
        "title": title or phrase,
        "status": status,
        "detail": detail,
    }
    if errors is not None:
        body["errors"] = errors
    return JSONResponse(body, status_code=status, media_type=PROBLEM_JSON, headers=headers)


def _loc(loc: Any) -> list[str | int]:
    return [x if isinstance(x, (int, str)) else str(x) for x in (loc or [])]


def validation_errors(exc: RequestValidationError) -> list[dict[str, Any]]:
    return [
        {"loc": _loc(e.get("loc")), "msg": str(e.get("msg", "")), "type": str(e.get("type", ""))} for e in exc.errors()
    ]


def summarize(errors: list[dict[str, Any]]) -> str:
    parts = []
    for e in errors[:5]:
        loc = [str(x) for x in e["loc"] if x not in ("body", "query", "path")]
        parts.append(f"{'.'.join(loc) or 'request'}: {e['msg']}")
    more = f" (+{len(errors) - 5} more)" if len(errors) > 5 else ""
    return "; ".join(parts) + more


def install(app: FastAPI) -> None:
    @app.exception_handler(Problem)
    async def _problem(_: Request, exc: Problem) -> JSONResponse:
        return problem_response(
            exc.status, exc.detail, type_=exc.type, title=exc.title, errors=exc.errors, headers=exc.headers
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        errors = validation_errors(exc)
        return problem_response(
            422, summarize(errors), type_="validation-error", title="Invalid request", errors=errors
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        return problem_response(exc.status_code, detail, headers=getattr(exc, "headers", None))

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        import logging

        logging.getLogger("qts_api").exception("unhandled error")
        return problem_response(500, f"Internal error: {type(exc).__name__}", type_="internal")
