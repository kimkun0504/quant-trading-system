"""FastAPI application: ``uvicorn qts_api.main:app --reload``.

``create_app(settings)`` builds an app with its own run registry, ledger reader and
job manager (tests create one per temporary runs root); ``app`` is the module-level
instance configured from the environment (:mod:`qts_api.settings`).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from qts_api import __version__, problems
from qts_api.problems import Problem
from qts_api.routers import config, health, jobs, ledger, runs
from qts_api.services.jobs import JobManager
from qts_api.services.ledger import LedgerService
from qts_api.services.registry import RunRegistry
from qts_api.settings import Settings

log = logging.getLogger("qts_api")

DESCRIPTION = """
Run registry, trial ledger, config schema and background pipeline jobs of the `qts`
research workbench. Errors are RFC 7807 problem details (`application/problem+json`)
with a string `detail`. Non-finite floats (NaN / inf) are serialised as `null`.
"""


def create_app(settings: Settings | None = None, *, start_jobs: bool = True) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.runs_root.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        app.state.jobs.shutdown()

    app = FastAPI(
        title="qts research workbench API",
        version=__version__,
        description=DESCRIPTION,
        openapi_url="/api/openapi.json",
        docs_url="/api/docs",
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.registry = RunRegistry(settings.runs_root)
    app.state.ledger = LedgerService(settings.ledger)
    app.state.jobs = JobManager(settings, start=start_jobs)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["Location"],
    )
    problems.install(app)
    for r in (runs.router, jobs.router, ledger.router, config.router, health.router):
        app.include_router(r)

    @app.api_route("/api/{rest:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"], include_in_schema=False)
    def api_not_found(rest: str) -> None:
        raise Problem(404, f"No API endpoint /api/{rest}", type_="endpoint-not-found")

    if settings.serve_ui:
        _mount_ui(app, settings.ui_dist)
    _add_problem_schemas(app)
    return app


def _add_problem_schemas(app: FastAPI) -> None:
    """Error responses reference ``ApiErrorBody`` under ``application/problem+json``;
    FastAPI only collects schemas of ``model=`` responses, so add it (and its parts)."""
    from fastapi.openapi.utils import get_openapi

    from qts_api.schemas import ApiErrorBody

    def openapi() -> dict:
        if app.openapi_schema:
            return app.openapi_schema
        doc = get_openapi(title=app.title, version=app.version, description=app.description, routes=app.routes)
        extra = ApiErrorBody.model_json_schema(ref_template="#/components/schemas/{model}", mode="validation")
        comps = doc.setdefault("components", {}).setdefault("schemas", {})
        comps.update(extra.pop("$defs", {}))
        for prop in extra.get("properties", {}).values():  # optional, not "defaults to null"
            if prop.get("default", 0) is None:
                prop.pop("default")
        comps["ApiErrorBody"] = extra
        app.openapi_schema = doc
        return doc

    app.openapi = openapi  # type: ignore[method-assign]


def _mount_ui(app: FastAPI, dist: Path) -> None:
    """Serve the built UI at ``/`` with an SPA fallback to ``index.html``."""
    index = dist / "index.html"
    if not index.exists():
        log.warning("QTS_SERVE_UI is set but %s does not exist (run `npm --prefix ui run build`)", index)
        return
    if (dist / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")
    root = dist.resolve()

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        f = (root / path).resolve()
        if path and f.is_file() and root in f.parents:
            return FileResponse(f)
        return FileResponse(index, headers={"Cache-Control": "no-cache"})


_app: FastAPI | None = None


def __getattr__(name: str) -> FastAPI:
    """``qts_api.main.app`` is built on first access (uvicorn's ``getattr``), so importing
    this module (tests, OpenAPI export) does not touch ``QTS_RUNS_ROOT``."""
    global _app
    if name == "app":
        if _app is None:
            _app = create_app()
        return _app
    raise AttributeError(name)
