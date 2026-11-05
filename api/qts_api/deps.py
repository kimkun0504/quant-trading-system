"""FastAPI dependencies: the per-app singletons live on ``app.state``."""

from __future__ import annotations

from fastapi import Request

from qts_api.services.jobs import JobManager
from qts_api.services.ledger import LedgerService
from qts_api.services.registry import RunRegistry
from qts_api.settings import Settings


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_registry(request: Request) -> RunRegistry:
    return request.app.state.registry


def get_jobs(request: Request) -> JobManager:
    return request.app.state.jobs


def get_ledger(request: Request) -> LedgerService:
    return request.app.state.ledger
