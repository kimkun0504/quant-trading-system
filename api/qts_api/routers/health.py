"""``/api/health``: liveness plus a little state."""

from __future__ import annotations

from typing import Annotated, Any

import qts
from fastapi import APIRouter, Depends

from qts_api import __version__
from qts_api import schemas as S
from qts_api.deps import get_jobs, get_registry
from qts_api.services.jobs import JobManager
from qts_api.services.registry import RunRegistry

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=S.Health)
def health(
    registry: Annotated[RunRegistry, Depends(get_registry)], jobs: Annotated[JobManager, Depends(get_jobs)]
) -> dict[str, Any]:
    return {
        "status": "ok",
        "version": __version__,
        "qts_version": str(getattr(qts, "__version__", "unknown")),
        "runs_root": str(registry.runs_root),
        "runs_root_exists": registry.runs_root.exists(),
        "n_runs": len(registry.run_dirs()),
        "jobs": jobs.counts(),
    }
