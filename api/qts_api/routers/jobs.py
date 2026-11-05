"""``/api/jobs``: progress and cancellation of launched runs."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends

from qts_api import schemas as S
from qts_api.deps import get_jobs
from qts_api.problems import Problem
from qts_api.routers._common import problems
from qts_api.services.jobs import JobManager, JobNotFound

router = APIRouter(prefix="/api/jobs", tags=["jobs"])
Jobs = Annotated[JobManager, Depends(get_jobs)]


def _job(j: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_id": j["job_id"],
        "run_id": j.get("run_id", ""),
        "status": j.get("status", "failed"),
        "progress": float(j.get("progress", 0.0) or 0.0),
        "message": str(j.get("message", "")),
    }


@router.get("/{job_id}", response_model=S.Job, responses=problems(404))
def get_job(job_id: str, jobs: Jobs) -> dict[str, Any]:
    try:
        return _job(jobs.get(job_id))
    except JobNotFound as e:
        raise Problem(404, f"No job with id {job_id}", type_="job-not-found") from e


@router.delete("/{job_id}", response_model=S.Job, responses=problems(404), summary="Cancel (best effort)")
def cancel_job(job_id: str, jobs: Jobs) -> dict[str, Any]:
    try:
        return _job(jobs.cancel(job_id))
    except JobNotFound as e:
        raise Problem(404, f"No job with id {job_id}", type_="job-not-found") from e
