"""``/api/jobs``: the run queue: listing, progress, log tail, cancellation, reordering."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query

from qts_api import schemas as S
from qts_api.deps import get_jobs
from qts_api.problems import Problem
from qts_api.routers._common import problems
from qts_api.services.jobs import RECENT_TERMINAL, JobManager, JobNotFound, NotQueued

router = APIRouter(prefix="/api/jobs", tags=["jobs"])
Jobs = Annotated[JobManager, Depends(get_jobs)]


def _missing(job_id: str) -> Problem:
    return Problem(404, f"No job with id {job_id[:64]}", type_="job-not-found")


@router.get("", response_model=S.JobQueue, summary="The queue: running, queued (in order), recently finished")
def list_jobs(
    jobs: Jobs,
    recent: Annotated[
        int, Query(ge=0, le=200, description="how many finished / failed / cancelled jobs to include")
    ] = RECENT_TERMINAL,
) -> dict[str, Any]:
    return jobs.listing(recent)


@router.get("/{job_id}", response_model=S.Job, responses=problems(404))
def get_job(job_id: str, jobs: Jobs) -> dict[str, Any]:
    try:
        return jobs.view(jobs.get(job_id))
    except JobNotFound as e:
        raise _missing(job_id) from e


@router.get(
    "/{job_id}/log",
    response_model=S.JobLog,
    responses=problems(404),
    summary="Tail of the job's log (stage lines and the pipeline's INFO log)",
)
def get_job_log(
    job_id: str,
    jobs: Jobs,
    tail: Annotated[int, Query(ge=1, le=2000, description="number of lines from the end")] = 200,
) -> dict[str, Any]:
    try:
        return jobs.log(job_id, tail)
    except JobNotFound as e:
        raise _missing(job_id) from e


@router.post(
    "/{job_id}/move",
    response_model=S.Job,
    responses=problems(404, 409, 422),
    summary="Move a queued job to a new place in the queue",
)
def move_job(job_id: str, body: S.JobMove, jobs: Jobs) -> dict[str, Any]:
    try:
        return jobs.view(jobs.move(job_id, body.position))
    except JobNotFound as e:
        raise _missing(job_id) from e
    except NotQueued as e:
        raise Problem(
            409, "Only a queued job can be moved; this one has started or finished.", type_="job-not-queued"
        ) from e


@router.delete(
    "/{job_id}",
    response_model=S.Job,
    responses=problems(404),
    summary="Cancel: a queued job leaves the queue, a running one is terminated (best effort)",
)
def cancel_job(job_id: str, jobs: Jobs) -> dict[str, Any]:
    try:
        return jobs.view(jobs.cancel(job_id))
    except JobNotFound as e:
        raise _missing(job_id) from e
