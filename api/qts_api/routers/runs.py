"""``/api/runs``: the run registry, its derived series, and launching new runs."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Response
from qts.core.manifest import git_info, new_run_id

from qts_api import schemas as S
from qts_api.deps import get_jobs, get_registry, get_settings
from qts_api.problems import Problem, summarize
from qts_api.routers._common import problems
from qts_api.routers.config import warmup_info
from qts_api.services.config_schema import ConfigError, validate_overrides
from qts_api.services.jobs import JobManager, QueueFull, default_label
from qts_api.services.registry import (
    COMPARE_MAX_RUNS,
    RUN_ID_MAX_LEN,
    RunNotFound,
    RunRegistry,
    RunUnreadable,
    valid_run_id,
)
from qts_api.settings import Settings

router = APIRouter(prefix="/api/runs", tags=["runs"])

Registry = Annotated[RunRegistry, Depends(get_registry)]
Jobs = Annotated[JobManager, Depends(get_jobs)]

RUNNING_DETAIL = "This run is still running; its results appear when the backtest finishes."
FAILED_DETAIL = "This run failed before evaluation; no results were written."


def _job_for_run(jobs: JobManager, run_id: str) -> dict[str, Any] | None:
    matches = [j for j in jobs.all() if j.get("run_id") == run_id]
    return max(matches, key=lambda j: str(j.get("created_at", ""))) if matches else None


def _pending_summary(j: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": j["run_id"],
        "created_at": str(j.get("created_at", "")),
        "label": str(j.get("label", "")),
        "kind": j.get("kind", "synthetic"),
        "phase": 2 if int(j.get("phase", 1)) == 2 else 1,
        "use_ml": bool(j.get("use_ml", False)),
        "config_hash": str(j.get("config_hash", "")),
        "data_hash": "",
        "git_sha": str(j.get("git_sha", "unknown")),
        "status": "failed" if j.get("status") == "failed" else "running",
        "headline": None,
        "config_leaves": {},
    }


def resolve(registry: RunRegistry, jobs: JobManager, run_id: str) -> str:
    """``run_id`` if its directory is readable; else 409 (pending / failed job) or 404.
    A malformed id (not a run-directory name, or longer than ``RUN_ID_MAX_LEN``) is a 404
    without touching the filesystem."""
    if not valid_run_id(run_id):
        shown = run_id if len(run_id) <= 64 else run_id[:64] + "…"
        raise Problem(
            404,
            f"No run with id {shown} (run ids are letters, digits, '.', '_' or '-', at most {RUN_ID_MAX_LEN})",
            type_="run-not-found",
        )
    try:
        registry.get(run_id)
        return run_id
    except RunUnreadable as e:
        raise Problem(500, f"Run directory {run_id} is unreadable: {e}", type_="run-unreadable") from e
    except RunNotFound:
        pass
    j = _job_for_run(jobs, run_id)
    if j is not None and j.get("status") != "done":
        failed = j.get("status") == "failed"
        detail = FAILED_DETAIL if failed else RUNNING_DETAIL
        if failed and j.get("message"):
            detail = f"{detail} {j['message']}"
        raise Problem(
            409,
            detail,
            type_="run-failed" if failed else "run-running",
            title="Run failed" if failed else "Run still running",
        )
    raise Problem(404, f"No run with id {run_id}", type_="run-not-found")


@router.get("", response_model=list[S.RunSummary], summary="All runs, newest first")
def list_runs(registry: Registry, jobs: Jobs) -> list[dict[str, Any]]:
    done = registry.list_summaries()
    have = {r["id"] for r in done}
    pending = [_pending_summary(j) for j in jobs.all() if j.get("status") != "done" and j.get("run_id") not in have]
    return sorted(done + pending, key=lambda r: r["created_at"], reverse=True)


@router.get(
    "/compare",
    response_model=S.RunCompare,
    responses=problems(404, 409, 422),
    summary="Several runs side by side: aligned NAV, fold Sharpes, gates, flattened configs",
)
def compare_runs(
    registry: Registry,
    jobs: Jobs,
    ids: Annotated[
        str,
        Query(description=f"comma-separated run ids, 1 to {COMPARE_MAX_RUNS} (duplicates ignored)"),
    ],
) -> dict[str, Any]:
    seen: list[str] = []
    for rid in (x.strip() for x in ids.split(",")):
        if rid and rid not in seen:
            seen.append(rid)
    if not seen or len(seen) > COMPARE_MAX_RUNS:
        issue = {
            "loc": ["query", "ids"],
            "msg": f"give 1 to {COMPARE_MAX_RUNS} run ids, comma-separated",
            "type": "value_error",
        }
        raise Problem(422, summarize([issue]), type_="validation-error", errors=[issue])
    return registry.compare([resolve(registry, jobs, rid) for rid in seen])


@router.get("/{run_id}", response_model=S.RunDetail, responses=problems(404, 409), summary="One run")
def get_run(run_id: str, registry: Registry, jobs: Jobs) -> dict[str, Any]:
    return registry.detail(resolve(registry, jobs, run_id))


@router.get(
    "/{run_id}/series",
    response_model=S.ColumnarSeries,
    responses=problems(404, 409, 422),
    summary="Return series on the realisation-date axis",
)
def get_series(
    run_id: str,
    registry: Registry,
    jobs: Jobs,
    names: Annotated[
        str | None,
        Query(description=f"comma-separated subset of {', '.join(S.SERIES_NAMES)} (default: all)"),
    ] = None,
) -> dict[str, Any]:
    requested = [n.strip() for n in (names or "").split(",") if n.strip()]
    unknown = [n for n in requested if n not in S.SERIES_NAMES]
    if unknown:
        raise Problem(
            422,
            f"Unknown series {unknown}; choose from {list(S.SERIES_NAMES)}",
            type_="validation-error",
            errors=[
                {"loc": ["query", "names"], "msg": f"unknown series {n!r}", "type": "value_error"} for n in unknown
            ],
        )
    return registry.series(resolve(registry, jobs, run_id), requested or list(S.SERIES_NAMES))


@router.get("/{run_id}/costs", response_model=S.RunCosts, responses=problems(404, 409))
def get_costs(run_id: str, registry: Registry, jobs: Jobs) -> dict[str, Any]:
    return registry.costs(resolve(registry, jobs, run_id))


@router.get("/{run_id}/exposures", response_model=S.RunExposures, responses=problems(404, 409))
def get_exposures(run_id: str, registry: Registry, jobs: Jobs) -> dict[str, Any]:
    return registry.exposures(resolve(registry, jobs, run_id))


@router.get("/{run_id}/holdings", response_model=S.RunHoldings, responses=problems(404, 409))
def get_holdings(
    run_id: str,
    registry: Registry,
    jobs: Jobs,
    date: Annotated[str | None, Query(description="YYYY-MM-DD, one of available_dates (default: latest)")] = None,
) -> dict[str, Any]:
    rid = resolve(registry, jobs, run_id)
    try:
        return registry.holdings(rid, date)
    except KeyError as e:
        raise Problem(404, f"No holdings on {date} for run {run_id}; see available_dates", type_="holdings-date") from e
    except RunNotFound as e:
        raise Problem(404, str(e), type_="holdings-empty") from e


@router.get("/{run_id}/ic", response_model=S.RunIc, responses=problems(404, 409))
def get_ic(run_id: str, registry: Registry, jobs: Jobs) -> dict[str, Any]:
    return registry.ic(resolve(registry, jobs, run_id))


@router.post(
    "",
    response_model=S.LaunchRunResponse,
    status_code=202,
    responses=problems(422, 429),
    summary="Launch a pipeline run as a background job",
)
def launch_run(
    body: S.LaunchRunRequest,
    response: Response,
    jobs: Jobs,
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    if body.phase == 2 and not body.ml:
        issues.append(
            {"loc": ["body", "ml"], "msg": "phase 2 tests the ML sleeve: set ml=true or phase=1", "type": "value_error"}
        )
    if body.kind == "store" and settings.store_path is None:
        issues.append(
            {
                "loc": ["body", "kind"],
                "msg": "store runs need QTS_STORE_PATH set on the API server",
                "type": "value_error",
            }
        )
    try:
        cfg = validate_overrides(body.config_overrides)
    except ConfigError as e:
        issues += [{"loc": loc, "msg": msg, "type": t} for loc, msg, t in e.issues]
        cfg = None
    if issues or cfg is None:
        raise Problem(422, summarize(issues), type_="validation-error", title="Invalid run request", errors=issues)
    cfg = cfg.with_overrides({"seed": body.seed, "execution.signal_lag_days": body.signal_lag_days})
    if body.kind == "synthetic":
        cfg = cfg.with_overrides({"universe.top_n_by_mcap": min(cfg.universe.top_n_by_mcap, body.stocks)})
        # P7: the floor is the run's own effective warm-up plus one year
        wu = warmup_info(cfg, body.ml)
        if body.days < wu["min_days"]:
            issue = {
                "loc": ["body", "days"],
                "msg": (
                    f"days must be at least {wu['min_days']} ({wu['warmup_days']} warm-up days "
                    f"for this configuration plus one year)"
                ),
                "type": "value_error",
            }
            raise Problem(
                422, summarize([issue]), type_="validation-error", title="Invalid run request", errors=[issue]
            )
    cfg.validate()
    config_hash = cfg.hash()
    run_id = new_run_id(config_hash)
    label = body.label or default_label(body.ml, body.seed, body.signal_lag_days)
    try:
        job = jobs.submit(
            request=body.model_dump(),
            config=cfg.to_dict(),
            config_hash=config_hash,
            run_id=run_id,
            label=label,
            git_sha=str(git_info()["git_sha"]),
        )
    except QueueFull as e:
        raise Problem(
            429, str(e), type_="queue-full", title="Too many queued runs", headers={"Retry-After": "60"}
        ) from e
    response.headers["Location"] = f"/api/jobs/{job['job_id']}"
    return {"job_id": job["job_id"], "run_id": run_id}
