"""Research tools: the holdout seal, trial families and impact, features, signal tests,
sweeps, the capacity curve, the risk calibration report and the mid-cap check.

Reads serve the library's own result files; launches queue background jobs
(:mod:`qts_api.services.tasks`) and answer 202 with the job id and the equivalent
``qts`` command.  Every launch that adds trials has a preview route that says, before
anything is queued, how many trials it adds, to which family, and (sweeps) the DSR
haircut on the family's current best trial (:func:`qts.research.sweep.trial_impact`).
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Response

from qts_api import schemas as S
from qts_api.deps import get_jobs, get_registry, get_settings
from qts_api.problems import Problem, summarize
from qts_api.routers._common import problems
from qts_api.services import research as R
from qts_api.services.jobs import JobManager, QueueFull
from qts_api.services.registry import RunNotFound, RunRegistry, RunUnreadable
from qts_api.services.tasks import task_command
from qts_api.settings import Settings

router = APIRouter(prefix="/api", tags=["research"])
Jobs = Annotated[JobManager, Depends(get_jobs)]
Cfg = Annotated[Settings, Depends(get_settings)]
Registry = Annotated[RunRegistry, Depends(get_registry)]

ACTIVE = ("queued", "running")


def _invalid(loc: list[str | int], msg: str) -> Problem:
    issue = {"loc": loc, "msg": msg, "type": "value_error"}
    return Problem(422, summarize([issue]), type_="validation-error", title="Invalid request", errors=[issue])


def _need_store(settings: Settings) -> None:
    if settings.store_path is None:
        raise _invalid(["body", "store"], "research tasks read the data store: set QTS_STORE_PATH on the API server")


def _coverage(
    settings: Settings, store: dict[str, Any], config_path: str, overrides: list[dict[str, Any]] | None = None
) -> None:
    """Refuse (422) a research launch whose store prefilter cannot cover the universe of
    the configuration(s) it runs: the library's load-coverage guard where it needs no store
    read (:func:`qts_api.services.launch.coverage_precheck`); a peak-cap prefilter is
    measured inside the job, which then fails with ``error_kind: coverage``."""
    from qts.core.config import SystemConfig
    from qts.research.common import resolve_repo_path
    from qts.research.data import StoreSpec

    from qts_api.services.launch import coverage_precheck

    try:
        spec = StoreSpec.from_dict(R.store_spec_dict(settings, store))
        base = SystemConfig.from_yaml(str(resolve_repo_path(config_path)))
    except (OSError, ValueError, KeyError):
        return  # an unreadable config or slice is reported by the job itself
    for ov in overrides or [{}]:
        try:
            cfg = base.with_overrides(ov)
        except (KeyError, TypeError, ValueError):
            continue
        issue = coverage_precheck(spec, cfg, False, [], ["body", "store", "max_rank"])
        if issue is not None:
            raise Problem(422, issue["msg"], type_="load-coverage", title="Incomplete store load", errors=[issue])


def _missing(what: str, rid: str) -> Problem:
    return Problem(404, f"No {what} with id {rid[:80]}", type_="result-not-found")


def _queue(
    jobs: JobManager,
    response: Response,
    *,
    task: str,
    params: dict[str, Any],
    label: str,
    subject: str | None,
    use_ml: bool = True,
    size_key: Any = None,
    config_hash: str = "",
) -> dict[str, Any]:
    s = jobs.settings
    command = task_command(task, params, str(s.store_path) if s.store_path else None, str(s.paper_root))
    try:
        job = jobs.submit_task(
            task=task,
            params=params,
            label=label,
            subject=subject,
            command=command,
            use_ml=use_ml,
            size_key=size_key,
            config_hash=config_hash,
        )
    except QueueFull as e:
        raise Problem(
            429, str(e), type_="queue-full", title="Too many queued jobs", headers={"Retry-After": "60"}
        ) from e
    response.headers["Location"] = f"/api/jobs/{job['job_id']}"
    return {"job_id": job["job_id"], "label": label, "command": command}


def _views(jobs: JobManager, task: str, subject: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
    hist = jobs.all()
    return [jobs.view(j, hist) for j in jobs.tasks(task, subject)[:limit]]


# ============================================================================= seal
@router.get(
    "/seal", response_model=S.SealStatus, tags=["seal"], summary="The holdout seal, every unseal, runs under it"
)
def get_seal(settings: Cfg) -> dict[str, Any]:
    from qts.research.seal import SealError

    try:
        return R.seal_view(settings)
    except SealError as e:
        raise Problem(500, f"The seal file is unreadable: {e}", type_="seal-unreadable") from e


@router.get(
    "/seal/unseals",
    response_model=S.UnsealCheck,
    tags=["seal"],
    summary="Earlier unseals of one configuration (a second needs force and is flagged)",
)
def get_unseals(
    settings: Cfg,
    config_hash: Annotated[str, Query(min_length=8, max_length=64, pattern=r"^[0-9a-f]+$")],
    phase: Annotated[int, Query(ge=1, le=2)] = 2,
) -> dict[str, Any]:
    return R.unseal_check(settings, config_hash, phase)


# ============================================================================= families
@router.get(
    "/ledger/impact",
    response_model=S.TrialImpact,
    tags=["ledger"],
    summary="What k new trials do to a family's deflated Sharpe ratio (trial_impact)",
)
def get_trial_impact(
    settings: Cfg,
    family: Annotated[str, Query(min_length=3, max_length=120)],
    n_new: Annotated[int, Query(ge=0, le=100)] = 1,
    n_reference: Annotated[int, Query(ge=0, le=100)] = 0,
) -> dict[str, Any]:
    """The family's trial count before / after, ``N_eff``, ``E[max SR]`` and the current
    best trial's DSR before and after ``n_new`` independent new trials
    (:func:`qts.research.sweep.trial_impact`, the numbers ``qts sweep`` prints)."""
    return R.trial_impact(settings, family, n_new, n_reference)


@router.post(
    "/families/resolve",
    response_model=S.FamilyResolution,
    tags=["ledger"],
    summary="The full-history market hash of a store slice, when cached (no store read)",
)
def resolve_family(body: S.StoreSlice, settings: Cfg) -> dict[str, Any]:
    return R.resolve_family(settings, body.model_dump())


@router.post(
    "/families/resolve/jobs",
    response_model=S.TaskLaunchResponse,
    status_code=202,
    responses=problems(422, 429),
    tags=["ledger"],
    summary="Compute and cache the family hash of a store slice (a background job, about a minute)",
)
def launch_resolve_family(body: S.StoreSlice, settings: Cfg, jobs: Jobs, response: Response) -> dict[str, Any]:
    _need_store(settings)
    store = body.model_dump()
    return _queue(
        jobs, response, task="market_hash", params={"store": store}, label="Resolve the trial family of a store slice",
        subject=None,
    )  # fmt: skip


# ============================================================================= features
@router.get("/features", response_model=S.FeatureList, tags=["features"], summary="The alpha features of a feature set")
def get_features(feature_set: Annotated[str, Query(pattern=r"^v[0-9]+$")] = "v1") -> dict[str, Any]:
    try:
        return R.feature_list(feature_set)
    except ValueError as e:
        raise _invalid(["query", "feature_set"], str(e)) from e


# ============================================================================= signals
@router.get("/signals", response_model=list[S.SignalDef], tags=["signals"], summary="Every testable signal")
def list_signals() -> list[dict[str, Any]]:
    from qts.research.signals import list_signals as _list

    return [s.to_dict() for s in _list()]


@router.get("/signals/tests", response_model=S.SignalTestList, tags=["signals"], summary="Signal tests, newest first")
def list_signal_tests(settings: Cfg) -> dict[str, Any]:
    return {"tests": R.signal_tests(settings)}


@router.get(
    "/signals/tests/{test_id}",
    response_model=S.SignalTestDetail,
    responses=problems(404),
    tags=["signals"],
    summary="One signal test, with its family's multiple-testing state now",
)
def get_signal_test(test_id: str, settings: Cfg) -> dict[str, Any]:
    try:
        return R.signal_test_detail(settings, test_id)
    except R.NotFound as e:
        raise _missing("signal test", test_id) from e


def _signal_params(body: S.SignalTestRequest) -> dict[str, Any]:
    return {**body.model_dump(), "store": body.store.model_dump()}


@router.post(
    "/signals/tests/preview",
    response_model=S.SignalTestPreview,
    responses=problems(422),
    tags=["signals"],
    summary="Before a test: the family it joins and that family's test count",
)
def preview_signal_test(body: S.SignalTestRequest, settings: Cfg) -> dict[str, Any]:
    from qts.research.signals import FAMILY_PREFIX, get_signal

    try:
        get_signal(body.feature)
    except KeyError as e:
        raise _invalid(["body", "feature"], str(e.args[0] if e.args else e)) from e
    res = R.resolve_family(settings, body.store.model_dump())
    family = f"{FAMILY_PREFIX}{res['market_hash_full'][:12]}" if res["known"] else None
    n = len(R.family_now(settings, family)["tests"]) if family else 0
    adds = 1 if body.log_trial else 0
    s = settings
    return {
        "family": family,
        "resolution": res,
        "n_tests": n,
        "n_tests_after": n + adds,
        "adds": adds,
        "hurdle": 3.0,
        "command": task_command("signal_test", _signal_params(body), str(s.store_path) if s.store_path else None),
    }


@router.post(
    "/signals/tests",
    response_model=S.TaskLaunchResponse,
    status_code=202,
    responses=problems(422, 429),
    tags=["signals"],
    summary="Run a signal test as a background job (one trial in signal:<hash>)",
)
def launch_signal_test(body: S.SignalTestRequest, settings: Cfg, jobs: Jobs, response: Response) -> dict[str, Any]:
    from qts.research.signals import get_signal

    _need_store(settings)
    try:
        get_signal(body.feature)
    except KeyError as e:
        raise _invalid(["body", "feature"], str(e.args[0] if e.args else e)) from e
    _coverage(settings, body.store.model_dump(), body.baseline)
    params = _signal_params(body)
    return _queue(
        jobs, response, task="signal_test", params=params, subject=body.feature, use_ml=body.use_ml,
        label=f"Signal test: {body.feature} (h {','.join(map(str, body.horizons))})",
    )  # fmt: skip


# ============================================================================= sweeps
def _plan(body: S.SweepRequest, settings: Settings) -> Any:
    """``qts.research.sweep.plan_sweep`` with the family resolved from the cached hash, so
    planning never reads the store (409 when the family is not known yet)."""
    from qts.research.common import resolve_repo_path
    from qts.research.data import StoreSpec
    from qts.research.sweep import SweepError, plan_sweep
    from qts.validation.ledger import research_family

    _need_store(settings)
    res = R.resolve_family(settings, body.store.model_dump())
    if not res["known"]:
        raise Problem(409, res["reason"], type_="family-unknown", title="Trial family not known yet")
    params = [(p.key, list(p.values)) for p in body.params]
    try:
        spec = StoreSpec.from_dict(R.store_spec_dict(settings, body.store.model_dump()))
        return plan_sweep(
            params,
            base_config=str(resolve_repo_path(body.base_config))
            if body.base_config.startswith("runs/")
            else body.base_config,
            mode=body.mode,
            spec=spec,
            family=research_family(2 if body.use_ml else 1, res["market_hash_full"]),
            use_ml=body.use_ml,
            nav=body.nav,
            max_runs=body.max_runs,
            max_days=body.max_days,
            ledger_path=str(settings.ledger),
            runs_root=str(settings.runs_root),
        )
    except (SweepError, KeyError, ValueError, OSError) as e:
        raise _invalid(["body", "params"], str(e.args[0] if e.args else e)) from e


def _sweep_command(plan: dict[str, Any], settings: Settings) -> str:
    return task_command(
        "sweep_member", R.sweep_task_params(plan), str(settings.store_path) if settings.store_path else None
    )


@router.post(
    "/sweeps/plan",
    response_model=S.SweepPlanView,
    responses=problems(409, 422),
    tags=["sweeps"],
    summary="Plan a sweep: members, family, trials added and the DSR haircut (nothing written)",
)
def plan_sweep(body: S.SweepRequest, settings: Cfg) -> dict[str, Any]:
    plan = _plan(body, settings)
    _coverage(settings, body.store.model_dump(), body.base_config, [dict(m.overrides) for m in plan.members])
    d = plan.to_dict()
    return {
        "plan": d,
        "exists": plan.path.exists(),
        "n_pending": sum(1 for m in plan.members if m.status != "done"),
        "command": _sweep_command(d, settings),
    }


def _active_members(jobs: JobManager, sweep_id: str) -> tuple[set[int], bool]:
    members: set[int] = set()
    finalize = False
    for j in jobs.tasks("sweep_member", sweep_id):
        if j.get("status") in ACTIVE:
            members.add(int((j.get("request") or {}).get("index", -1)))
    for j in jobs.tasks("sweep_finalize", sweep_id):
        finalize = finalize or j.get("status") in ACTIVE
    return members, finalize


def _queue_sweep(plan: dict[str, Any], jobs: JobManager, response: Response, settings: Settings) -> dict[str, Any]:
    active, finalizing = _active_members(jobs, plan["id"])
    todo = [m for m in plan["members"] if m.get("status") != "done" and int(m["index"]) not in active]
    need = len(todo) + (0 if finalizing else 1)
    with jobs._cv:
        free = jobs.max_queued - len(jobs._queue)
    if need > free:
        raise Problem(
            429,
            f"the sweep needs {need} jobs but the queue has room for {max(free, 0)} (QTS_MAX_QUEUED "
            f"{jobs.max_queued}); wait for queued jobs to finish or cancel some",
            type_="queue-full",
            title="Too many queued jobs",
            headers={"Retry-After": "60"},
        )
    base = R.sweep_task_params(plan)
    n = len(plan["members"])
    ids = []
    for m in todo:
        out = _queue(
            jobs, response, task="sweep_member", params={**base, "index": int(m["index"])}, subject=plan["id"],
            label=f"Sweep {plan['id']}: member {int(m['index']) + 1} of {n} {m.get('overrides')}",
            use_ml=bool(plan.get("use_ml", True)), config_hash=str(m.get("config_hash", "")),
            size_key={"max_days": plan.get("max_days"), "use_ml": plan.get("use_ml")},
        )  # fmt: skip
        ids.append(out["job_id"])
    fin = None
    if not finalizing:
        fin = _queue(
            jobs, response, task="sweep_finalize", params=base, subject=plan["id"],
            label=f"Sweep {plan['id']}: PBO and the metric-vs-parameter table", use_ml=bool(plan.get("use_ml", True)),
        )["job_id"]  # fmt: skip
    return {
        "sweep_id": plan["id"],
        "job_ids": ids,
        "finalize_job_id": fin,
        "n_queued": len(ids) + (1 if fin else 0),
        "command": _sweep_command(plan, settings),
    }


@router.post(
    "/sweeps",
    response_model=S.SweepLaunchResponse,
    status_code=202,
    responses=problems(409, 422, 429),
    tags=["sweeps"],
    summary="Create the confirmed sweep and queue one job per member, then the PBO job",
)
def create_sweep(body: S.SweepCreate, settings: Cfg, jobs: Jobs, response: Response) -> dict[str, Any]:
    """``create_sweep(plan)`` then one ``run_sweep_member`` job per member that is not
    done (members run one at a time with the stage cache) and a ``finalize_sweep`` job.
    The plan must be the one confirmed (``expected_id`` = the preview's id)."""
    from qts.research.sweep import create_sweep as _create

    plan = _plan(S.SweepRequest(**body.model_dump(exclude={"expected_id"})), settings)
    if plan.id != body.expected_id:
        raise Problem(
            409,
            f"the sweep changed since it was confirmed ({body.expected_id} -> {plan.id}); review the new plan",
            type_="plan-changed",
        )
    _coverage(settings, body.store.model_dump(), body.base_config, [dict(m.overrides) for m in plan.members])
    _create(plan)
    return _queue_sweep(plan.to_dict(), jobs, response, settings)


@router.get("/sweeps", response_model=S.SweepList, tags=["sweeps"], summary="Every sweep manifest, newest first")
def list_sweeps(settings: Cfg) -> dict[str, Any]:
    return {"sweeps": R.sweep_summaries(settings)}


def _sweep_detail(sweep_id: str, settings: Settings, jobs: JobManager) -> dict[str, Any]:
    try:
        d = R.sweep_manifest(settings, sweep_id)
    except R.NotFound as e:
        raise _missing("sweep", sweep_id) from e
    views = _views(jobs, "sweep_member", sweep_id, 40) + _views(jobs, "sweep_finalize", sweep_id, 5)
    views.sort(key=lambda j: j["created_at"], reverse=True)
    active, _ = _active_members(jobs, sweep_id)
    resumable = sum(1 for m in d.get("members") or [] if m.get("status") != "done" and int(m["index"]) not in active)
    return {
        "sweep": d,
        "jobs": views,
        "active": sum(1 for j in views if j["status"] in ACTIVE),
        "resumable": resumable,
        "command": _sweep_command(d, settings),
    }


@router.get(
    "/sweeps/{sweep_id}",
    response_model=S.SweepDetail,
    responses=problems(404),
    tags=["sweeps"],
    summary="One sweep: members, metrics, PBO, and its jobs",
)
def get_sweep(sweep_id: str, settings: Cfg, jobs: Jobs) -> dict[str, Any]:
    return _sweep_detail(sweep_id, settings, jobs)


@router.post(
    "/sweeps/{sweep_id}/resume",
    response_model=S.SweepLaunchResponse,
    status_code=202,
    responses=problems(404, 409, 429),
    tags=["sweeps"],
    summary="Queue the members that are not done (an interrupted sweep), then PBO again",
)
def resume_sweep(sweep_id: str, settings: Cfg, jobs: Jobs, response: Response) -> dict[str, Any]:
    _need_store(settings)
    try:
        d = R.sweep_manifest(settings, sweep_id)
    except R.NotFound as e:
        raise _missing("sweep", sweep_id) from e
    active, _ = _active_members(jobs, sweep_id)
    if not [m for m in d.get("members") or [] if m.get("status") != "done" and int(m["index"]) not in active]:
        raise Problem(409, "nothing to resume: every member is done or already queued", type_="nothing-to-resume")
    return _queue_sweep(d, jobs, response, settings)


# ============================================================================= capacity
@router.get("/capacity", response_model=S.CapacityList, tags=["capacity"], summary="Capacity curves, newest first")
def list_capacity(settings: Cfg) -> dict[str, Any]:
    return {"results": R.capacity_summaries(settings)}


@router.get("/capacity/{result_id}", response_model=S.CapacityResult, responses=problems(404), tags=["capacity"])
def get_capacity(result_id: str, settings: Cfg) -> dict[str, Any]:
    try:
        return R.result_file(settings.runs_root, R.CAPACITY_DIR, result_id)
    except R.NotFound as e:
        raise _missing("capacity curve", result_id) from e


@router.post(
    "/capacity",
    response_model=S.TaskLaunchResponse,
    status_code=202,
    responses=problems(422, 429),
    tags=["capacity"],
    summary="Re-run the baseline backtest at several NAVs (a background job; not trials)",
)
def launch_capacity(body: S.CapacityRequest, settings: Cfg, jobs: Jobs, response: Response) -> dict[str, Any]:
    _need_store(settings)
    _coverage(settings, body.store.model_dump(), body.baseline)
    params = {**body.model_dump(), "store": body.store.model_dump()}
    return _queue(
        jobs, response, task="capacity", params=params, subject=None, use_ml=body.use_ml,
        label=f"Capacity curve at {len(body.navs)} NAVs", size_key={"n": len(body.navs), "max_days": body.max_days},
    )  # fmt: skip


# ============================================================================= risk calibration
@router.get(
    "/risk-calibrations", response_model=S.RiskCalibrationList, tags=["risk"], summary="Risk calibration reports"
)
def list_risk(settings: Cfg) -> dict[str, Any]:
    return {"results": R.risk_summaries(settings)}


@router.get(
    "/risk-calibrations/{result_id}", response_model=S.RiskCalibrationResult, responses=problems(404), tags=["risk"]
)
def get_risk(result_id: str, settings: Cfg) -> dict[str, Any]:
    try:
        return R.result_file(settings.runs_root, R.RISK_DIR, result_id)
    except R.NotFound as e:
        raise _missing("risk calibration report", result_id) from e


@router.post(
    "/risk-calibrations",
    response_model=S.TaskLaunchResponse,
    status_code=202,
    responses=problems(422, 429),
    tags=["risk"],
    summary="Realised / ex-ante vol by year for a grid of nu (a background job; a report, not trials)",
)
def launch_risk(body: S.RiskCalibrationRequest, settings: Cfg, jobs: Jobs, response: Response) -> dict[str, Any]:
    _need_store(settings)
    _coverage(settings, body.store.model_dump(), body.baseline)
    params = {**body.model_dump(), "store": body.store.model_dump()}
    return _queue(
        jobs, response, task="risk_calibration", params=params, subject=None, use_ml=body.use_ml,
        label=f"Risk calibration over nu = {', '.join(f'{x:g}' for x in body.nu)}",
        size_key={"n": len(body.nu), "max_days": body.max_days},
    )  # fmt: skip


# ============================================================================= mid-cap check
def _midcap_status(settings: Settings, jobs: JobManager) -> dict[str, Any]:
    st = R.midcap_static(settings)
    sha = st["preregistration_sha256"]
    results = []
    for rid, src, d in R.midcap_results(settings):
        v = d.get("verdict") or {}
        results.append(
            {
                "run_id": rid,
                "outcome": v.get("outcome"),
                "forced": bool(d.get("forced")),
                "family": d.get("family"),
                "created_at": rid[:20] if rid[:8].isdigit() else None,
                "preregistration_sha256": str(d.get("preregistration_sha256") or ""),
                "sha_matches": bool(sha) and d.get("preregistration_sha256") == sha,
                "source": src,
            }
        )
    views = _views(jobs, "midcap_check", None, 10)
    running = [j for j in views if j["status"] in ACTIVE]
    ran = bool(st["previous_runs"])
    blocked = None
    if not sha:
        blocked = f"The pre-registration {st['preregistration']} is missing: the check must not run without it."
    elif settings.store_path is None:
        blocked = "The check reads the data store: set QTS_STORE_PATH on the API server."
    elif running:
        blocked = "A mid-cap check job is already queued or running."
    elif ran:
        blocked = (
            "The check of this configuration already ran: it is once-only, so a second run is no longer "
            "out-of-sample. Forcing it is possible but flags the record."
        )
    last_dry = None
    for j in jobs.tasks("midcap_check"):
        if j.get("status") == "done" and (j.get("request") or {}).get("dry_run") and j.get("result_payload"):
            last_dry = j["result_payload"]
            break
    return {
        **st,
        "results": results,
        "can_launch": blocked is None,
        "blocked_reason": blocked,
        "needs_force": ran,
        "force_phrase": R.MIDCAP_FORCE_PHRASE,
        "store_configured": settings.store_path is not None,
        "last_dry_run": last_dry,
        "jobs": views,
    }


@router.get(
    "/checks/midcap",
    response_model=S.MidcapStatus,
    tags=["midcap"],
    summary="The pre-registered criteria, results found on disk, and whether the check may run",
)
def get_midcap(settings: Cfg, jobs: Jobs) -> dict[str, Any]:
    return _midcap_status(settings, jobs)


@router.get(
    "/checks/midcap/preregistration",
    response_model=S.MidcapPreregistration,
    tags=["midcap"],
    summary="The pre-registration note and its SHA-256",
)
def get_preregistration(settings: Cfg) -> dict[str, Any]:
    return R.preregistration(settings)


@router.get(
    "/checks/midcap/results/{run_id}",
    response_model=S.MidcapResultDetail,
    responses=problems(404),
    tags=["midcap"],
    summary="One mid-cap check result with the mid-cap run and the large-cap reference",
)
def get_midcap_result(run_id: str, settings: Cfg, registry: Registry) -> dict[str, Any]:
    from pathlib import Path

    from qts.research.midcap import REFERENCE_RUN_ID, preregistration_sha256

    try:
        d = R.midcap_result(settings, run_id)
    except R.NotFound as e:
        raise _missing("mid-cap check result", run_id) from e

    def summary(rid: str | None) -> dict[str, Any] | None:
        if not rid:
            return None
        try:
            return registry.run_summary(registry.get(rid))
        except (RunNotFound, RunUnreadable):
            return None

    ref_dir = (d.get("verdict") or {}).get("reference_run")
    sha = preregistration_sha256()
    return {
        "result": d,
        "run": summary(run_id),
        "reference": summary(Path(ref_dir).name if ref_dir else REFERENCE_RUN_ID),
        "sha_matches": bool(sha) and d.get("preregistration_sha256") == sha,
    }


@router.post(
    "/checks/midcap",
    response_model=S.TaskLaunchResponse,
    status_code=202,
    responses=problems(409, 422, 429),
    tags=["midcap"],
    summary="Run the mid-cap check (or its coverage dry run) as a background job",
)
def launch_midcap(body: S.MidcapRequest, settings: Cfg, jobs: Jobs, response: Response) -> dict[str, Any]:
    """Once per configuration: a second run is refused (409) unless ``force`` with
    ``confirm`` equal to the status's ``force_phrase``; the repeat is flagged on its
    ledger record.  A dry run (coverage and plan only) is always allowed."""
    from qts.research.midcap import MIDCAP_DEFAULT_MAX_RANK

    _need_store(settings)
    st = _midcap_status(settings, jobs)
    if not st["preregistration_sha256"]:
        raise Problem(409, st["blocked_reason"] or "pre-registration missing", type_="midcap-refused")
    if any(j["status"] in ACTIVE for j in st["jobs"]):
        raise Problem(409, "A mid-cap check job is already queued or running.", type_="midcap-busy")
    if not body.dry_run:
        if st["needs_force"] and not body.force:
            raise Problem(409, st["blocked_reason"] or "already ran", type_="midcap-already-ran")
        if body.force and (body.confirm or "").strip() != R.MIDCAP_FORCE_PHRASE:
            raise _invalid(["body", "confirm"], f'to force a repeat, type "{R.MIDCAP_FORCE_PHRASE}"')
    store = body.store.model_dump()
    if not any(store.get(k) is not None for k in ("max_names", "min_mcap_ever", "max_rank")):
        store["max_rank"] = MIDCAP_DEFAULT_MAX_RANK
    params = {
        "store": store,
        "dry_run": body.dry_run,
        "force": bool(body.force and not body.dry_run),
        "cache": body.cache,
        "nav": body.nav,
        "config": "configs/baseline.yaml",
    }
    return _queue(
        jobs, response, task="midcap_check", params=params, subject="midcap",
        label="Mid-cap check: coverage dry run" if body.dry_run else ("Mid-cap check (forced repeat)" if body.force else "Mid-cap check"),
    )  # fmt: skip
