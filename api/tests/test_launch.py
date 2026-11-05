"""POST /api/runs validation (422 problem bodies) and the job bookkeeping, without
starting worker processes (the dispatcher is off in the ``client`` fixture)."""

from __future__ import annotations

import json

import pytest

from qts_api import schemas as S

LOCAL = "http://localhost"  # Host allow-list (see conftest.LOCAL)

BASE = {
    "kind": "synthetic",
    "ml": False,
    "phase": 1,
    "stocks": 50,
    "days": 764,
    "seed": 3,
    "signal_lag_days": 1,
    "config_overrides": {},
}


def post(client, **kw):
    return client.post("/api/runs", json={**BASE, **kw})


def issues(r):
    assert r.status_code == 422, r.text
    assert r.headers["content-type"].startswith("application/problem+json")
    body = S.ApiErrorBody.model_validate(r.json())
    assert isinstance(body.detail, str) and body.detail
    return {tuple(e.loc): e.msg for e in body.errors}


@pytest.mark.parametrize(
    "overrides, loc, needle",
    [
        ({"portfolio.no_such_field": 1}, "portfolio.no_such_field", "unknown config field"),
        ({"nosection.x": 1}, "nosection.x", "unknown config section"),
        ({"risk.styles": "beta"}, "risk.styles", "not a scalar field"),
        ({"portfolio.trade_aversion": "high"}, "portfolio.trade_aversion", "must be a number"),
        ({"portfolio.trade_aversion": True}, "portfolio.trade_aversion", "must be a number"),
        ({"portfolio.trade_aversion": 50}, "portfolio.trade_aversion", "at most 10"),
        ({"validation.n_splits": 2.5}, "validation.n_splits", "whole number"),
        ({"portfolio.book": "short_only"}, "portfolio.book", "must be one of"),
        ({"portfolio.beta_neutral": 1}, "portfolio.beta_neutral", "true or false"),
        ({"seed": 4}, "seed", "use the request's `seed` field"),
    ],
)
def test_invalid_overrides(client, overrides, loc, needle):
    got = issues(post(client, config_overrides=overrides))
    msg = got[("body", "config_overrides", loc)]
    assert needle in msg


def test_several_invalid_overrides_reported_together(client):
    got = issues(post(client, config_overrides={"a.b": 1, "portfolio.risk_aversion": -1}))
    assert len(got) == 2


@pytest.mark.parametrize(
    "kw, loc",
    [
        ({"stocks": 10}, ("body", "stocks")),
        ({"days": 500}, ("body", "days")),
        ({"phase": 3}, ("body", "phase")),
        ({"signal_lag_days": 2}, ("body", "signal_lag_days")),
        ({"kind": "csv"}, ("body", "kind")),
        ({"unexpected": 1}, ("body", "unexpected")),
        ({"phase": 2, "ml": False}, ("body", "ml")),
        ({"kind": "store"}, ("body", "kind")),
    ],
)
def test_invalid_request(client, kw, loc):
    assert loc in issues(post(client, **kw))


def test_launch_queues_a_job(client, fixture_root):
    r = post(client, config_overrides={"portfolio.trade_aversion": 2, "validation.n_splits": 4.0}, seed=9)
    assert r.status_code == 202, r.text
    out = S.LaunchRunResponse.model_validate(r.json())
    assert r.headers["location"] == f"/api/jobs/{out.job_id}"
    assert out.run_id.endswith(out.run_id.split("_")[-1]) and len(out.run_id.split("_")[-1]) == 8
    job = S.Job.model_validate(client.get(f"/api/jobs/{out.job_id}").json())
    assert job.status == "queued" and job.run_id == out.run_id and job.progress == 0
    stored = json.loads((fixture_root / "_jobs" / f"{out.job_id}.json").read_text())
    assert stored["request"]["config_overrides"] == {"portfolio.trade_aversion": 2, "validation.n_splits": 4.0}
    runs = {x["id"]: x for x in client.get("/api/runs").json()}
    assert runs[out.run_id]["status"] == "running" and runs[out.run_id]["label"] == "Composite, seed 9"
    # cancel before it starts
    c = S.Job.model_validate(client.delete(f"/api/jobs/{out.job_id}").json())
    assert c.status == "failed" and "Cancelled" in c.message
    assert client.get(f"/api/runs/{out.run_id}").status_code == 409


def test_same_config_same_run_hash_suffix(client):
    a = post(client).json()["run_id"]
    b = post(client).json()["run_id"]
    c = post(client, config_overrides={"portfolio.risk_aversion": 20}).json()["run_id"]
    assert a.split("_")[-1] == b.split("_")[-1] != c.split("_")[-1]


def test_unknown_job_404(client):
    for method in ("get", "delete"):
        r = getattr(client, method)("/api/jobs/job_nope")
        assert r.status_code == 404 and r.json()["detail"] == "No job with id job_nope"


def test_jobs_survive_restart(fixture_root):
    """A queued job of a previous API process is reported as interrupted (failed)."""
    from fastapi.testclient import TestClient

    from qts_api.main import create_app
    from qts_api.settings import Settings

    s = Settings(runs_root=fixture_root)
    with TestClient(create_app(s, start_jobs=False), base_url=LOCAL) as c1:
        out = c1.post("/api/runs", json=BASE).json()
    with TestClient(create_app(s, start_jobs=False), base_url=LOCAL) as c2:
        j = c2.get(f"/api/jobs/{out['job_id']}").json()
        assert j["status"] == "failed" and "Interrupted" in j["message"]
    for f in (fixture_root / "_jobs").glob("*.json"):
        f.unlink()


# ----------------------------------------------------------------------------- warm-up (P7)
def test_warmup_endpoint_derives_from_config(client):
    w = S.Warmup.model_validate(client.get("/api/config/warmup").json())
    # risk: 252 exposure warm-up + 252 factor-return days + 1; + signal lag 1
    assert (w.warmup_days, w.min_days, w.risk_days, w.ml_days, w.exact) == (506, 758, 505, None, True)
    w0 = S.Warmup.model_validate(client.get("/api/config/warmup?signal_lag_days=0").json())
    assert w0.warmup_days == 505 and w0.signal_lag_days == 0
    wm = S.Warmup.model_validate(client.get("/api/config/warmup?ml=true").json())
    # ML: universe lead 51 + train_min_days 756 + horizon 20 + lag 1 = 828; + lag 1
    assert (wm.warmup_days, wm.min_days, wm.ml_days, wm.exact) == (829, 1081, 828, False)
    assert client.get("/api/config/warmup?signal_lag_days=3").status_code == 422


def test_days_floor_follows_the_runs_config(client):
    assert post(client, days=758).status_code == 202  # 506 + 252
    got = issues(post(client, days=757))
    assert "at least 758 (506 warm-up days" in got[("body", "days")]
    got = issues(post(client, ml=True, days=1080))
    assert "at least 1081 (829 warm-up days" in got[("body", "days")]
    assert post(client, ml=True, days=1081).status_code == 202
    # overrides move the floor: a shorter risk history lowers it
    assert post(client, days=600, config_overrides={"risk.min_history_days": 126}).status_code == 202
