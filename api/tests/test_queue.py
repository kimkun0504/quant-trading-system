"""The run queue (persisted FIFO, statuses, reorder, restart recovery, log tail, ETA),
``POST /api/config/validate`` and the run's recorded data source: no worker processes
(the dispatcher is driven by hand with fake processes); the real lifecycle is in
``test_jobs.py``."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from qts_api import schemas as S
from qts_api.main import create_app
from qts_api.services.config_schema import leaf_index
from qts_api.services.jobs import JobManager, cli_command, estimate_seconds
from qts_api.services.registry import command_source
from qts_api.settings import Settings

from .conftest import OLD_RUN_ID

LOCAL = "http://localhost"

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


class FakeProc:
    """A worker that runs until told otherwise."""

    exitcode = 0

    def __init__(self) -> None:
        self.alive = True

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float | None = None) -> None:
        pass

    def terminate(self) -> None:
        self.alive = False


def manager(tmp_path, **kw) -> tuple[JobManager, dict[str, FakeProc]]:
    jm = JobManager(Settings(runs_root=tmp_path, **kw), start=False)
    procs: dict[str, FakeProc] = {}

    def start(job_id, payload):
        procs[job_id] = FakeProc()
        return procs[job_id]

    jm._start_process = start  # type: ignore[method-assign]
    return jm, procs


def submit(jm: JobManager, n: int, **req) -> dict:
    return jm.submit(
        request={**BASE, **req},
        config={},
        config_hash=f"hash{n:04d}",
        run_id=f"20990101T00000000000{n}Z_hash{n:04d}",
        label=f"Variant {n}",
        git_sha="x",
    )


def tick(jm: JobManager) -> None:
    with jm._cv:
        jm._reap()
        jm._dispatch()


# ----------------------------------------------------------------------------- queue lifecycle
def test_fifo_one_at_a_time_and_statuses(tmp_path):
    jm, procs = manager(tmp_path)
    a, b, c = (submit(jm, i)["job_id"] for i in (1, 2, 3))
    assert json.loads((tmp_path / "_jobs" / "queue.json").read_text())["order"] == [a, b, c]
    tick(jm)
    assert list(procs) == [a] and jm.counts() == {"queued": 2, "running": 1}  # one run at a time
    listing = S.JobQueue.model_validate(jm.listing())
    assert [(j.job_id, j.status, j.position) for j in listing.jobs] == [
        (a, "running", None),
        (b, "queued", 0),
        (c, "queued", 1),
    ]
    assert listing.max_workers == 1 and listing.counts == {"running": 1, "queued": 2}
    assert jm.get(c)["message"] == "Queued (1 job ahead)"

    # the worker finishes a: b starts, c moves up
    jm._mark(a, status="done", progress=1.0, message="Finished", finished_at="2099-01-01T00:00:10Z")
    procs[a].alive = False
    tick(jm)
    assert list(procs) == [a, b]
    assert [(j.job_id, j.status) for j in S.JobQueue.model_validate(jm.listing()).jobs] == [
        (b, "running"),
        (c, "queued"),
        (a, "done"),
    ]
    # a worker that dies without reporting is a failure, and the queue moves on
    procs[b].alive = False
    tick(jm)
    assert jm.get(b)["status"] == "failed" and "exited with code" in jm.get(b)["message"]
    assert jm.get(c)["status"] == "running"
    # cancelling the running job terminates it
    assert jm.cancel(c)["status"] == "cancelled" and not procs[c].alive
    assert jm.counts() == {"queued": 0, "running": 0}
    assert json.loads((tmp_path / "_jobs" / "queue.json").read_text())["order"] == []


def test_cancel_and_reorder_queued(tmp_path):
    jm, procs = manager(tmp_path)
    a, b, c, d = (submit(jm, i)["job_id"] for i in (1, 2, 3, 4))
    assert jm.move(d, 0)["position"] == 0
    assert jm.move(a, 99)["position"] == 3  # clamped to the end
    assert list(jm._queue) == [d, b, c, a]
    out = jm.cancel(b)
    assert out["status"] == "cancelled" and out["message"] == "Cancelled before it started"
    assert jm.cancel(b)["status"] == "cancelled"  # idempotent
    assert json.loads((tmp_path / "_jobs" / "queue.json").read_text())["order"] == [d, c, a]
    tick(jm)
    assert list(procs) == [d]  # the moved job runs first; the cancelled one never does
    tick(jm)
    assert list(procs) == [d]


def test_queue_routes(tmp_path):
    with TestClient(create_app(Settings(runs_root=tmp_path), start_jobs=False), base_url=LOCAL) as c:
        ids = [c.post("/api/runs", json={**BASE, "seed": i, "label": f"V{i}"}).json()["job_id"] for i in range(3)]
        q = S.JobQueue.model_validate(c.get("/api/jobs").json())
        assert [j.job_id for j in q.jobs] == ids and [j.position for j in q.jobs] == [0, 1, 2]
        assert [j.label for j in q.jobs] == ["V0", "V1", "V2"] and q.max_queued == 20
        j = q.jobs[0]
        assert (j.kind, j.ml, j.phase, j.base_run_id, j.n_overrides) == ("synthetic", False, 1, None, 0)
        assert j.started_at is None and j.eta_seconds is None and j.last_log is None and j.stage == ""

        moved = S.Job.model_validate(c.post(f"/api/jobs/{ids[2]}/move", json={"position": 0}).json())
        assert moved.position == 0
        assert [j.job_id for j in S.JobQueue.model_validate(c.get("/api/jobs").json()).jobs] == [
            ids[2],
            ids[0],
            ids[1],
        ]
        assert c.post(f"/api/jobs/{ids[2]}/move", json={"position": -1}).status_code == 422
        assert c.post("/api/jobs/job_nope/move", json={"position": 0}).status_code == 404

        gone = S.Job.model_validate(c.delete(f"/api/jobs/{ids[0]}").json())
        assert gone.status == "cancelled" and gone.position is None and gone.finished_at
        r = c.post(f"/api/jobs/{ids[0]}/move", json={"position": 0})
        assert r.status_code == 409 and r.json()["type"].endswith("job-not-queued")
        q = S.JobQueue.model_validate(c.get("/api/jobs").json())
        assert [(j.job_id, j.status) for j in q.jobs] == [(ids[2], "queued"), (ids[1], "queued"), (ids[0], "cancelled")]
        assert S.JobQueue.model_validate(c.get("/api/jobs?recent=0").json()).jobs[-1].status == "queued"
        assert c.get("/api/health").json()["jobs"] == {"queued": 2, "running": 0}


# ----------------------------------------------------------------------------- restart
def test_restart_requeues_queued_and_fails_running(tmp_path):
    jm, _ = manager(tmp_path)
    a, b, c = (submit(jm, i)["job_id"] for i in (1, 2, 3))
    jm.move(c, 0)  # queue: c, a, b
    tick(jm)  # c is running
    assert jm.get(c)["status"] == "running"
    (tmp_path / "_staging").mkdir()
    # the API process dies here (no shutdown); a new one starts on the same runs root
    jm2, procs2 = manager(tmp_path)
    assert jm2.get(c)["status"] == "failed"
    assert "the API restarted while this job was running" in jm2.get(c)["message"]
    assert list(jm2._queue) == [a, b] and jm2.counts() == {"queued": 2, "running": 0}
    assert not (tmp_path / "_staging").exists()
    assert [jm2.get(x)["position"] for x in (a, b)] == [0, 1]
    tick(jm2)
    assert list(procs2) == [a]  # the re-queued job runs, with its persisted payload
    assert jm2._payloads[a]["run_id"] == jm.get(a)["run_id"]


def test_shutdown_keeps_the_queue(tmp_path):
    jm, _ = manager(tmp_path)
    a, b = (submit(jm, i)["job_id"] for i in (1, 2))
    tick(jm)
    jm.shutdown(timeout=0.1)
    assert jm.get(a)["status"] == "failed" and "shut down" in jm.get(a)["message"]
    jm2, _ = manager(tmp_path)
    assert list(jm2._queue) == [b] and jm2.get(b)["status"] == "queued"


def test_restart_without_payload_fails_the_job(tmp_path):
    """A queued record written before payloads were persisted cannot be run."""
    jm, _ = manager(tmp_path)
    a = submit(jm, 1)["job_id"]
    rec = jm.get(a)
    rec.pop("payload")
    rec.pop("position", None)
    (tmp_path / "_jobs" / f"{a}.json").write_text(json.dumps(rec))
    jm2, _ = manager(tmp_path)
    assert jm2.get(a)["status"] == "failed" and list(jm2._queue) == []


# ----------------------------------------------------------------------------- log tail
def test_log_tail(tmp_path):
    with TestClient(create_app(Settings(runs_root=tmp_path), start_jobs=False), base_url=LOCAL) as c:
        job_id = c.post("/api/runs", json=BASE).json()["job_id"]
        empty = S.JobLog.model_validate(c.get(f"/api/jobs/{job_id}/log").json())
        assert empty.lines == [] and empty.total_lines == 0 and not empty.truncated
        (tmp_path / "_jobs" / f"{job_id}.log").write_text("".join(f"line {i}\n" for i in range(1, 501)))
        lg = S.JobLog.model_validate(c.get(f"/api/jobs/{job_id}/log?tail=3").json())
        assert lg.lines == ["line 498", "line 499", "line 500"] and lg.total_lines == 500 and lg.truncated
        assert len(c.get(f"/api/jobs/{job_id}/log").json()["lines"]) == 200  # default tail
        assert S.Job.model_validate(c.get(f"/api/jobs/{job_id}").json()).last_log == "line 500"
        assert c.get(f"/api/jobs/{job_id}/log?tail=0").status_code == 422
        r = c.get("/api/jobs/job_nope/log")
        assert r.status_code == 404 and r.headers["content-type"].startswith("application/problem+json")


# ----------------------------------------------------------------------------- ETA
def _done(n: int, seconds: float, **req) -> dict:
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return {
        "job_id": f"job_{n:012x}",
        "status": "done",
        "started_at": t0.isoformat(),
        "finished_at": (t0 + timedelta(seconds=seconds)).isoformat(),
        "request": {**BASE, **req},
    }


def test_eta_from_similar_finished_jobs():
    job = {"job_id": "job_new", "request": {**BASE, "stocks": 100, "days": 1000}}
    assert estimate_seconds(job, []) is None
    history = [
        _done(1, 100.0, stocks=100, days=1000),  # same size
        _done(2, 60.0, stocks=50, days=1000),  # half the size: scaled to 120
        _done(3, 5.0, stocks=10, days=1000),  # 10x smaller: not similar
        _done(4, 999.0, stocks=100, days=1000, ml=True),  # another model
        _done(5, 999.0, kind="store"),  # another data kind
        {**_done(6, 1.0, stocks=100, days=1000), "status": "failed"},
    ]
    assert estimate_seconds(job, history) == pytest.approx(110.0)  # median of 100 and 120
    # a store slice is only comparable with the same slice
    store = {"job_id": "job_s", "request": {**BASE, "kind": "store", "start": "2010-01-01", "max_names": 500}}
    hist = [_done(7, 300.0, kind="store", start="2010-01-01", max_names=500), _done(8, 900.0, kind="store")]
    assert estimate_seconds(store, hist) == 300.0


def test_running_job_reports_eta(tmp_path):
    jm, _ = manager(tmp_path)
    a = submit(jm, 1)["job_id"]
    tick(jm)
    now = datetime.now(timezone.utc)
    # no history: extrapolate from the backtest's own day counter (40 s in, 25% through)
    jm._mark(a, stage="backtest", backtest_started_at=(now - timedelta(seconds=40)).isoformat(), backtest_frac=0.25)
    v = S.Job.model_validate(jm.view(jm.get(a)))
    assert v.stage == "backtest" and v.estimated_seconds is None and v.eta_seconds == pytest.approx(120, abs=5)
    # with history: the similar job's duration minus the time already spent
    jm._mark(a, started_at=(now - timedelta(seconds=50)).isoformat())
    hist = [*jm.all(), _done(9, 200.0)]
    v = S.Job.model_validate(jm.view(jm.get(a), hist))
    assert v.estimated_seconds == 200.0 and v.eta_seconds == pytest.approx(150, abs=5)


# ----------------------------------------------------------------------------- validate
def validate(client, **kw) -> S.ConfigValidation:
    r = client.post("/api/config/validate", json={**BASE, **kw})
    assert r.status_code == 200, r.text
    return S.ConfigValidation.model_validate(r.json())


def test_validate_ok_matches_the_launch(client):
    over = {"portfolio.trade_aversion": 2, "alpha.horizon_days": 10}
    v = validate(client, config_overrides=over)
    assert v.valid and v.errors == [] and v.warnings == [] and v.overrides_applied == 2
    assert (v.warmup_days, v.min_days) == (506, 758)
    assert v.config_hash and v.base_config_hash and v.config_hash != v.base_config_hash
    assert client.app.state.jobs.counts() == {"queued": 0, "running": 0}  # nothing was queued
    run_id = client.post("/api/runs", json={**BASE, "config_overrides": over}).json()["run_id"]
    assert run_id.endswith("_" + v.config_hash[:8])
    # the warm-up follows the config: a shorter risk history lowers it
    assert validate(client, config_overrides={"risk.min_history_days": 126}).warmup_days == 254
    assert validate(client, ml=True, days=1081).warmup_days == 829


def test_validate_reports_errors_per_key(client):
    v = validate(
        client,
        config_overrides={"portfolio.trade_aversion": 50, "nosection.x": 1, "portfolio.rebalance_every_days": 0},
    )
    assert not v.valid and v.config_hash is None and v.base_config_hash
    got = {tuple(e.loc): e.msg for e in v.errors}
    assert "at most 10" in got[("body", "config_overrides", "portfolio.trade_aversion")]
    assert "unknown config section" in got[("body", "config_overrides", "nosection.x")]
    assert "at least 1" in got[("body", "config_overrides", "portfolio.rebalance_every_days")]
    # request-level problems are errors too, with the launch's own messages
    v = validate(client, days=600)
    assert [tuple(e.loc) for e in v.errors] == [("body", "days")] and "at least 758" in v.errors[0].msg
    assert v.min_days == 758
    assert [tuple(e.loc) for e in validate(client, phase=2).errors] == [("body", "ml")]
    assert [tuple(e.loc) for e in validate(client, kind="store").errors] == [("body", "kind")]
    # a section's own cross-field validation names the key that broke it
    v = validate(client, config_overrides={"cost.borrow_proxy_si_gc": 0.9})
    assert not v.valid and v.errors[0].loc[-1] == "cost.borrow_proxy_si_gc"
    # a malformed body is a 422, like the launch
    assert client.post("/api/config/validate", json={**BASE, "stocks": "60"}).status_code == 422
    assert client.post("/api/config/validate", json={**BASE, "start": "2010"}).status_code == 422


def test_validate_warnings(client):
    v = validate(client, signal_lag_days=0)
    assert v.valid and [w.type for w in v.warnings] == ["warning.lag0"]
    v = validate(client, config_overrides={"alpha.lgbm.num_leaves": 31})
    assert (
        v.valid and [w.type for w in v.warnings] == ["warning.ml_off"] and "alpha.lgbm.num_leaves" in v.warnings[0].msg
    )
    assert validate(client, ml=True, days=1100, config_overrides={"alpha.lgbm.num_leaves": 31}).warnings == []


def test_validate_null_override_only_for_nullable_fields(client):
    assert leaf_index()["portfolio.target_vol_annual"]["nullable"] is True
    v = validate(client, config_overrides={"portfolio.target_vol_annual": None})
    assert v.valid and v.config_hash != validate(client).config_hash
    v = validate(client, config_overrides={"portfolio.trade_aversion": None})
    assert not v.valid and "cannot be empty" in v.errors[0].msg


def test_validate_from_a_base_run(client, run_id):
    d = S.RunDetail.model_validate(client.get(f"/api/runs/{run_id}").json())
    req = {
        "stocks": d.config.source.stocks or 60,
        "days": d.config.source.days or 764,
        "seed": d.config.seed,
        "signal_lag_days": d.config.signal_lag_days,
        "base_run_id": run_id,
    }
    same = validate(client, **req)
    assert same.valid and same.config_hash == same.base_config_hash == d.manifest.config_hash
    assert [w.type for w in same.warnings] == ["warning.same_config"]
    edited = validate(client, **req, config_overrides={"portfolio.trade_aversion": 3})
    assert edited.valid and edited.config_hash != d.manifest.config_hash and edited.warnings == []
    # the override applies on top of the run's config, not of the defaults
    default_based = validate(client, **{**req, "base_run_id": None}, config_overrides={"portfolio.trade_aversion": 3})
    assert default_based.base_config_hash != edited.base_config_hash
    missing = validate(client, base_run_id="20000101T000000000000Z_nothere0")
    assert not missing.valid and tuple(missing.errors[0].loc) == ("body", "base_run_id")
    r = client.post("/api/runs", json={**BASE, "base_run_id": "20000101T000000000000Z_nothere0"})
    assert r.status_code == 422 and r.json()["errors"][0]["loc"] == ["body", "base_run_id"]


def test_store_request_validation(tmp_path):
    s = Settings(runs_root=tmp_path, store_path=tmp_path / "store")
    with TestClient(create_app(s, start_jobs=False), base_url=LOCAL) as c:
        assert c.get("/api/health").json()["store_path"] == str(tmp_path / "store")
        req = {**BASE, "kind": "store", "start": "2010-01-01", "end": "2017-12-31", "max_names": 1500}
        req = {**req, "min_mcap_ever": 2_000_000_000, "cache": True, "ml": True, "phase": 2}
        v = S.ConfigValidation.model_validate(c.post("/api/config/validate", json=req).json())
        assert v.valid, v.errors
        bad = S.ConfigValidation.model_validate(
            c.post("/api/config/validate", json={**req, "start": "2018-01-01"}).json()
        )
        assert [tuple(e.loc) for e in bad.errors] == [("body", "end")]
        out = c.post("/api/runs", json=req).json()
        stored = json.loads((tmp_path / "_jobs" / f"{out['job_id']}.json").read_text())
        assert stored["payload"]["store_path"] == str(tmp_path / "store")
        assert stored["request"]["max_names"] == 1500 and stored["request"]["cache"] is True
        assert cli_command(stored["request"], stored["payload"]["store_path"]) == (
            f"qts run --store {tmp_path / 'store'} --start 2010-01-01 --end 2017-12-31 --max-names 1500 "
            "--min-mcap-ever 2e+09 --ml --cache --set seed=3"
        )
    assert Settings(runs_root=tmp_path).store_path is None


# ----------------------------------------------------------------------------- run source
def test_cli_command_text():
    req = {**BASE, "config_overrides": {"portfolio.target_vol_annual": None, "portfolio.beta_neutral": False}}
    assert cli_command({**req, "signal_lag_days": 0, "base_run_id": "r1"}, None) == (
        "qts demo --stocks 50 --days 764 --seed 3 --config runs/r1/config.yaml --signal-lag 0 "
        "--set portfolio.target_vol_annual=null --set portfolio.beta_neutral=false"
    )


def test_command_source_parsing():
    src = command_source(
        "/usr/bin/qts run --store data/store --start 2001-01-01 --end=2025-06-30 --max-names 1500 "
        "--min-mcap-ever 2e9 --ml --cache --set alpha.horizon_days=10"
    )
    assert src == {
        "kind": "store",
        "stocks": None,
        "days": None,
        "store_path": "data/store",
        "start": "2001-01-01",
        "end": "2025-06-30",
        "max_names": 1500,
        "min_mcap_ever": 2e9,
        "max_days": None,
        "cache": True,
        "venue": None,
    }
    demo = command_source("qts/cli.py demo --days 900 --max-days 120")
    assert (demo["kind"], demo["stocks"], demo["days"], demo["max_days"]) == ("synthetic", 300, 900, 120)
    assert command_source("pytest -q") is None and command_source("qts run --yfinance AAPL") is None
    assert command_source('unbalanced "quote') is None


def test_run_source_in_detail(client, run_id):
    old = S.RunDetail.model_validate(client.get(f"/api/runs/{OLD_RUN_ID}").json()).config.source
    assert old.recorded and old.origin == "cli" and (old.kind, old.stocks, old.days) == ("synthetic", 60, 764)
    assert old.command == "qts/cli.py demo --stocks 60 --days 764 --seed 5"
    new = S.RunDetail.model_validate(client.get(f"/api/runs/{run_id}").json()).config.source
    # the fixture run was built in-process (no CLI command, no API job): not recorded
    assert not new.recorded and new.origin == "unknown" and new.kind == "synthetic" and new.stocks == 60
    # the family the run's DSR counted in, as the manifest records it (null on older manifests)
    fam = S.RunDetail.model_validate(client.get(f"/api/runs/{run_id}").json()).manifest.ledger_family
    assert fam in {f["family"] for f in client.get("/api/ledger").json()["families"]}
    assert "ledger_family" in S.RunManifest.model_fields


# ----------------------------------------------------------------------------- schema
def test_schema_is_generated_from_the_live_config():
    from dataclasses import fields

    from qts.core.config import AlphaConfig

    ix = leaf_index()
    scalar = {f.name for f in fields(AlphaConfig) if f.type in ("int", "float", "bool", "str", "float | None")}
    assert scalar and all(f"alpha.{name}" in ix for name in scalar)  # a new scalar field appears by itself
    for key in ("alpha.horizon_days", "universe.top_n_by_mcap", "cache.enabled", "portfolio.rebalance_every_days"):
        assert ix[key].get("description"), key
    assert ix["portfolio.rebalance_every_days"]["minimum"] == 1
    assert ix["alpha.score_smoothing_halflife_days"]["minimum"] == 0
    assert "nullable" not in ix["portfolio.trade_aversion"]
    assert "default" not in ix["alpha.embargo_days"] and ix["alpha.embargo_days"]["nullable"] is True
