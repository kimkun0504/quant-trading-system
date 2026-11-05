"""Local-server hardening (second-pass audit, findings 3, 4, 10-13): Host allow-list,
queued-job cap, dispatcher robustness, run-id validation, duplicate run directories,
strict launch types and semantic config bounds."""

from __future__ import annotations

import json
import shutil
import threading
import time

import pytest
from fastapi.testclient import TestClient

from qts_api import schemas as S
from qts_api.main import create_app
from qts_api.services.config_schema import leaf_index
from qts_api.services.jobs import JobManager
from qts_api.settings import DEFAULT_ALLOWED_HOSTS, Settings

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


def _problem(r, status: int) -> S.ApiErrorBody:
    assert r.status_code == status, r.text
    assert r.headers["content-type"].startswith("application/problem+json")
    return S.ApiErrorBody.model_validate(r.json())


# ----------------------------------------------------------------- #4 Host allow-list
def test_foreign_host_is_rejected(fixture_root):
    with TestClient(create_app(Settings(runs_root=fixture_root), start_jobs=False), base_url=LOCAL) as c:
        assert c.get("/api/runs").status_code == 200
        for host in ("attacker.example", "localhost.attacker.example", "127.0.0.1.nip.io"):
            r = c.get("/api/runs", headers={"Host": host})
            assert r.status_code == 400, host
        r = c.post("/api/runs", json=BASE, headers={"Host": "attacker.example:8000"})
        assert r.status_code == 400
        for host in ("localhost:8000", "127.0.0.1", "127.0.0.1:5173", "[::1]:8000"):
            assert c.get("/api/health", headers={"Host": host}).status_code == 200, host


def test_allowed_hosts_from_env():
    s = Settings.from_env({"QTS_ALLOWED_HOSTS": "qts.lan, *.corp.example ,localhost"})
    assert s.allowed_hosts == (*DEFAULT_ALLOWED_HOSTS, "qts.lan", "*.corp.example")
    assert Settings.from_env({}).allowed_hosts == DEFAULT_ALLOWED_HOSTS
    assert Settings.from_env({}).max_queued == 4 and Settings.from_env({"QTS_MAX_QUEUED": "1"}).max_queued == 1


def test_extra_allowed_host(tmp_path):
    s = Settings(runs_root=tmp_path, allowed_hosts=(*DEFAULT_ALLOWED_HOSTS, "qts.lan"))
    with TestClient(create_app(s, start_jobs=False), base_url="http://qts.lan") as c:
        assert c.get("/api/runs").status_code == 200
        assert c.get("/api/runs", headers={"Host": "other.lan"}).status_code == 400


# ----------------------------------------------------------------- queue cap
def test_queued_jobs_are_capped(tmp_path):
    with TestClient(create_app(Settings(runs_root=tmp_path, max_queued=2), start_jobs=False), base_url=LOCAL) as c:
        assert c.post("/api/runs", json=BASE).status_code == 202
        assert c.post("/api/runs", json=BASE).status_code == 202
        r = c.post("/api/runs", json=BASE)
        body = _problem(r, 429)
        assert "QTS_MAX_QUEUED" in body.detail and body.type.endswith("queue-full")
        assert r.headers["retry-after"] == "60"
        assert c.app.state.jobs.counts() == {"queued": 2, "running": 0}
        assert len(list((tmp_path / "_jobs").glob("*.json"))) == 2  # the refused job left no file


# ----------------------------------------------------------------- #10 dispatcher
def test_dispatcher_survives_process_start_failure(tmp_path):
    jm = JobManager(Settings(runs_root=tmp_path, max_workers=1), start=False)
    calls: list[str] = []
    started = threading.Event()

    class FakeProc:
        exitcode = 0

        def is_alive(self):
            return True

        def terminate(self):
            pass

        def join(self, timeout=None):
            pass

    def start(job_id, payload):
        calls.append(job_id)
        if len(calls) == 1:
            raise OSError(11, "Resource temporarily unavailable")  # EAGAIN
        started.set()
        return FakeProc()

    jm._start_process = start  # type: ignore[method-assign]
    kw = {"request": {"kind": "synthetic", "phase": 1, "ml": False}, "config": {}, "config_hash": "h", "git_sha": "x"}
    a = jm.submit(run_id="20260101T000000000000Z_aaaaaaaa", label="a", **kw)
    b = jm.submit(run_id="20260101T000000000000Z_bbbbbbbb", label="b", **kw)
    jm.start()
    try:
        assert started.wait(10), "dispatcher did not reach the second job"
        ja = jm.get(a["job_id"])
        assert ja["status"] == "failed" and "could not start" in ja["message"] and "Errno 11" in ja["message"]
        assert jm.counts() == {"queued": 0, "running": 1}
        assert jm._thread is not None and jm._thread.is_alive()
        assert calls == [a["job_id"], b["job_id"]]
    finally:
        jm.shutdown(timeout=1)


# ----------------------------------------------------------------- #11 run ids
@pytest.mark.parametrize(
    "rid",
    ["a" * 300, "x" * 129, "..", "_jobs", ".hidden", "a..b", "run%00id", "run id", "trial_ledger.jsonl.."],
)
def test_malformed_run_ids_are_404(client, rid):
    r = client.get(f"/api/runs/{rid}")
    body = _problem(r, 404)
    if rid != "..":  # the client normalises /api/runs/.. to /api/ (endpoint-not-found)
        assert body.type.endswith("run-not-found")
    for sub in ("series", "costs", "holdings", "ic", "exposures"):
        assert client.get(f"/api/runs/{rid}/{sub}").status_code == 404


def test_malformed_job_ids_are_404(client):
    for jid in ("job_" + "a" * 300, "../x", "job_ABCDEF123456"):
        assert client.get(f"/api/jobs/{jid}").status_code == 404


# ----------------------------------------------------------------- #12 duplicate ids
def test_copied_run_directory_is_listed_once_under_its_directory_name(fixture_root, run_id, tmp_path, caplog):
    root = tmp_path / "runs"
    root.mkdir()
    shutil.copytree(fixture_root / run_id, root / run_id)
    copy = run_id[:-8] + "c0pyc0py"
    shutil.copytree(fixture_root / run_id, root / copy)  # manifest still says run_id
    with TestClient(create_app(Settings(runs_root=root), start_jobs=False), base_url=LOCAL) as c:
        ids = [r["id"] for r in c.get("/api/runs").json()]
        assert sorted(ids) == sorted([run_id, copy]) and len(set(ids)) == 2
        d = c.get(f"/api/runs/{copy}").json()
        assert d["id"] == copy
        assert c.get(f"/api/runs/{copy}/series").status_code == 200
    assert any("listing it under the directory name" in r.getMessage() for r in caplog.records)


# ----------------------------------------------------------------- #13 strict request types
@pytest.mark.parametrize(
    "kw, loc",
    [
        ({"stocks": "60"}, ("body", "stocks")),
        ({"ml": "true"}, ("body", "ml")),
        ({"seed": "3"}, ("body", "seed")),
        ({"phase": "1"}, ("body", "phase")),
        ({"stocks": 60.0}, ("body", "stocks")),
        ({"label": "bad\x00label"}, ("body", "label")),
        ({"label": "tab\there"}, ("body", "label")),
    ],
)
def test_launch_request_is_strict(client, kw, loc):
    body = _problem(client.post("/api/runs", json={**BASE, **kw}), 422)
    assert loc in {tuple(e.loc) for e in body.errors}


# ----------------------------------------------------------------- #3 semantic bounds
@pytest.mark.parametrize(
    "key, value, needle",
    [
        ("cost.clearing_per_share", -0.0002, "at least 0"),
        ("cost.cat_fee_per_share", -1e-6, "at least 0"),
        ("cost.finra_taf_per_share_sold", -0.1, "at least 0"),
        ("cost.auction_exchange_fee_per_share", -0.001, "at least 0"),
        ("cost.auction_volume_share_default", 0, "greater than 0"),
        ("cost.impact_b_continuous", 1e12, "at most 10"),
        ("cost.impact_b_auction", 0, "greater than 0"),
        ("cost.impact_exponent", 1.0, "greater than 1"),
        ("execution.algo_max_pct_adv", 0, "greater than 0"),
        ("alpha.lgbm.n_estimators", 1_000_000_000, "at most 10000"),
        ("alpha.mlp.max_iter", 1_000_000, "at most 10000"),
    ],
)
def test_config_overrides_semantic_bounds(client, key, value, needle):
    body = _problem(client.post("/api/runs", json={**BASE, "config_overrides": {key: value}}), 422)
    msgs = {tuple(e.loc): e.msg for e in body.errors}
    assert needle in msgs[("body", "config_overrides", key)]


def test_cross_field_cost_rule_is_reported(client):
    r = client.post(
        "/api/runs",
        json={**BASE, "config_overrides": {"cost.borrow_proxy_si_gc": 0.5, "cost.borrow_proxy_si_htb": 0.2}},
    )
    body = _problem(r, 422)
    assert "borrow_proxy_si_gc must be < borrow_proxy_si_htb" in body.detail


def test_schema_publishes_the_config_bounds(client):
    body = client.get("/api/config/schema").json()
    cost = body["properties"]["cost"]["properties"]
    assert cost["clearing_per_share"]["minimum"] == 0 and "exclusiveMinimum" not in cost["clearing_per_share"]
    assert cost["impact_b_auction"]["exclusiveMinimum"] == 0 and cost["impact_b_auction"]["maximum"] == 10
    assert cost["impact_exponent"]["exclusiveMinimum"] == 1 and cost["impact_exponent"]["maximum"] == 2
    assert cost["auction_volume_share_default"]["exclusiveMinimum"] == 0
    ex = body["properties"]["execution"]["properties"]
    assert ex["signal_lag_days"]["minimum"] == 0 and ex["signal_lag_days"]["maximum"] == 10
    assert body["properties"]["alpha"]["properties"]["lgbm"]["properties"]["n_estimators"]["maximum"] == 10_000
    S.JsonSchemaObject.model_validate(body)
    # every cost / execution scalar carries a bound
    idx = leaf_index()
    for k, leaf in idx.items():
        if k.startswith(("cost.", "execution.")) and leaf["type"] in ("number", "integer"):
            assert {"minimum", "exclusiveMinimum"} & set(leaf), k
    json.dumps(body)


def test_defaults_still_launch(client):
    r = client.post("/api/runs", json=BASE)
    assert r.status_code == 202, r.text
    time.sleep(0)  # the dispatcher is off in this fixture: nothing runs
