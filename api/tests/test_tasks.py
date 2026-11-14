"""Task jobs: the worker entry point ``run_task`` against the research world (in-process,
so the library calls are the real ones without the spawn overhead), the failure kinds of
paper jobs (stale data = exit code 3, a locked book = exit code 4), one task through a
real worker process, and the holdout seal on API-launched store runs (real workers):
a sealed run never sees a date on or after ``holdout_start``, and an unseal is logged once
per configuration, a second one refused unless ``force`` and then flagged."""

from __future__ import annotations

import json
import shutil
import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from qts_api import schemas as S
from qts_api.main import create_app
from qts_api.research_fixture import HOLDOUT as _HOLDOUT
from qts_api.services.jobs import JobManager, cli_command
from qts_api.services.tasks import run_task, task_command
from qts_api.settings import Settings

from .conftest import LOCAL

HOLDOUT = pd.Timestamp(_HOLDOUT)


class Copy:
    """The world with its runs root (and ledger) and paper books copied: tasks write there."""

    def __init__(self, world, tmp_path):
        self.runs_root = tmp_path / "runs"
        shutil.copytree(world.runs_root, self.runs_root)
        self.ledger = self.runs_root / "trial_ledger.jsonl"
        self.paper_root = tmp_path / "paper"
        shutil.copytree(world.paper_root, self.paper_root)
        self.store, self.config, self.ids, self.run_ids = world.store, world.config, world.ids, world.run_ids


@pytest.fixture()
def wcopy(sealed, tmp_path):
    return Copy(sealed, tmp_path)


def run_inline(tmp_path, world, task: str, params: dict, paper_root=None) -> dict:
    """``run_task`` in this process on a job file of its own; returns the final job file."""
    jobs = tmp_path / "_jobs"
    jobs.mkdir(exist_ok=True)
    path = jobs / f"job_{abs(hash((task, json.dumps(params, sort_keys=True)))) % 16**12:012x}.json"
    path.write_text(json.dumps({"job_id": path.stem, "status": "running"}))
    payload = {
        "job_id": path.stem,
        "task": task,
        "runs_root": str(world.runs_root),
        "ledger_path": str(world.ledger),
        "store_path": str(world.store),
        "paper_root": str(paper_root or world.paper_root),
        "params": params,
        "label": task,
        "command": task_command(task, params, str(world.store)),
    }
    run_task(str(path), payload)
    return json.loads(path.read_text())


def test_signal_test_task(tmp_path, wcopy):
    out = run_inline(
        tmp_path, wcopy, "signal_test",
        {"feature": "max_ret_21", "store": {"max_names": 50}, "horizons": [5, 20], "use_ml": False,
         "baseline": str(wcopy.config)},
    )  # fmt: skip
    assert out["status"] == "done", out.get("error")
    r = out["result"]
    assert r["result_id"].endswith("_max_ret_21") and (wcopy.runs_root / "_signals" / f"{r['result_id']}.json").exists()
    assert r["n_tests"] == 3  # the family's third test: Holm over three
    assert out["progress"] == 1.0 and "signal test" in out["message"]
    log = (tmp_path / "_jobs" / f"{out['job_id']}.log").read_text()
    assert f"seal: holdout from {_HOLDOUT} sealed" in log  # the library's sealed load


def test_capacity_and_risk_tasks(tmp_path, wcopy):
    out = run_inline(
        tmp_path, wcopy, "capacity",
        {"store": {"max_names": 50}, "navs": [5e6, 1e9], "baseline": str(wcopy.config), "use_ml": False, "max_days": 40},
    )  # fmt: skip
    assert out["status"] == "done", out.get("error")
    cap = json.loads((wcopy.runs_root / "_capacity" / f"{out['result']['result_id']}.json").read_text())
    assert [p["nav"] for p in cap["points"]] == [5e6, 1e9] and cap["sealed_end"] < _HOLDOUT
    out = run_inline(
        tmp_path, wcopy, "risk_calibration",
        {"store": {"max_names": 50}, "nu": [0.0, 1.0], "baseline": str(wcopy.config), "use_ml": False, "max_days": 40},
    )  # fmt: skip
    assert out["status"] == "done", out.get("error")
    assert "recommended_nu_scale" in out["result"]


def test_sweep_member_and_finalize_tasks(tmp_path, wcopy):
    """A copy of the world's sweep with one member reset: the member job re-runs it, the
    finalize job recomputes PBO over both."""
    runs = wcopy.runs_root
    sid = wcopy.ids["sweep"]
    p = runs / "_sweeps" / f"{sid}.json"
    d = json.loads(p.read_text())
    d["runs_root"] = str(runs)
    for m in d["members"]:
        m["run_dir"] = str(runs / m["run_id"])
    d["members"][1].update(status="pending", run_dir=None, run_id=None, metrics={})
    p.write_text(json.dumps(d))
    out = run_inline(tmp_path, wcopy, "sweep_member", {"sweep_id": sid, "index": 1})
    assert out["status"] == "done", out.get("error")
    assert out["result"]["status"] == "done" and out["result"]["run_id"]
    assert out["result"]["reference"] is True  # the same configuration as the counted trial it replaces
    out = run_inline(tmp_path, wcopy, "sweep_finalize", {"sweep_id": sid})
    assert out["status"] == "done" and out["result"]["partial"] is False


def test_midcap_dry_run_task_keeps_the_payload(tmp_path, wcopy):
    out = run_inline(
        tmp_path, wcopy, "midcap_check", {"store": {"max_rank": 55}, "dry_run": True, "config": "configs/baseline.yaml"}
    )
    assert out["status"] == "done", out.get("error")
    assert out["result"]["dry_run"] is True and out["result"]["coverage_ok"] in (True, False)
    cov = out["result_payload"]["coverage"]
    assert cov["prefilter"] == "--max-rank 55" and cov["n_loaded_ids"] > 0
    assert len(list((wcopy.runs_root / "_checks").glob("*.json"))) == 1  # a dry run writes nothing


def test_paper_task_failures_are_classified(tmp_path, wcopy):
    from qts.paper.book import BookLock

    root, sealed = wcopy.paper_root, wcopy
    from qts.paper import history

    END = history("check", paper_root=str(root))[-1]["date"]  # the book is caught up to here
    # the synthetic store ends years ago: today's session is not there -> stale data, exit code 3
    out = run_inline(tmp_path, sealed, "paper_run", {"book": "check", "no_sync": True}, paper_root=root)
    assert out["status"] == "failed" and out["error_kind"] == "stale_data" and out["exit_code"] == 3
    assert out["message"].startswith("Failed: Data not yet published")
    # another process holds the book: locked, exit code 4
    with BookLock(root / "check", "test"):
        out = run_inline(
            tmp_path, sealed, "paper_run", {"book": "check", "until": END, "no_sync": True}, paper_root=root
        )
    assert out["error_kind"] == "book_locked" and out["exit_code"] == 4 and "locked" in out["message"]
    # caught up already: nothing to do
    out = run_inline(tmp_path, sealed, "paper_run", {"book": "check", "until": END, "no_sync": True}, paper_root=root)
    assert out["status"] == "done" and "nothing to do" in out["message"]


def test_reconcile_task_and_market_hash_task(tmp_path, wcopy):
    out = run_inline(tmp_path, wcopy, "paper_reconcile", {"book": "check"})
    assert out["status"] == "done", out.get("error")
    assert out["result_payload"]["sizing"] == "replica" and out["result_payload"]["per_day"]
    out = run_inline(tmp_path, wcopy, "market_hash", {"store": {"max_names": 45}})
    assert out["status"] == "done" and len(out["result"]["market_hash_full"]) == 16


def test_task_commands():
    assert task_command(
        "signal_test", {"feature": "x", "horizons": [5, 60], "store": {"start": "2000-01-01"}}, "/s"
    ) == ("qts signal test x --store /s --start 2000-01-01 --horizons 5,60")
    assert task_command(
        "capacity", {"navs": [5e6, 2.5e8], "store": {"max_names": 3000, "min_mcap_ever": 3e8}}, "/s"
    ) == ("qts capacity --store /s --max-names 3000 --min-mcap-ever 3e+08 --navs 5e+06,2.5e+08")
    assert task_command("risk_calibration", {"nu": [0, 0.05], "store": {}, "use_ml": False}, None) == (
        "qts risk calibrate --store '<store>' --nu 0,0.05 --no-ml"
    )
    sw = {"base_config": "configs/baseline.yaml", "params": [{"key": "a.b", "values": [0.5, 1.0, 2]}], "mode": "zip",
          "store": {}, "nav": 5e6, "max_days": 60, "use_ml": True, "sweep_id": "sw_0123456789ab"}  # fmt: skip
    assert task_command("sweep_member", sw, "/s") == (
        "qts sweep --config configs/baseline.yaml --param a.b=0.5,1.0,2 --zip --store /s --max-days 60 --id sw_0123456789ab"
    )
    assert task_command("midcap_check", {"store": {"max_rank": 2750}, "cache": True, "dry_run": True}, "/s") == (
        "qts check midcap --store /s --max-rank 2750 --cache --dry-run"
    )
    assert task_command("paper_run", {"book": "baseline", "until": "2026-11-09", "no_sync": True}, None, "paper") == (
        "qts paper run --name baseline --until 2026-11-09 --no-sync"
    )


# --------------------------------------------------------------------------- one real worker
def poll(c: TestClient, job_id: str, timeout: float = 300.0) -> dict:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        j = c.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("done", "failed", "cancelled"):
            return j
        time.sleep(0.3)
    raise AssertionError(f"job {job_id} timed out: {j}")


def test_a_task_through_a_real_worker_process(tmp_path, sealed):
    root = tmp_path / "paper"
    shutil.copytree(sealed.paper_root, root)
    s = Settings(runs_root=tmp_path / "runs", store_path=sealed.store, paper_root=root)
    with TestClient(create_app(s), base_url=LOCAL) as c:
        r = c.post("/api/paper/books/check/run", json={"no_sync": True})
        j = S.Job.model_validate(poll(c, r.json()["job_id"]))
        assert j.status == "failed" and j.task == "paper_run" and j.error_kind == "stale_data" and j.exit_code == 3
        assert j.command == r.json()["command"]


# --------------------------------------------------------------------------- the seal on store runs
RUN_STORE = {
    "kind": "store",
    "ml": False,
    "phase": 1,
    "stocks": 300,
    "days": 1512,
    "seed": 0,
    "signal_lag_days": 1,
    "max_names": 50,
    "config_overrides": {},
    "max_days": 20,
    "cache": True,
}


def _ledger(path) -> list[dict]:
    from qts.validation import TrialLedger

    return TrialLedger(str(path)).raw_records()


def test_api_store_runs_are_sealed_and_unseal_once_per_config(tmp_path, sealed):
    """Runs start from (a copy of) the world's sealed baseline run, in a runs root and
    ledger of their own."""
    base = sealed.run_ids["sealed"]
    (tmp_path / "runs").mkdir()
    shutil.copytree(sealed.runs_root / base, tmp_path / "runs" / base)
    RUN = {**RUN_STORE, "base_run_id": base}
    s = Settings(runs_root=tmp_path / "runs", store_path=sealed.store)
    with TestClient(create_app(s), base_url=LOCAL) as c:
        v = S.ConfigValidation.model_validate(c.post("/api/config/validate", json=RUN).json())
        assert v.valid and "warning.sealed" in [w.type for w in v.warnings]
        # 1) a sealed run: data through the day before the holdout, never after
        out = c.post("/api/runs", json=RUN).json()
        j = poll(c, out["job_id"])
        assert j["status"] == "done", j["message"]
        run_dir = tmp_path / "runs" / out["run_id"]
        summ = json.loads((run_dir / "summary.json").read_text())
        assert summ["sealed_end"] == sealed.sealed_end and summ["unsealed"] is False
        assert pd.Timestamp(summ["backtest"]["end"]) < HOLDOUT
        nav = pd.read_parquet(run_dir / "nav.parquet")
        assert pd.DatetimeIndex(nav.index).max() < HOLDOUT
        for f in ("returns.parquet", "weights.parquet", "ic.parquet"):
            idx = pd.read_parquet(run_dir / f).index
            dates = pd.to_datetime(idx.get_level_values(0) if isinstance(idx, pd.MultiIndex) else idx)
            assert not len(dates) or dates.max() < HOLDOUT, f
        rec = _ledger(s.ledger)[-1]
        assert (
            rec["sealed_end"] == sealed.sealed_end
            and rec["market_hash_full"]
            and rec["market_hash"] != rec["market_hash_full"]
        )
        assert rec["family"].endswith(rec["market_hash_full"][:12])  # the full-history family
        assert next(r for r in c.get("/api/runs").json() if r["id"] == out["run_id"])["seal"]["state"] == "sealed"
        log = c.get(f"/api/jobs/{out['job_id']}/log?tail=400").json()["lines"]
        load_end = str((HOLDOUT - pd.Timedelta(days=1)).date())
        assert any(f"seal: holdout from {_HOLDOUT} sealed; loading data through {load_end}" in ln for ln in log)

        # 2) unseal: needs a reason; the event is logged before the run, the run loads everything
        assert c.post("/api/runs", json={**RUN, "unseal_reason": "  "}).status_code == 422
        assert c.post("/api/runs", json={**RUN, "kind": "synthetic", "unseal_reason": "x"}).status_code == 422
        assert c.post("/api/runs", json={**RUN, "force": True}).status_code == 422
        unseal = {**RUN, "unseal_reason": "final check of the candidate"}
        v = c.post("/api/config/validate", json=unseal).json()
        assert v["valid"] and "warning.unseal" in [w["type"] for w in v["warnings"]]
        out2 = c.post("/api/runs", json=unseal).json()
        assert poll(c, out2["job_id"])["status"] == "done"
        recs = _ledger(s.ledger)
        events = [r for r in recs if r.get("kind") == "unseal"]
        assert len(events) == 1 and events[0]["unseal_reason"] == "final check of the candidate"
        assert events[0]["forced"] is False and recs.index(events[0]) == len(recs) - 2
        assert "--unseal 'final check of the candidate'" in events[0]["command"]
        summ2 = json.loads((tmp_path / "runs" / out2["run_id"] / "summary.json").read_text())
        assert summ2["unsealed"] is True and pd.Timestamp(summ2["backtest"]["end"]) > HOLDOUT
        assert summ2["holdout_already_used"] is False

        # 3) a second unseal of the same configuration: refused before anything is queued ...
        n = len(_ledger(s.ledger))
        r = c.post("/api/runs", json=unseal)
        assert r.status_code == 422 and "already unsealed" in r.json()["detail"]
        v = c.post("/api/config/validate", json=unseal).json()
        assert not v["valid"] and v["errors"][0]["loc"] == ["body", "unseal_reason"]
        check = c.get(f"/api/seal/unseals?config_hash={events[0]['config_hash']}&phase=1").json()
        assert check["requires_force"] is True
        assert len(_ledger(s.ledger)) == n
        # ... and flagged when forced: one more unseal event, forced
        forced = {**unseal, "force": True}
        v = c.post("/api/config/validate", json=forced).json()
        assert v["valid"] and "warning.holdout_used" in [w["type"] for w in v["warnings"]]
        out3 = c.post("/api/runs", json=forced).json()
        assert poll(c, out3["job_id"])["status"] == "done"
        events = [r for r in _ledger(s.ledger) if r.get("kind") == "unseal"]
        assert len(events) == 2 and events[1]["forced"] is True and events[1]["n_previous_unseals"] == 1
        summ3 = json.loads((tmp_path / "runs" / out3["run_id"] / "summary.json").read_text())
        assert summ3["holdout_already_used"] is True
        seal = next(r for r in c.get("/api/runs").json() if r["id"] == out3["run_id"])["seal"]
        assert seal["state"] == "unsealed" and seal["holdout_already_used"] is True
        st = c.get("/api/seal").json()
        assert len(st["unseals"]) == 2 and st["n_forced"] == 1


def test_unseal_without_a_seal_in_force_warns(tmp_path, world):
    s = Settings(runs_root=tmp_path / "runs", store_path=world.store)  # QTS_SEAL_FILE=none here
    with TestClient(create_app(s, start_jobs=False), base_url=LOCAL) as c:
        v = c.post("/api/config/validate", json={**RUN_STORE, "unseal_reason": "x"}).json()
        assert v["valid"] and [w["type"] for w in v["warnings"]] == ["warning.no_seal"]
        r = c.get("/api/runs")
        assert r.status_code == 200


def test_cli_command_records_the_unseal():
    req = {**RUN_STORE, "unseal_reason": "why", "force": True, "config_overrides": {"a.b": 1}}
    assert cli_command(req, "/s").endswith("--max-days 20 --cache --unseal why --force --set a.b=1")
    assert "--unseal" not in cli_command({**RUN_STORE, "kind": "synthetic", "unseal_reason": "x"}, None)


def test_rank_prefilter_and_partial_coverage_on_store_runs(tmp_path, world):
    """``max_rank`` (``qts run --max-rank``) and ``allow_partial_coverage`` reach the
    command line and the loader; a rank prefilter with the peak-cap flags is refused, and
    a load the coverage guard refuses fails its job with ``error_kind: coverage``."""
    req = {**RUN_STORE, "max_names": None, "max_rank": 1500, "allow_partial_coverage": True}
    assert cli_command(req, "/s").startswith("qts run --store /s --max-rank 1500 ")
    assert "--allow-partial-coverage" in cli_command(req, "/s")
    assert "--allow-partial-coverage" not in cli_command({**req, "kind": "synthetic"}, None)
    s = Settings(runs_root=tmp_path / "runs", store_path=world.store)
    with TestClient(create_app(s, start_jobs=False), base_url=LOCAL) as c:
        v = c.post("/api/config/validate", json={**RUN_STORE, "max_rank": 50, "max_names": 45}).json()
        assert not v["valid"] and v["errors"][0]["loc"] == ["body", "max_rank"]
        v = c.post("/api/config/validate", json={**RUN_STORE, "kind": "synthetic", "allow_partial_coverage": True})
        assert v.json()["errors"][0]["loc"] == ["body", "allow_partial_coverage"]
        assert c.post("/api/config/validate", json={**RUN_STORE, "max_names": None, "max_rank": 55}).json()["valid"]
    from qts.research.coverage import LoadCoverageError

    from qts_api.services.tasks import _error_kind

    assert _error_kind(LoadCoverageError("short")) == ("coverage", 2)


def test_uncovered_store_prefilters_are_refused_with_the_guards_error(tmp_path, sealed, monkeypatch):
    """The load-coverage guard (``qts.research.coverage``) on API store runs: a rank
    prefilter below the universe's need is refused before queueing (422, the guard's own
    advice); ``allow_partial_coverage`` turns it into a flagged warning; a peak-cap
    prefilter is measured in the job, which fails with ``error_kind: coverage``, and a run
    whose load was partial carries ``seal.load_coverage.ok = false``."""
    monkeypatch.setenv("QTS_LOAD_COVERAGE", "enforce")  # the suite runs with it off
    base = sealed.run_ids["sealed"]
    (tmp_path / "runs").mkdir()
    shutil.copytree(sealed.runs_root / base, tmp_path / "runs" / base)
    run = {**RUN_STORE, "base_run_id": base, "max_names": None}
    s = Settings(runs_root=tmp_path / "runs", store_path=sealed.store)
    with TestClient(create_app(s), base_url=LOCAL) as c:
        r = c.post("/api/runs", json={**run, "max_rank": 20})
        assert r.status_code == 422, r.text
        err = r.json()["errors"][0]
        assert err["loc"] == ["body", "max_rank"] and err["type"] == "value_error.load_coverage"
        assert "incomplete universe" in err["msg"] and "Use --max-rank 250" in err["msg"]
        v = c.post("/api/config/validate", json={**run, "max_rank": 20, "allow_partial_coverage": True}).json()
        assert v["valid"] and "warning.partial_coverage" in [w["type"] for w in v["warnings"]]
        assert c.post("/api/config/validate", json={**run, "max_rank": 55}).json()["valid"]
        # research launches: the same check before anything is queued
        r = c.post("/api/capacity", json={"navs": [5e6], "store": {"max_rank": 20}})
        assert r.status_code == 422 and r.json()["type"].endswith("load-coverage"), r.text
        # a peak-cap prefilter is measured inside the job (DuckDB) and refused there
        v = c.post("/api/config/validate", json={**run, "max_names": 45}).json()
        assert v["valid"] and "warning.coverage_measured" in [w["type"] for w in v["warnings"]]
        out = c.post("/api/runs", json={**run, "max_names": 45}).json()
        j = S.Job.model_validate(poll(c, out["job_id"]))
        assert j.status == "failed" and j.error_kind == "coverage", j.message
        assert j.message.startswith("Refused: the store prefilter would load an incomplete universe")
        # loaded anyway: the run says its load was partial
        out = c.post("/api/runs", json={**run, "max_names": 45, "allow_partial_coverage": True}).json()
        j = poll(c, out["job_id"])
        assert j["status"] == "done", j["message"]
        launch = json.loads((tmp_path / "runs" / out["run_id"] / "launch.json").read_text())
        assert "--allow-partial-coverage" in launch["command"] and launch["load_coverage"]["ok"] is False
        seal = next(r for r in c.get("/api/runs").json() if r["id"] == out["run_id"])["seal"]
        assert seal["load_coverage"]["ok"] is False and seal["load_coverage"]["method"] == "measured"
        rec = _ledger(s.ledger)[-1]
        assert rec["load_coverage"]["ok"] is False


def test_job_view_of_tasks_and_eta(tmp_path):
    jm = JobManager(Settings(runs_root=tmp_path), start=False)
    a = jm.submit_task(task="capacity", params={"navs": [1.0]}, label="c", command="qts capacity", size_key={"n": 1})
    v = jm.view(jm.get(a["job_id"]))
    assert v["task"] == "capacity" and v["command"] == "qts capacity" and v["kind"] == "store" and v["result"] is None
    assert v["estimated_seconds"] is None
    jm._mark(a["job_id"], status="done", started_at="2026-01-01T00:00:00Z", finished_at="2026-01-01T00:10:00Z",
             result={"result_id": "x", "nested": {"no": 1}})  # fmt: skip
    b = jm.submit_task(task="capacity", params={}, label="c2", command="", size_key={"n": 1})
    assert jm.view(jm.get(b["job_id"]))["estimated_seconds"] == pytest.approx(600)
    done = jm.view(jm.get(a["job_id"]))
    assert done["result"] == {"result_id": "x"}  # only flat values are served
