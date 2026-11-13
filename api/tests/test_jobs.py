"""Job lifecycle with a real worker process on a tiny synthetic market (~15 s + ~8 s;
not marked slow: both stay under 30 s)."""

from __future__ import annotations

import json
import time

from fastapi.testclient import TestClient

from qts_api import schemas as S
from qts_api.main import create_app
from qts_api.settings import Settings

LOCAL = "http://localhost"  # Host allow-list (see conftest.LOCAL)

REQ = {
    "kind": "synthetic",
    "ml": False,
    "phase": 1,
    "stocks": 50,
    "days": 764,
    "seed": 2,
    "signal_lag_days": 1,
    "config_overrides": {"portfolio.trade_aversion": 2},
    "max_days": 60,
    "label": "Lifecycle test",
}


def poll(c: TestClient, job_id: str, until, timeout: float = 420.0) -> list[dict]:
    seen: list[dict] = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        j = S.Job.model_validate(c.get(f"/api/jobs/{job_id}").json()).model_dump()
        if not seen or seen[-1] != j:
            seen.append(j)
        if until(j):
            return seen
        time.sleep(0.2)
    raise AssertionError(f"job {job_id} timed out; last state {seen[-1] if seen else None}")


def test_job_lifecycle(tmp_path):
    with TestClient(create_app(Settings(runs_root=tmp_path)), base_url=LOCAL) as c:
        out = c.post("/api/runs", json=REQ).json()
        seen = poll(c, out["job_id"], lambda j: j["status"] in ("done", "failed"))
        last = seen[-1]
        assert last["status"] == "done", last
        assert last["progress"] == 1.0 and "gates evaluated" in last["message"]
        progress = [j["progress"] for j in seen]
        assert progress == sorted(progress)  # monotone
        messages = " | ".join(j["message"] for j in seen)
        assert "Running the backtest: day" in messages, messages

        d = S.RunDetail.model_validate(c.get(f"/api/runs/{out['run_id']}").json())
        assert d.label == "Lifecycle test" and d.config.stocks == 50 and d.config.days == 764
        assert d.summary.n_days == 60 and d.execution_venue
        assert "trade_aversion: 2" in d.config.config_yaml
        run_dir = tmp_path / out["run_id"]
        assert (run_dir / "extras.json").exists() and not (tmp_path / "_staging" / out["run_id"]).exists()
        assert json.loads((run_dir / "manifest.json").read_text())["run_id"] == out["run_id"]
        runs = c.get("/api/runs").json()
        assert [r["id"] for r in runs] == [out["run_id"]] and runs[0]["status"] == "done"
        led = c.get("/api/ledger").json()
        assert len(led["trials"]) == 1

        # the finished job in the queue listing, its log, and the run's recorded source
        q = S.JobQueue.model_validate(c.get("/api/jobs").json())
        assert [j.job_id for j in q.jobs] == [out["job_id"]] and q.counts == {"done": 1}
        job = q.jobs[0]
        assert job.label == "Lifecycle test" and job.n_overrides == 1 and job.started_at and job.finished_at
        assert job.last_log and "finished" in job.last_log
        lg = S.JobLog.model_validate(c.get(f"/api/jobs/{out['job_id']}/log").json())
        text = "\n".join(lg.lines)
        assert "Fitting the risk model" in text and "Running the backtest" in text and not lg.truncated
        assert "nav=" in text, text  # the simulator's own progress line
        two = S.JobLog.model_validate(c.get(f"/api/jobs/{out['job_id']}/log?tail=2").json())
        assert two.lines == lg.lines[-2:] and two.truncated and two.total_lines == lg.total_lines
        src = d.config.source
        assert src.recorded and src.origin == "api" and (src.kind, src.stocks, src.days) == ("synthetic", 50, 764)
        assert src.max_days == 60 and not src.cache
        assert src.command == (
            "qts demo --stocks 50 --days 764 --seed 2 --max-days 60 --set portfolio.trade_aversion=2"
        )
        assert json.loads((run_dir / "launch.json").read_text())["request"]["label"] == "Lifecycle test"

        # clone it: same base, one more override; the dry run and the launch agree on the hash
        clone = {**REQ, "base_run_id": out["run_id"], "config_overrides": {"alpha.horizon_days": 10}, "label": None}
        v = S.ConfigValidation.model_validate(c.post("/api/config/validate", json=clone).json())
        assert v.valid and v.base_config_hash == d.manifest.config_hash and v.config_hash != v.base_config_hash
        same = S.ConfigValidation.model_validate(
            c.post("/api/config/validate", json={**clone, "config_overrides": {}}).json()
        )
        assert same.config_hash == d.manifest.config_hash  # an unchanged clone is the same config
        assert [w.type for w in same.warnings] == ["warning.same_config"] and out["run_id"] in same.warnings[0].msg
        # an ETA for a job of the same size, from the finished one
        nxt = c.post("/api/runs", json=clone).json()
        assert nxt["run_id"].endswith(v.config_hash[:8])
        est = S.Job.model_validate(c.get(f"/api/jobs/{nxt['job_id']}").json()).estimated_seconds
        assert est is not None and 0 < est < 420
        c.delete(f"/api/jobs/{nxt['job_id']}")

    # a fresh API process still reports the finished job
    with TestClient(create_app(Settings(runs_root=tmp_path), start_jobs=False), base_url=LOCAL) as c2:
        assert c2.get(f"/api/jobs/{out['job_id']}").json()["status"] == "done"


def test_cancel_running_job(tmp_path):
    with TestClient(create_app(Settings(runs_root=tmp_path)), base_url=LOCAL) as c:
        out = c.post("/api/runs", json={**REQ, "max_days": None}).json()
        poll(c, out["job_id"], lambda j: j["status"] == "running" and j["progress"] >= 0.05, timeout=120)
        j = c.delete(f"/api/jobs/{out['job_id']}").json()
        assert j["status"] == "cancelled" and j["message"] == "Cancelled"
        assert c.get(f"/api/runs/{out['run_id']}").status_code == 409
        assert not (tmp_path / out["run_id"]).exists()
        assert c.app.state.jobs.counts() == {"queued": 0, "running": 0}
