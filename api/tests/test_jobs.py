"""Job lifecycle with a real worker process on a tiny synthetic market (~15 s + ~8 s;
not marked slow: both stay under 30 s)."""

from __future__ import annotations

import json
import time

from fastapi.testclient import TestClient

from qts_api import schemas as S
from qts_api.main import create_app
from qts_api.settings import Settings

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
    with TestClient(create_app(Settings(runs_root=tmp_path))) as c:
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

    # a fresh API process still reports the finished job
    with TestClient(create_app(Settings(runs_root=tmp_path), start_jobs=False)) as c2:
        assert c2.get(f"/api/jobs/{out['job_id']}").json()["status"] == "done"


def test_cancel_running_job(tmp_path):
    with TestClient(create_app(Settings(runs_root=tmp_path))) as c:
        out = c.post("/api/runs", json={**REQ, "max_days": None}).json()
        poll(c, out["job_id"], lambda j: j["status"] == "running" and j["progress"] >= 0.05, timeout=120)
        j = c.delete(f"/api/jobs/{out['job_id']}").json()
        assert j["status"] == "failed" and j["message"] == "Cancelled"
        assert c.get(f"/api/runs/{out['run_id']}").status_code == 409
        assert not (tmp_path / out["run_id"]).exists()
        assert c.app.state.jobs.counts() == {"queued": 0, "running": 0}
