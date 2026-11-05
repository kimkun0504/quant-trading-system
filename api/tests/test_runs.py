"""Every read endpoint against the fixture run: shapes (Pydantic models) and values
(spot-checked against the run directory's own files)."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
from pydantic import TypeAdapter

from qts_api import schemas as S
from tests.conftest import OLD_RUN_ID


def close(a, b, tol=1e-9):
    return a is not None and b is not None and math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


# ----------------------------------------------------------------------------- list
def test_list_runs(client, run_id, summary):
    r = client.get("/api/runs")
    assert r.status_code == 200
    runs = TypeAdapter(list[S.RunSummary]).validate_python(r.json())
    ids = [x.id for x in runs]
    assert ids == [run_id, OLD_RUN_ID]  # newest first
    row = runs[0]
    assert row.status == "done" and row.kind == "synthetic" and row.phase == 1 and not row.use_ml
    assert row.label == "Composite, seed 5"
    assert row.headline is not None
    assert close(row.headline.sharpe_net, summary["backtest"]["sharpe_net"])
    assert close(row.headline.max_drawdown, summary["backtest"]["max_drawdown_net"])
    assert close(row.headline.dsr, summary["dsr"])
    assert row.headline.overall_gate == ("PASS" if summary["overall"] else "FAIL")
    assert row.config_hash == summary["config_hash"]


# ----------------------------------------------------------------------------- detail
def test_run_detail(client, run_id, run_dir, summary, returns):
    r = client.get(f"/api/runs/{run_id}")
    assert r.status_code == 200
    d = S.RunDetail.model_validate(r.json())
    b = summary["backtest"]
    for k in ("sharpe_net", "sharpe_gross", "cagr_net", "final_nav", "avg_cost_bps_per_side"):
        assert close(getattr(d.summary, k), b[k]), k
    assert d.summary.n_days == b["n_days"] == len(returns)
    assert d.status == "done" and d.label == "Composite, seed 5"
    assert d.config.kind == "synthetic" and d.config.stocks == 60 and d.config.days == 764
    assert d.config.seed == 5 and d.config.signal_lag_days == 1
    assert "trade_aversion" in d.config.config_yaml
    assert d.manifest.run_id == run_id and d.manifest.python
    assert d.hurdles.overall == ("PASS" if summary["overall"] else "FAIL")
    hurdles = json.loads((run_dir / "hurdles.json").read_text())
    assert [g.name for g in d.hurdles.gates] == [g["name"] for g in hurdles["gates"]]
    assert set(d.hurdles.required) == set(hurdles["required"])
    assert close(d.ic_report.mean_ic, summary["ic"]["mean_ic"]) and d.ic_report.horizon_days == 20
    assert sum(y.n for y in d.ic_report.by_year) == summary["ic"]["n"]
    assert close(d.dsr, summary["dsr"])
    net = returns["net"].dropna()
    assert close(d.dsr_inputs.skew, net.skew(), 1e-9) and close(d.dsr_inputs.kurtosis, net.kurt() + 3, 1e-9)
    assert d.dsr_inputs.n_obs == len(net)
    assert close(d.dsr_inputs.n_trials, summary["dsr_inputs"]["n_trials"])
    assert [f.sharpe_net for f in d.fold_sharpes] == pytest.approx([f["sharpe"] for f in summary["folds"]])
    assert all("–" in f.label for f in d.fold_sharpes)
    assert d.execution_venue and "closing auction" in d.execution_venue
    assert d.decile is not None and close(d.decile.t_stat, summary["decile"]["t_stat"])
    assert d.pbo is None and d.notes == summary["notes"]


def test_old_run_degrades(client):
    d = S.RunDetail.model_validate(client.get(f"/api/runs/{OLD_RUN_ID}").json())
    assert d.execution_venue is None
    # recovered from the recorded `qts.cli demo` command
    assert d.config.kind == "synthetic" and d.config.stocks == 60 and d.config.days == 764
    assert d.dsr_inputs.n_obs > 0 and np.isfinite(d.dsr_inputs.skew)  # recomputed from returns.parquet
    assert d.ic_report.by_year  # recomputed from ic.parquet
    s = client.get(f"/api/runs/{OLD_RUN_ID}/series", params={"names": "exante_vol,nav"}).json()
    assert all(v is None for v in s["series"]["exante_vol"]) and s["series"]["nav"][0] is not None
    e = S.RunExposures.model_validate(client.get(f"/api/runs/{OLD_RUN_ID}/exposures").json())
    assert e.factors == [] and e.latest is None and e.history.dates == []
    h = S.RunHoldings.model_validate(client.get(f"/api/runs/{OLD_RUN_ID}/holdings").json())
    assert h.positions and all(p.alpha is None and p.industry is None for p in h.positions)


# ----------------------------------------------------------------------------- series
def test_series(client, run_id, summary, returns):
    r = client.get(f"/api/runs/{run_id}/series")
    s = S.ColumnarSeries.model_validate(r.json())
    assert list(s.series) == list(S.SERIES_NAMES)
    n = summary["backtest"]["n_days"]
    assert len(s.dates) == n and all(len(v) == n for v in s.series.values())
    assert s.dates[0] == str(returns.index[0].date())
    assert close(s.series["nav"][-1], summary["backtest"]["final_nav"])
    assert close(min(s.series["drawdown"]), summary["backtest"]["max_drawdown_net"])
    assert s.series["net"] == pytest.approx(returns["net"].tolist())
    to = [v for v in s.series["turnover"] if v is not None]
    assert close(np.mean(to), summary["backtest"]["avg_daily_turnover"], 1e-6)
    ev = [v for v in s.series["exante_vol"] if v is not None]
    assert len(ev) == n and 0.005 < np.median(ev) < 0.5


def test_series_subset_and_unknown(client, run_id):
    s = client.get(f"/api/runs/{run_id}/series", params={"names": "nav,drawdown"}).json()
    assert list(s["series"]) == ["nav", "drawdown"]
    r = client.get(f"/api/runs/{run_id}/series", params={"names": "nav,bogus"})
    assert r.status_code == 422
    body = r.json()
    assert r.headers["content-type"].startswith("application/problem+json")
    assert "bogus" in body["detail"] and body["errors"][0]["loc"] == ["query", "names"]


# ----------------------------------------------------------------------------- costs
def test_costs(client, run_id, run_dir, summary):
    c = S.RunCosts.model_validate(client.get(f"/api/runs/{run_id}/costs").json())
    costs = pd.read_parquet(run_dir / "costs.parquet")
    assert len(c.daily.dates) == len(costs) == len(c.nav)
    total = sum(sum(v or 0 for v in getattr(c.daily.series, k)) for k in S.COST_COMPONENTS)
    assert close(total, summary["backtest"]["total_costs_usd"], 1e-6)
    assert [r.multiplier for r in c.cost_sensitivity] == [0.5, 1.0, 2.0, 3.0]
    one = next(r for r in c.cost_sensitivity if r.multiplier == 1.0)
    assert close(one.cost_bps_per_side, summary["backtest"]["avg_cost_bps_per_side"], 1e-6)
    assert c.nav[0] == pytest.approx(5e6)


# ----------------------------------------------------------------------------- exposures / holdings / ic
def test_exposures(client, run_id, summary):
    e = S.RunExposures.model_validate(client.get(f"/api/runs/{run_id}/exposures").json())
    assert "market" in e.factors and "momentum" in e.factors and "beta" in e.factors
    assert e.roles["market"] == "market" and e.roles["momentum"] == "alpha_theme" and e.roles["beta"] == "risk_only"
    assert e.latest is not None and e.latest.date == e.history.dates[-1]
    assert len(e.history.dates) == summary["backtest"]["n_days"]
    beta = [v for v in e.history.series["beta"] if v is not None]
    assert max(abs(v) for v in beta) < 0.2  # beta-neutral book (band 0.05 + slack for unwinds)


def test_holdings(client, run_id, run_dir):
    h = S.RunHoldings.model_validate(client.get(f"/api/runs/{run_id}/holdings").json())
    assert h.date == h.available_dates[-1] and len(h.available_dates) >= 5
    assert h.positions == sorted(h.positions, key=lambda p: -p.weight)
    assert all(p.alpha is not None and p.specific_vol and p.industry for p in h.positions)
    w = pd.read_parquet(run_dir / "weights.parquet").loc[pd.Timestamp(h.date)]
    assert sum(abs(p.weight) for p in h.positions) == pytest.approx(w.abs().sum(), abs=1e-6)
    first = h.available_dates[0]
    h0 = S.RunHoldings.model_validate(client.get(f"/api/runs/{run_id}/holdings", params={"date": first}).json())
    assert h0.date == first and h0.positions != h.positions
    r = client.get(f"/api/runs/{run_id}/holdings", params={"date": "1999-01-01"})
    assert r.status_code == 404 and "available_dates" in r.json()["detail"]


def test_ic(client, run_id, summary):
    ic = S.RunIc.model_validate(client.get(f"/api/runs/{run_id}/ic").json())
    assert len(ic.ic.dates) == summary["ic"]["n"] and ic.horizon_days == 20
    assert np.nanmean([v for v in ic.ic.series.rank_ic if v is not None]) == pytest.approx(summary["ic"]["mean_ic"])
    assert [p.horizon_days for p in ic.decay] == [5, 20, 60]


# ----------------------------------------------------------------------------- errors
@pytest.mark.parametrize("suffix", ["", "/series", "/costs", "/exposures", "/holdings", "/ic"])
def test_unknown_run_404(client, suffix):
    r = client.get(f"/api/runs/no_such_run{suffix}")
    assert r.status_code == 404
    body = S.ApiErrorBody.model_validate(r.json())
    assert body.status == 404 and body.detail == "No run with id no_such_run"
    assert r.headers["content-type"].startswith("application/problem+json")


def test_unknown_endpoint_404(client):
    r = client.get("/api/nothing/here")
    assert r.status_code == 404 and isinstance(r.json()["detail"], str)


def test_pending_and_failed_runs_409(client):
    jobs = client.app.state.jobs
    req = {
        "kind": "synthetic",
        "ml": False,
        "phase": 1,
        "stocks": 50,
        "days": 764,
        "seed": 1,
        "signal_lag_days": 1,
        "config_overrides": {},
    }
    q = jobs.submit(
        request=req,
        config={},
        config_hash="abc",
        run_id="20990101T000000000000Z_pending0",
        label="Pending",
        git_sha="x",
    )
    r = client.get(f"/api/runs/{q['run_id']}")
    assert r.status_code == 409 and "still running" in r.json()["detail"]
    for suffix in ("/series", "/costs", "/ic"):
        assert client.get(f"/api/runs/{q['run_id']}{suffix}").status_code == 409
    listed = {x["id"]: x for x in client.get("/api/runs").json()}
    assert listed[q["run_id"]]["status"] == "running" and listed[q["run_id"]]["headline"] is None
    jobs.cancel(q["job_id"])
    r = client.get(f"/api/runs/{q['run_id']}")
    assert r.status_code == 409 and "failed before evaluation" in r.json()["detail"]
    listed = {x["id"]: x for x in client.get("/api/runs").json()}
    assert listed[q["run_id"]]["status"] == "failed"
    TypeAdapter(list[S.RunSummary]).validate_python(list(listed.values()))


# ----------------------------------------------------------------------------- cache
def test_registry_cache_invalidation(client, run_id, run_dir):
    reg = client.app.state.registry
    e1 = reg.get(run_id)
    assert reg.get(run_id) is e1  # cached
    f = run_dir / "extras.json"
    original = f.read_text()
    try:
        d = json.loads(original)
        d["run"]["label"] = "Relabelled"
        f.write_text(json.dumps(d))
        assert client.get(f"/api/runs/{run_id}").json()["label"] == "Relabelled"
    finally:
        f.write_text(original)
    assert client.get(f"/api/runs/{run_id}").json()["label"] == "Composite, seed 5"


def test_nan_serialised_as_null(client, run_id, run_dir):
    f = run_dir / "summary.json"
    original = f.read_text()
    try:
        d = json.loads(original)
        d["backtest"]["transfer_coefficient_mean"] = "nan"
        f.write_text(json.dumps(d))
        r = client.get(f"/api/runs/{run_id}")
        assert r.status_code == 200 and r.json()["summary"]["transfer_coefficient_mean"] is None
    finally:
        f.write_text(original)
