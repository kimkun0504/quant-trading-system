"""Parameter sweeps (``qts.research.sweep``, ``qts sweep``): expansion, the trial-impact
estimate, PBO on planted grids, and a resumable sweep on a synthetic store whose
members re-use the cached fits."""

from __future__ import annotations

import json
import math
import warnings

import numpy as np
import pandas as pd
import pytest

from qts.core.config import SystemConfig
from qts.data.store import ParquetStore
from qts.data.synthetic import SyntheticSpec, generate
from qts.research import sweep as SW
from qts.validation import TrialLedger


def test_parse_and_expand():
    assert SW.parse_param("portfolio.trade_aversion=0.5, 1,2") == ("portfolio.trade_aversion", [0.5, 1, 2])
    assert SW.parse_param("alpha.feature_set=v1,v2")[1] == ["v1", "v2"]
    for bad in ("x", "=1", "a.b=", "a.b=1,1"):
        with pytest.raises(SW.SweepError):
            SW.parse_param(bad)
    g = SW.expand([("a", [1, 2]), ("b", ["x", "y", "z"])], "grid")
    assert len(g) == 6 and g[0] == {"a": 1, "b": "x"} and g[-1] == {"a": 2, "b": "z"}
    z = SW.expand([("a", [1, 2]), ("b", ["x", "y"])], "zip")
    assert z == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    with pytest.raises(SW.SweepError, match="same number"):
        SW.expand([("a", [1, 2]), ("b", ["x"])], "zip")
    with pytest.raises(SW.SweepError):
        SW.expand([], "grid")


def test_plan_caps_validates_and_estimates_the_haircut(tmp_path):
    led = TrialLedger(str(tmp_path / "l.jsonl"))
    months = pd.date_range("2010-01-31", periods=60, freq="ME")
    rng = np.random.default_rng(0)
    for sr in (0.3, 0.5, 0.7, 0.4):
        led.log(
            name="F", family="F", config={"sr": sr}, metrics={"sharpe_annual": sr, "n_obs": 5000, "skew": 0.0, "kurt": 4.0},
            returns_monthly=pd.Series(rng.normal(0.005, 0.02, 60), index=months),
        )  # fmt: skip
    cfg = SystemConfig()
    params = [("portfolio.trade_aversion", [0.5, 1.0, 2.0])]
    plan = SW.plan_sweep(
        params, config=cfg, family="F", ledger_path=led.path, runs_root=str(tmp_path / "runs")
    )
    assert len(plan.members) == 3 and plan.members[1].config_hash == cfg.hash()
    imp = plan.impact
    assert imp.n_trials_raw == 4 and imp.n_new == 3 and imp.n_eff_after == pytest.approx(imp.n_eff_before + 3)
    assert imp.exp_max_sr_after > imp.exp_max_sr_before > 0
    assert imp.best_sharpe == 0.7 and imp.dsr_best_after < imp.dsr_best_before
    assert "expected max Sharpe" in plan.format()
    with pytest.raises(SW.SweepError, match="max-runs"):
        SW.plan_sweep(
            [("portfolio.trade_aversion", list(range(1, 14)))],
            config=cfg,
            family="F",
            runs_root=str(tmp_path),
        )
    with pytest.raises(SW.SweepError, match="member 0"):
        SW.plan_sweep(
            [("portfolio.rebalance_every_days", [0, 1])], config=cfg, family="F", runs_root=str(tmp_path)
        )
    with pytest.raises(SW.SweepError, match="repeats"):
        SW.plan_sweep(
            [("portfolio.trade_aversion", [1, 1.0])], config=cfg, family="F", runs_root=str(tmp_path)
        )
    json.dumps(plan.to_dict())


# --------------------------------------------------------------------------- PBO
def _planted(n_days: int, n_cfg: int, skill: float | None, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 0.01, size=(n_days, n_cfg))
    if skill is not None:
        X[:, 0] += skill  # one configuration with a real edge in every block
    return pd.DataFrame(X, index=pd.bdate_range("2005-01-03", periods=n_days))


def test_pbo_flags_a_planted_overfit_grid_and_clears_a_real_edge():
    # pure noise: the in-sample winner is a coin flip out of sample
    noise = [SW.sweep_pbo(_planted(2500, 10, None, s))["pbo"] for s in range(5)]
    assert np.mean(noise) > 0.3
    # one configuration with a daily edge of 0.15 sd (annual SR ~2.4): the IS winner wins OOS
    for seed in range(3):
        edge = SW.sweep_pbo(_planted(2500, 10, 0.0015, seed))
        assert edge["pbo"] < 0.05 and edge["n_combinations"] == 252 and edge["n_blocks"] == 10
    assert SW.sweep_pbo(_planted(15, 3, None, 0))["pbo"] is None
    assert "fewer than two" in SW.sweep_pbo(_planted(500, 1, None, 0))["reason"]


# --------------------------------------------------------------------------- end to end
@pytest.fixture(scope="module")
def store(tmp_path_factory):
    md = generate(SyntheticSpec(n_stocks=60, n_days=640, seed=5))
    root = tmp_path_factory.mktemp("sweep") / "store"
    ParquetStore(str(root)).write_market_data(md)
    return root


def test_sweep_runs_reuses_fits_resumes_and_computes_pbo(store, tmp_path, capsys):
    from qts.cli import main

    cfg = SystemConfig().with_overrides({"universe.top_n_by_mcap": 50})
    base = tmp_path / "base.yaml"
    base.write_text(cfg.to_yaml())
    argv = [
        "sweep", "--config", str(base), "--param", "portfolio.trade_aversion=0.5,2",
        "--store", str(store), "--max-names", "60", "--no-ml", "--max-days", "60",
        "--ledger", str(tmp_path / "l.jsonl"), "--runs-root", str(tmp_path / "runs"),
    ]  # fmt: skip
    assert main([*argv, "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "dry run" in out and "Sweep sw_" in out and not (tmp_path / "l.jsonl").exists()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert main(argv) == 1  # not a terminal and no --yes: not run
        capsys.readouterr()
        assert main([*argv, "--yes"]) == 0
    out = capsys.readouterr().out
    sid = next(w for w in out.split() if w.startswith("sw_"))
    plan = SW.load_sweep(sid, str(tmp_path / "runs"))
    assert plan.status == "done" and all(m.run_dir for m in plan.members)
    # the second member reused both fits of the first (portfolio-only sweep)
    assert plan.members[0].metrics["cache"] == {"risk_fit": "stored", "alpha_fit": "stored"}
    assert plan.members[1].metrics["cache"] == {"risk_fit": "hit", "alpha_fit": "hit"}
    led = TrialLedger(str(tmp_path / "l.jsonl"))
    recs = led.raw_records()
    assert [r["sweep_member"] for r in recs] == [0, 1] and {r["sweep_id"] for r in recs} == {sid}
    assert len({r["family"] for r in recs}) == 1 and led.n_trials(recs[0]["family"]) == 2
    assert plan.pbo is not None and plan.pbo["pbo"] is not None and plan.pbo["partial"] is False
    assert "metric vs parameter" in out and "PBO (CSCV, S = 10" in out
    # resume after an interrupted member: only that one runs again
    d = json.loads(plan.path.read_text())
    d["members"][1]["status"] = "running"
    plan.path.write_text(json.dumps(d))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert main([*argv, "--yes"]) == 0
    out = capsys.readouterr().out
    assert "resuming (1 of 2 done)" in out and "member 1" in out and "member 0" not in out
    assert len(TrialLedger(str(tmp_path / "l.jsonl")).raw_records()) == 3
    # a completed sweep re-run does nothing
    assert main([*argv, "--yes"]) == 0
    assert "every member is done" in capsys.readouterr().out


def test_finalize_marks_partial(tmp_path):
    plan = SW.plan_sweep([("portfolio.trade_aversion", [0.5, 1.0])], config=SystemConfig(), family="F",
                         runs_root=str(tmp_path))  # fmt: skip
    SW.create_sweep(plan)
    res = SW.finalize_sweep(plan.id, str(tmp_path))
    assert res.pbo["pbo"] is None and res.pbo["partial"] is True
    assert math.isnan(float("nan")) and "PBO not computed" in res.format()
