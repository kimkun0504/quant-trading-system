"""Production-path smoke test: ``ResearchPipeline`` end to end on the synthetic
market (universe -> hybrid risk -> alpha -> cost -> optimizer -> simulator ->
hurdles), plus determinism and the eligibility contract."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from qts.core.config import SystemConfig
from qts.data.synthetic import SyntheticSpec, generate
from qts.pipeline import ResearchPipeline, format_result

warnings.filterwarnings("ignore", category=RuntimeWarning)

GATES = {"t_stat", "ic_sign", "incremental_ic", "net_sharpe_delta", "breakeven_cost", "dsr", "pbo",
         "phase_sharpe", "suspect"}
CORE = ("sharpe_gross", "sharpe_net", "cagr_net", "ann_vol_net", "max_drawdown_net", "avg_daily_turnover",
        "avg_cost_bps_per_side", "avg_gross_exposure", "avg_net_exposure")


@pytest.fixture(scope="module")
def md():
    return generate(SyntheticSpec(n_stocks=120, n_days=900, seed=7, ic_true=0.08))


@pytest.fixture(scope="module")
def cfg():
    c = SystemConfig()
    c.universe.top_n_by_mcap = 100          # time-varying universe inside the 120 names
    return c


@pytest.fixture(scope="module")
def result(md, cfg):
    pipe = ResearchPipeline(cfg, use_ml=False)
    res = pipe.run(md, max_days=120)
    return pipe, res


def test_pipeline_smoke(result, md):
    pipe, res = result
    bt = res.backtest
    assert len(bt.returns) == 120 and len(bt.nav) == 121
    for k in CORE:
        assert k in res.summary and np.isfinite(res.summary[k]), k
    assert np.isfinite(res.dsr) and 0.0 <= res.dsr <= 1.0
    assert res.decile is not None and np.isfinite(res.decile.t_stat)
    assert np.isfinite(res.ic_report["mean_ic"]) and res.ic_report["n"] >= 5
    # ICIR annualised at the sampled frequency (every H days), not 252
    assert res.ic_report["periods_per_year"] == pytest.approx(252 / pipe.cfg.alpha.horizon_days, rel=0.1)
    assert res.config_hash and res.data_hash
    assert not res.cost_sensitivity.isna().any().any()
    assert bt.diagnostics["n_failed_solves"] == 0
    assert "OVERALL" in format_result(res)


def test_pipeline_hurdle_report_shape(result):
    _, res = result
    h = res.hurdles
    assert set(h["status"]) == GATES
    assert set(h["status"].values()) <= {"PASS", "FAIL", "skipped"}
    # standalone run without a config grid: PBO and incremental IC are skipped, not failed
    assert h["status"]["pbo"] == "skipped" and h["status"]["incremental_ic"] == "skipped"
    assert set(h["required"]) == {"phase_sharpe", "suspect", "dsr"}
    for g in h["required"]:
        assert h["status"][g] in ("PASS", "FAIL")
    assert "pbo" not in h["failed"]


def test_pipeline_held_names_are_eligible(result):
    """Every *new* position is opened in a name eligible on its decision date
    (positions in names that later leave the universe may be unwound gradually)."""
    pipe, res = result
    elig = pipe.md_u.eligibility()
    W = res.backtest.weights
    prev = pd.Series(0.0, index=W.columns)
    for d, w in W.iterrows():
        opened = (w.abs() > 1e-8) & (prev.abs() <= 1e-8)
        bad = [t for t in W.columns[opened.to_numpy()] if not elig.loc[d, t]]
        assert not bad, (d, bad[:5])
        prev = w
    # alpha emitted only for eligible names at each decision date
    d = W.index[len(W) // 2]
    out = pipe.alpha.as_of(d)
    ok = elig.loc[d]
    assert set(out.score.index) <= set(ok[ok].index)
    # universe genuinely varies inside the sample (the test is not vacuous)
    assert elig.loc[W.index].sum(axis=1).max() < len(W.columns)


def test_pipeline_is_deterministic(md, cfg, result):
    _, res1 = result
    res2 = ResearchPipeline(cfg, use_ml=False).run(md, max_days=120)
    np.testing.assert_array_equal(res1.backtest.nav.to_numpy(), res2.backtest.nav.to_numpy())
    assert res1.config_hash == res2.config_hash and res1.data_hash == res2.data_hash


def test_pipeline_ledger_dedupes_reruns(md, cfg, tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    r1 = ResearchPipeline(cfg, use_ml=False, ledger_path=path).run(md, max_days=60)
    r2 = ResearchPipeline(cfg, use_ml=False, ledger_path=path).run(md, max_days=60)
    assert r1.dsr_inputs["n_trials_raw"] == 1 and r2.dsr_inputs["n_trials_raw"] == 1   # identical re-run: 1 trial
    assert r1.dsr == pytest.approx(r2.dsr)
    cfg2 = SystemConfig()
    cfg2.universe.top_n_by_mcap = 100
    cfg2.portfolio.risk_aversion = cfg.portfolio.risk_aversion * 2
    r3 = ResearchPipeline(cfg2, use_ml=False, ledger_path=path).run(md, max_days=60)
    assert r3.dsr_inputs["n_trials_raw"] == 2 and r3.dsr_inputs["var_trials_sr_annual"] > 0
