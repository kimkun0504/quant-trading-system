"""Tests for the daily-bar simulator and metrics."""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from qts.backtest import (
    BacktestResult,
    DailyBarSimulator,
    annualized_sharpe,
    breakeven_cost_bps,
    cost_sensitivity,
    max_drawdown,
    newey_west_tstat,
    sharpe_t_stat,
    turnover_stats,
)
from qts.core.config import SystemConfig
from qts.core.types import AlphaOutput
from qts.portfolio import SimpleCostModel, SinglePeriodOptimizer

SUMMARY_KEYS = [
    "cagr_gross", "cagr_net", "ann_vol_net", "sharpe_gross", "sharpe_net", "max_drawdown_net",
    "avg_daily_turnover", "effective_holding_days", "avg_cost_bps_per_side", "transfer_coefficient_mean",
    "realized_vs_exante_vol_ratio", "hit_rate_net", "breakeven_cost_bps",
]


def make_cfg(**pkw) -> SystemConfig:
    cfg = SystemConfig()
    cfg.portfolio.risk_aversion = 3.0
    for k, v in pkw.items():
        setattr(cfg.portfolio, k, v)
    return cfg


@pytest.fixture(scope="module")
def planted_run(md, risk_model, alpha_model):
    cfg = make_cfg()
    cm = SimpleCostModel(cfg.cost)
    opt = SinglePeriodOptimizer(cfg.portfolio, cm)
    sim = DailyBarSimulator(cfg, cm)
    dates = md.dates[-301:]            # 300 realised returns
    return sim.run(md, dates, alpha_model.as_of, risk_model.as_of, opt, nav0=1e6, rebalance_every=1)


def _check_identity(res: BacktestResult, md):
    R = md.close.loc[res.nav.index].pct_change(fill_method=None)
    r = np.nan_to_num(R.values[1:])
    lhs = res.nav.values[1:]
    rhs = res.nav.values[:-1] * (1.0 + (res.weights.values * r).sum(axis=1)) - res.costs["total"].values
    assert np.max(np.abs(lhs - rhs) / res.nav.values[:-1]) < 1e-8


def test_planted_alpha_positive_net_sharpe(planted_run, md):
    s = planted_run.summary()
    print("\nplanted-alpha backtest: gross Sharpe %.2f, net Sharpe %.2f, turnover %.3f/day, "
          "holding %.0f d, cost %.1f bps/side, TC %.2f, real/exante vol %.2f" % (
              s["sharpe_gross"], s["sharpe_net"], s["avg_daily_turnover"], s["effective_holding_days"],
              s["avg_cost_bps_per_side"], s["transfer_coefficient_mean"], s["realized_vs_exante_vol_ratio"]))
    assert s["n_days"] == 300
    assert s["sharpe_net"] > 0
    assert s["sharpe_gross"] > s["sharpe_net"]
    assert s["n_failed_solves"] == 0
    assert planted_run.costs["total"].sum() > 0
    assert (planted_run.costs[["spread", "impact", "fees", "borrow"]] >= 0).all().all()
    assert 0.5 < s["realized_vs_exante_vol_ratio"] < 2.0
    assert 0 < s["transfer_coefficient_mean"] <= 1
    assert abs(s["avg_net_exposure"]) <= 0.06
    assert s["avg_gross_exposure"] <= 2.0 + 1e-6
    assert s["ann_vol_net"] <= 0.10 * 1.5   # vol target 10% is a cap, realised may wobble


def test_nav_identity(planted_run, md):
    _check_identity(planted_run, md)
    # net return definition
    net = planted_run.nav.pct_change().dropna()
    assert np.allclose(net.values, planted_run.returns["net"].values, atol=1e-12)


def test_summary_keys_and_breakeven(planted_run):
    s = planted_run.summary()
    for k in SUMMARY_KEYS:
        assert k in s, k
    assert np.isfinite(s["sharpe_net"])
    if s["sharpe_net"] > 0:
        assert s["breakeven_cost_bps"] >= s["avg_cost_bps_per_side"]
    # effective holding days consistent with turnover and gross
    gross = planted_run.weights.abs().sum(axis=1).mean()
    assert s["effective_holding_days"] == pytest.approx(gross / (s["avg_daily_turnover"] / 2), rel=1e-9)


def test_zero_alpha_strategy(md, risk_model, alpha_model):
    cfg = make_cfg()
    cm = SimpleCostModel(cfg.cost)
    opt = SinglePeriodOptimizer(cfg.portfolio, cm)
    sim = DailyBarSimulator(cfg, cm)

    def zero_alpha(t):
        a = alpha_model.as_of(t)
        return dataclasses.replace(a, expected_return=a.expected_return * 0.0, score=a.score * 0.0)

    res = sim.run(md, md.dates[-41:], zero_alpha, risk_model.as_of, opt, nav0=1e6)
    s = res.summary()
    assert res.weights.abs().sum(axis=1).max() < 1e-5       # nothing to hold (solver noise only)
    assert res.costs["total"].sum() >= 0
    assert res.costs["total"].sum() < 1.0                 # empty book -> no costs beyond noise
    assert np.abs(res.returns["net"]).max() < 1e-6
    assert (not np.isfinite(s["sharpe_net"])) or abs(s["sharpe_net"]) < 0.5
    assert res.nav.iloc[-1] == pytest.approx(1e6, rel=1e-5)


def test_rebalance_every_and_drift(md, risk_model, alpha_model):
    cfg = make_cfg()
    cm = SimpleCostModel(cfg.cost)
    opt = SinglePeriodOptimizer(cfg.portfolio, cm)
    res = DailyBarSimulator(cfg, cm).run(md, md.dates[-31:], alpha_model.as_of, risk_model.as_of, opt,
                                         nav0=1e6, rebalance_every=5)
    to = res.turnover
    assert (to.values[1:5] == 0).all() and to.values[0] > 0 and to.values[5] > 0
    # on non-rebalance days post-trade weights equal drifted weights -> no trading cost, only borrow
    c = res.costs.iloc[1]
    assert c["spread"] == 0 and c["impact"] == 0 and c["fees"] == 0
    _check_identity(res, md)


def test_delisting_is_liquidated_and_flagged(md, risk_model, alpha_model):
    dates = md.dates[-41:]
    tick = "S0007"
    kill_from = dates[15]
    close = md.close.copy()
    close.loc[kill_from:, tick] = np.nan
    md2 = dataclasses.replace(md, close=close)
    cfg = make_cfg()
    cm = SimpleCostModel(cfg.cost)
    opt = SinglePeriodOptimizer(cfg.portfolio, cm)

    def alpha_fn(t):
        a = alpha_model.as_of(t)
        er = a.expected_return.copy()
        er[tick] = er.abs().max() * 3          # make sure we hold it
        return dataclasses.replace(a, expected_return=er)

    res = DailyBarSimulator(cfg, cm).run(md2, dates, alpha_fn, risk_model.as_of, opt, nav0=1e6)
    held_before = res.weights.loc[dates[13], tick]
    assert held_before > 0
    dl = res.diagnostics["delistings"]
    assert any(d["ticker"] == tick and d["date"] == kill_from for d in dl)
    assert (res.weights.loc[kill_from:, tick] == 0).all()
    assert (res.trades.loc[kill_from:, tick] == 0).all()
    assert res.nav.notna().all()
    _check_identity(res, md2)


# --------------------------------------------------------------- metrics
def test_metrics_basic():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.standard_normal(2520) * 0.01 + 0.0004)
    sr = annualized_sharpe(r)
    assert sr == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(252))
    assert abs(annualized_sharpe(r, lo_adjust=True) - sr) < 0.2 * abs(sr) + 0.05   # iid -> little change
    # positively autocorrelated series -> Lo adjustment lowers the Sharpe
    ar = pd.Series(np.convolve(rng.standard_normal(3000), np.ones(5) / 5, mode="valid") + 0.05)
    assert annualized_sharpe(ar, lo_adjust=True) < annualized_sharpe(ar)
    assert sharpe_t_stat(r) == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(len(r)), rel=0.05)
    assert abs(newey_west_tstat(r, lags=5) - r.mean() / r.std(ddof=1) * np.sqrt(len(r))) < 0.5

    dd = pd.Series([0.1, -0.5, 0.2, -0.2])
    # nav: 1.1, 0.55, 0.66, 0.528 -> max dd from 1.1 to 0.528
    assert max_drawdown(dd) == pytest.approx(0.528 / 1.1 - 1)


def test_breakeven_and_sensitivity():
    g = pd.Series([0.001] * 100)
    to = pd.Series([0.05] * 100)
    be = breakeven_cost_bps(g, to)
    assert be == pytest.approx(0.001 / 0.05 * 1e4)         # 200 bps
    cs = cost_sensitivity(g, to, cost_bps_per_side=be / 2, multipliers=(0.5, 1, 2, 3))
    assert cs.loc[2, "net_cagr"] == pytest.approx(0.0, abs=1e-12)
    assert cs["net_cagr"].is_monotonic_decreasing
    ts = turnover_stats(to)
    assert ts["mean_daily"] == pytest.approx(0.05) and ts["one_sided_mean_daily"] == pytest.approx(0.025)
