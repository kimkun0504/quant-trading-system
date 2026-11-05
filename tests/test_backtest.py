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
from qts.portfolio import SimpleCostModel, SinglePeriodOptimizer

SUMMARY_KEYS = [
    "cagr_gross",
    "cagr_net",
    "ann_vol_net",
    "sharpe_gross",
    "sharpe_net",
    "max_drawdown_net",
    "avg_daily_turnover",
    "effective_holding_days",
    "avg_cost_bps_per_side",
    "transfer_coefficient_mean",
    "realized_vs_exante_vol_ratio",
    "hit_rate_net",
    "breakeven_cost_bps",
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
    dates = md.dates[-301:]  # 300 realised returns
    return sim.run(md, dates, alpha_model.as_of, risk_model.as_of, opt, nav0=1e6, rebalance_every=1)


def _check_identity(res: BacktestResult, md, haircut: float = -0.30):
    R = md.close.loc[res.nav.index].pct_change(fill_method=None)
    raw = R.values[1:]
    w = res.weights.values
    # delisting: a held name with NaN next close earns the haircut against its sign
    r = np.where(np.isnan(raw), np.where(w != 0, haircut * np.sign(w), 0.0), raw)
    lhs = res.nav.values[1:]
    rhs = res.nav.values[:-1] * (1.0 + (res.weights.values * r).sum(axis=1)) - res.costs["total"].values
    assert np.max(np.abs(lhs - rhs) / res.nav.values[:-1]) < 1e-8


@pytest.mark.slow
def test_planted_alpha_positive_net_sharpe(planted_run, md):
    s = planted_run.summary()
    print(
        "\nplanted-alpha backtest: gross Sharpe {:.2f}, net Sharpe {:.2f}, turnover {:.3f}/day, "
        "holding {:.0f} d, cost {:.1f} bps/side, TC {:.2f}, real/exante vol {:.2f}".format(
            s["sharpe_gross"],
            s["sharpe_net"],
            s["avg_daily_turnover"],
            s["effective_holding_days"],
            s["avg_cost_bps_per_side"],
            s["transfer_coefficient_mean"],
            s["realized_vs_exante_vol_ratio"],
        )
    )
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
    assert s["ann_vol_net"] <= 0.10 * 1.5  # vol target 10% is a cap, realised may wobble


@pytest.mark.slow
def test_nav_identity(planted_run, md):
    _check_identity(planted_run, md)
    # net return definition
    net = planted_run.nav.pct_change().dropna()
    assert np.allclose(net.values, planted_run.returns["net"].values, atol=1e-12)


@pytest.mark.slow
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
    assert res.weights.abs().sum(axis=1).max() < 1e-5  # nothing to hold (solver noise only)
    assert res.costs["total"].sum() >= 0
    assert res.costs["total"].sum() < 1.0  # empty book -> no costs beyond noise
    assert np.abs(res.returns["net"]).max() < 1e-6
    assert (not np.isfinite(s["sharpe_net"])) or abs(s["sharpe_net"]) < 0.5
    assert res.nav.iloc[-1] == pytest.approx(1e6, rel=1e-5)


def test_rebalance_every_and_drift(md, risk_model, alpha_model):
    cfg = make_cfg()
    cm = SimpleCostModel(cfg.cost)
    opt = SinglePeriodOptimizer(cfg.portfolio, cm)
    res = DailyBarSimulator(cfg, cm).run(
        md, md.dates[-31:], alpha_model.as_of, risk_model.as_of, opt, nav0=1e6, rebalance_every=5
    )
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
        er[tick] = er.abs().max() * 3  # make sure we hold it
        return dataclasses.replace(a, expected_return=er)

    res = DailyBarSimulator(cfg, cm).run(md2, dates, alpha_fn, risk_model.as_of, opt, nav0=1e6)
    held_before = res.weights.loc[dates[13], tick]
    assert held_before > 0
    dl = res.diagnostics["delistings"]
    ev = [d for d in dl if d["ticker"] == tick and d["date"] == kill_from]
    assert len(ev) == 1
    ev = ev[0]
    last = dates[14]  # decision row on which the exit is booked
    w_last = res.weights.loc[last, tick]
    assert w_last > 0 and ev["weight"] == pytest.approx(w_last)
    assert ev["haircut_return"] == pytest.approx(cfg.cost.delisting_haircut)
    assert ev["liquidation_usd"] == pytest.approx(
        w_last * (1 + cfg.cost.delisting_haircut) * res.nav.loc[last]
    )
    assert ev["cost_usd"] > 0  # spread + fees charged on the forced exit
    # the forced liquidation is a recorded trade (on top of the day's rebalance) and counts in turnover
    prev = dates[13]
    w_pre_last = (
        res.weights.loc[prev, tick]
        * (1 + md2.close.loc[last, tick] / md2.close.loc[prev, tick] - 1)
        * res.nav.loc[prev]
        / res.nav.loc[last]
    )
    x_opt = w_last - w_pre_last
    assert res.trades.loc[last, tick] == pytest.approx(x_opt - w_last * (1 + cfg.cost.delisting_haircut))
    assert res.turnover.loc[last] == pytest.approx(res.trades.loc[last].abs().sum())
    assert (res.weights.loc[kill_from:, tick] == 0).all()
    assert (res.trades.loc[kill_from:, tick] == 0).all()
    assert res.nav.notna().all()
    _check_identity(res, md2, cfg.cost.delisting_haircut)


def test_cost_frame_raises_on_unrecognised_or_nan():
    idx = pd.Index(["A", "B"])
    with pytest.raises(ValueError, match=r"unrecognised|no recognised"):
        DailyBarSimulator._cost_frame(pd.DataFrame({"foo": [1.0, 2.0]}, index=idx), idx)
    with pytest.raises(ValueError, match="non-finite"):
        DailyBarSimulator._cost_frame(
            pd.DataFrame({"spread": [1.0, np.nan], "impact": [0.0, 0.0], "fees": [0.0, 0.0]}, index=idx), idx
        )
    with pytest.raises(ValueError, match="non-finite"):
        DailyBarSimulator._cost_frame(pd.Series([1.0, np.nan], index=idx), idx)
    ok = DailyBarSimulator._cost_frame(
        pd.DataFrame(
            {"spread_usd": [1.0, 2.0], "impact_cost": [0.5, 0.5], "fees_usd": [0.1, 0.1]}, index=idx
        ),
        idx,
    )
    assert ok["spread"].tolist() == [1.0, 2.0] and ok["impact"].tolist() == [0.5, 0.5]
    tot = DailyBarSimulator._cost_frame(pd.DataFrame({"total": [3.0, 4.0]}, index=idx), idx)
    assert tot["impact"].tolist() == [3.0, 4.0]
    assert DailyBarSimulator._cost_frame(None, idx).values.sum() == 0.0


def test_simulator_snapshot_is_the_cost_model_snapshot(md):
    from qts.core.costs import LiquiditySnapshot
    from qts.tcost import TransactionCostModel

    cfg = make_cfg()
    tcm = TransactionCostModel(cfg.cost)
    sim = DailyBarSimulator(cfg, tcm)
    t = md.dates[-50]
    snap = sim._snapshot(md, t)
    assert isinstance(snap, LiquiditySnapshot) and snap.date == t and snap.tickers.equals(md.tickers)
    raw = tcm.snapshot(md, t)
    for k in ("half_spread", "adv_shares", "adv_usd", "sigma_daily", "borrow_fee"):
        pd.testing.assert_series_equal(getattr(snap, k), getattr(raw, k))
    assert snap.auction_share == raw.auction_share and snap.borrow_fee_source == raw.borrow_fee_source
    # estimate() through the simulator's snapshot prices the requested venue with real spreads
    trades = pd.Series(0.0, index=md.tickers)
    trades.iloc[:3] = [1e4, -2e4, 3e4]
    est = tcm.estimate(trades, snap, "continuous")
    assert (est["venue"] == "continuous").all()
    pd.testing.assert_series_equal(
        est.bps()["spread_bps"].iloc[:3], raw.half_spread_bps.iloc[:3], check_names=False
    )
    # SimpleCostModel (test double) satisfies the protocol and is used directly
    cm = SimpleCostModel(cfg.cost)
    assert DailyBarSimulator(cfg, cm)._cm is cm


def test_run_with_tcost_model_venue_and_persistence(md, risk_model, alpha_model):
    from qts.tcost import TransactionCostModel

    cfg = make_cfg()
    tcm = TransactionCostModel(cfg.cost)
    opt = SinglePeriodOptimizer(cfg.portfolio, tcm)
    dates = md.dates[-21:]
    res_r = DailyBarSimulator(cfg, tcm).run(md, dates, alpha_model.as_of, risk_model.as_of, opt, nav0=1e6)
    assert res_r.diagnostics["venue"] == "routed"  # D3 default
    assert res_r.costs["fees"].sum() > 0
    _check_identity(res_r, md)
    res_a = DailyBarSimulator(cfg, tcm).run(
        md, dates, alpha_model.as_of, risk_model.as_of, opt, nav0=1e6, venue="auction"
    )
    assert res_a.costs["spread"].sum() == 0.0  # auction: no spread crossing
    _check_identity(res_a, md)
    res_c = DailyBarSimulator(cfg, tcm).run(
        md, dates, alpha_model.as_of, risk_model.as_of, opt, nav0=1e6, venue="continuous"
    )
    assert res_c.costs["spread"].sum() > 0.0  # continuous venue actually priced
    _check_identity(res_c, md)
    # routed spread <= continuous spread on the same first-day trade (part goes to the auction)
    assert res_r.costs["spread"].iloc[0] <= res_c.costs["spread"].iloc[0] + 1e-9
    # persistence off -> impact no larger than with persistence on
    cfg_off = make_cfg()
    cfg_off.cost.impact_persistence = False
    tcm_off = TransactionCostModel(cfg_off.cost)
    res_off = DailyBarSimulator(cfg_off, tcm_off).run(
        md,
        dates,
        alpha_model.as_of,
        risk_model.as_of,
        SinglePeriodOptimizer(cfg_off.portfolio, tcm_off),
        nav0=1e6,
    )
    assert res_off.costs["impact"].iloc[0] == pytest.approx(res_r.costs["impact"].iloc[0], rel=1e-6)
    assert res_off.costs["impact"].sum() <= res_r.costs["impact"].sum() + 1e-9


def test_signal_lag_shifts_decisions_by_one_day():
    """D4 on the hand-computed fixture: with zero costs and a date-keyed optimizer
    double, the lag-1 run's weights at t equal the lag-0 run's weights at t-1;
    nothing is traded on the first day (no signal date yet)."""
    from tests.test_simulator_handcomputed import (
        DATES,
        NAV0,
        TARGETS,
        ConstantBpsCost,
        PresetOptimizer,
        alpha_fn,
        make_md,
        risk_fn,
    )

    md = make_md()
    cfg = SystemConfig()
    seen = []

    def alpha_rec(t):
        seen.append(t)
        return alpha_fn(t)

    res0 = DailyBarSimulator(cfg, ConstantBpsCost()).run(
        md, DATES, alpha_fn, risk_fn, PresetOptimizer(TARGETS), nav0=NAV0
    )
    opt1 = PresetOptimizer(TARGETS)
    res1 = DailyBarSimulator(cfg, ConstantBpsCost()).run(
        md, DATES, alpha_rec, risk_fn, opt1, nav0=NAV0, signal_lag_days=1
    )
    assert res1.diagnostics["signal_lag_days"] == 1 and res1.diagnostics["n_no_signal_days"] == 1
    np.testing.assert_allclose(res1.weights.iloc[0].to_numpy(), 0.0)
    np.testing.assert_allclose(
        res1.weights.iloc[1:].to_numpy(), res0.weights.iloc[:-1].to_numpy(), atol=1e-12
    )
    # signals / risk / optimizer snapshot evaluated at t-1; execution still at the close of t
    assert seen == list(DATES[:3])
    assert [c["risk"].date for c in opt1.calls] == list(DATES[:3])
    assert list(res1.weights.index) == list(DATES[:4])
    # config default is 0 and the kwarg overrides it
    assert cfg.execution.signal_lag_days == 0
    cfg1 = SystemConfig()
    cfg1.execution.signal_lag_days = 1
    res_cfg = DailyBarSimulator(cfg1, ConstantBpsCost()).run(
        md, DATES, alpha_fn, risk_fn, PresetOptimizer(TARGETS), nav0=NAV0
    )
    pd.testing.assert_frame_equal(res_cfg.weights, res1.weights)
    with pytest.raises(ValueError):
        DailyBarSimulator(cfg, ConstantBpsCost()).run(
            md, DATES, alpha_fn, risk_fn, PresetOptimizer(TARGETS), signal_lag_days=-1
        )


# --------------------------------------------------------------- metrics
def test_metrics_basic():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.standard_normal(2520) * 0.01 + 0.0004)
    sr = annualized_sharpe(r)
    assert sr == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(252))
    assert abs(annualized_sharpe(r, lo_adjust=True) - sr) < 0.2 * abs(sr) + 0.05  # iid -> little change
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
    assert be == pytest.approx(0.001 / 0.05 * 1e4)  # 200 bps
    cs = cost_sensitivity(g, to, cost_bps_per_side=be / 2, multipliers=(0.5, 1, 2, 3))
    assert cs.loc[2, "net_cagr"] == pytest.approx(0.0, abs=1e-12)
    assert cs["net_cagr"].is_monotonic_decreasing
    ts = turnover_stats(to)
    assert ts["mean_daily"] == pytest.approx(0.05) and ts["one_sided_mean_daily"] == pytest.approx(0.025)
