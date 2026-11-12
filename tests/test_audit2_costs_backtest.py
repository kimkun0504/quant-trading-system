"""Fixes for the second-pass audit of costs / execution / backtest
(``docs/audit/second_pass_costs_backtest_api.md``): borrow in the breakeven gate (#1),
config bounds and non-negative cost estimates (#3), exact streak persistence (#5),
the optimizer snapshot's auction share under lag (#14) and holdings signal dates (#15).

Every expected number is computed by hand from the formulas (not by re-running the
simulator's own loop)."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from qts.backtest import DailyBarSimulator
from qts.backtest import metrics as M
from qts.core.config import CostConfig, ExecutionConfig, SystemConfig
from qts.core.costs import CostEstimate, LiquiditySnapshot, auction_share_for_date
from qts.core.types import MarketData
from qts.tcost import TransactionCostModel
from tests.test_simulator_handcomputed import DATES as HC_DATES
from tests.test_simulator_handcomputed import (
    NAV0,
    TICKERS,
    ConstantBpsCost,
    PresetOptimizer,
    alpha_fn,
    make_md,
    risk_fn,
)


# ----------------------------------------------------------------- #1 borrow in breakeven
def test_gross_after_borrow_and_breakeven_net_of_borrow():
    """Short-heavy book with 50%/yr borrow: ``gross_after_borrow = gross - borrow/nav_pre``
    on the realisation axis, and breakeven / cost sensitivity start from it."""
    w = {"A": 0.3, "B": -0.5, "C": -0.4}
    targets = dict.fromkeys(HC_DATES[:4], w)
    cfg = SystemConfig().with_overrides({"execution.signal_lag_days": 0, "portfolio.rebalance_every_days": 1})
    cost = ConstantBpsCost(spread_bps=1.0, impact_bps=2.0, fees_bps=0.5, borrow_fee_annual=0.5)
    res = DailyBarSimulator(cfg, cost).run(
        make_md(), HC_DATES, alpha_fn, risk_fn, PresetOptimizer(targets), nav0=NAV0
    )
    nav_pre = res.nav.to_numpy()[:-1]
    borrow = res.costs["borrow"].to_numpy()
    assert (borrow > 0).all()
    gab = res.returns["gross_after_borrow"].to_numpy()
    np.testing.assert_allclose(gab, res.returns["gross"].to_numpy() - borrow / nav_pre, rtol=0, atol=1e-15)
    # day 0 by hand: shorts 0.9 of NAV, 50%/yr / 252 days
    assert borrow[0] / NAV0 == pytest.approx(0.9 * 0.5 / 252, rel=1e-12)
    to = res.turnover_on_return_dates()
    s = res.summary()
    assert s["breakeven_cost_bps"] == pytest.approx(gab.mean() / to.mean() * 1e4, rel=1e-12)
    assert s["breakeven_cost_bps"] < M.breakeven_cost_bps(res.returns["gross"], to)
    # cost sensitivity on gross_after_borrow: each row = gab - c * turnover
    cs = res.cost_sensitivity(cost_bps_per_side=3.5)
    net1 = res.returns["gross_after_borrow"] - 3.5e-4 * to
    assert cs.loc[1, "net_sharpe"] == pytest.approx(M.annualized_sharpe(net1), rel=1e-12)
    # 3.5 bp/side is the realised per-side trading cost here: the 1x row is the realised net
    assert s["avg_cost_bps_per_side"] == pytest.approx(3.5, rel=1e-12)
    np.testing.assert_allclose(net1.to_numpy(), res.returns["net"].to_numpy(), atol=1e-15)


def test_gross_after_borrow_fallback_for_old_results():
    cfg = SystemConfig().with_overrides({"execution.signal_lag_days": 0, "portfolio.rebalance_every_days": 1})
    targets = {d: {"A": 0.2, "B": -0.2, "C": 0.0} for d in HC_DATES[:4]}
    res = DailyBarSimulator(cfg, ConstantBpsCost(borrow_fee_annual=0.1)).run(
        make_md(), HC_DATES, alpha_fn, risk_fn, PresetOptimizer(targets), nav0=NAV0
    )
    expected = res.returns["gross_after_borrow"].copy()
    res.returns = res.returns.drop(columns="gross_after_borrow")  # a result written before the column
    pd.testing.assert_series_equal(res.gross_after_borrow_returns, expected, check_names=False, atol=1e-15)


# ----------------------------------------------------------------- #3 bounds
@pytest.mark.parametrize(
    "kw",
    [
        {"clearing_per_share": -0.0002},
        {"cat_fee_per_share": -1e-6},
        {"finra_taf_per_share_sold": -0.1},
        {"auction_exchange_fee_per_share": -0.001},
        {"auction_volume_share_default": 0.0},
        {"auction_volume_share_default": 1.5},
        {"impact_b_continuous": 1e12},
        {"impact_b_auction": 0.0},
        {"impact_exponent": 1.0},
        {"impact_exponent": 2.5},
        {"max_participation_adv": 0.0},
        {"default_half_spread_bps": -1.0},
        {"commission_per_share": float("nan")},
        {"borrow_proxy_si_gc": 0.3, "borrow_proxy_si_htb": 0.2},
        {"tca_reversion_reference": "vwap"},
    ],
)
def test_cost_config_rejects_out_of_bounds(kw):
    with pytest.raises(ValueError, match="CostConfig"):
        CostConfig(**kw)


@pytest.mark.parametrize(
    "kw",
    [
        {"signal_lag_days": -1},
        {"signal_lag_days": 11},
        {"signal_lag_days": 1.5},
        {"signal_lag_days": True},
        {"auction_max_pct_of_auction_volume": 0.0},
        {"algo_max_pct_adv": 1.2},
        {"loc_offset_bps": -3.0},
    ],
)
def test_execution_config_rejects_out_of_bounds(kw):
    with pytest.raises(ValueError, match="ExecutionConfig"):
        ExecutionConfig(**kw)


def test_config_bounds_edges_and_round_trip():
    CostConfig(impact_exponent=2.0, impact_b_continuous=10.0, auction_volume_share_default=1.0)
    ExecutionConfig(signal_lag_days=0)
    ExecutionConfig(signal_lag_days=10)
    cfg = SystemConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):  # configs are frozen: no in-place bypass
        cfg.cost.clearing_per_share = -1.0  # type: ignore[misc]
    with pytest.raises(ValueError, match="clearing_per_share"):
        cfg.with_overrides({"cost.clearing_per_share": -1.0})  # replace() re-runs __post_init__
    with pytest.raises(ValueError):
        SystemConfig.from_dict({"cost": {"impact_b_auction": -0.3}})


def test_cost_estimate_rejects_negative_components():
    idx = pd.Index(["A", "B"])
    t = pd.Series([1e4, -1e4], index=idx)
    CostEstimate.from_components(idx, "routed", t, [1.0, 0.0], [0.0, -1e-9], [0.1, 0.1])  # rounding noise ok
    with pytest.raises(ValueError, match="negative cost estimate"):
        CostEstimate.from_components(idx, "routed", t, [1.0, 0.0], [0.0, 0.0], [0.1, -0.5])


# ----------------------------------------------------------------- #5 exact persistence
def _snap(adv_usd=5e7, sigma=0.02, price=50.0, share=0.09) -> LiquiditySnapshot:
    idx = pd.Index(["X"])
    return LiquiditySnapshot(
        date=pd.Timestamp("2026-11-03"),
        tickers=idx,
        price=pd.Series(price, index=idx),
        adv_usd=pd.Series(adv_usd, index=idx),
        adv_shares=pd.Series(adv_usd / price, index=idx),
        sigma_daily=pd.Series(sigma, index=idx),
        half_spread=pd.Series(2e-4, index=idx),
        borrow_fee=pd.Series(0.005, index=idx),
        auction_share=share,
    )


@pytest.mark.parametrize("exponent", [1.5, 1.8])
@pytest.mark.parametrize("path", ["halving", "growing", "equal"])
def test_streak_increment_is_exact_for_any_path(path, exponent):
    tcm = TransactionCostModel(CostConfig(impact_exponent=exponent))
    snap = _snap()
    A, sig, e = 5e7, 0.02, exponent
    qs = {"halving": [8e5, 4e5, 2e5, 1e5, 5e4], "growing": [1e5, 2e5, 4e5, 8e5, 1.6e6], "equal": [3e5] * 5}[
        path
    ]

    def C(Q: float) -> float:  # continuous: b_c = 1
        return sig * Q**e / A ** (e - 1)

    Q_prev, approx, exact = 0.0, [], []
    for k, q in enumerate(qs, start=1):
        x = pd.Series({"X": q})
        est = tcm.estimate(x, snap, venue="continuous", prior_streak_usd=pd.Series({"X": Q_prev}))
        exact.append(C(Q_prev + q) - C(Q_prev))
        assert est.frame["impact_usd"].iloc[0] == pytest.approx(exact[-1], rel=1e-12)
        approx.append(
            tcm.estimate(x, snap, venue="continuous", streak=pd.Series({"X": float(k)}))
            .frame["impact_usd"]
            .iloc[0]
        )
        Q_prev += q
    assert sum(exact) == pytest.approx(C(sum(qs)), rel=1e-12)  # telescoping: the whole metaorder
    ratio = sum(approx) / sum(exact)
    if path == "equal":
        assert ratio == pytest.approx(1.0, rel=1e-12)  # the old multiplier is exact only here
    elif path == "halving":
        assert ratio < 0.9  # the equal-days multiplier understated a converging build
    else:
        assert ratio > 1.1


def test_streak_increment_routed_and_auction():
    """Routed: [C_c(Q_k) - C_c(Q_{k-1})] + [C_a(q_a) - C_c(q_a)]; auction at (phi A, b_a)."""
    cfg = CostConfig(impact_b_auction=0.4)  # beta_a != b_c so the auction discount is visible
    tcm = TransactionCostModel(cfg)
    snap = _snap()
    A, sig, e, phi = 5e7, 0.02, 1.5, 0.09
    K = 0.02 * phi * A

    def C(Q, V, b):
        return b * sig * Q**e / V ** (e - 1)

    q, Qp = 5e5, 1.2e6
    qa = min(q, K)
    est = tcm.estimate(pd.Series({"X": q}), snap, venue="routed", prior_streak_usd=pd.Series({"X": Qp}))
    exp = (C(Qp + q, A, 1.0) - C(Qp, A, 1.0)) + (C(qa, phi * A, 0.4) - C(qa, A, 1.0))
    assert est.frame["impact_usd"].iloc[0] == pytest.approx(exp, rel=1e-12)
    est_a = tcm.estimate(pd.Series({"X": -q}), snap, venue="auction", prior_streak_usd=pd.Series({"X": Qp}))
    assert est_a.frame["impact_usd"].iloc[0] == pytest.approx(C(Qp + q, phi * A, 0.4) - C(Qp, phi * A, 0.4))
    # first day of a streak (Q_prev = 0) is the one-day cost; persistence off ignores Q_prev
    one = tcm.estimate(pd.Series({"X": q}), snap, venue="routed").frame["impact_usd"].iloc[0]
    assert tcm.estimate(
        pd.Series({"X": q}), snap, venue="routed", prior_streak_usd=pd.Series({"X": 0.0})
    ).frame["impact_usd"].iloc[0] == pytest.approx(one, rel=1e-15)
    off = TransactionCostModel(CostConfig(impact_b_auction=0.4, impact_persistence=False))
    assert off.estimate(
        pd.Series({"X": q}), snap, venue="routed", prior_streak_usd=pd.Series({"X": Qp})
    ).frame["impact_usd"].iloc[0] == pytest.approx(one, rel=1e-15)


def _md_long(dates: pd.DatetimeIndex, seed: int = 0) -> MarketData:
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0, 0.015, size=(len(dates), len(TICKERS)))
    close = pd.DataFrame(50.0 * np.cumprod(1.0 + r, axis=0), index=dates, columns=TICKERS)
    vol = pd.DataFrame(rng.uniform(5e5, 1.5e6, size=close.shape), index=dates, columns=TICKERS)
    md = MarketData(
        close=close, volume=vol, market_cap=close * 1e8, industry=pd.Series(["X", "Y", "X"], index=TICKERS)
    )
    md.validate()
    return md


@pytest.mark.parametrize(
    "weights", [[0.08, 0.12, 0.14, 0.15], [0.01, 0.03, 0.07, 0.15]], ids=["halving", "growing"]
)
def test_simulator_books_exact_streak_increment(weights):
    """Through ``DailyBarSimulator`` with the real cost model: A is bought on four
    consecutive days (sizes from the drifted book); the booked impact on day k is
    ``C(Q_k) - C(Q_{k-1})`` with ``Q`` the cumulative USD of the streak and ``C``
    priced on day k's snapshot (continuous venue)."""
    dates = pd.bdate_range("2026-01-05", periods=80)
    md = _md_long(dates)
    run_dates = dates[-5:]
    targets = {d: {"A": w, "B": 0.0, "C": 0.0} for d, w in zip(run_dates[:4], weights, strict=True)}
    cfg = SystemConfig().with_overrides({"execution.signal_lag_days": 0, "portfolio.rebalance_every_days": 1})
    tcm = TransactionCostModel(cfg.cost)
    res = DailyBarSimulator(cfg, tcm).run(
        md, run_dates, alpha_fn, risk_fn, PresetOptimizer(targets), nav0=NAV0, venue="continuous"
    )
    xa = res.trades["A"].to_numpy() * res.nav.to_numpy()[:-1]  # USD traded in A
    assert (xa > 0).all()
    e = cfg.cost.impact_exponent
    Q_prev = 0.0
    for k, t in enumerate(run_dates[:4]):
        s = tcm.snapshot(md, t)
        A_, sig = float(s.adv_usd["A"]), float(s.sigma_daily["A"])

        def C(Q, A_=A_, sig=sig):
            return sig * Q**e / A_ ** (e - 1)

        assert res.costs["impact"].iloc[k] == pytest.approx(C(Q_prev + xa[k]) - C(Q_prev), rel=1e-10)
        Q_prev += xa[k]


# ----------------------------------------------------------------- #14 auction share under lag
def test_optimizer_snapshot_uses_execution_date_auction_share():
    """Lag 1 across the 2026-03-20 quad witching: the optimizer's snapshot has the
    signal date's liquidity but the execution date's auction share."""
    dates = pd.bdate_range("2026-01-02", "2026-03-25")
    md = _md_long(dates, seed=1)
    run_dates = dates[-6:]  # 2026-03-18 .. 2026-03-25
    targets = {d: {"A": 0.05, "B": -0.05, "C": 0.0} for d in dates}
    cfg = SystemConfig().with_overrides(
        {"portfolio.rebalance_every_days": 1}
    )  # lag 1 (default), daily solves
    tcm = TransactionCostModel(cfg.cost)
    opt = PresetOptimizer(targets)
    DailyBarSimulator(cfg, tcm).run(md, run_dates, alpha_fn, risk_fn, opt, nav0=NAV0)
    assert len(opt.calls) == len(run_dates) - 1
    for t, call in zip(run_dates[:-1], opt.calls, strict=True):
        snap = call["snapshot"]
        sig_date = call["risk"].date
        assert sig_date < t and snap.date == sig_date
        assert snap.auction_share == auction_share_for_date(t, cfg.cost.auction_volume_share_default)
        ref = tcm.snapshot(md, sig_date)
        pd.testing.assert_series_equal(snap.adv_usd, ref.adv_usd)
    by_t = {t: c["snapshot"].auction_share for t, c in zip(run_dates[:-1], opt.calls, strict=True)}
    assert by_t[pd.Timestamp("2026-03-20")] == pytest.approx(0.20)
    assert by_t[pd.Timestamp("2026-03-19")] == pytest.approx(0.09)
    # the signal date of the quad-witching decision is the normal day before
    assert auction_share_for_date(pd.Timestamp("2026-03-19")) == pytest.approx(0.09)


# ----------------------------------------------------------------- #15 holdings signal date
def test_holdings_alpha_is_labelled_with_signal_date():
    from qts.runs import artifacts as AR

    W = pd.DataFrame(
        {"A": [0.1, 0.1, 0.2], "B": [-0.1, -0.1, -0.2]}, index=pd.bdate_range("2026-01-29", periods=3)
    )
    log_ = [
        {"date": W.index[0], "signal_date": pd.Timestamp("2026-01-28")},
        {"date": W.index[2], "signal_date": W.index[1]},
    ]
    result = SimpleNamespace(backtest=SimpleNamespace(weights=W, diagnostics={"solve_log": log_}))
    sd = AR.signal_dates(result)  # type: ignore[arg-type]
    assert list(sd) == [pd.Timestamp("2026-01-28"), pd.Timestamp("2026-01-28"), W.index[1]]
    seen: list[pd.Timestamp] = []

    class Alpha:
        def as_of(self, d):
            seen.append(pd.Timestamp(d))
            return SimpleNamespace(expected_return=pd.Series({"A": 0.01, "B": -0.01}))

    pipe = SimpleNamespace(cfg=SystemConfig(), alpha=Alpha(), risk=None, md_u=None)
    h = AR.holdings(result, pipe)  # type: ignore[arg-type]
    # decision 2026-01-30 (month end, a non-rebalance day): alpha as of the last solve's 2026-01-28;
    # decision 2026-02-02 (last row): alpha as of its own signal date 2026-01-30
    by = h.groupby("date")["signal_date"].first()
    assert by.to_dict() == {
        pd.Timestamp("2026-01-30"): pd.Timestamp("2026-01-28"),
        pd.Timestamp("2026-02-02"): pd.Timestamp("2026-01-30"),
    }
    assert seen == [pd.Timestamp("2026-01-28"), pd.Timestamp("2026-01-30")]
    assert list(h.columns) == ["date", "signal_date", "ticker", "weight", "alpha", "specific_vol", "industry"]


def test_sizing_summary_from_solve_log():
    """``vol_target_caps_share`` / ``mean_risk_aversion`` come from the solve log;
    NaN when no entry carries them (fixed gamma, or optimizers without the fields)."""
    from qts.backtest.simulator import BacktestResult

    idx = pd.bdate_range("2024-01-01", periods=3)
    empty = pd.DataFrame(index=idx)
    base = {
        "nav": pd.Series(1.0, index=idx),
        "returns": pd.DataFrame({"gross": 0.0, "net": 0.0}, index=idx),
        "weights": empty,
        "trades": empty,
        "costs": pd.DataFrame(0.0, index=idx, columns=["spread", "impact", "fees", "borrow", "total"]),
    }
    log_ = [
        {"vol_target_binding": "caps", "risk_aversion": 1.0},
        {"vol_target_binding": "target", "risk_aversion": 3.0},
        {"vol_target_binding": "caps", "risk_aversion": float("nan")},
        {"vol_target_binding": "", "risk_aversion": 2.0},  # not a target-vol solve
    ]
    s = BacktestResult(**base, diagnostics={"solve_log": log_}).sizing_summary()
    assert s["vol_target_caps_share"] == pytest.approx(2 / 3)
    assert s["mean_risk_aversion"] == pytest.approx(2.0)
    s0 = BacktestResult(**base, diagnostics={"solve_log": [{"status": "optimal"}]}).sizing_summary()
    assert np.isnan(s0["vol_target_caps_share"]) and np.isnan(s0["mean_risk_aversion"])
