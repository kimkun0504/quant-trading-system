"""D4 execution delay through the *real* stack: ``DailyBarSimulator`` with the
real :class:`SinglePeriodOptimizer` (cvxpy/Clarabel) and the real
:class:`TransactionCostModel` -- not the optimizer/cost doubles of
``test_backtest.test_signal_lag_shifts_decisions_by_one_day``.

Lag investigation 2026-11-05 (docs/notes/lag_investigation_2026-11-05.md):

1. With the optimizer's cost aversions at 0 (costs still booked) and slack ADV
   caps, the optimum does not depend on the current book, so the lag-1 weights decided for execution at ``t`` must equal
   the lag-0 weights decided at ``t-1`` (solver tolerance).
2. With real costs, every input the optimizer sees at execution date ``t`` is
   dated ``t-1`` (alpha, risk, cost snapshot) except the book and NAV, which
   are the drifted ``w_pre`` and ``nav`` marked at ``t``; costs are booked on
   the execution-date snapshot and positions earn ``r_{t+1}``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qts.backtest import DailyBarSimulator
from qts.core.config import SystemConfig
from qts.core.types import AlphaOutput, RiskModelOutput
from qts.data.synthetic import SyntheticSpec, generate
from qts.portfolio import SinglePeriodOptimizer
from qts.tcost import TransactionCostModel

N_DAYS_RUN = 6


@pytest.fixture(scope="module")
def md():
    return generate(SyntheticSpec(n_stocks=30, n_days=140, seed=3))


def _alpha_fn(md):
    z = md.extra["true_latent_score"]
    vol = md.returns.rolling(60, min_periods=40).std() * np.sqrt(20)

    def alpha_fn(d):
        s = z.loc[d]
        s = (s - s.mean()) / s.std(ddof=0)
        er = (0.05 * vol.loc[d] * s).fillna(0.0)
        return AlphaOutput(
            date=d,
            score=s.rename("score"),
            expected_return=er.rename("expected_return"),
            horizon_days=20,
            half_life_days=65.0,
        )

    return alpha_fn


def _risk_fn(md):
    """PIT one-factor model + a time-varying 'beta' style (trailing 20d return
    z-score, so a one-day shift visibly changes the exposures)."""
    R = md.returns
    sty = R.rolling(20, min_periods=15).sum()
    sty = sty.sub(sty.mean(axis=1), axis=0).div(sty.std(axis=1), axis=0).fillna(0.0)
    spec = R.rolling(60, min_periods=40).var()  # trailing, PIT

    def risk_fn(d, alpha=None):
        B = pd.DataFrame({"market": 1.0, "beta": sty.loc[d]}, index=md.tickers)
        F = pd.DataFrame([[1e-4, 0.0], [0.0, 2e-5]], index=B.columns, columns=B.columns)
        D = spec.loc[:d].iloc[-1].fillna(4e-4).clip(lower=1e-6)
        return RiskModelOutput(
            date=pd.Timestamp(d),
            exposures=B,
            factor_cov=F,
            specific_var=D,
            factor_returns=pd.DataFrame(columns=B.columns, dtype=float),
            factor_groups={"style": ["beta"], "industry": []},
        )

    return risk_fn


class Recorder:
    """Real optimizer wrapped to record the keyword arguments of every solve."""

    def __init__(self, opt):
        self.opt = opt
        self.horizon_days = opt.horizon_days
        self.calls: list[dict] = []

    def solve(self, **kw):
        self.calls.append(kw)
        return self.opt.solve(**kw)


def test_real_stack_lag1_weights_equal_lag0_weights_one_day_earlier(md):
    # costs are still booked by the real TransactionCostModel, but the optimizer
    # ignores them, so its optimum does not depend on the current book
    cfg = SystemConfig().with_overrides(
        {
            "portfolio.trade_aversion": 0.0,
            "portfolio.hold_aversion": 0.0,
            "portfolio.vol_target_projection_cost_weight": 0.0,
            # mechanics: a solve every day, nothing filtered after it
            "portfolio.rebalance_every_days": 1,
            "portfolio.min_trade_weight": 0.0,
        }
    )
    tc = TransactionCostModel(cfg.cost, cfg.execution)
    md0 = md
    dates = md.dates[-(N_DAYS_RUN + 1) :]
    a_fn, r_fn = _alpha_fn(md), _risk_fn(md)
    sim = DailyBarSimulator(cfg, tc)
    run = lambda lag: sim.run(  # noqa: E731
        md0, dates, a_fn, r_fn, SinglePeriodOptimizer(cfg.portfolio, tc), nav0=1e5, signal_lag_days=lag
    )
    res0, res1 = run(0), run(1)
    assert res1.diagnostics["n_failed_solves"] == 0 and res0.diagnostics["n_failed_solves"] == 0
    assert res1.weights.abs().sum(axis=1).min() > 0.5  # a real book, not a degenerate one
    # lag-1 decision for execution at dates[i] == lag-0 decision at dates[i-1]
    np.testing.assert_allclose(res1.weights.iloc[1:].to_numpy(), res0.weights.iloc[:-1].to_numpy(), atol=2e-5)
    # and the lag-1 book is NOT the lag-0 book of the same day (the shift is real)
    assert np.abs(res1.weights.iloc[2:].to_numpy() - res0.weights.iloc[2:].to_numpy()).max() > 1e-3
    # gross return identity: the lag-1 book held at t earns r_{t+1}
    R = md0.close.loc[dates].pct_change(fill_method=None)
    np.testing.assert_allclose(
        res1.returns["gross"].to_numpy(), (res1.weights.to_numpy() * R.to_numpy()[1:]).sum(axis=1), atol=1e-12
    )


def test_real_stack_lag1_inputs_are_dated_signal_date_and_book_is_current(md):
    cfg = SystemConfig().with_overrides({"portfolio.rebalance_every_days": 1})
    tc = TransactionCostModel(cfg.cost, cfg.execution)
    dates = md.dates[-(N_DAYS_RUN + 1) :]
    a_fn, r_fn = _alpha_fn(md), _risk_fn(md)
    rec = Recorder(SinglePeriodOptimizer(cfg.portfolio, tc))
    res = DailyBarSimulator(cfg, tc).run(md, dates, a_fn, r_fn, rec, nav0=1e6, signal_lag_days=1)
    assert len(rec.calls) == N_DAYS_RUN
    pos = md.dates.get_indexer(dates)
    R = md.close.pct_change(fill_method=None)
    for i, call in enumerate(rec.calls):
        t, s = dates[i], md.dates[pos[i] - 1]
        assert call["risk"].date == s
        assert pd.Timestamp(call["snapshot"].date) == s  # optimizer cost terms on the signal date
        exp = a_fn(s).expected_return.reindex(md.tickers).fillna(0.0)
        pd.testing.assert_series_equal(call["alpha"], exp, check_names=False, atol=0, rtol=0)
        # book and NAV are marked at the execution date t
        assert call["nav"] == pytest.approx(res.nav.loc[t], rel=1e-12)
        if i == 0:
            np.testing.assert_allclose(call["w_current"].to_numpy(), 0.0)
        else:
            w_prev, nav_prev = res.weights.iloc[i - 1], res.nav.iloc[i - 1]
            drift = w_prev * (1.0 + R.loc[t].reindex(md.tickers).fillna(0.0)) * nav_prev / res.nav.loc[t]
            np.testing.assert_allclose(call["w_current"].to_numpy(), drift.to_numpy(), atol=1e-12)
    # costs are booked on the execution-date snapshot (actual liquidity at t), not s
    for i in (1, 3):
        t = dates[i]
        snap_t = tc.snapshot(md, t)
        x_usd = res.trades.iloc[i] * res.nav.loc[t]
        est = tc.estimate(x_usd, snap_t, venue="routed").frame
        expected_trade = float(est[["spread_usd", "fees_usd"]].to_numpy().sum())
        got = float(res.costs.iloc[i][["spread", "fees"]].sum())
        assert got == pytest.approx(expected_trade, rel=1e-9, abs=1e-9)
    # NAV identity on the lagged path
    Rr = md.close.loc[dates].pct_change(fill_method=None).to_numpy()[1:]
    lhs = res.nav.to_numpy()[1:]
    rhs = (
        res.nav.to_numpy()[:-1] * (1 + (res.weights.to_numpy() * Rr).sum(axis=1))
        - res.costs["total"].to_numpy()
    )
    np.testing.assert_allclose(lhs, rhs, rtol=1e-12)
