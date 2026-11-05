"""Optimizer <-> simulator cost equivalence (finding #24, design decision D3).

(a) ``continuous`` / ``auction``: the optimizer's convex trade cost
    ``cvx_terms(x, w_post, nav, snap, venue)[0]`` evaluated at a trade equals
    ``estimate(x * nav, snap, venue).total / nav`` to 1e-10 (relative), holding
    cost excluded, for trades where neither the $0.35 per-order commission
    minimum nor the TAF per-trade cap binds (the only non-convex fee pieces,
    deliberately left out of the optimizer).
(b) ``routed``: the routed estimate is <= the continuous estimate for trades
    small enough to go entirely to the auction, and the optimizer's convex
    surrogate (``qts.tcost.model`` docstring) stays within a measured relative
    error of the exact routed estimate -- printed and bounded below.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qts.core.config import CostConfig
from qts.core.costs import LiquiditySnapshot
from qts.tcost import TransactionCostModel

cp = pytest.importorskip("cvxpy")

NAV = 1e7
DATE = pd.Timestamp("2026-06-15")  # SEC and TAF both non-zero (no fee holiday)


def _snapshot(n: int, seed: int, auction_share: float = 0.09) -> LiquiditySnapshot:
    rng = np.random.default_rng(seed)
    idx = pd.Index([f"T{i:03d}" for i in range(n)])
    return LiquiditySnapshot(
        date=DATE,
        tickers=idx,
        price=rng.uniform(5, 300, n),
        adv_usd=np.exp(rng.uniform(np.log(5e6), np.log(5e8), n)),
        adv_shares=None,
        sigma_daily=rng.uniform(0.008, 0.045, n),
        half_spread=rng.uniform(0.5e-4, 15e-4, n),
        borrow_fee=rng.uniform(0.003, 0.2, n),
        auction_share=auction_share,
    )


def _trades_no_min_no_cap(snap: LiquiditySnapshot, seed: int) -> pd.Series:
    """Signed USD trades of 150..45,000 shares (commission > $0.35 minimum; TAF
    below its 50,205-share cap), plus a few exact zeros."""
    rng = np.random.default_rng(seed)
    n = len(snap)
    shares = rng.uniform(150, 45_000, n) * np.where(rng.random(n) < 0.5, -1.0, 1.0)
    shares[rng.random(n) < 0.1] = 0.0
    return pd.Series(shares * snap.price.to_numpy(), index=snap.tickers)


def _cvx_value(tcm, trades_usd: pd.Series, snap, venue: str, include_explicit: bool = True) -> float:
    x = cp.Variable(len(trades_usd))
    x.value = trades_usd.to_numpy() / NAV
    trade, _ = tcm.cvx_terms(x, x, NAV, snap, venue, include_explicit=include_explicit)
    return float(trade.value)


@pytest.mark.parametrize("venue", ["continuous", "auction"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cvx_terms_equal_estimate(venue, seed):
    tcm = TransactionCostModel(CostConfig())
    snap = _snapshot(120, seed)
    trades = _trades_no_min_no_cap(snap, seed + 100)
    est = tcm.estimate(trades, snap, venue)
    exact = float(est["total_usd"].sum()) / NAV
    assert _cvx_value(tcm, trades, snap, venue) == pytest.approx(exact, rel=1e-10, abs=0)
    # spread + impact alone (no fees) also match, and per name
    no_fee = float((est["spread_usd"] + est["impact_usd"]).sum()) / NAV
    assert _cvx_value(tcm, trades, snap, venue, include_explicit=False) == pytest.approx(
        no_fee, rel=1e-10, abs=0
    )
    # the optimizer passes an old dict snapshot today: same numbers
    assert _cvx_value(tcm, trades, snap.to_dict(), venue) == pytest.approx(exact, rel=1e-10, abs=0)
    for j in np.flatnonzero(trades.to_numpy())[:5]:
        one = trades.iloc[[j]]
        one_snap = snap.reindex(one.index)
        assert _cvx_value(tcm, one, one_snap, venue) == pytest.approx(
            float(tcm.estimate(one, one_snap, venue)["total_usd"].iloc[0]) / NAV, rel=1e-10, abs=0
        )


def test_cvx_terms_equal_estimate_non_default_exponent_and_heavy_day():
    tcm = TransactionCostModel(CostConfig(impact_exponent=1.7, impact_b_auction=0.45))
    snap = _snapshot(60, 7, auction_share=0.34)
    trades = _trades_no_min_no_cap(snap, 8)
    for venue in ("continuous", "auction"):
        exact = float(tcm.estimate(trades, snap, venue)["total_usd"].sum()) / NAV
        assert _cvx_value(tcm, trades, snap, venue) == pytest.approx(exact, rel=1e-10, abs=0)


def _routed_errors(tcm, snap, trades):
    est = tcm.estimate(trades, snap, "routed")
    errs = []
    for j in np.flatnonzero(trades.to_numpy()):
        one = trades.iloc[[j]]
        s1 = snap.reindex(one.index)
        exact = float(est["total_usd"].iloc[j])
        approx = _cvx_value(tcm, one, s1, "routed") * NAV
        errs.append((approx - exact) / exact)
    return np.array(errs), est


def test_routed_small_trades_cheaper_and_surrogate_error_bounded():
    tcm = TransactionCostModel(CostConfig())
    snap = _snapshot(300, 11)
    rng = np.random.default_rng(12)
    n = len(snap)
    # (b1) trades inside the auction cap (<= 2% of auction volume): routed <= continuous
    cap = tcm.auction_cap_usd(snap)
    min_usd = 150 * snap.price.to_numpy()  # >= 150 shares: one commission minimum at most
    small = pd.Series(
        np.maximum(rng.uniform(0.05, 1.0, n) * cap, min_usd) * np.sign(rng.normal(size=n)), index=snap.tickers
    )
    inside = np.abs(small.to_numpy()) <= cap
    r = tcm.estimate(small, snap, "routed")["total_usd"].to_numpy()
    c = tcm.estimate(small, snap, "continuous")["total_usd"].to_numpy()
    assert inside.sum() > 0.8 * n
    assert (r[inside] <= c[inside] + 1e-9).all()
    assert (r <= c + 1e-9).mean() > 0.95

    # (b2) surrogate error over trades from 0.001% to 5% of ADV (both legs >= ~150 shares)
    part = np.exp(rng.uniform(np.log(1e-5), np.log(0.05), n))
    q = np.maximum(part * snap.adv_usd.to_numpy(), 2 * min_usd + cap * (part > 0.0018))
    trades = pd.Series(q * np.sign(rng.normal(size=n)), index=snap.tickers)
    errs, est = _routed_errors(tcm, snap, trades)
    split = est.auction_fraction.to_numpy()[np.flatnonzero(trades.to_numpy())] < 1.0
    print(
        f"\n[routed surrogate, D2 defaults, normal day] relative error (cvx*nav - estimate)/estimate over "
        f"{len(errs)} trades: max {errs.max():+.3%}, min {errs.min():+.3%}, "
        f"max |err| on split trades {np.abs(errs[split]).max():.3%}, on auction-only {np.abs(errs[~split]).max():.2e}"
    )
    # auction-only trades: the surrogate is exact (spread kink inactive, impact equal at D2)
    assert np.abs(errs[~split]).max() < 1e-9
    # split trades: only fee approximations differ (auction fee on the continuous remainder,
    # the second order's commission minimum) -> a few percent at most
    assert np.abs(errs).max() < 0.05

    # heavy day (34% auction share, beta_a = 0.51 < b_c): impact still exact via the kink term
    heavy = _snapshot(300, 11, auction_share=0.34)
    errs_h, _ = _routed_errors(tcm, heavy, trades)
    print(f"[routed surrogate, Russell day] max {errs_h.max():+.3%}, min {errs_h.min():+.3%}")
    assert np.abs(errs_h).max() < 0.05
    # beta_a > b_c (auction prior raised): convex fallback prices the whole trade at beta_a -> upper bound
    hi = TransactionCostModel(CostConfig(impact_b_auction=0.6))
    errs_hi, _ = _routed_errors(hi, snap, trades)
    print(
        f"[routed surrogate, b_auction=0.6 (beta_a=2.0 > b_c)] max {errs_hi.max():+.3%}, min {errs_hi.min():+.3%}"
    )
    assert errs_hi.min() > -0.02


@pytest.mark.parametrize("share", [0.09, 0.20, 0.34])
def test_routed_spread_and_impact_exact_when_beta_a_le_b_c(share):
    """With b_a/phi^(e-1) <= b_c (D2 default; heavy days) the routed spread and
    impact terms of the surrogate are exact -- only the linear fees approximate."""
    tcm = TransactionCostModel(CostConfig())
    snap = _snapshot(80, 21, auction_share=share)
    rng = np.random.default_rng(22)
    q = np.exp(rng.uniform(np.log(1e-5), np.log(0.05), 80)) * snap.adv_usd.to_numpy()
    trades = pd.Series(q * np.sign(rng.normal(size=80)), index=snap.tickers)
    est = tcm.estimate(trades, snap, "routed")
    assert 0 < (est.auction_fraction < 1).sum() < 80  # both regimes present
    exact = float((est["spread_usd"] + est["impact_usd"]).sum()) / NAV
    assert _cvx_value(tcm, trades, snap, "routed", include_explicit=False) == pytest.approx(
        exact, rel=1e-10, abs=0
    )


def test_routed_surrogate_is_convex_and_solves():
    tcm = TransactionCostModel(CostConfig())
    snap = _snapshot(50, 3)
    x = cp.Variable(50)
    trade, hold = tcm.cvx_terms(x, x, NAV, snap, "routed")
    assert trade.is_dcp() and trade.is_convex() and hold.is_convex()
    alpha = np.linspace(-0.005, 0.005, 50)
    prob = cp.Problem(cp.Minimize(trade + hold - alpha @ x), [cp.norm1(x) <= 1.0])
    prob.solve(solver="CLARABEL")
    assert prob.status in ("optimal", "optimal_inaccurate")
