"""Hand-computed 3-stock x 5-day fixture for ``DailyBarSimulator``.

Every expected number below is derived by hand (the arithmetic is spelled out
in the comments and in literal expressions) -- *not* by re-running a copy of
the simulator loop -- so the tests pin the simulator's accounting identity,
drift rule, cost booking, borrow and delisting semantics exactly (1e-10).

Simulator semantics pinned here (``qts/backtest/simulator.py``, read 2026-11-04):

* decision row ``t = dates[i]``; trades ``x = w_post - w_pre`` execute at the
  close of ``t``; costs are ``estimate(x * nav_t)`` in USD plus one day of
  borrow ``max(-w_post, 0) * nav_t * borrow_fee / 252`` (snapshot fallback
  path, used when the cost model has no ``holding_cost_usd``);
* ``nav_{t+1} = nav_t * (1 + w_post . r_{t+1}) - costs_t``; ``gross = w_post . r``,
  ``net = nav_{t+1} / nav_t - 1``; ``turnover_t = sum |x|``;
* drift: ``w_pre_{t+1} = w_post * (1 + r_{t+1}) * nav_t / nav_{t+1}``;
* a held name with NaN close on ``t+1`` is a *delisting* when
  ``md.delisting_dates`` says its last trading date is before ``t+1``, or (name
  not covered) when ``t+1`` is ``cfg.alpha.delisting_gap_days`` trading days
  after its last close; otherwise it is *halted*.
* delisting: its return on ``t+1`` is
  ``cfg.cost.delisting_haircut * sign(w)`` (default -0.30: a loss for longs
  *and* shorts), the residual value ``x_liq = -w * (1 + haircut_ret)`` is
  booked as an extra trade on row ``t`` (counted in trades / turnover), priced
  by ``estimate(x_liq * nav_t, venue="continuous")`` with the **impact column
  zeroed** (spread + fees only), and the name's drifted weight is set to 0.
* halt: the position is frozen at its last price (return 0 per missing day,
  weight carried); on resumption the return is measured from the last close
  before the halt.
* A name with no price on ``t`` goes on ``no_trade`` and its ``w_post`` is
  forced to ``w_pre`` whatever the optimizer returns.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from qts.backtest import DailyBarSimulator
from qts.core.config import SystemConfig
from qts.core.types import AlphaOutput, MarketData, RiskModelOutput

TICKERS = pd.Index(["A", "B", "C"], name="ticker")
DATES = pd.bdate_range("2024-01-01", periods=5)  # d0..d4 (Mon..Fri)
NAV0 = 1_000_000.0

# Closes chosen so every daily return is a round number:
#   A: 100 -> 110 -> 99  -> 99   -> 108.9   r = +10%, -10%,   0%, +10%
#   B:  50 ->  50 -> 55  -> 49.5 ->  49.5   r =   0%, +10%, -10%,   0%
#   C:  20 ->  21 -> 21  -> 23.1 ->  23.1   r =  +5%,   0%, +10%,   0%
CLOSE = {
    "A": [100.0, 110.0, 99.0, 99.0, 108.9],
    "B": [50.0, 50.0, 55.0, 49.5, 49.5],
    "C": [20.0, 21.0, 21.0, 23.1, 23.1],
}
R1, R2, R3, R4 = (
    np.array([0.10, 0.00, 0.05]),
    np.array([-0.10, 0.10, 0.00]),
    np.array([0.00, -0.10, 0.10]),
    np.array([0.10, 0.00, 0.00]),
)

ATOL = 1e-10  # weights, returns, turnover (fractions of NAV)
NAV_RTOL = 1e-12  # NAV / USD (1e6 scale: 1e-12 rel = 1e-6 USD)


def _lag0_config() -> SystemConfig:
    cfg = SystemConfig().with_overrides({"execution.signal_lag_days": 0, "portfolio.rebalance_every_days": 1})
    return cfg


def make_md(close: dict | None = None, delisting_dates: dict | None = None) -> MarketData:
    close_df = pd.DataFrame(close or CLOSE, index=DATES)[list(TICKERS)]
    close_df.columns = TICKERS
    vol = pd.DataFrame(1e6, index=DATES, columns=TICKERS)
    cap = close_df * 1e8
    dd = None
    if delisting_dates is not None:  # vendor table covering every ticker; NaT = alive
        dd = pd.Series(pd.NaT, index=TICKERS, dtype="datetime64[ns]")
        for k, v in delisting_dates.items():
            dd[k] = v
    md = MarketData(
        close=close_df,
        volume=vol,
        market_cap=cap,
        industry=pd.Series(["X", "Y", "X"], index=TICKERS),
        delisting_dates=dd,
    )
    md.validate()
    return md


class PresetOptimizer:
    """Optimizer double: returns preset target weights per decision date and
    records every call's keyword arguments."""

    horizon_days = 20

    def __init__(self, targets: dict):
        self.targets = targets
        self.calls: list[dict] = []

    def solve(self, **kw):
        self.calls.append(kw)
        t = kw["risk"].date
        w = pd.Series(self.targets[t], index=TICKERS, dtype=float)
        return SimpleNamespace(
            w_target=w,
            x=w - kw["w_current"],
            status="optimal",
            solve_time=0.0,
            transfer_coefficient=1.0,
            binding_constraints={},
            vol_scalar=1.0,
            expected_alpha=0.0,
            expected_risk=0.0,
            expected_cost=0.0,
            notes=[],
            n_out_of_model=0,
        )


class ConstantBpsCost:
    """Cost double: ``spread_bps`` / ``impact_bps`` / ``fees_bps`` of |USD traded|
    (zero everywhere when all are 0); borrow is left to the simulator's snapshot
    fallback (``borrow_fee`` annual, no ``holding_cost_usd`` method)."""

    def __init__(self, spread_bps=0.0, impact_bps=0.0, fees_bps=0.0, borrow_fee_annual=0.0):
        self.s, self.i, self.f = spread_bps / 1e4, impact_bps / 1e4, fees_bps / 1e4
        self.borrow = borrow_fee_annual
        self.venues: list[str] = []

    def snapshot_from_market(self, md, t):
        ones = pd.Series(1.0, index=md.tickers)
        return {
            "adv_usd": ones * 1e9,
            "sigma_daily": ones * 0.02,
            "half_spread": ones * self.s,
            "price": md.close.loc[:t].iloc[-1],
            "borrow_fee": ones * self.borrow,
        }

    def cvx_terms(self, *a, **k):  # pragma: no cover - the optimizer double never calls it
        raise NotImplementedError

    def estimate(self, trades_usd: pd.Series, snapshot, venue: str = "auction") -> pd.DataFrame:
        self.venues.append(venue)
        q = trades_usd.abs().astype(float)
        return pd.DataFrame({"spread": q * self.s, "impact": q * self.i, "fees": q * self.f})


def risk_fn(t, alpha=None) -> RiskModelOutput:
    # one zero-variance factor + specific variance 1e-4/day: ex-ante var = 1e-4 * sum w^2
    B = pd.DataFrame(0.0, index=TICKERS, columns=["mkt"])
    F = pd.DataFrame([[0.0]], index=["mkt"], columns=["mkt"])
    return RiskModelOutput(
        date=t,
        exposures=B,
        factor_cov=F,
        specific_var=pd.Series(1e-4, index=TICKERS),
        factor_returns=pd.DataFrame(columns=["mkt"]),
    )


def alpha_fn(t) -> AlphaOutput:
    z = pd.Series(0.0, index=TICKERS)
    return AlphaOutput(date=t, score=z, expected_return=z, horizon_days=20, half_life_days=60.0)


def run(cost, targets, md=None, gap_days: int | None = None):
    md = md or make_md()
    # mechanics test: trade at the signal close, solve every day
    cfg = SystemConfig().with_overrides({"execution.signal_lag_days": 0, "portfolio.rebalance_every_days": 1})
    if gap_days is not None:
        cfg = cfg.with_overrides({"alpha.delisting_gap_days": gap_days})
    opt = PresetOptimizer(targets)
    res = DailyBarSimulator(cfg, cost).run(md, DATES, alpha_fn, risk_fn, opt, nav0=NAV0)
    return res, opt, cfg


# targets used by the zero-cost and constant-bps cases
W_LS = [0.50, -0.25, 0.25]  # d0, d1
W_LO = [0.25, 0.00, 0.50]  # d2, d3
TARGETS = {DATES[0]: W_LS, DATES[1]: W_LS, DATES[2]: W_LO, DATES[3]: W_LO}


# ----------------------------------------------------------------------------- (a) zero cost
def test_zero_cost_nav_path_returns_turnover():
    res, opt, _ = run(ConstantBpsCost(), TARGETS)
    w_ls, w_lo = np.array(W_LS), np.array(W_LO)

    # d0: from cash. x0 = [.5, -.25, .25], turnover 1.0
    #     gross = .5*.10 + (-.25)*0 + .25*.05 = .0625 ; nav1 = 1e6 * 1.0625 = 1,062,500
    nav1 = NAV0 * (1 + 0.0625)
    # drift: w_post*(1+r1)*nav0/nav1 = [.55, -.25, .2625] / 1.0625
    w_pre1 = np.array([0.55, -0.25, 0.2625]) / 1.0625
    # d1: back to [.5,-.25,.25]; gross = .5*(-.1) + (-.25)*.1 + .25*0 = -.075
    #     nav2 = 1,062,500 * .925 = 982,812.5
    x1 = w_ls - w_pre1
    nav2 = nav1 * (1 - 0.075)
    w_pre2 = np.array([0.5 * 0.9, -0.25 * 1.1, 0.25 * 1.0]) / 0.925
    # d2: to [.25, 0, .5]; gross = .25*0 + 0*(-.1) + .5*.1 = .05 ; nav3 = 982,812.5*1.05 = 1,031,953.125
    x2 = w_lo - w_pre2
    nav3 = nav2 * 1.05
    w_pre3 = np.array([0.25 * 1.0, 0.0, 0.5 * 1.1]) / 1.05
    # d3: back to [.25, 0, .5]; gross = .25*.1 = .025 ; nav4 = 1,031,953.125*1.025 = 1,057,751.953125
    x3 = w_lo - w_pre3
    nav4 = nav3 * 1.025
    assert nav4 == pytest.approx(1_057_751.953125, rel=1e-15)

    np.testing.assert_allclose(res.nav.to_numpy(), [NAV0, nav1, nav2, nav3, nav4], rtol=NAV_RTOL, atol=0)
    gross = np.array([0.0625, -0.075, 0.05, 0.025])
    np.testing.assert_allclose(res.returns["gross"].to_numpy(), gross, atol=ATOL, rtol=0)
    np.testing.assert_allclose(res.returns["net"].to_numpy(), gross, atol=ATOL, rtol=0)  # no costs
    np.testing.assert_allclose(res.weights.to_numpy(), [w_ls, w_ls, w_lo, w_lo], atol=ATOL, rtol=0)
    np.testing.assert_allclose(res.trades.to_numpy(), [w_ls, x1, x2, x3], atol=ATOL, rtol=0)
    # turnover d1 = |.5-.517647| + |-.25+.235294| + |.25-.247059| = 2*.25*(1/1.0625 - ... ) etc.
    np.testing.assert_allclose(
        res.turnover.to_numpy(),
        [1.0, np.abs(x1).sum(), np.abs(x2).sum(), np.abs(x3).sum()],
        atol=ATOL,
        rtol=0,
    )
    # d1 turnover by hand: |.5 - .55/1.0625| + |-.25 + .25/1.0625| + |.25 - .2625/1.0625|
    #                    = .0176470588 + .0147058824 + .0029411765 = .0352941176 (= .0375/1.0625)
    assert res.turnover.iloc[1] == pytest.approx(0.0375 / 1.0625, abs=ATOL)
    assert (res.costs.to_numpy() == 0).all()
    # indices: decision dates d0..d3, realisation dates d1..d4
    assert list(res.weights.index) == list(DATES[:4]) and list(res.returns.index) == list(DATES[1:])
    # ex-ante variance from the stand-in risk: 1e-4 * sum w^2  (d0: 1e-4*(.25+.0625+.0625))
    np.testing.assert_allclose(
        res.exante_var.to_numpy(), [1e-4 * 0.375, 1e-4 * 0.375, 1e-4 * 0.3125, 1e-4 * 0.3125], atol=1e-16
    )
    # the optimizer saw the drifted book and the pre-trade NAV
    np.testing.assert_allclose(opt.calls[1]["w_current"].to_numpy(), w_pre1, atol=ATOL)
    assert opt.calls[2]["nav"] == pytest.approx(nav2, rel=NAV_RTOL)
    assert all(c["no_trade"] == [] for c in opt.calls)


# ----------------------------------------------------------------------------- (b) constant bps
SPREAD, IMPACT, FEES = 5.0, 2.0, 1.0  # bps of |USD traded|  -> 8 bps per side in total
BORROW = 0.0252  # annual -> 0.0252 / 252 = 1e-4 per day on short notional


def bps_model():
    return ConstantBpsCost(SPREAD, IMPACT, FEES, BORROW)


def test_constant_bps_costs_and_borrow():
    res, _, cfg = run(bps_model(), TARGETS)
    assert cfg.trading_days_per_year == 252
    w_ls, w_lo = np.array(W_LS), np.array(W_LO)
    k_trade, k_borrow = 8e-4, 1e-4

    # d0: x = [.5,-.25,.25]; traded 1.0 * 1e6 = 1e6 USD -> 800 USD trading cost
    #     (spread 500, impact 200, fees 100); borrow on the .25 short: .25*1e6*1e-4 = 25 USD
    c0 = k_trade * 1.0 * NAV0 + k_borrow * 0.25 * NAV0  # 825
    assert c0 == pytest.approx(825.0)
    nav1 = NAV0 * 1.0625 - c0  # 1,062,500 - 825 = 1,061,675
    # drift re-based to the post-cost NAV: [.55, -.25, .2625] * 1e6 / 1,061,675
    w_pre1 = np.array([0.55, -0.25, 0.2625]) * NAV0 / nav1
    x1 = w_ls - w_pre1
    c1 = k_trade * np.abs(x1).sum() * nav1 + k_borrow * 0.25 * nav1
    nav2 = nav1 * (1 - 0.075) - c1
    w_pre2 = np.array([0.45, -0.275, 0.25]) * nav1 / nav2
    x2 = w_lo - w_pre2  # covers the short: no borrow on d2
    c2 = k_trade * np.abs(x2).sum() * nav2
    nav3 = nav2 * 1.05 - c2
    w_pre3 = np.array([0.25, 0.0, 0.55]) * nav2 / nav3
    x3 = w_lo - w_pre3
    c3 = k_trade * np.abs(x3).sum() * nav3
    nav4 = nav3 * 1.025 - c3

    np.testing.assert_allclose(res.nav.to_numpy(), [NAV0, nav1, nav2, nav3, nav4], rtol=NAV_RTOL, atol=0)
    gross = np.array([0.0625, -0.075, 0.05, 0.025])  # unaffected by costs
    np.testing.assert_allclose(res.returns["gross"].to_numpy(), gross, atol=ATOL, rtol=0)
    net = np.array([nav1 / NAV0, nav2 / nav1, nav3 / nav2, nav4 / nav3]) - 1
    np.testing.assert_allclose(res.returns["net"].to_numpy(), net, atol=ATOL, rtol=0)
    # net = gross - cost/nav  (d0: .0625 - 825/1e6 = .061675)
    assert res.returns["net"].iloc[0] == pytest.approx(0.061675, abs=ATOL)
    np.testing.assert_allclose(res.trades.to_numpy(), [w_ls, x1, x2, x3], atol=ATOL, rtol=0)
    to = np.array([1.0, np.abs(x1).sum(), np.abs(x2).sum(), np.abs(x3).sum()])
    np.testing.assert_allclose(res.turnover.to_numpy(), to, atol=ATOL, rtol=0)

    navs = np.array([NAV0, nav1, nav2, nav3])
    C = res.costs
    np.testing.assert_allclose(C["spread"].to_numpy(), 5e-4 * to * navs, rtol=NAV_RTOL, atol=1e-9)
    np.testing.assert_allclose(C["impact"].to_numpy(), 2e-4 * to * navs, rtol=NAV_RTOL, atol=1e-9)
    np.testing.assert_allclose(C["fees"].to_numpy(), 1e-4 * to * navs, rtol=NAV_RTOL, atol=1e-9)
    np.testing.assert_allclose(
        C["borrow"].to_numpy(), [25.0, 0.25 * nav1 * 1e-4, 0.0, 0.0], rtol=NAV_RTOL, atol=1e-9
    )
    np.testing.assert_allclose(C["total"].to_numpy(), [c0, c1, c2, c3], rtol=NAV_RTOL, atol=1e-9)

    s = res.summary()
    # avg cost per side = (spread+impact+fees) / traded USD = 8 bps exactly (borrow excluded)
    assert s["avg_cost_bps_per_side"] == pytest.approx(8.0, abs=1e-9)
    assert s["total_costs_usd"] == pytest.approx(c0 + c1 + c2 + c3, rel=NAV_RTOL)
    assert s["avg_daily_turnover"] == pytest.approx(to.mean(), abs=ATOL)
    assert s["final_nav"] == pytest.approx(nav4, rel=NAV_RTOL)
    assert s["n_days"] == 4 and s["n_delistings"] == 0 and s["n_failed_solves"] == 0
    np.testing.assert_allclose(
        res.cost_bps_of_nav.to_numpy(), np.array([c0, c1, c2, c3]) / navs * 1e4, rtol=1e-12
    )


# ----------------------------------------------------------------------------- delisting
TARGETS_CONST = dict.fromkeys(DATES[:4], W_LS)


def test_delisting_long_haircut_and_liquidation_cost():
    """C (long .25) has no close on d3 and d4 (index 3 = 'day 4'); the vendor
    delisting table says its last trading date is d2, so the gap is a delisting
    decided on d3 (no waiting for the causal gap rule)."""
    close = {k: list(v) for k, v in CLOSE.items()}
    close["C"][3] = close["C"][4] = np.nan
    cost = bps_model()
    res, opt, cfg = run(cost, TARGETS_CONST, make_md(close, {"C": DATES[2]}))
    h = cfg.cost.delisting_haircut
    assert h == -0.30
    w = np.array(W_LS)

    # d0, d1 exactly as in the constant-bps case (same targets on those days)
    c0 = 8e-4 * NAV0 + 1e-4 * 0.25 * NAV0
    nav1 = NAV0 * 1.0625 - c0
    w_pre1 = np.array([0.55, -0.25, 0.2625]) * NAV0 / nav1
    x1 = w - w_pre1
    c1 = 8e-4 * np.abs(x1).sum() * nav1 + 1e-4 * 0.25 * nav1
    nav2 = nav1 * 0.925 - c1
    w_pre2 = np.array([0.45, -0.275, 0.25]) * nav1 / nav2
    # d2 -> d3: C delists. Its d3 return is haircut * sign(+.25) = -0.30
    #   gross = .5*0 + (-.25)*(-.10) + .25*(-.30) = .025 - .075 = -.05
    #   x_liq_C = -.25 * (1 - .30) = -.175  (sells the remaining 70% of the position)
    #   liquidation cost = (spread 5 + fees 1) bps * .175 * nav2  (impact column zeroed)
    x2 = w - w_pre2
    x_liq = np.array([0.0, 0.0, -0.175])
    c2_trade = 8e-4 * np.abs(x2).sum() * nav2
    c2_borrow = 1e-4 * 0.25 * nav2
    c2_liq = 6e-4 * 0.175 * nav2
    nav3 = nav2 * (1 - 0.05) - (c2_trade + c2_borrow + c2_liq)
    # C's drifted weight is zeroed; A and B drift normally
    w_pre3 = np.array([0.5 * 1.0, -0.25 * 0.9, 0.0]) * nav2 / nav3
    # d3: C has no price -> no_trade; the optimizer asks for .25 in C but w_post_C = w_pre_C = 0
    w3 = np.array([0.5, -0.25, 0.0])
    x3 = w3 - w_pre3
    c3 = 8e-4 * np.abs(x3).sum() * nav3 + 1e-4 * 0.25 * nav3
    nav4 = nav3 * (1 + 0.5 * 0.10) - c3  # d4: A +10%, B 0, C NaN but not held -> 0

    np.testing.assert_allclose(res.nav.to_numpy(), [NAV0, nav1, nav2, nav3, nav4], rtol=NAV_RTOL, atol=0)
    np.testing.assert_allclose(
        res.returns["gross"].to_numpy(), [0.0625, -0.075, -0.05, 0.05], atol=ATOL, rtol=0
    )
    np.testing.assert_allclose(res.weights.to_numpy(), [w, w, w, w3], atol=ATOL, rtol=0)
    # the liquidation is recorded as a trade on decision row d2 and counted in turnover
    np.testing.assert_allclose(res.trades.to_numpy(), [w, x1, x2 + x_liq, x3], atol=ATOL, rtol=0)
    assert res.turnover.iloc[2] == pytest.approx(np.abs(x2 + x_liq).sum(), abs=ATOL)
    C = res.costs.iloc[2]
    assert C["spread"] == pytest.approx(5e-4 * np.abs(x2).sum() * nav2 + 5e-4 * 0.175 * nav2, rel=NAV_RTOL)
    assert C["impact"] == pytest.approx(2e-4 * np.abs(x2).sum() * nav2, rel=NAV_RTOL)  # no liq impact
    assert C["fees"] == pytest.approx(1e-4 * np.abs(x2).sum() * nav2 + 1e-4 * 0.175 * nav2, rel=NAV_RTOL)
    assert C["total"] == pytest.approx(c2_trade + c2_borrow + c2_liq, rel=NAV_RTOL)
    # liquidation priced on the continuous venue; regular trades on the simulator's
    # default venue, "routed" (design decision D3; was "auction")
    assert cost.venues == ["routed", "routed", "routed", "continuous", "routed"]
    # optimizer told not to trade C on d3
    assert opt.calls[3]["no_trade"] == ["C"]
    (ev,) = res.diagnostics["delistings"]
    assert ev["date"] == DATES[3] and ev["ticker"] == "C"
    assert ev["weight"] == pytest.approx(0.25) and ev["haircut_return"] == pytest.approx(-0.30)
    assert ev["liquidation_usd"] == pytest.approx(0.175 * nav2, rel=NAV_RTOL)
    assert ev["cost_usd"] == pytest.approx(c2_liq, rel=NAV_RTOL)
    assert res.summary()["n_delistings"] == 1
    assert res.diagnostics["halts"] == [] and res.diagnostics["n_halted_name_days"] == 0


def test_delisting_short_haircut_is_a_loss():
    """B (short -.25) has no close on d3 and d4 (vendor last trading date d2): the
    haircut is applied *against* the position's sign (r = -0.30 * sign(-.25) =
    +0.30), i.e. a loss for the short."""
    close = {k: list(v) for k, v in CLOSE.items()}
    close["B"][3] = close["B"][4] = np.nan
    res, opt, _ = run(bps_model(), TARGETS_CONST, make_md(close, {"B": DATES[2]}))
    w = np.array(W_LS)
    nav1 = NAV0 * 1.0625 - 825.0
    w_pre1 = np.array([0.55, -0.25, 0.2625]) * NAV0 / nav1
    c1 = 8e-4 * np.abs(w - w_pre1).sum() * nav1 + 1e-4 * 0.25 * nav1
    nav2 = nav1 * 0.925 - c1
    w_pre2 = np.array([0.45, -0.275, 0.25]) * nav1 / nav2
    x2 = w - w_pre2
    # gross d3 = .5*0 + (-.25)*(+.30) + .25*(+.10) = -.075 + .025 = -.05
    # x_liq_B = -(-.25) * (1 + .30) = +.325 (buy to cover at the haircut price)
    x_liq = np.array([0.0, 0.325, 0.0])
    c2 = 8e-4 * np.abs(x2).sum() * nav2 + 1e-4 * 0.25 * nav2 + 6e-4 * 0.325 * nav2
    nav3 = nav2 * 0.95 - c2
    w_pre3 = np.array([0.5, 0.0, 0.25 * 1.1]) * nav2 / nav3
    w3 = np.array([0.5, 0.0, 0.25])
    x3 = w3 - w_pre3
    c3 = 8e-4 * np.abs(x3).sum() * nav3  # no short left -> no borrow on d3
    nav4 = nav3 * (1 + 0.5 * 0.10) - c3

    assert res.returns["gross"].iloc[2] == pytest.approx(-0.05, abs=ATOL)
    np.testing.assert_allclose(res.trades.iloc[2].to_numpy(), x2 + x_liq, atol=ATOL, rtol=0)
    assert res.costs["borrow"].iloc[3] == 0.0
    np.testing.assert_allclose(res.nav.to_numpy(), [NAV0, nav1, nav2, nav3, nav4], rtol=NAV_RTOL, atol=0)
    (ev,) = res.diagnostics["delistings"]
    assert ev["ticker"] == "B" and ev["haircut_return"] == pytest.approx(0.30)
    assert ev["liquidation_usd"] == pytest.approx(0.325 * nav2, rel=NAV_RTOL)
    assert opt.calls[3]["no_trade"] == ["B"]


# ----------------------------------------------------------------------------- halts
def _d0_d1_baseline():
    """NAV and drifted book through d2 for TARGETS_CONST (C's d2 return is 0 in
    the base closes, so a C halt starting on d2 or d3 leaves these unchanged)."""
    c0 = 8e-4 * NAV0 + 1e-4 * 0.25 * NAV0  # 825
    nav1 = NAV0 * 1.0625 - c0
    w_pre1 = np.array([0.55, -0.25, 0.2625]) * NAV0 / nav1
    x1 = np.array(W_LS) - w_pre1
    c1 = 8e-4 * np.abs(x1).sum() * nav1 + 1e-4 * 0.25 * nav1
    nav2 = nav1 * 0.925 - c1  # d2 gross: .5*(-.1) + (-.25)*(.1) + .25*0 = -.075
    return nav1, nav2


@pytest.mark.parametrize("vendor", [None, {"C": pd.NaT}], ids=["causal-rule", "vendor-NaT"])
def test_halt_then_resume_is_frozen_not_delisted(vendor):
    """C (long .25) has no close on d2 and d3 and resumes on d4 at 23.1.  Not a
    delisting (gap 2 < G = 63, or the vendor table says alive): the position is
    frozen at 21 (return 0 on d2 and d3, no trade, still weighted), and the d4
    return is 23.1 / 21 - 1 = +10% from the last close before the halt."""
    close = {k: list(v) for k, v in CLOSE.items()}
    close["C"][2] = close["C"][3] = np.nan
    res, opt, _ = run(bps_model(), TARGETS_CONST, make_md(close, vendor))
    nav1, nav2 = _d0_d1_baseline()
    # d2: C has no price -> no_trade, w_post_C = w_pre_C = .25 * nav1 / nav2 (drift with r_C = 0)
    w2c = 0.25 * nav1 / nav2
    w2 = np.array([0.5, -0.25, w2c])
    x2 = w2 - np.array([0.45, -0.275, 0.25]) * nav1 / nav2
    assert x2[2] == 0.0
    c2 = 8e-4 * np.abs(x2).sum() * nav2 + 1e-4 * 0.25 * nav2  # no liquidation cost
    # d3 returns: A 0, B -10%, C frozen 0 -> gross = (-.25)(-.1) = .025
    nav3 = nav2 * 1.025 - c2
    # d3: C still has no price -> frozen again: w_post_C = w2c * nav2 / nav3 = .25 * nav1 / nav3
    w3c = 0.25 * nav1 / nav3
    w3 = np.array([0.5, -0.25, w3c])
    x3 = w3 - np.array([0.5, -0.225, w2c]) * nav2 / nav3
    c3 = 8e-4 * np.abs(x3).sum() * nav3 + 1e-4 * 0.25 * nav3
    # d4 returns: A +10%, B 0, C 23.1/21 - 1 = +10% (whole halt-period move)
    g4 = 0.5 * 0.10 + w3c * 0.10
    nav4 = nav3 * (1 + g4) - c3

    np.testing.assert_allclose(res.nav.to_numpy(), [NAV0, nav1, nav2, nav3, nav4], rtol=NAV_RTOL, atol=0)
    np.testing.assert_allclose(
        res.returns["gross"].to_numpy(), [0.0625, -0.075, 0.025, g4], atol=ATOL, rtol=0
    )
    np.testing.assert_allclose(res.weights.to_numpy()[2:], [w2, w3], atol=ATOL, rtol=0)
    np.testing.assert_allclose(res.trades.to_numpy()[2:], [x2, x3], atol=ATOL, rtol=0)
    np.testing.assert_allclose(res.costs["total"].to_numpy()[2:], [c2, c3], rtol=NAV_RTOL, atol=1e-9)
    assert opt.calls[2]["no_trade"] == ["C"] and opt.calls[3]["no_trade"] == ["C"]
    assert res.diagnostics["delistings"] == []
    assert res.diagnostics["n_halted_name_days"] == 2
    (ep,) = res.diagnostics["halts"]
    assert ep["ticker"] == "C" and ep["outcome"] == "resumed" and ep["days"] == 2
    assert ep["start"] == DATES[2] and ep["end"] == DATES[4] and ep["weight"] == pytest.approx(0.25)


def test_causal_gap_rule_delists_after_g_days():
    """C (long .25) has no close on d3 and d4, no vendor table, G = 2.  On d3 the
    gap is 1 day (< G): frozen.  On d4 the gap reaches 2 = G: delisted, the -30%
    haircut is applied to the frozen price and the rest is liquidated on row d3."""
    close = {k: list(v) for k, v in CLOSE.items()}
    close["C"][3] = close["C"][4] = np.nan
    res, opt, _ = run(bps_model(), TARGETS_CONST, make_md(close), gap_days=2)
    nav1, nav2 = _d0_d1_baseline()
    # d2: normal rebalance back to W_LS
    x2 = np.array(W_LS) - np.array([0.45, -0.275, 0.25]) * nav1 / nav2
    c2 = 8e-4 * np.abs(x2).sum() * nav2 + 1e-4 * 0.25 * nav2
    # d3 returns: A 0, B -10%, C frozen (gap 1) -> gross = .025, no liquidation
    nav3 = nav2 * 1.025 - c2
    # d3: C has no price -> no_trade, frozen weight .25 * nav2 / nav3
    w3c = 0.25 * nav2 / nav3
    x3 = np.array([0.5, -0.25, w3c]) - np.array([0.5, -0.225, 0.25]) * nav2 / nav3
    # d4: gap 2 = G -> delisted: r_C = -.30; x_liq_C = -w3c * .70 (spread 5 + fees 1 bps)
    x_liq = np.array([0.0, 0.0, -0.7 * w3c])
    c3 = 8e-4 * np.abs(x3).sum() * nav3 + 1e-4 * 0.25 * nav3 + 6e-4 * 0.7 * w3c * nav3
    g4 = 0.5 * 0.10 + w3c * (-0.30)
    nav4 = nav3 * (1 + g4) - c3

    np.testing.assert_allclose(res.nav.to_numpy(), [NAV0, nav1, nav2, nav3, nav4], rtol=NAV_RTOL, atol=0)
    np.testing.assert_allclose(
        res.returns["gross"].to_numpy(), [0.0625, -0.075, 0.025, g4], atol=ATOL, rtol=0
    )
    np.testing.assert_allclose(res.trades.to_numpy()[3], x3 + x_liq, atol=ATOL, rtol=0)
    assert res.trades.iloc[2, 2] == pytest.approx(x2[2], abs=ATOL)  # d2: no liquidation yet
    assert opt.calls[3]["no_trade"] == ["C"]  # frozen on d3: no trading
    (ev,) = res.diagnostics["delistings"]
    assert ev["date"] == DATES[4] and ev["ticker"] == "C"
    assert ev["weight"] == pytest.approx(w3c) and ev["haircut_return"] == pytest.approx(-0.30)
    assert ev["liquidation_usd"] == pytest.approx(0.7 * w3c * nav3, rel=NAV_RTOL)
    assert res.diagnostics["n_halted_name_days"] == 1
    (ep,) = res.diagnostics["halts"]
    assert ep["outcome"] == "delisted" and ep["start"] == DATES[3] and ep["end"] == DATES[4]
    # with the default G = 63 the same gap is still a halt at the end of the panel
    res63, _, _ = run(bps_model(), TARGETS_CONST, make_md(close))
    assert res63.diagnostics["delistings"] == []
    assert [e["outcome"] for e in res63.diagnostics["halts"]] == ["open"]


def test_nav_nonpositive_stops_early():
    """A 3x levered long in A loses 10% on d2 after costs -> NAV <= 0 halts the run."""
    targets = {d: [3.0, 0.0, 0.0] for d in DATES[:4]}
    close = {k: list(v) for k, v in CLOSE.items()}
    close["A"] = [100.0, 100.0, 60.0, 60.0, 60.0]  # -40% on d2: nav2 = nav1 * (1 - 1.2) < 0
    res, _, _ = run(ConstantBpsCost(), targets, make_md(close))
    # d0: gross 0, nav1 = 1e6; d1: gross = 3 * -.4 = -1.2 -> nav2 = -200,000 -> stop
    np.testing.assert_allclose(res.nav.to_numpy(), [NAV0, NAV0, -0.2 * NAV0], rtol=NAV_RTOL)
    assert len(res.returns) == 2 and len(res.weights) == 2


# ----------------------------------------------------------------------------- half-life forwarding (fix #5)
class HalfLifeAwareOptimizer(PresetOptimizer):
    """Declares ``alpha_half_life_days`` so the simulator forwards it."""

    def solve(self, alpha_half_life_days=None, **kw):
        kw["alpha_half_life_days"] = alpha_half_life_days
        return super().solve(**kw)


def test_half_life_forwarded_only_when_solve_accepts_it():
    md = make_md()
    sim = DailyBarSimulator(_lag0_config(), ConstantBpsCost())
    aware = HalfLifeAwareOptimizer(TARGETS)
    sim.run(md, DATES, alpha_fn, risk_fn, aware, nav0=NAV0)
    assert aware.calls and all(c["alpha_half_life_days"] == 60.0 for c in aware.calls)
    plain = PresetOptimizer(TARGETS)  # solve(**kw): no named parameter -> not passed
    sim.run(md, DATES, alpha_fn, risk_fn, plain, nav0=NAV0)
    assert plain.calls and all("alpha_half_life_days" not in c for c in plain.calls)
