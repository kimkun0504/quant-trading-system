import numpy as np
import pandas as pd
import pytest

from qts.core.config import CostConfig
from qts.data.synthetic import SyntheticSpec, generate
from qts.tcost import (FeeSchedule, SquareRootImpact, TCALedger, TransactionCostModel,
                       holding_cost_fraction, cvx_holding_expression)

cp = pytest.importorskip("cvxpy")


# ----------------------------------------------------------------------- fees
def test_fee_schedule_1000_shares_50_dollars():
    fs = FeeSchedule(CostConfig())
    buy = fs.explicit_cost(1000, 50.0, "buy", date="2026-06-15")
    sell = fs.explicit_cost(1000, 50.0, "sell", date="2026-06-15")
    assert buy["commission"] == pytest.approx(3.50)
    assert buy["clearing"] == pytest.approx(0.20)
    assert buy["cat"] == pytest.approx(0.003)
    assert buy["sec"] == 0.0 and buy["taf"] == 0.0
    assert buy["total"] == pytest.approx(3.50 + 0.20 + 0.003)
    assert sell["sec"] == pytest.approx(50_000 * 20.60 / 1e6)   # 1.03
    assert sell["taf"] == pytest.approx(0.166)
    assert sell["total"] == pytest.approx(3.50 + 0.20 + 1.03 + 0.166 + 0.003)
    rt_bps = fs.round_trip_bps(50_000, 50.0, date="2026-06-15")
    assert 1.5 <= rt_bps <= 2.5          # "about 2 bp round trip for a $50 stock"
    assert rt_bps == pytest.approx((buy["total"] + sell["total"]) / 50_000 * 1e4)


def test_fee_min_and_cap():
    fs = FeeSchedule(CostConfig())
    tiny = fs.explicit_cost(10, 50.0, "buy")
    assert tiny["commission"] == pytest.approx(0.35)          # per-order minimum
    penny = fs.explicit_cost(10_000, 0.01, "buy")
    assert penny["commission"] == pytest.approx(0.01 * 100.0)  # 1% of value cap
    big_sell = fs.explicit_cost(100_000, 50.0, "sell", date="2026-06-15")
    assert big_sell["taf"] == pytest.approx(8.30)              # per-trade TAF cap


def test_fee_holiday_zeroes_taf():
    fs = FeeSchedule(CostConfig())
    assert fs.explicit_cost(1000, 50.0, "sell", date="2026-11-03")["taf"] == 0.0
    assert fs.explicit_cost(1000, 50.0, "sell", date="2026-09-30")["taf"] == pytest.approx(0.166)
    assert fs.explicit_cost(1000, 50.0, "sell", date="2027-01-04")["taf"] == pytest.approx(0.166)
    assert fs.fee_holiday("2026-12-31") == {"finra_taf_per_share_sold": 0.0}
    assert fs.fee_holiday("2026-03-01")["sec_fee_per_dollar_sold"] == 0.0  # SEC rate was 0 before 2026-04-04


# --------------------------------------------------------------------- impact
def test_impact_sqrt_scaling_and_auction_b():
    imp = SquareRootImpact(CostConfig())
    c1 = imp.cost_bps(1e5, 1e7, 0.02)
    c2 = imp.cost_bps(2e5, 1e7, 0.02)
    assert c2 / c1 == pytest.approx(np.sqrt(2))
    assert c1 == pytest.approx(1e4 * 1.0 * 0.02 * np.sqrt(0.01))   # 20 bps
    assert imp.cost_bps(1e5, 1e7, 0.02, venue="auction") == pytest.approx(CostConfig().impact_b_auction * c1)
    # integrated form: average cost per dollar equals cost_bps when a=0
    tot = imp.total_cost_fraction(1e5, 1e7, 0.02, half_spread_bps=0.0)
    assert tot / 1e5 * 1e4 == pytest.approx(c1)
    tot2 = imp.total_cost_fraction(1e5, 1e7, 0.02, half_spread_bps=2.0)
    assert tot2 == pytest.approx(tot + 2.0 / 1e4 * 1e5)
    # vectorised with Series
    s = imp.cost_bps(pd.Series([1e5, 2e5]), pd.Series([1e7, 1e7]), pd.Series([0.02, 0.02]))
    assert isinstance(s, pd.Series) and s.iloc[1] / s.iloc[0] == pytest.approx(np.sqrt(2))


def test_participation_cap_and_auction_volume():
    imp = SquareRootImpact(CostConfig())
    assert imp.participation_cap(1e7, 1e6, 0.05) == pytest.approx(0.5)
    assert imp.auction_volume_estimate(1e6, "2026-11-03") == pytest.approx(9e4)
    assert imp.auction_share_for_date("2026-06-26") == pytest.approx(0.34)   # Russell recon Friday
    assert imp.auction_share_for_date("2026-03-20") == pytest.approx(0.20)   # quad witching
    choice = imp.cost_for_venue_choice(1e5, 1e7, 0.02, half_spread_bps=5.0, date="2026-11-03")
    assert set(choice) >= {"continuous_bps", "auction_bps", "venue"}
    assert choice["auction_bps"] == pytest.approx(1e4 * CostConfig().impact_b_auction * 0.02 * np.sqrt(1e5 / (0.09 * 1e7)))


def test_cvx_cost_expression_is_dcp_and_solves():
    imp = SquareRootImpact(CostConfig())
    n = 5
    rng = np.random.default_rng(0)
    adv = rng.uniform(5e6, 5e7, n)
    sigma = rng.uniform(0.01, 0.03, n)
    hs = np.full(n, 2e-4)
    x = cp.Variable(n)
    expr = imp.cvx_cost_expression(x, nav=1e7, adv_usd=adv, sigma=sigma, half_spread=hs,
                                   explicit_buy_frac=np.full(n, 1e-4), explicit_sell_frac=np.full(n, 1.5e-4))
    assert expr.is_dcp() and expr.is_convex()
    alpha = np.array([0.01, -0.01, 0.005, 0.0, -0.003])
    prob = cp.Problem(cp.Minimize(expr - alpha @ x), [cp.norm1(x) <= 1.0])
    prob.solve(solver="CLARABEL")
    assert prob.status in ("optimal", "optimal_inaccurate")
    xv = x.value
    assert xv[0] > 0 and xv[1] < 0 and abs(xv[3]) < 1e-4


# --------------------------------------------------------------------- borrow
def test_holding_cost():
    w = pd.Series({"A": 0.1, "B": -0.1, "C": -0.05})
    fee = pd.Series({"A": 0.5, "B": 0.0252, "C": 0.252})
    c = holding_cost_fraction(w, fee, CostConfig(), days=1)
    assert c == pytest.approx(0.1 * 0.0252 / 252 + 0.05 * 0.252 / 252)
    assert holding_cost_fraction(w, None, CostConfig(), days=2) == pytest.approx(0.15 * 0.005 / 252 * 2)
    wv = cp.Variable(3)
    e = cvx_holding_expression(wv, fee.values)
    assert e.is_convex()


# ------------------------------------------------------------------------ TCA
def _synthetic_ledger(n=500, b_true=0.8, seed=1):
    rng = np.random.default_rng(seed)
    led = TCALedger()
    for i in range(n):
        side = "buy" if rng.random() < 0.5 else "sell"
        s = 1 if side == "buy" else -1
        px = rng.uniform(20, 200)
        adv = rng.uniform(5e5, 5e6)
        sigma = rng.uniform(0.01, 0.04)
        part = rng.uniform(0.002, 0.08)
        qty = max(1.0, round(part * adv))
        hs_bps = rng.uniform(1, 4)
        arrival = px * (1 + s * rng.normal(0, 0.001))
        slip = b_true * sigma * np.sqrt(qty / adv) + hs_bps / 1e4 + rng.normal(0, 0.4 * sigma * np.sqrt(part))
        fill = arrival * (1 + s * slip)
        close = fill * (1 + rng.normal(0, 0.005))
        nxt = close * (1 + rng.normal(0, 0.01))
        oid = f"o{i}"
        led.record_order(oid, f"T{i % 50}", side, qty, px, "2026-06-01 15:00", arrival, "2026-06-01 15:50")
        # two partial fills
        led.record_fill(oid, qty * 0.6, fill * (1 + s * 1e-5), "2026-06-01 16:00", "auction")
        led.record_fill(oid, qty * 0.4, fill * (1 - s * 1.5e-5), "2026-06-01 16:00", "auction")
        led.close_order(oid, close, nxt, adv, sigma, fees=qty * px * 1e-4, half_spread_bps=hs_bps)
    return led


def test_calibrate_b_recovers_truth():
    led = _synthetic_ledger(500, b_true=0.8)
    b, diag = led.calibrate_b(prior_b=1.0, prior_weight=20)
    assert diag["n"] == 500
    assert abs(b - 0.8) < 0.25
    assert abs(diag["b_ols"] - 0.8) < 0.2
    assert np.isfinite(diag["se"]) and np.isfinite(diag["r2"])
    # empty ledger returns the prior
    assert TCALedger().calibrate_b(prior_b=1.0)[0] == 1.0


def test_decomposition_sums_and_signs():
    led = TCALedger()
    # buy 100 @ decision 100, arrival 100.10, fills 60@100.30, 20@100.50, close 101, next open 100.8
    led.record_order("x", "AAA", "buy", 100, 100.0, "2026-06-01 10:00", 100.10, "2026-06-01 10:05")
    led.record_fill("x", 60, 100.30, "2026-06-01 11:00", "continuous")
    led.record_fill("x", 20, 100.50, "2026-06-01 12:00", "continuous")
    led.close_order("x", 101.0, 100.8, adv_shares=10_000, sigma_daily=0.02, fees=0.50)
    d = led.decompose().loc["x"]
    k = 1e4 / (100 * 100.0)
    vwap = (60 * 100.30 + 20 * 100.50) / 80
    assert d["delay_bps"] == pytest.approx(80 * 0.10 * k)
    assert d["impact_bps"] == pytest.approx(80 * (vwap - 100.10) * k)
    assert d["opportunity_bps"] == pytest.approx(20 * 1.0 * k)
    assert d["fees_bps"] == pytest.approx(0.50 * k)
    assert d["total_is_bps"] == pytest.approx(d["delay_bps"] + d["impact_bps"] + d["opportunity_bps"] + d["fees_bps"])
    # Perold: total = filled*(vwap-dec) + unfilled*(close-dec) + fees
    assert d["total_is_bps"] == pytest.approx((80 * (vwap - 100) + 20 * (101 - 100) + 0.5) * k)
    assert d["reversion_bps"] == pytest.approx(80 * (vwap - 100.8) * k)
    assert d["fill_rate"] == pytest.approx(0.8)
    # sell order mirrors signs
    led.record_order("y", "BBB", "sell", 100, 50.0, "2026-06-01 10:00", 49.95, "2026-06-01 10:05")
    led.record_fill("y", 100, 49.90, "2026-06-01 11:00", "continuous")
    led.close_order("y", 49.0, 49.5, adv_shares=10_000, sigma_daily=0.02, fees=0.0)
    e = led.decompose().loc["y"]
    assert e["delay_bps"] > 0 and e["impact_bps"] > 0 and e["opportunity_bps"] == 0
    summ = led.summary()
    assert not summ.empty and np.isfinite(summ["total_is_bps"]).all()


def test_ledger_persistence_roundtrip(tmp_path):
    led = _synthetic_ledger(20)
    led.save(tmp_path / "tca", fmt="parquet")
    led2 = TCALedger.load(tmp_path / "tca", fmt="parquet")
    pd.testing.assert_frame_equal(led.decompose(), led2.decompose())
    led.save(tmp_path / "tca_csv", fmt="csv")
    led3 = TCALedger.load(tmp_path / "tca_csv", fmt="csv")
    assert len(led3.orders) == 20 and len(led3.fills) == 40


# --------------------------------------------------------------------- facade
@pytest.fixture(scope="module")
def md():
    return generate(SyntheticSpec(n_stocks=40, n_days=300, seed=3))


def test_snapshot_point_in_time_and_estimate(md):
    tcm = TransactionCostModel(CostConfig())
    date = md.dates[150]
    snap = tcm.snapshot_from_market(md, date)
    for k in ("adv_usd", "adv_shares", "sigma_daily", "half_spread_bps", "half_spread", "price", "borrow_fee", "venue"):
        assert k in snap and len(snap[k]) == len(md.tickers)
    # point-in-time: identical when future data is removed
    snap2 = tcm.snapshot_from_market(md.slice_until(date), date)
    pd.testing.assert_series_equal(snap["adv_usd"], snap2["adv_usd"])
    pd.testing.assert_series_equal(snap["sigma_daily"], snap2["sigma_daily"])
    exp_adv = (md.close * md.volume).loc[:date].tail(63).mean()
    pd.testing.assert_series_equal(snap["adv_usd"], exp_adv.astype(float), check_names=False)

    trades = pd.Series(0.0, index=md.tickers)
    trades.iloc[:5] = [50_000, -50_000, 10_000, -200_000, 0.0]
    est = tcm.estimate(trades, snap)
    assert set(["spread_bps", "impact_bps", "explicit_bps", "total_bps", "total_usd"]) <= set(est.columns)
    assert (est["total_bps"] >= 0).all() and est.loc[trades.index[4], "total_usd"] == 0.0
    assert (est["spread_bps"] == 0).all()          # default venue = auction: no spread crossing
    est_c = tcm.estimate(trades, snap, venue="continuous")
    assert (est_c.loc[trades != 0, "spread_bps"] > 0).all()
    # explicit bps in a plausible range (sub-5 bp for $50-$200 stocks)
    assert est.loc[trades != 0, "explicit_bps"].between(0.1, 10).all()


def test_cvx_terms_solve(md):
    tcm = TransactionCostModel(CostConfig())
    date = md.dates[150]
    snap = tcm.snapshot_from_market(md, date)
    n = len(md.tickers)
    w_pre = np.zeros(n)
    x = cp.Variable(n)
    w_post = w_pre + x
    trade, hold = tcm.cvx_terms(x, w_post, nav=5e6, snapshot=snap)
    assert trade.is_convex() and hold.is_convex()
    alpha = np.linspace(-0.01, 0.01, n)
    cap = tcm.participation_cap(snap, nav=5e6).values
    prob = cp.Problem(cp.Minimize(trade + hold - alpha @ x),
                      [cp.norm1(x) <= 1.0, cp.abs(x) <= cap])
    prob.solve(solver="CLARABEL")
    assert prob.status in ("optimal", "optimal_inaccurate")
    assert np.all(np.abs(x.value) <= cap + 1e-6)
