import numpy as np
import pandas as pd
import pytest

from qts.core.config import CostConfig
from qts.data.synthetic import SyntheticSpec, generate
from qts.tcost import (FeeSchedule, SquareRootImpact, TCALedger, TransactionCostModel,
                       holding_cost_fraction, cvx_holding_expression, borrow_fee_proxy,
                       finra_taf_rate, sec_section31_rate)
from qts.tcost.model import persistence_multiplier

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
    assert sell["taf"] == pytest.approx(0.195)                   # 2026 TAF rate
    assert sell["exchange"] == 0.0                               # no venue -> no auction fee
    assert sell["total"] == pytest.approx(3.50 + 0.20 + 1.03 + 0.195 + 0.003)
    # closing-auction exchange fee applies to auction-venue fills only
    auc = fs.explicit_cost(1000, 50.0, "buy", date="2026-06-15", venue="auction")
    assert auc["exchange"] == pytest.approx(1000 * CostConfig().auction_exchange_fee_per_share)
    assert auc["total"] == pytest.approx(buy["total"] + auc["exchange"])
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
    assert big_sell["taf"] == pytest.approx(9.79)              # per-trade TAF cap (2026)
    assert fs.explicit_cost(100_000, 50.0, "sell", date="2025-06-15")["taf"] == pytest.approx(8.30)


def test_fee_holiday_zeroes_taf():
    fs = FeeSchedule(CostConfig())
    assert fs.explicit_cost(1000, 50.0, "sell", date="2026-11-03")["taf"] == 0.0
    assert fs.explicit_cost(1000, 50.0, "sell", date="2026-09-30")["taf"] == pytest.approx(0.195)
    assert fs.explicit_cost(1000, 50.0, "sell", date="2027-01-04")["taf"] == pytest.approx(0.195)
    assert fs.fee_holiday("2026-12-31")["finra_taf_per_share_sold"] == 0.0
    assert fs.fee_holiday("2026-03-01")["sec_fee_per_dollar_sold"] == 0.0  # SEC rate was 0 2025-05-14..2026-04-03
    assert finra_taf_rate("2024-06-01") == (0.000166, 8.30)
    assert finra_taf_rate(None, CostConfig()) == (0.000195, 9.79)


def test_sec_section31_dated_table():
    fs = FeeSchedule(CostConfig())
    per_m = lambda d: sec_section31_rate(d) * 1e6
    assert per_m("2026-04-04") == pytest.approx(20.60)
    assert per_m("2026-04-03") == 0.0 and per_m("2025-05-14") == 0.0
    assert per_m("2025-05-13") == pytest.approx(27.80) and per_m("2024-05-22") == pytest.approx(27.80)
    assert per_m("2024-05-21") == pytest.approx(8.00)
    assert per_m("2023-01-15") == pytest.approx(22.90)
    assert per_m("2022-01-15") == pytest.approx(5.10)
    assert per_m("2020-06-15") == pytest.approx(22.10)
    assert per_m("2019-06-15") == pytest.approx(20.70)
    assert per_m("2018-12-01") == pytest.approx(13.00)
    assert per_m("2017-09-01") == pytest.approx(23.10)
    assert per_m("2010-01-01") > 0.0                      # pre-table dates: then-current rate, never 0
    assert sec_section31_rate(None, CostConfig()) == pytest.approx(20.60 / 1e6)
    # flows through explicit_cost on sells for a historical date
    sell = fs.explicit_cost(1000, 50.0, "sell", date="2021-06-15")
    assert sell["sec"] == pytest.approx(50_000 * 5.10 / 1e6)
    assert fs.explicit_cost(1000, 50.0, "sell", date="2019-06-15")["sec"] == pytest.approx(50_000 * 20.70 / 1e6)


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
def test_borrow_fee_proxy_shape():
    cfg = CostConfig()
    gc, cap = cfg.default_borrow_fee_annual, cfg.htb_fee_cap_annual
    assert borrow_fee_proxy(0.01, cfg=cfg) == pytest.approx(gc)
    assert borrow_fee_proxy(0.05, cfg=cfg) == pytest.approx(gc)
    assert borrow_fee_proxy(0.20, cfg=cfg) == pytest.approx(cap)
    assert borrow_fee_proxy(0.50, cfg=cfg) == pytest.approx(cap)
    f10, f15 = borrow_fee_proxy(0.10, cfg=cfg), borrow_fee_proxy(0.15, cfg=cfg)
    assert gc < f10 < f15 < cap
    assert (f15 - f10) > (f10 - gc)                     # convex rise
    assert f10 == pytest.approx(gc + (cap - gc) * (1 / 3) ** 2)
    # small-cap haircut multiplier
    assert borrow_fee_proxy(0.10, cfg=cfg, market_cap=1e9) == pytest.approx(f10 * cfg.small_cap_borrow_multiplier)
    assert borrow_fee_proxy(0.10, cfg=cfg, market_cap=5e9) == pytest.approx(f10)
    # days-to-cover tightens; NaN SI -> GC; Series in -> Series out
    assert borrow_fee_proxy(0.01, days_to_cover=10.0, cfg=cfg) == pytest.approx(cap)
    s = borrow_fee_proxy(pd.Series({"A": np.nan, "B": 0.3}), cfg=cfg)
    assert isinstance(s, pd.Series) and s["A"] == pytest.approx(gc) and s["B"] == pytest.approx(cap)


def test_snapshot_uses_borrow_proxy_when_fee_absent(md):
    import dataclasses
    tcm = TransactionCostModel(CostConfig())
    date = md.dates[150]
    md_nofee = dataclasses.replace(md, borrow_fee=None)
    snap = tcm.snapshot_from_market(md_nofee, date)
    assert snap["borrow_fee_source"] == "proxy"
    si = md.short_interest_ratio.loc[:date].iloc[-1].reindex(md.tickers)
    mc = md.market_cap.loc[:date].iloc[-1].reindex(md.tickers)
    expected = borrow_fee_proxy(si, cfg=tcm.cfg, market_cap=mc)
    pd.testing.assert_series_equal(snap["borrow_fee"], expected.fillna(tcm.cfg.default_borrow_fee_annual),
                                   check_names=False)
    assert (snap["borrow_fee"] >= tcm.cfg.default_borrow_fee_annual - 1e-12).all()
    # data present -> data used
    assert tcm.snapshot_from_market(md, date)["borrow_fee_source"] == "data"
    md_none = dataclasses.replace(md, borrow_fee=None, short_interest_ratio=None)
    s3 = tcm.snapshot_from_market(md_none, date)
    assert s3["borrow_fee_source"] == "default" and (s3["borrow_fee"] == tcm.cfg.default_borrow_fee_annual).all()


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


def _venue_fill_ledger(b_auction=0.5, b_continuous=1.0, n=200, seed=7):
    """Noise-free fills through FillSimulator: MOC (auction) and ALGO (continuous)."""
    from qts.execution import FillSimulator, Order
    cfg = CostConfig(impact_b_auction=b_auction, impact_b_continuous=b_continuous)
    sim = FillSimulator(cfg, auction_share=0.09)
    rng = np.random.default_rng(seed)
    led = TCALedger()
    for day in range(n):
        date = pd.Timestamp("2026-06-01") + pd.Timedelta(days=day)
        px = rng.uniform(20, 200)
        adv = rng.uniform(5e5, 5e6)
        vol = adv * rng.uniform(0.8, 1.2)
        sigma = rng.uniform(0.01, 0.04)
        side = "BUY" if rng.random() < 0.5 else "SELL"
        realized = pd.DataFrame({"open": px, "close": px, "next_open": px, "volume": vol,
                                 "sigma_daily": sigma, "half_spread_bps": 2.5, "adv_shares": adv},
                                index=["AUC", "CON"])
        q_auc = max(1.0, round(rng.uniform(0.001, 0.02) * vol * 0.09))
        q_con = max(1.0, round(rng.uniform(0.002, 0.04) * vol))
        orders = [Order("AUC", side, q_auc, order_type="MOC", order_id="auc", meta={"decision_px": px}),
                  Order("CON", side, q_con, order_type="ALGO", algo="Vwap", order_id="con",
                        algo_params={"maxPctVol": 0.5}, meta={"decision_px": px})]
        sim.simulate(orders, realized, date=date, ledger=led)
    return led


def test_calibrate_b_is_venue_aware():
    led = _venue_fill_ledger(b_auction=0.5, b_continuous=1.0)
    d = led.decompose()
    assert set(d["venue"]) == {"auction", "continuous"}
    assert len(d) == 400                                        # unique ids across days
    by = led.calibrate_b_by_venue(prior_weight=0.0)
    assert abs(by["auction"][1]["b_ols"] - 0.5) < 0.15
    assert abs(by["continuous"][1]["b_ols"] - 1.0) < 0.15
    assert by["auction"][1]["n"] == 200 and by["continuous"][1]["n"] == 200
    # single-venue calibrate_b agrees; pooled regression is on per-venue V
    b_auc, diag = led.calibrate_b(prior_b=0.5, prior_weight=0.0, venue="auction")
    assert b_auc == pytest.approx(by["auction"][0]) and diag["venue"] == "auction"
    # the old full-ADV regressor would have recovered b_auction / sqrt(0.09) = 1.67
    auc = d[d["venue"] == "auction"]
    x_full = auc["sigma_daily"] * np.sqrt(auc["participation"])
    y = (auc["arrival_slippage_bps"] - auc["half_spread_bps"]) / 1e4
    assert abs(float(x_full @ y) / float(x_full @ x_full) - 0.5 / np.sqrt(0.09)) < 0.3
    assert (auc["venue_volume"] < auc["filled_qty"] / auc["participation"] * 0.5).all()


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


def test_estimate_prices_requested_venue_and_real_spreads(md):
    tcm = TransactionCostModel(CostConfig())
    date = md.dates[150]
    snap = tcm.snapshot_from_market(md, date)
    trades = pd.Series(0.0, index=md.tickers)
    trades.iloc[:5] = [50_000, -50_000, 10_000, -200_000, 20_000]
    est_c = tcm.estimate(trades, snap, venue="continuous")
    exp_hs = snap["half_spread_bps"].iloc[:5]
    pd.testing.assert_series_equal(est_c["spread_bps"].iloc[:5], exp_hs, check_names=False)
    # half_spread (fraction) fallback when half_spread_bps is missing
    snap2 = {k: v for k, v in snap.items() if k != "half_spread_bps"}
    est_c2 = tcm.estimate(trades, snap2, venue="continuous")
    pd.testing.assert_series_equal(est_c2["spread_bps"], est_c["spread_bps"])
    # auction venue: no spread but the exchange fee is in explicit
    est_a = tcm.estimate(trades, snap, venue="auction")
    assert (est_a["spread_bps"] == 0).all()
    assert (est_a["explicit_bps"].iloc[:5] > est_c["explicit_bps"].iloc[:5]).all()
    # NaN sigma: estimate and cvx_terms both use the cross-sectional median, never 0
    snap3 = dict(snap)
    sig = snap["sigma_daily"].copy(); sig.iloc[0] = np.nan
    snap3["sigma_daily"] = sig
    e3 = tcm.estimate(trades, snap3)
    assert np.isfinite(e3["impact_bps"].iloc[0]) and e3["impact_bps"].iloc[0] > 0
    x = cp.Variable(len(md.tickers))
    trade, _ = tcm.cvx_terms(x, x, nav=1e6, snapshot=snap3)
    assert trade.is_convex()


def test_impact_persistence_multiplier(md):
    assert persistence_multiplier([0, 1, 2, 3]) == pytest.approx([1.0, 1.0, 2 ** 1.5 - 1, 3 ** 1.5 - 2 ** 1.5])
    tcm = TransactionCostModel(CostConfig(impact_persistence=True))
    date = md.dates[150]
    snap = tcm.snapshot_from_market(md, date)
    trades = pd.Series(0.0, index=md.tickers); trades.iloc[0] = 50_000; trades.iloc[1] = -50_000
    base = tcm.estimate(trades, snap)
    streak = pd.Series(1.0, index=md.tickers); streak.iloc[0] = 3.0
    k3 = tcm.estimate(trades, snap, streak=streak)
    m3 = 3 ** 1.5 - 2 ** 1.5
    assert k3["impact_bps"].iloc[0] == pytest.approx(base["impact_bps"].iloc[0] * m3)
    assert k3["impact_bps"].iloc[1] == pytest.approx(base["impact_bps"].iloc[1])
    assert k3["spread_usd"].iloc[0] == base["spread_usd"].iloc[0] and k3["explicit_usd"].iloc[0] == base["explicit_usd"].iloc[0]
    # incremental cost of a k-day build sums to the metaorder cost b*sigma*(kQ)^1.5/sqrt(V)
    q = 50_000.0
    total = sum(tcm.estimate(trades.iloc[:1], snap, streak=pd.Series([k], index=trades.index[:1]))["impact_usd"].iloc[0]
                for k in (1, 2, 3))
    meta = tcm.estimate(trades.iloc[:1] * 3, snap)["impact_usd"].iloc[0]
    assert total == pytest.approx(meta, rel=1e-9)
    # switched off -> no scaling
    off = TransactionCostModel(CostConfig(impact_persistence=False)).estimate(trades, snap, streak=streak)
    assert off["impact_bps"].iloc[0] == pytest.approx(base["impact_bps"].iloc[0])


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
