import numpy as np
import pandas as pd
import pytest

from qts.core.config import CostConfig
from qts.core.costs import CostEstimate, CostModel, LiquiditySnapshot, as_snapshot, auction_share_for_date
from qts.data.synthetic import SyntheticSpec, generate
from qts.tcost import (
    FeeSchedule,
    SquareRootImpact,
    TCALedger,
    TransactionCostModel,
    borrow_fee_proxy,
    cvx_holding_expression,
    finra_taf_rate,
    holding_cost_fraction,
    sec_section31_rate,
)
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
    assert sell["sec"] == pytest.approx(50_000 * 20.60 / 1e6)  # 1.03
    assert sell["taf"] == pytest.approx(0.195)  # 2026 TAF rate
    assert sell["exchange"] == 0.0  # no venue -> no auction fee
    assert sell["total"] == pytest.approx(3.50 + 0.20 + 1.03 + 0.195 + 0.003)
    # closing-auction exchange fee applies to auction-venue fills only
    auc = fs.explicit_cost(1000, 50.0, "buy", date="2026-06-15", venue="auction")
    assert auc["exchange"] == pytest.approx(1000 * CostConfig().auction_exchange_fee_per_share)
    assert auc["total"] == pytest.approx(buy["total"] + auc["exchange"])
    rt_bps = fs.round_trip_bps(50_000, 50.0, date="2026-06-15")
    assert 1.5 <= rt_bps <= 2.5  # "about 2 bp round trip for a $50 stock"
    assert rt_bps == pytest.approx((buy["total"] + sell["total"]) / 50_000 * 1e4)


def test_fee_min_and_cap():
    fs = FeeSchedule(CostConfig())
    tiny = fs.explicit_cost(10, 50.0, "buy")
    assert tiny["commission"] == pytest.approx(0.35)  # per-order minimum
    penny = fs.explicit_cost(10_000, 0.01, "buy")
    assert penny["commission"] == pytest.approx(0.01 * 100.0)  # 1% of value cap
    big_sell = fs.explicit_cost(100_000, 50.0, "sell", date="2026-06-15")
    assert big_sell["taf"] == pytest.approx(9.79)  # per-trade TAF cap (2026)
    assert fs.explicit_cost(100_000, 50.0, "sell", date="2025-06-15")["taf"] == pytest.approx(8.30)


def test_fee_holiday_zeroes_taf():
    fs = FeeSchedule(CostConfig())
    assert fs.explicit_cost(1000, 50.0, "sell", date="2026-11-03")["taf"] == 0.0
    assert fs.explicit_cost(1000, 50.0, "sell", date="2026-09-30")["taf"] == pytest.approx(0.195)
    assert fs.explicit_cost(1000, 50.0, "sell", date="2027-01-04")["taf"] == pytest.approx(0.195)
    assert fs.fee_holiday("2026-12-31")["finra_taf_per_share_sold"] == 0.0
    assert (
        fs.fee_holiday("2026-03-01")["sec_fee_per_dollar_sold"] == 0.0
    )  # SEC rate was 0 2025-05-14..2026-04-03
    assert finra_taf_rate("2024-06-01") == (0.000166, 8.30)
    assert finra_taf_rate(None, CostConfig()) == (0.000195, 9.79)


def test_sec_section31_dated_table():
    fs = FeeSchedule(CostConfig())

    def per_m(d):
        return sec_section31_rate(d) * 1e6

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
    assert per_m("2010-01-01") > 0.0  # pre-table dates: then-current rate, never 0
    assert sec_section31_rate(None, CostConfig()) == pytest.approx(20.60 / 1e6)
    # flows through explicit_cost on sells for a historical date
    sell = fs.explicit_cost(1000, 50.0, "sell", date="2021-06-15")
    assert sell["sec"] == pytest.approx(50_000 * 5.10 / 1e6)
    assert fs.explicit_cost(1000, 50.0, "sell", date="2019-06-15")["sec"] == pytest.approx(
        50_000 * 20.70 / 1e6
    )


# --------------------------------------------------------------------- impact
def test_impact_sqrt_scaling_and_auction_b():
    imp = SquareRootImpact(CostConfig())
    c1 = imp.cost_bps(1e5, 1e7, 0.02)
    c2 = imp.cost_bps(2e5, 1e7, 0.02)
    assert c2 / c1 == pytest.approx(np.sqrt(2))
    assert c1 == pytest.approx(1e4 * 1.0 * 0.02 * np.sqrt(0.01))  # 20 bps
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
    assert imp.auction_share_for_date("2026-06-26") == pytest.approx(0.34)  # Russell recon Friday
    assert imp.auction_share_for_date("2026-03-20") == pytest.approx(0.20)  # quad witching
    choice = imp.cost_for_venue_choice(1e5, 1e7, 0.02, half_spread_bps=5.0, date="2026-11-03")
    assert set(choice) >= {"continuous_bps", "auction_bps", "venue"}
    assert choice["auction_bps"] == pytest.approx(
        1e4 * CostConfig().impact_b_auction * 0.02 * np.sqrt(1e5 / (0.09 * 1e7))
    )


def test_cvx_cost_expression_is_dcp_and_solves():
    imp = SquareRootImpact(CostConfig())
    n = 5
    rng = np.random.default_rng(0)
    adv = rng.uniform(5e6, 5e7, n)
    sigma = rng.uniform(0.01, 0.03, n)
    hs = np.full(n, 2e-4)
    x = cp.Variable(n)
    expr = imp.cvx_cost_expression(
        x,
        nav=1e7,
        adv_usd=adv,
        sigma=sigma,
        half_spread=hs,
        explicit_buy_frac=np.full(n, 1e-4),
        explicit_sell_frac=np.full(n, 1.5e-4),
    )
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
    assert (f15 - f10) > (f10 - gc)  # convex rise
    assert f10 == pytest.approx(gc + (cap - gc) * (1 / 3) ** 2)
    # small-cap haircut multiplier
    assert borrow_fee_proxy(0.10, cfg=cfg, market_cap=1e9) == pytest.approx(
        f10 * cfg.small_cap_borrow_multiplier
    )
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
    # days to cover = short interest (shares) / ADV (shares), from md.shares_outstanding
    so = md.shares_outstanding.loc[:date].iloc[-1].reindex(md.tickers)
    dtc = si * so / snap["adv_shares"].reindex(md.tickers)
    expected = borrow_fee_proxy(si, days_to_cover=dtc, cfg=tcm.cfg, market_cap=mc)
    pd.testing.assert_series_equal(
        snap["borrow_fee"], expected.fillna(tcm.cfg.default_borrow_fee_annual), check_names=False
    )
    # without a share count the proxy falls back to short interest + size only
    snap_noso = tcm.snapshot_from_market(dataclasses.replace(md_nofee, shares_outstanding=None), date)
    expected_noso = borrow_fee_proxy(si, cfg=tcm.cfg, market_cap=mc)
    pd.testing.assert_series_equal(
        snap_noso["borrow_fee"], expected_noso.fillna(tcm.cfg.default_borrow_fee_annual), check_names=False
    )
    assert (snap["borrow_fee"] >= tcm.cfg.default_borrow_fee_annual - 1e-12).all()
    # data present -> data used
    assert tcm.snapshot_from_market(md, date)["borrow_fee_source"] == "data"
    md_none = dataclasses.replace(md, borrow_fee=None, short_interest_ratio=None)
    s3 = tcm.snapshot_from_market(md_none, date)
    # no SI data: GC default, times the small-cap multiplier for small caps (audit 2 #8)
    gc, mult = tcm.cfg.default_borrow_fee_annual, tcm.cfg.small_cap_borrow_multiplier
    small = (mc < tcm.cfg.small_cap_threshold_usd).to_numpy()
    assert s3["borrow_fee_source"] == "default"
    np.testing.assert_allclose(s3["borrow_fee"].to_numpy(), np.where(small, gc * mult, gc))
    assert small.any() and (~small).any()
    s4 = tcm.snapshot_from_market(dataclasses.replace(md_none, market_cap=None), date)
    assert (s4["borrow_fee"] == gc).all()


def test_borrow_fee_source_per_name_and_all_nan_panel(md):
    """Audit 2 finding 8: an all-NaN borrow_fee row is not "data"; names with NaN SI
    still get the small-cap multiplier on the GC default; per-name sources."""
    import dataclasses

    tcm = TransactionCostModel(CostConfig())
    date = md.dates[150]
    nan_fee = md.borrow_fee * np.nan
    si = md.short_interest_ratio.copy()
    si.iloc[:, :5] = np.nan  # five names without SI data
    md2 = dataclasses.replace(md, borrow_fee=nan_fee, short_interest_ratio=si)
    snap = tcm.snapshot(md2, date)
    src = tcm.borrow_fee_sources(md2, date)
    assert snap.borrow_fee_source == "proxy+default"
    assert (src.iloc[:5] == "default").all() and (src.iloc[5:] == "proxy").all()
    mc = md.market_cap.loc[:date].iloc[-1].reindex(md.tickers)
    gc, mult = tcm.cfg.default_borrow_fee_annual, tcm.cfg.small_cap_borrow_multiplier
    small = (mc.iloc[:5] < tcm.cfg.small_cap_threshold_usd).to_numpy()
    np.testing.assert_allclose(snap.borrow_fee.iloc[:5].to_numpy(), np.where(small, gc * mult, gc))
    # one finite datum -> "data" appears, and that name uses it
    fee1 = nan_fee.copy()
    fee1.iloc[:, 7] = 0.031
    md3 = dataclasses.replace(md2, borrow_fee=fee1)
    snap3 = tcm.snapshot(md3, date)
    assert snap3.borrow_fee_source == "data+proxy+default"
    assert snap3.borrow_fee.iloc[7] == pytest.approx(0.031)
    assert tcm.borrow_fee_sources(md3, date).iloc[7] == "data"


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


def _venue_fill_ledger(b_auction=0.5, b_continuous=1.0, n=200, seed=7, exponent=1.5, vol_noise=True):
    """Noise-free fills through FillSimulator: MOC (auction) and ALGO (continuous)."""
    from qts.execution import FillSimulator, Order

    cfg = CostConfig(impact_b_auction=b_auction, impact_b_continuous=b_continuous, impact_exponent=exponent)
    sim = FillSimulator(cfg, auction_share=0.09)
    rng = np.random.default_rng(seed)
    led = TCALedger(cost_cfg=cfg)
    for day in range(n):
        date = pd.Timestamp("2026-06-01") + pd.Timedelta(days=day)
        px = rng.uniform(20, 200)
        adv = rng.uniform(5e5, 5e6)
        vol = adv * rng.uniform(0.8, 1.2) if vol_noise else adv
        sigma = rng.uniform(0.01, 0.04)
        side = "BUY" if rng.random() < 0.5 else "SELL"
        realized = pd.DataFrame(
            {
                "open": px,
                "close": px,
                "next_open": px,
                "volume": vol,
                "sigma_daily": sigma,
                "half_spread_bps": 2.5,
                "adv_shares": adv,
            },
            index=["AUC", "CON"],
        )
        q_auc = max(1.0, round(rng.uniform(0.001, 0.02) * vol * 0.09))
        q_con = max(1.0, round(rng.uniform(0.002, 0.04) * vol))
        orders = [
            Order("AUC", side, q_auc, order_type="MOC", order_id="auc", meta={"decision_px": px}),
            Order(
                "CON",
                side,
                q_con,
                order_type="ALGO",
                algo="Vwap",
                order_id="con",
                algo_params={"maxPctVol": 0.5},
                meta={"decision_px": px},
            ),
        ]
        sim.simulate(orders, realized, date=date, ledger=led)
    return led


def test_calibrate_b_is_venue_aware():
    led = _venue_fill_ledger(b_auction=0.5, b_continuous=1.0)
    d = led.decompose()
    assert set(d["venue"]) == {"auction", "continuous"}
    assert len(d) == 400  # unique ids across days
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


@pytest.mark.parametrize("exponent", [1.25, 1.5, 1.8])
def test_calibrate_b_uses_configured_exponent(exponent):
    """Audit 2 finding 2: the regressor is sigma*(F/V)^(e-1), the average impact per
    dollar of C(Q) = b sigma Q^e / V^(e-1).  Noise-free fills with realised volume = ADV
    recover both venue prefactors exactly for any e in (1, 2]."""
    led = _venue_fill_ledger(b_auction=0.3, b_continuous=1.0, exponent=exponent, vol_noise=False)
    by = led.calibrate_b_by_venue(prior_weight=0.0)
    assert by["auction"][1]["b_ols"] == pytest.approx(0.3, rel=1e-6)
    assert by["continuous"][1]["b_ols"] == pytest.approx(1.0, rel=1e-6)
    d = led.decompose()
    # hand check of the regressor on one order
    r = d.iloc[0]
    assert r["impact_x"] == pytest.approx(
        r["sigma_daily"] * (r["filled_qty"] / r["venue_volume"]) ** (exponent - 1)
    )
    if exponent != 1.5:  # the old sqrt regressor is biased away from e = 1.5
        auc = d[d["venue"] == "continuous"]
        x_sqrt = auc["sigma_daily"] * np.sqrt(auc["filled_qty"] / auc["venue_volume"])
        y = (auc["arrival_slippage_bps"] - auc["half_spread_bps"]) / 1e4
        assert abs(float(x_sqrt @ y) / float(x_sqrt @ x_sqrt) - 1.0) > 0.3


def test_decomposition_sums_and_signs():
    led = TCALedger()
    # buy 100 @ decision 100, arrival 100.10, fills 60@100.30, 20@100.50, close 101, next open 100.8
    led.record_order("x", "AAA", "buy", 100, 100.0, "2026-06-01 10:00", 100.10, "2026-06-01 10:05")
    led.record_fill("x", 60, 100.30, "2026-06-01 11:00", "continuous")
    led.record_fill("x", 20, 100.50, "2026-06-01 12:00", "continuous")
    led.close_order("x", 101.0, 100.8, adv_shares=10_000, sigma_daily=0.02, fees=0.50, next_close_px=100.2)
    d = led.decompose().loc["x"]
    k = 1e4 / (100 * 100.0)
    vwap = (60 * 100.30 + 20 * 100.50) / 80
    assert d["delay_bps"] == pytest.approx(80 * 0.10 * k)
    assert d["impact_bps"] == pytest.approx(80 * (vwap - 100.10) * k)
    assert d["opportunity_bps"] == pytest.approx(20 * 1.0 * k)
    assert d["fees_bps"] == pytest.approx(0.50 * k)
    assert d["total_is_bps"] == pytest.approx(
        d["delay_bps"] + d["impact_bps"] + d["opportunity_bps"] + d["fees_bps"]
    )
    # Perold: total = filled*(vwap-dec) + unfilled*(close-dec) + fees
    assert d["total_is_bps"] == pytest.approx((80 * (vwap - 100) + 20 * (101 - 100) + 0.5) * k)
    # reversion vs the NEXT CLOSE by default (finding #25); next open is configurable
    assert CostConfig().tca_reversion_reference == "next_close"
    assert d["reversion_bps"] == pytest.approx(80 * (vwap - 100.2) * k)
    led_open = TCALedger(
        orders=led.orders, fills=led.fills, cost_cfg=CostConfig(tca_reversion_reference="next_open")
    )
    assert led_open.decompose().loc["x", "reversion_bps"] == pytest.approx(80 * (vwap - 100.8) * k)
    with pytest.raises(ValueError):
        TCALedger(
            orders=led.orders, fills=led.fills, cost_cfg=CostConfig(tca_reversion_reference="vwap")
        ).decompose()
    assert d["fill_rate"] == pytest.approx(0.8)
    # sell order mirrors signs
    led.record_order("y", "BBB", "sell", 100, 50.0, "2026-06-01 10:00", 49.95, "2026-06-01 10:05")
    led.record_fill("y", 100, 49.90, "2026-06-01 11:00", "continuous")
    led.close_order("y", 49.0, 49.5, adv_shares=10_000, sigma_daily=0.02, fees=0.0)
    e = led.decompose().loc["y"]
    assert e["delay_bps"] > 0 and e["impact_bps"] > 0 and e["opportunity_bps"] == 0
    assert np.isnan(e["reversion_bps"])  # no next close given: NaN, never a silent 0
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
    snap = tcm.snapshot(md, date)
    assert isinstance(snap, LiquiditySnapshot) and snap.tickers.equals(md.tickers) and snap.date == date
    for k in ("price", "adv_usd", "adv_shares", "sigma_daily", "half_spread", "borrow_fee"):
        v = getattr(snap, k)
        assert len(v) == len(md.tickers) and np.isfinite(v.to_numpy()).all(), k
    assert snap.auction_share == pytest.approx(0.09) or snap.auction_share >= 0.2
    # point-in-time: identical when future data is removed
    snap2 = tcm.snapshot(md.slice_until(date), date)
    for k in ("adv_usd", "sigma_daily", "half_spread", "borrow_fee", "price"):
        pd.testing.assert_series_equal(getattr(snap, k), getattr(snap2, k))
    raw_adv = (md.close * md.volume).loc[:date].tail(63).mean()
    ok = raw_adv.notna()
    pd.testing.assert_series_equal(snap.adv_usd[ok], raw_adv[ok].astype(float), check_names=False)
    # backward-compatible mapping access and the deprecated dict builder agree
    assert snap["half_spread_bps"].equals(snap.half_spread * 1e4) and "date" in snap
    legacy = tcm.snapshot_from_market(md, date)
    assert isinstance(legacy, dict)
    pd.testing.assert_series_equal(legacy["adv_usd"], snap.adv_usd)

    trades = pd.Series(0.0, index=md.tickers)
    trades.iloc[:5] = [50_000, -50_000, 10_000, -200_000, 0.0]
    est = tcm.estimate(trades, snap)  # default venue: routed
    assert isinstance(est, CostEstimate) and est.venue == "routed"
    assert list(est.frame.columns) == [
        "spread_usd",
        "impact_usd",
        "fees_usd",
        "total_usd",
        "venue",
        "persistence_mult",
    ]
    assert (est.frame[["spread_usd", "impact_usd", "fees_usd"]] >= 0).all().all()
    assert est.frame.loc[trades.index[4], "total_usd"] == 0.0
    np.testing.assert_allclose(est["total_usd"], est["spread_usd"] + est["impact_usd"] + est["fees_usd"])
    tot = est.totals()
    assert tot["total"] == pytest.approx(est["total_usd"].sum())
    est_a = tcm.estimate(trades, snap, "auction")
    assert (est_a["spread_usd"] == 0).all()  # auction: no spread crossing
    est_c = tcm.estimate(trades, snap, venue="continuous")
    assert (est_c["spread_usd"][trades != 0] > 0).all()
    # explicit bps in a plausible range (sub-10 bp for these prices)
    assert est_c.bps().loc[trades != 0, "fees_bps"].between(0.1, 10).all()
    with pytest.raises(ValueError):
        tcm.estimate(trades, snap, venue="dark_pool")


def test_estimate_prices_requested_venue_and_real_spreads(md):
    tcm = TransactionCostModel(CostConfig())
    date = md.dates[150]
    snap = tcm.snapshot(md, date)
    trades = pd.Series(0.0, index=md.tickers)
    trades.iloc[:5] = [50_000, -50_000, 10_000, -200_000, 20_000]
    est_c = tcm.estimate(trades, snap, venue="continuous")
    exp_hs = snap.half_spread_bps.iloc[:5]
    pd.testing.assert_series_equal(est_c.bps()["spread_bps"].iloc[:5], exp_hs, check_names=False)
    # old dict snapshots are accepted (adapter): half_spread_bps or half_spread fraction
    d = snap.to_dict()
    est_d = tcm.estimate(trades, d, venue="continuous")
    pd.testing.assert_frame_equal(est_d.frame, est_c.frame)
    d2 = {k: v for k, v in d.items() if k != "half_spread_bps"}
    pd.testing.assert_frame_equal(tcm.estimate(trades, d2, venue="continuous").frame, est_c.frame)
    # auction venue: no spread but the exchange fee is in fees
    est_a = tcm.estimate(trades, snap, venue="auction")
    assert (est_a["spread_usd"] == 0).all()
    assert (est_a["fees_usd"].iloc[:5] > est_c["fees_usd"].iloc[:5]).all()
    # NaN sigma: estimate and cvx_terms both use the cross-sectional median, never 0
    snap3 = dict(d)
    sig = snap.sigma_daily.copy()
    sig.iloc[0] = np.nan
    snap3["sigma_daily"] = sig
    e3 = tcm.estimate(trades, snap3)
    assert np.isfinite(e3["impact_usd"].iloc[0]) and e3["impact_usd"].iloc[0] > 0
    assert as_snapshot(snap3).sigma_daily.iloc[0] == pytest.approx(sig.median())
    x = cp.Variable(len(md.tickers))
    trade, _ = tcm.cvx_terms(x, x, 1e6, snap3)
    assert trade.is_convex()


def test_routed_venue_hand_computed():
    """D3: q_a = min(q, kappa*phi*ADV) at the auction, the rest continuous; the
    continuous leg is the increment C(q) - C(q_a) of the day's metaorder."""
    cfg = CostConfig(impact_b_auction=0.4)  # != D2 so the auction leg is distinguishable
    tcm = TransactionCostModel(cfg)
    idx = pd.Index(["A"])
    snap = LiquiditySnapshot(
        date=pd.Timestamp("2026-06-15"),
        tickers=idx,
        price=50.0,
        adv_usd=5e7,
        adv_shares=1e6,
        sigma_daily=0.02,
        half_spread=3e-4,
        borrow_fee=0.005,
        auction_share=0.09,
    )
    K = 0.02 * 0.09 * 5e7  # 90,000 USD auction cap
    q = 500_000.0
    est = tcm.estimate(pd.Series([q], index=idx), snap, "routed")
    qa, qc = K, q - K
    imp_auc = 0.4 * 0.02 * qa**1.5 / (0.09 * 5e7) ** 0.5
    imp_cont = 1.0 * 0.02 * (q**1.5 - qa**1.5) / 5e7**0.5
    fees = (
        tcm.fees.explicit_cost(qa / 50.0, 50.0, "buy", "2026-06-15", "auction")["total"]
        + tcm.fees.explicit_cost(qc / 50.0, 50.0, "buy", "2026-06-15", "continuous")["total"]
    )
    assert est["spread_usd"].iloc[0] == pytest.approx(3e-4 * qc, rel=1e-12)
    assert est["impact_usd"].iloc[0] == pytest.approx(imp_auc + imp_cont, rel=1e-12)
    assert est["fees_usd"].iloc[0] == pytest.approx(fees, rel=1e-12)
    assert est.auction_fraction.iloc[0] == pytest.approx(K / q)
    # small trade: entirely at the auction
    small = tcm.estimate(pd.Series([-50_000.0], index=idx), snap, "routed")
    auc = tcm.estimate(pd.Series([-50_000.0], index=idx), snap, "auction")
    pd.testing.assert_series_equal(small["total_usd"], auc["total_usd"])
    # D2 defaults: routed impact == continuous impact (auction saves only the spread)
    d2 = TransactionCostModel(CostConfig())
    assert d2.effective_auction_b(0.09) == pytest.approx(1.0)
    r = d2.estimate(pd.Series([q], index=idx), snap, "routed")
    c = d2.estimate(pd.Series([q], index=idx), snap, "continuous")
    assert r["impact_usd"].iloc[0] == pytest.approx(c["impact_usd"].iloc[0], rel=1e-12)
    assert r["spread_usd"].iloc[0] == pytest.approx(c["spread_usd"].iloc[0] - 3e-4 * K, rel=1e-12)


def test_impact_persistence_multiplier(md):
    assert persistence_multiplier([0, 1, 2, 3]) == pytest.approx([1.0, 1.0, 2**1.5 - 1, 3**1.5 - 2**1.5])
    assert persistence_multiplier(3) == pytest.approx(2.368, abs=5e-4)  # m_3 (finding #23)
    tcm = TransactionCostModel(CostConfig(impact_persistence=True))
    date = md.dates[150]
    snap = tcm.snapshot(md, date)
    trades = pd.Series(0.0, index=md.tickers)
    trades.iloc[0] = 50_000
    trades.iloc[1] = -50_000
    base = tcm.estimate(trades, snap, "continuous")
    streak = pd.Series(1.0, index=md.tickers)
    streak.iloc[0] = 3.0
    k3 = tcm.estimate(trades, snap, "continuous", streak=streak)
    m3 = 3**1.5 - 2**1.5
    assert k3["impact_usd"].iloc[0] == pytest.approx(base["impact_usd"].iloc[0] * m3)
    assert k3["persistence_mult"].iloc[0] == pytest.approx(m3)
    assert k3["impact_usd"].iloc[1] == pytest.approx(base["impact_usd"].iloc[1])
    assert (
        k3["spread_usd"].iloc[0] == base["spread_usd"].iloc[0]
        and k3["fees_usd"].iloc[0] == base["fees_usd"].iloc[0]
    )
    # incremental cost of a k-day build sums to the metaorder cost b*sigma*(kQ)^1.5/sqrt(V)
    one = trades.iloc[:1]
    total = sum(
        tcm.estimate(one, snap, "continuous", streak=pd.Series([k], index=one.index))["impact_usd"].iloc[0]
        for k in (1, 2, 3)
    )
    meta = tcm.estimate(one * 3, snap, "continuous")["impact_usd"].iloc[0]
    assert total == pytest.approx(meta, rel=1e-9)
    # switched off -> no scaling
    off = TransactionCostModel(CostConfig(impact_persistence=False)).estimate(
        trades, snap, "continuous", streak=streak
    )
    assert off["impact_usd"].iloc[0] == pytest.approx(base["impact_usd"].iloc[0])


def test_impact_exponent_is_wired_and_validated():
    for bad in (1.0, 0.5, 2.5):
        with pytest.raises(ValueError):
            SquareRootImpact(CostConfig(impact_exponent=bad))
    imp = SquareRootImpact(CostConfig(impact_exponent=1.6))
    assert imp.cost_bps(1e5, 1e7, 0.02) == pytest.approx(1e4 * 0.02 * 0.01**0.6)
    assert imp.total_cost_fraction(1e5, 1e7, 0.02) == pytest.approx(0.02 * 1e5**1.6 / 1e7**0.6)
    tcm = TransactionCostModel(CostConfig(impact_exponent=1.6))
    idx = pd.Index(["A"])
    snap = LiquiditySnapshot(
        date=None,
        tickers=idx,
        price=50.0,
        adv_usd=1e7,
        adv_shares=2e5,
        sigma_daily=0.02,
        half_spread=0.0,
        borrow_fee=0.0,
    )
    est = tcm.estimate(pd.Series([1e5], index=idx), snap, "continuous")
    assert est["impact_usd"].iloc[0] == pytest.approx(0.02 * 1e5**1.6 / 1e7**0.6)
    assert persistence_multiplier(2, 1.6) == pytest.approx(2**1.6 - 1)
    x = cp.Variable(1)
    x.value = np.array([0.1])
    trade, _ = tcm.cvx_terms(x, x, 1e6, snap, "continuous", include_explicit=False)
    assert trade.value == pytest.approx(est["impact_usd"].iloc[0] / 1e6, rel=1e-10)


def test_auction_share_single_source():
    """CostConfig.auction_volume_share_default drives impact, router, fill
    simulator, dry-run broker and TCA (finding #20)."""
    from qts.core.config import ExecutionConfig
    from qts.execution import DryRunBroker, ExecutionRouter, FillSimulator, Order

    assert not hasattr(ExecutionConfig(), "auction_volume_share_default")
    cfg = CostConfig(auction_volume_share_default=0.16)
    imp = SquareRootImpact(cfg)
    assert imp.auction_share_for_date("2026-11-03") == pytest.approx(0.16)
    assert imp.auction_share_for_date("2026-03-20") == pytest.approx(0.20)  # heavy-day floor
    assert auction_share_for_date("2026-06-26") == pytest.approx(0.34)
    assert imp.cost_for_venue_choice(1e5, 1e7, 0.02, date="2026-11-03")["auction_share"] == pytest.approx(
        0.16
    )
    # router: missing auction_volume_est -> adv * date-aware share
    r = ExecutionRouter(cost_cfg=cfg)
    snap = {
        "adv_shares": pd.Series({"A": 1e6}),
        "sigma_daily": pd.Series({"A": 0.02}),
        "indicative_close_px": pd.Series({"A": 50.0}),
    }
    o = r.route([Order("A", "BUY", 1_000)], snap, 20.0, date="2026-11-03")[0]
    assert o.meta["pct_of_auction_volume"] == pytest.approx(1_000 / 160_000)
    o_q = r.route([Order("A", "BUY", 1_000)], snap, 20.0, date="2026-03-20")[0]
    assert o_q.meta["pct_of_auction_volume"] == pytest.approx(1_000 / 200_000)  # quad witching
    # fill simulator: auction volume from the configured share
    sim = FillSimulator(cfg)
    realized = pd.DataFrame({"open": 50.0, "close": 50.0, "volume": 1e6, "sigma_daily": 0.02}, index=["A"])
    f = sim.simulate([Order("A", "BUY", 1_000, order_type="MOC")], realized, date="2026-11-03")
    assert f["impact_bps"].iloc[0] == pytest.approx(
        1e4 * cfg.impact_b_auction * 0.02 * np.sqrt(1_000 / 160_000)
    )
    br = DryRunBroker(cfg)
    oid = br.place_order(Order("A", "BUY", 1_000, order_type="MOC"))
    fb = br.simulate_fill(oid, ref_px=50.0, adv_shares=1e6, sigma_daily=0.02, ts="2026-11-03 16:00")
    assert fb["impact_bps"] == pytest.approx(1e4 * cfg.impact_b_auction * 0.02 * np.sqrt(1_000 / 160_000))
    # TCA: auction V = share * ADV when no share/venue volume is recorded
    led = TCALedger(cost_cfg=cfg)
    led.record_order(
        "o", "A", "buy", 1_000, 50.0, "2026-11-03 15:00", 50.0, "2026-11-03 15:50", venue="auction"
    )
    led.record_fill("o", 1_000, 50.05, "2026-11-03 16:00")
    led.close_order("o", 50.0, 50.0, adv_shares=1e6, sigma_daily=0.02)
    assert led.decompose().loc["o", "venue_volume"] == pytest.approx(160_000)


def test_vectorised_fees_match_scalar():
    fs = FeeSchedule(CostConfig())
    rng = np.random.default_rng(4)
    n = 400
    shares = np.concatenate([rng.integers(1, 200, n // 4), rng.integers(100, 120_000, 3 * n // 4)]).astype(
        float
    )
    price = np.concatenate([rng.uniform(0.005, 2.0, n // 8), rng.uniform(2, 500, n - n // 8)])
    sell = rng.random(n) < 0.5
    auc = rng.random(n) < 0.5
    for date in (None, "2021-06-15", "2026-06-15", "2026-11-03"):
        vec = fs.explicit_cost_vec(shares, price, sell, date, auc)
        ref = [
            fs.explicit_cost(q, p, "sell" if s else "buy", date, "auction" if a else "continuous")["total"]
            for q, p, s, a in zip(shares, price, sell, auc, strict=True)
        ]
        np.testing.assert_allclose(vec, ref, rtol=1e-13, atol=1e-12)
        mf = fs.marginal_fraction_vec(price, sell, date, auc)
        mref = [
            fs.marginal_fraction(p, "sell" if s else "buy", date, "auction" if a else None)
            for p, s, a in zip(price, sell, auc, strict=True)
        ]
        np.testing.assert_allclose(mf, mref, rtol=1e-13)
    assert fs.explicit_cost_vec([0.0, 10.0], [50.0, 0.0], False).tolist() == [0.0, 0.0]


def test_cvx_terms_solve(md):
    tcm = TransactionCostModel(CostConfig())
    date = md.dates[150]
    snap = tcm.snapshot(md, date)
    n = len(md.tickers)
    w_pre = np.zeros(n)
    x = cp.Variable(n)
    w_post = w_pre + x
    for venue in ("routed", "auction", "continuous"):
        trade, hold = tcm.cvx_terms(x, w_post, 5e6, snap, venue)
        assert trade.is_dcp() and trade.is_convex() and hold.is_convex()
    alpha = np.linspace(-0.01, 0.01, n)
    cap = tcm.participation_cap(snap, nav=5e6).values
    prob = cp.Problem(cp.Minimize(trade + hold - alpha @ x), [cp.norm1(x) <= 1.0, cp.abs(x) <= cap])
    prob.solve(solver="CLARABEL")
    assert prob.status in ("optimal", "optimal_inaccurate")
    assert np.all(np.abs(x.value) <= cap + 1e-6)
    assert isinstance(tcm, CostModel)


@pytest.mark.slow
def test_benchmark_vectorised_estimate_1500_names():
    """Performance (WORK_PLAN 1.6): the vectorised estimate vs the previous per-name
    Python loops (re-implemented here with the scalar APIs; results must agree)."""
    import time

    rng = np.random.default_rng(0)
    n = 1500
    idx = pd.Index([f"S{i:04d}" for i in range(n)])
    snap = LiquiditySnapshot(
        date=pd.Timestamp("2026-06-15"),
        tickers=idx,
        price=rng.uniform(5, 300, n),
        adv_usd=rng.uniform(2e6, 5e8, n),
        adv_shares=None,
        sigma_daily=rng.uniform(0.01, 0.04, n),
        half_spread=rng.uniform(1e-4, 1e-3, n),
        borrow_fee=0.005,
    )
    trades = pd.Series(rng.normal(0, 50_000, n), index=idx)
    trades[rng.random(n) < 0.1] = 0.0
    streak = pd.Series(rng.integers(0, 5, n).astype(float), index=idx)
    tcm = TransactionCostModel(CostConfig())

    def loop_reference():  # the pre-refactor algorithm: per-name scalar impact + fee calls
        P, A, S = snap.price.values, snap.adv_usd.values * snap.auction_share, snap.sigma_daily.values
        imp = [
            float(tcm.impact.cost_bps(q, v, s, venue="auction")) if q != 0 else 0.0
            for q, v, s in zip(trades.values, A, S, strict=True)
        ]
        mult = persistence_multiplier(streak.values)
        fees = [
            tcm.fees.explicit_bps(q, p, "buy" if q > 0 else "sell", snap.date, venue="auction")
            if q != 0
            else 0.0
            for q, p in zip(trades.values, P, strict=True)
        ]
        aq = trades.abs().values
        return np.array(imp) * mult * aq / 1e4, np.array(fees) * aq / 1e4

    def best(fn, reps=5):
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            out = fn()
            ts.append(time.perf_counter() - t0)
        return min(ts), out

    t_loop, (imp_ref, fee_ref) = best(loop_reference)
    t_vec, est = best(lambda: tcm.estimate(trades, snap, "auction", streak=streak))
    np.testing.assert_allclose(
        est["impact_usd"].values, np.where(trades.values != 0, imp_ref, 0.0), rtol=1e-10
    )
    np.testing.assert_allclose(est["fees_usd"].values, fee_ref, rtol=1e-10, atol=1e-12)
    x = cp.Variable(n)
    t_cvx, _ = best(lambda: tcm.cvx_terms(x, x, 1e7, snap, "routed"))
    print(
        f"\n[benchmark] estimate on {n} names: per-name loops {t_loop * 1e3:.1f} ms, "
        f"vectorised {t_vec * 1e3:.2f} ms ({t_loop / t_vec:.0f}x); cvx_terms build {t_cvx * 1e3:.1f} ms"
    )


# ----------------------------------------------------------------------------- raw prices across splits (#34)
def test_snapshot_adv_and_fees_use_as_traded_prices_across_splits():
    """ADV (USD) and per-share fees are computed from the as-traded price and the
    actual share volume, so they do not depend on how history is split-adjusted:

    * vs the same tape without splits, ``adv_usd`` and ``adv_shares * price`` are identical
      on every date, including the days straddling each split;
    * re-adjusting history (a vendor rescales the adjusted close by a constant after a
      later corporate action) changes neither ADV nor fees nor sigma;
    * per-share fees use the raw price: before a k:1 split the share count is 1/k of what
      the adjusted price would imply."""
    import dataclasses

    spec = {"n_stocks": 30, "n_days": 300, "seed": 7}
    base = generate(SyntheticSpec(**spec))
    md = generate(SyntheticSpec(**spec, n_splits=3))
    tcm = TransactionCostModel(CostConfig())
    events = md.extra["true_splits"].stack().dropna()
    assert len(events) == 3
    for (d_split, tk), k in events.items():
        pos = md.dates.get_loc(d_split)
        for d in md.dates[[pos - 2, pos - 1, pos, pos + 1, min(pos + 70, len(md.dates) - 1)]]:
            s, s0 = tcm.snapshot(md, d), tcm.snapshot(base, d)
            np.testing.assert_allclose(s.adv_usd.to_numpy(), s0.adv_usd.to_numpy(), rtol=1e-12)
            np.testing.assert_allclose(
                (s.adv_shares * s.price).to_numpy(), (s0.adv_shares * s0.price).to_numpy(), rtol=1e-12
            )
            np.testing.assert_allclose(s.sigma_daily.to_numpy(), s0.sigma_daily.to_numpy(), rtol=1e-12)
            assert s.price[tk] == pytest.approx(md.close_raw.loc[d, tk], rel=1e-12)
        # ADV in USD has no jump of size k across the split
        before, after = tcm.snapshot(md, md.dates[pos - 1]), tcm.snapshot(md, d_split)
        assert 0.8 < after.adv_usd[tk] / before.adv_usd[tk] < 1.25
        # per-share fees on a $1m buy the day before the split: shares = USD / raw price
        d = md.dates[pos - 1]
        snap = tcm.snapshot(md, d)
        trade = pd.Series(0.0, index=md.tickers)
        trade[tk] = 1e6
        fees = tcm.estimate(trade, snap, venue="continuous")["fees_usd"][tk]
        P_raw = md.close_raw.loc[d, tk]
        expected = tcm.fees.explicit_cost_vec(
            np.array([1e6 / P_raw]), np.array([P_raw]), np.array([False]), d, is_auction=False
        )[0]
        assert fees == pytest.approx(expected, rel=1e-12)
        P_adj = md.close.loc[d, tk]
        legacy = tcm.fees.explicit_cost_vec(
            np.array([1e6 / P_adj]), np.array([P_adj]), np.array([False]), d, is_auction=False
        )[0]
        assert P_raw == pytest.approx(k * P_adj) and legacy > fees  # adjusted price overstated shares by k

    # re-adjustment of history leaves every dollar quantity unchanged
    rescale = pd.Series(1.0, index=md.tickers)
    rescale[events.index.get_level_values(1)] = 0.37
    md_readj = dataclasses.replace(md, close=md.close * rescale)
    d = md.dates[-1]
    trade = pd.Series(np.linspace(-5e5, 5e5, md.tickers.size), index=md.tickers)
    for dd in (md.dates[150], d):
        a, b = tcm.snapshot(md, dd), tcm.snapshot(md_readj, dd)
        for f in ("adv_usd", "adv_shares", "price", "sigma_daily"):
            np.testing.assert_allclose(getattr(a, f).to_numpy(), getattr(b, f).to_numpy(), rtol=1e-12)
        for venue in ("continuous", "auction", "routed"):
            ea, eb = tcm.estimate(trade, a, venue=venue), tcm.estimate(trade, b, venue=venue)
            np.testing.assert_allclose(ea["fees_usd"].to_numpy(), eb["fees_usd"].to_numpy(), rtol=1e-12)
            np.testing.assert_allclose(ea.total.to_numpy(), eb.total.to_numpy(), rtol=1e-12)
