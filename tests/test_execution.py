import numpy as np
import pandas as pd
import pytest

from qts.core.config import CostConfig, ExecutionConfig
from qts.execution import (DryRunBroker, ExecutionRouter, FillSimulator, Order, almgren_chriss_schedule,
                           auction_deadline, intraday_volume_curve, pov_schedule, targets_to_orders,
                           to_ib_order_dict, urgency_from_half_life, vwap_schedule)
from qts.tcost import TCALedger


# --------------------------------------------------------------------- orders
def test_targets_to_orders_rounding_and_dust():
    w_t = pd.Series({"A": 0.05, "B": -0.02, "C": 0.00001, "D": 0.0})
    w_c = pd.Series({"A": 0.00, "B": 0.00, "C": 0.0, "D": 0.01})
    px = pd.Series({"A": 33.33, "B": 100.0, "C": 50.0, "D": 10.0})
    orders = targets_to_orders(w_t, w_c, nav=1_000_000, prices=px, lot_size=1, min_notional=200)
    by = {o.ticker: o for o in orders}
    assert "C" not in by                      # dust ($10) skipped
    assert by["A"].side == "BUY" and by["A"].qty == np.floor(50_000 / 33.33)
    assert by["B"].side == "SELL" and by["B"].qty == 200
    assert by["D"].side == "SELL" and by["D"].qty == 1000
    assert by["A"].meta["decision_px"] == 33.33
    with pytest.raises(ValueError):
        Order("A", "HOLD", 1)


# --------------------------------------------------------------------- router
@pytest.fixture
def snapshot():
    t = ["SMALL", "BIG", "FAST", "REV", "HUGE"]
    return {
        "adv_shares": pd.Series(1_000_000.0, index=t),
        "auction_volume_est": pd.Series(90_000.0, index=t),
        "sigma_daily": pd.Series(0.02, index=t),
        "half_spread_bps": pd.Series(3.0, index=t),
        "indicative_close_px": pd.Series(100.0, index=t),
    }


def test_router_tactics(snapshot):
    cfg = ExecutionConfig()
    r = ExecutionRouter(cfg)
    orders = [
        Order("SMALL", "BUY", 1_000),                                            # 1.1% of auction -> LOC
        Order("SMALL", "SELL", 1_000, order_id="small-sell"),
        Order("BIG", "SELL", 10_000),                                            # 11% of auction -> ALGO
        Order("FAST", "BUY", 1_000, meta={"half_life_days": 2.0}),               # fast -> ArrivalPx
        Order("REV", "BUY", 1_000, meta={"sleeve": "reversal"}),                 # reversal -> passive LMT
        Order("HUGE", "BUY", 100_000),                                           # 10% ADV -> ALGO flagged
    ]
    routed = r.route(orders, snapshot, signal_half_life_days=20.0)
    assert len(routed) == 6
    small_b, small_s, big, fast, rev, huge = routed
    assert small_b.order_type == "LOC" and small_b.limit_px > 100.0
    exp_imp_bps = 1e4 * CostConfig().impact_b_auction * 0.02 * np.sqrt(1_000 / 90_000)
    assert small_b.meta["loc_offset_bps"] == pytest.approx(max(3.0, 1.5 * exp_imp_bps))
    assert small_s.order_type == "LOC" and small_s.limit_px < 100.0
    # with the impact floor disabled the limit is the fixed cfg offset through the indicative
    fixed = ExecutionRouter(cfg, loc_offset_impact_mult=0.0).route(orders[:2], snapshot, 20.0)
    assert fixed[0].limit_px == pytest.approx(100 * (1 + 3e-4), abs=0.005)
    assert fixed[1].limit_px == pytest.approx(100 * (1 - 3e-4), abs=0.005)
    assert "LOC" in small_b.meta["routing_reason"]
    assert big.order_type == "ALGO" and big.algo in ("ClosePx", "Vwap")
    assert big.algo_params["maxPctVol"] <= 0.05
    assert fast.order_type == "ALGO" and fast.algo == "ArrivalPx" and fast.venue_hint == "open"
    assert rev.order_type == "LMT" and rev.limit_px < 100.0 and rev.algo is None
    assert huge.order_type == "ALGO" and huge.meta.get("multi_day") is True
    assert "WARNING" in huge.meta["routing_reason"]
    # medium urgency -> Vwap rather than ClosePx
    routed2 = r.route([Order("BIG", "SELL", 10_000)], snapshot, signal_half_life_days=6.0)
    assert routed2[0].algo == "Vwap" and routed2[0].algo_params["noTakeLiq"] is False
    # all orders carry a reason string
    assert all(isinstance(o.meta.get("routing_reason"), str) and o.meta["routing_reason"] for o in routed)
    tbl = r.routing_table(routed)
    assert len(tbl) == 6 and "reason" in tbl


def test_loc_limit_rounds_to_tick_in_favourable_direction():
    r = ExecutionRouter(ExecutionConfig(), loc_offset_impact_mult=0.0)
    # 3 bp on a $7 stock is $0.0021: must still move one tick through the indicative
    assert r.loc_limit("BUY", 7.0, 3.0) == pytest.approx(7.01)
    assert r.loc_limit("SELL", 7.0, 3.0) == pytest.approx(6.99)
    assert r.loc_limit("BUY", 5.0, 3.0) > 5.0 and r.loc_limit("SELL", 10.0, 3.0) < 10.0
    assert r.loc_limit("BUY", 100.0, 3.0) == pytest.approx(100.03)
    # sub-dollar tick is $0.0001
    assert r.loc_limit("BUY", 0.5, 3.0) == pytest.approx(0.5002)
    assert r.loc_limit("SELL", 0.5, 3.0) == pytest.approx(0.4998)
    # passive limits round away from the indicative (stay passive)
    assert r.passive_limit("BUY", 7.0, 3.0) == pytest.approx(6.99)
    assert r.passive_limit("SELL", 7.0, 3.0) == pytest.approx(7.01)
    snap = {"adv_shares": pd.Series({"P": 1e6}), "auction_volume_est": pd.Series({"P": 9e4}),
            "sigma_daily": pd.Series({"P": 0.02}), "half_spread_bps": pd.Series({"P": 3.0}),
            "indicative_close_px": pd.Series({"P": 7.0})}
    routed = r.route([Order("P", "BUY", 100), Order("P", "SELL", 100)], snap, 20.0)
    assert routed[0].order_type == "LOC" and routed[0].limit_px > 7.0
    assert routed[1].order_type == "LOC" and routed[1].limit_px < 7.0


def test_order_ids_unique_across_days():
    w_t = pd.Series({"A": 0.05}); w_c = pd.Series({"A": 0.0}); px = pd.Series({"A": 50.0})
    o1 = targets_to_orders(w_t, w_c, 1e6, px, decision_ts="2026-06-01")[0]
    o2 = targets_to_orders(w_t, w_c, 1e6, px, decision_ts="2026-06-02")[0]
    assert o1.order_id != o2.order_id and o1.order_id.startswith("20260601-")
    # FillSimulator prefixes the date when the same id is replayed into one ledger
    led = TCALedger()
    sim = FillSimulator(CostConfig())
    realized = pd.DataFrame({"open": 50.0, "close": 50.0, "next_open": 50.0, "volume": 1e6,
                             "sigma_daily": 0.02, "half_spread_bps": 2.0, "adv_shares": 1e6}, index=["A"])
    same = Order("A", "BUY", 100, order_id="A-BUY-100")
    sim.simulate([same], realized, date="2026-06-01", ledger=led)
    sim.simulate([same], realized, date="2026-06-02", ledger=led)
    assert len(led.orders) == 2
    d = led.decompose()
    assert (d["venue"] == "auction").all()


# ----------------------------------------------------------------------- ibkr
def test_ib_payloads(snapshot):
    r = ExecutionRouter(ExecutionConfig())
    routed = r.route([Order("SMALL", "BUY", 1_000, order_id="o1"),
                      Order("BIG", "SELL", 10_000, order_id="o2"),
                      Order("FAST", "BUY", 1_000, order_id="o3", meta={"half_life_days": 1.0})],
                     snapshot, signal_half_life_days=20.0)
    loc = to_ib_order_dict(routed[0])
    assert loc["orderType"] == "LOC" and loc["action"] == "BUY" and loc["totalQuantity"] == 1000
    assert loc["lmtPrice"] == routed[0].limit_px and loc["tif"] == "DAY" and loc["orderRef"] == "o1"
    assert loc["contract"]["symbol"] == "SMALL" and loc["contract"]["secType"] == "STK"
    algo = to_ib_order_dict(routed[1])
    assert algo["algoStrategy"] in ("ClosePx", "Vwap") and algo["orderType"] == "MKT"
    tags = {p["tag"]: p["value"] for p in algo["algoParams"]}
    assert "maxPctVol" in tags and float(tags["maxPctVol"]) <= 0.05
    assert tags.get("noTakeLiq") in ("0", "1")
    arr = to_ib_order_dict(routed[2])
    assert arr["algoStrategy"] == "ArrivalPx"
    arr_tags = {p["tag"]: p["value"] for p in arr["algoParams"]}
    assert "riskAversion" in arr_tags and "maxPctVol" in arr_tags
    moc = to_ib_order_dict(Order("X", "SELL", 5, order_type="MOC"))
    assert moc["orderType"] == "MOC" and "lmtPrice" not in moc
    with pytest.raises(ValueError):
        to_ib_order_dict(Order("X", "BUY", 5, order_type="LOC"))        # no limit


def test_auction_deadline():
    d = auction_deadline("2026-11-03")
    assert d.hour == 15 and d.minute == 50 and str(d.tz) == "America/New_York"
    assert d.tz_convert("UTC").hour == 20   # EST


def test_dry_run_broker():
    br = DryRunBroker(CostConfig())
    oid = br.place_order(Order("A", "BUY", 1_000, order_type="MOC", order_id="a1"))
    f = br.simulate_fill(oid, ref_px=50.0, adv_shares=1e6, sigma_daily=0.02, ts="2026-11-03 16:00")
    assert f["qty"] == 1000 and f["px"] > 50.0 and f["impact_bps"] > 0
    oid2 = br.place_order(Order("A", "BUY", 1_000, order_type="LOC", limit_px=50.0, order_id="a2"))
    f2 = br.simulate_fill(oid2, ref_px=50.0, adv_shares=1e6, sigma_daily=0.02)
    assert f2["qty"] == 0          # impact pushes price through the limit


# ------------------------------------------------------------------ scheduler
def test_scheduler():
    c = intraday_volume_curve()
    assert len(c) == 13 and c.sum() == pytest.approx(1.0) and c[-1] > c[6] and c[0] > c[6]
    c2 = intraday_volume_curve(close_share=0.3)
    assert c2[-1] == pytest.approx(0.3) and c2.sum() == pytest.approx(1.0)
    v = vwap_schedule(10_000, c)
    assert v.sum() == pytest.approx(10_000)
    p = pov_schedule(10_000, np.full(13, 20_000.0), pct=0.05)
    assert p.sum() == pytest.approx(10_000) and p.max() <= 1000 + 1e-9
    assert urgency_from_half_life(1.0) > urgency_from_half_life(20.0)
    assert 0.01 <= urgency_from_half_life(100.0) <= 0.25
    ac = almgren_chriss_schedule(10_000, 10, sigma=0.5, eta=1e-3, lam=1e-4)
    assert ac["holdings"][0] == pytest.approx(10_000) and ac["holdings"][-1] == pytest.approx(0.0, abs=1e-9)
    assert ac["trades"].sum() == pytest.approx(10_000)
    assert ac["trades"][0] > ac["trades"][-1]                    # front-loaded when risk-averse
    twap = almgren_chriss_schedule(10_000, 10, sigma=0.5, eta=1e-3, lam=0.0)
    assert np.allclose(twap["trades"], 1000)


# ------------------------------------------------- fill simulator + ledger
def test_fill_simulator_roundtrip_with_ledger(snapshot):
    cfg = ExecutionConfig()
    r = ExecutionRouter(cfg)
    orders = [
        Order("SMALL", "BUY", 1_000, order_id="s1", meta={"decision_px": 99.5}),
        Order("BIG", "SELL", 10_000, order_id="b1", meta={"decision_px": 100.2}),
        Order("FAST", "BUY", 1_000, order_id="f1", meta={"half_life_days": 1.0, "decision_px": 100.0}),
        Order("REV", "BUY", 1_000, order_id="r1", meta={"sleeve": "reversal", "decision_px": 100.0}),
        Order("HUGE", "BUY", 100_000, order_id="h1", meta={"decision_px": 100.0}),
    ]
    routed = r.route(orders, snapshot, signal_half_life_days=20.0)
    realized = pd.DataFrame({
        "open": 100.0, "close": [100.02, 100.1, 100.1, 99.5, 100.1], "next_open": 100.3,
        "volume": 1_000_000.0, "sigma_daily": 0.02, "half_spread_bps": 3.0, "adv_shares": 1_000_000.0,
    }, index=["SMALL", "BIG", "FAST", "REV", "HUGE"])
    led = TCALedger()
    sim = FillSimulator(CostConfig())
    fills = sim.simulate(routed, realized, date="2026-11-03", ledger=led)
    assert len(fills) == 5
    f = fills.set_index("order_id")
    assert f.loc["s1", "status"] == "filled" and f.loc["s1", "fill_px"] > 100.02       # auction impact on buy
    assert f.loc["s1", "fill_px"] <= routed[0].limit_px
    assert f.loc["b1", "status"] == "filled" and f.loc["b1", "fill_px"] < 100.1
    assert f.loc["r1", "status"] == "filled" and f.loc["r1", "fill_px"] == routed[3].limit_px
    assert f.loc["h1", "status"] == "partial" and f.loc["h1", "filled_qty"] == pytest.approx(50_000)
    assert (f["fees_usd"] >= 0).all() and f.loc["s1", "fees_usd"] > 0
    d = led.decompose()
    assert len(d) == 5
    cols = ["delay_bps", "impact_bps", "opportunity_bps", "fees_bps", "total_is_bps", "reversion_bps"]
    assert np.isfinite(d[cols].values).all()
    assert np.allclose(d["total_is_bps"], d[["delay_bps", "impact_bps", "opportunity_bps", "fees_bps"]].sum(axis=1))
    assert d.loc["h1", "opportunity_bps"] > 0            # unfilled half, price rose
    b, diag = led.calibrate_b(prior_b=1.0, prior_weight=20)
    assert np.isfinite(b) and diag["n"] >= 1
    summ = led.summary()
    assert not summ.empty
    # LOC violated when the auction prints through the limit
    tight = routed[0].with_(limit_px=100.0)
    fills2 = sim.simulate([tight], realized, date="2026-11-03")
    assert fills2.iloc[0]["status"] == "unfilled"
