import numpy as np
import pandas as pd
import pytest

from qts.core.config import CostConfig, ExecutionConfig
from qts.execution import (
    DryRunBroker,
    ExecutionRouter,
    FillSimulator,
    Order,
    almgren_chriss_schedule,
    auction_deadline,
    intraday_volume_curve,
    pov_schedule,
    targets_to_orders,
    to_ib_order_dict,
    urgency_from_half_life,
    vwap_schedule,
)
from qts.tcost import TCALedger


# --------------------------------------------------------------------- orders
def test_targets_to_orders_rounding_and_dust():
    w_t = pd.Series({"A": 0.05, "B": -0.02, "C": 0.00001, "D": 0.0})
    w_c = pd.Series({"A": 0.00, "B": 0.00, "C": 0.0, "D": 0.01})
    px = pd.Series({"A": 33.33, "B": 100.0, "C": 50.0, "D": 10.0})
    orders = targets_to_orders(w_t, w_c, nav=1_000_000, prices=px, lot_size=1, min_notional=200)
    by = {o.ticker: o for o in orders}
    assert "C" not in by  # dust ($10) skipped
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
        Order("SMALL", "BUY", 1_000),  # 1.1% of auction -> LOC
        Order("SMALL", "SELL", 1_000, order_id="small-sell"),
        Order("BIG", "SELL", 10_000),  # 11% of auction -> split: LOC cap + ALGO rest
        Order("FAST", "BUY", 1_000, meta={"half_life_days": 2.0}),  # fast -> ArrivalPx
        Order("REV", "BUY", 1_000, meta={"sleeve": "reversal"}),  # reversal -> passive LMT
        Order("HUGE", "BUY", 100_000),  # 10% ADV -> split, ALGO child flagged
    ]
    routed = r.route(orders, snapshot, signal_half_life_days=20.0)
    assert len(routed) == 8
    small_b, small_s, big_auc, big, fast, rev, huge_auc, huge = routed
    # split (audit 2 #6): K = floor(2% x 90,000) = 1,800 shares to the auction, the rest to the algo
    assert big_auc.order_type == "LOC" and big_auc.qty == 1_800 and big_auc.limit_px < 100.0
    assert (
        big.qty == 10_000 - 1_800 and big_auc.meta["split"] == "auction" and big.meta["split"] == "continuous"
    )
    assert huge_auc.qty == 1_800 and huge.qty == 98_200 and huge_auc.side == huge.side == "BUY"
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
    # medium urgency -> Vwap rather than ClosePx (Vwap documents noTakeLiq; ClosePx does not)
    routed2 = r.route([Order("BIG", "SELL", 10_000)], snapshot, signal_half_life_days=6.0)[1:]  # algo child
    assert routed2[0].algo == "Vwap" and routed2[0].algo_params["noTakeLiq"] is False
    assert "noTakeLiq" not in big.algo_params
    # participation: urgency_from_half_life clipped to [1%, algo_max_pct_adv]
    assert routed2[0].algo_params["maxPctVol"] == pytest.approx(urgency_from_half_life(6.0))
    assert big.algo_params["maxPctVol"] == pytest.approx(urgency_from_half_life(20.0))
    assert fast.algo_params["maxPctVol"] == pytest.approx(cfg.algo_max_pct_adv)  # capped
    assert r.participation(None) == cfg.algo_max_pct_adv and r.participation(1e4) == pytest.approx(0.01)
    # all orders carry a reason string
    assert all(isinstance(o.meta.get("routing_reason"), str) and o.meta["routing_reason"] for o in routed)
    tbl = r.routing_table(routed)
    assert len(tbl) == 8 and "reason" in tbl
    # the old all-or-nothing routing is still available
    unsplit = ExecutionRouter(cfg, split_auction=False).route([Order("BIG", "SELL", 10_000)], snapshot, 20.0)
    assert len(unsplit) == 1 and unsplit[0].order_type == "ALGO" and unsplit[0].qty == 10_000


def test_router_split_matches_routed_cost_model():
    """The routed cost venue prices q_a = min(q, K) in the auction and the rest in the
    continuous session; the router's children carry exactly those quantities."""
    from qts.core.costs import LiquiditySnapshot
    from qts.tcost.model import TransactionCostModel

    px, adv_sh = 50.0, 2_000_000.0
    share = 0.09
    snap_r = {
        "adv_shares": pd.Series({"X": adv_sh}),
        "auction_volume_est": pd.Series({"X": adv_sh * share}),
        "sigma_daily": pd.Series({"X": 0.02}),
        "half_spread_bps": pd.Series({"X": 2.0}),
        "indicative_close_px": pd.Series({"X": px}),
    }
    q_sh = 20_000.0  # 11% of the auction
    kids = ExecutionRouter().route([Order("X", "BUY", q_sh, order_id="p")], snap_r, 20.0, date="2026-11-03")
    assert [k.order_id for k in kids] == ["p-auc", "p-algo"]
    assert sum(k.qty for k in kids) == q_sh and all(k.meta["parent_order_id"] == "p" for k in kids)
    tcm = TransactionCostModel()
    snap_c = LiquiditySnapshot(
        date=pd.Timestamp("2026-11-03"),
        tickers=pd.Index(["X"]),
        price=pd.Series({"X": px}),
        adv_usd=pd.Series({"X": adv_sh * px}),
        adv_shares=pd.Series({"X": adv_sh}),
        sigma_daily=pd.Series({"X": 0.02}),
        half_spread=pd.Series({"X": 2e-4}),
        borrow_fee=pd.Series({"X": 0.005}),
        auction_share=share,
    )
    est = tcm.estimate(pd.Series({"X": q_sh * px}), snap_c, venue="routed")
    assert est.auction_fraction["X"] * q_sh == pytest.approx(kids[0].qty, abs=1.0)  # whole shares


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
    snap = {
        "adv_shares": pd.Series({"P": 1e6}),
        "auction_volume_est": pd.Series({"P": 9e4}),
        "sigma_daily": pd.Series({"P": 0.02}),
        "half_spread_bps": pd.Series({"P": 3.0}),
        "indicative_close_px": pd.Series({"P": 7.0}),
    }
    routed = r.route([Order("P", "BUY", 100), Order("P", "SELL", 100)], snap, 20.0)
    assert routed[0].order_type == "LOC" and routed[0].limit_px > 7.0
    assert routed[1].order_type == "LOC" and routed[1].limit_px < 7.0


def test_order_ids_unique_across_days():
    w_t = pd.Series({"A": 0.05})
    w_c = pd.Series({"A": 0.0})
    px = pd.Series({"A": 50.0})
    o1 = targets_to_orders(w_t, w_c, 1e6, px, decision_ts="2026-06-01")[0]
    o2 = targets_to_orders(w_t, w_c, 1e6, px, decision_ts="2026-06-02")[0]
    assert o1.order_id != o2.order_id and o1.order_id.startswith("20260601-")
    # FillSimulator prefixes the date when the same id is replayed into one ledger
    led = TCALedger()
    sim = FillSimulator(CostConfig())
    realized = pd.DataFrame(
        {
            "open": 50.0,
            "close": 50.0,
            "next_open": 50.0,
            "volume": 1e6,
            "sigma_daily": 0.02,
            "half_spread_bps": 2.0,
            "adv_shares": 1e6,
        },
        index=["A"],
    )
    same = Order("A", "BUY", 100, order_id="A-BUY-100")
    sim.simulate([same], realized, date="2026-06-01", ledger=led)
    sim.simulate([same], realized, date="2026-06-02", ledger=led)
    assert len(led.orders) == 2
    d = led.decompose()
    assert (d["venue"] == "auction").all()


# ----------------------------------------------------------------------- ibkr
def test_ib_payloads(snapshot):
    r = ExecutionRouter(ExecutionConfig())
    routed = r.route(
        [
            Order("SMALL", "BUY", 1_000, order_id="o1"),
            Order("BIG", "SELL", 10_000, order_id="o2"),
            Order("FAST", "BUY", 1_000, order_id="o3", meta={"half_life_days": 1.0}),
        ],
        snapshot,
        signal_half_life_days=20.0,
    )
    assert [o.order_id for o in routed] == ["o1", "o2-auc", "o2-algo", "o3"]
    assert to_ib_order_dict(routed[1])["orderType"] == "LOC"
    routed = [routed[0], routed[2], routed[3]]
    loc = to_ib_order_dict(routed[0])
    assert loc["orderType"] == "LOC" and loc["action"] == "BUY" and loc["totalQuantity"] == 1000
    assert loc["lmtPrice"] == routed[0].limit_px and loc["tif"] == "DAY" and loc["orderRef"] == "o1"
    assert loc["contract"]["symbol"] == "SMALL" and loc["contract"]["secType"] == "STK"
    algo = to_ib_order_dict(routed[1])
    assert algo["algoStrategy"] in ("ClosePx", "Vwap") and algo["orderType"] == "MKT"
    tags = {p["tag"]: p["value"] for p in algo["algoParams"]}
    assert "maxPctVol" in tags and float(tags["maxPctVol"]) <= 0.05
    # ClosePx documents maxPctVol/riskAversion/startTime/forceCompletion only: no noTakeLiq (#22)
    assert algo["algoStrategy"] == "ClosePx" and "noTakeLiq" not in tags
    # participation from the signal half-life (urgency_from_half_life, #21)
    assert float(tags["maxPctVol"]) == pytest.approx(urgency_from_half_life(20.0))
    arr = to_ib_order_dict(routed[2])
    assert arr["algoStrategy"] == "ArrivalPx"
    arr_tags = {p["tag"]: p["value"] for p in arr["algoParams"]}
    assert "riskAversion" in arr_tags and "maxPctVol" in arr_tags
    moc = to_ib_order_dict(Order("X", "SELL", 5, order_type="MOC"))
    assert moc["orderType"] == "MOC" and "lmtPrice" not in moc
    with pytest.raises(ValueError):
        to_ib_order_dict(Order("X", "BUY", 5, order_type="LOC"))  # no limit


def test_auction_deadline():
    d = auction_deadline("2026-11-03")
    assert d.hour == 15 and d.minute == 50 and str(d.tz) == "America/New_York"
    assert d.tz_convert("UTC").hour == 20  # EST


def test_dry_run_broker():
    br = DryRunBroker(CostConfig())
    oid = br.place_order(Order("A", "BUY", 1_000, order_type="MOC", order_id="a1"))
    f = br.simulate_fill(oid, ref_px=50.0, adv_shares=1e6, sigma_daily=0.02, ts="2026-11-03 16:00")
    assert f["qty"] == 1000 and f["px"] > 50.0 and f["impact_bps"] > 0
    oid2 = br.place_order(Order("A", "BUY", 1_000, order_type="LOC", limit_px=50.0, order_id="a2"))
    f2 = br.simulate_fill(oid2, ref_px=50.0, adv_shares=1e6, sigma_daily=0.02)
    assert f2["qty"] == 0  # impact pushes price through the limit


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
    assert ac["trades"][0] > ac["trades"][-1]  # front-loaded when risk-averse
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
    realized = pd.DataFrame(
        {
            "open": 100.0,
            "close": [100.02, 100.1, 100.1, 99.5, 100.1],
            "next_open": 100.3,
            "volume": 1_000_000.0,
            "sigma_daily": 0.02,
            "half_spread_bps": 3.0,
            "adv_shares": 1_000_000.0,
            "next_close": 100.25,
        },
        index=["SMALL", "BIG", "FAST", "REV", "HUGE"],
    )
    led = TCALedger()
    sim = FillSimulator(CostConfig())
    fills = sim.simulate(routed, realized, date="2026-11-03", ledger=led)
    assert len(fills) == 7  # BIG and HUGE are split into auction + algo children
    f = fills.set_index("order_id")
    assert f.loc["b1-auc", "status"] == "filled" and f.loc["b1-auc", "filled_qty"] == 1_800
    assert f.loc["s1", "status"] == "filled" and f.loc["s1", "fill_px"] > 100.02  # auction impact on buy
    assert f.loc["s1", "fill_px"] <= routed[0].limit_px
    assert f.loc["b1-algo", "status"] == "filled" and f.loc["b1-algo", "fill_px"] < 100.1
    rev = next(o for o in routed if o.order_id == "r1")
    assert f.loc["r1", "status"] == "filled" and f.loc["r1", "fill_px"] == rev.limit_px
    # HUGE's algo child (10% ADV, half-life 20d) is capped at urgency_from_half_life(20) ~1.3% of volume
    assert f.loc["h1-algo", "status"] == "partial"
    # partial fills are whole shares: floor of the participation cap
    assert f.loc["h1-algo", "filled_qty"] == np.floor(urgency_from_half_life(20.0) * 1_000_000)
    assert (f["fees_usd"] >= 0).all() and f.loc["s1", "fees_usd"] > 0
    d = led.decompose()
    assert len(d) == 7
    cols = ["delay_bps", "impact_bps", "opportunity_bps", "fees_bps", "total_is_bps", "reversion_bps"]
    assert np.isfinite(d[cols].values).all()
    assert np.allclose(
        d["total_is_bps"], d[["delay_bps", "impact_bps", "opportunity_bps", "fees_bps"]].sum(axis=1)
    )
    assert d.loc["h1-algo", "opportunity_bps"] > 0  # unfilled half, price rose
    b, diag = led.calibrate_b(prior_b=1.0, prior_weight=20)
    assert np.isfinite(b) and diag["n"] >= 1
    summ = led.summary()
    assert not summ.empty
    # LOC violated when the auction prints through the limit
    tight = routed[0].with_(limit_px=100.0)
    fills2 = sim.simulate([tight], realized, date="2026-11-03")
    assert fills2.iloc[0]["status"] == "unfilled"


def test_fill_simulator_split_algo_child_priced_as_increment():
    """P14: the algo child of a split is charged C(q_a + q_c) - C(q_a) (routed cost
    model), partial fills are whole shares, ArrivalPx says it fills at the open."""
    from qts.execution.orders import Order as O

    cfg = CostConfig()
    sim = FillSimulator(cfg)
    realized = pd.DataFrame(
        {"open": 100.0, "close": 100.0, "volume": 1_000_000.0, "sigma_daily": 0.02, "half_spread_bps": 0.0},
        index=["X"],
    )
    auc = O(
        "X",
        "BUY",
        2_000,
        order_type="MOC",
        order_id="p-auc",
        meta={"split": "auction", "parent_order_id": "p", "parent_qty": 10_000},
    )
    algo = O(
        "X",
        "BUY",
        8_000,
        order_type="ALGO",
        algo="Vwap",
        order_id="p-algo",
        algo_params={"maxPctVol": 0.05},
        meta={"split": "continuous", "parent_order_id": "p", "parent_qty": 10_000},
    )
    f = sim.simulate([auc, algo], realized, date="2026-11-03").set_index("order_id")
    # hand computation: C(Q) = b_c * sigma * Q^1.5 / V^0.5 in USD, V = 1e8 USD, b_c = 1.0
    b, sig, V = cfg.impact_b_continuous, 0.02, 1e8
    C = lambda q: b * sig * q**1.5 / V**0.5  # noqa: E731
    inc_bps = 1e4 * (C(1e6) - C(2e5)) / 8e5  # q_a = 2,000 sh, q_c = 8,000 sh at 100
    standalone_bps = 1e4 * C(8e5) / 8e5
    assert f.loc["p-algo", "impact_bps"] == pytest.approx(inc_bps, rel=1e-12)
    assert inc_bps > standalone_bps  # superadditive: splitting is not free
    assert f.loc["p-algo", "fill_px"] == pytest.approx(100.0 * (1 + inc_bps / 1e4), rel=1e-12)
    assert "increment over 2000 sh auction child" in f.loc["p-algo", "reason"]
    # an unfilled auction child (LOC through its limit) leaves no metaorder to extend
    loc = auc.with_(order_type="LOC", limit_px=99.0)
    f2 = sim.simulate([loc, algo], realized, date="2026-11-03").set_index("order_id")
    assert f2.loc["p-auc", "status"] == "unfilled"
    assert f2.loc["p-algo", "impact_bps"] == pytest.approx(standalone_bps, rel=1e-12)
    # without the auction child in the batch the planned q_a = parent_qty - qty is used
    f3 = sim.simulate([algo], realized, date="2026-11-03").set_index("order_id")
    assert f3.loc["p-algo", "impact_bps"] == pytest.approx(inc_bps, rel=1e-12)
    # partial fill floored to whole shares; ArrivalPx fills at the open
    arr = O(
        "X",
        "SELL",
        60_000,
        order_type="ALGO",
        algo="ArrivalPx",
        order_id="a",
        algo_params={"maxPctVol": 0.0123456},
    )
    g = sim.simulate([arr], realized, date="2026-11-03").iloc[0]
    assert g["status"] == "partial" and g["filled_qty"] == 12_345.0
    assert g["reason"].startswith("ArrivalPx at arrival (open)")
