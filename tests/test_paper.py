"""Paper-trading book (``qts.paper``): batch-vs-paper equivalence, point in time,
corporate events, accounting, idempotence, catch-up, lock, replay, stop rules,
the mocked Sharadar refresh and the CLI.

The synthetic market is small (50 names x 520 days) with a shortened walk-forward
schedule (refit every 40 days, ridge sleeve) so the ML refit crosses the 40-day
paper window; a split, a dividend, a halt and a delisting are planted on names the
batch book holds in the window.
"""

from __future__ import annotations

import itertools
import json
import shutil
import warnings
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qts.alpha.models import RidgeModel
from qts.core.config import SystemConfig
from qts.core.types import MarketData
from qts.data.synthetic import SyntheticSpec, generate
from qts.paper import expectations as X
from qts.paper import runner
from qts.paper.book import (
    BookLock,
    BookLockedError,
    DataSpec,
    PaperError,
    PaperPaths,
    StaleDataError,
    read_journal,
)
from qts.paper.data import MemorySource, NYSECalendar, check_freshness, expected_last_session
from qts.paper.monitor import StopRules, evaluate
from qts.paper.serving import fit_signals
from qts.pipeline import ResearchPipeline

SPEC = {"n_stocks": 50, "n_days": 520, "seed": 3}
OVERRIDES = {
    "universe.top_n_by_mcap": 40,
    "alpha.train_min_days": 150,
    "alpha.refit_frequency_days": 40,
    "alpha.valid_days": 30,
    "alpha.horizon_days": 10,
    "alpha.secondary_horizons": [5, 10],
    "alpha.ic_window": 100,
    "alpha.ic_min_obs": 20,
    "alpha.half_life_lookback_days": 100,
    "risk.min_history_days": 100,
    "risk.statistical_window": 100,
}
NAV0 = 5e6
LATE = datetime(2030, 1, 2, 12, tzinfo=timezone.utc)  # long after every synthetic date
WINDOW = 41  # execution dates: 40 return days
DIV_YIELD = 0.03
EV = {"dividend": 8, "split": 10, "halt": (15, 5), "delist": 25}  # offsets into the window


def _quiet():
    warnings.simplefilter("ignore", RuntimeWarning)
    warnings.simplefilter("ignore", FutureWarning)


def cfg_max_weight() -> float:
    return float(_cfg().portfolio.max_position_weight)


def _cfg() -> SystemConfig:
    return SystemConfig.from_yaml("configs/baseline.yaml").with_overrides(OVERRIDES)


def _plant(md: MarketData, dates: pd.DatetimeIndex, names: dict[str, str]) -> MarketData:
    """Split 2:1, dividend, 5-day halt and a delisting (performance) on held names."""
    pos = {k: int(md.dates.get_loc(dates[v if isinstance(v, int) else v[0]])) for k, v in EV.items()}
    close, raw = md.close.copy(), md.close_raw.copy()
    vol, mcap = md.volume.copy(), md.market_cap.copy()
    sh = md.shares_outstanding.copy()
    a, b, c, d = names["split"], names["dividend"], names["halt"], names["delist"]
    p = pos["split"]  # 2:1 split effective on the open of dates[split]: raw halves, shares double
    raw.iloc[:p, raw.columns.get_loc(a)] *= 2.0
    vol.iloc[:p, vol.columns.get_loc(a)] /= 2.0
    sh.iloc[:p, sh.columns.get_loc(a)] /= 2.0
    p = pos[
        "dividend"
    ]  # ex-date dates[dividend]: the raw close drops by the yield, the total return does not
    raw.iloc[:p, raw.columns.get_loc(b)] /= 1.0 - DIV_YIELD
    p, n = pos["halt"], EV["halt"][1]
    for f in (close, raw, vol, mcap):
        f.iloc[p : p + n, f.columns.get_loc(c)] = np.nan
    p = pos["delist"]
    for f in (close, raw, vol, mcap):
        f.iloc[p:, f.columns.get_loc(d)] = np.nan
    dd = pd.Series(pd.NaT, index=md.tickers, dtype="datetime64[ns]")
    dd[d] = md.dates[p - 1]
    reasons = pd.Series({d: "performance"})
    out = replace(
        md,
        close=close,
        close_raw=raw,
        volume=vol,
        market_cap=mcap,
        shares_outstanding=sh,
        delisting_dates=dd,
        delisting_reasons=reasons,
    )
    out.validate()
    return out


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """Market with planted events, the batch book on it, expectations and a config file."""
    _quiet()
    tmp = tmp_path_factory.mktemp("paper")
    cfg = _cfg()
    cfg_path = tmp / "cfg.yaml"
    cfg_path.write_text(cfg.to_yaml())
    md0 = generate(SyntheticSpec(**SPEC))
    pipe0 = ResearchPipeline(cfg, use_ml=True, ml_factory=RidgeModel, nav0=NAV0)
    pipe0.fit(md0)
    dates = pipe0.backtest_dates()[-WINDOW:]
    bt0 = pipe0.run(md0, dates=dates).backtest
    W = bt0.weights
    held = W.columns[(W.abs() > 0.004).all()]
    longs = [t for t in held if (W[t] > 0).all()]
    shorts = [t for t in held if (W[t] < 0).all()]
    names = {"split": longs[0], "dividend": longs[1], "delist": longs[2], "halt": shorts[0]}
    md = _plant(md0, dates, names)
    pipe = ResearchPipeline(cfg, use_ml=True, ml_factory=RidgeModel, nav0=NAV0)
    full = pipe.run(md).backtest  # whole out-of-sample history: the expectation source
    exp = X.from_backtest(full, horizon_days=10)
    info, rows = X.calibrate(exp.returns.to_numpy(), StopRules())  # once: every init reuses it
    exp = X.Expectations({**exp.summary, "calibration": info}, pd.concat([exp.table, rows]), exp.returns)
    S = md.dates[md.dates.get_loc(dates[0]) - 1]
    return {
        "md": md,
        "md0": md0,
        "dates": dates,
        "start": str(S.date()),
        "names": names,
        "exp": exp,
        "cfg_path": str(cfg_path),
        "tmp": tmp,
        "until": str(dates[-1].date()),
    }


def _init(world, root, name, sizing="replica", rules=None, source=None, md=None):
    src = source or MemorySource((md if md is not None else world["md"]).slice_until(world["dates"][-1]))
    runner.init_book(
        name,
        world["cfg_path"],
        nav=NAV0,
        start=world["start"],
        paper_root=root,
        data=DataSpec(kind="memory"),
        sizing=sizing,
        calendar="index",
        expectations=world["exp"],
        source=src,
        rules=rules,
    )
    return src


def _run(name, root, src, **kw):
    kw.setdefault("until", None)
    kw.setdefault("now", LATE)
    return runner.run_book(
        name, paper_root=root, source=src, inner_factory=RidgeModel, progress=lambda m: None, **kw
    )


def _records(root, name):
    from qts.paper.book import effective_records

    raw, _ = read_journal(PaperPaths(Path(root) / name))
    return effective_records(raw)


def _events(recs, kind):
    return [e for r in recs for e in (r.get("events") or {}).get(kind, [])]


@pytest.fixture(scope="module")
def replica_daily(world):
    """The key run: replica sizing, a fresh fit on data through every signal date."""
    _quiet()
    root = world["tmp"] / "books"
    src = _init(world, root, "replica")
    rep = _run("replica", root, src, until=world["until"])
    return root, src, rep


# =========================================================================== equivalence
def test_replica_book_matches_the_batch_backtest(world, replica_daily):
    root, src, rep = replica_daily
    assert len(rep["days"]) == WINDOW + 1  # the start day + 41 execution days
    rc = runner.reconcile("replica", paper_root=root, source=src, inner_factory=RidgeModel)
    # same targets within solver tolerance (Clarabel's optimum moves ~1e-5 for a 1e-16 input
    # change, which can flip a min_trade_weight = 5e-4 drop decision), same P&L
    assert rc["max_abs_weight_diff"] < 2e-3
    assert rc["max_rel_nav_diff"] < 2e-4
    assert rc["daily_return_corr"] > 0.9999
    assert rc["max_abs_daily_return_diff"] < 1e-4
    assert abs(rc["cum_return_paper"] - rc["cum_return_batch"]) < 2e-4
    recs = _records(root, "replica")
    # the window crossed a walk-forward refit: one fold was trained on a job day inside it
    trained_days = [r["date"] for r in recs if (r.get("models") or {}).get("folds_trained")]
    assert any(d > recs[1]["date"] for d in trained_days)
    # rebalance decisions every 5 execution dates from the first one
    reb = [r["date"] for r in recs if r.get("rebalance_decision")]
    assert reb[0] == str(world["dates"][0].date())
    assert len(reb) == (WINDOW + 4) // 5


def test_planted_events_are_booked(world, replica_daily):
    root, _, _ = replica_daily
    recs = _records(root, "replica")
    n = world["names"]
    splits = _events(recs, "splits")
    assert [e["ticker"] for e in splits] == [n["split"]]
    assert splits[0]["ratio"] == pytest.approx(2.0, rel=1e-9)
    divs = [e for e in _events(recs, "distributions") if not e.get("missing")]
    assert [e["ticker"] for e in divs] == [n["dividend"]]
    assert divs[0]["factor"] == pytest.approx(1.0 / (1.0 - DIV_YIELD), rel=1e-9)
    opened = _events(recs, "halts_opened")
    closed = _events(recs, "halts_closed")
    assert n["halt"] in [e["ticker"] for e in opened]
    assert any(e["ticker"] == n["halt"] and e["outcome"] == "resumed" for e in closed)
    dl = _events(recs, "delistings")
    assert [e["ticker"] for e in dl] == [n["delist"]]
    assert dl[0]["reason"] == "performance" and dl[0]["haircut_return"] == pytest.approx(-0.30)
    # the delisted name is gone from the book afterwards
    last = recs[-1]["state"]["shares"]
    assert n["delist"] not in last
    # split: implied shares doubled on the split day, value continuous
    i = next(k for k, r in enumerate(recs) if any(e["ticker"] == n["split"] for e in r["events"]["splits"]))
    before, after = recs[i - 1]["state"], recs[i]["state"]
    assert after["shares"][n["split"]] / before["shares"][n["split"]] == pytest.approx(2.0, rel=0.05)


def test_accounting_identity_and_pnl_split(replica_daily):
    root, _, _ = replica_daily
    recs = _records(root, "replica")
    for prev, r in itertools.pairwise(recs):
        st = r["state"]
        v = sum(st["shares"][k] * st["px"][k] for k in st["shares"])
        assert st["cash"] + v == pytest.approx(r["nav"], rel=1e-10)
        p = r["pnl"]
        assert r["nav"] - prev["nav"] == pytest.approx(p["total"], abs=1e-6 * NAV0)
        assert p["trading_cost"] == pytest.approx(p["spread"] + p["impact"] + p["fees"])
        assert r["nav_pre"] == pytest.approx(prev["nav"] + p["total"] + p["trading_cost"], abs=1e-6 * NAV0)


def test_single_fit_serving_matches_daily_fits(world, replica_daily, tmp_path):
    """``--signal-mode single`` (one fit through the last day) books the same day-by-day
    as a fresh fit per signal date: the fits are point in time."""
    _quiet()
    root_d, _, _ = replica_daily
    src = _init(world, tmp_path, "single")
    _run("single", tmp_path, src, until=world["until"], signal_mode="single")
    a, b = _records(root_d, "replica"), _records(tmp_path, "single")
    nav_a = np.array([r["nav"] for r in a])
    nav_b = np.array([r["nav"] for r in b])
    assert np.max(np.abs(nav_a / nav_b - 1)) < 2e-4


def test_moc_book_tracks_the_batch_and_trades_whole_shares(world, tmp_path):
    _quiet()
    src = _init(world, tmp_path, "moc", sizing="moc")
    _run("moc", tmp_path, src, until=world["until"], signal_mode="single")
    recs = _records(tmp_path, "moc")
    for prev, r in itertools.pairwise(recs):
        orders = prev["orders"]
        q = np.array(orders.get("shares", []), dtype=float)
        if len(q):
            closing = np.array(orders["close_out"], dtype=bool)
            assert np.all(q[~closing] == np.trunc(q[~closing]))  # whole shares
        assert r["date"] == prev.get("orders_for") or not len(q)
        fills = dict(zip(r["fills"].get("ticker", []), r["fills"].get("shares", []), strict=True))
        unfilled = {e["ticker"] for e in r["events"]["unfilled"]}
        for tk, qty, co in zip(orders.get("ticker", []), q, orders.get("close_out", []), strict=True):
            if tk in unfilled:
                continue
            ratio = fills[tk] / qty
            assert ratio == pytest.approx(1.0, rel=1e-6) or ratio == pytest.approx(2.0, rel=1e-6) or co
    rc = runner.reconcile("moc", paper_root=tmp_path, source=src, inner_factory=RidgeModel)
    # one day of sizing drift + whole shares: close, not identical
    assert rc["daily_return_corr"] > 0.99
    assert rc["max_rel_nav_diff"] < 2e-3
    # one whole position apart at most: on the halt-resumption day the batch reads the
    # name's eligibility at the (halt-day) signal date and force-exits it, while the moc
    # book decided the day before, when the name had no price, and holds it (handbook 14)
    assert rc["max_abs_weight_diff"] < cfg_max_weight() + 5e-3
    big = [d for d in rc["per_day"] if (d["weight_diff"] or 0) > 5e-3]
    halt_end = str(world["dates"][EV["halt"][0] + EV["halt"][1]].date())
    assert not big or big[0]["date"] <= halt_end
    assert all(r["mode"] == "backfill" for r in recs)  # generated long after the fill dates


def test_moc_mode_is_live_before_the_cutoff(world, tmp_path):
    _quiet()
    src = _init(world, tmp_path, "live", sizing="moc")
    d = world["dates"][0]
    now = datetime(d.year, d.month, d.day, 12, 5, tzinfo=timezone.utc)  # 07:05 ET on the fill day
    rep = _run("live", tmp_path, src, date=world["start"], now=now, signal_mode="single")
    assert rep["days"][0]["mode"] == "live" and rep["days"][0]["orders_for"] == str(d.date())


# =========================================================================== point in time
def test_day_t_orders_do_not_depend_on_later_data(world, tmp_path):
    """Perturb every price after day t (and delist a held name after it): the
    journal through t, and so the orders for t+1, are unchanged."""
    _quiet()
    md = world["md"]
    t = world["dates"][12]
    rng = np.random.default_rng(1)
    after = md.close.index > t
    noise = np.exp(rng.normal(0, 0.05, md.close.shape)) * after[:, None] + (~after)[:, None]
    pert = replace(md, close=md.close * noise, close_raw=md.close_raw * noise)
    out = {}
    for tag, mkt in (("base", md), ("pert", pert)):
        src = MemorySource(mkt)  # the full market: the runner must not look past t
        _init(world, tmp_path, tag, sizing="moc", source=src)
        _run(tag, tmp_path, src, until=str(t.date()))
        out[tag] = _records(tmp_path, tag)
    assert [r["date"] for r in out["base"]] == [r["date"] for r in out["pert"]]
    for a, b in zip(out["base"], out["pert"], strict=True):
        assert a["orders"] == b["orders"]
        assert a["nav"] == b["nav"]


def test_fit_through_a_later_date_answers_as_of_s_point_in_time(world):
    """The serving fit on data through T answers ``as_of(s)`` as the fit on data
    through s does, and data after s does not move it (what ``single`` relies on)."""
    _quiet()
    md, cfg = world["md"], _cfg()
    s = world["dates"][20]
    end = world["dates"][-1]
    rng = np.random.default_rng(2)
    later = md.close.index > s
    noise = np.exp(rng.normal(0, 0.05, md.close.shape)) * later[:, None] + (~later)[:, None]
    pert = replace(md, close=md.close * noise, close_raw=md.close_raw * noise)
    fits = {
        "cut": fit_signals(
            md.slice_until(s), cfg, use_ml=True, store=None, inner_factory=RidgeModel, nav0=NAV0
        ),
        "full": fit_signals(
            md.slice_until(end), cfg, use_ml=True, store=None, inner_factory=RidgeModel, nav0=NAV0
        ),
        "pert": fit_signals(
            pert.slice_until(end), cfg, use_ml=True, store=None, inner_factory=RidgeModel, nav0=NAV0
        ),
    }
    ref = fits["cut"].alpha_as_of(s)
    r_ref = fits["cut"].risk_as_of(s, ref.expected_return)
    for k in ("full", "pert"):
        a = fits[k].alpha_as_of(s)
        pd.testing.assert_series_equal(a.expected_return, ref.expected_return, rtol=1e-10, atol=1e-14)
        r = fits[k].risk_as_of(s, a.expected_return)
        np.testing.assert_allclose(
            r.factor_cov.to_numpy(), r_ref.factor_cov.to_numpy(), rtol=1e-10, atol=1e-16
        )
        np.testing.assert_allclose(
            r.specific_var.to_numpy(), r_ref.specific_var.reindex(r.specific_var.index).to_numpy(), rtol=1e-10
        )


# =========================================================================== prices
def test_restated_adjusted_history_does_not_move_the_book(world, tmp_path):
    """Sharadar rescales the whole adjusted history of a name at each dividend.  Feed
    the book the snapshots a vendor would have delivered (before the ex-date the
    adjusted history is *not* yet scaled for the dividend): NAV and P&L are unchanged,
    because only the ratio of two closes of one snapshot is ever used."""
    _quiet()
    md = world["md"].slice_until(world["dates"][-1])
    b = world["names"]["dividend"]
    ex = world["dates"][EV["dividend"]]
    snaps = {}
    for d in md.dates[md.dates >= pd.Timestamp(world["start"])]:
        cut = md.slice_until(d)
        if d < ex:
            c = cut.close.copy()
            c[b] = c[b] / (1.0 - DIV_YIELD)
            cut = replace(cut, close=c)
        snaps[d] = cut
    plain = MemorySource(md)
    restated = MemorySource(md, snapshots=snaps)
    for tag, src in (("plain", plain), ("restated", restated)):
        _init(world, tmp_path, tag, sizing="replica", source=src)
        _run(tag, tmp_path, src, until=world["until"], signal_mode="single")
    a, r = _records(tmp_path, "plain"), _records(tmp_path, "restated")
    nav_a = np.array([x["nav"] for x in a])
    nav_r = np.array([x["nav"] for x in r])
    assert np.max(np.abs(nav_r / nav_a - 1)) < 1e-6
    # the ex-date P&L of the position is the total return, and the shares grew by the yield
    i = next(k for k, x in enumerate(a) if x["date"] == str(ex.date()))
    assert a[i]["state"]["shares"][b] / a[i - 1]["state"]["shares"][b] == pytest.approx(
        1 / (1 - DIV_YIELD), rel=2e-2
    )


# =========================================================================== operations
def test_rerun_is_idempotent_and_redo_supersedes(world, tmp_path):
    _quiet()
    src = _init(world, tmp_path, "idem", sizing="moc")
    until = str(world["dates"][6].date())
    _run("idem", tmp_path, src, until=until, signal_mode="single")
    raw0, _ = read_journal(PaperPaths(tmp_path / "idem"))
    rep = _run("idem", tmp_path, src, until=until, signal_mode="single")
    assert "nothing to do" in rep["message"]
    rep = _run("idem", tmp_path, src, date=world["start"], signal_mode="single")
    assert "nothing to do" in rep["message"]
    raw1, _ = read_journal(PaperPaths(tmp_path / "idem"))
    assert raw1 == raw0  # no record added, nothing filled twice
    rep = _run("idem", tmp_path, src, until=until, redo_last=True, signal_mode="single")
    raw2, _ = read_journal(PaperPaths(tmp_path / "idem"))
    assert len(raw2) == len(raw0) + 1 and raw2[-1]["supersedes"] == raw0[-1]["seq"]
    eff = _records(tmp_path, "idem")
    assert len(eff) == len(raw0) and eff[-1]["nav"] == pytest.approx(raw0[-1]["nav"], rel=1e-12)


def test_dry_run_writes_nothing_and_shows_the_orders(world, tmp_path):
    _quiet()
    src = _init(world, tmp_path, "dry", sizing="moc")
    rep = _run("dry", tmp_path, src, date=world["start"], dry_run=True, signal_mode="single")
    assert (
        rep["dry_run"]
        and rep["days"][0]["orders"]
        and rep["days"][0]["n_orders"] == len(rep["days"][0]["orders"])
    )
    paths = PaperPaths(tmp_path / "dry")
    assert not paths.journal.exists() and not any((paths.root / "days").iterdir())


def test_catch_up_after_missed_days_equals_running_daily(world, tmp_path):
    _quiet()
    days = [str(d.date()) for d in world["dates"][:9]]
    srcs = {}
    for tag in ("daily", "catch"):
        srcs[tag] = _init(world, tmp_path, tag, sizing="moc")
    for d in [world["start"], *days]:
        _run("daily", tmp_path, srcs["daily"], date=d)
    _run("catch", tmp_path, srcs["catch"], date=world["start"])
    _run("catch", tmp_path, srcs["catch"], date=days[-1])  # 9 missed days at once
    a, b = _records(tmp_path, "daily"), _records(tmp_path, "catch")
    assert [r["date"] for r in a] == [r["date"] for r in b]
    for x, y in zip(a, b, strict=True):
        assert x["nav"] == pytest.approx(y["nav"], rel=1e-9)
        assert x["orders"]["ticker"] == y["orders"]["ticker"]


def test_lock_blocks_a_second_job(world, tmp_path):
    src = _init(world, tmp_path, "locked", sizing="moc")
    with BookLock(tmp_path / "locked", "test holder"), pytest.raises(BookLockedError, match="test holder"):
        _run("locked", tmp_path, src, date=world["start"])


def test_replay_removes_a_torn_line_and_rebuilds_the_views(world, tmp_path):
    _quiet()
    src = _init(world, tmp_path, "rp", sizing="moc")
    _run("rp", tmp_path, src, until=str(world["dates"][3].date()), signal_mode="single")
    paths = PaperPaths(tmp_path / "rp")
    shutil.rmtree(paths.root / "days")
    with open(paths.journal, "a") as fh:
        fh.write('{"kind": "day", "date": "2099-01-0')  # crash mid-append
    with pytest.raises(PaperError, match="torn"):
        _run("rp", tmp_path, src, until=world["until"])
    rep = runner.replay("rp", paper_root=tmp_path)
    assert rep["ok"] and rep["torn_line_removed"] and rep["n_days"] == 5
    last = paths.day_dir(world["dates"][3])
    pos = pd.read_parquet(last / "positions.parquet")
    assert len(pos) > 10 and pos["value"].sum() != 0
    assert (last / "orders.parquet").exists() and (last / "fills.parquet").exists()
    o = runner.orders("rp", paper_root=tmp_path)
    assert o["n_orders"] == len(pd.read_parquet(last / "orders.parquet"))


def test_init_refuses_to_overwrite_and_checks_the_start(world, tmp_path):
    _init(world, tmp_path, "once")
    with pytest.raises(PaperError, match="already exists"):
        _init(world, tmp_path, "once")
    with pytest.raises(PaperError, match="not a trading session"):
        runner.init_book(
            "sat",
            world["cfg_path"],
            nav=NAV0,
            start="2019-11-02",
            paper_root=tmp_path,
            data=DataSpec(kind="memory"),
            calendar="nyse",
            expectations=world["exp"],
        )


# =========================================================================== stop rules
def _fake(rets, *, gross=2.0, exante=0.10, solve_ok=True, start="2020-01-02"):
    dates = pd.bdate_range(start, periods=len(rets))
    nav, out = NAV0, []
    for d, r in zip(dates, rets, strict=True):
        nav *= 1 + r
        out.append(
            {
                "date": str(d.date()),
                "nav": nav,
                "ret": r,
                "gross": gross,
                "net": 0.0,
                "exante_vol": exante,
                "turnover": 0.01,
                "traded_usd": 0.01 * nav,
                "pnl": {"trading_cost": 0.01 * nav * 3e-4},
                "solve": {"ok": solve_ok},
                "data_quality": {},
                "events": {},
            }
        )
    return out


@pytest.fixture(scope="module")
def ref_exp():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0004, 0.006, 3000), index=pd.bdate_range("2005-01-03", periods=3000))
    extra = {
        "avg_daily_turnover": 0.01,
        "avg_cost_bps_per_side": 3.0,
        "avg_gross_exposure": 2.0,
        "realized_vs_exante_vol_ratio": 1.0,
    }
    return X.Expectations(
        summary=X.summary_from_returns(r, extra), table=X.block_bootstrap_table(r.to_numpy()), returns=r
    )


def _level(recs, exp, rule, **kw):
    st = evaluate(recs, exp, StopRules(), recs[0]["date"], 2.0, **kw)
    return st, next(c for c in st["checks"] if c["rule"] == rule)["level"]


def test_stop_rule_drawdown(ref_exp):
    rng = np.random.default_rng(3)
    ok = list(rng.normal(0.0004, 0.006, 60))
    assert _level(_fake(ok), ref_exp, "drawdown")[1] == "OK"
    crash = ok[:40] + [-0.03] * 5 + ok[45:]
    st, lvl = _level(_fake(crash), ref_exp, "drawdown")
    assert lvl == "STOP" and st["level"] == "STOP" and "drawdown" in st["stop_rules"]
    mild = ok[:40] + [-0.012] * 3 + ok[43:]
    nav = NAV0 * np.cumprod(1 + np.array(mild))
    path = np.concatenate([[NAV0], nav])
    dd = float((path / np.maximum.accumulate(path) - 1).min())
    q01 = ref_exp.quantile_at("max_dd", 0.01, 60)
    q05 = ref_exp.quantile_at("max_dd", 0.05, 60)
    _, lvl = _level(_fake(mild), ref_exp, "drawdown")
    assert lvl == ("STOP" if dd < q01 else "WATCH" if dd < q05 else "OK")


def test_stop_rule_sharpe_needs_126_days(ref_exp):
    rng = np.random.default_rng(4)
    bad = list(rng.normal(-0.0012, 0.006, 130))
    _, lvl = _level(_fake(bad[:100]), ref_exp, "sharpe")
    assert lvl == "WATCH"  # below the 10 % quantile, but STOP needs 126 days
    _, lvl = _level(_fake(bad), ref_exp, "sharpe")
    assert lvl == "STOP"


def test_stop_rule_vol_ratio_needs_20_consecutive_days(ref_exp):
    rng = np.random.default_rng(5)
    calm = list(rng.normal(0.0, 0.10 / np.sqrt(252), 40))
    wild = list(rng.normal(0.0, 0.30 / np.sqrt(252), 45))
    _, lvl = _level(_fake(calm + wild[:22]), ref_exp, "vol_ratio")  # too few days above the ratio yet
    assert lvl != "STOP"
    _, lvl = _level(_fake(calm + wild), ref_exp, "vol_ratio")
    assert lvl == "STOP"


def test_stop_rule_failed_solves_and_gross(ref_exp):
    recs = _fake([0.0001] * 30)
    for r in recs[-3:]:
        r["solve"] = {"ok": False}
    assert _level(recs, ref_exp, "optimizer")[1] == "STOP"
    recs[-3]["solve"] = {"ok": True}
    assert _level(recs, ref_exp, "optimizer")[1] == "WATCH"
    recs = _fake([0.0001] * 30, gross=2.5)
    assert _level(recs, ref_exp, "gross")[1] == "STOP"
    recs = _fake([0.0001] * 30, gross=0.3)
    assert _level(recs, ref_exp, "gross")[1] == "WATCH"


def test_ack_suppresses_until_the_rule_clears(ref_exp):
    recs = _fake([0.0001] * 30, gross=2.5)
    st = evaluate(recs, ref_exp, StopRules(), recs[0]["date"], 2.0)
    assert st["level"] == "STOP"
    ack = [{"date": recs[-1]["date"], "rules": st["stop_rules"]}]
    st2 = evaluate(recs, ref_exp, StopRules(), recs[0]["date"], 2.0, acks=ack)
    assert st2["level"] == "WATCH" and st2["suppressed"] == ["gross"]
    # still breached the next day: stays suppressed
    more = _fake([0.0001] * 31, gross=2.5)
    st3 = evaluate(more, ref_exp, StopRules(), more[0]["date"], 2.0, prior_suppressed=st2["suppressed"])
    assert st3["level"] == "WATCH"
    # cleared, then breached again: STOP again
    cleared = _fake([0.0001] * 32, gross=1.8)
    st4 = evaluate(cleared, ref_exp, StopRules(), cleared[0]["date"], 2.0, prior_suppressed=st3["suppressed"])
    assert st4["suppressed"] == []
    again = _fake([0.0001] * 33, gross=2.5)
    st5 = evaluate(again, ref_exp, StopRules(), again[0]["date"], 2.0, prior_suppressed=st4["suppressed"])
    assert st5["level"] == "STOP"


def test_stop_halts_trading_until_ack(world, tmp_path):
    _quiet()
    rules = StopRules(gross_stop_mult=0.05)  # any invested book breaches: a planted STOP
    src = _init(world, tmp_path, "halt", sizing="moc", rules=rules)
    d = [str(x.date()) for x in world["dates"][:4]]
    _run("halt", tmp_path, src, date=d[0], signal_mode="single")  # first fills -> gross > 0.1 -> STOP
    r = _records(tmp_path, "halt")[-1]
    assert r["status"]["level"] == "STOP" and r["orders"]["ticker"] == []
    assert r.get("orders_cancelled_by_stop", 0) >= 0
    _run("halt", tmp_path, src, date=d[1], signal_mode="single")
    r = _records(tmp_path, "halt")[-1]
    assert r["halted"] and r["n_fills"] == 0 and r["orders"]["ticker"] == []
    st = runner.status("halt", paper_root=tmp_path)
    assert st["status"]["trading_halted"]
    runner.ack("halt", "planted STOP in a test; resume", paper_root=tmp_path)
    assert not runner.status("halt", paper_root=tmp_path)["status"]["trading_halted"]
    _run("halt", tmp_path, src, date=d[3], signal_mode="single")
    recs = _records(tmp_path, "halt")
    assert not recs[-2]["halted"] and recs[-2]["status"]["level"] == "WATCH"


# =========================================================================== data refresh
@pytest.fixture(scope="module")
def store_book(world, tmp_path_factory):
    from qts.data.store import ParquetStore

    tmp = tmp_path_factory.mktemp("store")
    md = world["md"].slice_until(world["dates"][3])
    md = replace(md, extra={})
    ParquetStore(str(tmp / "store")).write_market_data(md)
    data = DataSpec(kind="store", store=str(tmp / "store"), start=None, max_names=None, min_mcap_ever=None)
    root = tmp / "paper"
    cal_start = world["start"]
    runner.init_book(
        "s",
        world["cfg_path"],
        nav=NAV0,
        start=cal_start,
        paper_root=root,
        data=data,
        sizing="moc",
        calendar="nyse",
        expectations=world["exp"],
    )
    return {"root": root, "data": data, "last": world["dates"][3]}


class _FakeReport:
    mode, seconds, warnings, tables = "incremental", 0.1, [], {}


def test_sync_is_called_incrementally_and_freshness_is_checked(store_book, world):
    _quiet()
    calls = []

    def fake_sync(**kw):
        calls.append(kw)
        return _FakeReport()

    root = store_book["root"]
    last = store_book["last"]
    # the vendor should have published `last` + 1 session by now -> stale
    later = NYSECalendar().next_session(last)
    now_late = datetime(later.year, later.month, later.day, 14, tzinfo=timezone.utc) + pd.Timedelta(days=1)
    with pytest.raises(StaleDataError, match="should be published"):
        runner.run_book(
            "s",
            paper_root=root,
            sync_fn=fake_sync,
            now=now_late,
            inner_factory=RidgeModel,
            signal_mode="single",
        )
    assert calls and calls[0]["since"] == str((pd.Timestamp(world["start"]) - pd.Timedelta(days=5)).date())
    assert calls[0]["store_root"] == store_book["data"].store and "api_key" not in calls[0]
    # on time: processes through the latest session in the store
    now_ok = datetime(last.year, last.month, last.day, 12, tzinfo=timezone.utc) + pd.Timedelta(days=1)
    rep = runner.run_book(
        "s", paper_root=root, sync_fn=fake_sync, now=now_ok, inner_factory=RidgeModel, signal_mode="single"
    )
    assert rep["target"] == str(last.date()) and len(rep["days"]) == 5
    assert calls[-1]["since"] == str((pd.Timestamp(world["start"]) - pd.Timedelta(days=5)).date())
    rep = runner.run_book("s", paper_root=root, sync_fn=fake_sync, now=now_ok, inner_factory=RidgeModel)
    assert "nothing to do" in rep["message"]
    assert calls[-1]["since"] == str((last - pd.Timedelta(days=5)).date())


def test_expected_last_session_and_freshness():
    cal = NYSECalendar()
    # Monday 2026-11-16 07:05 ET -> Friday 2026-11-13; Sunday -> Friday as well
    assert expected_last_session(datetime(2026, 11, 16, 12, 5, tzinfo=timezone.utc), cal) == pd.Timestamp(
        "2026-11-13"
    )
    assert expected_last_session(datetime(2026, 11, 15, 12, tzinfo=timezone.utc), cal) == pd.Timestamp(
        "2026-11-13"
    )
    # Friday 2026-11-27 after Thanksgiving, 07:05 ET -> Wednesday 2026-11-25
    assert expected_last_session(datetime(2026, 11, 27, 12, 5, tzinfo=timezone.utc), cal) == pd.Timestamp(
        "2026-11-25"
    )
    # 05:00 ET Tuesday: Monday's prices are not due yet -> Friday
    assert expected_last_session(datetime(2026, 11, 17, 10, tzinfo=timezone.utc), cal) == pd.Timestamp(
        "2026-11-13"
    )
    with pytest.raises(StaleDataError):
        check_freshness(pd.Timestamp("2026-11-12"), pd.Timestamp("2026-11-13"), "store")
    check_freshness(pd.Timestamp("2026-11-13"), pd.Timestamp("2026-11-13"), "store")


# =========================================================================== CLI, schedule, expectations
def test_cli_status_history_orders_schedule(store_book, capsys):
    from qts.cli import main

    root = str(store_book["root"])
    assert main(["paper", "status", "--name", "s", "--paper-root", root, "--json"]) == 0
    st = json.loads(capsys.readouterr().out)
    assert st["status"]["level"] in ("OK", "WATCH", "STOP") and st["series"]["date"]
    assert {"checks", "metrics", "reasons"} <= set(st["status"])
    assert main(["paper", "history", "--name", "s", "--paper-root", root]) == 0
    assert "date" in capsys.readouterr().out
    assert main(["paper", "orders", "--name", "s", "--paper-root", root, "--json"]) == 0
    o = json.loads(capsys.readouterr().out)
    assert o["orders_for"] and isinstance(o["orders"], list)
    assert (
        main(["paper", "run", "--name", "s", "--paper-root", root, "--no-sync", "--date", "2030-01-02"]) == 1
    )
    assert main(["paper", "schedule", "--print", "--name", "s", "--paper-root", root]) == 0
    out = capsys.readouterr().out
    assert "<key>Hour</key><integer>7</integer><key>Minute</key><integer>5</integer>" in out
    assert out.count("<key>Weekday</key>") == 5 and "5 7 * * 1-5" in out and "paper run --name s" in out
    assert main(["paper", "ack", "--name", "s", "--paper-root", root, "--note", "checked"]) == 0


def test_calibrated_boundaries_hit_their_false_alarm_targets(ref_exp):
    rules = StopRules()
    info, rows = X.calibrate(ref_exp.returns.to_numpy(), rules)
    for name, target in (
        ("dd_stop", 0.025),
        ("dd_watch", 0.10),
        ("sharpe_stop", 0.025),
        ("sharpe_watch", 0.10),
    ):
        assert info["achieved"][name] == pytest.approx(target, abs=0.006)
        # the uncalibrated per-horizon quantile checked daily fires far more often
        assert info["fixed_quantile_rates"][name] > 2 * target
    assert info["levels"]["dd_stop"] < rules.dd_stop_quantile
    exp = X.Expectations(
        {**ref_exp.summary, "calibration": info}, pd.concat([ref_exp.table, rows]), ref_exp.returns
    )
    recs = _fake([0.0004] * 200)
    st = evaluate(recs, exp, rules, recs[0]["date"], 2.0)
    assert st["metrics"]["boundary_levels"]["dd_stop"] == info["levels"]["dd_stop"]
    # the STOP boundary sits below the WATCH boundary at every horizon
    for h in (21, 126, 252, 504):
        assert exp.quantile_at("max_dd", info["levels"]["dd_stop"], h) < exp.quantile_at(
            "max_dd", info["levels"]["dd_watch"], h
        )


def test_expectations_from_a_run_dir_and_tamper_detection(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2010-01-04", periods=800)
    pd.DataFrame(
        {"gross": 0.0, "net": rng.normal(0.0003, 0.005, 800), "gross_after_borrow": 0.0}, index=idx
    ).to_parquet(run / "returns.parquet")
    (run / "summary.json").write_text(
        json.dumps(
            {
                "backtest": {"avg_daily_turnover": 0.02, "avg_cost_bps_per_side": 4.0},
                "config_hash": "x",
                "use_ml": True,
            }
        )
    )
    pd.DataFrame({"rank_ic": rng.normal(0.02, 0.1, 100)}).to_parquet(run / "ic.parquet")
    exp = X.from_run_dir(run)
    assert exp.summary["n_days"] == 800 and exp.summary["ic_n"] == 100
    # quantiles are ordered and the drawdown quantile deepens with the horizon
    assert exp.quantile_at("max_dd", 0.01, 252) < exp.quantile_at("max_dd", 0.05, 252) < 0
    assert exp.quantile_at("max_dd", 0.01, 504) < exp.quantile_at("max_dd", 0.01, 63)
    digest = X.save(exp, tmp_path / "e")
    X.load(tmp_path / "e", digest)
    s = json.loads((tmp_path / "e" / "summary.json").read_text())
    s["sd_daily"] = 0.0001
    (tmp_path / "e" / "summary.json").write_text(json.dumps(s))
    with pytest.raises(PaperError, match="do not match"):
        X.load(tmp_path / "e", digest)
