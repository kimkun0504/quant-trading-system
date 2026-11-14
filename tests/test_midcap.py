"""Mid-cap confirmation (``universe.rank_offset``, ``qts check midcap``,
``qts.research.midcap``; pre-registration docs/notes/midcap_check_2026-11-13.md).

* rank offset: hand-computed daily ranks, offset 0 bitwise the old rule, the offset and
  the top-N universes are disjoint and tile the top ``offset + N``;
* config: validation, the default unchanged;
* load coverage (V2) by hand on a tiny store;
* the once-only rule (normalised config hash, ``--force`` flag on the record) and an
  end-to-end CLI run on a synthetic store (own family, record fields, verdict JSON,
  second run refused);
* cost sanity: a mid-cap-like name pays more per side and gets a smaller ADV cap.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qts.core.config import SystemConfig, UniverseConfig
from qts.data.store import ParquetStore
from qts.data.synthetic import SyntheticSpec, generate
from qts.data.universe import build_universe_mask, raw_eligibility
from qts.research import midcap as MC
from qts.research.data import StoreSpec
from qts.validation import TrialLedger


# --------------------------------------------------------------------------- rank offset
@pytest.fixture(scope="module")
def small_market():
    return generate(SyntheticSpec(n_stocks=90, n_days=120, seed=3))


def _ucfg(**kw) -> UniverseConfig:
    base = {"min_price": 0.0, "min_adv_usd": 0.0, "exclude_bottom_size_quantile": 0.0}
    return UniverseConfig(**{**base, **kw})


def test_rank_offset_selects_the_next_names_by_daily_market_cap(small_market):
    md = small_market
    got = raw_eligibility(md, _ucfg(top_n_by_mcap=20, rank_offset=30))
    for d in md.dates[60::17]:
        mc = md.market_cap.loc[d].where(md.close.loc[d].notna()).dropna()
        order = mc.sort_values(ascending=False, kind="mergesort").index  # ties: first by position
        want = set(order[30:50])
        assert set(got.columns[got.loc[d].to_numpy()]) == want


def test_rank_offset_zero_is_the_old_rule_bit_for_bit(small_market):
    md = small_market
    cfg = UniverseConfig(top_n_by_mcap=40)
    new = raw_eligibility(md, cfg)
    # the rule before the field existed
    price, mcap = md.trade_price(), md.market_cap
    adv = md.dollar_volume.rolling(cfg.adv_window, min_periods=max(5, cfg.adv_window // 2)).mean()
    base = (price >= cfg.min_price) & (adv >= cfg.min_adv_usd) & mcap.notna() & md.close.notna()
    m_elig = mcap.where(base)
    cut = m_elig.quantile(cfg.exclude_bottom_size_quantile, axis=1)
    base = base & (mcap.ge(cut, axis=0))
    rank = mcap.where(base).rank(axis=1, ascending=False, method="first")
    old = (base & (rank <= cfg.top_n_by_mcap)).fillna(False).astype(bool)
    assert new.equals(old)


def test_offset_and_top_universes_are_disjoint_and_tile_the_top(small_market):
    md = small_market
    top = raw_eligibility(md, _ucfg(top_n_by_mcap=25))
    mid = raw_eligibility(md, _ucfg(top_n_by_mcap=25, rank_offset=25))
    both = raw_eligibility(md, _ucfg(top_n_by_mcap=50))
    assert not (top & mid).to_numpy().any()
    assert (top | mid).equals(both)
    # point in time: the mask on a prefix equals the prefix of the mask
    cut = md.dates[80]
    early = build_universe_mask(md.slice_until(cut), _ucfg(top_n_by_mcap=25, rank_offset=25))
    full = build_universe_mask(md, _ucfg(top_n_by_mcap=25, rank_offset=25))
    assert early.equals(full.loc[:cut, early.columns])


def test_rank_offset_validation_and_default():
    assert UniverseConfig().rank_offset == 0
    for bad in (-1, 1.5, True):
        with pytest.raises(ValueError, match="rank_offset"):
            SystemConfig().with_overrides({"universe.rank_offset": bad})
    cfg = SystemConfig().with_overrides({"universe.rank_offset": 1000})
    assert cfg.universe.rank_offset == 1000
    assert SystemConfig.from_dict(cfg.to_dict()).hash() == cfg.hash()


def test_needed_loaded_names_follows_the_size_cut():
    cfg = SystemConfig().with_overrides({"universe.rank_offset": 1000})
    assert MC.needed_loaded_names(cfg) == 2500  # 2000 / (1 - 0.2)


# --------------------------------------------------------------------------- coverage (V2)
def _tiny_store(tmp_path: Path) -> ParquetStore:
    """Eight dates, five ids; with adv_window 5 the ADV screen is defined from d4.
    A passes (cap 100); B fails the price screen (cap 90); C fails the ADV screen (cap 80);
    D passes (cap 70 on d4-d5, 105 on d6-d7); E passes (cap 50).  Screened ranks:
    d4, d5: A 1, D 2, E 3;  d6, d7: D 1, A 2, E 3."""
    dates = pd.bdate_range("2020-01-01", periods=8)
    one = np.ones(8)
    price = pd.DataFrame(
        {"A": 10 * one, "B": 4 * one, "C": 10 * one, "D": 20 * one, "E": 10 * one}, index=dates
    )
    vol = pd.DataFrame(
        {"A": 1e6 * one, "B": 1e6 * one, "C": 1e5 * one, "D": 1e6 * one, "E": 1e6 * one}, index=dates
    )
    cap = pd.DataFrame(
        {"A": 100 * one, "B": 90 * one, "C": 80 * one, "D": [70.0] * 6 + [105.0] * 2, "E": 50 * one},
        index=dates,
    )
    st = ParquetStore(str(tmp_path / "tiny"))
    st.write_wide("prices_close", price)
    st.write_wide("prices_volume", vol)
    st.write_wide("market_cap", cap)
    return st


def _spec(st, **kw) -> StoreSpec:
    return StoreSpec(store=st.root, adv_window=5, **kw)


def test_screened_rank_prefilter_by_hand(tmp_path):
    st = _tiny_store(tmp_path)
    assert st.tickers_by_screened_rank(1, adv_window=5) == ["A", "D"]
    assert st.tickers_by_screened_rank(2, adv_window=5) == ["A", "D"]
    assert st.tickers_by_screened_rank(3, adv_window=5) == ["A", "D", "E"]
    assert st.tickers_by_screened_rank(10, adv_window=5) == ["A", "D", "E"]  # B and C never pass
    with pytest.raises(ValueError, match="max-rank"):
        StoreSpec(store=st.root, max_rank=3, max_names=2)


def test_prefilter_coverage_by_hand(tmp_path):
    st = _tiny_store(tmp_path)
    rep = MC.prefilter_coverage(_spec(st, max_rank=2), rank=2, needed_loaded=2)
    assert rep.n_loaded_ids == 2 and rep.n_dates == 4  # d4..d7 have screened names
    assert rep.share_dates_full == pytest.approx(1.0) and rep.min_loaded == 2 and rep.ok
    assert np.isfinite(rep.est_load_gb) and "--max-rank 2" in rep.format()
    # 3 needed: the store has 3 passing names, only 2 loaded -> short on every date
    short = MC.prefilter_coverage(_spec(st, max_rank=2), rank=2, needed_loaded=3)
    assert not short.ok and short.n_short_dates == 4 and short.min_store == 3
    # 4 needed: more than the store has -> loading all 3 passing names is enough
    full = MC.prefilter_coverage(_spec(st, max_rank=3), rank=2, needed_loaded=4)
    assert full.ok and full.n_short_dates == 0 and "V1 judges it" in full.format()
    # the peak-cap prefilter keeps D (105) and A (100): on d4-d7 top 2 = {A, D} -> full;
    # with one name (D only) A is missed on every date
    assert MC.prefilter_coverage(_spec(st, max_names=2), rank=2, needed_loaded=2).ok
    rep2 = MC.prefilter_coverage(_spec(st, max_names=1), rank=2, needed_loaded=1)
    assert rep2.share_dates_full == 0.0 and rep2.min_coverage == pytest.approx(0.5)
    assert not rep2.ok and len(rep2.worst_dates) == 4 and "NOT OK" in rep2.format()
    json.dumps(rep2.to_dict())


def test_screened_ranks_match_the_universe_rule_on_a_gap_free_market(tmp_path):
    md = generate(SyntheticSpec(n_stocks=50, n_days=120, seed=2))
    st = ParquetStore(str(tmp_path / "s"))
    st.write_market_data(md)
    k = 15
    ranked = st.duckdb_query(f"SELECT date, ticker FROM ({st.screened_rank_sql()}) WHERE rk <= {k}")
    ranked["date"] = pd.to_datetime(ranked["date"])
    want = raw_eligibility(md, UniverseConfig(top_n_by_mcap=k, exclude_bottom_size_quantile=0.0))
    for d in md.dates[40::9]:
        got = set(ranked.loc[ranked["date"] == d, "ticker"])
        assert got == set(want.columns[want.loc[d].to_numpy()]), d


# --------------------------------------------------------------------------- once-only
def _small_baseline(tmp_path: Path, top_n: int = 20) -> Path:
    cfg = SystemConfig().with_overrides({"universe.top_n_by_mcap": top_n})
    p = tmp_path / "baseline_small.yaml"
    p.write_text(cfg.to_yaml())
    return p


def test_once_only_rule_uses_the_normalised_config_hash(tmp_path):
    base = _small_baseline(tmp_path)
    cfg = MC.midcap_config(base, rank_offset=20)
    led = TrialLedger(str(tmp_path / "ledger.jsonl"))
    # an earlier record written by code without the field: its stored dict lacks
    # universe.rank_offset... except that the check always sets it; drop another new
    # field instead to mimic an older writer
    d = cfg.to_dict()
    d["alpha"].pop("disabled_features")
    led.log(
        name="confirm:midcap:abc", family="confirm:midcap:abc", config=d, config_hash="old-hash", metrics={}
    )
    prev = MC.previous_midcap_records(led.path, cfg.hash())
    assert len(prev) == 1
    spec = StoreSpec(store=str(tmp_path / "nostore"))
    with pytest.raises(MC.MidcapCheckError, match="already ran"):
        MC.run_midcap_check(
            spec, config_path=base, ledger_path=led.path, runs_root=None, rank_offset=20, md=object()
        )
    # --force passes the rule (dry run: nothing loaded, nothing logged)
    r = MC.run_midcap_check(
        spec,
        config_path=base,
        ledger_path=led.path,
        runs_root=None,
        rank_offset=20,
        md=object(),
        force=True,
        dry_run=True,
    )
    assert r.forced and r.previous_runs and r.dry_run
    assert len(led.raw_records()) == 1
    # another configuration is another check
    assert MC.previous_midcap_records(led.path, MC.midcap_config(base, rank_offset=10).hash()) == []


def test_midcap_refuses_a_baseline_with_an_offset(tmp_path):
    p = tmp_path / "off.yaml"
    p.write_text(SystemConfig().with_overrides({"universe.rank_offset": 5}).to_yaml())
    with pytest.raises(MC.MidcapCheckError, match="already sets"):
        MC.midcap_config(p)


def test_preregistration_hash_ignores_the_appended_result(tmp_path, monkeypatch):
    f = tmp_path / "note.md"
    f.write_text("# pre\ncriteria\n\n## Result\n\n(pending)\n")
    h1 = MC.preregistration_sha256(f)
    f.write_text("# pre\ncriteria\n\n## Result\n\nPASS run X\n")
    assert MC.preregistration_sha256(f) == h1 and h1
    f.write_text("# pre\ncriteria changed\n\n## Result\n")
    assert MC.preregistration_sha256(f) != h1
    assert MC.preregistration_sha256()  # the committed note exists


# --------------------------------------------------------------------------- end to end
@pytest.fixture(scope="module")
def synth_store(tmp_path_factory):
    md = generate(SyntheticSpec(n_stocks=60, n_days=640, seed=5, n_delistings=3))
    root = tmp_path_factory.mktemp("midcap") / "store"
    ParquetStore(str(root)).write_market_data(md)
    return root


def test_check_midcap_cli_end_to_end(synth_store, tmp_path, capsys):
    from qts.cli import main

    base = _small_baseline(tmp_path, top_n=20)
    led = tmp_path / "ledger.jsonl"
    runs = tmp_path / "runs"
    argv = [
        "check",
        "midcap",
        "--store",
        str(synth_store),
        "--config",
        str(base),
        "--max-names",
        "60",
        "--ledger",
        str(led),
        "--runs-root",
        str(runs),
    ]
    # the CLI runs the real check (offset 1000): on 60 names that leaves no universe, so
    # the scaled-down end-to-end goes through the library with offset 20 and the CLI
    # is exercised for its refusal paths and the dry run
    assert main([*argv, "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "Load coverage" in out and "dry run" in out and not led.exists()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = MC.run_midcap_check(
            StoreSpec(store=str(synth_store), max_names=60),
            config_path=base,
            ledger_path=str(led),
            runs_root=str(runs),
            use_ml=False,
            rank_offset=20,
            max_days=30,
            coverage_rank=40,
            log=lambda _m: None,
        )
    assert r.verdict is not None and r.verdict.outcome in (
        "PASS",
        "PASS, cost-fragile",
        "FAIL",
        "INCONCLUSIVE",
    )
    assert r.family is not None and r.family.startswith("confirm:midcap:")
    recs = TrialLedger(str(led)).raw_records()
    assert len(recs) == 1
    rec = recs[0]
    assert rec["family"] == r.family and rec["check"] == "midcap" and rec["seal_exempt"] == "midcap"
    assert rec["preregistration_sha256"] == MC.preregistration_sha256() and rec["midcap_forced"] is False
    assert rec["config"]["universe"]["rank_offset"] == 20
    # the confirmation is a trial of its own family only
    led_ = TrialLedger(str(led))
    assert led_.n_trials(r.family) == 1
    summ = json.loads((Path(r.run_dir) / "summary.json").read_text())
    assert (
        summ["ledger_family"] == r.family
        and summ["seal_exempt"] == "midcap"
        and summ["record_kind"] == "trial"
    )
    js = json.loads(Path(r.json_path).read_text())
    assert js["verdict"]["outcome"] == r.verdict.outcome and js["coverage"]["ok"] is True
    assert {c["id"] for c in js["verdict"]["criteria"]} == {"C1", "C2", "C3"}
    # the verdict follows the run's own statistics
    s = summ["backtest"]
    c1 = next(c for c in r.verdict.criteria if c.id == "C1")
    assert c1.passed == (s["sharpe_net"] > 0 and s["newey_west_t_net"] >= 2.0)
    c3 = next(c for c in r.verdict.criteria if c.id == "C3")
    n_pos = sum(1 for f in summ["folds"] if isinstance(f["sharpe"], float) and f["sharpe"] > 0)
    assert c3.value.startswith(f"{n_pos} of")
    assert "VERDICT" in r.report and "Side by side" in r.report
    # a second run of the same configuration is refused (library and CLI)
    with pytest.raises(MC.MidcapCheckError, match="already ran"):
        MC.run_midcap_check(
            StoreSpec(store=str(synth_store), max_names=60),
            config_path=base,
            ledger_path=str(led),
            runs_root=str(runs),
            use_ml=False,
            rank_offset=20,
            coverage_rank=40,
        )


def test_check_midcap_cli_refuses_low_coverage(synth_store, tmp_path, capsys):
    from qts.cli import main

    base = _small_baseline(tmp_path, top_n=20)
    rc = main(
        [
            "check",
            "midcap",
            "--store",
            str(synth_store),
            "--config",
            str(base),
            "--max-names",
            "30",
            "--ledger",
            str(tmp_path / "l.jsonl"),
            "--runs-root",
            str(tmp_path / "runs"),
        ]
    )
    assert rc == 2
    assert "coverage too low" in capsys.readouterr().err
    assert not (tmp_path / "l.jsonl").exists()


# --------------------------------------------------------------------------- cost sanity
def test_cost_model_prices_mid_caps_higher_and_caps_participation_by_adv():
    from qts.core.config import CostConfig, ExecutionConfig
    from qts.core.costs import LiquiditySnapshot
    from qts.tcost import TransactionCostModel

    tickers = pd.Index(["LARGE", "MID"])
    snap = LiquiditySnapshot(
        date=pd.Timestamp("2020-06-01"),
        tickers=tickers,
        price=pd.Series([150.0, 40.0], index=tickers),
        adv_usd=pd.Series([800e6, 25e6], index=tickers),
        adv_shares=pd.Series([800e6 / 150, 25e6 / 40], index=tickers),
        sigma_daily=pd.Series([0.012, 0.024], index=tickers),
        half_spread=pd.Series([2e-4, 2e-4], index=tickers),
        borrow_fee=pd.Series([0.005, 0.005], index=tickers),
    )
    tcm = TransactionCostModel(CostConfig(), ExecutionConfig())
    trade = pd.Series([100_000.0, 100_000.0], index=tickers)
    est = tcm.estimate(trade, snap, venue="continuous")
    bps = est.frame["total_usd"] / trade.abs() * 1e4
    assert bps["MID"] > bps["LARGE"]
    assert est.frame.loc["MID", "impact_usd"] > 2 * est.frame.loc["LARGE", "impact_usd"]
    # the flat default half-spread: identical spread charge per dollar (the optimistic part)
    assert est.frame.loc["MID", "spread_usd"] == pytest.approx(est.frame.loc["LARGE", "spread_usd"])
    cap = tcm.participation_cap(snap, nav=5e6)
    assert cap["MID"] / cap["LARGE"] == pytest.approx(25e6 / 800e6)
    assert np.isfinite(cap.to_numpy()).all()
