"""``qts breakdown industries`` (:mod:`qts.research.industry_breakdown`).

* the industry contributions (sum of w x r within each industry) add up to the run's
  recorded gross return, delistings included, and the cost allocation to its net;
* the industry labels are the risk model's point-in-time ones (the run's holdings file);
* a planted single-industry edge reads "concentrated", a uniform one "broad", one in
  two thirds of the industries "moderately concentrated", no edge "undetermined";
* a sealed run is refused on full-history data, and on a moved seal; a full-history run
  under the seal is broken down on the pre-holdout dates only; a cold stage cache is
  refused with ``refuse_refit``;
* the JSON round-trips; the CLI end to end; nothing is logged to the trial ledger.
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from qts.data.loaders import FF48_NAMES, FF48_TO_FF12
from qts.data.store import ParquetStore
from qts.data.synthetic import SyntheticSpec, generate
from qts.research import industry_breakdown as B
from qts.research import seal as S
from qts.research.data import StoreSpec

SPEC: dict[str, Any] = {"n_stocks": 60, "n_days": 640, "seed": 5, "n_delistings": 20}
HOLDOUT = "2020-03-02"  # synthetic market: 2018-01-02 .. 2020-06-15 (sealed load: .. 2020-02-28)
#: 12 synthetic industries relabelled as Fama-French 48 names, so the FF12 roll-up exists
LABELS = {f"IND{k:02d}": FF48_NAMES[4 * k] for k in range(12)}


def _write_seal(d: Path, start: str = HOLDOUT) -> Path:
    import yaml

    p = d / "seal.yaml"
    p.write_text(yaml.safe_dump({"holdout_start": start, "reason": "test"}))
    return p


@pytest.fixture(scope="module")
def store(tmp_path_factory) -> Path:
    md = generate(SyntheticSpec(**SPEC))
    md = dataclasses.replace(md, industry=md.industry.map(LABELS))
    root = tmp_path_factory.mktemp("ib") / "store"
    ParquetStore(str(root)).write_market_data(md)
    return root


def _spec(store: Path) -> StoreSpec:
    return StoreSpec(store=str(store), max_names=60)


def _qts_run(store: Path, root: Path) -> Path:
    from qts.cli import main
    from qts.pipeline import list_runs

    argv = [
        "run", "--store", str(store), "--max-names", "60", "--set", "universe.top_n_by_mcap=50",
        "--max-days", "200", "--ledger", str(root / "ledger.jsonl"), "--runs-root", str(root), "--cache",
    ]  # fmt: skip
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert main(argv) == 0
    return list_runs(root)[0]


@pytest.fixture(scope="module")
def full_run(store, tmp_path_factory) -> Path:
    """A full-history run (no seal in force: tests/conftest.py sets QTS_SEAL_FILE=none)."""
    return _qts_run(store, tmp_path_factory.mktemp("full") / "runs")


@pytest.fixture(scope="module")
def seal_file(tmp_path_factory) -> Path:
    return _write_seal(tmp_path_factory.mktemp("seal"))


@pytest.fixture(scope="module")
def sealed_run(store, seal_file, tmp_path_factory) -> Path:
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(S.SEAL_FILE_ENV, str(seal_file))
        return _qts_run(store, tmp_path_factory.mktemp("sealed") / "runs")


def _ledger_lines(run_dir: Path) -> int:
    p = run_dir.parent / "ledger.jsonl"
    return len(p.read_text().splitlines()) if p.exists() else 0


# --------------------------------------------------------------------------- accounting
def test_contributions_sum_to_the_books_gross(full_run, store):
    from qts.runs.registry import read_delistings

    dl = read_delistings(full_run)
    assert dl is not None and len(dl) > 0  # a held delisting inside the run: the override is exercised
    n_ledger = _ledger_lines(full_run)
    r = B.run_industry_breakdown(full_run, spec=_spec(store), write=False)
    d = r.to_dict()
    rec = d["reconciliation"]
    assert rec["ok"] and rec["max_abs_daily_residual"] < 1e-12 and rec["delistings"] == "delistings.parquet"
    assert rec["net_max_abs_daily_residual"] < 1e-10  # gross - allocated costs = the recorded net
    for level in ("industry", "ff12"):
        rows = d["levels"][level]["rows"]
        assert sum(x["pnl_total"] for x in rows) == pytest.approx(rec["book_gross_total"], abs=1e-12)
        assert sum(x["gross_share"] for x in rows) == pytest.approx(1.0, abs=1e-12)
        cost = sum(x["cost_ann"] for x in rows) + d["costs"]["unallocated_ann"]
        assert cost == pytest.approx(d["book"]["cost_ann"], rel=1e-9)
        pnl = [x["pnl_total"] for x in rows if x["industry"] != B.UNCLASSIFIED]
        assert pnl == sorted(pnl, reverse=True)  # sorted by P&L contribution
    # each FF12 row is the sum of its industries (FF48_TO_FF12, the risk model's parent map)
    by_parent: dict[str, list[float]] = {}
    for x in d["levels"]["industry"]["rows"]:
        if x["industry"] != B.UNCLASSIFIED:
            assert x["parent"] == FF48_TO_FF12[x["industry"]]
            by_parent.setdefault(x["parent"], [0.0, 0.0])
            by_parent[x["parent"]][0] += x["pnl_total"]
            by_parent[x["parent"]][1] += x["gross"]
    for x in d["levels"]["ff12"]["rows"]:
        if x["industry"] != B.UNCLASSIFIED:  # a sector never held can still have an IC row
            want = by_parent.get(x["industry"], [0.0, 0.0])
            assert [x["pnl_total"], x["gross"]] == pytest.approx(want, abs=1e-12)
    # without the delisting override sum(w r) misses the booked delisting return
    book = B.read_run_book(full_run)
    md = ParquetStore(str(store)).as_market_data(universe_tickers=list(book.tickers))
    r0 = B.booked_returns(md.close, book.run_dates, book.tickers, book.cfg, None)
    assert np.abs(book.gross - (book.W * r0).sum(axis=1)).max() > 1e-8
    # the re-fitted scores are the traded ones; nothing went to the ledger
    sc = d["ic"]["score_check"]
    assert sc["ok"] and sc["n_dates"] == sc["n_run_dates"] > 0
    assert d["ic"]["cache"] == {"risk_fit": "hit", "alpha_fit": "hit"}
    assert _ledger_lines(full_run) == n_ledger


def test_labels_are_the_risk_models_point_in_time_industries(full_run, store):
    book = B.read_run_book(full_run)
    md = ParquetStore(str(store)).as_market_data(universe_tickers=list(book.tickers))
    lab = B.industry_labels(md, book.dec, book.tickers)
    hold = pd.read_parquet(full_run / "holdings.parquet")  # md_u.industry_as_of(t), written by the run
    rows = book.dec.get_indexer(pd.DatetimeIndex(hold["date"]))
    cols = book.tickers.get_indexer(pd.Index(hold["ticker"].astype(str)))
    assert (rows >= 0).all() and (cols >= 0).all()
    assert list(lab[rows, cols]) == list(hold["industry"].astype(str))
    # a dated classification follows MarketData.industry_as_of (qts.risk.exposures.industry_panel)
    from qts.risk.exposures import industry_panel

    t = md.tickers[:6]
    ih = pd.DataFrame([["Banks"] * 3 + [None] * 3, ["Steel"] * 6], index=md.dates[[10, 300]], columns=t)
    md2 = dataclasses.replace(md, industry_history=ih)
    dates = md.dates[[5, 10, 200, 300, 400]]
    got = B.industry_labels(md2, dates, t)
    want = industry_panel(md2, dates)
    assert want is not None
    assert got.tolist() == want[t].astype(object).where(want[t].notna(), None).to_numpy().tolist()


# --------------------------------------------------------------------------- breadth verdicts
def _planted(alpha_inds: list[int], mu: float, n_days: int = 1000, per: int = 10, seed: int = 0):
    """A constant book of +-1/N on 12 industries x ``per`` names; names in ``alpha_inds``
    earn ``mu`` a day in the direction of their weight on top of 2 % daily noise."""
    rng = np.random.default_rng(seed)
    n_ind = 12
    N = n_ind * per
    ind = np.repeat(np.arange(n_ind), per)
    sign = np.where(rng.random(N) < 0.5, -1.0, 1.0)
    W = np.tile(sign / N, (n_days, 1))
    R = rng.normal(0.0, 0.02, (n_days, N)) + mu * np.isin(ind, alpha_inds) * sign
    labels = np.array([FF48_NAMES[4 * k] for k in ind], dtype=object)
    dates = pd.bdate_range("2010-01-04", periods=n_days + 1)
    return B.compute_breakdown(
        ret_dates=pd.DatetimeIndex(dates[1:]),
        W=W,
        R=R,
        gross=(W * R).sum(axis=1),
        industry=np.tile(labels, (n_days, 1)),
    )


def test_planted_single_industry_reads_concentrated_and_uniform_broad():
    one = _planted([3], mu=0.0056)
    lv = one["levels"]["industry"]
    assert lv["verdict"]["label"] == "concentrated", lv["verdict"]
    assert lv["breadth"]["top"][0] == FF48_NAMES[12] and lv["rows"][0]["industry"] == FF48_NAMES[12]
    assert lv["breadth"]["n_eff"] < 3 and lv["breadth"]["rest_share"] < B.CONCENTRATED_MAX_REST_SHARE
    assert [h["industry"] for h in lv["breadth"]["holm"]] == [FF48_NAMES[12]]

    uni = _planted(list(range(12)), mu=0.001)
    lv = uni["levels"]["industry"]
    assert lv["verdict"]["label"] == "broad", lv["verdict"]
    b = lv["breadth"]
    assert b["n_positive"] == b["n_active"] == 12 and b["binom_p"] < 1e-3 and b["n_eff"] > 10
    assert b["rest_share"] >= B.BROAD_MIN_REST_SHARE and b["n_t2"] > b["n_t2_chance"]
    assert uni["reconciliation"]["ok"] and uni["reconciliation"]["max_abs_daily_residual"] < 1e-15

    mid = _planted(list(range(8)), mu=0.001)
    assert mid["levels"]["industry"]["verdict"]["label"] == "moderately concentrated"

    none = _planted([], mu=0.0)
    v = none["levels"]["industry"]["verdict"]
    assert v["label"] == "undetermined" and "not enough edge" in v["text"]


def test_breadth_numbers_by_hand():
    rows = [
        {"industry": "A", "active": True, "pnl_total": 6.0, "t": 3.0, "gross": 1.0},
        {"industry": "B", "active": True, "pnl_total": 2.0, "t": 1.0, "gross": 1.0},
        {"industry": "C", "active": True, "pnl_total": -2.0, "t": -1.0, "gross": 1.0},
        {"industry": "D", "active": False, "pnl_total": 1.0, "t": 0.5, "gross": 0.001},
    ]
    pnl = np.array([[3.0, 1.0, -1.0, 0.5], [3.0, 1.0, -1.0, 0.5]])
    b = B.breadth_stats(rows, ["A", "B", "C", "D"], pnl, pnl.sum(axis=1), tdy=252)
    assert b["n_active"] == 3 and b["n_positive"] == 2 and b["n_held"] == 4
    assert b["binom_p"] == pytest.approx(0.5)  # P(X >= 2 | 3, 1/2)
    assert b["top"] == ["A", "B", "C"] and b["top_share"] == pytest.approx(6.0 / 7.0)
    assert b["n_eff"] == pytest.approx(1.0 / ((0.6**2) + 2 * 0.2**2))  # |P| shares 6/10, 2/10, 2/10
    assert b["n_t2"] == 1 and b["n_t2_chance"] == pytest.approx(0.15)
    assert b["ex_top"]["pnl_ann"] == pytest.approx(252 * 0.5)  # only D is left


# --------------------------------------------------------------------------- the seal
def test_sealed_run_is_refused_on_full_history_data(
    sealed_run, store, seal_file, monkeypatch, capsys, tmp_path
):
    man = json.loads((sealed_run / "manifest.json").read_text())
    assert man["sealed_end"] == "2020-02-28"
    # no seal in force (QTS_SEAL_FILE=none): refused before anything is loaded
    with pytest.raises(B.BreakdownError, match="made on the sealed window"):
        B.run_industry_breakdown(sealed_run, spec=_spec(store), with_ic=False, write=False)
    from qts.cli import main

    argv = ["breakdown", "industries", str(sealed_run), "--store", str(store), "--max-names", "60", "--no-ic"]
    assert main(argv) == 2
    assert "sealed window" in capsys.readouterr().err
    assert not (sealed_run / B.OUTPUT_FILE).exists()
    # a seal that moved is refused too
    monkeypatch.setenv(S.SEAL_FILE_ENV, str(_write_seal(tmp_path, "2020-01-02")))
    with pytest.raises(B.BreakdownError, match="holdout from 2020-03-02"):
        B.run_industry_breakdown(sealed_run, spec=_spec(store), with_ic=False, write=False)
    # under its own seal it is broken down on the sealed window
    monkeypatch.setenv(S.SEAL_FILE_ENV, str(seal_file))
    d = B.run_industry_breakdown(sealed_run, spec=_spec(store), with_ic=False, write=False).to_dict()
    assert d["data"]["window"] == "sealed" and d["data"]["sealed_end"] == "2020-02-28"
    assert d["reconciliation"]["ok"] and pd.Timestamp(d["period"]["end"]) < pd.Timestamp(HOLDOUT)


def test_full_history_run_under_the_seal_is_truncated_and_cold_refit_refused(
    full_run, store, seal_file, monkeypatch
):
    monkeypatch.setenv(S.SEAL_FILE_ENV, str(seal_file))
    d = B.run_industry_breakdown(full_run, spec=_spec(store), with_ic=False, write=False).to_dict()
    assert d["data"]["window"] == "truncated" and pd.Timestamp(d["period"]["end"]) <= pd.Timestamp(
        "2020-02-28"
    )
    assert d["reconciliation"]["ok"] and any("never reads the holdout" in n for n in d["notes"])
    # the sealed data is not the data the run was fitted on: the stage cache is cold
    with pytest.raises(B.BreakdownError, match="--no-refit"):
        B.run_industry_breakdown(full_run, spec=_spec(store), refuse_refit=True, write=False)


# --------------------------------------------------------------------------- JSON and CLI
def test_json_round_trip(full_run, store):
    r = B.run_industry_breakdown(full_run, spec=_spec(store), with_ic=False)
    assert r.json_path is not None and r.json_path == str(full_run / B.OUTPUT_FILE)
    back = B.IndustryBreakdown.read(r.json_path)
    assert back.to_dict() == r.to_dict() and back.format() == B.format_breakdown(r.to_dict())
    d = back.to_dict()
    assert d["schema"] == B.SCHEMA and d["thresholds"] == B.THRESHOLDS and set(B.HELP) <= set(d["help"])
    assert d["ic"]["available"] is False and d["ic"]["reason"] == "--no-ic"
    keys = {"industry", "parent", "active", "long", "short", "net", "gross", "gross_share", "n_long", "n_short",
            "pnl_total", "pnl_ann", "pnl_share", "sharpe", "t", "hit_rate", "n_months", "cost_ann",
            "net_pnl_ann", "ic_mean", "ic_t", "ic_n_dates", "ic_coverage", "ic_mean_names"}  # fmt: skip
    assert all(set(row) == keys for row in d["levels"]["industry"]["rows"])
    with pytest.raises(B.BreakdownError, match="not an industry breakdown"):
        B.IndustryBreakdown.from_dict({"schema": "other"})


def test_store_flags_from_the_run_command():
    s = B.store_spec_from_command(
        "/usr/bin/qts run --store data/store --start 2000-01-01 --max-rank 1500 --config configs/baseline.yaml "
        "--ml --cache -v --set alpha.horizon_days=60"
    )
    assert s == StoreSpec(store="data/store", start="2000-01-01", max_rank=1500)
    s = B.store_spec_from_command("qts run --store data/store --max-names 3000 --min-mcap-ever 3e8")
    assert s is not None and s.max_names == 3000 and s.min_mcap_ever == 3e8 and s.max_rank is None
    assert B.store_spec_from_command("qts demo --stocks 300") is None
    assert B.store_spec_from_command("qts run --yfinance AAPL MSFT") is None
    assert B.store_spec_from_command("pytest -q tests") is None


def test_cli_end_to_end(sealed_run, store, seal_file, monkeypatch, capsys, tmp_path):
    from qts.cli import main

    monkeypatch.setenv(S.SEAL_FILE_ENV, str(seal_file))
    n_ledger = _ledger_lines(sealed_run)
    out_json = tmp_path / "copy.json"
    argv = [
        "breakdown", "industries", sealed_run.name, "--runs-root", str(sealed_run.parent),
        "--store", str(store), "--max-names", "60", "--no-refit", "--json", str(out_json),
    ]  # fmt: skip
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert main(argv) == 0
    out = capsys.readouterr().out
    assert "=== Industry breakdown of run" in out and "BY INDUSTRY" in out and "BY FF12 SECTOR" in out
    assert "VERDICT:" in out and "What the numbers mean:" in out and "score check OK" in out
    d = json.loads((sealed_run / B.OUTPUT_FILE).read_text())
    assert d == json.loads(out_json.read_text())
    assert d["data"]["window"] == "sealed" and d["ic"]["available"] and d["ic"]["score_check"]["ok"]
    assert d["levels"]["industry"]["verdict"]["label"] in {
        "broad", "moderately concentrated", "concentrated", "undetermined",
    }  # fmt: skip
    assert _ledger_lines(sealed_run) == n_ledger
    assert main(["breakdown", "industries", sealed_run.name, "--start", "2000-01-01"]) == 2
    assert "need --store" in capsys.readouterr().err
