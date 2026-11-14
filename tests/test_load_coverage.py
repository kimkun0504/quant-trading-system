"""The store load-coverage guard (qts.research.coverage) and ``qts run --max-rank``.

Background: docs/notes/prefilter_lookahead_2026-11-14.md.  The peak-market-cap prefilter
(``--max-names``) can leave out names the point-in-time universe picks; the guard refuses
such a load, ``--max-rank`` covers the universe by construction, and with ``--max-rank``
the universe on a date does not depend on where the load window ends.
"""

from __future__ import annotations

import warnings

import pytest

from qts.core.config import SystemConfig, UniverseConfig
from qts.data.store import ParquetStore
from qts.data.synthetic import SyntheticSpec, generate
from qts.data.universe import raw_eligibility
from qts.research import coverage as C
from qts.research.data import StoreSpec, load_research_market, load_store_market

UCFG = UniverseConfig(top_n_by_mcap=20, min_price=1.0, min_adv_usd=0.0)


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    md = generate(SyntheticSpec(n_stocks=80, n_days=640, seed=5, n_delistings=3))
    root = tmp_path_factory.mktemp("cov") / "store"
    ParquetStore(str(root)).write_market_data(md)
    return root


@pytest.fixture
def enforce(monkeypatch):
    monkeypatch.setenv(C.ENV, "enforce")


def test_needed_and_recommended_ranks():
    base = SystemConfig()  # top 1000, bottom-size cut 0.2
    from qts.research.midcap import needed_loaded_names

    assert needed_loaded_names(base) == 1250
    assert C.recommended_max_rank(base) == 1500
    assert C.recommended_max_rank(base.universe) == 1500  # a UniverseConfig works too
    assert C.recommended_max_rank(UCFG) == 250  # 20 / 0.8 = 25 -> x1.2 -> rounded up to 250


def test_max_rank_is_covered_by_construction_without_touching_the_store(tmp_path, enforce):
    spec = StoreSpec(store=str(tmp_path / "does-not-exist"), max_rank=1500)
    cov = C.require_load_coverage(spec, SystemConfig())
    assert cov.ok and cov.method == "by construction" and cov.detail == ""
    low = C.check_load_coverage(StoreSpec(store="x", max_rank=1300), SystemConfig())
    assert low.ok and "under 1.2x" in low.detail  # allowed, with a warning
    with pytest.raises(C.LoadCoverageError, match="--max-rank 1500"):
        C.require_load_coverage(StoreSpec(store="x", max_rank=1000), SystemConfig())


def test_peak_cap_prefilter_that_misses_names_is_refused(store, enforce):
    # 22 ids by peak cap for a top-20 universe that needs 25 loaded names per date
    spec = StoreSpec(store=str(store), max_names=22, min_price=1.0, min_adv_usd=0.0)
    cov = C.check_load_coverage(spec, UCFG)
    assert cov.method == "measured" and not cov.ok and cov.n_short_dates and cov.n_short_dates > 0
    with pytest.raises(C.LoadCoverageError, match="incomplete universe"):
        load_research_market(spec, seal=None, universe=UCFG, hash_cache_dir=None)
    # allowed explicitly: loads, and the coverage goes on the ledger record
    msgs: list[str] = []
    ld = load_research_market(
        spec, seal=None, universe=UCFG, hash_cache_dir=None, allow_partial_coverage=True, log=msgs.append
    )
    assert any(m.startswith("WARNING") for m in msgs)
    extra = dict(ld.context(1).record_extra)
    assert extra["load_coverage"]["ok"] is False and extra["load_coverage"]["method"] == "measured"


def test_warn_and_off_modes(store, monkeypatch):
    spec = StoreSpec(store=str(store), max_names=22, min_price=1.0, min_adv_usd=0.0)
    monkeypatch.setenv(C.ENV, "warn")
    msgs: list[str] = []
    assert not C.require_load_coverage(spec, UCFG, log=msgs.append).ok
    assert any(m.startswith("WARNING") for m in msgs)
    monkeypatch.setenv(C.ENV, "off")
    assert C.require_load_coverage(spec, UCFG).method == "skipped"
    monkeypatch.setenv(C.ENV, "bogus")
    with pytest.raises(C.LoadCoverageError):
        C.require_load_coverage(spec, UCFG)


def test_covering_peak_cap_prefilter_and_no_prefilter_pass(store, enforce):
    cov = C.require_load_coverage(StoreSpec(store=str(store), max_names=80), UCFG)
    assert cov.ok and cov.share_dates_full == pytest.approx(1.0)
    assert C.require_load_coverage(StoreSpec(store=str(store)), UCFG).method == "no prefilter"


def test_max_rank_universe_does_not_depend_on_the_window_end(store):
    """The look-ahead itself: with --max-rank the PIT universe before a cut date is the
    same whether the load window ends at the cut or at the store's end."""
    full = load_store_market(StoreSpec(store=str(store), max_rank=40, min_price=1.0, min_adv_usd=0.0))
    cut = str(full.dates[400].date())
    short = load_store_market(
        StoreSpec(store=str(store), max_rank=40, min_price=1.0, min_adv_usd=0.0), end=cut
    )
    u_full = raw_eligibility(full, UCFG).loc[:cut]
    u_short = raw_eligibility(short, UCFG)
    for d in u_short.index[70:]:  # after the ADV window has filled
        a = set(u_full.columns[u_full.loc[d].to_numpy()])
        b = set(u_short.columns[u_short.loc[d].to_numpy()])
        assert a == b, d


def test_qts_run_max_rank_flag(store, tmp_path, monkeypatch):
    from qts.cli import build_parser, main

    a = build_parser().parse_args(["run", "--store", "x", "--max-rank", "1500", "--allow-partial-coverage"])
    assert a.max_rank == 1500 and a.allow_partial_coverage
    assert StoreSpec.from_args(a).max_rank == 1500
    # --max-rank with a peak-cap flag is refused
    rc = main(["run", "--store", str(store), "--max-rank", "40", "--max-names", "40", "--no-save"])
    assert rc == 2
    # a covered run goes through, an uncovered one is refused before anything is loaded
    monkeypatch.setenv(C.ENV, "enforce")
    argv = ["run", "--store", str(store), "--set", "universe.top_n_by_mcap=20", "--set", "universe.min_price=1",
            "--set", "universe.min_adv_usd=0", "--max-days", "3", "--no-save",
            "--ledger", str(tmp_path / "ledger.jsonl")]  # fmt: skip
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert main([*argv, "--max-names", "22"]) == 2
        assert main([*argv, "--max-rank", "40"]) == 0
