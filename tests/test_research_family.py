"""Research-family identity of the trial ledger (handbook 10 "Research family"):
``market_hash``, ``qts ledger list | rekey | merge-families | restate``."""

from __future__ import annotations

import dataclasses
import io
import json
import warnings
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qts.cli import data_hash_aliases, main, manifest_market_hashes
from qts.core.config import SystemConfig
from qts.core.manifest import RunManifest, hash_market, hash_market_data
from qts.data.synthetic import SyntheticSpec, generate
from qts.pipeline import ResearchPipeline
from qts.runs.artifacts import read_extras
from qts.runs.restate import find_run_record, restate_dsr
from qts.validation import TrialLedger
from qts.validation.deflated_sharpe import deflated_sharpe_ratio_annual
from qts.validation.ledger import rekeyed_family, research_family


@pytest.fixture(scope="module")
def market():
    return generate(SyntheticSpec(n_stocks=60, n_days=640, seed=5))


def _with_reasons(md):
    reasons = pd.Series({md.tickers[0]: "acquisition", md.tickers[1]: "performance"})
    return dataclasses.replace(md, delisting_reasons=reasons)


def _cli(*argv) -> tuple[int, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = main(list(argv))
    return rc, out.getvalue() + err.getvalue()


# ----------------------------------------------------------------------------- the hash
def test_market_hash_covers_the_adjusted_close_panel_only(market):
    md = market
    base, full = hash_market(md), hash_market_data(md)
    assert len(base) == 16 and base != full
    same_market = {
        "delisting_reasons": _with_reasons(md),
        "delisting_dates": dataclasses.replace(md, delisting_dates=pd.Series({md.tickers[0]: md.dates[-5]})),
        "volume": dataclasses.replace(md, volume=md.volume * 2.0),
        "market_cap": dataclasses.replace(md, market_cap=md.market_cap * 1.01),
        "industry": dataclasses.replace(md, industry=md.industry.iloc[::-1].set_axis(md.industry.index)),
        "fundamentals": dataclasses.replace(md, book_to_price=md.close * 0.0 + 0.5),
        "eligible": dataclasses.replace(
            md, eligible=md.close.notna() & (md.market_cap > md.market_cap.median())
        ),
        "extra": dataclasses.replace(md, extra={"x": md.close * 0.0}),
    }
    for name, m in same_market.items():
        assert hash_market(m) == base, name  # same price history -> same research family
        assert hash_market_data(m) != full, name  # ... although the input is not bit-identical
    close2 = md.close.copy()
    close2.iloc[100, 3] *= 1.0001
    other_market = {
        "one changed close": dataclasses.replace(md, close=close2),
        "fewer ids": dataclasses.replace(md, close=md.close.iloc[:, :-1]),
        "fewer dates": dataclasses.replace(md, close=md.close.iloc[:-1]),
        "relabelled ids": dataclasses.replace(
            md, close=md.close.set_axis([f"X{c}" for c in md.close.columns], axis=1)
        ),
    }
    for name, m in other_market.items():
        assert hash_market(m) != base, name
    assert hash_market(md) == hash_market(dataclasses.replace(md, close=md.close.copy()))  # deterministic


def test_data_hash_aliases_cover_the_hash_before_a_reference_table_existed(market):
    md = dataclasses.replace(
        _with_reasons(market), delisting_dates=pd.Series({market.tickers[0]: market.dates[-5]})
    )
    alias = data_hash_aliases(md)
    assert set(alias.values()) == {hash_market(md)}
    before_reasons = dataclasses.replace(md, delisting_reasons=None)
    before_both = dataclasses.replace(md, delisting_reasons=None, delisting_dates=None)
    for m in (md, before_reasons, before_both):
        assert hash_market_data(m) in alias  # what an earlier run on the same panels recorded
    assert len(alias) == 4
    assert hash_market_data(dataclasses.replace(md, volume=md.volume * 2)) not in alias


# ----------------------------------------------------------------------------- ledger rewrites
def _log(led, family, data_hash, sharpe, market_hash="", **kw):
    return led.log(
        family,
        {"x": sharpe},
        {"sharpe_annual": sharpe, "n_obs": 1000},
        data_hash=data_hash,
        family=family,
        market_hash=market_hash,
        **kw,
    )


def _legacy_ledger(tmp_path) -> TrialLedger:
    """Eight trials filed under the data hash D1, one under D2 (same market, a
    reference table added), one on another market, one hand-named family."""
    led = TrialLedger(str(tmp_path / "ledger.jsonl"))
    for i in range(8):
        _log(led, "pipeline_phase2:d1d1d1d1d1d1", "d1d1d1d1d1d1aaaa", 0.2 + 0.1 * i)
    _log(led, "pipeline_phase2:d2d2d2d2d2d2", "d2d2d2d2d2d2bbbb", 1.3)
    _log(led, "pipeline_phase2:e0e0e0e0e0e0", "e0e0e0e0e0e0cccc", 0.7)
    _log(led, "my_search", "d1d1d1d1d1d1aaaa", 0.4)
    return led


def test_rekeyed_family_rules():
    rec = {"family": "pipeline_phase2:d1d1d1d1d1d1", "data_hash": "d1d1d1d1d1d1aaaa"}
    assert rekeyed_family(rec) is None  # market hash unknown
    assert rekeyed_family(rec, {"d1d1d1d1d1d1aaaa": "m" * 16}) == ("pipeline_phase2:" + "m" * 12, "m" * 16)
    assert rekeyed_family({**rec, "market_hash": "f" * 16}) == ("pipeline_phase2:" + "f" * 12, "f" * 16)
    assert rekeyed_family({**rec, "family": "my_search"}, {"d1d1d1d1d1d1aaaa": "m" * 16}) is None
    # suffix is not the record's own data hash (already re-keyed or merged): left alone
    assert rekeyed_family({**rec, "family": "pipeline_phase2:" + "m" * 12, "market_hash": "m" * 16}) is None
    assert research_family(2, "abcdef0123456789") == "pipeline_phase2:abcdef012345"


def test_rekey_moves_only_records_whose_market_hash_is_known(tmp_path):
    led = _legacy_ledger(tmp_path)
    path = Path(led.path)
    before = path.read_text()
    M = "a1b2c3d4e5f60718"
    assert led.rekey_families() == [] and path.read_text() == before  # nothing known yet
    assert led.unkeyed_families() == {
        "pipeline_phase2:d1d1d1d1d1d1": 8,
        "pipeline_phase2:d2d2d2d2d2d2": 1,
        "pipeline_phase2:e0e0e0e0e0e0": 1,
    }
    # a new-format run on D2 teaches the ledger D2 -> M
    fam = research_family(2, M)
    _log(led, fam, "d2d2d2d2d2d2bbbb", 1.1, market_hash=M)
    before = path.read_text()
    dry = led.rekey_families(dry_run=True)
    assert [c["old"] for c in dry] == ["pipeline_phase2:d2d2d2d2d2d2"] and path.read_text() == before
    assert not list(tmp_path.glob("*.bak-*"))
    # ... and the store (or a manifest) supplies D1 -> M
    changes = led.rekey_families({"d1d1d1d1d1d1aaaa": M})
    assert len(changes) == 9 and {c["new"] for c in changes} == {fam}
    assert len(list(tmp_path.glob("ledger.jsonl.bak-*"))) == 1
    assert next(tmp_path.glob("ledger.jsonl.bak-*")).read_text() == before  # the backup is the old file
    raw = led.raw_records()
    assert len(raw) == 12  # no record dropped
    assert led.n_trials(fam) == 10  # 8 + 1 + the new run
    moved = [r for r in raw if r.get("family_rekeyed_from")]
    assert len(moved) == 9 and all(r["market_hash"] == M and r["name"] == fam for r in moved)
    assert {r["data_hash"] for r in moved} == {
        "d1d1d1d1d1d1aaaa",
        "d2d2d2d2d2d2bbbb",
    }  # kept for reproducibility
    assert led.n_trials("pipeline_phase2:e0e0e0e0e0e0") == 1 and led.n_trials("my_search") == 1
    assert led.unkeyed_families() == {"pipeline_phase2:e0e0e0e0e0e0": 1}
    assert led.rekey_families({"d1d1d1d1d1d1aaaa": M}) == []  # idempotent
    assert len(list(tmp_path.glob("ledger.jsonl.bak-*"))) == 1
    # the DSR inputs now see the whole search
    assert led.dsr_inputs(fam, 1000)["n_trials_raw"] == 10


def test_merge_families_is_explicit_backed_up_and_lossless(tmp_path):
    led = _legacy_ledger(tmp_path)
    path = Path(led.path)
    before = path.read_text()
    a, b, other = (
        "pipeline_phase2:d1d1d1d1d1d1",
        "pipeline_phase2:d2d2d2d2d2d2",
        "pipeline_phase2:e0e0e0e0e0e0",
    )
    with pytest.raises(KeyError):
        led.merge_families(b, [a, "pipeline_phase2:typo"])
    assert path.read_text() == before
    assert len(led.merge_families(b, [a], dry_run=True)) == 8 and path.read_text() == before
    changes = led.merge_families(b, [a, b])  # the target among the sources is ignored
    assert len(changes) == 8 and {(c["old"], c["new"]) for c in changes} == {(a, b)}
    assert next(tmp_path.glob("ledger.jsonl.bak-*")).read_text() == before
    assert len(led.raw_records()) == 11 and led.n_trials(b) == 9 and led.n_trials(a) == 0
    assert led.n_trials(other) == 1 and led.n_trials("my_search") == 1
    merged = [r for r in led.raw_records() if r.get("family_merged_from")]
    assert len(merged) == 8 and all(r["family_merged_from"] == a and r["name"] == b for r in merged)
    assert all(r["data_hash"] == "d1d1d1d1d1d1aaaa" for r in merged)
    want = sorted([0.2 + 0.1 * i for i in range(8)] + [1.3])
    np.testing.assert_allclose(sorted(led.trial_sharpes_annual(b)), want)
    # merged again into a named family: the original family is still on the record
    led.merge_families("us_equities", [b])
    assert led.n_trials("us_equities") == 9
    assert {r.get("family_merged_from") for r in led.records("us_equities")} == {a, b}
    with pytest.raises(KeyError):
        led.merge_families("us_equities", [a])  # no longer exists


def test_families_summary_and_cli_list(tmp_path):
    led = _legacy_ledger(tmp_path)
    rows = {r["family"]: r for r in led.families()}
    assert list(rows) == [
        "pipeline_phase2:d1d1d1d1d1d1",
        "pipeline_phase2:d2d2d2d2d2d2",
        "pipeline_phase2:e0e0e0e0e0e0",
        "my_search",
    ]
    r = rows["pipeline_phase2:d1d1d1d1d1d1"]
    assert r["n_records"] == r["n_trials"] == 8 and r["data_hashes"] == ["d1d1d1d1d1d1aaaa"]
    assert r["market_hashes"] == [] and r["best_sharpe"] == pytest.approx(0.9) and r["n_effective"] == 8.0
    rc, out = _cli("ledger", "list", "--path", led.path)
    assert rc == 0 and "pipeline_phase2:d1d1d1d1d1d1" in out and "my_search" in out
    assert "10 record(s) in 3 famil(ies) are keyed on a data hash" in out
    rc, out = _cli("ledger", "merge-families", "--path", led.path, "--into", "x", "nope")
    assert rc == 2 and "no such family" in out
    rc, out = _cli(
        "ledger", "merge-families", "--path", led.path, "--dry-run",
        "--into", "pipeline_phase2:d2d2d2d2d2d2", "pipeline_phase2:d1d1d1d1d1d1",
    )  # fmt: skip
    assert rc == 0 and "would merge 8 record(s)" in out and led.n_trials("pipeline_phase2:d1d1d1d1d1d1") == 8
    rc, out = _cli(
        "ledger", "merge-families", "--path", led.path,
        "--into", "pipeline_phase2:d2d2d2d2d2d2", "pipeline_phase2:d1d1d1d1d1d1",
    )  # fmt: skip
    assert (
        rc == 0
        and "merge 8 record(s); backup at" in out
        and led.n_trials("pipeline_phase2:d2d2d2d2d2d2") == 9
    )


# ----------------------------------------------------------------------------- end to end
def _run(md, cfg, ledger, runs_root):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pipe = ResearchPipeline(cfg, use_ml=False, ledger_path=ledger, runs_root=runs_root)
        res = pipe.run(md, max_days=30)
    return pipe, res


def _as_written_before_the_change(path: Path, data_hash: str) -> None:
    """Rewrite the records with ``data_hash`` the way the pipeline wrote them until
    2026-11-12: family keyed on the data hash, no market hash."""
    out = []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r["data_hash"] == data_hash:
            r.pop("market_hash", None)
            r["family"] = r["name"] = f"pipeline_phase1:{data_hash[:12]}"
        out.append(json.dumps(r))
    path.write_text("\n".join(out) + "\n")


@pytest.fixture(scope="module")
def story(market, tmp_path_factory):
    """The real-data sequence on a synthetic market: two trials on the panels
    without ``delisting_reasons`` (filed, as then, under the data hash), then the
    table is added and a third configuration is run."""
    root = tmp_path_factory.mktemp("family")
    ledger, runs = str(root / "ledger.jsonl"), str(root / "runs")
    cfg = SystemConfig().with_overrides({"universe.top_n_by_mcap": 50})
    p1, r1 = _run(market, cfg, ledger, runs)
    _, r2 = _run(market, cfg.with_overrides({"portfolio.risk_aversion": 5.0}), ledger, runs)
    # the family holds both trials although nothing but the config changed
    assert r1.ledger_family == r2.ledger_family and r2.dsr_inputs["n_trials_raw"] == 2
    # forget what the new code recorded for them (ledger and manifests), as for the owner's old runs
    _as_written_before_the_change(Path(ledger), r1.data_hash)
    for r in (r1, r2):
        mf = Path(r.run_dir) / "manifest.json"
        m = json.loads(mf.read_text())
        m.pop("market_hash"), m.pop("ledger_family")
        mf.write_text(json.dumps(m))
    md2 = _with_reasons(market)
    p3, r3 = _run(md2, cfg.with_overrides({"portfolio.risk_aversion": 8.0}), ledger, runs)
    return {"root": root, "ledger": ledger, "runs": runs, "md2": md2, "r": (r1, r2, r3), "p": (p1, p3)}


def test_auxiliary_table_changes_the_data_hash_but_not_the_family(story):
    r1, _, r3 = story["r"]
    assert r3.data_hash != r1.data_hash  # delisting_reasons is part of the reproducibility hash
    assert r3.market_hash == r1.market_hash == hash_market(story["md2"])
    assert r3.ledger_family == r1.ledger_family == research_family(1, r3.market_hash)
    m = RunManifest.load(r3.run_dir)
    assert (m.data_hash, m.market_hash, m.ledger_family) == (r3.data_hash, r3.market_hash, r3.ledger_family)
    summ = json.loads((Path(r3.run_dir) / "summary.json").read_text())
    assert summ["market_hash"] == r3.market_hash and summ["ledger_family"] == r3.ledger_family
    rec = TrialLedger(story["ledger"]).raw_records()[-1]
    assert rec["data_hash"] == r3.data_hash and rec["market_hash"] == r3.market_hash
    assert rec["family"] == r3.ledger_family
    old = RunManifest.load(r1.run_dir)  # a manifest written before the fields existed still loads
    assert old.market_hash == "" and old.ledger_family == ""


def test_rekey_then_restate_recovers_the_honest_trial_count(story):
    r1, _r2, r3 = story["r"]
    ledger, runs = story["ledger"], story["runs"]
    led = TrialLedger(ledger)
    old_family = f"pipeline_phase1:{r1.data_hash[:12]}"
    # run 3 saw only itself: the earlier trials sat in the data-hash family ...
    assert r3.dsr_inputs["n_trials_raw"] == 1 and r3.dsr_inputs["n_trials"] == 1.0
    assert led.n_trials(old_family) == 2 and led.n_trials(r3.ledger_family) == 1
    # ... and since the earlier records carry no market hash the pipeline could not know (no note)
    assert not any("NOT counted" in n for n in r3.notes)

    # records alone cannot map the old data hash; neither can the old manifests
    rc, out = _cli("ledger", "rekey", "--path", ledger, "--runs-root", runs, "--dry-run")
    assert rc == 0 and "nothing to re-key" in out and f"{old_family}  2 record(s)" in out
    assert "merge-families" in out
    assert manifest_market_hashes(runs) == {r3.data_hash: r3.market_hash}
    # the data can: the store's panels hash to the old data hash once delisting_reasons is left out
    alias = data_hash_aliases(story["md2"])
    assert alias[r1.data_hash] == r3.market_hash and alias[r3.data_hash] == r3.market_hash
    changes = led.rekey_families(alias)
    assert [(c["old"], c["new"]) for c in changes] == [(old_family, r3.ledger_family)] * 2
    assert led.n_trials(r3.ledger_family) == 3 and led.n_trials(old_family) == 0

    # restate the finished run at the family's current count: no data, no backtest
    summary_before = (Path(r3.run_dir) / "summary.json").read_text()
    extras_before = read_extras(r3.run_dir)
    out = restate_dsr(r3.run_dir, ledger)
    assert out["family"] == r3.ledger_family and out["run_in_family"]
    assert out["n_trials_raw"] == 3 and out["original"]["n_trials_raw"] == 1
    net = pd.read_parquet(Path(r3.run_dir) / "returns.parquet")["net"]
    inputs = led.dsr_inputs(r3.ledger_family, len(net))
    assert out["n_trials"] == inputs["n_trials"] == pytest.approx(led.effective_trials_for(r3.ledger_family))
    assert out["var_trials_sr_annual"] == inputs["var_trials_sr_annual"] >= 252 / len(net)
    by_hand = deflated_sharpe_ratio_annual(
        r3.summary["sharpe_net"],
        n_trials=inputs["n_trials"],
        var_trials_sr_annual=inputs["var_trials_sr_annual"],
        n_obs=len(net),
        skew=float(net.skew()),
        kurt=float(net.kurt() + 3),
    )
    assert out["dsr"] == pytest.approx(by_hand, rel=1e-12)
    assert out["dsr"] < r3.dsr and out["original"]["dsr"] == pytest.approx(r3.dsr)  # more trials, lower DSR
    assert out["expected_max_sharpe_annual"] > 0
    extras = read_extras(r3.run_dir)
    assert extras["dsr_restated"]["dsr"] == out["dsr"]
    assert {k: v for k, v in extras.items() if k != "dsr_restated"} == extras_before  # nothing else touched
    assert (Path(r3.run_dir) / "summary.json").read_text() == summary_before  # the run's own report stays

    # the same through the CLI, for an old run (no market hash in its manifest): family from its ledger record
    rc, text = _cli("ledger", "restate", Path(r1.run_dir).name, "--path", ledger, "--runs-root", runs)
    assert rc == 0 and "raw 3" in text and "Nyholt N_eff" in text and "var SR" in text and "written:" in text
    assert read_extras(r1.run_dir)["dsr_restated"]["family"] == r3.ledger_family
    rec = find_run_record(led, r1.config_hash, r1.data_hash, RunManifest.load(r1.run_dir).created_at)
    assert rec is not None and rec["family_rekeyed_from"] == old_family
    rc, text = _cli("ledger", "restate", "no_such_run", "--path", ledger, "--runs-root", runs)
    assert rc == 1 and "not a run directory" in text


def test_restating_an_up_to_date_run_reproduces_its_dsr(story, market):
    """With no change to the family since the run finished, the restated DSR is the reported one."""
    cfg = SystemConfig().with_overrides({"universe.top_n_by_mcap": 50, "portfolio.risk_aversion": 3.0})
    _, r4 = _run(story["md2"], cfg, story["ledger"], story["runs"])
    out = restate_dsr(r4.run_dir, story["ledger"], write=False)
    assert out["n_trials_raw"] == r4.dsr_inputs["n_trials_raw"]
    assert out["n_trials"] == pytest.approx(r4.dsr_inputs["n_trials"])
    assert out["dsr"] == pytest.approx(r4.dsr, rel=1e-9) and np.isfinite(out["dsr"])
    assert "dsr_restated" not in read_extras(r4.run_dir)  # write=False


def test_pipeline_notes_earlier_records_it_could_not_count(market, tmp_path):
    """A record of the same data hash under the old family is certain to belong to this
    run's family: the run says so instead of silently reporting n_trials = 1."""
    ledger = str(tmp_path / "ledger.jsonl")
    cfg = SystemConfig().with_overrides({"universe.top_n_by_mcap": 50})
    _, r1 = _run(market, cfg, ledger, None)
    _as_written_before_the_change(Path(ledger), r1.data_hash)
    _, r2 = _run(market, cfg.with_overrides({"portfolio.risk_aversion": 5.0}), ledger, None)
    assert r2.dsr_inputs["n_trials_raw"] == 1
    (note,) = [n for n in r2.notes if "NOT counted" in n]
    assert "1 earlier record(s)" in note and "qts ledger rekey" in note
    rc, out = _cli("ledger", "rekey", "--path", ledger, "--runs-root", str(tmp_path / "none"))
    assert rc == 0 and "re-key 1 record(s); backup at" in out
    assert TrialLedger(ledger).n_trials(r2.ledger_family) == 2
