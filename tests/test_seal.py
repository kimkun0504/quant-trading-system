"""Research holdout seal (``research/seal.yaml``, :mod:`qts.research.seal`).

* the seal file: parsing, strictness, ``is_sealed``, ``$QTS_SEAL_FILE``;
* point in time: a sealed load ends before ``holdout_start``; rewriting every stored
  value on or after the seal (prices, volume, market cap, fundamentals) changes neither
  the prefilter's ids, nor any loaded panel, nor any feature, label, risk input or
  score of the fitted pipeline -- bit for bit; labels whose window reaches the cut are NaN;
* the family: a sealed run logs into the family of the full-history panel (both hashes on
  the record); a sealed re-run of a counted configuration is a reference, not a trial;
* unseal: once per configuration, logged before the run, a second refused without
  ``--force`` and then flagged "holdout already used"; ``qts seal status``;
* exemptions: ``qts demo`` and ``qts check midcap``.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qts.core.config import SystemConfig
from qts.core.manifest import hash_market
from qts.data.store import ParquetStore
from qts.data.synthetic import SyntheticSpec, generate
from qts.research import seal as S
from qts.research.data import StoreSpec, full_history_market_hash, load_research_market
from qts.validation import TrialLedger
from qts.validation.ledger import research_family

SPEC = {"n_stocks": 60, "n_days": 640, "seed": 5}
HOLDOUT = "2020-03-02"  # synthetic market: 2018-01-02 .. 2020-06-15 (sealed load: .. 2020-02-28)


def _write_seal(tmp_path: Path, start: str = HOLDOUT, **extra) -> Path:
    import yaml

    p = tmp_path / "seal.yaml"
    p.write_text(yaml.safe_dump({"holdout_start": start, "reason": "test", **extra}))
    return p


@pytest.fixture(scope="module")
def market():
    return generate(SyntheticSpec(**SPEC))


@pytest.fixture
def store(market, tmp_path):
    root = tmp_path / "store"
    ParquetStore(str(root)).write_market_data(market)
    return root


@pytest.fixture
def sealed_env(tmp_path, monkeypatch):
    p = _write_seal(tmp_path)
    monkeypatch.setenv(S.SEAL_FILE_ENV, str(p))
    return p


# --------------------------------------------------------------------------- the file
def test_seal_file_parsing_and_is_sealed(tmp_path, monkeypatch):
    p = _write_seal(tmp_path, inherit_family="pipeline_phase2:abc")
    s = S.load_seal(p)
    assert s is not None and s.holdout_start_iso == HOLDOUT and s.load_end_iso == "2020-03-01"
    assert s.inherit_family == ("pipeline_phase2:abc",)
    assert s.is_sealed("2020-03-02") and s.is_sealed("2021-05-01") and not s.is_sealed("2020-03-01 23:59")
    assert S.is_sealed("2020-03-03", s) and not S.is_sealed("2020-02-28", s)
    bad = tmp_path / "bad.yaml"
    bad.write_text("holdout_start: 2020-01-02\nholdout_strat: x\n")
    with pytest.raises(S.SealError, match="unknown key"):
        S.load_seal(bad)
    bad.write_text("holdout_start: not-a-date\n")
    with pytest.raises(S.SealError, match="not a date"):
        S.load_seal(bad)
    bad.write_text("reason: no start\n")
    with pytest.raises(S.SealError, match="holdout_start"):
        S.load_seal(bad)
    monkeypatch.setenv(S.SEAL_FILE_ENV, "none")
    assert S.load_seal() is None and not S.is_sealed("2030-01-01")
    monkeypatch.setenv(S.SEAL_FILE_ENV, str(p))
    assert S.load_seal() == s
    json.dumps(s.to_dict())


def test_the_committed_seal():
    from qts.research.common import repo_root

    s = S.load_seal(repo_root() / S.DEFAULT_SEAL_FILE)
    assert s is not None and s.holdout_start_iso == "2023-11-09" and s.load_end_iso == "2023-11-08"
    assert "pipeline_phase2:c1da477b63d2" in s.inherit_family and s.reason


# --------------------------------------------------------------------------- point in time
def _perturb_after(root: Path, start: str) -> None:
    """Rewrite every wide panel's values dated >= start (the sealed part of the store)."""
    st = ParquetStore(str(root))
    rng = np.random.default_rng(0)
    for name in st.datasets():
        if st.meta(name).get("kind") != "wide" or name == "industry_history":
            continue
        w = st.read_wide(name)
        late = w.index >= pd.Timestamp(start)
        if not late.any():
            continue
        w.loc[late] = w.loc[late] * rng.uniform(0.2, 5.0, size=w.loc[late].shape)
        st.write_wide(name, w, mode="replace")


def _fit(md, top_n=40):
    from qts.pipeline import ResearchPipeline

    cfg = SystemConfig().with_overrides({"universe.top_n_by_mcap": top_n})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ResearchPipeline(cfg, use_ml=False).fit(md)


def test_sealed_load_and_fit_never_see_the_holdout(store, tmp_path):
    seal = S.load_seal(_write_seal(tmp_path))
    spec = StoreSpec(store=str(store), max_names=50)
    a = load_research_market(spec, seal=seal, hash_cache_dir=None)
    assert a.sealed and not a.unsealed and a.md.dates[-1] < seal.holdout_start
    assert a.md.dates[-1] == pd.Timestamp("2020-02-28")
    full_a = full_history_market_hash(spec)
    pipe_a = _fit(a.md)
    _perturb_after(store, HOLDOUT)
    b = load_research_market(spec, seal=seal, hash_cache_dir=None)
    # the prefilter's ids, every panel and the data identity are unchanged ...
    assert list(b.md.tickers) == list(a.md.tickers)
    from qts.pipeline import data_hash_of

    assert data_hash_of(b.md) == data_hash_of(a.md) and hash_market(b.md) == hash_market(a.md)
    # ... while the store's full history did change (the family identity follows it)
    assert full_history_market_hash(spec) != full_a
    pipe_b = _fit(b.md)
    assert pipe_b.risk.factor_returns.equals(pipe_a.risk.factor_returns)
    assert pipe_b.risk.specific_returns.equals(pipe_a.risk.specific_returns)
    assert pipe_b.alpha.X_.equals(pipe_a.alpha.X_)
    assert pipe_b.alpha.target_.equals(pipe_a.alpha.target_)
    assert pipe_b.alpha.traded_score_.equals(pipe_a.alpha.traded_score_)
    d = pipe_a.md_u.dates[-5]
    ra, rb = pipe_a.risk.as_of(d), pipe_b.risk.as_of(d)
    assert ra.factor_cov.equals(rb.factor_cov) and ra.specific_var.equals(rb.specific_var)
    # labels whose forward window would reach the cut are missing, not filled from the holdout
    h = pipe_a.cfg.alpha.horizon_days + pipe_a.alpha.signal_lag_days
    assert pipe_a.alpha.target_.iloc[-h:].isna().all().all()


def test_apply_seal_cuts_reference_tables(tmp_path):
    md = generate(SyntheticSpec(n_stocks=30, n_days=600, seed=1, with_earnings=True, n_delistings=4))
    seal = S.load_seal(_write_seal(tmp_path, start=str(md.dates[400].date())))
    out = S.apply_seal(md, seal)
    assert out.dates[-1] < seal.holdout_start
    assert pd.to_datetime(out.earnings_dates["announce_date"]).max() < seal.holdout_start
    assert pd.to_datetime(out.earnings_fundamentals["datekey"]).max() < seal.holdout_start
    dd = pd.to_datetime(out.delisting_dates).dropna()
    assert (dd < seal.holdout_start).all()


def test_full_history_hash_is_cached_and_invalidated(store, tmp_path):
    from qts.research.data import cached_full_history_market_hash

    spec = StoreSpec(store=str(store), max_names=50)
    cache = tmp_path / "mh"
    assert cached_full_history_market_hash(spec, str(cache)) is None  # nothing computed yet
    assert cached_full_history_market_hash(spec, None) is None
    h1 = full_history_market_hash(spec, cache_dir=str(cache))
    assert len(list(cache.glob("*.json"))) == 1
    assert full_history_market_hash(spec, cache_dir=str(cache)) == h1
    assert cached_full_history_market_hash(spec, str(cache)) == h1  # the lookup reads no panel
    assert cached_full_history_market_hash(StoreSpec(store=str(store), max_names=40), str(cache)) is None
    _perturb_after(store, HOLDOUT)  # rewrites the files: new fingerprint, new entry
    assert cached_full_history_market_hash(spec, str(cache)) is None  # a stale entry is never served
    h2 = full_history_market_hash(spec, cache_dir=str(cache))
    assert h2 != h1 and len(list(cache.glob("*.json"))) == 2
    md = ParquetStore(str(store)).as_market_data(
        universe_tickers=ParquetStore(str(store)).tickers_by_market_cap(max_names=50)
    )
    assert h2 == hash_market(md)  # what an unsealed `qts run` with the same flags hashes


# --------------------------------------------------------------------------- the CLI
def _run_argv(store, tmp_path, *extra):
    return [
        "run",
        "--store", str(store),
        "--max-names", "50",
        "--set", "universe.top_n_by_mcap=40",
        "--max-days", "5",
        "--ledger", str(tmp_path / "ledger.jsonl"),
        "--runs-root", str(tmp_path / "runs"),
        "--cache",
        *extra,
    ]  # fmt: skip


def _last_summary(tmp_path) -> dict:
    from qts.pipeline import list_runs

    d = list_runs(tmp_path / "runs")[0]
    return json.loads((d / "summary.json").read_text()), json.loads((d / "manifest.json").read_text())


def test_sealed_run_family_reference_unseal_once_and_status(store, tmp_path, sealed_env, capsys):
    from qts.cli import main

    led_path = tmp_path / "ledger.jsonl"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # 1) a sealed run: data through the day before the holdout, full-history family
        assert main(_run_argv(store, tmp_path)) == 0
        out = capsys.readouterr().out
        assert "seal: holdout from 2020-03-02 sealed" in out and "seal: data through 2020-02-28" in out
        summ, man = _last_summary(tmp_path)
        assert (
            summ["sealed_end"] == "2020-02-28"
            and summ["unsealed"] is False
            and summ["holdout_start"] == HOLDOUT
        )
        assert (
            man["sealed_end"] == "2020-02-28" and man["unsealed"] is False and man["holdout_start"] == HOLDOUT
        )
        assert pd.Timestamp(summ["backtest"]["end"]) < pd.Timestamp(HOLDOUT)
        full = full_history_market_hash(StoreSpec(store=str(store), max_names=50))
        fam = research_family(1, full)
        rec = TrialLedger(str(led_path)).raw_records()[-1]
        assert rec["family"] == fam == summ["ledger_family"]
        assert (
            rec["market_hash_full"] == full
            and rec["market_hash"] != full
            and rec["sealed_end"] == "2020-02-28"
        )
        assert summ["market_hash"] == rec["market_hash"] and rec.get("reference") is None
        led = TrialLedger(str(led_path))
        assert led.n_trials(fam) == 1
        # 2) the same configuration again under the seal: a reference, not a new trial
        assert main(_run_argv(store, tmp_path, "--nav", "2e6")) == 0
        out = capsys.readouterr().out
        assert "logged as a reference" in out
        summ, _ = _last_summary(tmp_path)
        assert summ["reference"] is True
        assert TrialLedger(str(led_path)).n_trials(fam) == 1
        # a different configuration under the seal is a trial of the same family
        assert main(_run_argv(store, tmp_path, "--set", "portfolio.trade_aversion=2")) == 0
        capsys.readouterr()
        assert TrialLedger(str(led_path)).n_trials(fam) == 2
        # 3) unseal: needs a reason; logged before the run; full history; banner
        assert main(_run_argv(store, tmp_path, "--unseal", " ")) == 2
        assert "needs a reason" in capsys.readouterr().err
        assert main(_run_argv(store, tmp_path, "--unseal", "final check")) == 0
        out = capsys.readouterr().out
        assert "UNSEALED RUN" in out and "UNSEALED: full history" in out
        summ, man = _last_summary(tmp_path)
        assert (
            summ["unsealed"] is True and summ["sealed_end"] is None and summ["holdout_already_used"] is False
        )
        assert man["unsealed"] is True and pd.Timestamp(summ["backtest"]["end"]) > pd.Timestamp(HOLDOUT)
        raw = TrialLedger(str(led_path)).raw_records()
        ev = [r for r in raw if r.get("kind") == "unseal"]
        assert len(ev) == 1 and ev[0]["unseal_reason"] == "final check" and ev[0]["family"] == fam
        assert raw.index(ev[0]) == len(raw) - 2  # the event precedes the run's record
        assert raw[-1]["unsealed"] is True and raw[-1]["family"] == fam
        # 4) a second unseal of the same configuration is refused ...
        n = len(raw)
        assert main(_run_argv(store, tmp_path, "--unseal", "again")) == 2
        assert "already unsealed" in capsys.readouterr().err
        assert len(TrialLedger(str(led_path)).raw_records()) == n
        # ... and flagged when forced
        assert main(_run_argv(store, tmp_path, "--unseal", "again", "--force")) == 0
        out = capsys.readouterr().out
        assert "HOLDOUT ALREADY USED" in out
        summ, _ = _last_summary(tmp_path)
        assert summ["holdout_already_used"] is True and "holdout already used" in " ".join(summ["notes"])
        raw = TrialLedger(str(led_path)).raw_records()
        assert raw[-1]["holdout_already_used"] is True and raw[-2]["forced"] is True
        # 5) status
        st = S.seal_status(str(led_path))
        assert st.runs_under_seal == 3 and len(st.unseals) == 2
        assert st.by_family[fam] == {"reference": 1, "trial": 2}
        assert main(["seal", "status", "--ledger", str(led_path), "--json", str(tmp_path / "st.json")]) == 0
        out = capsys.readouterr().out
        assert "research runs under the seal: 3" in out and "unseals: 2" in out and "FORCED" in out
        assert json.loads((tmp_path / "st.json").read_text())["runs_under_seal"] == 3


def test_demo_and_diagnostics_loader(store, tmp_path, sealed_env):
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from diagnose_signal import load_market

    a = argparse.Namespace(
        store=str(store), synthetic=False, start=None, end=None, max_names=50, min_mcap_ever=None,
        runs_root=str(tmp_path / "runs"),
    )  # fmt: skip
    md = load_market(a)
    assert md.dates[-1] < pd.Timestamp(HOLDOUT)
    # the synthetic demo is exempt
    from qts.cli import main

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rc = main(
            ["demo", "--stocks", "40", "--days", "600", "--max-days", "3", "--no-save", "--ledger", "",
             "--set", "universe.top_n_by_mcap=30"]
        )  # fmt: skip
    assert rc == 0


def test_family_note_on_a_different_full_history_family(tmp_path):
    seal = S.load_seal(_write_seal(tmp_path, inherit_family=["pipeline_phase2:aaaaaaaaaaaa"]))
    assert S.family_note(seal, "pipeline_phase2:aaaaaaaaaaaa") is None
    assert S.family_note(seal, "pipeline_phase1:bbbbbbbbbbbb") is None  # other phase: nothing inherited
    note = S.family_note(seal, "pipeline_phase2:bbbbbbbbbbbb")
    assert note and "merge-families" in note
