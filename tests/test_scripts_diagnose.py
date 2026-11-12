"""Smoke tests of the diagnostic scripts on the synthetic market."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    return importlib.import_module(name)


@pytest.mark.slow
def test_diagnose_constraints_synthetic(capsys):
    mod = _load("diagnose_constraints")
    rc = mod.main(["--synthetic", "--n-dates", "3"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "solve from a FLAT book" in out and "PREVIOUS sampled solution" in out
    assert "relaxation solves" in out and "pos x2" in out and "part x2" in out
    assert "vol_target_binding" in out


def test_diagnose_constraints_relaxed_optimizer_scales_caps():
    mod = _load("diagnose_constraints")
    from qts.core.config import SystemConfig
    from qts.pipeline import ResearchPipeline

    cfg = SystemConfig()
    pipe = ResearchPipeline(cfg, runs_root=None)
    # only the pieces relaxed_optimizer touches are needed: no fit
    from qts.tcost import TransactionCostModel

    pipe.cost = TransactionCostModel(cfg.cost, cfg.execution)
    from qts.portfolio import SinglePeriodOptimizer

    pipe.optimizer = SinglePeriodOptimizer(cfg.portfolio, pipe.cost, horizon_days=cfg.alpha.horizon_days)
    rep = object.__new__(mod.SolveReplayer)
    rep.pipe, rep.cfg, rep.optimizer = pipe, cfg, pipe.optimizer
    opt = rep.relaxed_optimizer(mod.RELAXATIONS["pos+part x2"])
    assert opt.cfg.max_position_weight == pytest.approx(2 * cfg.portfolio.max_position_weight)
    assert opt.cfg.max_adv_participation == pytest.approx(2 * cfg.portfolio.max_adv_participation)
    assert opt.cfg.industry_band == cfg.portfolio.industry_band
    assert opt.horizon_days == pipe.optimizer.horizon_days


def test_find_outlier_returns_synthetic(capsys):
    mod = _load("find_outlier_returns")
    rc = mod.main(["--synthetic", "--top", "5", "--windows", "2018-03-01..2018-03-07"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "top 5 |daily return|" in out and "cause" in out
    assert "rule_a_caps" in out and "rule_a_price_floor" in out
    assert "cross-sectional sqrt-cap-weighted mean |r| spikes" in out
    assert "2018-03-01..2018-03-07" in out


def test_find_outlier_returns_tags_causes():
    mod = _load("find_outlier_returns")
    assert "sub-penny" in mod._likely_cause(5.0, 0.5, 3.0, 0.5, 3.0, 1, False)
    assert "restatement" in mod._likely_cause(9.0, 20.0, 20.5, 2.0, 20.0, 1, False)
    assert "split-like" in mod._likely_cause(-0.5, 100.0, 50.0, 100.0, 50.0, 1, False)
    assert "delisting-day" in mod._likely_cause(-0.97, 10.0, 0.3, 10.0, 0.3, 1, True)
    assert mod._likely_cause(0.4, 10.0, 14.0, 10.0, 14.0, 4, False) == "gap(4d)"
    assert mod._likely_cause(0.4, 10.0, 14.0, 10.0, 14.0, 1, False) == "unknown"


def test_attribute_run_synthetic(capsys):
    """Builds a 30-day run in a temp runs root and prints all five tables."""
    mod = _load("attribute_run")
    rc = mod.main(["--synthetic", "--max-days", "30", "--risk-dates", "4"])
    out = capsys.readouterr().out
    assert rc == 0
    for title in (
        "1. SIGNAL -> IDEAL PORTFOLIOS",
        "2. REALISED BOOK ATTRIBUTION",
        "3. RISK-FORECAST GAP",
        "4. TURNOVER ANATOMY",
        "5. COST REALITY CHECK",
    ):
        assert title in out
    for label in mod.IDEAL_LABELS.values():
        assert label in out
    assert "REALISED BOOK gross" in out and "specific (total - factors)" in out
    # the accounting identities the tables rest on, on a market without halts / delistings
    assert "recorded gross return: corr 1.0000" in out
    assert "delistings by reason and side (from the run's delistings.parquet)" in out
    assert "gap +0.00 %/yr = delistings +0.00 %/yr + remainder +0.00 %/yr" in out
    assert "recomputed ex-ante vol / recorded = 1.000" in out
    assert "re-priced / recorded = 1.000" in out
    assert "WARNING" not in out


def test_attribute_run_splits_the_reconciliation_gap_by_delisting_reason():
    """``delisting_events`` / ``delisting_split``: from the run's own records, and
    reconstructed from the book for a run written before ``delistings.parquet``."""
    from types import SimpleNamespace

    import numpy as np
    import pandas as pd

    from qts.core.config import SystemConfig

    mod = _load("attribute_run")
    tickers = pd.Index(["ACQ", "BKR", "SHT", "HLT", "LIV"])
    dates = pd.bdate_range("2024-01-01", periods=6)
    md = SimpleNamespace(
        tickers=tickers,
        delisting_reasons=pd.Series({"ACQ": "acquisition", "BKR": "performance", "SHT": "performance"}),
    )
    # ACQ, BKR, SHT have no close from day 3; HLT is halted on day 3 only
    has_close = np.ones((6, 5), dtype=bool)
    has_close[3:, :3] = False
    has_close[3, 3] = False
    W = np.zeros((5, 5))
    W[:3] = [0.02, 0.01, -0.02, 0.01, 0.05]  # held through decision row 2
    W[3:] = [0.0, 0.0, 0.0, 0.01, 0.05]  # liquidated names are 0; the halted one keeps its weight
    mk = SimpleNamespace(cfg=SystemConfig(), md=md, tickers=tickers, has_close=has_close, pos=np.arange(5))
    run = SimpleNamespace(W=W, dec=dates[:5], ret_dates=dates[1:], delistings=None, old_haircut=-0.30)
    ev, note = mod.delisting_events(mk, run)
    assert "RECONSTRUCTED" in note
    got = ev.set_index("ticker")
    assert sorted(got.index) == ["ACQ", "BKR", "SHT"] and (ev["date"] == dates[3]).all()
    assert got["reason"].to_dict() == {"ACQ": "acquisition", "BKR": "performance", "SHT": "performance"}
    # booked by the old run: -30 % against the position on both sides
    np.testing.assert_allclose(got.loc[["ACQ", "BKR", "SHT"], "pnl"], [-0.006, -0.003, -0.006])
    # the same events under the current rule: acquisition 0, long loses 30 %, short gains 30 %
    np.testing.assert_allclose(
        got.loc[["ACQ", "BKR", "SHT"], "pnl_default"], [0.0, -0.003, 0.006], atol=1e-15
    )
    split = mod.delisting_split(ev, n_days=5)
    k = 100 * 252 / 5
    assert list(split.index) == ["acquisition long", "performance long", "performance short", "total"]
    assert split.loc["acquisition long", "n"] == 1 and split.loc["total", "n"] == 3
    assert split.loc["total", "booked %/yr"] == pytest.approx(k * -0.015)
    assert split.loc["total", "current rule %/yr"] == pytest.approx(k * 0.003)
    assert split.loc["performance short", "current rule %/yr"] == pytest.approx(k * 0.006)
    # a run that recorded its delistings: the simulator's numbers are used as they are
    rec = pd.DataFrame(
        {
            "date": [dates[3]],
            "ticker": ["SHT"],
            "weight": [-0.02],
            "reason": ["performance"],
            "side": ["short"],
            "stock_return": [-0.30],
            "haircut_return": [-0.30],
            "pnl": [0.006],
        }
    )
    ev2, note2 = mod.delisting_events(mk, SimpleNamespace(**{**run.__dict__, "delistings": rec}))
    assert "delistings.parquet" in note2 and "RECONSTRUCTED" not in note2
    assert ev2["pnl"].tolist() == [0.006] and ev2["pnl_default"].tolist() == pytest.approx([0.006])
    empty, _ = mod.delisting_events(mk, SimpleNamespace(**{**run.__dict__, "delistings": rec.iloc[:0]}))
    assert mod.delisting_split(empty, 5).loc["total", "booked %/yr"] == 0.0


def test_attribute_run_helpers():
    import numpy as np

    mod = _load("attribute_run")
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(50), rng.normal(size=(50, 3)), np.zeros(50)])  # with a dead column
    res = mod._residual(X, rng.normal(size=(50, 2)))
    assert np.abs(X.T @ res).max() < 1e-9
    assert np.abs(mod._normalise(res[:, 0])).sum() == pytest.approx(2.0)
    # ex-ante split is additive: pieces sum to w' (B F B' + D + nu y y') w
    B = rng.normal(size=(50, 4))
    A = rng.normal(size=(4, 4))
    F = A @ A.T
    D = rng.uniform(0.5, 1.5, size=50)
    y = rng.normal(size=50)
    w = rng.normal(size=50)
    parts = mod._exante_split(w, B, F, D, y, 0.3, {"market": [0], "style": [1, 2, 3], "statistical": []})
    total = w @ (B @ F @ B.T + np.diag(D) + 0.3 * np.outer(y, y)) @ w
    assert sum(parts.values()) == pytest.approx(total)
    assert set(parts) == {"factor:market", "factor:style", "specific", "alignment"}


def test_diagnostic_loader_applies_set_overrides():
    import argparse

    mod = _load("diagnose_signal")
    from qts.cli import ConfigOverrideError

    ap = argparse.ArgumentParser()
    mod.add_data_args(ap)
    a = ap.parse_args(["--synthetic", "--set", "alpha.horizon_days=60", "--set", "portfolio.beta_band=0.1"])
    cfg = mod.load_config(a)
    assert cfg.alpha.horizon_days == 60 and cfg.portfolio.beta_band == pytest.approx(0.1)
    # same result (hence the same stage-cache key) as the CLI's own override path
    from qts.cli import _apply_set_overrides, _load_config

    assert cfg.hash() == _apply_set_overrides(_load_config(None), a.set_overrides).hash()
    with pytest.raises(ConfigOverrideError):
        mod.load_config(ap.parse_args(["--synthetic", "--set", "alpha.no_such_key=1"]))
