"""Smoke tests of the diagnostic scripts on the synthetic market (slow tier)."""

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
