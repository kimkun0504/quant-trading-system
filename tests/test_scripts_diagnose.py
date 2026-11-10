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
