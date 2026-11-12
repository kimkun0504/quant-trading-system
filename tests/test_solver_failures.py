"""Optimizer failure handling: narrow solver catch, failed-solve diagnostics and
the ``failed_solves`` hurdle gate.

* A cvxpy ``SolverError`` (or a numerical error) on every solver in the chain is
  recorded in ``OptimizationResult.solver_errors`` and the current book is held.
* Any other exception (a programming error) propagates: it must not become a
  silent "hold the book".
* The simulator lists failed solves in ``diagnostics["failed_solves"]`` and
  counts rebalance solves in ``diagnostics["n_rebalance_solves"]``.
* ``HurdleReport`` FAILs ``failed_solves`` when the failed share exceeds
  ``ValidationConfig.max_failed_solve_frac`` (default 2 %).
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from qts.core.config import PortfolioConfig, ValidationConfig
from qts.portfolio import SimpleCostModel, SinglePeriodOptimizer, snapshot_from_market
from qts.portfolio.optimizer import SOLVER_FAILURES, solve_with_chain, solver_chain
from qts.validation.hurdles import HurdleReport
from tests.test_simulator_handcomputed import (
    DATES,
    NAV0,
    TARGETS,
    TICKERS,
    ConstantBpsCost,
    PresetOptimizer,
    alpha_fn,
    make_md,
    risk_fn,
)

PASSING = {"decile_tstat": 4.0, "breakeven_cost_bps": 50.0, "modeled_cost_bps": 10.0, "dsr": 0.98}


# ----------------------------------------------------------------------------- hurdle gate
@pytest.mark.parametrize(
    ("frac", "status"),
    [
        (0.0, "PASS"),
        (0.02, "PASS"),
        (0.0201, "FAIL"),
        (0.5, "FAIL"),
        (None, "skipped"),
        (float("nan"), "skipped"),
    ],
)
def test_failed_solves_gate_threshold(frac, status):
    r = HurdleReport(ValidationConfig()).evaluate_signal(
        **PASSING, phase=1, net_sharpe=0.8, failed_solve_frac=frac
    )
    g = r.gate("failed_solves")
    assert g.status == status
    assert g.threshold == pytest.approx(0.02)
    # the other gates pass: overall follows the solver gate (skipped does not fail it)
    assert r.overall is (status != "FAIL"), r.reasons


def test_failed_solves_gate_threshold_is_configurable_and_round_trips():
    hr = HurdleReport(ValidationConfig(max_failed_solve_frac=0.1))
    r = hr.evaluate_signal(**PASSING, phase=1, net_sharpe=0.8, failed_solve_frac=0.05)
    assert r["status"]["failed_solves"] == "PASS"
    again = type(r).from_dict(r.to_dict())
    assert again.status == r.status and again.gate("failed_solves").value == pytest.approx(0.05)
    # the gate can be made required: then a missing input fails overall
    r2 = hr.evaluate_signal(**PASSING, phase=1, net_sharpe=0.8, required=["failed_solves"])
    assert r2["status"]["failed_solves"] == "FAIL" and not r2.overall


# ----------------------------------------------------------------------------- optimizer
@pytest.fixture(scope="module")
def octx(md, risk_model, alpha_model, test_date):
    return {
        "risk": risk_model.as_of(test_date),
        "alpha": alpha_model.as_of(test_date).expected_return,
        "snap": snapshot_from_market(md, test_date),
        "w0": pd.Series(0.01, index=md.tickers),
    }


def _fixed_opt() -> SinglePeriodOptimizer:
    cfg = PortfolioConfig(
        book="long_short", risk_aversion=2.0, target_vol_annual=None, risk_aversion_mode="fixed"
    )
    return SinglePeriodOptimizer(cfg, SimpleCostModel(zero=True))


def test_solver_error_is_recorded_and_book_held(octx, monkeypatch):
    def boom(self, *a, **k):
        raise cp.error.SolverError("synthetic solver crash")

    monkeypatch.setattr(cp.Problem, "solve", boom)
    opt = _fixed_opt()
    res = opt.solve(octx["alpha"], octx["risk"], octx["w0"], 1e6, octx["snap"])
    assert not res.ok and res.status == "error:SolverError"
    chain = opt._solver_chain()
    assert len(res.solver_errors) == len(chain) and all(
        "synthetic solver crash" in e for e in res.solver_errors
    )
    pd.testing.assert_series_equal(res.w_target, octx["w0"].reindex(res.w_target.index), check_names=False)
    assert res.turnover == 0.0


def test_numerical_error_falls_through_to_next_solver(octx, monkeypatch):
    real = cp.Problem.solve
    calls: list[str] = []

    def first_fails(self, *a, solver=None, **k):
        calls.append(solver)
        if len(calls) == 1:
            raise ZeroDivisionError("pivot")
        return real(self, *a, solver=solver, **k)

    monkeypatch.setattr(cp.Problem, "solve", first_fails)
    opt = _fixed_opt()
    if len(opt._solver_chain()) < 2:
        pytest.skip("needs two installed solvers")
    res = opt.solve(octx["alpha"], octx["risk"], octx["w0"], 1e6, octx["snap"])
    assert res.ok and res.solver == calls[1]
    assert res.solver_errors == [f"{calls[0]}: ZeroDivisionError: pivot"]


def test_programming_error_propagates(octx, monkeypatch):
    def bug(self, *a, **k):
        raise TypeError("not a solver failure")

    monkeypatch.setattr(cp.Problem, "solve", bug)
    with pytest.raises(TypeError, match="not a solver failure"):
        _fixed_opt().solve(octx["alpha"], octx["risk"], octx["w0"], 1e6, octx["snap"])


def test_solver_failures_tuple_is_narrow():
    assert cp.error.SolverError in SOLVER_FAILURES and ArithmeticError in SOLVER_FAILURES
    assert Exception not in SOLVER_FAILURES and TypeError not in SOLVER_FAILURES
    assert solver_chain("clarabel", ["CLARABEL", "SCS"]) == ["CLARABEL", "SCS"]
    assert solver_chain("MOSEK", ["SCS"]) == ["SCS"]


def test_solve_with_chain_stops_at_terminal_status():
    x = cp.Variable()
    prob = cp.Problem(cp.Minimize(x), [x >= 1, x <= 0])  # infeasible
    att = solve_with_chain(prob, ["CLARABEL", "SCS"], {}, primary_only=False)
    assert att.status == "infeasible" and att.solver == "CLARABEL" and att.errors == ()


# ----------------------------------------------------------------------------- simulator
class FlakyOptimizer(PresetOptimizer):
    """Preset targets, but the solve on ``fail_dates`` reports a failure
    (``w_target`` = current book, as the real optimizer does)."""

    def __init__(self, targets: dict, fail_dates: set):
        super().__init__(targets)
        self.fail_dates = fail_dates

    def solve(self, **kw):
        res = super().solve(**kw)
        if kw["risk"].date in self.fail_dates:
            res.status = "error:SolverError"
            res.w_target = kw["w_current"].astype(float)
            res.solver_errors = ["CLARABEL: SolverError: synthetic"]
        return res


def test_simulator_records_failed_solves_and_gate_fails():
    from qts.backtest import DailyBarSimulator
    from qts.core.config import SystemConfig

    cfg = SystemConfig().with_overrides({"execution.signal_lag_days": 0, "portfolio.rebalance_every_days": 1})
    opt = FlakyOptimizer(TARGETS, fail_dates={DATES[1]})
    res = DailyBarSimulator(cfg, ConstantBpsCost()).run(make_md(), DATES, alpha_fn, risk_fn, opt, nav0=NAV0)
    d = res.diagnostics
    assert d["n_rebalance_solves"] == 4 and d["n_failed_solves"] == 1
    assert d["failed_solves"] == [
        {
            "date": DATES[1],
            "status": "error:SolverError",
            "solver_errors": ["CLARABEL: SolverError: synthetic"],
        }
    ]
    assert res.failed_solve_fraction() == pytest.approx(0.25)
    # the failed day held its (drifted) book: no trade on DATES[1]
    assert np.allclose(res.trades.loc[DATES[1], list(TICKERS)].to_numpy(), 0.0)
    gate = HurdleReport(ValidationConfig()).evaluate_signal(failed_solve_frac=res.failed_solve_fraction())
    assert gate["status"]["failed_solves"] == "FAIL"


def test_failed_solve_fraction_without_solves_is_none():
    from qts.backtest import BacktestResult

    empty = pd.DataFrame()
    res = BacktestResult(nav=pd.Series(dtype=float), returns=empty, weights=empty, trades=empty, costs=empty)
    assert res.failed_solve_fraction() is None
    res.diagnostics = {"solve_log": [{}, {}], "n_failed_solves": 1}  # older results: count the log
    assert res.failed_solve_fraction() == pytest.approx(0.5)
