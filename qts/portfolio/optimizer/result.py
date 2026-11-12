"""The optimizer's output for one date."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from qts.portfolio.optimizer.constants import OK_STATUSES


@dataclass(frozen=True)
class OptimizationResult:
    w_target: pd.Series
    x: pd.Series
    expected_alpha: float  # alpha' w_target over the horizon (after any GP shrink)
    expected_risk: float  # annualised ex-ante vol of w_target
    expected_cost: float  # trade cost + H * holding cost (fraction of NAV); nan on failure
    turnover: float  # sum |x| (two-sided, fraction of NAV)
    transfer_coefficient: float
    status: str
    solve_time: float
    binding_constraints: dict[str, int] = field(default_factory=dict)
    objective_value: float = float("nan")
    vol_scalar: float = 1.0  # s used in the vol-target projection (1 = none / constraint mode)
    solver: str = ""
    notes: list[str] = field(default_factory=list)
    exante_var_daily: float = float("nan")
    n_out_of_model: int = 0  # names outside the risk model with w_target != 0 (forced exits in progress)
    alignment_share: float = float("nan")  # share of ex-ante variance from the alignment term
    gp_shrink: float = 1.0  # Garleanu-Pedersen factor applied to alpha (1 = none)
    problem_reused: bool = False  # True when the cached DPP problem was re-solved
    risk_aversion: float = float("nan")  # annual gamma of the returned book (searched or fixed)
    # risk_aversion_mode="target_vol": "target" (vol within 2 % of target), "caps" (gross /
    # position / ADV caps bind below target; capped book accepted), "bracket" (gamma at a
    # GAMMA_BRACKET end), "max_solves", "no_alpha", "solve_failed:<status>"; "" otherwise.
    # With vol_target_is_cap: "below_target" (the base-gamma book is under the cap, gamma
    # untouched) or "target" (gamma raised until the vol came down to the cap)
    vol_target_binding: str = ""
    n_solves: int = 0  # convex solves for this date (search steps / projection included)
    n_excluded_alpha: int = 0  # names with alpha outside the risk model (excluded, w = 0)
    # min-trade projection (PortfolioConfig.min_trade_weight): trades dropped after the
    # solve, and the largest constraint violation of the returned book (0 = none) with
    # the name of that constraint
    n_small_trades_dropped: int = 0
    max_constraint_violation: float = 0.0
    violated_constraint: str = ""
    # solver exceptions caught while solving this date ("<SOLVER>: <ExcType>: <message>");
    # only cvxpy SolverError / numerical errors are caught, anything else propagates
    solver_errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status in OK_STATUSES
