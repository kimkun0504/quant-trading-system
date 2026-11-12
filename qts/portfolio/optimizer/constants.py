"""Optimizer constants (tolerances, mode names, gamma-search settings)."""

from __future__ import annotations

from collections.abc import Callable

import cvxpy as cp
import pandas as pd

OK_STATUSES = ("optimal", "optimal_inaccurate")
#: solver statuses that end the solver chain (a problem property, not a solver failure)
TERMINAL_STATUSES = ("infeasible", "unbounded")
#: user constraint hook: ``fn(w, x, tickers) -> [constraints]``
ExtraConstraint = Callable[[cp.Variable, cp.Expression, pd.Index], list[cp.Constraint]]
SOLVER_TOL = 1e-6
MAX_SCALE_UP = 5.0  # sanity cap on the vol-target projection anchor s = target / predicted
_BIG = 10.0  # "no cap" value for the per-name no-trade / forced-exit cap parameters
VOL_TARGET_MODES = ("projection", "constraint")
RISK_AVERSION_MODES = ("target_vol", "fixed")
TRADING_DAYS = 252  # gamma is an annual risk aversion; vols are annualised with sqrt(252)
# --- gamma search (risk_aversion_mode="target_vol")
GAMMA_BRACKET = (0.05, 200.0)  # search range for the annual risk aversion
GAMMA_MAX_SOLVES = 8  # solves per date, the closed-form start included
GAMMA_SEARCH_VOL_TOL = 0.02  # stop when |vol / target - 1| < 2 %
GAMMA_NUDGE = 1.02  # gamma multiplier for one retry after a numerical solver failure
CAPS_ELASTICITY = 0.1  # |d log vol / d log gamma| below this (vol under target) ...
POSITION_CAPS_SHARE = 0.1  # ... with per-name / ADV caps binding on >= 10 % of held names = caps bind
GROSS_CAP_CONSTRAINTS = ("gross", "long_leg", "short_leg")  # leverage limits that cap the vol
# --- gamma cap search (vol_target_is_cap): upward only from the base risk aversion
GAMMA_CAP_FIRST_STEP = 4.0  # gamma multiplier per step while the leverage cap still binds
GAMMA_CAP_GROSS_TOL = 0.03  # stop when the un-capped gross is within 3 % of the leverage limit
GAMMA_CAP_MIN_RATIO = 1.05  # ... or the gamma bracket is narrower than 5 %
GAMMA_CAP_ELASTICITY = -0.7  # prior d log gross / d log gamma before two admissible books exist
