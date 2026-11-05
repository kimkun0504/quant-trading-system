"""Solver chain with narrow failure handling.

:func:`solve_with_chain` tries the configured solver, then the fallbacks
(Clarabel, SCS, OSQP; only installed ones).  It catches only what a solver
legitimately raises -- :class:`cvxpy.error.SolverError` (solver crashed, not
applicable to the problem class, or reported an error status) and numerical
errors (``ArithmeticError``, ``numpy.linalg.LinAlgError``) -- records each one
as ``"<SOLVER>: <ExcType>: <message>"`` and moves on.  Any other exception is
a programming error (a malformed problem, a bad cost expression) and
propagates instead of turning into a silent "hold the book".
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from typing import Any, NamedTuple

import cvxpy as cp
import numpy as np
from cvxpy.error import SolverError

from qts.portfolio.optimizer.constants import OK_STATUSES, TERMINAL_STATUSES

log = logging.getLogger("qts.portfolio.optimizer")

#: exceptions treated as "this solver failed on this problem; try the next one"
SOLVER_FAILURES: tuple[type[BaseException], ...] = (
    SolverError,
    ArithmeticError,
    np.linalg.LinAlgError,
)
FALLBACK_SOLVERS = ("CLARABEL", "SCS", "OSQP")


class SolveAttempt(NamedTuple):
    """Outcome of one pass through the solver chain."""

    status: str  # solver status, or "error:<ExcType>" when every solver raised
    solver: str  # solver that produced ``status`` ("" when none did)
    errors: tuple[str, ...]  # exceptions caught on the way


def solver_chain(primary: str | None, installed: Sequence[str]) -> list[str]:
    """``[primary, CLARABEL, SCS, OSQP]`` restricted to installed solvers, de-duplicated."""
    have = {s.upper() for s in installed}
    out: list[str] = []
    for s in [primary, *FALLBACK_SOLVERS]:
        if s and s.upper() not in out and s.upper() in have:
            out.append(s.upper())
    return out


def solve_with_chain(
    prob: cp.Problem,
    chain: Sequence[str],
    solver_opts: dict[str, Any],
    verbose: bool = False,
    dpp: bool = False,
    primary_only: bool = False,
) -> SolveAttempt:
    """Solve with ``chain`` (``primary_only``: its first solver only) and stop at
    the first ok or terminal (infeasible / unbounded) status.  ``dpp=False``
    canonicalises with the parameters' current values as constants
    (``ignore_dpp``): for a one-off solve that is far cheaper than a DPP compile."""
    last = "failed"
    errors: list[str] = []
    kw = dict(solver_opts)
    if not dpp:
        kw["ignore_dpp"] = True
    for s in chain[:1] if primary_only else chain:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                prob.solve(solver=s, verbose=verbose, **kw)
        except SOLVER_FAILURES as e:
            log.debug("solver %s failed: %s", s, e)
            errors.append(f"{s}: {type(e).__name__}: {e}"[:300])
            last = f"error:{type(e).__name__}"
            continue
        last = prob.status
        if prob.status in OK_STATUSES or prob.status in TERMINAL_STATUSES:
            return SolveAttempt(prob.status, s, tuple(errors))
    return SolveAttempt(last, "", tuple(errors))
