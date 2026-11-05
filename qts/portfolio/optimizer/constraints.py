"""The constraint menu of the alpha / projection problems, and binding counts.

:func:`build_constraints` returns named constraints over the problem's
variables (``w`` post-trade weights, ``x`` trades, ``f = B'w`` factor
exposures) and parameters ``P`` (see :class:`~qts.portfolio.optimizer.compiled._Compiled`):

* definitional (names start with ``_``, never reported as binding):
  ``_link`` ``x = w - w0``; ``_f_static`` / ``_f_dynamic`` ``f = B'w`` (the
  market / industry block as a constant sparse matrix, styles as a parameter);
* book: ``long_only`` + ``fully_invested``; ``long_leg`` / ``short_leg`` /
  ``net_hi`` / ``net_lo`` (130/30); ``gross`` / ``net_hi`` / ``net_lo`` (long/short);
* ``max_position`` ``|w| <= cfg.max_position_weight``;
* ``beta_neutral`` ``|f_beta| <= beta_band``;
* ``industry_band`` / ``style_band`` ``|f_k - f_k(benchmark)| <= band``;
* ``no_trade`` ``|x| <= nt`` and ``force_to_zero`` ``|w| <= fz`` (per-name caps;
  ``_BIG`` = no cap), present only when some name uses them.

The ADV participation cap ``|x| <= part`` is returned separately because it
can be dropped (``cfg.relax_participation_on_infeasible``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from qts.core.config import PortfolioConfig
from qts.portfolio.optimizer.constants import SOLVER_TOL
from qts.portfolio.optimizer.problem import _ProblemData


def build_constraints(
    w: cp.Variable,
    x: cp.Variable,
    f: cp.Variable,
    P: Mapping[str, cp.Parameter],
    layout: Mapping[str, Any],
    book: str,
    cfg: PortfolioConfig,
    beta_band: float,
) -> tuple[dict[str, cp.Constraint], cp.Constraint]:
    """``(named constraints, participation constraint)`` (module docstring)."""
    Ks, Kd = len(layout["static"]), len(layout["dynamic"])
    cons: dict[str, cp.Constraint] = {"_link": x == w - P["w0"]}
    if Ks:
        cons["_f_static"] = f[:Ks] == sp.csc_matrix(layout["static_block"]).T @ w
    if Kd:
        cons["_f_dynamic"] = f[Ks:] == P["Bd"].T @ w
    cons.update(_book_constraints(w, book, cfg))
    cons["max_position"] = cp.abs(w) <= cfg.max_position_weight
    pos = {c: i for i, c in enumerate(layout["static"] + layout["dynamic"])}
    if layout["beta"] is not None:
        cons["beta_neutral"] = cp.abs(f[pos[layout["beta"]]]) <= beta_band
    if layout["ind"]:
        cons["industry_band"] = cp.abs(f[[pos[c] for c in layout["ind"]]] - P["oi"]) <= cfg.industry_band
    if layout["sty"]:
        cons["style_band"] = cp.abs(f[[pos[c] for c in layout["sty"]]] - P["os"]) <= cfg.style_band_sd
    if layout.get("has_no_trade", True):
        cons["no_trade"] = cp.abs(x) <= P["nt"]
    if layout.get("has_force", True):
        cons["force_to_zero"] = cp.abs(w) <= P["fz"]
    return cons, cp.abs(x) <= P["part"]


def _book_constraints(w: cp.Variable, book: str, cfg: PortfolioConfig) -> dict[str, cp.Constraint]:
    # cvxpy's stubs type comparisons of expressions as bool; they are Constraints
    if book == "long_only":
        return cast(dict[str, cp.Constraint], {"long_only": w >= 0, "fully_invested": cp.sum(w) == 1})
    if book == "130_30":
        return cast(
            dict[str, cp.Constraint],
            {
                "long_leg": cp.sum(cp.pos(w)) <= 1.3,
                "short_leg": cp.sum(cp.neg(w)) <= 0.3,
                "net_hi": cp.sum(w) <= 1.0 + cfg.net_exposure_band,
                "net_lo": cp.sum(w) >= 1.0 - cfg.net_exposure_band,
            },
        )
    return cast(
        dict[str, cp.Constraint],
        {  # long_short
            "gross": cp.norm1(w) <= cfg.gross_leverage_max,
            "net_hi": cp.sum(w) <= cfg.net_exposure_target + cfg.net_exposure_band,
            "net_lo": cp.sum(w) >= cfg.net_exposure_target - cfg.net_exposure_band,
        },
    )


def n_binding(c: cp.Constraint, mask: np.ndarray | None = None) -> int:
    """Entries of ``c`` with a dual above ``SOLVER_TOL`` (restricted to ``mask``)."""
    dv = c.dual_value
    if dv is None:
        return 0
    dv = np.atleast_1d(np.asarray(dv, dtype=float))
    if mask is not None and dv.shape == mask.shape:
        dv = dv[mask]
    return int((np.abs(dv) > SOLVER_TOL).sum())


def binding_counts(
    cons: Mapping[str, cp.Constraint],
    participation: cp.Constraint,
    pdat: _ProblemData,
    with_part: bool,
) -> dict[str, int]:
    """Binding-entry count per named constraint after a solve (definitional
    ones skipped; per-name caps counted only on the names they apply to)."""
    binding: dict[str, int] = {}
    for name, c in cons.items():
        if name.startswith("_"):  # internal definitional equalities (x, f)
            continue
        mask = None
        if name == "no_trade":
            if not pdat.no_trade_mask.any():
                continue
            mask = pdat.no_trade_mask
        elif name == "force_to_zero":
            if not pdat.force_mask.any():
                continue
            mask = pdat.force_mask
        cnt = n_binding(c, mask)
        if cnt:
            binding[name] = cnt
    if with_part:
        cnt = n_binding(participation)
        if cnt:
            binding["participation"] = cnt
    return binding
