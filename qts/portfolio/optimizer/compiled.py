"""One parametrised (DPP) cvxpy problem and its structure key.

A :class:`_Compiled` problem is built once per structure (ticker set,
constraint layout, cost-expression structure, config) and re-solved with new
parameter values while :func:`structure_key` is unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd

from qts.core.config import PortfolioConfig
from qts.portfolio.costs import CostTerms
from qts.portfolio.optimizer.constants import _BIG, ExtraConstraint
from qts.portfolio.optimizer.constraints import binding_counts, build_constraints
from qts.portfolio.optimizer.dpp import _leaf_sig, _Parametrizer
from qts.portfolio.optimizer.problem import _ProblemData


def structure_key(
    cfg: PortfolioConfig,
    beta_band: float,
    kind: str,
    pdat: _ProblemData,
    H: int,
    with_part: bool,
    vol_cap: bool,
    venue: str,
) -> tuple:
    """Everything that changes the *structure* of the compiled problem (a
    changed key forces a rebuild; values that only change parameters are not in it)."""
    cfg_sig = (
        cfg.book,
        cfg.gross_leverage_max,
        cfg.net_exposure_target,
        cfg.net_exposure_band,
        cfg.max_position_weight,
        cfg.industry_band,
        cfg.style_band_sd,
        cfg.trade_aversion,
        cfg.hold_aversion,
        cfg.turnover_penalty,
        beta_band,
        getattr(cfg, "vol_target_projection_cost_weight", 0.0),
    )
    static = np.ascontiguousarray(pdat.static_block())
    return (
        kind,
        with_part,
        vol_cap,
        H,
        venue,
        len(pdat.tickers),
        hash(tuple(pdat.tickers)),
        tuple(pdat.static_cols),
        tuple(pdat.dynamic_cols),
        hash(static.tobytes()),
        pdat.beta_col,
        tuple(pdat.ind_cols),
        tuple(pdat.sty_cols),
        bool(pdat.no_trade_mask.any()),
        bool(pdat.force_mask.any()),
        cfg_sig,
    )


def problem_layout(pdat: _ProblemData) -> dict[str, Any]:
    """Factor layout and optional constraint blocks of the problem for ``pdat``."""
    return {
        "static": list(pdat.static_cols),
        "dynamic": list(pdat.dynamic_cols),
        "static_block": pdat.static_block(),
        "beta": pdat.beta_col,
        "ind": list(pdat.ind_cols),
        "sty": list(pdat.sty_cols),
        "has_no_trade": bool(pdat.no_trade_mask.any()),
        "has_force": bool(pdat.force_mask.any()),
    }


class _Compiled:
    """One parametrised cvxpy problem (alpha solve or vol-target projection).

    The risk aversion enters only through parameter *values*: the risk term is
    ``||Lk' f||^2 + ||sdk o w||^2 + (yk'w)^2`` with ``Lk = sqrt(kappa) L``,
    ``sdk = sqrt(kappa) sd``, ``yk = sqrt(kappa nu) y`` and ``kappa = gamma * H``,
    so the gamma search re-solves the same problem (and the problem stays DPP:
    ``gamma * sum_squares(L' f)`` with both ``gamma`` and ``L`` parameters would
    not be).  The SOC vol cap uses the unscaled ``L`` / ``sd`` / ``y``."""

    def __init__(
        self,
        kind: str,
        key: tuple,
        n: int,
        K: int,
        layout: dict[str, Any],
        book: str,
        cfg: PortfolioConfig,
        beta_band: float,
        coefs: dict[str, float],
        with_part: bool,
        vol_cap: bool,
        cost_model: CostTerms,
        cost_args: tuple,
        extra_fns: Sequence[ExtraConstraint] | None,
        tickers: pd.Index,
        exact_power: bool = True,
    ):
        self.kind, self.key = kind, key
        w = cp.Variable(n, name="w")
        x = cp.Variable(n, name="x")
        f = cp.Variable(K, name="f")  # factor exposures B'w
        self.w, self.x, self.f = w, x, f
        self.P = P = self._parameters(n, K, layout)
        self.cons, self.participation = build_constraints(w, x, f, P, layout, book, cfg, beta_band)
        extra: list[cp.Constraint] = []
        for fn in extra_fns or []:
            extra.extend(fn(w, x, tickers))
        clist = list(self.cons.values()) + extra + ([self.participation] if with_part else [])

        # cost terms: the cost model's own expressions, constants -> parameters
        tc, hc = cost_model.cvx_terms(x, w, *cost_args)
        self.cost_sig = (_leaf_sig(tc), _leaf_sig(hc))
        pz = _Parametrizer(exact_power)
        tc_p = pz(tc)
        hc_p = pz(hc)
        self.cost_params: list[cp.Parameter] = pz.params
        self.cost_recipes: list[tuple] = pz.recipes
        self.risk_cap: cp.Constraint | None = None
        if kind == "alpha":
            obj = self._alpha_objective(coefs, tc_p, hc_p)
            if vol_cap:
                self.risk_cap = self._risk_norm(w) <= P["tgt"]
                clist.append(self.risk_cap)
            self.problem = cp.Problem(cp.Maximize(obj), clist)
        else:  # projection: kappa * tracking variance to the anchor + c * TC (both NAV fractions)
            obj = self._projection_objective(coefs, tc_p)
            self.risk_cap = self._risk_norm(w) <= P["tgt"]
            clist.append(self.risk_cap)
            self.problem = cp.Problem(cp.Minimize(obj), clist)
        self.trade_cost_expr = tc_p
        self.is_dpp = self.problem.is_dpp()
        self.n_solves = 0

    @staticmethod
    def _parameters(n: int, K: int, layout: Mapping[str, Any]) -> dict[str, cp.Parameter]:
        """Every day-varying value of the problem (DPP parameters)."""
        Kd = len(layout["dynamic"])
        P = {
            "a": cp.Parameter(n, name="alpha"),
            "w0": cp.Parameter(n, name="w0"),
            "L": cp.Parameter((K, K), name="L"),
            "sd": cp.Parameter(n, nonneg=True, name="sd"),
            "y": cp.Parameter(n, name="y_sqrt_nu"),
            "Lk": cp.Parameter((K, K), name="L_sqrt_kappa"),
            "sdk": cp.Parameter(n, nonneg=True, name="sd_sqrt_kappa"),
            "yk": cp.Parameter(n, name="y_sqrt_nu_kappa"),
            "part": cp.Parameter(n, nonneg=True, name="part_cap"),
            "nt": cp.Parameter(n, nonneg=True, name="no_trade_cap"),
            "fz": cp.Parameter(n, nonneg=True, name="force_cap"),
            "tgt": cp.Parameter(nonneg=True, name="vol_cap_daily"),
            # projection anchor s w*, entered through its risk-scaled images
            "fa": cp.Parameter(K, name="anchor_factor"),  # Lk' B' (s w*)
            "sda": cp.Parameter(n, name="anchor_specific"),  # sdk o (s w*)
            "ya": cp.Parameter(name="anchor_alignment"),  # yk' (s w*)
        }
        if Kd:
            P["Bd"] = cp.Parameter((n, Kd), name="B_dynamic")
        if layout["ind"]:
            P["oi"] = cp.Parameter(len(layout["ind"]), name="ind_off")
        if layout["sty"]:
            P["os"] = cp.Parameter(len(layout["sty"]), name="sty_off")
        return P

    def _alpha_objective(self, coefs: Mapping[str, float], tc_p: cp.Expression, hc_p: cp.Expression):
        """``alpha'w - kappa w'Sigma w - c_trade TC - c_hold HC [- c_turn ||x||_1]``."""
        P, w = self.P, self.w
        risk = (
            cp.sum_squares(P["Lk"].T @ self.f)
            + cp.sum_squares(cp.multiply(P["sdk"], w))
            + cp.square(P["yk"] @ w)
        )
        obj = P["a"] @ w - risk - coefs["trade"] * tc_p - coefs["hold"] * hc_p
        if coefs["turnover"] > 0:
            obj = obj - coefs["turnover"] * cp.norm1(self.x)
        return obj

    def _projection_objective(self, coefs: Mapping[str, float], tc_p: cp.Expression):
        """``kappa (w - s w*)' Sigma (w - s w*) [+ c_proj TC]``."""
        P, w = self.P, self.w
        obj = (
            cp.sum_squares(P["Lk"].T @ self.f - P["fa"])
            + cp.sum_squares(cp.multiply(P["sdk"], w) - P["sda"])
            + cp.square(P["yk"] @ w - P["ya"])
        )
        if coefs["proj_cost"] > 0:
            obj = obj + coefs["proj_cost"] * tc_p
        return obj

    def binding(self, pdat: _ProblemData, with_part: bool) -> dict[str, int]:
        """Binding-entry counts after a solve (:func:`binding_counts`)."""
        return binding_counts(self.cons, self.participation, pdat, with_part)

    def _risk_norm(self, w: cp.Variable) -> cp.Expression:
        """``sqrt(w' Sigma w)`` (daily) as a second-order-cone expression."""
        P = self.P
        return cp.norm(
            cp.hstack([P["L"].T @ self.f, cp.multiply(P["sd"], w), cp.reshape(P["y"] @ w, (1,), order="C")]),
            2,
        )

    def set_kappa(self, pdat: _ProblemData, kappa: float, anchor: np.ndarray | None = None) -> None:
        """Risk-aversion-dependent values: ``kappa = gamma * H`` (and the
        projection anchor's risk-scaled images)."""
        P = self.P
        rk = float(np.sqrt(max(kappa, 0.0)))
        Lo = pdat.L_ordered()
        P["Lk"].value = rk * Lo
        P["sdk"].value = rk * pdat.sd
        P["yk"].value = rk * pdat.y_scaled()
        n = len(pdat.tickers)
        anc = np.zeros(n) if anchor is None else np.asarray(anchor, dtype=float)
        P["fa"].value = rk * (Lo.T @ (pdat.B_ordered().T @ anc))
        P["sda"].value = rk * pdat.sd * anc
        P["ya"].value = rk * float(pdat.y_scaled() @ anc)

    def set_values(
        self,
        pdat: _ProblemData,
        anchor: np.ndarray | None = None,
        tgt: float | None = None,
        kappa: float = 0.0,
    ) -> None:
        P = self.P
        P["a"].value = pdat.a
        P["w0"].value = pdat.w0
        P["L"].value = pdat.L_ordered()
        if "Bd" in P:
            P["Bd"].value = pdat.cols(pdat.dynamic_cols)
        P["sd"].value = pdat.sd
        P["y"].value = pdat.y_scaled()
        P["part"].value = np.clip(pdat.part_cap, 0.0, None)
        P["nt"].value = np.where(pdat.no_trade_mask, 0.0, _BIG)
        P["fz"].value = np.where(pdat.force_mask, np.clip(pdat.force_caps, 0.0, None), _BIG)
        if "oi" in P:
            P["oi"].value = pdat.offsets(pdat.ind_cols)
        if "os" in P:
            P["os"].value = pdat.offsets(pdat.sty_cols)
        P["tgt"].value = float(tgt) if tgt is not None else 1.0
        self.set_kappa(pdat, kappa, anchor)
