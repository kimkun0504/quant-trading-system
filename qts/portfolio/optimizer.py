"""Single-period convex portfolio optimizer (Boyd et al. SPO) on a factor
risk model.

Objective (all terms in *horizon* units, H = alpha horizon in days)::

    maximize  alpha' w_post
              - gamma_risk * H * ( f' F f + sum_i d_i w_i^2 + nu (y' w)^2 )
              - gamma_trade * TC(x) - gamma_hold * H * HC(w_post)
              - turnover_penalty * ||x||_1
    where     f = B' w_post,  x = w_post - w_current

``alpha`` is the expected return over H days, ``F``/``d`` are daily variances
so ``H *`` puts risk on the same footing (equivalently an annual risk
aversion applied to annualised quantities).  The factor form keeps the
problem O(N K^2): the full N x N covariance is never built.  ``F`` is made
PSD by eigenvalue clipping and entered as ``||L' f||^2`` with ``F = L L'``.

Constraint menu (driven by ``PortfolioConfig``): book type, gross/net,
per-name cap, ADV participation, beta neutrality, industry bands, style
bands, no-trade list, extra user constraints.  Participation bounds are
dropped and the problem re-solved if they make the problem infeasible
(status records this).  Infeasible / failed solves return ``w_current``.
"""
from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import cvxpy as cp
import numpy as np
import pandas as pd

from qts.core.config import PortfolioConfig
from qts.core.types import RiskModelOutput
from qts.portfolio.costs import CostTerms, Snapshot, align_snapshot
from qts.portfolio.diagnostics import transfer_coefficient
from qts.portfolio.vol_target import portfolio_vol_scalar

log = logging.getLogger(__name__)

BETA_FACTOR_NAMES = ("beta", "market", "mkt")
ExtraConstraint = Callable[[cp.Variable, cp.Expression, pd.Index], List[cp.Constraint]]


@dataclass
class OptimizationResult:
    w_target: pd.Series
    x: pd.Series
    expected_alpha: float          # alpha' w_target over the horizon
    expected_risk: float           # annualised ex-ante vol of w_target
    expected_cost: float           # trade cost + H * holding cost (fraction of NAV)
    turnover: float                # sum |x| (two-sided, fraction of NAV)
    transfer_coefficient: float
    status: str
    solve_time: float
    binding_constraints: Dict[str, int] = field(default_factory=dict)
    objective_value: float = float("nan")
    vol_scalar: float = 1.0
    solver: str = ""
    notes: List[str] = field(default_factory=list)
    exante_var_daily: float = float("nan")


def _psd_factor(F: np.ndarray, floor: float = 0.0) -> np.ndarray:
    """Return L with L L' = PSD-clipped symmetric part of F."""
    Fs = 0.5 * (F + F.T)
    vals, vecs = np.linalg.eigh(Fs)
    vals = np.clip(vals, floor, None)
    return vecs * np.sqrt(vals)[None, :]


class SinglePeriodOptimizer:
    def __init__(self, cfg: PortfolioConfig, cost_model: CostTerms, horizon_days: int = 20,
                 beta_band: float = 0.05, solver_opts: Optional[dict] = None, verbose: bool = False):
        self.cfg = cfg
        self.cost_model = cost_model
        self.horizon_days = int(horizon_days)
        self.beta_band = beta_band
        self.solver_opts = solver_opts or {}
        self.verbose = verbose

    # ------------------------------------------------------------------ utils
    def _solver_chain(self) -> List[str]:
        chain = [self.cfg.solver, "CLARABEL", "SCS", "OSQP"]
        out: List[str] = []
        for s in chain:
            if s and s.upper() not in out and s.upper() in cp.installed_solvers():
                out.append(s.upper())
        return out

    @staticmethod
    def _find_beta_column(factors: pd.Index) -> Optional[str]:
        for c in factors:
            if str(c).lower() in BETA_FACTOR_NAMES:
                return c
        return None

    # ------------------------------------------------------------------ solve
    def solve(
        self,
        alpha: pd.Series,
        risk: RiskModelOutput,
        w_current: pd.Series,
        nav: float,
        snapshot: Snapshot,
        constraints_extra: Optional[Sequence[ExtraConstraint]] = None,
        venue: str = "auction",
        neutral_styles: Optional[Sequence[str]] = None,
        industry_columns: Optional[Sequence[str]] = None,
        no_trade: Optional[Sequence[str]] = None,
        benchmark_weights: Optional[pd.Series] = None,
        horizon_days: Optional[int] = None,
    ) -> OptimizationResult:
        cfg = self.cfg
        H = int(horizon_days or self.horizon_days)
        t0 = time.perf_counter()
        notes: List[str] = []

        # ---- universe = union of alpha, current holdings and risk model
        tickers = pd.Index(alpha.index).union(pd.Index(w_current.index)).union(risk.tickers)
        tickers = pd.Index(sorted(tickers))
        n = len(tickers)
        a = pd.Series(alpha, dtype=float).reindex(tickers).fillna(0.0).values
        w0 = pd.Series(w_current, dtype=float).reindex(tickers).fillna(0.0).values
        snap = align_snapshot(snapshot, tickers)

        B = risk.exposures.reindex(tickers).fillna(0.0)
        factors = list(B.columns)
        Bv = B.values
        F = risk.factor_cov.loc[factors, factors].values
        L = _psd_factor(F)
        d = pd.Series(risk.specific_var, dtype=float).reindex(tickers)
        d = d.fillna(d.median() if np.isfinite(d.median()) else 1e-4).clip(lower=1e-10).values
        sd = np.sqrt(d)
        y = None
        if risk.alignment_vector is not None and risk.alignment_var > 0:
            y = pd.Series(risk.alignment_vector, dtype=float).reindex(tickers).fillna(0.0).values

        # ---- variables & objective
        w = cp.Variable(n, name="w")
        x = w - w0
        f = Bv.T @ w
        risk_expr = cp.sum_squares(L.T @ f) + cp.sum_squares(cp.multiply(sd, w))
        if y is not None:
            risk_expr = risk_expr + risk.alignment_var * cp.square(y @ w)
        trade_cost, hold_cost = self.cost_model.cvx_terms(x, w, nav, snap, venue)
        obj = a @ w - cfg.risk_aversion * H * risk_expr \
            - cfg.trade_aversion * trade_cost - cfg.hold_aversion * H * hold_cost
        if cfg.turnover_penalty > 0:
            obj = obj - cfg.turnover_penalty * cp.norm1(x)

        # ---- constraints
        cons: Dict[str, cp.Constraint] = {}
        book = cfg.book
        if book == "long_only":
            cons["long_only"] = w >= 0
            cons["fully_invested"] = cp.sum(w) == 1
        elif book == "130_30":
            cons["long_leg"] = cp.sum(cp.pos(w)) <= 1.3
            cons["short_leg"] = cp.sum(cp.neg(w)) <= 0.3
            cons["net_hi"] = cp.sum(w) <= 1.0 + cfg.net_exposure_band
            cons["net_lo"] = cp.sum(w) >= 1.0 - cfg.net_exposure_band
        else:  # long_short
            cons["gross"] = cp.norm1(w) <= cfg.gross_leverage_max
            cons["net_hi"] = cp.sum(w) <= cfg.net_exposure_target + cfg.net_exposure_band
            cons["net_lo"] = cp.sum(w) >= cfg.net_exposure_target - cfg.net_exposure_band
        cons["max_position"] = cp.abs(w) <= cfg.max_position_weight

        part_cap = cfg.max_adv_participation * snap["adv_usd"].values / max(nav, 1.0)
        participation = cp.abs(x) <= part_cap

        beta_col = self._find_beta_column(B.columns)
        if cfg.beta_neutral:
            if beta_col is not None:
                j = factors.index(beta_col)
                cons["beta_neutral"] = cp.abs(Bv[:, j] @ w) <= self.beta_band
            else:
                notes.append("beta_neutral requested but no beta/market factor in risk model; skipped")

        wb = None
        if benchmark_weights is not None:
            wb = pd.Series(benchmark_weights, dtype=float).reindex(tickers).fillna(0.0).values
        ind_cols = list(industry_columns) if industry_columns else [c for c in factors if str(c).upper().startswith("IND")]
        ind_cols = [c for c in ind_cols if c in factors]
        if ind_cols and cfg.industry_band is not None and cfg.industry_band > 0:
            idx = [factors.index(c) for c in ind_cols]
            Bi = Bv[:, idx]
            rel = w if wb is None else (w - wb)
            cons["industry_band"] = cp.abs(Bi.T @ rel) <= cfg.industry_band
        sty = [c for c in (neutral_styles or []) if c in factors and c != beta_col]
        if sty and cfg.style_band_sd is not None and cfg.style_band_sd > 0:
            idx = [factors.index(c) for c in sty]
            rel = w if wb is None else (w - wb)
            cons["style_band"] = cp.abs(Bv[:, idx].T @ rel) <= cfg.style_band_sd
        nt = [t for t in (no_trade or []) if t in tickers]
        if nt:
            mask = np.isin(tickers, nt)
            cons["no_trade"] = x[mask] == 0
        extra: List[cp.Constraint] = []
        for k, fn in enumerate(constraints_extra or []):
            extra.extend(fn(w, x, tickers))

        # ---- solve with fallbacks: participation hard -> relaxed
        status, solver_used, obj_val = "failed", "", float("nan")
        attempts = [("participation", True), ("participation_relaxed", False)]
        for label, with_part in attempts:
            clist = list(cons.values()) + extra + ([participation] if with_part else [])
            prob = cp.Problem(cp.Maximize(obj), clist)
            status, solver_used = self._try_solvers(prob)
            if status in ("optimal", "optimal_inaccurate"):
                obj_val = float(prob.value)
                if not with_part:
                    notes.append("ADV participation bounds relaxed (infeasible with them)")
                break
            notes.append(f"{label}: {status}")

        solve_time = time.perf_counter() - t0
        if status not in ("optimal", "optimal_inaccurate") or w.value is None:
            log.warning("optimizer failed (%s); returning current weights", status)
            w_t = pd.Series(w0, index=tickers)
            var_d = float(risk.portfolio_variance(w_t))
            return OptimizationResult(
                w_target=w_t, x=pd.Series(0.0, index=tickers), expected_alpha=float(a @ w0),
                expected_risk=float(np.sqrt(max(var_d, 0) * 252)), expected_cost=0.0, turnover=0.0,
                transfer_coefficient=float("nan"), status=status, solve_time=solve_time,
                binding_constraints={}, objective_value=obj_val, solver=solver_used, notes=notes,
                exante_var_daily=var_d,
            )

        wv = np.asarray(w.value).ravel()
        wv[np.abs(wv) < 1e-6] = 0.0
        # keep tiny solver noise from creating trades for untouched names
        wv = np.where(np.abs(wv - w0) < 1e-6, w0, wv)

        # ---- binding constraints via duals
        binding: Dict[str, int] = {}
        for name, c in cons.items():
            dv = c.dual_value
            if dv is None:
                continue
            dv = np.atleast_1d(np.asarray(dv, dtype=float))
            cnt = int((np.abs(dv) > 1e-6).sum())
            if cnt:
                binding[name] = cnt
        if participation in prob.constraints and participation.dual_value is not None:
            cnt = int((np.abs(np.atleast_1d(participation.dual_value)) > 1e-6).sum())
            if cnt:
                binding["participation"] = cnt

        # ---- vol targeting (scale down only; never beyond caps)
        w_t = pd.Series(wv, index=tickers)
        var_d = float(risk.portfolio_variance(w_t))
        pred_vol = float(np.sqrt(max(var_d, 0.0) * 252))
        # Scale-up is allowed (cfg.vol_target_scale_up) only as far as the hard
        # caps permit: gross leverage, per-name weight and ADV participation.
        max_up = 1.0
        if getattr(cfg, "vol_target_scale_up", False):
            gross = float(np.abs(wv).sum())
            bounds = [cfg.gross_leverage_max / gross if gross > 1e-12 else np.inf,
                      cfg.max_position_weight / float(np.abs(wv).max()) if np.abs(wv).max() > 1e-12 else np.inf]
            part = None
            try:
                adv = np.asarray(snap["adv_usd"], dtype=float)
                cap = cfg.max_adv_participation * adv / float(nav)
                dw = wv - w0
                pos = (np.abs(dw) > 1e-12) & (np.abs(wv) > 1e-12)
                # |s*w - w0| <= cap  ->  s <= (cap + sign(dw)*w0) / |w| when scaling moves away from w0
                part = float(np.min((cap[pos] + np.abs(w0[pos])) / np.abs(wv[pos]))) if pos.any() else np.inf
            except Exception:
                part = np.inf
            bounds.append(part if part is not None else np.inf)
            max_up = float(max(1.0, min(bounds)))
        scalar = portfolio_vol_scalar(pred_vol, cfg.target_vol_annual, max_leverage_scalar=max_up)
        if abs(scalar - 1.0) > 1e-9:
            w_t = w_t * scalar
            var_d = float(risk.portfolio_variance(w_t))
            pred_vol = float(np.sqrt(max(var_d, 0.0) * 252))
            notes.append(f"vol-target scalar {scalar:.3f}")
        x_t = w_t - pd.Series(w0, index=tickers)

        # ---- expected cost at the final (possibly scaled) trade
        exp_cost = float("nan")
        try:
            xc = cp.Constant(x_t.values)
            wc = cp.Constant(w_t.values)
            tc, hc = self.cost_model.cvx_terms(xc, wc, nav, snap, venue)
            exp_cost = float(np.asarray(tc.value).ravel()[0] if np.ndim(tc.value) else tc.value) \
                + H * float(np.asarray(hc.value).ravel()[0] if np.ndim(hc.value) else hc.value)
        except Exception as e:  # pragma: no cover - diagnostic only
            notes.append(f"expected_cost unavailable: {e}")

        tc_val = transfer_coefficient(pd.Series(a, index=tickers), w_t, risk=risk) if np.any(a != 0) else float("nan")
        return OptimizationResult(
            w_target=w_t, x=x_t, expected_alpha=float(a @ w_t.values), expected_risk=pred_vol,
            expected_cost=exp_cost, turnover=float(np.abs(x_t.values).sum()),
            transfer_coefficient=tc_val, status=status, solve_time=solve_time,
            binding_constraints=binding, objective_value=obj_val, vol_scalar=scalar,
            solver=solver_used, notes=notes, exante_var_daily=var_d,
        )

    def _try_solvers(self, prob: cp.Problem):
        last = "failed"
        for s in self._solver_chain():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    prob.solve(solver=s, verbose=self.verbose, **self.solver_opts)
            except (cp.SolverError, Exception) as e:  # solver not applicable / crashed
                log.debug("solver %s failed: %s", s, e)
                last = f"error:{type(e).__name__}"
                continue
            last = prob.status
            if prob.status in ("optimal", "optimal_inaccurate"):
                return prob.status, s
            if prob.status in ("infeasible", "unbounded"):
                return prob.status, s
        return last, ""
