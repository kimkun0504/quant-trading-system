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
bands, no-trade list, forced exits, extra user constraints.

Factor roles come from ``RiskModelOutput.factor_groups`` (``"style"`` /
``"industry"``), never from column-name prefixes: beta neutrality binds the
style factor ``cfg.beta_factor_name`` (default ``"beta"``) inside
``+-cfg.beta_band`` -- *not* the unit-exposure market/country factor -- and
industry bands cover every column in ``factor_groups["industry"]`` (so
Fama-French labels such as ``Banks`` work).  Prefix detection is used only
when ``factor_groups`` is empty, with a logged warning.

Vol targeting is a **projection re-solve**, not a post-hoc multiplication:
when the predicted vol of the optimum ``w*`` differs from the target the book
is re-solved as ``minimize ||w - s w*||^2`` subject to the identical
constraint set plus ``sqrt(w' Sigma w) <= target``, so every band the first
solve honoured still holds after scaling (``s = target / predicted``, capped
by the gross and per-name limits when levering up).

Failure semantics: a solve that is not ``optimal`` / ``optimal_inaccurate``
returns ``status`` as reported by the solver, ``w_target = w_current``,
``expected_cost = nan`` and logs a warning.  Dropping the ADV participation
caps on infeasibility is opt-in (``cfg.relax_participation_on_infeasible``).
"""
from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import cvxpy as cp
import numpy as np
import pandas as pd

from qts.core.config import PortfolioConfig
from qts.core.types import RiskModelOutput
from qts.portfolio.costs import CostTerms, Snapshot, align_snapshot
from qts.portfolio.diagnostics import transfer_coefficient

log = logging.getLogger(__name__)

OK_STATUSES = ("optimal", "optimal_inaccurate")
ExtraConstraint = Callable[[cp.Variable, cp.Expression, pd.Index], List[cp.Constraint]]
SOLVER_TOL = 1e-6
MAX_SCALE_UP = 5.0   # sanity cap on the vol-target projection anchor s = target / predicted


@dataclass
class OptimizationResult:
    w_target: pd.Series
    x: pd.Series
    expected_alpha: float          # alpha' w_target over the horizon
    expected_risk: float           # annualised ex-ante vol of w_target
    expected_cost: float           # trade cost + H * holding cost (fraction of NAV); nan on failure
    turnover: float                # sum |x| (two-sided, fraction of NAV)
    transfer_coefficient: float
    status: str
    solve_time: float
    binding_constraints: Dict[str, int] = field(default_factory=dict)
    objective_value: float = float("nan")
    vol_scalar: float = 1.0        # s used in the vol-target projection (1 = none)
    solver: str = ""
    notes: List[str] = field(default_factory=list)
    exante_var_daily: float = float("nan")
    n_out_of_model: int = 0        # material names (alpha or position != 0) outside the risk universe
    alignment_share: float = float("nan")  # share of ex-ante variance from the alignment term

    @property
    def ok(self) -> bool:
        return self.status in OK_STATUSES


def _psd_factor(F: np.ndarray, floor: float = 0.0) -> np.ndarray:
    """Return L with L L' = PSD-clipped symmetric part of F."""
    Fs = 0.5 * (F + F.T)
    vals, vecs = np.linalg.eigh(Fs)
    vals = np.clip(vals, floor, None)
    return vecs * np.sqrt(vals)[None, :]


@dataclass
class _ProblemData:
    """Everything the constraint builder and the two solves share."""
    tickers: pd.Index
    a: np.ndarray
    w0: np.ndarray
    snap: Snapshot
    factors: List[str]
    Bv: np.ndarray
    F: np.ndarray
    L: np.ndarray
    d: np.ndarray
    sd: np.ndarray
    y: Optional[np.ndarray]
    nu: float
    nav: float
    beta_col: Optional[str]
    ind_cols: List[str]
    sty_cols: List[str]
    wb: Optional[np.ndarray]
    no_trade_mask: np.ndarray
    force_mask: np.ndarray
    force_caps: np.ndarray
    part_cap: np.ndarray
    n_out_of_model: int

    def variance(self, w: np.ndarray) -> float:
        fw = self.Bv.T @ w
        v = float(fw @ self.F @ fw + (w ** 2 * self.d).sum())
        if self.y is not None:
            v += self.nu * float(self.y @ w) ** 2
        return v

    def alignment_share(self, w: np.ndarray) -> float:
        v = self.variance(w)
        if self.y is None or v <= 0:
            return 0.0 if v > 0 else float("nan")
        return self.nu * float(self.y @ w) ** 2 / v

    def risk_expr(self, w: cp.Variable) -> cp.Expression:
        f = self.Bv.T @ w
        e = cp.sum_squares(self.L.T @ f) + cp.sum_squares(cp.multiply(self.sd, w))
        if self.y is not None:
            e = e + self.nu * cp.square(self.y @ w)
        return e

    def risk_norm(self, w: cp.Variable) -> cp.Expression:
        """``sqrt(w' Sigma w)`` as a second-order-cone expression."""
        parts = [self.L.T @ (self.Bv.T @ w), cp.multiply(self.sd, w)]
        if self.y is not None:
            parts.append(cp.reshape(np.sqrt(self.nu) * (self.y @ w), (1,), order="C"))
        return cp.norm(cp.hstack(parts), 2)


class SinglePeriodOptimizer:
    def __init__(self, cfg: PortfolioConfig, cost_model: CostTerms, horizon_days: int = 20,
                 beta_band: Optional[float] = None, solver_opts: Optional[dict] = None, verbose: bool = False):
        self.cfg = cfg
        self.cost_model = cost_model
        self.horizon_days = int(horizon_days)
        self.beta_band = float(getattr(cfg, "beta_band", 0.05) if beta_band is None else beta_band)
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

    # ------------------------------------------------------------ factor roles
    def _resolve_factor_roles(self, risk: RiskModelOutput, factors: List[str],
                              neutral_styles: Optional[Sequence[str]],
                              industry_columns: Optional[Sequence[str]],
                              notes: List[str]) -> Tuple[Optional[str], List[str], List[str]]:
        """``(beta_col, industry_cols, style_band_cols)`` from
        ``risk.factor_groups``; prefix fallback only when the groups are empty."""
        cfg = self.cfg
        groups = getattr(risk, "factor_groups", None) or {}
        has_groups = any(len(v) for v in groups.values()) if isinstance(groups, dict) else False
        beta_name = str(getattr(cfg, "beta_factor_name", "beta"))
        if has_groups:
            style = [c for c in groups.get("style", []) if c in factors]
            industry = [c for c in groups.get("industry", []) if c in factors]
            unknown = [s for s in (neutral_styles or []) if s not in style]
            if unknown:
                raise ValueError(f"neutral_styles {unknown} are not in factor_groups['style'] = {style}")
            beta_col = beta_name if beta_name in style else None
            if cfg.beta_neutral and beta_col is None:
                notes.append(f"beta_neutral requested but style '{beta_name}' not in "
                             f"factor_groups['style']; skipped")
            if industry_columns:
                ind_cols = [c for c in industry_columns if c in factors]
            else:
                ind_cols = industry
        else:
            log.warning("risk model output has no factor_groups; falling back to name-based factor "
                        "role detection (beta = exact '%s', industry = 'IND*' prefix)", beta_name)
            notes.append("factor_groups missing: prefix fallback for factor roles")
            low = {str(c).lower(): c for c in factors}
            beta_col = low.get(beta_name.lower())
            if cfg.beta_neutral and beta_col is None:
                notes.append(f"beta_neutral requested but no '{beta_name}' factor in risk model; skipped")
            if industry_columns:
                ind_cols = [c for c in industry_columns if c in factors]
            else:
                ind_cols = [c for c in factors if str(c).upper().startswith("IND")]
            style = [c for c in (neutral_styles or []) if c in factors]
        if cfg.industry_band is not None and cfg.industry_band > 0 and not ind_cols:
            notes.append("industry_band > 0 but no industry factors identified; band not applied")
        # beta already has its own (tighter) band when beta_neutral is on
        sty = [c for c in (neutral_styles or []) if c in factors and not (cfg.beta_neutral and c == beta_col)]
        return beta_col, ind_cols, sty

    # ------------------------------------------------------------ problem data
    def _prepare(self, alpha: pd.Series, risk: RiskModelOutput, w_current: pd.Series, nav: float,
                 snapshot: Snapshot, neutral_styles, industry_columns, no_trade, force_to_zero,
                 benchmark_weights, notes: List[str]) -> _ProblemData:
        cfg = self.cfg
        tickers = pd.Index(alpha.index).union(pd.Index(w_current.index)).union(risk.tickers)
        tickers = pd.Index(sorted(tickers))
        a = pd.Series(alpha, dtype=float).reindex(tickers).fillna(0.0).values
        w0 = pd.Series(w_current, dtype=float).reindex(tickers).fillna(0.0).values
        snap = align_snapshot(snapshot, tickers)

        B = risk.exposures.reindex(tickers).fillna(0.0)
        factors = list(B.columns)
        Bv = B.values
        F = risk.factor_cov.loc[factors, factors].values
        L = _psd_factor(F)
        d_known = pd.Series(risk.specific_var, dtype=float)
        d = d_known.reindex(tickers)
        in_model = tickers.isin(risk.tickers)
        # names outside the risk universe: zero factor exposure, so give them a
        # *conservative* specific variance (90th percentile) rather than the median
        fin = d_known.dropna()
        fill = float(np.percentile(fin.values, 90)) if len(fin) else 1e-4
        d = d.fillna(fill).clip(lower=1e-10).values
        material = (~in_model) & ((a != 0) | (w0 != 0))
        n_out = int(material.sum())
        if n_out:
            notes.append(f"{n_out} material names outside the risk model universe; "
                         f"specific var set to 90th pct ({fill:.2e}), zero factor exposure")
            log.debug("optimizer: %d names outside risk universe on %s", n_out, getattr(risk, "date", "?"))
        sd = np.sqrt(d)
        y = None
        nu = 0.0
        if risk.alignment_vector is not None and risk.alignment_var > 0:
            y = pd.Series(risk.alignment_vector, dtype=float).reindex(tickers).fillna(0.0).values
            nu = float(risk.alignment_var)

        beta_col, ind_cols, sty = self._resolve_factor_roles(risk, factors, neutral_styles, industry_columns, notes)
        wb = None
        if benchmark_weights is not None:
            wb = pd.Series(benchmark_weights, dtype=float).reindex(tickers).fillna(0.0).values

        part_cap = cfg.max_adv_participation * np.asarray(snap["adv_usd"].values, dtype=float) / max(nav, 1.0)
        nt = [t for t in (no_trade or []) if t in tickers]
        no_trade_mask = np.isin(tickers, nt)
        force_mask = np.zeros(len(tickers), dtype=bool)
        force_caps = np.zeros(len(tickers))
        if force_to_zero:
            fz = pd.Series(dict(force_to_zero), dtype=float).reindex(tickers)
            force_mask = fz.notna().values & ~no_trade_mask
            req = np.nan_to_num(fz.values, nan=np.inf)
            # never ask for more than the ADV cap allows in one day (keeps the problem feasible)
            force_caps = np.where(force_mask, np.maximum(np.maximum(req, 0.0), np.abs(w0) - part_cap), 0.0)
        return _ProblemData(tickers=tickers, a=a, w0=w0, snap=snap, factors=factors, Bv=Bv, F=F, L=L,
                            d=d, sd=sd, y=y, nu=nu, nav=float(nav), beta_col=beta_col, ind_cols=ind_cols,
                            sty_cols=sty, wb=wb, no_trade_mask=no_trade_mask, force_mask=force_mask,
                            force_caps=force_caps, part_cap=part_cap, n_out_of_model=n_out)

    # ------------------------------------------------------------- constraints
    def _build_constraints(self, w: cp.Variable, pdat: _ProblemData,
                           constraints_extra: Optional[Sequence[ExtraConstraint]]
                           ) -> Tuple[Dict[str, cp.Constraint], cp.Constraint, List[cp.Constraint]]:
        """Full constraint set: ``(named, participation, extra)``.  Used
        unchanged by the alpha solve and by the vol-target projection."""
        cfg = self.cfg
        x = w - pdat.w0
        Bv, factors = pdat.Bv, pdat.factors
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

        participation = cp.abs(x) <= pdat.part_cap

        if cfg.beta_neutral and pdat.beta_col is not None:
            j = factors.index(pdat.beta_col)
            cons["beta_neutral"] = cp.abs(Bv[:, j] @ w) <= self.beta_band

        rel = w if pdat.wb is None else (w - pdat.wb)
        if pdat.ind_cols and cfg.industry_band is not None and cfg.industry_band > 0:
            idx = [factors.index(c) for c in pdat.ind_cols]
            cons["industry_band"] = cp.abs(Bv[:, idx].T @ rel) <= cfg.industry_band
        if pdat.sty_cols and cfg.style_band_sd is not None and cfg.style_band_sd > 0:
            idx = [factors.index(c) for c in pdat.sty_cols]
            cons["style_band"] = cp.abs(Bv[:, idx].T @ rel) <= cfg.style_band_sd
        if pdat.no_trade_mask.any():
            cons["no_trade"] = x[pdat.no_trade_mask] == 0
        if pdat.force_mask.any():
            cons["force_to_zero"] = cp.abs(w[pdat.force_mask]) <= pdat.force_caps[pdat.force_mask]
        extra: List[cp.Constraint] = []
        for fn in (constraints_extra or []):
            extra.extend(fn(w, x, pdat.tickers))
        return cons, participation, extra

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
        force_to_zero: Optional[Mapping[str, float]] = None,
    ) -> OptimizationResult:
        """``force_to_zero`` maps ticker -> maximum ``|w|`` allowed today (0 =
        fully out); the simulator uses it to unwind names that left the
        eligible universe over ``cfg.exit_ineligible_days``.  ``no_trade``
        names keep ``x = 0``; a name in both lists is treated as no-trade."""
        cfg = self.cfg
        H = int(horizon_days or self.horizon_days)
        t0 = time.perf_counter()
        notes: List[str] = []
        pdat = self._prepare(alpha, risk, w_current, nav, snapshot, neutral_styles, industry_columns,
                             no_trade, force_to_zero, benchmark_weights, notes)
        tickers, a, w0, snap = pdat.tickers, pdat.a, pdat.w0, pdat.snap
        n = len(tickers)

        # ---- variables & objective
        w = cp.Variable(n, name="w")
        x = w - w0
        trade_cost, hold_cost = self.cost_model.cvx_terms(x, w, nav, snap, venue)
        obj = a @ w - cfg.risk_aversion * H * pdat.risk_expr(w) \
            - cfg.trade_aversion * trade_cost - cfg.hold_aversion * H * hold_cost
        if cfg.turnover_penalty > 0:
            obj = obj - cfg.turnover_penalty * cp.norm1(x)

        cons, participation, extra = self._build_constraints(w, pdat, constraints_extra)

        # ---- solve (participation relaxation is opt-in)
        status, solver_used, obj_val, with_part = "failed", "", float("nan"), True
        attempts: List[Tuple[str, bool]] = [("participation", True)]
        if getattr(cfg, "relax_participation_on_infeasible", False):
            attempts.append(("participation_relaxed", False))
        for label, with_part in attempts:
            clist = list(cons.values()) + extra + ([participation] if with_part else [])
            prob = cp.Problem(cp.Maximize(obj), clist)
            status, solver_used = self._try_solvers(prob)
            if status in OK_STATUSES:
                obj_val = float(prob.value)
                if not with_part:
                    notes.append("ADV participation bounds relaxed (infeasible with them)")
                    log.warning("optimizer: ADV participation caps dropped on %s (infeasible with them)",
                                getattr(risk, "date", "?"))
                break
            notes.append(f"{label}: {status}")

        solve_time = time.perf_counter() - t0
        if status not in OK_STATUSES or w.value is None:
            log.warning("optimizer failed on %s (status=%s, solver=%s); holding current weights",
                        getattr(risk, "date", "?"), status, solver_used or "none")
            w_t = pd.Series(w0, index=tickers)
            var_d = pdat.variance(w0)
            return OptimizationResult(
                w_target=w_t, x=pd.Series(0.0, index=tickers), expected_alpha=float(a @ w0),
                expected_risk=float(np.sqrt(max(var_d, 0) * 252)), expected_cost=float("nan"), turnover=0.0,
                transfer_coefficient=float("nan"), status=status, solve_time=solve_time,
                binding_constraints={}, objective_value=obj_val, solver=solver_used, notes=notes,
                exante_var_daily=var_d, n_out_of_model=pdat.n_out_of_model,
                alignment_share=pdat.alignment_share(w0),
            )

        wv = self._clean(np.asarray(w.value).ravel(), w0)

        # ---- binding constraints via duals (read before any re-solve overwrites them)
        binding: Dict[str, int] = {}
        for name, c in cons.items():
            cnt = self._n_binding(c)
            if cnt:
                binding[name] = cnt
        if with_part:
            cnt = self._n_binding(participation)
            if cnt:
                binding["participation"] = cnt

        # ---- vol targeting by projection onto the same feasible set
        scalar = 1.0
        wv, scalar, proj_note = self._vol_target_projection(w, wv, pdat, cons, participation, extra, with_part,
                                                            date=getattr(risk, "date", "?"))
        if proj_note:
            notes.append(proj_note)
        var_d = pdat.variance(wv)
        pred_vol = float(np.sqrt(max(var_d, 0.0) * 252))
        w_t = pd.Series(wv, index=tickers)
        x_t = w_t - pd.Series(w0, index=tickers)
        share = pdat.alignment_share(wv)
        if pdat.y is not None and np.isfinite(share):
            notes.append(f"alignment share of ex-ante variance {share:.3f}")

        # ---- expected cost at the final trade
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
            transfer_coefficient=tc_val, status=status, solve_time=time.perf_counter() - t0,
            binding_constraints=binding, objective_value=obj_val, vol_scalar=scalar,
            solver=solver_used, notes=notes, exante_var_daily=var_d,
            n_out_of_model=pdat.n_out_of_model, alignment_share=share,
        )

    # ------------------------------------------------------------ vol target
    def _vol_target_projection(self, w: cp.Variable, wv: np.ndarray, pdat: _ProblemData,
                               cons: Dict[str, cp.Constraint], participation: cp.Constraint,
                               extra: List[cp.Constraint], with_part: bool, date=None
                               ) -> Tuple[np.ndarray, float, Optional[str]]:
        """Re-solve ``min ||w - s w*||^2`` over the full constraint set plus the
        risk cap ``sqrt(w' Sigma w) <= target`` whenever the predicted vol of
        ``w*`` is off target (down always; up only with ``vol_target_scale_up``
        and only for long/short books, capped by gross and per-name limits).
        Returns ``(w, s, note)``; on a failed projection ``w*`` is kept."""
        cfg = self.cfg
        target = cfg.target_vol_annual
        if target is None or not np.isfinite(target) or target <= 0:
            return wv, 1.0, None
        var0 = pdat.variance(wv)
        pred = float(np.sqrt(max(var0, 0.0) * 252))
        if pred <= 0 or not np.isfinite(pred):
            return wv, 1.0, None
        s = target / pred
        if abs(s - 1.0) <= 1e-4:
            return wv, 1.0, None
        if s > 1.0:
            if not getattr(cfg, "vol_target_scale_up", False):
                return wv, 1.0, None
            if cfg.book != "long_short":
                return wv, 1.0, f"vol-target scale-up skipped for book '{cfg.book}' (fully invested)"
            gross = float(np.abs(wv).sum())
            if gross >= cfg.gross_leverage_max * (1.0 - 1e-6):
                return wv, 1.0, "vol-target scale-up blocked: gross leverage already at cap"
            # s is only the projection *anchor*: names at their per-name / ADV
            # caps are clipped by the constraints while the rest of the book is
            # levered, and the risk cap keeps the result at or below target.
            s = float(min(s, MAX_SCALE_UP))
        var_cap = (target ** 2) / 252.0
        risk_cap = pdat.risk_norm(w) <= np.sqrt(var_cap) * (1.0 + 1e-7)
        clist = list(cons.values()) + extra + ([participation] if with_part else []) + [risk_cap]
        prob = cp.Problem(cp.Minimize(cp.sum_squares(w - s * wv)), clist)
        status, solver = self._try_solvers(prob)
        if status not in OK_STATUSES or w.value is None:
            log.warning("vol-target projection failed on %s (status=%s); keeping un-scaled book",
                        date, status)
            return wv, 1.0, f"vol-target projection failed ({status}); book left at s=1"
        w_new = self._clean(np.asarray(w.value).ravel(), pdat.w0)
        got = float(np.sqrt(max(pdat.variance(w_new), 0.0) * 252))
        return w_new, float(s), f"vol-target projection s={s:.3f}: vol {pred:.4f} -> {got:.4f} (target {target:.4f}, {solver})"

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _clean(wv: np.ndarray, w0: np.ndarray) -> np.ndarray:
        wv = wv.copy()
        wv[np.abs(wv) < SOLVER_TOL] = 0.0
        # keep tiny solver noise from creating trades for untouched names
        return np.where(np.abs(wv - w0) < SOLVER_TOL, w0, wv)

    @staticmethod
    def _n_binding(c: cp.Constraint) -> int:
        dv = c.dual_value
        if dv is None:
            return 0
        dv = np.atleast_1d(np.asarray(dv, dtype=float))
        return int((np.abs(dv) > SOLVER_TOL).sum())

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
            if prob.status in OK_STATUSES:
                return prob.status, s
            if prob.status in ("infeasible", "unbounded"):
                return prob.status, s
        return last, ""
