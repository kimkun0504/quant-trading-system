""":class:`SinglePeriodOptimizer`: orchestrates one date's solve.

``solve`` is a fixed sequence of steps, each a small method or a function of
a sibling module:

1. :func:`~qts.portfolio.optimizer.problem.prepare_problem` -- numeric inputs;
2. :meth:`SinglePeriodOptimizer._gp_factor` -- Garleanu-Pedersen alpha shrink;
3. :meth:`SinglePeriodOptimizer._sizing_plan` -- risk aversion / vol-target mode;
4. :meth:`SinglePeriodOptimizer._solve_alpha` -- the alpha problem
   (participation relaxation opt-in), compiled or reused
   (:class:`~qts.portfolio.optimizer.compiled._Compiled`);
5. :func:`~qts.portfolio.optimizer.sizing.gamma_search` (``target_vol``) or
   :meth:`SinglePeriodOptimizer._vol_target_projection` (``fixed`` +
   ``projection``);
5b. :meth:`SinglePeriodOptimizer._drop_small_trades` -- the min-trade projection
   (``cfg.min_trade_weight``);
6. :meth:`SinglePeriodOptimizer._final_result` -- diagnostics and the result.

A failed alpha solve returns the current book (:meth:`_hold_result`).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd

from qts.core.config import ExecutionConfig, PortfolioConfig
from qts.core.costs import VENUE_ROUTED
from qts.core.types import RiskModelOutput
from qts.portfolio.alpha_scaling import cost_implied_holding_days, gp_sleeve_shrink_factors
from qts.portfolio.costs import CostTerms, Snapshot
from qts.portfolio.diagnostics import transfer_coefficient
from qts.portfolio.optimizer.compiled import _Compiled, problem_layout, structure_key
from qts.portfolio.optimizer.constants import (
    GAMMA_BRACKET,
    GAMMA_NUDGE,
    MAX_SCALE_UP,
    OK_STATUSES,
    RISK_AVERSION_MODES,
    SOLVER_TOL,
    TERMINAL_STATUSES,
    TRADING_DAYS,
    VOL_TARGET_MODES,
    ExtraConstraint,
)
from qts.portfolio.optimizer.constraints import max_constraint_violation, n_binding
from qts.portfolio.optimizer.dpp import _assign, _constants, _leaf_sig
from qts.portfolio.optimizer.problem import (
    _ProblemData,
    prepare_problem,
    resolve_factor_roles,
    static_columns,
)
from qts.portfolio.optimizer.result import OptimizationResult
from qts.portfolio.optimizer.sizing import (
    GammaSearchResult,
    _objective,
    caps_bind,
    clean_weights,
    closed_form_gamma,
    gamma_cap_search,
    gamma_search,
    position_caps_bind,
)
from qts.portfolio.optimizer.solvers import solve_with_chain, solver_chain

log = logging.getLogger("qts.portfolio.optimizer")


@dataclass(frozen=True)
class _SizingPlan:
    """How this date's book is sized (``cfg.risk_aversion_mode`` / ``vol_target_mode``)."""

    target: float | None  # annual vol target (None / non-positive = none)
    search: bool  # target_vol mode: gamma search
    vol_cap: bool  # fixed mode + "constraint": SOC risk cap in the alpha problem
    projection: bool  # fixed mode + "projection": re-solve toward s w*
    tgt_daily: float | None  # daily vol cap (slightly loosened)
    gamma0: float  # starting annual risk aversion
    gamma_floor: float | None = None  # vol_target_is_cap: gamma is never searched below this


@dataclass
class _AlphaSolve:
    """Outcome of the alpha problem (step 4)."""

    comp: _Compiled
    status: str
    solver: str
    obj: float
    with_part: bool
    gamma: float  # after a possible nudge
    reused: bool
    n_solves: int


@dataclass
class _Book:
    """The book being refined by steps 5-6."""

    w: np.ndarray
    binding: dict[str, int]
    obj: float
    gamma: float
    status: str
    solver: str
    n_solves: int
    vt_binding: str = ""
    scalar: float = 1.0
    n_dropped: int = 0  # trades removed by the min-trade projection
    violation: float = 0.0  # largest constraint violation after it
    violated: str = ""
    notes: list[str] = field(default_factory=list)


class SinglePeriodOptimizer:
    def __init__(
        self,
        cfg: PortfolioConfig,
        cost_model: CostTerms,
        horizon_days: int = 20,
        beta_band: float | None = None,
        solver_opts: dict | None = None,
        verbose: bool = False,
        reuse_problem: bool | None = None,
        exact_power: bool = True,
        exec_cfg: ExecutionConfig | None = None,
    ):
        """``reuse_problem`` overrides ``cfg.reuse_problem`` (cache the DPP problem
        across solves while the ticker set / constraint structure is unchanged).
        ``exact_power`` enters the cost model's 3/2-power impact terms as exact
        power cones instead of cvxpy's SOC approximation
        (:class:`~qts.portfolio.optimizer.dpp._Parametrizer`).
        ``exec_cfg`` (default: the cost model's ``exec_cfg``, else ``ExecutionConfig()``)
        supplies ``algo_max_pct_adv`` for ``cfg.participation_from_urgency``."""
        self.cfg = cfg
        self.exec_cfg = exec_cfg
        self.cost_model = cost_model
        self.horizon_days = int(horizon_days)
        self.beta_band = float(getattr(cfg, "beta_band", 0.05) if beta_band is None else beta_band)
        self.solver_opts = solver_opts or {}
        self.verbose = verbose
        self.reuse_problem = bool(
            getattr(cfg, "reuse_problem", False) if reuse_problem is None else reuse_problem
        )
        self.exact_power = bool(exact_power)
        self._chain: list[str] | None = None
        mode = getattr(cfg, "vol_target_mode", "projection")
        if mode not in VOL_TARGET_MODES:
            raise ValueError(f"vol_target_mode must be one of {VOL_TARGET_MODES}, got {mode!r}")
        ra_mode = getattr(cfg, "risk_aversion_mode", "fixed")
        if ra_mode not in RISK_AVERSION_MODES:
            raise ValueError(f"risk_aversion_mode must be one of {RISK_AVERSION_MODES}, got {ra_mode!r}")
        self._cache: dict[tuple[str, bool], _Compiled] = {}
        self._solve_errors: list[str] = []  # solver exceptions caught during the current solve()
        self.n_builds = 0
        self.n_reuses = 0

    # ------------------------------------------------------------------ solvers
    def _solver_chain(self) -> list[str]:
        if self._chain is None:  # cp.installed_solvers() costs ~10 ms per call
            self._chain = solver_chain(self.cfg.solver, cp.installed_solvers())
        return self._chain

    def _try_solvers(
        self, prob: cp.Problem, dpp: bool = False, primary_only: bool = False
    ) -> tuple[str, str]:
        """``(status, solver)`` from :func:`~qts.portfolio.optimizer.solvers.solve_with_chain`;
        caught solver exceptions are kept for ``OptimizationResult.solver_errors``."""
        att = solve_with_chain(
            prob, self._solver_chain(), self.solver_opts, self.verbose, dpp=dpp, primary_only=primary_only
        )
        self._solve_errors.extend(att.errors)
        return att.status, att.solver

    def _solve_fn(self, comp: _Compiled):
        dpp = self.reuse_problem and comp.is_dpp
        return lambda primary_only: self._try_solvers(comp.problem, dpp=dpp, primary_only=primary_only)

    # ------------------------------------------------------------------ problem data
    def participation_rate(self, half_life_days: float | None = None) -> float:
        """Per-name daily trade cap as a fraction of ADV: ``cfg.max_adv_participation``,
        or with ``cfg.participation_from_urgency`` the router's urgency-derived rate
        :func:`qts.execution.router.participation_for_half_life` (``half_life_days`` =
        ``AlphaOutput.half_life_days``; unknown -> ``algo_max_pct_adv``)."""
        if not getattr(self.cfg, "participation_from_urgency", False):
            return float(self.cfg.max_adv_participation)
        from qts.execution.router import participation_for_half_life

        ec = self.exec_cfg or getattr(self.cost_model, "exec_cfg", None) or ExecutionConfig()
        return participation_for_half_life(half_life_days, float(ec.algo_max_pct_adv))

    def _resolve_factor_roles(
        self,
        risk: RiskModelOutput,
        factors: list[str],
        neutral_styles: Sequence[str] | None,
        industry_columns: Sequence[str] | None,
        notes: list[str],
        has_benchmark: bool = False,
    ) -> tuple[str | None, list[str], list[str]]:
        """:func:`~qts.portfolio.optimizer.problem.resolve_factor_roles` with this optimizer's config."""
        return resolve_factor_roles(
            self.cfg, risk, factors, neutral_styles, industry_columns, notes, has_benchmark
        )

    def _prepare(
        self,
        alpha: pd.Series,
        risk: RiskModelOutput,
        w_current: pd.Series,
        nav: float,
        snapshot: Snapshot,
        neutral_styles: Sequence[str] | None,
        industry_columns: Sequence[str] | None,
        no_trade: Sequence[str] | None,
        force_to_zero: Mapping[str, float] | None,
        benchmark_weights: pd.Series | None,
        notes: list[str],
        part_rate: float | None = None,
    ) -> _ProblemData:
        """:func:`~qts.portfolio.optimizer.problem.prepare_problem` with this optimizer's config."""
        return prepare_problem(
            self.cfg,
            alpha,
            risk,
            w_current,
            nav,
            snapshot,
            neutral_styles,
            industry_columns,
            no_trade,
            force_to_zero,
            benchmark_weights,
            notes,
            part_rate=part_rate,
        )

    _static_columns = staticmethod(static_columns)
    closed_form_gamma = staticmethod(closed_form_gamma)
    _caps_bind = staticmethod(caps_bind)
    _position_caps_bind = staticmethod(position_caps_bind)
    _clean = staticmethod(clean_weights)
    _n_binding = staticmethod(n_binding)

    # ------------------------------------------------------------------ compiled problems
    def _compiled(
        self,
        kind: str,
        pdat: _ProblemData,
        H: int,
        with_part: bool,
        vol_cap: bool,
        venue: str,
        extra_fns: Sequence[ExtraConstraint] | None,
    ) -> tuple[_Compiled, bool]:
        """The cached problem for this structure with today's values set, or a
        fresh one.  Returns ``(compiled, reused)``."""
        cfg = self.cfg
        key = structure_key(cfg, self.beta_band, kind, pdat, H, with_part, vol_cap, venue)
        cost_args = (pdat.nav, pdat.snap, venue)
        slot = (kind, with_part)
        cached = self._cache.get(slot) if (self.reuse_problem and not extra_fns) else None
        if cached is not None and cached.key == key and cached.is_dpp:
            tc, hc = self.cost_model.cvx_terms(cached.x, cached.w, *cost_args)
            if (_leaf_sig(tc), _leaf_sig(hc)) == cached.cost_sig and _assign(
                cached.cost_params, _constants(tc, []) + _constants(hc, []), cached.cost_recipes
            ):
                self.n_reuses += 1
                return cached, True
        coefs = {
            "trade": cfg.trade_aversion,
            "hold": cfg.hold_aversion * H,
            "turnover": cfg.turnover_penalty,
            "proj_cost": float(getattr(cfg, "vol_target_projection_cost_weight", 0.0) or 0.0),
        }
        comp = _Compiled(
            kind,
            key,
            len(pdat.tickers),
            len(pdat.factors),
            problem_layout(pdat),
            cfg.book,
            cfg,
            self.beta_band,
            coefs,
            with_part,
            vol_cap,
            self.cost_model,
            cost_args,
            extra_fns,
            pdat.tickers,
            exact_power=self.exact_power,
        )
        self.n_builds += 1
        if not comp.is_dpp:
            log.debug(
                "optimizer problem is not DPP (cost expression structure); it will be rebuilt per solve"
            )
        if self.reuse_problem and not extra_fns:
            self._cache[slot] = comp
        return comp, False

    def _binding(self, comp: _Compiled, pdat: _ProblemData, with_part: bool) -> dict[str, int]:
        return comp.binding(pdat, with_part)

    # ------------------------------------------------------------------ step 2: GP shrink
    def _gp_factor(
        self, pdat: _ProblemData, H: int, half_life: float | None, venue: str, notes: list[str]
    ) -> float:
        """Garleanu-Pedersen alpha shrink ``g = 1 / (1 + h / hl)`` with the
        cost-implied holding period ``h = round-trip cost / alpha per day``
        (:func:`qts.portfolio.alpha_scaling.cost_implied_holding_days`),
        evaluated on a unit-gross book proportional to alpha traded from flat."""
        if not getattr(self.cfg, "use_gp_shrink", False):
            return 1.0
        if half_life is None or not np.isfinite(half_life) or half_life <= 0:
            notes.append("use_gp_shrink on but no alpha half-life given; GP shrink skipped")
            return 1.0
        a = pdat.a
        gross = float(np.abs(a).sum())
        if gross <= 0:
            return 1.0
        w_dir = a / gross  # unit gross, alpha-proportional
        alpha_per_day = float(a @ w_dir) / max(H, 1)
        try:
            tc, _ = self.cost_model.cvx_terms(
                cp.Constant(w_dir), cp.Constant(w_dir), pdat.nav, pdat.snap, venue
            )
            one_way = _scalar(tc.value)
        except Exception as e:  # pragma: no cover - diagnostic only (GP shrink is skipped, noted)
            notes.append(f"GP shrink skipped: cost unavailable ({e})")
            return 1.0
        hold_days = cost_implied_holding_days(alpha_per_day, 2.0 * one_way)
        g = float(gp_sleeve_shrink_factors({"alpha": half_life}, hold_days)["alpha"])
        notes.append(f"GP shrink g={g:.3f} (holding {hold_days:.1f}d vs half-life {half_life:.1f}d)")
        return g

    # ------------------------------------------------------------------ step 3: sizing plan
    def _sizing_plan(self, pdat: _ProblemData, H: int, notes: list[str]) -> _SizingPlan:
        cfg = self.cfg
        raw = cfg.target_vol_annual
        target = float(raw) if raw is not None and np.isfinite(raw) and raw > 0 else None
        search = target is not None and getattr(cfg, "risk_aversion_mode", "fixed") == "target_vol"
        mode = getattr(cfg, "vol_target_mode", "projection")
        gamma = float(cfg.risk_aversion)
        floor: float | None = None
        if search and target is not None and getattr(cfg, "vol_target_is_cap", False):
            # the target is a cap: start at the base gamma and only ever raise it
            gamma = float(np.clip(gamma, *GAMMA_BRACKET))
            floor = gamma
            notes.append(f"vol target {target:.3f} is a cap: base gamma {gamma:.3f}, raised only")
        elif search and target is not None:
            g0, ir = closed_form_gamma(pdat, H, target)
            if np.isfinite(g0) and g0 > 0:
                gamma = float(np.clip(g0, *GAMMA_BRACKET))
            notes.append(f"gamma0 = IR {ir:.3f} / (2 x {target:.3f}) = {g0:.3f}")
        fixed = target is not None and not search
        return _SizingPlan(
            target=target,
            search=search,
            vol_cap=fixed and mode == "constraint",
            projection=fixed and mode == "projection",
            tgt_daily=target / np.sqrt(TRADING_DAYS) * (1.0 + 1e-7) if target is not None else None,
            gamma0=gamma,
            gamma_floor=floor,
        )

    # ------------------------------------------------------------------ step 4: alpha problem
    def _solve_alpha(
        self,
        pdat: _ProblemData,
        H: int,
        plan: _SizingPlan,
        venue: str,
        constraints_extra: Sequence[ExtraConstraint] | None,
        risk: RiskModelOutput,
        notes: list[str],
    ) -> _AlphaSolve:
        """Solve the alpha problem; with ``cfg.relax_participation_on_infeasible``
        retry once without the ADV caps.  In ``target_vol`` mode gamma is free,
        so a numerical failure of the primary solver is retried at
        ``GAMMA_NUDGE`` x gamma before the (slow) fallback solvers."""
        attempts: list[tuple[str, bool]] = [("participation", True)]
        if getattr(self.cfg, "relax_participation_on_infeasible", False):
            attempts.append(("participation_relaxed", False))
        gamma = plan.gamma0
        out: _AlphaSolve | None = None
        n_solves = 0
        for label, with_part in attempts:
            comp, reused = self._compiled("alpha", pdat, H, with_part, plan.vol_cap, venue, constraints_extra)
            comp.set_values(pdat, tgt=plan.tgt_daily, kappa=gamma * H)
            solve = self._solve_fn(comp)
            if plan.search:
                status, solver = solve(True)
                n_solves += 1
                if status not in OK_STATUSES and status not in TERMINAL_STATUSES:
                    gamma *= GAMMA_NUDGE
                    comp.set_kappa(pdat, gamma * H)
                    status, solver = solve(False)
                    n_solves += 1
            else:
                status, solver = solve(False)
                n_solves += 1
            out = _AlphaSolve(comp, status, solver, float("nan"), with_part, gamma, reused, n_solves)
            if status in OK_STATUSES:
                out.obj = _objective(comp)
                if not with_part:
                    notes.append("ADV participation bounds relaxed (infeasible with them)")
                    log.warning(
                        "optimizer: ADV participation caps dropped on %s (infeasible with them)",
                        getattr(risk, "date", "?"),
                    )
                break
            notes.append(f"{label}: {status}")
        assert out is not None
        return out

    def _hold_result(
        self, pdat: _ProblemData, s: _AlphaSolve, g: float, risk: RiskModelOutput, t0: float, notes: list[str]
    ) -> OptimizationResult:
        """Failure semantics (module docstring): hold the current book."""
        log.warning(
            "optimizer failed on %s (status=%s, solver=%s); holding current weights",
            getattr(risk, "date", "?"),
            s.status,
            s.solver or "none",
        )
        solve_time = time.perf_counter() - t0
        w0 = pdat.w0
        w_t = _full(pdat, w0)
        var_d = pdat.variance(w0)
        return OptimizationResult(
            w_target=w_t,
            x=pd.Series(0.0, index=w_t.index),
            expected_alpha=float(pdat.a @ w0),
            expected_risk=float(np.sqrt(max(var_d, 0) * TRADING_DAYS)),
            expected_cost=float("nan"),
            turnover=0.0,
            transfer_coefficient=float("nan"),
            status=s.status,
            solve_time=solve_time,
            binding_constraints={},
            objective_value=s.obj,
            solver=s.solver,
            notes=notes,
            exante_var_daily=var_d,
            n_out_of_model=_n_out(pdat, w0),
            alignment_share=pdat.alignment_share(w0),
            gp_shrink=g,
            problem_reused=s.reused,
            risk_aversion=s.gamma,
            n_solves=s.n_solves,
            n_excluded_alpha=len(pdat.excluded_alpha),
            solver_errors=list(self._solve_errors),
        )

    # ------------------------------------------------------------------ step 5: sizing
    def _apply_gamma_search(
        self,
        book: _Book,
        s: _AlphaSolve,
        pdat: _ProblemData,
        H: int,
        target: float,
        gamma_floor: float | None = None,
    ) -> _Book:
        if gamma_floor is not None:  # vol_target_is_cap: upward only from the base gamma
            cfg = self.cfg
            lever = bool(getattr(cfg, "leverage_cap_raises_gamma", False)) and cfg.book == "long_short"
            sr = gamma_cap_search(
                s.comp,
                pdat,
                H,
                s.with_part,
                target,
                book.gamma,
                book.w,
                book.binding,
                book.obj,
                self._solve_fn(s.comp),
                gross_cap=float(cfg.gross_leverage_max) if lever else None,
            )
            return self._take_search(book, sr)
        sr = gamma_search(
            s.comp,
            pdat,
            H,
            s.with_part,
            target,
            book.gamma,
            book.w,
            book.binding,
            book.obj,
            self._solve_fn(s.comp),
        )
        return self._take_search(book, sr)

    @staticmethod
    def _take_search(book: _Book, sr: GammaSearchResult) -> _Book:
        book.w, book.binding, book.obj, book.gamma = sr.w, sr.binding, sr.obj, sr.gamma
        book.status, book.vt_binding = sr.status, sr.outcome
        book.solver = sr.solver or book.solver
        book.n_solves += sr.n_extra
        book.notes.append(sr.note)
        return book

    def _vol_target_projection(
        self,
        wv: np.ndarray,
        pdat: _ProblemData,
        H: int,
        with_part: bool,
        venue: str,
        extra_fns: Sequence[ExtraConstraint] | None,
        date=None,
        kappa: float | None = None,
    ) -> tuple[np.ndarray, float, str | None, int]:
        """Re-solve ``min kappa (w - s w*)' Sigma (w - s w*) + c TC(w - w0)``
        over the full constraint set plus the risk cap ``sqrt(w' Sigma w) <=
        target`` whenever the predicted vol of ``w*`` is off target (down
        always; up only with ``vol_target_scale_up`` and only for long/short
        books, capped by gross and per-name limits).  ``kappa = gamma H``
        (default ``cfg.risk_aversion * H``): the distance is the *risk-aversion
        weighted tracking variance* to the anchor, i.e. the utility the main
        objective charges for that much risk, so it and the trade cost are both
        fractions of NAV over the horizon (second-pass finding 2; the former
        Euclidean ``||w - s w*||^2`` was in weight^2 units, giving cost ~5 % of
        the trade-off).  Returns ``(w, s, note, n_solves)``; on a failed
        projection ``w*`` is kept."""
        cfg = self.cfg
        target = cfg.target_vol_annual
        if target is None or not np.isfinite(target) or target <= 0:
            return wv, 1.0, None, 0
        pred = float(np.sqrt(max(pdat.variance(wv), 0.0) * TRADING_DAYS))
        if pred <= 0 or not np.isfinite(pred):
            return wv, 1.0, None, 0
        s = target / pred
        if abs(s - 1.0) <= 1e-4:
            return wv, 1.0, None, 0
        if s > 1.0:
            if not getattr(cfg, "vol_target_scale_up", False):
                return wv, 1.0, None, 0
            if cfg.book != "long_short":
                return wv, 1.0, f"vol-target scale-up skipped for book '{cfg.book}' (fully invested)", 0
            if float(np.abs(wv).sum()) >= cfg.gross_leverage_max * (1.0 - 1e-6):
                return wv, 1.0, "vol-target scale-up blocked: gross leverage already at cap", 0
            # s is only the projection *anchor*: names at their per-name / ADV
            # caps are clipped by the constraints while the rest of the book is
            # levered, and the risk cap keeps the result at or below target.
            s = float(min(s, MAX_SCALE_UP))
        k = float(cfg.risk_aversion * H if kappa is None else kappa)
        if not float(getattr(cfg, "vol_target_projection_cost_weight", 0.0) or 0.0) > 0:
            # no cost term: the argmin does not depend on kappa; normalise the
            # objective to O(1) so the solver's absolute tolerance does not
            # dominate a ~1e-4 tracking variance
            k = 1.0 / max(pdat.variance(s * wv), 1e-16)
        comp, _ = self._compiled("projection", pdat, H, with_part, True, venue, extra_fns)
        comp.set_values(
            pdat, anchor=s * wv, tgt=float(target) / np.sqrt(TRADING_DAYS) * (1.0 + 1e-7), kappa=max(k, 1e-12)
        )
        status, solver = self._try_solvers(comp.problem, dpp=self.reuse_problem and comp.is_dpp)
        if status not in OK_STATUSES or comp.w.value is None:
            log.warning(
                "vol-target projection failed on %s (status=%s); keeping un-scaled book", date, status
            )
            return wv, 1.0, f"vol-target projection failed ({status}); book left at s=1", 1
        w_new = clean_weights(np.asarray(comp.w.value).ravel(), pdat.w0)
        got = float(np.sqrt(max(pdat.variance(w_new), 0.0) * TRADING_DAYS))
        return (
            w_new,
            float(s),
            f"vol-target projection s={s:.3f}: vol {pred:.4f} -> {got:.4f} (target {target:.4f}, {solver})",
            1,
        )

    # ------------------------------------------------------------------ step 5b: min trade
    def _drop_small_trades(self, book: _Book, pdat: _ProblemData) -> _Book:
        """Post-solve projection of ``cfg.min_trade_weight``: a trade with ``0 < |x_i|
        < min_trade_weight`` is cancelled (the name keeps ``w0_i``) **unless** it
        closes the position (target weight 0) or the name is being forced out
        (``force_to_zero``: universe exits must proceed however small the daily
        step).  On the real run two thirds of all trades were below 5 bp of NAV:
        the solver's continuous optimum re-trims every name every day, and each of
        those orders pays the per-order minimum commission and the spread for no
        measurable change in the book.

        This is a projection, not a constraint (a hard minimum trade size makes
        the problem a mixed-integer one): the returned book is no longer the
        optimum of the solved problem and can sit marginally outside a band the
        solve honoured -- by at most the sum of the cancelled trades' exposures.
        The number of cancelled trades and the largest violation
        (:func:`~qts.portfolio.optimizer.constraints.max_constraint_violation`)
        go on the result; they are not repaired."""
        thr = float(getattr(self.cfg, "min_trade_weight", 0.0) or 0.0)
        if thr <= 0:
            return book
        w, w0 = book.w, pdat.w0
        x = w - w0
        small = (np.abs(x) > 0) & (np.abs(x) < thr)
        keep = (w == 0.0) | pdat.force_mask  # exits and forced unwinds always trade
        drop = small & ~keep
        n = int(drop.sum())
        if n:
            book.w = np.where(drop, w0, w)
        book.n_dropped = n
        book.violation, book.violated = max_constraint_violation(pdat, self.cfg, self.beta_band, book.w)
        if book.violation <= SOLVER_TOL:  # solver-tolerance slack is not a violation
            book.violation, book.violated = 0.0, ""
        book.notes.append(
            f"min trade {thr:.4%}: {n} trades dropped"
            + (f", max violation {book.violation:.2e} ({book.violated})" if book.violated else "")
        )
        return book

    # ------------------------------------------------------------------ step 6: result
    def _expected_cost(
        self, pdat: _ProblemData, wv: np.ndarray, H: int, nav: float, venue: str, notes: list[str]
    ) -> float:
        """Trade cost + ``H`` x holding cost of the final book (fraction of NAV)."""
        try:
            tc, hc = self.cost_model.cvx_terms(
                cp.Constant(wv - pdat.w0), cp.Constant(wv), nav, pdat.snap, venue
            )
            return _scalar(tc.value) + H * _scalar(hc.value)
        except Exception as e:  # pragma: no cover - diagnostic only (the book is already chosen)
            notes.append(f"expected_cost unavailable: {e}")
            return float("nan")

    def _final_result(
        self,
        pdat: _ProblemData,
        risk: RiskModelOutput,
        book: _Book,
        H: int,
        nav: float,
        venue: str,
        g: float,
        reused: bool,
        t0: float,
        notes: list[str],
    ) -> OptimizationResult:
        wv, a, tickers = book.w, pdat.a, pdat.tickers
        var_d = pdat.variance(wv)
        pred_vol = float(np.sqrt(max(var_d, 0.0) * TRADING_DAYS))
        share = pdat.alignment_share(wv)
        if pdat.y is not None and np.isfinite(share):
            notes.append(f"alignment share of ex-ante variance {share:.3f}")
        exp_cost = self._expected_cost(pdat, wv, H, nav, venue, notes)
        tc_val = (
            transfer_coefficient(pd.Series(a, index=tickers), pd.Series(wv, index=tickers), risk=risk)
            if np.any(a != 0)
            else float("nan")
        )
        w_t = _full(pdat, wv)
        x_t = w_t - _full(pdat, pdat.w0)
        return OptimizationResult(
            w_target=w_t,
            x=x_t,
            expected_alpha=float(a @ wv),
            expected_risk=pred_vol,
            expected_cost=exp_cost,
            turnover=float(np.abs(x_t.values).sum()),
            transfer_coefficient=tc_val,
            status=book.status,
            solve_time=time.perf_counter() - t0,
            binding_constraints=book.binding,
            objective_value=book.obj,
            vol_scalar=book.scalar,
            solver=book.solver,
            notes=notes,
            exante_var_daily=var_d,
            n_out_of_model=_n_out(pdat, wv),
            alignment_share=share,
            gp_shrink=g,
            problem_reused=reused,
            risk_aversion=book.gamma,
            vol_target_binding=book.vt_binding,
            n_solves=book.n_solves,
            n_excluded_alpha=len(pdat.excluded_alpha),
            n_small_trades_dropped=book.n_dropped,
            max_constraint_violation=book.violation,
            violated_constraint=book.violated,
            solver_errors=list(self._solve_errors),
        )

    # ------------------------------------------------------------------ solve
    def solve(
        self,
        alpha: pd.Series,
        risk: RiskModelOutput,
        w_current: pd.Series,
        nav: float,
        snapshot: Snapshot,
        constraints_extra: Sequence[ExtraConstraint] | None = None,
        venue: str = VENUE_ROUTED,
        neutral_styles: Sequence[str] | None = None,
        industry_columns: Sequence[str] | None = None,
        no_trade: Sequence[str] | None = None,
        benchmark_weights: pd.Series | None = None,
        horizon_days: int | None = None,
        force_to_zero: Mapping[str, float] | None = None,
        alpha_half_life_days: float | None = None,
    ) -> OptimizationResult:
        """``force_to_zero`` maps ticker -> maximum ``|w|`` allowed today (0 =
        fully out); the simulator uses it to unwind names that left the
        eligible universe over ``cfg.exit_ineligible_days``.  ``no_trade``
        names keep ``x = 0``; a name in both lists is treated as no-trade.
        ``alpha_half_life_days`` (``AlphaOutput.half_life_days``) drives the
        Garleanu-Pedersen shrink when ``cfg.use_gp_shrink`` is on.
        ``neutral_styles=None`` -> ``RISK_ONLY_STYLES`` when
        ``cfg.neutralize_risk_only_styles`` (``[]`` = no style bands).
        ``venue`` defaults to ``"routed"``, the venue the pipeline and the cost
        model price at (design decision D3).

        Only names in ``risk.tickers`` are held or traded; held names outside
        the model are exited (``force_to_zero``), other names -- even with
        alpha -- are excluded and get ``w = 0`` (reported in ``notes`` and
        ``n_excluded_alpha``).  ``w_target`` is indexed by alpha | w_current |
        risk tickers."""
        H = int(horizon_days or self.horizon_days)
        t0 = time.perf_counter()
        self._solve_errors = []
        notes: list[str] = []
        part_rate = self.participation_rate(alpha_half_life_days)
        pdat = self._prepare(
            alpha,
            risk,
            w_current,
            nav,
            snapshot,
            neutral_styles,
            industry_columns,
            no_trade,
            force_to_zero,
            benchmark_weights,
            notes,
            part_rate=part_rate,
        )
        if getattr(self.cfg, "participation_from_urgency", False):
            notes.append(
                f"ADV participation cap {part_rate:.2%} from urgency (half-life {alpha_half_life_days})"
            )
        g = self._gp_factor(pdat, H, alpha_half_life_days, venue, notes)
        if g != 1.0:
            pdat.a = pdat.a * g
        plan = self._sizing_plan(pdat, H, notes)
        s = self._solve_alpha(pdat, H, plan, venue, constraints_extra, risk, notes)
        comp = s.comp
        if s.status not in OK_STATUSES or comp.w.value is None:
            return self._hold_result(pdat, s, g, risk, t0, notes)

        wv = clean_weights(np.asarray(comp.w.value).ravel(), pdat.w0)
        # binding constraints via duals (read before any re-solve overwrites them)
        binding = comp.binding(pdat, s.with_part)
        if plan.vol_cap and comp.risk_cap is not None and n_binding(comp.risk_cap):
            binding["vol_cap"] = 1
            notes.append(f"vol cap binding (target {plan.target:.4f})")
        book = _Book(wv, binding, s.obj, s.gamma, s.status, s.solver, s.n_solves, notes=notes)
        if plan.search:
            assert plan.target is not None
            book = self._apply_gamma_search(book, s, pdat, H, plan.target, plan.gamma_floor)
        elif plan.projection:  # vol targeting by projection onto the same feasible set (fixed gamma)
            book.w, book.scalar, proj_note, k = self._vol_target_projection(
                wv,
                pdat,
                H,
                s.with_part,
                venue,
                constraints_extra,
                date=getattr(risk, "date", "?"),
                kappa=book.gamma * H,
            )
            book.n_solves += k
            if proj_note:
                notes.append(proj_note)
        book = self._drop_small_trades(book, pdat)
        return self._final_result(pdat, risk, book, H, nav, venue, g, s.reused, t0, notes)


# --------------------------------------------------------------------------- helpers
def _scalar(v: Any) -> float:
    """A cvxpy expression value (scalar or 1-element array) as float."""
    return float(np.asarray(v).ravel()[0] if np.ndim(v) else v)


def _full(pdat: _ProblemData, v: np.ndarray) -> pd.Series:
    """Problem-space vector -> Series on the full result index (0 for excluded names)."""
    s = pd.Series(v, index=pdat.tickers, dtype=float)
    return s if pdat.full_tickers is None else s.reindex(pdat.full_tickers).fillna(0.0)


def _n_out(pdat: _ProblemData, wv: np.ndarray) -> int:
    """Names outside the risk model with a non-zero weight in ``wv`` (after the solve)."""
    if pdat.in_model is None:
        return 0
    return int(((~pdat.in_model) & (np.abs(wv) > 0)).sum())
