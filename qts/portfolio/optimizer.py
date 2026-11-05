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
PSD by eigenvalue clipping and entered as ``||L' B' w||^2`` with ``F = L L'``.

Constraint menu (driven by ``PortfolioConfig``): book type, gross/net,
per-name cap, ADV participation, beta neutrality, industry bands, style
bands, no-trade list, forced exits, extra user constraints.

Factor roles come from ``RiskModelOutput.factor_groups`` (``"style"`` /
``"industry"``), never from column-name prefixes: beta neutrality binds the
style factor ``cfg.beta_factor_name`` (default ``"beta"``) inside
``+-cfg.beta_band`` -- *not* the unit-exposure market/country factor -- and
industry bands cover every column in ``factor_groups["industry"]`` (so
Fama-French labels such as ``Banks`` work).  Prefix detection is used only
when ``factor_groups`` is empty, with a logged warning.  With
``neutral_styles=None`` and ``cfg.neutralize_risk_only_styles`` the style
bands cover ``qts.core.roles.RISK_ONLY_STYLES`` (design decision D1: the label
is residualised on exactly this set); pass ``neutral_styles=[]`` for none.

DPP parametrisation (performance)
---------------------------------
The cvxpy problem is built **once** per (ticker set, constraint structure,
cost-expression structure) with ``cp.Parameter`` for everything that changes
day to day -- alpha, ``w_current``, the factor loadings ``L'B'`` (a
``K x N`` parameter matrix), specific sd, the alignment vector, ADV caps,
no-trade / forced-exit caps, band exposures and every array coefficient of
the cost model's convex terms -- and re-solved with new parameter values
while the structure is unchanged, so cvxpy canonicalises once (Disciplined
Parametrised Programming).  The cost model is untouched: its
``cvx_terms(x, w_post, nav, snapshot, venue)`` is called every day with the
problem's own variables and its array-valued ``Constant`` leaves are mapped,
in tree order, onto the cached problem's parameters (a changed expression
tree, scalar constant or sign triggers a rebuild).  ``x`` is a variable tied
by ``x == w - w_current`` so cost terms stay DPP.  ``cfg.reuse_problem=False``
rebuilds every solve (same formulation).

Vol targeting (``cfg.vol_target_mode``)
---------------------------------------
* ``"projection"``: when the predicted vol of the optimum ``w*`` differs from
  the target the book is re-solved as ``minimize ||w - s w*||^2 +
  c * TC(w - w0)`` (``c = cfg.vol_target_projection_cost_weight``; finding
  #28: the old projection ignored the cost of the extra trades) subject to the
  identical constraint set plus ``sqrt(w' Sigma w) <= target``, so every band
  the first solve honoured still holds after scaling (``s = target /
  predicted``, capped by the gross and per-name limits when levering up).
* ``"constraint"``: one solve -- the SOC risk cap ``sqrt(w' Sigma w) <=
  target`` is added to the alpha problem, so alpha, risk penalty, costs and
  the cap are traded off jointly.  The target is a ceiling (no scale-up).

Failure semantics: a solve that is not ``optimal`` / ``optimal_inaccurate``
returns ``status`` as reported by the solver, ``w_target = w_current``,
``expected_cost = nan`` and logs a warning.  Dropping the ADV participation
caps on infeasibility is opt-in (``cfg.relax_participation_on_infeasible``).
"""
from __future__ import annotations

import logging
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd

from qts.core.config import PortfolioConfig
from qts.core.roles import RISK_ONLY_STYLES
from qts.core.types import RiskModelOutput
from qts.portfolio.alpha_scaling import cost_implied_holding_days, gp_sleeve_shrink_factors
from qts.portfolio.costs import CostTerms, Snapshot, align_snapshot
from qts.portfolio.diagnostics import transfer_coefficient

log = logging.getLogger(__name__)

OK_STATUSES = ("optimal", "optimal_inaccurate")
ExtraConstraint = Callable[[cp.Variable, cp.Expression, pd.Index], list[cp.Constraint]]
SOLVER_TOL = 1e-6
MAX_SCALE_UP = 5.0   # sanity cap on the vol-target projection anchor s = target / predicted
_BIG = 10.0          # "no cap" value for the per-name no-trade / forced-exit cap parameters
VOL_TARGET_MODES = ("projection", "constraint")


@dataclass
class OptimizationResult:
    w_target: pd.Series
    x: pd.Series
    expected_alpha: float          # alpha' w_target over the horizon (after any GP shrink)
    expected_risk: float           # annualised ex-ante vol of w_target
    expected_cost: float           # trade cost + H * holding cost (fraction of NAV); nan on failure
    turnover: float                # sum |x| (two-sided, fraction of NAV)
    transfer_coefficient: float
    status: str
    solve_time: float
    binding_constraints: dict[str, int] = field(default_factory=dict)
    objective_value: float = float("nan")
    vol_scalar: float = 1.0        # s used in the vol-target projection (1 = none / constraint mode)
    solver: str = ""
    notes: list[str] = field(default_factory=list)
    exante_var_daily: float = float("nan")
    n_out_of_model: int = 0        # material names (alpha or position != 0) outside the risk universe
    alignment_share: float = float("nan")  # share of ex-ante variance from the alignment term
    gp_shrink: float = 1.0         # Garleanu-Pedersen factor applied to alpha (1 = none)
    problem_reused: bool = False   # True when the cached DPP problem was re-solved

    @property
    def ok(self) -> bool:
        return self.status in OK_STATUSES


def _psd_factor(F: np.ndarray, floor: float = 0.0) -> np.ndarray:
    """Return L with L L' = PSD-clipped symmetric part of F."""
    Fs = 0.5 * (F + F.T)
    vals, vecs = np.linalg.eigh(Fs)
    vals = np.clip(vals, floor, None)
    return vecs * np.sqrt(vals)[None, :]


def _snap_get(snapshot: Any, key: str):
    """Read a per-ticker field from a dict snapshot or an attribute-style snapshot object."""
    try:
        return snapshot[key]
    except (TypeError, KeyError):
        return getattr(snapshot, key)


@dataclass
class _ProblemData:
    """Everything the constraint builder and the solves share (numeric, one date)."""
    tickers: pd.Index
    a: np.ndarray
    w0: np.ndarray
    snap: Snapshot
    factors: list[str]
    Bv: np.ndarray
    F: np.ndarray
    L: np.ndarray
    d: np.ndarray
    sd: np.ndarray
    y: np.ndarray | None
    nu: float
    nav: float
    beta_col: str | None
    ind_cols: list[str]
    sty_cols: list[str]
    wb: np.ndarray | None
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

    # factor layout for the parametrised problem ------------------------------
    static_cols: list[str] = field(default_factory=list)   # market + industry dummies (constant, sparse)
    dynamic_cols: list[str] = field(default_factory=list)  # styles / statistical (parameter, dense)

    def ordered(self) -> list[str]:
        return list(self.static_cols) + list(self.dynamic_cols)

    def static_block(self) -> np.ndarray:
        return self.cols(self.static_cols)

    def L_ordered(self) -> np.ndarray:
        """``L`` with ``L L' = F`` in the ``[static | dynamic]`` factor order."""
        idx = [self.factors.index(c) for c in self.ordered()]
        return _psd_factor(self.F[np.ix_(idx, idx)])

    def y_scaled(self) -> np.ndarray:
        n = len(self.tickers)
        return np.zeros(n) if self.y is None else np.sqrt(self.nu) * self.y

    def cols(self, names: Sequence[str]) -> np.ndarray:
        if not names:
            return np.zeros((len(self.tickers), 0))
        return self.Bv[:, [self.factors.index(c) for c in names]]

    def offsets(self, names: Sequence[str]) -> np.ndarray:
        if self.wb is None or not names:
            return np.zeros(len(names))
        return self.cols(names).T @ self.wb


# --------------------------------------------------------------------------- DPP helpers
def _leaf_sig(e) -> tuple:
    """Structure signature of a cvxpy expression tree: node types, shapes and
    atom data; array constants contribute shape + sign only (they become
    parameters), scalar constants their value."""
    if isinstance(e, cp.Constant):
        v = np.asarray(e.value, dtype=float)
        if v.size > 1:
            return ("C", v.shape)
        return ("c", float(v.reshape(-1)[0]) if v.size else 0.0)
    if isinstance(e, (cp.Variable, cp.Parameter)):
        return (type(e).__name__, e.id)
    data = e.get_data() if hasattr(e, "get_data") else None
    return (type(e).__name__, e.shape, repr(data), tuple(_leaf_sig(a) for a in e.args))


def _constants(e, out: list[np.ndarray]) -> list[np.ndarray]:
    """Array-valued constant leaves of ``e`` in tree order."""
    if isinstance(e, cp.Constant):
        v = np.asarray(e.value, dtype=float)
        if v.size > 1:
            out.append(v)
        return out
    for a in e.args:
        _constants(a, out)
    return out


def _parametrize(e, params: list[cp.Parameter], exact_power: bool = True):
    """Copy of ``e`` with each array constant replaced by a Parameter carrying
    the constant's value and sign (appended to ``params`` in tree order).
    With ``exact_power`` every ``power(., p)`` rational SOC approximation
    (cvxpy's default ``PowerApprox``) is replaced by the exact power-cone atom
    (``approx=False``; Clarabel has native 3-d power cones): one cone per name
    instead of a ~10-cone SOC tower -- a smaller, more accurate problem."""
    if isinstance(e, cp.Constant):
        v = np.asarray(e.value, dtype=float)
        if v.size <= 1:
            return e
        kw = {"nonneg": True} if np.all(v >= 0) else ({"nonpos": True} if np.all(v <= 0) else {})
        p = cp.Parameter(v.shape, **kw)
        p.value = v
        params.append(p)
        return p
    if not e.args:
        return e
    args = [_parametrize(a, params, exact_power) for a in e.args]
    if exact_power and type(e).__name__ == "PowerApprox":
        return cp.power(args[0], float(e.get_data()[0]), approx=False)
    return e.copy(args)


def _assign(params: Sequence[cp.Parameter], values: Sequence[np.ndarray]) -> bool:
    """Set parameter values; False if a value violates a parameter's sign."""
    if len(params) != len(values):
        return False
    for p, v in zip(params, values, strict=True):
        if p.shape != v.shape:
            return False
        if p.is_nonneg() and np.any(v < 0):
            return False
        if p.is_nonpos() and np.any(v > 0):
            return False
    for p, v in zip(params, values, strict=True):
        p.value = v
    return True


class _Compiled:
    """One parametrised cvxpy problem (alpha solve or vol-target projection)."""

    def __init__(self, kind: str, key: tuple, n: int, K: int, layout: dict[str, Any],
                 book: str, cfg: PortfolioConfig, beta_band: float, coefs: dict[str, float],
                 with_part: bool, vol_cap: bool, cost_model: CostTerms, cost_args: tuple,
                 extra_fns: Sequence[ExtraConstraint] | None, tickers: pd.Index, exact_power: bool = True):
        self.kind, self.key = kind, key
        n_ind, n_sty, has_beta = len(layout["ind"]), len(layout["sty"]), layout["beta"] is not None
        w = cp.Variable(n, name="w")
        x = cp.Variable(n, name="x")
        f = cp.Variable(K, name="f")                         # factor exposures B'w
        self.w, self.x, self.f = w, x, f
        Ks, Kd = len(layout["static"]), len(layout["dynamic"])
        P = {
            "a": cp.Parameter(n, name="alpha"), "w0": cp.Parameter(n, name="w0"),
            "L": cp.Parameter((K, K), name="L"), "sd": cp.Parameter(n, nonneg=True, name="sd"),
            "y": cp.Parameter(n, name="y_sqrt_nu"), "part": cp.Parameter(n, nonneg=True, name="part_cap"),
            "nt": cp.Parameter(n, nonneg=True, name="no_trade_cap"),
            "fz": cp.Parameter(n, nonneg=True, name="force_cap"),
            "tgt": cp.Parameter(nonneg=True, name="vol_cap_daily"),
            "anchor": cp.Parameter(n, name="anchor"),
        }
        if Kd:
            P["Bd"] = cp.Parameter((n, Kd), name="B_dynamic")
        if n_ind:
            P["oi"] = cp.Parameter(n_ind, name="ind_off")
        if n_sty:
            P["os"] = cp.Parameter(n_sty, name="sty_off")
        self.P = P
        cons: dict[str, cp.Constraint] = {"_link": x == w - P["w0"]}
        # f = B'w: the market/industry block is a constant sparse matrix (its
        # values are part of the structure key), the dense style block a parameter
        if Ks:
            import scipy.sparse as sp
            cons["_f_static"] = f[:Ks] == sp.csc_matrix(layout["static_block"]).T @ w
        if Kd:
            cons["_f_dynamic"] = f[Ks:] == P["Bd"].T @ w
        pos = {c: i for i, c in enumerate(layout["static"] + layout["dynamic"])}
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
        if has_beta:
            cons["beta_neutral"] = cp.abs(f[pos[layout["beta"]]]) <= beta_band
        if n_ind:
            cons["industry_band"] = cp.abs(f[[pos[c] for c in layout["ind"]]] - P["oi"]) <= cfg.industry_band
        if n_sty:
            cons["style_band"] = cp.abs(f[[pos[c] for c in layout["sty"]]] - P["os"]) <= cfg.style_band_sd
        if layout.get("has_no_trade", True):
            cons["no_trade"] = cp.abs(x) <= P["nt"]
        if layout.get("has_force", True):
            cons["force_to_zero"] = cp.abs(w) <= P["fz"]
        self.participation = cp.abs(x) <= P["part"]
        self.cons = cons
        extra: list[cp.Constraint] = []
        for fn in (extra_fns or []):
            extra.extend(fn(w, x, tickers))
        clist = list(cons.values()) + extra + ([self.participation] if with_part else [])

        # cost terms: the cost model's own expressions, constants -> parameters
        tc, hc = cost_model.cvx_terms(x, w, *cost_args)
        self.cost_params: list[cp.Parameter] = []
        self.cost_sig = (_leaf_sig(tc), _leaf_sig(hc))
        tc_p = _parametrize(tc, self.cost_params, exact_power)
        hc_p = _parametrize(hc, self.cost_params, exact_power)
        self.risk_cap = None
        if kind == "alpha":
            risk = cp.sum_squares(P["L"].T @ f) + cp.sum_squares(cp.multiply(P["sd"], w)) + cp.square(P["y"] @ w)
            obj = P["a"] @ w - coefs["kappa"] * risk - coefs["trade"] * tc_p - coefs["hold"] * hc_p
            if coefs["turnover"] > 0:
                obj = obj - coefs["turnover"] * cp.norm1(x)
            if vol_cap:
                self.risk_cap = self._risk_norm(w) <= P["tgt"]
                clist.append(self.risk_cap)
            self.problem = cp.Problem(cp.Maximize(obj), clist)
        else:  # projection
            obj = cp.sum_squares(w - P["anchor"])
            if coefs["proj_cost"] > 0:
                obj = obj + coefs["proj_cost"] * tc_p
            self.risk_cap = self._risk_norm(w) <= P["tgt"]
            clist.append(self.risk_cap)
            self.problem = cp.Problem(cp.Minimize(obj), clist)
        self.trade_cost_expr = tc_p
        self.is_dpp = self.problem.is_dpp()
        self.n_solves = 0

    def _risk_norm(self, w: cp.Variable) -> cp.Expression:
        """``sqrt(w' Sigma w)`` (daily) as a second-order-cone expression."""
        P = self.P
        return cp.norm(cp.hstack([P["L"].T @ self.f, cp.multiply(P["sd"], w),
                                  cp.reshape(P["y"] @ w, (1,), order="C")]), 2)

    def set_values(self, pdat: _ProblemData, anchor: np.ndarray | None = None,
                   tgt: float | None = None) -> None:
        P = self.P
        n = len(pdat.tickers)
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
        P["anchor"].value = anchor if anchor is not None else np.zeros(n)


class SinglePeriodOptimizer:
    def __init__(self, cfg: PortfolioConfig, cost_model: CostTerms, horizon_days: int = 20,
                 beta_band: float | None = None, solver_opts: dict | None = None, verbose: bool = False,
                 reuse_problem: bool | None = None, exact_power: bool = True):
        """``reuse_problem`` overrides ``cfg.reuse_problem`` (cache the DPP problem
        across solves while the ticker set / constraint structure is unchanged).
        ``exact_power`` enters the cost model's 3/2-power impact terms as exact
        power cones instead of cvxpy's SOC approximation (see :func:`_parametrize`)."""
        self.cfg = cfg
        self.cost_model = cost_model
        self.horizon_days = int(horizon_days)
        self.beta_band = float(getattr(cfg, "beta_band", 0.05) if beta_band is None else beta_band)
        self.solver_opts = solver_opts or {}
        self.verbose = verbose
        self.reuse_problem = bool(getattr(cfg, "reuse_problem", False) if reuse_problem is None else reuse_problem)
        self.exact_power = bool(exact_power)
        self._chain: list[str] | None = None
        mode = getattr(cfg, "vol_target_mode", "projection")
        if mode not in VOL_TARGET_MODES:
            raise ValueError(f"vol_target_mode must be one of {VOL_TARGET_MODES}, got {mode!r}")
        self._cache: dict[tuple[str, bool], _Compiled] = {}
        self.n_builds = 0
        self.n_reuses = 0

    # ------------------------------------------------------------------ utils
    def _solver_chain(self) -> list[str]:
        if self._chain is None:            # cp.installed_solvers() costs ~10 ms per call
            installed = set(cp.installed_solvers())
            out: list[str] = []
            for s in [self.cfg.solver, "CLARABEL", "SCS", "OSQP"]:
                if s and s.upper() not in out and s.upper() in installed:
                    out.append(s.upper())
            self._chain = out
        return self._chain

    # ------------------------------------------------------------ factor roles
    def _resolve_factor_roles(self, risk: RiskModelOutput, factors: list[str],
                              neutral_styles: Sequence[str] | None,
                              industry_columns: Sequence[str] | None,
                              notes: list[str], has_benchmark: bool = False
                              ) -> tuple[str | None, list[str], list[str]]:
        """``(beta_col, industry_cols, style_band_cols)`` from
        ``risk.factor_groups``; prefix fallback only when the groups are empty.
        ``neutral_styles=None`` with ``cfg.neutralize_risk_only_styles`` means
        ``RISK_ONLY_STYLES`` (those present in the model; no error for absent
        ones) for long/short books or benchmark-relative solves; an explicit
        list must name existing style factors."""
        cfg = self.cfg
        groups = getattr(risk, "factor_groups", None) or {}
        has_groups = any(len(v) for v in groups.values()) if isinstance(groups, dict) else False
        beta_name = str(getattr(cfg, "beta_factor_name", "beta"))
        # D1 default: only where neutrality is meaningful -- a dollar-neutral book, or
        # active weights against a benchmark (a fully invested long-only book cannot be
        # style-neutral in absolute terms)
        defaulted = (neutral_styles is None and bool(getattr(cfg, "neutralize_risk_only_styles", False))
                     and (cfg.book == "long_short" or has_benchmark))
        if has_groups:
            style = [c for c in groups.get("style", []) if c in factors]
            industry = [c for c in groups.get("industry", []) if c in factors]
            if defaulted:
                neutral_styles = [s for s in RISK_ONLY_STYLES if s in style]
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
            if defaulted:
                neutral_styles = [s for s in RISK_ONLY_STYLES if s in factors]
        if cfg.industry_band is not None and cfg.industry_band > 0 and not ind_cols:
            notes.append("industry_band > 0 but no industry factors identified; band not applied")
        if not (cfg.industry_band is not None and cfg.industry_band > 0):
            ind_cols = []
        # beta already has its own (tighter) band when beta_neutral is on
        sty = [c for c in (neutral_styles or []) if c in factors and not (cfg.beta_neutral and c == beta_col)]
        if not (cfg.style_band_sd is not None and cfg.style_band_sd > 0):
            sty = []
        if not cfg.beta_neutral:
            beta_col_used = None
        else:
            beta_col_used = beta_col
        return beta_col_used, ind_cols, sty

    # ------------------------------------------------------------ problem data
    def _prepare(self, alpha: pd.Series, risk: RiskModelOutput, w_current: pd.Series, nav: float,
                 snapshot: Snapshot, neutral_styles, industry_columns, no_trade, force_to_zero,
                 benchmark_weights, notes: list[str]) -> _ProblemData:
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

        beta_col, ind_cols, sty = self._resolve_factor_roles(risk, factors, neutral_styles, industry_columns, notes,
                                                             has_benchmark=benchmark_weights is not None)
        wb = None
        if benchmark_weights is not None:
            wb = pd.Series(benchmark_weights, dtype=float).reindex(tickers).fillna(0.0).values

        adv = np.asarray(pd.Series(_snap_get(snap, "adv_usd")).reindex(tickers).values, dtype=float)
        part_cap = cfg.max_adv_participation * adv / max(nav, 1.0)
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
        static_cols = self._static_columns(risk, factors, Bv)
        dynamic_cols = [c for c in factors if c not in set(static_cols)]
        return _ProblemData(tickers=tickers, a=a, w0=w0, snap=snap, factors=factors, Bv=Bv, F=F, L=L,
                            d=d, sd=sd, y=y, nu=nu, nav=float(nav), beta_col=beta_col, ind_cols=ind_cols,
                            sty_cols=sty, wb=wb, no_trade_mask=no_trade_mask, force_mask=force_mask,
                            force_caps=force_caps, part_cap=part_cap, n_out_of_model=n_out,
                            static_cols=static_cols, dynamic_cols=dynamic_cols)

    @staticmethod
    def _static_columns(risk: RiskModelOutput, factors: list[str], Bv: np.ndarray) -> list[str]:
        """Factor columns entered as a *constant sparse* block of the cached
        problem: the market and industry groups (0/1 dummies that change only on
        reclassification / universe change, which then triggers a rebuild).
        Without ``factor_groups``: any column whose values are all 0 or 1."""
        groups = getattr(risk, "factor_groups", None) or {}
        named = set(groups.get("market", [])) | set(groups.get("industry", [])) if isinstance(groups, dict) else set()
        out = []
        for j, c in enumerate(factors):
            col = Bv[:, j]
            dummy = bool(np.all((col == 0.0) | (col == 1.0)))
            if (c in named and dummy) or (not named and dummy):
                out.append(c)
        return out

    # ------------------------------------------------------- compiled problems
    def _structure_key(self, kind: str, pdat: _ProblemData, H: int, with_part: bool, vol_cap: bool,
                       venue: str) -> tuple:
        cfg = self.cfg
        cfg_sig = (cfg.book, cfg.gross_leverage_max, cfg.net_exposure_target, cfg.net_exposure_band,
                   cfg.max_position_weight, cfg.industry_band, cfg.style_band_sd, cfg.risk_aversion,
                   cfg.trade_aversion, cfg.hold_aversion, cfg.turnover_penalty, self.beta_band,
                   getattr(cfg, "vol_target_projection_cost_weight", 0.0))
        static = np.ascontiguousarray(pdat.static_block())
        return (kind, with_part, vol_cap, H, venue, len(pdat.tickers), hash(tuple(pdat.tickers)),
                tuple(pdat.static_cols), tuple(pdat.dynamic_cols), hash(static.tobytes()), pdat.beta_col,
                tuple(pdat.ind_cols), tuple(pdat.sty_cols), bool(pdat.no_trade_mask.any()),
                bool(pdat.force_mask.any()), cfg_sig)

    def _compiled(self, kind: str, pdat: _ProblemData, H: int, with_part: bool, vol_cap: bool, venue: str,
                  extra_fns: Sequence[ExtraConstraint] | None) -> tuple[_Compiled, bool]:
        """The cached problem for this structure with today's values set, or a
        fresh one.  Returns ``(compiled, reused)``."""
        cfg = self.cfg
        key = self._structure_key(kind, pdat, H, with_part, vol_cap, venue)
        cost_args = (pdat.nav, pdat.snap, venue)
        slot = (kind, with_part)
        cached = self._cache.get(slot) if (self.reuse_problem and not extra_fns) else None
        if cached is not None and cached.key == key and cached.is_dpp:
            tc, hc = self.cost_model.cvx_terms(cached.x, cached.w, *cost_args)
            if (_leaf_sig(tc), _leaf_sig(hc)) == cached.cost_sig and \
                    _assign(cached.cost_params, _constants(tc, []) + _constants(hc, [])):
                self.n_reuses += 1
                return cached, True
        coefs = {"kappa": cfg.risk_aversion * H, "trade": cfg.trade_aversion, "hold": cfg.hold_aversion * H,
                 "turnover": cfg.turnover_penalty,
                 "proj_cost": float(getattr(cfg, "vol_target_projection_cost_weight", 0.0) or 0.0)}
        layout = {"static": list(pdat.static_cols), "dynamic": list(pdat.dynamic_cols),
                  "static_block": pdat.static_block(), "beta": pdat.beta_col,
                  "ind": list(pdat.ind_cols), "sty": list(pdat.sty_cols),
                  "has_no_trade": bool(pdat.no_trade_mask.any()), "has_force": bool(pdat.force_mask.any())}
        comp = _Compiled(kind, key, len(pdat.tickers), len(pdat.factors), layout, cfg.book, cfg, self.beta_band,
                         coefs, with_part, vol_cap, self.cost_model, cost_args, extra_fns, pdat.tickers,
                         exact_power=self.exact_power)
        self.n_builds += 1
        if not comp.is_dpp:
            log.debug("optimizer problem is not DPP (cost expression structure); it will be rebuilt per solve")
        if self.reuse_problem and not extra_fns:
            self._cache[slot] = comp
        return comp, False

    # ------------------------------------------------------------------ GP
    def _gp_factor(self, pdat: _ProblemData, H: int, half_life: float | None, venue: str,
                   notes: list[str]) -> float:
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
        w_dir = a / gross                                   # unit gross, alpha-proportional
        alpha_per_day = float(a @ w_dir) / max(H, 1)
        try:
            tc, _ = self.cost_model.cvx_terms(cp.Constant(w_dir), cp.Constant(w_dir), pdat.nav, pdat.snap, venue)
            one_way = float(np.asarray(tc.value).ravel()[0] if np.ndim(tc.value) else tc.value)
        except Exception as e:  # pragma: no cover - diagnostic only
            notes.append(f"GP shrink skipped: cost unavailable ({e})")
            return 1.0
        hold_days = cost_implied_holding_days(alpha_per_day, 2.0 * one_way)
        g = float(gp_sleeve_shrink_factors({"alpha": half_life}, hold_days)["alpha"])
        notes.append(f"GP shrink g={g:.3f} (holding {hold_days:.1f}d vs half-life {half_life:.1f}d)")
        return g

    # ------------------------------------------------------------------ solve
    def solve(
        self,
        alpha: pd.Series,
        risk: RiskModelOutput,
        w_current: pd.Series,
        nav: float,
        snapshot: Snapshot,
        constraints_extra: Sequence[ExtraConstraint] | None = None,
        venue: str = "auction",
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
        ``cfg.neutralize_risk_only_styles`` (``[]`` = no style bands)."""
        cfg = self.cfg
        H = int(horizon_days or self.horizon_days)
        t0 = time.perf_counter()
        notes: list[str] = []
        pdat = self._prepare(alpha, risk, w_current, nav, snapshot, neutral_styles, industry_columns,
                             no_trade, force_to_zero, benchmark_weights, notes)
        tickers, w0 = pdat.tickers, pdat.w0
        g = self._gp_factor(pdat, H, alpha_half_life_days, venue, notes)
        if g != 1.0:
            pdat.a = pdat.a * g
        a = pdat.a

        target = cfg.target_vol_annual
        has_target = target is not None and np.isfinite(target) and target > 0
        mode = getattr(cfg, "vol_target_mode", "projection")
        vol_cap = bool(has_target and mode == "constraint")
        tgt_daily = float(target) / np.sqrt(252.0) * (1.0 + 1e-7) if has_target else None

        # ---- solve (participation relaxation is opt-in)
        status, solver_used, obj_val, with_part = "failed", "", float("nan"), True
        attempts: list[tuple[str, bool]] = [("participation", True)]
        if getattr(cfg, "relax_participation_on_infeasible", False):
            attempts.append(("participation_relaxed", False))
        comp: _Compiled | None = None
        reused = False
        for label, with_part in attempts:
            comp, reused = self._compiled("alpha", pdat, H, with_part, vol_cap, venue, constraints_extra)
            comp.set_values(pdat, tgt=tgt_daily)
            status, solver_used = self._try_solvers(comp.problem, dpp=self.reuse_problem and comp.is_dpp)
            if status in OK_STATUSES:
                obj_val = float(comp.problem.value)
                if not with_part:
                    notes.append("ADV participation bounds relaxed (infeasible with them)")
                    log.warning("optimizer: ADV participation caps dropped on %s (infeasible with them)",
                                getattr(risk, "date", "?"))
                break
            notes.append(f"{label}: {status}")

        assert comp is not None
        w = comp.w
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
                alignment_share=pdat.alignment_share(w0), gp_shrink=g, problem_reused=reused,
            )

        wv = self._clean(np.asarray(w.value).ravel(), w0)

        # ---- binding constraints via duals (read before any re-solve overwrites them)
        binding = self._binding(comp, pdat, with_part)
        if vol_cap and comp.risk_cap is not None and self._n_binding(comp.risk_cap):
            binding["vol_cap"] = 1
            notes.append(f"vol cap binding (target {target:.4f})")

        # ---- vol targeting by projection onto the same feasible set
        scalar = 1.0
        if has_target and mode == "projection":
            wv, scalar, proj_note = self._vol_target_projection(wv, pdat, H, with_part, venue, constraints_extra,
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
            tc, hc = self.cost_model.cvx_terms(xc, wc, nav, pdat.snap, venue)
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
            n_out_of_model=pdat.n_out_of_model, alignment_share=share, gp_shrink=g, problem_reused=reused,
        )

    def _binding(self, comp: _Compiled, pdat: _ProblemData, with_part: bool) -> dict[str, int]:
        binding: dict[str, int] = {}
        for name, c in comp.cons.items():
            if name.startswith("_"):          # internal definitional equalities (x, f)
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
            cnt = self._n_binding(c, mask)
            if cnt:
                binding[name] = cnt
        if with_part:
            cnt = self._n_binding(comp.participation)
            if cnt:
                binding["participation"] = cnt
        return binding

    # ------------------------------------------------------------ vol target
    def _vol_target_projection(self, wv: np.ndarray, pdat: _ProblemData, H: int, with_part: bool, venue: str,
                               extra_fns: Sequence[ExtraConstraint] | None, date=None
                               ) -> tuple[np.ndarray, float, str | None]:
        """Re-solve ``min ||w - s w*||^2 + c TC(w - w0)`` over the full
        constraint set plus the risk cap ``sqrt(w' Sigma w) <= target`` whenever
        the predicted vol of ``w*`` is off target (down always; up only with
        ``vol_target_scale_up`` and only for long/short books, capped by gross
        and per-name limits).  Returns ``(w, s, note)``; on a failed projection
        ``w*`` is kept."""
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
        comp, _ = self._compiled("projection", pdat, H, with_part, True, venue, extra_fns)
        comp.set_values(pdat, anchor=s * wv, tgt=float(target) / np.sqrt(252.0) * (1.0 + 1e-7))
        status, solver = self._try_solvers(comp.problem, dpp=self.reuse_problem and comp.is_dpp)
        if status not in OK_STATUSES or comp.w.value is None:
            log.warning("vol-target projection failed on %s (status=%s); keeping un-scaled book",
                        date, status)
            return wv, 1.0, f"vol-target projection failed ({status}); book left at s=1"
        w_new = self._clean(np.asarray(comp.w.value).ravel(), pdat.w0)
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
    def _n_binding(c: cp.Constraint, mask: np.ndarray | None = None) -> int:
        dv = c.dual_value
        if dv is None:
            return 0
        dv = np.atleast_1d(np.asarray(dv, dtype=float))
        if mask is not None and dv.shape == mask.shape:
            dv = dv[mask]
        return int((np.abs(dv) > SOLVER_TOL).sum())

    def _try_solvers(self, prob: cp.Problem, dpp: bool = False):
        """Solve with the solver chain.  ``dpp=False`` canonicalises with the
        parameters' current values as constants (``ignore_dpp``): for a one-off
        solve that is far cheaper than a DPP compile."""
        last = "failed"
        kw = dict(self.solver_opts)
        if not dpp:
            kw["ignore_dpp"] = True
        for s in self._solver_chain():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    prob.solve(solver=s, verbose=self.verbose, **kw)
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
