"""Single-period convex portfolio optimizer (Boyd et al. SPO) on a factor
risk model.

Objective (all terms in *horizon* units, H = alpha horizon in days)::

    maximize  alpha' w_post
              - gamma * H * ( f' F f + sum_i d_i w_i^2 + nu (y' w)^2 )
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
tree, scalar constant or sign triggers a rebuild).  The routed venue's kink
terms ``C * pos(A - K)`` are rewritten as ``pos(C * A - C K)`` (``C >= 0``) so
they stay DPP (:class:`_Parametrizer`, second-pass finding 6).  The risk
aversion is a parameter value too (``sqrt(gamma H)`` scales ``L``, ``sd``,
``y``), so the gamma search re-solves one compiled problem.  ``x`` is a variable tied
by ``x == w - w_current`` so cost terms stay DPP.  ``cfg.reuse_problem=False``
rebuilds every solve (same formulation).

Sizing: risk aversion and the vol target (``cfg.risk_aversion_mode``)
--------------------------------------------------------------------
``gamma`` is an *annual* risk aversion: the objective equals ``(H/252)
[alpha_ann'w - gamma w'Sigma_ann w] - TC - ...``.  For the unconstrained,
cost-free problem ``w* = Sigma^-1 alpha_H / (2 gamma H)``, so
``sigma_ann(w*) = IR_ann / (2 gamma)`` with ``IR_ann = sqrt(252)/H *
sqrt(alpha_H' Sigma_d^-1 alpha_H)`` (Grinold & Kahn ch. 14).

* ``"target_vol"`` (default; second-pass audit finding 1): per solve, gamma
  is chosen so that the optimal book of the *alpha-aware* objective has
  ex-ante annual vol ``cfg.target_vol_annual`` subject to every constraint.
  Start at the closed form ``gamma0 = IR_ann / (2 sigma_target)``
  (:meth:`SinglePeriodOptimizer.closed_form_gamma`, ``Sigma^-1`` in factor
  form), then a safeguarded root search on ``log gamma`` in ``GAMMA_BRACKET``
  = [0.05, 200] (Newton with the measured vol elasticity, then log-log secant /
  bisection once bracketed; at most ``GAMMA_MAX_SOLVES`` = 8 solves of the same
  compiled problem, re-parametrised).  Stops at ``|vol/target - 1| < 2 %``,
  or when the gross / position / ADV caps bind below target: the capped book
  is accepted and ``OptimizationResult.vol_target_binding = "caps"``.  The
  book is on the cost-aware efficient frontier, so costs, the GP alpha shrink
  and the risk penalty all act at the traded size (the former fixed gamma = 10
  solved for a book 3-5x smaller and levered it with an alpha-blind
  projection).  ``vol_target_mode`` is not used.
* ``"fixed"``: ``gamma = cfg.risk_aversion`` (default 2, ~ the closed form on
  the measured alpha); a vol target is then enforced by ``cfg.vol_target_mode``:

  - ``"projection"``: when the predicted vol of the optimum ``w*`` differs from
    the target the book is re-solved as ``minimize gamma H (w - s w*)' Sigma
    (w - s w*) + c * TC(w - w0)`` (``c = cfg.vol_target_projection_cost_weight``)
    subject to the identical constraint set plus ``sqrt(w' Sigma w) <=
    target``, so every band the first solve honoured still holds after scaling
    (``s = target / predicted``, capped by the gross and per-name limits when
    levering up).  The distance is the risk-aversion-weighted *tracking
    variance* to the anchor, i.e. the utility the main objective charges for
    that much risk, so both terms are fractions of NAV (second-pass finding 2;
    the former Euclidean ``||w - s w*||^2`` was in weight^2 units).
  - ``"constraint"``: one solve -- the SOC risk cap ``sqrt(w' Sigma w) <=
    target`` is added to the alpha problem, so alpha, risk penalty, costs and
    the cap are traded off jointly.  The target is a ceiling (no scale-up).

  With ``target_vol_annual=None`` both modes solve once at ``cfg.risk_aversion``.

Universe: only names in ``risk.tickers`` are held or traded (second-pass
finding 3).  Names outside the risk model that are not held are dropped from
the variable set (``w = 0``; those with alpha are counted in
``n_excluded_alpha``); held ones are forced toward zero (``force_to_zero`` at
the ADV cap) with zero exposure and the conservative out-of-model specific
variance (:func:`qts.core.risk_fill.out_of_model_specific_var`).
``n_out_of_model`` counts out-of-model names still held *after* the solve.

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

from qts.core.config import ExecutionConfig, PortfolioConfig
from qts.core.costs import VENUE_ROUTED
from qts.core.risk_fill import out_of_model_specific_var
from qts.core.roles import RISK_ONLY_STYLES
from qts.core.types import RiskModelOutput
from qts.portfolio.alpha_scaling import cost_implied_holding_days, gp_sleeve_shrink_factors
from qts.portfolio.costs import CostTerms, Snapshot, align_snapshot
from qts.portfolio.diagnostics import transfer_coefficient

log = logging.getLogger(__name__)

OK_STATUSES = ("optimal", "optimal_inaccurate")
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


@dataclass
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
    # GAMMA_BRACKET end), "max_solves", "no_alpha", "solve_failed:<status>"; "" otherwise
    vol_target_binding: str = ""
    n_solves: int = 0  # convex solves for this date (search steps / projection included)
    n_excluded_alpha: int = 0  # names with alpha outside the risk model (excluded, w = 0)

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
    n_out_of_model: int  # held names outside the risk model in the problem (forced exits)
    in_model: np.ndarray | None = None  # (n,) bool: name in risk.tickers
    full_tickers: pd.Index | None = None  # alpha | w_current | risk tickers (result index)
    excluded_alpha: list[str] = field(default_factory=list)  # out-of-model names with alpha, excluded
    _L_ord: np.ndarray | None = field(default=None, repr=False)

    def variance(self, w: np.ndarray) -> float:
        fw = self.Bv.T @ w
        v = float(fw @ self.F @ fw + (w**2 * self.d).sum())
        if self.y is not None:
            v += self.nu * float(self.y @ w) ** 2
        return v

    def alignment_share(self, w: np.ndarray) -> float:
        v = self.variance(w)
        if self.y is None or v <= 0:
            return 0.0 if v > 0 else float("nan")
        return self.nu * float(self.y @ w) ** 2 / v

    # factor layout for the parametrised problem ------------------------------
    static_cols: list[str] = field(default_factory=list)  # market + industry dummies (constant, sparse)
    dynamic_cols: list[str] = field(default_factory=list)  # styles / statistical (parameter, dense)

    def ordered(self) -> list[str]:
        return list(self.static_cols) + list(self.dynamic_cols)

    def static_block(self) -> np.ndarray:
        return self.cols(self.static_cols)

    def L_ordered(self) -> np.ndarray:
        """``L`` with ``L L' = F`` in the ``[static | dynamic]`` factor order (cached)."""
        if self._L_ord is None:
            idx = [self.factors.index(c) for c in self.ordered()]
            self._L_ord = _psd_factor(self.F[np.ix_(idx, idx)])
        return self._L_ord

    def B_ordered(self) -> np.ndarray:
        """Exposures ``(n, K)`` in the ``[static | dynamic]`` factor order."""
        return self.cols(self.ordered())

    def sigma_inv_quad(self, a: np.ndarray) -> float:
        """``a' Sigma^-1 a`` for ``Sigma = B F B' + D + nu y y'`` (daily), in
        factor form (Woodbury with ``U = [B L, sqrt(nu) y]``, ``F = L L'``):
        ``Sigma^-1 = D^-1 - D^-1 U (I + U' D^-1 U)^-1 U' D^-1``; valid for a
        singular (PSD-clipped) ``F``.  O(N K^2)."""
        U = self.Bv @ self.L
        if self.y is not None and self.nu > 0:
            U = np.column_stack([U, np.sqrt(self.nu) * self.y])
        dinv = 1.0 / self.d
        Da = dinv * a
        UtDa = U.T @ Da
        M = np.eye(U.shape[1]) + (U * dinv[:, None]).T @ U
        try:
            corr = float(UtDa @ np.linalg.solve(M, UtDa))
        except np.linalg.LinAlgError:
            corr = float(UtDa @ np.linalg.lstsq(M, UtDa, rcond=None)[0])
        return max(float(a @ Da) - corr, 0.0)

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


def _is_array_const(e) -> bool:
    return isinstance(e, cp.Constant) and np.asarray(e.value).size > 1


def _is_zero_scalar(e) -> bool:
    if not isinstance(e, cp.Constant):
        return False
    v = np.asarray(e.value, dtype=float)
    return v.size == 1 and float(v.reshape(-1)[0]) == 0.0


def _match_scaled_hinge(e):
    """``multiply(C, maximum(A - K, 0))`` with array constants ``C >= 0`` and
    ``K`` -> ``(C, A, K)`` else ``None``.  This is the routed venue's kink term
    (``spread * pos(|x| - k)``, ``impact_kink * pos(|x|^e - k^e)``)."""
    if type(e).__name__ != "multiply" or len(e.args) != 2:
        return None
    c, m = e.args
    if not _is_array_const(c) or np.any(np.asarray(c.value) < 0):
        return None
    if type(m).__name__ != "maximum" or len(m.args) != 2 or not _is_zero_scalar(m.args[1]):
        return None
    add = m.args[0]
    if type(add).__name__ != "AddExpression" or len(add.args) != 2:
        return None
    neg = add.args[1]
    if type(neg).__name__ != "NegExpression" or not _is_array_const(neg.args[0]):
        return None
    return c, add.args[0], neg.args[0]


class _Parametrizer:
    """Copy of a cost expression with every array constant replaced by a
    ``cp.Parameter`` (so the cached problem can be re-solved with new
    coefficients, DPP).  ``recipes[i]`` says how parameter ``i``'s value is
    computed from the expression's array constants listed in tree order
    (:func:`_constants`): ``("c", j)`` = constant ``j``; ``("mul", j, k)`` =
    constant ``j`` * constant ``k``.

    DPP rewrite (second-pass finding 6): the routed venue's kink terms
    ``multiply(C, pos(A - K))`` are a parameter times a parameter-dependent
    expression once both constants become parameters, which is *not* DPP (the
    problem was then rebuilt every solve).  For ``C >= 0`` the identity
    ``C * pos(A - K) = pos(C * A - C * K)`` gives the DPP form
    ``pos(multiply(P_C, A) - P_CK)`` with ``P_CK = C * K`` a parameter of its own.

    With ``exact_power`` every ``power(., p)`` rational SOC approximation
    (cvxpy's default ``PowerApprox``) is replaced by the exact power-cone atom
    (``approx=False``; Clarabel has native 3-d power cones): one cone per name
    instead of a ~10-cone SOC tower -- a smaller, more accurate problem."""

    def __init__(self, exact_power: bool = True):
        self.exact_power = exact_power
        self.params: list[cp.Parameter] = []
        self.recipes: list[tuple[Any, ...]] = []
        self.n_consts = 0

    def _param(self, v: np.ndarray, recipe: tuple) -> cp.Parameter:
        kw = {"nonneg": True} if np.all(v >= 0) else ({"nonpos": True} if np.all(v <= 0) else {})
        p = cp.Parameter(v.shape, **kw)
        p.value = v
        self.params.append(p)
        self.recipes.append(recipe)
        return p

    def __call__(self, e):
        if isinstance(e, cp.Constant):
            v = np.asarray(e.value, dtype=float)
            if v.size <= 1:
                return e
            j = self.n_consts
            self.n_consts += 1
            return self._param(v, ("c", j))
        if not e.args:
            return e
        hinge = _match_scaled_hinge(e)
        if hinge is not None:
            c, a, k = hinge
            jc = self.n_consts
            self.n_consts += 1
            a_p = self(a)
            jk = self.n_consts
            self.n_consts += 1
            cv = np.asarray(c.value, dtype=float)
            kv = np.asarray(k.value, dtype=float)
            pc = self._param(cv, ("c", jc))
            pck = self._param(cv * kv, ("mul", jc, jk))
            return cp.maximum(cp.multiply(pc, a_p) - pck, 0.0)
        args = [self(a) for a in e.args]
        if self.exact_power and type(e).__name__ == "PowerApprox":
            return cp.power(args[0], float(e.get_data()[0]), approx=False)
        return e.copy(args)


def _parametrize(e, params: list[cp.Parameter], exact_power: bool = True):
    """Back-compatible wrapper: parametrise ``e`` and append its parameters to
    ``params`` (recipes are discarded; use :class:`_Parametrizer` directly to
    re-assign values)."""
    pz = _Parametrizer(exact_power)
    out = pz(e)
    params.extend(pz.params)
    return out


def _assign(
    params: Sequence[cp.Parameter],
    values: Sequence[np.ndarray],
    recipes: Sequence[tuple[Any, ...]] | None = None,
) -> bool:
    """Set parameter values from the expression's array constants ``values``
    (via ``recipes``; default one constant per parameter, in order); False if
    the counts / shapes do not match or a value violates a parameter's sign."""
    if recipes is None and len(params) != len(values):
        return False
    rec: list[tuple[Any, ...]] = (
        [("c", j) for j in range(len(values))] if recipes is None else [tuple(r) for r in recipes]
    )
    if len(params) != len(rec):
        return False
    new: list[np.ndarray] = []
    for r in rec:
        idx = [int(j) for j in r[1:]]
        if any(j >= len(values) for j in idx):
            return False
        v = np.asarray(values[idx[0]], dtype=float)
        if r[0] == "mul":
            other = np.asarray(values[idx[1]], dtype=float)
            if other.shape != v.shape:
                return False
            v = v * other
        new.append(v)
    for p, v in zip(params, new, strict=True):
        if p.shape != v.shape:
            return False
        if p.is_nonneg() and np.any(v < 0):
            return False
        if p.is_nonpos() and np.any(v > 0):
            return False
    for p, v in zip(params, new, strict=True):
        p.value = v
    return True


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
        n_ind, n_sty, has_beta = len(layout["ind"]), len(layout["sty"]), layout["beta"] is not None
        w = cp.Variable(n, name="w")
        x = cp.Variable(n, name="x")
        f = cp.Variable(K, name="f")  # factor exposures B'w
        self.w, self.x, self.f = w, x, f
        Ks, Kd = len(layout["static"]), len(layout["dynamic"])
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
        for fn in extra_fns or []:
            extra.extend(fn(w, x, tickers))
        clist = list(cons.values()) + extra + ([self.participation] if with_part else [])

        # cost terms: the cost model's own expressions, constants -> parameters
        tc, hc = cost_model.cvx_terms(x, w, *cost_args)
        self.cost_sig = (_leaf_sig(tc), _leaf_sig(hc))
        pz = _Parametrizer(exact_power)
        tc_p = pz(tc)
        hc_p = pz(hc)
        self.cost_params: list[cp.Parameter] = pz.params
        self.cost_recipes: list[tuple] = pz.recipes
        self.risk_cap = None
        if kind == "alpha":
            risk = (
                cp.sum_squares(P["Lk"].T @ f)
                + cp.sum_squares(cp.multiply(P["sdk"], w))
                + cp.square(P["yk"] @ w)
            )
            obj = P["a"] @ w - risk - coefs["trade"] * tc_p - coefs["hold"] * hc_p
            if coefs["turnover"] > 0:
                obj = obj - coefs["turnover"] * cp.norm1(x)
            if vol_cap:
                self.risk_cap = self._risk_norm(w) <= P["tgt"]
                clist.append(self.risk_cap)
            self.problem = cp.Problem(cp.Maximize(obj), clist)
        else:  # projection: kappa * tracking variance to the anchor + c * TC (both NAV fractions)
            track = (
                cp.sum_squares(P["Lk"].T @ f - P["fa"])
                + cp.sum_squares(cp.multiply(P["sdk"], w) - P["sda"])
                + cp.square(P["yk"] @ w - P["ya"])
            )
            obj = track
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
        power cones instead of cvxpy's SOC approximation (see :func:`_parametrize`).
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
        self.n_builds = 0
        self.n_reuses = 0

    # ------------------------------------------------------------------ utils
    def _solver_chain(self) -> list[str]:
        if self._chain is None:  # cp.installed_solvers() costs ~10 ms per call
            installed = set(cp.installed_solvers())
            out: list[str] = []
            for s in [self.cfg.solver, "CLARABEL", "SCS", "OSQP"]:
                if s and s.upper() not in out and s.upper() in installed:
                    out.append(s.upper())
            self._chain = out
        return self._chain

    # ------------------------------------------------------------ factor roles
    def _resolve_factor_roles(
        self,
        risk: RiskModelOutput,
        factors: list[str],
        neutral_styles: Sequence[str] | None,
        industry_columns: Sequence[str] | None,
        notes: list[str],
        has_benchmark: bool = False,
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
        defaulted = (
            neutral_styles is None
            and bool(getattr(cfg, "neutralize_risk_only_styles", False))
            and (cfg.book == "long_short" or has_benchmark)
        )
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
                notes.append(
                    f"beta_neutral requested but style '{beta_name}' not in factor_groups['style']; skipped"
                )
            if industry_columns:
                ind_cols = [c for c in industry_columns if c in factors]
            else:
                ind_cols = industry
        else:
            log.warning(
                "risk model output has no factor_groups; falling back to name-based factor "
                "role detection (beta = exact '%s', industry = 'IND*' prefix)",
                beta_name,
            )
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

    def _prepare(
        self,
        alpha: pd.Series,
        risk: RiskModelOutput,
        w_current: pd.Series,
        nav: float,
        snapshot: Snapshot,
        neutral_styles,
        industry_columns,
        no_trade,
        force_to_zero,
        benchmark_weights,
        notes: list[str],
        part_rate: float | None = None,
    ) -> _ProblemData:
        cfg = self.cfg
        full = pd.Index(alpha.index).union(pd.Index(w_current.index)).union(risk.tickers)
        full = pd.Index(sorted(full))
        a_full = pd.Series(alpha, dtype=float).reindex(full).fillna(0.0).values
        w0_full = pd.Series(w_current, dtype=float).reindex(full).fillna(0.0).values
        in_model_full = full.isin(risk.tickers)
        # second-pass finding 3: the optimizer holds / trades only names the risk
        # model covers.  A name outside it has no factor exposure in the model,
        # so it would act as a "free" hedge for the net / beta bands while its
        # real market beta is ~1.  Not-held names are dropped from the variable
        # set; held ones stay only to be exited (force_to_zero below), alpha 0.
        keep = in_model_full | (w0_full != 0)
        excl = (~keep) & (a_full != 0)
        excluded_alpha = [str(t) for t in full[excl]]
        if excluded_alpha:
            shown = ", ".join(excluded_alpha[:5]) + (" ..." if len(excluded_alpha) > 5 else "")
            notes.append(
                f"{len(excluded_alpha)} names with alpha outside the risk model excluded "
                f"(not held, not traded): {shown}"
            )
        tickers = full[keep]
        in_model = in_model_full[keep]
        a = np.where(in_model, a_full[keep], 0.0)
        w0 = w0_full[keep]
        snap = align_snapshot(snapshot, tickers)

        B = risk.exposures.reindex(tickers).fillna(0.0)
        factors = list(B.columns)
        Bv = B.values
        F = risk.factor_cov.loc[factors, factors].values
        L = _psd_factor(F)
        d_known = pd.Series(risk.specific_var, dtype=float)
        d = d_known.reindex(tickers)
        # names outside the risk universe (here: held names being exited): zero
        # factor exposure and the shared conservative fill (90th percentile)
        fill = out_of_model_specific_var(d_known)
        d = d.fillna(fill).clip(lower=1e-10).values
        held_out = (~in_model) & (w0 != 0)
        n_out = int(held_out.sum())
        if n_out:
            notes.append(
                f"{n_out} held names outside the risk model universe forced toward zero; "
                f"specific var set to 90th pct ({fill:.2e}), zero factor exposure"
            )
            log.debug(
                "optimizer: %d held names outside risk universe on %s", n_out, getattr(risk, "date", "?")
            )
        sd = np.sqrt(d)
        y = None
        nu = 0.0
        if risk.alignment_vector is not None and risk.alignment_var > 0:
            y = pd.Series(risk.alignment_vector, dtype=float).reindex(tickers).fillna(0.0).values
            nu = float(risk.alignment_var)

        beta_col, ind_cols, sty = self._resolve_factor_roles(
            risk,
            factors,
            neutral_styles,
            industry_columns,
            notes,
            has_benchmark=benchmark_weights is not None,
        )
        wb = None
        if benchmark_weights is not None:
            wb = pd.Series(benchmark_weights, dtype=float).reindex(tickers).fillna(0.0).values

        adv = np.asarray(pd.Series(_snap_get(snap, "adv_usd")).reindex(tickers).values, dtype=float)
        rate = float(cfg.max_adv_participation if part_rate is None else part_rate)
        part_cap = rate * adv / max(nav, 1.0)
        nt = [t for t in (no_trade or []) if t in tickers]
        no_trade_mask = np.isin(tickers, nt)
        req_map = dict(force_to_zero or {})
        for t in tickers[held_out]:  # out-of-model holdings: exit (caller's cap wins if tighter)
            req_map[t] = min(float(req_map.get(t, 0.0)), 0.0)
        force_mask = np.zeros(len(tickers), dtype=bool)
        force_caps = np.zeros(len(tickers))
        if req_map:
            fz = pd.Series(req_map, dtype=float).reindex(tickers)
            force_mask = fz.notna().values & ~no_trade_mask
            req = np.nan_to_num(fz.values, nan=np.inf)
            # never ask for more than the ADV cap allows in one day (keeps the problem feasible)
            force_caps = np.where(force_mask, np.maximum(np.maximum(req, 0.0), np.abs(w0) - part_cap), 0.0)
        static_cols = self._static_columns(risk, factors, Bv)
        dynamic_cols = [c for c in factors if c not in set(static_cols)]
        return _ProblemData(
            tickers=tickers,
            a=a,
            w0=w0,
            snap=snap,
            factors=factors,
            Bv=Bv,
            F=F,
            L=L,
            d=d,
            sd=sd,
            y=y,
            nu=nu,
            nav=float(nav),
            beta_col=beta_col,
            ind_cols=ind_cols,
            sty_cols=sty,
            wb=wb,
            no_trade_mask=no_trade_mask,
            force_mask=force_mask,
            force_caps=force_caps,
            part_cap=part_cap,
            n_out_of_model=n_out,
            in_model=in_model,
            full_tickers=full,
            excluded_alpha=excluded_alpha,
            static_cols=static_cols,
            dynamic_cols=dynamic_cols,
        )

    @staticmethod
    def _static_columns(risk: RiskModelOutput, factors: list[str], Bv: np.ndarray) -> list[str]:
        """Factor columns entered as a *constant sparse* block of the cached
        problem: the market and industry groups (0/1 dummies that change only on
        reclassification / universe change, which then triggers a rebuild).
        Without ``factor_groups``: any column whose values are all 0 or 1."""
        groups = getattr(risk, "factor_groups", None) or {}
        named = (
            set(groups.get("market", [])) | set(groups.get("industry", []))
            if isinstance(groups, dict)
            else set()
        )
        out = []
        for j, c in enumerate(factors):
            col = Bv[:, j]
            dummy = bool(np.all((col == 0.0) | (col == 1.0)))
            if (c in named and dummy) or (not named and dummy):
                out.append(c)
        return out

    # ------------------------------------------------------- compiled problems
    def _structure_key(
        self, kind: str, pdat: _ProblemData, H: int, with_part: bool, vol_cap: bool, venue: str
    ) -> tuple:
        cfg = self.cfg
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
            self.beta_band,
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
        key = self._structure_key(kind, pdat, H, with_part, vol_cap, venue)
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
        layout = {
            "static": list(pdat.static_cols),
            "dynamic": list(pdat.dynamic_cols),
            "static_block": pdat.static_block(),
            "beta": pdat.beta_col,
            "ind": list(pdat.ind_cols),
            "sty": list(pdat.sty_cols),
            "has_no_trade": bool(pdat.no_trade_mask.any()),
            "has_force": bool(pdat.force_mask.any()),
        }
        comp = _Compiled(
            kind,
            key,
            len(pdat.tickers),
            len(pdat.factors),
            layout,
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

    # ------------------------------------------------------------------ GP
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
        cfg = self.cfg
        H = int(horizon_days or self.horizon_days)
        t0 = time.perf_counter()
        notes: list[str] = []
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
            part_rate=self.participation_rate(alpha_half_life_days),
        )
        if getattr(cfg, "participation_from_urgency", False):
            notes.append(
                f"ADV participation cap {self.participation_rate(alpha_half_life_days):.2%} "
                f"from urgency (half-life {alpha_half_life_days})"
            )
        tickers, w0 = pdat.tickers, pdat.w0
        g = self._gp_factor(pdat, H, alpha_half_life_days, venue, notes)
        if g != 1.0:
            pdat.a = pdat.a * g
        a = pdat.a

        target = cfg.target_vol_annual
        has_target = target is not None and np.isfinite(target) and target > 0
        search = bool(has_target and getattr(cfg, "risk_aversion_mode", "fixed") == "target_vol")
        mode = getattr(cfg, "vol_target_mode", "projection")
        vol_cap = bool(has_target and not search and mode == "constraint")
        tgt_daily = float(target) / np.sqrt(TRADING_DAYS) * (1.0 + 1e-7) if has_target else None
        gamma = float(cfg.risk_aversion)
        if search:
            g0, ir = self.closed_form_gamma(pdat, H, float(target))
            if np.isfinite(g0) and g0 > 0:
                gamma = float(np.clip(g0, *GAMMA_BRACKET))
            notes.append(f"gamma0 = IR {ir:.3f} / (2 x {float(target):.3f}) = {g0:.3f}")

        # ---- solve (participation relaxation is opt-in)
        status, solver_used, obj_val, with_part = "failed", "", float("nan"), True
        attempts: list[tuple[str, bool]] = [("participation", True)]
        if getattr(cfg, "relax_participation_on_infeasible", False):
            attempts.append(("participation_relaxed", False))
        comp: _Compiled | None = None
        reused = False
        n_solves = 0
        for label, with_part in attempts:
            comp, reused = self._compiled("alpha", pdat, H, with_part, vol_cap, venue, constraints_extra)
            comp.set_values(pdat, tgt=tgt_daily, kappa=gamma * H)
            dpp = self.reuse_problem and comp.is_dpp
            if search:
                # gamma is free in this mode: on a numerical failure of the primary
                # solver nudge it by GAMMA_NUDGE before the (slow) fallback solvers
                status, solver_used = self._try_solvers(comp.problem, dpp=dpp, primary_only=True)
                n_solves += 1
                if status not in OK_STATUSES and status not in ("infeasible", "unbounded"):
                    gamma *= GAMMA_NUDGE
                    comp.set_kappa(pdat, gamma * H)
                    status, solver_used = self._try_solvers(comp.problem, dpp=dpp)
                    n_solves += 1
            else:
                status, solver_used = self._try_solvers(comp.problem, dpp=dpp)
                n_solves += 1
            if status in OK_STATUSES:
                obj_val = float(comp.problem.value)
                if not with_part:
                    notes.append("ADV participation bounds relaxed (infeasible with them)")
                    log.warning(
                        "optimizer: ADV participation caps dropped on %s (infeasible with them)",
                        getattr(risk, "date", "?"),
                    )
                break
            notes.append(f"{label}: {status}")

        assert comp is not None
        w = comp.w
        solve_time = time.perf_counter() - t0
        if status not in OK_STATUSES or w.value is None:
            log.warning(
                "optimizer failed on %s (status=%s, solver=%s); holding current weights",
                getattr(risk, "date", "?"),
                status,
                solver_used or "none",
            )
            w_t = self._full(pdat, w0)
            var_d = pdat.variance(w0)
            return OptimizationResult(
                w_target=w_t,
                x=pd.Series(0.0, index=w_t.index),
                expected_alpha=float(a @ w0),
                expected_risk=float(np.sqrt(max(var_d, 0) * TRADING_DAYS)),
                expected_cost=float("nan"),
                turnover=0.0,
                transfer_coefficient=float("nan"),
                status=status,
                solve_time=solve_time,
                binding_constraints={},
                objective_value=obj_val,
                solver=solver_used,
                notes=notes,
                exante_var_daily=var_d,
                n_out_of_model=self._n_out(pdat, w0),
                alignment_share=pdat.alignment_share(w0),
                gp_shrink=g,
                problem_reused=reused,
                risk_aversion=gamma,
                n_solves=n_solves,
                n_excluded_alpha=len(pdat.excluded_alpha),
            )

        wv = self._clean(np.asarray(w.value).ravel(), w0)

        # ---- binding constraints via duals (read before any re-solve overwrites them)
        binding = self._binding(comp, pdat, with_part)
        if vol_cap and comp.risk_cap is not None and self._n_binding(comp.risk_cap):
            binding["vol_cap"] = 1
            notes.append(f"vol cap binding (target {target:.4f})")

        vt_binding = ""
        if search:
            sr = self._gamma_search(comp, pdat, H, with_part, float(target), gamma, wv, binding, obj_val)
            wv, binding, obj_val, gamma = sr["w"], sr["binding"], sr["obj"], sr["gamma"]
            status, vt_binding = sr["status"], sr["outcome"]
            solver_used = sr["solver"] or solver_used
            n_solves += sr["n_extra"]
            notes.append(sr["note"])

        # ---- vol targeting by projection onto the same feasible set (fixed-gamma mode only)
        scalar = 1.0
        if has_target and not search and mode == "projection":
            wv, scalar, proj_note, k = self._vol_target_projection(
                wv,
                pdat,
                H,
                with_part,
                venue,
                constraints_extra,
                date=getattr(risk, "date", "?"),
                kappa=gamma * H,
            )
            n_solves += k
            if proj_note:
                notes.append(proj_note)
        var_d = pdat.variance(wv)
        pred_vol = float(np.sqrt(max(var_d, 0.0) * TRADING_DAYS))
        share = pdat.alignment_share(wv)
        if pdat.y is not None and np.isfinite(share):
            notes.append(f"alignment share of ex-ante variance {share:.3f}")

        # ---- expected cost at the final trade
        exp_cost = float("nan")
        try:
            xc = cp.Constant(wv - w0)
            wc = cp.Constant(wv)
            tc, hc = self.cost_model.cvx_terms(xc, wc, nav, pdat.snap, venue)
            exp_cost = float(np.asarray(tc.value).ravel()[0] if np.ndim(tc.value) else tc.value) + H * float(
                np.asarray(hc.value).ravel()[0] if np.ndim(hc.value) else hc.value
            )
        except Exception as e:  # pragma: no cover - diagnostic only
            notes.append(f"expected_cost unavailable: {e}")

        tc_val = (
            transfer_coefficient(pd.Series(a, index=tickers), pd.Series(wv, index=tickers), risk=risk)
            if np.any(a != 0)
            else float("nan")
        )
        w_t = self._full(pdat, wv)
        x_t = w_t - self._full(pdat, w0)
        return OptimizationResult(
            w_target=w_t,
            x=x_t,
            expected_alpha=float(a @ wv),
            expected_risk=pred_vol,
            expected_cost=exp_cost,
            turnover=float(np.abs(x_t.values).sum()),
            transfer_coefficient=tc_val,
            status=status,
            solve_time=time.perf_counter() - t0,
            binding_constraints=binding,
            objective_value=obj_val,
            vol_scalar=scalar,
            solver=solver_used,
            notes=notes,
            exante_var_daily=var_d,
            n_out_of_model=self._n_out(pdat, wv),
            alignment_share=share,
            gp_shrink=g,
            problem_reused=reused,
            risk_aversion=gamma,
            vol_target_binding=vt_binding,
            n_solves=n_solves,
            n_excluded_alpha=len(pdat.excluded_alpha),
        )

    @staticmethod
    def _full(pdat: _ProblemData, v: np.ndarray) -> pd.Series:
        """Problem-space vector -> Series on the full result index (0 for excluded names)."""
        s = pd.Series(v, index=pdat.tickers, dtype=float)
        return s if pdat.full_tickers is None else s.reindex(pdat.full_tickers).fillna(0.0)

    @staticmethod
    def _n_out(pdat: _ProblemData, wv: np.ndarray) -> int:
        """Names outside the risk model with a non-zero weight in ``wv`` (after the solve)."""
        if pdat.in_model is None:
            return 0
        return int(((~pdat.in_model) & (np.abs(wv) > 0)).sum())

    def _binding(self, comp: _Compiled, pdat: _ProblemData, with_part: bool) -> dict[str, int]:
        binding: dict[str, int] = {}
        for name, c in comp.cons.items():
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
            cnt = self._n_binding(c, mask)
            if cnt:
                binding[name] = cnt
        if with_part:
            cnt = self._n_binding(comp.participation)
            if cnt:
                binding["participation"] = cnt
        return binding

    # ------------------------------------------------------- risk aversion
    @staticmethod
    def closed_form_gamma(pdat: _ProblemData, H: int, target_annual: float) -> tuple[float, float]:
        """``(gamma0, IR_ann)``: the risk aversion at which the *unconstrained*,
        cost-free optimum has annual vol ``target_annual``.

        The objective is ``alpha_H'w - gamma H w'Sigma_d w`` (``alpha_H`` over
        ``H`` days, ``Sigma_d`` daily), i.e. ``(H/252) [alpha_ann'w - gamma
        w'Sigma_ann w]`` with ``alpha_ann = (252/H) alpha_H`` and ``Sigma_ann =
        252 Sigma_d``: ``gamma`` is an *annual* risk aversion.  Unconstrained,
        ``w* = Sigma_ann^-1 alpha_ann / (2 gamma)`` so

            sigma_ann(w*) = sqrt(alpha_ann' Sigma_ann^-1 alpha_ann) / (2 gamma)
                          = IR_ann / (2 gamma),
            IR_ann = sqrt(252) / H * sqrt(alpha_H' Sigma_d^-1 alpha_H),

        and ``gamma0 = IR_ann / (2 sigma_target)`` (Grinold & Kahn, ch. 14:
        ``sigma* = IR / (2 lambda)``).  ``Sigma`` includes the alignment term;
        ``alpha' Sigma^-1 alpha`` is computed in factor form
        (:meth:`_ProblemData.sigma_inv_quad`).  Constraints and costs shrink the
        optimum, so the target-vol gamma is usually *below* ``gamma0``; it is the
        search's starting point."""
        q = pdat.sigma_inv_quad(pdat.a)
        ir = float(np.sqrt(TRADING_DAYS) / max(H, 1) * np.sqrt(q))
        if not (ir > 0 and target_annual > 0):
            return float("nan"), ir
        return ir / (2.0 * target_annual), ir

    @staticmethod
    def _caps_bind(binding: Mapping[str, int]) -> bool:
        """A leverage limit (gross / 130-30 legs) binds."""
        return any(binding.get(k, 0) for k in GROSS_CAP_CONSTRAINTS)

    @staticmethod
    def _position_caps_bind(binding: Mapping[str, int], w: np.ndarray) -> bool:
        """Per-name or ADV caps bind on at least ``POSITION_CAPS_SHARE`` of the held names."""
        n_held = int((np.abs(w) > SOLVER_TOL).sum())
        n_cap = int(binding.get("max_position", 0)) + int(binding.get("participation", 0))
        return n_held > 0 and n_cap >= max(1, POSITION_CAPS_SHARE * n_held)

    def _gamma_search(
        self,
        comp: _Compiled,
        pdat: _ProblemData,
        H: int,
        with_part: bool,
        target: float,
        gamma: float,
        wv: np.ndarray,
        binding: dict[str, int],
        obj: float,
    ) -> dict[str, Any]:
        """Choose ``gamma`` so that the alpha problem's optimum has ex-ante
        annual vol ``target`` (``risk_aversion_mode="target_vol"``).

        ``sigma(w*(gamma))`` is non-increasing in ``gamma`` for this objective
        (comparing the optimality of ``w1 = w*(g1)`` and ``w2 = w*(g2)`` gives
        ``(g2 - g1)(R(w1) - R(w2)) >= 0``), so a root search on ``log gamma``
        in ``GAMMA_BRACKET`` is safe.  Steps: start at the closed form
        ``gamma0`` (already solved, :meth:`closed_form_gamma`); while only one
        side of the target is known, a Newton step on ``log sigma`` vs ``log
        gamma`` with the measured elasticity (``-1`` before two points exist,
        the unconstrained value); once the target is bracketed, the log-log
        secant (regula falsi) safeguarded to the inner 80 % of the bracket,
        falling back to bisection.  Stops when ``|sigma / target - 1| <
        GAMMA_SEARCH_VOL_TOL``, after ``GAMMA_MAX_SOLVES`` solves, or when the
        caps bind: vol below target with the gross / leg limits binding, or a
        vol elasticity to gamma below ``CAPS_ELASTICITY`` in magnitude while
        per-name / ADV caps bind on at least ``POSITION_CAPS_SHARE`` of the held
        names (lowering gamma no longer adds risk).  A low elasticity *without*
        binding caps means the cost term dominates (small alpha); the search
        then keeps stepping (up to 10x per step) toward the bracket end.  A
        numerical solver failure is retried once at ``GAMMA_NUDGE`` x gamma.  Then the
        capped book is accepted (``outcome="caps"``).  Without convergence the
        book with the highest vol not above ``target (1 + tol)`` is kept (else
        the lowest-vol book).  Each step re-solves the same compiled problem
        with new ``kappa`` parameter values."""
        lo_b, hi_b = GAMMA_BRACKET
        tol = GAMMA_SEARCH_VOL_TOL

        def vol(w: np.ndarray) -> float:
            return float(np.sqrt(max(pdat.variance(w), 0.0) * TRADING_DAYS))

        cur: dict[str, Any] = {
            "g": gamma,
            "vol": vol(wv),
            "w": wv,
            "binding": binding,
            "obj": obj,
            "status": "optimal",
            "solver": "",
        }
        hist = [cur]
        n_failed = 0  # nudged re-solves (count toward the solve budget)
        outcome = "max_solves"
        if not np.any(pdat.a != 0) or cur["vol"] <= 0:
            outcome = "no_alpha"
        while outcome == "max_solves":
            r = cur["vol"] / target
            if abs(r - 1.0) < tol:
                outcome = "target"
                break
            if r < 1.0 and self._caps_bind(cur["binding"]):
                outcome = "caps"
                break
            below = [h for h in hist if h["vol"] < target and h is not cur and h["g"] > cur["g"]]
            elast = float("nan")
            if len(hist) >= 2:
                prev = min(hist[:-1], key=lambda h: abs(np.log(h["g"] / cur["g"])))
                if prev["vol"] > 0 and abs(np.log(prev["g"] / cur["g"])) > 1e-9:
                    elast = float(np.log(cur["vol"] / prev["vol"]) / np.log(cur["g"] / prev["g"]))
            if (
                r < 1.0
                and below
                and np.isfinite(elast)
                and abs(elast) < CAPS_ELASTICITY
                and self._position_caps_bind(cur["binding"], cur["w"])
            ):
                outcome = "caps"
                break
            if len(hist) + n_failed >= GAMMA_MAX_SOLVES:
                break
            over = [h for h in hist if h["vol"] > target]  # gamma too small
            under = [h for h in hist if h["vol"] < target]  # gamma too large
            if over and under:
                o = max(over, key=lambda h: h["g"])
                u = min(under, key=lambda h: h["g"])
                lgo, lgu = np.log(o["g"]), np.log(u["g"])
                lvo, lvu = np.log(o["vol"]), np.log(u["vol"])
                lg = 0.5 * (lgo + lgu)
                if lvo > lvu:
                    cand = lgo + (np.log(target) - lvo) * (lgu - lgo) / (lvu - lvo)
                    m = 0.1 * (lgu - lgo)
                    if lgo + m <= cand <= lgu - m:
                        lg = cand
                g_new = float(np.exp(lg))
            else:
                e = elast if np.isfinite(elast) and elast < -0.01 else -1.0  # measured, else unconstrained
                step = float(np.clip(r ** (-1.0 / e), 0.1, 10.0))
                g_new = cur["g"] * step
            g_new = float(np.clip(g_new, lo_b, hi_b))
            if abs(np.log(g_new / cur["g"])) < 1e-9 or any(abs(np.log(g_new / h["g"])) < 1e-9 for h in hist):
                outcome = "bracket"
                break
            comp.set_kappa(pdat, g_new * H)
            st, solver = self._try_solvers(
                comp.problem, dpp=self.reuse_problem and comp.is_dpp, primary_only=True
            )
            if st not in OK_STATUSES or comp.w.value is None:  # numerical failure: nudge once
                g_new *= GAMMA_NUDGE
                comp.set_kappa(pdat, g_new * H)
                st, solver = self._try_solvers(comp.problem, dpp=self.reuse_problem and comp.is_dpp)
                n_failed += 1
            if st not in OK_STATUSES or comp.w.value is None:
                outcome = f"solve_failed:{st}"
                break
            w_new = self._clean(np.asarray(comp.w.value).ravel(), pdat.w0)
            cur = {
                "g": g_new,
                "vol": vol(w_new),
                "w": w_new,
                "binding": self._binding(comp, pdat, with_part),
                "obj": float(comp.problem.value),
                "status": st,
                "solver": solver,
            }
            hist.append(cur)
        if outcome in ("target", "caps", "no_alpha"):
            best = cur
        else:
            ok = [h for h in hist if h["vol"] <= target * (1.0 + tol)]
            best = max(ok, key=lambda h: h["vol"]) if ok else min(hist, key=lambda h: h["vol"])
        path = ", ".join(f"{h['g']:.3g}:{h['vol']:.4f}" for h in hist)
        note = (
            f"gamma search ({outcome}): gamma {best['g']:.3f}, vol {best['vol']:.4f} vs target "
            f"{target:.4f} after {len(hist)} solves [gamma:vol {path}]"
        )
        return {
            "w": best["w"],
            "binding": best["binding"],
            "obj": best["obj"],
            "gamma": float(best["g"]),
            "status": best["status"],
            "solver": best["solver"],
            "outcome": outcome,
            "n_extra": len(hist) - 1 + n_failed,
            "note": note,
        }

    # ------------------------------------------------------------ vol target
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
        var0 = pdat.variance(wv)
        pred = float(np.sqrt(max(var0, 0.0) * TRADING_DAYS))
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
            gross = float(np.abs(wv).sum())
            if gross >= cfg.gross_leverage_max * (1.0 - 1e-6):
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
        w_new = self._clean(np.asarray(comp.w.value).ravel(), pdat.w0)
        got = float(np.sqrt(max(pdat.variance(w_new), 0.0) * TRADING_DAYS))
        return (
            w_new,
            float(s),
            f"vol-target projection s={s:.3f}: vol {pred:.4f} -> {got:.4f} (target {target:.4f}, {solver})",
            1,
        )

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

    def _try_solvers(self, prob: cp.Problem, dpp: bool = False, primary_only: bool = False):
        """Solve with the solver chain (``primary_only``: its first solver only).
        ``dpp=False`` canonicalises with the parameters' current values as
        constants (``ignore_dpp``): for a one-off solve that is far cheaper than
        a DPP compile."""
        last = "failed"
        kw = dict(self.solver_opts)
        if not dpp:
            kw["ignore_dpp"] = True
        chain = self._solver_chain()
        for s in chain[:1] if primary_only else chain:
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
