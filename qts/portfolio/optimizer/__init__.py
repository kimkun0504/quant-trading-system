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

* ``"target_vol"`` (default; second-pass audit finding 1): gamma is chosen
  per solve against ``cfg.target_vol_annual``, in one of two forms.

  **The target as a cap** (``cfg.vol_target_is_cap``, default since
  2026-11-12; :func:`~qts.portfolio.optimizer.sizing.gamma_cap_search`;
  docs/notes/construction_fix_2026-11-12.md): gamma starts at the base
  ``cfg.risk_aversion`` and is only ever *raised* -- until the ex-ante vol is
  at or under the target and, with ``cfg.leverage_cap_raises_gamma``
  (long/short books), until the gross limit stops binding.  A target the
  caps make unreachable is not chased: lowering gamma against a binding
  gross limit cannot add leverage, it only concentrates the book into the
  names with the largest alpha, while raising gamma scales the whole
  mean-variance book down to fit.  ``vol_target_binding`` is
  ``"below_target"`` (base book admissible), ``"leverage"`` or ``"target"``
  (gamma raised for the gross limit / the vol cap).

  **The target as a level** (``cfg.vol_target_is_cap=False``): gamma is
  chosen so that the optimal book of the *alpha-aware* objective has
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

Minimum trade size (``cfg.min_trade_weight``, default 5 bp of NAV): after
sizing, trades below the threshold are cancelled unless they close a position
or belong to a forced exit (:meth:`SinglePeriodOptimizer._drop_small_trades`).
A post-solve projection: the returned book can sit marginally outside a band;
``n_small_trades_dropped`` / ``max_constraint_violation`` /
``violated_constraint`` on the result say by how much.

Failure semantics: a solve that is not ``optimal`` / ``optimal_inaccurate``
returns ``status`` as reported by the solver, ``w_target = w_current``,
``expected_cost = nan`` and logs a warning.  Dropping the ADV participation
caps on infeasibility is opt-in (``cfg.relax_participation_on_infeasible``).

Package layout
--------------
==========================  ==============================================
module                      contents
==========================  ==============================================
``constants``               tolerances, mode names, gamma-search settings
``result``                  :class:`OptimizationResult`
``problem``                 :class:`_ProblemData`, :func:`prepare_problem`,
                            factor roles
``dpp``                     expression parametrisation (:class:`_Parametrizer`)
``constraints``             the constraint menu, binding counts
``compiled``                :class:`_Compiled` (one DPP problem), structure key
``sizing``                  closed-form gamma, gamma searches (cap form and
                            two-sided), weight clean-up
``solvers``                 solver chain with narrow exception handling
``spo``                     :class:`SinglePeriodOptimizer` (orchestration)
==========================  ==============================================

``from qts.portfolio.optimizer import ...`` keeps working for every name that
lived in the former single module.
"""

from qts.portfolio.optimizer.compiled import _Compiled, problem_layout, structure_key
from qts.portfolio.optimizer.constants import (
    _BIG,
    CAPS_ELASTICITY,
    GAMMA_BRACKET,
    GAMMA_MAX_SOLVES,
    GAMMA_NUDGE,
    GAMMA_SEARCH_VOL_TOL,
    GROSS_CAP_CONSTRAINTS,
    MAX_SCALE_UP,
    OK_STATUSES,
    POSITION_CAPS_SHARE,
    RISK_AVERSION_MODES,
    SOLVER_TOL,
    TERMINAL_STATUSES,
    TRADING_DAYS,
    VOL_TARGET_MODES,
    ExtraConstraint,
)
from qts.portfolio.optimizer.constraints import binding_counts, build_constraints, n_binding
from qts.portfolio.optimizer.dpp import _assign, _constants, _leaf_sig, _match_scaled_hinge, _Parametrizer
from qts.portfolio.optimizer.problem import (
    _ProblemData,
    _psd_factor,
    _snap_get,
    prepare_problem,
    resolve_factor_roles,
    static_columns,
)
from qts.portfolio.optimizer.result import OptimizationResult
from qts.portfolio.optimizer.sizing import (
    GammaSearchResult,
    caps_bind,
    clean_weights,
    closed_form_gamma,
    gamma_search,
    position_caps_bind,
)
from qts.portfolio.optimizer.solvers import SOLVER_FAILURES, SolveAttempt, solve_with_chain, solver_chain
from qts.portfolio.optimizer.spo import SinglePeriodOptimizer

__all__ = [
    "CAPS_ELASTICITY",
    "GAMMA_BRACKET",
    "GAMMA_MAX_SOLVES",
    "GAMMA_NUDGE",
    "GAMMA_SEARCH_VOL_TOL",
    "GROSS_CAP_CONSTRAINTS",
    "MAX_SCALE_UP",
    "OK_STATUSES",
    "POSITION_CAPS_SHARE",
    "RISK_AVERSION_MODES",
    "SOLVER_FAILURES",
    "SOLVER_TOL",
    "TERMINAL_STATUSES",
    "TRADING_DAYS",
    "VOL_TARGET_MODES",
    "_BIG",
    "ExtraConstraint",
    "GammaSearchResult",
    "OptimizationResult",
    "SinglePeriodOptimizer",
    "SolveAttempt",
    "_Compiled",
    "_Parametrizer",
    "_ProblemData",
    "_assign",
    "_constants",
    "_leaf_sig",
    "_match_scaled_hinge",
    "_psd_factor",
    "_snap_get",
    "binding_counts",
    "build_constraints",
    "caps_bind",
    "clean_weights",
    "closed_form_gamma",
    "gamma_search",
    "n_binding",
    "position_caps_bind",
    "prepare_problem",
    "problem_layout",
    "resolve_factor_roles",
    "solve_with_chain",
    "solver_chain",
    "static_columns",
    "structure_key",
]
