"""Sizing: the risk aversion that hits the vol target (``risk_aversion_mode="target_vol"``),
or -- with ``PortfolioConfig.vol_target_is_cap`` -- the smallest risk aversion at or
above the configured base that keeps the book inside the vol cap (and, with
``leverage_cap_raises_gamma``, off the leverage cap).

* :func:`gamma_cap_search` -- the cap form: upward-only search from the base gamma.
* :func:`closed_form_gamma` -- the start: the gamma at which the unconstrained,
  cost-free optimum has the target vol (Grinold & Kahn ch. 14).
* :func:`gamma_search` -- safeguarded root search on ``log gamma`` that
  re-solves one compiled problem with new ``kappa = gamma H`` values.
* :func:`clean_weights` -- solver-noise clean-up shared by every solve.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import cast

import numpy as np

from qts.portfolio.optimizer.compiled import _Compiled
from qts.portfolio.optimizer.constants import (
    CAPS_ELASTICITY,
    GAMMA_BRACKET,
    GAMMA_CAP_ELASTICITY,
    GAMMA_CAP_FIRST_STEP,
    GAMMA_CAP_GROSS_TOL,
    GAMMA_CAP_MIN_RATIO,
    GAMMA_MAX_SOLVES,
    GAMMA_NUDGE,
    GAMMA_SEARCH_VOL_TOL,
    GROSS_CAP_CONSTRAINTS,
    OK_STATUSES,
    POSITION_CAPS_SHARE,
    SOLVER_TOL,
    TRADING_DAYS,
)
from qts.portfolio.optimizer.problem import _ProblemData

#: ``solve(primary_only) -> (status, solver)`` on the compiled problem's current parameter values
SolveFn = Callable[[bool], tuple[str, str]]


def _objective(comp: _Compiled) -> float:
    """The compiled problem's optimal objective value after a successful solve."""
    return float(cast(float, comp.problem.value))


def clean_weights(wv: np.ndarray, w0: np.ndarray) -> np.ndarray:
    """Zero ``|w| < SOLVER_TOL`` and keep tiny solver noise from creating
    trades for untouched names."""
    wv = wv.copy()
    wv[np.abs(wv) < SOLVER_TOL] = 0.0
    return np.where(np.abs(wv - w0) < SOLVER_TOL, w0, wv)


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


def caps_bind(binding: Mapping[str, int]) -> bool:
    """A leverage limit (gross / 130-30 legs) binds."""
    return any(binding.get(k, 0) for k in GROSS_CAP_CONSTRAINTS)


def position_caps_bind(binding: Mapping[str, int], w: np.ndarray) -> bool:
    """Per-name or ADV caps bind on at least ``POSITION_CAPS_SHARE`` of the held names."""
    n_held = int((np.abs(w) > SOLVER_TOL).sum())
    n_cap = int(binding.get("max_position", 0)) + int(binding.get("participation", 0))
    return n_held > 0 and n_cap >= max(1, POSITION_CAPS_SHARE * n_held)


@dataclass(frozen=True, eq=False)
class GammaSearchResult:
    """The book chosen by :func:`gamma_search`."""

    w: np.ndarray
    binding: dict[str, int]
    obj: float
    gamma: float
    status: str
    solver: str
    # target / caps / bracket / max_solves / no_alpha / solve_failed:<status>; cap form
    # (gamma_cap_search): below_target / target / leverage / bracket / max_solves / ...
    outcome: str
    n_extra: int  # solves beyond the starting one (nudged retries included)
    note: str


@dataclass
class _Point:
    g: float
    vol: float
    w: np.ndarray
    binding: dict[str, int]
    obj: float
    status: str
    solver: str


def _elasticity(hist: list[_Point], cur: _Point) -> float:
    """``d log vol / d log gamma`` between ``cur`` and the nearest earlier point (nan if undefined)."""
    if len(hist) < 2:
        return float("nan")
    prev = min(hist[:-1], key=lambda h: abs(np.log(h.g / cur.g)))
    if prev.vol > 0 and abs(np.log(prev.g / cur.g)) > 1e-9:
        return float(np.log(cur.vol / prev.vol) / np.log(cur.g / prev.g))
    return float("nan")


def _next_gamma(hist: list[_Point], cur: _Point, target: float, r: float, elast: float) -> float:
    """Log-log secant inside the bracket (safeguarded to its inner 80 %, else
    bisection) once the target is bracketed; a Newton step with the measured
    elasticity (``-1`` = unconstrained) before."""
    over = [h for h in hist if h.vol > target]  # gamma too small
    under = [h for h in hist if h.vol < target]  # gamma too large
    if over and under:
        o = max(over, key=lambda h: h.g)
        u = min(under, key=lambda h: h.g)
        lgo, lgu = np.log(o.g), np.log(u.g)
        lvo, lvu = np.log(o.vol), np.log(u.vol)
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
        g_new = cur.g * step
    return float(np.clip(g_new, *GAMMA_BRACKET))


def gamma_search(
    comp: _Compiled,
    pdat: _ProblemData,
    H: int,
    with_part: bool,
    target: float,
    gamma: float,
    wv: np.ndarray,
    binding: dict[str, int],
    obj: float,
    solve: SolveFn,
) -> GammaSearchResult:
    """Choose ``gamma`` so that the alpha problem's optimum has ex-ante
    annual vol ``target`` (``risk_aversion_mode="target_vol"``).

    ``sigma(w*(gamma))`` is non-increasing in ``gamma`` for this objective
    (comparing the optimality of ``w1 = w*(g1)`` and ``w2 = w*(g2)`` gives
    ``(g2 - g1)(R(w1) - R(w2)) >= 0``), so a root search on ``log gamma``
    in ``GAMMA_BRACKET`` is safe.  Steps: start at the closed form
    ``gamma0`` (already solved, :func:`closed_form_gamma`); while only one
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
    tol = GAMMA_SEARCH_VOL_TOL

    def vol(w: np.ndarray) -> float:
        return float(np.sqrt(max(pdat.variance(w), 0.0) * TRADING_DAYS))

    cur = _Point(gamma, vol(wv), wv, binding, obj, "optimal", "")
    hist = [cur]
    n_failed = 0  # nudged re-solves (count toward the solve budget)
    outcome = "max_solves"
    if not np.any(pdat.a != 0) or cur.vol <= 0:
        outcome = "no_alpha"
    while outcome == "max_solves":
        r = cur.vol / target
        if abs(r - 1.0) < tol:
            outcome = "target"
            break
        if r < 1.0 and caps_bind(cur.binding):
            outcome = "caps"
            break
        below = [h for h in hist if h.vol < target and h is not cur and h.g > cur.g]
        elast = _elasticity(hist, cur)
        if (
            r < 1.0
            and below
            and np.isfinite(elast)
            and abs(elast) < CAPS_ELASTICITY
            and position_caps_bind(cur.binding, cur.w)
        ):
            outcome = "caps"
            break
        if len(hist) + n_failed >= GAMMA_MAX_SOLVES:
            break
        g_new = _next_gamma(hist, cur, target, r, elast)
        if abs(np.log(g_new / cur.g)) < 1e-9 or any(abs(np.log(g_new / h.g)) < 1e-9 for h in hist):
            outcome = "bracket"
            break
        comp.set_kappa(pdat, g_new * H)
        st, solver = solve(True)
        if st not in OK_STATUSES or comp.w.value is None:  # numerical failure: nudge once
            g_new *= GAMMA_NUDGE
            comp.set_kappa(pdat, g_new * H)
            st, solver = solve(False)
            n_failed += 1
        if st not in OK_STATUSES or comp.w.value is None:
            outcome = f"solve_failed:{st}"
            break
        w_new = clean_weights(np.asarray(comp.w.value).ravel(), pdat.w0)
        cur = _Point(g_new, vol(w_new), w_new, comp.binding(pdat, with_part), _objective(comp), st, solver)
        hist.append(cur)
    if outcome in ("target", "caps", "no_alpha"):
        best = cur
    else:
        ok = [h for h in hist if h.vol <= target * (1.0 + tol)]
        best = max(ok, key=lambda h: h.vol) if ok else min(hist, key=lambda h: h.vol)
    path = ", ".join(f"{h.g:.3g}:{h.vol:.4f}" for h in hist)
    note = (
        f"gamma search ({outcome}): gamma {best.g:.3f}, vol {best.vol:.4f} vs target "
        f"{target:.4f} after {len(hist)} solves [gamma:vol {path}]"
    )
    return GammaSearchResult(
        w=best.w,
        binding=best.binding,
        obj=best.obj,
        gamma=float(best.g),
        status=best.status,
        solver=best.solver,
        outcome=outcome,
        n_extra=len(hist) - 1 + n_failed,
        note=note,
    )


def _cap_step(adm: list[_Point], need: Callable[[_Point], float], aim: float) -> float:
    """Next gamma of :func:`gamma_cap_search` once an admissible book exists
    (``adm``: the admissible points by increasing gamma): the gamma at which
    ``need`` -- the book's distance to the tighter cap, 1 = at it -- would reach
    ``aim``, from the log-log secant through the two lowest-gamma admissible points,
    or with the prior elasticity ``GAMMA_CAP_ELASTICITY`` while there is only one
    (unconstrained and cost-free it is -1; trade costs anchor the book to the
    current holdings and make it flatter)."""
    p = adm[0]
    e = GAMMA_CAP_ELASTICITY
    if len(adm) >= 2:
        q = adm[1]
        num, dg = np.log(need(p) / need(q)), np.log(p.g / q.g)
        if abs(dg) > 1e-9 and num / dg < -0.05:
            e = float(num / dg)
    return float(p.g * (aim / need(p)) ** (1.0 / e))


def gamma_cap_search(
    comp: _Compiled,
    pdat: _ProblemData,
    H: int,
    with_part: bool,
    target: float,
    gamma: float,
    wv: np.ndarray,
    binding: dict[str, int],
    obj: float,
    solve: SolveFn,
    gross_cap: float | None = None,
) -> GammaSearchResult:
    """The vol target as a **cap** (``PortfolioConfig.vol_target_is_cap``): return
    the book of the smallest ``gamma >= base`` (``gamma``, the configured
    ``risk_aversion``, already solved: ``wv``) that is *admissible*:

    * ex-ante annual vol ``<= target (1 + tol)``, and
    * with ``gross_cap`` given (``leverage_cap_raises_gamma``, long/short books):
      the leverage limit is not binding (no dual on ``gross``).

    Gamma is never lowered below the base.  Why upward only: when the caps make the
    vol target unreachable, lowering gamma cannot add leverage -- the gross limit
    already binds -- it only moves the same gross into the names with the largest
    alpha (the gross limit's multiplier ``lam`` acts as an L1 penalty: a name is
    held only if ``|alpha_i| > lam``), i.e. into few, volatile names.  Raising gamma
    instead scales the mean-variance book down *proportionally* (cost-free and
    otherwise unconstrained, ``w*(gamma) = w*(1) / gamma``), so the book that just
    fits under the leverage limit keeps every name of the diversified optimum.

    Search: the base book is admissible -> done (``outcome="below_target"``).
    Otherwise ``lo`` = base (not admissible) and gamma is stepped up -- a Newton
    step on ``log vol`` with the measured elasticity (-1 before two points exist)
    when the vol cap is the violated one, else by ``GAMMA_CAP_FIRST_STEP`` -- until an admissible ``hi`` is found; then the
    bracket ``[lo, hi]`` is closed from above with :func:`_cap_step` (log-log
    secant on the distance to the tighter cap, ``max(vol / target, gross /
    gross_cap)``), falling back to the geometric mean when the step leaves the
    bracket.  It stops when ``hi`` is *tight* -- vol within ``tol`` of the
    target (``"target"``: the vol cap decides) or gross within
    ``GAMMA_CAP_GROSS_TOL`` of the limit (``"leverage"``: the leverage cap
    decides) -- or the bracket ratio is below ``GAMMA_CAP_MIN_RATIO``, or after
    ``GAMMA_MAX_SOLVES`` solves with an admissible ``hi`` in hand (same labels:
    the book is admissible, only less tight than asked).  Without any admissible
    book the most risk-averse one solved is returned (``"max_solves"``, or
    ``"bracket"`` at the end of ``GAMMA_BRACKET``).  A numerical solver failure is
    retried once at ``GAMMA_NUDGE`` x gamma."""
    tol = GAMMA_SEARCH_VOL_TOL
    g_max = GAMMA_BRACKET[1]

    def vol(w: np.ndarray) -> float:
        return float(np.sqrt(max(pdat.variance(w), 0.0) * TRADING_DAYS))

    def over_vol(p: _Point) -> bool:
        return p.vol > target * (1.0 + tol)

    def levered(p: _Point) -> bool:
        return gross_cap is not None and caps_bind(p.binding)

    def admissible(p: _Point) -> bool:
        return not over_vol(p) and not levered(p)

    def need(p: _Point) -> float:
        """Factor by which an unconstrained book at ``p`` would have to scale gamma
        to sit exactly at the tighter of the two caps (< 1: room to lower gamma)."""
        f = p.vol / target
        if gross_cap is not None and gross_cap > 0:
            f = max(f, float(np.abs(p.w).sum()) / gross_cap)
        return f

    def tight(p: _Point) -> str:
        if p.vol >= target * (1.0 - tol):
            return "target"
        if gross_cap is not None and float(np.abs(p.w).sum()) >= gross_cap * (1.0 - GAMMA_CAP_GROSS_TOL):
            return "leverage"
        return ""

    # aim inside the tolerance band of whichever cap is tighter
    aim = 1.0 - 0.5 * min(tol, GAMMA_CAP_GROSS_TOL if gross_cap is not None else tol)
    cur = _Point(gamma, vol(wv), wv, binding, obj, "optimal", "")
    hist = [cur]
    n_failed = 0
    lo: _Point | None = None  # highest gamma known not admissible
    hi: _Point | None = None  # lowest gamma known admissible
    outcome = "max_solves"
    if not np.any(pdat.a != 0) or cur.vol <= 0:
        outcome = "no_alpha"
    elif admissible(cur):
        outcome = "below_target" if cur.vol < target * (1.0 - tol) else "target"
    else:
        lo = cur
    while outcome == "max_solves":
        assert lo is not None
        if hi is not None:
            label = tight(hi)
            if not label and hi.g / lo.g < GAMMA_CAP_MIN_RATIO:
                label = "leverage" if levered(lo) and not over_vol(lo) else "target"
            if label:
                outcome = label
                break
        if len(hist) + n_failed >= GAMMA_MAX_SOLVES:
            if hi is not None:  # admissible but not yet tight: accepted, named by the cap that decided
                outcome = "leverage" if levered(lo) and not over_vol(lo) else "target"
            break
        if hi is None:
            step = 1.0
            if over_vol(lo):  # Newton on log vol with the measured elasticity (-1 before two points)
                e = _elasticity(hist, lo)
                e = e if np.isfinite(e) and e < -0.05 else -1.0
                step = float(np.clip((lo.vol / (aim * target)) ** (-1.0 / e), 1.0 + 2.0 * tol, 10.0))
            if levered(lo):
                step = max(step, GAMMA_CAP_FIRST_STEP)
            g_new = min(lo.g * step, g_max)
            if g_new <= lo.g * (1.0 + 1e-9):
                outcome = "bracket"
                break
        else:
            g_new = _cap_step(sorted((h for h in hist if admissible(h)), key=lambda h: h.g), need, aim)
            if not (lo.g * GAMMA_NUDGE <= g_new <= hi.g / GAMMA_NUDGE):  # outside the bracket: bisect
                g_new = float(np.sqrt(lo.g * hi.g))
        comp.set_kappa(pdat, g_new * H)
        st, solver = solve(True)
        if st not in OK_STATUSES or comp.w.value is None:  # numerical failure: nudge once
            g_new *= GAMMA_NUDGE
            comp.set_kappa(pdat, g_new * H)
            st, solver = solve(False)
            n_failed += 1
        if st not in OK_STATUSES or comp.w.value is None:
            outcome = f"solve_failed:{st}"
            break
        w_new = clean_weights(np.asarray(comp.w.value).ravel(), pdat.w0)
        cur = _Point(g_new, vol(w_new), w_new, comp.binding(pdat, with_part), _objective(comp), st, solver)
        hist.append(cur)
        if admissible(cur):
            hi = cur if hi is None or cur.g < hi.g else hi
        else:
            lo = cur if cur.g > lo.g else lo
    if hi is not None:
        best = hi
    elif outcome in ("below_target", "target", "no_alpha"):
        best = cur
    else:  # never reached an admissible book: the most risk-averse one solved
        best = max(hist, key=lambda h: h.g)
    path = ", ".join(f"{h.g:.3g}:{h.vol:.4f}/{float(np.abs(h.w).sum()):.2f}" for h in hist)
    note = (
        f"gamma cap search ({outcome}): gamma {best.g:.3f} (base {gamma:.3f}), vol {best.vol:.4f} vs cap "
        f"{target:.4f}, gross {float(np.abs(best.w).sum()):.3f} after {len(hist)} solves "
        f"[gamma:vol/gross {path}]"
    )
    return GammaSearchResult(
        w=best.w,
        binding=best.binding,
        obj=best.obj,
        gamma=float(best.g),
        status=best.status,
        solver=best.solver,
        outcome=outcome,
        n_extra=len(hist) - 1 + n_failed,
        note=note,
    )


__all__ = [
    "GammaSearchResult",
    "caps_bind",
    "clean_weights",
    "closed_form_gamma",
    "gamma_cap_search",
    "gamma_search",
    "position_caps_bind",
]
