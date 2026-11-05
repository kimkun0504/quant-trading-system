"""DPP machinery: cost expressions with their array constants turned into
``cp.Parameter`` so one compiled problem can be re-solved with new values.

* :func:`_leaf_sig` -- structure signature of an expression tree (array
  constants contribute shape only); a changed signature forces a rebuild.
* :func:`_constants` -- the array constants of a tree, in tree order.
* :class:`_Parametrizer` -- copies an expression with every array constant
  replaced by a parameter, recording a *recipe* per parameter: ``("c", j)`` =
  constant ``j``; ``("mul", j, k)`` = constant ``j`` * constant ``k``.
* :func:`_assign` -- re-evaluates the recipes on a new day's constants and
  sets the parameter values (False when shapes / signs no longer fit).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import cvxpy as cp
import numpy as np


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
        kw: dict[str, Any] = (
            {"nonneg": True} if np.all(v >= 0) else ({"nonpos": True} if np.all(v <= 0) else {})
        )
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
