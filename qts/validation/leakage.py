"""Leakage checks that are enforced by code, not convention (BUILD_PLAN s.8).

* :func:`pit_shift_test` — the universal point-in-time test: a feature builder
  applied to the full history and to ``md.slice_until(cut_date)`` must agree on
  every date ``<= cut_date``.  Any use of future rows (``shift(-k)``,
  full-sample statistics, centred rolling windows, look-ahead fills) breaks
  this equality.
* :func:`label_overlap_check` — train/test label windows must not overlap.
* :func:`planted_oracle_test` — runs the backtest with a feature equal to the
  future return.  The pipeline must flag the resulting Sharpe as suspect
  (> 2).  This makes operational the fact that DSR/PBO *cannot* detect
  leakage: a leaked oracle scores DSR = 1.00, so the "too good" gate and the
  PIT test are the only defences.
* :func:`feature_registry` — decorator that records which ``MarketData``
  panels a feature function reads and refuses functions whose source contains
  ``.shift(-k)`` with ``k > 0`` (best-effort AST inspection).

Limits of the AST check (documented, by design): it inspects only the
decorated function's own source (not helpers it calls), only literal negative
shift arguments (``shift(-1)``, ``shift(periods=-5)``), and cannot see
``shift(k)`` with a variable ``k``, ``.iloc`` tricks, ``rolling(center=True)``
or ``bfill``.  It is a cheap tripwire; :func:`pit_shift_test` is the real gate
and must be run on every registered feature.
"""
from __future__ import annotations

import ast
import functools
import hashlib
import inspect
import textwrap
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from qts.core.types import MarketData, Wide

SUSPECT_SHARPE = 2.0


class LeakageError(AssertionError):
    pass


# --------------------------------------------------------------------------- PIT test
def pit_shift_test(build_fn: Callable[[MarketData], Wide], md: MarketData, cut_date,
                   atol: float = 1e-10, raise_on_fail: bool = False) -> bool:
    """True iff ``build_fn`` is point-in-time at ``cut_date``.

    Compares ``build_fn(md)`` restricted to dates ``<= cut_date`` against
    ``build_fn(md.slice_until(cut_date))``.  NaN positions must agree and
    non-NaN values must be equal within ``atol``.
    """
    cut = pd.Timestamp(cut_date)
    full = build_fn(md)
    part = build_fn(md.slice_until(cut))
    full_c = full.loc[:cut]
    part_c = part.loc[:cut]
    ok = True
    reason = ""
    if not full_c.index.equals(part_c.index) or not full_c.columns.equals(part_c.columns):
        ok, reason = False, "index/columns differ between full and truncated builds"
    else:
        a, b = full_c.to_numpy(dtype=float), part_c.to_numpy(dtype=float)
        nan_a, nan_b = np.isnan(a), np.isnan(b)
        if not np.array_equal(nan_a, nan_b):
            ok, reason = False, f"NaN pattern differs on {int((nan_a != nan_b).sum())} cells"
        else:
            diff = np.abs(np.where(nan_a, 0.0, a - b))
            if diff.size and diff.max() > atol:
                bad = np.argwhere(diff > atol)
                d0 = full_c.index[bad[0][0]]
                ok, reason = False, f"{len(bad)} cells differ; first at {d0.date()} (max |diff| {diff.max():.3g})"
    if not ok and raise_on_fail:
        raise LeakageError(f"pit_shift_test failed at cut {cut.date()}: {reason}")
    return ok


def label_overlap_check(train_dates, test_dates, horizon: int, raise_on_fail: bool = False,
                        calendar: Optional[pd.DatetimeIndex] = None) -> bool:
    """True iff no train label window [s, s+horizon] overlaps any test label window.

    ``horizon`` is counted in trading days on ``calendar`` (default: business
    days spanning both inputs, unioned with the inputs themselves)."""
    train = pd.DatetimeIndex(train_dates)
    test = pd.DatetimeIndex(test_dates)
    if len(train) == 0 or len(test) == 0:
        return True
    both = train.union(test)
    if calendar is None:
        calendar = pd.bdate_range(both.min(), both.max())
    cal = pd.DatetimeIndex(calendar).union(both).sort_values()
    pos = pd.Series(np.arange(len(cal)), index=cal)
    tr = pos.loc[train].to_numpy()
    te = pos.loc[test].to_numpy()
    h = int(horizon)
    te_lo, te_hi = te.min(), te.max() + h
    # a test set may be non-contiguous (CPCV); check against each contiguous run
    runs = np.split(te, np.where(np.diff(te) > 1)[0] + 1)
    ok = True
    for run in runs:
        lo, hi = run.min(), run.max() + h
        if np.any((tr <= hi) & (tr + h >= lo)):
            ok = False
            break
    if not ok and raise_on_fail:
        raise LeakageError(f"train label windows overlap test windows (horizon={horizon})")
    return ok


# --------------------------------------------------------------------------- oracle
def forward_return(md: MarketData, horizon: int) -> Wide:
    """LABEL (not a feature): close-to-close return from t to t+horizon."""
    return md.close.shift(-horizon) / md.close - 1.0


def simple_rank_backtest(md: MarketData, feature: Wide, horizon: int = 20, quantile: float = 0.2,
                         periods_per_year: int = 252) -> Dict[str, float]:
    """Minimal long-short quantile backtest used by :func:`planted_oracle_test`.

    Each day rank ``feature``, go long the top ``quantile`` and short the
    bottom ``quantile`` equal-weighted, hold for one day (positions from the
    signal at ``t`` earn ``r_{t+1}``).  Returns annual Sharpe and mean return.
    """
    r = md.returns
    rk = feature.rank(axis=1, pct=True)
    w = pd.DataFrame(0.0, index=feature.index, columns=feature.columns)
    w[rk >= 1 - quantile] = 1.0
    w[rk <= quantile] = -1.0
    w = w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    pnl = (w.shift(1) * r).sum(axis=1).dropna()
    pnl = pnl[pnl != 0]
    sr = float(pnl.mean() / pnl.std() * np.sqrt(periods_per_year)) if len(pnl) > 2 and pnl.std() > 0 else 0.0
    return {"sharpe": sr, "mean_daily": float(pnl.mean()) if len(pnl) else 0.0, "n_obs": int(len(pnl))}


def planted_oracle_test(backtest_fn: Optional[Callable[[MarketData, Wide], object]], md: MarketData,
                        horizon: int = 20, suspect_sharpe: float = SUSPECT_SHARPE) -> Dict[str, object]:
    """Run ``backtest_fn(md, oracle_feature)`` with the oracle = future return.

    ``backtest_fn`` may return a float Sharpe or a dict containing ``"sharpe"``.
    Returns ``{"sharpe", "flagged", "passed"}`` where ``passed`` means the
    pipeline correctly flagged the oracle (Sharpe > ``suspect_sharpe``).
    If the oracle is *not* flagged the pipeline's suspect gate is broken (or
    the backtest is not using the feature) and ``passed`` is False.
    """
    fn = backtest_fn or (lambda m, f: simple_rank_backtest(m, f, horizon))
    oracle = forward_return(md, horizon)
    out = fn(md, oracle)
    sr = float(out["sharpe"]) if isinstance(out, dict) else float(out)
    flagged = sr > suspect_sharpe
    return {"sharpe": sr, "flagged": flagged, "passed": flagged,
            "message": ("oracle correctly flagged as suspect" if flagged else
                        "ORACLE NOT FLAGGED: the suspect-Sharpe gate or the backtest wiring is broken")}


# --------------------------------------------------------------------------- registry
FEATURE_REGISTRY: Dict[str, Dict[str, object]] = {}

_PANEL_NAMES = {"close", "volume", "market_cap", "industry", "spread_bps", "short_interest_ratio",
                "borrow_fee", "book_to_price", "earnings_yield", "gross_profitability", "asset_growth",
                "returns", "dollar_volume", "extra"}


def _negative_shift_calls(tree: ast.AST):
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "shift"):
            continue
        args = [node.args[0]] if node.args else []
        args += [kw.value for kw in node.keywords if kw.arg == "periods"]
        for a in args:
            val = None
            if isinstance(a, ast.UnaryOp) and isinstance(a.op, ast.USub) and isinstance(a.operand, ast.Constant):
                val = -a.operand.value
            elif isinstance(a, ast.Constant) and isinstance(a.value, (int, float)):
                val = a.value
            if val is not None and val < 0:
                found.append((getattr(node, "lineno", -1), val))
    return found


def _panel_reads(tree: ast.AST, md_param: Optional[str]):
    reads = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if (md_param is None or node.value.id == md_param) and node.attr in _PANEL_NAMES:
                reads.add(node.attr)
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == "extra":
            key = node.slice
            if isinstance(key, ast.Constant):
                reads.add(f"extra[{key.value!r}]")
    return sorted(reads)


def feature_registry(fn: Optional[Callable] = None, *, name: Optional[str] = None, reads=None):
    """Register a feature function and refuse obvious look-ahead.

    Usage::

        @feature_registry
        def mom_12_1(md: MarketData) -> Wide: ...

    Records ``FEATURE_REGISTRY[name] = {"reads": [...], "source_sha": ...,
    "fn": fn}``.  Raises :class:`LeakageError` at decoration time if the
    function source calls ``.shift(-k)`` with ``k > 0``.  See module docstring
    for the limits of the static check.
    """
    def deco(f):
        try:
            src = textwrap.dedent(inspect.getsource(f))
            tree = ast.parse(src)
        except (OSError, TypeError, SyntaxError):  # e.g. defined in a REPL
            src, tree = "", None
        if tree is not None:
            bad = _negative_shift_calls(tree)
            if bad:
                ln, k = bad[0]
                raise LeakageError(f"{f.__name__}: .shift({k}) at line {ln} looks ahead; refused by feature_registry")
        params = list(inspect.signature(f).parameters)
        md_param = params[0] if params else None
        detected = _panel_reads(tree, md_param) if tree is not None else []
        entry = {
            "fn": f,
            "reads": sorted(set(detected) | set(reads or [])),
            "source_sha": hashlib.sha1(src.encode()).hexdigest()[:12] if src else None,
            "static_check": tree is not None,
        }
        FEATURE_REGISTRY[name or f.__name__] = entry

        @functools.wraps(f)
        def wrapper(*a, **k):
            return f(*a, **k)
        wrapper.__feature_entry__ = entry
        return wrapper

    return deco(fn) if fn is not None else deco
