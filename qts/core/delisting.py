"""Delistings by reason: one rule shared by the labels and the backtest.

A held name that stops trading for good (``MarketData.delisting_dates`` or the
causal gap rule, see :mod:`qts.alpha.targets`) earns one final return relative
to its last close.  What that return is depends on *why* the name left:

==============  ================================================  ==========
reason          what happened                                     default
==============  ================================================  ==========
``acquisition`` acquired / merged into another company: holders   0 %
                receive cash or stock worth about the last price
``performance`` bankruptcy, liquidation, exchange-initiated       -30 %
                (regulatory) delisting (Shumway 1997)
``voluntary``   the issuer left the exchange (going dark, move    -10 %
                to OTC): trading continues, illiquid
``unknown``     no reason on file (also every name without        -30 %
                ``MarketData.delisting_reasons``)
==============  ================================================  ==========

The defaults live in :class:`qts.core.config.CostConfig`
(``delisting_return_<reason>``).  The return is a return **on the stock**: a
long of weight ``w > 0`` earns ``w * r`` and a short of weight ``w < 0`` earns
``w * r`` as well, i.e. it *gains* when ``r < 0``.  Two knobs change the booked
(position-side) return, both applied by :func:`position_returns`:

* ``delisting_short_gain_haircut`` in ``[0, 1]`` scales a short's gain on a
  negative return (1 = symmetric; 0 = shorts get nothing).  A short's loss on a
  positive return and every long are untouched.
* ``delisting_against_position = True`` applies the reason's return against the
  position's sign -- a loss for longs and shorts alike.  This is the accounting
  used before 2026-11-12 (``delisting_haircut``); kept to reproduce old runs.

Point in time: the reason is read when the delisting itself is decided (the day
after the vendor's last trading date).  The vendor action it comes from may be
dated a few days later (:data:`qts.data.sharadar.DELISTING_REASON_WINDOW_DAYS`),
so the *size* of the booked return can rest on a record published a few days
after the booking day; no signal, weight or trade depends on it.  ``MarketData.slice_until`` blanks the
reason of a name whose delisting date is after the cut.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from qts.core.config import CostConfig
    from qts.core.types import MarketData

#: Reason vocabulary (``MarketData.delisting_reasons`` values).
DELISTING_REASONS: tuple[str, ...] = ("acquisition", "performance", "voluntary", "unknown")
#: Which reason wins when several vendor actions sit at one delisting (first = strongest).
REASON_PRECEDENCE: tuple[str, ...] = ("performance", "acquisition", "voluntary", "unknown")
UNKNOWN = "unknown"
SIDES: tuple[str, ...] = ("long", "short")


def reasons_by_name(md: MarketData) -> np.ndarray:
    """Per ticker of ``md.tickers``: its delisting reason (object array of
    :data:`DELISTING_REASONS`); ``"unknown"`` where ``md.delisting_reasons`` is
    absent, does not list the ticker, or holds a value outside the vocabulary."""
    n = len(md.tickers)
    out = np.full(n, UNKNOWN, dtype=object)
    src = md.delisting_reasons
    if src is None or len(src) == 0:
        return out
    s = src[~src.index.duplicated(keep="last")].reindex(md.tickers)
    vals = np.asarray(s.to_numpy(dtype=object), dtype=object)
    ok = np.array([isinstance(v, str) and v in DELISTING_REASONS for v in vals.tolist()], dtype=bool)
    out[ok] = vals[ok]
    return out


def returns_by_name(md: MarketData, returns: Mapping[str, float]) -> np.ndarray:
    """Per ticker of ``md.tickers``: the delisting return on the stock for its
    reason.  ``returns``: reason -> return (``CostConfig.delisting_returns()``);
    a missing reason falls back to ``returns["unknown"]``."""
    dflt = float(returns.get(UNKNOWN, 0.0))
    table = {k: float(returns.get(k, dflt)) for k in DELISTING_REASONS}
    return np.array([table[str(r)] for r in reasons_by_name(md).tolist()], dtype=float)


def position_returns(
    stock_return: np.ndarray,
    weight: np.ndarray,
    short_gain_haircut: float = 1.0,
    against_position: bool = False,
) -> np.ndarray:
    """The return to multiply each weight by (so P&L = ``weight * result``).

    Default: the stock's return, with a short's gain (``weight < 0`` and
    ``stock_return < 0``) scaled by ``short_gain_haircut``.  With
    ``against_position``: ``stock_return * sign(weight)`` (the pre-2026-11-12
    both-sides rule; ``short_gain_haircut`` is not used)."""
    r = np.asarray(stock_return, dtype=float)
    w = np.asarray(weight, dtype=float)
    if against_position:
        return r * np.sign(w) + 0.0
    gain = (w < 0) & (r < 0)
    return np.where(gain, r * float(short_gain_haircut), r) + 0.0  # + 0.0: no negative zero


def cost_position_returns(cost: CostConfig, stock_return: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """:func:`position_returns` with the two knobs read from ``cost``."""
    return position_returns(
        stock_return,
        weight,
        short_gain_haircut=float(cost.delisting_short_gain_haircut),
        against_position=bool(cost.delisting_against_position),
    )


def summarize_delistings(
    events: list[dict[str, Any]], n_days: int | None = None, periods_per_year: int = 252
) -> dict[str, Any]:
    """Counts and P&L of ``BacktestResult.diagnostics["delistings"]`` by reason and side.

    Returns ``{"n", "pnl", "pnl_ann", "by_reason": {reason: {"n", "pnl", "pnl_ann",
    "long": {"n", "pnl"}, "short": {"n", "pnl"}}}}``.  ``pnl`` is the sum of the
    events' ``pnl`` (weight x booked return: a fraction of that day's NAV, so the
    sum is an arithmetic return contribution); ``pnl_ann = pnl * periods_per_year /
    n_days`` (``None`` without ``n_days``).  Events written before the reason was
    recorded count as ``unknown`` with ``pnl = weight * haircut_return``."""

    def ann(x: float) -> float | None:
        return None if not n_days else float(x) * periods_per_year / int(n_days)

    by: dict[str, Any] = {
        r: {"n": 0, "pnl": 0.0, "long": {"n": 0, "pnl": 0.0}, "short": {"n": 0, "pnl": 0.0}}
        for r in DELISTING_REASONS
    }
    total = 0.0
    for ev in events:
        w = float(ev.get("weight", 0.0))
        reason = str(ev.get("reason", UNKNOWN))
        if reason not in by:
            reason = UNKNOWN
        pnl_raw = ev.get("pnl")
        pnl = float(pnl_raw) if pnl_raw is not None else w * float(ev.get("haircut_return", 0.0))
        side = "long" if w > 0 else "short"
        slot = by[reason]
        slot["n"] += 1
        slot["pnl"] += pnl
        slot[side]["n"] += 1
        slot[side]["pnl"] += pnl
        total += pnl
    for slot in by.values():
        slot["pnl_ann"] = ann(slot["pnl"])
    return {"n": len(events), "pnl": total, "pnl_ann": ann(total), "by_reason": by}


def format_delisting_line(summary: Mapping[str, Any]) -> str:
    """Report-card line: ``acquisition N (P&L x%)  performance N (...)  ...`` with
    annualised P&L (total P&L when the summary carries no annualisation)."""
    parts: list[str] = []
    by = cast(Mapping[str, Any], summary.get("by_reason", {}))
    for r in DELISTING_REASONS:
        slot = by.get(r)
        if not slot or not slot.get("n"):
            continue
        v = slot.get("pnl_ann")
        txt = f"{100 * float(v):+.2f}%/yr" if v is not None else f"{100 * float(slot['pnl']):+.2f}%"
        parts.append(
            f"{r} {int(slot['n'])} (P&L {txt}; {int(slot['long']['n'])}L/{int(slot['short']['n'])}S)"
        )
    return "  ".join(parts) if parts else "none"


__all__ = [
    "DELISTING_REASONS",
    "REASON_PRECEDENCE",
    "SIDES",
    "UNKNOWN",
    "cost_position_returns",
    "format_delisting_line",
    "position_returns",
    "reasons_by_name",
    "returns_by_name",
    "summarize_delistings",
]
