"""Return cleaning shared by the risk model, the label builder and the simulator.

Why this exists (docs/notes/risk_outlier_fix_2026-11-10.md)
-----------------------------------------------------------
Vendor price panels contain a handful of absurd daily returns: a restated
adjusted close, a sub-penny print, a split applied to one day but not the
next.  The synthetic market never produces them, so nothing downstream was
defended.  On real data (Sharadar, 2018-03-05) one such return entered the
sqrt-cap-weighted cross-sectional factor regression, blew up that day's
factor returns, and through the EWMA factor covariance (``vol_half_life`` =
84 days) the ex-ante vol of the book went from 5 % to 5,110,263 %.  The
target-vol optimizer liquidated everything and the book stayed empty for
~19 half-lives.

Three rules, all using only same-day and past data (PIT-safe)
--------------------------------------------------------------
(a) **data-error rule** -- a return that cannot be a traded price move:
    ``r > cap_up`` (default +300 %) or ``r < cap_down`` (default -95 %).  The
    bounds are asymmetric on purpose: genuine -80 % days exist (fraud
    revelations, failed trials, bank runs), genuine +300 % days essentially
    do not for an investable name, and a -95 % day on a name that was
    eligible the day before is, in practice, always a bad print.  Optionally
    (``min_prev_price``) a return whose previous *as-traded* close was below
    a floor (default 0.25 USD) is also dropped: at sub-penny levels the tick
    size alone creates +-50 % "returns".  Flagged cells become NaN.
(b) **cross-sectional winsorisation** for *estimation inputs only*: per day,
    clip at ``median +- mad_k * 1.4826 * MAD`` (default ``mad_k`` = 8).
    This is the Barra USE4 practice of winsorising the regression's dependent
    variable so one fat-tailed name does not dominate a factor return.  It is
    deliberately loose (8 robust sigmas): it is a guard against the next
    data error that rule (a) does not catch, not a model of returns.
(c) **flags** are returned with the cleaned panel so callers can count and
    report what was changed.

Who applies what
----------------
* Risk model (factor regression, specific-return EWMA, APCA companion): (a)
  with the price floor + (b).  Estimation only -- nothing a book earns.
* Alpha labels (``qts.alpha.targets``): (a) without the price floor, as a bad
  *price*; a genuine crash is a genuine label, so no winsorising (the label
  builder already winsorises the cross-section at 1 % for the regression).
* Simulator P&L: (a) without the price floor, as a bad *price*: the day is
  treated like a halt (no return that day; the next day's return is measured
  from the last good close), counted in ``diagnostics['n_data_error_returns']``.
  A +50,000 % print is not tradable P&L, and neither is its -99.8 % reversal.

:func:`flag_bad_prices` is the price-level form of rule (a) used by the
label builder and the simulator: a spike-and-reverse pair (``+99,900 %`` then
``-99.9 %``) is *one* bad close, not two bad returns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

#: ``1 / Phi^{-1}(3/4)``: scales the median absolute deviation to a Gaussian sd.
MAD_TO_SD = 1.4826

DEFAULT_CAP_UP = 3.0
DEFAULT_CAP_DOWN = -0.95
DEFAULT_MIN_PREV_PRICE = 0.25
DEFAULT_WINSOR_MAD_K = 8.0


@dataclass(frozen=True)
class CleanedReturns:
    """Result of :func:`clean_returns`.

    ``returns``: the cleaned panel (same index / columns as the input).
    ``data_error``: boolean panel, True where rule (a) set the cell to NaN.
    ``winsorized``: boolean panel, True where rule (b) clipped the cell.
    """

    returns: pd.DataFrame
    data_error: pd.DataFrame
    winsorized: pd.DataFrame

    @property
    def n_data_errors(self) -> int:
        return int(self.data_error.to_numpy().sum())

    @property
    def n_winsorized(self) -> int:
        return int(self.winsorized.to_numpy().sum())

    def data_errors_by_date(self) -> pd.Series:
        """Count of rule-(a) cells per date (dates with none are dropped)."""
        s = self.data_error.sum(axis=1)
        return s[s > 0].astype(int)

    def data_errors_by_name(self) -> pd.Series:
        """Count of rule-(a) cells per ticker (tickers with none are dropped)."""
        s = self.data_error.sum(axis=0)
        return s[s > 0].astype(int)


def data_error_mask(
    returns: pd.DataFrame,
    close_raw: pd.DataFrame | None = None,
    *,
    cap_up: float = DEFAULT_CAP_UP,
    cap_down: float = DEFAULT_CAP_DOWN,
    min_prev_price: float | None = DEFAULT_MIN_PREV_PRICE,
) -> pd.DataFrame:
    """Rule (a): True where ``r > cap_up``, ``r < cap_down`` or (when
    ``close_raw`` and ``min_prev_price`` are given) the previous day's
    as-traded close was below ``min_prev_price``.  NaN returns are never
    flagged.  ``close_raw`` is aligned to ``returns`` and shifted by one day,
    so only past data is used."""
    if cap_up <= 0 or not (-1.0 <= cap_down < 0):
        raise ValueError(f"need cap_up > 0 and -1 <= cap_down < 0, got {cap_up}, {cap_down}")
    r = returns.to_numpy(dtype=np.float64)
    finite = np.isfinite(r)
    bad = finite & ((r > cap_up) | (r < cap_down))
    if close_raw is not None and min_prev_price is not None and min_prev_price > 0:
        prev = (
            close_raw.reindex(index=returns.index, columns=returns.columns)
            .shift(1)
            .to_numpy(dtype=np.float64)
        )
        bad |= finite & np.isfinite(prev) & (prev < float(min_prev_price))
    return pd.DataFrame(bad, index=returns.index, columns=returns.columns)


def flat_run_mask(returns: pd.DataFrame, tol: float = 1e-6, min_days: int = 20) -> pd.DataFrame:
    """Non-trading days (docs/notes/risk_new_listing_fix_2026-11-10.md): True on
    day ``t`` when ``|r| < tol`` on each of the ``min_days`` days ending at ``t``.

    A SPAC sitting at its trust value, a halted name whose last price is carried
    forward and a stale vendor price all look like this: a long run of returns
    that are exactly (or to rounding) zero.  They are not observations of the
    stock's risk.  The rule is causal -- day ``t`` is flagged from data ``<= t``
    only, so the first ``min_days - 1`` days of a run are *not* flagged (they were
    not known to be a run yet) -- and a NaN return breaks a run.  ``min_days <= 0``
    returns an all-False panel."""
    r = returns.to_numpy(dtype=np.float64)
    out = np.zeros(r.shape, dtype=bool)
    k = int(min_days)
    if k > 0 and r.shape[0] >= k:
        with np.errstate(invalid="ignore"):
            flat = np.isfinite(r) & (np.abs(r) < float(tol))
        if flat.any():
            c = np.cumsum(flat, axis=0, dtype=np.int64)
            run = c.copy()
            run[k:] = c[k:] - c[:-k]
            out = flat & (run >= k)
    return pd.DataFrame(out, index=returns.index, columns=returns.columns)


def winsorize_cross_section(
    returns: pd.DataFrame, mad_k: float = DEFAULT_WINSOR_MAD_K
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rule (b): per row (date), clip at ``median +- mad_k * MAD_TO_SD * MAD``.
    Rows with fewer than 20 finite values or a zero MAD are left unchanged.
    Returns ``(clipped, flags)``."""
    if mad_k <= 0:
        raise ValueError(f"need mad_k > 0, got {mad_k}")
    r = returns.to_numpy(dtype=np.float64)
    out = r.copy()
    flags = np.zeros(r.shape, dtype=bool)
    n = np.isfinite(r).sum(axis=1)
    rows = np.flatnonzero(n >= 20)
    med = np.full(r.shape[0], np.nan)
    mad = np.full(r.shape[0], np.nan)
    if len(rows):
        med[rows] = np.nanmedian(r[rows], axis=1)
        mad[rows] = np.nanmedian(np.abs(r[rows] - med[rows, None]), axis=1)
    ok = (n >= 20) & np.isfinite(mad) & (mad > 0)
    if ok.any():
        half = float(mad_k) * MAD_TO_SD * mad[ok]
        lo = (med[ok] - half)[:, None]
        hi = (med[ok] + half)[:, None]
        sub = r[ok]
        fin = np.isfinite(sub)
        clipped = np.clip(sub, lo, hi)
        flags[ok] = fin & (clipped != sub)
        out[ok] = np.where(fin, clipped, sub)
    idx, cols = returns.index, returns.columns
    return pd.DataFrame(out, index=idx, columns=cols), pd.DataFrame(flags, index=idx, columns=cols)


def clean_returns(
    returns: pd.DataFrame,
    close_raw: pd.DataFrame | None = None,
    *,
    cap_up: float = DEFAULT_CAP_UP,
    cap_down: float = DEFAULT_CAP_DOWN,
    min_prev_price: float | None = DEFAULT_MIN_PREV_PRICE,
    mad_k: float | None = DEFAULT_WINSOR_MAD_K,
) -> CleanedReturns:
    """Rules (a) and (b) in that order (``mad_k=None`` skips (b)); see the
    module docstring.  Point-in-time: row ``t`` of the output depends only on
    rows ``<= t`` of the inputs (rule (b) is purely cross-sectional, rule (a)
    uses ``close_raw`` lagged one day)."""
    bad = data_error_mask(returns, close_raw, cap_up=cap_up, cap_down=cap_down, min_prev_price=min_prev_price)
    cleaned = returns.mask(bad)
    if mad_k is None:
        flags = pd.DataFrame(False, index=returns.index, columns=returns.columns)
    else:
        cleaned, flags = winsorize_cross_section(cleaned, mad_k)
    return CleanedReturns(returns=cleaned, data_error=bad, winsorized=flags)


def flag_bad_prices(
    close: pd.DataFrame,
    *,
    cap_up: float = DEFAULT_CAP_UP,
    cap_down: float = DEFAULT_CAP_DOWN,
    max_passes: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Price-level rule (a): find the closes that produce data-error returns and
    blank them, so that consumers who work on prices (labels, simulator) see a
    missing price -- the same path as a halt -- rather than an absurd move.

    The return of day ``t`` is measured from the last non-missing close before
    ``t`` (as the simulator does), so after a close is blanked the next day's
    return spans the gap.  Each pass blanks the *first* day of every run of
    consecutive flagged returns per name and recomputes; a spike-and-reverse
    pair is therefore one bad close (its reversal disappears with it), while a
    level shift that persists (restated history) keeps being flagged on the
    following day and is blanked day by day for up to ``max_passes`` days.
    Only same-day and past data decide whether day ``t`` is blanked.

    Returns ``(close_clean, bad)`` with ``bad`` the boolean panel of blanked closes.
    """
    c = close.to_numpy(dtype=np.float64).copy()
    bad = np.zeros(c.shape, dtype=bool)
    for _ in range(max(int(max_passes), 1)):
        r = _gap_returns(c)
        flag = np.isfinite(r) & ((r > cap_up) | (r < cap_down))
        if not flag.any():
            break
        first = flag.copy()
        first[1:] &= ~flag[:-1]  # first day of each run of flagged returns
        bad |= first
        c[first] = np.nan
    return (
        pd.DataFrame(c, index=close.index, columns=close.columns),
        pd.DataFrame(bad, index=close.index, columns=close.columns),
    )


def _gap_returns(c: NDArray[np.float64]) -> NDArray[np.float64]:
    """``c_t / (last finite close before t) - 1``; NaN where ``c_t`` is missing
    or no earlier close exists."""
    T, _ = c.shape
    out = np.full(c.shape, np.nan)
    if T < 2:
        return out
    last = pd.DataFrame(c).ffill().to_numpy(dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[1:] = c[1:] / last[:-1] - 1.0
    out[~np.isfinite(c)] = np.nan
    return out
