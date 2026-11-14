"""Point-in-time investable universe.

:func:`build_universe_mask` returns a boolean wide panel (date x ticker) that
is True when a name is tradable *as of that date* using only information
available on or before it:

* as-traded price (``MarketData.trade_price()``: the unadjusted close when
  available) >= ``min_price`` -- a split-adjusted close understates the price
  a name actually traded at before a later split, so screening on it drops
  names that were tradable at the time;
* trailing ``adv_window``-day average dollar volume (``MarketData.dollar_volume``
  = as-traded price x actual shares traded) >= ``min_adv_usd``;
* market cap above the ``exclude_bottom_size_quantile`` quantile of the
  eligible cross-section (an NYSE-breakpoint proxy: the quantile is computed
  on names that already pass the price/liquidity screens, which drops the
  microcap tail where anomaly returns live but cannot be captured);
* within the top ``top_n_by_mcap`` by market cap among the survivors -- or, with
  ``UniverseConfig.rank_offset = k > 0``, ranked ``k + 1 .. k + top_n_by_mcap``
  (the ``k`` largest survivors are skipped; same daily PIT ranks, e.g. the mid caps
  1001-2000 of ``qts check midcap``).

Membership is refreshed on a fixed calendar grid and held constant in between:
with ``UniverseConfig.refresh_anchor = "month_start"`` (default) on the first
trading day of every calendar month (plus the panel's first date, so the mask
is defined from the start), so the refresh dates do not depend on the load
``start`` (audit 2 #18; the hysteresis state still needs
``exit_after_n_fails`` refreshes to forget the start); ``"panel_start"`` is the
legacy grid every ``refresh_days`` trading days from the first panel date.  To limit turnover, a name that is
in the universe leaves only after failing ``UniverseConfig.exit_after_n_fails``
(default 3) consecutive refreshes; a name enters as soon as it passes.  A name whose close is NaN
(not yet listed / delisted) is never in the universe regardless of hysteresis.

:func:`survivorship_check` is a cheap sanity test on the *source*: a
delisting-complete price history has tickers whose series end in NaN before
the panel's last date.  If none do, the source is almost surely
survivorship-biased (e.g. "current S&P 500 constituents from yfinance") and
every backtest on it is inflated.
"""
from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Dict, Optional, TypedDict

import numpy as np
import pandas as pd

from qts.core.config import UniverseConfig
from qts.core.types import MarketData, Wide


def raw_eligibility(md: MarketData, cfg: UniverseConfig) -> Wide:
    """Per-day eligibility *before* refresh/hysteresis (all PIT quantities)."""
    price, mcap = md.trade_price(), md.market_cap
    adv = md.dollar_volume.rolling(cfg.adv_window, min_periods=max(5, cfg.adv_window // 2)).mean()
    base = (price >= cfg.min_price) & (adv >= cfg.min_adv_usd) & mcap.notna() & md.close.notna()
    m_elig = mcap.where(base)
    q = cfg.exclude_bottom_size_quantile
    if q > 0:
        cut = m_elig.quantile(q, axis=1)            # row-wise, on eligible names only
        base = base & (mcap.ge(cut, axis=0))
        m_elig = mcap.where(base)
    rank = m_elig.rank(axis=1, ascending=False, method="first")
    offset = int(getattr(cfg, "rank_offset", 0) or 0)
    if offset > 0:
        # mid caps etc.: skip the ``offset`` largest survivors, then take the next top_n
        return (base & (rank > offset) & (rank <= offset + cfg.top_n_by_mcap)).fillna(False).astype(bool)
    return (base & (rank <= cfg.top_n_by_mcap)).fillna(False).astype(bool)


def refresh_positions(dates: pd.DatetimeIndex, refresh_days: int, anchor: str = "month_start") -> np.ndarray:
    """Positions of the universe refresh dates in ``dates``.

    ``"month_start"``: position 0 and the first date of every calendar month;
    ``"panel_start"``: every ``refresh_days`` positions from 0 (legacy)."""
    if len(dates) == 0:
        return np.zeros(0, dtype=int)
    if anchor == "panel_start":
        return np.arange(0, len(dates), int(refresh_days))
    if anchor != "month_start":
        raise ValueError(f"refresh_anchor must be 'month_start' or 'panel_start', got {anchor!r}")
    per = dates.to_period("M")
    new_month = np.r_[True, per[1:] != per[:-1]]
    return np.flatnonzero(new_month)


def build_universe_mask(md: MarketData, cfg: Optional[UniverseConfig] = None,
                        refresh_days: Optional[int] = None, exit_after_n_fails: Optional[int] = None) -> Wide:
    """PIT universe mask (see module docstring).  ``refresh_days`` and
    ``exit_after_n_fails`` default to ``cfg.refresh_days`` /
    ``cfg.exit_after_n_fails``; passing them overrides the config."""
    cfg = cfg or UniverseConfig()
    refresh_days = int(cfg.refresh_days if refresh_days is None else refresh_days)
    exit_after_n_fails = int(cfg.exit_after_n_fails if exit_after_n_fails is None else exit_after_n_fails)
    if refresh_days < 1 or exit_after_n_fails < 1:
        raise ValueError("refresh_days and exit_after_n_fails must be >= 1")
    raw = raw_eligibility(md, cfg)
    dates = raw.index
    refresh_pos = refresh_positions(pd.DatetimeIndex(dates), refresh_days, getattr(cfg, "refresh_anchor", "month_start"))
    R = raw.iloc[refresh_pos].to_numpy()
    N = R.shape[1]
    in_u = np.zeros(N, dtype=bool)
    fails = np.zeros(N, dtype=int)
    out = np.zeros_like(R)
    for i in range(R.shape[0]):
        passed = R[i]
        fails = np.where(passed, 0, fails + 1)
        in_u = passed | (in_u & (fails < exit_after_n_fails))
        out[i] = in_u
    at_refresh = pd.DataFrame(out, index=dates[refresh_pos], columns=raw.columns)
    mask = at_refresh.reindex(dates).ffill().fillna(False).astype(bool)
    mask &= md.close.notna()
    mask.columns.name = raw.columns.name
    return mask


def apply_mask(md: MarketData, mask: Wide) -> MarketData:
    """Attach the point-in-time universe ``mask`` to ``md`` as ``md.eligible``.

    **No panel is masked** (audit 2 #6): prices, volume, market cap, shares
    outstanding, fundamentals, short interest and ``extra`` keep their full
    history, so trailing statistics of a name that just entered the universe
    (63-day turnover / Amihud / dollar volume, risk liquidity, the cost model's
    ADV) are its true trailing values instead of NaN / truncated windows.
    Universe membership acts only through ``md.eligibility()`` (``eligible`` and
    a close at ``t``), which every cross-sectional statistic and every emitted
    score uses (features ``process``, composite, labels, IC, risk-style
    standardisation, the pipeline's alpha wrapper, the simulator).  Code that
    still needs "NaN outside the universe" panels (e.g. a risk model whose
    estimation universe is "has a market cap") should use
    :func:`masked_levels`.

    Tickers that are never in the universe are dropped **from the column set
    only** -- a storage convenience, not an eligibility statement.
    ``industry_history`` is column-restricted; ``extra`` panels whose columns
    are not tickers (e.g. ``true_style_factor_returns``) pass through unchanged.
    """
    keep = mask.columns[mask.any(axis=0)]
    m = mask[keep].astype(bool)
    tickers = set(md.tickers)

    def cols(f: Optional[Wide]) -> Optional[Wide]:
        return None if f is None else f[keep]

    def cols_extra(f: Wide) -> Wide:
        if not set(f.columns) <= tickers:
            return f                       # not a ticker panel
        return f.reindex(columns=keep)

    ih = md.industry_history
    return replace(
        md,
        close=md.close[keep], volume=cols(md.volume), market_cap=cols(md.market_cap),
        industry=md.industry.reindex(keep), spread_bps=cols(md.spread_bps),
        short_interest_ratio=cols(md.short_interest_ratio), borrow_fee=cols(md.borrow_fee),
        book_to_price=cols(md.book_to_price), earnings_yield=cols(md.earnings_yield),
        gross_profitability=cols(md.gross_profitability), asset_growth=cols(md.asset_growth),
        extra={k: cols_extra(v) for k, v in md.extra.items()},
        eligible=m & md.close[keep].notna(),
        close_raw=cols(md.close_raw),
        shares_outstanding=cols(md.shares_outstanding),
        industry_history=None if ih is None else ih.reindex(columns=keep),
        # delisting_dates kept as is: a ticker absent from it means "not covered"
    )


def masked_levels(md: MarketData) -> MarketData:
    """View of ``md`` with ``market_cap``, ``volume`` and ``shares_outstanding``
    NaN where the name is not eligible (the pre-audit-2 ``apply_mask`` levels).

    For consumers whose estimation universe is still defined by "has a market
    cap" rather than ``md.eligibility()`` (the pipeline passes this view to the
    risk model so its estimation universe is unchanged)."""
    if md.eligible is None:
        return md
    e = md.eligibility()

    def cut(f: Optional[Wide]) -> Optional[Wide]:
        return None if f is None else f.where(e.reindex_like(f).fillna(False).astype(bool))

    return replace(md, market_cap=cut(md.market_cap), volume=cut(md.volume),
                   shares_outstanding=cut(md.shares_outstanding))


class SurvivorshipReport(TypedDict):
    """:func:`survivorship_check`."""

    n_tickers: int
    n_delisted: int
    n_late_listed: int
    suspect: bool


def survivorship_check(md: MarketData, warn: bool = True) -> SurvivorshipReport:
    """Flag a likely survivorship-biased source.

    Returns ``{"n_tickers", "n_delisted", "n_late_listed", "suspect"}``.  A
    ticker is "delisted" if its last valid close precedes the panel's last
    date; "late listed" if its first valid close follows the panel's first
    date.  ``suspect`` is True when no ticker is delisted.
    """
    last_valid = md.close.apply(lambda s: s.last_valid_index())
    first_valid = md.close.apply(lambda s: s.first_valid_index())
    n_del = int((last_valid < md.dates[-1]).sum())
    n_late = int((first_valid > md.dates[0]).sum())
    suspect = n_del == 0
    if suspect and warn:
        warnings.warn("survivorship_check: no ticker has trailing NaNs — the price source contains no "
                      "delistings and is probably survivorship-biased (backtests will be inflated).",
                      UserWarning, stacklevel=2)
    return {"n_tickers": int(md.close.shape[1]), "n_delisted": n_del, "n_late_listed": n_late, "suspect": suspect}


def universe_turnover(mask: Wide) -> pd.Series:
    """Fraction of names entering or leaving per date (diagnostic)."""
    m = mask.astype(int)
    ch = (m.diff().abs().sum(axis=1)) / m.sum(axis=1).replace(0, np.nan)
    return ch.fillna(0.0)
