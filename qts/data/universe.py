"""Point-in-time investable universe.

:func:`build_universe_mask` returns a boolean wide panel (date x ticker) that
is True when a name is tradable *as of that date* using only information
available on or before it:

* price >= ``min_price``;
* trailing ``adv_window``-day average dollar volume >= ``min_adv_usd``;
* market cap above the ``exclude_bottom_size_quantile`` quantile of the
  eligible cross-section (an NYSE-breakpoint proxy: the quantile is computed
  on names that already pass the price/liquidity screens, which drops the
  microcap tail where anomaly returns live but cannot be captured);
* within the top ``top_n_by_mcap`` by market cap among the survivors.

Membership is refreshed every ``refresh_days`` (default 21 trading days) and
held constant in between.  To limit turnover, a name that is in the universe
leaves only after failing ``exit_after_n_fails`` (default 3) consecutive
refreshes; a name enters as soon as it passes.  A name whose close is NaN
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
from typing import Dict, Optional

import numpy as np
import pandas as pd

from qts.core.config import UniverseConfig
from qts.core.types import MarketData, Wide


def raw_eligibility(md: MarketData, cfg: UniverseConfig) -> Wide:
    """Per-day eligibility *before* refresh/hysteresis (all PIT quantities)."""
    close, mcap = md.close, md.market_cap
    adv = md.dollar_volume.rolling(cfg.adv_window, min_periods=max(5, cfg.adv_window // 2)).mean()
    base = (close >= cfg.min_price) & (adv >= cfg.min_adv_usd) & mcap.notna() & close.notna()
    m_elig = mcap.where(base)
    q = cfg.exclude_bottom_size_quantile
    if q > 0:
        cut = m_elig.quantile(q, axis=1)            # row-wise, on eligible names only
        base = base & (mcap.ge(cut, axis=0))
        m_elig = mcap.where(base)
    rank = m_elig.rank(axis=1, ascending=False, method="first")
    return (base & (rank <= cfg.top_n_by_mcap)).fillna(False).astype(bool)


def build_universe_mask(md: MarketData, cfg: Optional[UniverseConfig] = None, refresh_days: int = 21,
                        exit_after_n_fails: int = 3) -> Wide:
    cfg = cfg or UniverseConfig()
    raw = raw_eligibility(md, cfg)
    dates = raw.index
    refresh_pos = np.arange(0, len(dates), refresh_days)
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
    """Return a MarketData with every panel NaN where ``mask`` is False.

    Prices are kept intact (so returns and labels remain computable for
    names that just left the universe); market cap, volume and features are
    masked.  Tickers that are never in the universe are dropped **from the
    column set only** — that is a storage convenience, not an eligibility
    statement.  The per-date membership is stored on ``md.eligible`` and every
    cross-sectional operation downstream must go through
    ``MarketData.eligibility()`` so that a name that enters the universe in
    2015 is *not* in the 2010 cross-section (future-membership leakage).
    """
    keep = mask.columns[mask.any(axis=0)]
    m = mask[keep].astype(bool)

    def cut(f):
        if f is None:
            return None
        return f[keep].where(m)

    return MarketData(
        close=md.close[keep], volume=cut(md.volume), market_cap=cut(md.market_cap),
        industry=md.industry.reindex(keep), spread_bps=cut(md.spread_bps),
        short_interest_ratio=cut(md.short_interest_ratio), borrow_fee=cut(md.borrow_fee),
        book_to_price=cut(md.book_to_price), earnings_yield=cut(md.earnings_yield),
        gross_profitability=cut(md.gross_profitability), asset_growth=cut(md.asset_growth),
        extra={k: cut(v) for k, v in md.extra.items()},
        eligible=m & md.close[keep].notna(),
    )


def survivorship_check(md: MarketData, warn: bool = True) -> Dict[str, object]:
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
