"""Shared data conventions and typed containers for the quant trading system.

Conventions (every module must follow these):

* **Wide panels** are ``pd.DataFrame`` indexed by ``pd.DatetimeIndex`` (trading
  days, ascending, tz-naive) with one column per ticker.  Used for prices,
  returns, volume, market cap, spreads, short interest, etc.
* **Long panels** are ``pd.DataFrame`` with a two-level ``MultiIndex``
  ``(date, ticker)`` and one column per feature.  Used for feature matrices
  and alpha scores.
* **Returns** are simple (not log) close-to-close returns for the day,
  ``r_t = P_t / P_{t-1} - 1``.  Anything that uses a return "as of date t"
  may only use ``r_s`` for ``s <= t``.
* **Point-in-time rule**: any object produced "as of date t" must be a
  function of data with timestamp ``<= t`` only.  Labels (forward returns)
  are the one exception and are *never* used as features.
* **Weights** are fractions of NAV (long positive, short negative).
  Trades ``x`` are weight changes ``w_post - w_pre``.

``MarketData`` bundles the wide panels; ``RiskModelOutput`` is the contract
between the risk layer and the portfolio layer; ``AlphaOutput`` is the
contract between the alpha layer and the portfolio layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

Wide = pd.DataFrame  # index: date, columns: ticker
Long = pd.DataFrame  # index: MultiIndex(date, ticker), columns: feature


@dataclass
class MarketData:
    """Wide daily panels for a universe.  All frames share the same index/columns."""

    close: Wide                       # adjusted close
    volume: Wide                      # shares traded
    market_cap: Wide                  # USD
    industry: pd.Series               # ticker -> industry label (quasi-static)
    spread_bps: Optional[Wide] = None  # quoted half-spread proxy in bps
    short_interest_ratio: Optional[Wide] = None  # short interest / shares out
    borrow_fee: Optional[Wide] = None  # annualised borrow fee (fraction)
    book_to_price: Optional[Wide] = None
    earnings_yield: Optional[Wide] = None
    gross_profitability: Optional[Wide] = None
    asset_growth: Optional[Wide] = None
    extra: Dict[str, Wide] = field(default_factory=dict)
    #: Optional boolean panel (date x ticker): True where the name is in the
    #: investable universe *as of that date* (set by ``qts.data.universe.apply_mask``).
    #: ``None`` means "every name with a price is eligible".  Every cross-sectional
    #: statistic in the alpha layer (ranks, z-scores, labels, IC, emitted scores)
    #: is computed over ``eligibility()`` only, so future universe membership can
    #: never leak into the cross-section at ``t``.
    eligible: Optional[Wide] = None

    # ------------------------------------------------------------------ derived
    def eligibility(self) -> Wide:
        """Per-date boolean panel: in the universe at ``t`` *and* has a close at ``t``.

        Falls back to ``close.notna()`` when no universe mask has been applied.
        This is the only function the alpha layer should use to decide which
        names enter a cross-section.
        """
        alive = self.close.notna()
        if self.eligible is None:
            return alive
        e = self.eligible.reindex(index=alive.index, columns=alive.columns)
        return e.fillna(False).astype(bool) & alive
    @property
    def returns(self) -> Wide:
        return self.close.pct_change(fill_method=None)

    @property
    def dollar_volume(self) -> Wide:
        return self.close * self.volume

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    @property
    def tickers(self) -> pd.Index:
        return self.close.columns

    def validate(self) -> None:
        for name in ("volume", "market_cap"):
            f = getattr(self, name)
            if not f.index.equals(self.close.index) or not f.columns.equals(self.close.columns):
                raise ValueError(f"{name} does not align with close")
        if not self.close.index.is_monotonic_increasing:
            raise ValueError("dates must be ascending")
        missing = set(self.close.columns) - set(self.industry.index)
        if missing:
            raise ValueError(f"industry missing for {sorted(missing)[:5]}...")

    def slice_until(self, date: pd.Timestamp) -> "MarketData":
        """Point-in-time view: everything with timestamp <= date."""
        def cut(f):
            return None if f is None else f.loc[:date]
        return MarketData(
            close=cut(self.close), volume=cut(self.volume), market_cap=cut(self.market_cap),
            industry=self.industry, spread_bps=cut(self.spread_bps),
            short_interest_ratio=cut(self.short_interest_ratio), borrow_fee=cut(self.borrow_fee),
            book_to_price=cut(self.book_to_price), earnings_yield=cut(self.earnings_yield),
            gross_profitability=cut(self.gross_profitability), asset_growth=cut(self.asset_growth),
            extra={k: cut(v) for k, v in self.extra.items()},
            eligible=cut(self.eligible),
        )


@dataclass
class RiskModelOutput:
    """Risk model as of one date.

    ``exposures``: tickers x factors (B).  ``factor_cov``: factors x factors (F).
    ``specific_var``: per-ticker idiosyncratic variance (diag D), daily units.
    Total covariance = B F B' + D.  ``factor_returns`` is the history of
    estimated factor returns up to (and including) ``date``.
    ``alignment_var`` is the optional alpha-alignment term (Saxena-Stubbs):
    a unit-norm vector ``y`` and scalar ``nu`` such that the augmented
    covariance is ``B F B' + D + nu * y y'``.
    ``factor_groups`` tells consumers which *role* each factor column plays
    (keys ``"market"``, ``"industry"``, ``"style"``, ``"statistical"``,
    ``"alignment"``; values are lists of column names of ``exposures``).  The
    portfolio layer uses it to pick the beta style, the industry block and the
    style block without string matching; an empty dict means "unknown".
    """

    date: pd.Timestamp
    exposures: pd.DataFrame
    factor_cov: pd.DataFrame
    specific_var: pd.Series
    factor_returns: pd.DataFrame
    alignment_vector: Optional[pd.Series] = None
    alignment_var: float = 0.0
    factor_groups: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def tickers(self) -> pd.Index:
        return self.exposures.index

    @property
    def factors(self) -> pd.Index:
        return self.exposures.columns

    def covariance(self, tickers: Optional[pd.Index] = None, include_alignment: bool = True) -> pd.DataFrame:
        t = self.tickers if tickers is None else tickers
        # tickers outside the model universe get zero factor exposure and the
        # median specific variance (conservative; callers should intersect first)
        B = self.exposures.reindex(t).fillna(0.0).values
        F = self.factor_cov.loc[self.factors, self.factors].values
        D = np.diag(self.specific_var.reindex(t).fillna(self.specific_var.median()).values)
        S = B @ F @ B.T + D
        if include_alignment and self.alignment_vector is not None and self.alignment_var > 0:
            y = self.alignment_vector.reindex(t).fillna(0.0).values[:, None]
            S = S + self.alignment_var * (y @ y.T)
        return pd.DataFrame(S, index=t, columns=t)

    def portfolio_variance(self, w: pd.Series, include_alignment: bool = True) -> float:
        t = w.index
        B = self.exposures.reindex(t).fillna(0.0).values
        F = self.factor_cov.loc[self.factors, self.factors].values
        d = self.specific_var.reindex(t).fillna(self.specific_var.median()).values
        fw = B.T @ w.values
        v = float(fw @ F @ fw + (w.values ** 2 * d).sum())
        if include_alignment and self.alignment_vector is not None and self.alignment_var > 0:
            y = self.alignment_vector.reindex(t).fillna(0.0).values
            v += self.alignment_var * float(y @ w.values) ** 2
        return v


@dataclass
class AlphaOutput:
    """Alpha layer output as of one date.

    ``score``: cross-sectional z-score per ticker (mean 0, sd 1 across the
    universe).  ``expected_return``: Grinold-scaled expected residual return
    per period (``IC * resid_vol * score``), same horizon as the target.
    ``horizon_days`` and ``half_life_days`` are used by the portfolio layer
    for Garleanu-Pedersen shrinkage.  ``components`` optionally keeps the
    per-sleeve scores for attribution.
    """

    date: pd.Timestamp
    score: pd.Series
    expected_return: pd.Series
    horizon_days: int
    half_life_days: float
    components: Dict[str, pd.Series] = field(default_factory=dict)
    ic_estimate: float = float("nan")


def to_long(wide: Wide, name: str) -> pd.Series:
    """Wide (date x ticker) -> long Series indexed by (date, ticker)."""
    s = wide.stack(future_stack=True)
    s.index.names = ["date", "ticker"]
    s.name = name
    return s


def to_wide(long: pd.Series) -> Wide:
    return long.unstack("ticker")


def assert_wide_aligned(*frames: Wide) -> None:
    ref = frames[0]
    for f in frames[1:]:
        if not (f.index.equals(ref.index) and f.columns.equals(ref.columns)):
            raise ValueError("wide frames are not aligned")
