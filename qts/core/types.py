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

import math
from dataclasses import dataclass, field, fields, replace
from typing import Any, ClassVar, cast

import numpy as np
import pandas as pd

from qts.core.risk_fill import fill_specific_var

Wide = pd.DataFrame  # index: date, columns: ticker
Long = pd.DataFrame  # index: MultiIndex(date, ticker), columns: feature


@dataclass
class MarketData:
    """Wide daily panels for a universe.  All price/volume frames share the same index/columns.

    Prices
    ------
    ``close`` is the split- **and** dividend-adjusted close: use it for
    returns (``returns``), momentum, volatility and anything else that
    compares prices across time.  ``close_raw`` is the *as-traded* close
    (what was on the tape that day; changes discontinuously at splits): use
    it, through :meth:`trade_price`, for anything that is a *level* on one
    date -- the $5 price screen, dollar volume (``volume`` is in actual
    shares traded that day), share counts, fixed per-share fees, and
    per-share fundamentals (B/P = book per share / raw close).  When
    ``close_raw`` is ``None`` (e.g. a source that only has adjusted prices)
    :meth:`trade_price` falls back to ``close`` -- historically wrong across
    splits, so real-data loaders should provide it.

    ``shares_outstanding`` is the actual share count on each date (consistent
    with ``close_raw``: ``market_cap ~= close_raw * shares_outstanding``).

    Industry
    --------
    ``industry`` is a static ticker -> label map.  ``industry_history`` is an
    optional *dated* classification (wide panel date x ticker of labels; a
    row holds the label in force from that date on).  Consumers (risk
    exposures, alpha industry demeaning, label residualisation) should call
    :meth:`industry_as_of` so that a reclassification is not applied before
    it happened.
    """

    close: Wide  # split/dividend-adjusted close
    volume: Wide  # shares traded (actual shares, not split-adjusted)
    market_cap: Wide  # USD
    industry: pd.Series  # ticker -> industry label (static fallback)
    spread_bps: Wide | None = None  # quoted half-spread proxy in bps
    short_interest_ratio: Wide | None = None  # short interest / shares out
    borrow_fee: Wide | None = None  # annualised borrow fee (fraction)
    book_to_price: Wide | None = None
    earnings_yield: Wide | None = None
    gross_profitability: Wide | None = None
    asset_growth: Wide | None = None
    extra: dict[str, Wide] = field(default_factory=dict[str, Wide])
    #: Optional boolean panel (date x ticker): True where the name is in the
    #: investable universe *as of that date* (set by ``qts.data.universe.apply_mask``).
    #: ``None`` means "every name with a price is eligible".  Every cross-sectional
    #: statistic in the alpha layer (ranks, z-scores, labels, IC, emitted scores)
    #: is computed over ``eligibility()`` only, so future universe membership can
    #: never leak into the cross-section at ``t``.
    eligible: Wide | None = None
    #: As-traded (unadjusted) close; see the class docstring.  Aligned with ``close``.
    close_raw: Wide | None = None
    #: Actual shares outstanding per date (consistent with ``close_raw``).  Aligned with ``close``.
    shares_outstanding: Wide | None = None
    #: Dated industry classification (date x ticker labels, ascending dates; the
    #: index need not match ``close``).  ``None`` -> the static ``industry`` applies.
    industry_history: Wide | None = None
    #: Optional dated delisting events: ticker -> last trading date (``NaT`` =
    #: known not delisted).  Vendors provide it (CRSP ``dlstdt``, Norgate
    #: "delisted" date, Sharadar TICKERS/ACTIONS); ParquetStore dataset
    #: ``delisting_dates``.  Used by the label builder
    #: (:func:`qts.alpha.targets.forward_returns`) to tell a true delisting from a
    #: temporary halt without looking ahead.  ``None`` -> the causal gap rule
    #: (``AlphaConfig.delisting_gap_days``) applies.  Tickers absent from the
    #: series are treated as unknown (causal rule).  Not date-indexed:
    #: :meth:`slice_until` keeps it but blanks (``NaT``) events after the cut.
    delisting_dates: pd.Series | None = None

    #: Panels that share ``close``'s index/columns (checked by ``validate``).
    ALIGNED_PANELS: ClassVar[tuple[str, ...]] = ("volume", "market_cap", "close_raw", "shares_outstanding")

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

    def delisting_positions(self) -> np.ndarray:
        """Per ticker: index position in ``dates`` of the vendor's last trading
        date (``delisting_dates``).  ``inf`` = covered and known not delisted
        (``NaT``); ``nan`` = not covered (the causal gap rule
        ``AlphaConfig.delisting_gap_days`` applies).  Shared by the label builder
        (:mod:`qts.alpha.targets`) and the simulator so both classify a missing
        close the same way."""
        n = len(self.tickers)
        out = np.full(n, np.nan)
        dd_src = self.delisting_dates
        if dd_src is None:
            return out
        dd = pd.to_datetime(dd_src.reindex(self.tickers))
        covered = np.asarray(self.tickers.isin(dd_src.index))
        known = dd.notna().to_numpy()
        pos = self.dates.searchsorted(pd.DatetimeIndex(dd[known]), side="right") - 1
        out[covered] = np.inf
        out[np.flatnonzero(known)] = pos
        return out

    @property
    def returns(self) -> Wide:
        """Simple total returns from the *adjusted* close."""
        return self.close.pct_change(fill_method=None)

    def trade_price(self) -> Wide:
        """As-traded price: ``close_raw`` if present, else ``close`` (adjusted)."""
        return self.close_raw if self.close_raw is not None else self.close

    @property
    def dollar_volume(self) -> Wide:
        """USD traded per day = as-traded price x actual shares traded."""
        return self.trade_price() * self.volume

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index  # type: ignore[return-value]

    @property
    def tickers(self) -> pd.Index:
        return self.close.columns

    def industry_as_of(self, date: pd.Timestamp, fallback: bool = True) -> pd.Series:
        """Industry label of every ticker in force at ``date`` (point-in-time).

        With ``industry_history`` the label is the last row dated ``<= date``
        (NaN entries -- not yet classified -- are filled from the static
        ``industry`` when ``fallback``); without it the static ``industry`` is
        returned.  Indexed by ``tickers``.
        """
        static = self.industry.reindex(self.tickers)
        if self.industry_history is None:
            return static
        hist = self.industry_history.loc[: pd.Timestamp(date)]
        if hist.empty:
            row = pd.Series(np.nan, index=self.tickers, dtype=object)
        else:
            row = hist.ffill().iloc[-1].reindex(self.tickers)
        if fallback:
            row = row.where(row.notna(), static)
        row.name = "industry"
        row.index.name = static.index.name
        return row

    def validate(self) -> None:
        for name in self.ALIGNED_PANELS:
            f = getattr(self, name)
            if f is None:
                continue
            if not f.index.equals(self.close.index) or not f.columns.equals(self.close.columns):
                raise ValueError(f"{name} does not align with close")
        if not self.close.index.is_monotonic_increasing:
            raise ValueError("dates must be ascending")
        missing = set(self.close.columns) - set(self.industry.index)
        if missing:
            raise ValueError(f"industry missing for {sorted(missing)[:5]}...")
        if self.industry_history is not None:
            ih = self.industry_history
            if not isinstance(ih.index, pd.DatetimeIndex) or not ih.index.is_monotonic_increasing:
                raise ValueError("industry_history needs an ascending DatetimeIndex")
        if self.delisting_dates is not None and not isinstance(cast(object, self.delisting_dates), pd.Series):
            raise ValueError("delisting_dates must be a Series ticker -> last trading date")

    def slice_until(self, date: pd.Timestamp) -> MarketData:
        """Point-in-time view: every date-indexed panel cut at ``<= date``."""

        def cut(f: Wide | None) -> Wide | None:
            return None if f is None else f.loc[:date]

        changes: dict[str, Any] = {}
        for fl in fields(self):
            v = getattr(self, fl.name)
            if fl.name == "industry":
                continue
            if fl.name == "delisting_dates":
                # ticker-indexed: events after the cut are not known yet
                if v is not None:
                    dd = cast(pd.Series, pd.to_datetime(cast(pd.Series, v)))
                    changes[fl.name] = dd.where(dd <= pd.Timestamp(date))
                continue
            if fl.name == "extra":
                changes["extra"] = {k: cut(x) for k, x in v.items()}
            else:
                changes[fl.name] = cut(v)
        return replace(self, **changes)


@dataclass(frozen=True)
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
    alignment_vector: pd.Series | None = None
    alignment_var: float = 0.0
    factor_groups: dict[str, list[str]] = field(default_factory=dict[str, list[str]])

    def __post_init__(self) -> None:
        """Cheap structural checks (frozen contract): every exposure column has a
        row and a column in ``factor_cov``, the alignment variance is a finite
        non-negative number and ``factor_groups`` only names exposure columns."""
        cols = set(self.exposures.columns)
        missing = cols - set(self.factor_cov.index) | cols - set(self.factor_cov.columns)
        if missing:
            raise ValueError(
                f"RiskModelOutput: factor_cov lacks exposure columns {sorted(map(str, missing))[:5]}"
            )
        if not (math.isfinite(self.alignment_var) and self.alignment_var >= 0):
            raise ValueError(
                f"RiskModelOutput: alignment_var must be finite and >= 0, got {self.alignment_var}"
            )
        unknown = {c for names in self.factor_groups.values() for c in names} - cols
        if unknown:
            raise ValueError(
                f"RiskModelOutput: factor_groups name non-exposure columns {sorted(unknown)[:5]}"
            )

    @property
    def tickers(self) -> pd.Index:
        return self.exposures.index

    @property
    def factors(self) -> pd.Index:
        return self.exposures.columns

    def covariance(self, tickers: pd.Index | None = None, include_alignment: bool = True) -> pd.DataFrame:
        t = self.tickers if tickers is None else tickers
        # tickers outside the model universe get zero factor exposure and the
        # out-of-model specific variance (90th percentile, qts.core.risk_fill; the
        # optimizer's rule, so callers that do not intersect first stay consistent)
        B = self.exposures.reindex(t).fillna(0.0).to_numpy(dtype=np.float64)
        F = self.factor_cov.loc[self.factors, self.factors].to_numpy(dtype=np.float64)
        D = np.diag(fill_specific_var(self.specific_var, t))
        cov = B @ F @ B.T + D
        if include_alignment and self.alignment_vector is not None and self.alignment_var > 0:
            y = self.alignment_vector.reindex(t).fillna(0.0).to_numpy(dtype=np.float64)[:, None]
            cov = cov + self.alignment_var * (y @ y.T)
        return pd.DataFrame(cov, index=t, columns=t)

    def portfolio_variance(self, w: pd.Series, include_alignment: bool = True) -> float:
        t = w.index
        B = self.exposures.reindex(t).fillna(0.0).to_numpy(dtype=np.float64)
        F = self.factor_cov.loc[self.factors, self.factors].to_numpy(dtype=np.float64)
        d = fill_specific_var(self.specific_var, t)  # out-of-model rule as in covariance()
        wv = w.to_numpy(dtype=np.float64)
        fw = B.T @ wv
        v = float(fw @ F @ fw + (wv**2 * d).sum())
        if include_alignment and self.alignment_vector is not None and self.alignment_var > 0:
            y = self.alignment_vector.reindex(t).fillna(0.0).to_numpy(dtype=np.float64)
            v += self.alignment_var * float(y @ wv) ** 2
        return v


@dataclass(frozen=True)
class AlphaOutput:
    """Alpha layer output as of one date.

    ``score``: cross-sectional z-score per ticker (mean 0, sd 1 across the
    universe).  ``expected_return``: Grinold-scaled expected residual return
    per period (``IC * resid_vol * score``), same horizon as the target.
    ``horizon_days`` and ``half_life_days`` are used by the portfolio layer
    for Garleanu-Pedersen shrinkage.  ``components`` optionally keeps the
    per-sleeve scores for attribution; ``components_half_life`` holds the
    per-sleeve signal half-lives (trading days) estimated as of ``date``.
    """

    date: pd.Timestamp
    score: pd.Series
    expected_return: pd.Series
    horizon_days: int
    half_life_days: float
    components: dict[str, pd.Series] = field(default_factory=dict[str, pd.Series])
    ic_estimate: float = float("nan")
    components_half_life: dict[str, float] = field(default_factory=dict[str, float])

    def __post_init__(self) -> None:
        if int(self.horizon_days) < 1:
            raise ValueError(f"AlphaOutput: horizon_days must be >= 1, got {self.horizon_days}")


def to_long(wide: Wide, name: str) -> pd.Series:
    """Wide (date x ticker) -> long Series indexed by (date, ticker)."""
    # a DataFrame with a flat column index stacks to a Series (the stubs return Series | DataFrame)
    s = cast(pd.Series, wide.stack(future_stack=True))
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
