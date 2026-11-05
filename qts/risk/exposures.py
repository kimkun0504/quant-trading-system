"""Factor exposure construction (Barra/USE4 style).

Factor block layout for every exposure matrix (tickers x factors):

* ``market``      -- the USE4 *Country* factor: unit exposure for every stock.
* one dummy per industry label in ``MarketData.industry`` (0/1, rows sum to 1).
* style factors from ``RiskConfig.styles`` -- standardized Barra-style.

Style definitions (all computed from ``MarketData`` with data ``<= t`` only)
------------------------------------------------------------------------------
beta
    Slope of a rolling 252-day regression of the stock's return on the
    equal-weighted market return, shrunk toward 1 (Blume / Vasicek style
    prior):  ``beta = 1 + (1 - shrink) * (beta_hat - 1)``, ``shrink = 1/3``.
    Rolling cov/var is computed from rolling means of products so the whole
    panel is handled in a few vectorized passes.
size
    ``log(market_cap)``.
momentum
    12-1 month return: ``P_{t-21} / P_{t-252} - 1`` (skips the most recent
    month, which belongs to reversal).
value
    ``log(book_to_price)`` when available, otherwise ``earnings_yield``.
residual_vol
    252-day idiosyncratic volatility relative to the market regression:
    ``sqrt(var(r) - cov(r, m)^2 / var(m))``.
liquidity
    ``log`` of the 63-day mean turnover, where turnover is proxied by
    ``dollar_volume / market_cap`` (= volume / shares outstanding).
short_term_reversal
    21-day return ``P_t / P_{t-21} - 1``.
quality
    ``gross_profitability``.
investment
    ``-asset_growth`` (conservative investment loads positively).
short_interest
    ``short_interest_ratio``.

Standardization (USE4 Methodology Notes, section on descriptor standardization):
cap-weighted mean 0 and equal-weighted standard deviation 1, outliers trimmed
at +-3 standard deviations.  We apply: cap-weighted z-score -> clip to [-3, 3]
-> re-standardize (so the sd is ~1 after trimming) -> fill missing with 0
(a missing style exposure is treated as "average", the USE4 convention).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from qts.core import stats
from qts.core.config import RiskConfig
from qts.core.crosssection import zscore
from qts.core.types import MarketData

MARKET_FACTOR = "market"
BETA_WINDOW = 252
RESVOL_WINDOW = 252
MOMENTUM_LONG = 252
MOMENTUM_SKIP = 21
REVERSAL_WINDOW = 21
LIQUIDITY_WINDOW = 63
BETA_SHRINK = 1.0 / 3.0
WINSOR_SD = 3.0

SUPPORTED_STYLES = [
    "beta",
    "size",
    "momentum",
    "value",
    "residual_vol",
    "liquidity",
    "short_term_reversal",
    "quality",
    "investment",
    "short_interest",
]


# --------------------------------------------------------------------------- raw styles
def raw_style_panels(md: MarketData, styles: Iterable[str]) -> dict[str, pd.DataFrame]:
    """Compute un-standardized style descriptors as wide panels (date x ticker).

    Every descriptor is causal (rolling windows / lags only), so slicing a
    panel at date ``t`` uses data ``<= t`` only."""
    styles = list(styles)
    unknown = set(styles) - set(SUPPORTED_STYLES)
    if unknown:
        raise ValueError(f"unsupported styles: {sorted(unknown)}")
    out: dict[str, pd.DataFrame] = {}
    returns = md.returns
    need_reg = {"beta", "residual_vol"} & set(styles)
    beta_hat: pd.DataFrame | None = None
    resid_var: pd.DataFrame | None = None
    if need_reg:
        # regression on the equal-weighted market return over the names eligible
        # at each date (md.eligibility(): universe member with a price).  An
        # all-columns mean would include names whose *only* reason to be in the
        # panel is that apply_mask kept them for entering the universe later --
        # a future-membership look-ahead (lag investigation 2026-11-05).
        market = returns.where(md.eligibility()).mean(axis=1)
        reg = stats.rolling_beta(returns, market, BETA_WINDOW, min_periods=BETA_WINDOW // 4)
        beta_hat, resid_var = reg.beta, reg.resid_var
    for s in styles:
        if s == "beta":
            assert beta_hat is not None
            out[s] = 1.0 + (1.0 - BETA_SHRINK) * (beta_hat - 1.0)
        elif s == "size":
            out[s] = np.log(md.market_cap.where(md.market_cap > 0))
        elif s == "momentum":
            out[s] = md.close.shift(MOMENTUM_SKIP) / md.close.shift(MOMENTUM_LONG) - 1.0
        elif s == "value":
            if md.book_to_price is not None:
                out[s] = np.log(md.book_to_price.where(md.book_to_price > 0))
            elif md.earnings_yield is not None:
                out[s] = md.earnings_yield.copy()
            else:
                out[s] = pd.DataFrame(np.nan, index=md.dates, columns=md.tickers)
        elif s == "residual_vol":
            assert resid_var is not None
            out[s] = np.sqrt(resid_var)
        elif s == "liquidity":
            turnover = md.dollar_volume / md.market_cap.replace(0.0, np.nan)
            out[s] = np.log(
                turnover.rolling(LIQUIDITY_WINDOW, min_periods=LIQUIDITY_WINDOW // 3)
                .mean()
                .where(lambda x: x > 0)
            )
        elif s == "short_term_reversal":
            out[s] = md.close / md.close.shift(REVERSAL_WINDOW) - 1.0
        elif s == "quality":
            out[s] = (
                md.gross_profitability.copy()
                if md.gross_profitability is not None
                else pd.DataFrame(np.nan, index=md.dates, columns=md.tickers)
            )
        elif s == "investment":
            out[s] = (
                -md.asset_growth
                if md.asset_growth is not None
                else pd.DataFrame(np.nan, index=md.dates, columns=md.tickers)
            )
        elif s == "short_interest":
            out[s] = (
                md.short_interest_ratio.copy()
                if md.short_interest_ratio is not None
                else pd.DataFrame(np.nan, index=md.dates, columns=md.tickers)
            )
    return out


def standardize_style(raw: pd.DataFrame, market_cap: pd.DataFrame) -> pd.DataFrame:
    """Barra standardization of a wide descriptor panel, row by row:
    cap-weighted mean 0 / equal-weighted sd 1, trim at +-3 sd, re-standardize,
    fill missing with 0."""
    raw = raw.replace([np.inf, -np.inf], np.nan)
    z = zscore(raw, weights=market_cap).clip(-WINSOR_SD, WINSOR_SD)
    z = zscore(z, weights=market_cap)
    return z.fillna(0.0)


def standardized_style_panels(md: MarketData, styles: Iterable[str]) -> dict[str, pd.DataFrame]:
    """Standardised style panels; the per-date cross-section (cap-weighted mean,
    equal-weighted sd) is restricted to ``md.eligibility()`` at that date, so a
    name kept in the column set only because it enters the universe later never
    moves today's standardisation (PIT; lag investigation 2026-11-05)."""
    raw = raw_style_panels(md, styles)
    elig = md.eligibility()
    return {
        k: standardize_style(v.where(elig.reindex_like(v).fillna(False).astype(bool)), md.market_cap)
        for k, v in raw.items()
    }


# --------------------------------------------------------------------------- assembly
def industry_dummies(industry: pd.Series, tickers: pd.Index) -> pd.DataFrame:
    """Ticker x industry 0/1 matrix (:func:`qts.core.stats.industry_dummies`, all industries)."""
    return stats.industry_dummies(industry, tickers)


def industry_panel(md: MarketData, dates: pd.DatetimeIndex | None = None) -> pd.DataFrame | None:
    """Dated industry labels (date x ticker) on ``dates`` (default ``md.dates``),
    or ``None`` when ``md.industry_history`` is absent (static classification).

    Row ``t`` equals ``md.industry_as_of(t)`` (last classification dated
    ``<= t``, unclassified names filled from the static ``md.industry``) --
    computed for all dates in one forward-fill instead of one call per date."""
    ih = md.industry_history
    if ih is None:
        return None
    dates = md.dates if dates is None else pd.DatetimeIndex(dates)
    ih = ih.reindex(columns=md.tickers)
    grid = ih.index.union(dates)
    lab = ih.reindex(grid).ffill().reindex(dates)
    static = md.industry.reindex(md.tickers)
    lab = lab.where(
        lab.notna(), pd.DataFrame([static.to_numpy()] * len(dates), index=dates, columns=md.tickers)
    )
    lab.columns = md.tickers
    return lab


def _industry_labels(md: MarketData) -> list[str]:
    """Every industry label the exposure matrices can carry (static + dated), sorted.

    This is the label set over the *whole* sample.  A per-date exposure frame
    carries only the labels already carried by an eligible name on or before
    that date (``ExposureHistory.industry_first_row``, second-pass finding 4)."""
    labs = set(md.industry.reindex(md.tickers).dropna().unique())
    if md.industry_history is not None:
        labs |= set(pd.unique(md.industry_history.reindex(columns=md.tickers).stack().dropna()))
    return sorted(labs)


def factor_names(md: MarketData, cfg: RiskConfig) -> list[str]:
    return [MARKET_FACTOR, *_industry_labels(md), *list(cfg.styles)]


class ExposureHistory(dict):
    """Exposure matrices for many dates stored as arrays, not one DataFrame per date.

    Layout (``T`` stored dates, ``N`` tickers, ``S`` styles, ``K = 1 + I + S``):

    * ``static``  -- ``(N, 1 + I)`` market column + industry dummies when the
      classification is static (``MarketData.industry``), stored once instead
      of ``T`` times;
    * ``industry_codes`` -- ``None`` for a static classification; with
      ``MarketData.industry_history`` (finding #43) a ``(T, N)`` int array of
      industry indices into ``factors[1:1 + I]`` (``-1`` = unclassified) as of
      each date (``MarketData.industry_as_of``), and ``static`` is then only the
      ``(N, 1)`` market column;
    * ``industry_first_row`` -- ``(I,)`` int: the first stored row on which an
      eligible name carries each label (static or dated classification).  The frame for row ``i`` carries
      only the industry columns with ``industry_first_row <= i`` (columns
      *as of* the date, audit second pass finding 4), so ``hist[t]`` never
      reveals a label introduced after ``t`` and a risk model fitted on the
      full sample equals one fitted on data truncated at ``t``.  ``factors``
      is the union over all dates;
    * ``styles``  -- ``(T, N, S)`` standardised style exposures (``dtype``,
      float64 by default; float32 halves the memory at ~1e-7 relative error);
    * ``valid``   -- ``(T, N)`` bool: ticker in the exposure universe on the date
      (``md.eligibility()`` and positive market cap).

    ``hist[date]`` / ``hist.get(date)`` / ``hist.as_of(date)`` rebuild the
    ``tickers x factors`` DataFrame for one date (``as_of`` = most recent stored
    date ``<= date``, O(log T) via ``searchsorted``) with exactly the values the
    former dict-of-DataFrames held.  The class subclasses ``dict`` purely so
    existing ``isinstance(x, dict)`` / ``.keys()`` / ``.items()`` consumers keep
    working; no frames are held in the dict storage itself.  At 1,500 names x
    3,780 days x (1 + 48 + 10) factors the arrays take ~0.5 GB (float64) /
    ~0.25 GB (float32) against ~2.7 GB for the frames (``memory_report``).
    """

    def __init__(
        self,
        dates: pd.DatetimeIndex,
        tickers: pd.Index,
        factors: pd.Index,
        static: np.ndarray,
        styles: np.ndarray,
        valid: np.ndarray,
        industry_codes: np.ndarray | None = None,
        n_industries: int = 0,
        industry_first_row: np.ndarray | None = None,
    ):
        super().__init__()
        self.dates = pd.DatetimeIndex(dates)
        self.tickers = pd.Index(tickers)
        self.factors = pd.Index(factors, name="factor")
        self.static = static
        self.styles = styles
        self.valid = valid
        self.industry_codes = industry_codes
        self.n_industries = int(n_industries)
        self.industry_first_row = industry_first_row
        self._pos = {d: i for i, d in enumerate(self.dates)}

    # ---------------------------------------------------------------- mapping
    def frame_at(self, i: int) -> pd.DataFrame:
        mask = self.valid[i]
        sty = self.styles[i][mask].astype(float, copy=False)
        if self.industry_codes is None:
            arr = np.concatenate([self.static[mask], sty], axis=1)
        else:
            codes = self.industry_codes[i][mask]
            ind = np.zeros((len(codes), self.n_industries))
            ok = codes >= 0
            ind[np.flatnonzero(ok), codes[ok]] = 1.0
            arr = np.concatenate([self.static[mask], ind, sty], axis=1)
        cols = self.factors
        if self.industry_first_row is not None:
            active = self.industry_first_row <= i
            if not active.all():  # labels not yet observed in the estimation universe
                keep = np.r_[
                    0,  # market
                    1 + np.flatnonzero(active),
                    np.arange(1 + self.n_industries, len(cols)),
                ]
                arr, cols = arr[:, keep], cols[keep]
        return pd.DataFrame(arr, index=self.tickers[mask], columns=cols)

    def __getitem__(self, date) -> pd.DataFrame:
        i = self._pos.get(pd.Timestamp(date))
        if i is None:
            raise KeyError(date)
        return self.frame_at(i)

    def get(self, date, default=None):
        i = self._pos.get(pd.Timestamp(date)) if date is not None else None
        return default if i is None else self.frame_at(i)

    def __contains__(self, date) -> bool:
        try:
            return pd.Timestamp(date) in self._pos
        except (TypeError, ValueError):
            return False

    def __iter__(self):
        return iter(self.dates)

    def __len__(self) -> int:
        return len(self.dates)

    def __bool__(self) -> bool:
        return len(self.dates) > 0

    def keys(self):  # type: ignore[override]
        return list(self.dates)

    def values(self):  # type: ignore[override]
        return (self.frame_at(i) for i in range(len(self.dates)))

    def items(self):  # type: ignore[override]
        return ((d, self.frame_at(i)) for i, d in enumerate(self.dates))

    def __eq__(self, other) -> bool:  # identity: the dict storage is empty by design
        return self is other

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return (
            f"ExposureHistory(T={len(self.dates)}, N={len(self.tickers)}, K={len(self.factors)}, "
            f"styles dtype={self.styles.dtype}, {self.nbytes / 1e6:.1f} MB)"
        )

    # ---------------------------------------------------------------- extras
    def as_of(self, date) -> pd.DataFrame:
        """Exposures on the most recent stored date ``<= date``."""
        i = int(self.dates.searchsorted(pd.Timestamp(date), side="right")) - 1
        if i < 0:
            raise KeyError(f"no exposures on or before {date}")
        return self.frame_at(i)

    @property
    def first_date(self) -> pd.Timestamp | None:
        return self.dates[0] if len(self.dates) else None

    @property
    def nbytes(self) -> int:
        codes = 0 if self.industry_codes is None else self.industry_codes.nbytes
        return int(self.static.nbytes + self.styles.nbytes + self.valid.nbytes + codes)

    def memory_report(self) -> dict[str, float]:
        """Bytes held by the arrays vs the dict-of-DataFrames they replace
        (values + index, measured on the stored frames' sizes)."""
        K = len(self.factors)
        n_valid = int(self.valid.sum())
        frames_values = n_valid * K * 8
        # per-frame overhead: ticker index (object, ~8 B pointer + shared strings) + BlockManager
        frames_index = n_valid * 8 + len(self.dates) * 2_000
        return {
            "tensor_bytes": float(self.nbytes),
            "frames_bytes_est": float(frames_values + frames_index),
            "ratio": float((frames_values + frames_index) / max(self.nbytes, 1)),
        }


def exposure_history(
    md: MarketData,
    dates: Iterable[pd.Timestamp] | None = None,
    cfg: RiskConfig | None = None,
    dtype=np.float64,
) -> ExposureHistory:
    """Exposure matrices for many dates, as an :class:`ExposureHistory`.

    All rolling statistics are computed *once* on the wide panels (vectorized),
    stacked into a ``(T, N, S)`` style array plus a static ``(N, 1 + I)``
    market/industry block (or, when ``md.industry_history`` is present, a
    ``(T, N)`` array of point-in-time industry codes, row ``t`` =
    ``md.industry_as_of(t)``); ``hist[date]`` rebuilds the per-date frame.  Because
    every descriptor is causal this is point-in-time correct.  A ticker is
    included on date ``t`` if it is eligible (``md.eligibility()``: universe
    member with a finite close) and has a positive market cap on ``t``.  Dates not in ``md.dates`` are skipped."""
    cfg = cfg or RiskConfig()
    dates = md.dates if dates is None else pd.DatetimeIndex(list(dates))
    styles = list(cfg.styles)
    panels = standardized_style_panels(md, styles)
    tickers_all = md.tickers
    N = len(tickers_all)
    rows = md.dates.get_indexer(dates)
    keep = rows >= 0
    rows = rows[keep]
    sel_dates = dates[keep]
    codes: np.ndarray | None = None
    static_codes = np.full(N, -1)
    if md.industry_history is None:  # static classification: fast path
        dummies = industry_dummies(md.industry, md.tickers)
        inds = list(dummies.columns)
        static = np.column_stack([np.ones(N)] + [dummies[c].values for c in dummies.columns])
        static_codes = np.where(dummies.values.any(axis=1), dummies.values.argmax(axis=1), -1)
    else:  # dated classification (finding #43)
        inds = _industry_labels(md)
        panel = industry_panel(md, sel_dates)
        assert panel is not None
        codes = (
            pd.Categorical(panel.to_numpy().ravel(), categories=inds)
            .codes.reshape(len(sel_dates), N)
            .astype(np.int32)
        )
        static = np.ones((N, 1))
    cols = pd.Index([MARKET_FACTOR, *inds, *styles], name="factor")
    if styles:
        style_arr = np.empty((len(rows), N, len(styles)), dtype=dtype)
        for j, st in enumerate(styles):
            style_arr[:, :, j] = panels[st].reindex(index=md.dates, columns=tickers_all).values[rows]
    else:
        style_arr = np.zeros((len(rows), N, 0), dtype=dtype)
    # estimation universe: eligible on the date (universe member with a price,
    # ``md.eligibility()``) and a positive market cap.  ``apply_mask`` no longer
    # NaNs volume / caps outside the universe, so ``close.notna()`` alone would
    # admit names that are only in the panel because they join the universe later.
    valid = (md.eligibility() & (md.market_cap > 0)).values[rows]
    # industry columns *as of* each date (second-pass finding 4): a label is a
    # column from the first stored date on which an eligible name carries it
    # (dated label assigned and/or a member in the estimation universe), so the
    # column set at t never depends on classifications or universe entries after t
    code_arr = codes if codes is not None else np.broadcast_to(static_codes, valid.shape)
    first_row = np.full(len(inds), len(rows), dtype=np.int64)
    for k in range(len(inds)):
        seen = ((code_arr == k) & valid).any(axis=1)
        if seen.any():
            first_row[k] = int(np.argmax(seen))
    return ExposureHistory(
        sel_dates,
        tickers_all,
        cols,
        static,
        style_arr,
        valid,
        industry_codes=codes,
        n_industries=len(inds),
        industry_first_row=first_row,
    )


def build_exposures(md: MarketData, date: pd.Timestamp, cfg: RiskConfig | None = None) -> pd.DataFrame:
    """Exposure matrix (tickers x factors) as of ``date`` using data ``<= date``.

    For many dates prefer :func:`exposure_history`, which shares the rolling
    computations across dates."""
    cfg = cfg or RiskConfig()
    date = pd.Timestamp(date)
    sub = md.slice_until(date)
    hist = exposure_history(sub, [sub.dates[-1]], cfg)
    if len(hist) == 0:
        raise ValueError(f"no data on or before {date}")
    return hist[sub.dates[-1]]


def split_factor_columns(columns: Iterable[str], cfg: RiskConfig):
    """Return ``(market, industries, styles)`` column lists for an exposure frame."""
    cols = list(columns)
    styles = [c for c in cols if c in cfg.styles]
    market = [c for c in cols if c == MARKET_FACTOR]
    industries = [c for c in cols if c not in styles and c != MARKET_FACTOR]
    return market, industries, styles
