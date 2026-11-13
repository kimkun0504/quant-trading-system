"""Feature set 2 (``AlphaConfig.feature_set = "v2"``): the eight raw panels added to
the Tier-1 features.  Definitions, references and signs: handbook 04 "Feature set 2".

Point-in-time rules (every panel at row ``t`` is a function of data stamped ``<= t``):

* **Announcement features** (``ear_3d``, ``abn_volume_earn``, ``days_since_earnings``).
  An announcement dated ``a`` maps to the first trading day ``p >= a``.  It may have
  been made after the close, so the reaction window is the three trading days
  ``[p-1, p+1]`` and the values become visible at the close of ``p+2``
  (:data:`EARNINGS_VISIBLE_LAG`): on ``p`` and ``p+1`` the panels still show the
  previous announcement.  Only ``MarketData.earnings_dates`` rows
  (``ticker, announce_date``) are read; the fiscal-period columns of that table are
  not.  Values are carried until the next announcement becomes visible and expire
  :data:`EARNINGS_MAX_AGE_DAYS` (126) trading days after ``p``.
* **Filing features** (``sue``, ``accruals``).  A filing dated ``datekey`` becomes
  visible on the first trading day ``>= datekey + 1`` calendar day, the convention
  of every fundamental (``qts.data.loaders.sharadar_sf1_to_pit_fundamentals``).  A
  quarter enters another quarter's statistic only if it was filed on or before it.
* **Price / volume features** (``abn_volume_5d``, ``net_issuance``, ``high_52w``):
  trailing windows ending at ``t``.

All functions here take explicit inputs and return ``date x ticker`` panels; nothing
is cross-sectionally processed (``FeatureBuilder.process`` does that).
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from qts.core.roles import FEATURE_SET_2_FEATURES
from qts.core.types import MarketData, Wide

#: trading days between the announcement day ``p`` and the first signal date that
#: may use it (reaction window ends at ``p+1``; visible from the close of ``p+2``)
EARNINGS_VISIBLE_LAG = 2
#: cap of ``days_since_earnings`` and the age (trading days since the announcement /
#: since a filing became visible) beyond which an earnings value is missing
EARNINGS_MAX_AGE_DAYS = 126
#: calendar-day lag from a filing's ``datekey`` to its first usable date
FILING_LAG_DAYS = 1
#: SUE: seasonal lag (quarters), trailing window of seasonal differences, minimum
SUE_SEASONAL_LAG = 4
SUE_WINDOW = 8
SUE_MIN_OBS = 4
#: abnormal volume: base window / minimum observations, short window / minimum
VOLUME_BASE_WINDOW = 252
VOLUME_BASE_MIN_OBS = 126
VOLUME_SHORT_WINDOW = 5
VOLUME_SHORT_MIN_OBS = 3
#: issuance and 52-week-high look-backs (trading days)
ISSUANCE_LOOKBACK = 252
HIGH_WINDOW = 252
HIGH_MIN_OBS = 126
#: filings dated more than this many calendar days before the first panel date are
#: ignored (a stale pre-sample filing is not carried into the panel)
PRE_SAMPLE_FILING_DAYS = 190

DERIVE_HINT = (
    "run `qts data sharadar derive-earnings --store <store>` (it builds the store datasets "
    "earnings_dates and earnings_fundamentals from the tables already in the store)"
)
NCFO_HINT = (
    "the stored SF1 has no operating cash flow (`ncfo`): re-read SF1 from the bulk folder with "
    "`qts data sharadar sync --store <store> --tables SF1 --from-files <bulk dir>` and then "
    "`qts data sharadar derive-earnings --store <store>`"
)


class FeatureSetDataError(ValueError):
    """``feature_set = "v2"`` was requested on a ``MarketData`` without its inputs."""


def require_v2_inputs(md: MarketData) -> None:
    """Raise :class:`FeatureSetDataError` (naming the command that fixes it) unless
    ``md`` carries everything the v2 features read."""
    if md.earnings_dates is None or md.earnings_fundamentals is None:
        missing = [n for n in ("earnings_dates", "earnings_fundamentals") if getattr(md, n) is None]
        raise FeatureSetDataError(
            f"alpha.feature_set='v2' needs MarketData.{' / '.join(missing)}, which this data set "
            f"does not have: {DERIVE_HINT}"
        )
    ef = md.earnings_fundamentals
    art = ef[ef["dimension"] == "ART"]
    if "ncfo" not in ef.columns or not art["ncfo"].notna().any():
        raise FeatureSetDataError(f"alpha.feature_set='v2' cannot build `accruals`: {NCFO_HINT}")
    arq = ef[ef["dimension"] == "ARQ"]
    if not arq["eps"].notna().any():
        raise FeatureSetDataError(
            f"alpha.feature_set='v2' cannot build `sue`: earnings_fundamentals has no ARQ eps; {DERIVE_HINT}"
        )


# ----------------------------------------------------------------------------- helpers
def _carry_last(
    n_rows: int, n_cols: int, rows: np.ndarray, cols: np.ndarray, max_age: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """Scatter events ``k`` at ``(rows[k], cols[k])`` and carry each forward down the
    rows until the next event of the column.  Events must be given in the order in
    which a later one overrides an earlier one at the same cell.  Returns
    ``(last_k, age)``: per cell the index of the event in force (-1: none) and the
    rows elapsed since its row; with ``max_age`` an event older than that is dropped
    (``last_k = -1``)."""
    cell = np.full((n_rows, n_cols), -1, dtype=np.int32)
    cell[rows, cols] = np.arange(len(rows), dtype=np.int32)
    row_idx = np.arange(n_rows, dtype=np.int32)[:, None]
    src = np.where(cell >= 0, row_idx, -1)
    np.maximum.accumulate(src, axis=0, out=src)
    have = src >= 0
    last = np.where(have, cell[np.where(have, src, 0), np.arange(n_cols)[None, :]], -1)
    age = np.where(have, row_idx - src, -1)
    if max_age is not None:
        stale = age > int(max_age)
        last = np.where(stale, -1, last)
    return last, age


def _take(values: np.ndarray, last: np.ndarray) -> np.ndarray:
    """``values[last]`` with NaN where ``last < 0``."""
    out = np.full(last.shape, np.nan)
    ok = last >= 0
    out[ok] = values[last[ok]]
    return out


def _wide(arr: np.ndarray, md: MarketData) -> Wide:
    return pd.DataFrame(arr, index=md.dates, columns=md.tickers)


def _safe_log_ratio(num: Any, den: Any) -> Any:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(num / den)
    return np.where(np.isfinite(out), out, np.nan)


# ----------------------------------------------------------------------------- announcements
def announcement_positions(md: MarketData) -> pd.DataFrame:
    """``MarketData.earnings_dates`` -> one row per announcement inside the panel:
    ``col`` (ticker position), ``p`` (position of the first trading day >=
    ``announce_date``), sorted by ``(col, p)``, one row per ``(col, p)``.
    Announcements dated before the first panel date or after the last are dropped."""
    ed = md.earnings_dates
    assert ed is not None
    dates = md.dates
    when = pd.to_datetime(ed["announce_date"]).to_numpy()
    col = md.tickers.get_indexer(pd.Index(ed["ticker"].astype(str)))
    p = dates.searchsorted(when, side="left")
    ok = (col >= 0) & ~np.isnat(when) & (p < len(dates))
    if len(dates):
        ok &= when >= dates[0].to_datetime64()
    out = pd.DataFrame({"col": col[ok], "p": p[ok]})
    return out.drop_duplicates().sort_values(["col", "p"], kind="mergesort").reset_index(drop=True)


def earnings_reaction_panels(md: MarketData, market: pd.Series, base_dollar_volume: Wide) -> dict[str, Wide]:
    """``ear_3d``, ``abn_volume_earn`` and ``days_since_earnings`` (raw trading days,
    capped at :data:`EARNINGS_MAX_AGE_DAYS`; NaN before the first visible announcement).

    ``market``: the equal-weight universe return per day (the eligible names of that
    day).  ``base_dollar_volume``: trailing ``VOLUME_BASE_WINDOW``-day mean dollar
    volume.  For an announcement on trading day ``p``:

    * ``ear_3d = close[p+1] / close[p-2] - 1  -  (prod_{s=p-1..p+1} (1 + market_s) - 1)``;
    * ``abn_volume_earn = log(mean dollar volume over p-1..p+1 (>= 2 days) /
      base_dollar_volume[p-2])`` (the 252 days ending the day before the window);

    both visible from row ``p + EARNINGS_VISIBLE_LAG``."""
    T, N = md.close.shape
    ann = announcement_positions(md)
    col, p = ann["col"].to_numpy(), ann["p"].to_numpy()
    close = md.close.to_numpy(dtype=float)
    dv = md.dollar_volume.to_numpy(dtype=float)
    base = base_dollar_volume.to_numpy(dtype=float)
    mkt_level = np.cumprod(1.0 + market.fillna(0.0).to_numpy(dtype=float))
    ear = np.full(len(ann), np.nan)
    avol = np.full(len(ann), np.nan)
    full = (p - 2 >= 0) & (p + 1 < T)
    if full.any():
        c, q = col[full], p[full]
        with np.errstate(divide="ignore", invalid="ignore"):
            stock = close[q + 1, c] / close[q - 2, c] - 1.0
            mk = mkt_level[q + 1] / mkt_level[q - 2] - 1.0
            window = np.stack([dv[q - 1, c], dv[q, c], dv[q + 1, c]])
            n_ok = np.isfinite(window).sum(axis=0)
            mean_dv = np.where(n_ok >= 2, np.nansum(window, axis=0) / np.maximum(n_ok, 1), np.nan)
        ear[full] = stock - mk
        avol[full] = _safe_log_ratio(mean_dv, base[q - 2, c])
    vis = p + EARNINGS_VISIBLE_LAG
    seen = vis < T
    idx = np.flatnonzero(seen)
    last, age = _carry_last(T, N, vis[seen], col[seen], max_age=None)
    # age counts rows since the visible row; trading days since the announcement = + lag
    days = np.where(last >= 0, age + EARNINGS_VISIBLE_LAG, -1)
    fresh = np.where(days <= EARNINGS_MAX_AGE_DAYS, last, -1)
    k = np.where(fresh >= 0, idx[np.where(fresh >= 0, fresh, 0)], -1) if len(idx) else fresh
    days_f = np.where(last >= 0, np.minimum(days, EARNINGS_MAX_AGE_DAYS).astype(float), np.nan)
    return {
        "ear_3d": _wide(_take(ear, k), md),
        "abn_volume_earn": _wide(_take(avol, k), md),
        "days_since_earnings": _wide(days_f, md),
    }


# ----------------------------------------------------------------------------- filings
def _quarter_index(calendardate: pd.Series) -> np.ndarray:
    d = pd.to_datetime(calendardate)
    return (d.dt.year * 4 + (d.dt.quarter - 1)).to_numpy(dtype=np.int64)


def _filing_grid(
    rows: pd.DataFrame, value_cols: list[str]
) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    """One row per ``(ticker, fiscal quarter)`` (the first filing of the period) laid
    on a dense ``ticker x quarter`` grid: returns the de-duplicated rows with ``_r``
    / ``_q`` grid coordinates, the value grids and the ``datekey`` grid (int64 ns;
    the maximum int64 where the quarter has no filing)."""
    r = rows.sort_values(["ticker", "calendardate", "datekey"], kind="mergesort")
    r = r.drop_duplicates(["ticker", "calendardate"], keep="first").reset_index(drop=True)
    qi = _quarter_index(r["calendardate"])
    codes, _ = pd.factorize(r["ticker"], sort=True)
    q0 = int(qi.min()) if len(qi) else 0
    r["_r"], r["_q"] = codes, qi - q0
    shape = (int(codes.max()) + 1 if len(r) else 0, int(r["_q"].max()) + 1 if len(r) else 0)
    grids = {}
    for c in value_cols:
        g = np.full(shape, np.nan)
        g[r["_r"], r["_q"]] = r[c].to_numpy(dtype=float)
        grids[c] = g
    dk = np.full(shape, np.iinfo(np.int64).max, dtype=np.int64)
    dk[r["_r"], r["_q"]] = pd.to_datetime(r["datekey"]).to_numpy().astype("datetime64[ns]").astype(np.int64)
    return r, grids, dk


def _lagged(grid: np.ndarray, dk: np.ndarray, lag: int) -> np.ndarray:
    """``grid`` shifted ``lag`` quarters back, NaN where the lagged quarter has no
    filing or was filed after the current one (not known at the current filing)."""
    out = np.full(grid.shape, np.nan)
    if grid.shape[1] > lag:
        known = dk[:, :-lag] <= dk[:, lag:]
        out[:, lag:] = np.where(known, grid[:, :-lag], np.nan)
    return out


def sue_by_filing(arq: pd.DataFrame) -> pd.DataFrame:
    """Standardised unexpected earnings per quarterly filing (seasonal random walk).

    ``arq``: ``ticker, datekey, calendardate, eps, split_cum`` (ARQ rows of
    ``MarketData.earnings_fundamentals``).  With ``e_q = eps_q * split_cum_q``
    (EPS per original share: as-reported EPS re-expressed on one share basis using
    splits dated on or before the filing) and ``u_q = e_q - e_{q-4}``,

        ``sue_q = u_q / sd(u_{q-7}, ..., u_q)``   (sample sd, ``ddof = 1``),

    over the quarters of the trailing :data:`SUE_WINDOW` that have a value and were
    filed on or before quarter ``q``; NaN below :data:`SUE_MIN_OBS` values or a zero
    sd.  The ratio is unchanged by any rescaling common to the window, so later
    splits cannot alter it.  Returns ``ticker, datekey, calendardate, sue``."""
    rows = arq[["ticker", "datekey", "calendardate"]].copy()
    rows["e"] = pd.to_numeric(arq["eps"], errors="coerce").to_numpy(dtype=float) * pd.to_numeric(
        arq["split_cum"], errors="coerce"
    ).to_numpy(dtype=float)
    r, grids, dk = _filing_grid(rows, ["e"])
    e = grids["e"]
    u = e - _lagged(e, dk, SUE_SEASONAL_LAG)
    n_q = u.shape[1]
    stack = np.full((SUE_WINDOW, *u.shape), np.nan)
    for off in range(SUE_WINDOW):
        if off == 0:
            stack[0] = u
        elif n_q > off:
            known = dk[:, :-off] <= dk[:, off:]
            stack[off][:, off:] = np.where(known, u[:, :-off], np.nan)
    n = np.isfinite(stack).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN windows
        sd = np.nanstd(stack, axis=0, ddof=1)
        sue = np.where((n >= SUE_MIN_OBS) & (sd > 0), u / sd, np.nan)
    r["sue"] = sue[r["_r"], r["_q"]] if len(r) else np.nan
    return r[["ticker", "datekey", "calendardate", "sue"]]


def accruals_by_filing(art: pd.DataFrame) -> pd.DataFrame:
    """Sloan accruals per trailing-twelve-month filing:
    ``(netinc - ncfo) / ((assets + assets one year earlier) / 2)`` from the ART rows
    (``ticker, datekey, calendardate, netinc, ncfo, assets``).  The year-earlier
    assets come from the first filing of the period four quarters back, used only
    if filed on or before the current filing; NaN without it or with non-positive
    average assets.  Returns ``ticker, datekey, calendardate, accruals``."""
    r, g, dk = _filing_grid(
        art[["ticker", "datekey", "calendardate", "netinc", "ncfo", "assets"]], ["netinc", "ncfo", "assets"]
    )
    avg = 0.5 * (g["assets"] + _lagged(g["assets"], dk, 4))
    with np.errstate(invalid="ignore", divide="ignore"):
        acc = np.where(avg > 0, (g["netinc"] - g["ncfo"]) / avg, np.nan)
    r["accruals"] = acc[r["_r"], r["_q"]] if len(r) else np.nan
    return r[["ticker", "datekey", "calendardate", "accruals"]]


def filing_panel(md: MarketData, by_filing: pd.DataFrame, value: str, max_age: int | None) -> Wide:
    """Per-filing values -> PIT panel: a filing is visible from the first trading
    day ``>= datekey + FILING_LAG_DAYS`` and carried until the next filing of the
    name becomes visible (or ``max_age`` trading days).  A filing of a period older
    than one already filed does not replace the current value (the convention of
    ``sharadar_sf1_to_pit_fundamentals``); a newer filing with a NaN value does
    (the panel then shows NaN, not the superseded quarter)."""
    dates = md.dates
    T, N = md.close.shape
    f = by_filing.sort_values(["ticker", "datekey", "calendardate"], kind="mergesort").reset_index(drop=True)
    latest = f.groupby("ticker")["calendardate"].cummax()
    f = f[f["calendardate"] >= latest]
    avail = (pd.to_datetime(f["datekey"]) + pd.Timedelta(days=FILING_LAG_DAYS)).to_numpy()
    col = md.tickers.get_indexer(pd.Index(f["ticker"].astype(str)))
    pos = dates.searchsorted(avail, side="left")
    ok = (col >= 0) & (pos < T)
    if T:
        ok &= avail >= (dates[0] - pd.Timedelta(days=PRE_SAMPLE_FILING_DAYS)).to_datetime64()
    vals = f[value].to_numpy(dtype=float)[ok]
    last, _ = _carry_last(T, N, pos[ok], col[ok], max_age=max_age)
    return _wide(_take(vals, last), md)


# ----------------------------------------------------------------------------- price / volume
def base_dollar_volume(md: MarketData) -> Wide:
    """Trailing ``VOLUME_BASE_WINDOW``-day mean dollar volume (>= ``VOLUME_BASE_MIN_OBS`` days)."""
    return md.dollar_volume.rolling(VOLUME_BASE_WINDOW, min_periods=VOLUME_BASE_MIN_OBS).mean()


def abnormal_volume_5d(md: MarketData, base: Wide) -> Wide:
    """``log(mean dollar volume of the last 5 days / mean of the 252 days before them)``."""
    short = md.dollar_volume.rolling(VOLUME_SHORT_WINDOW, min_periods=VOLUME_SHORT_MIN_OBS).mean()
    prior = base.shift(VOLUME_SHORT_WINDOW)
    return _wide(_safe_log_ratio(short.to_numpy(dtype=float), prior.to_numpy(dtype=float)), md)


def net_issuance(md: MarketData) -> Wide:
    """``-log(S_t / S_{t-252})`` with ``S = market_cap / close`` (adjusted close): the
    share count on the adjusted-price basis, which a split leaves unchanged.  Because
    ``close`` is a total-return price, ``log S_t - log S_{t-252}`` equals the log
    growth of market cap minus the log total return (Daniel-Titman composite
    issuance): cash dividends count as negative issuance.  Sign: issuance is bad."""
    s = md.market_cap / md.close
    growth = _safe_log_ratio(s.to_numpy(dtype=float), s.shift(ISSUANCE_LOOKBACK).to_numpy(dtype=float))
    # 1e-10 resolution: an unchanged share count is exactly 0, not floating-point noise
    return _wide(-np.round(growth, 10) + 0.0, md)


def high_52w(md: MarketData) -> Wide:
    """``close / max(close over the trailing 252 days, today included)`` in (0, 1]."""
    return md.close / md.close.rolling(HIGH_WINDOW, min_periods=HIGH_MIN_OBS).max()


# ----------------------------------------------------------------------------- entry point
def build_v2_features(md: MarketData, market: pd.Series) -> dict[str, Wide]:
    """The eight v2 panels keyed by :data:`qts.core.roles.FEATURE_SET_2_FEATURES`.
    ``market``: the equal-weight return of the names eligible each day (the same
    series the Tier-1 beta uses)."""
    require_v2_inputs(md)
    ef = md.earnings_fundamentals
    assert ef is not None
    ef = ef[ef["ticker"].astype(str).isin(set(md.tickers))]
    base = base_dollar_volume(md)
    out = earnings_reaction_panels(md, market, base)
    arq = ef[ef["dimension"] == "ARQ"]
    art = ef[ef["dimension"] == "ART"]
    out["sue"] = filing_panel(md, sue_by_filing(arq), "sue", max_age=EARNINGS_MAX_AGE_DAYS)
    out["accruals"] = filing_panel(md, accruals_by_filing(art), "accruals", max_age=None)
    out["abn_volume_5d"] = abnormal_volume_5d(md, base)
    out["net_issuance"] = net_issuance(md)
    out["high_52w"] = high_52w(md)
    return {k: out[k] for k in FEATURE_SET_2_FEATURES}
