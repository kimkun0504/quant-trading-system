"""Baseline A: linear rank composite over factor themes (BUILD_PLAN §2.3).

* Themes are fixed lists of (feature, sign).  Within a theme, processed
  (rank-gaussian) features are averaged and the result is z-scored per date;
  across themes a weighted average is taken and re-z-scored.
* Dynamic weighting (review items 6 and 8):
    - **factor momentum tilt**: theme weights are multiplied by
      ``1 + clip(t-stat-like trailing theme return * max_tilt, ±max_tilt)``
      (one-month lookback by default, capped at ±25%).
    - **momentum crash control**: the momentum theme weight is scaled down
      (to as low as 0.25x) when the trailing 24-month equal-weight market
      return is negative *and* trailing 21-day market vol is above its 80th
      percentile (bear + high-vol state; Daniel-Moskowitz).
    - **momentum vol targeting**: the momentum theme weight is scaled by
      ``target_vol / trailing 63d vol of the momentum long-short portfolio``,
      clipped to [0.5, 1.5].  Only the momentum sleeve is vol-managed
      (Cederburg et al.).
  All dynamic quantities at date ``t`` use information ``<= t`` only.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from qts.core.config import AlphaConfig
from qts.core.crosssection import zscore
from qts.core.types import Long, MarketData, Wide

THEMES: Dict[str, List[Tuple[str, float]]] = {
    "value": [("value_bp", 1.0), ("value_ep", 1.0)],
    "momentum": [("mom_12_1", 1.0), ("mom_6_1", 1.0), ("industry_momentum", 1.0)],
    "quality": [("quality_gpa", 1.0)],
    "investment": [("investment_ag", 1.0)],
    "low_risk": [("beta_252", -1.0), ("idio_vol_252", -1.0), ("max_ret_21", -1.0)],
    "short_interest": [("short_interest", -1.0), ("borrow_fee", -1.0)],
}


# ----------------------------------------------------------------------------- helpers
def _long_to_wide(s: pd.Series) -> Wide:
    w = s.unstack("ticker")
    w.index.name, w.columns.name = "date", "ticker"
    return w


def theme_scores_long(processed: Long, themes: Dict[str, List[Tuple[str, float]]] = THEMES) -> Long:
    """Per-theme signed average of processed features (long frame, one column per theme).
    Themes with no available features are omitted."""
    cols = {}
    for theme, members in themes.items():
        avail = [(f, s) for f, s in members if f in processed.columns]
        if not avail:
            continue
        acc = sum(processed[f] * s for f, s in avail) / len(avail)
        cols[theme] = acc
    return pd.DataFrame(cols, index=processed.index)


def theme_scores_wide(processed: Long, themes=THEMES) -> Dict[str, Wide]:
    """Per-theme z-scored wide panels (date x ticker)."""
    tl = theme_scores_long(processed, themes)
    return {th: zscore(_long_to_wide(tl[th])) for th in tl.columns}


def theme_portfolio_returns(processed: Long, md: MarketData, n_deciles: int = 10,
                            themes=THEMES) -> pd.DataFrame:
    """Realised daily long-short (top minus bottom decile, equal-weight) return of
    each theme.  Portfolios are formed on the score at ``t`` and earn the return
    of ``t+1``; the series is indexed at the *realisation* date ``t+1`` so that
    ``.loc[:t]`` is point-in-time.  Fully vectorised."""
    rets = md.returns
    out = {}
    for th, score in theme_scores_wide(processed, themes).items():
        score = score.reindex(index=rets.index, columns=rets.columns)
        pct = score.rank(axis=1, pct=True)
        long = (pct > 1 - 1 / n_deciles).astype(float)
        short = (pct <= 1 / n_deciles).astype(float)
        nxt = rets.shift(-1)
        lr = (nxt * long).sum(axis=1) / long.sum(axis=1).replace(0, np.nan)
        sr = (nxt * short).sum(axis=1) / short.sum(axis=1).replace(0, np.nan)
        out[th] = (lr - sr).shift(1)  # index at realisation date
    df = pd.DataFrame(out)
    df.index.name = "date"
    return df


# ----------------------------------------------------------------------------- dynamic weights
def factor_momentum_tilt_series(theme_returns: pd.DataFrame, lookback_months: int,
                                max_tilt: float, scale_window: int = 252) -> pd.DataFrame:
    """Date x theme multiplier ``1 + tilt`` with
    ``tilt = clip(max_tilt * trailing_sum / (sd * sqrt(L)), -max_tilt, +max_tilt)``.
    The standardisation turns the trailing return into a ~t-statistic so the
    tilt saturates at ±max_tilt only for clearly positive/negative months."""
    L = max(1, int(lookback_months * 21))
    tr = theme_returns.rolling(L, min_periods=L).sum()
    sd = theme_returns.rolling(scale_window, min_periods=63).std()
    tstat = tr / (sd * np.sqrt(L)).replace(0, np.nan)
    tilt = (max_tilt * tstat).clip(-max_tilt, max_tilt).fillna(0.0)
    return 1.0 + tilt


def factor_momentum_tilt(theme_returns_history: pd.DataFrame, lookback_months: int,
                         max_tilt: float) -> pd.Series:
    """Multiplier per theme as of the last date in ``theme_returns_history``."""
    if len(theme_returns_history) == 0:
        return pd.Series(dtype=float)
    return factor_momentum_tilt_series(theme_returns_history, lookback_months, max_tilt).iloc[-1]


def momentum_crash_scalar_series(md: MarketData, lookback_days: int = 504,
                                 vol_window: int = 21, vol_pct: float = 0.80,
                                 floor: float = 0.25) -> pd.Series:
    """Per-date scalar in [floor, 1]: 1 normally; in a bear + high-vol state
    (trailing ``lookback_days`` EW market return < 0 and trailing 21d market vol
    above its expanding 80th percentile) ``clip(0.5 * vol_p80 / vol_21, floor, 1)``."""
    mkt = md.returns.mean(axis=1)
    cum = np.exp(np.log1p(mkt).rolling(lookback_days, min_periods=min(252, lookback_days)).sum()) - 1.0
    vol = mkt.rolling(vol_window, min_periods=vol_window).std()
    p80 = vol.expanding(min_periods=126).quantile(vol_pct)
    bear = cum < 0
    highvol = vol > p80
    scal = pd.Series(1.0, index=mkt.index)
    mask = (bear & highvol).fillna(False)
    scal[mask] = (0.5 * p80 / vol).clip(floor, 1.0)[mask]
    return scal


def momentum_crash_scalar(md: MarketData, date: pd.Timestamp, lookback_days: int = 504) -> float:
    s = momentum_crash_scalar_series(md.slice_until(date), lookback_days)
    return float(s.iloc[-1]) if len(s) else 1.0


def momentum_vol_scalar_series(theme_returns: pd.DataFrame, target_vol_annual: float,
                               window: int = 63, lo: float = 0.5, hi: float = 1.5) -> pd.Series:
    if "momentum" not in theme_returns.columns:
        return pd.Series(1.0, index=theme_returns.index)
    vol = theme_returns["momentum"].rolling(window, min_periods=window).std() * np.sqrt(252)
    return (target_vol_annual / vol.replace(0, np.nan)).clip(lo, hi).fillna(1.0)


# ----------------------------------------------------------------------------- composite
class ThemeComposite:
    """Fixed-theme rank composite with optional dynamic theme weights."""

    def __init__(self, cfg: AlphaConfig, themes: Dict[str, List[Tuple[str, float]]] = THEMES):
        self.cfg = cfg
        self.themes = themes

    # -- weights -------------------------------------------------------------
    def static_weights(self, available: List[str]) -> pd.Series:
        w = pd.Series({th: self.cfg.theme_weights.get(th, 0.0) for th in available}, dtype=float)
        return w

    def dynamic_weights(self, theme_returns: pd.DataFrame, md: MarketData,
                        available: List[str]) -> pd.DataFrame:
        """Date x theme weights = static x factor-momentum tilt x (momentum only:
        crash scalar x vol scalar).  All point-in-time."""
        base = self.static_weights(available)
        idx = theme_returns.index
        W = pd.DataFrame(np.tile(base.values, (len(idx), 1)), index=idx, columns=available)
        tilt = factor_momentum_tilt_series(theme_returns.reindex(columns=available).fillna(0.0),
                                           self.cfg.factor_momentum_lookback_months,
                                           self.cfg.factor_momentum_max_tilt)
        W = W * tilt.reindex(index=idx, columns=available).fillna(1.0)
        if "momentum" in available:
            crash = momentum_crash_scalar_series(md, self.cfg.momentum_crash_market_lookback_days)
            volsc = momentum_vol_scalar_series(theme_returns, self.cfg.momentum_vol_target_annual)
            W["momentum"] = W["momentum"] * crash.reindex(idx).fillna(1.0) * volsc.reindex(idx).fillna(1.0)
        return W

    # -- scoring -------------------------------------------------------------
    def score_all(self, processed: Long, md: Optional[MarketData] = None,
                  dynamic: bool = False, theme_returns: Optional[pd.DataFrame] = None,
                  return_components: bool = False):
        """Composite z-score panel for every date in ``processed``.

        ``dynamic=True`` requires ``md`` (and optionally precomputed
        ``theme_returns``) and applies the time-varying theme weights.
        """
        themes = theme_scores_wide(processed, self.themes)
        available = list(themes.keys())
        if dynamic:
            if md is None:
                raise ValueError("dynamic weighting needs MarketData")
            if theme_returns is None:
                theme_returns = theme_portfolio_returns(processed, md, themes=self.themes)
            W = self.dynamic_weights(theme_returns, md, available)
            ref = next(iter(themes.values()))
            W = W.reindex(ref.index).ffill().fillna(self.static_weights(available))
            num = sum(themes[th].mul(W[th], axis=0).fillna(0.0) for th in available)
            den = sum(themes[th].notna().mul(W[th].abs(), axis=0) for th in available).replace(0, np.nan)
        else:
            w = self.static_weights(available)
            num = sum(themes[th].fillna(0.0) * w[th] for th in available)
            den = sum(themes[th].notna() * abs(w[th]) for th in available).replace(0, np.nan)
        comp = zscore(num / den)
        comp.index.name, comp.columns.name = "date", "ticker"
        if return_components:
            return comp, themes
        return comp

    def score(self, processed: Long, date: pd.Timestamp, md: Optional[MarketData] = None,
              dynamic: bool = False, theme_returns: Optional[pd.DataFrame] = None) -> pd.Series:
        """Composite z-score Series (ticker index) for one date — AlphaOutput-ready."""
        sub = processed.loc[processed.index.get_level_values("date") <= date] if dynamic else \
            processed.loc[[date]]
        if dynamic and theme_returns is not None:
            theme_returns = theme_returns.loc[:date]
        comp = self.score_all(sub, md=None if md is None else md.slice_until(date),
                              dynamic=dynamic, theme_returns=theme_returns)
        return comp.loc[date].dropna()
