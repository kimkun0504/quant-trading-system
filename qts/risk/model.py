"""``BarraStyleRiskModel``: end-to-end USE4-style fundamental risk model.

Workflow
--------
``fit(md, dates)`` precomputes (vectorized, once):

* exposures for every date (``qts.risk.exposures.exposure_history``);
* daily factor / specific returns by constrained WLS, each day ``t`` using
  the exposures of ``t-1`` (``qts.risk.factor_returns``).

``as_of(date, alpha)`` then only touches data with timestamp ``<= date``:

* factor covariance from the factor-return history up to and including
  ``date`` (EWMA/NW, eigenfactor adjustment, VRA);
* specific variances from the specific-return history up to ``date`` and the
  exposures as of ``date``;
* the optional Saxena-Stubbs alignment term if ``alpha`` is given and
  ``cfg.alignment`` is on.

Speed: the EWMA/NW factor covariance is a weighted ``X'X`` on a
(T x K) matrix (microseconds-milliseconds); the specific-risk step is a
handful of vectorized passes over a (window x N) matrix.  The expensive piece
is the Monte-Carlo eigenfactor adjustment, so its per-eigenfactor bias vector
is cached and refreshed every ``eigen_refresh_days`` calendar steps (the
bias ratios depend only on the estimator and the eigenvalue spectrum, which
drifts slowly).
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from qts.core.config import RiskConfig
from qts.core.types import MarketData, RiskModelOutput
from qts.risk.alignment import alpha_alignment
from qts.risk.covariance import factor_covariance, specific_risk, shrunk_residual_covariance
from qts.risk.exposures import exposure_history
from qts.risk.factor_returns import estimate_factor_return_history


class BarraStyleRiskModel:
    def __init__(self, cfg: Optional[RiskConfig] = None, warmup_days: Optional[int] = None,
                 eigen_refresh_days: int = 21, seed: int = 0,
                 constraint_weights: str = "cap", weight_cap_quantile: Optional[float] = None):
        """``constraint_weights`` / ``weight_cap_quantile`` are forwarded to the
        cross-sectional regression (see ``estimate_factor_returns``); defaults
        are the pure USE4 choices."""
        self.cfg = cfg or RiskConfig()
        self.constraint_weights = constraint_weights
        self.weight_cap_quantile = weight_cap_quantile
        self.warmup_days = self.cfg.min_history_days if warmup_days is None else int(warmup_days)
        self.eigen_refresh_days = int(eigen_refresh_days)
        self.seed = seed
        self.md: Optional[MarketData] = None
        self.exposures: Dict[pd.Timestamp, pd.DataFrame] = {}
        self.factor_returns: Optional[pd.DataFrame] = None
        self.specific_returns: Optional[pd.DataFrame] = None
        self.r2: Optional[pd.Series] = None
        self._eigen_cache: Dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------ fit
    def fit(self, md: MarketData, dates: Optional[Iterable[pd.Timestamp]] = None) -> "BarraStyleRiskModel":
        """Precompute exposures and factor-return history.

        ``dates`` restricts the exposure dates (default: all dates after
        ``warmup_days``, which the rolling descriptors need).  Factor returns
        exist for every date whose previous day has exposures."""
        self.md = md
        if dates is None:
            dates = md.dates[min(self.warmup_days, len(md.dates) - 1):]
        self.exposures = exposure_history(md, dates, self.cfg)
        fr, sr, r2 = estimate_factor_return_history(
            md, self.exposures, self.cfg, constraint_weights=self.constraint_weights,
            weight_cap_quantile=self.weight_cap_quantile)
        self.factor_returns, self.specific_returns, self.r2 = fr, sr, r2
        self._eigen_cache.clear()
        return self

    # ------------------------------------------------------------------ helpers
    def _check_fitted(self):
        if self.md is None or self.factor_returns is None:
            raise RuntimeError("call fit() first")

    def exposures_as_of(self, date: pd.Timestamp) -> pd.DataFrame:
        date = pd.Timestamp(date)
        if date in self.exposures:
            return self.exposures[date]
        avail = [d for d in self.exposures if d <= date]
        if not avail:
            raise KeyError(f"no exposures on or before {date}")
        return self.exposures[max(avail)]

    def _eigen_bias(self, F_hist: pd.DataFrame) -> Optional[np.ndarray]:
        if not self.cfg.eigenfactor_adjust:
            return None
        key = len(F_hist) // self.eigen_refresh_days
        v = self._eigen_cache.get(key)
        if v is None:
            det = factor_covariance(F_hist, self.cfg, eigen_bias=None, seed=self.seed, return_details=True)
            v = det.eigen_bias
            self._eigen_cache[key] = v
        return v

    # ------------------------------------------------------------------ as_of
    def as_of(self, date: pd.Timestamp, alpha: Optional[pd.Series] = None,
              nu_scale: float = 1.0) -> RiskModelOutput:
        """Risk model as of ``date`` (factor returns through ``date`` inclusive,
        exposures as of ``date``)."""
        self._check_fitted()
        date = pd.Timestamp(date)
        F_hist = self.factor_returns.loc[:date]
        if len(F_hist) < self.cfg.min_history_days:
            raise ValueError(f"only {len(F_hist)} factor-return days <= {date}; "
                             f"need {self.cfg.min_history_days}")
        B = self.exposures_as_of(date)
        F_hist = F_hist.reindex(columns=B.columns).fillna(0.0)
        v = self._eigen_bias(F_hist)
        F = factor_covariance(F_hist, self.cfg, eigen_bias=v, seed=self.seed)
        mcap = self.md.market_cap.loc[:date].iloc[-1]
        U_hist = self.specific_returns.loc[:date]
        D = specific_risk(U_hist, B, mcap, self.cfg)
        out = RiskModelOutput(date=date, exposures=B, factor_cov=F, specific_var=D, factor_returns=F_hist)
        if alpha is not None and self.cfg.alignment:
            y, nu = alpha_alignment(alpha.reindex(B.index), B, D, nu_scale=nu_scale,
                                    factor_cov=F, cfg=self.cfg)
            out.alignment_vector, out.alignment_var = y, nu
        return out

    def residual_covariance_as_of(self, date: pd.Timestamp, window: Optional[int] = None) -> pd.DataFrame:
        """Full shrunk residual covariance (QIS / Ledoit-Wolf) for evaluation."""
        self._check_fitted()
        return shrunk_residual_covariance(self.specific_returns.loc[:pd.Timestamp(date)], self.cfg, window)
