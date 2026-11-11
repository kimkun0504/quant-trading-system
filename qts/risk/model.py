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
is cached and refreshed every ``eigen_refresh_days`` *factor-return
observations* (trading days with an estimated factor return, not calendar
days; finding #18).  The bias of a bucket is estimated from the history up to
the bucket's first observation only, so ``as_of(t)`` never uses data after
``t`` and does not depend on query order (finding #46).  The bias ratios
depend only on the estimator and the eigenvalue spectrum, which drifts slowly.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from qts.core.config import RiskConfig
from qts.core.returns import CleanedReturns, clean_returns, flag_bad_prices
from qts.core.types import MarketData, RiskModelOutput
from qts.risk.alignment import alpha_alignment
from qts.risk.covariance import factor_covariance, shrunk_residual_covariance, specific_risk
from qts.risk.exposures import ExposureHistory, exposure_history, split_factor_columns
from qts.risk.factor_returns import estimate_factor_return_history

log = logging.getLogger(__name__)

#: ``as_of`` records a covariance outlier when the reference book's ex-ante vol
#: jumps by more than this factor against the previous recorded date.
REFERENCE_VOL_JUMP = 5.0
TRADING_DAYS = 252


def estimation_inputs(
    md: MarketData, cfg: RiskConfig
) -> tuple[MarketData, pd.DataFrame, CleanedReturns | None, pd.DataFrame | None]:
    """``(md_est, returns, cleaning, estimation_universe)`` the risk model estimates on.

    With ``cfg.clean_returns``:

    * ``md_est`` is ``md`` with data-error *closes* blanked
      (:func:`qts.core.returns.flag_bad_prices`, the price-level form of rule
      (a)); the style descriptors (beta, momentum, residual vol, reversal and
      the equal-weighted market proxy inside them) are built from it, so a
      +99,900 % print does not become every name's beta for a year;
    * ``returns`` is ``md_est.returns`` after :func:`qts.core.returns.clean_returns`
      (rule (a) with the price floor, then rule (b)); the regression's dependent
      variable and the APCA companion's input;
    * ``cleaning.data_error`` marks every cell that was a finite return in
      ``md.returns`` and is NaN in ``returns`` (a blanked close removes the return
      into and out of it: a spike-and-reverse pair counts 2).

    Otherwise everything is ``md`` / ``md.returns`` untouched and ``cleaning`` is
    ``None``.  The estimation universe is ``md.eligible`` (the PIT universe mask)
    when ``cfg.estimation_universe_eligible_only`` and the mask exists, else
    ``None`` (every name with a market cap)."""
    cleaning: CleanedReturns | None = None
    md_est = md
    returns = md.returns
    if cfg.clean_returns:
        close_clean, bad_px = flag_bad_prices(
            md.close, cap_up=cfg.return_abs_cap_up, cap_down=cfg.return_abs_cap_down
        )
        if bad_px.to_numpy().any():
            md_est = replace(md, close=close_clean)
        cleaning = clean_returns(
            md_est.returns,
            md.close_raw,
            cap_up=cfg.return_abs_cap_up,
            cap_down=cfg.return_abs_cap_down,
            min_prev_price=cfg.return_min_prev_price,
            mad_k=cfg.return_winsor_mad_k if cfg.return_winsor_mad_k > 0 else None,
        )
        data_error = (returns.notna() & cleaning.returns.isna()) | cleaning.data_error
        cleaning = replace(cleaning, data_error=data_error)
        returns = cleaning.returns
    estu = md.eligible if (cfg.estimation_universe_eligible_only and md.eligible is not None) else None
    return md_est, returns, cleaning, estu


def reference_book(market_cap: pd.Series, tickers: pd.Index) -> pd.Series:
    """Long-only sqrt-cap-weighted book over ``tickers`` (gross 1): the
    "equal-ish" reference whose ex-ante vol should sit near the market's."""
    cap = market_cap.reindex(tickers).astype(float)
    w = np.sqrt(cap.clip(lower=0.0)).fillna(0.0)
    tot = float(w.sum())
    if tot <= 0:
        w = pd.Series(1.0, index=tickers)
        tot = float(len(tickers))
    return w / tot


def factor_groups_for(columns, cfg: RiskConfig, statistical=None) -> dict[str, list[str]]:
    """``RiskModelOutput.factor_groups`` for an exposure frame built by this
    package: ``market`` / ``industry`` / ``style`` from
    :func:`split_factor_columns`, plus the (optional) ``statistical`` block.
    ``alignment`` is always empty: the Saxena-Stubbs term is carried as
    ``alignment_vector`` / ``alignment_var``, not as a column."""
    stat = [str(c) for c in (statistical or [])]
    cols = [c for c in columns if c not in stat]
    market, industries, styles = split_factor_columns(cols, cfg)
    return {
        "market": list(market),
        "industry": list(industries),
        "style": list(styles),
        "statistical": stat,
        "alignment": [],
    }


class BarraStyleRiskModel:
    def __init__(
        self,
        cfg: RiskConfig | None = None,
        warmup_days: int | None = None,
        eigen_refresh_days: int = 21,
        seed: int = 0,
        constraint_weights: str = "cap",
        weight_cap_quantile: float | None = None,
    ):
        """``constraint_weights`` / ``weight_cap_quantile`` are forwarded to the
        cross-sectional regression (see ``estimate_factor_returns``); defaults
        are the pure USE4 choices."""
        self.cfg = cfg or RiskConfig()
        self.constraint_weights = constraint_weights
        self.weight_cap_quantile = weight_cap_quantile
        self.warmup_days = self.cfg.min_history_days if warmup_days is None else int(warmup_days)
        self.eigen_refresh_days = int(eigen_refresh_days)
        self.seed = seed
        self.md: MarketData | None = None
        self.exposures: dict[pd.Timestamp, pd.DataFrame] = {}  # ExposureHistory after fit()
        self.factor_returns: pd.DataFrame | None = None
        self.specific_returns: pd.DataFrame | None = None
        self.r2: pd.Series | None = None
        self._eigen_cache: dict[tuple, np.ndarray] = {}
        #: cleaned returns the regression ran on (``md.returns`` when cleaning is off)
        self.returns_clean: pd.DataFrame | None = None
        #: :class:`~qts.core.returns.CleanedReturns` flags (``None`` when cleaning is off)
        self.cleaning: CleanedReturns | None = None
        #: ``reference_vol`` (date -> annualised ex-ante vol of :func:`reference_book`),
        #: ``covariance_outliers`` (list of dicts), ``n_data_error_returns``,
        #: ``n_winsorized_returns``; filled by ``fit`` / ``as_of``.
        self.diagnostics: dict[str, Any] = {
            "reference_vol": {},
            "covariance_outliers": [],
            "n_data_error_returns": 0,
            "n_winsorized_returns": 0,
        }

    # ------------------------------------------------------------------ fit
    def fit(self, md: MarketData, dates: Iterable[pd.Timestamp] | None = None) -> BarraStyleRiskModel:
        """Precompute exposures and factor-return history.

        ``dates`` restricts the exposure dates (default: all dates after
        ``warmup_days``, which the rolling descriptors need).  Factor returns
        exist for every date whose previous day has exposures."""
        self.md = md
        if dates is None:
            dates = md.dates[min(self.warmup_days, len(md.dates) - 1) :]
        md_est, returns, cleaning, estu = estimation_inputs(md, self.cfg)
        self.returns_clean, self.cleaning = returns, cleaning
        self.exposures = exposure_history(
            md_est, dates, self.cfg, dtype=np.dtype(getattr(self.cfg, "exposure_dtype", "float64"))
        )
        self.diagnostics = {
            "reference_vol": {},
            "covariance_outliers": [],
            "n_data_error_returns": 0 if cleaning is None else cleaning.n_data_errors,
            "n_winsorized_returns": 0 if cleaning is None else cleaning.n_winsorized,
        }
        if cleaning is not None and cleaning.n_data_errors:
            log.warning(
                "risk model: %d data-error returns removed (rule (a)), %d winsorised (rule (b)); "
                "see model.cleaning / scripts/find_outlier_returns.py",
                cleaning.n_data_errors,
                cleaning.n_winsorized,
            )
        fr, sr, r2 = estimate_factor_return_history(
            md_est,
            self.exposures,
            self.cfg,
            returns=returns,
            estimation_universe=estu,
            constraint_weights=self.constraint_weights,
            weight_cap_quantile=self.weight_cap_quantile,
        )
        self.factor_returns, self.specific_returns, self.r2 = fr, sr, r2
        self._eigen_cache.clear()
        return self

    # ------------------------------------------------------------------ helpers
    def _check_fitted(self):
        if self.md is None or self.factor_returns is None or self.specific_returns is None:
            raise RuntimeError("call fit() first")

    def _fitted_state(self):
        """``(md, factor_returns, specific_returns)`` with the Optionals narrowed."""
        self._check_fitted()
        assert self.md is not None and self.factor_returns is not None and self.specific_returns is not None
        return self.md, self.factor_returns, self.specific_returns

    def exposures_as_of(self, date: pd.Timestamp) -> pd.DataFrame:
        """Exposures on the most recent exposure date ``<= date`` (O(log T))."""
        date = pd.Timestamp(date)
        ex = self.exposures
        if isinstance(ex, ExposureHistory):
            return ex.as_of(date)
        if date in ex:
            return ex[date]
        avail = [d for d in ex if d <= date]
        if not avail:
            raise KeyError(f"no exposures on or before {date}")
        return ex[max(avail)]

    def _eigen_bucket_length(self, n_hist: int) -> int:
        """History length the eigenfactor bias of an ``as_of`` with ``n_hist``
        factor-return rows is estimated on: the *first* row count of its refresh
        bucket ``n_hist // eigen_refresh_days`` (floored at ``min_history_days``,
        the first length ``as_of`` can serve).  Every date in the bucket maps to
        the same length ``<= n_hist``, so the bias only uses factor returns
        available at the bucket's first date (no look-ahead) and ``as_of(t)`` is
        independent of the order in which dates are queried (finding #46)."""
        R = max(1, self.eigen_refresh_days)
        n0 = max((n_hist // R) * R, int(self.cfg.min_history_days), 2)
        return int(min(n0, n_hist))

    def _eigen_bias(self, F_hist: pd.DataFrame) -> np.ndarray | None:
        """Raw simulated eigenfactor bias ``v`` (USE4 step 3) for the refresh
        bucket of ``F_hist``, cached by ``(bucket start length, factor columns)``
        and computed from ``F_hist.iloc[:bucket start]`` only.  The USE4
        inflation ``a`` / parabola smoothing are applied downstream in
        :func:`factor_covariance` (``cfg.eigen_inflation_a``)."""
        if not self.cfg.eigenfactor_adjust:
            return None
        n0 = self._eigen_bucket_length(len(F_hist))
        key = (n0, tuple(F_hist.columns))
        v = self._eigen_cache.get(key)
        if v is None:
            det = factor_covariance(
                F_hist.iloc[:n0], self.cfg, eigen_bias=None, seed=self.seed, return_details=True
            )
            v = det.eigen_bias
            self._eigen_cache[key] = v
        return v

    # ------------------------------------------------------------------ as_of
    def as_of(
        self, date: pd.Timestamp, alpha: pd.Series | None = None, nu_scale: float | None = None
    ) -> RiskModelOutput:
        """Risk model as of ``date`` (factor returns through ``date`` inclusive,
        exposures as of ``date``)."""
        md, factor_returns, specific_returns = self._fitted_state()
        date = pd.Timestamp(date)
        F_hist = factor_returns.loc[:date]
        if len(F_hist) < self.cfg.min_history_days:
            raise ValueError(
                f"only {len(F_hist)} factor-return days <= {date}; need {self.cfg.min_history_days}"
            )
        B = self.exposures_as_of(date)
        F_hist = F_hist.reindex(columns=B.columns).fillna(0.0)
        v = self._eigen_bias(F_hist)
        F = factor_covariance(F_hist, self.cfg, eigen_bias=v, seed=self.seed)
        mcap = md.market_cap.loc[:date].iloc[-1]
        U_hist = specific_returns.loc[:date]
        D = specific_risk(U_hist, B, mcap, self.cfg, market_cap_history=md.market_cap.loc[:date])
        self._check_reference_vol(date, B, F, D, mcap)
        y: pd.Series | None = None
        nu = 0.0
        if alpha is not None and self.cfg.alignment:
            y, nu = alpha_alignment(
                alpha.reindex(B.index), B, D, nu_scale=nu_scale, factor_cov=F, cfg=self.cfg
            )
        return RiskModelOutput(
            date=date,
            exposures=B,
            factor_cov=F,
            specific_var=D,
            factor_returns=F_hist,
            alignment_vector=y,
            alignment_var=float(nu),
            factor_groups=factor_groups_for(B.columns, self.cfg),
        )

    # ------------------------------------------------------------------ guard
    def _check_reference_vol(
        self, date: pd.Timestamp, B: pd.DataFrame, F: pd.DataFrame, D: pd.Series, mcap: pd.Series
    ) -> float:
        """Blow-up guard: annualised ex-ante vol of :func:`reference_book` under
        ``B F B' + D``, recorded in ``diagnostics['reference_vol'][date]``.  Above
        ``cfg.max_reference_vol`` or more than :data:`REFERENCE_VOL_JUMP` times the
        previously recorded date's value it is appended to
        ``diagnostics['covariance_outliers']`` with the top factor / specific
        contributors and logged as a warning.  A healthy long-only sqrt-cap book
        sits near the market's vol (15-25 %); hundreds of percent mean a factor
        return or a specific variance absorbed a data error."""
        w = reference_book(mcap, B.index)
        Bv = B.to_numpy(dtype=float)
        Fv = F.reindex(index=B.columns, columns=B.columns).fillna(0.0).to_numpy(dtype=float)
        d = D.reindex(B.index).fillna(0.0).to_numpy(dtype=float)
        wv = w.to_numpy(dtype=float)
        fw = Bv.T @ wv
        fac_contrib = fw * (Fv @ fw)
        spec_contrib = wv**2 * d
        var = float(fac_contrib.sum() + spec_contrib.sum())
        vol = float(np.sqrt(max(var, 0.0) * TRADING_DAYS))
        ref: dict[pd.Timestamp, float] = self.diagnostics["reference_vol"]
        earlier = [k for k in ref if k < date]
        prev_vol = ref[max(earlier)] if earlier else None
        ref[date] = vol
        jump = prev_vol is not None and prev_vol > 0 and vol > REFERENCE_VOL_JUMP * prev_vol
        if vol > float(self.cfg.max_reference_vol) or jump:
            kf = np.argsort(-np.abs(fac_contrib))[:5]
            ks = np.argsort(-spec_contrib)[:5]
            entry = {
                "date": date,
                "reference_vol": vol,
                "previous_vol": prev_vol,
                "top_factors": [(str(B.columns[k]), float(fac_contrib[k] * TRADING_DAYS)) for k in kf],
                "top_names": [(str(B.index[k]), float(spec_contrib[k] * TRADING_DAYS)) for k in ks],
            }
            self.diagnostics["covariance_outliers"].append(entry)
            log.warning(
                "risk model: reference-book ex-ante vol %.1f%% on %s (previous %s; limit %.0f%%); "
                "top factors %s; top names %s",
                100 * vol,
                date.date(),
                "n/a" if prev_vol is None else f"{100 * prev_vol:.1f}%",
                100 * float(self.cfg.max_reference_vol),
                [(f, round(v, 4)) for f, v in entry["top_factors"]],
                [(n, round(v, 4)) for n, v in entry["top_names"]],
            )
        return vol

    def residual_covariance_as_of(
        self, date: pd.Timestamp, window: int | None = None, method: str = "qis"
    ) -> pd.DataFrame:
        """Full shrunk residual covariance (``method`` = ``"qis"`` / ``"lw"``),
        evaluation only -- see :func:`shrunk_residual_covariance`."""
        _, _, specific_returns = self._fitted_state()
        return shrunk_residual_covariance(
            specific_returns.loc[: pd.Timestamp(date)], self.cfg, window, method
        )


def build_risk_model(cfg: RiskConfig | None = None, **kwargs):
    """Factory: ``HybridRiskModel`` (fundamental + APCA companion on the
    residuals, see ``qts.risk.hybrid``) when ``cfg.statistical_companion`` is
    on, otherwise a plain ``BarraStyleRiskModel``.  ``kwargs`` (``warmup_days``,
    ``eigen_refresh_days``, ``seed``, ``constraint_weights``,
    ``weight_cap_quantile``) go to the fundamental model either way.  Both
    classes share ``fit(md) / as_of(date, alpha) / exposures_as_of(date) /
    factor_returns / specific_returns``."""
    cfg = cfg or RiskConfig()
    fundamental = BarraStyleRiskModel(cfg, **kwargs)
    if cfg.statistical_companion:
        from qts.risk.hybrid import HybridRiskModel  # local import: hybrid imports this module

        return HybridRiskModel(fundamental, cfg)
    return fundamental
