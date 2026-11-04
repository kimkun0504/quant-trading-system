"""AlphaModel orchestrator: features -> targets -> composite (Baseline A) ->
walk-forward ML sleeve -> combined score -> Grinold expected returns.

``fit(md)`` does all heavy lifting once over the history; ``as_of(date)`` is a
cheap point-in-time lookup that assembles an :class:`AlphaOutput`.

Point-in-time guarantees in ``as_of(date)``:
* composite and ML scores at ``date`` are functions of features ``<= date``
  (features are PIT by construction; ML predictions are out-of-sample from the
  walk-forward trainer with embargo);
* the trailing OOS IC used for Grinold scaling only uses labels whose forward
  window ended on or before ``date``;
* the half-life is estimated from scores ``<= date``.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from qts.core.config import AlphaConfig
from qts.core.crosssection import zscore
from qts.core.types import AlphaOutput, Long, MarketData, Wide

from .composite import ThemeComposite, theme_portfolio_returns
from .evaluation import rank_ic_series, signal_half_life
from .features import FeatureBuilder
from .models import EnsembleModel, LGBMModel, MLPModel, RidgeModel
from .targets import ExposuresFn, residualize_forward_returns
from .trainer import WalkForwardTrainer


def default_ml_factory(cfg: AlphaConfig) -> Callable[[], object]:
    """LightGBM + MLP rank-averaged (ridge kept as a diagnostic member with low weight)."""
    def factory():
        return EnsembleModel(
            [LGBMModel(n_seeds=cfg.model_seeds),
             MLPModel(n_seeds=cfg.model_seeds, max_train_rows=400_000),
             RidgeModel()],
            weights=[0.45, 0.35, 0.20])
    return factory


class AlphaModel:
    def __init__(self, cfg: Optional[AlphaConfig] = None, composite_weight: float = 0.5,
                 ml_factory: Optional[Callable[[], object]] = None, use_ml: bool = True,
                 dynamic_weights: bool = True, ic_window: int = 252):
        self.cfg = cfg or AlphaConfig()
        self.composite_weight = float(composite_weight)
        self.ml_factory = ml_factory
        self.use_ml = use_ml
        self.dynamic_weights = dynamic_weights
        self.ic_window = ic_window
        self.features = FeatureBuilder(self.cfg)
        self.composite = ThemeComposite(self.cfg)
        self.trainer = WalkForwardTrainer(self.cfg)
        self.fitted_ = False

    # ------------------------------------------------------------------ fit
    def fit(self, md: MarketData, risk_exposures_fn: Optional[ExposuresFn] = None) -> "AlphaModel":
        h = self.cfg.horizon_days
        self.md_ = md
        self.raw_features_ = self.features.build(md)
        self.X_: Long = self.features.process(self.raw_features_, md)
        # labels: used ONLY as training targets / for OOS evaluation
        self.target_: Wide = residualize_forward_returns(md, h, exposures_fn=risk_exposures_fn,
                                                         features=self.raw_features_)
        self.theme_returns_ = theme_portfolio_returns(self.X_, md)
        self.composite_score_, self.theme_scores_ = self.composite.score_all(
            self.X_, md=md, dynamic=self.dynamic_weights, theme_returns=self.theme_returns_,
            return_components=True)
        self.ml_score_: Optional[Wide] = None
        if self.use_ml:
            y_long = self.target_.stack(future_stack=True)
            y_long.index.names = ["date", "ticker"]
            y_long = y_long.reindex(self.X_.index)
            factory = self.ml_factory or default_ml_factory(self.cfg)
            ml = self.trainer.run(self.X_, y_long, factory)
            self.ml_score_ = zscore(ml) if ml.notna().any().any() else None
        self.combined_score_ = self._combine_history()
        # per-date OOS rank IC of the combined score (label realised at date + h)
        self.oos_ic_ = rank_ic_series(self.combined_score_, self.target_)
        # daily residual vol -> horizon residual vol
        self.resid_vol_h_ = self.raw_features_["idio_vol_252"] * np.sqrt(h)
        self.fitted_ = True
        return self

    def _combine_history(self) -> Wide:
        comp = self.composite_score_
        if self.ml_score_ is None:
            return comp
        ml = self.ml_score_.reindex_like(comp)
        wc, wm = self.composite_weight, 1.0 - self.composite_weight
        have_ml = ml.notna()
        num = comp.fillna(0.0) * wc + ml.fillna(0.0) * wm
        den = comp.notna() * wc + have_ml * wm
        return zscore(num / den.replace(0, np.nan))

    # ------------------------------------------------------------------ as_of
    def trailing_ic(self, date: pd.Timestamp) -> float:
        """Mean OOS rank IC over the trailing ``ic_window`` dates whose labels are
        fully realised by ``date`` (label at s needs s + h <= date)."""
        h = self.cfg.horizon_days
        dates = self.combined_score_.index
        pos = dates.searchsorted(date, side="right") - 1
        end = pos - h
        if end < 0:
            return float("nan")
        ic = self.oos_ic_.iloc[max(0, end - self.ic_window + 1): end + 1].dropna()
        return float(ic.mean()) if len(ic) >= 60 else float("nan")

    def as_of(self, date: pd.Timestamp) -> AlphaOutput:
        if not self.fitted_:
            raise RuntimeError("call fit() first")
        date = pd.Timestamp(date)
        if date not in self.combined_score_.index:
            date = self.combined_score_.index[self.combined_score_.index <= date][-1]
        score = self.combined_score_.loc[date].dropna()
        score = (score - score.mean()) / score.std(ddof=0)
        # Grinold: alpha = IC * sigma_resid * z
        ic_tr = self.trailing_ic(date)
        ic_shrunk = self.cfg.ic_shrink * ic_tr if np.isfinite(ic_tr) else self.cfg.default_ic
        ic_shrunk = max(ic_shrunk, 0.0)
        vol = self.resid_vol_h_.loc[date].reindex(score.index)
        vol = vol.fillna(vol.median()).fillna(0.0)
        expected = (ic_shrunk * vol * score).rename("expected_return")
        hist = self.combined_score_.loc[:date]
        hl = signal_half_life(hist.iloc[-504:])
        if not np.isfinite(hl):
            hl = float(self.cfg.horizon_days * 3)
        components: Dict[str, pd.Series] = {
            f"theme_{th}": s.loc[date].reindex(score.index) for th, s in self.theme_scores_.items()}
        components["composite"] = self.composite_score_.loc[date].reindex(score.index)
        if self.ml_score_ is not None:
            components["ml"] = self.ml_score_.loc[date].reindex(score.index)
        return AlphaOutput(date=date, score=score.rename("score"), expected_return=expected,
                           horizon_days=self.cfg.horizon_days, half_life_days=float(hl),
                           components=components, ic_estimate=float(ic_shrunk))
