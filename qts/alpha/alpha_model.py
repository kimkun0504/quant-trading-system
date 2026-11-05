"""AlphaModel orchestrator: features -> targets -> composite (Baseline A) ->
walk-forward ML sleeve -> combined score -> Grinold expected returns.

``fit(md)`` does all heavy lifting once over the history; ``as_of(date)`` is a
cheap point-in-time lookup that assembles an :class:`AlphaOutput`.

Point-in-time guarantees in ``as_of(date)``:
* composite and ML scores at ``date`` are functions of features ``<= date``
  (features are PIT by construction; ML predictions are out-of-sample from the
  walk-forward trainer with embargo);
* the trailing OOS IC used for Grinold scaling only uses label rows that are
  decided by ``date``: forward window ``(s + L, s + L + h]`` ended
  (``L = signal_lag_days``, the execution lag the labels are aligned to) *and*
  every halt/delisting in the row resolved (``targets.label_realisation``);
* the half-life is estimated from scores ``<= date`` (recomputed once per
  ``refit_frequency_days`` block, from scores up to the block's first date, and
  cached — so it is PIT regardless of the order in which ``as_of`` is called);
* only names eligible at ``date`` (``md.eligibility()``: in the universe and
  with a price) receive a score; ineligible / dead names are absent from the
  output, never zero-filled.

Grinold scaling and the IC-shrink convention (design decision D5)
-----------------------------------------------------------------
``expected_return = IC_used * sigma_resid,h * z`` with
``IC_used = max(0, ic_shrink * IC_hat)``: ``ic_shrink`` is the fraction of the
estimated IC that is *kept* (0.5 => half).  ``IC_hat`` is the trailing OOS rank
IC over ``cfg.ic_window`` realised dates (needs ``cfg.ic_min_obs``); when it is
unavailable ``IC_hat := cfg.default_ic``, which is stated **pre-shrink** and is
shrunk the same way (0.03 x 0.5 = 0.015).  ``AlphaOutput.ic_estimate`` is
``IC_used``.  The portfolio layer must not shrink again.

Signal half-life and Garleanu-Pedersen shrink (handbook fix #5)
---------------------------------------------------------------
``AlphaOutput.half_life_days`` is the half-life of the *combined* score's
cross-sectional rank autocorrelation; :meth:`AlphaModel.component_half_lives_as_of`
gives the same statistic per sleeve (``composite``, ``ml``, ``theme_*``) and is
attached to the output as ``AlphaOutput.components_half_life``.  Both are PIT (estimated once per ``refit_frequency_days`` block from
the trailing ``half_life_lookback_days`` of scores up to the block start, cached).
The alpha layer does **not** apply Garleanu-Pedersen shrinkage itself: the
portfolio layer applies it in *alpha space* (aim portfolio of the
half-life-weighted alpha) when ``PortfolioConfig.use_gp_shrink`` is true.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import pandas as pd

from qts.core.config import AlphaConfig, CostConfig
from qts.core.crosssection import zscore
from qts.core.types import AlphaOutput, Long, MarketData, Wide

from .composite import ThemeComposite, theme_portfolio_returns
from .evaluation import rank_ic_series, signal_half_life
from .features import FeatureBuilder
from .models import EnsembleModel, LGBMModel, MLPModel, RidgeModel
from .targets import ExposuresFn, label_realisation, residualize_forward_returns
from .trainer import WalkForwardTrainer


def default_ml_factory(cfg: AlphaConfig) -> Callable[[], object]:
    """LightGBM + MLP rank-averaged (ridge kept as a diagnostic member with low
    weight).  Hyper-parameters: ``cfg.lgbm``, ``cfg.mlp``; member weights:
    ``cfg.ensemble_weights`` (keys ``lgbm``, ``mlp``, ``ridge``; a zero or
    missing weight drops the member)."""
    w = dict(cfg.ensemble_weights)

    def factory():
        members, weights = [], []
        if w.get("lgbm", 0.0) > 0:
            members.append(LGBMModel(n_seeds=cfg.model_seeds, params=cfg.lgbm))
            weights.append(w["lgbm"])
        if w.get("mlp", 0.0) > 0:
            members.append(MLPModel(n_seeds=cfg.model_seeds, params=cfg.mlp))
            weights.append(w["mlp"])
        if w.get("ridge", 0.0) > 0:
            members.append(RidgeModel())
            weights.append(w["ridge"])
        if not members:
            raise ValueError("AlphaConfig.ensemble_weights selects no ML member")
        return EnsembleModel(members, weights=weights)

    return factory


class AlphaModel:
    """Features -> labels -> composite + walk-forward ML -> Grinold expected returns.

    Every knob lives in :class:`~qts.core.config.AlphaConfig`; the keyword
    arguments ``composite_weight``, ``dynamic_weights`` and ``ic_window`` are
    optional per-instance overrides (``None`` -> config value).  ``delisting_return``
    defaults to ``CostConfig().delisting_haircut`` (the pipeline passes
    ``cfg.cost.delisting_haircut``).
    """

    def __init__(
        self,
        cfg: AlphaConfig | None = None,
        composite_weight: float | None = None,
        ml_factory: Callable[[], object] | None = None,
        use_ml: bool = True,
        dynamic_weights: bool | None = None,
        ic_window: int | None = None,
        delisting_return: float | None = None,
        signal_lag_days: int = 0,
    ):
        self.cfg = cfg or AlphaConfig()
        #: execution lag L (``ExecutionConfig.signal_lag_days``): the label of
        #: decision date s is the return close_{s+L} -> close_{s+L+h} (what the
        #: book trading the signal earns); purge and IC windows grow by L.
        self.signal_lag_days = int(signal_lag_days)
        #: label haircut for names delisting inside the forward window; ``None`` ->
        #: ``CostConfig().delisting_haircut`` (single source; pass 0.0 to disable)
        self.delisting_return = float(
            CostConfig().delisting_haircut if delisting_return is None else delisting_return
        )
        self.composite_weight = float(
            self.cfg.composite_weight if composite_weight is None else composite_weight
        )
        self.ml_factory = ml_factory
        self.use_ml = use_ml
        self.dynamic_weights = bool(self.cfg.dynamic_weights if dynamic_weights is None else dynamic_weights)
        self.ic_window = int(self.cfg.ic_window if ic_window is None else ic_window)
        self.features = FeatureBuilder(self.cfg)
        self.composite = ThemeComposite(self.cfg)
        self.trainer = WalkForwardTrainer(self.cfg, label_lag_days=self.signal_lag_days)
        self.fitted_ = False
        self._half_life_cache: dict[tuple, float] = {}

    # ------------------------------------------------------------------ fit
    def fit(self, md: MarketData, risk_exposures_fn: ExposuresFn | None = None) -> AlphaModel:
        h = self.cfg.horizon_days
        self.md_ = md
        self.eligible_ = md.eligibility()
        self.raw_features_ = self.features.build(md)
        self.X_: Long = self.features.process(self.raw_features_, md, valid=self.eligible_)
        # labels: used ONLY as training targets / for OOS evaluation
        L = self.signal_lag_days
        G = int(self.cfg.delisting_gap_days)
        self.target_: Wide = residualize_forward_returns(
            md,
            h,
            exposures_fn=risk_exposures_fn,
            features=self.raw_features_,
            delisting_return=self.delisting_return,
            label_lag_days=L,
            delisting_gap_days=G,
        )
        # date at which each label row is fully decided (NaT: not within the data)
        self.label_ready_: pd.Series = label_realisation(
            md, h, label_lag_days=L, delisting_gap_days=G, valid=self.eligible_
        )
        self.theme_returns_ = theme_portfolio_returns(self.X_, md)
        self.composite_score_, self.theme_scores_ = self.composite.score_all(
            self.X_,
            md=md,
            dynamic=self.dynamic_weights,
            theme_returns=self.theme_returns_,
            return_components=True,
        )
        self.ml_score_: Wide | None = None
        if self.use_ml:
            y_long = self.target_.stack(future_stack=True)
            y_long.index.names = ["date", "ticker"]
            y_long = y_long.reindex(self.X_.index)
            factory = self.ml_factory or default_ml_factory(self.cfg)
            ml = self.trainer.run(self.X_, y_long, factory, label_ready=self.label_ready_)
            self.ml_score_ = zscore(ml) if ml.notna().any().any() else None
        self.combined_score_ = self._combine_history().where(
            self.eligible_.reindex_like(self.composite_score_)
        )
        self._half_life_cache = {}
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
        fully realised by ``date``: label at s needs ``s + h + L <= date`` and
        its row decided (``label_ready_[s] <= date``; halts / delistings)."""
        h = self.cfg.horizon_days + self.signal_lag_days
        dates = self.combined_score_.index
        pos = dates.searchsorted(date, side="right") - 1
        end = pos - h
        if end < 0:
            return float("nan")
        win = self.oos_ic_.iloc[max(0, end - self.ic_window + 1) : end + 1]
        ready = self.label_ready_.reindex(win.index)
        cutoff = dates[pos]
        ic = win[(ready.notna() & (ready <= cutoff)).to_numpy()].dropna()
        return float(ic.mean()) if len(ic) >= int(self.cfg.ic_min_obs) else float("nan")

    def ic_used(self, date: pd.Timestamp) -> float:
        """D5: ``max(0, ic_shrink * IC_hat)``, ``IC_hat`` = trailing OOS IC or,
        when unavailable, the pre-shrink prior ``default_ic``."""
        ic_hat = self.trailing_ic(date)
        if not np.isfinite(ic_hat):
            ic_hat = float(self.cfg.default_ic)
        return max(float(self.cfg.ic_shrink) * ic_hat, 0.0)

    def _sleeve_scores(self) -> dict[str, Wide]:
        out: dict[str, Wide] = {"combined": self.combined_score_, "composite": self.composite_score_}
        if self.ml_score_ is not None:
            out["ml"] = self.ml_score_
        for th, sc in self.theme_scores_.items():
            out[f"theme_{th}"] = sc
        return out

    def _half_life(self, sleeve: str, panel: Wide, date: pd.Timestamp) -> float:
        dates = self.combined_score_.index
        pos = int(dates.searchsorted(date, side="right") - 1)
        block = max(1, int(self.cfg.refit_frequency_days))
        key = (sleeve, pos // block)
        hl = self._half_life_cache.get(key)
        if hl is None:
            start = dates[max(pos, 0) // block * block]
            hist = panel.loc[:start]
            hl = signal_half_life(hist.iloc[-int(self.cfg.half_life_lookback_days) :])
            if not np.isfinite(hl):
                hl = float(self.cfg.horizon_days * 3)
            self._half_life_cache[key] = float(hl)
        return float(hl)

    def half_life_as_of(self, date: pd.Timestamp) -> float:
        """Signal half-life (days) of the combined score, estimated once per
        ``refit_frequency_days`` block from the trailing
        ``half_life_lookback_days`` (504) dates of scores up to the block's
        first date (PIT for every date in the block) and cached.  Not finite
        (no decay / too little history) -> ``3 * horizon_days``."""
        return self._half_life("combined", self.combined_score_, date)

    def component_half_lives_as_of(self, date: pd.Timestamp) -> dict[str, float]:
        """Per-sleeve half-lives (same estimator / caching as :meth:`half_life_as_of`):
        keys ``combined``, ``composite``, ``ml`` (when fitted) and ``theme_<name>``."""
        return {k: self._half_life(k, v, date) for k, v in self._sleeve_scores().items()}

    def as_of(self, date: pd.Timestamp) -> AlphaOutput:
        if not self.fitted_:
            raise RuntimeError("call fit() first")
        date = cast(pd.Timestamp, pd.Timestamp(date))
        if date not in self.combined_score_.index:
            earlier = self.combined_score_.index[self.combined_score_.index <= date]
            if len(earlier) == 0:
                raise KeyError(f"no scores on or before {date}")
            date = pd.Timestamp(earlier[-1])
        # eligible at `date` (in universe and alive): never score dead / out-of-universe names
        ok = self.eligible_.loc[date]
        score = (
            self.combined_score_.loc[date]
            .where(ok.reindex(self.combined_score_.columns).fillna(False).astype(bool))
            .dropna()
        )
        score = (score - score.mean()) / score.std(ddof=0)
        # Grinold: alpha = IC * sigma_resid * z
        ic_shrunk = self.ic_used(date)  # D5: shrink applies to the prior too
        vol = self.resid_vol_h_.loc[date].reindex(score.index)
        vol = vol.fillna(vol.median()).fillna(0.0)
        expected = (ic_shrunk * vol * score).rename("expected_return")
        hl = self.half_life_as_of(date)
        components: dict[str, pd.Series] = {
            f"theme_{th}": s.loc[date].reindex(score.index) for th, s in self.theme_scores_.items()
        }
        components["composite"] = self.composite_score_.loc[date].reindex(score.index)
        if self.ml_score_ is not None:
            components["ml"] = self.ml_score_.loc[date].reindex(score.index)
        return AlphaOutput(
            date=date,
            score=score.rename("score"),
            expected_return=expected,
            horizon_days=self.cfg.horizon_days,
            half_life_days=float(hl),
            components=components,
            ic_estimate=float(ic_shrunk),
            components_half_life=self.component_half_lives_as_of(date),
        )
