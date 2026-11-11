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

Grinold scaling and the IC-shrink convention (design decisions D5, D6)
----------------------------------------------------------------------
``expected_return = IC_used * sigma_resid,h * z`` with
``IC_used = max(ic_floor_frac * ic_shrink * default_ic, ic_shrink * IC_hat)``:
``ic_shrink`` is the fraction of the estimated IC that is *kept* (0.5 => half).
``IC_hat`` (D6) is the precision-weighted blend
``(n * IC_trailing + k * default_ic) / (n + k)`` of the trailing OOS rank IC
over ``cfg.ic_window`` realised dates (``n`` of them; ``n := 0`` below
``cfg.ic_min_obs``) and the pre-shrink prior ``cfg.default_ic`` weighted as
``k = cfg.ic_prior_obs`` dates.  With no realised IC the prior is shrunk the
same way (0.03 x 0.5 = 0.015).  The floor (a quarter of the shrunk prior by
default) means a negative trailing IC shrinks positions rather than emptying
the book, which the former ``max(0, .)`` floor did on about half of the
real-data dates (docs/notes/ic_floor_fix_2026-11-09.md).  ``AlphaOutput.ic_estimate``
is ``IC_used``.  The portfolio layer must not shrink again.

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
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd

from qts.core import stats
from qts.core.config import AlphaConfig, CostConfig
from qts.core.crosssection import zscore
from qts.core.types import AlphaOutput, Long, MarketData, Wide

from .composite import THEMES, ThemeComposite, theme_portfolio_returns
from .evaluation import newey_west_lags, rank_ic_series, signal_half_life
from .features import FeatureBuilder
from .models import EnsembleModel, LGBMModel, MLPModel, RidgeModel
from .targets import DEFAULT_BAD_PRICE_CAPS, ExposuresFn, label_realisation, residualize_forward_returns
from .trainer import FoldInfo, WalkForwardTrainer

#: Sleeve name of each ML member class in the refit record (``alpha_refits.parquet``).
SLEEVE_OF_MODEL: dict[str, str] = {"LGBMModel": "lgbm", "RidgeModel": "ridge", "MLPModel": "mlp"}


@dataclass(frozen=True)
class RefitRecord:
    """What one walk-forward refit learned: the window it was trained on and the
    feature importance of every sleeve (``lgbm``: mean gain share over the seed
    bag; ``ridge``: |coef|; ``ensemble``: the member average; ``mlp``: none, a
    permutation importance would cost another full pass over the validation
    block, so it is omitted and ``notes`` says so).  ``refit_date`` is the first
    date the model predicts for; its training labels end at ``train_end``,
    ``embargo`` days earlier (see :mod:`qts.alpha.trainer`)."""

    refit_date: pd.Timestamp
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    valid_start: pd.Timestamp | None
    pred_end: pd.Timestamp
    n_train: int
    n_valid: int
    importance: dict[str, pd.Series] = field(default_factory=dict)
    blend_weights: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _sleeve_name(model: object) -> str:
    name = type(model).__name__
    return SLEEVE_OF_MODEL.get(name, name.lower().removesuffix("model") or name.lower())


def refit_record(fold: FoldInfo, model: Any) -> RefitRecord:
    """Per-sleeve feature importance of one fitted walk-forward model (ensemble
    members by sleeve name plus ``ensemble``; a bare model under its own name)."""
    importance: dict[str, pd.Series] = {}
    blend: dict[str, float] = {}
    notes: list[str] = []
    members = list(getattr(model, "models", None) or [])
    if members:
        weights = list(getattr(model, "weights", [1.0] * len(members)))
        for m, w in zip(members, weights, strict=False):
            name = _sleeve_name(m)
            blend[name] = float(w)
            _add_importance(importance, notes, name, m)
        _add_importance(importance, notes, "ensemble", model)
    else:
        _add_importance(importance, notes, _sleeve_name(model), model)
    return RefitRecord(
        refit_date=pd.Timestamp(fold.pred_start),
        train_start=pd.Timestamp(fold.train_start),
        train_end=pd.Timestamp(fold.train_end),
        valid_start=None if fold.valid_start is None else pd.Timestamp(fold.valid_start),
        pred_end=pd.Timestamp(fold.pred_end),
        n_train=int(fold.n_train_rows),
        n_valid=int(fold.n_valid_rows),
        importance=importance,
        blend_weights=blend,
        notes=notes,
    )


def _add_importance(out: dict[str, pd.Series], notes: list[str], name: str, model: Any) -> None:
    fn = getattr(model, "feature_importance", None)
    imp = None
    if callable(fn):
        try:
            imp = fn()
        except Exception as e:  # a member without a fitted importance never fails the run
            notes.append(f"{name}: feature importance unavailable ({type(e).__name__})")
    if imp is None:
        if name == "mlp":
            notes.append("mlp: no feature importance recorded (permutation importance is not computed)")
        return
    s = pd.Series(cast(Any, imp), dtype=float)
    s = s[np.isfinite(s.to_numpy())]
    if len(s):
        out[name] = s


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
        bad_price_caps: tuple[float, float] | None = DEFAULT_BAD_PRICE_CAPS,
    ):
        self.cfg = cfg or AlphaConfig()
        #: data-error return caps applied to the label closes (``RiskConfig``
        #: ``return_abs_cap_up/down``; the pipeline threads the run's values)
        self.bad_price_caps = bad_price_caps
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
        #: one :class:`RefitRecord` per walk-forward refit of the ML sleeve (empty without ML)
        self.refit_records_: list[RefitRecord] = []

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
            bad_price_caps=self.bad_price_caps,
        )
        # date at which each label row is fully decided (NaT: not within the data)
        self.label_ready_: pd.Series = label_realisation(
            md,
            h,
            label_lag_days=L,
            delisting_gap_days=G,
            valid=self.eligible_,
            bad_price_caps=self.bad_price_caps,
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
            # what each refit learned (recording only: never read back by the scoring)
            self.refit_records_ = [
                refit_record(f, m) for f, m in zip(self.trainer.folds_, self.trainer.models_, strict=True)
            ]
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
    def trailing_ic_obs(self, date: pd.Timestamp) -> tuple[float, int]:
        """``(mean, n)`` of the OOS rank IC over the trailing ``ic_window`` dates
        whose labels are fully realised by ``date``: label at s needs
        ``s + h + L <= date`` and its row decided (``label_ready_[s] <= date``;
        halts / delistings).  ``n`` is the number of realised IC dates; the mean
        is NaN when ``n < ic_min_obs``."""
        h = self.cfg.horizon_days + self.signal_lag_days
        dates = self.combined_score_.index
        pos = dates.searchsorted(date, side="right") - 1
        end = pos - h
        if end < 0:
            return float("nan"), 0
        win = self.oos_ic_.iloc[max(0, end - self.ic_window + 1) : end + 1]
        ready = self.label_ready_.reindex(win.index)
        cutoff = dates[pos]
        ic = win[(ready.notna() & (ready <= cutoff)).to_numpy()].dropna()
        n = len(ic)
        return (float(ic.mean()) if n >= int(self.cfg.ic_min_obs) else float("nan")), n

    def trailing_ic(self, date: pd.Timestamp) -> float:
        """Mean trailing OOS rank IC (NaN below ``ic_min_obs`` realised dates)."""
        return self.trailing_ic_obs(date)[0]

    def ic_hat(self, date: pd.Timestamp) -> float:
        """D6: precision-weighted blend of the trailing OOS IC and the pre-shrink
        prior, ``(n IC_trailing + k default_ic) / (n + k)`` with
        ``k = ic_prior_obs``; ``n := 0`` (prior only) below ``ic_min_obs``."""
        mean, n = self.trailing_ic_obs(date)
        k = float(self.cfg.ic_prior_obs)
        prior = float(self.cfg.default_ic)
        if not np.isfinite(mean) or n <= 0:
            return prior
        return (n * mean + k * prior) / (n + k)

    def ic_used(self, date: pd.Timestamp) -> float:
        """D5/D6: ``max(ic_floor_frac * ic_shrink * default_ic, ic_shrink * IC_hat)``.
        The floor keeps a weak or temporarily negative signal invested at a
        fraction of its prior size instead of liquidating the book."""
        keep = float(self.cfg.ic_shrink)
        floor = float(self.cfg.ic_floor_frac) * keep * float(self.cfg.default_ic)
        return max(keep * self.ic_hat(date), floor)

    def _sleeve_scores(self) -> dict[str, Wide]:
        out: dict[str, Wide] = {"combined": self.combined_score_, "composite": self.composite_score_}
        if self.ml_score_ is not None:
            out["ml"] = self.ml_score_
        for th, sc in self.theme_scores_.items():
            out[f"theme_{th}"] = sc
        return out

    # ------------------------------------------------------------- recordings
    def sleeve_ic_series(self) -> pd.DataFrame:
        """Daily OOS rank IC of every sleeve (``combined``, ``composite``, ``ml`` when
        fitted, ``theme_<name>``) against the training label (``target_``): date x
        sleeve.  Diagnostic only (``alpha_ic_by_sleeve.parquet``); the ``combined``
        column equals ``oos_ic_``."""
        if not self.fitted_:
            raise RuntimeError("call fit() first")
        cols = {name: rank_ic_series(panel, self.target_) for name, panel in self._sleeve_scores().items()}
        out = pd.DataFrame(cols)
        out.index = pd.DatetimeIndex(out.index, name="date")
        return out

    def feature_ic_table(self) -> pd.DataFrame:
        """Full-sample daily rank IC of every processed feature, with the sign the
        composite uses (``THEMES``; +1 for features only the ML sleeve sees):
        columns ``feature, sign, theme, mean_ic, t_stat, n``.  ``t_stat`` is
        Newey-West with the label overlap (``horizon_days + signal_lag_days``) as
        the bandwidth floor."""
        if not self.fitted_:
            raise RuntimeError("call fit() first")
        signs: dict[str, tuple[float, str]] = {}
        for theme, members in THEMES.items():
            for f, sgn in members:
                signs[f] = (float(sgn), theme)
        overlap = int(self.cfg.horizon_days) + int(self.signal_lag_days)
        rows = []
        for f in self.X_.columns:
            sgn, theme = signs.get(str(f), (1.0, ""))
            panel = self.X_[f].unstack("ticker") * sgn
            ic = rank_ic_series(panel, self.target_).dropna()
            n = len(ic)
            if n:
                lags = newey_west_lags(n, overlap)
                t = float(stats.newey_west_tstat(ic.to_numpy(dtype=float), lags=int(lags)))
                mean = float(ic.mean())
            else:
                t = mean = float("nan")
            rows.append(
                {"feature": str(f), "sign": sgn, "theme": theme, "mean_ic": mean, "t_stat": t, "n": n}
            )
        return pd.DataFrame(rows, columns=["feature", "sign", "theme", "mean_ic", "t_stat", "n"])

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
