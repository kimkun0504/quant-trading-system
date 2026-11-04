"""Estimators for the alpha layer.

All models share the interface

    fit(X: Long, y: Series aligned to X.index, valid_mask: Optional[bool Series]) -> self
    predict(X: Long) -> Series aligned to X.index
    feature_importance() -> Series | None

``X`` is the processed long feature frame (rank-gaussian features, no NaN);
``y`` is the residualised, per-date z-scored forward return.  ``valid_mask``
marks the trailing validation block supplied by the walk-forward trainer; it
is used for early stopping / hyper-parameter selection only.  Rows with
``valid_mask == False`` are the training rows.  When ``valid_mask`` is None the
last 20% of dates are used as the validation block.
"""
from __future__ import annotations

import inspect
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from qts.core.config import AlphaConfig

from .composite import ThemeComposite
from .features import feature_signature


# ----------------------------------------------------------------------------- utilities
def _dates(X: pd.DataFrame) -> pd.Index:
    return X.index.get_level_values("date")


def default_valid_mask(X: pd.DataFrame, frac: float = 0.2) -> pd.Series:
    d = _dates(X)
    uniq = d.unique().sort_values()
    cut = uniq[int(len(uniq) * (1 - frac))]
    return pd.Series(d >= cut, index=X.index)


def rank_by_date(s: pd.Series) -> pd.Series:
    """Per-date percentile rank in (0, 1)."""
    return s.groupby(level="date").rank(pct=True)


def zscore_by_date(s: pd.Series) -> pd.Series:
    g = s.groupby(level="date")
    sd = g.transform("std").replace(0, np.nan)
    return ((s - g.transform("mean")) / sd).fillna(0.0)


def mean_rank_ic(pred: pd.Series, y: pd.Series) -> float:
    """Mean per-date Spearman correlation."""
    return float(np.nanmean(rank_ic_by_date(pred, y).values))


def rank_ic_by_date(pred: pd.Series, y: pd.Series) -> pd.Series:
    """Per-date Spearman correlation between two long Series (vectorised)."""
    df = pd.DataFrame({"p": pred, "y": y}).dropna()
    g = df.groupby(level="date")
    rp, ry = g["p"].rank(), g["y"].rank()
    rp = rp - rp.groupby(level="date").transform("mean")
    ry = ry - ry.groupby(level="date").transform("mean")
    num = (rp * ry).groupby(level="date").sum()
    den = np.sqrt((rp ** 2).groupby(level="date").sum() * (ry ** 2).groupby(level="date").sum())
    n = g.size()
    return (num / den.replace(0, np.nan)).where(n >= 20)


def make_rank_ic_metric(dates):
    """LightGBM sklearn-API eval metric: mean per-date Spearman IC (higher is better).
    ``dates`` are the date labels of the validation rows, in row order."""
    codes, _ = pd.factorize(np.asarray(dates))
    n = np.bincount(codes).astype(float)
    cache = {}

    def _ranks(v):
        return pd.Series(v).groupby(codes).rank().to_numpy()

    def _center(r):
        mu = np.bincount(codes, r) / n
        return r - mu[codes]

    def metric(y_true, y_pred):
        key = id(y_true)
        if key not in cache:
            cache.clear()
            cache[key] = _center(_ranks(y_true))
        rt = cache[key]
        rp = _center(_ranks(y_pred))
        num = np.bincount(codes, rt * rp)
        den = np.sqrt(np.bincount(codes, rt ** 2) * np.bincount(codes, rp ** 2))
        with np.errstate(invalid="ignore", divide="ignore"):
            ic = num / den
        ic = ic[(n >= 20) & np.isfinite(ic)]
        return "rank_ic", float(ic.mean()) if len(ic) else 0.0, True

    return metric


class _Base:
    feature_names_: Optional[List[str]] = None

    def _check_features(self, X: pd.DataFrame) -> None:
        if self.feature_names_ is not None and list(X.columns) != self.feature_names_:
            raise ValueError("feature columns/order differ from those used in fit "
                             f"({feature_signature(X.columns)} vs {feature_signature(self.feature_names_)})")

    @property
    def feature_hash_(self) -> Optional[str]:
        return None if self.feature_names_ is None else feature_signature(self.feature_names_)

    def feature_importance(self) -> Optional[pd.Series]:
        return None


# ----------------------------------------------------------------------------- A: rank composite
class RankCompositeModel(_Base):
    """Baseline A: fixed-weight theme composite.  No fitting."""

    def __init__(self, cfg: Optional[AlphaConfig] = None):
        self.cfg = cfg or AlphaConfig()
        self.composite = ThemeComposite(self.cfg)

    def fit(self, X, y=None, valid_mask=None):
        self.feature_names_ = list(X.columns)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        comp = self.composite.score_all(X)
        s = comp.stack(future_stack=True)
        s.index.names = ["date", "ticker"]
        return s.reindex(X.index).fillna(0.0).rename("pred")


# ----------------------------------------------------------------------------- B: ridge
class RidgeModel(_Base):
    """sklearn Ridge; alpha chosen on the trailing validation block by mean rank IC,
    then refit on train + validation."""

    def __init__(self, alphas: Sequence[float] = (1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0)):
        self.alphas = list(alphas)
        self.model_ = None
        self.best_alpha_ = None

    def fit(self, X, y, valid_mask=None):
        from sklearn.linear_model import Ridge
        self.feature_names_ = list(X.columns)
        vm = default_valid_mask(X) if valid_mask is None else valid_mask.reindex(X.index).fillna(False)
        keep = y.notna()
        Xt, yt = X[keep & ~vm], y[keep & ~vm]
        Xv, yv = X[keep & vm], y[keep & vm]
        best, best_ic = self.alphas[0], -np.inf
        if len(Xv) > 0:
            for a in self.alphas:
                m = Ridge(alpha=a).fit(Xt.values, yt.values)
                ic = mean_rank_ic(pd.Series(m.predict(Xv.values), index=Xv.index), yv)
                if np.isfinite(ic) and ic > best_ic:
                    best, best_ic = a, ic
        self.best_alpha_, self.valid_ic_ = best, best_ic
        self.model_ = Ridge(alpha=best).fit(X[keep].values, y[keep].values)
        return self

    def predict(self, X):
        self._check_features(X)
        return pd.Series(self.model_.predict(X.values), index=X.index, name="pred")

    def feature_importance(self):
        return pd.Series(self.model_.coef_, index=self.feature_names_).abs().sort_values(ascending=False)


# ----------------------------------------------------------------------------- LightGBM
class LGBMModel(_Base):
    """LightGBM regressor, seed-bagged, early-stopped on the validation block.

    After early stopping the model is refit on train + validation with the
    best iteration count (``refit_on_full``), so the most recent year of labels
    is not wasted.
    """

    def __init__(self, n_seeds: int = 3, n_estimators: int = 600, learning_rate: float = 0.03,
                 num_leaves: int = 31, min_data_in_leaf: int = 200, feature_fraction: float = 0.7,
                 bagging_fraction: float = 0.8, bagging_freq: int = 1, lambda_l2: float = 1.0,
                 early_stopping_rounds: int = 100, refit_on_full: bool = True, n_jobs: int = 4,
                 base_seed: int = 0, **extra):
        self.params = dict(objective="regression", n_estimators=n_estimators, learning_rate=learning_rate,
                           num_leaves=num_leaves, min_child_samples=min_data_in_leaf,
                           colsample_bytree=feature_fraction, subsample=bagging_fraction,
                           subsample_freq=bagging_freq, reg_lambda=lambda_l2, n_jobs=n_jobs,
                           metric="None", verbose=-1, **extra)
        self.n_seeds = n_seeds
        self.early_stopping_rounds = early_stopping_rounds
        self.refit_on_full = refit_on_full
        self.base_seed = base_seed
        self.models_: List = []
        self.best_iterations_: List[int] = []

    def fit(self, X, y, valid_mask=None):
        import lightgbm as lgb
        self.feature_names_ = list(X.columns)
        vm = default_valid_mask(X) if valid_mask is None else valid_mask.reindex(X.index).fillna(False)
        keep = y.notna()
        Xt, yt = X[keep & ~vm].values, y[keep & ~vm].values
        Xv, yv = X[keep & vm].values, y[keep & vm].values
        # Early stopping on per-date rank IC of the validation block: the l2 of a
        # z-scored, very noisy target keeps *rising* with boosting rounds (scale
        # miscalibration) even while the ranking improves, so l2 stops at ~0 trees.
        ic_metric = make_rank_ic_metric(_dates(X[keep & vm]))
        self.models_, self.best_iterations_ = [], []
        for s in range(self.n_seeds):
            seed = self.base_seed + s
            m = lgb.LGBMRegressor(random_state=seed, **self.params)
            if len(Xv) > 0:
                cb = [lgb.early_stopping(self.early_stopping_rounds, first_metric_only=True, verbose=False)]
                if "eval_X" in inspect.signature(m.fit).parameters:   # lightgbm >= 4.7
                    m.fit(Xt, yt, eval_X=Xv, eval_y=yv, eval_metric=ic_metric, callbacks=cb)
                else:
                    m.fit(Xt, yt, eval_set=[(Xv, yv)], eval_metric=ic_metric, callbacks=cb)
                best = int(m.best_iteration_ or self.params["n_estimators"])
            else:
                m.fit(Xt, yt)
                best = self.params["n_estimators"]
            best = max(best, 10)
            self.best_iterations_.append(best)
            if self.refit_on_full and len(Xv) > 0:
                p = dict(self.params, n_estimators=best)
                m = lgb.LGBMRegressor(random_state=seed, **p).fit(X[keep].values, y[keep].values)
            self.models_.append(m)
        return self

    def predict(self, X):
        self._check_features(X)
        p = np.mean([m.predict(X.values) for m in self.models_], axis=0)
        return pd.Series(p, index=X.index, name="pred")

    def feature_importance(self):
        imp = np.mean([m.booster_.feature_importance(importance_type="gain") for m in self.models_], axis=0)
        s = pd.Series(imp, index=self.feature_names_)
        return (s / s.sum()).sort_values(ascending=False)


# ----------------------------------------------------------------------------- MLP
class MLPModel(_Base):
    """sklearn MLPRegressor (64/32), standardised inputs, early stopping, seed-bagged."""

    def __init__(self, n_seeds: int = 3, hidden=(64, 32), alpha: float = 1e-3, max_iter: int = 200,
                 batch_size: int = 2048, learning_rate_init: float = 1e-3, base_seed: int = 0,
                 max_train_rows: Optional[int] = None):
        self.n_seeds, self.hidden, self.alpha, self.max_iter = n_seeds, hidden, alpha, max_iter
        self.batch_size, self.lr, self.base_seed = batch_size, learning_rate_init, base_seed
        self.max_train_rows = max_train_rows
        self.models_: List = []

    def fit(self, X, y, valid_mask=None):
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        self.feature_names_ = list(X.columns)
        keep = y.notna()
        Xa, ya = X[keep], y[keep]
        if self.max_train_rows is not None and len(Xa) > self.max_train_rows:
            rs = np.random.RandomState(self.base_seed)
            idx = np.sort(rs.choice(len(Xa), self.max_train_rows, replace=False))
            Xa, ya = Xa.iloc[idx], ya.iloc[idx]
        self.scaler_ = StandardScaler().fit(Xa.values)
        Xs = self.scaler_.transform(Xa.values)
        self.models_ = []
        for s in range(self.n_seeds):
            # sklearn's early_stopping uses a random 10% holdout; validation_fraction
            # of time-contiguous data is not supported, so the trainer's embargoed
            # window boundary is the real protection against leakage here.
            m = MLPRegressor(hidden_layer_sizes=self.hidden, alpha=self.alpha, max_iter=self.max_iter,
                             early_stopping=True, n_iter_no_change=10, validation_fraction=0.1,
                             batch_size=min(self.batch_size, len(Xs)), learning_rate_init=self.lr,
                             random_state=self.base_seed + s)
            m.fit(Xs, ya.values)
            self.models_.append(m)
        return self

    def predict(self, X):
        self._check_features(X)
        Xs = self.scaler_.transform(X.values)
        p = np.mean([m.predict(Xs) for m in self.models_], axis=0)
        return pd.Series(p, index=X.index, name="pred")


# ----------------------------------------------------------------------------- ensemble
class EnsembleModel(_Base):
    """Rank-average of member predictions per date, then per-date z-score.
    Weights are fixed ex ante (not fitted)."""

    def __init__(self, models: Sequence, weights: Optional[Sequence[float]] = None):
        self.models = list(models)
        w = np.ones(len(self.models)) if weights is None else np.asarray(weights, float)
        self.weights = w / w.sum()

    def fit(self, X, y, valid_mask=None):
        self.feature_names_ = list(X.columns)
        for m in self.models:
            m.fit(X, y, valid_mask)
        return self

    def predict(self, X):
        self._check_features(X)
        acc = None
        for w, m in zip(self.weights, self.models):
            r = rank_by_date(m.predict(X)) - 0.5
            acc = r * w if acc is None else acc + r * w
        return zscore_by_date(acc).rename("pred")

    def feature_importance(self):
        parts = [m.feature_importance() for m in self.models]
        parts = [p for p in parts if p is not None]
        if not parts:
            return None
        return pd.concat(parts, axis=1).mean(axis=1).sort_values(ascending=False)

    def member_predictions(self, X) -> Dict[str, pd.Series]:
        return {type(m).__name__: m.predict(X) for m in self.models}
