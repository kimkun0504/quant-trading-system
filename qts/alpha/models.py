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
last 20% of dates (MLP: ``MLPParams.valid_fraction``) are used as the
validation block — always a *time-contiguous* block at the end of the window,
never a random holdout (a random holdout of overlapping-label panel rows leaks
the label of a neighbouring date into early stopping).

Hyper-parameters live in ``AlphaConfig.lgbm`` (:class:`~qts.core.config.LGBMParams`)
and ``AlphaConfig.mlp`` (:class:`~qts.core.config.MLPParams`); constructor
keyword arguments override individual fields (e.g. ``LGBMModel(n_estimators=150)``).

Performance: the LightGBM early-stopping metric (per-date rank IC of the
validation block, evaluated every boosting round) is fully vectorised —
date groups are laid out once as a padded ``(n_dates, max_names)`` array and
ranked with ``scipy.stats.rankdata(axis=1)``; the label ranks are cached.
"""
from __future__ import annotations

import copy
import inspect
import os
from collections.abc import Sequence
from dataclasses import fields, replace
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from qts.core.config import AlphaConfig, LGBMParams, MLPParams
from qts.core.stats import IC_MIN_OBS

from .composite import ThemeComposite
from .features import feature_signature


# ----------------------------------------------------------------------------- utilities
def _dates(X: pd.DataFrame) -> pd.Index:
    return X.index.get_level_values("date")


def default_valid_mask(X: pd.DataFrame, frac: float = 0.2) -> pd.Series:
    """True on the last ``frac`` of the distinct dates (a time-contiguous block)."""
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
    return (num / den.replace(0, np.nan)).where(n >= IC_MIN_OBS)


class DateGroupedRankIC:
    """Vectorised mean per-date Spearman IC for row-ordered data.

    ``dates`` are the date labels of the rows (any order).  The layout of the
    rows into a padded ``(n_dates, max_names)`` array is computed once; each
    call ranks the padded array row-wise (``scipy.stats.rankdata``, average
    ties — identical to ``groupby(date).rank()``), centres the ranks per date
    and takes the per-date Pearson correlation of ranks.  Dates with fewer
    than ``min_n`` rows or a constant side (IC undefined) are skipped; the mean
    over the rest is returned (0.0 if none).

    Padding uses ``+inf`` so that pads rank after every real value and never
    share a tie with one; pad cells are masked out of every sum.
    """

    def __init__(self, dates, min_n: int = IC_MIN_OBS):
        codes, _ = pd.factorize(np.asarray(dates))
        self.n_rows = len(codes)
        counts = np.bincount(codes) if len(codes) else np.zeros(0, int)
        order = np.argsort(codes, kind="stable")
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]]) if len(counts) else np.zeros(0, int)
        sc = codes[order]
        self._order = order
        self._row = sc
        self._col = np.arange(len(codes)) - starts[sc] if len(codes) else np.zeros(0, int)
        self._shape = (len(counts), int(counts.max()) if len(counts) else 0)
        self._mask = np.zeros(self._shape, bool)
        self._mask[self._row, self._col] = True
        self._n = counts.astype(float)
        self._ok = counts >= min_n
        self._cache_key: tuple[int, int] | None = None
        self._cache_val: np.ndarray | None = None

    def centred_ranks(self, v: np.ndarray) -> np.ndarray:
        """Padded (n_dates, max_names) array of per-date ranks minus their mean (pads 0)."""
        P = np.full(self._shape, np.inf)
        P[self._row, self._col] = np.asarray(v, dtype=float)[self._order]
        R = rankdata(P, method="average", axis=1)
        R = np.where(self._mask, R, 0.0)
        mu = R.sum(axis=1) / np.maximum(self._n, 1.0)
        return np.where(self._mask, R - mu[:, None], 0.0)

    def per_date(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Per-date IC array (NaN where undefined or n < min_n)."""
        key = (id(y_true), len(y_true))
        if self._cache_key != key or self._cache_val is None:
            self._cache_val = self.centred_ranks(y_true)
            self._cache_key = key
        rt = self._cache_val
        rp = self.centred_ranks(y_pred)
        num = (rt * rp).sum(axis=1)
        den = np.sqrt((rt * rt).sum(axis=1) * (rp * rp).sum(axis=1))
        with np.errstate(invalid="ignore", divide="ignore"):
            ic = num / den
        ic[~self._ok] = np.nan
        return ic

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ic = self.per_date(y_true, y_pred)
        ic = ic[np.isfinite(ic)]
        return float(ic.mean()) if len(ic) else 0.0


def make_rank_ic_metric(dates):
    """LightGBM sklearn-API eval metric: mean per-date Spearman IC (higher is better).
    ``dates`` are the date labels of the validation rows, in row order.
    Vectorised (:class:`DateGroupedRankIC`); the label ranks are cached."""
    scorer = DateGroupedRankIC(dates)

    def metric(y_true, y_pred):
        return "rank_ic", scorer(y_true, y_pred), True

    return metric


def _split_overrides(cls, overrides: dict[str, Any], aliases: dict[str, str] | None = None):
    """Split constructor kwargs into (dataclass field overrides, passthrough extras)."""
    names = {f.name for f in fields(cls)}
    aliases = aliases or {}
    known, extra = {}, {}
    for k, v in overrides.items():
        k2 = aliases.get(k, k)
        (known if k2 in names else extra)[k2 if k2 in names else k] = v
    return known, extra


class _Base:
    feature_names_: list[str] | None = None

    def _check_features(self, X: pd.DataFrame) -> None:
        if self.feature_names_ is not None and list(X.columns) != self.feature_names_:
            raise ValueError("feature columns/order differ from those used in fit "
                             f"({feature_signature(X.columns)} vs {feature_signature(self.feature_names_)})")

    @property
    def feature_hash_(self) -> str | None:
        return None if self.feature_names_ is None else feature_signature(self.feature_names_)

    def feature_importance(self) -> pd.Series | None:
        return None


# ----------------------------------------------------------------------------- A: rank composite
class RankCompositeModel(_Base):
    """Baseline A: fixed-weight theme composite.  No fitting."""

    def __init__(self, cfg: AlphaConfig | None = None):
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
        if self.model_ is None:
            raise RuntimeError("RidgeModel.predict called before fit()")
        return pd.Series(self.model_.predict(X.values), index=X.index, name="pred")

    def feature_importance(self):
        if self.model_ is None:
            raise RuntimeError("RidgeModel.feature_importance called before fit()")
        return pd.Series(self.model_.coef_, index=self.feature_names_).abs().sort_values(ascending=False)


# ----------------------------------------------------------------------------- LightGBM
class LGBMModel(_Base):
    """LightGBM regressor, seed-bagged, early-stopped on the validation block.

    After early stopping the model is refit on train + validation with the
    best iteration count (``refit_on_full``), so the most recent year of labels
    is not wasted.  Hyper-parameters: ``params`` (default ``LGBMParams()``,
    i.e. ``AlphaConfig().lgbm``); any ``LGBMParams`` field may be overridden
    by keyword, other keywords are passed to ``LGBMRegressor`` unchanged.
    """

    def __init__(self, n_seeds: int = 3, params: LGBMParams | None = None, base_seed: int = 0,
                 **overrides):
        known, extra = _split_overrides(LGBMParams, overrides)
        hp = replace(params or LGBMParams(), **known)
        self.hp = hp
        n_jobs = hp.n_jobs if hp.n_jobs else (os.cpu_count() or 1)
        self.params = dict(objective="regression", n_estimators=hp.n_estimators, learning_rate=hp.learning_rate,
                           num_leaves=hp.num_leaves, min_child_samples=hp.min_data_in_leaf,
                           colsample_bytree=hp.feature_fraction, subsample=hp.bagging_fraction,
                           subsample_freq=hp.bagging_freq, reg_lambda=hp.lambda_l2, n_jobs=n_jobs,
                           metric="None", verbose=-1, **extra)
        self.n_seeds = n_seeds
        self.early_stopping_rounds = hp.early_stopping_rounds
        self.refit_on_full = hp.refit_on_full
        self.base_seed = base_seed
        self.models_: list = []
        self.best_iterations_: list[int] = []

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
    """sklearn ``MLPRegressor`` (default 64/32), standardised inputs, seed-bagged,
    with **time-contiguous** early stopping (audit #13 / handbook fix #8).

    sklearn's built-in ``early_stopping`` holds out a *random* 10 % of rows; on
    a panel with overlapping 20-day labels that leaks neighbouring dates'
    labels into the stopping rule.  Instead the net is trained epoch by epoch
    with ``partial_fit`` on the training rows and scored after every epoch on
    the validation block — the walk-forward trainer's purged trailing block
    (``valid_mask``) or, when none is given, the last ``valid_fraction`` of
    dates — by mean per-date rank IC.  Training stops after
    ``n_iter_no_change`` epochs without improvement; the best epoch count
    ``k*`` is kept.  With ``refit_on_full`` a fresh net (same seed) is trained
    on train + validation for ``k*`` epochs (mirrors ``LGBMModel``), otherwise
    the best-epoch weights are restored.

    Hyper-parameters: ``params`` (default ``MLPParams()``); keyword arguments
    override fields (``hidden``, ``alpha``, ``max_iter``, ``batch_size``,
    ``learning_rate_init``, ``max_train_rows``, ...).
    """

    def __init__(self, n_seeds: int = 3, params: MLPParams | None = None, base_seed: int = 0,
                 **overrides):
        known, extra = _split_overrides(MLPParams, overrides)
        if extra:
            raise TypeError(f"MLPModel: unknown argument(s) {sorted(extra)}")
        if "hidden" in known:
            known["hidden"] = tuple(known["hidden"])
        self.hp = replace(params or MLPParams(), **known)
        self.n_seeds, self.base_seed = n_seeds, base_seed
        self.models_: list = []
        self.best_epochs_: list[int] = []
        self.valid_ic_: list[float] = []

    def _new_net(self, seed: int, n_rows: int):
        from sklearn.neural_network import MLPRegressor
        hp = self.hp
        return MLPRegressor(hidden_layer_sizes=tuple(hp.hidden), alpha=hp.alpha,
                            batch_size=max(1, min(hp.batch_size, n_rows)), learning_rate_init=hp.learning_rate_init,
                            shuffle=True, random_state=seed)

    def fit(self, X, y, valid_mask=None):
        import warnings

        from sklearn.preprocessing import StandardScaler
        hp = self.hp
        self.feature_names_ = list(X.columns)
        keep = y.notna().to_numpy()
        vm = (default_valid_mask(X, frac=hp.valid_fraction) if valid_mask is None
              else valid_mask.reindex(X.index).fillna(False).astype(bool)).to_numpy()
        tr_idx = np.flatnonzero(keep & ~vm)
        va_idx = np.flatnonzero(keep & vm)
        all_idx = np.flatnonzero(keep)
        rs = np.random.RandomState(self.base_seed)
        cap = hp.max_train_rows
        if cap is not None and len(tr_idx) > cap:
            tr_idx = np.sort(rs.choice(tr_idx, cap, replace=False))
        if cap is not None and len(all_idx) > cap:
            all_idx = np.sort(rs.choice(all_idx, cap, replace=False))
        Xv_all = X.to_numpy(dtype=float)
        yv_all = y.to_numpy(dtype=float)
        self.scaler_ = StandardScaler().fit(Xv_all[tr_idx if len(tr_idx) else all_idx])
        Xt, yt = self.scaler_.transform(Xv_all[tr_idx]), yv_all[tr_idx]
        have_valid = len(va_idx) > 0 and len(tr_idx) > 0
        if have_valid:
            Xv, yv = self.scaler_.transform(Xv_all[va_idx]), yv_all[va_idx]
            scorer = DateGroupedRankIC(_dates(X)[va_idx])
        self.models_, self.best_epochs_, self.valid_ic_ = [], [], []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")          # ConvergenceWarning per partial_fit epoch
            for s in range(self.n_seeds):
                seed = self.base_seed + s
                if not have_valid:                   # no validation block: fixed max_iter epochs
                    m = self._new_net(seed, len(all_idx))
                    Xa = self.scaler_.transform(Xv_all[all_idx])
                    for _ in range(hp.max_iter):
                        m.partial_fit(Xa, yv_all[all_idx])
                    self.models_.append(m)
                    self.best_epochs_.append(hp.max_iter)
                    self.valid_ic_.append(float("nan"))
                    continue
                m = self._new_net(seed, len(tr_idx))
                best_ic, best_ep, best_state, wait = -np.inf, 0, None, 0
                for ep in range(1, hp.max_iter + 1):
                    m.partial_fit(Xt, yt)
                    ic = scorer(yv, m.predict(Xv))
                    if ic > best_ic + 1e-6:
                        best_ic, best_ep, wait = ic, ep, 0
                        best_state = (copy.deepcopy(m.coefs_), copy.deepcopy(m.intercepts_))
                    else:
                        wait += 1
                        if wait >= hp.n_iter_no_change:
                            break
                best_ep = max(best_ep, 1)
                if hp.refit_on_full:
                    m = self._new_net(seed, len(all_idx))
                    Xa = self.scaler_.transform(Xv_all[all_idx])
                    for _ in range(best_ep):
                        m.partial_fit(Xa, yv_all[all_idx])
                elif best_state is not None:
                    m.coefs_, m.intercepts_ = best_state
                self.models_.append(m)
                self.best_epochs_.append(best_ep)
                self.valid_ic_.append(float(best_ic))
        return self

    def predict(self, X):
        self._check_features(X)
        Xs = self.scaler_.transform(X.to_numpy(dtype=float))
        p = np.mean([m.predict(Xs) for m in self.models_], axis=0)
        return pd.Series(p, index=X.index, name="pred")


# ----------------------------------------------------------------------------- ensemble
class EnsembleModel(_Base):
    """Rank-average of member predictions per date, then per-date z-score.
    Weights are fixed ex ante (not fitted)."""

    def __init__(self, models: Sequence, weights: Sequence[float] | None = None):
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
        for w, m in zip(self.weights, self.models, strict=True):
            r = rank_by_date(m.predict(X)) - 0.5
            acc = r * w if acc is None else acc + r * w
        return zscore_by_date(acc).rename("pred")

    def feature_importance(self):
        parts = [m.feature_importance() for m in self.models]
        parts = [p for p in parts if p is not None]
        if not parts:
            return None
        return pd.concat(parts, axis=1).mean(axis=1).sort_values(ascending=False)

    def member_predictions(self, X) -> dict[str, pd.Series]:
        return {type(m).__name__: m.predict(X) for m in self.models}
