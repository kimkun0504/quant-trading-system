"""Recency weighting of the ML training rows
(``AlphaConfig.sample_weight_halflife_years``; handbook 04 "Recency weighting")."""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pandas as pd
import pytest

from qts.alpha.alpha_model import refit_record
from qts.alpha.models import (
    DateGroupedRankIC,
    EnsembleModel,
    LGBMModel,
    MLPModel,
    RidgeModel,
    mean_rank_ic,
    unit_mean,
    weighted_mean_rank_ic,
)
from qts.alpha.trainer import (
    WalkForwardTrainer,
    accepts_sample_weight,
    effective_sample_size,
    recency_weights,
)
from qts.core.config import AlphaConfig, SystemConfig
from qts.data.synthetic import SyntheticSpec, generate
from qts.pipeline import ResearchPipeline


def _panel(n_dates=420, n_names=25, n_feat=3, seed=0, flip_at=None, beta=0.5):
    """Long (date, ticker) panel; y = sign * beta * f0 + noise, sign flipping at ``flip_at``."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_dates)
    idx = pd.MultiIndex.from_product([dates, [f"S{i:02d}" for i in range(n_names)]], names=["date", "ticker"])
    X = pd.DataFrame(
        rng.standard_normal((len(idx), n_feat)), index=idx, columns=[f"f{i}" for i in range(n_feat)]
    )
    pos = np.repeat(np.arange(n_dates), n_names)
    sign = np.where(pos < flip_at, 1.0, -1.0) if flip_at is not None else 1.0
    y = pd.Series(sign * beta * X["f0"].to_numpy() + rng.standard_normal(len(idx)), index=idx)
    return X, y, dates


class _Spy:
    """Records how the trainer called ``fit``."""

    def __init__(self, log):
        self.log = log

    def fit(self, X, y, valid_mask=None, **kw):
        self.log.append({"kw": kw, "X": X, "valid": valid_mask})
        return self

    def predict(self, X):
        return pd.Series(0.0, index=X.index)


def _cfg(**kw) -> AlphaConfig:
    base = {"horizon_days": 5, "train_min_days": 200, "refit_frequency_days": 100, "valid_days": 60}
    return dataclasses.replace(AlphaConfig(), **{**base, **kw})


# ----------------------------------------------------------------------------- formula
def test_recency_weights_formula_and_effective_sample_size():
    age = np.array([0, 252, 504, 2016])
    np.testing.assert_allclose(recency_weights(age, 1.0), [1.0, 0.5, 0.25, 2.0**-8])
    np.testing.assert_allclose(recency_weights(age, 8.0), 0.5 ** (age / (8 * 252)))
    assert recency_weights(np.array([2016]), 8.0)[0] == pytest.approx(0.5)  # 8 years back = half weight
    with pytest.raises(ValueError):
        recency_weights(age, 0.0)
    with pytest.raises(ValueError):
        recency_weights(np.array([-1]), 1.0)  # a row after T would be look-ahead
    assert effective_sample_size(np.ones(10)) == pytest.approx(10.0)
    w = np.array([1.0, 1.0, 0.0, 0.0])
    assert effective_sample_size(w) == pytest.approx(2.0)
    w = recency_weights(np.arange(5000), 1.0)
    assert effective_sample_size(w) == pytest.approx(w.sum() ** 2 / (w**2).sum())
    np.testing.assert_allclose(unit_mean(w).mean(), 1.0)


def test_config_field_default_validation_and_hash():
    assert AlphaConfig().sample_weight_halflife_years == 0.0
    for bad in (-1.0, float("nan"), True, "8"):
        with pytest.raises((ValueError, TypeError)):
            dataclasses.replace(AlphaConfig(), sample_weight_halflife_years=bad)
    a = SystemConfig()
    b = a.with_overrides({"alpha.sample_weight_halflife_years": 8})
    assert b.alpha.sample_weight_halflife_years == 8
    assert a.hash() != b.hash() and a.alpha.hash() != b.alpha.hash()
    assert a.risk.hash() == b.risk.hash() and a.universe.hash() == b.universe.hash()


# ----------------------------------------------------------------------------- trainer
def test_trainer_passes_no_weights_at_zero_and_exact_weights_otherwise():
    X, y, _ = _panel()
    log0: list = []
    t0 = WalkForwardTrainer(_cfg())
    t0.run(X, y, lambda: _Spy(log0))
    assert log0 and all(c["kw"] == {} for c in log0)  # the unweighted call, no keyword at all
    for f in t0.folds_:
        assert f.extra["sample_weight_halflife_years"] == 0.0
        assert f.extra["effective_sample_size"] == f.extra["n_rows"]

    hl = 0.5
    log1: list = []
    t1 = WalkForwardTrainer(_cfg(sample_weight_halflife_years=hl))
    t1.run(X, y, lambda: _Spy(log1))
    assert len(log1) == len(log0) == len(t1.folds_)
    uniq = pd.DatetimeIndex(X.index.get_level_values("date").unique())
    for call, c0, fold in zip(log1, log0, t1.folds_, strict=True):
        w = call["kw"]["sample_weight"]
        assert call["X"].index.equals(c0["X"].index) and w.index.equals(call["X"].index)  # same rows
        d = w.index.get_level_values("date")
        T = d.max()
        assert T == fold.train_end < fold.pred_start  # PIT: weights only on dates <= T
        age = uniq.get_loc(T) - uniq.get_indexer(d)
        raw = 0.5 ** (age / (hl * 252))
        np.testing.assert_allclose(w.to_numpy(), raw / raw.mean(), rtol=1e-12)
        assert w.mean() == pytest.approx(1.0)
        assert w.groupby(level="date").nunique().max() == 1  # one weight per date
        assert w[d == T].iloc[0] == w.max()  # newest date carries the largest weight
        # validation rows are weighted by the same formula (they are part of the rows)
        assert call["valid"].any() and w[call["valid"].to_numpy()].min() > w[~call["valid"].to_numpy()].max()
        ess = fold.extra["effective_sample_size"]
        assert ess == pytest.approx(raw.sum() ** 2 / (raw**2).sum())
        assert ess < fold.extra["n_rows"] == len(w)
        rec = refit_record(fold, call and _Spy([]))
        assert rec.sample_weight_halflife_years == hl and rec.effective_sample_size == pytest.approx(ess)


def test_weights_do_not_depend_on_data_after_the_refit():
    """PIT: truncating the panel after a refit's prediction block leaves that refit's weights unchanged."""
    X, y, dates = _panel()
    cfg = _cfg(sample_weight_halflife_years=0.5)
    full: list = []
    WalkForwardTrainer(cfg).run(X, y, lambda: _Spy(full))
    cut = dates[330]
    keep = X.index.get_level_values("date") <= cut
    part: list = []
    WalkForwardTrainer(cfg).run(X[keep], y[keep], lambda: _Spy(part))
    assert 1 <= len(part) < len(full)
    for a, b in zip(part, full, strict=False):
        pd.testing.assert_series_equal(a["kw"]["sample_weight"], b["kw"]["sample_weight"])


def test_model_without_sample_weight_parameter_is_an_error_not_silently_unweighted():
    class NoWeights:
        def fit(self, X, y, valid_mask=None):
            return self

        def predict(self, X):
            return pd.Series(0.0, index=X.index)

    X, y, _ = _panel()
    assert not accepts_sample_weight(NoWeights()) and accepts_sample_weight(RidgeModel())
    WalkForwardTrainer(_cfg()).run(X, y, NoWeights)  # fine at 0
    with pytest.raises(TypeError, match="sample_weight"):
        WalkForwardTrainer(_cfg(sample_weight_halflife_years=1.0)).run(X, y, NoWeights)
    with pytest.raises(TypeError, match="sample_weight"):
        EnsembleModel([NoWeights()]).fit(X, y, None, sample_weight=pd.Series(1.0, index=X.index))


# ----------------------------------------------------------------------------- sleeves
def _sleeves():
    return {
        "ridge": lambda: RidgeModel(),
        "lgbm": lambda: LGBMModel(n_seeds=1, n_estimators=40, min_data_in_leaf=20, n_jobs=1),
        "mlp": lambda: MLPModel(
            n_seeds=1, max_iter=40, hidden=(16,), batch_size=128, learning_rate_init=3e-3
        ),
    }


@pytest.mark.parametrize("name", ["ridge", "lgbm", "mlp"])
def test_each_sleeve_uses_the_weights(name):
    """Constant weights reproduce the unweighted fit; zero weight on the first
    regime makes the fit follow the second one."""
    X, y, dates = _panel(n_dates=240, flip_at=120, beta=1.0)
    make = _sleeves()[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = make().fit(X, y).predict(X)
        same = make().fit(X, y, None, sample_weight=pd.Series(3.0, index=X.index)).predict(X)
        # constant weights are normalised to 1: the same model up to float round-off
        tol = {"ridge": 1e-8, "lgbm": 1e-8, "mlp": 1e-6}[name]
        np.testing.assert_allclose(same.to_numpy(), base.to_numpy(), atol=tol)
        d = X.index.get_level_values("date")
        w = pd.Series(np.where(d >= dates[120], 1.0, 1e-9), index=X.index)
        late = make().fit(X, y, None, sample_weight=w).predict(X)
    f0 = X["f0"]
    assert np.corrcoef(late, f0)[0, 1] < -0.5  # learned the second regime (y = -f0)
    assert np.corrcoef(late, f0)[0, 1] < np.corrcoef(base, f0)[0, 1] - 0.3


def test_mlp_row_cap_samples_uniformly_then_weights_the_loss():
    """``max_train_rows``: the drawn rows are the unweighted fit's rows (uniform
    draw) and the weights act on the loss, so the cap does not undo the weighting."""
    X, y, dates = _panel(n_dates=240, flip_at=120, beta=1.0)
    d = X.index.get_level_values("date")
    w = pd.Series(np.where(d >= dates[120], 1.0, 1e-9), index=X.index)
    kw = {
        "n_seeds": 1,
        "max_iter": 40,
        "hidden": (16,),
        "batch_size": 128,
        "learning_rate_init": 3e-3,
        "max_train_rows": 2000,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        capped = MLPModel(**kw).fit(X, y, None, sample_weight=w)
        plain = MLPModel(**kw).fit(X, y)
    assert capped.weighting_ == "loss" and plain.weighting_ == "none"
    assert np.corrcoef(capped.predict(X), X["f0"])[0, 1] < -0.5
    # the same uniformly drawn rows as the unweighted fit (a weighted draw would differ)
    assert capped.train_rows_signature_ == plain.train_rows_signature_
    assert capped.train_rows_signature_[0] == 2000


def test_mlp_falls_back_to_weighted_resampling_without_native_sample_weight(monkeypatch):
    """scikit-learn < 1.7 has no ``sample_weight`` in the MLP: rows are drawn in
    proportion to their weight instead, and the model says so."""
    X, y, dates = _panel(n_dates=240, flip_at=120, beta=1.0)
    d = X.index.get_level_values("date")
    w = pd.Series(np.where(d >= dates[120], 1.0, 1e-9), index=X.index)
    monkeypatch.setattr(MLPModel, "_native_weights", staticmethod(lambda: False))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = MLPModel(n_seeds=1, max_iter=40, hidden=(16,), batch_size=128, learning_rate_init=3e-3)
        m.fit(X, y, None, sample_weight=w)
    assert m.weighting_ == "resample"
    assert np.corrcoef(m.predict(X), X["f0"])[0, 1] < -0.5


def test_validation_score_is_date_weighted():
    X, y, dates = _panel(n_dates=60, flip_at=30, beta=1.0)
    pred = X["f0"]
    d = X.index.get_level_values("date")
    w = pd.Series(np.where(d >= dates[30], 1.0, 0.0), index=X.index)
    assert abs(mean_rank_ic(pred, y)) < 0.15  # the two regimes cancel
    assert weighted_mean_rank_ic(pred, y, w) < -0.4  # only the second regime counts
    plain = DateGroupedRankIC(d)(y.to_numpy(), pred.to_numpy())
    wtd = DateGroupedRankIC(d, weights=w.to_numpy())(y.to_numpy(), pred.to_numpy())
    assert plain == pytest.approx(mean_rank_ic(pred, y)) and wtd == pytest.approx(
        weighted_mean_rank_ic(pred, y, w)
    )
    ones = DateGroupedRankIC(d, weights=np.ones(len(d)))(y.to_numpy(), pred.to_numpy())
    assert ones == pytest.approx(plain)


# ----------------------------------------------------------------------------- regime switch
def test_regime_switch_weighted_fit_beats_equal_weights_out_of_sample():
    """Planted relationship flips sign halfway.  The refit sees more old-regime
    rows than new ones: equal weights still predict the old sign, a short
    half-life follows the switch, and its OOS rank IC after the switch is higher."""
    flip = 300
    X, y, dates = _panel(n_dates=560, n_names=30, flip_at=flip, beta=0.5, seed=3)
    base = {"train_min_days": 420, "refit_frequency_days": 1000, "valid_days": 60, "horizon_days": 5}

    def oos_ic(hl):
        tr = WalkForwardTrainer(_cfg(**base, sample_weight_halflife_years=hl))
        pred = tr.run(X, y, RidgeModel)
        (fold,) = tr.folds_
        assert fold.train_end > dates[flip] and fold.pred_start > fold.train_end
        n_old = (X.index.get_level_values("date") < dates[flip]).sum()
        assert n_old > fold.extra["n_rows"] / 2  # the old regime is the majority of the rows
        long = pred.stack().dropna().rename("p")
        assert long.index.get_level_values(0).min() == fold.pred_start  # OOS only
        return mean_rank_ic(long, y.reindex(long.index)), fold

    ic_eq, f_eq = oos_ic(0.0)
    ic_w, f_w = oos_ic(0.15)  # ~38 trading days
    assert ic_eq < -0.02  # equal weights: still on the old sign
    assert ic_w > 0.10  # weighted: on the new sign
    assert ic_w > ic_eq + 0.15
    assert f_w.extra["effective_sample_size"] < 0.3 * f_eq.extra["effective_sample_size"]


# ----------------------------------------------------------------------------- cache key
def test_stage_cache_key_changes_for_the_alpha_fit_only(tmp_path):
    md = generate(SyntheticSpec(n_stocks=30, n_days=300, seed=1))

    def keys(hl):
        cfg = SystemConfig().with_overrides(
            {"cache.enabled": True, "cache.root": str(tmp_path), "alpha.sample_weight_halflife_years": hl}
        )
        p = ResearchPipeline(cfg, use_ml=True)
        cache = p._stage_cache()
        risk = p._cache_key(cache, "risk_fit", md, "x" * 64)
        return risk, p._cache_key(cache, "alpha_fit", md, "x" * 64, parent=risk)

    r0, a0 = keys(0.0)
    r8, a8 = keys(8.0)
    assert r0 == r8  # the risk fit does not read the alpha section
    assert a0 != a8
    assert keys(8.0) == (r8, a8)
