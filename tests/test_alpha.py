"""Tests for the alpha layer (qts.alpha) on the synthetic market with planted truth."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from qts.alpha import (AlphaModel, EnsembleModel, FeatureBuilder, LGBMModel, RankCompositeModel,
                       RidgeModel, ThemeComposite, WalkForwardTrainer, forward_returns, ic_decay_curve,
                       ic_report, incremental_ic, rank_ic_series, residualize_forward_returns,
                       signal_half_life, theme_portfolio_returns, train_live_parity_check)
from qts.alpha.composite import factor_momentum_tilt, momentum_crash_scalar
from qts.alpha.evaluation import decile_spread
from qts.alpha.trainer import embargoed_train_mask
from qts.core.config import AlphaConfig
from qts.core.types import AlphaOutput
from qts.data.synthetic import SyntheticSpec, generate

warnings.filterwarnings("ignore", category=RuntimeWarning)

HORIZON = 20
WARMUP = 330  # dates after which every rolling feature is populated (mom_change needs 252+63)


@pytest.fixture(scope="module")
def cfg():
    return AlphaConfig(train_min_days=504, refit_frequency_days=252, model_seeds=1)


@pytest.fixture(scope="module")
def md():
    return generate(SyntheticSpec(n_stocks=200, n_days=1260, seed=2))


@pytest.fixture(scope="module")
def raw(md, cfg):
    return FeatureBuilder(cfg).build(md)


@pytest.fixture(scope="module")
def X(md, cfg, raw):
    return FeatureBuilder(cfg).process(raw, md)


@pytest.fixture(scope="module")
def target(md, raw):
    return residualize_forward_returns(md, HORIZON, features=raw)


@pytest.fixture(scope="module")
def y_long(X, target):
    y = target.stack(future_stack=True)
    y.index.names = ["date", "ticker"]
    return y.reindex(X.index)


# --------------------------------------------------------------------------- features
def test_features_point_in_time(md, cfg, raw, X):
    """Features at dates <= k must be identical whether or not data after k exists."""
    fb = FeatureBuilder(cfg)
    k = md.dates[800]
    md_k = md.slice_until(k)
    raw_k = fb.build(md_k)
    assert set(raw_k) == set(raw)
    for name, panel in raw.items():
        a = panel.loc[:k]
        b = raw_k[name]
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-10, atol=1e-12)
    # processing is per-date, so the processed frame is PIT as well
    X_k = fb.process(raw_k, md_k)
    full = X.loc[X.index.get_level_values("date") <= k]
    pd.testing.assert_frame_equal(full, X_k, check_exact=False, rtol=1e-10, atol=1e-12)


def test_processed_frame_standardised(X, md):
    assert not X.isna().any().any()
    sub = X.loc[X.index.get_level_values("date") >= md.dates[WARMUP]]
    g = sub.groupby(level="date")
    mean, sd = g.mean(), g.std(ddof=0)
    # borrow_fee is floored in the generator -> massive ties -> sd well below 1; exclude it
    cols = [c for c in X.columns if c != "borrow_fee"]
    assert (mean[cols].abs() < 0.05).all().all(), mean[cols].abs().max()
    assert ((sd[cols] - 1.0).abs() < 0.12).all().all(), sd[cols].mean()


# --------------------------------------------------------------------------- targets
def test_residualized_target_properties(md, target, raw):
    valid = target.dropna(how="all")
    assert len(valid) == len(md.dates) - HORIZON
    assert (valid.mean(axis=1).abs() < 1e-8).all()
    assert ((valid.std(axis=1, ddof=0) - 1).abs() < 1e-8).all()
    # neutral to market beta (true and estimated) and size in the cross-section
    beta = md.extra["true_beta"]
    c_beta = rank_ic_series(target, beta).dropna()
    assert abs(c_beta.mean()) < 0.02, c_beta.mean()
    lm = raw["log_mcap"]
    assert abs(rank_ic_series(target, lm).mean()) < 0.02
    # pooled correlation with the forward market return is ~0 (labels are demeaned per date)
    fwd_mkt = forward_returns(md, HORIZON).mean(axis=1)
    pooled = pd.DataFrame({"t": target.stack(future_stack=True),
                           "m": np.repeat(fwd_mkt.values, target.shape[1])}).dropna()
    assert abs(pooled["t"].corr(pooled["m"])) < 0.01
    # exposures hook: an external exposures function is honoured
    ext = residualize_forward_returns(
        md.slice_until(md.dates[80]), 10,
        exposures_fn=lambda d: pd.DataFrame({"b": beta.loc[d]}))
    assert ext.dropna(how="all").shape[0] == 81 - 10
    # ... and the result is beta-neutral per date
    assert abs(rank_ic_series(ext, beta).mean()) < 0.03


def test_planted_truth_ic(md, target):
    ic = rank_ic_series(md.extra["true_latent_score"], target).dropna()
    m = ic.mean()
    print(f"\n[planted] latent score vs residualised {HORIZON}d target: mean rank IC = {m:.4f}")
    assert 0.025 <= m <= 0.055, m


# --------------------------------------------------------------------------- composite
def test_composite_and_dynamic_weights(md, cfg, X, target):
    comp = ThemeComposite(cfg)
    static = comp.score_all(X)
    ic_static = rank_ic_series(static, target).mean()
    tr = theme_portfolio_returns(X, md)
    assert tr.shape[1] == 6 and tr.dropna().shape[0] > 1000
    dyn = comp.score_all(X, md=md, dynamic=True, theme_returns=tr)
    ic_dyn = rank_ic_series(dyn, target).mean()
    print(f"\n[composite] static IC = {ic_static:.4f}, dynamic IC = {ic_dyn:.4f}")
    assert ic_static > 0.01
    row = dyn.loc[md.dates[-5]].dropna()
    assert abs(row.mean()) < 1e-8 and abs(row.std(ddof=0) - 1) < 1e-8
    tilt = factor_momentum_tilt(tr, cfg.factor_momentum_lookback_months, cfg.factor_momentum_max_tilt)
    assert ((tilt >= 0.75 - 1e-12) & (tilt <= 1.25 + 1e-12)).all()
    s = momentum_crash_scalar(md, md.dates[-1], cfg.momentum_crash_market_lookback_days)
    assert 0.25 <= s <= 1.0
    one = comp.score(X, md.dates[-5], md=md, dynamic=True, theme_returns=tr)
    pd.testing.assert_series_equal(one, row, check_names=False)


# --------------------------------------------------------------------------- walk-forward
def test_walk_forward_oos_ic(md, cfg, X, y_long, target):
    wf = WalkForwardTrainer(cfg)
    factory = lambda: EnsembleModel([RankCompositeModel(cfg),
                                     LGBMModel(n_seeds=1, n_estimators=150, n_jobs=4)])
    pred = wf.run(X, y_long, factory)
    first = wf.folds_[0].pred_start
    assert pred.loc[:first].iloc[:-1].isna().all().all()
    assert pred.loc[first:].notna().sum(axis=1).min() == md.tickers.size
    rep = ic_report(pred, target)
    print(f"\n[walk-forward] composite+LGBM OOS mean rank IC = {rep['mean_ic']:.4f} "
          f"(ICIR {rep['icir']:.2f}, t {rep['t_stat']:.2f}, n={rep['n']})")
    print(rep["by_year"].round(4).to_string())
    lgbm = wf.models_[-1].models[1]
    print("[walk-forward] LGBM best iterations:", [m.best_iterations_ for m in (mm.models[1] for mm in wf.models_)])
    print("[walk-forward] LGBM top features:\n", lgbm.feature_importance().head(6).round(3).to_string())
    assert rep["mean_ic"] > 0.01
    # ridge baseline runs and is OOS-positive
    pr = wf.run(X, y_long, RidgeModel)
    ic_r = rank_ic_series(pr, target).mean()
    print(f"[walk-forward] ridge OOS mean rank IC = {ic_r:.4f}, alphas={[m.best_alpha_ for m in wf.models_]}")
    assert np.isfinite(ic_r)
    # train/live parity
    last = X.loc[[md.dates[-1]]]
    assert train_live_parity_check(wf.models_[-1], last)
    with pytest.raises(AssertionError):
        train_live_parity_check(wf.models_[-1], last[list(reversed(last.columns))])


class _SpyModel:
    """Records the training / prediction dates it is given."""
    log = []

    def fit(self, X, y, valid_mask=None):
        self.feature_names_ = list(X.columns)
        self.train_dates = X.index.get_level_values("date")
        self.valid_dates = X.index.get_level_values("date")[valid_mask.values]
        return self

    def predict(self, X):
        _SpyModel.log.append((self.train_dates.max(), self.valid_dates.min() if len(self.valid_dates) else None,
                              X.index.get_level_values("date").min()))
        return pd.Series(0.0, index=X.index)


def test_embargo_excludes_overlapping_labels():
    h = 5
    dates = pd.bdate_range("2020-01-01", periods=120)
    tickers = [f"T{i}" for i in range(30)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((len(idx), 3)), index=idx, columns=list("abc"))
    y = pd.Series(rng.standard_normal(len(idx)), index=idx)
    cfg = AlphaConfig(horizon_days=h, train_min_days=40, refit_frequency_days=20, embargo_days=None)
    tr = WalkForwardTrainer(cfg, valid_days=10)
    _SpyModel.log.clear()
    tr.run(X, y, _SpyModel)
    assert len(_SpyModel.log) >= 3
    pos = {d: i for i, d in enumerate(dates)}
    for train_max, valid_min, pred_min in _SpyModel.log:
        # label at s spans (s, s+h]; it must be fully realised before the first prediction date
        assert pos[train_max] + h < pos[pred_min], (train_max, pred_min)
        # the tightest allowed: exactly h+1 positions before the prediction start
        assert pos[pred_min] - pos[train_max] == h + 1
    # first fit happens once train_min_days usable labels exist
    assert pos[_SpyModel.log[0][2]] == cfg.train_min_days + h
    # direct check of the mask helper
    m = embargoed_train_mask(np.arange(20), pred_start_pos=10, horizon=3)
    assert m.tolist() == [True] * 7 + [False] * 13
    # custom embargo overrides horizon
    tr2 = WalkForwardTrainer(AlphaConfig(horizon_days=h, embargo_days=9, train_min_days=40,
                                         refit_frequency_days=20), valid_days=10)
    _SpyModel.log.clear()
    tr2.run(X, y, _SpyModel)
    assert all(pos[p] - pos[t] == 10 for t, _, p in _SpyModel.log)


# --------------------------------------------------------------------------- evaluation
def test_half_life_and_decay(md, raw, cfg, X, target):
    z = md.extra["true_latent_score"]
    hl = signal_half_life(z)
    assert np.isfinite(hl) and hl > 0
    comp = ThemeComposite(cfg).score_all(X)
    hl_c = signal_half_life(comp)
    print(f"\n[half-life] latent {hl:.1f} d, composite {hl_c:.1f} d")
    assert np.isfinite(hl_c) and hl_c > 0
    curve = ic_decay_curve(z, md, horizons=(1, 5, 20, 60), features=raw)
    print("[decay]", curve.round(4).to_dict())
    assert curve.loc[1] < curve.loc[5] < curve.loc[20]  # latent score is persistent: IC accumulates
    assert curve.loc[60] > 0
    # incremental IC of the latent truth over the composite stays positive
    inc = incremental_ic(z, comp, target)
    assert inc["mean_ic"] > 0.01
    ds = decile_spread(comp, forward_returns(md, HORIZON), horizon=HORIZON)
    assert np.isfinite(ds.sharpe) and np.isfinite(ds.t_stat) and len(ds.returns) > 1000


# --------------------------------------------------------------------------- orchestrator
def test_alpha_model_as_of(md, cfg):
    am = AlphaModel(cfg, ml_factory=lambda: LGBMModel(n_seeds=1, n_estimators=100, n_jobs=4)).fit(md)
    d = md.dates[-10]
    out = am.as_of(d)
    assert isinstance(out, AlphaOutput)
    assert out.date == d and out.horizon_days == cfg.horizon_days
    assert len(out.score) == md.tickers.size
    assert abs(out.score.mean()) < 1e-8 and abs(out.score.std(ddof=0) - 1) < 1e-8
    assert np.isfinite(out.expected_return).all()
    assert out.expected_return.abs().max() > 0
    assert np.isfinite(out.half_life_days) and out.half_life_days > 0
    assert out.ic_estimate > 0
    assert {"composite", "ml", "theme_value", "theme_momentum"} <= set(out.components)
    # Grinold scaling: expected_return / (IC * score) == horizon residual vol
    vol = out.expected_return / (out.ic_estimate * out.score.replace(0, np.nan))
    assert (vol.dropna() > 0).all()
    # ML sleeve is only available out-of-sample
    early = am.as_of(md.dates[100])
    assert "ml" not in early.components or early.components["ml"].isna().all()
    print(f"\n[alpha_model] trailing OOS IC used at {d.date()}: {am.trailing_ic(d):.4f}; "
          f"shrunk IC {out.ic_estimate:.4f}; half-life {out.half_life_days:.1f} d")


# --------------------------------------------------------------------------- universe eligibility (audit finding 1)
def test_cross_section_uses_only_eligible_names():
    """With a time-varying universe, feature rows, labels and emitted scores must
    contain only names eligible *at that date*, and ``as_of(t)`` must not depend
    on future universe membership."""
    from qts.core.config import UniverseConfig
    from qts.core.types import to_long
    from qts.data.universe import apply_mask, build_universe_mask

    md_full = generate(SyntheticSpec(n_stocks=300, n_days=700, seed=11))
    ucfg = UniverseConfig(top_n_by_mcap=120, min_adv_usd=0.0, exclude_bottom_size_quantile=0.0)
    mask = build_universe_mask(md_full, ucfg)
    md_u = apply_mask(md_full, mask)
    assert md_u.eligible is not None
    n_in = md_u.eligible.sum(axis=1)
    assert n_in.iloc[50:].max() < md_u.tickers.size            # names ever in > names in at t (hysteresis allows >120)
    acfg = AlphaConfig(train_min_days=300, refit_frequency_days=126, model_seeds=1)
    am = AlphaModel(acfg, use_ml=False).fit(md_u)
    elig_long = to_long(md_u.eligibility(), "e")
    # every feature row is eligible at its own date
    rows = elig_long.reindex(am.X_.index)
    assert rows.notna().all() and rows.all()
    # labels only for eligible names
    assert not (am.target_.notna() & ~md_u.eligibility()).any().any()
    # emitted scores only for eligible names at t
    for t in (md_u.dates[400], md_u.dates[-30]):
        out = am.as_of(t)
        ok = md_u.eligibility().loc[t]
        assert set(out.score.index) <= set(ok[ok].index)
        assert len(out.score) == int(ok.sum())
    # as_of(t) is invariant to altering the mask after t (no future-membership leakage)
    t = md_u.dates[450]
    ref = am.as_of(t)
    mask2 = mask.copy()
    fut = mask2.index > t
    mask2.loc[fut] = ~mask2.loc[fut].to_numpy()            # flip all future membership
    mask2.loc[fut] &= md_full.close.notna().loc[fut]
    am2 = AlphaModel(acfg, use_ml=False).fit(apply_mask(md_full, mask2))
    alt = am2.as_of(t)
    pd.testing.assert_series_equal(ref.score, alt.score.reindex(ref.score.index), rtol=1e-9, atol=1e-10)
    pd.testing.assert_series_equal(ref.expected_return, alt.expected_return.reindex(ref.score.index),
                                   rtol=1e-9, atol=1e-10)
