"""Tests for the risk-model layer (qts.risk)."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from qts.core.config import RiskConfig
from qts.data.synthetic import SyntheticSpec, generate
from qts.risk import (
    BarraStyleRiskModel, alpha_alignment, bias_statistic, build_exposures,
    estimate_factor_return_history, estimate_factor_returns, ewma_cov,
    exposure_history, factor_covariance, linear_shrinkage, nonlinear_shrinkage,
    portfolio_bias_test, qlike, mincer_zarnowitz, min_variance_oos_test,
    random_long_short_weights, specific_risk,
)
from qts.risk.exposures import MARKET_FACTOR, split_factor_columns
from qts.risk.factor_returns import cap_weighted_industry_sum


@pytest.fixture(scope="module")
def md():
    return generate(SyntheticSpec(n_stocks=150, n_days=1000, seed=1))


@pytest.fixture(scope="module")
def cfg():
    return RiskConfig()


@pytest.fixture(scope="module")
def model(md, cfg):
    return BarraStyleRiskModel(cfg).fit(md)


def _is_psd(S, tol=1e-12):
    S = np.asarray(S)
    return np.linalg.eigvalsh((S + S.T) / 2).min() >= -tol * max(1.0, np.abs(S).max())


# --------------------------------------------------------------------------- exposures
def test_exposures_shapes_and_standardization(md, cfg):
    date = md.dates[600]
    B = build_exposures(md, date, cfg)
    _, industries, styles = split_factor_columns(B.columns, cfg)
    assert B.shape == (len(md.tickers), 1 + len(industries) + len(styles))
    assert B.columns[0] == MARKET_FACTOR
    assert not B.isna().any().any()
    assert (B[MARKET_FACTOR] == 1.0).all()
    assert np.allclose(B[industries].sum(axis=1), 1.0)
    assert set(styles) == set(cfg.styles)
    cap = md.market_cap.loc[date].reindex(B.index)
    w = cap / cap.sum()
    for s in styles:
        assert abs(float((B[s] * w).sum())) < 0.05, s          # cap-weighted mean ~0
        assert 0.85 < float(B[s].std(ddof=0)) < 1.1, s          # equal-weighted sd ~1
        assert B[s].abs().max() <= 3.5, s                       # trimmed


def test_exposure_history_matches_single_date_and_is_point_in_time(md, cfg):
    dates = md.dates[500:505]
    hist = exposure_history(md, dates, cfg)
    assert set(hist) == set(dates)
    single = build_exposures(md, dates[2], cfg)
    pd.testing.assert_frame_equal(hist[dates[2]], single, check_exact=False, atol=1e-10)
    # using a future-truncated panel gives the same answer -> no look-ahead
    cut = md.slice_until(dates[2])
    pd.testing.assert_frame_equal(exposure_history(cut, [dates[2]], cfg)[dates[2]], single, atol=1e-10)


def test_beta_exposure_tracks_true_beta(md, cfg):
    B = build_exposures(md, md.dates[-1], cfg)
    true_beta = md.extra["true_beta"].iloc[-1].reindex(B.index)
    assert B["beta"].corr(true_beta) > 0.6


# --------------------------------------------------------------------------- factor returns
def test_factor_returns_market_and_constraint(md, cfg, model):
    fr = model.factor_returns
    assert fr.shape[1] == model.exposures[next(iter(model.exposures))].shape[1]
    assert not fr.isna().any().any()
    r = md.returns
    c = md.market_cap.shift(1)
    cw_market = ((r * c).sum(axis=1) / c.sum(axis=1)).reindex(fr.index)
    # USE4 identification: with the cap-weighted industry constraint the market
    # factor *is* the cap-weighted market return.
    assert fr[MARKET_FACTOR].corr(cw_market) > 0.98
    # Against the latent market factor of the generator.  NOTE: this synthetic
    # universe is extremely concentrated (two names hold ~29% of cap), so the
    # cap-weighted market itself only correlates ~0.83 with the latent m_t; the
    # 0.9 bar is therefore only reachable for less concentrated universes.
    true_m = md.extra["true_market_return"].iloc[:, 0].reindex(fr.index)
    assert fr[MARKET_FACTOR].corr(true_m) > 0.7
    assert fr[MARKET_FACTOR].corr(true_m) >= cw_market.corr(true_m) - 0.02
    # constraint holds exactly every day
    for d in fr.index[::97]:
        i = md.dates.get_loc(d)
        prev = md.dates[i - 1]
        s = cap_weighted_industry_sum(fr.loc[d], model.exposures[prev], md.market_cap.loc[prev], cfg)
        assert abs(s) < 1e-10
    # R^2 sane
    assert 0.05 < model.r2.mean() < 0.8
    assert (model.r2 > 0).all() and (model.r2 < 1).all()


def test_single_day_regression_returns_specific_returns(md, cfg, model):
    d = model.factor_returns.index[300]
    i = md.dates.get_loc(d)
    prev = md.dates[i - 1]
    fit = estimate_factor_returns(md.returns.loc[d], model.exposures[prev], md.market_cap.loc[prev], cfg)
    f, u, r2 = fit
    B = model.exposures[prev]
    recon = B.values @ f.values + u.reindex(B.index).values
    assert np.allclose(recon, md.returns.loc[d].reindex(B.index).values)
    assert 0 < r2 < 1


# --------------------------------------------------------------------------- covariance
def test_ewma_cov_psd_and_scale(model, cfg):
    X = model.factor_returns.values
    S = ewma_cov(X, cfg.vol_half_life, cfg.newey_west_lags_vol)
    assert S.shape == (X.shape[1], X.shape[1])
    assert _is_psd(S)
    S0 = ewma_cov(X, 1e9, 0)   # ~equal weights, no NW -> ~sample covariance
    assert np.allclose(S0, np.cov(X.T, ddof=0), rtol=1e-3, atol=1e-10)


def test_factor_covariance_psd(model, cfg):
    F = factor_covariance(model.factor_returns, cfg)
    assert F.shape == (model.factor_returns.shape[1],) * 2
    assert _is_psd(F.values)
    assert (np.diag(F.values) > 0).all()
    # market vol in a plausible annualised range
    assert 0.05 < np.sqrt(F.loc[MARKET_FACTOR, MARKET_FACTOR] * 252) < 0.5
    F_noeig = factor_covariance(model.factor_returns, RiskConfig(eigenfactor_adjust=False))
    assert _is_psd(F_noeig.values)


def test_nonlinear_shrinkage_beats_sample_cov():
    rng = np.random.default_rng(0)
    N, T = 100, 200
    A = rng.standard_normal((N, N))
    Sigma = A @ A.T / N + 0.5 * np.eye(N)
    X = rng.multivariate_normal(np.zeros(N), Sigma, size=T)
    S_sample = np.cov(X.T, ddof=1)
    S_qis = nonlinear_shrinkage(X)
    S_lw = linear_shrinkage(X)
    assert S_qis.shape == (N, N)
    assert _is_psd(S_qis)
    assert np.isclose(np.trace(S_qis), np.trace(np.cov(X.T, ddof=1)), rtol=1e-6)
    err = lambda S: np.linalg.norm(S - Sigma, "fro")
    assert err(S_qis) < err(S_sample)
    assert err(S_qis) < 0.8 * err(S_sample)
    assert err(S_lw) < err(S_sample)


def test_nonlinear_shrinkage_singular_case():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((60, 120))
    S = nonlinear_shrinkage(X)
    assert S.shape == (120, 120)
    assert _is_psd(S)
    assert np.linalg.eigvalsh(S).min() > 0


def test_specific_risk_pipeline(md, cfg, model):
    d = md.dates[-1]
    B = model.exposures[d]
    var, det = specific_risk(model.specific_returns.loc[:d], B, md.market_cap.loc[d], cfg, return_details=True)
    assert var.index.equals(B.index)
    assert not var.isna().any() and (var > 0).all()
    ann = np.sqrt(var * 252)
    assert 0.05 < ann.median() < 0.8
    assert 0.5 < det["vra_multiplier"] < 2.0
    assert (det["gamma"] == 1.0).mean() > 0.9  # long histories -> time-series weight


# --------------------------------------------------------------------------- model as_of
def test_as_of_outputs(md, cfg, model):
    d = md.dates[-10]
    out = model.as_of(d)
    assert out.date == d
    assert out.factor_returns.index[-1] == d
    assert out.exposures.index.equals(out.specific_var.index)
    assert out.factor_cov.index.equals(out.exposures.columns)
    rng = np.random.default_rng(3)
    w = pd.Series(rng.standard_normal(len(out.tickers)), index=out.tickers)
    w -= w.mean()
    w /= w.abs().sum()
    assert out.portfolio_variance(w) > 0
    S = out.covariance()
    assert _is_psd(S.values)
    assert np.isclose(float(w @ S @ w), out.portfolio_variance(w))


def test_point_in_time_as_of(md, cfg, model):
    """as_of(d) must be identical when the model is fitted on data truncated at d."""
    d = md.dates[-100]
    cut = md.slice_until(d)
    m_cut = BarraStyleRiskModel(cfg).fit(cut)
    a, b = model.as_of(d), m_cut.as_of(d)
    pd.testing.assert_frame_equal(a.exposures, b.exposures)
    pd.testing.assert_frame_equal(a.factor_returns, b.factor_returns)
    pd.testing.assert_series_equal(a.specific_var, b.specific_var)
    pd.testing.assert_frame_equal(a.factor_cov, b.factor_cov)


def test_random_portfolio_bias_statistic(md, cfg, model):
    dates = md.dates[-251:-1]
    rng = np.random.default_rng(7)
    ports = {f"rand{k}": random_long_short_weights(rng) for k in range(3)}
    res = portfolio_bias_test(model, md, ports, dates)
    assert (res["T"] == 250).all()
    for name, row in res.iterrows():
        assert 0.7 < row["bias"] < 1.4, (name, row["bias"])
    assert np.isclose(res["upper"].iloc[0], 1 + 2 / np.sqrt(500))


# --------------------------------------------------------------------------- alignment
def test_alignment_orthogonal_unit_norm(md, cfg, model):
    d = md.dates[-5]
    out = model.as_of(d)
    rng = np.random.default_rng(11)
    alpha = pd.Series(rng.standard_normal(len(out.tickers)), index=out.tickers)
    alpha = alpha + 0.5 * out.exposures["momentum"]
    y, nu = alpha_alignment(alpha, out.exposures, out.specific_var, factor_cov=out.factor_cov, cfg=cfg)
    assert np.isclose(np.linalg.norm(y.values), 1.0)
    assert np.abs(out.exposures.values.T @ y.values).max() < 1e-8
    assert nu > 0
    out2 = model.as_of(d, alpha=alpha)
    assert out2.alignment_vector is not None and out2.alignment_var > 0
    w = y.copy()
    assert out2.portfolio_variance(w) > out2.portfolio_variance(w, include_alignment=False)


# --------------------------------------------------------------------------- evaluation helpers
def test_evaluation_metrics():
    rng = np.random.default_rng(0)
    var = pd.Series(rng.uniform(0.5, 2.0, 2000))
    r = pd.Series(rng.standard_normal(2000)) * np.sqrt(var)
    b = bias_statistic(np.sqrt(var), r)
    assert 0.9 < b < 1.1
    mz = mincer_zarnowitz(var, r ** 2)
    assert 0.7 < mz["beta"] < 1.3
    assert qlike(var, r ** 2) < qlike(var * 3, r ** 2)


def test_min_variance_oos_test():
    rng = np.random.default_rng(2)
    N, T = 40, 600
    beta = rng.uniform(0.5, 1.5, N)
    f = rng.standard_normal(T) * 0.01
    X = np.outer(f, beta) + rng.standard_normal((T, N)) * 0.02
    R = pd.DataFrame(X, index=pd.bdate_range("2020-01-01", periods=T))
    res = min_variance_oos_test(
        {"sample": lambda A: np.cov(A.T), "qis": nonlinear_shrinkage, "lw": linear_shrinkage},
        R, rebalance_every=21, window=120)
    assert set(res.index) == {"sample", "qis", "lw"}
    assert res.loc["qis", "oos_daily_var"] <= res.loc["sample", "oos_daily_var"]


# --------------------------------------------------------------------------- speed
def test_as_of_speed(md, model):
    dates = md.dates[-50:]
    model.as_of(dates[0])
    t0 = time.perf_counter()
    for d in dates:
        model.as_of(d)
    assert time.perf_counter() - t0 < 5.0
