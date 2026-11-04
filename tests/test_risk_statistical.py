"""Tests for the statistical (APCA) companion risk model and the hybrid model."""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from qts.core.config import RiskConfig
from qts.core.types import MarketData
from qts.data.synthetic import SyntheticSpec, generate
from qts.risk import (
    BarraStyleRiskModel, HybridRiskModel, StatisticalRiskModel, apca, build_risk_model,
    missing_factor_monitor, portfolio_bias_test, random_long_short_weights, select_n_factors,
)

INJECT_START = 500          # first day of the unobserved factor
LATENT_VOL = 0.01           # daily vol of the injected factor
N_EVAL = 150                # bias-test days at the end of the sample


def _is_psd(S, tol=1e-12):
    S = np.asarray(S)
    return np.linalg.eigvalsh((S + S.T) / 2).min() >= -tol * max(1.0, np.abs(S).max())


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def md() -> MarketData:
    return generate(SyntheticSpec(n_stocks=150, n_days=900, seed=4))


@pytest.fixture(scope="module")
def latent(md):
    """Random latent exposures ``g`` and a factor return path that is zero
    before ``INJECT_START`` and N(0, LATENT_VOL^2) afterwards."""
    rng = np.random.default_rng(42)
    g = rng.standard_normal(len(md.tickers))
    f = np.zeros(len(md.dates))
    f[INJECT_START:] = LATENT_VOL * rng.standard_normal(len(md.dates) - INJECT_START)
    return g, f


@pytest.fixture(scope="module")
def md_injected(md, latent) -> MarketData:
    """Same market with ``g_i f_t`` added to every return (no observable footprint)."""
    g, f = latent
    r = md.returns.values.copy()
    r[1:] = r[1:] + np.outer(f, g)[1:]
    close = 100 * np.exp(np.cumsum(np.log1p(np.nan_to_num(r)), axis=0))
    close = pd.DataFrame(close, index=md.dates, columns=md.tickers)
    shares = md.market_cap / md.close
    return dataclasses.replace(md, close=close, market_cap=close * shares)


@pytest.fixture(scope="module")
def cfg():
    return RiskConfig()


@pytest.fixture(scope="module")
def hybrid(md_injected, cfg):
    return HybridRiskModel(BarraStyleRiskModel(cfg), cfg).fit(md_injected)


@pytest.fixture(scope="module")
def aligned_book(md, latent):
    g, _ = latent
    w = pd.Series(g, index=md.tickers)
    w = w - w.mean()
    return w / w.abs().sum() * 2.0


# --------------------------------------------------------------------------- APCA core
def test_apca_recovers_planted_factor_structure():
    rng = np.random.default_rng(0)
    N, T, k = 200, 300, 3
    B = rng.standard_normal((N, k))
    f = 0.01 * rng.standard_normal((T, k))
    e = 0.02 * np.exp(0.3 * rng.standard_normal(N)) * rng.standard_normal((T, N))  # heteroskedastic noise
    X = f @ B.T + e
    res = apca(X, k, n_iter=3)
    assert res.F.shape == (T, k) and res.B.shape == (N, k)
    assert np.allclose(res.F.T @ res.F / T, np.eye(k), atol=1e-8)
    # projection of the true systematic covariance onto the recovered loading space
    Bh = res.B
    P = Bh @ np.linalg.solve(Bh.T @ Bh, Bh.T)
    Sig = B @ np.cov(f.T) @ B.T
    explained = np.trace(P @ Sig @ P) / np.trace(Sig)
    assert explained > 0.90, explained
    k_bn = select_n_factors(X, method="bai_ng")
    k_mp = select_n_factors(X, method="marchenko_pastur")
    assert abs(k_bn - 3) <= 1, k_bn
    assert k_mp >= 3, k_mp
    assert select_n_factors(X, method="fixed", n_factors=7) == 7
    # pure noise: neither criterion invents factors
    assert select_n_factors(e, method="bai_ng") == 0
    assert select_n_factors(e, method="marchenko_pastur") <= 1


def test_apca_handles_missing_returns():
    rng = np.random.default_rng(1)
    N, T = 120, 260
    B = rng.standard_normal((N, 2))
    X = 0.01 * rng.standard_normal((T, 2)) @ B.T + 0.02 * rng.standard_normal((T, N))
    X[rng.random((T, N)) < 0.05] = np.nan
    X[:100, 0] = np.nan                       # one stock with a short history
    res = apca(X, 2, n_iter=2)
    assert np.isfinite(res.B).all() and np.isfinite(res.specific_var).all()
    assert (res.specific_var > 0).all()
    assert res.ok.all()


# --------------------------------------------------------------------------- point in time
def test_statistical_model_is_point_in_time(md):
    R = md.returns
    kw = dict(n_factors=4, window=252, min_obs=200, refit_every=21)
    full = StatisticalRiskModel(**kw).fit(R)
    for pos in (700, 710, 899):               # a refit date, mid-cycle, last day
        d = md.dates[pos]
        a = full.as_of(d)
        b = StatisticalRiskModel(**kw).fit(R.loc[:d]).as_of(d)
        assert a.date == d and a.factor_returns.index[-1] == d
        pd.testing.assert_frame_equal(a.exposures, b.exposures, atol=1e-12, rtol=0)
        pd.testing.assert_frame_equal(a.factor_cov, b.factor_cov, atol=1e-14, rtol=0)
        pd.testing.assert_series_equal(a.specific_var, b.specific_var, atol=1e-18, rtol=0)
        pd.testing.assert_frame_equal(a.factor_returns, b.factor_returns, atol=1e-12, rtol=0)
    assert _is_psd(a.covariance().values)
    with pytest.raises(ValueError):
        full.as_of(md.dates[100])             # before the first eligible date


def test_factor_sign_and_rotation_stability(md):
    """Consecutive refits estimate the same factors with the same sign."""
    m = StatisticalRiskModel(n_factors=4, window=252, min_obs=252, refit_every=21).fit(md.returns)
    rd = m.refit_dates()
    assert len(rd) > 10
    for d1, d2 in zip(rd[:-1], rd[1:]):
        F1, F2 = m.refit_factor_returns(d1), m.refit_factor_returns(d2)
        ov = F1.index.intersection(F2.index)
        assert len(ov) > 200
        for c in F1.columns:
            assert F1.loc[ov, c].corr(F2.loc[ov, c]) > 0.0, (d2, c)
    # the real-time chain (cross-sectional estimates between refits) tracks the refit solution
    d = rd[-1]
    F_ref = m.refit_factor_returns(d)
    live = m.factor_returns.loc[F_ref.index]
    for c in F_ref.columns:
        assert live[c].corr(F_ref[c]) > 0.5, c


# --------------------------------------------------------------------------- hybrid
def test_hybrid_prices_unobserved_factor(md_injected, hybrid, aligned_book, cfg):
    fund = hybrid.fundamental
    dates = md_injected.dates[-(N_EVAL + 1):-1]
    assert hybrid.statistical_available(dates[0])
    assert 1 <= hybrid.n_statistical_factors <= cfg.max_statistical_factors

    def ports():
        p = {"aligned": aligned_book}
        for k in range(3):
            p[f"rand{k}"] = random_long_short_weights(np.random.default_rng(k))
        return p

    res_f = portfolio_bias_test(fund, md_injected, ports(), dates)
    res_h = portfolio_bias_test(hybrid, md_injected, ports(), dates)
    print("\nbias statistics (fundamental / hybrid):")
    for name in res_f.index:
        print(f"  {name:8s} {res_f.loc[name, 'bias']:.3f}  {res_h.loc[name, 'bias']:.3f}")
    assert res_f.loc["aligned", "bias"] > 1.3
    assert 0.8 <= res_h.loc["aligned", "bias"] <= 1.25
    for k in range(3):
        assert 0.8 <= res_h.loc[f"rand{k}", "bias"] <= 1.25

    # a diversified random book does not get riskier on average (diagonal is pinned)
    vh, vf = [], []
    for d in dates[::30]:
        a, b = fund.as_of(d), hybrid.as_of(d)
        assert np.allclose(np.diag(b.covariance().values), np.diag(a.covariance().values), rtol=1e-8)
        for k in range(200):
            w = random_long_short_weights(np.random.default_rng(1000 + k))(d, a.tickers)
            vf.append(a.portfolio_variance(w))
            vh.append(b.portfolio_variance(w))
    ratio = float(np.mean(vh) / np.mean(vf))
    print(f"  random-book predicted variance, hybrid / fundamental: {ratio:.4f}")
    assert ratio <= 1.05


def test_hybrid_output_structure(md_injected, hybrid, cfg):
    d = md_injected.dates[-5]
    fund_out = hybrid.fundamental.as_of(d)
    out = hybrid.as_of(d)
    k = hybrid.n_statistical_factors
    assert list(out.exposures.columns[:fund_out.exposures.shape[1]]) == list(fund_out.exposures.columns)
    assert out.exposures.shape[1] == fund_out.exposures.shape[1] + k
    assert out.factor_cov.index.equals(out.exposures.columns)
    K1 = fund_out.factor_cov.shape[0]
    assert np.allclose(out.factor_cov.values[:K1, :K1], fund_out.factor_cov.values)
    assert np.allclose(out.factor_cov.values[:K1, K1:], 0.0)       # block-diagonal
    assert _is_psd(out.factor_cov.values) and _is_psd(out.covariance().values)
    assert (out.specific_var <= fund_out.specific_var + 1e-18).all()
    assert (out.specific_var >= 0.1 * fund_out.specific_var - 1e-18).all()
    assert (hybrid.explained_share(d) > 0.3).sum() > 10             # the latent factor is seen
    # alignment passthrough
    rng = np.random.default_rng(5)
    alpha = pd.Series(rng.standard_normal(len(out.tickers)), index=out.tickers)
    out2 = hybrid.as_of(d, alpha=alpha)
    assert out2.alignment_vector is not None and out2.alignment_var > 0
    assert np.abs(out2.exposures.values.T @ out2.alignment_vector.values).max() < 1e-8
    # the statistical block is ready by the fundamental model's own first date
    # (min_obs = 0.8 * window residual days vs min_history_days factor-return days)
    first_fund = hybrid.fundamental.factor_returns.index[cfg.min_history_days - 1]
    assert hybrid.statistical_available(first_fund)
    assert hybrid.as_of(first_fund).exposures.shape[1] == fund_out.exposures.shape[1] + k
    assert not hybrid.statistical_available(md_injected.dates[cfg.min_history_days + 10])
    # pipeline passthroughs
    assert hybrid.exposures_as_of(d).equals(hybrid.fundamental.exposures_as_of(d))
    assert hybrid.factor_returns is hybrid.fundamental.factor_returns


def test_missing_factor_monitor_flags_injected_period(md_injected, hybrid, cfg):
    mon = missing_factor_monitor(md_injected, hybrid, lookback=252, step=5)
    assert {"pc1_share", "pck_share", "mp_edge", "flag", "n_factors_mp"} <= set(mon.columns)
    clean = mon.loc[: md_injected.dates[INJECT_START - 1]]
    injected = mon.loc[md_injected.dates[INJECT_START + 252]:]
    assert len(clean) > 20 and len(injected) > 20
    print(f"\nmonitor pc1 share: clean {clean['pc1_share'].mean():.3f}  injected {injected['pc1_share'].mean():.3f}"
          f"  alarm {cfg.missing_factor_alarm_share}")
    assert not clean["flag"].any()
    assert injected["flag"].all()
    assert (injected["pc1_share"] > clean["pc1_share"].max()).all()
    assert (clean["pc1_share"] < 2 * clean["mp_edge_share"].max() + 0.02).all()


def test_build_risk_model_factory(cfg):
    assert isinstance(build_risk_model(RiskConfig(statistical_companion=True)), HybridRiskModel)
    assert isinstance(build_risk_model(RiskConfig(statistical_companion=False)), BarraStyleRiskModel)
    assert isinstance(build_risk_model(cfg), HybridRiskModel)        # default on
    m = build_risk_model(RiskConfig(statistical_companion=True), warmup_days=300)
    assert m.fundamental.warmup_days == 300


def test_hybrid_factor_groups(md_injected, hybrid, cfg):
    d = md_injected.dates[-5]
    out = hybrid.as_of(d)
    g = out.factor_groups
    k = hybrid.n_statistical_factors
    assert len(g["statistical"]) == k
    assert g["statistical"] == list(out.factors[-k:])
    assert g["style"] == list(cfg.styles)
    assert sorted(g["market"] + g["industry"] + g["style"] + g["statistical"]) == sorted(out.factors)
    fund = hybrid.fundamental.as_of(d)
    assert fund.factor_groups["statistical"] == []
    assert fund.factor_groups["industry"] == g["industry"]
