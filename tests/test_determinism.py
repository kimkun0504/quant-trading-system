"""Determinism: the same seed / inputs give bit-identical outputs, and
different seeds give different outputs, for every stochastic or
order-sensitive stage (synthetic generator, alpha ``as_of``, risk ``as_of``
incl. the eigenfactor Monte Carlo and the APCA companion, optimizer solve,
and a short end-to-end pipeline run).

Why: a refactor that introduces hidden global RNG state, dict-order or
thread-order dependence, or a cache whose content depends on call history
(e.g. ``BarraStyleRiskModel._eigen_cache``) makes results irreproducible; the
golden test would then flake instead of catching drift.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from qts.alpha import AlphaModel
from qts.core.config import RiskConfig, SystemConfig
from qts.data.synthetic import SyntheticSpec, generate
from qts.portfolio import SinglePeriodOptimizer
from qts.risk import BarraStyleRiskModel, build_risk_model
from qts.tcost import TransactionCostModel

warnings.filterwarnings("ignore", category=RuntimeWarning)

SPEC = {"n_stocks": 60, "n_days": 420, "ic_true": 0.08}


def small_risk_cfg(**kw) -> RiskConfig:
    cfg = RiskConfig(min_history_days=60, n_eigen_sims=50, statistical_window=120, statistical_n_factors=2)
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def assert_frames_identical(a: pd.DataFrame | pd.Series, b: pd.DataFrame | pd.Series) -> None:
    """Bitwise identity (NaN == NaN), with identical labels."""
    assert a.shape == b.shape
    assert a.index.equals(b.index)
    if isinstance(a, pd.DataFrame):
        assert a.columns.equals(b.columns)
    np.testing.assert_array_equal(np.asarray(a, dtype=float), np.asarray(b, dtype=float))


@pytest.fixture(scope="module")
def md():
    return generate(SyntheticSpec(seed=21, **SPEC))


# ----------------------------------------------------------------------------- generator
def test_generate_same_seed_identical_different_seed_differs(md):
    md2 = generate(SyntheticSpec(seed=21, **SPEC))
    for name in ("close", "volume", "market_cap", "spread_bps", "borrow_fee", "book_to_price"):
        assert_frames_identical(getattr(md, name), getattr(md2, name))
    pd.testing.assert_series_equal(md.industry, md2.industry)
    for k in md.extra:
        assert_frames_identical(md.extra[k], md2.extra[k])
    md3 = generate(SyntheticSpec(seed=22, **SPEC))
    assert not np.allclose(md.close.to_numpy(), md3.close.to_numpy(), equal_nan=True)
    assert not np.allclose(
        md.extra["true_latent_score"].to_numpy(), md3.extra["true_latent_score"].to_numpy(), equal_nan=True
    )


# ----------------------------------------------------------------------------- alpha
def _alpha(md):
    return AlphaModel(use_ml=False).fit(md)


def test_alpha_as_of_deterministic(md):
    a1, a2 = _alpha(md), _alpha(md)
    for d in (md.dates[-1], md.dates[300]):
        o1, o2 = a1.as_of(d), a2.as_of(d)
        assert_frames_identical(o1.score, o2.score)
        assert_frames_identical(o1.expected_return, o2.expected_return)
        assert o1.half_life_days == o2.half_life_days
        assert o1.ic_estimate == o2.ic_estimate or (np.isnan(o1.ic_estimate) and np.isnan(o2.ic_estimate))
    # different data seed -> different scores
    o3 = _alpha(generate(SyntheticSpec(seed=22, **SPEC))).as_of(md.dates[-1])
    common = o3.score.index.intersection(a1.as_of(md.dates[-1]).score.index)
    assert not np.allclose(o3.score[common], a1.as_of(md.dates[-1]).score[common])


# ----------------------------------------------------------------------------- risk
@pytest.fixture(scope="module")
def risk_pair(md):
    cfg = small_risk_cfg(statistical_companion=False)
    m1 = BarraStyleRiskModel(cfg, warmup_days=60, seed=0).fit(md)
    m2 = BarraStyleRiskModel(cfg, warmup_days=60, seed=0).fit(md)
    return cfg, m1, m2


def _assert_risk_identical(o1, o2):
    assert_frames_identical(o1.exposures, o2.exposures)
    assert_frames_identical(o1.factor_cov, o2.factor_cov)
    assert_frames_identical(o1.specific_var, o2.specific_var)


def test_risk_as_of_deterministic_and_seed_sensitive(md, risk_pair):
    cfg, m1, m2 = risk_pair
    d = md.dates[-1]
    _assert_risk_identical(m1.as_of(d), m2.as_of(d))
    # the eigenfactor-adjustment Monte Carlo is the seeded stage: another seed moves F
    m3 = BarraStyleRiskModel(cfg, warmup_days=60, seed=1).fit(md)
    o1, o3 = m1.as_of(d), m3.as_of(d)
    assert_frames_identical(o1.exposures, o3.exposures)  # exposures are not random
    assert not np.allclose(o1.factor_cov.to_numpy(), o3.factor_cov.to_numpy(), rtol=1e-10, atol=0)


def _same_bucket_dates(model, md):
    """Two dates whose factor-return history lengths fall in the same
    ``eigen_refresh_days`` bucket (the ``_eigen_cache`` key)."""
    fr = model.factor_returns
    L = {d: len(fr.loc[:d]) for d in md.dates[-60:]}
    key = L[md.dates[-1]] // model.eigen_refresh_days
    ds = [d for d in md.dates[-60:] if L[d] // model.eigen_refresh_days == key]
    assert len(ds) >= 2
    return ds[0], ds[-1]


def test_risk_as_of_independent_of_call_order_across_buckets(md):
    """Dates in different eigen-cache buckets: order of queries does not matter."""
    cfg = small_risk_cfg(statistical_companion=False)
    fwd = BarraStyleRiskModel(cfg, warmup_days=60).fit(md)
    d_early, _ = _same_bucket_dates(fwd, md)
    d_late = md.dates[-1]
    d_early = md.dates[md.dates.get_loc(d_early) - 1]  # previous bucket
    a_early, a_late = fwd.as_of(d_early), fwd.as_of(d_late)
    rev = BarraStyleRiskModel(cfg, warmup_days=60).fit(md)
    b_late, b_early = rev.as_of(d_late), rev.as_of(d_early)
    _assert_risk_identical(a_late, b_late)
    _assert_risk_identical(a_early, b_early)


def test_risk_as_of_independent_of_call_order_within_bucket(md):
    """Finding #46 (fixed): the eigenfactor bias of a refresh bucket is keyed on
    the bucket's first history length and estimated from history up to it, so
    querying a later date first cannot leak its history into an earlier date."""
    cfg = small_risk_cfg(statistical_companion=False)
    fwd = BarraStyleRiskModel(cfg, warmup_days=60).fit(md)
    a, b = _same_bucket_dates(fwd, md)
    fa = fwd.as_of(a)  # in-order: bucket filled from history <= a
    rev = BarraStyleRiskModel(cfg, warmup_days=60).fit(md)
    rev.as_of(b)  # out-of-order: bucket filled from history <= b > a
    ra = rev.as_of(a)
    _assert_risk_identical(fa, ra)


def test_hybrid_risk_as_of_deterministic(md):
    cfg = small_risk_cfg(statistical_companion=True)
    h1 = build_risk_model(cfg, warmup_days=60).fit(md)
    h2 = build_risk_model(cfg, warmup_days=60).fit(md)
    d = md.dates[-1]
    _assert_risk_identical(h1.as_of(d), h2.as_of(d))


# ----------------------------------------------------------------------------- optimizer
def test_optimizer_solve_deterministic(md, risk_pair):
    _, m1, _ = risk_pair
    d = md.dates[-1]
    alpha = _alpha(md).as_of(d).expected_return
    risk = m1.as_of(d, alpha=alpha.reindex(md.tickers).fillna(0.0))
    sys_cfg = SystemConfig()
    cost = TransactionCostModel(sys_cfg.cost)
    snap = cost.snapshot_from_market(md, d)
    w0 = pd.Series(0.0, index=md.tickers)
    a = alpha.reindex(md.tickers).fillna(0.0) * 20  # strong enough to trade through costs
    r1 = SinglePeriodOptimizer(sys_cfg.portfolio, cost).solve(a, risk, w0, 1e6, snap)
    r2 = SinglePeriodOptimizer(sys_cfg.portfolio, cost).solve(a, risk, w0, 1e6, snap)
    assert r1.status == r2.status and r1.status in ("optimal", "optimal_inaccurate")
    assert r1.w_target.abs().sum() > 0.1
    assert_frames_identical(r1.w_target, r2.w_target)
    assert r1.expected_cost == r2.expected_cost
    # a different alpha (different data seed) gives a different book
    r3 = SinglePeriodOptimizer(sys_cfg.portfolio, cost).solve(-a, risk, w0, 1e6, snap)
    assert not np.allclose(r1.w_target.to_numpy(), r3.w_target.to_numpy(), atol=1e-6)


# ----------------------------------------------------------------------------- pipeline
@pytest.mark.slow
def test_pipeline_run_deterministic(golden_market, golden_pipeline_run):
    """Re-running the golden pipeline from scratch reproduces NAV, weights and
    costs bit for bit (shares the session run with tests/test_golden.py)."""
    from qts.pipeline import ResearchPipeline
    from tests.conftest import GOLDEN_MAX_DAYS, golden_config

    _, res1 = golden_pipeline_run
    res2 = ResearchPipeline(golden_config(), use_ml=False).run(golden_market, max_days=GOLDEN_MAX_DAYS)
    assert_frames_identical(res1.backtest.nav, res2.backtest.nav)
    assert_frames_identical(res1.backtest.weights, res2.backtest.weights)
    assert_frames_identical(res1.backtest.costs, res2.backtest.costs)
    assert res1.hurdles["status"] == res2.hurdles["status"]
    assert res1.ic_report["mean_ic"] == res2.ic_report["mean_ic"]
