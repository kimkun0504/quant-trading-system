"""Property-based point-in-time (PIT) tests.

For a random cut date ``t`` (hypothesis, derandomized so CI is stable) every
object the system computes "as of ``t``" must be identical whether it was
computed on the full history or on ``md.slice_until(t)`` -- i.e. it may not
read anything stamped after ``t``.  Covered: every raw panel of
``FeatureBuilder.build`` and the processed long frame, ``build_universe_mask``,
``TransactionCostModel.snapshot_from_market``, the risk model ``as_of`` and
``AlphaModel(use_ml=False).as_of``.  Labels are the legitimate exception:
``residualize_forward_returns`` at row ``t`` must *change* when prices after
``t`` change, while features at rows ``<= t`` must not.

The market is a small time-varying universe (60 names x 420 days, top-45 by
cap) so that eligibility enters every cross-section.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from qts.alpha import AlphaModel, FeatureBuilder, residualize_forward_returns
from qts.core.config import AlphaConfig, RiskConfig, UniverseConfig
from qts.core.types import MarketData
from qts.data.synthetic import SyntheticSpec, generate
from qts.data.universe import apply_mask, build_universe_mask
from qts.risk import BarraStyleRiskModel
from qts.tcost import TransactionCostModel

warnings.filterwarnings("ignore", category=RuntimeWarning)

N_DAYS = 420
HORIZON = 20
UCFG = UniverseConfig(top_n_by_mcap=45)

FAST = settings(max_examples=15, deadline=None, derandomize=True,
                suppress_health_check=[HealthCheck.too_slow], database=None)
MEDIUM = settings(max_examples=8, deadline=None, derandomize=True,
                  suppress_health_check=[HealthCheck.too_slow], database=None)
SLOW = settings(max_examples=6, deadline=None, derandomize=True,
                suppress_health_check=[HealthCheck.too_slow], database=None)

# cut positions: leave >= 1 day after the cut so the truncation is real
cut_pos = st.integers(min_value=40, max_value=N_DAYS - 2)
late_cut_pos = st.integers(min_value=200, max_value=N_DAYS - 2)   # risk/alpha need history


def _eq(a, b, what: str, rtol: float | None = None) -> None:
    """Equality of two frames/series on their common (truncated) rows: exact by
    default, or to ``rtol`` (atol 1e-14) where only float summation order differs."""
    kw = dict(check_exact=True) if rtol is None else dict(check_exact=False, rtol=rtol, atol=1e-14)
    if isinstance(a, pd.DataFrame):
        pd.testing.assert_frame_equal(a, b, check_freq=False, obj=what, **kw)
    else:
        pd.testing.assert_series_equal(a, b, check_freq=False, obj=what, **kw)


@pytest.fixture(scope="module")
def raw_md() -> MarketData:
    return generate(SyntheticSpec(n_stocks=60, n_days=N_DAYS, seed=5, ic_true=0.08))


@pytest.fixture(scope="module")
def md(raw_md) -> MarketData:
    return apply_mask(raw_md, build_universe_mask(raw_md, UCFG))


@pytest.fixture(scope="module")
def acfg() -> AlphaConfig:
    return AlphaConfig()


@pytest.fixture(scope="module")
def full_features(md, acfg):
    fb = FeatureBuilder(acfg)
    raw = fb.build(md)
    return raw, fb.process(raw, md, valid=md.eligibility())


# ----------------------------------------------------------------------------- (a) features
@FAST
@given(p=cut_pos)
def test_features_pit(md, acfg, full_features, p):
    t = md.dates[p]
    raw_full, X_full = full_features
    fb = FeatureBuilder(acfg)
    cut = md.slice_until(t)
    raw_cut = fb.build(cut)
    assert list(raw_cut) == list(raw_full)            # same registered panels
    for name, panel in raw_full.items():
        _eq(raw_cut[name], panel.loc[:t], f"feature {name!r} cut at {t.date()}")
    X_cut = fb.process(raw_cut, cut, valid=cut.eligibility())
    X_ref = X_full.loc[X_full.index.get_level_values("date") <= t]
    _eq(X_cut, X_ref, f"processed features cut at {t.date()}")


# ----------------------------------------------------------------------------- (b) universe
@FAST
@given(p=cut_pos)
def test_universe_mask_pit(raw_md, p):
    t = raw_md.dates[p]
    full = build_universe_mask(raw_md, UCFG)
    cut = build_universe_mask(raw_md.slice_until(t), UCFG)
    _eq(cut, full.loc[:t], f"universe mask cut at {t.date()}")
    # and the masked market's eligibility panel
    # apply_mask drops never-eligible tickers from the *column set* (documented storage
    # convenience), so the truncated market can have fewer columns: compare per-date
    # membership on the full column set
    e_full = apply_mask(raw_md, full).eligibility().loc[:t]
    e_cut = apply_mask(raw_md.slice_until(t), cut).eligibility()
    assert set(e_cut.columns) <= set(e_full.columns)
    _eq(e_cut.reindex(columns=e_full.columns, fill_value=False), e_full, "eligibility")


# ----------------------------------------------------------------------------- (c) cost snapshot
@FAST
@given(p=cut_pos)
def test_cost_snapshot_pit(md, p):
    t = md.dates[p]
    tcm = TransactionCostModel()
    a = tcm.snapshot_from_market(md, t)
    b = tcm.snapshot_from_market(md.slice_until(t), t)
    assert set(a) == set(b)
    for k in a:
        if isinstance(a[k], (pd.Series, pd.DataFrame)):
            _eq(b[k], a[k], f"snapshot[{k!r}] at {t.date()}")
        else:
            assert a[k] == b[k], k


# ----------------------------------------------------------------------------- (d) risk as_of
RISK_CFG = dict(min_history_days=60, n_eigen_sims=30, statistical_companion=False)


@pytest.fixture(scope="module")
def risk_full(md):
    return BarraStyleRiskModel(RiskConfig(**RISK_CFG), warmup_days=60).fit(md)


@SLOW
@given(p=late_cut_pos)
def test_risk_as_of_pit(md, risk_full, p):
    t = md.dates[p]
    m_cut = BarraStyleRiskModel(RiskConfig(**RISK_CFG), warmup_days=60).fit(md.slice_until(t))
    # fresh full model per example: _eigen_cache is call-order dependent (see
    # test_determinism.py::test_risk_as_of_independent_of_call_order_within_bucket)
    m_full = BarraStyleRiskModel(RiskConfig(**RISK_CFG), warmup_days=60)
    m_full.md, m_full.exposures = risk_full.md, risk_full.exposures
    m_full.factor_returns, m_full.specific_returns, m_full.r2 = (
        risk_full.factor_returns, risk_full.specific_returns, risk_full.r2)
    a, b = m_full.as_of(t), m_cut.as_of(t)
    _eq(b.exposures, a.exposures, f"risk exposures at {t.date()}")
    _eq(b.specific_var, a.specific_var, f"specific_var at {t.date()}")
    _eq(b.factor_returns, a.factor_returns, "factor return history")
    np.testing.assert_array_equal(b.factor_cov.to_numpy(), a.factor_cov.to_numpy())


# ----------------------------------------------------------------------------- (e) alpha as_of
@pytest.fixture(scope="module")
def alpha_full(md, acfg):
    return AlphaModel(acfg, use_ml=False).fit(md)


@SLOW
@given(p=late_cut_pos)
def test_alpha_as_of_pit(md, acfg, alpha_full, p):
    t = md.dates[p]
    a = alpha_full.as_of(t)
    b = AlphaModel(acfg, use_ml=False).fit(md.slice_until(t)).as_of(t)
    assert a.date == b.date
    # The *order* of the emitted names differs between the full and truncated fit
    # (it follows when names entered the universe), and with it the float summation
    # order of the cross-sectional z-score: compare sorted, to 1e-12.
    _eq(b.score.sort_index(), a.score.sort_index(), f"alpha score at {t.date()}", rtol=1e-12)
    _eq(b.expected_return.sort_index(), a.expected_return.sort_index(), f"expected return at {t.date()}",
        rtol=1e-12)
    assert a.ic_estimate == pytest.approx(b.ic_estimate, rel=1e-12, nan_ok=True)
    assert a.half_life_days == pytest.approx(b.half_life_days, rel=1e-12)
    assert set(a.components) == set(b.components)
    for k in a.components:
        _eq(b.components[k].sort_index(), a.components[k].sort_index(), f"component {k!r}", rtol=1e-12)


# ----------------------------------------------------------------------------- labels vs features
def _perturb_after(md: MarketData, p: int, seed: int) -> MarketData:
    """Copy of ``md`` with every close strictly after ``dates[p]`` multiplied by a
    per-name random factor (a different future, same past)."""
    rng = np.random.default_rng(seed)
    close = md.close.copy()
    mult = rng.uniform(0.7, 1.3, size=close.shape[1])
    close.iloc[p + 1:] = close.iloc[p + 1:].to_numpy() * mult[None, :]
    return MarketData(close=close, volume=md.volume, market_cap=md.market_cap, industry=md.industry,
                      spread_bps=md.spread_bps, short_interest_ratio=md.short_interest_ratio,
                      borrow_fee=md.borrow_fee, book_to_price=md.book_to_price,
                      earnings_yield=md.earnings_yield, gross_profitability=md.gross_profitability,
                      asset_growth=md.asset_growth, extra=md.extra, eligible=md.eligible)


@MEDIUM
@given(p=st.integers(min_value=60, max_value=N_DAYS - HORIZON - 1), seed=st.integers(0, 2**16))
def test_labels_depend_on_future_features_do_not(md, acfg, full_features, p, seed):
    t = md.dates[p]
    md2 = _perturb_after(md, p, seed)
    y1 = residualize_forward_returns(md, HORIZON)
    y2 = residualize_forward_returns(md2, HORIZON)
    # the label at t uses close[t + H] -> must move
    r1, r2 = y1.loc[t], y2.loc[t]
    ok = r1.notna() & r2.notna()
    assert ok.sum() >= 20
    assert not np.allclose(r1[ok], r2[ok]), "label at t ignored the future it is defined on"
    # labels whose window ended by t do not move
    p_old = p - HORIZON
    if p_old >= 0:
        _eq(y2.iloc[: p_old + 1], y1.iloc[: p_old + 1], "labels with windows ending <= t")
    # features on rows <= t are unchanged
    raw_full, _ = full_features
    raw2 = FeatureBuilder(acfg).build(md2)
    for name, panel in raw_full.items():
        _eq(raw2[name].loc[:t], panel.loc[:t], f"feature {name!r} under a perturbed future")
