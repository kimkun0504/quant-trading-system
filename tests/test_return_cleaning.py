"""Return cleaning (qts.core.returns) and the outlier defences built on it:
the risk model's cleaned estimation inputs and reference-book guard, the
label builder's bad-price rule, the simulator's data-error path and the
``risk_model_sanity`` hurdle (docs/notes/risk_outlier_fix_2026-11-10.md)."""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
import pandas as pd
import pytest

from qts.alpha.targets import forward_returns
from qts.backtest.simulator import DailyBarSimulator
from qts.core.config import RiskConfig, SystemConfig
from qts.core.returns import (
    MAD_TO_SD,
    clean_returns,
    data_error_mask,
    flag_bad_prices,
    winsorize_cross_section,
)
from qts.data.synthetic import SyntheticSpec, generate
from qts.portfolio import SimpleCostModel, SinglePeriodOptimizer
from qts.risk.model import BarraStyleRiskModel
from qts.validation.hurdles import HurdleReport
from tests.conftest import StandInAlpha, StandInRiskModel

# --------------------------------------------------------------------------- unit: rules


def _panel(values, n_dates=None):
    arr = np.asarray(values, dtype=float)
    idx = pd.date_range("2020-01-01", periods=arr.shape[0], freq="B")
    cols = [f"S{j}" for j in range(arr.shape[1])]
    return pd.DataFrame(arr, index=idx, columns=cols)


def test_rule_a_caps_are_asymmetric_and_skip_nan():
    r = _panel([[0.1, 3.5, -0.96, np.nan], [2.99, -0.94, 0.0, 10.0]])
    bad = data_error_mask(r, None)
    assert bad.to_numpy().tolist() == [[False, True, True, False], [False, False, False, True]]
    out = clean_returns(r, None, mad_k=None)
    assert out.n_data_errors == 3 and out.n_winsorized == 0
    assert np.isnan(out.returns.iat[0, 1]) and out.returns.iat[1, 0] == 2.99
    assert out.data_errors_by_date().tolist() == [2, 1]
    assert out.data_errors_by_name().to_dict() == {"S1": 1, "S2": 1, "S3": 1}


def test_rule_a_price_floor_uses_previous_close_only():
    r = _panel([[0.01, 0.01], [0.02, 0.02], [0.03, 0.03]])
    raw = _panel([[0.10, 5.0], [5.0, 5.0], [5.0, 0.10]])  # S0 cheap on day 0, S1 cheap on day 2
    bad = data_error_mask(r, raw, min_prev_price=0.25)
    # S0: the return of day 1 (off the 0.10 close) is dropped, day 0 / 2 kept
    # S1: the cheap close on day 2 only affects day 3 (not in the panel) -> nothing flagged
    assert bad["S0"].tolist() == [False, True, False]
    assert bad["S1"].tolist() == [False, False, False]
    assert not data_error_mask(r, raw, min_prev_price=None).to_numpy().any()


def test_rule_b_clips_at_k_mads_and_flags():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0, 0.01, size=(3, 60))
    r[1, 5] = 0.9  # ~90 robust sigmas on day 1
    r[2, :] = np.nan
    r[2, :10] = 0.5  # only 10 finite names: row untouched
    panel = _panel(r)
    clipped, flags = winsorize_cross_section(panel, mad_k=8.0)
    med = np.median(r[1])
    mad = np.median(np.abs(r[1] - med))
    assert clipped.iat[1, 5] == pytest.approx(med + 8.0 * MAD_TO_SD * mad)
    assert flags.iat[1, 5] and flags.to_numpy().sum() == 1
    assert (clipped.iloc[2, :10] == 0.5).all()
    assert clipped.iloc[0].equals(panel.iloc[0])


def test_clean_returns_is_point_in_time():
    rng = np.random.default_rng(1)
    r = _panel(rng.normal(0, 0.01, size=(30, 40)))
    raw = _panel(np.full((30, 40), 20.0))
    a = clean_returns(r, raw)
    r2 = r.copy()
    r2.iloc[20:] = 50.0  # garbage in the future
    raw2 = raw.copy()
    raw2.iloc[20:] = 0.01
    b = clean_returns(r2, raw2)
    pd.testing.assert_frame_equal(a.returns.iloc[:20], b.returns.iloc[:20])
    pd.testing.assert_frame_equal(a.data_error.iloc[:20], b.data_error.iloc[:20])


def test_invalid_thresholds_raise():
    r = _panel([[0.0, 0.0]])
    with pytest.raises(ValueError):
        data_error_mask(r, None, cap_up=-1.0)
    with pytest.raises(ValueError):
        data_error_mask(r, None, cap_down=0.5)
    with pytest.raises(ValueError):
        winsorize_cross_section(r, mad_k=0.0)
    with pytest.raises(ValueError):
        RiskConfig(return_abs_cap_down=0.1)


def test_flag_bad_prices_spike_is_one_bad_close():
    c = _panel(np.full((10, 2), 100.0))
    c.iloc[4, 0] = 100_000.0  # +99,900 % then -99.9 %
    clean, bad = flag_bad_prices(c)
    assert bad.to_numpy().sum() == 1 and bad.iat[4, 0]
    assert np.isnan(clean.iat[4, 0]) and clean.iat[5, 0] == 100.0
    # a restated level that persists is blanked day by day, up to max_passes days
    c2 = _panel(np.full((10, 1), 100.0))
    c2.iloc[4:, 0] = 100_000.0
    _, bad2 = flag_bad_prices(c2, max_passes=3)
    assert bad2.iloc[:, 0].tolist() == [False] * 4 + [True] * 3 + [False] * 3
    # genuine crash and recovery inside the caps: untouched
    c3 = _panel([[100.0], [25.0], [60.0], [200.0]])
    _, bad3 = flag_bad_prices(c3)
    assert not bad3.to_numpy().any()


# --------------------------------------------------------------------------- injected outlier
N_STOCKS, N_DAYS, SPIKE_DAY = 150, 900, 620
SPIKED = ("S0003", "S0077")


@pytest.fixture(scope="module")
def clean_md():
    return generate(SyntheticSpec(n_stocks=N_STOCKS, n_days=N_DAYS, seed=5, ic_true=0.08))


@pytest.fixture(scope="module")
def dirty_md(clean_md):
    close = clean_md.close.copy()
    d = clean_md.dates[SPIKE_DAY]
    for tk in SPIKED:
        close.loc[d, tk] *= 1000.0
    return dataclasses.replace(clean_md, close=close)


@pytest.fixture(scope="module")
def fitted(clean_md, dirty_md):
    cfg = RiskConfig(statistical_companion=False)
    return (
        BarraStyleRiskModel(cfg).fit(clean_md),
        BarraStyleRiskModel(cfg).fit(dirty_md),
        BarraStyleRiskModel(dataclasses.replace(cfg, clean_returns=False)).fit(dirty_md),
    )


def test_synthetic_market_has_no_data_errors(clean_md):
    out = clean_returns(clean_md.returns, clean_md.close_raw)
    assert out.n_data_errors == 0
    assert not flag_bad_prices(clean_md.close)[1].to_numpy().any()


def test_risk_model_reference_vol_survives_injected_spike(clean_md, dirty_md, fitted):
    clean, dirty, raw = fitted
    assert dirty.diagnostics["n_data_error_returns"] == 2 * len(SPIKED)  # spike + reversal per name
    assert dirty.cleaning is not None and dirty.cleaning.data_errors_by_name().to_dict() == dict.fromkeys(
        SPIKED, 2
    )
    d = clean_md.dates[SPIKE_DAY]
    days = [t for t in clean_md.dates[SPIKE_DAY : SPIKE_DAY + 120] if t >= d]
    worst = 0.0
    for t in days:
        v_clean = clean.as_of(t)
        v_dirty = dirty.as_of(t)
        rv_c, rv_d = clean.diagnostics["reference_vol"][t], dirty.diagnostics["reference_vol"][t]
        worst = max(worst, abs(rv_d / rv_c - 1.0))
        for tk in SPIKED:
            if t == d:  # the blanked close: out of the model that day, like a halt
                assert tk not in v_dirty.specific_var.index
            else:
                assert np.isfinite(v_dirty.specific_var[tk]) and v_dirty.specific_var[tk] > 0
        assert np.isfinite(v_clean.specific_var).all()
    assert worst < 0.20, f"reference vol moved {worst:.1%} after the spike"
    assert dirty.diagnostics["covariance_outliers"] == []
    # without cleaning the covariance explodes and the guard records it
    raw.as_of(clean_md.dates[SPIKE_DAY + 1])
    out = raw.diagnostics["covariance_outliers"]
    assert out and out[0]["reference_vol"] > 1.0
    assert (
        raw.diagnostics["reference_vol"][clean_md.dates[SPIKE_DAY + 1]]
        > 20 * clean.diagnostics["reference_vol"][clean_md.dates[SPIKE_DAY + 1]]
    )


def test_label_of_spiked_cell_is_nan_and_others_unchanged(clean_md, dirty_md):
    h = 20
    fc = forward_returns(clean_md, h)
    fd = forward_returns(dirty_md, h)
    d = clean_md.dates[SPIKE_DAY]
    for tk in SPIKED:
        assert np.isnan(fd.loc[d, tk])
        # a window ending on the blanked day takes the halt path: it runs to the
        # last good close and is an ordinary-sized label, not an absurd one
        v = fd.loc[clean_md.dates[SPIKE_DAY - h], tk]
        assert np.isfinite(v) and abs(v) < 1.0
    others = [c for c in clean_md.tickers if c not in SPIKED]
    pd.testing.assert_frame_equal(fc[others], fd[others])
    # with the rule off the labels carry the absurd move
    fraw = forward_returns(dirty_md, h, bad_price_caps=None)
    assert fraw.loc[d, SPIKED[0]] < -0.99


def test_simulator_blanks_bad_prices_and_nav_unchanged_when_not_held(clean_md, dirty_md):
    cfg = SystemConfig().with_overrides({"portfolio.risk_aversion": 3.0})
    risk = StandInRiskModel(clean_md)
    alpha = StandInAlpha(clean_md, risk)

    def alpha_fn(t):
        a = alpha.as_of(t)
        er = a.expected_return.copy()
        er[list(SPIKED)] = 0.0
        return dataclasses.replace(a, expected_return=er)

    dates = clean_md.dates[SPIKE_DAY - 20 : SPIKE_DAY + 20]
    others = clean_md.tickers.difference(pd.Index(SPIKED))
    runs = []
    for md in (clean_md, dirty_md):
        cm = SimpleCostModel(cfg.cost)
        opt = SinglePeriodOptimizer(cfg.portfolio, cm)
        runs.append(
            DailyBarSimulator(cfg, cm).run(
                md,
                dates,
                alpha_fn,
                risk.as_of,
                opt,
                nav0=1e6,
                rebalance_every=5,
                eligible_fn=lambda d: others,
            )
        )
    clean_run, dirty_run = runs
    assert clean_run.diagnostics["n_data_error_returns"] == 0
    assert dirty_run.diagnostics["n_data_error_returns"] == len(SPIKED)
    cells = {(e["date"], e["ticker"]) for e in dirty_run.diagnostics["data_error_returns"]}
    assert cells == {(clean_md.dates[SPIKE_DAY], tk) for tk in SPIKED}
    assert (dirty_run.weights[list(SPIKED)] == 0).all().all()
    # the excluded names still sit in the optimizer's problem data (cost-model vol
    # snapshot), so the solver path differs at its tolerance; the P&L does not
    np.testing.assert_allclose(clean_run.nav.to_numpy(), dirty_run.nav.to_numpy(), rtol=1e-6)
    assert dirty_run.diagnostics["n_exante_vol_alerts"] == 0


def test_risk_model_sanity_gate_fires_without_cleaning(clean_md, dirty_md, fitted):
    _, dirty, raw = fitted
    cfg = SystemConfig().with_overrides({"portfolio.risk_aversion": 3.0})
    stand_in = StandInRiskModel(clean_md)
    alpha = StandInAlpha(clean_md, stand_in)
    dates = clean_md.dates[SPIKE_DAY - 5 : SPIKE_DAY + 40]
    fracs = {}
    for name, model in (("cleaned", dirty), ("raw", raw)):
        cm = SimpleCostModel(cfg.cost)
        opt = SinglePeriodOptimizer(cfg.portfolio, cm)
        res = DailyBarSimulator(cfg, cm).run(
            dirty_md, dates, alpha.as_of, model.as_of, opt, nav0=1e6, rebalance_every=5
        )
        fracs[name] = res.exante_vol_alert_fraction()
        if name == "raw":
            alerts = res.diagnostics["exante_vol_alerts"]
            assert alerts and alerts[0]["exante_vol"] > cfg.validation.max_exante_vol
            assert len(alerts[0]["top_factors"]) == 5 and len(alerts[0]["top_names"]) <= 5
    assert fracs["cleaned"] == 0.0
    assert fracs["raw"] > cfg.validation.max_exante_vol_alert_frac
    rep = HurdleReport(cfg.validation)
    assert rep.evaluate_signal(exante_vol_alert_frac=fracs["raw"])["status"]["risk_model_sanity"] == "FAIL"
    assert (
        rep.evaluate_signal(exante_vol_alert_frac=fracs["cleaned"])["status"]["risk_model_sanity"] == "PASS"
    )
    assert rep.evaluate_signal()["status"]["risk_model_sanity"] == "skipped"


def test_risk_model_warns_on_data_errors(dirty_md, caplog):
    cfg = RiskConfig(statistical_companion=False)
    with caplog.at_level(logging.WARNING, logger="qts.risk.model"):
        BarraStyleRiskModel(cfg).fit(dirty_md)
    assert any("data-error returns removed" in m for m in caplog.messages)
