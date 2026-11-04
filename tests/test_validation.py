"""Fast offline tests for qts.validation."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from qts.core.config import ValidationConfig
from qts.core.types import MarketData
from qts.data.synthetic import SyntheticSpec, generate
from qts.validation import (
    CombinatorialPurgedCV, HurdleReport, LeakageError, PurgedKFold, PurgedWalkForward, TrialLedger,
    annualize_sharpe, check_no_overlap, deflated_sharpe_ratio, dsr_from_trial_returns, expected_max_sharpe,
    feature_registry, label_overlap_check, label_windows_overlap, min_backtest_length, min_track_record_length,
    nagel_collapse_test, pit_shift_test, planted_oracle_test, probabilistic_sharpe,
    probability_of_backtest_overfitting, sharpe_ratio_se,
)

DATES = pd.bdate_range("2015-01-01", periods=1500)


# --------------------------------------------------------------------------- splits
def _explicit_overlap(train, test, h):
    tw = set()
    for t in test:
        tw.update(range(t, t + h + 1))
    for s in train:
        if any((s + k) in tw for k in range(h + 1)):
            return True
    return False


@pytest.mark.parametrize("splitter", [
    PurgedWalkForward(n_splits=5, embargo_days=20, label_horizon=20, min_train_days=300),
    PurgedKFold(n_splits=5, embargo_days=20, label_horizon=20),
    CombinatorialPurgedCV(n_groups=6, k_test=2, embargo=20, label_horizon=20),
])
def test_no_train_label_overlaps_test(splitter):
    n_folds = 0
    for train, test in splitter.split(DATES):
        n_folds += 1
        assert len(train) > 0 and len(test) > 0
        assert len(np.intersect1d(train, test)) == 0
        assert not _explicit_overlap(train, test, 20)
        assert not label_windows_overlap(train, test, 20)
        # embargo: no train obs within 20 days after the test forward reach
        for t_end in np.split(test, np.where(np.diff(test) > 1)[0] + 1):
            hi = t_end.max() + 20
            assert not np.any((train > hi) & (train <= hi + 20))
    assert n_folds == splitter.get_n_splits()
    assert check_no_overlap(splitter, DATES)


def test_walk_forward_is_forward_only():
    wf = PurgedWalkForward(n_splits=4, embargo_days=5, label_horizon=10, min_train_days=200)
    prev_end = -1
    for train, test in wf.split(DATES):
        assert train.max() < test.min()
        assert test.min() > prev_end
        prev_end = test.max()
    assert prev_end == len(DATES) - 1


def test_cpcv_counts_and_paths():
    cv = CombinatorialPurgedCV(n_groups=8, k_test=2, embargo=20, label_horizon=20)
    assert cv.n_splits == math.comb(8, 2) == 28
    assert cv.n_paths == 28 * 2 // 8 == 7
    splits = list(cv.split(DATES))
    assert len(splits) == 28
    paths = cv.backtest_paths()
    assert paths.shape == (8, 7)
    combos = cv.test_combinations()
    for g in range(8):
        for p in range(7):
            assert g in combos[paths[g, p]]
        assert len(set(paths[g])) == 7  # distinct splits per group across paths
    # every split used exactly k_test times overall
    counts = np.bincount(paths.ravel(), minlength=28)
    assert (counts == 2).all()
    # assemble paths: each path covers every date exactly once
    preds = [pd.Series(i, index=test) for i, (_, test) in enumerate(splits)]
    assembled = cv.assemble_paths(DATES, preds)
    assert len(assembled) == 7
    for s in assembled:
        assert len(s) == len(DATES) and s.index.is_unique


# --------------------------------------------------------------------------- DSR
def test_sharpe_se_and_psr_basics():
    assert sharpe_ratio_se(0.0, 1001) == pytest.approx(1 / math.sqrt(1000))
    assert probabilistic_sharpe(0.1, 0.0, 1000) > 0.99
    assert probabilistic_sharpe(0.0, 0.0, 1000) == pytest.approx(0.5)
    assert expected_max_sharpe(1, 1.0) == 0.0
    assert expected_max_sharpe(100, 1.0) > expected_max_sharpe(10, 1.0) > 0


def test_dsr_rejects_best_of_noise_and_accepts_true_signal():
    fails = 0
    for seed in range(5):
        rng = np.random.default_rng(seed)
        R = rng.standard_normal((1000, 100)) * 0.01
        res = dsr_from_trial_returns(R)
        if res["dsr"] < 0.95:
            fails += 1
    assert fails >= 4
    # true SR = 1.0 annual, N = 1 trial, 10 years of daily data
    rng = np.random.default_rng(123)
    sr_d = 1.0 / math.sqrt(252)
    r = rng.standard_normal(2520) * 0.01 + 0.01 * sr_d
    res = dsr_from_trial_returns(r[:, None])
    assert res["n_trials"] == 1
    assert res["dsr"] > 0.95
    assert deflated_sharpe_ratio(sr_d, 1, 0.0, 2520) > 0.95


def test_min_lengths():
    assert min_track_record_length(0.1, 0.0) > 0
    assert math.isinf(min_track_record_length(0.0, 0.0))
    assert min_backtest_length(1, 1.0) == 0.0
    n100 = min_backtest_length(100, 1.0)
    n10 = min_backtest_length(10, 1.0)
    assert n100 > n10 > 0
    # check definitional property: E[max SR] of N noise trials of length n_min ~= target
    n = min_backtest_length(50, 1.0)
    assert annualize_sharpe(expected_max_sharpe(50, 1.0 / n)) == pytest.approx(1.0, rel=1e-6)


# --------------------------------------------------------------------------- PBO
def test_pbo_noise_vs_signal():
    rng = np.random.default_rng(0)
    T, N = 1600, 30
    noise = pd.DataFrame(rng.standard_normal((T, N)) * 0.01)
    res = probability_of_backtest_overfitting(noise, n_groups=16)
    assert res["n_combinations"] == 5000 or res["n_combinations"] == math.comb(16, 8)
    assert abs(res["pbo"] - 0.5) < 0.15
    assert np.isfinite(res["degradation_slope"])
    # one real signal among noise configs
    sig = noise.copy()
    sig[0] = sig[0] + 0.002  # daily SR 0.2 -> clearly best IS and OOS
    res2 = probability_of_backtest_overfitting(sig, n_groups=16)
    assert res2["pbo"] < 0.1
    assert res2["prob_oos_loss"] < 0.1


def test_pbo_caps_combinations_with_warning():
    rng = np.random.default_rng(1)
    M = pd.DataFrame(rng.standard_normal((800, 5)))
    with pytest.warns(RuntimeWarning):
        res = probability_of_backtest_overfitting(M, n_groups=16, max_combinations=500)
    assert res["n_combinations"] == 500


# --------------------------------------------------------------------------- ledger
def test_ledger_round_trip(tmp_path):
    led = TrialLedger(tmp_path / "sub" / "trials.jsonl")
    rng = np.random.default_rng(0)
    for i in range(10):
        r = rng.standard_normal(1000) * 0.01
        sr_d = r.mean() / r.std()
        led.log(f"mom:lb={i}", {"lb": i, "arr": np.arange(2)}, {"sharpe_daily": sr_d, "n_obs": 1000},
                data_hash="abc", code_version="v0", notes="test")
    led.log("value:raw", {"x": 1}, {"sharpe_annual": 0.8, "n_obs": 1500})
    assert led.n_trials() == 11
    assert led.n_trials("mom") == 10
    rep = led.report()
    assert len(rep) == 11 and "m.n_obs" in rep.columns and "c.lb" in rep.columns
    # re-open and the records persist (append-only)
    led2 = TrialLedger(str(tmp_path / "sub" / "trials.jsonl"))
    assert led2.n_trials() == 11
    out = led2.dsr_for("mom:lb=3")
    assert 0.0 <= out["dsr"] <= 1.0 and out["n_trials"] == 10
    C = np.eye(5)
    assert TrialLedger.effective_trials(C) == pytest.approx(5.0)
    C1 = np.ones((5, 5))
    assert TrialLedger.effective_trials(C1) == pytest.approx(1.0)
    assert TrialLedger.effective_trials(None, n_trials=7) == 7.0


# --------------------------------------------------------------------------- hurdles
def test_hurdles_pass_and_fail():
    hr = HurdleReport(ValidationConfig())
    folds = pd.Series([0.01, 0.012, 0.008, 0.02, 0.006, 0.011])
    good = hr.evaluate_signal(
        ic_by_fold=pd.Series([0.03, 0.02, 0.04, 0.03, 0.02, 0.03]), decile_tstat=3.5,
        incremental_ic_by_fold=folds, net_sharpe_delta_by_fold=pd.Series([0.1, 0.08, 0.06, 0.2, 0.04, 0.07]),
        breakeven_cost_bps=40.0, modeled_cost_bps=15.0, dsr=0.97, pbo=0.2, phase=1, net_sharpe=0.6)
    assert good["overall"], good["reasons"]
    bad = hr.evaluate_signal(
        ic_by_fold=pd.Series([0.03, -0.02, 0.04, -0.03, 0.02, 0.03]), decile_tstat=2.0,
        incremental_ic_by_fold=pd.Series([0.001, 0.0, 0.002, 0.01, -0.01, 0.0]),
        net_sharpe_delta_by_fold=pd.Series([0.01, 0.0, -0.1, 0.2, 0.0, 0.0]),
        breakeven_cost_bps=20.0, modeled_cost_bps=15.0, dsr=0.5, pbo=0.7, phase=2, net_sharpe=0.4)
    assert not bad["overall"]
    for g in ("t_stat", "incremental_ic", "net_sharpe_delta", "breakeven_cost", "dsr", "pbo", "phase_sharpe"):
        assert bad["gates"][g] is False, g
    # suspect: too-good Sharpe fails even when everything else passes
    sus = hr.evaluate_signal(decile_tstat=5.0, incremental_ic_by_fold=folds,
                             net_sharpe_delta_by_fold=folds + 0.1, breakeven_cost_bps=100, modeled_cost_bps=10,
                             dsr=1.0, pbo=0.0, phase=2, net_sharpe=2.5)
    assert not sus["overall"] and sus["gates"]["suspect"] is False
    # t-stat gate not applied to incremental features
    inc = hr.evaluate_signal(decile_tstat=1.0, incremental_ic_by_fold=folds, net_sharpe_delta_by_fold=folds + 0.1,
                             breakeven_cost_bps=100, modeled_cost_bps=10, dsr=0.99, pbo=0.1, phase=1,
                             net_sharpe=0.5, standalone=False)
    assert inc["overall"]
    assert "OVERALL" in HurdleReport.format(inc)


# --------------------------------------------------------------------------- leakage
@pytest.fixture(scope="module")
def md_small() -> MarketData:
    return generate(SyntheticSpec(n_stocks=60, n_days=400, seed=3))


def test_pit_shift_test_catches_lookahead(md_small):
    cut = md_small.dates[250]

    def good(md):
        return md.returns.rolling(20).mean()

    def bad(md):
        return md.returns.shift(-1).rolling(20).mean()

    def bad_fullsample(md):
        return md.returns - md.returns.mean()

    assert pit_shift_test(good, md_small, cut)
    assert not pit_shift_test(bad, md_small, cut)
    assert not pit_shift_test(bad_fullsample, md_small, cut)
    with pytest.raises(LeakageError):
        pit_shift_test(bad, md_small, cut, raise_on_fail=True)


def test_label_overlap_check():
    d = DATES
    assert label_overlap_check(d[:100], d[130:200], horizon=20)
    assert not label_overlap_check(d[:120], d[130:200], horizon=20)
    assert not label_overlap_check(d[210:300], d[130:200], horizon=20)  # test reach [130, 219]
    assert label_overlap_check(d[221:300], d[130:200], horizon=20)


def test_planted_oracle_is_flagged(md_small):
    res = planted_oracle_test(None, md_small, horizon=20)
    assert res["sharpe"] > 2 and res["flagged"] and res["passed"]
    # a backtest that ignores the feature must NOT pass the oracle test
    res2 = planted_oracle_test(lambda md, f: 0.5, md_small, horizon=20)
    assert not res2["passed"]


def test_feature_registry_refuses_negative_shift():
    from qts.validation.leakage import FEATURE_REGISTRY

    @feature_registry
    def mom_feature(md):
        return md.close.pct_change(252).shift(21) / md.volume.rolling(5).mean()

    assert "mom_feature" in FEATURE_REGISTRY
    assert set(FEATURE_REGISTRY["mom_feature"]["reads"]) >= {"close", "volume"}

    with pytest.raises(LeakageError):
        @feature_registry
        def leaky(md):
            return md.close.shift(-1) / md.close

    with pytest.raises(LeakageError):
        @feature_registry
        def leaky_kw(md):
            return md.close.shift(periods=-5)


# --------------------------------------------------------------------------- Nagel audit
def test_nagel_collapse_test_runs_and_sane():
    from sklearn.linear_model import Ridge
    est = Ridge(alpha=1.0)
    res = nagel_collapse_test(lambda Xtr, ytr, Xte: est.fit(Xtr, ytr).predict(Xte),
                              n_assets=40, n_days=600, seed=0, n_lags=20, horizon=10)
    assert set(res) >= {"ic_mean", "ic_tstat", "corr_with_recency_mean", "passed", "collapsed"}
    assert abs(res["ic_tstat"]) < 4.0
    # a model that literally is the recency-weighted mean must be flagged as collapsed
    from qts.validation.synthetic_audit import recency_weighted_mean
    res2 = nagel_collapse_test(lambda Xtr, ytr, Xte: recency_weighted_mean(Xte, 10.0),
                               n_assets=40, n_days=600, seed=0, n_lags=20, horizon=10)
    assert res2["collapsed"] and res2["corr_with_recency_mean"] > 0.99
