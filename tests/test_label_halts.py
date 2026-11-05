"""Labels under halts, delistings and an execution lag (audit 2, findings 1, 3, 19).

* A temporary halt must not be classified with prices after the date the label
  is consumed: residualised labels realised by ``t``, the trailing IC estimate
  and the expected returns at ``t`` are identical between a fit on the full
  data and a fit on ``md.slice_until(t)`` (15 names halted for 10 days, then
  resumed).
* The causal delisting rule (``AlphaConfig.delisting_gap_days``) and vendor
  ``MarketData.delisting_dates`` apply the haircut to true delistings only.
* With ``label_lag_days = L`` the label of ``s`` is ``close_{s+L} -> close_{s+L+h}``.
"""

from __future__ import annotations

import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from qts.alpha import AlphaModel
from qts.alpha.models import RidgeModel
from qts.alpha.targets import forward_returns, label_realisation, residualize_forward_returns
from qts.alpha.trainer import WalkForwardTrainer
from qts.core.config import AlphaConfig
from qts.core.types import MarketData
from qts.data.synthetic import SyntheticSpec, generate

warnings.filterwarnings("ignore", category=RuntimeWarning)

H = 20
HALT_START = 560
HALT_DAYS = 10


def _with_halts(md: MarketData, names: pd.Index, start: int, days: int) -> MarketData:
    blank = pd.DataFrame(False, index=md.dates, columns=md.tickers)
    blank.loc[md.dates[start : start + days], names] = True
    return replace(
        md,
        close=md.close.mask(blank),
        close_raw=None if md.close_raw is None else md.close_raw.mask(blank),
        volume=md.volume.mask(blank),
        market_cap=md.market_cap.mask(blank),
        spread_bps=None if md.spread_bps is None else md.spread_bps.mask(blank),
    )


@pytest.fixture(scope="module")
def halted_md() -> MarketData:
    md = generate(SyntheticSpec(n_stocks=80, n_days=700, seed=11, ic_true=0.08))
    return _with_halts(md, md.tickers[:15], HALT_START, HALT_DAYS)


def _cfg() -> AlphaConfig:
    # short walk-forward schedule / IC windows so a 700-day market exercises the
    # trainer purge and the trailing IC around the halt
    return AlphaConfig(
        train_min_days=300, refit_frequency_days=63, valid_days=63, ic_window=126, ic_min_obs=20
    )


def _fit(md: MarketData, lag: int) -> AlphaModel:
    am = AlphaModel(_cfg(), use_ml=True, ml_factory=RidgeModel, signal_lag_days=lag)
    return am.fit(md)


@pytest.mark.parametrize("lag", [0, 1])
@pytest.mark.parametrize("t_pos", [HALT_START + 4, HALT_START + HALT_DAYS + 3, HALT_START + 45])
def test_halt_labels_ic_and_expected_return_are_pit(halted_md, lag, t_pos):
    md = halted_md
    t = md.dates[t_pos]
    full = _fit(md, lag)
    live = _fit(md.slice_until(t), lag)
    # labels realised by t (row decided by t) are identical
    ready = full.label_ready_
    rows = ready.index[(ready.notna() & (ready <= t)).to_numpy()]
    assert len(rows) > 100
    pd.testing.assert_index_equal(
        rows, live.label_ready_.index[(live.label_ready_.notna() & (live.label_ready_ <= t)).to_numpy()]
    )
    pd.testing.assert_frame_equal(
        full.target_.loc[rows], live.target_.loc[rows], check_exact=False, rtol=1e-10, atol=1e-12
    )
    # rows touching the halt that are not decided yet are excluded on both sides
    if t_pos == HALT_START + 4:
        undecided = full.label_ready_.index[(full.label_ready_ > t).to_numpy()]
        assert undecided.min() < t
    # IC estimate and expected returns at t
    a, b = full.as_of(t), live.as_of(t)
    assert a.ic_estimate == pytest.approx(b.ic_estimate, rel=1e-12, abs=1e-15)
    assert full.trailing_ic(t) == pytest.approx(live.trailing_ic(t), rel=1e-10, nan_ok=True)
    pd.testing.assert_series_equal(a.score, b.score, check_exact=False, rtol=1e-10, atol=1e-12)
    pd.testing.assert_series_equal(
        a.expected_return, b.expected_return, check_exact=False, rtol=1e-10, atol=1e-15
    )


def test_halt_is_not_a_delisting_and_rule_is_causal(halted_md):
    md = halted_md
    name = md.tickers[0]
    s = md.dates[HALT_START - H + 3]  # window end falls inside the halt
    full = forward_returns(md, H)  # resumption seen within 63 days -> halt, no haircut
    last = md.close[name].iloc[:HALT_START].iloc[-1]
    assert full.loc[s, name] == pytest.approx(last / md.close.loc[s, name] - 1.0)
    # sliced inside the halt: undecided -> NaN (neither haircut nor plain return)
    cut = md.slice_until(md.dates[HALT_START + 5])
    assert np.isnan(forward_returns(cut, H).loc[s, name])
    # the halt resolves when the name resumes
    ready = label_realisation(md, H)
    assert ready.loc[s] == md.dates[HALT_START + HALT_DAYS]


def test_causal_delisting_rule_and_vendor_dates():
    md = generate(SyntheticSpec(n_stocks=40, n_days=400, seed=3))
    gone = md.tickers[:3]
    d = 250
    blank = pd.DataFrame(False, index=md.dates, columns=md.tickers)
    blank.loc[md.dates[d:], gone] = True
    md = replace(md, close=md.close.mask(blank), market_cap=md.market_cap.mask(blank))
    s = md.dates[d - 5]
    plain = md.close.loc[md.dates[d - 1], gone] / md.close.loc[s, gone] - 1.0
    G = 63
    fr = forward_returns(md, H, delisting_return=-0.3, delisting_gap_days=G)
    np.testing.assert_allclose(fr.loc[s, gone], (1 + plain) * 0.7 - 1)
    # decided only G days after the last close: a slice before that is undecided
    assert np.isnan(forward_returns(md.slice_until(md.dates[d - 1 + G - 1]), H).loc[s, gone]).all()
    cut = forward_returns(md.slice_until(md.dates[d - 1 + G]), H, delisting_gap_days=G)
    np.testing.assert_allclose(cut.loc[s, gone], (1 + plain) * 0.7 - 1)
    assert label_realisation(md, H, delisting_gap_days=G).loc[s] == md.dates[d - 1 + G]
    # vendor delisting dates: decided at the window end, no G-day wait
    dd = pd.Series(pd.NaT, index=md.tickers, dtype="datetime64[ns]")
    dd[gone] = md.dates[d - 1]
    mv = replace(md, delisting_dates=dd)
    assert label_realisation(mv, H).loc[s] == md.dates[md.dates.get_loc(s) + H]
    np.testing.assert_allclose(forward_returns(mv, H).loc[s, gone], (1 + plain) * 0.7 - 1)
    # a covered name with NaT is known alive: a gap is a halt (no haircut)
    mv2 = replace(md, delisting_dates=pd.Series(pd.NaT, index=md.tickers, dtype="datetime64[ns]"))
    np.testing.assert_allclose(forward_returns(mv2, H).loc[s, gone], plain)
    # slice_until blanks vendor events after the cut
    assert mv.slice_until(md.dates[100]).delisting_dates.isna().all()


def test_label_lag_shifts_window_by_L():
    md = generate(SyntheticSpec(n_stocks=30, n_days=200, seed=4))
    c = md.close
    f0 = forward_returns(md, H)
    f1 = forward_returns(md, H, label_lag_days=1)
    pd.testing.assert_frame_equal(f0, c.shift(-H) / c - 1.0, check_freq=False)  # L=0 unchanged
    exp1 = (c.shift(-(H + 1)) / c.shift(-1) - 1.0).where(c.notna())
    pd.testing.assert_frame_equal(f1, exp1, check_freq=False)
    # label of s with L=1 == label of s+1 with L=0 (one-day shift)
    pd.testing.assert_frame_equal(f1.iloc[:-1], f0.shift(-1).iloc[:-1], check_freq=False)
    # realisation dates grow by L; residualised labels are defined on the same rows
    r0, r1 = label_realisation(md, H), label_realisation(md, H, label_lag_days=1)
    assert r0.iloc[0] == md.dates[H] and r1.iloc[0] == md.dates[H + 1]
    y1 = residualize_forward_returns(md, H, label_lag_days=1)
    assert y1.iloc[-(H + 1) :].isna().all().all() and y1.iloc[-(H + 2)].notna().any()


def test_trainer_embargo_grows_with_lag():
    cfg = AlphaConfig()
    assert WalkForwardTrainer(cfg).embargo == cfg.horizon_days
    assert WalkForwardTrainer(cfg, label_lag_days=1).embargo == cfg.horizon_days + 1


def test_alpha_model_lag0_unchanged_and_lag1_ic_window():
    md = generate(SyntheticSpec(n_stocks=60, n_days=500, seed=2, ic_true=0.08))
    cfg = AlphaConfig(ic_window=126, ic_min_obs=20)
    a0 = AlphaModel(cfg, use_ml=False).fit(md)
    a1 = AlphaModel(cfg, use_ml=False, signal_lag_days=1).fit(md)
    # L=0 labels equal the classic residualised label; L=1 uses the shifted return
    pd.testing.assert_frame_equal(a0.target_, residualize_forward_returns(md, H, features=a0.raw_features_))
    assert not a0.target_.equals(a1.target_)
    # the trailing IC of L=1 at t is the L=0 estimator's window shifted one day back
    t = md.dates[400]
    ic = a1.oos_ic_
    end = 400 - H - 1
    assert a1.trailing_ic(t) == pytest.approx(float(ic.iloc[end - 126 + 1 : end + 1].dropna().mean()))


def test_synthetic_halts_and_delistings_flags():
    base = generate(SyntheticSpec(n_stocks=40, n_days=300, seed=1))
    assert base.delisting_dates is None and "true_halts" not in base.extra
    md = generate(SyntheticSpec(n_stocks=40, n_days=300, seed=1, n_halts=4, halt_days=7, n_delistings=3))
    halted = md.extra["true_halts"].notna()
    assert int(halted.any().sum()) == 4 and int(halted.sum().max()) == 7
    assert md.close.where(halted).isna().all().all()
    dd = md.delisting_dates.dropna()
    assert len(dd) == 3
    for tk, last in dd.items():
        assert md.close[tk].last_valid_index() == last
    # every other panel and the non-event names are untouched
    keep = md.tickers[~(halted.any() | md.tickers.isin(dd.index))]
    pd.testing.assert_frame_equal(md.close[keep], base.close[keep])
    pd.testing.assert_frame_equal(md.book_to_price, base.book_to_price)
