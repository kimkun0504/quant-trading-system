"""Construction fix of 2026-11-12 (docs/notes/construction_fix_2026-11-12.md):

* score neutralisation and causal smoothing before Grinold scaling
  (``AlphaConfig.neutralize_score`` / ``score_smoothing_halflife_days``);
* the vol target as a cap: gamma is only ever raised from the base
  (``PortfolioConfig.vol_target_is_cap`` / ``leverage_cap_raises_gamma``);
* trading slower: the rebalance schedule (``rebalance_every_days``) with forced
  exits on the days in between, and the min-trade projection (``min_trade_weight``);
* the diversification numbers of the summary and the default config's book
  against the rank-weighted neutral book.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pandas as pd
import pytest

from qts.alpha import AlphaModel
from qts.alpha.evaluation import rank_ic_series
from qts.alpha.score_processing import process_scores, smoothing_decay
from qts.alpha.targets import default_exposures, internal_design
from qts.backtest import DailyBarSimulator
from qts.backtest.result import BacktestResult
from qts.core.config import AlphaConfig, PortfolioConfig, RiskConfig, SystemConfig
from qts.data.synthetic import SyntheticSpec, generate
from qts.data.universe import apply_mask, build_universe_mask
from qts.portfolio import SimpleCostModel, SinglePeriodOptimizer, snapshot_from_market
from qts.portfolio.optimizer import GAMMA_MAX_SOLVES, GAMMA_SEARCH_VOL_TOL
from qts.portfolio.optimizer.spo import _Book

NAV = 1e7
STYLES = ["size", "liquidity"]


# ============================================================================ score processing
@pytest.fixture(scope="module")
def umd():
    raw = generate(SyntheticSpec(n_stocks=60, n_days=420, seed=5, ic_true=0.08))
    ucfg = SystemConfig().with_overrides({"universe.top_n_by_mcap": 45}).universe
    return apply_mask(raw, build_universe_mask(raw, ucfg))


@pytest.fixture(scope="module")
def alpha_fit(umd):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return AlphaModel(AlphaConfig(), use_ml=False).fit(umd)


def _design(md, i: int, m: np.ndarray) -> np.ndarray:
    D_of_row, S = internal_design(md, default_exposures(md))
    return np.hstack([np.ones((int(m.sum()), 1)), D_of_row[i][m], S[i][m]])


def test_neutralised_score_is_orthogonal_to_the_exposures_and_unit_sd(umd, alpha_fit):
    comb = alpha_fit.combined_score_
    sp = process_scores(umd, comb, neutralize=True, halflife_days=0.0)
    sm = process_scores(umd, comb, neutralize=True, halflife_days=10.0).score
    for d in comb.index[[-120, -60, -1]]:
        i = umd.dates.get_loc(d)
        row = sp.score.loc[d].to_numpy()
        m = np.isfinite(row)
        assert m.sum() >= 20
        # same names as the input; unit sd; no loading on intercept / industries / risk-only styles
        assert (m == comb.loc[d].notna().to_numpy()).all()
        assert row[m].std() == pytest.approx(1.0, abs=1e-12)
        X = _design(umd, i, m)
        assert np.abs(X.T @ row[m]).max() < 1e-8
        assert 0.0 <= sp.r2.loc[d] < 1.0
        # the smoothed score is projected on the day's exposures again
        row = sm.loc[d].to_numpy()
        assert np.abs(X.T @ row[m]).max() < 1e-8
        assert row[m].std() == pytest.approx(1.0, abs=1e-12)
    # and the raw combined score did load on them
    d = comb.index[-1]
    m = comb.loc[d].notna().to_numpy()
    assert np.abs(_design(umd, umd.dates.get_loc(d), m).T @ comb.loc[d].to_numpy()[m]).max() > 0.1


def test_neutralisation_uses_the_exposures_fn_when_it_has_a_model(umd, alpha_fit):
    """Dates where ``exposures_fn`` returns a frame are residualised on it (plus the
    intercept); dates where it returns ``None`` fall back to the built-in proxies."""
    comb = alpha_fit.combined_score_
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((len(umd.tickers), 3)), index=umd.tickers, columns=list("abc"))
    cut, before, after = comb.index[-40], comb.index[-70], comb.index[-20]

    def fn(d):
        return X if d >= cut else None

    sp = process_scores(umd, comb, neutralize=True, halflife_days=0.0, exposures_fn=fn)
    row = sp.score.loc[after].to_numpy()
    m = np.isfinite(row)
    assert np.abs(X.to_numpy()[m].T @ row[m]).max() < 1e-8 and abs(row[m].sum()) < 1e-8
    ref = process_scores(umd, comb, neutralize=True, halflife_days=0.0)
    pd.testing.assert_series_equal(sp.score.loc[before], ref.score.loc[before])
    assert not np.allclose(sp.score.loc[after].dropna(), ref.score.loc[after].dropna())


def test_smoothing_is_the_documented_ewma_with_entries_and_gaps():
    """Hand computation of ``s_t = num_t / den_t`` on a 6-name panel: a name that
    enters late starts at its own first (standardised) score, a name with a gap has
    its old scores discounted through the gap."""
    dates = pd.bdate_range("2024-01-01", periods=6)
    tickers = pd.Index(list("ABCDEF"))
    rng = np.random.default_rng(1)
    raw = pd.DataFrame(rng.standard_normal((6, 6)), index=dates, columns=tickers)
    raw.loc[dates[:3], "E"] = np.nan  # E enters on day 3
    raw.loc[dates[2:4], "F"] = np.nan  # F away on days 2-3
    md = dataclasses.replace(
        generate(SyntheticSpec(n_stocks=6, n_days=6, seed=0)),
        close=pd.DataFrame(1.0, index=dates, columns=tickers),
    )
    hl = 2.0
    lam = smoothing_decay(hl)
    assert lam == pytest.approx(0.5**0.5)
    got = process_scores(md, raw, neutralize=False, halflife_days=hl).score

    def unit(v):
        v = v - np.nanmean(v)
        return v / np.nanstd(v)

    z = np.vstack([unit(r) for r in raw.to_numpy()])  # each day standardised over its names
    num, den = np.zeros(6), np.zeros(6)
    for t in range(6):
        num, den = lam * num, lam * den
        ok = np.isfinite(z[t])
        num[ok] += z[t][ok]
        den[ok] += 1.0
        exp = np.full(6, np.nan)
        exp[ok] = num[ok] / den[ok]
        np.testing.assert_allclose(got.iloc[t].to_numpy(), unit(exp), rtol=1e-12, atol=1e-12)
    # the entrant's first smoothed value is its own score (before the re-standardisation)
    assert den[4] == pytest.approx(1 + lam + lam**2)  # E: three observations
    assert den[5] == pytest.approx(1 + lam + lam**4 + lam**5)  # F: days 0, 1, 4, 5


def test_processed_score_is_point_in_time(umd, alpha_fit):
    """Row t of the traded score is unchanged when everything after t changes: the
    score panel, and the data the neutralisation's exposures are built from."""
    comb = alpha_fit.combined_score_
    t = comb.index[-40]
    p = umd.dates.get_loc(t)
    rng = np.random.default_rng(3)
    future = comb.copy()
    n_after = int((comb.index > t).sum())
    future.iloc[-n_after:] = rng.standard_normal(future.iloc[-n_after:].shape)
    close = umd.close.copy()
    close.iloc[p + 1 :] = close.iloc[p + 1 :].to_numpy() * rng.uniform(0.5, 1.5, close.shape[1])[None, :]
    mcap = umd.market_cap.copy()
    mcap.iloc[p + 1 :] = mcap.iloc[p + 1 :].to_numpy() * rng.uniform(0.5, 1.5, mcap.shape[1])[None, :]
    md2 = dataclasses.replace(umd, close=close, market_cap=mcap)
    later = comb.index[-30]
    for neut, hl in ((True, 0.0), (False, 10.0), (True, 10.0)):
        a = process_scores(umd, comb, neutralize=neut, halflife_days=hl).score
        b = process_scores(md2, future, neutralize=neut, halflife_days=hl).score
        pd.testing.assert_frame_equal(a.loc[:t], b.loc[:t])
        assert not np.allclose(a.loc[later].fillna(0.0), b.loc[later].fillna(0.0))


def test_alpha_model_trades_the_processed_score_and_measures_its_ic(umd, alpha_fit):
    am = alpha_fit
    assert am.cfg.neutralize_score and am.cfg.score_smoothing_halflife_days == 10.0
    # the traded score is the processed combined score; IC and half-life are measured on it
    ref = process_scores(
        umd, am.combined_score_, neutralize=True, halflife_days=10.0, features=am.raw_features_
    ).score
    pd.testing.assert_frame_equal(am.traded_score_, ref)
    pd.testing.assert_series_equal(am.oos_ic_, rank_ic_series(am.traded_score_, am.target_))
    assert not am.oos_ic_.equals(rank_ic_series(am.combined_score_, am.target_))
    d = umd.dates[-1]
    out = am.as_of(d)
    row = am.traded_score_.loc[d].dropna()
    np.testing.assert_allclose(out.score.to_numpy(), row.reindex(out.score.index).to_numpy(), atol=1e-12)
    # Grinold: alpha = IC_used * sigma * z of the traded score
    vol = am.resid_vol_h_.loc[d].reindex(out.score.index)
    vol = vol.fillna(vol.median()).fillna(0.0)
    np.testing.assert_allclose(out.expected_return.to_numpy(), (am.ic_used(d) * vol * out.score).to_numpy())


def test_alpha_as_of_is_point_in_time_with_neutralisation_and_smoothing(umd, alpha_fit):
    """``as_of(t)`` from the full fit equals ``as_of(t)`` from a fit that never saw
    anything after t (traded score, expected return, IC used)."""
    t = alpha_fit.combined_score_.index[-25]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        b = AlphaModel(AlphaConfig(), use_ml=False).fit(umd.slice_until(t)).as_of(t)
    a = alpha_fit.as_of(t)
    pd.testing.assert_series_equal(a.score.sort_index(), b.score.sort_index(), rtol=1e-9, atol=1e-10)
    pd.testing.assert_series_equal(
        a.expected_return.sort_index(), b.expected_return.sort_index(), rtol=1e-9, atol=1e-12
    )
    assert a.ic_estimate == pytest.approx(b.ic_estimate, rel=1e-12)


def test_both_knobs_off_is_the_raw_combined_score(umd):
    cfg = dataclasses.replace(AlphaConfig(), neutralize_score=False, score_smoothing_halflife_days=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        am = AlphaModel(cfg, use_ml=False).fit(umd)
    assert am.traded_score_ is am.combined_score_
    with pytest.raises(ValueError, match="score_smoothing_halflife_days"):
        dataclasses.replace(AlphaConfig(), score_smoothing_halflife_days=-1.0)


# ============================================================================ sizing: the vol target as a cap
@pytest.fixture(scope="module")
def gctx(md):
    from qts.risk import BarraStyleRiskModel
    from qts.tcost import TransactionCostModel

    model = BarraStyleRiskModel(RiskConfig()).fit(md)
    date = md.dates[-5]
    r0 = model.as_of(date)
    rng = np.random.default_rng(7)
    sd_h = np.sqrt(r0.specific_var * 20)
    z = 0.5 * r0.exposures["momentum"] + pd.Series(rng.standard_normal(len(r0.tickers)), index=r0.tickers)
    alpha = 0.05 * sd_h * z
    return {
        "risk": model.as_of(date, alpha=alpha),
        "alpha": alpha,
        "snap": snapshot_from_market(md, date),
        "w0": pd.Series(0.0, index=r0.tickers),
        "cost": TransactionCostModel(),
    }


def _solve(gctx, cfg: PortfolioConfig, scale: float = 1.0, w0: pd.Series | None = None, **kw):
    opt = SinglePeriodOptimizer(cfg, gctx["cost"])
    res = opt.solve(
        gctx["alpha"] * scale,
        gctx["risk"],
        gctx["w0"] if w0 is None else w0,
        NAV,
        gctx["snap"],
        neutral_styles=STYLES,
        **kw,
    )
    assert res.ok
    return res


def _gross(res) -> float:
    return float(res.w_target.abs().sum())


def test_defaults_of_the_construction_fix():
    p, a = PortfolioConfig(), AlphaConfig()
    assert p.vol_target_is_cap and p.leverage_cap_raises_gamma and p.risk_aversion == 2.0
    assert p.rebalance_every_days == 5 and p.min_trade_weight == 0.0005
    assert a.neutralize_score and a.score_smoothing_halflife_days == 10.0
    for bad in ({"rebalance_every_days": 0}, {"rebalance_every_days": 2.5}, {"min_trade_weight": -1e-4}):
        with pytest.raises(ValueError):
            PortfolioConfig(**bad)


@pytest.mark.parametrize("target", [0.40, 0.10, 0.03, 0.01])
@pytest.mark.parametrize("base", [0.5, 2.0, 8.0])
def test_vol_cap_never_lowers_gamma_below_the_base(gctx, target, base):
    """Whatever the target, the returned gamma is >= the base; the book is under the
    cap (within the search tolerance) whenever a raise was needed."""
    cfg = PortfolioConfig(
        target_vol_annual=target, risk_aversion=base, leverage_cap_raises_gamma=False, min_trade_weight=0.0
    )
    res = _solve(gctx, cfg)
    base_book = _solve(
        gctx,
        PortfolioConfig(
            risk_aversion_mode="fixed", risk_aversion=base, target_vol_annual=None, min_trade_weight=0.0
        ),
    )
    assert res.risk_aversion >= base and res.n_solves <= GAMMA_MAX_SOLVES + 1
    if base_book.expected_risk <= target * (1 - GAMMA_SEARCH_VOL_TOL):
        # the base book is already under the cap: it is returned untouched
        assert res.vol_target_binding == "below_target" and res.risk_aversion == base and res.n_solves == 1
        np.testing.assert_allclose(res.w_target.to_numpy(), base_book.w_target.to_numpy(), atol=1e-9)
    elif base_book.expected_risk > target * (1 + GAMMA_SEARCH_VOL_TOL):
        assert res.vol_target_binding == "target" and res.risk_aversion > base
        assert res.expected_risk == pytest.approx(target, rel=GAMMA_SEARCH_VOL_TOL * 1.01)
        assert res.expected_risk < base_book.expected_risk


def test_unreachable_target_is_not_chased(gctx):
    """The pathology this replaces: with a 40 % target the two-sided search drives
    gamma far below the base and stops at the caps; the cap form leaves gamma alone."""
    old = _solve(gctx, PortfolioConfig(target_vol_annual=0.40, vol_target_is_cap=False, min_trade_weight=0.0))
    new = _solve(
        gctx, PortfolioConfig(target_vol_annual=0.40, leverage_cap_raises_gamma=False, min_trade_weight=0.0)
    )
    assert old.vol_target_binding == "caps" and old.risk_aversion < 2.0
    assert new.vol_target_binding == "below_target" and new.risk_aversion == 2.0
    assert new.expected_risk < old.expected_risk  # less risk for the same caps


def test_leverage_cap_is_resolved_by_raising_gamma(gctx):
    """A strong signal (unconstrained gross far above the limit): at the base gamma
    the gross limit binds and thresholds the book; with ``leverage_cap_raises_gamma``
    gamma is raised until the limit is slack, the book sits just under it and is
    spread over more names with less specific risk."""
    kw = {
        "target_vol_annual": 0.40,
        "max_adv_participation": 1.0,
        "max_position_weight": 0.08,  # 120 names: leave room for the gross limit to concentrate
        "min_trade_weight": 0.0,
    }
    off = _solve(gctx, PortfolioConfig(leverage_cap_raises_gamma=False, **kw), scale=8.0)
    on = _solve(gctx, PortfolioConfig(**kw), scale=8.0)
    assert off.binding_constraints.get("gross", 0) == 1 and off.risk_aversion == 2.0
    assert on.vol_target_binding == "leverage" and on.risk_aversion > 2.0
    assert on.binding_constraints.get("gross", 0) == 0
    assert 2.0 * 0.95 <= _gross(on) <= 2.0 + 1e-6

    def eff_n(res):
        p = res.w_target.abs() / res.w_target.abs().sum()
        return 1.0 / float((p**2).sum())

    def spec_vol(res):
        return float(np.sqrt(gctx["risk"].specific_variance(res.w_target) * 252))

    assert eff_n(on) > 1.1 * eff_n(off)
    assert spec_vol(on) / _gross(on) < 0.95 * spec_vol(off) / _gross(off)
    assert on.expected_risk / _gross(on) < off.expected_risk / _gross(off)
    # a weak signal never reaches the limit: nothing is raised
    weak = _solve(gctx, PortfolioConfig(**kw), scale=0.05)
    assert weak.vol_target_binding == "below_target" and weak.risk_aversion == 2.0


# ============================================================================ min trade
def test_min_trade_projection_rules(gctx):
    """Trades below the threshold are cancelled unless they close the position or
    the name is forced out; the count and the violation are reported."""
    thr = 5e-4
    cfg = PortfolioConfig(target_vol_annual=None, risk_aversion_mode="fixed", min_trade_weight=thr)
    opt = SinglePeriodOptimizer(cfg, gctx["cost"])
    tk = list(gctx["risk"].tickers)
    w0 = pd.Series(0.0, index=tk)
    w0[tk[0]], w0[tk[1]], w0[tk[2]], w0[tk[3]], w0[tk[4]] = 0.0100, 0.0003, 0.0003, 0.0100, 0.0100
    notes: list[str] = []
    pdat = opt._prepare(
        gctx["alpha"], gctx["risk"], w0, NAV, gctx["snap"], STYLES, None, None, {tk[2]: 0.0001}, None, notes
    )
    pos = {t: i for i, t in enumerate(pdat.tickers)}
    w = pdat.w0.copy()
    w[pos[tk[0]]] += 0.0002  # small trim of a held name -> cancelled
    w[pos[tk[1]]] = 0.0  # small trade that closes the position -> kept
    w[pos[tk[2]]] = 0.0001  # small trade of a forced exit -> kept
    w[pos[tk[3]]] -= 0.0030  # ordinary trade -> kept
    w[pos[tk[5]]] = 0.0002  # small opening trade -> cancelled
    w[pos[tk[6]]] = -0.0020  # ordinary opening trade -> kept
    book = opt._drop_small_trades(_Book(w.copy(), {}, 0.0, 2.0, "optimal", "", 1), pdat)
    x = book.w - pdat.w0
    assert book.n_dropped == 2
    assert x[pos[tk[0]]] == 0.0 and x[pos[tk[5]]] == 0.0
    assert book.w[pos[tk[1]]] == 0.0 and book.w[pos[tk[2]]] == pytest.approx(0.0001)
    assert x[pos[tk[3]]] == pytest.approx(-0.0030) and book.w[pos[tk[6]]] == pytest.approx(-0.0020)
    # every surviving trade is at least the threshold, an exit, or forced
    alive = np.abs(x) > 0
    assert ((np.abs(x[alive]) >= thr) | (book.w[alive] == 0.0) | pdat.force_mask[alive]).all()
    assert book.violation >= 0.0 and (book.violated != "") == (book.violation > 0)
    # off: untouched
    off = SinglePeriodOptimizer(dataclasses.replace(cfg, min_trade_weight=0.0), gctx["cost"])
    b0 = off._drop_small_trades(_Book(w.copy(), {}, 0.0, 2.0, "optimal", "", 1), pdat)
    np.testing.assert_array_equal(b0.w, w)
    assert b0.n_dropped == 0


def test_min_trade_never_blocks_exits_in_a_solve(gctx):
    """End to end: a forced exit smaller than the threshold and a held name the
    optimizer wants out of are both traded; no other trade is below the threshold."""
    thr = 5e-4
    first = _solve(gctx, PortfolioConfig(min_trade_weight=0.0))
    w0 = first.w_target.copy()
    held = w0[w0.abs() > 0.004].index
    forced, tiny = held[0], w0[w0 == 0].index[0]
    w0[forced] = np.sign(w0[forced]) * 0.0004  # a forced exit below the threshold
    w0[tiny] = 0.0002  # a held crumb with no reason to stay
    cfg = PortfolioConfig(min_trade_weight=thr)
    alpha = gctx["alpha"].copy()
    alpha[tiny] = -abs(alpha).max()  # the optimizer wants it gone (or short)
    opt = SinglePeriodOptimizer(cfg, gctx["cost"])
    res = opt.solve(
        alpha, gctx["risk"], w0, NAV, gctx["snap"], neutral_styles=STYLES, force_to_zero={forced: 0.0}
    )
    assert res.ok
    assert res.w_target[forced] == 0.0  # forced exit done although |x| = 4 bp < threshold
    assert res.w_target[tiny] <= 0.0  # closed (or flipped), not left as a crumb
    x = res.x[res.x.abs() > 0]
    small = x[x.abs() < thr].index
    assert all(res.w_target[t] == 0.0 or t == forced for t in small)
    assert res.n_small_trades_dropped >= 0 and res.max_constraint_violation >= 0.0
    unfiltered = SinglePeriodOptimizer(PortfolioConfig(min_trade_weight=0.0), gctx["cost"]).solve(
        alpha, gctx["risk"], w0, NAV, gctx["snap"], neutral_styles=STYLES, force_to_zero={forced: 0.0}
    )
    assert res.turnover <= unfiltered.turnover + 1e-12


# ============================================================================ rebalance schedule
def _sim_cfg(**pkw) -> SystemConfig:
    return SystemConfig().with_overrides(
        {"portfolio.risk_aversion": 3.0, **{f"portfolio.{k}": v for k, v in pkw.items()}}
    )


@pytest.mark.parametrize("k", [1, 3, 5])
def test_rebalance_schedule_is_exact(md, risk_model, alpha_model, k):
    """Solves happen on dates i % k == 0 and only there; the book drifts in between.
    The config default drives the simulator; the run kwarg overrides it."""
    cfg = _sim_cfg(rebalance_every_days=k)
    cm = SimpleCostModel(cfg.cost)
    dates = md.dates[-23:]
    res = DailyBarSimulator(cfg, cm).run(
        md, dates, alpha_model.as_of, risk_model.as_of, SinglePeriodOptimizer(cfg.portfolio, cm), nav0=1e6
    )
    solved = [e["date"] for e in res.diagnostics["solve_log"]]
    assert solved == list(dates[:-1][::k])
    assert res.diagnostics["rebalance_every"] == k and res.diagnostics["n_rebalance_solves"] == len(solved)
    off = ~res.turnover.index.isin(solved)
    assert (res.turnover[off] == 0).all() and (res.turnover[~off] > 0).all()
    # drift: on a hold day the post-trade weights are yesterday's, moved by the returns
    if k > 1:
        i = 1
        r = md.returns.loc[dates[i]].fillna(0.0)
        nav_ratio = res.nav.iloc[i - 1] / res.nav.iloc[i]
        np.testing.assert_allclose(
            res.weights.iloc[i].to_numpy(),
            (res.weights.iloc[i - 1] * (1 + r) * nav_ratio).to_numpy(),
            atol=1e-12,
        )
    # kwarg overrides the config
    res1 = DailyBarSimulator(cfg, cm).run(
        md,
        dates[:8],
        alpha_model.as_of,
        risk_model.as_of,
        SinglePeriodOptimizer(cfg.portfolio, cm),
        nav0=1e6,
        rebalance_every=2,
    )
    assert [e["date"] for e in res1.diagnostics["solve_log"]] == list(dates[:7][::2])


def test_forced_exits_run_on_hold_days(md, risk_model, alpha_model):
    """A held name that leaves the universe on a non-rebalance day is unwound on its
    linear schedule every day, without waiting for the next solve; nothing else trades."""
    k, exit_days = 10, 4
    cfg = _sim_cfg(rebalance_every_days=k, exit_ineligible_days=exit_days, min_trade_weight=0.0)
    cm = SimpleCostModel(cfg.cost)
    dates = md.dates[-16:]
    base = DailyBarSimulator(cfg, cm).run(
        md, dates, alpha_model.as_of, risk_model.as_of, SinglePeriodOptimizer(cfg.portfolio, cm), nav0=1e6
    )
    name = base.weights.iloc[0].abs().idxmax()
    lag = cfg.execution.signal_lag_days
    drop_i = 3  # execution-date index on which the name is first seen as ineligible
    elig = pd.DataFrame(True, index=md.dates, columns=md.tickers)
    elig.loc[md.dates[md.dates.get_loc(dates[drop_i]) - lag] :, name] = False
    md2 = dataclasses.replace(md, eligible=elig)
    res = DailyBarSimulator(cfg, cm).run(
        md2, dates, alpha_model.as_of, risk_model.as_of, SinglePeriodOptimizer(cfg.portfolio, cm), nav0=1e6
    )
    assert [e["date"] for e in res.diagnostics["solve_log"]] == [dates[0], dates[10]]
    w = res.weights[name].abs()
    w_start = abs(res.weights[name].iloc[drop_i - 1] * (1 + md.returns.loc[dates[drop_i], name]))
    for j in range(exit_days):
        i = drop_i + j
        cap = float(res.diagnostics["forced_exits"][0]["weight"]).__abs__() * (1 - (j + 1) / exit_days)
        assert w.iloc[i] <= cap + 1e-12
        assert res.trades[name].iloc[i] != 0.0
        others = res.trades.iloc[i].drop(name)
        assert (others == 0).all()  # hold day: only the forced name trades
    assert w.iloc[drop_i + exit_days - 1] == 0.0 and w_start > 0
    assert res.diagnostics["forced_exits"][0]["ticker"] == name
    # trading costs are charged on the forced trades of a hold day
    assert res.costs[["spread", "impact", "fees"]].iloc[drop_i].sum() > 0


# ============================================================================ diversification
def test_diversification_summary_hand_computed():
    dates = pd.bdate_range("2024-01-01", periods=4)
    w = pd.DataFrame(
        [[0.5, -0.5, 0.0, 0.0], [0.25, 0.25, -0.25, -0.25], [0.0, 0.0, 0.0, 0.0]],
        index=dates[:3],
        columns=list("ABCD"),
    )
    res = BacktestResult(
        nav=pd.Series(1.0, index=dates),
        returns=pd.DataFrame({"gross": 0.0, "net": 0.0}, index=dates[1:]),
        weights=w,
        trades=w * 0.0,
        costs=pd.DataFrame(0.0, index=dates[:3], columns=["spread", "impact", "fees", "borrow", "total"]),
        exante_var=pd.Series([4e-4 / 252, 1e-4 / 252, 0.0], index=dates[:3]),
        exante_specific_var=pd.Series([1e-4 / 252, 0.5e-4 / 252, 0.0], index=dates[:3]),
    )
    d = res.diversification()
    # day 0: 2 equal names, day 1: 4 equal names; the empty day 2 is not an invested day
    assert d["effective_n_names"] == pytest.approx((2 + 4) / 2)
    assert d["specific_var_share"] == pytest.approx((0.25 + 0.5) / 2)
    # gross 1 on both days: vol at gross 2 is twice the ex-ante vol (2 % and 1 %)
    assert d["exante_vol_at_gross2"] == pytest.approx((0.04 + 0.02) / 2)
    s = res.summary()
    for key in ("effective_n_names", "specific_var_share", "exante_vol_at_gross2"):
        assert s[key] == pytest.approx(d[key])
    empty = dataclasses.replace(res, weights=w * 0.0)
    assert all(np.isnan(v) for v in empty.diversification().values())


def test_golden_run_reports_diversification(golden_pipeline_run):
    _, res = golden_pipeline_run
    s = res.summary
    assert 10 < s["effective_n_names"] <= 80
    assert 0.0 < s["specific_var_share"] < 1.0
    assert 0.01 < s["exante_vol_at_gross2"] < 0.5
    from qts.pipeline import format_result

    card = format_result(res)
    assert "diversification   effective names" in card and "solve every 5d" in card


# ---------------------------------------------------------------------------- default config vs the rank book
@pytest.fixture(scope="module")
def big_pipeline():
    """480 investable names with a healthy planted signal: enough breadth for the
    optimal book to want far more than gross 2 at the base gamma (as on real data)."""
    from qts.pipeline import ResearchPipeline

    md = generate(SyntheticSpec(n_stocks=600, n_days=800, seed=11, ic_true=0.08))
    cfg = SystemConfig().with_overrides({"universe.top_n_by_mcap": 480})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pipe = ResearchPipeline(cfg, use_ml=False, nav0=5e6).fit(md)
    return md, pipe


def _book_stats(w: pd.Series, risk) -> dict[str, float]:
    w = w[w != 0]
    gross = float(w.abs().sum())
    p = w.abs() / gross
    return {
        "eff_n": 1.0 / float((p**2).sum()),
        "gross": gross,
        "vol_at_gross2": float(np.sqrt(risk.portfolio_variance(w) * 252)) * 2.0 / gross,
        "specific_vol_at_gross2": float(np.sqrt(risk.specific_variance(w) * 252)) * 2.0 / gross,
    }


def _run_tail(md, pipe, overrides: dict, days: int = 30):
    from qts.tcost import TransactionCostModel

    cfg = pipe.cfg
    try:
        pipe.cfg = cfg.with_overrides(overrides)
        pipe.cost = TransactionCostModel(pipe.cfg.cost, pipe.cfg.execution)
        pipe.optimizer = SinglePeriodOptimizer(
            pipe.cfg.portfolio,
            pipe.cost,
            horizon_days=pipe.cfg.alpha.horizon_days,
            exec_cfg=pipe.cfg.execution,
        )
        pipe.rebalance_every = pipe.cfg.portfolio.rebalance_every_days
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return pipe.run(md, max_days=days)
    finally:
        pipe.cfg = cfg


def test_default_config_does_not_concentrate_against_the_rank_weighted_neutral_book(big_pipeline):
    """The acceptance check of the construction fix: on the synthetic market the
    default config's book is, at gross 2, about as diversified as the *rank-weighted
    neutral book* (weights proportional to the demeaned rank of the traded score --
    no optimizer, every name held): at least half its effective number of names, at
    most 1.5x its specific vol and no more total vol.  (Its specific *share* of
    variance is higher than the rank book's, because the optimizer hedges the factor
    risk the rank book carries -- which is why the comparison is on specific vol at
    gross 2, not on the share.)  The same run with the gross limit left as a plain
    constraint at the base gamma (``leverage_cap_raises_gamma=False``) fails the
    check: that is the concentration the default removes."""
    md, pipe = big_pipeline
    res = _run_tail(md, pipe, {})
    off = _run_tail(md, pipe, {"portfolio.leverage_cap_raises_gamma": False})
    d = res.backtest.weights.index[-1]
    lag = pipe.signal_lag_days()
    sig = pipe.md_u.dates[pipe.md_u.dates.get_loc(d) - lag]
    risk = pipe.risk.as_of(sig)
    score = pipe.alpha.as_of(sig).score.reindex(risk.tickers).dropna()
    rank_book = _book_stats(score.rank() - score.rank().mean(), risk)
    book = _book_stats(res.backtest.weights.loc[d], risk)
    conc = _book_stats(off.backtest.weights.loc[d], risk)
    assert rank_book["eff_n"] > 300 and book["gross"] > 1.8  # a full book against a broad reference

    def passes(b: dict[str, float]) -> bool:
        return (
            b["eff_n"] >= 0.5 * rank_book["eff_n"]
            and b["specific_vol_at_gross2"] <= 1.5 * rank_book["specific_vol_at_gross2"]
            and b["vol_at_gross2"] <= rank_book["vol_at_gross2"]
        )

    assert passes(book), (book, rank_book)
    assert not passes(conc), (conc, rank_book)
    assert book["eff_n"] > 1.3 * conc["eff_n"]
    assert book["specific_vol_at_gross2"] < 0.8 * conc["specific_vol_at_gross2"]
    # the summary carries the same picture, and the sizing never went below the base gamma
    s, s_off = res.summary, off.summary
    assert s["effective_n_names"] > 1.2 * s_off["effective_n_names"]
    assert s["exante_vol_at_gross2"] < 0.8 * s_off["exante_vol_at_gross2"]
    assert 0.0 < s["specific_var_share"] < 1.0
    gammas = [e["risk_aversion"] for e in res.backtest.diagnostics["solve_log"]]
    assert min(gammas) >= pipe.cfg.portfolio.risk_aversion
    assert set(res.backtest.vol_target_outcomes()) <= {"below_target", "target", "leverage"}
    assert res.backtest.vol_target_outcomes().get("leverage", 0.0) > 0.5
    assert res.summary["n_failed_solves"] == 0
