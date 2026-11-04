"""Tests for the portfolio construction layer (N=120 synthetic universe)."""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from qts.core.config import PortfolioConfig
from qts.portfolio import (
    SimpleCostModel,
    SinglePeriodOptimizer,
    brinson_like_attribution,
    combine_in_forecast_space,
    effective_holding_days,
    factor_exposure_report,
    garleanu_pedersen_shrink,
    gp_sleeve_weights,
    grinold_alpha,
    portfolio_vol_scalar,
    realized_vs_exante_risk,
    shrink_ic,
    snapshot_from_market,
    transfer_coefficient,
)

TOL = 1e-5


@pytest.fixture(scope="module")
def ctx(md, risk_model, alpha_model, test_date):
    r = risk_model.as_of(test_date)
    a = alpha_model.as_of(test_date)
    snap = snapshot_from_market(md, test_date)
    w0 = pd.Series(0.0, index=md.tickers)
    return dict(risk=r, alpha=a.expected_return, snap=snap, w0=w0, date=test_date)


def ls_cfg(**kw) -> PortfolioConfig:
    base = dict(book="long_short", risk_aversion=2.0, target_vol_annual=None)
    base.update(kw)
    return PortfolioConfig(**base)


# ----------------------------------------------------------------- optimizer
def test_zero_alpha_no_cost_zero_trade(ctx):
    opt = SinglePeriodOptimizer(ls_cfg(), SimpleCostModel(zero=True))
    res = opt.solve(ctx["alpha"] * 0.0, ctx["risk"], ctx["w0"], 1e6, ctx["snap"])
    assert res.status in ("optimal", "optimal_inaccurate")
    assert res.turnover < 1e-5


def test_resolve_at_optimum_is_fixed_point(ctx):
    # no ADV cap so the first solve reaches the unconstrained-by-liquidity optimum
    opt = SinglePeriodOptimizer(ls_cfg(max_adv_participation=1e3), SimpleCostModel(zero=True))
    r1 = opt.solve(ctx["alpha"] * 5, ctx["risk"], ctx["w0"], 1e6, ctx["snap"])
    assert r1.turnover > 0.5
    r2 = opt.solve(ctx["alpha"] * 5, ctx["risk"], r1.w_target, 1e6, ctx["snap"])
    assert r2.turnover < 5e-3 * r1.turnover + 1e-4


def test_long_short_constraints_and_speed(ctx):
    cfg = ls_cfg()
    opt = SinglePeriodOptimizer(cfg, SimpleCostModel())
    res = opt.solve(ctx["alpha"] * 5, ctx["risk"], ctx["w0"], 1e6, ctx["snap"])
    w = res.w_target
    assert res.status == "optimal"
    assert res.solve_time < 2.0
    assert w.abs().sum() <= cfg.gross_leverage_max + TOL
    assert abs(w.sum() - cfg.net_exposure_target) <= cfg.net_exposure_band + TOL
    assert w.abs().max() <= cfg.max_position_weight + TOL
    B = ctx["risk"].exposures.reindex(w.index).fillna(0.0)
    expo = B.T @ w
    assert abs(expo["beta"]) <= opt.beta_band + TOL
    ind = [c for c in B.columns if c.startswith("IND")]
    assert expo[ind].abs().max() <= cfg.industry_band + TOL
    # alpha actually used: gross book deployed and positive expected alpha
    assert w.abs().sum() > 0.5
    assert res.expected_alpha > 0
    assert 0 < res.transfer_coefficient <= 1
    assert res.expected_risk > 0 and res.expected_cost > 0
    assert isinstance(res.binding_constraints, dict)


def test_participation_bounds(ctx):
    cfg = ls_cfg(max_adv_participation=0.01)
    opt = SinglePeriodOptimizer(cfg, SimpleCostModel())
    nav = 5e7
    res = opt.solve(ctx["alpha"] * 5, ctx["risk"], ctx["w0"], nav, ctx["snap"])
    assert res.status == "optimal"
    cap = cfg.max_adv_participation * ctx["snap"]["adv_usd"].reindex(res.x.index) / nav
    assert ((res.x.abs() - cap) <= TOL).all()
    assert "participation" in res.binding_constraints


def test_trade_aversion_monotone_turnover(ctx, risk_model, alpha_model, md):
    cm = SimpleCostModel()
    start = SinglePeriodOptimizer(ls_cfg(), cm).solve(ctx["alpha"] * 5, ctx["risk"], ctx["w0"], 1e6, ctx["snap"])
    d2 = md.dates[420]
    r2, a2 = risk_model.as_of(d2), alpha_model.as_of(d2).expected_return * 5
    snap2 = snapshot_from_market(md, d2)
    tos = []
    for ta in (0.25, 1.0, 4.0, 16.0):
        res = SinglePeriodOptimizer(ls_cfg(trade_aversion=ta), cm).solve(a2, r2, start.w_target, 1e6, snap2)
        assert res.status in ("optimal", "optimal_inaccurate")
        tos.append(res.turnover)
    assert all(tos[i + 1] <= tos[i] + 1e-6 for i in range(len(tos) - 1)), tos
    assert tos[-1] < 0.5 * tos[0]


def test_turnover_penalty_reduces_trading(ctx):
    cm = SimpleCostModel(zero=True)
    base = SinglePeriodOptimizer(ls_cfg(), cm).solve(ctx["alpha"] * 5, ctx["risk"], ctx["w0"], 1e6, ctx["snap"])
    pen = SinglePeriodOptimizer(ls_cfg(turnover_penalty=0.002), cm).solve(ctx["alpha"] * 5, ctx["risk"], ctx["w0"], 1e6, ctx["snap"])
    assert pen.turnover < base.turnover


def test_long_only_and_130_30(ctx):
    lo = PortfolioConfig(book="long_only", max_position_weight=0.05, beta_neutral=False, industry_band=0.0,
                         risk_aversion=2.0, target_vol_annual=None)
    res = SinglePeriodOptimizer(lo, SimpleCostModel()).solve(ctx["alpha"] * 5, ctx["risk"], ctx["w0"], 1e6, ctx["snap"])
    assert res.status in ("optimal", "optimal_inaccurate")
    assert (res.w_target >= -TOL).all()
    assert abs(res.w_target.sum() - 1) < 1e-4
    assert res.w_target.max() <= 0.05 + TOL

    c = PortfolioConfig(book="130_30", max_position_weight=0.05, beta_neutral=False, industry_band=0.0,
                        risk_aversion=2.0, target_vol_annual=None)
    res = SinglePeriodOptimizer(c, SimpleCostModel()).solve(ctx["alpha"] * 5, ctx["risk"], ctx["w0"], 1e6, ctx["snap"])
    assert res.status in ("optimal", "optimal_inaccurate")
    w = res.w_target
    assert w.clip(lower=0).sum() <= 1.3 + 1e-4
    assert (-w.clip(upper=0)).sum() <= 0.3 + 1e-4
    assert abs(w.sum() - 1) <= c.net_exposure_band + 1e-4


def test_no_trade_and_style_band(ctx):
    nt = list(ctx["alpha"].abs().sort_values().index[-5:])  # the five strongest names
    cfg = ls_cfg(style_band_sd=0.02)
    res = SinglePeriodOptimizer(cfg, SimpleCostModel()).solve(
        ctx["alpha"] * 5, ctx["risk"], ctx["w0"], 1e6, ctx["snap"], no_trade=nt, neutral_styles=["size"])
    assert res.status in ("optimal", "optimal_inaccurate")
    assert (res.w_target[nt].abs() < 1e-7).all()
    B = ctx["risk"].exposures.reindex(res.w_target.index).fillna(0.0)
    assert abs(B["size"] @ res.w_target) <= cfg.style_band_sd + TOL


def test_vol_target_scales_down(ctx):
    hi = SinglePeriodOptimizer(ls_cfg(), SimpleCostModel()).solve(ctx["alpha"] * 5, ctx["risk"], ctx["w0"], 1e6, ctx["snap"])
    target = 0.5 * hi.expected_risk
    lo = SinglePeriodOptimizer(ls_cfg(target_vol_annual=target), SimpleCostModel()).solve(
        ctx["alpha"] * 5, ctx["risk"], ctx["w0"], 1e6, ctx["snap"])
    assert lo.vol_scalar == pytest.approx(0.5, rel=1e-3)
    assert lo.expected_risk == pytest.approx(target, rel=1e-3)
    assert lo.w_target.abs().sum() == pytest.approx(0.5 * hi.w_target.abs().sum(), rel=1e-3)


def test_infeasible_returns_current(ctx):
    cfg = PortfolioConfig(book="long_only", max_position_weight=0.001, beta_neutral=False, industry_band=0.0,
                          target_vol_annual=None)
    res = SinglePeriodOptimizer(cfg, SimpleCostModel()).solve(ctx["alpha"], ctx["risk"], ctx["w0"], 1e6, ctx["snap"])
    assert res.status not in ("optimal", "optimal_inaccurate")
    assert res.turnover == 0.0
    pd.testing.assert_series_equal(res.w_target, ctx["w0"].reindex(res.w_target.index), check_names=False)


def test_beta_neutral_skipped_without_beta_factor(ctx):
    r = ctx["risk"]
    r2 = dataclasses.replace(r, exposures=r.exposures.drop(columns=["beta"]),
                             factor_cov=r.factor_cov.drop(index="beta", columns="beta"),
                             factor_returns=r.factor_returns.drop(columns=["beta"]))
    res = SinglePeriodOptimizer(ls_cfg(), SimpleCostModel()).solve(ctx["alpha"] * 5, r2, ctx["w0"], 1e6, ctx["snap"])
    assert res.status in ("optimal", "optimal_inaccurate")
    assert any("beta_neutral" in n for n in res.notes)


def test_alignment_term_increases_predicted_risk(ctx):
    r = ctx["risk"]
    a = ctx["alpha"] * 5
    y = a / np.linalg.norm(a.values)
    r2 = dataclasses.replace(r, alignment_vector=y, alignment_var=float(r.specific_var.median()) * 50)
    o = SinglePeriodOptimizer(ls_cfg(), SimpleCostModel(zero=True))
    base = o.solve(a, r, ctx["w0"], 1e6, ctx["snap"])
    al = o.solve(a, r2, ctx["w0"], 1e6, ctx["snap"])
    assert al.status in ("optimal", "optimal_inaccurate")
    assert r2.portfolio_variance(base.w_target) > r.portfolio_variance(base.w_target)
    assert al.w_target.abs().sum() < base.w_target.abs().sum()


# ------------------------------------------------------------- alpha scaling
def test_grinold_shapes_and_missing(ctx):
    score = ctx["alpha"].copy() * 0 + np.random.default_rng(0).standard_normal(len(ctx["alpha"]))
    score.iloc[:3] = np.nan
    vol = np.sqrt(ctx["risk"].specific_var * 20)
    a = grinold_alpha(score, vol, ic=0.05)
    assert a.index.equals(score.index)
    assert (a.iloc[:3] == 0).all()
    assert a.notna().all()
    # alpha scale: IC * vol * z
    z = (score - score.mean()) / score.std(ddof=0)
    pd.testing.assert_series_equal(a.iloc[3:], (0.05 * vol * z).iloc[3:], check_names=False, atol=1e-12)


def test_gp_weights():
    w = gp_sleeve_weights({"fast": 2.0, "slow": 200.0}, cost_implied_holding_days=50.0)
    assert w.sum() == pytest.approx(1.0)
    assert w["fast"] < w["slow"]
    # fast signal gets shrunk more as holding period lengthens
    w2 = gp_sleeve_weights({"fast": 2.0, "slow": 200.0}, cost_implied_holding_days=200.0)
    assert w2["fast"] < w["fast"]
    idx = pd.Index(["A", "B", "C"])
    al = {"fast": pd.Series([1.0, 0.0, -1.0], idx), "slow": pd.Series([0.0, 1.0, -1.0], idx)}
    comb = garleanu_pedersen_shrink(al, {"fast": 2.0, "slow": 200.0}, 50.0)
    assert comb.index.equals(idx)
    expected = w["fast"] * al["fast"] + w["slow"] * al["slow"]
    pd.testing.assert_series_equal(comb, expected, check_names=False)


def test_combine_and_shrink_ic():
    s = {"a": pd.Series({"X": 1.0, "Y": 2.0}), "b": pd.Series({"Y": 1.0, "Z": -1.0})}
    c = combine_in_forecast_space(s, {"a": 0.5, "b": 0.5})
    assert c.to_dict() == {"X": 0.5, "Y": 1.5, "Z": -0.5}
    assert shrink_ic(0.06) == pytest.approx(0.03)
    assert shrink_ic(0.5) == 0.1
    assert shrink_ic(-0.2) == 0.0
    assert shrink_ic(float("nan")) == 0.0


# --------------------------------------------------------------- diagnostics
def test_transfer_coefficient_bounds(ctx):
    res = SinglePeriodOptimizer(ls_cfg(), SimpleCostModel()).solve(ctx["alpha"] * 5, ctx["risk"], ctx["w0"], 1e6, ctx["snap"])
    tc_exact = transfer_coefficient(ctx["alpha"], res.w_target, risk=ctx["risk"])
    tc_cov = transfer_coefficient(ctx["alpha"], res.w_target, cov=ctx["risk"].covariance())
    tc_approx = transfer_coefficient(ctx["alpha"], res.w_target, risk=ctx["risk"], exact_max_n=10)
    assert 0 < tc_exact <= 1 and 0 < tc_approx <= 1
    assert tc_exact == pytest.approx(tc_cov, rel=1e-6)
    # unconstrained Markowitz has TC == 1
    S = ctx["risk"].covariance()
    w_star = pd.Series(np.linalg.solve(S.values, ctx["alpha"].reindex(S.index).values), index=S.index)
    assert transfer_coefficient(ctx["alpha"], w_star, cov=S) == pytest.approx(1.0, abs=1e-6)


def test_effective_holding_days_inverse_to_turnover():
    to1 = pd.Series(0.04, index=range(100))
    to2 = pd.Series(0.08, index=range(100))
    assert effective_holding_days(to1) == pytest.approx(50.0)
    assert effective_holding_days(to2) == pytest.approx(25.0)
    assert effective_holding_days(to1, gross_series=pd.Series(2.0, index=range(100))) == pytest.approx(100.0)


def test_vol_scalar():
    assert portfolio_vol_scalar(0.20, 0.10) == pytest.approx(0.5)
    assert portfolio_vol_scalar(0.05, 0.10) == 1.0
    assert portfolio_vol_scalar(0.05, 0.10, max_leverage_scalar=1.5) == 1.5
    assert portfolio_vol_scalar(0.2, None) == 1.0


def test_exposure_report_and_realized_risk(ctx):
    res = SinglePeriodOptimizer(ls_cfg(), SimpleCostModel()).solve(ctx["alpha"] * 5, ctx["risk"], ctx["w0"], 1e6, ctx["snap"])
    rep = factor_exposure_report(res.w_target, ctx["risk"])
    assert "__total__" in rep.index and "beta" in rep.index
    assert rep.loc["__total__", "var_contribution"] == pytest.approx(ctx["risk"].portfolio_variance(res.w_target), rel=1e-9)
    assert rep["pct_of_var"].drop("__total__").sum() == pytest.approx(1.0, rel=1e-9)
    rng = np.random.default_rng(1)
    v = pd.Series(1e-4, index=range(500))
    r = pd.Series(rng.standard_normal(500) * 0.01, index=range(500))
    out = realized_vs_exante_risk(r, v)
    assert out["ratio"] == pytest.approx(1.0, abs=0.1) and out["bias_stat"] == pytest.approx(1.0, abs=0.1)


def test_brinson_attribution_sums(md, risk_model):
    dates = md.dates[400:410]
    rng = np.random.default_rng(0)
    W = pd.DataFrame(rng.standard_normal((len(dates), len(md.tickers))) * 0.01, index=dates, columns=md.tickers)
    expos = {d: risk_model.exposures(d) for d in dates}
    att = brinson_like_attribution(W, risk_model.factor_returns, expos, total_returns=md.returns)
    assert len(att) == len(dates)
    # with u = r - B f the decomposition is exact: total == w'r
    for d_dec, d_real in zip(dates, md.dates[401:411]):
        assert att.loc[d_real, "total"] == pytest.approx(float(W.loc[d_dec] @ md.returns.loc[d_real]), rel=1e-9)
    fac_cols = [c for c in att.columns if c not in ("specific", "total")]
    assert np.allclose(att[fac_cols].sum(axis=1) + att["specific"], att["total"])
