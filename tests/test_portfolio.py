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
    gp_sleeve_shrink_factors,
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
    return {"risk": r, "alpha": a.expected_return, "snap": snap, "w0": w0, "date": test_date}


def ls_cfg(**kw) -> PortfolioConfig:
    base = {"book": "long_short", "risk_aversion": 2.0, "target_vol_annual": None}
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


def test_long_short_constraints(ctx):
    cfg = ls_cfg()
    opt = SinglePeriodOptimizer(cfg, SimpleCostModel())
    res = opt.solve(ctx["alpha"] * 5, ctx["risk"], ctx["w0"], 1e6, ctx["snap"])
    w = res.w_target
    assert res.status == "optimal"
    # (solve time is reported by the @slow benchmark below, never asserted: a
    # wall-clock bound flakes on a loaded CI runner)
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


@pytest.mark.slow
def test_long_short_solve_time_benchmark(ctx):
    """Benchmark only: reports the median solve time; no timing assertion."""
    import statistics
    import time

    opt = SinglePeriodOptimizer(ls_cfg(), SimpleCostModel())
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        res = opt.solve(ctx["alpha"] * 5, ctx["risk"], ctx["w0"], 1e6, ctx["snap"])
        times.append(time.perf_counter() - t0)
        assert res.status in ("optimal", "optimal_inaccurate")
    print(
        f"\n[benchmark] long-short solve: median {statistics.median(times):.3f}s "
        f"(solver-reported {res.solve_time:.3f}s)"
    )
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
    start = SinglePeriodOptimizer(ls_cfg(), cm).solve(
        ctx["alpha"] * 5, ctx["risk"], ctx["w0"], 1e6, ctx["snap"]
    )
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
    base = SinglePeriodOptimizer(ls_cfg(), cm).solve(
        ctx["alpha"] * 5, ctx["risk"], ctx["w0"], 1e6, ctx["snap"]
    )
    pen = SinglePeriodOptimizer(ls_cfg(turnover_penalty=0.002), cm).solve(
        ctx["alpha"] * 5, ctx["risk"], ctx["w0"], 1e6, ctx["snap"]
    )
    assert pen.turnover < base.turnover


def test_long_only_and_130_30(ctx):
    lo = PortfolioConfig(
        book="long_only",
        max_position_weight=0.05,
        beta_neutral=False,
        industry_band=0.0,
        risk_aversion=2.0,
        target_vol_annual=None,
    )
    res = SinglePeriodOptimizer(lo, SimpleCostModel()).solve(
        ctx["alpha"] * 5, ctx["risk"], ctx["w0"], 1e6, ctx["snap"]
    )
    assert res.status in ("optimal", "optimal_inaccurate")
    assert (res.w_target >= -TOL).all()
    assert abs(res.w_target.sum() - 1) < 1e-4
    assert res.w_target.max() <= 0.05 + TOL

    c = PortfolioConfig(
        book="130_30",
        max_position_weight=0.05,
        beta_neutral=False,
        industry_band=0.0,
        risk_aversion=2.0,
        target_vol_annual=None,
    )
    res = SinglePeriodOptimizer(c, SimpleCostModel()).solve(
        ctx["alpha"] * 5, ctx["risk"], ctx["w0"], 1e6, ctx["snap"]
    )
    assert res.status in ("optimal", "optimal_inaccurate")
    w = res.w_target
    assert w.clip(lower=0).sum() <= 1.3 + 1e-4
    assert (-w.clip(upper=0)).sum() <= 0.3 + 1e-4
    assert abs(w.sum() - 1) <= c.net_exposure_band + 1e-4


def test_no_trade_and_style_band(ctx):
    nt = list(ctx["alpha"].abs().sort_values().index[-5:])  # the five strongest names
    cfg = ls_cfg(style_band_sd=0.02)
    res = SinglePeriodOptimizer(cfg, SimpleCostModel()).solve(
        ctx["alpha"] * 5, ctx["risk"], ctx["w0"], 1e6, ctx["snap"], no_trade=nt, neutral_styles=["size"]
    )
    assert res.status in ("optimal", "optimal_inaccurate")
    assert (res.w_target[nt].abs() < 1e-7).all()
    B = ctx["risk"].exposures.reindex(res.w_target.index).fillna(0.0)
    assert abs(B["size"] @ res.w_target) <= cfg.style_band_sd + TOL


def test_vol_target_scales_down(ctx):
    hi = SinglePeriodOptimizer(ls_cfg(), SimpleCostModel()).solve(
        ctx["alpha"] * 5, ctx["risk"], ctx["w0"], 1e6, ctx["snap"]
    )
    target = 0.5 * hi.expected_risk
    # without the cost term the projection is the exact scaled book
    lo = SinglePeriodOptimizer(
        ls_cfg(target_vol_annual=target, vol_target_mode="projection", vol_target_projection_cost_weight=0.0),
        SimpleCostModel(),
    ).solve(ctx["alpha"] * 5, ctx["risk"], ctx["w0"], 1e6, ctx["snap"])
    assert lo.vol_scalar == pytest.approx(0.5, rel=1e-3)
    assert lo.expected_risk == pytest.approx(target, rel=1e-3)
    assert lo.w_target.abs().sum() == pytest.approx(0.5 * hi.w_target.abs().sum(), rel=1e-3)
    # finding #28: with the trade-cost term the projection trades less (never above target)
    lc = SinglePeriodOptimizer(
        ls_cfg(target_vol_annual=target, vol_target_mode="projection", vol_target_projection_cost_weight=1.0),
        SimpleCostModel(),
    ).solve(ctx["alpha"] * 5, ctx["risk"], ctx["w0"], 1e6, ctx["snap"])
    assert lc.ok and lc.expected_risk <= target * (1 + 1e-3)
    assert lc.turnover < lo.turnover
    assert lc.expected_cost < lo.expected_cost


def test_vol_target_constraint_mode_one_solve(ctx):
    """``vol_target_mode="constraint"``: the SOC cap sits inside the alpha solve."""
    hi = SinglePeriodOptimizer(ls_cfg(), SimpleCostModel()).solve(
        ctx["alpha"] * 5, ctx["risk"], ctx["w0"], 1e6, ctx["snap"]
    )
    target = 0.5 * hi.expected_risk
    opt = SinglePeriodOptimizer(
        ls_cfg(target_vol_annual=target, vol_target_mode="constraint"), SimpleCostModel()
    )
    res = opt.solve(ctx["alpha"] * 5, ctx["risk"], ctx["w0"], 1e6, ctx["snap"])
    assert res.ok and res.vol_scalar == 1.0
    assert res.expected_risk == pytest.approx(target, rel=1e-3)  # cap binds
    assert res.binding_constraints.get("vol_cap") == 1
    assert not any("projection" in n for n in res.notes)
    assert opt.n_builds == 1  # no second (projection) problem
    # never worse on the joint objective than the projected book at the same risk
    proj = SinglePeriodOptimizer(
        ls_cfg(target_vol_annual=target, vol_target_mode="projection"), SimpleCostModel()
    ).solve(ctx["alpha"] * 5, ctx["risk"], ctx["w0"], 1e6, ctx["snap"])
    assert res.expected_alpha - res.expected_cost >= proj.expected_alpha - proj.expected_cost - 1e-6
    with pytest.raises(ValueError, match="vol_target_mode"):
        SinglePeriodOptimizer(ls_cfg(vol_target_mode="bogus"), SimpleCostModel())


def test_infeasible_returns_current(ctx):
    cfg = PortfolioConfig(
        book="long_only",
        max_position_weight=0.001,
        beta_neutral=False,
        industry_band=0.0,
        target_vol_annual=None,
    )
    res = SinglePeriodOptimizer(cfg, SimpleCostModel()).solve(
        ctx["alpha"], ctx["risk"], ctx["w0"], 1e6, ctx["snap"]
    )
    assert res.status not in ("optimal", "optimal_inaccurate")
    assert res.turnover == 0.0
    pd.testing.assert_series_equal(res.w_target, ctx["w0"].reindex(res.w_target.index), check_names=False)


def test_beta_neutral_skipped_without_beta_factor(ctx):
    r = ctx["risk"]
    r2 = dataclasses.replace(
        r,
        exposures=r.exposures.drop(columns=["beta"]),
        factor_cov=r.factor_cov.drop(index="beta", columns="beta"),
        factor_returns=r.factor_returns.drop(columns=["beta"]),
    )
    res = SinglePeriodOptimizer(ls_cfg(), SimpleCostModel()).solve(
        ctx["alpha"] * 5, r2, ctx["w0"], 1e6, ctx["snap"]
    )
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
    comb_mix = garleanu_pedersen_shrink(al, {"fast": 2.0, "slow": 200.0}, 50.0, renormalize=True)
    assert comb_mix.index.equals(idx)
    expected = w["fast"] * al["fast"] + w["slow"] * al["slow"]
    pd.testing.assert_series_equal(comb_mix, expected, check_names=False)
    # default: the alpha *level* is shrunk (GP), not just the mix
    g = gp_sleeve_shrink_factors({"fast": 2.0, "slow": 200.0}, 50.0)
    comb = garleanu_pedersen_shrink(al, {"fast": 2.0, "slow": 200.0}, 50.0)
    expected_lvl = 0.5 * g["fast"] * al["fast"] + 0.5 * g["slow"] * al["slow"]
    pd.testing.assert_series_equal(comb, expected_lvl, check_names=False)
    assert comb.abs().sum() < comb_mix.abs().sum()
    one = garleanu_pedersen_shrink({"slow": al["slow"]}, {"slow": 200.0}, 200.0)
    pd.testing.assert_series_equal(one, 0.5 * al["slow"], check_names=False)  # hl == H -> halved


def test_combine_and_shrink_ic():
    s = {"a": pd.Series({"X": 1.0, "Y": 2.0}), "b": pd.Series({"Y": 1.0, "Z": -1.0})}
    c = combine_in_forecast_space(s, {"a": 0.5, "b": 0.5})
    assert c.to_dict() == {"X": 0.5, "Y": 1.5, "Z": -0.5}
    assert shrink_ic(0.06) == pytest.approx(0.03)
    assert shrink_ic(0.5) == 0.1
    assert shrink_ic(-0.2) == 0.0
    assert shrink_ic(float("nan")) == 0.0
    # D5: ``keep`` is the fraction kept (same as AlphaConfig.ic_shrink), not removed
    assert shrink_ic(0.06, keep=0.25) == pytest.approx(0.015)
    assert shrink_ic(0.06, keep=1.0) == pytest.approx(0.06)
    assert shrink_ic(0.06, keep=0.0) == 0.0
    from qts.core.config import AlphaConfig

    assert shrink_ic(0.04, keep=AlphaConfig().ic_shrink) == pytest.approx(AlphaConfig().ic_shrink * 0.04)
    with pytest.raises(ValueError):
        shrink_ic(0.06, keep=1.5)


# --------------------------------------------------------------- diagnostics
def test_transfer_coefficient_bounds(ctx):
    res = SinglePeriodOptimizer(ls_cfg(), SimpleCostModel()).solve(
        ctx["alpha"] * 5, ctx["risk"], ctx["w0"], 1e6, ctx["snap"]
    )
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
    res = SinglePeriodOptimizer(ls_cfg(), SimpleCostModel()).solve(
        ctx["alpha"] * 5, ctx["risk"], ctx["w0"], 1e6, ctx["snap"]
    )
    rep = factor_exposure_report(res.w_target, ctx["risk"])
    assert "__total__" in rep.index and "beta" in rep.index
    assert rep.loc["__total__", "var_contribution"] == pytest.approx(
        ctx["risk"].portfolio_variance(res.w_target), rel=1e-9
    )
    assert rep["pct_of_var"].drop("__total__").sum() == pytest.approx(1.0, rel=1e-9)
    rng = np.random.default_rng(1)
    v = pd.Series(1e-4, index=range(500))
    r = pd.Series(rng.standard_normal(500) * 0.01, index=range(500))
    out = realized_vs_exante_risk(r, v)
    assert out["ratio"] == pytest.approx(1.0, abs=0.1) and out["bias_stat"] == pytest.approx(1.0, abs=0.1)


def test_brinson_attribution_sums(md, risk_model):
    dates = md.dates[400:410]
    rng = np.random.default_rng(0)
    W = pd.DataFrame(
        rng.standard_normal((len(dates), len(md.tickers))) * 0.01, index=dates, columns=md.tickers
    )
    expos = {d: risk_model.exposures(d) for d in dates}
    att = brinson_like_attribution(W, risk_model.factor_returns, expos, total_returns=md.returns)
    assert len(att) == len(dates)
    # with u = r - B f the decomposition is exact: total == w'r
    for d_dec, d_real in zip(dates, md.dates[401:411], strict=True):
        assert att.loc[d_real, "total"] == pytest.approx(
            float(W.loc[d_dec] @ md.returns.loc[d_real]), rel=1e-9
        )
    fac_cols = [c for c in att.columns if c not in ("specific", "total")]
    assert np.allclose(att[fac_cols].sum(axis=1) + att["specific"], att["total"])


# ------------------------------------------------------- real risk model regressions
FF48_LIKE = [
    "Agric",
    "Banks",
    "Drugs",
    "Softw",
    "Oil",
    "Rtail",
    "Autos",
    "Chems",
    "Insur",
    "Telcm",
    "Steel",
    "Other",
]
BARRA_TOL = 2e-4  # solver tolerance on linear bands (Clarabel default ~1e-8 rel; projection adds a little)


def _rename_industries(md):
    labels = sorted(md.industry.unique())
    mapping = {old: FF48_LIKE[i % len(FF48_LIKE)] for i, old in enumerate(labels)}
    return dataclasses.replace(md, industry=md.industry.map(mapping))


@pytest.fixture(scope="module")
def barra_ctx(md):
    """Real ``BarraStyleRiskModel`` on the synthetic market with Fama-French-like
    industry labels (no ``IND`` prefix) and an alpha tilted toward high beta."""
    from qts.core.config import RiskConfig
    from qts.risk import BarraStyleRiskModel

    md2 = _rename_industries(md)
    model = BarraStyleRiskModel(RiskConfig()).fit(md2)
    date = md2.dates[-5]
    r = model.as_of(date)
    rng = np.random.default_rng(5)
    B = r.exposures
    alpha = 0.01 * (B["beta"] + 0.5 * pd.Series(rng.standard_normal(len(B)), index=B.index))
    snap = snapshot_from_market(md2, date)
    return {
        "md": md2,
        "model": model,
        "risk": r,
        "alpha": alpha,
        "snap": snap,
        "w0": pd.Series(0.0, index=md2.tickers),
        "date": date,
    }


def _exposures(w, risk):
    return risk.exposures.reindex(w.index).fillna(0.0).T @ w


def _assert_all_constraints(res, cfg, risk, snap, nav, w0, opt, neutral_styles=(), tol=BARRA_TOL):
    w = res.w_target
    groups = risk.factor_groups
    assert w.abs().sum() <= cfg.gross_leverage_max + tol
    assert abs(w.sum() - cfg.net_exposure_target) <= cfg.net_exposure_band + tol
    assert w.abs().max() <= cfg.max_position_weight + tol
    cap = cfg.max_adv_participation * snap["adv_usd"].reindex(w.index) / nav
    assert ((w - w0.reindex(w.index).fillna(0.0)).abs() <= cap + tol).all()
    expo = _exposures(w, risk)
    if cfg.beta_neutral:
        assert abs(expo["beta"]) <= opt.beta_band + tol, expo["beta"]
    if cfg.industry_band:
        assert expo[groups["industry"]].abs().max() <= cfg.industry_band + tol
    for s in neutral_styles:
        if not (cfg.beta_neutral and s == "beta"):
            assert abs(expo[s]) <= cfg.style_band_sd + tol, (s, expo[s])


def test_factor_groups_drive_beta_and_ff48_industry_bands(barra_ctx):
    r, cfg = barra_ctx["risk"], ls_cfg(beta_neutral=True)
    assert r.factor_groups["style"] and "beta" in r.factor_groups["style"]
    assert set(r.factor_groups["industry"]) == set(FF48_LIKE)
    opt = SinglePeriodOptimizer(cfg, SimpleCostModel(zero=True))
    res = opt.solve(barra_ctx["alpha"], r, barra_ctx["w0"], 1e6, barra_ctx["snap"])
    assert res.status == "optimal" and res.w_target.abs().sum() > 0.5
    expo = _exposures(res.w_target, r)
    # beta neutrality binds the *beta style*, not the unit-exposure market factor
    assert abs(expo["beta"]) <= opt.beta_band + BARRA_TOL, expo["beta"]
    assert "beta_neutral" in res.binding_constraints
    # industry bands work on FF48-like labels (nothing starts with "IND")
    assert expo[FF48_LIKE].abs().max() <= cfg.industry_band + BARRA_TOL
    assert not any("prefix fallback" in n for n in res.notes)
    # the test has teeth: without the constraint the high-beta tilt is large
    free = SinglePeriodOptimizer(
        ls_cfg(beta_neutral=False, industry_band=0.0), SimpleCostModel(zero=True)
    ).solve(barra_ctx["alpha"], r, barra_ctx["w0"], 1e6, barra_ctx["snap"], neutral_styles=[])
    assert abs(_exposures(free.w_target, r)["beta"]) > 3 * opt.beta_band
    assert _exposures(free.w_target, r)[FF48_LIKE].abs().max() > cfg.industry_band


def test_beta_band_from_config_and_unknown_neutral_style_raises(barra_ctx):
    cfg = ls_cfg(beta_band=0.01)
    opt = SinglePeriodOptimizer(cfg, SimpleCostModel(zero=True))
    assert opt.beta_band == 0.01
    res = opt.solve(
        barra_ctx["alpha"],
        barra_ctx["risk"],
        barra_ctx["w0"],
        1e6,
        barra_ctx["snap"],
        neutral_styles=["beta", "size", "liquidity"],
    )
    assert res.status == "optimal"
    expo = _exposures(res.w_target, barra_ctx["risk"])
    assert abs(expo["beta"]) <= 0.01 + BARRA_TOL
    assert expo[["size", "liquidity"]].abs().max() <= cfg.style_band_sd + BARRA_TOL
    with pytest.raises(ValueError, match="neutral_styles"):
        opt.solve(
            barra_ctx["alpha"],
            barra_ctx["risk"],
            barra_ctx["w0"],
            1e6,
            barra_ctx["snap"],
            neutral_styles=["not_a_style"],
        )


def test_prefix_fallback_only_without_factor_groups(ctx, caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="qts.portfolio.optimizer"):
        res = SinglePeriodOptimizer(ls_cfg(), SimpleCostModel(zero=True)).solve(
            ctx["alpha"] * 5, ctx["risk"], ctx["w0"], 1e6, ctx["snap"]
        )
    assert res.status == "optimal"
    assert any("prefix fallback" in n for n in res.notes)
    assert any("factor_groups" in rec.getMessage() for rec in caplog.records)


def test_vol_target_scale_up_is_a_projection(barra_ctx):
    r, snap, w0, a = barra_ctx["risk"], barra_ctx["snap"], barra_ctx["w0"], barra_ctx["alpha"]
    nav = 1e6
    styles = ["size", "liquidity"]
    # a timid book (high risk aversion) well inside every cap
    base_cfg = ls_cfg(risk_aversion=50.0, max_position_weight=0.05, style_band_sd=0.05)
    base = SinglePeriodOptimizer(base_cfg, SimpleCostModel(zero=True)).solve(
        a, r, w0, nav, snap, neutral_styles=styles
    )
    assert base.status == "optimal" and base.w_target.abs().sum() < 1.0
    target = min(
        2.0 * base.expected_risk,
        0.5 * base.expected_risk * base_cfg.gross_leverage_max / base.w_target.abs().sum(),
    )  # reachable without hitting the gross cap
    cfg = dataclasses.replace(base_cfg, target_vol_annual=target, vol_target_scale_up=True)
    opt = SinglePeriodOptimizer(cfg, SimpleCostModel(zero=True))
    up = opt.solve(a, r, w0, nav, snap, neutral_styles=styles)
    assert up.status == "optimal" and up.vol_scalar > 1.2
    assert any("projection" in n for n in up.notes)
    # predicted vol lands on the target (caps not binding) and never above it
    assert up.expected_risk <= target * (1 + 1e-3)
    assert up.expected_risk >= 0.9 * target
    _assert_all_constraints(up, cfg, r, snap, nav, w0, opt, neutral_styles=styles)

    # with binding caps (tight ADV, pre-existing opposite-sign book, style band at
    # the edge): scaling is still feasible and every band holds after it
    rng = np.random.default_rng(2)
    w_prev = -base.w_target * 0.5 + pd.Series(rng.standard_normal(len(w0)), index=w0.index) * 1e-3
    cfg2 = dataclasses.replace(cfg, max_adv_participation=0.01, net_exposure_band=0.02, beta_band=0.02)
    opt2 = SinglePeriodOptimizer(cfg2, SimpleCostModel(zero=True))
    up2 = opt2.solve(a, r, w_prev, 5e7, snap, neutral_styles=styles)
    assert up2.status == "optimal"
    assert up2.expected_risk <= target * (1 + 1e-3)
    _assert_all_constraints(up2, cfg2, r, snap, 5e7, w_prev, opt2, neutral_styles=styles)


def test_vol_target_scale_down_respects_no_trade_and_participation(barra_ctx):
    r, snap, a = barra_ctx["risk"], barra_ctx["snap"], barra_ctx["alpha"]
    nav = 2e7
    hi = SinglePeriodOptimizer(ls_cfg(max_position_weight=0.05), SimpleCostModel(zero=True)).solve(
        a, r, barra_ctx["w0"], nav, snap
    )
    nt = list(hi.w_target.abs().sort_values().index[-3:])
    w_prev = hi.w_target.copy()
    cfg = ls_cfg(
        max_position_weight=0.05, target_vol_annual=0.5 * hi.expected_risk, max_adv_participation=0.02
    )
    opt = SinglePeriodOptimizer(cfg, SimpleCostModel(zero=True))
    lo = opt.solve(a, r, w_prev, nav, snap, no_trade=nt)
    assert lo.status == "optimal"
    assert lo.expected_risk <= cfg.target_vol_annual * (1 + 1e-3)
    assert (lo.w_target[nt] == w_prev[nt]).all()  # no-trade names untouched by the scaling
    _assert_all_constraints(lo, cfg, r, snap, nav, w_prev, opt)


def test_failed_solve_semantics_and_optin_participation_relax(barra_ctx, caplog):
    import logging

    r, snap, a = barra_ctx["risk"], barra_ctx["snap"], barra_ctx["alpha"]
    # current book violates the per-name cap and the ADV cap is far too small to fix it in one day
    w_prev = barra_ctx["w0"].copy()
    w_prev.iloc[0] = 0.10
    cfg = ls_cfg(max_position_weight=0.02)
    nav = 1e10
    with caplog.at_level(logging.WARNING, logger="qts.portfolio.optimizer"):
        res = SinglePeriodOptimizer(cfg, SimpleCostModel()).solve(a, r, w_prev, nav, snap)
    assert res.status not in ("optimal", "optimal_inaccurate")
    assert not res.ok
    assert np.isnan(res.expected_cost)
    assert res.turnover == 0.0
    pd.testing.assert_series_equal(res.w_target, w_prev.reindex(res.w_target.index), check_names=False)
    assert any("optimizer failed" in rec.getMessage() for rec in caplog.records)
    # opt-in relaxation: solves without the caps and says so
    cfg2 = ls_cfg(max_position_weight=0.02, relax_participation_on_infeasible=True)
    res2 = SinglePeriodOptimizer(cfg2, SimpleCostModel()).solve(a, r, w_prev, nav, snap)
    assert res2.ok
    assert any("relaxed" in n for n in res2.notes)
    assert res2.w_target.iloc[0] <= 0.02 + BARRA_TOL


def test_force_to_zero_and_out_of_model_names(barra_ctx):
    r, snap, a = barra_ctx["risk"], barra_ctx["snap"], barra_ctx["alpha"].copy()
    tk = a.abs().idxmax()
    w_prev = barra_ctx["w0"].copy()
    w_prev[tk] = 0.015
    cfg = ls_cfg()
    opt = SinglePeriodOptimizer(cfg, SimpleCostModel(zero=True))
    free = opt.solve(a, r, w_prev, 1e6, snap)
    assert abs(free.w_target[tk]) > 0.005  # the alpha wants to hold it
    half = opt.solve(a, r, w_prev, 1e6, snap, force_to_zero={tk: 0.0075})
    assert half.status == "optimal" and abs(half.w_target[tk]) <= 0.0075 + BARRA_TOL
    out = opt.solve(a, r, w_prev, 1e6, snap, force_to_zero={tk: 0.0})
    assert out.status == "optimal" and abs(out.w_target[tk]) <= BARRA_TOL
    # a name the risk model does not know: zero exposure, 90th-pct specific var, counted
    a2 = pd.concat([a, pd.Series({"ZZZ_NOT_IN_MODEL": a.abs().max()})])
    res = opt.solve(a2, r, w_prev, 1e6, snap)
    assert res.status == "optimal"
    assert res.n_out_of_model == 1
    assert any("outside the risk model universe" in n for n in res.notes)
    assert free.n_out_of_model == 0


# ------------------------------------------------------- simulator eligibility hook
def test_simulator_eligibility_hook_unwinds_and_blocks(md, risk_model, alpha_model):
    from qts.backtest import DailyBarSimulator
    from qts.core.config import SystemConfig

    cfg = SystemConfig()
    cfg.portfolio.risk_aversion = 3.0
    cfg.portfolio.exit_ineligible_days = 4
    cm = SimpleCostModel(zero=True)
    opt = SinglePeriodOptimizer(cfg.portfolio, cm)
    dates = md.dates[-26:]
    # names the planted alpha wants long on the first day
    a0 = alpha_model.as_of(dates[0]).expected_return
    held_tk = a0.idxmax()
    blocked_tk = a0.sort_values().index[-2]
    cut = dates[8]

    def eligible_fn(t):
        if t < cut:
            return md.tickers.drop(blocked_tk)
        return md.tickers.drop([held_tk, blocked_tk])

    res = DailyBarSimulator(cfg, cm).run(
        md, dates, alpha_model.as_of, risk_model.as_of, opt, nav0=1e6, eligible_fn=eligible_fn
    )
    W = res.weights
    assert (W[blocked_tk].abs() < 1e-9).all()  # never opened
    assert W.loc[dates[7], held_tk] > 0.002  # held while eligible
    w_at_cut = W.loc[dates[7], held_tk]
    post = W.loc[cut:, held_tk]
    assert (post.abs() <= w_at_cut + 1e-9).all()
    assert abs(W.loc[dates[8 + 3], held_tk]) < 1e-9  # flat after exit_ineligible_days
    assert (W.loc[dates[12] :, held_tk].abs() < 1e-9).all()  # never re-opened
    # linear schedule: cap_k = w0 * (1 - k/4); each day stays within it
    for k in range(1, 4):
        assert abs(W.loc[dates[8 + k - 1], held_tk]) <= w_at_cut * (1 - k / 4) + 1e-6
    fe = res.diagnostics["forced_exits"]
    assert any(d["ticker"] == held_tk and d["date"] == cut for d in fe)
    assert res.diagnostics["n_ineligible_name_days"] > 0
    assert res.summary()["n_failed_solves"] == 0

    # same thing through md.eligible (wide bool panel), which takes precedence
    elig = pd.DataFrame(True, index=md.dates, columns=md.tickers)
    elig[blocked_tk] = False
    elig.loc[cut:, held_tk] = False
    md_e = dataclasses.replace(md, eligible=elig)
    res2 = DailyBarSimulator(cfg, cm).run(md_e, dates, alpha_model.as_of, risk_model.as_of, opt, nav0=1e6)
    pd.testing.assert_frame_equal(res2.weights, W, atol=1e-6)


# ------------------------------------------------------- DPP reuse, roles, GP (work plan 1.6, D1, #5)
def test_dpp_reuse_matches_rebuild_across_days(barra_ctx):
    """The cached DPP problem re-solved with new parameter values gives the same
    book as a fresh build (to solver tolerance) on several days; the ticker set
    is unchanged so it is built once.  A changed ticker set triggers a rebuild."""
    md, model, snap0 = barra_ctx["md"], barra_ctx["model"], barra_ctx["snap"]
    from qts.tcost import TransactionCostModel

    cost = TransactionCostModel()
    cfg = ls_cfg(target_vol_annual=0.08, vol_target_projection_cost_weight=1.0)
    styles = ["size", "liquidity"]
    reuse = SinglePeriodOptimizer(cfg, cost, reuse_problem=True)
    fresh = SinglePeriodOptimizer(cfg, cost, reuse_problem=False)
    w_prev = barra_ctx["w0"]
    rng = np.random.default_rng(11)
    for d in md.dates[-8:-2:2]:
        r = model.as_of(d)
        a = 0.01 * pd.Series(rng.standard_normal(len(r.tickers)), index=r.tickers)
        snap = cost.snapshot_from_market(md, d)
        x = reuse.solve(a, r, w_prev, 1e7, snap, neutral_styles=styles)
        y = fresh.solve(a, r, w_prev, 1e7, snap, neutral_styles=styles)
        assert x.ok and y.ok
        assert float((x.w_target - y.w_target).abs().max()) < 2e-5
        assert x.expected_cost == pytest.approx(y.expected_cost, rel=1e-3, abs=1e-8)
        w_prev = x.w_target
    assert reuse.n_reuses >= 2 and fresh.n_reuses == 0
    builds = reuse.n_builds
    r = model.as_of(md.dates[-2])
    sub = r.tickers[:-3]  # different ticker set -> rebuild
    r2 = dataclasses.replace(r, exposures=r.exposures.loc[sub], specific_var=r.specific_var.loc[sub])
    a = 0.01 * pd.Series(rng.standard_normal(len(sub)), index=sub)
    res = reuse.solve(
        a,
        r2,
        pd.Series(0.0, index=sub),
        1e7,
        {k: (v.reindex(sub) if isinstance(v, pd.Series) else v) for k, v in snap0.items()},
        neutral_styles=styles,
    )
    assert res.ok and not res.problem_reused and reuse.n_builds > builds


def test_default_neutral_styles_from_roles(barra_ctx):
    """D1: ``neutral_styles=None`` bands qts.core.roles.RISK_ONLY_STYLES; ``[]`` bands none."""
    from qts.core.roles import RISK_ONLY_STYLES

    r, a = barra_ctx["risk"], barra_ctx["alpha"]
    # tilt the alpha toward size so an un-banded book is clearly size-exposed
    a = a + 0.01 * r.exposures["size"]
    cfg = ls_cfg(style_band_sd=0.05)
    opt = SinglePeriodOptimizer(cfg, SimpleCostModel(zero=True))
    dflt = opt.solve(a, r, barra_ctx["w0"], 1e6, barra_ctx["snap"])
    none = opt.solve(a, r, barra_ctx["w0"], 1e6, barra_ctx["snap"], neutral_styles=[])
    assert dflt.ok and none.ok
    banded = [s for s in RISK_ONLY_STYLES if s in r.factor_groups["style"] and s != "beta"]
    assert banded
    e_d, e_n = _exposures(dflt.w_target, r), _exposures(none.w_target, r)
    assert e_d[banded].abs().max() <= cfg.style_band_sd + BARRA_TOL
    assert abs(e_n["size"]) > cfg.style_band_sd + 0.01  # the test has teeth
    assert "style_band" in dflt.binding_constraints
    # alpha themes are never defaulted into the bands
    off = SinglePeriodOptimizer(
        ls_cfg(style_band_sd=0.05, neutralize_risk_only_styles=False), SimpleCostModel(zero=True)
    ).solve(a, r, barra_ctx["w0"], 1e6, barra_ctx["snap"])
    assert abs(_exposures(off.w_target, r)["size"]) > cfg.style_band_sd + 0.01


def test_gp_shrink_wiring(barra_ctx):
    """``use_gp_shrink``: alpha scaled by 1 / (1 + h / hl) with the cost-implied
    holding period h; a faster signal (shorter half-life) is shrunk more; off
    by default and skipped (with a note) without a half-life."""
    from qts.tcost import TransactionCostModel

    r, a, snap = barra_ctx["risk"], barra_ctx["alpha"], barra_ctx["snap"]
    cost = TransactionCostModel()
    assert PortfolioConfig().use_gp_shrink is False
    base = SinglePeriodOptimizer(ls_cfg(), cost).solve(
        a, r, barra_ctx["w0"], 1e7, snap, alpha_half_life_days=5.0
    )
    assert base.gp_shrink == 1.0
    opt = SinglePeriodOptimizer(ls_cfg(use_gp_shrink=True), cost)
    slow = opt.solve(a, r, barra_ctx["w0"], 1e7, snap, alpha_half_life_days=200.0)
    fast = opt.solve(a, r, barra_ctx["w0"], 1e7, snap, alpha_half_life_days=2.0)
    assert 0 < fast.gp_shrink < slow.gp_shrink < 1.0
    assert any("GP shrink" in n for n in fast.notes)
    assert fast.w_target.abs().sum() < slow.w_target.abs().sum()
    # the factor is exactly gp_sleeve_shrink_factors at the reported holding period
    import re

    h = float(re.search(r"holding ([0-9.]+)d", next(n for n in fast.notes if "GP shrink" in n)).group(1))
    assert fast.gp_shrink == pytest.approx(float(gp_sleeve_shrink_factors({"s": 2.0}, h)["s"]), rel=1e-2)
    none = opt.solve(a, r, barra_ctx["w0"], 1e7, snap)
    assert none.gp_shrink == 1.0 and any("no alpha half-life" in n for n in none.notes)
