"""Second-pass audit (risk / portfolio) fixes in the optimizer:

* finding 1: ``risk_aversion_mode="target_vol"`` -- gamma searched per solve so
  the optimal book's ex-ante vol hits the target (closed-form start);
* finding 2: the ``"projection"`` vol mode measures distance as
  risk-aversion-weighted tracking variance (same units as the cost term);
* finding 3: names outside the risk model are never opened;
* finding 5: the GP alpha shrink survives vol targeting;
* findings 6 / 7: routed cost expression is DPP; ``solve()`` defaults to routed.
"""

from __future__ import annotations

import dataclasses
import inspect
import itertools

import numpy as np
import pandas as pd
import pytest

from qts.core.config import PortfolioConfig, RiskConfig
from qts.portfolio import SimpleCostModel, SinglePeriodOptimizer, snapshot_from_market
from qts.portfolio.optimizer import GAMMA_MAX_SOLVES, GAMMA_SEARCH_VOL_TOL

TOL = 2e-4
NAV = 1e7
STYLES = ["size", "liquidity"]


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
    alpha = 0.05 * sd_h * z  # IC 0.05 Grinold-style alpha over H = 20 days
    r = model.as_of(date, alpha=alpha)  # with the alignment term
    return {
        "md": md,
        "model": model,
        "risk": r,
        "alpha": alpha,
        "snap": snapshot_from_market(md, date),
        "w0": pd.Series(0.0, index=r.tickers),
        "cost": TransactionCostModel(),
        "date": date,
    }


def _vol(res) -> float:
    return float(res.expected_risk)


def _check_bands(res, cfg, risk, opt, nav, w0, snap):
    w = res.w_target
    assert w.abs().sum() <= cfg.gross_leverage_max + TOL
    assert abs(w.sum() - cfg.net_exposure_target) <= cfg.net_exposure_band + TOL
    assert w.abs().max() <= cfg.max_position_weight + TOL
    cap = cfg.max_adv_participation * snap["adv_usd"].reindex(w.index) / nav
    assert ((w - w0.reindex(w.index).fillna(0.0)).abs() <= cap + TOL).all()
    expo = risk.exposures.reindex(w.index).fillna(0.0).T @ w
    assert abs(expo["beta"]) <= opt.beta_band + TOL
    assert expo[risk.factor_groups["industry"]].abs().max() <= cfg.industry_band + TOL
    for s in STYLES:
        assert abs(expo[s]) <= cfg.style_band_sd + TOL


def _fixed(g: float, **kw) -> PortfolioConfig:
    return PortfolioConfig(risk_aversion_mode="fixed", risk_aversion=g, target_vol_annual=None, **kw)


# ----------------------------------------------------------------- closed form
def test_closed_form_gamma_matches_dense_reference(gctx):
    """gamma0 = IR_ann / (2 sigma_target), IR_ann = sqrt(252)/H sqrt(a' Sigma_d^-1 a):
    the Woodbury factor form equals a dense inverse, and the *unconstrained,
    cost-free* optimum w* = Sigma^-1 a / (2 gamma0 H) has exactly the target vol."""
    r, a = gctx["risk"], gctx["alpha"]
    H, target = 20, 0.10
    opt = SinglePeriodOptimizer(PortfolioConfig(), SimpleCostModel(zero=True), horizon_days=H)
    pdat = opt._prepare(a, r, gctx["w0"], NAV, gctx["snap"], STYLES, None, None, None, None, [])
    g0, ir = opt.closed_form_gamma(pdat, H, target)
    S = r.covariance(pdat.tickers, include_alignment=True).to_numpy()
    av = a.reindex(pdat.tickers).to_numpy()
    q = float(av @ np.linalg.solve(S, av))
    assert pdat.sigma_inv_quad(av) == pytest.approx(q, rel=1e-8)
    assert ir == pytest.approx(np.sqrt(252) / H * np.sqrt(q), rel=1e-8)
    assert g0 == pytest.approx(ir / (2 * target), rel=1e-12)
    w_star = np.linalg.solve(S, av) / (2 * g0 * H)
    assert np.sqrt(252 * w_star @ S @ w_star) == pytest.approx(target, rel=1e-8)


def test_vol_is_monotone_in_gamma(gctx):
    """The search relies on sigma(w*(gamma)) being non-increasing in gamma."""
    r, a, snap, cost = gctx["risk"], gctx["alpha"], gctx["snap"], gctx["cost"]
    vols = [
        _vol(SinglePeriodOptimizer(_fixed(g), cost).solve(a, r, gctx["w0"], NAV, snap, neutral_styles=STYLES))
        for g in (0.5, 2.0, 8.0, 32.0)
    ]
    assert all(v1 >= v2 - 1e-6 for v1, v2 in itertools.pairwise(vols)), vols
    assert vols[0] > 1.5 * vols[-1]


# ----------------------------------------------------------------- gamma search
def test_target_vol_hits_reachable_target_within_bands(gctx):
    r, a, snap, cost = gctx["risk"], gctx["alpha"], gctx["snap"], gctx["cost"]
    lo = SinglePeriodOptimizer(_fixed(64.0), cost).solve(a, r, gctx["w0"], NAV, snap, neutral_styles=STYLES)
    hi = SinglePeriodOptimizer(_fixed(1.0), cost).solve(a, r, gctx["w0"], NAV, snap, neutral_styles=STYLES)
    target = float(np.sqrt(_vol(lo) * _vol(hi)))  # strictly inside the reachable range
    cfg = PortfolioConfig(target_vol_annual=target)
    assert cfg.risk_aversion_mode == "target_vol"  # the default
    opt = SinglePeriodOptimizer(cfg, cost)
    res = opt.solve(a, r, gctx["w0"], NAV, snap, neutral_styles=STYLES)
    assert res.ok and res.vol_target_binding == "target"
    assert abs(_vol(res) / target - 1) < GAMMA_SEARCH_VOL_TOL
    assert 1.0 < res.risk_aversion < 64.0
    assert 2 <= res.n_solves <= GAMMA_MAX_SOLVES
    assert res.vol_scalar == 1.0 and not any("projection" in n for n in res.notes)
    assert any("gamma search (target)" in n for n in res.notes)
    _check_bands(res, cfg, r, opt, NAV, gctx["w0"], snap)
    # the returned book is the fixed-gamma optimum at the reported gamma
    same = SinglePeriodOptimizer(_fixed(res.risk_aversion), cost).solve(
        a, r, gctx["w0"], NAV, snap, neutral_styles=STYLES
    )
    assert float((same.w_target - res.w_target).abs().max()) < 1e-5


def test_target_vol_accepts_capped_book(gctx):
    """An unreachable target: the search stops once the caps bind (vol below
    target, gross at its limit or vol insensitive to gamma) and keeps that book."""
    r, a, snap, cost = gctx["risk"], gctx["alpha"], gctx["snap"], gctx["cost"]
    cfg = PortfolioConfig(target_vol_annual=0.40)
    opt = SinglePeriodOptimizer(cfg, cost)
    res = opt.solve(a, r, gctx["w0"], NAV, snap, neutral_styles=STYLES)
    assert res.ok and res.vol_target_binding == "caps"
    assert _vol(res) < 0.40 and res.n_solves <= GAMMA_MAX_SOLVES
    _check_bands(res, cfg, r, opt, NAV, gctx["w0"], snap)


def test_risk_aversion_mode_fallbacks(gctx):
    r, a, snap = gctx["risk"], gctx["alpha"], gctx["snap"]
    assert PortfolioConfig().risk_aversion == 2.0  # audit: gamma ~ IR / (2 sigma_target)
    res = SinglePeriodOptimizer(PortfolioConfig(target_vol_annual=None), SimpleCostModel(zero=True)).solve(
        a, r, gctx["w0"], NAV, snap
    )
    assert res.ok and res.risk_aversion == 2.0 and res.n_solves == 1 and res.vol_target_binding == ""
    zero = SinglePeriodOptimizer(PortfolioConfig(), SimpleCostModel(zero=True)).solve(
        a * 0.0, r, gctx["w0"], NAV, snap
    )
    assert zero.ok and zero.vol_target_binding == "no_alpha" and zero.turnover < 1e-6
    with pytest.raises(ValueError, match="risk_aversion_mode"):
        SinglePeriodOptimizer(PortfolioConfig(risk_aversion_mode="bogus"), SimpleCostModel())


def test_gp_shrink_survives_gamma_search(gctx):
    """Finding 5: with the alpha-blind projection the GP shrink was undone (same
    gross / vol with and without it).  With gamma-search sizing both books sit at
    the target vol but the shrunk alpha trades less against the cost term."""
    r, a, snap, cost = gctx["risk"], gctx["alpha"], gctx["snap"], gctx["cost"]
    rng = np.random.default_rng(3)
    w_prev = pd.Series(np.clip(0.004 * rng.standard_normal(len(r.tickers)), -0.01, 0.01), index=r.tickers)
    w_prev -= w_prev.mean()
    lo = SinglePeriodOptimizer(_fixed(64.0), cost).solve(a, r, w_prev, NAV, snap, neutral_styles=STYLES)
    hi = SinglePeriodOptimizer(_fixed(1.0), cost).solve(a, r, w_prev, NAV, snap, neutral_styles=STYLES)
    target = float(np.sqrt(_vol(lo) * _vol(hi)))
    kw = {"neutral_styles": STYLES, "alpha_half_life_days": 20.0}
    off = SinglePeriodOptimizer(PortfolioConfig(target_vol_annual=target), cost).solve(
        a, r, w_prev, NAV, snap, **kw
    )
    on = SinglePeriodOptimizer(PortfolioConfig(target_vol_annual=target, use_gp_shrink=True), cost).solve(
        a, r, w_prev, NAV, snap, **kw
    )
    assert off.ok and on.ok and on.gp_shrink < 0.9 and off.gp_shrink == 1.0
    assert (
        abs(_vol(on) / target - 1) < GAMMA_SEARCH_VOL_TOL
        and abs(_vol(off) / target - 1) < GAMMA_SEARCH_VOL_TOL
    )
    assert on.turnover < 0.97 * off.turnover
    assert on.expected_cost < off.expected_cost
    assert on.w_target.abs().sum() <= off.w_target.abs().sum() + 1e-6


# ----------------------------------------------------------------- out-of-model names
def test_no_positions_outside_risk_model_from_flat(gctx):
    """Finding 3: from flat, names the risk model does not cover are never
    opened (they used to serve as zero-exposure hedges for the net band); the
    net and beta bands hold on the modelled book; excluded names are counted."""
    r, a, snap, cost = gctx["risk"], gctx["alpha"], gctx["snap"], gctx["cost"]
    extra = pd.Index([f"ZZ{i:03d}" for i in range(30)])
    a2 = pd.concat([a, pd.Series(np.linspace(-1, 1, 30) * a.abs().max(), index=extra)])
    snap2 = {
        k: (v.reindex(a2.index).fillna(v.median()) if isinstance(v, pd.Series) else v)
        for k, v in snap.items()
    }
    w0 = pd.Series(0.0, index=a2.index)
    cfg = PortfolioConfig()
    opt = SinglePeriodOptimizer(cfg, cost)
    res = opt.solve(a2, r, w0, NAV, snap2, neutral_styles=STYLES)
    assert res.ok
    assert (res.w_target.reindex(extra) == 0.0).all()
    assert res.n_out_of_model == 0 and res.n_excluded_alpha == 30
    modelled = res.w_target.reindex(r.tickers)
    assert abs(modelled.sum()) <= cfg.net_exposure_band + TOL
    assert abs(r.exposures["beta"] @ modelled) <= opt.beta_band + TOL
    assert abs(r.exposures["market"] @ modelled) <= cfg.net_exposure_band + TOL


# ----------------------------------------------------------------- projection units
def test_projection_objective_is_weighted_tracking_variance_plus_cost(gctx):
    """Finding 2 (fixed-gamma ``"projection"`` mode): the projection minimises
    ``gamma H (w - s w*)' Sigma (w - s w*) + c TC(w - w0)`` -- checked against a
    dense evaluation of that objective at the returned book."""
    import cvxpy as cp

    r, a, snap, cost = gctx["risk"], gctx["alpha"], gctx["snap"], gctx["cost"]
    H, g = 20, 10.0
    base = SinglePeriodOptimizer(_fixed(g), cost, reuse_problem=True).solve(
        a, r, gctx["w0"], NAV, snap, neutral_styles=STYLES
    )
    target = 0.6 * _vol(base)
    cfg = dataclasses.replace(_fixed(g), target_vol_annual=target, vol_target_mode="projection")
    opt = SinglePeriodOptimizer(cfg, cost, reuse_problem=True)
    res = opt.solve(a, r, gctx["w0"], NAV, snap, neutral_styles=STYLES)
    assert res.ok and res.vol_scalar == pytest.approx(0.6, rel=1e-6)
    t = res.w_target.index
    S = r.covariance(t, include_alignment=True).to_numpy()
    d = res.w_target.to_numpy() - 0.6 * base.w_target.reindex(t).to_numpy()
    tc, _ = cost.cvx_terms(
        cp.Constant(res.x.to_numpy()), cp.Constant(res.w_target.to_numpy()), NAV, snap, "routed", t
    )
    expected = g * H * float(d @ S @ d) + cfg.vol_target_projection_cost_weight * float(tc.value)
    comp = opt._cache[("projection", True)]
    assert float(comp.problem.value) == pytest.approx(expected, rel=1e-4)


# ----------------------------------------------------------------- venue / DPP
def test_solve_default_venue_is_routed():
    assert inspect.signature(SinglePeriodOptimizer.solve).parameters["venue"].default == "routed"


def test_routed_cost_problem_is_dpp_and_reuse_matches_rebuild(gctx):
    """Finding 6: the routed kink terms C * pos(A - K) are rewritten as
    pos(C * A - C K) so the cached problem is DPP and is actually re-solved."""
    import cvxpy as cp

    from qts.portfolio.optimizer import _constants, _Parametrizer

    md, model, cost = gctx["md"], gctx["model"], gctx["cost"]
    snap = gctx["snap"]
    t = gctx["risk"].tickers
    x, w = cp.Variable(len(t)), cp.Variable(len(t))
    tc, _ = cost.cvx_terms(x, w, NAV, snap, "routed", t)
    pz = _Parametrizer()
    tc_p = pz(tc)
    assert tc_p.is_dpp() and pz.n_consts == len(_constants(tc, []))
    rng = np.random.default_rng(1)
    x.value, w.value = 0.01 * rng.standard_normal(len(t)), 0.01 * rng.standard_normal(len(t))
    assert float(tc_p.value) == pytest.approx(float(tc.value), rel=1e-9)  # same function

    cfg = PortfolioConfig(target_vol_annual=0.08)
    reuse = SinglePeriodOptimizer(cfg, cost, reuse_problem=True)
    fresh = SinglePeriodOptimizer(cfg, cost, reuse_problem=False)
    w_prev = gctx["w0"]
    for dd in md.dates[-8:-4:2]:
        r = model.as_of(dd)
        a = gctx["alpha"].reindex(r.tickers).fillna(0.0) * (1 + 0.2 * rng.standard_normal(len(r.tickers)))
        s = cost.snapshot_from_market(md, dd)
        xr = reuse.solve(a, r, w_prev, NAV, s, neutral_styles=STYLES)
        yr = fresh.solve(a, r, w_prev, NAV, s, neutral_styles=STYLES)
        assert xr.ok and yr.ok
        assert float((xr.w_target - yr.w_target).abs().max()) < 2e-5
        w_prev = xr.w_target
    assert all(c.is_dpp for c in reuse._cache.values())
    assert reuse.n_reuses >= 1 and xr.problem_reused
