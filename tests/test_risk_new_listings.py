"""New listings and de-SPACs (docs/notes/risk_new_listing_fix_2026-11-10.md).

The real-data pathology (Sharadar, 2020-06 .. 2021-08: DKNG, MP, QS, OPEN, ALIT,
SOFI): a de-SPAC enters the universe with (i) ~250 days of history at exactly
10.00 and (ii) the fundamentals of a shell -- total assets a year earlier were
the sponsor's few thousand dollars, so year-on-year asset growth is of order
+100,000 %.  (i) alone is harmless (``test_flat_price_history_alone_is_benign``).
(ii) is the cause: "clip at 3 sd, then re-standardise" does not survive a gross
outlier, the ``investment`` exposure of that one name comes out at ``-sqrt(N)``
with every other name inside a band of width ~0.01, the style becomes a dummy
for the name (regression leverage 0.999997, residual ~0), and the structural
specific-risk model -- ``exp`` of a regression whose ``investment`` coefficient
is fitted on that band -- returns a specific vol of 1e55.  The size-decile
shrinkage target, a cap-weighted mean, then hands (nearly) the same huge value
to every other name of the decile.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from qts.core.config import RiskConfig, UniverseConfig
from qts.core.returns import flat_run_mask
from qts.data.synthetic import SyntheticSpec, generate
from qts.data.universe import apply_mask, build_universe_mask, masked_levels
from qts.risk import covariance as cov_mod
from qts.risk import model as model_mod
from qts.risk.covariance import (
    bayesian_shrink_specific,
    repair_covariance,
    robust_factor_variances,
    specific_risk,
    structural_specific_vol,
)
from qts.risk.exposures import standardize_style, winsorize_raw_descriptor
from qts.risk.hybrid import SPECIFIC_FLOOR, HybridRiskModel
from qts.risk.model import BarraStyleRiskModel, estimation_inputs

#: escape hatches: every new-listing protection off (the estimator before this fix)
PROTECTIONS_OFF = {
    "style_raw_winsor_mad_k": 0.0,
    "style_exposure_cap": math.inf,
    "flat_run_min_days": 0,
    "structural_clip_pctl": 0.0,
    "specific_var_cap_mult": math.inf,
    "covariance_repair": False,
}
N_STOCKS, N_DAYS, TOP_N = 200, 1100, 120
MERGER_ROWS = (560, 640, 720)  # T of the three SPACs
FLAT_DAYS = 250
SHELL_GROWTH = 1500.0  # +150,000 % year-on-year asset growth (trust vs pre-IPO shell)
SHELL_DAYS = 75  # trading days after T until the next filing replaces it


def _base_market():
    return generate(SyntheticSpec(n_stocks=N_STOCKS, n_days=N_DAYS, seed=1))


def spac_market(noise: float = 0.0, shell: bool = True):
    """The base market with its last three names turned into de-SPACs: no price
    before ``T - 250``; adjusted and raw close flat at 10.00 for 250 days (returns
    exactly 0, or +-``noise`` daily); market cap 3e8 and 1 MUSD a day traded (not
    eligible); on day ``T`` +60 %, the cap jumps to 3e10 (top decile) and the
    name trades 300 MUSD a day at 60 % annual vol.  With ``shell`` the reported
    asset growth is ``SHELL_GROWTH`` from ``T - 60`` to ``T + SHELL_DAYS``."""
    md = _base_market()
    rng = np.random.default_rng(99)
    close, raw, cap, vol, sh, ag = (
        x.copy()
        for x in (md.close, md.close_raw, md.market_cap, md.volume, md.shares_outstanding, md.asset_growth)
    )
    names = list(md.tickers[-len(MERGER_ROWS) :])
    t_idx = np.arange(N_DAYS)
    for s, t0 in zip(names, MERGER_ROWS, strict=True):
        p = np.full(N_DAYS, np.nan)
        p[t0 - FLAT_DAYS : t0] = (
            10.0 * np.cumprod(1 + noise * rng.standard_normal(FLAT_DAYS)) if noise else 10.0
        )
        r = 0.60 / np.sqrt(252) * rng.standard_normal(N_DAYS - t0)
        r[0] = 0.60
        p[t0:] = p[t0 - 1] * np.cumprod(1 + r)
        listed = np.isfinite(p)
        c = np.where(listed, np.where(t_idx < t0, 3e8, 3e10 * p / p[t0]), np.nan)
        close[s], raw[s], cap[s], sh[s] = p, p, c, c / p
        vol[s] = np.where(listed, np.where(t_idx < t0, 1e6, 3e8) / p, np.nan)
        if shell:
            ag.iloc[t0 - 60 : t0 + SHELL_DAYS, ag.columns.get_loc(s)] = SHELL_GROWTH
    md = replace(
        md, close=close, close_raw=raw, market_cap=cap, volume=vol, shares_outstanding=sh, asset_growth=ag
    )
    return md, names


def _pipeline_view(md):
    """What the pipeline hands the risk model: PIT universe mask, caps NaN outside it."""
    mask = build_universe_mask(md, UniverseConfig(top_n_by_mcap=TOP_N))
    return masked_levels(apply_mask(md, mask))


def _fit(md, **over) -> HybridRiskModel:
    cfg = RiskConfig(reference_vol_check_in_fit=False, **over)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        return HybridRiskModel(BarraStyleRiskModel(cfg), cfg).fit(md)


def _grid(dense: int, step: int) -> list[int]:
    """Rows T-5 .. T+dense daily, then every ``step``-th row to T+300, for every T."""
    rows: set[int] = set()
    for t0 in MERGER_ROWS:
        rows |= set(range(t0 - 5, t0 + dense + 1)) | set(range(t0 + dense, t0 + 301, step))
    return sorted(r for r in rows if r < N_DAYS)


FAST_GRID = _grid(dense=12, step=16)
FULL_GRID = _grid(dense=300, step=1)


def _scan(model: HybridRiskModel, rows, names=()):
    """as_of on every row: reference vols (fundamental, hybrid), the largest
    statistical factor variance, and the bound checks on the specific variances."""
    fm = model.fundamental
    dates = fm.md.dates
    out = {"fund": [], "hyb": [], "stat_var": [], "max_abs_style": []}
    for i in rows:
        d = dates[i]
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            h = model.as_of(d)
            f = fm.as_of(d)
        out["fund"].append(fm.diagnostics["reference_vol"][d])
        out["hyb"].append(fm.diagnostics["reference_vol_hybrid"][d])
        stat = h.factor_groups["statistical"]
        out["stat_var"].append(float(np.diag(h.factor_cov.loc[stat, stat].to_numpy()).max()))
        out["max_abs_style"].append(float(f.exposures[fm.cfg.styles].abs().to_numpy().max()))
        lo, hi = fm.diagnostics["specific_var_bounds"][d]
        if lo is not None and hi is not None:
            assert np.isfinite(f.specific_var).all() and np.isfinite(h.specific_var).all()
            assert f.specific_var.min() >= lo * (1 - 1e-9) and f.specific_var.max() <= hi * (1 + 1e-9)
            assert h.specific_var.min() >= SPECIFIC_FLOOR * lo * (1 - 1e-9)
            assert h.specific_var.max() <= hi * (1 + 1e-9)
            for s in names:
                if s in f.specific_var.index:
                    assert lo * (1 - 1e-9) <= f.specific_var[s] <= hi * (1 + 1e-9)
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


@pytest.fixture(scope="module")
def base_fit():
    return _fit(_pipeline_view(_base_market()))


@pytest.fixture(scope="module")
def base_scan(base_fit):
    return _scan(base_fit, FAST_GRID)


@pytest.fixture(scope="module")
def spac0():
    md, names = spac_market(noise=0.0)
    return _pipeline_view(md), names


@pytest.fixture(scope="module")
def fit_off(spac0):
    return _fit(spac0[0], **PROTECTIONS_OFF)


@pytest.fixture(scope="module")
def fit_on(spac0):
    return _fit(spac0[0])


# --------------------------------------------------------------------------- the pathology
def test_flat_price_history_alone_is_benign(base_fit):
    """250 days at exactly 10.00, a +60 % jump and a cap in the top decile do NOT
    break the pre-fix estimator: with ordinary fundamentals the reference vol
    stays at the no-SPAC level.  The price path is not the cause."""
    md, _ = spac_market(noise=0.0, shell=False)
    off = _fit(_pipeline_view(md), **PROTECTIONS_OFF)
    rows = [t0 + k for t0 in MERGER_ROWS for k in (6, 40, 73)]
    got, ref = _scan(off, rows), _scan(base_fit, rows)
    assert (np.abs(got["fund"] / ref["fund"] - 1.0) < 0.25).all()
    assert not off.diagnostics["covariance_outliers"]


def test_pathology_reproduces_with_the_protections_off(fit_off, base_scan, spac0):
    md, names = spac0
    fm = fit_off.fundamental
    got = _scan(fit_off, FAST_GRID)
    # the alarm: reference vol astronomically above the no-SPAC fit, fundamental and hybrid
    assert got["fund"].max() > 1e6 * base_scan["fund"].max()
    assert got["hyb"].max() > 1e6 * base_scan["hyb"].max()
    assert fm.diagnostics["covariance_outliers"]
    assert not fm.diagnostics["covariance_repairs"]  # switched off
    # the mechanism, on T + 6 of the first SPAC
    s, d = names[0], md.dates[MERGER_ROWS[0] + 6]
    B = fm.exposures_as_of(d)
    n = len(B)
    inv = B["investment"]
    assert abs(inv[s]) > 0.9 * math.sqrt(n)  # the clipped outlier is back at ~sqrt(N) sd
    assert inv.drop(s).std() < 0.01  # ... and everybody else is one point
    u = fm.specific_returns[s].loc[:d].dropna()
    r = fm.returns_clean[s].reindex(u.index)
    assert u.abs().max() < 1e-3 * r.abs().max()  # the style is a dummy for the name: residual ~ 0
    mcap = md.market_cap.loc[:d].iloc[-1]
    with np.errstate(all="ignore"):
        var, det = specific_risk(
            fm.specific_returns.loc[:d],
            B,
            mcap,
            fm.cfg,
            return_details=True,
            market_cap_history=md.market_cap.loc[:d],
        )
    assert det["gamma"][s] == 0.0 and det["sigma_str"][s] > 1e20  # exp() of the extrapolation
    assert det["sigma_ts"][s] < 0.02  # the EWMA input itself is tiny, not huge
    # the shrinkage target (a cap-weighted mean) hands the same value to the decile
    med = float(var.median())
    polluted = var.drop(s)[var.drop(s) > 1e6 * med]
    assert len(polluted) >= 5
    assert polluted.max() / polluted.min() < 1.01  # near-identical across ordinary names


# --------------------------------------------------------------------------- the fix
def _assert_sane(model, scan, base, names):
    fm = model.fundamental
    for key in ("fund", "hyb"):
        ratio = scan[key] / base[key]
        assert np.isfinite(ratio).all() and (np.abs(ratio - 1.0) < 0.25).all(), (
            key,
            ratio.min(),
            ratio.max(),
        )
    assert np.isfinite(scan["stat_var"]).all()
    assert (scan["stat_var"] <= 10.0 * np.maximum(base["stat_var"], 1e-12)).all()
    assert (scan["max_abs_style"] <= fm.cfg.style_exposure_cap + 1e-12).all()
    # the cause is fixed: the last-line repair never had to act, the alarm never rang
    assert fm.diagnostics["covariance_repairs"] == []
    assert fm.diagnostics["covariance_outliers"] == []


def test_defaults_keep_the_fit_sane(fit_on, base_scan, spac0):
    _, names = spac0
    _assert_sane(fit_on, _scan(fit_on, FAST_GRID, names), base_scan, names)


def test_defaults_keep_the_fit_sane_with_noisy_flat_prices(base_scan):
    """+-0.1 % daily noise around 10.00: the non-trading rule does not fire, the
    other protections carry it."""
    md, names = spac_market(noise=0.001)
    on = _fit(_pipeline_view(md))
    assert on.diagnostics["n_flat_return_days"] == 0
    _assert_sane(on, _scan(on, FAST_GRID, names), base_scan, names)


@pytest.mark.slow
@pytest.mark.parametrize("noise", [0.0, 0.001])
def test_defaults_sane_on_every_date_to_t_plus_300(noise):
    base = _scan(_fit(_pipeline_view(_base_market())), FULL_GRID)
    md, names = spac_market(noise=noise)
    on = _fit(_pipeline_view(md))
    _assert_sane(on, _scan(on, FULL_GRID, names), base, names)


def test_the_exposure_is_bounded_and_the_cross_section_keeps_its_spread(fit_on, fit_off, spac0):
    md, names = spac0
    fm = fit_on.fundamental
    s, d = names[0], md.dates[MERGER_ROWS[0] + 6]
    B = fm.exposures_as_of(d)
    assert B.loc[s, "investment"] == pytest.approx(-3.0)
    assert B["investment"].drop(s).std() > 0.8  # the other names are a cross-section again
    u = fm.specific_returns[s].loc[:d].dropna()
    r = fm.returns_clean[s].reindex(u.index)
    assert u.abs().max() > 0.1 * r.abs().max()  # the name has a real residual
    # non-trading days are not observations: beta / residual vol are missing (0 exposure)
    # until the name has BETA_WINDOW // 4 real returns; without the rule they are computed
    # on 250 zeros and one jump
    assert B.loc[s, "beta"] == 0.0 and B.loc[s, "residual_vol"] == 0.0
    B_off = fit_off.fundamental.exposures_as_of(d)
    assert B_off.loc[s, "beta"] < -1.0 and B_off.loc[s, "residual_vol"] > 1.0
    late = md.dates[MERGER_ROWS[0] + 80]
    assert fm.exposures_as_of(late).loc[s, "residual_vol"] > 1.0  # 80 real returns: a high-vol name
    assert fm.diagnostics["n_flat_return_days"] > 0


def test_each_layer_alone(spac0, base_fit):
    """Which protection is the cure: the raw-descriptor winsorisation removes the
    cause; the structural clip and the cap each bound the damage on their own;
    winsorising the *standardised* exposure at 3 alone does not (the style is
    still a dummy for the outlier, its structural coefficient still fitted on a
    collapsed cross-section)."""
    md, _ = spac0
    rows = [MERGER_ROWS[0] + k for k in (6, 40, 73)]
    ref = _scan(base_fit, rows)["fund"]
    worst = {}
    for label, on in {
        "mad": {"style_raw_winsor_mad_k": 10.0},
        "struct": {"structural_clip_pctl": 0.01},
        "cap": {"specific_var_cap_mult": 25.0},
        "clip3": {"style_exposure_cap": 3.0},
    }.items():
        m = _fit(md, **{**PROTECTIONS_OFF, **on})
        worst[label] = float((_scan(m, rows)["fund"] / ref).max())
    assert worst["mad"] < 1.25 and worst["struct"] < 1.25 and worst["cap"] < 1.25
    assert worst["clip3"] > 1e6


# --------------------------------------------------------------------------- units
def test_raw_winsorisation_and_standardisation():
    rng = np.random.default_rng(0)
    n = 400
    x = rng.standard_normal(n)
    x[7] = 1e6
    raw = pd.DataFrame([x], columns=[f"n{i}" for i in range(n)])
    cap = pd.DataFrame(np.ones((1, n)), columns=raw.columns)
    old = standardize_style(raw, cap, raw_winsor_mad_k=0.0, exposure_cap=math.inf).iloc[0]
    assert old["n7"] > 0.9 * math.sqrt(n) and old.drop("n7").std() < 0.01
    clip_only = standardize_style(raw, cap, raw_winsor_mad_k=0.0, exposure_cap=3.0).iloc[0]
    assert clip_only["n7"] == 3.0 and clip_only.drop("n7").std() < 0.01  # still a dummy
    new = standardize_style(raw, cap).iloc[0]
    assert new["n7"] == 3.0 and new.abs().max() <= 3.0
    assert new.drop("n7").std() > 0.9
    assert np.corrcoef(new.drop("n7"), x[np.arange(n) != 7])[0, 1] > 0.999
    # no gross outlier: the pre-winsorisation is the identity (same object back)
    clean = pd.DataFrame([rng.standard_normal(n)], columns=raw.columns)
    assert winsorize_raw_descriptor(clean, 10.0) is clean
    # MAD = 0 (most names share one value, the rest on one side): the IQR is the
    # fallback scale; with both zero the row is left alone
    y = np.zeros(n)
    y[:180] = np.abs(rng.standard_normal(180))
    y[3] = 1e9
    w = winsorize_raw_descriptor(pd.DataFrame([y]), 10.0).iloc[0]
    assert w[3] < 100.0 and np.array_equal(w.drop(3).to_numpy(), np.delete(y, 3))
    const = pd.DataFrame([np.r_[np.zeros(n - 1), 5.0]])
    assert winsorize_raw_descriptor(const, 10.0) is const


def test_flat_run_mask_is_causal():
    r = pd.DataFrame(
        {
            "flat": [0.01] + [0.0] * 30 + [0.6, 0.02],
            "gap": [0.0] * 10 + [np.nan] + [0.0] * 22,
            "noisy": [1e-3 * (-1) ** i for i in range(33)],
        }
    )
    m = flat_run_mask(r, tol=1e-6, min_days=20)
    assert not m["flat"].iloc[:20].any()  # day 20 of the run is the first flagged
    assert m["flat"].iloc[20:31].all() and not m["flat"].iloc[31:].any()
    assert not m["gap"].iloc[:30].any() and m["gap"].iloc[30:].all()  # a NaN breaks the run
    assert not m["noisy"].any()
    assert not flat_run_mask(r, 1e-6, 0).to_numpy().any()
    # causal: the flags of a prefix equal the prefix of the flags
    assert flat_run_mask(r.iloc[:25], 1e-6, 20).equals(m.iloc[:25])


def test_estimation_inputs_drop_non_trading_days(spac0):
    md, names = spac0
    _, ret_on, _, _ = estimation_inputs(md, RiskConfig())
    _, ret_off, _, _ = estimation_inputs(md, RiskConfig(flat_run_min_days=0))
    t0 = MERGER_ROWS[0]
    s = names[0]
    assert ret_off[s].iloc[t0 - 200 : t0].eq(0.0).all()
    assert ret_on[s].iloc[t0 - 200 : t0].isna().all()
    # the merger day itself is an observation (+60 %, winsorised by return rule (b))
    assert ret_on[s].iloc[t0] == ret_off[s].iloc[t0] > 0.10
    others = [c for c in ret_on.columns if c not in names]
    pd.testing.assert_frame_equal(ret_on[others], ret_off[others])


def test_structural_prediction_is_clipped_to_the_fitted_band():
    rng = np.random.default_rng(1)
    n = 300
    idx = pd.Index([f"n{i}" for i in range(n)])
    X = pd.DataFrame({"market": 1.0, "s": rng.standard_normal(n)}, index=idx)
    X.loc["n0", "s"] = 400.0  # far outside the fitted cross-section
    sigma = pd.Series(0.02 * np.exp(0.3 * X["s"].clip(-3, 3) + 0.1 * rng.standard_normal(n)), index=idx)
    gamma = pd.Series(1.0, index=idx)
    gamma["n0"] = 0.0
    cap = pd.Series(1e9, index=idx)
    with np.errstate(all="ignore"):
        raw = structural_specific_vol(sigma, gamma, X, cap)
        clipped = structural_specific_vol(sigma, gamma, X, cap, clip_pctl=0.01)
    assert raw["n0"] > 1e30
    lo, hi = np.quantile(sigma.drop("n0"), [0.01, 0.99])
    assert clipped["n0"] == pytest.approx(hi)
    assert clipped.min() >= lo * (1 - 1e-12) and clipped.max() <= hi * (1 + 1e-12)
    inner = (raw > lo) & (raw < hi)
    assert inner.sum() > 250 and np.allclose(raw[inner], clipped[inner])


def test_cap_bounds_the_shrinkage_input_and_the_output():
    rng = np.random.default_rng(2)
    n, T = 200, 600
    idx = pd.Index([f"n{i}" for i in range(n)])
    dates = pd.bdate_range("2020-01-01", periods=T)
    U = pd.DataFrame(0.02 * rng.standard_normal((T, n)), index=dates, columns=idx)
    U["n5"] *= 1e6  # one name with an absurd residual history
    B = pd.DataFrame({"market": 1.0, "size": rng.standard_normal(n)}, index=idx)
    cap = pd.Series(np.exp(rng.normal(21, 1, n)), index=idx)
    off = RiskConfig(specific_var_cap_mult=math.inf)
    v_off = specific_risk(U, B, cap, off)
    v_on, det = specific_risk(U, B, cap, RiskConfig(), return_details=True)
    med = float(v_on.median())
    assert v_off["n5"] > 1e9 * med
    assert v_off.drop("n5").max() > 1e6 * med  # the decile mean is polluted
    lo, hi = det["var_bounds"]
    assert v_on.max() <= hi * (1 + 1e-12) and v_on.min() >= lo * (1 - 1e-12)
    # capped at 5x the median vol before the shrinkage, which then pulls it part of the
    # way to its decile mean: far above the median name, never above the cap
    assert 10 * med < v_on["n5"] <= hi and det["n_at_cap"] == 1
    assert hi == pytest.approx(
        RiskConfig().specific_var_cap_mult
        * det["vra_multiplier"] ** 2
        * float((det["sigma_shrunk"] ** 2).median()),
        rel=1e-9,
    )
    assert v_on.drop("n5").max() < 5 * med  # nobody else is dragged along
    # the shrinkage target itself: arithmetic, hence the bound on its input
    s = pd.Series(0.02, index=idx)
    s["n5"] = 1e20
    _, target = bayesian_shrink_specific(s, cap, 0.1, return_target=True)
    assert target.max() > 1e15


# --------------------------------------------------------------------------- belt and braces
def test_repair_covariance_unit():
    rng = np.random.default_rng(3)
    cols = ["market", "a", "b", "empty"]
    fr = pd.DataFrame(0.01 * rng.standard_normal((400, 4)), columns=cols)
    fr["empty"] = 0.0
    F = pd.DataFrame(np.diag([1e-4, 1e-4, 1e-4, 0.0]), index=cols, columns=cols)
    D = pd.Series([1e-4, 2e-4, 3e-4], index=["x", "y", "z"])
    F2, D2, rep = repair_covariance(F, D, fr, (1e-5, 1e-3), 100.0)
    assert rep == [] and F2 is F and D2 is D  # nothing to do: inputs back untouched
    ref = robust_factor_variances(fr)
    assert ref["market"] == pytest.approx(1e-4, rel=0.3) and np.isnan(ref["empty"])
    Fb = F.copy()
    Fb.loc["a", "a"] = 1e20
    Fb.loc["a", "b"] = Fb.loc["b", "a"] = 1e8  # correlation 1e-2... kept under the rescale
    Fb.loc["b", "market"] = Fb.loc["market", "b"] = np.nan
    Db = pd.Series([1e20, np.nan, 1e-9], index=D.index)
    F3, D3, rep = repair_covariance(Fb, Db, fr, (1e-5, 1e-3), 100.0)
    what = {r["what"] for r in rep}
    assert what == {
        "specific_var_not_finite",
        "specific_var_below_floor",
        "specific_var_above_cap",
        "factor_var_above_bound",
        "factor_cov_not_finite",
    }
    assert np.isfinite(F3.to_numpy()).all() and np.isfinite(D3).all()
    assert F3.loc["a", "a"] == pytest.approx(100.0 * ref["a"], rel=1e-6)
    assert D3["x"] == 1e-3 and D3["z"] == 1e-5 and 1e-5 <= D3["y"] <= 1e-3
    assert np.linalg.eigvalsh(F3.to_numpy()).min() > -1e-12
    big = next(r for r in rep if r["what"] == "factor_var_above_bound")
    assert big["worst"] == ("a", 1e20)
    # a non-finite variance: the factor takes its reference variance, no covariances
    Fn = F.copy()
    Fn.loc["b", "b"] = np.inf
    F4, _, rep = repair_covariance(Fn, D, fr, None, 100.0)
    assert [r["what"] for r in rep] == ["factor_var_not_finite"]
    assert F4.loc["b", "b"] == pytest.approx(ref["b"]) and F4.loc["b", "a"] == 0.0


def test_repair_fires_when_1e20_is_injected_by_hand(monkeypatch, fit_on, spac0):
    """The repair step is live: poison the factor covariance and one specific
    variance after estimation and ``as_of`` clips both and records them."""
    md, _ = spac0
    cfg = RiskConfig(reference_vol_check_in_fit=False)
    fm = fit_on.fundamental
    d = md.dates[900]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clean = fm.as_of(d)
    assert fm.diagnostics["covariance_repairs"] == []
    victim = clean.specific_var.index[10]
    real_fc, real_sr = model_mod.factor_covariance, model_mod.specific_risk

    def bad_fc(*a, **k):
        F = real_fc(*a, **k).copy()
        F.loc["market", "market"] = 1e20
        return F

    def bad_sr(*a, **k):
        var, det = real_sr(*a, **k)
        var = var.copy()
        var[victim] = 1e20
        return var, det

    monkeypatch.setattr(model_mod, "factor_covariance", bad_fc)
    monkeypatch.setattr(model_mod, "specific_risk", bad_sr)
    n_out = len(fm.diagnostics["covariance_outliers"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = fm.as_of(d)
    reps = [r for r in fm.diagnostics["covariance_repairs"] if r["date"] == d]
    assert {r["what"] for r in reps} == {"factor_var_above_bound", "specific_var_above_cap"}
    assert all(r["model"] == "fundamental" for r in reps)
    assert next(r for r in reps if r["what"] == "specific_var_above_cap")["worst"] == (victim, 1e20)
    ref = robust_factor_variances(fm.factor_returns.loc[:d])["market"]
    assert out.factor_cov.loc["market", "market"] == pytest.approx(cfg.factor_var_repair_mult * ref, rel=1e-6)
    _, hi = fm.diagnostics["specific_var_bounds"][d]
    assert out.specific_var[victim] == pytest.approx(hi)
    assert out.specific_var.drop(victim).equals(clean.specific_var.drop(victim))
    # the repaired covariance is what the alarm sees: bounded (market vol x10), not 1e20
    assert fm.diagnostics["reference_vol"][d] < 3.0
    assert len(fm.diagnostics["covariance_outliers"]) >= n_out
    # switched off: the poison goes through and the alarm is the only defence
    fm.diagnostics["covariance_repairs"].clear()
    fm.diagnostics["covariance_outliers"].clear()
    monkeypatch.setattr(fm, "cfg", replace(fm.cfg, covariance_repair=False))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = fm.as_of(d)
    assert raw.factor_cov.loc["market", "market"] == 1e20 and raw.specific_var[victim] == 1e20
    assert fm.diagnostics["covariance_repairs"] == [] and fm.diagnostics["covariance_outliers"]
    fm.diagnostics["covariance_outliers"].clear()


# --------------------------------------------------------------------------- config
def test_config_bounds_and_escape_hatches():
    RiskConfig(**PROTECTIONS_OFF)
    for bad in (
        {"style_exposure_cap": 0.0},
        {"style_raw_winsor_mad_k": -1.0},
        {"specific_var_cap_mult": 0.5},
        {"structural_clip_pctl": 0.5},
        {"flat_run_min_days": -1},
        {"factor_var_repair_mult": 0.0},
        {"min_specific_obs": -5},
    ):
        with pytest.raises(ValueError):
            RiskConfig(**bad)
    assert cov_mod.REPAIR_WINDOWS[0] == 252
