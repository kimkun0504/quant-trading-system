"""Thin industries, the specific-variance floor and VRA clipping
(docs/notes/risk_thin_industry_fix_2026-11-10.md).

The real-data pathology: an industry with one eligible member absorbs that
stock's whole return into its dummy, the residual is ~1e-16, and the day a
second member joins (or the ESTU changes) the specific VRA bias statistic
divides a real residual by a 1e-32 forecast variance -- the cap-weighted
multiplier applied to *every* name explodes (reference-book vol 4.4e11 % on
2003-09-30).  Reproduced here on the synthetic market by relabelling three
names into single-member industries and two names into a two-member industry
whose second member joins late (dated ``industry_history``).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from qts.core.config import RiskConfig
from qts.data.loaders import FF12_NAMES, FF48_NAMES, FF48_TO_FF12
from qts.data.synthetic import SyntheticSpec, generate
from qts.risk.exposures import MARKET_FACTOR
from qts.risk.factor_returns import (
    ALL_INDUSTRIES,
    cap_weighted_industry_sum,
    estimate_factor_returns,
    industry_parent,
    shrink_thin_industries,
)
from qts.risk.model import BarraStyleRiskModel

#: escape hatches: every defence off (the pre-2026-11-10 estimator)
FIXES_OFF = {
    "thin_industry_min_members": 0,
    "specific_var_floor_frac": 0.0,
    "vra_z2_clip": math.inf,
    "vra_min": 0.0,
    "vra_max": math.inf,
}
JOIN_ROW = 600  # the second PAIR member joins on this date index
SINGLES = ("SOLO0", "SOLO1", "SOLO2")


@pytest.fixture(scope="module")
def clean_md():
    return generate(SyntheticSpec(n_stocks=150, n_days=900, seed=1))


@pytest.fixture(scope="module")
def thin_md(clean_md):
    """Three single-member industries and one pair whose second member joins
    at ``JOIN_ROW`` (before that it keeps its original label)."""
    md = clean_md
    t = list(md.tickers)
    ind = md.industry.copy()
    for i, s in enumerate(t[:3]):
        ind[s] = SINGLES[i]
    ind[t[3]] = ind[t[4]] = "PAIR"
    hist = pd.DataFrame(index=md.dates[[0, JOIN_ROW]], columns=md.tickers, dtype=object)
    hist.iloc[0] = ind.reindex(md.tickers).to_numpy()
    hist.iloc[0, 4] = md.industry[t[4]]
    hist.iloc[1] = ind.reindex(md.tickers).to_numpy()
    return replace(md, industry=ind, industry_history=hist)


def _fit(md, **over):
    cfg = RiskConfig(reference_vol_check_in_fit=False, **over)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return BarraStyleRiskModel(cfg).fit(md)


def _as_of_grid(model, dates):
    vols, lams, min_sigma_ts, floor_hits = [], [], [], []
    from qts.risk.covariance import specific_risk

    for d in dates:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = model.as_of(d)
            B = model.exposures_as_of(d)
            mcap = model.md.market_cap.loc[:d].iloc[-1]
            _, det = specific_risk(
                model.specific_returns.loc[:d],
                B,
                mcap,
                model.cfg,
                return_details=True,
                market_cap_history=model.md.market_cap.loc[:d],
            )
        vols.append(model.diagnostics["reference_vol"][d])
        lams.append(model.diagnostics["specific_vra_multiplier"][d])
        floor_hits.append(model.diagnostics["n_at_specific_floor"][d])
        estu = det["n_obs"] > 0
        min_sigma_ts.append(float(det["sigma_ts"][estu].min()))
        assert float(out.specific_var.min()) > 0
    return np.array(vols), np.array(lams), np.array(min_sigma_ts), np.array(floor_hits)


@pytest.fixture(scope="module")
def fits(clean_md, thin_md):
    return {
        "clean": _fit(clean_md),
        "on": _fit(thin_md),
        "off": _fit(thin_md, **FIXES_OFF),
    }


@pytest.fixture(scope="module")
def grid(thin_md):
    return thin_md.dates[520::40]


# --------------------------------------------------------------------------- the pathology
def test_pathology_reproduces_without_the_fixes(fits, grid):
    off, clean = fits["off"], fits["clean"]
    sr = off.specific_returns
    # single-member industry: the dummy absorbs the stock, residual is float noise
    for s in sr.columns[:3]:
        assert sr[s].abs().max() < 1e-12
    assert sr[sr.columns[3]].loc[: off.md.dates[JOIN_ROW - 1]].abs().max() < 1e-12
    vols, lams, min_ts, _ = _as_of_grid(off, grid)
    vols_c, lams_c, _, _ = _as_of_grid(clean, grid)
    assert min_ts.min() < 1e-8  # the 1e-16 specific vol
    # the VRA multiplier / reference vol explode on some date after the join
    assert (lams > 10 * lams_c).any() or (vols > 10 * vols_c).any()
    assert off.diagnostics["covariance_outliers"]


# --------------------------------------------------------------------------- the fixes
def test_fixes_keep_the_fit_sane(fits, grid):
    on, clean = fits["on"], fits["clean"]
    cfg = on.cfg
    sr = on.specific_returns
    # the single-member stock keeps a real residual
    for s in sr.columns[:3]:
        assert sr[s].abs().max() > 1e-3
    assert sr[sr.columns[3]].loc[: on.md.dates[JOIN_ROW - 1]].abs().max() > 1e-3
    assert set(on.diagnostics["thin_industry_days"]) == {*SINGLES, "PAIR"}
    assert on.diagnostics["thin_industry_days"]["SOLO0"] == len(sr)
    cells = on.diagnostics["thin_industry_cells"]
    assert cells[1]["PAIR"] + cells[2]["PAIR"] == len(sr) and cells[2]["PAIR"] > 0
    vols, lams, min_ts, floor_hits = _as_of_grid(on, grid)
    vols_c, _, _, _ = _as_of_grid(clean, grid)
    assert ((lams >= cfg.vra_min) & (lams <= cfg.vra_max)).all()
    assert min_ts.min() > 1e-3  # the raw EWMA specific vol is real, not float noise
    assert (np.abs(vols / vols_c - 1.0) < 0.25).all()
    assert not on.diagnostics["covariance_outliers"]
    assert (floor_hits >= 0).all()
    # min raw EWMA specific vol over the ESTU >= the floor (10 % of the median variance)
    for d in grid:
        from qts.risk.covariance import specific_risk

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            var, det = specific_risk(
                on.specific_returns.loc[:d],
                on.exposures_as_of(d),
                on.md.market_cap.loc[:d].iloc[-1],
                cfg,
                return_details=True,
                market_cap_history=on.md.market_cap.loc[:d],
            )
        estu = det["n_obs"] > 0
        sh2 = det["sigma_shrunk"][estu] ** 2
        assert float(sh2.min()) >= cfg.specific_var_floor_frac * float(sh2.median()) * (1 - 1e-9)
        assert float(var.min()) > 0 and np.isfinite(var).all()


def test_thin_industry_shrinkage_formula_and_constraint(fits, thin_md):
    """Hand-computed day: f_k = lam f_k^OLS + (1 - lam) f_parent - m, lam = n/(n+5),
    the parent of a non-FF48 label is the whole block (cap-weighted mean 0 by the
    constraint), m moves to the market factor, and the constraint still holds."""
    on, off = fits["on"], fits["off"]
    md = thin_md
    for row in (400, JOIN_ROW + 50):  # PAIR has 1 member, then 2
        d = md.dates[row]
        prev = md.dates[row - 1]
        B = on.exposures_as_of(prev)
        cap = md.market_cap.loc[prev]
        fit_on = estimate_factor_returns(md.returns.loc[d], B, cap, on.cfg)
        fit_off = estimate_factor_returns(md.returns.loc[d], B, cap, off.cfg)
        f_on, f_off = fit_on.factor_returns, fit_off.factor_returns
        n_pair = int((B["PAIR"] != 0).sum())
        assert n_pair == (1 if row < JOIN_ROW else 2)
        assert fit_on.thin_industries == {"SOLO0": 1, "SOLO1": 1, "SOLO2": 1, "PAIR": n_pair}
        m = fit_on.constraint_shift
        assert f_on[MARKET_FACTOR] == pytest.approx(f_off[MARKET_FACTOR] + m, abs=1e-14)
        prior = on.cfg.thin_industry_prior_obs
        for lab, n in fit_on.thin_industries.items():
            lam = n / (n + prior)
            # parent = all industries; their cap-weighted mean is 0 under the constraint
            assert f_on[lab] == pytest.approx(lam * f_off[lab] - m, abs=1e-12)
        fat = [c for c in fit_on.thin_industries.keys() ^ set(B.columns) if c.startswith("IND")]
        for lab in fat:
            assert f_on[lab] == pytest.approx(f_off[lab] - m, abs=1e-14)
        # style factors untouched; fitted values unchanged for every name except
        # through the thin industries
        for s in on.cfg.styles:
            assert f_on[s] == f_off[s]
        assert abs(cap_weighted_industry_sum(f_on, B, cap, on.cfg)) < 1e-10
        # residual identity r = B f + u
        recon = B.to_numpy() @ f_on.reindex(B.columns).to_numpy() + fit_on.specific_returns.reindex(B.index)
        assert np.allclose(recon, md.returns.loc[d].reindex(B.index).to_numpy())
    # the regression itself keeps the constraint on every sampled day of the history
    fr = on.factor_returns
    for d in fr.index[::97]:
        i = md.dates.get_loc(d)
        prev = md.dates[i - 1]
        assert (
            abs(
                cap_weighted_industry_sum(
                    fr.loc[d], on.exposures_as_of(prev), md.market_cap.loc[prev], on.cfg
                )
            )
            < 1e-10
        )


def test_shrink_toward_ff12_parent_sector():
    """FF48 labels shrink toward the cap-weighted mean of their FF12 sector."""
    cols = [MARKET_FACTOR, "Banks", "Insur", "Ships", "Aero", "Mach", "beta"]
    industries = ["Banks", "Insur", "Ships", "Aero", "Mach"]
    f = np.array([0.01, 0.004, -0.002, 0.05, 0.003, 0.001, 0.0])
    n = np.array([0, 40, 30, 1, 12, 20, 0])
    c = np.array([0.0, 0.5, 0.3, 0.004, 0.1, 0.096, 0.0])
    # constraint holds for the unshrunk f (make it so)
    ind = slice(1, 6)
    f[ind] -= (c[ind] * f[ind]).sum() / c[ind].sum()
    g, thin, m = shrink_thin_industries(f, cols, industries, n, c, min_members=5, prior_obs=5.0)
    assert thin == {"Ships": 1}
    manuf = [k for k in range(1, 6) if FF48_TO_FF12[cols[k]] == "Manuf"]
    f_parent = (c[manuf] * f[manuf]).sum() / c[manuf].sum()
    shrunk = f[3] / 6 + 5 / 6 * f_parent
    assert g[3] == pytest.approx(shrunk - m)
    assert g[0] == pytest.approx(f[0] + m)
    assert abs((c[ind] * g[ind]).sum()) < 1e-15
    # an empty FF48 industry takes its parent's return exactly (before the shift)
    n0 = n.copy()
    n0[3] = 0
    g0, thin0, m0 = shrink_thin_industries(f, cols, industries, n0, c, 5, 5.0)
    assert thin0 == {"Ships": 0}
    assert g0[3] == pytest.approx(f_parent - m0)
    # off switch
    g_off, thin_off, m_off = shrink_thin_industries(f, cols, industries, n, c, 0, 5.0)
    assert thin_off == {} and m_off == 0.0 and np.array_equal(g_off, f)


def test_ff48_to_ff12_is_a_complete_crosswalk():
    assert set(FF48_TO_FF12) == set(FF48_NAMES)
    assert set(FF48_TO_FF12.values()) == set(FF12_NAMES) and len(FF12_NAMES) == 12
    assert FF48_TO_FF12["Other"] == "Other"
    assert industry_parent("Banks") == "Money" and industry_parent("IND03") == ALL_INDUSTRIES
    assert industry_parent("Banks", {"Banks": "X"}) == "X"


# --------------------------------------------------------------------------- config
def test_config_bounds():
    with pytest.raises(ValueError):
        RiskConfig(specific_var_floor_frac=1.0)
    with pytest.raises(ValueError):
        RiskConfig(vra_min=2.0, vra_max=1.0)
    with pytest.raises(ValueError):
        RiskConfig(vra_z2_clip=0.0)
    with pytest.raises(ValueError):
        RiskConfig(thin_industry_min_members=-1)
    RiskConfig(**FIXES_OFF)  # the escape hatches are valid
