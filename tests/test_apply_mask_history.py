"""``apply_mask`` keeps pre-entry history (audit 2 #6): trailing liquidity
features of a name entering the universe equal their true trailing values, the
cost model's ADV uses the full window, and features stay point-in-time."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from qts.alpha import FeatureBuilder
from qts.core.config import AlphaConfig, UniverseConfig
from qts.data.synthetic import SyntheticSpec, generate
from qts.data.universe import apply_mask, build_universe_mask
from qts.validation.leakage import pit_shift_test

warnings.filterwarnings("ignore", category=RuntimeWarning)

LIQ = ("turnover_63", "amihud_63", "log_dollar_vol_63", "log_mcap")


def _md():
    md = generate(SyntheticSpec(n_stocks=60, n_days=400, seed=7))
    mask = build_universe_mask(md, UniverseConfig(top_n_by_mcap=45))
    # force fresh entrants: five names join on day 200
    late = md.tickers[:5]
    mask.loc[md.dates[:200], late] = False
    mask.loc[md.dates[200:], late] = md.close.loc[md.dates[200:], late].notna()
    return md, mask, late


def test_new_entrant_features_equal_true_trailing_values():
    md, mask, late = _md()
    md_u = apply_mask(md, mask)
    fb = FeatureBuilder(AlphaConfig())
    f_u = fb.build(md_u)
    f_true = fb.build(md)  # computed on the full, unmasked panels
    elig = md_u.eligibility()
    for k in LIQ:
        a = f_u[k].where(elig)
        b = f_true[k].reindex_like(a).where(elig)
        pd.testing.assert_frame_equal(a, b, check_freq=False, obj=k)
        # the entrants have a value from their first eligible day (no 40-day NaN warm-up)
        first = md.dates[200]
        assert f_u[k].loc[first, late].notna().all(), k
    # processed features of the entrants are not median-filled zeros on entry
    X = fb.process(f_u, md_u)
    row = X.xs(md.dates[200], level="date").loc[late, "turnover_63"]
    assert (row != 0.0).all()


def test_cost_snapshot_adv_uses_full_window():
    from qts.tcost import TransactionCostModel

    md, mask, late = _md()
    md_u = apply_mask(md, mask)
    tc = TransactionCostModel()
    d = md.dates[205]
    snap_u = tc.snapshot(md_u, d)
    snap = tc.snapshot(md, d)
    np.testing.assert_allclose(snap_u.adv_usd.reindex(late), snap.adv_usd.reindex(late), rtol=1e-12)


def test_features_still_point_in_time_after_mask():
    md, mask, _ = _md()
    md_u = apply_mask(md, mask)
    fb = FeatureBuilder(AlphaConfig())
    for k in (*LIQ, "industry_momentum", "beta_252"):
        for cut in (md.dates[150], md.dates[210], md.dates[320]):
            assert pit_shift_test(lambda m, k=k: fb.build(m)[k], md_u, cut, raise_on_fail=True), (k, cut)
