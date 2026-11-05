"""Oracle fairness of the synthetic generator (audit finding 8, WORK_PLAN 1.8).

The latent score revealed at ``t`` (``md.extra["true_latent_score"]``, also
leaked into the characteristics) must predict returns strictly *after* ``t``;
and its realised residual rank IC must sit inside the band documented in the
``qts.data.synthetic`` docstring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qts.alpha.evaluation import rank_ic_series
from qts.alpha.targets import forward_returns, residualize_forward_returns
from qts.data.synthetic import SyntheticSpec, generate


def _innovation(z: pd.DataFrame) -> pd.DataFrame:
    """Per-date cross-sectional OLS residual of z_t on z_{t-1}: the part of the
    score that becomes known *at* t."""
    zt, zl = z.to_numpy(dtype=float), z.shift(1).to_numpy(dtype=float)
    out = np.full(zt.shape, np.nan)
    for i in range(1, len(zt)):
        x, y = zl[i], zt[i]
        b = float(x @ y) / float(x @ x)
        out[i] = y - b * x
    return pd.DataFrame(out, index=z.index, columns=z.columns)


def _timing_ics(md):
    z = md.extra["true_latent_score"]
    e = _innovation(z)
    r = md.returns
    same = rank_ic_series(e, r).dropna()  # news at t vs r_t
    fwd = rank_ic_series(e, forward_returns(md, 20)).dropna()  # news at t vs r_{t+1..t+20}
    return same, fwd


@pytest.mark.parametrize("seed", [0, 1])
def test_score_at_t_predicts_only_after_t(seed):
    md = generate(SyntheticSpec(n_stocks=300, n_days=1512, seed=seed, ic_true=0.08))
    same, fwd = _timing_ics(md)
    t_same = same.mean() / (same.std() / np.sqrt(len(same)))
    t_fwd = fwd.mean() / (fwd.std() / np.sqrt(len(fwd)))
    print(
        f"\n[oracle timing seed {seed}] innovation IC same-day {same.mean():+.4f} (t {t_same:+.2f}), "
        f"next 20d {fwd.mean():+.4f} (t {t_fwd:+.2f})"
    )
    assert abs(t_same) < 3.0 and abs(same.mean()) < 0.005
    assert fwd.mean() > 0 and t_fwd > 3.0
    # persistence alone: the level of z_t predicts the next-day return at least as
    # well as the same-day return (legacy generator: same-day >= next-day)
    z = md.extra["true_latent_score"]
    r = md.returns
    assert rank_ic_series(z, r.shift(-1)).mean() >= rank_ic_series(z, r).mean() - 1e-4


def test_legacy_timing_is_detected():
    """Control: the same statistic flags the legacy contemporaneous generator
    (the score revealed on the first day of a month drives that day's return)."""
    md = generate(SyntheticSpec(n_stocks=300, n_days=1512, seed=0, ic_true=0.08, alpha_lag_days=0))
    same, _ = _timing_ics(md)
    month_start = same.iloc[np.flatnonzero(np.arange(len(same)) % 21 == 20)]  # innovation days
    assert month_start.mean() > 0.01


# documented in qts/data/synthetic.py: ic_true=0.04 -> 0.036 +- 0.006 over seeds 0-4
IC_BAND_PER_SEED = (0.020, 0.052)
IC_BAND_MEAN = (0.029, 0.043)


def test_planted_residual_ic_band():
    ics = []
    for seed in range(5):
        md = generate(SyntheticSpec(n_stocks=300, n_days=1512, seed=seed, ic_true=0.04))
        tgt = residualize_forward_returns(md, 20)
        ics.append(float(rank_ic_series(md.extra["true_latent_score"], tgt).mean()))
    print(
        "\n[planted IC] ic_true=0.04 residual 20d rank IC per seed",
        np.round(ics, 4),
        f"mean {np.mean(ics):.4f}",
    )
    assert all(IC_BAND_PER_SEED[0] <= x <= IC_BAND_PER_SEED[1] for x in ics), ics
    assert IC_BAND_MEAN[0] <= np.mean(ics) <= IC_BAND_MEAN[1]
    assert np.mean(ics) < 0.04  # never more than the planted ic_true
