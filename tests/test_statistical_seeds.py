"""Seed-robust versions of statistical checks that used to hinge on one synthetic seed.

A threshold tuned on a single seed passes or fails with that seed's sampling
noise.  Here each check runs over several seeds and requires a majority, plus
a tighter bound on the across-seed mean (whose noise shrinks with sqrt(#seeds)).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from qts.alpha import rank_ic_series, residualize_forward_returns
from qts.data.synthetic import SyntheticSpec, generate

warnings.filterwarnings("ignore", category=RuntimeWarning)

SEEDS = (2, 3, 4)


@pytest.mark.statistical
def test_planted_truth_ic_majority_of_seeds():
    """The generator plants ``ic_true=0.04`` between the latent score and the
    residualised 20d forward return.  Per seed (200 names x 800 days) the mean
    rank IC has s.e. ~0.01; we require >= 2 of 3 seeds inside 0.04 +- 0.025
    and the 3-seed mean inside 0.04 +- 0.015 (residualisation attenuates
    the measured IC slightly below ``ic_true``)."""
    ics = {}
    for s in SEEDS:
        md = generate(SyntheticSpec(n_stocks=200, n_days=800, seed=s, ic_true=0.04))
        target = residualize_forward_returns(md, 20)
        ics[s] = float(rank_ic_series(md.extra["true_latent_score"], target).dropna().mean())
    print("\n[statistical] planted IC by seed:", {k: round(v, 4) for k, v in ics.items()})
    inside = [0.015 <= v <= 0.065 for v in ics.values()]
    assert sum(inside) >= 2, ics
    assert 0.025 <= np.mean(list(ics.values())) <= 0.055, ics
