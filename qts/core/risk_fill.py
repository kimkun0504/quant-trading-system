"""Specific-variance fill for names outside the risk-model universe.

One rule shared by the optimizer (:mod:`qts.portfolio.optimizer`), the risk
model's own ``RiskModelOutput.covariance`` / ``portfolio_variance`` (the
simulator's ex-ante variance) and the risk decomposition: such a name gets zero
factor exposure and the ``OUT_OF_MODEL_SPECIFIC_VAR_PCTL`` (90th) percentile of
the finite modelled specific variances -- conservative, since a name the model
does not cover (typically a held name being exited) is more likely a small,
volatile one than a median one.  ``1e-4`` ((1 % daily vol)^2) when no finite
specific variance is known.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OUT_OF_MODEL_SPECIFIC_VAR_PCTL = 90.0
OUT_OF_MODEL_FALLBACK_VAR = 1e-4  # (1 % daily vol)^2 when no specific variance is known


def out_of_model_specific_var(specific_var: pd.Series) -> float:
    """Daily specific variance for names outside the risk-model universe: the
    ``OUT_OF_MODEL_SPECIFIC_VAR_PCTL`` percentile of the finite values of
    ``specific_var``, ``OUT_OF_MODEL_FALLBACK_VAR`` if there are none."""
    v = pd.Series(specific_var, dtype=float).to_numpy(dtype=np.float64)
    v = v[np.isfinite(v)]
    if not len(v):
        return OUT_OF_MODEL_FALLBACK_VAR
    return float(np.percentile(v, OUT_OF_MODEL_SPECIFIC_VAR_PCTL))


def fill_specific_var(specific_var: pd.Series, tickers: pd.Index) -> np.ndarray:
    """``specific_var`` aligned to ``tickers``; missing / non-finite entries get
    :func:`out_of_model_specific_var`."""
    s = pd.Series(specific_var, dtype=float)
    d = s.reindex(tickers).to_numpy(dtype=np.float64, copy=True)
    bad = ~np.isfinite(d)
    if bad.any():
        d[bad] = out_of_model_specific_var(s)
    return d
