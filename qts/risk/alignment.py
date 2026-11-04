"""Alpha alignment factor (Saxena & Stubbs, "Alpha Alignment Factor: A
Solution to the Underestimation of Risk for Optimized Active Portfolios",
Axioma research; see also Saxena & Stubbs 2012, J. Portfolio Management).

Mechanism
---------
Decompose the alpha into a component spanned by the risk-model exposures and
an orthogonal remainder:

    alpha = B b + alpha_perp,        y = alpha_perp / ||alpha_perp||.

The optimizer sees no systematic risk in ``alpha_perp`` and over-loads on it,
which is why optimized books realize bias statistics well above 1.  The fix
augments the covariance with a rank-one term

    Sigma_aug = B F B' + D + nu * y y'

which keeps the problem convex and (empirically) brings the bias statistic of
the optimized portfolio back toward 1.

Choice of ``nu``
----------------
Saxena-Stubbs leave ``nu`` as a calibration parameter (they tune it so the
bias statistic of the "alignment portfolio" is ~1).  Here ``nu`` is set so
that the alignment factor has the variance of a *typical style factor*:
``nu = nu_scale * median_k F_kk`` over the style factors, **after** rescaling
to the unit-norm vector ``y``.  Note that a style factor with unit-sd
exposures ``x`` has portfolio contribution ``F_kk (x'w)^2``, while the
alignment term contributes ``nu (y'w)^2`` with ``||y|| = 1``; to make the two
comparable we scale by ``N`` (``||x||^2 ≈ N`` for a standardized style):
``nu = nu_scale * N * median_style_var``.  If no factor covariance is
supplied the fallback is ``nu = nu_scale * median(specific_var)``, i.e. the
alignment factor is treated like one more idiosyncratic source (documented
fallback, less principled).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from qts.core.config import RiskConfig
from qts.risk.covariance import style_variances


def alpha_alignment(alpha: pd.Series, exposures: pd.DataFrame, specific_var: pd.Series,
                    nu_scale: float = 1.0, factor_cov: Optional[pd.DataFrame] = None,
                    cfg: Optional[RiskConfig] = None, weighted: bool = False
                    ) -> Tuple[pd.Series, float]:
    """Return ``(y, nu)``: unit-norm alignment vector orthogonal to the
    exposure columns.

    By default the regression is OLS (Euclidean projection), which is what
    Saxena-Stubbs use and what makes ``B'y = 0`` exact.  ``weighted=True``
    uses ``1/specific_var`` WLS weights (the "implied alpha" metric), in which
    case ``y`` is orthogonal to ``B`` in the weighted inner product instead.
    """
    cfg = cfg or RiskConfig()
    idx = alpha.dropna().index.intersection(exposures.index)
    a = alpha.loc[idx].values.astype(float)
    B = exposures.loc[idx].values.astype(float)
    keep = np.abs(B).sum(axis=0) > 0
    B = B[:, keep]
    if weighted:
        w = 1.0 / specific_var.reindex(idx).fillna(specific_var.median()).clip(lower=1e-12).values
        w = w / w.mean()
        sw = np.sqrt(w)[:, None]
        coef, *_ = np.linalg.lstsq(B * sw, a * sw[:, 0], rcond=None)
    else:
        coef, *_ = np.linalg.lstsq(B, a, rcond=None)
    resid = a - B @ coef
    norm = float(np.linalg.norm(resid))
    if norm < 1e-14:
        y = pd.Series(0.0, index=idx)
        return y.reindex(alpha.index).fillna(0.0), 0.0
    y = pd.Series(resid / norm, index=idx)
    N = len(idx)
    if factor_cov is not None:
        sv = style_variances(factor_cov, cfg)
        base = float(np.median(sv.values)) if len(sv) else float(np.median(np.diag(factor_cov.values)))
        nu = nu_scale * N * base
    else:
        nu = nu_scale * float(specific_var.reindex(idx).median())
    return y.reindex(alpha.index).fillna(0.0), float(nu)


def misalignment_gauge(alpha: pd.Series, exposures: pd.DataFrame) -> float:
    """``1 - R^2`` of alpha regressed on the exposures: the fraction of alpha
    variance that the risk model cannot see (BUILD_PLAN section 3 monitor)."""
    idx = alpha.dropna().index.intersection(exposures.index)
    a = alpha.loc[idx].values.astype(float)
    B = exposures.loc[idx].values.astype(float)
    coef, *_ = np.linalg.lstsq(B, a, rcond=None)
    resid = a - B @ coef
    tss = float(((a - a.mean()) ** 2).sum())
    return float((resid ** 2).sum() / tss) if tss > 0 else np.nan
