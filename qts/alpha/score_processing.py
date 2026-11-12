"""Score processing between the combined score and Grinold scaling: cross-sectional
neutralisation and causal smoothing (``AlphaConfig.neutralize_score``,
``AlphaConfig.score_smoothing_halflife_days``).

Why (docs/notes/construction_fix_2026-11-12.md)
-----------------------------------------------
The optimizer bands industries, beta and the risk-only styles.  A score that still
loads on them asks the optimizer for a book it is not allowed to hold: on the real
ML run the style bands sat at their limits on 48-80 % of days and the industry band
on every day, and the book that came out had a correlation of 0.61 with the paper
book built from the *residual* of the score (gross Sharpe 0.20 vs 1.04).  Removing
the hedged directions from the score itself means Grinold scaling, the trailing IC
and the optimizer all see the part of the signal the book can express.

Neutralisation (per date ``t``, point in time)
----------------------------------------------
Over the names with a score on ``t`` (the eligible names), ordinary least squares
of the score on an intercept + industry dummies + beta + the risk-only styles
(``qts.core.roles.RISK_ONLY_STYLES``) **as of** ``t`` -- the same exposure source
the labels are residualised on: ``exposures_fn(t)`` (the fitted risk model's
market / industry / risk-only columns) and, on dates before the risk model has
exposures, the alpha layer's own proxies (:func:`qts.alpha.targets.default_exposures`).
The residual is re-standardised to unit cross-sectional sd.  The regression is
**equal-weighted** on purpose: it reproduces the paper book (c) of
``scripts/attribute_run.py`` (the evidence the default rests on), and the
optimizer's bands are plain sums ``B'w`` with no cap weighting.  A sqrt-cap WLS
would leave an equal-weighted exposure in the residual.  Names the risk model does
not cover have zero exposures (only the intercept is removed), as in the labels.
A date with fewer than :data:`MIN_NAMES` scored names is left as it is.

Smoothing (causal, per name)
----------------------------
An exponentially weighted mean over trading days of each name's neutralised score,
with weights normalised over the days the name actually had a score::

    num_t = lam * num_{t-1} + r_t        den_t = lam * den_{t-1} + 1     (r_t finite)
    num_t = lam * num_{t-1}              den_t = lam * den_{t-1}         (no score on t)
    s_t   = num_t / den_t   for names with a score on t,   lam = 0.5 ** (1 / halflife)

so a name entering the universe starts at its own first score (no ramp from zero),
a name that was away for a while has its old scores discounted by the whole gap,
and ``s_t`` depends on scores ``<= t`` only.  The smoothed cross-section is
projected on the exposures of ``t`` once more (old days were neutral to *their*
exposures, and reversal or beta loadings move) and re-standardised.  Smoothing
slows the score down to the pace the horizon needs and blends across the ML
sleeve's refit days, where the raw score jumps (refit-day turnover 37 % vs 13 %
on the real run).  ``halflife = 0`` switches it off.

Both steps are functions of data ``<= t`` row by row, so the output panel is point
in time by construction (``tests/test_alpha.py`` perturbs the future and checks).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from qts.core.types import MarketData, Wide

from .targets import ExposuresFn, default_exposures, internal_design

#: fewer scored names on a date -> the row is not neutralised (kept as it is)
MIN_NAMES = 20
#: singular values below this fraction of the largest are treated as zero (collinear
#: columns: the risk model's unit market column duplicates the intercept, an industry
#: without a scored member is an all-zero column)
_RANK_TOL = 1e-10

__all__ = ["MIN_NAMES", "ScoreProcessing", "process_scores", "smoothing_decay"]


@dataclass(frozen=True, eq=False)
class ScoreProcessing:
    """Output of :func:`process_scores`.

    ``score``: the processed panel (date x ticker), unit cross-sectional sd, NaN
    where the input had no score.  ``r2``: per date, the share of the input score's
    cross-sectional variance that the exposures explained (NaN where the row was
    not neutralised) -- how much of the raw signal was a hedged direction."""

    score: Wide
    r2: pd.Series


def smoothing_decay(halflife_days: float) -> float:
    """Per-day decay ``lam = 0.5 ** (1 / halflife)``; ``0`` for ``halflife <= 0`` (off)."""
    hl = float(halflife_days)
    return float(0.5 ** (1.0 / hl)) if hl > 0 else 0.0


def _unit(v: np.ndarray) -> np.ndarray:
    """Demean and scale to unit sd (ddof 0); a constant vector becomes all-NaN."""
    v = v - v.mean()
    sd = v.std()
    return v / sd if sd > 0 else np.full_like(v, np.nan)


def _projector(X: np.ndarray) -> np.ndarray:
    """Orthonormal basis ``U`` of the column space of ``X`` (thin SVD, numerically
    zero directions dropped): the least-squares residual of ``y`` is ``y - U (U'y)``."""
    U, sv, _ = np.linalg.svd(X, full_matrices=False)
    if not len(sv) or sv[0] <= 0:
        return U[:, :0]
    return U[:, sv > _RANK_TOL * sv[0]]


def process_scores(
    md: MarketData,
    score: Wide,
    *,
    neutralize: bool,
    halflife_days: float,
    exposures_fn: ExposuresFn | None = None,
    features: dict[str, Wide] | None = None,
) -> ScoreProcessing:
    """Neutralise and smooth a score panel (module docstring).  ``score`` is indexed
    like ``md.close`` (rows may be a subset of ``md.dates``); ``exposures_fn(date)``
    returns the ticker x factor exposures as of ``date`` or ``None`` (then, and when
    no function is given, the built-in risk-only proxies are used -- built from
    ``features`` when supplied).  With both steps off the input is returned
    re-standardised."""
    dates, tickers = md.dates, md.tickers
    Y = score.reindex(columns=tickers).to_numpy(dtype=float)
    rows = dates.get_indexer(score.index)
    if (rows < 0).any():
        raise ValueError("score rows must be dates of md")
    lam = smoothing_decay(halflife_days)
    smooth = lam > 0
    out = np.full(Y.shape, np.nan)
    r2 = np.full(len(Y), np.nan)
    D_of_row: dict[int, np.ndarray] = {}
    S = np.zeros((len(dates), len(tickers), 0))
    if neutralize:
        D_of_row, S = internal_design(md, default_exposures(md, features))
    ones = np.ones((len(tickers), 1))
    num = np.zeros(len(tickers))
    den = np.zeros(len(tickers))
    for k, row_pos in enumerate(rows):
        i = int(row_pos)
        y = Y[k]
        m = np.isfinite(y)
        if smooth:  # decay every day, whether or not the name has a score today
            num *= lam
            den *= lam
        n = int(m.sum())
        if n == 0:
            continue
        v = y[m]
        U = None
        if neutralize and n >= MIN_NAMES:
            X = None if exposures_fn is None else exposures_fn(dates[i])
            if X is not None and len(X):
                Xm = np.hstack([ones[m], X.reindex(tickers).fillna(0.0).to_numpy(dtype=float)[m]])
            else:
                Xm = np.hstack([ones[m], D_of_row[i][m], S[i][m]])
            U = _projector(Xm)
            dev = v - v.mean()
            tot = float(dev @ dev)
            v = v - U @ (U.T @ v)
            r2[k] = 1.0 - float(v @ v) / tot if tot > 0 else np.nan
        v = _unit(v)
        if not np.isfinite(v).all():  # a constant cross-section carries no score
            continue
        if smooth:
            idx = np.flatnonzero(m)
            num[idx] += v
            den[idx] += 1.0
            sm = num[idx] / den[idx]
            if U is not None:  # neutral to today's exposures, not only to each past day's
                sm = sm - U @ (U.T @ sm)
            v = _unit(sm)
        out[k, m] = v
    panel = pd.DataFrame(out, index=score.index, columns=tickers)
    panel.index.name, panel.columns.name = score.index.name, score.columns.name
    return ScoreProcessing(score=panel, r2=pd.Series(r2, index=score.index, name="neutralization_r2"))
