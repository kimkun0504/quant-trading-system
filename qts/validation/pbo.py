"""Probability of Backtest Overfitting via Combinatorially Symmetric Cross
Validation (CSCV; Bailey, Borwein, López de Prado & Zhu 2017).

Input: a ``T x N`` matrix ``M`` of period returns for ``N`` model
configurations observed over the same ``T`` periods (one column per trial of a
parameter sweep).  Procedure:

1. Partition the rows into ``S`` equal contiguous groups.
2. For each of the ``C(S, S/2)`` ways to pick ``S/2`` groups as in-sample
   (IS), the complement is out-of-sample (OOS).
3. Compute the performance metric per configuration IS and OOS, pick the
   best IS configuration ``n*``, and find its OOS relative rank
   ``w = rank(OOS_n*) / (N + 1)`` (rank 1 = worst).
4. Logit ``lambda = log(w / (1 - w))``.  ``PBO = P[lambda < 0]``, i.e. the
   probability that the IS-optimal configuration is strictly below-median OOS.

NaNs in the input raise (align the trials first); so do fewer than two
distinct configurations (PBO is undefined when every column is identical).

Also reported: the performance-degradation regression (OOS metric of the
IS-best configuration on its IS metric across all combinations), and the
probability that the IS-best configuration loses money OOS.

Everything is vectorized with per-group sufficient statistics (sum, sum of
squares, count), so the cost is ``O(C(S,S/2) * N)`` rather than
``O(C(S,S/2) * T * N)``.  Combinations are capped at ``max_combinations``
(default 5000) with uniform random subsampling and a warning.
"""
from __future__ import annotations

import warnings
from itertools import combinations
from math import comb
from typing import Dict, Optional

import numpy as np
import pandas as pd


def _group_stats(M: np.ndarray, n_groups: int):
    T = M.shape[0]
    edges = np.linspace(0, T, n_groups + 1).astype(int)
    s1 = np.stack([M[edges[g]:edges[g + 1]].sum(axis=0) for g in range(n_groups)])          # S x N
    s2 = np.stack([(M[edges[g]:edges[g + 1]] ** 2).sum(axis=0) for g in range(n_groups)])   # S x N
    cnt = np.array([edges[g + 1] - edges[g] for g in range(n_groups)], dtype=float)         # S
    return s1, s2, cnt


def _metric_from_stats(sum1: np.ndarray, sum2: np.ndarray, n: np.ndarray, metric: str) -> np.ndarray:
    """sum1/sum2: (C x N), n: (C,) -> metric per (combination, config)."""
    n = n[:, None]
    mean = sum1 / n
    var = np.maximum(sum2 / n - mean ** 2, 0.0) * n / np.maximum(n - 1, 1)
    sd = np.sqrt(var)
    if metric == "sharpe":
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(sd > 0, mean / sd, 0.0)
        return out
    if metric == "mean":
        return mean
    raise ValueError(f"unknown metric {metric!r}")


def probability_of_backtest_overfitting(perf_matrix: pd.DataFrame, n_groups: int = 16,
                                        metric: str = "sharpe", max_combinations: int = 5000,
                                        seed: int = 0) -> Dict[str, object]:
    """CSCV estimate of the probability of backtest overfitting.

    Returns a dict with ``pbo``, ``logits`` (array), ``degradation_slope``,
    ``degradation_intercept``, ``degradation_r2``, ``prob_oos_loss``,
    ``is_best_metric``/``oos_best_metric`` arrays and ``n_combinations``.
    """
    if isinstance(perf_matrix, pd.DataFrame):
        M = perf_matrix.to_numpy(dtype=float)
    else:
        M = np.asarray(perf_matrix, dtype=float)
    if M.ndim != 2 or M.shape[1] < 2:
        raise ValueError("perf_matrix must be T x N with N >= 2 configurations")
    if not np.isfinite(M).all():
        raise ValueError("perf_matrix has NaN/inf entries; align the trials on common dates first")
    T, N = M.shape
    # identical configurations carry no selection information: PBO is undefined
    n_distinct = np.unique(np.round(M, 12), axis=1).shape[1]
    if n_distinct < 2:
        raise ValueError("perf_matrix needs at least 2 distinct configurations (all columns identical)")
    if n_groups % 2:
        raise ValueError("n_groups must be even")
    if T < n_groups * 2:
        raise ValueError("too few observations for the requested number of groups")

    s1, s2, cnt = _group_stats(M, n_groups)
    half = n_groups // 2
    n_total = comb(n_groups, half)
    rng = np.random.default_rng(seed)
    if n_total > max_combinations:
        warnings.warn(f"CSCV: {n_total} combinations capped to {max_combinations} random subsamples",
                      RuntimeWarning, stacklevel=2)
        chosen = set()
        while len(chosen) < max_combinations:
            chosen.add(tuple(sorted(rng.choice(n_groups, half, replace=False).tolist())))
        combos = sorted(chosen)
    else:
        combos = list(combinations(range(n_groups), half))
    C = len(combos)
    mask = np.zeros((C, n_groups), dtype=bool)
    for i, c in enumerate(combos):
        mask[i, list(c)] = True

    is_s1, is_s2, is_n = mask @ s1, mask @ s2, mask @ cnt
    oos_mask = ~mask
    oos_s1, oos_s2, oos_n = oos_mask @ s1, oos_mask @ s2, oos_mask @ cnt
    is_metric = _metric_from_stats(is_s1, is_s2, is_n, metric)      # C x N
    oos_metric = _metric_from_stats(oos_s1, oos_s2, oos_n, metric)  # C x N

    best = np.argmax(is_metric, axis=1)                              # C
    rows = np.arange(C)
    oos_best = oos_metric[rows, best]
    is_best = is_metric[rows, best]
    # relative rank of the IS-best configuration among all configs OOS (1 = worst)
    rank = 1 + (oos_metric < oos_best[:, None]).sum(axis=1) + 0.5 * ((oos_metric == oos_best[:, None]).sum(axis=1) - 1)
    w = rank / (N + 1.0)
    logits = np.log(w / (1.0 - w))
    # strict: "IS-best is *below* the OOS median" (Bailey et al.); an exact
    # median rank (logit == 0) is not counted as overfit
    pbo = float(np.mean(logits < 0.0))

    # performance degradation: OOS metric of IS-best ~ a + b * IS metric
    if np.std(is_best) > 0:
        b, a = np.polyfit(is_best, oos_best, 1)
        resid = oos_best - (a + b * is_best)
        r2 = 1.0 - resid.var() / oos_best.var() if oos_best.var() > 0 else float("nan")
    else:
        a, b, r2 = float("nan"), float("nan"), float("nan")
    return {
        "pbo": pbo,
        "logits": logits,
        "n_combinations": C,
        "n_configs": N,
        "degradation_slope": float(b),
        "degradation_intercept": float(a),
        "degradation_r2": float(r2),
        "prob_oos_loss": float(np.mean(oos_best < 0.0)),
        "is_best_metric": is_best,
        "oos_best_metric": oos_best,
        "mean_oos_rank_pct": float(np.mean(w)),
    }
