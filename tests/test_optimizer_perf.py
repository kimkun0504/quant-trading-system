"""Optimizer daily-solve benchmark (work plan 1.6): solve time at N = 300 / 1000
/ 1500, with and without DPP problem reuse, and for the two vol-target modes.

Slow-marked: run with ``python -m pytest tests/test_optimizer_perf.py -m slow -s``.
Timings are *reported*, never asserted (wall-clock bounds flake on loaded CI
runners); the assertions are only that the paths agree to solver tolerance and
that the reuse path really re-solved the cached problem.

The risk model is a synthetic factor model of realistic shape (market + 48
industries + 10 styles, K = 59) so the benchmark isolates the optimizer from
the risk-model fit.  Each "day" perturbs alpha, the current book, the style
exposures, specific variances and the liquidity snapshot (same ticker set),
which is what the daily backtest loop does.

DPP reuse is measured only up to ``DPP_MAX_N``: cvxpy 1.9's DPP compile of this
problem needs O(N^2) memory (~0.9 GB at N=300, ~2.9 GB at N=600, more than a
7 GB box at N=1000 with the vol-cap SOC), which is why ``reuse_problem`` is off
by default (see ``PortfolioConfig.reuse_problem``).
"""

from __future__ import annotations

import statistics
import time

import numpy as np
import pandas as pd
import pytest

from qts.core.config import PortfolioConfig
from qts.core.roles import RISK_ONLY_STYLES
from qts.core.types import RiskModelOutput
from qts.portfolio import SimpleCostModel, SinglePeriodOptimizer

N_IND, STYLES = (
    48,
    [
        "beta",
        "size",
        "momentum",
        "value",
        "residual_vol",
        "liquidity",
        "short_term_reversal",
        "quality",
        "investment",
        "short_interest",
    ],
)


def synthetic_day(N: int, day: int, seed: int = 0):
    """``(alpha, risk, w0, snapshot)`` for one synthetic day; the ticker set is
    fixed by ``N`` and ``seed``, values move with ``day``."""
    base = np.random.default_rng(seed)
    rng = np.random.default_rng(seed * 1000 + day + 1)
    tickers = pd.Index([f"T{i:05d}" for i in range(N)])
    inds = [f"Ind{j:02d}" for j in range(N_IND)]
    ind_of = base.integers(0, N_IND, N)
    D = np.zeros((N, N_IND))
    D[np.arange(N), ind_of] = 1.0
    S0 = base.standard_normal((N, len(STYLES)))
    S = S0 + 0.05 * rng.standard_normal(S0.shape)
    B = pd.DataFrame(np.column_stack([np.ones(N), D, S]), index=tickers, columns=["market", *inds, *STYLES])
    K = B.shape[1]
    A = base.standard_normal((K, K)) * 0.004
    F = A @ A.T / K + np.diag(np.full(K, 1e-5))
    F *= 1.0 + 0.02 * rng.standard_normal()
    F = pd.DataFrame(F, index=B.columns, columns=B.columns)
    spec = pd.Series(
        (0.015 * np.exp(0.3 * base.standard_normal(N)) * (1 + 0.02 * rng.standard_normal(N))) ** 2,
        index=tickers,
    )
    risk = RiskModelOutput(
        date=pd.Timestamp("2024-01-02") + pd.Timedelta(days=day),
        exposures=B,
        factor_cov=F,
        specific_var=spec,
        factor_returns=pd.DataFrame(columns=B.columns),
        factor_groups={
            "market": ["market"],
            "industry": inds,
            "style": STYLES,
            "statistical": [],
            "alignment": [],
        },
    )
    z = base.standard_normal(N) + 0.3 * rng.standard_normal(N)
    alpha = pd.Series(0.01 * z * np.sqrt(spec.values * 20) / 0.015, index=tickers)
    w0 = pd.Series(np.clip(0.004 * base.standard_normal(N), -0.01, 0.01), index=tickers)
    adv = pd.Series(np.exp(base.normal(17, 1, N)) * (1 + 0.05 * rng.standard_normal(N)), index=tickers)
    snap = {
        "adv_usd": adv.clip(lower=1e5),
        "sigma_daily": pd.Series(np.sqrt(spec.values) * 1.3, index=tickers),
        "half_spread": pd.Series(np.full(N, 3e-4), index=tickers),
        "price": pd.Series(np.exp(base.normal(4, 0.6, N)), index=tickers),
        "borrow_fee": pd.Series(np.full(N, 0.005), index=tickers),
    }
    return alpha, risk, w0, snap


DPP_MAX_N = 300


def run_days(N: int, reuse: bool, n_days: int = 6, cfg: PortfolioConfig | None = None) -> dict[str, object]:
    cfg = cfg or PortfolioConfig()
    opt = SinglePeriodOptimizer(cfg, SimpleCostModel(), reuse_problem=reuse)
    times: list[float] = []
    books = []
    for day in range(n_days):
        alpha, risk, w0, snap = synthetic_day(N, day)
        t0 = time.perf_counter()
        res = opt.solve(alpha, risk, w0, 1e8, snap, neutral_styles=list(RISK_ONLY_STYLES))
        times.append(time.perf_counter() - t0)
        assert res.ok, res.status
        books.append(res.w_target)
    return {"first": times[0], "steady": statistics.median(times[1:]), "books": books, "opt": opt}


@pytest.mark.slow
@pytest.mark.parametrize("N", [300, 1000, 1500])
def test_daily_solve_time(N):
    no_vt = PortfolioConfig(target_vol_annual=None)
    fresh = run_days(N, reuse=False, cfg=no_vt)
    line = f"\n[benchmark] N={N}: alpha solve {fresh['steady']:.3f}s/day (rebuild, default)"
    if N <= DPP_MAX_N:
        reused = run_days(N, reuse=True, cfg=no_vt)
        assert reused["opt"].n_reuses >= 4 and reused["opt"].n_builds == 1
        for a, b in zip(fresh["books"], reused["books"], strict=True):
            assert float((a - b).abs().max()) < 1e-5
        line += (
            f"; DPP reuse first {reused['first']:.2f}s, steady {reused['steady']:.3f}s/day "
            f"(x{fresh['steady'] / reused['steady']:.2f})"
        )
    proj = run_days(N, reuse=False, cfg=PortfolioConfig(target_vol_annual=0.10, vol_target_mode="projection"))
    cons = run_days(N, reuse=False, cfg=PortfolioConfig(target_vol_annual=0.10, vol_target_mode="constraint"))
    line += (
        f"; with vol target: projection {proj['steady']:.3f}s/day, constraint {cons['steady']:.3f}s/day "
        f"(x{proj['steady'] / cons['steady']:.2f})"
    )
    print(line)
