"""Calibrate the Saxena-Stubbs alignment variance scale ``nu_scale`` on the
production optimizer (WORK_PLAN item 1.8; audit risk_portfolio finding 5).

For each seed: build the market, fit the research pipeline (universe, hybrid
fundamental + APCA risk model, composite alpha without the ML sleeve, cost
model, ``SinglePeriodOptimizer`` with the default ``PortfolioConfig``), take the
last ``--n-dates`` out-of-sample dates and run
``qts.risk.calibrate_alignment_nu`` with a ``solve_fn`` closure over the
production optimizer.  The closure carries the previous day's book as
``w_current`` (so turnover/trade-aversion terms act as in the simulator) and
every ``nu_scale`` path starts from the same warm book (``--burn-in`` days solved
at ``nu_scale=1`` before the window).

    python scripts/calibrate_alignment_nu.py                       # synthetic, seeds 0 1 2
    python scripts/calibrate_alignment_nu.py --seeds 0 --n-dates 120
    python scripts/calibrate_alignment_nu.py --json runs/nu_cal.json

On real data replace ``generate(...)`` by the loaded ``MarketData`` (same code
path from ``ResearchPipeline.fit`` on) and use the median calibrated value as
``RiskConfig.alignment_nu_scale``.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import time
from typing import Dict

import numpy as np
import pandas as pd

from qts.backtest.simulator import DailyBarSimulator
from qts.core.config import SystemConfig
from qts.data.synthetic import SyntheticSpec, generate
from qts.pipeline import ResearchPipeline
from qts.risk.alignment import calibrate_alignment_nu
from qts.risk.evaluation import bias_statistic

GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


def calibrate_seed(seed: int, n_stocks: int, n_days: int, n_dates: int, burn_in: int, nav: float,
                   n_iter: int, tol: float, md=None) -> Dict[str, object]:
    t0 = time.time()
    md = md if md is not None else generate(SyntheticSpec(n_stocks=n_stocks, n_days=n_days, seed=seed))
    cfg = SystemConfig()
    cfg.universe.top_n_by_mcap = min(cfg.universe.top_n_by_mcap, md.close.shape[1])
    pipe = ResearchPipeline(cfg, use_ml=False, nav0=nav).fit(md)
    md_u, risk, opt = pipe.md_u, pipe.risk, pipe.optimizer
    assert md_u is not None and risk is not None and opt is not None
    t_fit = time.time() - t0

    oos = pipe.backtest_dates()
    if len(oos) < n_dates + burn_in:
        raise ValueError(f"only {len(oos)} out-of-sample dates; need {n_dates + burn_in}")
    dates = oos[-n_dates:]
    burn = oos[-(n_dates + burn_in):-n_dates]
    tickers = md_u.tickers
    alive_px = md_u.close.notna()
    elig = md_u.eligibility()
    alpha_fn = pipe._eligible_alpha_fn()
    sim = DailyBarSimulator(cfg, pipe.cost)
    neutral = [s for s in cfg.risk.styles if s in pipe.RISK_ONLY_STYLES]

    alpha_cache: Dict[pd.Timestamp, pd.Series] = {}
    snap_cache: Dict[pd.Timestamp, dict] = {}

    def alpha_at(d: pd.Timestamp) -> pd.Series:
        if d not in alpha_cache:
            out = alpha_fn(d)
            er = pd.Series(out.expected_return, dtype=float).reindex(tickers)
            ok = alive_px.loc[d] & elig.loc[d]
            er[~ok.to_numpy()] = 0.0
            alpha_cache[d] = er.fillna(0.0)
        return alpha_cache[d]

    def snap_at(d: pd.Timestamp) -> dict:
        if d not in snap_cache:
            snap_cache[d] = sim._snapshot(md_u, d)
        return snap_cache[d]

    def solve_once(d, rm, a, w_prev):
        alive = alive_px.loc[d].to_numpy()
        dead = list(tickers[~alive])
        # held names that left the eligible universe are exited (the simulator
        # unwinds them over cfg.portfolio.exit_ineligible_days; immediate here)
        inel = alive & ~elig.loc[d].to_numpy() & (w_prev.abs().to_numpy() > 1e-12)
        force = {tk: 0.0 for tk in tickers[inel]}
        res = opt.solve(alpha=a, risk=rm, w_current=w_prev, nav=nav, snapshot=snap_at(d),
                        neutral_styles=neutral, no_trade=dead or None, force_to_zero=force or None)
        if res.status not in ("optimal", "optimal_inaccurate"):
            return w_prev.copy(), res
        return res.w_target.reindex(tickers).fillna(0.0), res

    # ---- warm book: burn-in at nu_scale = 1
    w = pd.Series(0.0, index=tickers)
    for d in burn:
        a = alpha_at(d)
        w, _ = solve_once(d, risk.as_of(d, alpha=a, nu_scale=1.0), a, w)
    w_init = w.copy()

    # ---- stateful solve_fn: previous book carried, reset at the first date
    base_var: Dict[pd.Timestamp, float] = {}
    books: Dict[float, Dict[pd.Timestamp, pd.Series]] = {}
    shares: Dict[float, list] = {}
    state = {"w": w_init.copy(), "last": None}

    def solve_fn(d, rm, a):
        d = pd.Timestamp(d)
        if state["last"] is None or d <= state["last"]:
            state["w"] = w_init.copy()
        if d not in base_var:      # first call per date is the nu_scale=1 model (alignment_var unscaled)
            base_var[d] = float(risk_nu1[d].alignment_var)
        nu_s = round(float(rm.alignment_var) / base_var[d], 8) if base_var[d] > 0 else 0.0
        w_new, res = solve_once(d, rm, a, state["w"])
        state["w"], state["last"] = w_new, d
        books.setdefault(nu_s, {})[d] = w_new
        shares.setdefault(nu_s, []).append(float(getattr(res, "alignment_share", np.nan)))
        return w_new

    # precompute the nu_scale=1 risk models once (calibrate_alignment_nu also
    # calls as_of once per date; cache through a thin proxy so it is not repeated)
    risk_nu1: Dict[pd.Timestamp, object] = {}

    class _CachedRisk:
        def as_of(self, d, alpha=None, nu_scale=None):
            d = pd.Timestamp(d)
            if d not in risk_nu1:
                risk_nu1[d] = risk.as_of(d, alpha=alpha, nu_scale=1.0)
            return risk_nu1[d]

    t1 = time.time()
    res = calibrate_alignment_nu(_CachedRisk(), md_u, alpha_at, dates, solve_fn=solve_fn,
                                 n_iter=n_iter, tol=tol, grid=GRID)
    t_cal = time.time() - t1

    # ---- diagnostics per evaluated nu_scale
    rets = md_u.returns
    nxt = {d: md_u.dates[md_u.dates.get_loc(d) + 1] for d in dates}
    diag = {}
    for nu_s, bk in sorted(books.items()):
        pa, pu, real, gross, yw = [], [], [], [], []
        for d, wb in bk.items():
            rm = risk_nu1[d]
            rm_s = dataclasses.replace(rm, alignment_var=float(rm.alignment_var) * nu_s)
            pa.append(np.sqrt(max(rm_s.portfolio_variance(wb, include_alignment=True), 0.0)))
            pu.append(np.sqrt(max(rm_s.portfolio_variance(wb, include_alignment=False), 0.0)))
            real.append(float((wb * rets.loc[nxt[d]].reindex(wb.index).fillna(0.0)).sum()))
            gross.append(float(wb.abs().sum()))
            yv = rm.alignment_vector.reindex(wb.index).fillna(0.0)
            yw.append(float(yv @ wb))
        pa, pu, real = map(np.asarray, (pa, pu, real))
        diag[nu_s] = {
            "bias_aligned": float(bias_statistic(pd.Series(pa), pd.Series(real))),
            "bias_unaligned_forecast": float(bias_statistic(pd.Series(pu), pd.Series(real))),
            "exante_vol_ann": float(np.mean(pa) * np.sqrt(252)),
            "realised_vol_ann": float(np.std(real) * np.sqrt(252)),
            "gross": float(np.mean(gross)),
            "mean_abs_y_w": float(np.mean(np.abs(yw))),
            "alignment_share": float(np.nanmean(shares.get(nu_s, [np.nan]))),
            "realised_sharpe": float(np.mean(real) / np.std(real) * np.sqrt(252)) if np.std(real) > 0 else np.nan,
        }
    return {"seed": seed, "calibration": res, "diagnostics": diag, "n_dates": len(dates),
            "window": [str(dates[0].date()), str(dates[-1].date())], "fit_seconds": t_fit,
            "calibration_seconds": t_cal, "default_nu_formula": "nu = nu_scale * N * median style variance"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--stocks", type=int, default=300)
    ap.add_argument("--days", type=int, default=1512)
    ap.add_argument("--n-dates", type=int, default=250)
    ap.add_argument("--burn-in", type=int, default=30)
    ap.add_argument("--nav", type=float, default=1_000_000.0)
    ap.add_argument("--n-iter", type=int, default=8)
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--json", default=None, help="write the full results here")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    out = []
    for s in a.seeds:
        r = calibrate_seed(s, a.stocks, a.days, a.n_dates, a.burn_in, a.nav, a.n_iter, a.tol)
        out.append(r)
        c = r["calibration"]
        print(f"\nseed {s}: window {r['window'][0]}..{r['window'][1]} ({r['n_dates']} dates), "
              f"fit {r['fit_seconds']:.0f}s, calibration {r['calibration_seconds']:.0f}s")
        print("  grid (nu_scale -> bias):", ", ".join(f"{g:g}: {b:.3f}" for g, b in c["grid"]))
        print(f"  calibrated nu_scale = {c['nu_scale']:.3f} (bias {c['bias']:.3f}, band "
              f"[{c['lower']:.3f}, {c['upper']:.3f}], converged {c['converged']})")
        print("  history:", ", ".join(f"{n:.3g}:{b:.3f}" for n, b in c["history"]))
        print(f"  {'nu_scale':>9} {'bias':>6} {'bias_noal':>9} {'exante':>7} {'realised':>8} "
              f"{'|y.w|':>7} {'al_share':>8} {'SR':>6}")
        for nu_s, dg in r["diagnostics"].items():
            print(f"  {nu_s:9.3f} {dg['bias_aligned']:6.3f} {dg['bias_unaligned_forecast']:9.3f} "
                  f"{dg['exante_vol_ann']:7.3f} {dg['realised_vol_ann']:8.3f} {dg['mean_abs_y_w']:7.4f} "
                  f"{dg['alignment_share']:8.3f} {dg['realised_sharpe']:6.2f}")
    cal = [r["calibration"]["nu_scale"] for r in out]
    print(f"\ncalibrated nu_scale per seed: {np.round(cal, 3).tolist()}  median {np.median(cal):.3f}")
    if a.json:
        def conv(o):
            if isinstance(o, dict):
                return {str(k): conv(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [conv(v) for v in o]
            if isinstance(o, (np.floating, np.integer)):
                return o.item()
            return o
        with open(a.json, "w") as f:
            json.dump(conv(out), f, indent=1)


if __name__ == "__main__":
    main()
