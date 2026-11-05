"""Scale benchmark of the research pipeline on the synthetic market.

Measures wall time per stage of ``ResearchPipeline(use_ml=False).run`` (phase-1
composite; ``--ml`` adds the walk-forward ML sleeve) for each requested
universe size x history length, the backtest seconds per out-of-sample day,
and -- with ``--cache`` -- the same run again with a warm stage cache
(``cache.enabled``; the second run loads the fitted risk and alpha models).
Then it extrapolates to the WORK_PLAN target (1,000 names x 15 years < 30 min):
fit stages scale linearly with the number of days, the backtest with the
number of out-of-sample days at the measured per-day cost.

Usage::

    python scripts/bench_scale.py --names 300 1000 --years 6 --max-days 120
    python scripts/bench_scale.py --names 1000 --years 15            # full run (long)
    python scripts/bench_scale.py --names 300 --years 6 --cache --json out.json

``--max-days N`` backtests only the last N out-of-sample days (the fit stages
still see the whole history), which bounds a scale point's run time; the
per-day cost is then extrapolated to the full out-of-sample span.  Timings are
wall-clock on whatever machine runs this (record the CPU count with them);
nothing is asserted.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TRADING_DAYS = 252
TARGET_NAMES, TARGET_YEARS, TARGET_MINUTES = 1000, 15, 30.0


def run_point(
    n_names: int, years: float, max_days: int | None, ml: bool, cache_root: str | None, seed: int
) -> dict[str, Any]:
    from qts.core.config import SystemConfig
    from qts.data.synthetic import SyntheticSpec, generate
    from qts.pipeline import ResearchPipeline

    n_days = round(years * TRADING_DAYS)
    t0 = time.perf_counter()
    md = generate(SyntheticSpec(n_stocks=n_names, n_days=n_days, seed=seed))
    t_gen = time.perf_counter() - t0
    ov: dict[str, Any] = {"universe.top_n_by_mcap": n_names}
    if cache_root:
        ov.update({"cache.enabled": True, "cache.root": cache_root})
    cfg = SystemConfig().with_overrides(ov)
    out: dict[str, Any] = {"names": n_names, "years": years, "n_days": n_days, "ml": ml, "generate": t_gen}
    passes = ["cold", "warm"] if cache_root else ["nocache"]
    for p in passes:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe = ResearchPipeline(cfg, use_ml=ml)
            t0 = time.perf_counter()
            res = pipe.run(md, max_days=max_days)
            wall = time.perf_counter() - t0
        n_bt = len(res.backtest.nav)
        n_oos = len(pipe.backtest_dates())
        log = cast(list[dict[str, Any]], res.backtest.diagnostics.get("solve_log", []) or [])
        rec: dict[str, Any] = {k: round(float(v), 3) for k, v in res.timings.items()}
        rec.update(
            wall=round(wall, 2),
            backtest_days=n_bt,
            oos_days_available=n_oos,
            backtest_sec_per_day=round(res.timings["backtest"] / max(n_bt, 1), 4),
            n_rebalances=len(log),
            n_convex_solves=sum(int(e.get("n_solves", 0) or 0) for e in log),
            cache=dict(getattr(pipe, "cache_info_", {})),
            sharpe_net=float(res.summary.get("sharpe_net", float("nan"))),
        )
        out[p] = rec
    return out


def extrapolate(point: dict[str, Any], target_years: float = TARGET_YEARS) -> dict[str, float]:
    """Minutes for ``point``'s universe at ``target_years`` of data: fit stages
    (data hash, universe, risk, alpha, evaluate) linear in days, backtest =
    per-day cost x out-of-sample days (target days minus this point's warm-up)."""
    r = point.get("nocache") or point["cold"]
    scale = target_years * TRADING_DAYS / point["n_days"]
    fit = sum(r.get(k, 0.0) for k in ("data_hash", "universe", "risk_fit", "alpha_fit", "evaluate")) * scale
    warmup = point["n_days"] - r["oos_days_available"]
    oos = target_years * TRADING_DAYS - warmup
    bt = r["backtest_sec_per_day"] * oos
    out = {"fit_min": fit / 60, "backtest_min": bt / 60, "total_min": (fit + bt) / 60, "oos_days": oos}
    if "warm" in point:
        w = point["warm"]
        fit_w = (
            sum(w.get(k, 0.0) for k in ("data_hash", "universe", "risk_fit", "alpha_fit", "evaluate")) * scale
        )
        out["total_min_warm_cache"] = (fit_w + w["backtest_sec_per_day"] * oos) / 60
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", type=int, nargs="+", default=[300, 1000])
    ap.add_argument("--years", type=float, nargs="+", default=[6.0])
    ap.add_argument("--max-days", type=int, default=None, help="backtest only the last N OOS days")
    ap.add_argument("--ml", action="store_true", help="phase 2 (walk-forward ML sleeve)")
    ap.add_argument("--cache", action="store_true", help="also time a warm-stage-cache re-run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None, help="write results here")
    a = ap.parse_args(argv)

    results: list[dict[str, Any]] = []
    env = {"cpus": os.cpu_count(), "python": platform.python_version(), "platform": platform.platform()}
    print(f"# bench_scale  {env}", flush=True)
    for n in a.names:
        for y in a.years:
            with tempfile.TemporaryDirectory(prefix="qts-bench-cache-") as tmp:
                p = run_point(n, y, a.max_days, a.ml, tmp if a.cache else None, a.seed)
            p["extrapolated_to_target_years"] = extrapolate(p)
            results.append(p)
            for k in ("nocache", "cold", "warm"):
                if k in p:
                    r = p[k]
                    print(
                        f"{n:5d} names x {y:4.1f} y [{k:7s}] risk_fit {r['risk_fit']:7.1f}s  alpha_fit "
                        f"{r['alpha_fit']:7.1f}s  backtest {r['backtest']:8.1f}s over {r['backtest_days']} d "
                        f"({r['backtest_sec_per_day']:.3f} s/d, {r['n_convex_solves']} convex solves)  "
                        f"total {r['total_run']:8.1f}s  cache {r['cache']}",
                        flush=True,
                    )
            e = p["extrapolated_to_target_years"]
            print(
                f"      -> {TARGET_YEARS} y of data ({e['oos_days']:.0f} OOS days): fit {e['fit_min']:.1f} min + "
                f"backtest {e['backtest_min']:.1f} min = {e['total_min']:.1f} min"
                + (
                    f" (warm cache {e['total_min_warm_cache']:.1f} min)"
                    if "total_min_warm_cache" in e
                    else ""
                ),
                flush=True,
            )
    if a.json:
        Path(a.json).write_text(json.dumps({"env": env, "results": results}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
