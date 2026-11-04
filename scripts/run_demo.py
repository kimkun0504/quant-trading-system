"""End-to-end demo on the synthetic market (no data subscriptions needed).

    python scripts/run_demo.py                # phase 1: rank composite only
    python scripts/run_demo.py --ml           # phase 2: composite + walk-forward LightGBM/ridge
    python scripts/run_demo.py --stocks 500 --days 2520 --ml
"""
import argparse
import logging
import time

from qts.core.config import SystemConfig
from qts.data.synthetic import SyntheticSpec, generate
from qts.pipeline import ResearchPipeline, format_result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stocks", type=int, default=300)
    ap.add_argument("--days", type=int, default=1512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ml", action="store_true", help="include the ML sleeve (phase 2)")
    ap.add_argument("--rebalance-every", type=int, default=1)
    ap.add_argument("--ledger", default="runs/trial_ledger.jsonl")
    ap.add_argument("--nav", type=float, default=5e6)
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO if a.v else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    t0 = time.time()
    md = generate(SyntheticSpec(n_stocks=a.stocks, n_days=a.days, seed=a.seed))
    print(f"synthetic market: {md.close.shape[1]} stocks x {md.close.shape[0]} days  ({time.time()-t0:.1f}s)")

    cfg = SystemConfig()
    cfg.universe.top_n_by_mcap = min(cfg.universe.top_n_by_mcap, a.stocks)
    pipe = ResearchPipeline(cfg, use_ml=a.ml, ledger_path=a.ledger, rebalance_every=a.rebalance_every, nav0=a.nav)
    res = pipe.run(md, progress=a.v)
    print(format_result(res))


if __name__ == "__main__":
    main()
