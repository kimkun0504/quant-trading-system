"""End-to-end demo on the synthetic market (no data subscriptions needed).

Thin wrapper around ``python -m qts.cli demo`` (same flags):

    python scripts/run_demo.py                # phase 1: rank composite only
    python scripts/run_demo.py --ml           # phase 2: composite + walk-forward LightGBM/MLP/ridge
    python scripts/run_demo.py --stocks 500 --days 2520 --ml
    python scripts/run_demo.py --compare-lag  # D4: Sharpe at signal lag 0 vs 1
"""
import sys

from qts.cli import main

if __name__ == "__main__":
    sys.exit(main(["demo", *sys.argv[1:]]))
