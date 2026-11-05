"""Run the research pipeline on real data held in a ParquetStore.

Thin wrapper around ``python -m qts.cli run`` (same flags).

Expected datasets in the store (see qts/data/store.py docstring):
    prices_close, prices_volume, market_cap, industry, and optionally
    spread_bps, short_interest_ratio, borrow_fee, book_to_price,
    earnings_yield, gross_profitability, asset_growth.

Example (after loading Norgate/Sharadar panels into ./data/store):
    python scripts/run_real.py --store data/store --start 2006-01-01 --ml

Prototype with yfinance (survivorship-biased; for plumbing only):
    python scripts/run_real.py --yfinance AAPL MSFT ... --start 2015-01-01
"""

import sys

from qts.cli import main

if __name__ == "__main__":
    sys.exit(main(["run", *sys.argv[1:]]))
