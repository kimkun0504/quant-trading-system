"""Run the research pipeline on real data held in a ParquetStore.

Thin wrapper around ``python -m qts.cli run`` (same flags).

Expected datasets in the store (see qts/data/store.py docstring):
    required:  prices_close, prices_volume, market_cap, industry
    strongly recommended for real data:
        prices_close_raw   as-traded close (price screens, dollar volume, ADV)
        shares_outstanding actual shares outstanding (consistent with prices_close_raw)
        industry_history   dated industry labels (else dates before the first
                           classification read "UNKNOWN")
        delisting_dates    ticker -> last trading date (tells delistings from
                           halts in labels and in the simulator; else the
                           63-day causal gap rule applies)
    optional:  spread_bps, short_interest_ratio, borrow_fee, book_to_price,
               earnings_yield, gross_profitability, asset_growth.

Example (after loading Norgate/Sharadar panels into ./data/store, e.g. with
qts.data.loaders.norgate_to_store and sharadar_sf1_to_store):
    python scripts/run_real.py --store data/store --start 2006-01-01 --ml

Prototype with yfinance (survivorship-biased; for plumbing only):
    python scripts/run_real.py --yfinance AAPL MSFT ... --start 2015-01-01
"""

import sys

from qts.cli import main

if __name__ == "__main__":
    sys.exit(main(["run", *sys.argv[1:]]))
