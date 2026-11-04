"""Run the research pipeline on real data held in a ParquetStore.

Expected datasets in the store (see qts/data/store.py docstring):
    prices_close, prices_volume, market_cap, industry, and optionally
    spread_bps, short_interest_ratio, borrow_fee, book_to_price,
    earnings_yield, gross_profitability, asset_growth.

Example (after loading Norgate/Sharadar panels into ./data/store):
    python scripts/run_real.py --store data/store --start 2006-01-01 --ml

Prototype with yfinance (survivorship-biased; for plumbing only):
    python scripts/run_real.py --yfinance AAPL MSFT ... --start 2015-01-01
"""
import argparse
import logging

from qts.core.config import SystemConfig
from qts.pipeline import ResearchPipeline, format_result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=None, help="ParquetStore root")
    ap.add_argument("--yfinance", nargs="*", default=None, help="tickers for a yfinance prototype run")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--ml", action="store_true")
    ap.add_argument("--nav", type=float, default=5e6)
    ap.add_argument("--ledger", default="runs/trial_ledger.jsonl")
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO if a.v else logging.WARNING)

    if a.store:
        from qts.data.store import ParquetStore
        md = ParquetStore(a.store).as_market_data(start=a.start, end=a.end)
    elif a.yfinance:
        from qts.data.loaders import yfinance_to_market_data
        md = yfinance_to_market_data(a.yfinance, start=a.start or "2015-01-01", end=a.end)
    else:
        ap.error("give --store or --yfinance")

    from qts.data.universe import survivorship_check
    survivorship_check(md)
    cfg = SystemConfig()
    cfg.universe.top_n_by_mcap = min(cfg.universe.top_n_by_mcap, md.close.shape[1])
    res = ResearchPipeline(cfg, use_ml=a.ml, ledger_path=a.ledger, nav0=a.nav).run(md, progress=a.v)
    print(format_result(res))


if __name__ == "__main__":
    main()
