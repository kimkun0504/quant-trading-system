"""Does a store prefilter load every name the universe can pick?  (look-ahead check)

``qts run --max-names N --min-mcap-ever X`` keeps the N ids with the largest *peak*
market cap over the whole load window.  A name that was in the point-in-time top
``top_n`` on some date but never grew enough to rank among the N largest peaks is not
loaded, so the universe on that date is chosen partly by how big names became later
(look-ahead).  It also makes the traded universe depend on the window's end: the
holdout seal shortens the window, a different set of N ids is loaded, and the sealed
re-run of the same configuration trades a different universe on the same dates.

This script measures, with DuckDB over the store (no panel is pivoted, a minute or
two per line on the full store), for each prefilter and window:

* the share of dates on which every one of the store's screened top-``top_n`` names
  (ranked by market cap among the names passing the universe's price / ADV screens,
  the ranks :func:`qts.data.universe.raw_eligibility` uses) is loaded;
* the worst date's coverage and the worst dates;
* whether enough screened names are loaded per date for the universe's bottom-size
  cut (a quantile of the loaded cross-section) to stay below rank ``top_n``.

The ``--max-rank K`` prefilter (load every id that ranked <= K by screened market cap
on at least one date) covers the universe by construction when K >= top_n /
(1 - exclude_bottom_size_quantile); it is printed for comparison.

Usage (repo root, venv active; reads market caps / prices / volume only):

    python scripts/check_load_coverage.py --store data/store --start 2000-01-01 \
        --max-names 3000 --min-mcap-ever 3e8 --ends full,2023-11-08 --max-rank 1500
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qts.core.config import SystemConfig
from qts.research.data import StoreSpec
from qts.research.midcap import prefilter_coverage


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--store", required=True)
    ap.add_argument("--config", default="configs/baseline.yaml", help="for the universe's screens and top_n")
    ap.add_argument("--start", default="2000-01-01")
    ap.add_argument("--max-names", type=int, default=3000)
    ap.add_argument("--min-mcap-ever", type=float, default=3e8)
    ap.add_argument(
        "--ends",
        default="full,2023-11-08",
        help="comma-separated window ends; 'full' = the store's last date (default: full and the seal's last date)",
    )
    ap.add_argument("--max-rank", type=int, default=None, help="also measure the --max-rank K prefilter")
    a = ap.parse_args(argv)

    cfg = SystemConfig.from_yaml(a.config) if Path(a.config).exists() else SystemConfig()
    u = cfg.universe
    top_n = int(u.rank_offset) + int(u.top_n_by_mcap)
    needed = math.ceil(top_n / max(1.0 - float(u.exclude_bottom_size_quantile), 1e-9))
    print(
        f"universe: screened top {top_n} (price >= {u.min_price:g}, ADV >= {u.min_adv_usd:g}), "
        f"bottom-size cut {u.exclude_bottom_size_quantile:g} -> needs >= {needed} loaded passing names per date\n"
    )
    specs = [
        (
            "peak cap",
            StoreSpec(store=a.store, start=a.start, max_names=a.max_names, min_mcap_ever=a.min_mcap_ever),
        )
    ]
    if a.max_rank is not None:
        specs.append(("screened rank", StoreSpec(store=a.store, start=a.start, max_rank=a.max_rank)))
    ok_all = True
    for end_s in [e.strip() for e in a.ends.split(",") if e.strip()]:
        end = None if end_s == "full" else end_s
        for label, spec in specs:
            rep = prefilter_coverage(spec.with_screens(u), rank=top_n, needed_loaded=needed, end=end)
            print(f"--- {label} prefilter, window {a.start} .. {end_s} ---")
            print(rep.format().replace("=== Load coverage (V2, checked before the run) ===\n", ""))
            print()
            ok_all = ok_all and rep.ok
    print("ALL COVERED" if ok_all else "SOME PREFILTER MISSES UNIVERSE NAMES (see above)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
