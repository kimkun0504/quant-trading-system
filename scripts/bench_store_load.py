"""Memory and time of loading N ids from a ParquetStore (``qts run --store --max-names N``)
and of building the point-in-time universe on them: the measurement behind the
``qts check midcap`` default ``--max-names`` (docs/notes/midcap_check_2026-11-13.md).

Writes a synthetic market of ``--stocks`` names x ``--days`` days to a temporary store
(or uses ``--store``), then for each ``--names`` value loads the ``N`` largest ids by
peak market cap in a fresh subprocess and reports wall time and peak RSS of
``as_market_data`` and of ``build_universe_mask`` (top 1000 after skipping the first
1000 when N > 2000), plus ``bytes per (id x day)`` so the result extrapolates to the
real store (about 6,700 days from 2000-01-01)::

    python scripts/bench_store_load.py --stocks 5000 --days 1000 --names 1000 3000 5000
    python scripts/bench_store_load.py --store data/store --start 2000-01-01 --names 3000 5000

Only the standard library and the package are used; peak RSS is the child's ``VmHWM``
(``ru_maxrss`` where ``/proc`` is unavailable, i.e. macOS).
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

CHILD = r"""
import json, resource, sys, time

def peak_mb():
    # VmHWM is per address space (reset at exec); ru_maxrss survives exec on Linux
    try:
        for ln in open("/proc/self/status"):
            if ln.startswith("VmHWM:"):
                return int(ln.split()[1]) / 1024
    except OSError:
        pass
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 1024

from qts.core.config import UniverseConfig
from qts.data.store import ParquetStore
from qts.data.universe import build_universe_mask
store, start, n = sys.argv[1], (sys.argv[2] or None), int(sys.argv[3])
st = ParquetStore(store)
t0 = time.perf_counter()
ids = st.tickers_by_market_cap(start, None, max_names=n)
t1 = time.perf_counter()
md = st.as_market_data(universe_tickers=ids, start=start)
t2 = time.perf_counter()
rss_load = peak_mb()
off = 1000 if n > 2000 else 0
mask = build_universe_mask(md, UniverseConfig(top_n_by_mcap=1000, rank_offset=off))
t3 = time.perf_counter()
rss = peak_mb()
print(json.dumps({"names": len(ids), "days": len(md.dates), "prefilter_s": t1 - t0,
                  "load_s": t2 - t1, "universe_s": t3 - t2, "rss_load_mb": rss_load,
                  "rss_peak_mb": rss, "mean_universe": float(mask.sum(axis=1).mean()),
                  "rank_offset": off}))
"""


def _write_synthetic(root: Path, n_stocks: int, n_days: int) -> None:
    from qts.data.store import ParquetStore
    from qts.data.synthetic import SyntheticSpec, generate

    md = generate(SyntheticSpec(n_stocks=n_stocks, n_days=n_days, seed=0, n_delistings=n_stocks // 20))
    ParquetStore(str(root)).write_market_data(md)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--store", default=None, help="existing store (default: a synthetic one)")
    ap.add_argument("--start", default=None)
    ap.add_argument("--stocks", type=int, default=5000)
    ap.add_argument("--days", type=int, default=1000)
    ap.add_argument("--names", type=int, nargs="+", default=[1000, 3000, 5000])
    ap.add_argument("--target-days", type=int, default=6700, help="days to extrapolate to")
    a = ap.parse_args(argv)
    tmp = None
    root = a.store
    if root is None:
        tmp = tempfile.TemporaryDirectory()
        root = tmp.name
        t0 = time.perf_counter()
        _write_synthetic(Path(root), a.stocks, a.days)
        print(f"synthetic store {a.stocks} x {a.days} written in {time.perf_counter() - t0:.0f}s")
    print(
        f"{'names':>6} {'days':>6} {'prefilt s':>9} {'load s':>7} {'univ s':>7} {'RSS load MB':>11} "
        f"{'RSS peak MB':>11} {'B/(id*day)':>10} {f'GB @{a.target_days}d':>10} {'universe':>8}"
    )
    rows = []
    for n in a.names:
        out = subprocess.run(
            [sys.executable, "-c", CHILD, str(root), a.start or "", str(n)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        r = json.loads(out.strip().splitlines()[-1])
        rows.append(r)
        cells = max(r["names"] * r["days"], 1)
        bpc = r["rss_peak_mb"] * 2**20 / cells
        gb = bpc * r["names"] * a.target_days / 2**30
        print(
            f"{r['names']:6d} {r['days']:6d} {r['prefilter_s']:9.1f} {r['load_s']:7.1f} {r['universe_s']:7.1f} "
            f"{r['rss_load_mb']:11.0f} {r['rss_peak_mb']:11.0f} {bpc:10.0f} {gb:10.1f} {r['mean_universe']:8.0f}"
        )
    if len(rows) >= 2:
        # RSS = fixed interpreter / library cost + slope x (ids x days): the slope from the
        # smallest and largest load extrapolates to the real store's length
        lo, hi = rows[0], rows[-1]
        d_cells = hi["names"] * hi["days"] - lo["names"] * lo["days"]
        if d_cells > 0:
            slope = (hi["rss_peak_mb"] - lo["rss_peak_mb"]) * 2**20 / d_cells
            fixed = lo["rss_peak_mb"] * 2**20 - slope * lo["names"] * lo["days"]
            print(f"\npeak RSS ~ {fixed / 2**30:.2f} GB + {slope:.0f} B x ids x days")
            for r in rows:
                est = (fixed + slope * r["names"] * a.target_days) / 2**30
                print(f"  {r['names']:6d} ids x {a.target_days} days -> ~{est:.1f} GB peak (load + universe)")
    if tmp is not None:
        tmp.cleanup()
    _ = resource  # parent RSS not reported
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
