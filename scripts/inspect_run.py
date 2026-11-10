"""Inspect a finished run directory: where was the book empty, and what did the
risk model forecast around those stretches?

Prints (1) every stretch of consecutive days with gross exposure below 1e-4, with
the last day's trades before it (what was sold and how much), (2) a monthly table of
mean gross exposure, mean ex-ante vol, realised vol of the net return, the
realised / ex-ante ratio, mean |alpha| of held names (bp over the horizon) and the
number of held names, and (3) the exante vol on the first empty day versus the
last invested day of each stretch.

Usage::

    python scripts/inspect_run.py runs/<run_id>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

EMPTY_GROSS = 1e-4


def _stretches(mask: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    out: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []
    start = None
    prev = None
    for d, m in mask.items():
        if m and start is None:
            start = d
        if not m and start is not None:
            out.append((start, prev, int(mask.loc[start:prev].sum())))
            start = None
        prev = d
    if start is not None:
        out.append((start, prev, int(mask.loc[start:prev].sum())))
    return out


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__)
        return 2
    rd = Path(argv[0])
    w = pd.read_parquet(rd / "weights.parquet")
    t = pd.read_parquet(rd / "trades.parquet")
    ret = pd.read_parquet(rd / "returns.parquet")
    ev = pd.read_parquet(rd / "exante_vol.parquet")["exante_vol"]
    hold = pd.read_parquet(rd / "holdings.parquet")
    gross = w.abs().sum(axis=1)
    empty = gross < EMPTY_GROSS
    print(f"run {rd.name}: {len(gross)} days, empty on {empty.mean():.0%} of days")

    print("\n== empty stretches (>= 20 days) ==")
    for s, e, n in _stretches(empty):
        if n < 20:
            continue
        pos = gross.index.get_loc(s)
        last_inv = gross.index[pos - 1] if pos > 0 else None
        line = f"{s.date()} .. {e.date()}  {n:4d} days"
        if last_inv is not None:
            sold = t.loc[last_inv].abs().sum()
            ev_before = ev.reindex([last_inv]).iloc[0]
            ev_first = ev.reindex([s]).iloc[0]
            g_before = gross.loc[last_inv]
            line += (
                f"   last invested {last_inv.date()}: gross {g_before:.2f}, traded {sold:.2f} of NAV,"
                f" exante vol {ev_before:.1%} -> {ev_first:.1%} on first empty day"
            )
        print(line)

    print("\n== monthly: gross exposure, ex-ante vol, realised vol, ratio, held names, |alpha| bp ==")
    rv = ret["net"].rolling(21).std() * np.sqrt(252)
    hold["date"] = pd.to_datetime(hold["date"])
    hm = hold.groupby("date").agg(
        n_held=("ticker", "size"), abs_alpha_bp=("alpha", lambda a: 1e4 * np.abs(a).mean())
    )
    df = pd.DataFrame({"gross": gross, "exante_vol": ev, "realised_vol": rv}).join(hm, how="left")
    m = df.resample("QE").agg(
        gross=("gross", "mean"),
        exante_vol=("exante_vol", "mean"),
        realised_vol=("realised_vol", "mean"),
        n_held=("n_held", "mean"),
        abs_alpha_bp=("abs_alpha_bp", "mean"),
    )
    m["ratio"] = m["realised_vol"] / m["exante_vol"]
    with pd.option_context(
        "display.width", 160, "display.max_rows", 400, "display.float_format", "{:.3f}".format
    ):
        print(m.to_string())

    print("\n== ex-ante vol distribution on invested days ==")
    inv = ev[~empty.reindex(ev.index).fillna(True)]
    print(inv.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
