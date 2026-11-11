"""Find the absurd daily returns in a ParquetStore (or the synthetic market).

Why: on real data a handful of bad prints (restated adjusted closes, sub-penny
prices, one-sided split adjustments) enter the risk model's cross-sectional
regression and blow up the factor covariance for years
(docs/notes/risk_outlier_fix_2026-11-10.md).  This script shows *which* cells
they are, so the owner can see the culprits before and after a rerun.

Prints

* the top-``--top`` |daily return| cells (from ``prices_close`` = adjusted
  close): id, date, r, raw close (t-1, t), adjusted close (t-1, t), volume,
  market cap, eligible that day (``UniverseConfig``), and a *likely cause* tag
  (``sub-penny`` prev raw close < 1 USD; ``restatement`` raw return sane but
  adjusted return absurd; ``split-like`` adjusted ratio within 2 % of a round
  factor; ``delisting-day`` last close of the name; ``gap`` return spans a
  halt; ``unknown``);
* counts of |r| > 0.5 / 1 / 3 / 10 per year, and how many of those cells the
  rule-(a) cleaning (``RiskConfig`` caps) and the price floor would remove;
* the dates where the cross-sectional sqrt-cap-weighted mean |r| spikes
  (> ``--spike-mult`` x its trailing-252-day median), with the top contributor;
* per-date max |r| (and its id) inside the windows around the known empty
  stretches of the real run (``--windows`` to override).

Usage (same data flags as ``qts run``)::

    python scripts/find_outlier_returns.py --store data/store --start 2000-01-01 \\
        --max-names 3000 --min-mcap-ever 3e8
    python scripts/find_outlier_returns.py --synthetic      # smoke test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_signal import add_data_args, load_config, load_market
from qts.core.returns import data_error_mask
from qts.core.types import MarketData
from qts.data.universe import build_universe_mask

DEFAULT_WINDOWS = [
    ("2003-09-02", "2003-09-05"),
    ("2008-03-03", "2008-03-06"),
    ("2010-06-01", "2010-06-04"),
    ("2011-12-01", "2011-12-06"),
    ("2014-07-03", "2014-07-09"),
    ("2018-03-01", "2018-03-07"),
    ("2023-05-02", "2023-05-05"),
    ("2025-06-02", "2025-06-05"),
]
ROUND_FACTORS = np.array([2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 25, 30, 40, 50, 100, 1000], dtype=float)


def _likely_cause(
    r: float, raw_prev: float, raw_t: float, adj_prev: float, adj_t: float, gap: int, last_day: bool
) -> str:
    tags: list[str] = []
    if last_day:
        tags.append("delisting-day")
    if np.isfinite(raw_prev) and raw_prev < 1.0:
        tags.append("sub-penny")
    if np.isfinite(raw_prev) and np.isfinite(raw_t) and raw_prev > 0:
        raw_r = raw_t / raw_prev - 1.0
        if abs(raw_r) < 0.5 and abs(r) > 1.0:
            tags.append("restatement")
    if np.isfinite(adj_prev) and adj_prev > 0 and np.isfinite(adj_t) and adj_t > 0:
        ratio = adj_t / adj_prev
        cand = np.concatenate([ROUND_FACTORS, 1.0 / ROUND_FACTORS])
        if np.any(np.abs(ratio / cand - 1.0) < 0.02):
            tags.append("split-like")
    if gap > 1:
        tags.append(f"gap({gap}d)")
    return ",".join(tags) if tags else "unknown"


def _fmt(df: pd.DataFrame) -> str:
    with pd.option_context(
        "display.width",
        250,
        "display.max_columns",
        50,
        "display.max_rows",
        500,
        "display.float_format",
        "{:.4g}".format,
    ):
        return df.to_string(index=False)


def outlier_table(md: MarketData, elig: pd.DataFrame, top: int) -> pd.DataFrame:
    rets = md.returns
    R = rets.to_numpy(dtype=float)
    flat = np.abs(np.where(np.isfinite(R), R, 0.0)).ravel()
    k = min(top, int((flat > 0).sum()))
    order = np.argpartition(-flat, k - 1)[:k] if k else np.array([], dtype=int)
    order = order[np.argsort(-flat[order])]
    close = md.close
    raw = md.close_raw if md.close_raw is not None else md.close
    have = close.notna().to_numpy()
    pos = pd.DataFrame(np.where(have, np.arange(len(md.dates))[:, None], np.nan)).ffill().to_numpy()
    last_valid = close.apply(lambda s: s.last_valid_index())
    rows = []
    for flat_i in order:
        i, j = divmod(int(flat_i), R.shape[1])
        tk, d = md.tickers[j], md.dates[i]
        prev_i = int(pos[i - 1, j]) if i > 0 and np.isfinite(pos[i - 1, j]) else i - 1
        gap = i - prev_i
        rows.append(
            {
                "id": tk,
                "date": d.date(),
                "r": float(R[i, j]),
                "raw_prev": float(raw.iat[prev_i, j]) if prev_i >= 0 else np.nan,
                "raw_t": float(raw.iat[i, j]),
                "adj_prev": float(close.iat[prev_i, j]) if prev_i >= 0 else np.nan,
                "adj_t": float(close.iat[i, j]),
                "volume": float(md.volume.iat[i, j]),
                "mcap_musd": float(md.market_cap.iat[i, j]) / 1e6,
                "eligible": bool(elig.iat[i, j]),
                "cause": _likely_cause(
                    float(R[i, j]),
                    float(raw.iat[prev_i, j]) if prev_i >= 0 else np.nan,
                    float(raw.iat[i, j]),
                    float(close.iat[prev_i, j]) if prev_i >= 0 else np.nan,
                    float(close.iat[i, j]),
                    gap,
                    last_valid.iloc[j] == d,
                ),
            }
        )
    return pd.DataFrame(rows)


def yearly_counts(md: MarketData, cfg_risk) -> pd.DataFrame:
    rets = md.returns
    a = rets.abs()
    out = pd.DataFrame(
        {
            ">0.5": (a > 0.5).sum(axis=1),
            ">1": (a > 1.0).sum(axis=1),
            ">3": (a > 3.0).sum(axis=1),
            ">10": (a > 10.0).sum(axis=1),
        }
    )
    caps = data_error_mask(
        rets,
        None,
        cap_up=cfg_risk.return_abs_cap_up,
        cap_down=cfg_risk.return_abs_cap_down,
        min_prev_price=None,
    )
    floor = data_error_mask(
        rets,
        md.close_raw,
        cap_up=np.inf,
        cap_down=-1.0,
        min_prev_price=cfg_risk.return_min_prev_price,
    )
    out["rule_a_caps"] = caps.sum(axis=1)
    out["rule_a_price_floor"] = floor.sum(axis=1)
    out["n_returns"] = rets.notna().sum(axis=1)
    return out.groupby(out.index.year).sum()


def cross_sectional_spikes(md: MarketData, elig: pd.DataFrame, mult: float, n: int = 25) -> pd.DataFrame:
    rets = md.returns.where(elig)
    w = np.sqrt(md.market_cap.where(elig).clip(lower=0.0)).where(rets.notna())
    w = w.div(w.sum(axis=1).replace(0, np.nan), axis=0)
    m = (rets.abs() * w).sum(axis=1, min_count=1)
    med = m.rolling(252, min_periods=60).median().shift(1)
    ratio = m / med
    spikes = ratio[ratio > mult].sort_values(ascending=False).head(n)
    rows = []
    for d, x in spikes.items():
        row = rets.loc[d].abs()
        tk = row.idxmax()
        rows.append(
            {
                "date": d.date(),
                "xs_mean_abs_r": float(m[d]),
                "vs_trailing_median": float(x),
                "max_abs_r": float(row.max()),
                "max_id": tk,
                "n_abs_r>1": int((row > 1).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("date") if rows else pd.DataFrame()


def window_maxima(md: MarketData, elig: pd.DataFrame, windows) -> pd.DataFrame:
    rets = md.returns
    rows = []
    for a, b in windows:
        sub = rets.loc[a:b]
        for d, row in sub.iterrows():
            ar = row.abs()
            if ar.notna().any():
                tk = ar.idxmax()
                e = ar.where(elig.loc[d]).max()
                rows.append(
                    {
                        "window": f"{a}..{b}",
                        "date": d.date(),
                        "max_abs_r": float(ar.max()),
                        "id": tk,
                        "r": float(row[tk]),
                        "eligible": bool(elig.at[d, tk]),
                        "max_abs_r_eligible": float(e) if np.isfinite(e) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(ap)
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--spike-mult", type=float, default=5.0)
    ap.add_argument("--windows", nargs="*", help="START..END pairs (default: the known empty stretches)")
    a = ap.parse_args(argv)
    cfg = load_config(a)
    md = load_market(a)
    if md is None:
        print("give --store or --synthetic", file=sys.stderr)
        return 2
    print(
        f"loaded: {md.close.shape[1]} ids x {md.close.shape[0]} days ({md.dates[0].date()}..{md.dates[-1].date()})"
    )
    elig = (
        build_universe_mask(md, cfg.universe)
        .reindex(index=md.dates, columns=md.tickers)
        .fillna(False)
        .astype(bool)
    )
    elig &= md.close.notna()

    print(f"\n== top {a.top} |daily return| (adjusted close) ==")
    print(_fmt(outlier_table(md, elig, a.top)))

    print(
        "\n== counts per year: cells with |r| above thresholds; rule-(a) removals at the configured caps =="
    )
    print(
        f"(caps: r > {cfg.risk.return_abs_cap_up:+.2f} or r < {cfg.risk.return_abs_cap_down:+.2f}; "
        f"price floor: previous raw close < {cfg.risk.return_min_prev_price} USD)"
    )
    print(yearly_counts(md, cfg.risk).to_string())

    print(
        f"\n== cross-sectional sqrt-cap-weighted mean |r| spikes (> {a.spike_mult}x trailing-252d median), eligible names =="
    )
    sp = cross_sectional_spikes(md, elig, a.spike_mult)
    print(_fmt(sp) if len(sp) else "(none)")

    windows = DEFAULT_WINDOWS if not a.windows else [tuple(w.split("..")) for w in a.windows]
    print("\n== per-date max |r| inside the windows around the empty stretches ==")
    wm = window_maxima(md, elig, windows)
    print(_fmt(wm) if len(wm) else "(no dates in the panel)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
