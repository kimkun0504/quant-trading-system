"""Signal diagnostics on a ParquetStore (or the synthetic market): why is the book empty?

Fits the pipeline exactly as ``qts run`` does (stage cache reused when enabled) and
prints, without running the backtest:

* eligible names per year (the PIT universe);
* rank IC per year of the combined score and of every theme and raw feature,
  against the pipeline's own residualised labels, with t-stats;
* the Grinold ``IC_used`` the simulator would see on each rebalance date and the
  share of dates where it is zero (trailing IC <= 0 => alpha 0 => empty book);
* industry label counts of the loaded names ("Other" / "UNKNOWN" share).

Usage (same data flags as ``qts run``)::

    python scripts/diagnose_signal.py --store data/store --start 2000-01-01 \
        --max-names 3000 --min-mcap-ever 3e8 --cache
    python scripts/diagnose_signal.py --synthetic            # smoke test
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from qts.core.config import SystemConfig
from qts.core.stats import rank_ic_series
from qts.pipeline import ResearchPipeline


def _yearly(ic: pd.Series) -> pd.DataFrame:
    ic = ic.dropna()
    if ic.empty:
        return pd.DataFrame(columns=["mean", "t", "n"])
    g = ic.groupby(ic.index.year)
    out = pd.DataFrame({"mean": g.mean(), "n": g.size()})
    out["t"] = g.mean() / g.std(ddof=1).replace(0, np.nan) * np.sqrt(out["n"])
    return out[["mean", "t", "n"]]


def _fmt(df: pd.DataFrame, cols: list[str]) -> str:
    with pd.option_context(
        "display.width", 200, "display.max_columns", 50, "display.float_format", "{:.3f}".format
    ):
        return df[cols].to_string()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--max-names", type=int)
    ap.add_argument("--min-mcap-ever", type=float)
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--config")
    ap.add_argument("--ml", action="store_true")
    a = ap.parse_args(argv)

    cfg = SystemConfig.from_yaml(a.config) if a.config else SystemConfig()
    if a.cache:
        cfg = cfg.with_overrides({"cache.enabled": True, "cache.root": f"{a.runs_root}/_cache"})

    if a.synthetic:
        from qts.data.synthetic import SyntheticSpec, generate

        md = generate(SyntheticSpec(n_stocks=120, n_days=900, seed=0))
    elif a.store:
        from qts.data.store import ParquetStore

        store = ParquetStore(a.store)
        tickers = None
        if a.min_mcap_ever is not None or a.max_names is not None:
            tickers = store.tickers_by_market_cap(
                a.start, a.end, min_mcap_ever=a.min_mcap_ever, max_names=a.max_names
            )
        md = store.as_market_data(universe_tickers=tickers, start=a.start, end=a.end)
    else:
        print("give --store or --synthetic", file=sys.stderr)
        return 2
    print(
        f"loaded: {md.close.shape[1]} ids x {md.close.shape[0]} days ({md.dates[0].date()}..{md.dates[-1].date()})"
    )

    pipe = ResearchPipeline(cfg, use_ml=a.ml, runs_root=None)
    pipe.fit(md)
    alpha = pipe.alpha
    md_u = pipe.md_u
    assert alpha is not None and md_u is not None

    # -- universe --------------------------------------------------------------
    elig = md_u.eligibility()
    per_year = elig.sum(axis=1).groupby(elig.index.year).agg(["mean", "min", "max"]).round(0).astype(int)
    print("\n== eligible names per year (mean / min / max) ==")
    print(per_year.to_string())

    # -- industries ------------------------------------------------------------
    ind = md_u.industry.reindex(md_u.tickers)
    vc = ind.value_counts(dropna=False)
    print(f"\n== industry labels ({len(vc)} distinct) ==")
    print(vc.head(12).to_string())
    other = float(ind.isin(["Other", "UNKNOWN"]).mean())
    print(f"share Other/UNKNOWN: {other:.1%}")

    # -- IC by year: combined, composite, themes, raw features -----------------
    target = alpha.target_
    print(f"\nlabels: {target.shape}, non-NaN {target.notna().to_numpy().mean():.1%}")
    rows: dict[str, pd.Series] = {"combined": alpha.combined_score_, "composite": alpha.composite_score_}
    ml = getattr(alpha, "ml_score_", None)
    if ml is not None:
        rows["ml"] = ml
    for th, sc in alpha.theme_scores_.items():
        rows[f"theme_{th}"] = sc
    table = {}
    for name, panel in rows.items():
        ic = rank_ic_series(panel, target)
        y = _yearly(ic)
        table[name] = y["mean"]
        full = ic.dropna()
        t_full = full.mean() / full.std(ddof=1) * np.sqrt(len(full)) if len(full) > 2 else np.nan
        print(f"{name:28s} full-sample IC {full.mean():+.4f}  t {t_full:+.2f}  n {len(full)}")
    print("\n== rank IC by year (mean) ==")
    print(_fmt(pd.DataFrame(table), list(table)))

    # raw features (processed X_, long -> wide per feature)
    X = alpha.X_
    if isinstance(X, pd.DataFrame) and set(X.index.names) >= {"date", "ticker"}:
        feats = list(X.columns)
        ftab = {}
        for f in feats:
            w = X[f].unstack("ticker")
            ic = rank_ic_series(w, target).dropna()
            if len(ic) > 20:
                ftab[f] = (ic.mean(), ic.mean() / ic.std(ddof=1) * np.sqrt(len(ic)))
        if ftab:
            fdf = pd.DataFrame(ftab, index=["IC", "t"]).T.sort_values("t")
            print(
                "\n== processed features: full-sample rank IC vs label (sign as used: + means higher value -> higher return) =="
            )
            print(_fmt(fdf, ["IC", "t"]))

    # -- IC_used over time -----------------------------------------------------
    dates = alpha.combined_score_.index
    step = max(1, len(dates) // 120)
    sample = dates[::step]
    used = pd.Series({d: alpha.ic_used(d) for d in sample})
    trailing = pd.Series({d: alpha.trailing_ic(d) for d in sample})
    print("\n== Grinold IC_used (what the optimizer sees), sampled ==")
    print(
        f"IC_used == 0 on {float((used == 0).mean()):.0%} of sampled dates; "
        f"trailing IC available on {float(trailing.notna().mean()):.0%}"
    )
    yr = pd.DataFrame({"trailing_ic": trailing, "ic_used": used})
    print(yr.groupby(yr.index.year).mean().round(4).to_string())
    print(
        f"\nconfig: ic_shrink={cfg.alpha.ic_shrink} ic_window={cfg.alpha.ic_window} "
        f"default_ic={cfg.alpha.default_ic} ic_min_obs={cfg.alpha.ic_min_obs}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
