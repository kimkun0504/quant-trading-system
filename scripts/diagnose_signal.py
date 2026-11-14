"""Signal diagnostics on a ParquetStore (or the synthetic market): why is the book empty?

Fits the pipeline exactly as ``qts run`` does (stage cache reused when enabled) and
prints, without running the backtest:

* eligible names per year (the PIT universe);
* rank IC per year of the combined score and of every theme and raw feature,
  against the pipeline's own residualised labels, with t-stats;
* the Grinold ``IC_used`` the simulator would see on each rebalance date and the
  share of dates where it is zero (trailing IC <= 0 => alpha 0 => empty book);
* industry label counts of the loaded names ("Other" / "UNKNOWN" share);
* **coverage per year** of every raw feature: the share of eligible (date, name)
  cells with a non-missing value before the cross-sectional fill (the feature-set-2
  features are marked).  ``--coverage-only`` prints this table and stops: it
  applies the universe and builds the raw feature panels but fits nothing (no risk
  model, no labels, no ML), so it takes a minute or two on the full store -- run it
  before a long ``alpha.feature_set=v2`` run to check the earnings data.

Usage (same data flags as ``qts run``)::

    python scripts/diagnose_signal.py --store data/store --start 2000-01-01 \
        --max-names 3000 --min-mcap-ever 3e8 --cache [--ml --set alpha.horizon_days=60]
    python scripts/diagnose_signal.py --store data/store --start 2000-01-01 \
        --max-names 3000 --min-mcap-ever 3e8 --set alpha.feature_set=v2 --coverage-only
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


def add_data_args(ap: argparse.ArgumentParser) -> None:
    """The data / config flags shared with ``qts run`` (and scripts/diagnose_constraints.py)."""
    ap.add_argument("--store")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--max-names", type=int)
    ap.add_argument("--min-mcap-ever", type=float)
    ap.add_argument("--max-rank", type=int, help="screened-rank store prefilter, as `qts run --max-rank`")
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--config")
    ap.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        metavar="KEY=VALUE",
        default=None,
        help="dotted config override applied after --config (repeatable), exactly as "
        "`qts run --set` (same parser, so the stage-cache key matches the run)",
    )
    ap.add_argument("--ml", action="store_true")


def load_config(a: argparse.Namespace) -> SystemConfig:
    """``--config`` (or defaults), then every ``--set KEY=VALUE`` (``qts.cli``'s own
    parser: YAML scalars, unknown keys / rejected values raise
    ``qts.cli.ConfigOverrideError``), with the stage cache switched on by
    ``--cache``, exactly as ``qts run`` does."""
    from qts.cli import _apply_set_overrides

    cfg = SystemConfig.from_yaml(a.config) if a.config else SystemConfig()
    cfg = _apply_set_overrides(cfg, getattr(a, "set_overrides", None))
    if a.cache:
        cfg = cfg.with_overrides({"cache.enabled": True, "cache.root": f"{a.runs_root}/_cache"})
    return cfg


def load_market(a: argparse.Namespace, with_earnings: bool = False, cfg: SystemConfig | None = None):
    """``MarketData`` from ``--store`` (with the peak-market-cap prefilter, under the
    research holdout seal) or ``--synthetic`` (``with_earnings``: also the synthetic earnings tables, which
    ``alpha.feature_set=v2`` reads); ``None`` when neither was given."""
    if a.synthetic:
        from qts.data.synthetic import SyntheticSpec, generate

        return generate(SyntheticSpec(n_stocks=120, n_days=900, seed=0, with_earnings=with_earnings))
    if a.store:
        # the research seal (qts.research.seal; research/seal.yaml): diagnostics load data
        # through the day before the holdout, exactly as a sealed `qts run` with the same
        # flags (same panels, same stage-cache key)
        from qts.research.data import StoreSpec, load_research_market

        load = load_research_market(
            StoreSpec.from_args(a),
            hash_cache_dir=f"{getattr(a, 'runs_root', 'runs')}/_cache/market_hash",
            log=print,
            universe=cfg,
        )
        return load.md
    return None


def clamp_universe(cfg: SystemConfig, md) -> SystemConfig:
    """``universe.top_n_by_mcap`` capped at the number of loaded ids, as
    ``qts.cli._run_pipeline`` does before it builds the pipeline (a no-op unless
    fewer ids were loaded than the configured top-N; without it the stage-cache
    key of such a run would not match)."""
    return cfg.with_overrides({"universe.top_n_by_mcap": min(cfg.universe.top_n_by_mcap, md.close.shape[1])})


def feature_coverage_by_year(raw: dict[str, pd.DataFrame], eligible: pd.DataFrame) -> pd.DataFrame:
    """Year x feature: share of the eligible (date, name) cells of the year whose
    *raw* feature value is present (before winsorising / ranking / the fill of
    missing values).  The last row, ``all``, is the whole sample."""
    elig = eligible.to_numpy(dtype=bool)
    years = np.asarray(eligible.index.year)
    uniq = np.unique(years)
    n_year = np.array([elig[years == y].sum() for y in uniq], dtype=float)
    out = {}
    for name, panel in raw.items():
        have = panel.reindex(index=eligible.index, columns=eligible.columns).notna().to_numpy() & elig
        per = np.array([have[years == y].sum() for y in uniq], dtype=float)
        col = np.append(per / np.where(n_year > 0, n_year, np.nan), have.sum() / max(elig.sum(), 1))
        out[name] = col
    return pd.DataFrame(out, index=[*map(str, uniq), "all"])


def print_coverage(raw: dict[str, pd.DataFrame], eligible: pd.DataFrame) -> None:
    from qts.core.roles import FEATURE_SET_2_FEATURES

    cov = feature_coverage_by_year(raw, eligible)
    new = [f for f in FEATURE_SET_2_FEATURES if f in cov.columns]
    old = [f for f in cov.columns if f not in new]
    pct = "{:.1%}".format
    with pd.option_context("display.width", 250, "display.max_columns", 60):
        if new:
            print("\n== coverage per year: feature set 2 (share of eligible names with a value) ==")
            print(cov[new].to_string(float_format=pct))
            low = [f for f in new if cov.loc["all", f] < 0.5]
            if low:
                print(f"LOW COVERAGE (< 50% over the sample): {', '.join(low)}")
        print("\n== coverage per year: v1 features ==")
        print(cov[old].to_string(float_format=pct))


def print_earnings_sources(md) -> None:
    """Announcements per year by source (EVENTS 8-K vs filing-date fallback) for the loaded names."""
    ed = getattr(md, "earnings_dates", None)
    if ed is None or ed.empty:
        return
    ed = ed[ed["ticker"].isin(set(md.tickers))]
    tab = ed.groupby([pd.to_datetime(ed["announce_date"]).dt.year, "source"]).size().unstack(fill_value=0)
    tab["share_events"] = tab.get("events", 0) / tab.sum(axis=1)
    print("\n== earnings announcements per year by source (loaded names) ==")
    print(tab.to_string(float_format="{:.1%}".format))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(ap)
    ap.add_argument(
        "--coverage-only",
        action="store_true",
        help="print the per-year feature coverage (and the earnings-date sources) and stop: universe "
        "+ raw features only, no risk / alpha fit",
    )
    a = ap.parse_args(argv)

    cfg = load_config(a)
    md = load_market(a, with_earnings=cfg.alpha.feature_set == "v2", cfg=cfg)
    if md is None:
        print("give --store or --synthetic", file=sys.stderr)
        return 2
    print(
        f"loaded: {md.close.shape[1]} ids x {md.close.shape[0]} days ({md.dates[0].date()}..{md.dates[-1].date()})"
    )
    print(f"feature set: {cfg.alpha.feature_set}")
    if cfg.alpha.feature_set == "v2":
        from qts.alpha.features_v2 import FeatureSetDataError, require_v2_inputs

        try:
            require_v2_inputs(md)
        except FeatureSetDataError as e:
            print(f"error: {str(e).replace('<store>', a.store or '<store>')}", file=sys.stderr)
            return 2

    if a.coverage_only:
        from qts.alpha.features import FeatureBuilder
        from qts.data.universe import apply_mask, build_universe_mask

        cfg_u = clamp_universe(cfg, md)
        md_cov = apply_mask(md, build_universe_mask(md, cfg_u.universe))
        elig_cov = md_cov.eligibility()
        per_year = elig_cov.sum(axis=1).groupby(elig_cov.index.year).agg(["mean", "min", "max"])
        print("\n== eligible names per year (mean / min / max) ==")
        print(per_year.round(0).astype(int).to_string())
        print_earnings_sources(md_cov)
        print_coverage(FeatureBuilder(cfg_u.alpha).build(md_cov), elig_cov)
        return 0

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

    # -- coverage of the raw features -------------------------------------------
    print_earnings_sources(md_u)
    print_coverage(alpha.raw_features_, elig)

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
