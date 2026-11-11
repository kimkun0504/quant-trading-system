"""Risk-model diagnostics on a ParquetStore (or the synthetic market): where does the
factor covariance blow up, and why?

Fits only the risk model (no alpha, no backtest) exactly as the pipeline does, then
prints:

* the 25 days with the largest |factor return| (factor, value, robust z against the
  trailing year, number of names in the regression, condition number of the
  weighted design matrix that day);
* per year: max |factor return| and the factor;
* thin industries: for every industry, the minimum number of eligible members over
  time and the first date it fell to <= 2; and the number of (date, industry) cells
  with 0, 1 or 2 members;
* specific-variance extremes per year (min / max annualised specific vol);
* the reference-book ex-ante vol (``as_of`` on month ends): every month where it
  exceeds 100% or jumps > 5x, with the top factors by variance contribution.

Usage::

    python scripts/diagnose_risk.py --store data/store --start 2000-01-01 \
        --max-names 3000 --min-mcap-ever 3e8
    python scripts/diagnose_risk.py --synthetic
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd

from qts.data.universe import apply_mask, build_universe_mask, masked_levels
from qts.risk import build_risk_model

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from diagnose_signal import add_data_args, load_config, load_market


def _fundamental(risk):
    return getattr(risk, "fundamental", risk)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(ap)
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args(argv)
    cfg = load_config(a)
    md = load_market(a)
    print(
        f"loaded: {md.close.shape[1]} ids x {md.close.shape[0]} days ({md.dates[0].date()}..{md.dates[-1].date()})"
    )

    mask = build_universe_mask(md, cfg.universe)
    md_u = apply_mask(md, mask)
    md_risk = masked_levels(md_u)
    t0 = time.perf_counter()
    risk = build_risk_model(cfg.risk).fit(md_risk)
    print(f"risk model fitted in {time.perf_counter() - t0:.0f}s; clean_returns={cfg.risk.clean_returns}")
    fm = _fundamental(risk)
    fr: pd.DataFrame = fm.factor_returns
    sr: pd.DataFrame = fm.specific_returns
    elig = md_u.eligibility()

    # -- factor-return outliers --------------------------------------------------
    absfr = fr.abs()
    mad = (fr - fr.rolling(252, min_periods=60).median()).abs().rolling(252, min_periods=60).median() * 1.4826
    z = (fr.abs() / mad.replace(0, np.nan)).fillna(0)
    rows = []
    for d in absfr.max(axis=1).nlargest(a.top).index:
        k = absfr.loc[d].idxmax()
        n_names = int(sr.loc[d].notna().sum()) if d in sr.index else -1
        cond = float("nan")
        try:
            X = fm.exposures_as_of(d - pd.Timedelta(days=1))
            names = elig.loc[d][elig.loc[d]].index.intersection(X.index)
            Xn = X.loc[names].fillna(0.0).to_numpy(dtype=float)
            w = np.sqrt(md.market_cap.loc[:d].iloc[-2].reindex(names).fillna(0).clip(lower=0).to_numpy())
            s = np.linalg.svd(Xn * w[:, None], compute_uv=False)
            cond = float(s[0] / max(s[-1], 1e-300))
        except Exception:
            pass
        rows.append(
            {
                "date": d.date(),
                "factor": k,
                "f": float(fr.loc[d, k]),
                "robust_z": float(z.loc[d, k]),
                "n_in_regression": n_names,
                "design_cond": cond,
            }
        )
    with pd.option_context("display.width", 160, "display.float_format", "{:.4g}".format):
        print("\n== largest |factor return| days ==")
        print(pd.DataFrame(rows).to_string(index=False))
        yr = pd.DataFrame({"max_abs_f": absfr.max(axis=1), "factor": absfr.idxmax(axis=1)})
        yr = yr.groupby(yr.index.year).apply(lambda g: g.loc[g["max_abs_f"].idxmax()])
        print("\n== per year: max |factor return| ==")
        print(yr.to_string())

    # -- thin industries ----------------------------------------------------------
    ind = md_u.industry.reindex(md_u.tickers)
    counts = {}
    for lab in sorted(ind.dropna().unique()):
        cols = ind.index[ind == lab]
        counts[lab] = elig[cols].sum(axis=1)
    cnt = pd.DataFrame(counts)
    thin = []
    for lab in cnt.columns:
        s = cnt[lab]
        first = s.index[s <= 2]
        thin.append(
            {
                "industry": lab,
                "min_members": int(s.min()),
                "median": float(s.median()),
                "first_<=2": first[0].date() if len(first) else None,
                "days_<=2": int((s <= 2).sum()),
            }
        )
    thin_df = pd.DataFrame(thin).sort_values("min_members")
    print("\n== thin industries (eligible members over time) ==")
    print(thin_df.head(15).to_string(index=False))
    print(
        f"(date, industry) cells with 0 / 1 / 2 members: {int((cnt == 0).sum().sum())} / {int((cnt == 1).sum().sum())} / {int((cnt == 2).sum().sum())}"
    )

    # -- specific variance extremes -----------------------------------------------
    sv = sr.rolling(63, min_periods=20).std() * np.sqrt(252)
    print("\n== specific vol (63d) per year: min / median / max across names ==")
    yrs = sv.index.year
    tab = pd.DataFrame(
        {
            "min": sv.groupby(yrs).min().min(axis=1),
            "median": sv.groupby(yrs).median().median(axis=1),
            "max": sv.groupby(yrs).max().max(axis=1),
        }
    )
    print(tab.to_string())

    # -- reference vol on month ends ----------------------------------------------
    months = pd.Series(md.dates).groupby(pd.Series(md.dates).dt.to_period("M")).last().tolist()
    months = [d for d in months if d >= fr.index[0] + pd.Timedelta(days=400)]
    ref = {}
    alerts = []
    prev = None
    for d in months:
        try:
            out = risk.as_of(d)
        except Exception as e:
            print(f"as_of({d.date()}) failed: {e}")
            continue
        B, F, D = out.exposures, out.factor_cov, out.specific_var
        names = B.index
        mc = md.market_cap.loc[:d].iloc[-1].reindex(names).fillna(0).clip(lower=0)
        w = np.sqrt(mc.to_numpy())
        w = w / w.sum() if w.sum() > 0 else w
        Bw = B.fillna(0).to_numpy(dtype=float).T @ w
        fvar = Bw @ F.to_numpy() @ Bw
        svar = float(np.nansum(D.reindex(names).fillna(0).to_numpy() * w**2))
        vol = float(np.sqrt(max(fvar + svar, 0) * 252))
        ref[d] = vol
        if vol > 1.0 or (prev and vol > 5 * prev):
            contrib = pd.Series(Bw * (F.to_numpy() @ Bw), index=F.columns).sort_values(ascending=False)
            alerts.append((d.date(), vol, prev, contrib.head(5).round(6).to_dict()))
        prev = vol
    refs = pd.Series(ref)
    print("\n== reference-book ex-ante vol (month ends) ==")
    print(refs.describe(percentiles=[0.05, 0.5, 0.95]).to_string())
    print("\n== alerts (vol > 100% or > 5x previous month) ==")
    for d, v, p, c in alerts:
        print(
            f"{d}: vol {v:.2%} (prev {p if p is None else f'{p:.2%}'}) top factor variance contributions: {c}"
        )
    if not alerts:
        print("none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
