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
* the reference-book ex-ante vol on **all** month ends ``as_of`` can serve, for
  the fundamental model and (when the statistical companion is on) the hybrid:
  every month where it exceeds 100% or jumps > 5x, with the top factor and
  specific variance contributions and, for the top-5 names, every stage of the
  specific-risk pipeline (raw EWMA variance, number of residuals, structural
  prediction, shrinkage target, final variance, largest style exposure); ends
  with one ``alerts: none`` / ``alerts: <n>`` line;
* new-listing protections: non-trading cells dropped, names at the
  specific-variance cap per year, covariance repairs (expected 0) and the
  largest raw-descriptor outliers in robust sigmas (which name / descriptor the
  pre-winsorisation acts on);
* the specific VRA multiplier on the same month ends (extremes with dates, and
  how many month ends sit at the configured ``[vra_min, vra_max]`` clip), the
  number of names at the specific-variance floor per year, and the industries
  most often shrunk toward their parent sector (``thin_industry_days``).

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


def reference_contributions(out, market_cap: pd.Series) -> tuple[float, pd.Series, pd.Series]:
    """``(annualised vol, factor contributions, specific contributions)`` of the
    sqrt-cap reference book under ``out`` (a ``RiskModelOutput``); contributions
    are annualised variances, sorted descending."""
    B, F, D = out.exposures, out.factor_cov, out.specific_var
    names = B.index
    w = np.sqrt(market_cap.reindex(names).fillna(0).clip(lower=0).to_numpy(dtype=float))
    w = w / w.sum() if w.sum() > 0 else w
    Bw = B.fillna(0).to_numpy(dtype=float).T @ w
    Fv = F.reindex(index=B.columns, columns=B.columns).fillna(0.0).to_numpy(dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        fac = pd.Series(Bw * (Fv @ Bw) * 252, index=B.columns).sort_values(ascending=False)
        spec = pd.Series(D.reindex(names).fillna(0).to_numpy(dtype=float) * w**2 * 252, index=names)
        spec = spec.sort_values(ascending=False)
        var = float(fac.sum() + spec.sum())
    vol = float(np.sqrt(max(var, 0.0))) if np.isfinite(var) else float("inf")
    return vol, fac, spec


def specific_trace(fm, d: pd.Timestamp, names: list[str]) -> pd.DataFrame:
    """Every stage of :func:`qts.risk.covariance.specific_risk` for ``names`` on
    ``d``: raw EWMA variance, number of residuals, blend weight, structural
    prediction, size-decile shrinkage target, final variance -- next to the
    cross-sectional median of each -- plus the largest |style exposure| and the
    spread of that style over the other names (a spread << 1 means the style is
    a dummy for the outlier)."""
    from qts.risk.covariance import specific_risk

    B = fm.exposures_as_of(d)
    mcap = fm.md.market_cap.loc[:d].iloc[-1]
    with np.errstate(over="ignore", invalid="ignore"):
        var, det = specific_risk(
            fm.specific_returns.loc[:d],
            B,
            mcap,
            fm.cfg,
            return_details=True,
            market_cap_history=fm.md.market_cap.loc[:d],
        )
    styles = [c for c in B.columns if c in fm.cfg.styles]
    cols = {
        "ewma_var_raw": det["var_ts_raw"],
        "n_obs": det["n_obs"],
        "gamma": det["gamma"],
        "structural_var": det["sigma_str"] ** 2,
        "shrink_target_var": det["shrink_target"] ** 2,
        "final_var": var,
    }
    tab = pd.DataFrame({k: v.reindex(names) for k, v in cols.items()})
    if styles:
        absx = B[styles].abs()
        worst = absx.reindex(names).idxmax(axis=1)
        tab["max_abs_style"] = absx.reindex(names).max(axis=1)
        tab["style"] = worst
        tab["others_sd"] = [
            float(B[st].drop(index=n, errors="ignore").std()) if isinstance(st, str) else np.nan
            for n, st in worst.items()
        ]
    med = {k: float(np.nanmedian(v.to_numpy(dtype=float))) for k, v in cols.items()}
    tab.loc["(median, all names)"] = pd.Series(med)
    return tab


def raw_descriptor_outliers(md, rcfg, months, top: int = 25) -> pd.DataFrame:
    """Largest raw-descriptor outliers on the month ends, in robust sigmas
    (|x - median| / 1.4826 MAD over the eligible names): the cells the
    pre-winsorisation (``style_raw_winsor_mad_k``) acts on.  A de-SPAC's asset
    growth against its pre-IPO shell shows up here at 1e3-1e5 robust sigmas."""
    from qts.risk.exposures import raw_style_panels

    elig = md.eligibility()
    rows = []
    for st, panel in raw_style_panels(md, rcfg.styles, rcfg).items():
        sub = panel.where(elig).reindex(months).replace([np.inf, -np.inf], np.nan)
        if not sub.notna().to_numpy().any():
            continue
        med = sub.median(axis=1)
        sig = 1.4826 * sub.sub(med, axis=0).abs().median(axis=1)
        z = sub.sub(med, axis=0).div(sig.where(sig > 0), axis=0).abs().stack()
        n_beyond = int((z > rcfg.style_raw_winsor_mad_k).sum()) if rcfg.style_raw_winsor_mad_k > 0 else 0
        for (d, t), v in z.nlargest(5).items():
            rows.append(
                {
                    "style": st,
                    "date": d.date(),
                    "name": t,
                    "raw": float(panel.at[d, t]),
                    "robust_z": float(v),
                    "cells_beyond_k": n_beyond,
                }
            )
    print(
        f"largest raw-descriptor outliers on month ends (robust z; winsorised at "
        f"{rcfg.style_raw_winsor_mad_k:g} robust sigmas before standardisation):"
    )
    if not rows:
        return pd.DataFrame(columns=["style", "date", "name", "raw", "robust_z", "cells_beyond_k"])
    return pd.DataFrame(rows).sort_values("robust_z", ascending=False).head(top)


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
    shrunk = fm.diagnostics.get("thin_industry_days", {})
    print(
        f"thin-industry shrinkage (< {cfg.risk.thin_industry_min_members} members, prior "
        f"{cfg.risk.thin_industry_prior_obs:g}): {len(shrunk)} industries shrunk on >= 1 day; most often: "
        f"{list(shrunk.items())[:10]}"
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

    # -- reference vol on ALL month ends, fundamental and hybrid --------------------
    # every month end as_of can serve (>= min_history_days factor-return rows)
    first_ok = fr.index[min(cfg.risk.min_history_days, len(fr)) - 1]
    months = pd.Series(md.dates).groupby(pd.Series(md.dates).dt.to_period("M")).last().tolist()
    months = [d for d in months if first_ok <= d <= fr.index[-1]]
    is_hybrid = fm is not risk
    models = ["fundamental", "hybrid"] if is_hybrid else ["fundamental"]
    ref: dict[str, dict] = {m: {} for m in models}
    alerts: dict[str, list] = {m: [] for m in models}
    prev: dict[str, float | None] = dict.fromkeys(models)
    for d in months:
        try:
            out_h = risk.as_of(d)
            out_f = fm.as_of(d) if is_hybrid else out_h
        except Exception as e:
            print(f"as_of({d.date()}) failed: {e}")
            continue
        for m, out in zip(models, [out_f, out_h] if is_hybrid else [out_f], strict=True):
            vol, fac, spec = reference_contributions(out, md.market_cap.loc[:d].iloc[-1])
            ref[m][d] = vol
            p = prev[m]
            if vol > 1.0 or (p and vol > 5 * p) or not np.isfinite(vol):
                alerts[m].append((d, vol, p, fac.head(5), spec.head(5)))
            prev[m] = vol
    n_alerts = 0
    for m in models:
        refs = pd.Series(ref[m])
        print(f"\n== reference-book ex-ante vol, {m} ({len(refs)} month ends) ==")
        print(refs.describe(percentiles=[0.05, 0.5, 0.95]).to_string())
        print(f"\n== alerts, {m} (vol > 100% or > 5x previous month) ==")
        for d, v, p, fac, spec in alerts[m]:
            n_alerts += 1
            print(f"{d.date()}: vol {v:.4g} ({v:.2%}; prev {p if p is None else f'{p:.2%}'})")
            print(f"  top factor variance contributions (annualised): {fac.map('{:.4g}'.format).to_dict()}")
            print(
                f"  top specific variance contributions (annualised): {spec.map('{:.4g}'.format).to_dict()}"
            )
            with pd.option_context("display.width", 200, "display.float_format", "{:.4g}".format):
                print("  specific-risk trace of those names (daily variances; vols are daily):")
                print(specific_trace(fm, d, list(spec.index)).to_string().replace("\n", "\n  "))
        if not alerts[m]:
            print("none")
    print(
        f"\nalerts: {'none' if n_alerts == 0 else n_alerts} (all {len(months)} month ends, {' + '.join(models)})"
    )

    # -- new listings / de-SPACs: caps, repairs, non-trading days, raw-descriptor outliers
    print("\n== new-listing protections ==")
    print(f"non-trading (date, name) cells dropped: {fm.diagnostics.get('n_flat_return_days', 0)}")
    nc = pd.Series(fm.diagnostics.get("n_at_specific_cap", {})).sort_index()
    if len(nc):
        g = nc.groupby(nc.index.year)
        print(
            f"names at the specific-variance cap ({cfg.risk.specific_var_cap_mult:g} x median), month ends per year:"
        )
        print(pd.DataFrame({"mean": g.mean().round(1), "max": g.max()}).to_string())
    reps = fm.diagnostics.get("covariance_repairs", [])
    print(
        f"covariance repairs: {len(reps)}"
        + (" (expected 0: a repair means a protection has a hole)" if reps else "")
    )
    for r in reps[:20]:
        print(f"  {r['date'].date()} {r['model']}: {r['what']} x{r['n']}, worst {r['worst']}")
    print(raw_descriptor_outliers(md_risk, cfg.risk, months, top=a.top).to_string(index=False))

    # -- specific VRA multiplier and the specific-variance floor ------------------
    lam = pd.Series(fm.diagnostics.get("specific_vra_multiplier", {})).sort_index()
    lam = lam.loc[[d for d in lam.index if d in ref["fundamental"]]] if len(lam) else lam
    print("\n== specific VRA multiplier lambda_S (month ends) ==")
    if len(lam):
        lo, hi = cfg.risk.vra_min, cfg.risk.vra_max
        print(lam.describe(percentiles=[0.05, 0.5, 0.95]).to_string())
        print(f"max {lam.max():.3f} on {lam.idxmax().date()}; min {lam.min():.3f} on {lam.idxmin().date()}")
        print(
            f"month ends at the clip [{lo:g}, {hi:g}]: {int((lam <= lo).sum())} at vra_min, "
            f"{int((lam >= hi).sum())} at vra_max (z^2 clip {cfg.risk.vra_z2_clip:g})"
        )
        print("5 largest:", lam.nlargest(5).round(3).to_dict())
    else:
        print("none recorded")
    nf = pd.Series(fm.diagnostics.get("n_at_specific_floor", {})).sort_index()
    print(
        f"\n== names at the specific-variance floor ({cfg.risk.specific_var_floor_frac:g} x median) per year "
        f"(month ends: mean / max) =="
    )
    if len(nf):
        g = nf.groupby(nf.index.year)
        print(pd.DataFrame({"mean": g.mean().round(1), "max": g.max()}).to_string())
    else:
        print("none recorded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
