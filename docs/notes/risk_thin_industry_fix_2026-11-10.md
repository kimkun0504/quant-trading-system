# Risk model: thin industries, the specific-variance floor and VRA clipping

**Found:** 2026-11-10, real-data run (Sharadar, ~1,000 eligible names, FF48 industries), after the
return cleaning of `risk_outlier_fix_2026-11-10.md`. `scripts/diagnose_risk.py`: factor returns
are sane (max |f| = 0.26), but the reference-book ex-ante vol reaches 4.4e11 % — 2003-09-30:
14.1 % → 438,258,410,932 %; 2008-03-31: 21 % → 2.8e10 %; also 2010-06-30, 2011-12-30, 2002-02 —
each the start of a multi-year empty-book stretch decaying with the EWMA half-lives. The top
contributors are the *specific* variances of mega-caps (MSFT, ORCL, INTC, CSCO at 1e15–1e17) and,
in the hybrid, the APCA factors `stat_01`/`stat_02` with variance 1e16.

**Mechanism.** Thin-industry table: 22,983 (date, industry) cells with exactly one eligible member,
10,525 with zero, 17,322 with two (Ships, Smoke, RlEst, Rubbr, Aero, Mines, Other, …). With one
member the industry dummy absorbs the stock's whole return: its residual is exactly 0 (~1e-16 in
floating point), its EWMA specific vol ~1e-16 (the per-year minimum specific vol is 3e-16 in every
year). The day that stock gets a real residual (a second member joins, the ESTU changes) every step
that divides by its specific variance explodes: the specific VRA bias statistic z² = u²/σ² ~ 1e26,
the cap-weighted average over all names — applied to *every* name — scales the whole specific block
by ~1e13, and the APCA input scaling R/σ sees a 1e16-sigma observation. Reproduced on the synthetic
market (`tests/test_risk_thin_industry.py`: three single-member industries, one pair whose second
member joins late): λ_S = 1.3e12, reference vol 3.2e10 with the defences off; 0.98–1.01 and 11–18 %
with them on.

## What was built

1. **Thin-industry shrinkage** (`qts/risk/factor_returns.py::shrink_thin_industries`,
   `RiskConfig.thin_industry_min_members = 5`, `thin_industry_prior_obs = 5`). After the constrained
   WLS on day t, every industry with n_t < 5 members is shrunk toward the cap-weighted average return
   of its parent sector, f_k ← λ f_k + (1 − λ) f_parent, λ = n_t / (n_t + 5) (n = 0 → the parent
   exactly, n = 1 → 1/6 own). Parent = Fama-French 12 sector via the new
   `qts/data/loaders.py::FF48_TO_FF12` (all 48 names → 12 sectors, following French's sector
   descriptions; `FF12_NAMES`); a label outside the table shrinks toward the whole industry block.
   Order: solve → parent averages from the *unshrunk* returns → shrink → re-impose the cap-weighted
   constraint by subtracting the cap-weighted mean m of the shrunk industry returns from every
   industry and adding m to `market` (a re-identification: no fitted value changes) → recompute
   u = r − B f_shrunk and R². `CrossSectionalFit` gained `thin_industries` / `constraint_shift` (still
   unpacks as the 3-tuple); `diagnostics["thin_industry_days"]` counts days per industry,
   `thin_industry_cells` splits them by member count. Barra estimates an industry on its own only
   above a minimum membership of about five names (USE4 Methodology Notes, "Thin industries").
2. **Specific-variance floor** (`covariance.py::specific_risk`, `specific_var_floor_frac = 0.10`): the
   final shrunk variance is floored at 10 % of its cross-sectional median over the names observed
   in the window (specific vol ≥ ~32 % of the median name); the raw EWMA input is floored the same
   way *before* the structural fit, so an identically-zero residual series enters as a small real
   vol, never 1e-16. `n_at_floor` / `floor_var` in the details; `diagnostics["n_at_specific_floor"]`.
3. **VRA robustness** (`specific_vra_multiplier`, `factor_vra_multiplier`; `vra_z2_clip = 25`,
   `vra_min = 0.5`, `vra_max = 4.0`): z² clipped at 25 before the cross-sectional average, names
   whose forecast variance is below the floor excluded, the multiplier clipped to [0.5, 4]. The
   function defaults are off (the pandas-equivalence test in `test_stage_cache.py` stays bitwise);
   `specific_risk` / `factor_covariance` pass the config values. `diagnostics["specific_vra_multiplier"]`
   per `as_of`.
4. **APCA / hybrid / alignment.** `statistical.apca` checks the scaled input is finite and winsorises
   |R/σ| at `MAX_ABS_SCALED = 50` (`APCAResult.n_winsorized`, `StatisticalRiskModel.n_scaled_winsorized`,
   `diagnostics["apca_scaled_winsorized"]`). The hybrid's `stat_*` block is expressed in the floored,
   clipped D_fund units; the alignment's optional 1/σ² weights read the same D.
5. **Config.** Seven new `RiskConfig` fields with bounds (`specific_var_floor_frac` in [0, 1),
   `vra_min <= vra_max`, `vra_z2_clip` / `vra_max` > 0 with `.inf` = off, `thin_industry_min_members
   >= 0`); `configs/default.yaml` regenerated (hash changes; cached risk/alpha fits of earlier real
   runs will miss).
6. **Guards.** `reference_vol_check_in_fit = True`: `fit` ends with `as_of` on every month end after
   the warm-up, so the reference-vol guard fires at fit time (logs median / max / outlier count).
   `scripts/diagnose_risk.py` prints the specific VRA multiplier series (extremes with dates, month
   ends at the clip), names at the floor per year and the industries most often shrunk.
7. **Tests** (`tests/test_risk_thin_industry.py`, 6): pathology reproduces with the escape hatches
   (residual < 1e-12, raw specific vol < 1e-8, λ_S or reference vol > 10× the clean fit, a covariance
   outlier recorded); with defaults the raw EWMA specific vol over the ESTU is real on every date and
   the shrunk variance ≥ the floor, λ_S in [0.5, 4], reference vol within 25 % of a fit on the
   original labels on every grid date, no outliers, the single-member stock's residual > 1e-3, the
   shrunk returns equal λ f_OLS − m with λ = 1/6 (one member) and 2/7 (two members) on hand-computed
   days, the fat industries move by exactly −m and `market` by +m, styles untouched, r = B f + u, the
   constraint holds to 1e-10 on the hand days and every 97th day of the history; `shrink_thin_industries`
   on FF48 labels shrinks Ships toward the cap-weighted Manuf mean (Ships, Aero, Mach) and an empty
   industry takes the parent exactly; the FF48→FF12 crosswalk is complete; config bounds.
   Fast tier 443 passed; `pyright qts/core` 0; ruff clean.

## Golden impact

The golden market (100 names, 12 synthetic industries, top-80 universe) has industries with exactly
four members on some days, so the shrinkage at k = 5 touches it. Regenerated: net Sharpe
2.1642 → 2.1571 (−0.32 %), gross 2.4592 → 2.4525, CAGR net 15.50 % → 15.46 %, ann vol net
6.766 % → 6.770 %, max DD −1.899 % → −1.903 %, Calmar 8.16 → 8.12, Sortino 3.367 → 3.354,
cost/side 6.98 → 6.99 bp, realised/ex-ante 0.965 → 0.967, transfer coefficient 0.7895 → 0.7885,
`mean_risk_aversion` 0.2343 → 0.2329 (largest, −0.6 %), DSR 0.8506 → 0.8498; IC statistics, decile
spread, gate statuses and the data hash unchanged. A/B on the same run: shrinkage off → 2.1634,
VRA clipping off → 2.1577, floor off → 2.1571, k = 4 → 2.1634 (nothing is thin at k = 4 on this
market). The move is well under the 5 % threshold; k = 5 is kept because it is the Barra threshold,
not because of the golden move.

## Commands for the owner

```
# 1. confirm the reference vol stays sane before spending two hours on the backtest
#    (fits the risk model only; prints the thin-industry table, the month-end reference vol,
#    the specific VRA multiplier extremes and the names at the floor per year)
python scripts/diagnose_risk.py --store data/store --start 2000-01-01 \
    --max-names 3000 --min-mcap-ever 3e8
# expected: reference-vol max of order 20-40 %, "alerts: none", lambda_S within [0.5, 4] and
# not sitting at the clip, floor hits a handful of names at most

# 2. rerun (the risk config section changed: the risk/alpha stage cache will miss)
qts run --store data/store --start 2000-01-01 --max-names 3000 --min-mcap-ever 3e8 --cache --ml \
    --set alpha.horizon_days=60

# 3. A/B against the old behaviour if wanted (all escape hatches)
qts run ... --set risk.thin_industry_min_members=0 --set risk.specific_var_floor_frac=0 \
    --set risk.vra_z2_clip=.inf --set risk.vra_min=0 --set risk.vra_max=.inf
```

## Not verified without real data

- Whether all five empty stretches disappear, or some were (also) the IC floor / other causes.
- How often the floor and the VRA clip bind on Sharadar (the diagnostic prints both); a multiplier
  sitting at `vra_max` for months would mean the base half-life, not the clip, is the problem.
- `thin_industry_prior_obs = 5` is a prior, not a calibration; the per-industry `thin_industry_days`
  counts say how much of the history each FF48 industry spends borrowing from its sector.

## Open items

- `FF48_TO_FF12` assigns the five straddling industries (Hshld, ElcEq, LabEq, PerSv, Toys) to the
  majority sector; a SIC-level FF12 lookup would be exact but needs the SIC, which the static
  `industry` label no longer carries.
- The fit-time monthly check calls the fundamental `as_of` only; the hybrid's `stat_*` block is not
  checked at fit time (its diagonal equals the fundamental's, so a blow-up there is the same blow-up).
