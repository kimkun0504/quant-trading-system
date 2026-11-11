# Risk model outlier defence: return cleaning, ESTU, blow-up guards

**Found:** 2026-11-10, real-data run (Sharadar, 3,000 names loaded, top-1,000 eligible, 2000–2026),
`scripts/inspect_run.py`. On 2018-03-05 the ex-ante vol of the book jumped from 5 % to
**5,110,263 %**; the target-vol optimizer liquidated everything and the book stayed empty until
2022-09. Seven more empty stretches (2003-09..2005-11, 2008-03..2010-02, 2010-06..2011-09,
2011-12..2013-08, 2014-07..2015-04, 2023-05..08, 2025-06..) have the same signature with smaller
shocks. The synthetic market never produces such a return, so the golden tests could not see it.

**Mechanism.** The risk model had no return cleaning. One absurd daily return (restated adjusted
close, sub-penny print, one-sided split) enters the sqrt-cap-weighted cross-sectional regression
(`qts/risk/factor_returns.py`); WLS has to explain a squared error 10^5–10^9 times every other
observation's, so that day's factor returns blow up; the EWMA factor covariance
(`vol_half_life` = 84) then carries a squared contribution 10^8× normal, which decays as
2^(−n/84) and reaches order 1 only after n ≈ 84 log2(10^8) ≈ 2,230 trading days ≈ 4.4 years —
the 2018-03 → 2022-09 gap. The specific-return EWMA (same half-life) and the style descriptors
(beta, residual vol, momentum, reversal: 252-day windows and an equal-weighted market proxy)
carry the same print.

## What was built

1. **`scripts/find_outlier_returns.py`** (same loader as `diagnose_signal.py`): top-50 |r| with id,
   date, raw/adjusted closes (t−1, t), volume, market cap, eligible flag and a likely-cause tag
   (`sub-penny`, `restatement`, `split-like`, `delisting-day`, `gap(nd)`, `unknown`); counts of
   |r| > 0.5/1/3/10 per year and how many cells rule (a) / the price floor would remove; dates
   where the cross-sectional sqrt-cap-weighted mean |r| spikes vs its trailing-252d median; per-date
   max |r| (and culprit) inside the eight windows above (`--windows` to override). Smoke-tested on
   the synthetic market.
2. **`qts/core/returns.py`** — three documented, PIT-safe rules:
   (a) data error: `r > return_abs_cap_up` (3.0) or `r < return_abs_cap_down` (−0.95) → NaN;
   optionally a return off a previous as-traded close `< return_min_prev_price` (0.25 USD);
   (b) cross-sectional winsorisation at median ± `return_winsor_mad_k` (8) × 1.4826 MAD per day;
   (c) flags (`CleanedReturns.data_error` / `.winsorized`, counts by date / name).
   `flag_bad_prices` is the price-level form of (a): a spike-and-reverse pair is one bad close; a
   persisting level shift is blanked for up to 3 days.
   Who applies what: **risk model** (a)+floor+(b) on the regression's dependent variable, the
   specific returns, the APCA input and the hybrid's total-return loadings, with the style
   descriptors built from the price-blanked view (`BarraStyleRiskModel.fit` →
   `estimation_inputs`); **alpha labels** (`qts/alpha/targets.py`) caps only, as a bad price
   (label of that cell NaN; a window ending on it takes the halt path) — genuine crashes are
   genuine labels, the residualiser already winsorises at 1 %; **simulator** caps only, as a
   missing price (= the halt path: frozen, no return, no trade; next return measured from the
   last good close), counted in `diagnostics["n_data_error_returns"]` with the cells in
   `data_error_returns`. Config home: `RiskConfig` (`clean_returns`, the four thresholds), read by
   the labels and the simulator through `SystemConfig` (the label builder reads the caps as
   module defaults from `RiskConfig()`; `AlphaModel` does not thread the config — see open items).
3. **Estimation universe.** Checked: in the pipeline the regression already used only eligible
   names, because `ResearchPipeline.fit` passes `masked_levels(md_u)` (market cap NaN outside the
   universe) and the regression intersects on names with a cap. New
   `RiskConfig.estimation_universe_eligible_only = True` makes it explicit
   (`estimate_factor_return_history(estimation_universe=md.eligible)`: regression weights,
   industry constraint and specific returns from eligible-at-t−1 names only; exposures for every
   priced name; names outside the ESTU get the structural specific-risk prior). No golden change.
4. **Guards.** (a) `BarraStyleRiskModel.as_of` records the annualised ex-ante vol of a long-only
   sqrt-cap reference book in `diagnostics["reference_vol"]`; above `max_reference_vol` (1.0) or
   a > 5× jump vs the previous recorded date → `diagnostics["covariance_outliers"]` (top-5 factor
   and specific contributions) + warning. (b) The simulator checks the incoming book's ex-ante vol
   (or the reference book's when the book is empty) on every rebalance against
   `ValidationConfig.max_exante_vol` (2.0): warning with date and top-5 names/factors,
   `diagnostics["exante_vol_alerts"]`, `n_exante_vol_alerts`; new hurdle **`risk_model_sanity`**
   fails when alerts exceed `max_exante_vol_alert_frac` (0.005) of rebalance solves. (c) The
   report card prints `data guards: data-error closes blanked N, ex-ante vol alerts N of M solves,
   failed solves N`.
5. **Tests** (`tests/test_return_cleaning.py`, 12; `tests/test_scripts_diagnose.py`, +2): rule
   (a) asymmetry and NaN handling; price floor lagged one day; rule (b) clip value and flags, rows
   with < 20 names untouched; PIT (future garbage leaves earlier rows unchanged); invalid
   thresholds raise; `flag_bad_prices` spike = one close, level shift = 3 days, genuine crash
   untouched. Injected outlier (150 names × 900 days, one close × 1000 on day 620 in 2 names):
   the cleaned fit's reference vol stays within 20 % of the clean fit on every later day (the raw
   fit records a covariance outlier > 20× the clean vol), the spiked names' specific variance is
   finite on every day they are priced, the label of the spiked cell is NaN and every other name's
   labels are identical, the simulator counts exactly the two blanked closes and a book that does
   not hold the names has the same NAV (solver tolerance 1e-6); the `risk_model_sanity` gate FAILs
   with `clean_returns=False` and PASSes with it.

## Golden impact

Rules (a), the price floor, the ESTU flag, the simulator path and both guards leave the golden
synthetic run **bitwise unchanged** (verified with `return_winsor_mad_k = 0`: zero diffs).
Rule (b) does touch it: the synthetic cross-section is fat-tailed by construction and 48 of
~70,000 cells lie beyond 8 MADs, so the golden file was regenerated. Moves: net Sharpe
2.1682 → 2.1642, gross 2.4638 → 2.4592, CAGR net 15.53 % → 15.50 %, max DD −1.898 % → −1.899 %,
cost/side 6.99 → 6.98 bp, realised/ex-ante 0.962 → 0.965, `mean_risk_aversion` 0.2381 → 0.2343
(largest, 1.6 %); IC statistics, decile spread, DSR to 3 decimals, gate statuses and the data hash
unchanged. The owner asked for "report instead of silently changing": this is the report; the
pre-change numbers are recovered exactly with `--set risk.return_winsor_mad_k=0`.

Every config hash changes (seven new `RiskConfig` fields, two new `ValidationConfig` fields);
`configs/default.yaml` regenerated. The stage cache keys include the risk config section, so the
cached `risk_fit` / `alpha_fit` of earlier real-data runs will miss (expected: the fit changes).

## Commands for the owner

```
# 1. see the culprits (no fit; a minute or two on the full panel)
python scripts/find_outlier_returns.py --store data/store --start 2000-01-01 \
    --max-names 3000 --min-mcap-ever 3e8 --top 50
# 2. rerun (risk/alpha cache will miss: the risk config section changed)
qts run --store data/store --start 2000-01-01 --max-names 3000 --min-mcap-ever 3e8 --cache
# 3. A/B against the old behaviour if wanted
qts run ... --set risk.clean_returns=false
```

Read the report card's `data guards` line and the `risk_model_sanity` gate; the log lists each
blanked close and each ex-ante vol alert with its top names/factors.

## Not verified without real data

- The thresholds. +300 % / −95 % / 0.25 USD / 8 MAD are standard-practice defaults; the
  `find_outlier_returns` counts per year say how many cells they remove on Sharadar and whether
  any genuine mega-move (e.g. a biotech +400 % on approval, 2020 meme names) is caught. A
  `restatement`-tagged cell whose raw return is sane points at the vendor's adjustment, which
  the loader could also fix at source (`closeadj` vs `closeunadj × cumulative factor`).
- Whether the 2018-03-05 print is a one-day spike (one blanked close) or a level shift (the
  three-day rule then accepts the new level and the gap return is still absurd → rule (a) drops
  it from the regression but the simulator's halt path would realise it on day 4; the alert and
  the count will show it).
- Whether, after cleaning, the other empty stretches disappear or were (also) the IC floor.

## Open items

- `AlphaModel` passes no caps to the label builder; the labels use the module defaults
  (`RiskConfig()` caps), so `--set risk.return_abs_cap_up=...` does not reach the labels until a
  one-line pass-through in `qts/alpha/alpha_model.py` (not touched here: another agent was editing
  that file concurrently).
- Rule (b) in the risk model is applied before the ESTU restriction (per-day median/MAD over all
  priced names); restricting the statistic to the ESTU would be marginally cleaner.
- `qts/risk/alignment.py`, `evaluation.py` and the alpha features still read `md.returns` raw
  (evaluation and calibration paths; the alpha feature winsorisation handles its own tails).
