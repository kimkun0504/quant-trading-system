# Risk model: new listings and de-SPACs (gross descriptor outliers)

**Found:** 2026-11-10, real-data run (Sharadar, ~1,000 eligible names) after
`risk_thin_industry_fix_2026-11-10.md`. The 2002–2012 blow-ups are gone (λ_S within 0.71–1.45,
floor hits ≈ 0, reference-vol median 15 %, 95th percentile 39 %). Five month ends still alert,
all de-SPACs:

| Month end | Fundamental reference vol | Top specific contributors |
|---|---|---|
| 2020-06-30 | 2.7e15 % (prev 46.7 %) | DKNG 7.4e26; JBHT / QRVO / SUI / EXPD all ≈ 3.2e19 |
| 2020-07-31 | 1.6e9 % | DKNG 2.6e14; others ≈ 9.8e6 each |
| 2021-06-30 | 6.0e5 % | MP 1.3e7, QS 1.3e7, OPEN 8.1e6, ALIT 2.2e6; next name 0.43 |
| 2021-07-30 | 6.6e4 % | MP, OPEN, ALIT |
| 2021-08-31 | 1.2e36 % | SOFI 1.3e68; others ≈ 6.7e60 each |

## Reproduction, and what did not reproduce

The brief's reproduction — a name flat at 10.00 for 250 days (returns exactly 0, or ±0.1 % noise),
cap 3e8, then +60 %, cap into the top decile, 60 % annual vol; three names, T = 560 / 640 / 720 of
1,100 days, 200 names, top-120 universe — **does not blow up** the pre-fix estimator, eligible or
not before the merger: reference vol 12–17 % on every date, the same as the no-SPAC fit. A name
with r = 0 has the residual u = −B f (daily sd 0.8–1.2 %), so nothing divides by zero; before it is
eligible it has no residual and takes the structural value from exposures that are bounded.

What a de-SPAC brings besides the price path is a shell's fundamentals. `investment` =
−(`assets / assets_prev − 1`); a SPAC's assets a year before the trust was funded are the
sponsor's few thousand dollars. Planting asset growth = 1500 (+150,000 %) on the same three names
from T − 60 to T + 75 reproduces every signature of the real run. Numbers at T + 6 of the first
name (N = 127 eligible), protections off:

| Step | Value |
|---|---|
| raw `investment` | −1500 against a cross-section with sd 0.11 (18,646 robust sigmas) |
| after z-score → clip ±3 → z-score | **−11.07** (√127 = 11.3); the other 126 names: sd **0.0029** |
| WLS leverage of the name | 0.999997 (the style is a dummy for it) |
| its residual | sd 3e-5 against a return sd of 0.057; raw EWMA variance 8e-11, lifted by the raw-input floor |
| structural coefficient on `investment` | **−11.6** (every other style < 0.3 in absolute value) |
| x·b for the name | 128, of which `investment` 128.9 |
| structural specific vol (γ = 0, n_obs = 2) | **4.5e55** per day (median name 0.0166) |
| after shrinkage | the name 3.4e55; five other names of its size decile **3.882e52 each** |
| reference vol, fundamental | 8.0e54 (peak 5.2e116 at T + 73, where b = −24.4) |
| hybrid | `stat_01` variance *contribution* 1.4e103, largest hybrid specific variance 1.2e111; the stat factor variance itself is 0.97 |
| VRA multiplier | 0.997; names at the floor 0 |

So, precisely: **(1)** "clip at 3 sd, then re-standardise" is not robust to a gross outlier — the
outlier owns the first sd, is clipped to 3, owns the second sd (3/√N) too and is put back at √N
sd, with the rest of the cross-section collapsed to a point; **(2)** the regression treats the
style as that name's dummy (residual ≈ 0: the "tiny but not zero" minimum specific vol);
**(3)** the structural log-vol regression fits the style's coefficient on the collapsed band, so
an ordinary slope becomes −11.6 per unit, and `exp` of the extrapolation to x = −11 is 1e55;
**(4)** the Bayesian shrinkage target is a cap-weighted arithmetic mean, so every other name of
the decile gets (nearly) the same huge value — the near-identical JBHT / QRVO / SUI / EXPD;
**(5)** the hybrid carries the same numbers in its `stat_*` block because that block is expressed
in fundamental specific-variance units. Steps not involved: return cleaning, thin industries, the
VRA (on the as_of date or any other), the blend weight, the APCA extraction, the total-return
loadings.

The filing calendar fits the five alerts. DEAC IPO'd in May 2019: DKNG's year-ago quarter is
pre-IPO until the Q2-2020 filing (mid-August 2020) — alerts in June and July 2020, not August.
FVAC / KCAC / IPOB / WPF all IPO'd in Q2 2020: MP / QS / OPEN / ALIT are absurd from the Q1-2021
filing to the Q2-2021 one — alerts in June and July 2021. IPOE was formed in July 2020: SOFI's
Q2-2021 filing has no real year-ago balance sheet — alert in August 2021. One outlier gives √N
(2020-06, 2021-08: 1e15 %, 1e36 %); four at once give √(N/4) each (2021-06: 1e5 %).

## What was built

1. **Robust standardisation** (`style_raw_winsor_mad_k = 10`, `style_exposure_cap = 3`;
   `exposures.py::winsorize_raw_descriptor`, `standardize_style`). Raw descriptors are winsorised
   per date at median ± 10 × 1.4826 MAD (IQR / 1.349 when the MAD is zero; a row where both are
   zero is left alone) before the z-scores, and the final standardised exposure is winsorised at
   ±3 (USE4). After the fix: exposure −3.000, others sd 0.956, leverage 0.48, residual sd 0.016,
   structural vol 0.0162, reference vol 13.1 %.
2. **Non-trading days** (`flat_return_tol = 1e-6`, `flat_run_min_days = 20`;
   `core/returns.py::flat_run_mask`, `model.py::flat_return_days`). |r| < 1e-6 on each of the last
   20 days → the day is NaN in the regression's returns (no specific return, nothing in the APCA
   panel or the hybrid's total-return loadings) and is removed from the beta / residual-vol
   windows. Causal (the first 19 days of a run are not flagged). `diagnostics['n_flat_return_days']`.
3. **Seasoning** (`min_specific_obs = 60`): the former constant in γ.
4. **Structural clip** (`structural_clip_pctl = 0.01`): prediction clipped to the [1 %, 99 %] band
   of the vols it was fitted on; a non-finite prediction takes the fit-set median.
5. **Specific-variance ceiling** (`specific_var_cap_mult = 25`, vol ≤ 5× the ESTU median), on the
   raw EWMA input, on the blend before the shrinkage and on the final forecast; `n_at_cap`,
   `cap_var`, `var_bounds` in the details, `diagnostics['n_at_specific_cap']` /
   `['specific_var_bounds']`. The shrinkage target stays the USE4 cap-weighted mean; its input is
   bounded instead (a median target would change every decile every day).
6. **Repair** (`covariance_repair = True`, `factor_var_repair_mult = 100`;
   `covariance.py::repair_covariance`, `robust_factor_variances`, `BarraStyleRiskModel.repair`),
   after the final assembly in both `as_of`: specific variances finite and in [floor, cap]
   (hybrid: [0.1 × floor, cap]); factor variances finite and ≤ 100 × median(f²)/0.4549 over the
   trailing year (or quarter / month if larger). Violations clipped and recorded in
   `diagnostics['covariance_repairs']` (date, model, what, n, worst).
7. **Alarm on the hybrid**: `diagnostics['reference_vol_hybrid']`; outlier entries carry `model`
   (closes the open item of the thin-industry note).
8. **`scripts/diagnose_risk.py`**: alert scan on every month end `as_of` can serve, fundamental
   and hybrid, ending in `alerts: none (all N month ends, fundamental + hybrid)`; per alert the
   top-5 names' raw EWMA variance, n_obs, γ, structural prediction, shrinkage target, final
   variance, largest |style exposure| and that style's spread over the other names; counts of
   non-trading cells, names at the cap, repairs; the largest raw-descriptor outliers in robust
   sigmas.

Which layer does what (same reproduction, worst reference vol / no-SPAC fit over T − 5 … T + 300):
all off 3.9e117; defaults 1.03; raw winsorisation only 1.03; structural clip only 1.02; ceiling
only 1.03 (one name at the cap, specific vol 118 %); **±3 on the standardised exposure only:
1.2e35** (still a dummy, e^{3b}); non-trading rule only 9e119; repair only 3.9e117 (no cap → no
bound to repair to). The raw winsorisation is the cure; 4 and 5 each bound the damage alone; 2 and
3 are correctness for the descriptors and the residual history, not what stops the blow-up.

Not changed: the APCA extraction and the total-return loadings. The trace shows neither is in the
path (stat factor variance 0.97–1.00 throughout; the existing `min_stock_obs` and |R/σ| ≤ 50
guards hold), and the non-trading rule already removes flat days from their inputs.

## Tests (`tests/test_risk_new_listings.py`, 14 fast + 2 slow)

Flat price alone is benign with the protections off; with the shell fundamentals the pathology
reproduces with the escape hatches (reference vol > 1e6 × the no-SPAC fit for both models, a
recorded outlier, exposure > 0.9 √N, others' sd < 0.01, residual < 1e-3 of the return, γ = 0 with
structural vol > 1e20 and EWMA vol < 0.02, ≥ 5 other names > 1e6 × median and equal to 1 %); with
defaults, zero-return and ±0.1 % variants, on T − 5 … T + 12 daily then every 16th day to T + 300
(slow tier: every date): fundamental and hybrid reference vol within 25 % of the no-SPAC fit,
every specific variance in [floor, cap], stat factor variances finite and ≤ 10× the no-SPAC fit,
|style exposure| ≤ 3, **no covariance repairs and no outliers**; each layer alone; the repair
fires on a hand-injected 1e20 (factor variance and one specific variance; off switch lets it
through); units for the raw winsorisation, the flat-run mask (causal), the estimation inputs, the
structural clip, the ceiling before the shrinkage, `repair_covariance`, config bounds.
Fast tier 457 passed; `pyright qts/core` 0; ruff clean.

## Golden impact

Regenerated. Net Sharpe 2.1571 → 2.2161 (+2.73 %), gross 2.4525 → 2.5138 (+2.50 %), CAGR net
15.46 % → 15.76 %, gross 17.82 % → 18.13 %, ann vol net 6.770 % → 6.706 %, max DD −1.903 % →
−1.809 %, Calmar 8.12 → 8.71 (+7.3 %), Sortino 3.354 → 3.459, `mean_risk_aversion` 0.2329 → 0.2441
(+4.8 %), realised/ex-ante 0.967 → 0.954, cost/side 6.99 → 6.93 bp, turnover 9.57 % → 9.64 %,
transfer coefficient 0.7885 → 0.7881, DSR 0.8498 → 0.8562, `sharpe_net_lo` 7.41 → 7.97; IC over the
three IC dates: mean 0.0506 → 0.0545, ICIR 1.333 → 1.433, hit rate 1/3 → 2/3 (one date changes
sign); gate statuses and the data hash unchanged.

A/B on the same run (net Sharpe): all six off 2.1571 (the previous file, exactly);
`style_exposure_cap = inf` 2.1491; `style_raw_winsor_mad_k = 0` 2.2095; `flat_run_min_days = 0`,
`structural_clip_pctl = 0`, `specific_var_cap_mult = inf`, `covariance_repair = false` each 2.2161
(bitwise the default run). **The ±3 winsorisation of the standardised exposure is the move**
(+0.067), the raw winsorisation adds +0.007 (the generator's short-interest descriptor has a few
cells beyond 10 robust sigmas); the other four are neutral. The synthetic styles reach 3.2–4.7 sd
after the re-standardisation, so the cap binds on a few names per date. Headline net Sharpe is
inside the 5 % threshold; Calmar, mean IC / ICIR and `sharpe_net_lo` are not (60 days, three IC
dates). The cap is kept at 3 because it is the USE4 rule; `--set risk.style_exposure_cap=.inf`
keeps everything else and gives 2.1491.

## Commands for the owner

```
# 1. risk model only (minutes): every month end, fundamental and hybrid
python scripts/diagnose_risk.py --store data/store --start 2000-01-01 \
    --max-names 3000 --min-mcap-ever 3e8
# expected: "alerts: none (all N month ends, fundamental + hybrid)", "covariance repairs: 0",
# names at the cap 0 or a handful, and in the raw-descriptor table DKNG / MP / QS / OPEN / ALIT /
# SOFI under `investment` at thousands of robust sigmas on 2020-06/07 and 2021-06/07/08

# 2. the two-hour run (risk config changed: the risk / alpha stage cache misses once)
qts run --store data/store --start 2000-01-01 --max-names 3000 --min-mcap-ever 3e8 --cache -v \
    --ml --set alpha.horizon_days=60

# 3. A/B against the previous estimator if wanted
qts run ... --set risk.style_raw_winsor_mad_k=0 --set risk.style_exposure_cap=.inf \
    --set risk.flat_run_min_days=0 --set risk.structural_clip_pctl=0 \
    --set risk.specific_var_cap_mult=.inf --set risk.covariance_repair=false
```

## Not verified without real data

- That asset growth is the descriptor. It is an inference from the reproduction (which matches
  all three signatures) and the filing calendar; the raw-descriptor table of step 1 settles it.
  If the table names another descriptor (`quality` = gp / assets on a near-zero asset base is the
  other candidate) the mechanism and the fix are the same.
- Why 2020-05-29 (46.7 %) and 2021-05-28 did not alert although the Q1 filings were out: probably
  the PIT lag of the fundamentals or the names' eligibility date; not checked.
- How often the ceiling and the ±3 cap bind on Sharadar, and whether the repair ever fires (it
  should not; if it does, the entry says where).

## Open items

- A descriptor with ≥ 75 % identical values (MAD and IQR both zero) is not pre-winsorised; a gross
  outlier there ends at ±3 with the rest collapsed. Bounded by layers 4–6, and such a descriptor
  carries no cross-section anyway, but it is still a quasi-dummy in the regression.
- The alpha features use the same fundamentals; whether a +100,000 % asset growth reaches the
  alpha model un-winsorised was not looked at (rank transforms would neutralise it).
- The repair's 100× bound is against the factor's own robust trailing variance; a factor whose
  returns are zero on more than ~92 % of the trailing year (an almost-always-empty industry) has
  no reference and is only checked for finiteness.
- `specific_var_cap_mult = inf` with `covariance_repair = true` leaves the repair without an upper
  bound on specific variance (it clips to the configured cap, and there is none).
