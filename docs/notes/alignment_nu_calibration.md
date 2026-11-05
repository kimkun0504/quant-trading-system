# Alignment-factor variance `nu`: calibration on the synthetic market

WORK_PLAN item 1.8; resolves audit `risk_portfolio.md` finding 5 ("`nu` is uncalibrated").
**Recalibrated 2026-11-05** after the second-pass audit (`second_pass_risk_portfolio.md`
finding 10): the first calibration (below, "2026-11-04 run") used fixed `gamma = 10` plus the
alpha-blind vol-target projection, execution lag 0 (production is 1) and books that could hold
names outside the risk model. **Current default: `alignment_nu_scale = 0.05`.**

## 2026-11-05 recalibration (production settings)

Changes against the first run:

* Sizing: default `PortfolioConfig` = `risk_aversion_mode="target_vol"` (gamma searched per
  date so the optimal book has 10 % ex-ante vol, capped by gross 2 / 2 % names / ADV).
* Lag: `calibrate_alignment_nu(..., signal_lag_days=1)` pairs the book built from data as of `t`
  with the return of `t + 2`, as `DailyBarSimulator` does with `ExecutionConfig.signal_lag_days = 1`.
* Universe: the optimizer only holds risk-model names (finding 3), and the script puts
  ineligible names that are not held on `no_trade`, like the simulator.
* Risk model: Bayesian shrinkage Delta about the cap-weighted mean, specific VRA with lagged
  daily caps, industry columns as of `t` (findings 4 and 9).

Run (CPU-constrained, so a shorter window than the first run): 300 names, 1512 days, last
**120** out-of-sample dates (2023-05-03 .. 2023-10-17), 20 burn-in days, `--n-iter 5`,
grid 0 / 0.25 / 0.5 / 1 / 2 (plus the search ends 0.05 and 50):

    python scripts/calibrate_alignment_nu.py --seeds S --n-dates 120 --burn-in 20 --n-iter 5 --grid 0 0.25 0.5 1 2

Bias statistic of the optimized book (band `1 ± 2/sqrt(2T)` = [0.870, 1.130], T = 120):

| nu_scale | seed 0 | seed 1 | seed 2 |
|---:|---:|---:|---:|
| 0    | 0.870 | 0.861 | 0.897 |
| 0.05 | 0.867 | 0.846 | 0.883 |
| 0.25 | 0.859 | 0.831 | 0.849 |
| 0.5  | 0.850 | 0.817 | 0.830 |
| 1    | 0.843 | 0.796 | 0.810 |
| 2    | 0.833 | 0.784 | 0.801 |
| 50   | 0.833 | 0.788 | 0.789 |
| **calibrated** | **0.05** (floor) | **0.05** (floor) | **0.05** (floor) |

| at nu = 0 | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| ex-ante / realised vol | 8.5 % / 7.4 % | 8.8 % / 7.6 % | 8.3 % / 7.5 % |
| `abs(y'w)` at nu = 0 → 0.25 → 1 | .051 → .046 → .037 | .072 → .063 → .046 | .083 → .069 → .046 |
| alignment share of ex-ante variance at 0.25 / 1 | 2.3 % / 5.7 % | 3.5 % / 7.2 % | 4.6 % / 7.5 % |

**Reading.** With the book sized at the vol target by the alpha-aware objective the risk model
already *over*-forecasts the optimized book (bias 0.86–0.90 at `nu = 0`, at or just below the
lower band edge on all three seeds). The alignment term only raises the forecast further, so
the bias is monotone decreasing in `nu` and the calibration lands on the search floor (0.05)
on every seed. The under-forecast that motivated the term (bias 1.02–1.12 in the first run) was
produced by the old sizing: a book optimized at 1/3–1/5 of its traded size and then levered by
a projection that ignored `alpha_perp`, plus phantom hedges outside the model whose market
moves the forecast did not see. **Default set to the calibrated median, 0.05** (the term stays
on but carries ~1 % of the ex-ante variance). The remaining over-forecast (~12 %) is a
risk-model calibration question (eigen inflation `a = 1.2`, VRA), not an alignment one; it is
on the conservative side for the vol target. On real data the unexplained alpha may carry real
unpriced risk, so **recalibrate on real data** before relying on this value.

---

# 2026-11-04 run (superseded)

## What is being calibrated

The Saxena–Stubbs alignment term adds `nu * y y'` to the covariance, where `y` is the unit-norm
part of the alpha that the risk-model exposures cannot explain (`qts/risk/alignment.py`). The
variance is

    nu = nu_scale * N * median_k F_kk      (median over the style factors)

so `nu_scale = 1` gives the alignment factor the variance of a typical unit-sd style factor.
Saxena–Stubbs treat `nu` as a free parameter. They tune it so that the **optimized** book's
bias statistic `sd(r_{t+1} / sigma_t)` is about 1. Below 1, the model over-forecasts risk. Above 1,
it under-forecasts risk because the optimizer loads on directions the model cannot see.

The setting is `RiskConfig.alignment_nu_scale`. It is used whenever `as_of(...)` or
`alpha_alignment(...)` is called with `nu_scale=None`, which is the default for both the
fundamental and the hybrid model. `calibrate_alignment_nu` still passes `nu_scale` explicitly.

## Method (`scripts/calibrate_alignment_nu.py`)

* Market: `generate(SyntheticSpec(seed=s))`, with 300 names and 1512 days (6 years).
  This uses the lagged planted alpha with `alpha_lag_days=1`. Seeds tested: 0, 1 and 2.
* Production path: `ResearchPipeline(SystemConfig(), use_ml=False).fit(md)` builds:
  * the universe mask
  * the hybrid fundamental + APCA risk model
  * the composite alpha (`AlphaModel(use_ml=False)`)
  * the `TransactionCostModel`
  * a `SinglePeriodOptimizer` with the default `PortfolioConfig`: 150/50 gross 2.0, beta-, industry- and
    style-neutral bands, 2 % position cap, ADV caps, 10 % vol target and CLARABEL.
* Window: the last 250 out-of-sample dates (2022-11-02 .. 2023-10-17), plus 30 burn-in days
  solved at `nu_scale = 1` to build a warm starting book.
* `solve_fn` is a closure over `SinglePeriodOptimizer.solve`. The inputs are:
  * the eligible-names expected return, as in `DailyBarSimulator`
  * the simulator's cost snapshot
  * the previous day's book as `w_current`
  * the risk-only styles as `neutral_styles`
  * held names that became ineligible, forced to zero
  The closure therefore follows the simulator's sequential book. Each `nu_scale` path starts from the same
  warm book.
* `calibrate_alignment_nu(model, md_u, alpha, dates, solve_fn=..., grid=(0, .25, .5, 1, 2, 4))`
  works in four steps:
  1. It builds `as_of(t, alpha_t, nu_scale=1)` once per date.
  2. It rescales `alignment_var` for each candidate.
  3. It forecasts `sqrt(w' Sigma_aug w)` on the **whole** book and compares it with the realised
     next-day return of the whole book.
  4. It bisects on `log nu_scale` in `[0.05, 50]` until `|bias − 1| ≤ 0.02`.

  Each `nu_scale` is evaluated at most once.
* Changes to `calibrate_alignment_nu` made for this run:
  * memoisation
  * a `grid` output
  * dates sorted, so stateful solvers see them in order
  * realised P&L and forecast now cover the whole book. Before, they were truncated to the
    risk-model tickers.

Runtime is about 12–20 minutes per seed on 1 CPU. The run used 2 processes for 3 seeds.

## Results

Bias statistic of the optimized book (ex-ante includes the alignment term). The ±2-sd band is
`1 ± 2/sqrt(2T)`.

| nu_scale | seed 0 (T=139) | seed 1 (T=250) | seed 2 (T=250) | mean abs(bias−1) |
|---:|---:|---:|---:|---:|
| 0    | 1.115 | 1.017 | 1.079 | 0.070 |
| 0.25 | 1.038 | 0.921 | 0.985 | 0.044 |
| 0.5  | 1.019 | **0.910** | 0.982 | 0.042 |
| 1    | 1.038 | **0.885** | 0.936 | 0.072 |
| 2    | 1.030 | **0.901** | **0.908** | 0.074 |
| 4    | 0.996 | 0.915 | **0.904** | 0.062 |
| band | [0.880, 1.120] | [0.911, 1.089] | [0.911, 1.089] | |
| **calibrated** | **1.581** (bias 1.001) | **0.05** (0.979, at the search floor) | **0.281** (0.983) | |

Bold entries inside the grid are outside the band.

Calibrated values: 1.58, 0.05 and 0.28. The **median is 0.28**, so the default is
**`alignment_nu_scale = 0.25`**. The old implicit default was 1.0.

Other diagnostics from the same runs. The numbers are annualised and come from the
`diagnostics` block of the script's `--json` output:

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| bias with no alignment term (`nu=0`) | 1.115 | 1.017 | 1.079 |
| `abs(y'w)`, nu = 0 → 0.25 → 1 → 4 | .037 → .019 → .006 → .003 | .098 → .069 → .032 → .012 | .107 → .073 → .035 → .014 |
| alignment share of ex-ante variance at 0.25 / 1 | 1.0 % / 0.5 % | 7.6 % / 6.8 % | 8.3 % / 7.4 % |
| ex-ante / realised vol at nu = 0.25 | 6.0 % / 6.2 % | 7.1 % / 6.6 % | 7.4 % / 7.3 % |

## Reading the numbers

* The term does what it should. With `nu = 0` the optimized book's risk is under-forecast on all
  three seeds (bias 1.02–1.12). A small `nu` removes most of that. Raising `nu` further mainly
  pushes the book off `alpha_perp` (`abs(y'w)` falls steadily). Past about 0.5 it raises the
  forecast more than the realised risk, and seeds 1 and 2 start to over-forecast (bias < 1).
* The bias curve is flat and noisy for `nu_scale` ≥ 0.25. The sampling sd of the bias over 250 days
  is about 0.045, about the same size as the differences between grid points. So the calibrated
  values per seed (1.58, 0.05, 0.28) carry mostly noise beyond "small, positive". Of the grid
  points, only 0.25 keeps all three seeds inside their bands. At 0.5, seed 1 is just outside
  (0.910 against a lower limit of 0.911). At 1.0, seed 1 has bias 0.885. Across the three
  seeds, 0.25 and 0.5 are tied on mean abs(bias−1). The rounded median is 0.25, so that is
  the default.
* Seed 0 uses only 139 of the 250 dates. On the first 111 dates of its window the composite
  alpha's expected return is identically zero. Its trailing IC estimate is ≤ 0, so Grinold
  scaling gives a zero alpha and there is no alignment vector. `calibrate_alignment_nu` skips
  those dates, which is correct because there is nothing to align.
* The alignment share of ex-ante variance stays below 10 %, so the vol-target scalar is
  barely affected at the new default. Audit finding 5 had warned that the term feeds the vol target.

## Caveats

* **Synthetic data.** The market has 300 names and a static universe. Its latent factors are all
  observable except the planted alpha (the APCA block sees the rest), so `alpha_perp` is mostly
  noise plus the planted score. On real data the unexplained part of the alpha may carry real
  unpriced risk, such as crowding or missing industries. The right `nu` may then be much larger.
  **Re-run `scripts/calibrate_alignment_nu.py` on the real-data `MarketData`** (same code path
  from `ResearchPipeline.fit` on) over the warm-up window. Then set
  `RiskConfig.alignment_nu_scale` to the median result. Recalibrate after changes to the
  alpha (especially enabling the ML sleeve), the risk-model factor set or the portfolio
  constraints.
* Daily bias over 250 days has wide confidence bands (about ±0.09). Use at least a year of dates. Prefer
  the grid table and the mean abs(bias−1) across windows or seeds over a single bisection result.
* The burn-in book is built at `nu_scale = 1`. Each candidate path then evolves on its own.
  Ineligible held names are exited immediately, where the simulator takes 5 days. On this
  synthetic universe the effect is negligible.
* The calibration uses the composite-only alpha (phase 1). The ML sleeve changes `alpha_perp`
  and needs its own run.
