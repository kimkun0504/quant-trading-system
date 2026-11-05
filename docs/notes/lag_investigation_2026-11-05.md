# Signal-lag investigation, 2026-11-05

**Question.** In `runs/demo_post_refactor.txt` (default synthetic market: 300 names x 1512 days, seed 0,
composite alpha, `use_ml=False`), delaying execution by one day (`ExecutionConfig.signal_lag_days=1`)
dropped gross Sharpe from 1.33 to 0.49 and net Sharpe from 0.96 to 0.11. The planted alpha is a daily AR(1)
score with a one-day autocorrelation of about 0.989, so one day of staleness should cost about 1 % of
the edge. Which explanation holds: (A) a broken lag implementation, (B) same-day leakage at lag 0,
(C) a real short-horizon effect, or (D) something else?

**Answer: (D). The −0.84 Sharpe gap does not come from the signal lag.**

* The gap does not reproduce on the code in this repository. The demo ran from an uncommitted working
  tree (`git_dirty: true` at `63ce32b`). That tree became commit `ab8cea0` and was modified again before
  the commit. On the current code, the same run gives **gross 0.68 → 0.60, net 0.23 → 0.14** (lag 0 → lag
  1). A paired block bootstrap puts that difference at −0.08 ± 0.15 (1 SE).
* The demo's own lag-0 book did not depend on a one-day edge. Held one day stale (`w_{t-1}·r_{t+1}`),
  it still earns a Sharpe of 1.30 instead of 1.33. The demo's lag-1 run traded into a *different* book:
  its daily gross returns correlate only 0.75 with the lag-0 run's. With ~50-day holding periods and
  cost-aware trading, the optimizer is path dependent, so books that start apart stay apart. One
  1000-day path cannot separate a Sharpe difference of this size from that path noise.
* Across five market seeds (current code), lag 1 is **not worse** than lag 0. The mean gross difference
  is **+0.16** (lag 1 minus lag 0; 4 of 5 seeds positive). The composite's rank IC at lag 1 is 93–98 %
  of its lag-0 IC at every horizon. The true-score oracle decays exactly as AR(1) theory predicts
  across seeds.
* No same-day leakage was found in the lag-0 path. The simulator's lag path is correct (it is now
  covered by real-stack regression tests).
* The audit found one genuine **point-in-time bug, fixed here**. The risk model's style exposures
  (beta, residual vol, and the standardisation of every style) used names that join the universe only
  *after* `t`. This is a future-membership leak. It is unrelated to the lag gap: it biases lag 0 and
  lag 1 equally. It also moved the label residualisation, so the IC estimate used for Grinold scaling
  carried a tiny look-ahead.
* A robustness bug is also fixed. `signal_lag_days >= 2` crashed the pipeline on its first backtest
  date (risk warm-up).

Scripts and raw outputs are in `/tmp/lag_investigation/` (`ic_by_lag.py`, `generator_checks.py`,
`run_bt.py`, `run_seed.py`, `pit_equivalence*.py`, `demo_weights_analysis.py`, `paired_diff.py`,
`z_decay.py`, `loo_themes.py`, `summarize.py`, and their `*.txt` / `*.csv` outputs). Unless stated
otherwise, every backtest below uses the real `DailyBarSimulator`, the real `SinglePeriodOptimizer` and
`TransactionCostModel`, the routed venue and NAV 1e6, called exactly as `ResearchPipeline.run` calls
them. I checked bit-for-bit that `run_bt.py` reproduces `pipe.run(md, dates=...)`.

---

## 1. Is the demo number reproducible?

| run | code | gross SR lag 0 | gross SR lag 1 | net SR lag 0 | net SR lag 1 | turnover/day |
|---|---|---:|---:|---:|---:|---:|
| `demo_post_refactor.txt` | dirty tree at 02:28 (not recoverable) | 1.328 | 0.490 | 0.956 | 0.111 | 8.0 % |
| this investigation (before the fix) | `HEAD` (`ddf9317`) | 0.679 | 0.596 | 0.233 | 0.143 | 12.3 % |
| this investigation (after the exposure fix, §7) | working tree | 0.712 | 1.138 | 0.259 | 0.673 | 12.4 % |

After the fix, the same full-window pair moves the *other* way: lag 1 − lag 0 = +0.43 gross, paired
bootstrap SE 0.24, return corr 0.88. A fix that changes no timing flips the sign of the "lag effect",
which shows the scale of the path noise.

`config.yaml` of both demo runs matches the current defaults, apart from `top_n_by_mcap`: the CLI caps
it at 300, which gives a universe mask identical to 1000 here (checked). The data hash is the same
(`acc387f0eca55816`). On the first backtest day the weights differ (gross 0.40 vs 0.55), so the code
that produced the demo differs from `HEAD` in the optimizer, cost or risk path. `ab8cea0` was
committed 13 minutes after the demo's lag-1 run finished, and no stash or copy of that intermediate
state survives.

**What the stored demo weights show** (`demo_weights_analysis.py`). Gross returns are recomputed from
the stored weights:

| book | SR of `w_t·r_{t+1}` (as booked) | SR of `w_{t-1}·r_{t+1}` (book held 1 day stale) |
|---|---:|---:|
| demo lag 0 | 1.328 | 1.299 |
| demo lag 1 | 0.490 | 0.473 |
| current lag 0 | 0.679 | 0.726 |
| current lag 1 | 0.596 | 0.623 |

The demo lag-0 book loses almost nothing when it is held one day stale, so its edge was not a one-day
effect. The demo lag-1 book (corr 0.74 with the lag-0 book shifted by one day) is a different
portfolio path, not a lagged copy of the same one.

**How noisy are single-path comparisons?** (`paired_diff.py`, 21-day block bootstrap of paired daily
gross returns):

| pair | SR a | SR b | diff | bootstrap SE | NW t (mean diff) | return corr |
|---|---:|---:|---:|---:|---:|---:|
| demo lag 0 vs demo lag 1 | 1.33 | 0.49 | +0.84 | 0.32 | 2.46 | 0.75 |
| current lag 0 vs current lag 1 | 0.68 | 0.60 | +0.08 | 0.15 | 0.47 | 0.92 |
| demo lag 0 vs current lag 0 (same lag, different code) | 1.33 | 0.68 | +0.65 | 0.42 | 1.47 | 0.66 |

Two code versions running the *same* lag differ by 0.65 Sharpe. That matches the golden note's finding
that small optimizer and risk changes move a single-path Sharpe by more than 0.5. The demo's lag
"effect" falls within that range of path variation.

## 2. Oracle test (true latent score as alpha)

`alpha = 0.02 · σ_resid,20d · zscore(z_t over eligible names)`, half-life 65 d, fixed IC (no
estimation). Common window 2019-12-18 → 2023-10-17 (1000 days; lags 2 and 5 start 5 days later because
of the warm-up, see §6). Δ = Sharpe difference vs lag 0, paired bootstrap SE.

| costs | lag | gross SR | net SR | Δ gross | SE | return corr vs lag 0 |
|---|---:|---:|---:|---:|---:|---:|
| real | 0 | 0.859 | 0.396 | | | |
| real | 1 | 0.698 | 0.229 | −0.161 | 0.101 | 0.978 |
| real | 2 | 0.624 | 0.145 | −0.235 | 0.102 | 0.975 |
| real | 5 | 0.600 | 0.119 | −0.259 | 0.159 | 0.953 |
| zero | 0 | 1.147 | 1.136 | | | |
| zero | 1 | 1.070 | 1.059 | −0.077 | 0.081 | 0.988 |
| zero | 2 | 0.986 | 0.975 | −0.161 | 0.111 | 0.976 |
| zero | 5 | 0.820 | 0.810 | −0.326 | 0.186 | 0.941 |

The curve is smooth and monotone, with no cliff between lag 0 and lag 1. The decay looks steeper than
φ^k = 0.989^k because of **seed 0's draw**, not the mechanics. `z_decay.py` measures the realised IC of
`z_{t-k}` vs `r_{t+1}` per seed:

| seed | IC(k=0) | IC(k=1)/IC(0) | IC(k=5)/IC(0) |
|---:|---:|---:|---:|
| 0 | 0.0081 | 0.96 | 0.78 |
| 1 | 0.0077 | 1.01 | 0.89 |
| 2 | 0.0080 | 1.03 | 1.07 |
| 3 | 0.0075 | 1.02 | 0.96 |
| 4 | 0.0108 | 0.98 | 0.96 |
| **mean** | 0.0084 | **1.00** | **0.93** |
| theory φ^k | | 0.989 | 0.948 |

Seed 0 happens to sit in the steep tail. Applied to the zero-cost lag-0 Sharpe, its realised ratios
predict ≈1.10 at lag 1 and ≈0.90 at lag 5 (observed 1.07 and 0.82). With costs the path dependence
adds noise (§1). Neither run shows a mechanical defect.

## 3. Composite decay curve (same settings, `use_ml=False`)

Same 1000-day window (pre-fix risk model):

| lag | gross SR | net SR | Δ gross vs 0 | SE | return corr vs lag 0 | turnover/day |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.700 | 0.262 | | | | 12.3 % |
| 1 | 0.619 | 0.175 | −0.080 | 0.159 | 0.923 | 12.3 % |
| 2 | 0.891 | 0.444 | +0.192 | 0.171 | 0.942 | — |
| 5 | 0.650 | 0.196 | −0.050 | 0.211 | 0.904 | — |

Lag 2 *beats* lag 0 by 0.19. A single path cannot rank lags 0–5 on this market.

**Five seeds** (`run_seed.py`: the last 600 out-of-sample days per seed, the same window for both
lags, current code including the exposure fix):

| seed | gross lag 0 | gross lag 1 | Δ gross | net lag 0 | net lag 1 | Δ net |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.242 | 0.504 | +0.262 | −0.201 | 0.042 | +0.243 |
| 1 | 2.155 | 2.111 | −0.044 | 1.637 | 1.590 | −0.046 |
| 2 | 0.798 | 0.846 | +0.049 | 0.328 | 0.375 | +0.047 |
| 3 | 1.371 | 1.768 | +0.397 | 0.965 | 1.358 | +0.393 |
| 4 | 0.449 | 0.590 | +0.141 | −0.046 | 0.101 | +0.148 |
| **mean** | 1.003 | 1.164 | **+0.161** (sd 0.174) | 0.537 | 0.693 | +0.157 |

Lagging the composite by one day costs nothing measurable on the synthetic market. The positive mean
(t ≈ 2.1 over 5 seeds, below the 4-df 5 % critical value of 2.78) and the post-fix full-window pair (+0.43
± 0.24) do not show that "lag helps". The IC table (§4) shows the lag-1 *signal* is slightly weaker at
every horizon. If lag 1 really were better at the portfolio level, the mechanism would have to sit in
the risk and constraint layer (e.g. the ±0.1 sd band on short-term reversal, whose exposure includes
`r_t`). That is an open question and needs more seeds to settle. What the data do rule out is a
systematic one-day edge at lag 0 of anything like the demo's −0.84.

## 4. IC by lag (`ic_by_lag.py`)

Mean daily Spearman IC (×1000) of `score_{t−k}` vs the residualised forward return from `t` (labels
from `residualize_forward_returns` on the risk model's risk-only exposures). Backtest dates, eligible
names. `feat_*` rows are the processed features with **positive** sign; in the composite the low-risk
features enter with sign −1, and `strev_21` is not in any composite theme (it is an ML input only).

**h = 1**

| score | k=0 | 1 | 2 | 3 | 4 | 5 | k1/k0 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| composite | 9.03 | 8.38 | 8.40 | 8.27 | 7.93 | 8.29 | 0.93 |
| true_z | 8.43 | 7.75 | 7.24 | 6.94 | 6.99 | 5.99 | 0.92 |
| theme_value | 3.19 | 3.07 | 3.14 | 3.31 | 3.16 | 3.50 | 0.96 |
| theme_momentum | 1.47 | 1.24 | 0.79 | 1.27 | 0.92 | 0.96 | 0.85 |
| theme_quality | 10.44 | 10.00 | 9.64 | 9.27 | 9.35 | 8.94 | 0.96 |
| theme_investment | 11.06 | 10.74 | 10.87 | 10.18 | 10.18 | 10.46 | 0.97 |
| theme_low_risk | −2.45 | −2.60 | −2.21 | −1.90 | −2.38 | −1.86 | |
| theme_short_interest | −0.29 | −0.51 | −0.70 | −0.61 | −0.88 | −0.83 | |
| feat_strev_21 | 1.08 | −0.63 | −1.14 | −0.51 | −2.70 | −2.00 | |
| feat_max_ret_21 | 1.94 | 2.60 | 1.41 | 0.50 | 1.76 | 0.56 | |
| feat_beta_252 | 0.55 | 0.51 | 0.56 | 0.62 | 0.68 | 0.94 | |
| feat_idio_vol_252 | 1.24 | 1.17 | 1.38 | 1.49 | 1.54 | 1.47 | |
| feat_industry_momentum | 0.64 | 0.59 | 0.52 | 0.59 | 0.54 | 0.60 | |
| feat_mom_change | 4.25 | 5.01 | 4.07 | 4.59 | 4.09 | 4.24 | |

**h = 5**

| score | k=0 | 1 | 2 | 3 | 4 | 5 | k1/k0 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| composite | 20.37 | 19.93 | 19.45 | 18.93 | 18.69 | 18.30 | 0.98 |
| true_z | 16.61 | 15.54 | 14.72 | 14.12 | 13.74 | 13.22 | 0.94 |
| theme_quality | 21.69 | 21.02 | 20.44 | 19.88 | 19.74 | 19.13 | 0.97 |
| theme_investment | 23.51 | 23.25 | 22.77 | 22.23 | 22.30 | 21.94 | 0.99 |
| theme_value | 7.50 | 7.60 | 7.62 | 7.45 | 7.27 | 7.28 | 1.01 |
| theme_low_risk | 0.79 | 1.12 | 1.44 | 1.37 | 1.16 | 1.15 | |
| feat_strev_21 | 1.83 | 0.42 | −0.39 | −0.85 | −1.80 | −1.94 | |

**h = 20**

| score | k=0 | 1 | 2 | 3 | 4 | 5 | k1/k0 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| composite | 37.62 | 36.97 | 36.51 | 35.95 | 35.35 | 34.95 | 0.98 |
| true_z | 27.84 | 26.84 | 26.22 | 25.80 | 25.37 | 25.05 | 0.96 |
| theme_value | 18.64 | 18.73 | 19.06 | 19.32 | 19.44 | 19.68 | 1.00 |
| theme_momentum | 1.69 | 1.23 | 0.77 | 0.41 | −0.19 | −0.70 | 0.73 |
| theme_quality | 37.83 | 37.31 | 37.00 | 36.66 | 36.37 | 36.15 | 0.99 |
| theme_investment | 39.47 | 39.15 | 38.81 | 38.42 | 37.94 | 37.68 | 0.99 |
| theme_low_risk | 5.60 | 5.25 | 4.95 | 4.64 | 4.37 | 4.01 | 0.94 |
| theme_short_interest | −7.39 | −7.54 | −7.70 | −7.78 | −7.86 | −7.89 | |
| feat_strev_21 | 0.17 | −2.41 | −4.34 | −6.06 | −7.87 | −9.84 | |
| feat_max_ret_21 | −5.40 | −4.84 | −4.44 | −3.96 | −3.45 | −2.88 | |
| feat_idio_vol_252 | −11.47 | −11.33 | −11.17 | −11.06 | −10.99 | −10.93 | |

At lag 1 the composite keeps 93 % of its 1-day IC and 98 % of its 5- and 20-day IC. No theme carries a
short-horizon component that vanishes at k = 1. The features that read day-*t* returns (`beta_252`,
`idio_vol_252`, `max_ret_21`; the `mom_*` and industry-momentum features skip the last 21 days and do
*not*) have small ICs that barely change with k. The one feature that does change with k is `strev_21`,
and it is not in the composite. Hypothesis (C) is rejected for the composite.

**Where the composite's edge comes from** (`loo_themes.py`: static-weight composite with one theme
removed; IC ×1000):

| composite | h1 k0 | h1 k1 | h20 k0 | h20 k1 | h20 k1/k0 |
|:--|--:|--:|--:|--:|--:|
| all six | 9.40 | 8.86 | 38.52 | 37.73 | 0.979 |
| minus value | 8.60 | 8.15 | 33.22 | 32.42 | 0.976 |
| minus momentum | 9.98 | 9.25 | 41.28 | 40.51 | 0.982 |
| minus quality | 6.90 | 6.40 | 30.63 | 29.95 | 0.978 |
| minus investment | 5.41 | 4.81 | 24.73 | 23.96 | 0.969 |
| minus low_risk | 10.57 | 9.81 | 37.32 | 36.62 | 0.981 |
| minus short_interest | 10.10 | 9.51 | 44.29 | 43.63 | 0.985 |

At 20 days, investment, quality and value carry the edge: dropping one of them costs 36 %, 20 % and
14 % of the IC. Momentum, low-risk and short interest add nothing or *subtract*: dropping them raises
the IC. No subset of themes depends on the lag. Side finding, not a bug: the composite's 20-day IC
(0.038) exceeds the true score's (0.028). Value, quality and investment also load on the synthetic
**style-factor premia** (2 %/yr each). The label is residualised only on risk-only styles, so it keeps
those premia.

## 5. Synthetic generator: same-day structure (`generator_checks.py`)

Mean daily cross-sectional Spearman correlation (market-stripped returns, dates 260+):

| panel | vs r_{t−1} | vs r_t | vs r_{t+1} | Δpanel_t vs r_t | Δpanel_t vs r_{t+1} |
|---|---:|---:|---:|---:|---:|
| book_to_price | 0.0034 | 0.0031 | 0.0031 | −0.0006 | 0.0009 |
| earnings_yield | 0.0037 | 0.0028 | 0.0028 | −0.0006 | 0.0014 |
| gross_prof | 0.0096 | 0.0098 | 0.0097 | −0.0008 | 0.0019 |
| −asset_growth | 0.0093 | 0.0091 | 0.0091 | −0.0006 | 0.0023 |
| spread_bps | 0.0004 | −0.0002 | 0.0007 | −0.0020 | 0.0014 |
| volume | 0.0005 | 0.0004 | −0.0001 | −0.0004 | 0.0006 |
| dollar_volume | 0.0126 | 0.0121 | 0.0003 | 0.0214 | 0.0008 |
| short_interest | −0.0017 | −0.0017 | −0.0019 | 0.0091 | −0.0024 |
| borrow_fee | −0.0018 | −0.0017 | −0.0019 | 0.0025 | −0.0044 |
| true_z | 0.0089 | 0.0084 | 0.0083 | −0.0003 | 0.0015 |

* Characteristics correlate with `r_t` only as much as with `r_{t±1}`, which is the persistence of z.
  There is no contemporaneous spike.
* `dollar_volume` co-moves with `r_t` mechanically (price × volume). It carries no information about
  `r_{t+1}` (0.0003).
* There is no mechanical reversal. The cross-sectional correlation of `r_t` with `r_{t+1}` is 0.0010
  (t 0.5), and for the 21-day return vs `r_{t+1}` it is −0.0002. The time-series autocorrelation of
  residual returns is −0.0003. The clip at [−50 %, +100 %] is never hit.
* One quirk with no effect on the lag: the expo and noise of the characteristics are constant within a
  month, so the *daily change* in a characteristic reveals Δz exactly (corr of Δlog B/P with Δz ≈
  0.3). It predicts `r_{t+1}` only weakly (≤ 0.0023), and the composite uses levels, not changes.

The generator plants no same-day structure.

## 6. Simulator lag path audit

`qts/backtest/simulator.py` with `L = signal_lag_days` and `s = md.dates[pos(t) − L]`:
`alpha_fn(s)`, `risk_fn(s, alpha)` and the optimizer's `snapshot` are dated `s`; eligibility is as of
`s`; liveness (`price_ok`) is as of `t`; `w_current = w_pre` (drifted, marked at `t`); `nav = nav_t`;
costs are booked on `snapshot_t`; positions earn `R[t+1]`. This is the documented design and the code
matches it.

The existing test (`test_backtest.py::test_signal_lag_shifts_decisions_by_one_day`) used only an
optimizer double and a cost double. **New tests on the real stack** (`tests/test_signal_lag_real.py`,
fast tier):

1. `test_real_stack_lag1_weights_equal_lag0_weights_one_day_earlier` uses the real
   `SinglePeriodOptimizer` (Clarabel) and the real `TransactionCostModel`. Costs are booked but the
   optimizer's `trade_aversion` / `hold_aversion` / projection cost weight are 0 and the ADV caps are
   slack, so the optimum does not depend on the book. The lag-1 weights at `t` equal the lag-0 weights
   at `t−1` (atol 2e-5) and differ from the lag-0 weights at `t`. Gross returns equal `w_t · r_{t+1}`
   exactly.
2. `test_real_stack_lag1_inputs_are_dated_signal_date_and_book_is_current` records every solve
   (real optimizer, real costs) and checks five things: `risk.date == s`, `snapshot.date == s`, `alpha
   == alpha_fn(s)` (exactly), `nav == nav_t`, and `w_current` equal to the drift
   `w_post(t−1)·(1+r_t)·nav_{t−1}/nav_t`. It also checks that spread and fees equal
   `estimate(x·nav_t, snapshot_t)`, and that the NAV identity holds on the lagged path.

No off-by-one was found. In particular, the lag-1 book does not earn the wrong day's return, and
expected return and risk come from the same date.

Two side observations, neither a bug: (a) at lag ≥ 1 the optimizer prices costs on the `s`
snapshot while the simulator books them on `t`, as documented; (b) the first-day weights of the
zero-cost run depend slightly on the book, because the dated SEC/TAF fee schedules ignore the zeroed
`CostConfig` fee fields. That is expected; `tests/test_signal_lag_real.py` sets the optimizer's cost
aversions to 0 instead.

**Robustness bug fixed.** `ResearchPipeline.backtest_dates()` started at the first risk-model date for
every lag. For `L >= 2` the first signal date fell inside the risk warm-up and `risk.as_of` raised
(`only 251 factor-return days <= 2019-12-06; need 252`). L = 1 worked only because it landed exactly on
the boundary. `backtest_dates()` now drops the first `L` out-of-sample dates
(`qts/pipeline.py`). Test: `tests/test_pipeline.py::test_backtest_dates_leave_room_for_the_signal_lag`
(slow tier, it reuses the pipeline fixture). Lag 0 is unchanged; lag 1 starts one day later.

## 7. Lag-0 leakage audit (`pit_equivalence.py`)

For four decision dates (2019-12-24, 2021-06-22, 2023-01-03, 2023-10-17) the whole pipeline was refit
on `md.slice_until(t)` and every decision input at `t` was compared with the full-sample fit. Each
cell shows the max abs diff:

| input at t | before fix (4 dates) | after fix |
|---|---|---|
| universe mask (forward ADV?) | 0 | 0 |
| composite score (dynamic weights, crash scalar, mom vol scaling) | ≤ 9e-16 | ≤ 9e-16 |
| half-life | identical | identical |
| `ic_used` (Grinold) | 0.0133369 vs 0.0133573 (2019-12-24) | identical |
| expected_return | 7.6e-6, 2.8e-5, 0, 0 | ≤ 3e-18 |
| risk exposures B | **0.065, 0.044, 0.011, 0** | ≤ 3e-15 |
| factor cov F | 9.8e-4, 1.9e-4, 1.8e-5, 0 | ≤ 1e-15 |
| specific var D | 5.5e-5, 3.4e-7, 6.6e-7, 0 | ≤ 1e-17 |
| alignment vector | 9.0e-3, 1.3e-3, 0, 0 | ≤ 2e-15 |
| cost snapshot (ADV, σ, spread, price, borrow) | 0 (values) | 0 (values) |

The snapshot differs only in its *column set*: names that join the universe after `t` exist in the
full-sample `md_u` and get no-trade rows. Their values are identical.

**Root cause of the leak.** `apply_mask` keeps a ticker's full price history in the column set if the
ticker is *ever* in the universe. `qts.risk.exposures.raw_style_panels` regressed every name on
`returns.mean(axis=1)`, the mean over **all columns**, which includes names that join the universe
later. `standardize_style` then took the equal-weighted sd over every name with a raw descriptor,
which includes the same future entrants. At date `t`, the beta and residual-vol exposures and the
standardisation of momentum and short-term reversal therefore depended on *which names enter the
universe after `t`*: future-membership look-ahead. The alpha layer already used
`rets.where(valid).mean(axis=1)`. The labels inherited the leak through the risk exposures they are
residualised on, which is why `ic_used` moved.

**Fix** (`qts/risk/exposures.py`): the market proxy is `returns.where(md.eligibility()).mean(axis=1)`,
and the raw descriptors are masked to `md.eligibility()` before standardisation. Ineligible names are
never rows of `B` (`valid` needs a positive universe-masked market cap), so nothing downstream reads
the masked values. On raw `MarketData` (no universe) the behaviour is unchanged, because
`eligibility() == close.notna()`. Regression test:
`tests/test_risk.py::test_style_exposures_ignore_future_universe_members`. It fails on the old code and
passes on the new.

The leak moves lag 0 and lag 1 identically, so it does not explain the lag gap.

## 8. Golden-file change

`tests/test_golden.py` failed after the exposure fix. I reverted `qts/risk/exposures.py` alone and the
golden test passed again, so the exposure fix is the only cause (the `backtest_dates` change does
nothing at lag 0). I regenerated the file with `pytest tests/test_golden.py --update-golden`. Fields
that moved (rtol 1e-6):

| field | before | after |
|---|---:|---:|
| `ic_report.mean_ic` | 0.0643872 | 0.0631309 |
| `ic_report.ic_sd` | 0.149971 | 0.149101 |
| `ic_report.icir` | 1.52397 | 1.50296 |
| `ic_report.t_stat` | 0.743623 | 0.733369 |
| `dsr` | 0.395072 | 0.450455 |
| `summary.final_nav` | 992,807 | 996,384 |
| `summary.cagr_gross` | −0.00693 | 0.00814 |
| `summary.cagr_net` | −0.0299 | −0.0151 |
| `summary.ann_vol_gross` | 0.05287 | 0.05372 |
| `summary.ann_vol_net` | 0.05283 | 0.05371 |
| `summary.sharpe_gross` | −0.1055 | 0.1773 |
| `summary.sharpe_net` | −0.5479 | −0.2569 |
| `summary.sharpe_net_lo` | −1.1248 | −0.7823 |
| `summary.sharpe_t_stat_net` | −0.2673 | −0.1253 |
| `summary.newey_west_t_net` | −0.2819 | −0.1323 |
| `summary.max_drawdown_net` | −0.03256 | −0.03152 |
| `summary.calmar_net` | −0.9171 | −0.4792 |
| `summary.sortino_net` | −0.7358 | −0.3497 |
| `summary.hit_rate_net` | 0.4667 | 0.5167 |
| `summary.avg_gross_exposure` | 1.10684 | 1.11097 |
| `summary.avg_net_exposure` | 0.04484 | 0.04317 |
| `summary.avg_daily_turnover` | 0.113693 | 0.113897 |
| `summary.effective_holding_days` | 19.4707 | 19.5083 |
| `summary.avg_cost_bps_per_side` | 7.2451 | 7.1840 |
| `summary.avg_daily_cost_bps_of_nav` | 0.92723 | 0.92558 |
| `summary.total_costs_usd` | 5,588.8 | 5,583.7 |
| `summary.transfer_coefficient_mean` | 0.82778 | 0.82869 |
| `summary.realized_vs_exante_vol_ratio` | 0.96874 | 0.98093 |
| `summary.breakeven_cost_bps` | −1.947 | 3.319 |
| `backtest.nav_quartiles[0..2]` | 1,002,542 / 1,002,576 / 1,011,752 | 1,002,260 / 1,003,104 / 1,013,720 |
| `backtest.sum_abs_weights` | 66.41 | 66.66 |
| `backtest.sum_costs_by_kind.spread` | 0.046 | 0.320 |
| `backtest.sum_costs_by_kind.impact` | 3,219.9 | 3,192.2 |
| `backtest.sum_costs_by_kind.fees` | 1,743.0 | 1,741.8 |
| `backtest.sum_costs_by_kind.borrow` | 625.9 | 649.4 |

Why each group moved:
* **`ic_report.*`**: the composite scores are unchanged (`decile.*` did not move), but the 20-day
  label is residualised on the risk model's exposures, whose beta and residual-vol columns changed.
* **`dsr`**: it depends on the backtest Sharpe.
* **Backtest metrics**: new exposures change B, F, D, the beta-neutrality and style bands, and the
  alignment factor, so the 60-day path changes. As the earlier golden note says, a 60-day Sharpe has
  an SE of about 2; the sign change is path noise, not evidence of a better strategy.

`data_hash`, `decile.*`, `hurdles.*`, `summary.n_days`, `n_delistings` and `n_failed_solves` did not
change.

**Test runs after the fixes:** `pytest tests -q -p no:warnings -m "not slow"`: 244 passed, including
the regenerated golden file. `-m slow`: 24 passed.

## 9. Recommendation on `signal_lag_days`

* **Use lag 1 as the headline research setting.** Keep lag 0 as a reported diagnostic (the
  `--compare-lag` pair). Three reasons:
  1. Lag 0 is not executable. MOC/LOC orders must be in by 15:50 ET and the official close prints
     around 16:00 (handbook §"simulator timeline"), so a lag-0 signal uses a price the live system
     cannot have.
  2. On this market, lag 1 costs **nothing measurable** for the composite (IC retention 93–98 %; five
     seeds give a mean Sharpe difference of +0.16 ± 0.08 SE, with lag 1 *not* worse). The conservative
     setting is close to free for slow signals. Lag 1 is somewhat pessimistic compared with the real
     15:45 snapshot, by about one day of a 60–100-day half-life.
  3. On real data, the direction of the lag-0 bias is known and optimistic: bid–ask bounce and
     auction-imbalance reversal on `P_t`. Lag 1 removes it, and no synthetic test can measure it.
     I have not flipped the config default here. It is design decision D4, and changing it moves every
     run, the golden file and the hurdle history. The evidence above supports flipping it, ideally
     together with renaming the CLI comparison to "lag 1 (headline) vs lag 0 (optimistic bound)".
* **Do not lag the day-*t*-return features by construction.** In the composite, only `beta_252`,
  `idio_vol_252` and `max_ret_21` read `r_t`. They are 252- and 21-day statistics whose IC does not
  change between k = 0 and k = 1 (§4). Lagging them would add a second, inconsistent timing convention
  for no measured benefit. The one feature to watch is **`strev_21`** (an ML input, risk-only style):
  it is the only feature whose IC changes sign within 1–2 days. Once the ML sleeve is on, the lag-0 vs
  lag-1 comparison should be rerun with `use_ml=True` before trusting a fast sleeve.
* **Judge lag effects by paired, multi-seed comparisons, not one path.** On this market, small code or
  setting changes move a 1000-day Sharpe by 0.5–0.8 (§1, §3). Report the paired bootstrap SE of the
  difference (`/tmp/lag_investigation/paired_diff.py`) next to any lag comparison.

## 10. Follow-ups (not done here)

* `docs/handbook/09_backtesting.md` still says "There is **no execution-delay option** in the simulator
  today". `ExecutionConfig.signal_lag_days` / `DailyBarSimulator.run(signal_lag_days=)` now exist. The
  "How to remove it" paragraph should point to them, and to the `backtest_dates` shift for L ≥ 2.
* `runs/demo_post_refactor.txt` should be regenerated on committed code. Stored demo outputs should
  come only from clean trees (`git_dirty: false`).
* Path sensitivity: books from two runs whose inputs differ only slightly correlate 0.5–0.75 over four
  years. That is worth a separate look at the optimizer's no-trade region and vol-target projection.
  It is what makes single-path Sharpe comparisons unreliable here.
