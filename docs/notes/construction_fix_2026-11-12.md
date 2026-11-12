# Construction fix: neutral score, vol target as a cap, slower trading (2026-11-12)

**Found:** 2026-11-12, `scripts/attribute_run.py` on the real-data run
`20261112T031754882789Z_425a06bb` (Sharadar, ML sleeve, `alpha.horizon_days=60`). The signal is
there (ICIR 1.2); the portfolio built from it by the optimizer does not earn it.

## Evidence (real data; nothing below is from a synthetic market)

"Ideal book (c)" is the paper portfolio of table 1 of the script: every eligible name held, weights
proportional to the cross-sectional residual of the combined score's rank on industries + beta + the
risk-only styles, no optimizer, gross of costs.

| Measurement | Ideal book (c) | The optimizer's book |
|---|---|---|
| Gross Sharpe | **1.04** (positive in 6 of 6 folds) | **0.20** |
| … sampled every 21 days | 0.92 (turnover 2.2 % of gross a day) | — |
| … on a 20-day EWMA of the score | 0.97 | — |
| Ex-post vol at gross 2 | 3.4 % | 7.5 % |
| Correlation with (c) | 1 | 0.61 |
| Specific P&L | — | −1.34 % a year, on 82 % of the variance |
| Realised / ex-ante vol | 0.95 (the risk model's forecast for (c)) | 1.46, all of it in the specific piece |
| Vol target (10 %) | — | "capped" on 100 % of solves (gross limit 2.0); mean γ 4.55, 5th percentile 0.10 |
| Risk-only style exposures | neutral by construction | at their ±0.1 bands on 48–80 % of days; industry band on 100 % |
| Turnover | 2.2 % of gross a day is enough (21-day row) | 13.6 % of NAV a day |
| Trades below 5 bp of NAV | — | 67 % of all trades |
| Daily trading that survives netting over 21 days | — | 44 % |
| Turnover on ML refit days vs other days | — | 37 % vs 13 % |

Three separate failures:

1. **The score asks for directions the book may not hold.** The label is residualised on industries,
   beta and the risk-only styles, the score is not. The optimizer spends its bands on the leftover
   loading (styles pinned at the band on half to four fifths of days) and bends the rest of the book
   around it (correlation 0.61 with the residual-score book).
2. **Sizing chases a vol target the caps make unreachable, and concentrates.** A diversified
   market-neutral book at gross 2 runs at 3.4 % vol; the target is 10 %. The two-sided γ search ends
   with the gross limit binding at a γ where the alpha wants several times the allowed gross. The
   limit's multiplier then works as an L1 penalty: only names with |alpha| above it are held — the
   high-score, high-volatility names, at the position cap. Result: 7.5 % vol at gross 2, specific risk
   82 % of variance, and a realised/ex-ante gap of 1.46 that sits entirely in the specific piece (the
   same risk model forecasts the diversified book (c) at 0.95).
3. **It trades six times more than the horizon needs.** Daily re-trimming of every name (two thirds
   of trades under 5 bp), most of it undone within a month, plus a full re-rank on every ML refit day.

## What was built (all config-flagged; defaults in bold)

| # | Flag | Default | What it does |
|---|---|---|---|
| 1 | `alpha.neutralize_score` | **True** | Per date, the combined score is residualised (equal-weighted OLS over the eligible names) on intercept + industry dummies + beta + the risk-only styles *as of that date* — the label residualiser's exposure source — and re-standardised, before Grinold scaling. `qts/alpha/score_processing.py`. |
| 3b | `alpha.score_smoothing_halflife_days` | **10.0** | Causal per-name EWMA of the neutralised score (weights normalised over the days the name had a score: an entrant starts at its own first score), projected on the day's exposures again, re-standardised. 0 = off. |
| — | (consequence) | | `AlphaModel.traded_score_` is what `as_of` scales; `oos_ic_`, the trailing IC, `IC_used` and the half-life are computed on it. `combined_score_` stays the raw blend. |
| 2 | `portfolio.vol_target_is_cap` | **True** | γ starts at `risk_aversion` (2.0, unchanged) and is only raised, to bring the ex-ante vol down to the target; never lowered to reach it. `vol_target_binding`: `below_target` / `target`. `sizing.gamma_cap_search`. False = the old two-sided search. |
| 2b | `portfolio.leverage_cap_raises_gamma` | **True** | *Added during the build, see the next section.* With the cap form, γ is also raised until `gross_leverage_max` stops binding: the mean-variance book is scaled to fit under the limit instead of being thresholded by it. `vol_target_binding = "leverage"`. |
| 3a | `portfolio.rebalance_every_days` | **5** | The simulator solves on dates `i % 5 == 0` and lets the book drift in between; forced exits of ineligible names and delistings still run every day (`DailyBarSimulator._forced_exit_step`). The pipeline / CLI `rebalance_every` argument now defaults to this. |
| 3c | `portfolio.min_trade_weight` | **0.0005** | After the solve, trades below 5 bp of NAV are cancelled unless they close the position or belong to a forced exit. A projection: the book can end marginally outside a band; `n_small_trades_dropped`, `max_constraint_violation`, `violated_constraint` per solve, summed in the simulator diagnostics and printed on the report card. |
| 4 | summary | | `effective_n_names`, `specific_var_share`, `exante_vol_at_gross2` in `BacktestResult.summary()`, the report card (`diversification` line), the API `BacktestSummary` and the UI run dossier. |

Old behaviour, bit for bit (verified: reproduces the previous golden file):

```bash
--set alpha.neutralize_score=false --set alpha.score_smoothing_halflife_days=0 \
--set portfolio.vol_target_is_cap=false --set portfolio.rebalance_every_days=1 \
--set portfolio.min_trade_weight=0
```

## Finding during the build: `risk_aversion = 2` does not give a diversified book by itself

The brief was "γ is searched only upward from the base `risk_aversion` (default stays 2.0 → check it
yields a diversified book)". It does not, whenever the alpha asks for much more gross than the limit —
which is the real-data situation (the old run's mean γ of 4.55 is already *above* 2, and that book was
the concentrated one). Measured on the synthetic market, 800 eligible names, 60 out-of-sample days,
everything else at the new defaults:

| | γ = 2, gross limit a plain constraint | γ raised until the limit is slack | Rank-weighted neutral paper book |
|---|---|---|---|
| mean γ | 2.0 | 9.0 | n/a |
| average gross | 2.00 | 1.94 | 2 |
| effective names | 156 | 306 | 587 |
| ex-ante vol at gross 2 | 8.5 % | 4.4 % | 6.9 % |
| specific vol at gross 2 | 5.8 % | 3.5 % | 2.5 % |

and in a single solve on 120 names with an alpha 8× the usual scale (position cap widened to 8 % so
the gross limit, not the position cap, decides): effective names 31 → 68, ex-ante vol at gross 2
15.6 % → 7.0 %.

So `leverage_cap_raises_gamma` (default True) extends the "only upward" rule to the leverage limit:
raising γ divides the whole mean-variance book by a common factor, so the book that just fits under
gross 2 keeps every name of the diversified optimum. The base stays 2.0. The acceptance test
(`test_default_config_does_not_concentrate_against_the_rank_weighted_neutral_book`, 480 names)
passes with the flag on and is asserted to fail with it off.

The specific **share** of variance is not the right yardstick against the rank book: the optimizer
hedges the factor risk the rank book carries, so its share is higher (62 % vs 14 % above) while its
specific *vol* and total vol at gross 2 are what the table shows. The test compares effective names,
specific vol at gross 2 and total vol at gross 2.

**This flag is the one decision in this change that was not in the brief.** It is cheap to A/B on real
data (the alpha fit is cached; only the backtest re-runs):

```bash
... --set portfolio.leverage_cap_raises_gamma=false
```

## Golden file and A/Bs (synthetic)

`tests/golden/pipeline_summary.json` regenerated (config hash changes: 6 new fields). The golden run
is 60 out-of-sample days of an 80-name market, so its Sharpe has a standard error of about 2; the
moves below are mechanics, not evidence about performance.

Golden run (60 days), one feature switched on at a time from the old behaviour:

| Variant | Net Sharpe | Turnover / day | Cost / side | Holding days | Mean γ | Effective names | Vol at gross 2 |
|---|---|---|---|---|---|---|---|
| all off (= previous golden) | 2.216 | 9.6 % | 6.9 bp | 26 | 0.24 | 65 | 11.4 % |
| + `neutralize_score` | 0.066 | 10.4 % | 6.6 bp | 25 | 0.20 | 67 | 11.1 % |
| + smoothing (half-life 10) | −0.462 | 7.5 % | 7.1 bp | 33 | 0.40 | 66 | 10.8 % |
| + `vol_target_is_cap` (± leverage flag: identical, the gross limit never binds here) | 1.099 | 8.3 % | 6.7 bp | 23 | 2.00 | 55 | 12.0 % |
| + `rebalance_every_days` = 5 | 1.891 | 5.3 % | 4.9 bp | 45 | 0.20 | 64 | 11.4 % |
| + `min_trade_weight` = 5 bp | 2.293 | 9.0 % | 5.7 bp | 28 | 0.24 | 65 | 11.4 % |
| **all on (new golden)** | **−1.840** | 4.3 % | 3.9 bp | 50 | 2.00 | 60 | 10.4 % |

The same market over its whole out-of-sample span (192 days; SE of a Sharpe ≈ 1.1):

| Variant | Gross / net Sharpe | Turnover / day | Cost / side | Mean IC of the traded score |
|---|---|---|---|---|
| all off | 2.94 / 2.66 | 8.2 % | 7.0 bp | 0.112 |
| + `neutralize_score` | 2.30 / 1.99 | 8.2 % | 7.4 bp | 0.125 |
| + smoothing | 2.54 / 2.30 | 6.2 % | 7.5 bp | 0.124 |
| + `vol_target_is_cap` | 2.04 / 1.78 | 6.5 % | 6.9 bp | 0.112 |
| + `rebalance_every_days` = 5 | 2.32 / 2.17 | 4.6 % | 4.6 bp | 0.112 |
| + `min_trade_weight` | 2.94 / 2.72 | 7.5 % | 5.6 bp | 0.112 |
| all on | 1.97 / 1.87 | 3.2 % | 3.8 bp | 0.138 |

Larger synthetic markets, old behaviour (all off) against the new defaults and against the new
defaults with `leverage_cap_raises_gamma` off (composite alpha, NAV 5M, each run a full fit +
backtest):

| Market | Seed | Old: gross / net | New: gross / net | New, leverage flag off: gross / net |
|---|---|---|---|---|
| 480 names, 292 days, `ic_true` 0.08 | 11 | 0.66 / 0.27 | −0.23 / −0.42 | 0.25 / 0.10 |
| | 3 | 1.32 / 0.96 | 0.23 / 0.07 | 0.43 / 0.30 |
| | 5 | 1.95 / 1.56 | 2.26 / 2.09 | 2.16 / 2.03 |
| | 7 | 1.80 / 1.32 | 1.06 / 0.83 | 1.08 / 0.90 |
| | mean | 1.43 / 1.03 | 0.83 / 0.64 | 0.98 / 0.83 |
| 240 names, 1,004 days, `ic_true` 0.04 (the demo market) | 0 | 1.12 / 0.80 | 0.76 / 0.64 | 0.69 / 0.57 |
| | 1 | 2.21 / 1.84 | 2.50 / 2.32 | 2.42 / 2.25 |
| | mean | 1.67 / 1.32 | 1.63 / 1.48 | 1.55 / 1.41 |

Means of the other columns (old → new → new with the leverage flag off):

| Market | Turnover / day | Cost / side | Ann. vol | Effective names | Ex-ante vol at gross 2 | Realised / ex-ante | Mean γ |
|---|---|---|---|---|---|---|---|
| 480 names (4 seeds) | 11.7 % → 6.3 % → 5.3 % | 8.3 → 4.4 → 6.0 bp | 7.2 % → 6.2 % → 9.1 % | 182 → 210 → 136 | 7.2 % → 5.8 % → 9.0 % | 1.00 → 1.09 → 1.02 | 4.3 → 6.2 → 2.0 |
| 240 names (2 seeds) | 9.3 % → 4.0 % → 3.8 % | 10.8 → 6.1 → 6.8 bp | 8.5 % → 7.1 % → 7.8 % | 122 → 129 → 122 | 9.0 % → 7.2 % → 8.0 % | 0.95 → 1.03 → 1.01 | 1.4 → 3.3 → 2.0 |

Paired net-Sharpe difference new − old: −0.38 on the 480-name market (four seeds: −0.69, −0.89,
+0.53, −0.49; SE 0.31) and +0.16 on the 240-name market (two seeds: −0.16, +0.48). The correlation
of the new and old books' daily gross returns is 0.82–0.85.

What the synthetic A/Bs do and do not show:

- **Mechanics work as designed**: turnover falls by half to two thirds, cost per side by 40–50 %, the
  holding period reaches the 50-day target, γ never goes below 2, effective names rise and vol at
  gross 2 falls where the gross limit binds.
- **The score processing is not what costs Sharpe.** Paper portfolios (rank-weighted, daily, lag 1,
  480 names, four seeds, 292 days each): raw score 1.53, neutralised 1.75, neutralised + smoothed
  1.70, at a turnover of 8 % of gross a day instead of 23 %; the mean daily IC is unchanged
  (0.049 → 0.048).
- **Net Sharpe on the synthetic market: lower with the new defaults on the golden market and on the
  480-name market (−0.38, about 1.2 standard errors), higher on the 240-name / 1,004-day market
  (+0.16). None of it is distinguishable from path noise** (paired SE of a Sharpe difference on one
  292-day path ≈ 0.6–0.8). *Gross* Sharpe is lower on the 480-name market by 0.60 (≈ 1.9 SE): slower
  trading gives up some of a clean planted signal there, and the lower costs buy back about a third
  of it.
- **The leverage flag on synthetic data**: fewer bets' worth of concentration (effective names 210 vs
  136, vol at gross 2 5.8 % vs 9.0 % on the 480-name market) and a net Sharpe that is 0.19 lower on
  the 480-name market and 0.07 higher on the 240-name market — again inside the noise. The synthetic market does not have the pathologies the fix targets: its risk model
  is the true one (specific risk really is idiosyncratic, so concentrating in the highest-alpha names
  is not punished), its planted alpha is proportional to σ·z (exactly what a thresholded book loads
  on), costs are low and the signal is clean every day. On that market the old construction was
  already close to efficient. The claim "this fixes the real run" rests on the real-data table at the
  top and has to be confirmed by the re-run below, not by these numbers.
- Realised / ex-ante vol of the default book on the 480-name market is 1.01–1.20 (old: 0.92–1.08).
  The alignment variance (`risk.alignment_nu_scale = 0.05`) was calibrated under the old sizing;
  re-check it on real data after the re-run (`scripts/calibrate_alignment_nu.py`).

Min-trade violations on the golden run (default config, 12 solves): 212 trades cancelled; largest
violations 0.0019 on the ±0.05 net band, 0.0013 SD on a ±0.10 style band, 0.0006 on a ±0.03 industry
band, 0.0005 on the 2 % position cap.

## Tests

`tests/test_construction.py` (31 tests): neutralised score orthogonal to the design and unit SD;
exposure-function path and warm-up fallback; hand-computed EWMA with an entrant and a gap; point in
time (future scores, prices and caps perturbed → rows ≤ t unchanged; `as_of(t)` equals a fit on data
≤ t); `oos_ic_` is the traded score's; both knobs off → `traded_score_ is combined_score_`; γ ≥ base
for 12 (target, base) pairs with the base book returned untouched when it is under the cap; the
unreachable-target case against the old search; leverage cap resolved by raising γ; min-trade rules
on a constructed book and end to end (a forced exit and a closing trade below the threshold both
trade); rebalance dates exactly `dates[::k]` for k = 1, 3, 5 with exact drift in between and the kwarg
overriding the config; forced exits on hold days; diversification numbers by hand; the default
config against the rank-weighted neutral book.

Existing tests that assumed a solve every day or an exact optimum now say so explicitly
(`portfolio.rebalance_every_days=1`, `min_trade_weight=0`, `vol_target_is_cap=False` for the tests of
the two-sided search); none was deleted.

## Owner: re-run

```bash
qts run --store data/store --start 2000-01-01 --max-names 3000 --min-mcap-ever 3e8 --cache -v --ml \
    --set alpha.horizon_days=60 2>&1 | tee runs/real_ml_h60_construction.log
python scripts/attribute_run.py --store data/store --start 2000-01-01 --max-names 3000 \
    --min-mcap-ever 3e8 --cache --ml --set alpha.horizon_days=60 --run runs/<new_run_id>
```

Cache: **both fits rebuild.** The stage-cache key includes a hash of every `qts/**/*.py` source file,
and this change edits files under `qts/`, so the risk fit misses even though the `risk` config
section is untouched (the alpha fit would miss anyway: new `alpha` fields). Both are stored again
under the new key, so the attribution command and any `--set portfolio.*` A/B afterwards hit the
cache.

What to look at, in order:

1. Report card: `diversification` line (effective names in the hundreds; ex-ante vol at gross 2 near
   (c)'s 3–4 %), `sizing` line (mostly "leverage"/under the cap; mean γ well above 2), `trading` line,
   turnover per day (target: a few % of NAV), realised / ex-ante vol (was 1.46).
2. Attribution table 1: row (g) — the rank book of the model's traded score — against (c); the line
   under the table prints `corr((g), (c))`, `corr(realised gross, (c))` (was 0.61) and
   `corr(realised gross, (g))`.
3. Table 2: specific P&L (was −1.34 % a year), exposures vs bands (styles were at the band on 48–80 %
   of days).
4. If the gross Sharpe is still far below (c)'s: A/B `portfolio.leverage_cap_raises_gamma=false` and
   `alpha.score_smoothing_halflife_days=0` (backtest only, fits cached).
