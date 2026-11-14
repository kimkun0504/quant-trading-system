# Mid-cap out-of-sample check of the frozen baseline — pre-registration (2026-11-13)

**Status: pre-registered. Written before the check was run; nothing in this file was
changed after any mid-cap result was seen.** The result is appended at the end, under
"Result", with the run id, after the one run. `qts check midcap` prints the SHA-256 of
this file and stores it on its ledger record, so the text in force at run time is on
record.

## Why this check

Every design choice of the baseline (`configs/baseline.yaml`; reference run
`runs/20261113T023313333450Z_e4667d90`, net Sharpe 0.68, ledger family
`pipeline_phase2:c1da477b63d2`, 9 trials, restated DSR ≈ 0.62–0.70) was made after
seeing the large-cap panel (top 1,000 by market cap) over 2003-06..2026-11. There is
no clean holdout in time. There is one in the cross-section: **no run so far has
traded a stock ranked 1001–2000 by market cap.** This check runs the frozen baseline
*method* — same configuration, same code, same features and hyper-parameters, the
walk-forward ML sleeve re-trained on the mid-cap cross-section exactly as it was on
large caps — once, on that unseen cross-section, over the full history.

What it tests: whether the strategy's edge is a property of the method (it transfers
to 1,000 stocks it was never tuned on) or of the 1,000 stocks it was tuned on.
What it does **not** test: the time dimension (same 2003–2026 period, same market
regimes; the 3-year seal of `research/seal.yaml` is the time holdout for new research),
nor the fitted large-cap coefficients (the models are re-fitted on mid caps).

## What is run (fixed)

- Configuration: `configs/baseline.yaml` with exactly one override,
  `universe.rank_offset = 1000` (with `universe.top_n_by_mcap = 1000`: per date, the
  names ranked 1001–2000 by market cap among those passing the baseline's price ≥ $5,
  ADV ≥ $2M and bottom-20 %-size screens; monthly refresh, 3-refresh exit hysteresis,
  all as for large caps). Everything else — horizon 60, recency half-life 8 years,
  feature set v1, construction, costs, vol target — unchanged.
- Data: the same Sharadar store from `--start 2000-01-01` as the large-cap runs, full
  history (exempt from the seal by design), loaded with the *screened-rank* prefilter
  `--max-rank 2750` instead of the large-cap runs' peak-cap prefilter (see "Load" below:
  a peak-cap cut can miss names that were mid caps once and never grew). The command
  checks the load before it runs and refuses when it is short (V2).
- ML sleeve on (`--ml`, phase 2), NAV $5M, signal lag 1, as the reference run.
- Ledger: its own family `confirm:midcap:<market_hash[:12]>`, one trial. It does not
  enter the large-cap family's trial count. The command refuses a second run of the same
  configuration (`--force` overrides with a warning that a repeat is no longer
  out-of-sample, and the repeat is flagged on its record).

## Pre-registered criteria

All statistics are the ones the report card already prints, computed by the same code
as for the reference run.

**PASS** requires all three:

| # | Criterion | Statistic | Threshold |
|---|---|---|---|
| C1 | The strategy makes money net of costs | Newey–West t of the mean daily **net** return (`newey_west_t_net`) | net Sharpe > 0 **and** NW t ≥ 2.0 |
| C2 | The signal transfers | mean rank IC of the traded score vs the risk-residualised 60-day label, per purged walk-forward fold (the 6 blocks of the `ic_sign` gate) | IC > 0 in ≥ 4 of 6 folds |
| C3 | The P&L is not one episode | net Sharpe per purged walk-forward fold (the 6 blocks of the fold gate) | > 0 in ≥ 4 of 6 folds |

Outcome **FAIL** if any of C1–C3 fails.

**Validity conditions.** If any fails, the outcome is **INCONCLUSIVE** (the run did not
test the hypothesis; the reason is recorded; a repeat needs `--force`, which the
record shows):

| # | Condition | Threshold |
|---|---|---|
| V1 | Universe size | median number of eligible names per backtest day ≥ 900 |
| V2 | Load coverage (checked before the run) | every name ranked ≤ 2,000 by market cap among the names passing the price / ADV screens loaded on ≥ 99 % of dates, and on every date at least 2,500 loaded names passing the screens (or all of the store's, when it has fewer) |
| V3 | Engine health | failed solves ≤ 2 % of rebalances, `risk_model_sanity` gate not FAIL, book invested on ≥ 90 % of days |

A criterion that cannot be computed (no fold long enough) also makes the outcome
INCONCLUSIVE — it cannot happen on the 22-year run, only on a truncated one.

**Reported, not part of the verdict:**

- S1 cost robustness: net Sharpe at **2×** the modelled cost per side (the
  cost-sensitivity row), and the breakeven cost per side. The cost model has no
  spread data for this store: every name pays the flat `cost.default_half_spread_bps`
  = 2 bp half-spread. Quoted half-spreads of US mid caps are wider than of large caps
  (several bp in the 2010s, more before decimalisation's effects settled and in
  2008–09), so the modelled mid-cap costs are optimistic in the spread term. Impact
  (`b σ √(Q/ADV)`) and the ADV participation cap (5 %) do scale with each name's own
  volatility and dollar volume, and borrow doubles below $2B market cap, so those
  terms are not optimistic. A PASS with S1 net Sharpe ≤ 0 is reported as "PASS,
  cost-fragile".
- S2 the single-trial PSR (DSR with N = 1).
- S3 side by side with the reference large-cap run: gross / net Sharpe, CAGR, IC,
  cost per side, turnover, effective names, share of traded names at the ADV cap.

### Why these thresholds

- **t ≥ 2 (not the 3.0 discovery hurdle, not a DSR).** This is one pre-registered
  confirmation of a strategy already selected elsewhere, not a search: there is one
  test, so the conventional two-sided 5 % level is the right bar. The multiple-testing
  penalty belongs to the discovery (it is in the large-cap family's DSR), not to the
  replication — the standard in the replication literature (Hou, Xue & Zhang 2020 use
  |t| ≥ 1.96 for replications; Harvey, Liu & Zhu's 3.0 is for new factors chosen from
  many). With ~22 years of daily returns, t ≥ 2 ⇔ net Sharpe ≳ 0.43.
- **4 of 6 folds** is the repository's own gate rule (`validation.min_fold_pass_fraction
  = 2/3`) on the same purged blocks, so a mid-cap PASS means the same thing as the
  large-cap gates.
- **Power.** If the true mid-cap net Sharpe were the large-cap 0.68, P(t ≥ 2) ≈ 0.88;
  at 0.5 it is ≈ 0.64; at 0.43 it is 0.5. A C1 failure with C2 and C3 passing is
  therefore weak evidence against the method; a C2 failure is strong evidence.

### What each outcome means and what will be done

| Outcome | Reading | Action |
|---|---|---|
| PASS | The method's edge transfers to an unseen cross-section: the large-cap result is less likely to be a selection artefact of the 9 trials. | None to the baseline. Mid caps become a candidate second universe for *new* research under the seal (a new trial family). |
| PASS, cost-fragile | Signal and P&L transfer at modelled costs, not at twice the spread. | Mid-cap trading is not considered before spread data (TAQ / IBKR fills) exists. |
| FAIL on C1 only | The signal transfers, the net P&L does not clear t = 2 (costs or noise; see power). | Report; no change. |
| FAIL on C2 | The signal does not transfer. The large-cap edge may be specific to large caps or a product of the search. | Treat the large-cap DSR as optimistic; weigh the paper-trading results more heavily. |
| INCONCLUSIVE | The run did not test the hypothesis. | Fix the cause, document it here, repeat once with `--force`. |

**Commitments.** No parameter of the baseline is changed because of this result. The
check is not repeated (except the INCONCLUSIVE case). Mid-cap results are not used to
tune anything. The ledger record and this note's "Result" section are the record.

## Load

The universe takes, per date, ranks 1001–2000 by market cap among the *loaded* names
that pass the price ≥ $5 and ADV ≥ $2M screens and the bottom-20 % size cut (a
quantile of the loaded names passing the screens). Two things must therefore hold for
the load not to change the universe: (a) every name that ranks ≤ 2,000 among the
store's names passing the screens on some date is loaded; (b) on every date at least
2,000 / 0.8 = 2,500 loaded names pass the screens, so the size cut stays below rank
2,000 (loading more names below does not change ranks 1001–2000).

The large-cap runs' peak-cap prefilter (`--max-names 3000 --min-mcap-ever 3e8`) does
not guarantee (a) for mid caps: a name that was ranked 1,800 in 2003 at $400M and was
acquired in 2005 has a peak far below that of thousands of later names, and
`--min-mcap-ever 3e8` drops names that ranked ≤ 2,000 in early 2009 at $250M. The check
therefore loads with `--max-rank 2750` (`ParquetStore.tickers_by_screened_rank`): every
id that ranks ≤ 2,750 among the names passing the same screens on at least one date,
computed in DuckDB over the long tables. 2,500 gives (b) by construction; the 10 %
margin covers the SQL screens' approximation (their ADV window runs over each name's own
rows, the universe's over the panel's dates; they differ only for names with gaps).
`qts check midcap` measures (a) and (b) before it runs
(`qts.research.midcap.prefilter_coverage`, printed first; `--dry-run` prints only it,
with the number of ids and an estimate of the load's memory) and refuses when V2 fails.

Memory (`scripts/bench_store_load.py`, synthetic store, this sandbox: 2 vCPU, 7 GB):
peak RSS of `as_market_data` + the universe mask ≈ 0.3 GB + 200 bytes × ids × days
(1,000 / 3,000 / 5,000 ids × 1,000 days: 0.48 / 0.85 / 1.23 GB). At about 6,700 days
from 2000-01-01 that is ≈ 1.3 GB per 1,000 ids. The number of ids `--max-rank 2750`
loads on the real store is unknown here (every US company that was ever roughly a
Russell-2500 constituent since 2000: several thousand); `--dry-run` prints it and the
estimate.

## Command

```bash
qts check midcap --store data/store --start 2000-01-01 --dry-run     # coverage + plan only
qts check midcap --store data/store --start 2000-01-01 --cache -v 2>&1 | tee runs/midcap_check.log
```

(`--config` defaults to `configs/baseline.yaml`; `--ml` is implied: the baseline is
phase 2. `--dry-run` first prints the coverage table and the plan without running or
logging anything.)

## Result

(to be appended after the run: run id, the verdict lines as printed, the side-by-side
table; nothing above this heading changes.)
