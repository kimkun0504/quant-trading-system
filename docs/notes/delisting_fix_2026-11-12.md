# Delisting fix: return by reason, on the stock (2026-11-12)

**Found:** 2026-11-12, `scripts/attribute_run.py` on the real-data run
`20261112T031754882789Z_425a06bb` (Sharadar, top-1000 US universe, ML sleeve,
`alpha.horizon_days=60`).

## Evidence (real data)

| Measurement | Value |
|---|---|
| Σ w·r of the run's book, annualised (a delisted name earns 0 in this sum) | +5.02 % |
| Recorded gross return, annualised | +1.49 % |
| Reconciliation gap = delisting haircuts | **−3.53 % a year** |

The simulator charged every delisting of a held name −30 % **against the position**: a long lost 30 %
and a short lost 30 % too. At gross 2.0 a gap of 3.53 % a year means about 5.9 % of gross exposure a
year sat in names that delisted while held (2.0 × 5.9 % × 30 %). Two things were wrong with the rule:

1. In a large-cap universe most delistings are cash or stock acquisitions, which complete at about the
   last traded price. There is no 30 % to lose.
2. The −30 % figure (Shumway 1997) is the average return **on the stock** for performance-related
   delistings (bankruptcy, exchange-initiated). A short gains it.

The rule removed more than the strategy's whole gross alpha. The labels were less wrong: they applied
−30 % on the stock (not against a position), but for every reason, acquisitions included.

## What changed

**1. Delisting reason as data.** `qts/data/sharadar.py::derive_delisting_reasons` builds a
`delisting_reasons` table (`id, delisting_date, reason, source_action`) from the stored
`corporate_actions` + `security_master`: the ACTIONS rows of the id's ticker within ±5 business days of
its delisting date, mapped by the module constant `DELISTING_REASON_ACTIONS`:

| ACTIONS code | reason |
|---|---|
| `acquisitionby`, `mergerto` | acquisition |
| `bankruptcyliquidation`, `regulatorydelisting` | performance |
| `voluntarydelisting` | voluntary |
| plain `delisted`, or no row | unknown |

Precedence when several: performance > acquisition > voluntary > unknown. `qts data sharadar derive
--store <store>` rebuilds it without a download and prints the counts; `sync` writes it too.
`ParquetStore.as_market_data` loads it into the new optional `MarketData.delisting_reasons`.

*Not verified:* the reading "`acquisitionby` = the row's `ticker` was acquired by `contraticker`"
follows the action names and was not checked against Sharadar's field documentation. The derive report
prints every action code found next to a delisting so the real counts settle it (an acquirer does not
delist the week it buys something; if `acquisitionof` dominates there, the mapping is inverted and the
constant must be corrected).

**2. Simulator and labels.** `CostConfig` now carries a return on the stock per reason and two booking
knobs:

| field | default | meaning |
|---|---|---|
| `delisting_return_acquisition` | 0.0 | deal completes at about the last price |
| `delisting_return_performance` | −0.30 | Shumway (1997) |
| `delisting_return_voluntary` | −0.10 | going dark / OTC |
| `delisting_return_unknown` | −0.30 | no reason on file: treated as performance |
| `delisting_short_gain_haircut` | 1.0 | share of a short's gain on a negative return that is booked; 0 = shorts get nothing |
| `delisting_against_position` | False | True = the old rule (return × sign of the position) |
| `delisting_haircut` | None | deprecated; a value sets performance and unknown, with a `DeprecationWarning` |

A long earns w·r; a short earns w·r as well, so it gains when r < 0. The labels
(`qts/alpha/targets.py`) use the same per-reason map, so an acquisition's label is the return to its
last price. Diagnostics: `diagnostics["delistings"]` rows gained `reason`, `side`, `stock_return`,
`pnl`, `pnl_usd`; `diagnostics["delisting_summary"]` and `BacktestResult.delisting_summary()` give
counts and P&L by reason and side; summary field `delisting_pnl_ann`; report card line `delistings`;
run directory file `delistings.parquet`.

**3. Stage-cache key.** The code part of each stage's key was the hash of every `qts/**/*.py` file, so
a simulator-only edit forced both fits (25 minutes on the real universe). It is now the hash of the
packages the stage can import: `risk_fit` → `qts/core`, `qts/data`, `qts/risk`; `alpha_fit` → those +
`qts/alpha` (verified by a static import-graph test), plus the source of the pipeline's fit methods.
`qts/validation` is not imported by either fit. The data part: the panel hash no longer includes
`delisting_reasons`; that table is keyed on the alpha fit alone.

**4. `scripts/attribute_run.py`.** The reconciliation line is followed by the split of its gap into
delisting P&L by reason and side and the remainder, with the same events re-priced under the current
rule. Old runs (no `delistings.parquet`) are reconstructed from the book.

## Golden file

`tests/golden/pipeline_summary.json` did **not** move and was not regenerated. The golden market
(`tests/conftest.py::GOLDEN_SPEC`) plants no delistings (`n_delistings` in the golden summary is 0), a
market without a reasons table hashes as before, and with every delisting `unknown` at −30 % the labels
are bit-identical to the old ones. Only the config hash changes (new `CostConfig` fields), which the
golden file does not pin.

A/B on a synthetic market that does have delistings (golden spec + `n_delistings=12`, 250 backtest
days, composite only; no reasons table, so every event is `unknown`; 1 long and 3 shorts were held
when they delisted):

| | new default | `delisting_against_position=True` (old rule) | `delisting_short_gain_haircut=0` |
|---|---|---|---|
| Sharpe gross / net | 1.92 / 1.82 | 1.07 / 0.97 | 1.53 / 1.43 |
| CAGR gross | 10.04 % | 5.41 % | 7.70 % |
| `delisting_pnl_ann` | +1.37 % | −2.96 % | −0.79 % |

The labels (hence both fits) are identical in the three columns: the knobs act in the simulator only,
and both fits were served from the stage cache for columns 2 and 3. This is a synthetic market with
planted delistings and proves the mechanics only; it says nothing about the size of the effect on real
data.

## What is not known yet

- The real count table (`qts data sharadar derive`) has not been produced: how many delistings are
  acquisitions, and how many stay `unknown`, is open. `unknown` is charged −30 % on the stock, so with
  the symmetric default a short in an `unknown` delisting books a 30 % gain. If the unknown bucket is
  large, read the action codes the report lists for it before trusting the P&L.
- Whether the real result improves by the full 3.53 % a year is not established: the labels change
  too (acquisitions no longer carry −30 %), so the refit alpha model and its book differ.
- The symmetric short gain is the optimistic end. `--set cost.delisting_short_gain_haircut=0` is the
  conservative end; both reuse the cached fits.

## Tests

`tests/test_simulator_handcomputed.py` (long and short × each reason, gain haircut, escape hatch
reproducing the old numbers, deprecated field), `tests/test_label_halts.py` (labels by reason),
`tests/test_sharadar.py` (derivation rule, fixture store, `derive` CLI),
`tests/test_stage_cache.py` (keys vs touched packages, import graph, delisting inputs),
`tests/test_scripts_diagnose.py` (gap split).
