# The peak-cap store prefilter left out universe names (look-ahead) — 2026-11-14

## Finding

Every large-cap real-data run so far loaded the Sharadar store with
`qts run --store data/store --start 2000-01-01 --max-names 3000 --min-mcap-ever 3e8`.
That prefilter keeps the 3,000 ids with the largest **peak** market cap over the whole
load window, before the point-in-time universe rules run.

How it showed up: the baseline re-run under the holdout seal
(`runs/20261114T043028256534Z_6ec7fd65`, data through 2023-11-08) had net Sharpe
**0.47** on 2003-06-05 .. 2023-11-07. The reference run of the *same configuration*
(`runs/20261113T023313333450Z_e4667d90`, full history) had **0.65** on the same dates,
and the two daily net-return series correlated only **0.92**. Same config, same dates,
same code path: they should have traded the same universe. The only input that differed
was the load window, and with it which 3,000 ids the peak-cap prefilter kept.

`scripts/check_load_coverage.py` on the store, for the baseline universe (screened
top 1,000; the bottom-size cut is 0.2, so ≥ 1,250 loaded names passing the screens are
needed per date):

| prefilter, window | ids | screened top 1,000 fully loaded | worst date | dates short of names |
|---|---|---|---|---|
| `--max-names 3000 --min-mcap-ever 3e8`, 2000-01 .. 2026-11 | 3,000 | **23.7 %** of dates | 82.0 % (2000-03-09) | 1,080 |
| `--max-names 3000 --min-mcap-ever 3e8`, 2000-01 .. 2023-11-08 | 3,000 | **34.4 %** of dates | 86.0 % (2002-10-09) | 943 |
| `--max-rank 1500`, 2000-01 .. 2026-11 | 5,637 | 100 % | 100 % | 0 |
| `--max-rank 1500`, 2000-01 .. 2023-11-08 | 5,336 | 100 % | 100 % | 0 |

Two effects, both look-ahead:

1. **Missing names.** A name in the top 1,000 on some date whose peak cap never reached
   the 3,000 largest peaks of the window was never loaded. Whether a 2002 name could be
   traded depended on how large it became up to 2026. Up to 18 % of a day's universe was
   missing (early 2000s, when today's large peaks crowd out the period's mid-sized
   leaders).
2. **Moved bottom-size cut.** `exclude_bottom_size_quantile` is a quantile of the
   *loaded* cross-section. With fewer than 1,250 loaded names passing the screens (the
   minimum was 1,015), the 20 % cut fell inside the top 1,000 and dropped more names.

The universe also depended on where the window ended, which is why the sealed re-run
traded a different book on the same dates.

## What it affects

- The reference run (net Sharpe 0.68) and all nine real-data trials in family
  `pipeline_phase2:c1da477b63d2`. The sign of the bias is not known in advance: the
  missing names are ones that did *not* become large, i.e. relative losers, which a
  long-short book could have shorted.
- **Not** the mid-cap check (`runs/20261114T033438682619Z_9109eaca`, PASS). It loaded
  with the screened-rank prefilter `--max-rank 2750` and measured 100 % coverage (V2)
  before it ran.
- The paper-trading defaults (`--max-names 3000 --min-mcap-ever 3e8`), now changed.

## Fix

- **`--max-rank K`** for `qts run`, `qts ledger rekey` and `scripts/diagnose_signal.py`.
  The research tools already had it. It loads every id that ranked ≤ K by market cap,
  among the names passing the universe's price / ADV screens, on at least one date of the
  window. That covers a top-N universe by construction when
  K ≥ N / (1 − `exclude_bottom_size_quantile`) (1,250 for the baseline). **K = 1,500**
  leaves a 20 % margin, because the SQL ADV window runs over a name's own rows and the
  panel's over the date index, so names with gaps differ slightly. Under `--max-rank` the
  universe on a date no longer depends on where the window ends
  (`tests/test_load_coverage.py::test_max_rank_universe_does_not_depend_on_the_window_end`).
- **Load-coverage guard** (`qts/research/coverage.py`), called by
  `qts.research.data.load_research_market` before anything is loaded. That covers
  `qts run --store`, signal tests, sweeps (the widest member's universe), capacity, risk
  calibration, the diagnostics loader and the API's store jobs; paper books call it from
  `StoreSource.load`.
  - `--max-rank` is checked arithmetically.
  - A peak-cap prefilter is measured with DuckDB over the window actually loaded, and
    refused unless the screened top N is fully loaded on ≥ 99 % of dates and no date is
    short of names.
  - `qts run --allow-partial-coverage` loads anyway, with a warning.
  - The ledger record carries `load_coverage` (method, ok, shares) on every run.
  - `$QTS_LOAD_COVERAGE=enforce|warn|off`; the test suite uses `off`, and
    `tests/test_load_coverage.py` turns it on.
- **Paper books** default to `--max-rank 1500`. A `book.json` written earlier keeps its
  peak-cap prefilter and is refused; re-init it.

## Next

1. Re-run the frozen baseline under the seal with the corrected load (one run, a
   correction of a data bug, not a new research choice):
   `qts run --store data/store --start 2000-01-01 --max-rank 1500 --config configs/baseline.yaml --ml --cache -v`.
   The new close panel starts a new ledger family. Merge it into
   `pipeline_phase2:c1da477b63d2` (`qts ledger merge-families`) so its DSR counts the
   nine trials that shaped the configuration, then `qts ledger restate` the run.
2. That run replaces `runs/20261114T043028256534Z_6ec7fd65` as the pre-seal benchmark
   and becomes the paper book's reference run.
3. The full-history numbers of the old reference run are not re-measured (that would
   unseal). The 2003-06 .. 2026-11 large-cap figures in earlier notes stay as recorded,
   with this caveat.

## Result of the corrected sealed baseline (run `20261114T064935329466Z_6ec7fd65`, 2026-11-14)

Same frozen configuration (`configs/baseline.yaml`, config hash `6ec7fd650490ce4e`), `--max-rank 1500`
(5,336 ids; coverage OK by construction), data through 2023-11-08, 2003-06-05 .. 2023-11-07:

| | old load, reference run (full history) | old load, sealed | **corrected, sealed** | mid caps (`--max-rank 2750`) |
|---|---|---|---|---|
| net Sharpe on 2003-06 .. 2023-11 | 0.65 | 0.47 | **0.54** | 0.74 |
| net return / yr | 2.9 % | 2.1 % | **2.5 %** | 5.0 % |
| correlation with the corrected run | 0.88 | 0.89 | 1 | 0.25 |
| positive calendar years (of 21) | 14 | 13 | **14** | 16 |

Corrected run in full:
- net Sharpe 0.54 (gross 0.74), NW t 2.34, vol 4.5 %, max drawdown −16.9 %;
- rank IC 0.023 (NW t 3.80), IC positive in 6/6 folds, net Sharpe by fold 0.61 / −0.51 / 0.27 / 1.51 / 0.81 / 0.77;
- cost per side 4.7 bp, breakeven 31 bp, net Sharpe 0.45 at 2× cost.

Gates: FAIL on the decile long-short t (2.62 < 3.0) only.

DSR: the report card's 0.993 counts one trial (new family `pipeline_phase2:34d0b5de91a6`). With the nine
earlier trials merged in (10 trials, Nyholt N_eff 6.3, expected maximum Sharpe 0.66), it restates to **0.29**.
That figure comes from a dry run on a copy of the owner's ledger; the owner runs the merge and the restate.

Reading:
- The old load inflated the large-cap Sharpe on these dates by about 0.1 (0.65 against 0.54).
- The corrected large-cap result is modest and, after the trial count, not statistically convincing on its own.
- The stronger evidence remains the mid-cap confirmation (PASS on pre-registered criteria, clean load). The two
  books stay weakly correlated (0.25).
- An ex-post equal-risk mix of the two on the pre-seal dates would have had net Sharpe 0.81. That is hypothetical,
  a new research choice to test under the seal as a counted trial.

`research/seal.yaml` `inherit_family` now names `pipeline_phase2:34d0b5de91a6`.
