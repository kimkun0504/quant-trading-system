# Per-industry breakdown of a finished run (2026-11-14)

## Why

The strategy already treats industries specially at every layer. Features are
industry-demeaned, labels are residualised on industry dummies, and the traded score is
neutralised on industry. The optimizer keeps each industry's net weight inside a band, and
the risk model has FF48 industry factors, shrunk toward their FF12 sector when thin. So
the book is close to industry-neutral by construction. That says nothing about *where the
P&L comes from*. A stock-selection edge that works in most industries is robust. One that
lives in three industries is a bet on those industries' regime, even with zero net
industry exposure. `qts breakdown industries RUN` answers that question for any finished
run. It is a diagnostic: no trial is logged and no strategy number changes.

## What it computes

Full definitions are in the module docstring of `qts/research/industry_breakdown.py`
(including the JSON shape, `qts.industry_breakdown/1`). In short, per risk-model industry
(FF48 on the real store) and per FF12 sector:

1. **Positioning.** Mean long, short, net and gross weight; names held long and short;
   share of the book's gross.
2. **P&L.** The contribution $\sum_{i \in \text{industry}} w_{i,t}\, r_{i,t+1}$ uses the
   run's post-trade weights and the per-name returns the simulator booked. Those are the
   cleaned closes, with a halt's move landing on the resumption day and held delistings at
   their `delistings.parquet` return. The contributions are checked against the recorded
   gross return. On the synthetic runs the residual is about 1e-18 per day; without the
   delisting override it would be 1.6e-4. The command also reports costs (allocated pro
   rata, below), share of the total, Sharpe, Newey–West t and monthly hit rate.
3. **Signal quality.** The rank IC of the traded score against the residualised forward
   target, computed inside each industry on the run's own IC dates. An industry is skipped
   on dates when it has fewer than 5 scored names, and the coverage is reported. The fits
   come from the stage cache. A cold cache is announced, and `--no-refit` refuses it. The
   score check compares the re-fitted book IC with the run's `ic.parquet` (max
   |difference| 0 to 6e-17 on the synthetic runs).
4. **Breadth.** These statistics count *active* industries only:
   - the positive count, with a one-sided binomial p against a fair coin;
   - the top-5 P&L share and the rest;
   - the effective number of industries, $1/\sum_k s_k^2$ over absolute P&L shares;
   - the count with $|t| \ge 2$ against $0.05 \times$ count, plus the Holm list;
   - the book without its top 5;
   - the verdict.

## Verdict thresholds, fixed before any real result

These were set on 2026-11-14 in code (constants at the top of
`qts/research/industry_breakdown.py`, echoed in the JSON's `thresholds`). At that point
the breakdown had been run only on synthetic markets. **Do not change them after looking
at a real breakdown**: that would fit the verdict to the answer. If a threshold must
change, write the reason here first, then re-run.

| constant | value | meaning |
|---|---|---|
| `ACTIVE_MIN_GROSS_SHARE` | 0.25 % | an industry counts in the breadth statistics when it held at least this share of the book's gross |
| `TOP_K` | 5 | "the best few industries" |
| `VERDICT_MIN_ACTIVE` | 5 | fewer active industries → undetermined |
| `VERDICT_MIN_BOOK_T` | 2.0 | the book's gross P&L must be positive with Newey–West t ≥ 2, else undetermined |
| `CONCENTRATED_MAX_REST_SHARE` | 15 % | concentrated if less than this share of the gross P&L comes from outside the top 5 ... |
| `CONCENTRATED_MAX_NEFF_RATIO` | 0.10 | ... or the effective number of industries is below max(2, 0.10 × active) |
| `BROAD_MIN_REST_SHARE` | 40 % | broad needs at least this share from outside the top 5 ... |
| `BROAD_MIN_POSITIVE_SHARE` | 60 % | ... and at least this share of active industries with positive P&L ... |
| `BROAD_MIN_NEFF_RATIO` | 0.25 | ... and an effective number of at least 0.25 × active |
| otherwise | | moderately concentrated |
| `T_CLEAR`, `CHANCE_RATE`, `HOLM_ALPHA` | 2.0, 5 %, 5 % | the $\|t\| \ge 2$ count, its chance level, the Holm list's family-wise level |
| `IC_MIN_NAMES` | 5 | fewer scored names in an industry on an IC date → no IC that date |

**Why these values (back-of-envelope, written before any real run).**

Take a book whose gross P&L has t ≈ 4 over 20 years, spread uniformly over 48 equally
sized industries. Each industry's P&L stream then has t ≈ 4/√48 ≈ 0.6, so about 70 % of
industries should be positive. The five best are the five luckiest, so they hold roughly
35–40 % of the P&L, and the rest about 60 %.

Now put the same edge in three industries. The rest is noise, and its sum is about zero.
The top 5 are the three carriers plus two lucky noise industries, so the rest share is
around zero or negative.

The 40 % and 15 % cut-offs sit between these two cases with margin. Real industries are
not equal-sized, which lowers the uniform case's rest share somewhat, so 40 % is not set
higher. The effective-number conditions guard the cases the rest share can miss, such as
a single industry dominating the absolute P&L.

Below a book t of 2 the uniform and the concentrated cases overlap, so the honest answer
is "undetermined". The breadth numbers are still printed for the record.

**Synthetic check** (`tests/test_industry_breakdown.py`, constant ±1/N book on 12
industries × 10 names, 1,000 days, 2 % daily noise):

| planted edge | rest share | positive | N_eff | book t | verdict |
|---|---|---|---|---|---|
| one industry, 56 bp/day | −9 % | 7 / 12 | 1.6 | 8.0 | concentrated |
| all 12 industries, 10 bp/day | 52 % | 12 / 12 | 11.7 | 16.6 | broad |
| 8 of 12 industries, 10 bp/day | 29 % | 8 / 12 | 8.5 | 11.2 | moderately concentrated |
| none | | | | | undetermined |

## Sealing and loading

- **The loader.** Market data comes only through `qts.research.data.load_research_market`,
  with the run's store flags. They are parsed from `manifest.json`'s `command`; the ledger
  records do not store loader flags. Explicit `--store …` flags replace the parsed ones.
  The load-coverage guard applies; `--allow-partial-coverage` passes through as in
  `qts run`.
- **Data check.** The loaded close panel must hash to the run's `market_hash`. Otherwise
  the breakdown refuses, because other flags or a refreshed store are not the run's market.
- **Sealed run.** A run with `sealed_end` set is refused when no seal is in force, before
  anything is loaded. It is also refused when the seal's `holdout_start` moved, or when
  the sealed load does not end on the run's `sealed_end`.
- **Full-history run under the seal.** Runs from before the seal, `--unseal` runs and the
  mid-cap check are broken down on their pre-holdout dates only (`window: truncated`). Their
  stage cache is cold, because the sealed data is not the data the run was fitted on.
- **Nothing is logged** to the trial ledger. The pipeline is built with `ledger_path=None`
  and only `fit` is called, never `run`. The tests check that the ledger is unchanged.

## Limits

- **Costs per industry are approximate.** `costs.parquet` holds costs per day. Trading
  costs are allocated pro rata to traded value and borrow costs pro rata to short value.
  Impact is convex in the trade size, and fees and borrow rates differ by name, so
  illiquid industries are under-charged. The totals are exact: the net reconciliation is
  within 3e-16 on the synthetic run.
- **Small industries.** An IC from 5 names is noisy. Read `IC n` and `cov%` before reading
  `IC`.
- **Top 5 is picked after the fact.** The top 5 by P&L include the luckiest industries, so
  "without the top 5" is pessimistic by design.
- **Runs before 2026-11-12** have no `delistings.parquet`. Held delistings are then booked
  at 0, the reconciliation line says NOT RECONCILED, and a note explains why.
- **Peak-cap runs.** Runs made with `--max-names 3000 --min-mcap-ever 3e8`, before
  `docs/notes/prefilter_lookahead_2026-11-14.md`, are refused by the coverage guard unless
  `--allow-partial-coverage` is given, and their numbers carry that caveat.

## Owner command

For the corrected sealed baseline, once it exists:

```bash
qts breakdown industries runs/<run_id> --store data/store --start 2000-01-01 --max-rank 1500 \
    2>&1 | tee runs/<run_id>/industry_breakdown.txt
```

The store flags can be left out when the run was made with `qts run --store …`. They are
read from the manifest.

**Expected runtime (owner's Mac, warm cache):** about 6–10 minutes. Most of it is the
store load (about 5,300 ids at `--max-rank 1500`) and reading the two cached fits. The
breakdown itself takes about 10 s at 5,100 days × 2,500 names × 48 industries. With
`--no-ic` (no fit) it takes about 4–7 minutes. A cold cache adds a full refit, about
45 minutes with ML.

## Not verified

None of this has been run on real data; the sandbox has no store. In particular, the
real store's labels, the runtime and the size of the cost-allocation error per industry
are unverified.
