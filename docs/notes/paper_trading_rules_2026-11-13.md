# Paper trading: pre-registered monitoring and stop rules (2026-11-13)

Written before the first paper run of `configs/baseline.yaml`. The numbers below are
frozen into `paper/<book>/book.json` (`stop_rules`) by `qts paper init`, together with
the SHA-256 of the expectation tables they are evaluated against
(`expectations_sha256`); `qts paper run` refuses to evaluate against edited tables.
Changing a rule means starting a new book (new name), and this note gets a dated
addendum saying why. Code: `qts/paper/monitor.py::StopRules`, `evaluate`;
`qts/paper/expectations.py::calibrate`.

## What the rules are for

A paper book of a backtested strategy is a **pipeline and plumbing check**, not
evidence that the edge is real: with a backtest net Sharpe of 0.68 the standard
error of a one-year live Sharpe estimate is about 1.0, so a year of paper trading
cannot tell 0.68 from 0. Paper fills are priced with the same cost model as the
backtest, so costs are not validated either (that needs real fills). What paper
trading *can* catch quickly is a book that is not doing what the backtest says it
does: wrong data, a broken serving path, a risk model that no longer describes the
book, an optimizer that fails, accounting errors, or a drawdown so deep that the
backtest says it essentially never happens. The rules are built for that.

Levels: **OK**, **WATCH** (look at it this week), **STOP** (freeze trading). STOP
does not liquidate: the book keeps its positions and is marked to market daily, but
the day's orders are cancelled and no new orders are generated until
`qts paper ack --name <book> --note "<why it is safe to continue>"`. An
acknowledgement covers the STOP rules active on the acknowledged day; each of them is
then reported as WATCH ("STOP acknowledged") until it clears; a rule that clears and
fires again, or a different rule, stops the book again.

## The yardstick

The reference backtest run (`runs/20261113T023313333450Z_e4667d90` for the
baseline): its daily **net** returns (`returns.parquet`) and its summary (turnover,
cost per side, gross, realised / ex-ante vol ratio), its ex-ante vol series and its
sampled rank IC. From the daily net returns a moving-block bootstrap (block 21 days,
2,000 paths, seed 0) gives, for every horizon from 1 day to 5 years, the distribution
of the cumulative log return, the maximum drawdown and the annualised Sharpe over
the first `n` days. Blocks keep the volatility clustering and autocorrelation an
i.i.d. resample would destroy. "Horizon" is the number of execution days since the
book's first fill.

## Why the quantile levels are calibrated (and not "the 99th percentile")

The rules are checked every day. A boundary at the 1 % quantile of the drawdown
distribution *at each horizon* is crossed on some day of the first year far more often
than 1 % of the time, because the daily checks are 252 looks at the same path.
Measured on a baseline-like return series (Sharpe 0.68, 5 % annual vol, Student-t(4)
daily shocks; `qts/paper/expectations.py::calibrate`, 10,000 bootstrap paths):

| Rule (checked daily, first year) | fixed per-horizon quantile | P(false alarm in year 1) |
|---|---|---|
| drawdown STOP at the 1 % quantile | 1 % | **8.2 %** |
| drawdown WATCH at the 5 % quantile | 5 % | 30 % |
| Sharpe STOP at the 2.5 % quantile (from day 126) | 2.5 % | **7.7 %** |
| Sharpe WATCH at the 10 % quantile (from day 63) | 10 % | 30 % |

A 1-in-12 chance of stopping a book that is doing exactly what the backtest says is
too high. So the boundary of each return rule is a **simultaneous band**: the quantile
level `α` is chosen at init on the reference run's own bootstrap so that a book whose
returns behave like the backtest's crosses the boundary on at least one day of its
first year with the target probability:

| Rule | target P(false alarm in year 1) | calibrated level on the test series |
|---|---|---|
| drawdown STOP | 2.5 % | 0.26 % quantile at each horizon |
| drawdown WATCH | 10 % | 1.26 % |
| Sharpe STOP (from day 126) | 2.5 % | 0.70 % |
| Sharpe WATCH (from day 63) | 10 % | 2.29 % |
| either return STOP | — | 4.2 % (achieved) |

The calibrated levels, the achieved rates and the fixed-quantile rates for comparison
are printed by `qts paper init` and stored in `expectations/summary.json`
(`calibration`). For the baseline-like series the STOP boundaries are, in maximum
drawdown / annualised Sharpe: 21 days −4.2 % / −8.9; 63 days −7.3 % / −4.7; 126 days
−9.5 % / −3.2; 252 days −12.7 % / −2.0; 504 days −17.3 % / −1.2 (the real boundaries
come from the reference run and are shown by `qts paper status`). The fixed quantiles
(`dd_stop_quantile` 1 %, `dd_watch_quantile` 5 %, `sharpe_stop_quantile` 2.5 %,
`sharpe_watch_quantile` 10 %) are only the fallback for a book without a
calibration.

## The rules

| # | Rule | WATCH | STOP | Why this number |
|---|---|---|---|---|
| 1 | **Drawdown**: live maximum drawdown to date (net NAV) vs the bootstrap max-drawdown distribution at the same horizon | below the boundary with a 10 % year-1 false-alarm rate | below the boundary with a 2.5 % year-1 false-alarm rate | the one rule that protects capital; 2.5 % keeps a correct book running in 39 of 40 first years |
| 2 | **Sharpe**: live annualised net Sharpe vs the bootstrap Sharpe at the same horizon | from day 63, 10 % band | from day **126**, 2.5 % band | before six months the estimate's standard error is > 1.4, so only a catastrophic Sharpe could be called; the band then sits near −3 at 6 months and −2 at a year: a broken book, not an unlucky one |
| 3 | **Cumulative return** vs the bootstrap 2.5 % / 97.5 % cone (and the analytic cone `n μ ± 1.96 σ √n` of daily log returns, reported) | below the 2.5 % quantile (from day 21) | — | informational: rule 1 already catches the paths that matter |
| 4 | **Realised vs ex-ante vol**: 21-day realised vol of daily net returns / mean ex-ante vol of the book held, relative to the backtest's own ratio (`realized_vs_exante_vol_ratio`, floored at 1) | above 1.5× for 10 consecutive days | above **2×** for **20** consecutive days | a calibrated risk model's 21-day vol estimate has a sampling sd of ~15 %, so 2× is ~7 sd; 20 consecutive days is almost a full non-overlapping window: the risk model no longer describes the book (or the book is not what the optimizer built). Relative to the backtest's ratio because a known, stable miscalibration (the real backtest ran at 1.3–1.5) is part of what was backtested |
| 5 | **Turnover**: 21-day mean daily turnover, after the first 10 execution days (the build), vs the backtest's mean | > 2× | — | paper costs come from the cost model, so a turnover blow-up costs money only through the model; WATCH, then look at `qts paper orders` |
| 6 | **Cost per side**: 21-day trading cost / traded notional (bp), after the build | > 2× the backtest's | — | same reason; a doubling means the trades moved to less liquid names |
| 7 | **Invested gross** | below 0.5× the backtest's mean gross for 5 consecutive days after the build | above **1.1×** the config's gross limit (2.0 → 2.2) | an empty book was the failure mode of four real backtests (risk-model blow-ups); a gross above the optimizer's limit by 10 % is beyond what five days of drift can do and means a sizing or accounting bug |
| 8 | **Optimizer** | any failed rebalance solve, or a risk-model ex-ante vol alert (> 200 % on the incoming book) | **3** consecutive failed rebalance solves | a failed solve holds the book; three in a row is 15 trading days without a rebalance (the batch gate allows 2 % of solves) |
| 9 | **Realised rank IC** of the traded score, once labels mature (60-day horizon + 1 day, so the first values arrive after ~3 months) | z < −2 against the backtest's mean IC with an overlap-adjusted standard error (`n_eff = n / 60`) | — | overlapping 60-day labels make a few months of IC almost uninformative; WATCH only |
| 10 | **Data quality**: share of NAV in held names that are missing from the store or have an unchanged as-traded close for ≥ 5 sessions; unfilled orders | > 2 % of NAV, or any unfilled order | > **10 %** of NAV for **3** consecutive days | a halt is normal (held, marked flat, recorded as an event); a tenth of the book without prices for three days means the store is not being updated |

Not a rule: an unusually *good* path. There is no upper cut-off; an implausibly good
path is investigated through the same reconciliation tools (`qts paper reconcile`,
`scripts/attribute_run.py`), not stopped.

## Operational rules

* The job runs at 07:05 ET on weekdays (`qts paper schedule --print`). It refuses to
  run when the store does not yet hold the session the vendor should have published
  (exit code 3, "stale data"); the next run catches up day by day.
* A day already in the journal is never processed again; `--redo-last` re-processes
  the last day (after a vendor correction) and the new record supersedes the old one.
* Orders generated after the fill day's 15:50 ET MOC cutoff are labelled
  `mode: backfill` in the journal (they could not have been submitted); only
  `mode: live` days count as paper-trading evidence. A replica-sizing book is always
  `backfill`.
* The frozen walk-forward models are never retrained: a model is trained once, when
  the data first cross its scheduled refit date, and served unchanged afterwards. A
  vendor restatement of its training rows is recorded as a warning, not a refit.
* Code changes after init are recorded (`code_version`, config hash) and reported as a
  warning on the next run; they do not stop the book. A change that alters the
  strategy's behaviour needs a new book.
