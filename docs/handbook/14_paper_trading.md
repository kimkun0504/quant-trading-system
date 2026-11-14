# Paper trading

<callout icon="🧭" color="gray_bg">
The paper book runs the frozen strategy (`configs/baseline.yaml`) once per trading day as if it were trading: at 07:05 ET it refreshes the Sharadar store, fills yesterday's orders at yesterday's close with the system's own cost model, marks the book, handles corporate events, computes today's signal with the frozen models and writes the market-on-close orders for the next close. Everything goes into an append-only journal, so the book can be rebuilt, audited and served to the UI. Every day the live numbers are compared with what the reference backtest predicts, and pre-registered stop rules turn a book that does not behave like its backtest into WATCH or STOP. A STOP freezes trading until someone acknowledges it with a note. The book is evidence that the pipeline works on live data. It is **not** evidence that the edge is real (a year of paper P&L cannot separate a 0.68 Sharpe from zero), and since fills are priced by the cost model, it does not validate costs either.
</callout>

## 1. What it is

### What it is
A *paper book* is a simulated trading account that follows a strategy in real time without sending orders to a broker. Each trading day it does what a live account would do: receive fills for the orders it placed, revalue its positions, and decide tomorrow's trades. The only thing that is simulated is the fill itself: the book assumes its market-on-close (MOC) orders filled at the official close, and charges the cost the transaction cost model predicts (→ see **Transaction costs**).

### Why we need it here
A backtest is one long batch computation over history. Running the same strategy one day at a time, on data that arrives one day at a time, exercises everything the batch never does: the daily vendor refresh, a model that must be served without being refit, order sizing in whole shares, holidays, missed days, corporate actions arriving live, a job that crashes half-way. It also builds the first stretch of genuinely out-of-sample history for the frozen configuration. Paper trading comes before real money because a pipeline error found on paper costs nothing.

### What it is evidence of, and what it is not
- **It is** a test that the live pipeline reproduces the backtest (the replica check in §7 matches the batch simulator day by day), that data arrive on time, that the book is invested, sized and hedged as designed, and that the risk model still describes it.
- **It is not** evidence of alpha. The backtest's net Sharpe is 0.68; the standard error of a one-year Sharpe estimate is about $\sqrt{(1 + S^2/2)/T} \approx 1.0$ (annualised, $T$ = 1 year), so after a year the live Sharpe could be anywhere between −1.3 and 2.7 for a book that is exactly as good as the backtest.
- **It is not** a cost test. Paper fills are priced with the same model the backtest used. Real fills (TCA, → see **Transaction costs › TCA**) are the only way to validate the cost model.
- **Backfilled days are not out-of-sample evidence.** A day whose orders were generated after that fill day's 15:50 ET MOC cutoff could not have been traded; the journal labels it `mode: backfill`. Only `mode: live` days count.

## 2. The daily timeline

The job for trading day $t$ runs the next morning, after Sharadar has published day $t$'s prices, and uses data through $t$ only. With `execution.signal_lag_days = 1` (→ see **Execution › Why the signal cannot use the official close**), the signal of the close of $t$ is traded at the close of $t+1$:

```text
        close t-1                   close t                        close t+1
    ------|---------------------------|------------------------------|------->
          | job "t-1" (07:05, day t):  | job "t" (07:05, day t+1):     |
          |  orders O(t-1) for close t |  1. mark t-1 -> t             | O(t) filled by
          |                            |  2. fill O(t-1) at close t    | job "t+1"
          |                            |  3. borrow, delistings, splits|
          |                            |  4. signal at t -> O(t)       |
          |                            |  5. status, journal           |
```

1. **Refresh.** `qts data sharadar sync` incremental (rows updated since the last data date minus 5 days, so late vendor corrections are re-read). The API key comes from `NASDAQ_DATA_LINK_API_KEY` or `.env` and is read only by `qts/data/sharadar.py`, which never prints it.
2. **Freshness.** The store must hold the latest session whose prices are due: session $d$ is due from 06:00 New York time on the calendar day after $d$ (`qts/paper/data.py::expected_last_session`, NYSE calendar from `qts/data/calendar.py`). Otherwise the job stops with "stale data" (exit code 3) and the next run catches up.
3. **Catch-up.** Every session after the last journal day through the target day is processed in order, each from data cut at that day. A day already in the journal is never processed twice.
4. **Per day** (`qts/paper/engine.py::PaperEngine.run_day`): mark to market, apply the pending orders at the close, accrue one day of borrow on the shorts, process delistings and halts, then decide the next close's orders.
5. **Status** against the backtest (§5), then the journal line is appended. That append is the day's commit point.

**Rebalance schedule.** `portfolio.rebalance_every_days = 5`: the optimizer runs for every fifth execution date counted from the book's first fill date (the simulator's `i % 5 == 0` with $i$ = position in the run's dates). On the other days nothing is traded except names that left the universe, which keep unwinding on their linear `exit_ineligible_days = 5` schedule, exactly as in the simulator.

## 3. Serving the models without a daily refit

### The problem
The batch pipeline fits the risk model (~3 min on the real store) and the alpha model (~21 min) in one go. Most of the alpha time is the walk-forward ML sleeve: one LightGBM + MLP + ridge ensemble (three seeds each) per refit, refits every 252 trading days from a fixed anchor (the first feature date). A new data day changes the data hash, so the stage cache misses, and a naive daily job would spend 25 minutes every morning.

### How it works
The model of the refit that starts predicting at position $p_0$ is trained only on label rows that are fully realised strictly before $p_0$ (embargo $h + L$ = 61 days; → see **Alpha model › Walk-forward training**). It is therefore *the same model* whether the data end one day or one year after $p_0$. `qts/paper/serving.py` exploits this by composition, with no change to the pipeline or the trainer:

- `FrozenFoldStore` persists every fold model the first time it is trained (`paper/<book>/models/fold_<train_end>.pkl` + a JSON sidecar with the training window, feature columns, a fingerprint of the training labels and the book identity).
- `fit_signals(md, ...)` runs the ordinary `ResearchPipeline.fit` on the data through the signal date, with an `ml_factory` whose models (`FoldModelProxy`) load themselves from the store when their fold exists and train and store themselves when it does not.
- Everything else is recomputed by the unchanged batch code on the data through the signal date: features, labels, the composite, the neutralised and smoothed traded score, the trailing IC, the half-life, Grinold scaling and the whole risk model. The risk model is refit on each rebalance day (about 3 minutes on the real store); there is no incremental update, because the batch risk model's VRA, eigenfactor and statistical-companion state are cheapest to reproduce exactly by refitting.

So:

| Day | What the job fits | Real store, estimate |
|---|---|---|
| hold day (4 of 5) | nothing (marks, fills, forced exits; ex-ante vol from the last solve's risk model) | load + mark: 1–2 min |
| rebalance day | risk model + alpha model with frozen folds + one optimizer solve | 6–9 min |
| refit day (once a year) | as a rebalance day + training one new fold | + 2–4 min |
| the book's first run | every historical fold is trained once (the batch fit) | ~25 min |

(Measured on the synthetic benchmark of §9; the real-store numbers are scaled estimates.)

**Frozen means frozen.** A stored fold is served as long as the book's frozen config text and the feature columns match. If the vendor restates rows a fold was trained on, the fold is still served unchanged and the day records a warning (row count and label moments compared with the sidecar). A changed feature set is an error: the book cannot score inputs its models were not trained on; start a new book.

**Point in time.** `tests/test_paper.py::test_fit_through_a_later_date_answers_as_of_s_point_in_time` checks that the fit on data through a later date answers `as_of(s)` exactly as the fit on data through $s$, and that data after $s$ do not move it; `test_day_t_orders_do_not_depend_on_later_data` perturbs every price after day $t$ and checks that the journal through $t$ is identical. `--signal-mode single` (one fit through the last day for a long catch-up) relies on the first property.

## 4. Prices, corporate events and the two sizing conventions

### Prices
Positions are valued at the **as-traded** close (`close_raw`, SEP `closeunadj`) times an *implied share count*, and each day's P&L is the total return from the **adjusted** pair (`close`, SEP `closeadj`) of $t-1$ and $t$ read from the **same** snapshot:

$$
V_{i,t} = V_{i,t-1}\,\frac{P^{adj}_{i,t}}{P^{adj}_{i,t-1}}, \qquad q_{i,t} = \frac{V_{i,t}}{P^{raw}_{i,t}}
$$

where:
- $V_{i,t}$ is the dollar value of the position in stock $i$ at the close of $t$;
- $P^{adj}$ is the split- and dividend-adjusted close and $P^{raw}$ the as-traded close;
- $q_{i,t}$ is the implied share count.

Why: Sharadar rescales a company's whole adjusted history every time it pays a dividend. A book that stored adjusted prices would see its valuation jump at every restatement. Using only the *ratio* of two adjusted closes from one snapshot makes the restatement irrelevant (a constant rescaling cancels), and the NAV, weights and P&L are exactly the simulator's. The implied share count is multiplied by $k$ at a $k$:1 split and grows by the yield at a dividend: the distribution is reinvested at the ex-date close, the backtest's total-return convention. A real account would receive the dividend as cash on the pay date: same NAV, a weight lower by the yield until the next rebalance. Splits need no corporate-action table: the event factor $(1 + r^{adj}) / (1 + r^{raw})$ is recorded as a split when it is ≥ 1.25 or ≤ 0.8 and as a distribution otherwise. `test_restated_adjusted_history_does_not_move_the_book` feeds the book the snapshots a vendor would deliver (adjusted history not yet scaled before the ex-date) and checks that NAV and P&L do not move.

### Halts and delistings
Exactly as the simulator (→ see **Backtesting › Halts vs delistings**): a held name with no close is halted (frozen at its last price, return 0, borrow still charged, the whole halt-period move realised on resumption) or delisted (vendor delisting date, else the causal 63-day gap rule). A delisting books its reason's return on the stock (acquisition 0, performance −30 %, voluntary −10 %, unknown −30 %) and is liquidated at spread + fees on the previous day's liquidity. Data-error closes (returns above +300 % or below −95 %) are treated as halts.

### Sizing: `moc` (default) and `replica`
- **`moc`** is what live trading does. Orders are whole shares, rounded toward zero, sized at the job's close $t$: $\mathrm{trunc}((w^{target} - w^{now}) \cdot NAV_t / P^{raw}_t)$; a target of zero sells the exact holding. They fill at the close of $t+1$ whatever the price did in between. An order for a name without a close on the fill day expires unfilled (MOC is a day order; the next rebalance re-targets it); an order for a name that splits before the fill is scaled by the split ratio; an order for a name that delists is cancelled. The optimizer sees the book as it stands at close $t$.
- **`replica`** is the backtest's convention: the target for close $t$ is solved with the book marked at close $t$ from the signal of $t-1$ and filled there in fractional shares. It cannot be traded (the order size depends on the fill's own close), but it reproduces `DailyBarSimulator` day by day and is the convention for the backfill pipeline check.

The two differ by one day of sizing drift, whole-share rounding and one documented timing case: when a halted name resumes, the batch reads its eligibility at the halt-day signal date and treats it as having left the universe (a forced exit on the resumption day), while the `moc` book decided the day before, when the name had no price, and keeps holding it.

## 5. Monitoring and stop rules

The rules, their numbers and their justification are pre-registered in `docs/notes/paper_trading_rules_2026-11-13.md` and frozen into `book.json` at init. In short:

| Rule | WATCH | STOP |
|---|---|---|
| Drawdown vs the bootstrap max-drawdown band at the same horizon | 10 % year-1 false-alarm band | 2.5 % year-1 false-alarm band |
| Sharpe vs the bootstrap Sharpe band | 10 % band, from day 63 | 2.5 % band, from day 126 |
| Cumulative return vs the bootstrap cone | below the 2.5 % quantile | — |
| Realised / ex-ante vol (21 days, relative to the backtest's ratio) | > 1.5× for 10 days | > 2× for 20 consecutive days |
| Turnover, cost per side (21 days, after the build) | > 2× the backtest | — |
| Invested gross | < 0.5× the backtest's mean for 5 days | > 1.1× the gross limit |
| Optimizer | any failed solve or ex-ante vol alert | 3 consecutive failed solves |
| Realised rank IC (lagged, once labels mature) | z < −2 | — |
| Data quality (missing / stale held names, unfilled orders) | > 2 % of NAV, any unfilled | > 10 % of NAV for 3 days |

### The yardstick
`qts paper init` reads the reference run's `returns.parquet` (daily net returns), `summary.json`, `exante_vol.parquet` and `ic.parquet` and freezes two tables under `paper/<book>/expectations/`: scalar expectations (daily mean / sd, turnover, cost per side, gross, the realised / ex-ante vol ratio, mean ex-ante vol, IC mean / sd) and a moving-block bootstrap (21-day blocks, 2,000 paths) of the cumulative log return, maximum drawdown and Sharpe at horizons from 1 day to 5 years. The analytic cone $n\mu \pm 1.96\,\sigma\sqrt{n}$ of daily log returns is also reported.

<callout icon="💡" color="blue_bg">
**Why the bands are calibrated.** A rule checked every day looks at the same path 252 times a year. A boundary at "the 1 % quantile of drawdowns at that horizon" is crossed on some day of the first year by about 8 % of books that behave exactly like the backtest. The drawdown and Sharpe boundaries are therefore *simultaneous bands*: at init, on the reference run's own bootstrap (10,000 paths), the quantile level is chosen so that a backtest-like book crosses the STOP band at least once in its first year with probability 2.5 % (WATCH: 10 %). `qts paper init` prints the achieved rates.
</callout>

### STOP and acknowledgement
A STOP freezes trading: that day's orders are cancelled, the book keeps its positions and is marked to market, and no orders are generated until `qts paper ack --note "..."`. The acknowledgement covers the STOP rules of the acknowledged day; each is reported as WATCH ("STOP acknowledged") until it clears; a rule that clears and fires again, or a different rule, stops the book again. Nothing is ever liquidated automatically.

### Reading `qts paper status`
```text
book baseline  last day 2026-11-20  (6 days)
  NAV 5,003,120  gross 1.94  net +0.012  402L/398S  orders for 2026-11-23: 0  mode live
  cum 0.06%  max DD -0.31%  Sharpe None  vol ratio None
STATUS OK
  [OK   ] drawdown           max drawdown -0.31% after 5 days; WATCH / STOP boundaries -2.70% / -3.80% ...
  [OK   ] sharpe             Sharpe not judged before 63 days
  [OK   ] vol_ratio          21-day realised / ex-ante vol nan (backtest ratio 1.38); ...
  ...
```
The first line says which day was processed last and whether its orders were `live` (generated before the cutoff). `trading halted until qts paper ack` appears after an unacknowledged STOP. `--json` prints the full payload (§8).

## 6. The files of a book

```text
paper/<book>/
    book.json              header frozen at init (config + hash, start, NAV, data spec, reference run,
                           stop rules, expectation-table hash, code version)
    config.yaml            frozen copy of the strategy config: the only config the book reads
    expectations/          summary.json + bootstrap.parquet (the yardstick)
    journal.jsonl          append-only, one JSON record per trading day (the source of truth)
    acks.jsonl             acknowledgements
    days/<YYYY-MM-DD>/     positions.parquet, fills.parquet, orders.parquet (rebuilt by replay)
    models/                frozen walk-forward fold models
    state/last_risk.pkl    the last solve's risk model (ex-ante vol on hold days)
    logs/run.log           the scheduled job's log
    .lock                  advisory lock held by every writer
```

A journal record holds: `date`, `data_asof`, `mode` (live / backfill), `nav`, `nav_pre` (after the mark, before the day's fills: the simulator's NAV), `cash`, `gross`, `net`, long / short dollars and counts, the P&L split `pnl` (positions, delisting, spread, impact, fees, trading cost, borrow, liquidation cost, total), `ret` (NAV after costs) and `ret_sim` (the simulator's net return), `turnover`, `fills`, `orders` (for `orders_for`), `events` (splits, distributions, halts, delistings, forced exits, unfilled orders), `solve` (status, risk aversion, vol-target outcome, expected alpha / risk / cost, IC, half-life, the fold used, fit time), `models` (fits performed, folds trained, stage code hashes), `signal.realised_ic`, `exante_vol`, `data_quality`, `warnings`, `status`, `config_hash`, `config_text_sha256`, `code_version`, `seq` / `supersedes`, and the full engine `state` (implied shares, valuation prices, pending orders, impact streaks, forced exits, open halts), so the journal alone restores the book.

**Write discipline.** Views are written atomically (temporary file, fsync, rename) before the journal line; the journal line (one `write` on an `O_APPEND` descriptor, then fsync) is the commit point. A crash between the two leaves views that the next run overwrites. A torn last line is ignored by readers and removed by `qts paper replay`. **Idempotence:** a day in the journal is never processed again; `--redo-last` re-processes the last day after a vendor correction and the new record supersedes the old one (nothing is deleted). **Lock:** `fcntl.flock` on `.lock`; a second job fails at once with exit code 4 and names the holder; the OS releases the lock when a crashed job exits.

## 7. Commands

| Command | What it does |
|---|---|
| `qts paper init --config configs/baseline.yaml --nav 5e6 --start D --name baseline [--reference-run runs/...] [--sizing moc\|replica]` | create the book; `D` is the first job day: its close produces the first orders, which fill at the next session's close |
| `qts paper run [--name N] [--date D] [--until D] [--no-sync] [--dry-run] [--redo-last] [--signal-mode daily\|single]` | the daily job (§2); `--until` = backfill day by day without the freshness check |
| `qts paper status [--json]` | status, every check, the latest day |
| `qts paper history [--json] [--tail N]` | one line per day |
| `qts paper orders [--date D] [--json] [--csv F]` | orders generated on a day (for the next close) and that day's fills |
| `qts paper ack --note "..."` | acknowledge the latest status |
| `qts paper replay` | validate the journal (accounting identity, chain), remove a torn line, rebuild every view |
| `qts paper reconcile [--json]` | paper vs the batch simulator over the book's dates (flat start, same NAV and schedule) |
| `qts paper schedule --print` | a launchd plist and a crontab line (nothing is installed) |

Exit codes: 0 ok, 1 error, 3 stale data (re-run later), 4 locked.

### The backfill check
A backfill runs the frozen serving path over recent history. It is a pipeline test, not evidence: every record is `mode: backfill`. With `--sizing replica` the book must reproduce the batch simulator; `qts paper reconcile` prints the comparison. On the synthetic test market the replica book matches the batch over 40 days with a planted split, dividend, halt, delisting and an ML refit inside the window to a maximum weight difference below 2e-3, NAV within 2e-4 and a daily-return correlation above 0.9999 (`test_replica_book_matches_the_batch_backtest`). The residual is the solver: identical inputs to 1e-16 still move Clarabel's optimum by ~1e-5, which can flip a `min_trade_weight` (5e-4) decision.

```bash
qts paper init --name check --config configs/baseline.yaml --nav 5e6 --start 2026-10-01 --sizing replica
qts paper run --name check --until 2026-11-09 --no-sync
qts paper reconcile --name check
```

## 8. Library entry points (for the API and the UI)

All in `qts/paper/runner.py` (re-exported by `qts.paper`); every function takes the book name (under `paper_root="paper"`) or a book directory and returns JSON-serialisable dicts (non-finite floats as `null`, dates as `YYYY-MM-DD`).

| Function | Returns |
|---|---|
| `init_book(name, config_path, *, nav, start, reference_run=None, paper_root, data=DataSpec(...), sizing="moc", use_ml=True, rules=None)` | `{"root", "header": <book.json>, "expectations": <summary incl. calibration>}` |
| `run_book(name, *, date=None, until=None, no_sync=False, dry_run=False, redo_last=False, signal_mode="daily", progress=None)` | `{"book", "root", "dry_run", "sync"?, "target", "load_seconds", "days": [{"date", "nav", "ret", "n_fills", "n_orders", "orders_for", "status", "mode", "seconds", "warnings"}], "status": {"level", "reasons", "stop_rules", "suppressed"}, "seconds"}` or `{"message": "nothing to do ..."}` |
| `status(name)` | `{"book", "root", "header", "n_days", "last_date", "acks", "latest": {date, data_asof, mode, nav, nav_pre, cash, gross, net, n_long, n_short, ret, orders_for, n_orders, exante_vol, warnings, data_quality, models, code_version, config_hash}, "status": {"level", "reasons", "stop_rules", "suppressed", "acknowledged", "trading_halted", "checks": [{"rule", "level", "value", "threshold", "message"}], "metrics": {cum_return, cum_log, max_drawdown, drawdown, sharpe, ann_vol, cone, bootstrap_cone, vol_ratio, vol_ratio_backtest, turnover, cost_bps_per_side, gross, realised_ic, boundary_levels}}, "series": {"date", "nav", "ret", "cum_log", "drawdown", "gross", "net", "turnover", "exante_vol", "realised_vol_21d", "status", "cone_analytic": {expected, lower, upper}, "cone_bootstrap": {lower, median, upper, max_dd_q01, max_dd_q05}}}` |
| `history(name)` | `[{"date", "mode", "nav", "ret", "gross", "net", "n_fills", "n_orders", "orders_for", "pnl_positions", "trading_cost", "borrow", "turnover", "exante_vol", "status", "rebalance", "n_warnings"}]` |
| `orders(name, date=None)` | `{"date", "orders_for", "mode", "status", "orders": [{ticker, side, shares, px, usd, target_w, current_w, kind, close_out, exec_date, est_cost_usd, est_cost_bps}], "fills": [{ticker, side, shares, px, usd, spread, impact, fees, kind, date}], "n_orders", "buy_usd", "sell_usd", "est_cost_usd", "cancelled_by_stop"}` |
| `ack(name, note)` | the appended acknowledgement `{"date", "ts", "level", "rules", "reasons", "note", "user"}` |
| `replay(name)` | `{"torn_line_removed", "backup"?, "n_records", "n_days", "last_date", "problems": [...], "ok"}` |
| `reconcile(name)` | `{"sizing", "dates", "max_abs_weight_diff", "max_rel_nav_diff", "cum_return_paper", "cum_return_batch", "daily_return_corr", "tracking_error_ann", "max_abs_daily_return_diff", "batch_summary", "per_day": [{date, paper_nav_pre, batch_nav, weight_diff}], "note"}` |
| `schedule_text(name)` | `{"label", "plist", "crontab", "log", "install"}` |

Errors are `qts.paper.PaperError` (subclasses `StaleDataError`, `BookLockedError`) with a message that says what to do. `run_book` takes a lock; the read-side functions do not (they read the committed journal). `run_book` and `reconcile` are long-running (minutes): an API should run them as background jobs.

## 9. Scheduling on macOS

`qts paper schedule --print` prints a launchd agent and a crontab line for 07:05 on weekdays (Monday's job processes Friday; a job after a holiday finds nothing new and exits 0). Nothing is installed.

```bash
qts paper schedule --print --name baseline > /tmp/qts_paper.txt   # inspect it
# save the <plist> part as ~/Library/LaunchAgents/com.qts.paper.baseline.plist, then
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.qts.paper.baseline.plist
launchctl kickstart -k gui/$(id -u)/com.qts.paper.baseline     # run once now, as a test
tail -f paper/baseline/logs/run.log
```

launchd uses the Mac's local time: set the Mac to America/New_York or shift the hour. The Mac must be awake (System Settings › Battery › Options, or `sudo pmset repeat wakeorpoweron MTWRF 07:00:00`); launchd runs a job missed during sleep at the next wake, cron does not. The plist uses the absolute path of the Python that printed it, so run `schedule --print` from the environment the job should use. The log shows the refresh, each processed day (`NAV …, N fills, M orders for D, status S [live, 412s]`), and on a rebalance day a line saying that the fit is running.

### Runtime
Measured with `python -m qts.paper.bench --stocks 800 --days 1150 --start-offset 30` on the 2-vCPU development box (a synthetic ParquetStore, the baseline config with the full LightGBM + MLP + ridge ensemble, three seeds):

| Day | Measured (800 names × 1,150 days) | Real store on the Mac (estimate) |
|---|---|---|
| first run (trains every fold; two here) | 144 s (folds 43.5 s + 71 s, fit 25 s) | ~25 min (the batch alpha fit) + load |
| rebalance day (frozen folds) | 36 s (fit 31.5 s) | 6–9 min: store load 1–2 min, risk fit ~3 min, alpha without training 2–4 min, solve < 30 s |
| hold day | 2.6 s | 1–2 min (mostly loading the store) |
| refit day (once a year) | rebalance day + one fold (71 s here) | + 2–4 min (the latest, largest fold) |

The real-store column scales the measured fit by rows (≈ 2,500 universe ids × 6,700 days against 800 × 1,150, about 18×) and by the Mac's faster cores (2–3×); it is an estimate until the first real run prints its timings (each journal record has `timings`, and `models.fits[].stages` splits a fit into `risk_fit` / `alpha_fit`).

## 10. Pitfalls & checks
- **Look-ahead.** The job for $t$ sees data through $t$ only (`MemorySource` / `StoreSource.load(end)` cut there; the perturbation tests). The vendor's restatement of adjusted history after a later dividend is a constant rescaling of a name's history; features and P&L use ratios, so it carries no information.
- **The book is not the backtest's book.** A book started today starts flat and builds over the first days; the backtest's book at the same date is its fifteenth year. Compare distributions (the bootstrap), not the two NAV paths.
- **A STOP is a question, not a verdict.** Check `qts paper status`, `history` and `orders`, then decide; the note in `ack` is the record of that decision.
- **Code drift.** The book records the code version and config hash every day and warns when they differ from init. A code change that alters behaviour (a new feature, a risk-model fix) needs a new book: the frozen fold models would no longer match the strategy the batch would run.
- **Tests:** `tests/test_paper.py` (batch-vs-paper equivalence with a refit crossing, a split, a dividend, a halt and a delisting; point in time by perturbation; restated adjusted history; accounting identity and P&L split; whole-share orders; live vs backfill labels; idempotent re-run and `--redo-last`; catch-up equals daily runs; lock; replay after a torn line; stop rules on planted scenarios; the calibration; acknowledgement semantics; trading halts until ack; the mocked Sharadar sync and freshness; the CLI).

## References
- [Bailey & López de Prado (2012), The Sharpe ratio efficient frontier](https://doi.org/10.21314/JOR.2012.255) (standard error of the Sharpe ratio)
- [Künsch (1989), The jackknife and the bootstrap for general stationary observations](https://doi.org/10.1214/aos/1176347265) (moving-block bootstrap)
- [NYSE, Closing auction process fact sheet](https://www.nyse.com/publicdocs/nyse/NYSE_Auctions_Closing_Process_Fact_Sheet.pdf) (15:50 ET MOC cutoff)
