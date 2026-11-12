# Backtesting

<callout icon="🧭" color="gray_bg">
A backtest replays history day by day. On each day the system builds forecasts, risk and costs using only the data available then, chooses a portfolio, trades it at the close, earns the next day's returns, and pays its costs. The output is the daily return series the strategy would have produced. This page documents the `DailyBarSimulator` in `qts/backtest/simulator.py` precisely: its timeline (including the default one-day execution delay, which exists because a live closing-auction order must be sent before the close it trades at is known), its accounting identity, how it books spread, impact, fees, borrow, delistings and forced exits, the per-solve log, and every metric it reports. It summarises the November signal-lag investigation, which found that a large "lag effect" in one demo was path noise, and draws the rule that follows: compare variants with paired, multi-seed tests, never one path. It then covers the backtest biases and how this engine guards against each, how to read the `ResearchPipeline` report card line by line, the run registry every run writes, and the known limitations.
</callout>

<callout icon="🆕" color="green_bg">
**What changed (Nov 2026)**
- **Execution delay is the default.** `ExecutionConfig.signal_lag_days` = 1 (design decision D4): alpha, risk, eligibility and the optimizer's cost snapshot are as of $t-1$; the trade fills at the close of $t$ on day $t$'s liquidity. Lag 0 is kept as an optimistic bound (`--compare-lag`).
- **Lag investigation** (`docs/notes/lag_investigation_2026-11-05.md`): the demo's 0.96 → 0.11 net Sharpe drop at lag 1 came from an uncommitted intermediate code tree and path noise, not from the lag. Across five seeds lag 1 is not worse. A real future-universe-membership leak in the risk model's style exposures was found and fixed on the way.
- **Third return column `gross_after_borrow`.** Breakeven cost and the cost-sensitivity table now start from it, so borrow is no longer left out of the 2× breakeven gate.
- **Routed venue** by default, exact multi-day impact increment, typed cost contract (→ see **Transaction cost model**).
- **Solve log** records `risk_aversion`, `vol_target_binding`, `n_solves` and `n_excluded_alpha`; the summary gains `mean_risk_aversion` and `vol_target_caps_share`, printed as a "sizing" line.
- **Fold gates** use purged walk-forward test blocks instead of calendar years; **run registry** (manifest, config, summary, parquet frames, extras) is written for every run.
- Report-card walk-through updated to `runs/demo_post_refactor.txt` (`runs/demo_audit2.txt` did not exist when this page was written).
</callout>

## What a backtest is and isn't

### What it is
A backtest is a **simulation of a decision process**, not a measurement of a strategy's future. It answers: "If this exact code had run every day from 2019 to 2023, seeing only what it could have seen, with these cost assumptions, what would the account have done?" Think of a flight simulator. It is good for finding crashes before they happen and for comparing two designs under identical conditions. It is not proof that the real plane will fly the same way in tomorrow's weather.

### Why we need it here
Every model in this system makes assumptions that only show up when they interact. The alpha's magnitude only matters relative to costs. The risk model's errors only matter through the optimiser's positions. Costs only matter at the turnover the optimiser chooses. The backtest is the only place all five models (alpha, risk, cost, portfolio, execution assumptions) run together. Its net Sharpe is the number the plan's hurdle gates use (→ see **Validation & statistics › The hurdle gates**).

### The backtest-to-live gap
Expect live results to be much worse than the backtest. The best large-sample evidence is Suhonen, Lennkh & Perez: across 215 bank-sponsored alternative-beta indices, the **median Sharpe ratio fell by 73%** from backtest to live, and the most complex strategies lost over 30 percentage points more than the simplest (BUILD_PLAN §1; live data). Their sample is product backtests, which carry heavy selection bias, so the haircut for an honest in-house process should be smaller. But the direction is not in doubt. The causes are the subject of the biases section below: data snooping, optimistic costs, look-ahead and regime change.

<callout icon="💡" color="blue_bg">
Practical calibration from the plan: treat **net Sharpe 0.5–1.0** in an honest backtest as success, and treat anything above ~2 net as a bug until proven otherwise. The pipeline automatically flags net Sharpe > 2 as "suspect" (BUILD_PLAN §1, §2.4).
</callout>

### References
- [Suhonen, Lennkh & Perez via Alpha Architect, on backtest-to-live performance of alternative-beta indices](https://alphaarchitect.com/?p=36892)

## The simulator timeline

### What it is
`DailyBarSimulator.run` steps through a list of trading dates. Let $L$ = `signal_lag_days` (default **1**, from `cfg.execution.signal_lag_days`; the `run(signal_lag_days=)` argument overrides it) and let $s$ be the trading day $L$ positions before $t$ in `md.dates` (the **signal date**; $s = t$ when $L = 0$). For each execution date $t$ the simulator does exactly the following, in this order:

```text
   close of s = t-L          close of t (execution)                     close of t+1
 ──────┬──────────────────────────┬──────────────────────────────────────────┬─────────
       │ 1. snapshot_s            │                                          │
       │    (ADV, sigma, spread,  │                                          │
       │     price, borrow <= s)  │                                          │
       │ 2. eligibility as of s   │ 3. alive = has a price at t              │
       │ 4. alpha_fn(s)           │    snapshot_t; forced-exit caps          │
       │ 5. risk_fn(s, alpha)     │ 6. optimizer.solve(alpha_s, risk_s,      │
       │                          │      w_pre marked at t, nav_t,           │
       │                          │      snapshot_s + auction share of t)    │
       │                          │ 7. trade x = w_post - w_pre at CLOSE t   │
       │                          │    cost = estimate(x*nav_t, snapshot_t)  │
       │                          │         + one day of borrow on snapshot_t│
       │                          │ 8. hold w_post ── earns r_{t+1} ────────▶│ 9. nav_{t+1}
       │                          │                                          │10. drift
```

Step by step:
1. **Signal-date liquidity.** The optimizer prices costs on the snapshot as of $s$ (ADV, σ, spread, price, borrow: data ≤ $s$).
2. **Eligibility** (universe membership) is read as of $s$: the investable set the signal was formed on.
3. **Liveness** is read as of $t$: a name without a price at $t$ cannot trade and goes on the no-trade list. The cost snapshot for booking is built as of $t$. Forced-exit caps for held names that left the universe are updated.
4. `alpha_fn(s)` gives expected returns from data ≤ $s$. Ineligible and dead names get expected return 0.
5. `risk_fn(s, alpha)` gives the risk model (exposures B, factor covariance F, specific variance D, alignment) as of $s$.
6. The optimizer gets the signal-date alpha, risk and snapshot, but the **current** book $w^{\text{pre}}$ (drifted, marked at the close of $t$) and the **current** NAV. Positions are known in real time; only the signal inputs are stale. The snapshot carries the **execution date's** auction share, because the auction calendar (quad witching, Russell reconstitution) is published in advance (second-pass audit finding 14).
7. The trade $x = w^{\text{post}} - w^{\text{pre}}$ executes at the close of $t$. Costs are booked on the execution date's snapshot (actual liquidity) at the requested venue (default `routed`), plus one day of borrow on post-trade shorts.
8. Positions earn the close-to-close return $t \to t+1$.
9. NAV is updated by the identity below.
10. Weights drift to tomorrow's pre-trade weights.

On the first $L$ days of `md` there is no signal date; the book is held and `diagnostics["n_no_signal_days"]` counts them. `ResearchPipeline.backtest_dates()` avoids this by dropping the first $L$ out-of-sample dates, so the signal date of the first backtest day is itself out of sample (→ Pipeline stages below).

Steps 1–5 use only data with timestamp ≤ $s$, enforced by every model's `as_of(s)` and point-in-time slicing (→ see **Architecture & data flow › Point-in-time discipline**). The position decided at $t$ never earns the return of day $t$. It first earns $t\to t+1$. The audit reproduced a hand-computed NAV path exactly (A +10%/−10%/0/+21% with $w_A=0.5$: NAV 1.00 → 1.05 → 0.945 → 0.98795 → 1.0927M), and the second-pass audit reproduced a 12-name lag-1 run with a split, a delisting, a forced exit and dated fees by hand with **zero** error in spread, impact, borrow and NAV.

**Indexing.** Panels recorded per decision (`weights`, `trades`, `costs`, `exante_var`, `turnover`, `transfer_coefficient`) are indexed by the execution date $t$. `returns` (gross, net, gross_after_borrow) and `nav` (after the first entry) are indexed by the realisation date $t+1$. Use `BacktestResult.turnover_on_return_dates()` to line up turnover with returns positionally. Never date-join the two: that was audit tcost #4, which shifted costs by one day and dropped the initial build. `solve_log` entries carry both `date` ($t$) and `signal_date` ($s$).

### Why the signal is lagged by default (design decision D4)
The simulator uses the **closing price** $P_t$ as the *execution price* of the trade at $t$. That is realistic: a market-on-close (MOC) or limit-on-close (LOC) order fills at the official auction price, and that price *is* $P_t$.

With $L$ = 0 the same price is also an *input* to the signal, risk model and cost snapshot at $t$: the 21-day reversal feature `strev_21` uses $P_t$; `max_ret_21`, rolling beta and residual vol use $r_t$; the ADV window includes day $t$'s volume. That is **not achievable live**. On NYSE, MOC/LOC orders must be entered by **15:50 ET**; after that only offsetting orders against a published imbalance are accepted (BUILD_PLAN §6). The closing price is set around 16:00. When a live system must commit its orders, the close is not yet known (→ see **Execution › Why the signal cannot use the official close**).

**Bias direction at lag 0: optimistic.** The signal knows the last ~10–15 minutes of trading plus the auction print. That is harmless for slow features (12-1 momentum skips the most recent month; value, quality and investment are fundamental) and optimistic for anything that uses $r_t$ or $P_t$ directly, most of all the **short-term reversal** feature. Part of the auction move is temporary (an imbalance-driven print that partly reverses the next day), so a reversal signal that "sees" the print buys the dip *at the dipped price* and earns the bounce: a bid–ask-bounce-style bias that no synthetic test can measure.

**Lag 1 removes it** by using only information to the previous close. It is somewhat pessimistic compared with the real ~15:45 snapshot: it gives up a day of signal where reality gives up about fifteen minutes, roughly $1 - 2^{-1/\text{hl}}$ of the alpha, about 1% for a 60-day half-life. The lag investigation (below) found that for the composite this costs nothing measurable. So the conservative setting is close to free for slow signals, and it is the default. Lag 0 is kept as an **optimistic bound**: `python -m qts.cli demo --compare-lag` runs both and prints the pair.

<callout icon="⚠️" color="yellow_bg">
For any signal with a half-life of a few days (text/event signals, short-term reversal as an alpha rather than a risk factor), compare lag 0 and lag 1 before trusting it. The plan cites evidence that a one-period delay flips event-signal returns negative (Li, Qiao & Zheng, BUILD_PLAN §6). `strev_21` is the one feature whose IC changes sign within one or two days of lag; once the ML sleeve (which uses it) is on, rerun the lag comparison with `use_ml=True`.
</callout>

### The lag investigation (November 2026)
**The question.** `runs/demo_post_refactor.txt` (300 synthetic names × 1,512 days, seed 0, composite alpha) showed gross Sharpe 1.33 → 0.49 and **net Sharpe 0.96 → 0.11** when execution was delayed by one day. The planted alpha is an AR(1) score with one-day autocorrelation ≈ 0.989, so one day of staleness should cost about 1% of the edge. Was it (A) a broken lag implementation, (B) same-day leakage at lag 0, (C) a real short-horizon effect, or (D) something else? `docs/notes/lag_investigation_2026-11-05.md` has the full report; every backtest in it uses the real simulator, optimizer and cost model, called exactly as the pipeline calls them.

**The answer: (D). The gap did not come from the lag.**
- **It does not reproduce.** The demo ran from an uncommitted working tree (`git_dirty: true` at `63ce32b`) that was modified again before it became commit `ab8cea0`; no copy of that intermediate state survives. On committed code the same pair gives gross 0.68 → 0.60 and net 0.23 → 0.14 (lag 0 → lag 1), a difference of −0.08 ± 0.15 (paired block bootstrap, 1 SE).
- **The demo's lag-0 book had no one-day edge.** Held one day stale ($w_{t-1}\cdot r_{t+1}$), it still earns gross Sharpe 1.30 instead of 1.33. The demo's lag-1 run traded into a *different* book whose daily returns correlate only 0.75 with the lag-0 run's. With ~50-day holding periods and cost-aware trading the optimizer is path dependent: books that start apart stay apart.
- **Single paths are noisy at this scale.** Two code versions running the *same* lag differed by 0.65 Sharpe. After the exposure fix below, which changes no timing, the full-window "lag effect" flipped sign to **+0.43** (SE 0.24).
- **Five seeds** (last 600 out-of-sample days each, current code): lag 1 minus lag 0 gross Sharpe = +0.26, −0.04, +0.05, +0.40, +0.14; mean **+0.16** (sd 0.17). Lag 1 is not worse. The positive mean is not significant (t ≈ 2.1 on 4 degrees of freedom; the 5% critical value is 2.78) and is not evidence that lag "helps".
- **IC by lag.** The composite keeps 93% of its 1-day rank IC and 98% of its 5- and 20-day IC at lag 1. No theme has a short-horizon component that vanishes at $k$ = 1.
- **No implementation defect and no same-day leakage.** The simulator's lag path matches the design, now covered by real-stack tests (`tests/test_signal_lag_real.py`). Refitting the whole pipeline on data truncated at $t$ reproduced every decision input at lag 0.
- **One genuine point-in-time bug was found and fixed.** The risk model's style exposures (beta, residual vol, and the standardisation of every style) used names that join the universe only *after* $t$: a future-membership leak. It biases lag 0 and lag 1 equally, so it does not explain the gap (→ see **Risk model**). A robustness bug was fixed too: `signal_lag_days` ≥ 2 crashed on the first backtest day (risk warm-up); `backtest_dates()` now drops the first $L$ out-of-sample dates.

**The oracle check.** Using the true latent score as alpha (fixed IC, no estimation), Sharpe decays smoothly with lag, with no cliff between 0 and 1:

| Lag $k$ | 0 | 1 | 2 | 5 |
|---|---|---|---|---|
| Gross Sharpe, zero costs (seed 0) | 1.147 | 1.070 | 0.986 | 0.820 |
| Δ vs lag 0 (paired SE) | | −0.077 (0.081) | −0.161 (0.111) | −0.326 (0.186) |
| Realised IC($k$)/IC(0), mean of 5 seeds | 1 | 1.00 | | 0.93 |
| AR(1) theory $\phi^k$ = 0.989$^k$ | 1 | 0.989 | 0.978 | 0.948 |

Seed 0's draw sits in the steep tail (its IC ratio is 0.96 at $k$ = 1 and 0.78 at $k$ = 5), which explains why its decay looks steeper than theory; averaged over five seeds the oracle decays exactly as AR(1) predicts.

<callout icon="💡" color="blue_bg">
**The general lesson: compare variants with paired, multi-seed tests.** On this market, small code or setting changes move a 1,000-day Sharpe by 0.5–0.8. One path cannot rank two variants whose true Sharpe differs by less than that. For any A/B comparison (lag 0 vs 1, a cost parameter, a constraint): (1) run both on the **same** market and dates and compute the **paired** difference of daily returns, with a block bootstrap standard error (the investigation used 21-day blocks; script `paired_diff.py` in the note); (2) report the return correlation between the two books, since a low correlation means path divergence, not a treatment effect; (3) repeat over several market seeds and look at the mean and spread of the differences. A difference smaller than about two paired SEs, or one that flips sign across seeds, is noise.
</callout>

### How to run the comparison
- CLI: `python -m qts.cli demo --compare-lag` runs lag 0 and lag 1 on the same fitted models and prints an "Execution-delay check" table (Sharpe gross/net, net CAGR, turnover). `--signal-lag N` sets the lag for a single run.
- Code: set `cfg.execution.signal_lag_days`, or pass `DailyBarSimulator.run(..., signal_lag_days=L)`. Changing the config lag makes `ResearchPipeline.run` refit, because labels, the walk-forward purge and the trailing-IC windows are aligned to the lag at fit time.
- The trial ledger keys a run on the effective lag (`run_spec`), so lag 0 and lag 1 are two trials of the same search, as they should be for the DSR.

### Pitfalls & checks
- At $L \ge 1$ the optimizer prices costs on the $s$ snapshot while the simulator books them on $t$: by design, and the reason expected and realised cost differ slightly.
- The lag is in trading days of `md.dates`, not calendar days.
- Tests: `tests/test_signal_lag_real.py::test_real_stack_lag1_weights_equal_lag0_weights_one_day_earlier`, `test_real_stack_lag1_inputs_are_dated_signal_date_and_book_is_current`; `tests/test_backtest.py::test_signal_lag_shifts_decisions_by_one_day`; `tests/test_pipeline.py::test_backtest_dates_leave_room_for_the_signal_lag`, `test_signal_lag_is_forwarded_when_supported`; `tests/test_audit2_costs_backtest.py::test_optimizer_snapshot_uses_execution_date_auction_share`.

### References
- [NYSE, Closing Auction Process fact sheet](https://www.nyse.com/publicdocs/nyse/NYSE_Auctions_Closing_Process_Fact_Sheet.pdf)
- [Li, Qiao & Zheng (2025), LLM event extraction and execution delay](https://arxiv.org/pdf/2512.19484)

## Accounting

### The NAV identity
NAV (net asset value) evolves by one exact identity:

$$
\text{nav}_{t+1} = \text{nav}_t\Big(1 + \sum_i w^{\text{post}}_{i,t}\, r_{i,t+1}\Big) - C_t
$$

where:
- $w^{\text{post}}_{i,t}$ = post-trade weight of stock $i$ decided at $t$ (fraction of $\text{nav}_t$);
- $r_{i,t+1} = P_{i,t+1}/P_{i,t} - 1$ = simple close-to-close return from the **adjusted** close (dividends and splits folded in);
- $C_t$ = total USD cost booked at $t$: spread + impact + fees + one day of borrow (+ delisting liquidation costs).

### Three return series
The simulator records three columns in `BacktestResult.returns`, all indexed by $t+1$:

| Column | Formula | What it is for |
|---|---|---|
| `gross` | $g_t = \sum_i w^{\text{post}}_{i,t} r_{i,t+1}$ | signal quality through the optimiser, before every cost |
| `gross_after_borrow` | $g_t - B_t/\text{nav}_t$, with $B_t$ the day's borrow in USD | after the holding cost, before trading costs: the input to breakeven and cost sensitivity |
| `net` | $n_t = \text{nav}_{t+1}/\text{nav}_t - 1 = g_t - C_t/\text{nav}_t$ | what you keep |

Why the middle column? Breakeven cost and the cost-sensitivity table ask "what per-side trading cost would kill the strategy?" Borrow is not a per-side cost: it does not scale with turnover. Starting from `gross` left borrow out of those gates entirely, which flatters short-heavy books (second-pass audit finding 1). `BacktestResult.gross_after_borrow_returns` recomputes the column from `costs` and `nav` for results saved before it existed.

The cash leg (the residual $1 - \sum w$, plus short proceeds) earns **zero**. There is no interest on cash and no short rebate (see Limitations). The identity holds to machine precision (`test_nav_identity`; `tests/test_simulator_handcomputed.py`).

### Weight drift
Between closes, prices move and weights drift. The next day's pre-trade weights are the post-trade weights grown by their returns and re-based to the new NAV:

$$
w^{\text{pre}}_{i,t+1} = w^{\text{post}}_{i,t}\,(1 + r_{i,t+1})\,\frac{\text{nav}_t}{\text{nav}_{t+1}}
$$

Cash absorbs the costs, which is why $\text{nav}_{t+1}$ (after costs) is in the denominator. Tomorrow's trade is measured from these drifted weights. A winner that grew from 2% to 2.2% of NAV must be trimmed back if the per-name cap is 2%: that is real turnover the backtest correctly charges.

### Cost booking
At $t$ the trade $x = w^{\text{post}} - w^{\text{pre}}$ is converted to dollars at $\text{nav}_t$ and priced by `cost_model.estimate(x · nav_t, snapshot_t, venue, streak, prior_streak_usd)`, which returns a typed `CostEstimate` (→ see **Transaction cost model › The cost model end to end**):

| Component | What it is | Driver (default `routed` venue) |
|---|---|---|
| spread | half the bid–ask spread on the traded dollars | 0 on the auction slice; per-name half-spread on the continuous remainder |
| impact | power-law market impact, $C(Q) = b\,\sigma\,Q^{e}/V^{e-1}$ with $e$ = 1.5 | auction slice up to 2% of auction volume ($V$ = auction share × ADV, $b_{\text{auc}}$ = 0.3), remainder priced as the increment $C(q) - C(q_a)$ at $V$ = ADV, $b_c$ = 1.0; plus the multi-day streak increment |
| fees | commission, clearing, SEC §31 (sells), FINRA TAF (sells), CAT, auction exchange fee | IBKR Pro tiered; auction and continuous slices are two orders; dated SEC/TAF schedules use the snapshot date |
| borrow | one day of borrow on post-trade shorts, $\max(-w,0)\cdot\text{nav}_t\cdot f/252$ | per-name fee: `md.borrow_fee` → short-interest proxy → GC default, small-cap multiplier on the last two |

**Multi-day impact persistence.** Square-root impact is concave per share, so building a position over several days costs more than independent one-day trades if the earlier impact has not decayed. The simulator tracks, per name, the **cumulative dollars** traded in the current same-direction streak, $Q_{k-1}$ (reset on a sign flip or a day without a trade), and the cost model charges day $k$ the exact increment

$$
C(Q_{k-1} + q_k) - C(Q_{k-1})
$$

on that day's impact. This treats impact within a streak as fully persistent, which is conservative (Frazzini et al. estimate ~70% permanent). It is on by default (`CostConfig.impact_persistence = True`). The earlier equal-size multiplier $m_k = k^{3/2} - (k-1)^{3/2}$ ($m_2$ = 1.83, $m_3$ = 2.37) is the special case where every day of the streak trades the same notional; it is only used when a cost model cannot accept `prior_streak_usd`, and it is not a bound (it under-charged a halving build by about 20%). The **optimiser** still sees only the one-day marginal cost, because it cannot know the future streak.

**Venue.** `run(..., venue="routed")` is the default, as in the pipeline: each trade is priced the way the router would execute it (→ see **Execution › The routing table**). `"auction"` and `"continuous"` remain available for research. The optimiser receives the same venue for its cost terms.

**No silent zero cost.** `CostEstimate` raises on missing columns, non-finite or negative costs; a legacy cost model's output goes through `cost_frame`, which raises on unrecognised columns (audit tcost #8).

### Halts and delistings
A held name whose next close is missing is either **halted** (prices will come back) or **delisted**. The simulator decides which with the same causal rule as the labels (→ see **Data & universe › Delisting returns**), never by looking at later prices:

- if `md.delisting_dates` covers the name, it is delisted on $t+1$ if and only if the vendor's last trading date is before $t+1$ (a `NaT` or later date means halted);
- otherwise it is delisted on the first missing day that is $G$ = `AlphaConfig.delisting_gap_days` (63) trading days after its last close, and halted before that.

A **halted** position is frozen at its last price: it earns 0 on each missing day, cannot be traded (it is on the optimiser's no-trade list), keeps its weight, and a frozen short still pays borrow. When a close reappears the return is measured from the last close before the halt, so the whole halt-period move lands on the resumption day. Frozen name-days are counted in `diagnostics["n_halted_name_days"]`; each episode is recorded in `diagnostics["halts"]` as `{ticker, start, end, days, weight, outcome}` with `outcome` = `resumed`, `delisted` or `open` (still halted at the end of the backtest).

A **delisted** name would otherwise earn 0 and vanish for free. The simulator instead books a final return that depends on *why* the name left (next subsection), force-liquidates what is left, and records the event.

### Delistings by reason
**What it is.** The final return a delisted position earns, relative to its last close. It is a return **on the stock**, chosen by the vendor's delisting reason:

| Reason | What happened | Return on the stock | Config field (`CostConfig`) |
|---|---|---|---|
| `acquisition` | acquired or merged into another company: holders receive cash or stock worth about the last traded price | 0 | `delisting_return_acquisition = 0.0` |
| `performance` | bankruptcy, liquidation, or removal by the exchange for failing listing rules | −30% | `delisting_return_performance = -0.30` |
| `voluntary` | the issuer left the exchange (going dark, moving to OTC): trading continues, illiquid | −10% | `delisting_return_voluntary = -0.10` |
| `unknown` | no reason on file, or no `MarketData.delisting_reasons` at all | −30% | `delisting_return_unknown = -0.30` |

**Why we need it here.** Until 2026-11-12 one number, `delisting_haircut = -0.30`, was charged **against the position's sign**: a long lost 30% and a short lost 30% too, whatever the reason. That was meant to be conservative. On real data (Sharadar, top-1000 US universe, run `20261112T031754882789Z_425a06bb`) it removed more than the strategy's whole gross alpha:

| Measurement (`scripts/attribute_run.py`) | Value |
|---|---|
| $\sum_i w_i r_i$ of the book, annualised (delisted names earn 0 here) | +5.02% |
| recorded gross return, annualised | +1.49% |
| difference = delisting haircuts | **−3.53% a year** |

At gross exposure 2.0, −3.53% a year at 30% per event means about 5.9% of gross exposure a year sat in names that delisted while held (2.0 × 5.9% × 30% = 3.5%). In a large-cap universe most of those are acquisitions, which complete at about the last traded price: there is nothing to lose. And Shumway's −30% is a return on the stock for performance delistings: a short *gains* it.

**How it works.** Let $r_i$ be the return of the name's reason and $w_i$ its post-trade weight. The booked return is

$$
r^{\text{booked}}_i = \begin{cases} h_s\, r_i & w_i < 0 \text{ and } r_i < 0 \\ r_i & \text{otherwise} \end{cases}
\qquad \text{P\&L}_i = w_i\, r^{\text{booked}}_i
$$

where:
- $r_i$ = `delisting_return_<reason>` of the name's reason (table above), a fraction of the last close;
- $h_s$ = `delisting_short_gain_haircut` in $[0, 1]$, default 1: the share of a short's gain that is booked. 1 is the symmetric textbook return. 0 is the conservative choice "shorts get nothing": in practice a short in a failing company is often bought in, recalled or impossible to borrow before the final drop. A short's *loss* (a positive $r_i$) and every long are never scaled;
- P&L is a fraction of NAV at $t$.

Then, as before:
1. the remaining value is force-liquidated, $x_{\text{liq}} = -w(1+r^{\text{booked}})$, at continuous-venue **spread + fees** (no impact, no persistence);
2. the liquidation is booked as a trade on decision row $t$, so it appears in `trades`, turnover and costs;
3. the event is recorded in `diagnostics["delistings"]` as `{date, ticker, weight, reason, side, stock_return, haircut_return, pnl, pnl_usd, liquidation_usd, cost_usd}` (`haircut_return` is the booked return; `pnl` = weight × booked return); the weight is 0 afterwards.

`diagnostics["delisting_summary"]` (`qts/core/delisting.py::summarize_delistings`) holds the counts and P&L by reason and side; the summary field `delisting_pnl_ann` is their annualised sum, and the report card prints one line. Its format, from a 100-name synthetic market with 12 planted delistings (no reasons table, so all `unknown`; one long and three shorts were held when they delisted):

```
delistings        unknown 4 (P&L +1.37%/yr; 1L/3S)  total +1.37%/yr
```

Each reason with at least one event is listed as `<reason> N (P&L x%/yr; aL/bS)`: N events, their annualised P&L, and how many were longs and shorts.

**The labels use the same map.** `qts/alpha/targets.py::forward_returns` multiplies the label of a name that delists inside the window by $1 + r_i$ with the same per-reason returns (`CostConfig.delisting_returns()`), so the model is trained on the outcome the backtest books: an acquisition's label is the return to its last price, a bankruptcy's carries −30%. The labels already applied the haircut on the stock (never against a position), but used −30% for every reason; that part changed. The short-gain haircut and the escape hatch below affect the simulator's booking only.

**Where the reason comes from.** `MarketData.delisting_reasons` (ticker → reason), loaded from the store's `delisting_reasons` table, which is derived from Sharadar's `ACTIONS` rows around each delisting date (→ see **Data & universe › Delisting reasons**). A market without the table (the synthetic market, any other source) books every delisting as `unknown`.

<callout icon="⚠️" color="yellow_bg">
**Timing.** The reason is read on the day the delisting is decided (the day after the vendor's last trading date). The vendor action it comes from may be dated up to 5 business days later, so the *size* of the booked return can rest on a record published a few days after the booking. No signal, weight or trade depends on it, and a merger is announced months before it closes. `MarketData.slice_until` blanks the reason of any delisting after the cut.
</callout>

**The escape hatch.** `CostConfig.delisting_against_position = True` applies the reason's return against the position's sign, $r^{\text{booked}} = r \cdot \text{sign}(w)$: the old rule. With all four returns at −0.30 it reproduces the pre-2026-11-12 accounting exactly (`tests/test_simulator_handcomputed.py::test_delisting_escape_hatch_reproduces_the_old_both_sides_haircut`). `delisting_haircut` is deprecated: an old config that still sets it loads with a `DeprecationWarning`, and its value is copied into `delisting_return_performance` and `delisting_return_unknown`. It does **not** switch the escape hatch on.

**Worked example.** A book holds +2% in A (acquired), +1% in B (bankrupt) and −2% in C (bankrupt), each at its last close. New rule: A earns 0; B earns 0.01 × (−0.30) = −0.30% of NAV; C earns (−0.02) × (−0.30) = +0.60%. Total +0.30%. Old rule: −0.60% − 0.30% − 0.60% = −1.50%. With `delisting_short_gain_haircut = 0`: 0 − 0.30% + 0 = −0.30%.

**Pitfalls & checks.**
- `unknown` defaults to −30%, the same as `performance`. If `qts data sharadar derive` reports many unknowns, read the action codes it prints for them before trusting the P&L line: a large `unknown` bucket that is really acquisitions overstates the long-side loss and the short-side gain.
- The symmetric default lets shorts in bankruptcies earn 30% of the position. Check the `performance … S` count and P&L on the report card; rerun with `--set cost.delisting_short_gain_haircut=0` to see how much of the result rests on it. Both knobs act after the fits, so the stage cache serves the rerun.
- Tests: `tests/test_simulator_handcomputed.py::test_delisting_return_by_reason_and_side` (long and short × each reason, the gain haircut, the escape hatch, by hand), `tests/test_label_halts.py::test_labels_apply_the_delisting_return_of_the_reason`.

Without a vendor delisting table a real delisting is held frozen for 63 trading days before the return is booked, which delays (but does not remove) the loss; supply `delisting_dates` for real data.

Names with no price at $t$ go on the optimiser's no-trade list, and their weights are carried unchanged.

### Data-error prices
A close that would produce a return above `RiskConfig.return_abs_cap_up` (+300 %) or below `return_abs_cap_down` (−95 %) is a bad print, not tradable P&L: nobody earned +50,000 % on a restated adjusted close, and nobody lost 99.8 % on its reversal the next day. With `RiskConfig.clean_returns` on, the simulator blanks such closes before the run (`qts.core.returns.flag_bad_prices`: a spike-and-reverse pair is one bad close; a level shift that persists is blanked for up to three days and then accepted). The name then takes the **halt** path above on that day — frozen, return 0, no trade — and the next day's return is measured from the last good close, so the P&L across the gap is the true move. Blanked closes inside the run are counted in `diagnostics["n_data_error_returns"]` and listed (first 100, with the raw close) in `diagnostics["data_error_returns"]`; the report card prints the count. Nothing else is cleaned: a genuine −80 % day is P&L, and the risk model's cross-sectional winsorisation (→ see **Risk model › Return cleaning**) never touches the simulator. The motivating incident — one return of this kind emptying a real-data book for four years — is in `docs/notes/risk_outlier_fix_2026-11-10.md`.

### Risk-model sanity
On every rebalance the simulator computes the annualised ex-ante vol of the *incoming* book under the day's risk model (or, when the book is empty, of a long-only $\sqrt{\text{cap}}$-weighted reference book over the model's names, so a broken covariance is still visible after a liquidation). Above `ValidationConfig.max_exante_vol` (200 %) it logs the date, the vol and the five largest contributing names (specific, $w_i^2 d_i$) and factors ($(B^\top w)_k (F B^\top w)_k$), and appends the entry to `diagnostics["exante_vol_alerts"]` (`n_exante_vol_alerts` counts them). `BacktestResult.exante_vol_alert_fraction()` divides by `n_rebalance_solves`; the `risk_model_sanity` hurdle gate fails the run above `max_exante_vol_alert_frac` (0.5 % of rebalances). A target-vol optimiser facing a 10,000 % forecast liquidates everything and the book then stays empty for many covariance half-lives — a flat NAV that would otherwise read as "no signal".

### Forced exits (eligibility)
If a held name is not in the investable universe as of the signal date $s$ (`md.eligible`, or the `eligible_fn` hook), its expected return is set to 0 and the optimiser receives a `force_to_zero` cap on $\lvert w\rvert$:

$$
\text{cap}_k = \lvert w_0\rvert \Big(1 - \frac{k}{n}\Big)^{+}
$$

where $w_0$ is the weight when the exit started, $k$ the number of days since, and $n$ = `PortfolioConfig.exit_ineligible_days` = 5 (→ see **Portfolio construction › Constraints**). The second-pass audit's hand-built run reproduced the sequence exactly: 0.0373, 0.0280, 0.0186, 0.0093, 0. A name that re-enters the universe, or is already flat, leaves the forced set. Ineligible names that are not held go on the no-trade list. Events go to `diagnostics["forced_exits"]`; `n_ineligible_name_days` counts the exposure.

### Rebalance schedule

**What.** The optimiser is not solved every day. `PortfolioConfig.rebalance_every_days` = $k$ = **5** (default since 2026-11-12): it runs on the backtest dates with index $i$ where $i \bmod k = 0$ — the first date, then every fifth — and on the days in between the book **drifts**: weights move with prices (→ **Weight drift**), nothing is traded, borrow is still charged daily. `BacktestSpec.rebalance_every`, `DailyBarSimulator.run(rebalance_every=...)`, `ResearchPipeline(rebalance_every=...)` and `qts run --rebalance-every` override the config for one run; `None` (their default) means "use the config". $k = 1$ is the old daily schedule.

**Two things still happen every day:**
- **Forced exits.** A held name that leaves the eligible universe is unwound on its linear schedule (previous subsection) without waiting for the next solve: on a hold day its $\lvert w\rvert$ is cut to the day's cap (`DailyBarSimulator._forced_exit_step`; never increased, never flipped) and the trade is costed like any other. Nothing else trades on that day.
- **Delistings and halts** are handled in the mark-to-market step, which runs daily regardless.

**Why.** The first real ML run traded 13.6% of NAV a day. Only 44% of a day's trading survived netting over the next 21 days — the rest was undone — and the paper portfolio built from the same score lost little when it was sampled every 21 days instead of daily (gross Sharpe 0.92 against 1.04, at 2.2% of gross a day). A 20–60-day forecast does not change enough in a day to pay for a round of trades; solving weekly removes four fifths of the opportunities to re-trim. The two other pace controls are the score's smoothing half-life (→ see **Alpha model › Smoothing the score**) and the minimum trade size (→ see **Portfolio construction › Minimum trade size**).

<callout icon="⚠️" color="yellow_bg">
What a hold day does *not* do. No risk model is evaluated and no band is re-checked: the recorded ex-ante variance of a hold day uses the last solve's risk model with the drifted weights, and after five days of drift the book can sit slightly outside a band (gross 2.01 against a limit of 2.0 is normal). `diagnostics["solve_log"]`, `constraints.parquet` and `n_rebalance_solves` have one row per **solve**, not per day; the `failed_solves` and `risk_model_sanity` gates divide by solves.
</callout>

Tests: `tests/test_construction.py::test_rebalance_schedule_is_exact` (solve dates are exactly `dates[::k]`, zero turnover and exact drift in between, the kwarg overrides the config) and `test_forced_exits_run_on_hold_days`.

### In the code
`DailyBarSimulator(cfg, cost_model, adv_window=63, vol_window=63).run_spec(md, spec, alpha_fn, risk_fn, optimizer)` is the whole engine; `spec` is a frozen `qts/backtest/spec.py::BacktestSpec` (validated on construction). `run(md, dates, alpha_fn, risk_fn, optimizer, nav0=..., ...)` still accepts the dates plus keyword arguments and packs them into a spec. `cost_model` is a `qts.core.costs.CostModel` (production: `TransactionCostModel(cfg.cost, cfg.execution)`); a legacy model without `snapshot()` is wrapped once by `qts/backtest/cost_adapter.py::_LegacyCostModel`, and the window arguments only apply to that legacy path. The pipeline builds the spec for you:

```python
spec = BacktestSpec.create(
    dates,
    nav0=self.nav0,
    rebalance_every=self.rebalance_every,
    venue=self.venue,
    neutral_styles=self.neutral_styles(),
    progress=progress,
    eligible_fn=lambda d: elig.columns[elig.loc[d].to_numpy()],
    signal_lag_days=lag,
)
bt = sim.run_spec(self.md_u, spec, alpha_fn, risk_fn, self.optimizer)
```

The day loop in `run_spec` is a fixed pipeline of small steps, one method each: `_day` (dates, signal date, execution-date snapshot) → `_universe_step` (eligibility, forced exits, no-trade list) → `_target_step` (alpha, risk, optimizer) → `_cost_step` (trade costs, impact streaks, borrow) → `_mark_to_market_step` (returns, halts, delistings and their liquidation) → `_accounting_step` (NAV identity, recording, drift). Day-to-day state lives in `_BookState`; events in `_EventLog`; the output arrays in `_Recorder`. `BacktestResult` lives in `qts/backtest/result.py`. The spec fields that matter:

| Argument | Default | Effect |
|---|---|---|
| `nav0` | 1e6 (pipeline 1e6; CLI demo `--nav` 5e6) | starting NAV in USD; drives ADV caps and impact (capacity) |
| `rebalance_every` | `None` → `cfg.portfolio.rebalance_every_days` (5) | optimise every k-th date; drift in between (forced exits daily) |
| `venue` | `"routed"` | venue the optimiser and the cost estimate price |
| `signal_lag_days` | `None` → `cfg.execution.signal_lag_days` (1) | execution delay $L$ |
| `neutral_styles` | `None` | styles given the style band (pipeline: risk-only styles) |
| `warm_start_weights` | `None` | start from an existing book instead of cash |
| `eligible_fn` | `None` | `t -> Index` of investable names; ignored when `md.eligible` exists |
| `exit_ineligible_days` | `None` → `cfg.portfolio.exit_ineligible_days` (5) | forced-exit length |

`dates` must be a subset of `md.dates` with at least two entries; consecutive trading days are assumed (see the borrow note below).

**Diagnostics.** `BacktestResult.diagnostics` holds `solve_log`, `delistings`, `delisting_summary` (counts and P&L by reason and side), `halts`, `n_halted_name_days`, `forced_exits`, `n_failed_solves`, `failed_solves` (one entry per failed solve: `date`, `status`, `solver_errors`), `n_rebalance_solves` (the denominator of `BacktestResult.failed_solve_fraction()` and of the `failed_solves` hurdle gate), `n_no_signal_days`, `n_ineligible_name_days`, `n_out_of_model_name_days` (names outside the risk model still held after a solve, i.e. forced exits in progress), `n_data_error_returns` / `data_error_returns` (blanked closes, see *Data-error prices*), `n_exante_vol_alerts` / `exante_vol_alerts` (see *Risk-model sanity*), `signal_lag_days`, `venue`, `rebalance_every` and `run_time_sec`. One `solve_log` entry per solve:

| Field | Meaning |
|---|---|
| `date`, `signal_date` | execution date $t$ and signal date $s$ |
| `status`, `solve_time` | solver status (`optimal`, `optimal_inaccurate`, or a failure: the current book is then held and `n_failed_solves` incremented) |
| `binding` | which constraints were binding (per constraint family) |
| `vol_scalar` | vol-target projection scalar (1 = none) |
| `expected_alpha`, `expected_risk`, `expected_cost` | the optimiser's view of the chosen book: alpha over the horizon, annualised ex-ante vol, trade cost + horizon × holding cost (fraction of NAV) |
| `risk_aversion` | the annual $\gamma$ of the returned book (searched in `target_vol` mode, fixed otherwise) |
| `vol_target_binding` | in `target_vol` mode: `target` (ex-ante vol within 2% of target), `caps` (gross / position / ADV caps bind below target; capped book accepted), `bracket` ($\gamma$ at an end of the search bracket), `max_solves`, `no_alpha`, `solve_failed:<status>`; empty in `fixed` mode |
| `n_solves` | convex solves for this date (gamma-search steps and projection included) |
| `n_out_of_model`, `n_excluded_alpha` | held names outside the risk model; names with alpha but no risk-model coverage (excluded, $w$ = 0) |
| `notes` | free-text notes from the optimiser |

When a number in the summary surprises you, the `solve_log` entry for the day in question is where to look. A high share of `caps` means the book cannot reach its vol target inside the leverage and position limits (→ see **Portfolio construction › Sizing**).

### Pitfalls & checks
- The engine stops if NAV ≤ 0 and truncates the results.
- Borrow is charged once per *step*. If `dates` is a sparse subset (multi-day steps), borrow is under-charged proportionally (audit tcost #19). Run on consecutive trading days.
- Tests: `test_nav_identity`, `test_rebalance_every_and_drift`, `test_delisting_is_liquidated_and_flagged`, `test_cost_frame_raises_on_unrecognised_or_nan`, `test_simulator_snapshot_is_the_cost_model_snapshot`, `test_run_with_tcost_model_venue_and_persistence`; `tests/test_simulator_handcomputed.py` (zero-cost path, constant-bps costs and borrow, long and short delistings, halt-then-resume, causal gap-rule delisting, NAV ≤ 0 stop); `tests/test_audit2_costs_backtest.py::test_simulator_books_exact_streak_increment`, `test_gross_after_borrow_and_breakeven_net_of_borrow`.

### References
- [Frazzini, Israel & Moskowitz, Trading Costs](https://conference.nber.org/conf_papers/f68262/f68262.pdf)

## Metrics

All metrics are in `qts/backtest/metrics.py` and are assembled by `BacktestResult.summary()`. Inputs are daily returns; $q = 252$ periods per year. Worked numbers use `runs/demo_post_refactor.txt` and its run directory `runs/20261105T072821098301Z_2d9f80cd` (synthetic data, 1,005 days, lag 0, NAV \$5M). That run came from an uncommitted code tree and predates the `gross_after_borrow` fix; where that matters the current definition is shown next to it.

### CAGR
$$
\text{CAGR} = \Big(\prod_{t=1}^{T}(1+r_t)\Big)^{q/T} - 1
$$

The compound annual growth rate is the constant annual return that turns the starting NAV into the ending NAV. It returns −1 if the product is ≤ 0. Demo: gross 9.08%, net 6.40%. The 2.7-point gap is the annual cost drag (trading + borrow).

### Annualised volatility
$\sigma_{\text{ann}} = \mathrm{sd}(r_t)\sqrt q$ (sample sd, ddof = 1). Annualising by $\sqrt{q}$ assumes uncorrelated daily returns. Demo: net 6.7%.

### Sharpe ratio (gross and net)
$$
\text{SR} = \frac{\overline{r}}{\mathrm{sd}(r)}\sqrt q
$$

where $r$ is the daily return in excess of the risk-free rate. The code's `rf` defaults to **0**: the book is dollar-neutral and earns no cash interest, so its return is already roughly an excess return. The Sharpe ratio is return per unit of risk, and it is invariant to leverage, which is why the plan's hurdles are stated in it. **Gross** uses pre-cost returns and measures signal quality through the optimiser. **Net** is what you keep. Demo: 1.33 gross, 0.96 net. Costs took 28% of the Sharpe.

**Lo autocorrelation adjustment** (`sharpe_net_lo`). $\sqrt q$ scaling is wrong when daily returns are autocorrelated. Positive autocorrelation (e.g. from smoothed or partially traded positions) makes annual risk larger than $\sqrt{q}\times$ daily risk. Lo (2002) gives

$$
\text{SR}_q = \text{SR}_1\cdot\frac{q}{\sqrt{q + 2\sum_{k=1}^{L}(q-k)\rho_k}}
$$

where:
- $\text{SR}_1$ = daily Sharpe; $\rho_k$ = lag-$k$ autocorrelation of daily returns;
- $L = \min(q-1, 20, T-2)$ lags (code default 20). If the denominator is ≤ 0 the unadjusted value is returned.

Worked: with only $\rho_1 = 0.1$, the denominator is $\sqrt{252 + 2\cdot251\cdot0.1} = \sqrt{302.2} = 17.38$, versus $\sqrt{252}=15.87$. The adjusted Sharpe is $15.87/17.38 = 0.913\times$ the naive one. The adjustment can also go up: the demo's `sharpe_net_lo` is 1.69 against 0.96, because its daily returns are slightly negatively autocorrelated on balance. With 20 noisy autocorrelations the adjustment is itself noisy; read it as a flag, not a better estimate.

**Sharpe t-statistic** (`sharpe_t_stat_net`): $\text{SR}_1/\sqrt{(1+\text{SR}_1^2/2)/T}$, using Lo's i.i.d. standard error. Demo: 1.91.

### Sortino ratio
$$
\text{Sortino} = \frac{\overline r}{\sqrt{\tfrac1T\sum_t \min(r_t,0)^2}}\sqrt q
$$

Like Sharpe, but only downside deviations count as risk. The denominator is the root-mean-square of the negative part over **all** $T$ days (full-sample downside deviation), target 0. It is useful when returns are skewed; for a symmetric return stream it is about $\sqrt2\times$ Sharpe. Demo: 1.42 (≈ 1.48 × Sharpe).

### Maximum drawdown and Calmar
Drawdown at $t$ is $\text{NAV}_t/\max_{s\le t}\text{NAV}_s - 1$ on compounded net returns. Max drawdown is its minimum, the worst peak-to-trough loss. Calmar = CAGR / |max drawdown|. Demo: max drawdown −5.3%, Calmar $= 6.40/5.31 = 1.21$. Max drawdown is a single path statistic and very noisy. Over a long enough history every strategy's drawdown grows, so compare drawdowns only over equal-length windows.

### Hit rate
The fraction of non-zero daily net returns that are positive. Stock-selection books typically show 51–54%. A high hit rate with a low Sharpe means small wins and large losses. Demo: 53.7%.

### Turnover (as defined here)
**Two-sided daily turnover** is $\text{TO}_t = \sum_i\lvert x_{i,t}\rvert$ as a fraction of NAV. Buying 1% of one name and selling 1% of another is 2%. A full round trip of a gross-2 book is 4. **One-sided** turnover is half of that. `avg_daily_turnover` is the mean two-sided figure; `turnover_stats` also gives the annualised total ($\times252$). Demo: 8.03%/day two-sided, i.e. $8.03\%\times252 = 20.2\times$ NAV per year.

### Effective holding days
$$
\text{EHD} = \frac{\text{mean gross exposure}}{\text{mean two-sided turnover}/2}
$$

Demo: $2.00/(0.0803/2) = 50$ days. It should be on the order of the signal's half-life (→ see **Portfolio construction › Diagnostics**).

### Cost per side
$$
\text{cost per side (bp)} = \frac{\sum_t(\text{spread}_t + \text{impact}_t + \text{fees}_t)}{\sum_t \text{TO}_t\cdot\text{nav}_t}\times10^4
$$

Trading cost per dollar traded. **Borrow is excluded** because it is a holding cost, not a trading cost. Demo: 10.5 bp. `avg_daily_cost_bps_of_nav` is total cost including borrow, as bp of NAV per day (demo: 0.99). The demo's \$553,545 of total costs split into impact 73%, borrow 15%, fees 11% and spread 0.6%: almost every trade fit inside the auction cap, so it paid almost no spread.

### Breakeven cost: derivation
Suppose every dollar traded costs $c$ (a fraction, per side). The daily trading cost is then $c\cdot\text{TO}_t$, and the mean net return is $\overline{g}^{\,b} - c\,\overline{\text{TO}}$, where $\overline{g}^{\,b}$ is the mean of `gross_after_borrow`: borrow is already charged as realised, because it does not scale with turnover. Setting the mean to zero:

$$
c^\star = \frac{\overline{g}^{\,b}}{\overline{\text{TO}}}\times10^4\ \text{bp per side}
$$

where $\overline{\text{TO}}$ is the mean two-sided daily turnover, aligned so that turnover decided at $t$ meets the return earned $t\to t+1$. **Breakeven** is the per-side cost at which the strategy's edge after borrow is exactly used up. The gate asks that breakeven be at least **2×** the modelled cost, leaving room for the cost model to be wrong by a factor of two.

Demo: the report prints 44.1 bp, computed from plain `gross` by the code of the time. Recomputed from its saved parquet files with the current definition it is **42.3 bp**: borrow (0.15 bp of NAV a day) takes 1.9 bp off. Either way it passes: 2 × 10.5 = 20.9 bp. The implied mean edge after borrow is $42.3\text{ bp}\times0.0803 = 3.4$ bp/day, ≈ 8.5%/yr arithmetic.

### Cost sensitivity table
`cost_sensitivity(r, turnover, cost_bps_per_side=c, multipliers=(0.5, 1, 2, 3))` recomputes net returns as $r_t - m\,c\,\text{TO}_t$ with $r$ = `gross_after_borrow` and reports net Sharpe, net CAGR and mean daily cost for each multiplier $m$. The pipeline uses the modelled cost per side for $c$; `BacktestResult.cost_sensitivity()` does the same. Read it as "how fast does the strategy die if costs are worse than modelled?"

| Multiplier | 0.5× | 1× | 2× | 3× |
|---|---|---|---|---|
| Demo as printed (from `gross`) | 1.170 | 1.013 | 0.697 | 0.382 |
| Recomputed from `gross_after_borrow` (current code) | 1.115 | 0.957 | 0.641 | 0.326 |

<callout icon="⚠️" color="yellow_bg">
In the printed demo the 1× row (1.013) sat well above the actual net Sharpe (0.956), because borrow had vanished from the table. Recomputed the current way, the 1× row is 0.957, essentially the actual net Sharpe. A small gap can remain: the table charges a *flat average* cost per side, while the true net series has per-name, per-day costs (impact is convex, so large trades cost disproportionately more). Use the table for its *slope*: here each extra 10 bp per side costs about 0.31 of Sharpe.
</callout>

### Newey–West t-statistic
$$
t_{\text{NW}} = \frac{\overline r}{\sqrt{\hat S/T}},\qquad \hat S = \hat\gamma_0 + 2\sum_{k=1}^{L}\Big(1-\frac{k}{L+1}\Big)\hat\gamma_k,\qquad L = \big\lfloor 4(T/100)^{2/9}\big\rfloor
$$

where:
- $\hat\gamma_k$ = lag-$k$ autocovariance of daily returns (divided by $T$);
- the Bartlett weights $1-k/(L+1)$ keep $\hat S$ non-negative.

This tests "is the mean daily return different from zero?" while allowing for autocorrelation and heteroskedasticity. With $L=0$ it reduces to the ordinary i.i.d. t-statistic. The implementation is shared with the alpha layer (`qts.core.stats.newey_west_tstat`, at least 5 observations). Demo: $T=1005$ gives $L=6$ and $t = 2.06$. As a check, the i.i.d. approximation $\text{SR}\sqrt{T/252} = 0.96\times\sqrt{3.99} = 1.92$. A t of 2 is far below the plan's threshold of 3 (Harvey–Liu–Zhu): four years is too short to prove a Sharpe near 1.

### Other summary fields
| Field | Meaning |
|---|---|
| `start`, `end`, `n_days`, `final_nav` | first/last NAV date, number of net returns, ending NAV |
| `avg_gross_exposure`, `avg_net_exposure` | mean $\sum\lvert w\rvert$ and $\sum w$ of post-trade books |
| `total_costs_usd` | sum of all booked costs including borrow and delisting liquidations |
| `transfer_coefficient_mean` | mean per-solve TC (→ see **Portfolio construction › Diagnostics**) |
| `realized_vs_exante_vol_ratio` | realised gross vol / mean ex-ante vol over **invested** decision days only (post-trade gross exposure > `INVESTED_GROSS_EPS` = 1e-4 of NAV), forecast at $t$ aligned to the return $t\to t+1$; NaN when fewer than `MIN_INVESTED_DAYS_FOR_VOL_RATIO` = 20 such days have a forecast. An empty-book day holds solver-noise weights (|w| ~ 1e-9–1e-6), whose positive forecast and ~0 return would otherwise pass the `v > 0` filter; `BacktestResult.realized_vs_exante()` also returns the two annual vols and `n_invested_days` |
| `effective_n_names`, `specific_var_share`, `exante_vol_at_gross2` | the diversification numbers (next subsection) |
| `invested_day_frac` | share of decision days whose post-trade gross exposure exceeds 1e-4 of NAV (`BacktestResult.invested_days()`); the report card prints it next to the vol ratio and the pipeline adds a note below 0.5 |
| `breakeven_cost_bps` | per-side breakeven from `gross_after_borrow` (above) |
| `n_delistings`, `n_failed_solves` | counts from `diagnostics` |
| `delisting_pnl_ann` | annualised sum over the delistings of held names of weight × booked delisting return (`BacktestResult.delisting_summary()`; → see **Accounting › Delistings by reason**) |
| `mean_risk_aversion` | mean annual $\gamma$ over the solve log (`sizing_summary`); NaN if no solve reported one |
| `vol_target_caps_share` | share of target-vol solves whose book was capped below the vol target (`vol_target_binding == "caps"`); NaN in `fixed` mode; always 0 with `vol_target_is_cap` (the default), where `BacktestResult.vol_target_outcomes()` gives the share of each outcome (`below_target` / `leverage` / `target`) and the report card prints them |

### Diversification
Three summary fields answer one question: is the gross exposure spread over the universe, or parked in a handful of names? All are means over **invested** decision days (`BacktestResult.diversification()`).

**`effective_n_names`** — the inverse Herfindahl index of the absolute weights:

$$
N_{\text{eff}} = \frac{1}{\sum_i p_i^2}, \qquad p_i = \frac{\lvert w_i\rvert}{\sum_j \lvert w_j\rvert}
$$

where $w_i$ are the post-trade weights. $n$ equal positions give $N_{\text{eff}} = n$; one position gives 1. With a 2% position cap and gross 2 the floor is 100 (every name at the cap); a rank-weighted book over $n$ names has $N_{\text{eff}} \approx 0.75\,n$.

**`specific_var_share`** — the stock-specific share of the risk model's ex-ante variance of the book, $\sum_i w_i^2 d_i \,/\, w^\top \Sigma w$ (`RiskModelOutput.specific_variance` over `portfolio_variance`; recorded per day in `BacktestResult.exante_specific_var`). Read it together with the next number: a factor-hedged book *should* have a high specific share, because the factor part has been hedged away. It is a warning only when the risk per unit of gross is high too.

**`exante_vol_at_gross2`** — the predicted annual vol the book would have at 100% long / 100% short:

$$
\sigma_{\text{gross 2}} = \hat\sigma \cdot \frac{2}{\sum_i \lvert w_i\rvert}
$$

Risk per unit of leverage, comparable across runs of different size. A diversified market-neutral book of several hundred names runs at roughly 3–5% here; the concentrated book of the first real ML run ran at 7.5% against 3.4% for the paper portfolio built from the same score (`docs/notes/construction_fix_2026-11-12.md`).

The report card prints the three on the `diversification` line, and the solve schedule, the number of cancelled small trades and the largest constraint violation they left on the `trading` line.

### References
- [Harvey, Liu & Zhu via CXO Advisory, Taming the Factor Zoo](https://www.cxoadvisory.com/big-ideas/taming-the-factor-zoo/)
- *Lo (2002), The statistics of Sharpe ratios, Financial Analysts Journal.*

## Backtest biases and how this engine guards each

| Bias | What goes wrong | Guard in this system | Residual risk |
|---|---|---|---|
| Look-ahead | a feature, label or parameter uses data after the decision | every model exposes `as_of(t)`; `MarketData.slice_until`; refit-on-truncated-data audits of alpha, risk, exposures and cost snapshots; trainer purge/embargo; **signal lag 1 by default**, so nothing from the execution day's close reaches the decision | lag 1 is a proxy for a 15:45 snapshot; at lag 0 the same-close bias (Timeline) returns |
| Survivorship | universe built from today's survivors, so dead losers vanish | universe mask built per date from point-in-time market cap/price/ADV; delisting return by reason (acquisition 0, performance −30%, voluntary −10%, unknown −30%, on the stock) plus liquidation costs; labels for names delisting inside the horizon use last price × (1 + the same return), with vendor delisting dates when available | needs a data source that includes delisted names (Norgate/Sharadar per plan §7); synthetic data has none of this problem by construction |
| Universe membership | cross-sectional statistics computed over names that join the universe *later* | `MarketData.eligible` panel; features, labels and scores on eligible names only (CRITICAL first-audit fix); **risk-model style exposures** now use the eligible-only market proxy and standardisation (November lag-investigation fix: beta and residual vol had depended on future entrants) | — |
| Restatement / filing timing | fundamentals joined at their *restated* values or before they were public | `merge_asof` on filing date (HIGH audit fix for asset growth); `sharadar_sf1_to_pit_fundamentals(lag_days=...)` | check the loader's `lag_days` on real data: filings published after 16:00 must not be visible at that day's close |
| Overfitting / data snooping | trying many configurations and reporting the best | trial ledger keyed by config hash, data hash, clean code version and run spec (including the signal lag); deflated Sharpe with effective trial count; PBO over a config grid; t > 3 hurdle; fold consistency (→ see **Validation & statistics**) | ledger only knows trials run through the pipeline; dirty-tree runs are never de-duplicated |
| Path noise | one path's Sharpe difference read as a treatment effect | paired multi-seed comparisons (lag investigation); paired block bootstrap SE | not automated in the pipeline |
| Unrealistic fills | assuming any size trades at the close price | ADV participation cap (5%); per-name cap; routed venue (auction only up to 2% of auction volume); power-law impact; exact streak increment | no partial fills or LOC rejections in the daily simulator; trades are priced, never refused |
| Ignoring costs/borrow | gross Sharpe reported as if it were net | the same cost model in optimiser and simulator (equivalence tested); borrow on shorts with per-name source; breakeven and sensitivity net of borrow; 2× breakeven gate | borrow proxy is a placeholder until IBKR SLB history exists |
| Capacity | strategy works at \$1M, not at \$100M | participation caps and impact scale with `nav0` (both in weight units via ADV/NAV) | run the pipeline at several `nav0` values; not automated |
| Regime dependence | one bull market makes everything look good | purged walk-forward fold Sharpes and the "≥ 2/3 folds" gate; IC by year in the run extras | plan's VIX-regime and size-tercile splits (Gate 4) are not in the pipeline yet |

### A checklist for reading any backtest critically
Before believing a result, answer these in order. Each question maps to a field you already have.

1. **Is the gross Sharpe plausible?** Above ~2 net (or ~3 gross) on daily equity data, assume a leak first. Check the `suspect` gate.
2. **Where does the money come from?** Run `brinson_like_attribution`, and look at `exposures.parquet` in the run directory. Steady PnL from `market` or an industry is an unintended bet, not alpha.
3. **How much do costs take, and how fragile is that?** Compare gross and net Sharpe, then read the slope of the cost-sensitivity table. If 2× costs kill the strategy, the cost model has to be right to within a factor of two, and it has never been calibrated on your own fills.
4. **Is turnover consistent with the signal?** Effective holding days far below the signal half-life means you are paying to trade noise.
5. **Is the risk real and sized as intended?** `realized_vs_exante_vol_ratio` near 1. Well above 1 means the book is riskier than it was sized to be — and if `effective_n_names` is low and `exante_vol_at_gross2` high, the gap is concentration: specific risk is the part a risk model forecasts worst for a book the optimizer picked. A high `vol_target_caps_share` (two-sided search only) means the book could not reach its vol target. Read it together with `invested_day_frac`: the ratio is estimated on invested days only, so a run whose book was empty on half the days (an alpha that is exactly zero, e.g. an IC floor) still gets an honest ratio for the days it held risk, and `n/a (too few invested days)` instead of a noise number when it barely traded.
6. **Did the machinery work every day?** `n_failed_solves` (above 2% of rebalances the `failed_solves` gate fails the run), `n_exante_vol_alerts` (above 0.5% the `risk_model_sanity` gate fails: the risk model forecast an absurd vol and the book was sized on it), `n_data_error_returns`, `n_delistings`, `n_out_of_model_name_days`, `n_no_signal_days` and `forced_exits` should all be small and explainable.
7. **Is it consistent?** Fold Sharpes: one great block and three flat ones is a regime bet.
8. **How many things did you try?** The DSR only knows about trials logged in the ledger. Running variants outside the pipeline and reporting the best is exactly the snooping the gates exist to catch.
9. **Does it depend on timing?** Compare lag 0 (optimistic bound) and lag 1 (default) with a paired test. If lag 1 loses most of the edge, the strategy depends on information the live system will not have.
10. **Is the difference you care about bigger than path noise?** For any A/B comparison, the paired SE and a few seeds, not one path.

<callout icon="🤔" color="purple_bg">
Why not use an event-driven / limit-order-book simulator? For a daily closing-auction rebalancer the extra realism buys little. Square-root impact is nearly independent of the execution schedule at modest participation, and auction fills are single prints. BUILD_PLAN §6 recommends a daily-bar backtester with a cvxportfolio-style cost model on realised volume, and keeping an event-driven layer (e.g. NautilusTrader fill models) for when intraday algos are added.
</callout>

## The ResearchPipeline

### What it is
`qts/pipeline.py::ResearchPipeline` is the thin vertical slice that wires every layer together: universe → risk → alpha → cost → portfolio → backtest → validation. One call, `ResearchPipeline(cfg, use_ml=False, runs_root="runs").run(md)`, produces a `PipelineResult`, writes a run directory, and `format_result(res)` renders the report card. Constructor defaults: `venue="routed"`, `nav0=1e6`, `rebalance_every=1`, phase 1 without ML and 2 with it.

### Stages
1. **`fit(md)`**
   - `build_universe_mask` + `apply_mask`: the point-in-time investable universe (`md.eligible`).
   - `build_risk_model(cfg.risk).fit(md_u)`: fundamental factor model (+ statistical companion).
   - `AlphaModel.fit(md_u, risk_exposures_fn=...)`: labels residualised on exactly the set the optimiser neutralises (market, industries and the risk-only styles beta, size, residual vol, liquidity, short-term reversal; design decision D1), never on the alpha themes. With lag $L$ the label of signal date $s$ starts at $\text{close}_{s+L}$, so it measures the return of the trade actually made.
   - `TransactionCostModel(cfg.cost, cfg.execution)` and `SinglePeriodOptimizer(cfg.portfolio, cost, horizon_days=H)`.
2. **`backtest_dates()`**: out-of-sample dates start after the risk model has `min_history_days` (252) of factor returns and after the first out-of-sample ML prediction (if ML is on). The first $L$ of them are dropped so the first signal date is out of sample too, and the last date is dropped because it has no $t+1$ return.
3. **`run`**: `DailyBarSimulator.run` with an eligibility-filtered `alpha_fn`, `risk_fn(d, alpha)` (alpha passed for the alignment term), `neutral_styles` = risk-only styles, `venue="routed"`, `signal_lag_days` = $L$. A changed lag triggers a refit.
4. **`evaluate`**:
   - `bt.summary()`;
   - IC report of the **traded** score (`alpha.as_of(t − L)`) against its residualised target, on **non-overlapping** execution dates every $H$ days, eligible names only, with IC decay at 5, 20 and 60 days;
   - top-minus-bottom decile spread of raw forward returns (value-weighted, Newey–West t), sharing one sampling frequency with the ICIR;
   - fold net Sharpes on **purged walk-forward test blocks** (`ValidationConfig.n_splits` = 6, `embargo_days` = 20, blocks shorter than `min_fold_days` merged away), replacing calendar years;
   - cost sensitivity and breakeven from `gross_after_borrow` (turnover aligned positionally);
   - trial-ledger logging (family = phase + data hash), deflated Sharpe ratio (DSR) from the ledger family, PBO if a grid is configured;
   - hurdle gates; notes.

### Reading the report card
Below is `runs/demo_post_refactor.txt`: 300 synthetic stocks × 1,512 days, composite alpha only, NAV \$5M, run with `--compare-lag`. **Synthetic data has a planted signal; none of these numbers say anything about real markets.**

<callout icon="⚠️" color="yellow_bg">
**Provenance.** `runs/demo_audit2.txt` did not exist when this page was written, so this is the most recent stored demo. It was produced at 02:28 on 2026-11-05 from an **uncommitted** working tree (`git_dirty: true` at `63ce32b`) that the lag investigation could not reproduce on committed code. It also predates three changes: it ran at **lag 0** (the report shows no "signal lag" on the period line), its breakeven and sensitivity come from plain `gross`, and it has no "sizing" line. Read it for *how to read a report card*, not for numbers to compare against. Stored demos should come from clean trees (`git_dirty: false`).
</callout>

```text
=== Backtest summary ===
period            2019-12-10 -> 2023-10-17  (1005 days)
Sharpe gross/net  1.33 / 0.96   (NW t = 2.06)
CAGR gross/net    9.08% / 6.40%   vol 6.7%
max drawdown      -5.3%   calmar 1.21
turnover/day      8.03%   effective holding 50 d
cost per side     10.5 bp   gross exp 2.00  net exp +0.04
transfer coeff    0.53   realized/ex-ante vol 1.04
```

- **period**: first NAV date → last; 1,005 net daily returns (~4 years) after warm-up. A run at lag $L \ge 1$ appends `signal lag 1d`.
- **Sharpe gross/net (NW t)**: net 0.96 is in the plan's success band, but $t = 2.06$ says four years cannot distinguish it from zero at the plan's t > 3 standard.
- **CAGR / vol**: 6.40% net at **6.7%** vol, below the 10% target because gross leverage is at its 2.0 cap (→ see **Portfolio construction › Sizing**). The current report adds a `sizing` line right below `transfer coeff`: `mean gamma` (the mean risk aversion of the returned books) and `vol target capped on X% of solves` (`vol_target_caps_share`). A book at the gross cap below its vol target would show a high capped share. Since 2026-11-12 (vol target as a cap) the line reads `vol cap binding on X% of solves, book under the cap on Y%`, and two more lines follow: `diversification` (effective names, specific share of ex-ante variance, ex-ante vol at gross 2; → **Diversification**) and `trading` (solve every k days, small trades dropped, largest constraint violation after the min-trade projection).
- **max drawdown / calmar**: −5.3%, under one year's vol: benign for this path.
- **turnover / effective holding**: 8.0%/day two-sided; 50 days, at the 50-day monitored target, so no "raise trade_aversion" note (the note fires below 60% of target).
- **cost per side / exposures**: 10.5 bp per side (routed, all-in, excluding borrow); gross 2.00, net +0.04: a dollar-neutral 100/100 book.
- **transfer coeff / realised vs ex-ante**: 0.53, so constraints and cost penalties leak about half the forecast; 1.04, so the risk model forecasts the book's vol well.

The run also logged `WARNING ... 220 name-days with alpha/position outside the risk-model universe` (and 240 for the lag-1 run). Those are held names outside the risk model during forced exits, priced at the 90th-percentile specific variance with zero factor exposure. A few hundred out of ~300,000 name-days is harmless. Thousands would mean the universe and risk-model coverage disagree.

```text
=== Alpha IC (score vs residualised target, sampled every horizon) ===
mean rank IC 0.0392  ICIR 1.45 (annualised at 12.6/yr)  t 2.89  NW t 3.16 (L=3)  n 50
IC decay (horizon d: mean IC)  5d: 0.0180  20d: 0.0392  60d: 0.0590
decile L/S (VW)  mean 1.25%/period  Sharpe 0.92  NW t 1.67 (L=3)
DSR 0.704  (n_trials 5.0, var SR 0.332)  PBO skipped
config 2d9f80cdc0532202  data acc387f0eca55816  git 63ce32bf1f8c  run 20261105T072821098301Z_2d9f80cd
```

- **mean rank IC**: average Spearman correlation of score vs the 20-day residualised target over 50 non-overlapping dates. **ICIR** = mean/sd of IC, annualised at $252/20 = 12.6$ periods per year. $t$ = 2.89 is the plain t-stat; **NW t** = 3.16 uses Newey–West with 3 lags.
- **IC decay**: the same score against 5-, 20- and 60-day labels. IC rising with horizon (0.018 → 0.039 → 0.059) says the signal is slow, consistent with a 50-day holding period and with lag 1 costing little.
- **decile L/S (VW)**: value-weighted top-minus-bottom decile of *raw* forward returns per 20-day period. Its NW t (1.67) is the input to the t-stat gate. A significant IC on the residualised target with an insignificant raw decile spread is normal: the decile portfolio carries unhedged factor noise that the residualised IC removes.
- **DSR**: deflated Sharpe ratio, the probability that the true Sharpe exceeds the best you would expect by chance from `n_trials` trials (5 effective) with cross-trial Sharpe variance 0.332. It must exceed 0.95. **PBO** is skipped without a config grid.
- **config / data / git / run**: identify the trial in the ledger and the run directory. `git 63ce32bf1f8c` with a dirty tree is exactly why this run cannot be reproduced.

```text
=== Fold (purged walk-forward test block) net Sharpe ===
     start        end  n_days  sharpe
2019-12-11 2020-07-30     167    0.76
2020-07-31 2021-03-23     168    0.86
2021-03-24 2021-11-11     167    1.13
2021-11-12 2022-07-05     168    1.35
2022-07-06 2023-02-23     167    0.94
2023-02-24 2023-10-17     168    0.73
```
Six contiguous test blocks of `PurgedWalkForward` over the return dates. Every block is positive and none dominates: the consistency across folds matters more than the average.

```text
=== Cost sensitivity (net Sharpe at cost multiples) ===
            cost_bps_per_side  net_sharpe  net_cagr  mean_daily_cost_bps
multiplier
0.5                     5.233       1.170     0.079                 0.42
1.0                    10.467       1.013     0.068                 0.84
2.0                    20.933       0.697     0.046                 1.68
3.0                    31.400       0.382     0.024                 2.52
```
Described under Metrics: on current code the rows start from `gross_after_borrow` and read 1.115 / 0.957 / 0.641 / 0.326.

```text
=== Plan hurdles ===
OVERALL: FAIL (phase 1)
[FAIL] t_stat: decile L/S t = 1.67 <= 3.0
[PASS] ic_sign: IC > 0 in 4/5 folds (need >= 0.67)
[skipped] incremental_ic: not applied (standalone signal)
[PASS] net_sharpe_delta: net Sharpe by fold (vs zero baseline) >= +0.05 in 6/6 folds (need >= 0.67)
[PASS] breakeven_cost: breakeven 44.1 bps >= 2x modeled 10.5 bps (= 20.9 bps)
[FAIL] dsr: DSR = 0.704 <= 0.95
[skipped] pbo: PBO not supplied (no configuration grid run; set pipeline_pbo_configs)
[PASS] phase_sharpe: phase-1 net Sharpe 0.96 >= 0.3
[PASS] suspect: net Sharpe within plausible range
skipped (not required): incremental_ic, pbo
```

`ic_sign` counts blocks that contain sampled IC dates (5 of the 6 here). Then the execution-delay table from `--compare-lag`:

```text
=== Execution-delay check (D4: signal lag 0 vs 1 day) ===
                       lag 0     lag 1      diff
sharpe_gross           1.328     0.490    -0.839
sharpe_net             0.956     0.111    -0.846
cagr_net               0.064     0.005    -0.059
avg_daily_turnover     0.080     0.082    +0.001
```

This is the pair the lag investigation took apart. Read it with the rule from that investigation: a single-path difference needs a paired standard error (here 0.32) and a return correlation (here 0.75) next to it, and more than one seed, before it means anything.

### The run registry
With `runs_root` set (the CLI default is `runs/`), every run writes `<runs_root>/<run_id>/`, where `run_id` = UTC timestamp + `_` + the first 8 characters of the config hash (e.g. `20261105T072821098301Z_2d9f80cd`). The staging is atomic when the API launches the job.

| File | Contents | Written by |
|---|---|---|
| `manifest.json` | `RunManifest`: run id, created_at, config hash, data hash (every `MarketData` panel), git sha and dirty flag, library versions, python, platform, timings, seeds, command | `save_run` |
| `config.yaml` | the full `SystemConfig` (reload with `SystemConfig.from_yaml`) | `save_run` |
| `summary.json` | backtest summary, IC scalars and decay, decile, DSR and its inputs, PBO, breakeven, folds, cost sensitivity, overall verdict, flags (`use_ml`, `phase`, `signal_lag_days`), hashes, timings, notes | `save_run` |
| `nav.parquet`, `returns.parquet` | NAV; `gross`, `net`, `gross_after_borrow` by realisation date | `save_run` |
| `weights.parquet`, `trades.parquet`, `costs.parquet` | post-trade weights, weight changes, USD costs (`spread, impact, fees, borrow, total`) by execution date | `save_run` |
| `ic.parquet`, `hurdles.json` | sampled rank IC series; the hurdle result | `save_run` |
| `delistings.parquet` | one row per delisting of a held name: `date, ticker, weight, reason, side, stock_return, haircut_return, pnl, pnl_usd, liquidation_usd, cost_usd` (empty table when there were none; absent in runs written before 2026-11-12). `scripts/attribute_run.py` reads it to split the reconciliation gap | `save_run` |
| `extras.json` | run facts (data kind, universe size, seed, venue description, label), DSR moments, IC by calendar year, factor roles | `qts/runs/artifacts.py::write_extra_artifacts` |
| `exante_vol.parquet`, `exposures.parquet` | annualised ex-ante vol of the post-trade book; net book exposure to the market and every style factor, by decision date | same |
| `holdings.parquet` | month-end books (latest 60): date, **signal_date**, ticker, weight, alpha as the optimiser saw it (`alpha.as_of(signal_date)`, second-pass finding 15), specific vol, industry | same |

The extras are best effort: a failure is logged and never fails the run. The trial ledger is a separate append-only file (`runs/trial_ledger.jsonl` by default). Reading runs back: `qts.pipeline.load_run(run_dir)` returns a typed `RunRecord`; `list_runs(root)` lists run directories newest first; the CLI has `python -m qts.cli runs list`, `runs show ID` and `ledger show`; and the workbench API (`api/`) serves the same directories to the UI. Runs from a dirty tree are kept but should not be quoted as reference results.

### Hurdle gates in brief
Each gate is PASS, FAIL or skipped. A skipped gate fails `overall` only if it is in `ValidationConfig.required_gates` (`phase_sharpe`, `suspect`, `dsr`).

| Gate | Test | Default |
|---|---|---|
| `t_stat` | decile L/S Newey–West t | > 3.0 |
| `ic_sign` | mean sampled IC positive in enough purged walk-forward blocks | ≥ 2/3 of blocks |
| `incremental_ic` | IC after regressing on the existing composite | skipped for standalone runs |
| `net_sharpe_delta` | per-block net Sharpe improvement (standalone: vs 0) | ≥ +0.05 in ≥ 2/3 of blocks |
| `breakeven_cost` | breakeven (net of borrow) vs modelled cost per side | ≥ 2× |
| `dsr` | deflated Sharpe | > 0.95 |
| `pbo` | probability of backtest overfitting | < 0.5 |
| `phase_sharpe` | net Sharpe | ≥ 0.3 (phase 1), ≥ 0.5 (phase 2) |
| `suspect` | implausibly high net Sharpe | ≤ 2 |
| `failed_solves` | share of rebalance solves that failed (book held) | ≤ 2 % |
| `risk_model_sanity` | share of rebalances with the risk model's ex-ante vol above 200 % | ≤ 0.5 % |

The demo fails, correctly: four years of a ~1-Sharpe strategy is not enough evidence. Details, formulas and the trial ledger → see **Validation & statistics › The hurdle gates**.

<callout icon="⚠️" color="yellow_bg">
`runs/demo_phase2.txt` and `runs/demo_phase1_post_audit.txt` were produced before the November refactor (auction venue, $b_{\text{auc}}$ = 0.5, calendar-year folds, lag 0). Do not compare their numbers with current runs.
</callout>

## Known limitations

- **Lag 1 is a proxy.** The default execution delay uses the previous close where a live system would use a ~15:45 snapshot. Slightly pessimistic for slow signals; for fast signals the right answer needs intraday data.
- **Single-path comparisons are unreliable.** The optimiser is path dependent: books from two runs whose inputs differ slightly correlate only 0.5–0.75 over four years, and their Sharpes can differ by 0.5–0.8. The pipeline does not yet automate paired multi-seed comparisons.
- **No partial fills in the daily simulator.** Every trade fills completely at the close, at the modelled cost. In reality an LOC can go unfilled if the auction prints through its limit, and a large algo child may need several days. The execution layer's `FillSimulator` models partial ALGO fills and LOC rejections, but it is not in the backtest loop. The routed venue does price the router's auction/continuous split.
- **Permanent impact is a cost, not a price.** The streak increment charges multi-day builds more, but the price path itself is not moved. The next day's return is still computed from unimpacted closes, so the strategy cannot "see" its own footprint in later prices.
- **Adjusted-close dividend assumption.** Returns from adjusted closes assume dividends are reinvested at the close on the ex-date with no withholding or tax drag. For shorts, the adjusted price implicitly charges the dividend, which is economically right (short sellers pay manufactured dividends) but ignores timing and any tax differences.
- **No cash interest, short rebate or margin financing.** The cash and short-proceeds legs earn 0. For a dollar-neutral book this is conservative in high-rate regimes, because short proceeds would earn a rebate net of borrow. A true 150/50 book would owe margin interest, which is not modelled.
- **Borrow fee proxy.** Without `md.borrow_fee`, shorts are financed by a short-interest proxy (GC below 5% SI, rising to a 25%/yr cap at 20% SI, ×2 for market caps under \$2B, including data-poor small caps on the GC default). It is a placeholder until IBKR borrow history exists; check `borrow_fee_source`.
- **Sparse dates under-charge borrow** (one day per step).
- **Out-of-model names** are traded with a conservative but crude risk treatment.
- **Capacity is not reported automatically.** You must re-run at larger `nav0`.

## Roadmap
- An intraday ~15:45 snapshot data path (e.g. Databento) so research features match live features exactly, replacing the lag-1 proxy.
- Paired multi-seed comparison tooling in the CLI (paired block-bootstrap SE and return correlation next to every A/B), generalising `--compare-lag`.
- Regenerate the reference demo from a clean tree (`runs/demo_audit2.txt`) and keep stored demos `git_dirty: false`.
- Fill-probability model for LOC orders in the daily loop (unfilled fraction carried to the next day or routed to an algo).
- Cash interest and short rebate at a configurable rate, plus margin interest for net-long books.
- Per-name permanent-impact state with decay fed into effective prices (audit tcost #6 option ii).
- Automatic capacity sweep over `nav0` in the pipeline report.
- Regime splits (VIX high/low, size terciles) in `evaluate`, per plan Gate 4.
- Factor/specific PnL attribution printed in the report card (`brinson_like_attribution` exists as a library function).
