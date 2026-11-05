# Backtesting

<callout icon="🧭" color="gray_bg">
A backtest replays history day by day. On each day the system builds forecasts, risk and costs using only the data available then, chooses a portfolio, trades it at the closing auction, earns the next day's returns, and pays its costs. The output is the daily return series the strategy would have produced. This page documents the `DailyBarSimulator` in `qts/backtest/simulator.py` precisely: its timeline, its accounting identity, how it books spread, impact, fees, borrow, delistings and forced exits, and every metric it reports. It then covers the backtest biases and how this engine guards against each, how to read the `ResearchPipeline` report card line by line using the real post-audit demo output, and the known limitations. It is honest about the one timing assumption that is optimistic: the signal and the trade use the same closing price.
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
`DailyBarSimulator.run` steps through a list of trading dates. For each decision date $t$ it does exactly the following, in this order:

```text
           close of day t                                      close of day t+1
──────────────┬─────────────────────────────────────────────────────┬──────────────
              │ 1. snapshot_t   (ADV, sigma, spread, price, borrow)  │
              │ 2. eligibility_t, forced-exit caps                   │
              │ 3. alpha_fn(t)  -> expected returns (data <= t)      │
              │ 4. risk_fn(t, alpha) -> B, F, D, alignment (<= t)    │
              │ 5. optimizer.solve -> w_post (fractions of nav_t)    │
              │ 6. trade x = w_post - w_pre at the CLOSE of t        │
              │    cost_t = spread + impact + fees (USD) + borrow    │
              │                                                      │
              │ 7. hold w_post  ── earns r_{t+1} = P_{t+1}/P_t - 1 ──▶│
              │                                                      │ 8. nav_{t+1}
              │                                                      │ 9. drift: w_pre(t+1)
```

Steps 1–5 use only data with timestamp ≤ $t$, enforced by every model's `as_of(t)` and point-in-time slicing (→ see **Architecture & data flow › Point-in-time discipline**). The position decided at $t$ never earns the return of day $t$. It first earns $t\to t+1$. The audit reproduced a hand-computed NAV path exactly (A +10%/−10%/0/+21% with $w_A=0.5$: NAV 1.00 → 1.05 → 0.945 → 0.98795 → 1.0927M).

**Indexing.** Panels recorded per decision (`weights`, `trades`, `costs`, `exante_var`, `turnover`, `transfer_coefficient`) are indexed by $t$. `returns` (gross, net) and `nav` (after the first entry) are indexed by the realisation date $t+1$. Use `BacktestResult.turnover_on_return_dates()` to line up turnover with returns positionally. Never date-join the two: that was audit tcost #4, which shifted costs by one day and dropped the initial build.

### Why trading at the signal's close is (mostly) OK here, and where it is not
The simulator uses the **same closing price** $P_t$ in two roles:

1. as an *input* to the signal, risk model and cost snapshot at $t$ (e.g. the 21-day reversal feature `strev_21` uses $P_t$; `max_ret_21`, rolling beta and residual vol use $r_t$; the ADV window includes day $t$'s volume);
2. as the *execution price* of the trade at $t$.

The second role is realistic: a market-on-close (MOC) or limit-on-close (LOC) order fills at the official auction price, and that price *is* $P_t$. The first role is not literally achievable. On NYSE, MOC/LOC orders must be entered by **15:50 ET**; after that only offsetting orders against a published imbalance are accepted (BUILD_PLAN §6). The closing price is set around 16:00. **When a live system must commit its orders, the close is not yet known.** The production design computes signals from an intraday snapshot shortly before the cutoff (a ~15:30–15:45 snapshot; no code or config fixes the exact time yet) and submits by 15:50. Note that BUILD_PLAN §6 says to price LOC limits off the "3:55 indicative", which is published *after* the 15:50 entry cutoff; the usable reference is the 15:50 imbalance/indicative publication. The daily-bar backtest uses the official close **as a proxy** for that snapshot.

**Bias direction: optimistic.** The backtest's signal knows the last ~10–15 minutes of trading plus the auction print. The proxy is harmless for slow features: 12-1 momentum skips the most recent month entirely, and value, quality and investment are fundamental. It is optimistic for anything that uses $r_t$ or $P_t$ directly, most of all the **short-term reversal** feature. Part of the auction move is temporary (an imbalance-driven print that partly reverses the next day), so a reversal signal that "sees" the print buys the dip *at the dipped price* and earns the bounce. That is a small bid–ask-bounce-style bias. For a 20-day-horizon composite, the size is likely small but **unmeasured**. The ADV-includes-day-$t$ effect is negligible (audit tcost #18).

**How to remove it.** There is **no execution-delay option in the simulator today**. Two routes:

- *Research workaround with the existing API:* `alpha_fn` and `risk_fn` are plain callables, so you can wrap them to read the **previous** trading day: `alpha_fn = lambda t: model.as_of(prev_trading_day(t))`. Trades still execute at $P_t$, but on information to $t-1$. This is conservative: it gives up one day of signal, about $1 - 2^{-1/\text{hl}}$ of the alpha, roughly 1% for a 60-day half-life. The difference between the two runs bounds the same-close bias. Run it before trusting any strategy with a fast sleeve.
- *Roadmap:* a `signal_lag_days` simulator option, and intraday snapshot prices (e.g. Databento) so features can be computed exactly as live at ~15:45.

<callout icon="⚠️" color="yellow_bg">
For any signal with a half-life of a few days (text/event signals, short-term reversal as an alpha rather than a risk factor), the same-close assumption is **not** acceptable. The plan cites evidence that a one-period delay flips event-signal returns negative (Li, Qiao & Zheng, BUILD_PLAN §6). Run the lagged variant, and route such signals to next-open execution in the execution model.
</callout>

### References
- [NYSE, Closing Auction Process fact sheet](https://www.nyse.com/publicdocs/nyse/NYSE_Auctions_Closing_Process_Fact_Sheet.pdf)

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

**Gross return** is $g_t = \sum_i w^{\text{post}}_{i,t} r_{i,t+1}$. **Net return** is $n_t = \text{nav}_{t+1}/\text{nav}_t - 1 = g_t - C_t/\text{nav}_t$. The cash leg (the residual $1 - \sum w$, plus short proceeds) earns **zero**. There is no interest on cash and no short rebate (see Limitations). The identity holds to machine precision (`test_nav_identity`; audit "verified correct").

### Weight drift
Between closes, prices move and weights drift. The next day's pre-trade weights are the post-trade weights grown by their returns and re-based to the new NAV:

$$
w^{\text{pre}}_{i,t+1} = w^{\text{post}}_{i,t}\,(1 + r_{i,t+1})\,\frac{\text{nav}_t}{\text{nav}_{t+1}}
$$

Cash absorbs the costs, which is why $\text{nav}_{t+1}$ (after costs) is in the denominator. Tomorrow's trade is measured from these drifted weights. A winner that grew from 2% to 2.2% of NAV must be trimmed back if the per-name cap is 2%: that is real turnover the backtest correctly charges.

### Cost booking
At $t$ the trade $x = w^{\text{post}} - w^{\text{pre}}$ is converted to dollars at $\text{nav}_t$ and priced by the cost model's point estimate (→ see **Transaction cost model**). The cost interface is being refactored; the components are:

| Component | What it is | Driver |
|---|---|---|
| spread | half the bid–ask spread on the traded dollars | 0 for the auction venue; per-name half-spread for continuous |
| impact | square-root market impact, $b\,\sigma\sqrt{Q/V}$ per dollar | $V$ = auction share × ADV for the auction (≈9%), full ADV for continuous |
| fees | commission, clearing, SEC §31 (sells), FINRA TAF (sells), CAT, auction exchange fee | IBKR Pro tiered; dated SEC/TAF schedules use the decision date |
| borrow | one day of borrow fee on post-trade shorts, $\lvert w\rvert\cdot\text{nav}_t\cdot\text{fee}/252$ | `md.borrow_fee` if present, else a short-interest proxy, else GC 0.5%/yr |

**Impact persistence.** Square-root impact is concave in size, so building a position over $k$ days costs more than $k$ independent one-day trades if the earlier impact has not decayed. The simulator tracks a per-name **streak** of consecutive same-direction trading days and charges day $k$ of a streak the *incremental* cost of extending a $k$-day metaorder:

$$
m_k = k^{3/2} - (k-1)^{3/2}
$$

as a multiplier on that day's impact ($m_1 = 1$, $m_2 = 1.83$, $m_3 = 2.37$, $m_4 = 2.80$). This treats impact within a streak as fully persistent, which is conservative (Frazzini et al. estimate ~70% permanent). It is on by default (`CostConfig.impact_persistence = True`). The **optimiser** still sees only the one-day marginal cost, because it cannot know the future streak.

**Venue.** `run(..., venue="auction")` (the pipeline default) prices every trade with the auction model. The snapshot now carries `venue`, `date` and real spreads through to the cost estimate (audit tcost #3 fix). `_cost_frame` raises on unrecognised cost columns or non-finite costs, so a cost-model refactor cannot silently make the backtest free (audit tcost #8 fix).

### Delistings
A held name whose next close is missing (delisting, permanent halt) would otherwise earn 0 and vanish for free. The simulator instead:
1. applies `CostConfig.delisting_haircut = -0.30` **against the position's sign**: a long earns −30% and a short earns −30% too (the price is treated as +30%). This is conservative for both. The haircut follows the CRSP-style convention for performance delistings with missing returns (Shumway 1997);
2. force-liquidates the remaining value, $x_{\text{liq}} = -w(1+r)$, at continuous-venue **spread + fees** (no impact);
3. books the liquidation as a trade on decision row $t$, so it appears in `trades`, turnover and costs;
4. records `{date, ticker, weight, haircut_return, liquidation_usd, cost_usd}` in `diagnostics["delistings"]`.

Names with no price at $t$ go on the optimiser's no-trade list, and their weights are carried unchanged.

### Eligibility exits
If a held name leaves the investable universe (`md.eligible`), its expected return is set to 0 and the optimiser receives a `force_to_zero` cap that shrinks linearly to 0 over `PortfolioConfig.exit_ineligible_days = 5` (→ see **Portfolio construction › Constraints**). Ineligible names that are not held go on the no-trade list. Events go to `diagnostics["forced_exits"]`, and `n_ineligible_name_days` counts the exposure.

### `rebalance_every`
With `rebalance_every = k > 1`, the optimiser runs only every $k$-th date. On other days the book drifts with no trade, but borrow is still charged daily, and forced-exit caps only advance on rebalance days.

### In the code
`DailyBarSimulator(cfg, cost_model, adv_window=63, vol_window=63).run(md, dates, alpha_fn, risk_fn, optimizer, ...)` is the whole engine. The pipeline builds the arguments for you, but the simulator can be driven directly, which is how the timing-lag experiment above is run:

```python
sim = DailyBarSimulator(cfg, self.cost)
bt = sim.run(self.md_u, dates, alpha_fn, risk_fn, self.optimizer, nav0=self.nav0,
             rebalance_every=self.rebalance_every, venue=self.venue,
             neutral_styles=neutral_styles, progress=progress, **extra)
```

The keyword arguments that matter:

| Argument | Default | Effect |
|---|---|---|
| `nav0` | 1e6 | starting NAV in USD; drives ADV caps and impact (capacity) |
| `rebalance_every` | 1 | optimise every k-th date; drift in between |
| `venue` | `"auction"` | venue the optimiser and the cost estimate price |
| `neutral_styles` | `None` | styles given the style band (pipeline: risk-only styles) |
| `warm_start_weights` | `None` | start from an existing book instead of cash |
| `eligible_fn` | `None` | `t -> Index` of investable names; ignored when `md.eligible` exists |
| `exit_ineligible_days` | `None` → `cfg.portfolio.exit_ineligible_days` (5) | forced-exit length |

`dates` must be a subset of `md.dates` with at least two entries; consecutive trading days are assumed (see the borrow note below). The result's `diagnostics` holds `solve_log` (per-solve status, binding constraints, vol scalar, expected alpha/risk/cost, notes), `delistings`, `forced_exits`, `n_failed_solves`, `n_ineligible_name_days`, `n_out_of_model_name_days` and `run_time_sec`. When a number in the summary surprises you, the `solve_log` entry for the day in question is where to look.

### Pitfalls & checks
- The engine stops if NAV ≤ 0 and truncates the results.
- Borrow is charged once per *step*. If `dates` is a sparse subset (multi-day steps), borrow is under-charged proportionally (audit tcost #19). Run on consecutive trading days.
- Tests: `test_nav_identity`, `test_rebalance_every_and_drift`, `test_delisting_is_liquidated_and_flagged`, `test_cost_frame_raises_on_unrecognised_or_nan`, `test_simulator_snapshot_keeps_venue_date_and_spreads`, `test_run_with_tcost_model_venue_and_persistence`.

### References
- [Frazzini, Israel & Moskowitz, Trading Costs](https://conference.nber.org/conf_papers/f68262/f68262.pdf)

## Metrics

All metrics are in `qts/backtest/metrics.py` and are assembled by `BacktestResult.summary()`. Inputs are daily returns; $q = 252$ periods per year. Worked numbers use the post-audit Phase-1 demo (`runs/demo_phase1_post_audit.txt`, synthetic data, 1,005 days).

### CAGR
$$
\text{CAGR} = \Big(\prod_{t=1}^{T}(1+r_t)\Big)^{q/T} - 1
$$

The compound annual growth rate is the constant annual return that turns the starting NAV into the ending NAV. It returns −1 if the product is ≤ 0. Demo: gross 8.20%, net 4.76%. The 3.4-point gap is the annual cost drag (trading + borrow).

### Annualised volatility
$\sigma_{\text{ann}} = \mathrm{sd}(r_t)\sqrt q$ (sample sd, ddof = 1). Annualising by $\sqrt{q}$ assumes uncorrelated daily returns. Demo: net 6.6%.

### Sharpe ratio (gross and net)
$$
\text{SR} = \frac{\overline{r}}{\mathrm{sd}(r)}\sqrt q
$$

where $r$ is the daily return in excess of the risk-free rate. The code's `rf` defaults to **0**: the book is dollar-neutral and earns no cash interest, so its return is already roughly an excess return. The Sharpe ratio is return per unit of risk, and it is invariant to leverage, which is why the plan's hurdles are stated in it. **Gross** uses pre-cost returns and measures signal quality through the optimiser. **Net** is what you keep. Demo: 1.24 gross, 0.74 net. Costs took 40% of the Sharpe, which is typical and is why cost modelling is first-class here.

**Lo autocorrelation adjustment** (`sharpe_net_lo`). $\sqrt q$ scaling is wrong when daily returns are autocorrelated. Positive autocorrelation (e.g. from smoothed or partially traded positions) makes annual risk larger than $\sqrt{q}\times$ daily risk. Lo (2002) gives

$$
\text{SR}_q = \text{SR}_1\cdot\frac{q}{\sqrt{q + 2\sum_{k=1}^{L}(q-k)\rho_k}}
$$

where:
- $\text{SR}_1$ = daily Sharpe; $\rho_k$ = lag-$k$ autocorrelation of daily returns;
- $L = \min(q-1, 20, T-2)$ lags (code default 20). If the denominator is ≤ 0 the unadjusted value is returned.

Worked: with only $\rho_1 = 0.1$, the denominator is $\sqrt{252 + 2\cdot251\cdot0.1} = \sqrt{302.2} = 17.38$, versus $\sqrt{252}=15.87$. The adjusted Sharpe is $15.87/17.38 = 0.913\times$ the naive one, so 0.74 → 0.68.

**Sharpe t-statistic** (`sharpe_t_stat_net`): $\text{SR}_1/\sqrt{(1+\text{SR}_1^2/2)/T}$, using Lo's i.i.d. standard error.

### Sortino ratio
$$
\text{Sortino} = \frac{\overline r}{\sqrt{\tfrac1T\sum_t \min(r_t,0)^2}}\sqrt q
$$

Like Sharpe, but only downside deviations count as risk. The denominator is the root-mean-square of the negative part over **all** $T$ days (full-sample downside deviation), target 0. It is useful when returns are skewed; for a symmetric return stream it is about $\sqrt2\times$ Sharpe.

### Maximum drawdown and Calmar
Drawdown at $t$ is $\text{NAV}_t/\max_{s\le t}\text{NAV}_s - 1$ on compounded net returns. Max drawdown is its minimum, the worst peak-to-trough loss. Calmar = CAGR / |max drawdown|. Demo: max drawdown −6.3%, Calmar $= 4.76/6.3 = 0.75$. Max drawdown is a single path statistic and very noisy. Over a long enough history every strategy's drawdown grows, so compare drawdowns only over equal-length windows.

### Hit rate
The fraction of non-zero daily net returns that are positive. Stock-selection books typically show 51–54%. A high hit rate with a low Sharpe means small wins and large losses.

### Turnover (as defined here)
**Two-sided daily turnover** is $\text{TO}_t = \sum_i\lvert x_{i,t}\rvert$ as a fraction of NAV. Buying 1% of one name and selling 1% of another is 2%. A full round trip of a gross-2 book is 4. **One-sided** turnover is half of that. `avg_daily_turnover` is the mean two-sided figure; `turnover_stats` also gives the annualised total ($\times252$). Demo: 6.90%/day two-sided, i.e. $6.9\%\times252 = 17.4\times$ NAV per year.

### Effective holding days
$$
\text{EHD} = \frac{\text{mean gross exposure}}{\text{mean two-sided turnover}/2}
$$

Demo: $2.00/(0.069/2) = 58$ days. It should be on the order of the signal's half-life (→ see **Portfolio construction › Diagnostics**).

### Cost per side
$$
\text{cost per side (bp)} = \frac{\sum_t(\text{spread}_t + \text{impact}_t + \text{fees}_t)}{\sum_t \text{TO}_t\cdot\text{nav}_t}\times10^4
$$

Trading cost per dollar traded. **Borrow is excluded** because it is a holding cost, not a trading cost. Demo: 16.3 bp. `avg_daily_cost_bps_of_nav` is total cost including borrow, as bp of NAV per day.

### Breakeven cost: derivation
Suppose every dollar traded costs $c$ (a fraction, per side). The daily cost is then $c\cdot\text{TO}_t$, and the mean net return is $\overline{g} - c\,\overline{\text{TO}}$. Setting it to zero:

$$
c^\star = \frac{\overline{g}}{\overline{\text{TO}}}\times10^4\ \text{bp per side}
$$

where $\overline g$ is the mean daily gross return and $\overline{\text{TO}}$ the mean two-sided daily turnover, aligned so that turnover decided at $t$ meets the return earned $t\to t+1$. **Breakeven** is the per-side cost at which the strategy's gross edge is exactly used up. Demo: 46.6 bp. So the implied mean gross return is $46.6\text{ bp}\times0.069 = 3.2$ bp/day (≈8.1%/yr arithmetic, consistent with the 8.2% gross CAGR). The gate asks that breakeven be at least **2×** the modelled cost (2 × 16.3 = 32.7 bp → PASS), leaving room for the cost model to be wrong by a factor of two.

### Cost sensitivity table
`cost_sensitivity(gross, turnover, cost_bps_per_side=c, multipliers=(0.5, 1, 2, 3))` recomputes net returns as $g_t - m\,c\,\text{TO}_t$ and reports net Sharpe, net CAGR and mean daily cost for each multiplier $m$. The pipeline uses the modelled cost per side for $c$. Read it as "how fast does the strategy die if costs are worse than modelled?" Demo: Sharpe 1.02 / 0.80 / 0.37 / −0.07 at 0.5×/1×/2×/3×.

<callout icon="⚠️" color="yellow_bg">
The 1× row (net Sharpe 0.80) is higher than the reported net Sharpe (0.74). That is expected: the sensitivity table charges a *flat average* cost per side and **excludes borrow**, while the true net series has per-name, per-day costs (impact is convex, so large trades cost disproportionately more) plus borrow. Use the table for its *slope*, not its level.
</callout>

### Newey–West t-statistic
$$
t_{\text{NW}} = \frac{\overline r}{\sqrt{\hat S/T}},\qquad \hat S = \hat\gamma_0 + 2\sum_{k=1}^{L}\Big(1-\frac{k}{L+1}\Big)\hat\gamma_k,\qquad L = \big\lfloor 4(T/100)^{2/9}\big\rfloor
$$

where:
- $\hat\gamma_k$ = lag-$k$ autocovariance of daily returns (divided by $T$);
- the Bartlett weights $1-k/(L+1)$ keep $\hat S$ non-negative.

This tests "is the mean daily return different from zero?" while allowing for autocorrelation and heteroskedasticity. With $L=0$ it reduces to the ordinary i.i.d. t-statistic. Demo: $T=1005$ gives $L=6$ and $t = 1.54$. As a check, the i.i.d. approximation $\text{SR}\sqrt{T/252} = 0.74\times\sqrt{3.99} = 1.48$. A t of 1.5 is far below the plan's threshold of 3 (Harvey–Liu–Zhu): four years is simply too short to prove a Sharpe of 0.74.

### Other summary fields
| Field | Meaning |
|---|---|
| `start`, `end`, `n_days`, `final_nav` | first/last NAV date, number of net returns, ending NAV |
| `avg_gross_exposure`, `avg_net_exposure` | mean $\sum\lvert w\rvert$ and $\sum w$ of post-trade books |
| `total_costs_usd` | sum of all booked costs including borrow and delisting liquidations |
| `transfer_coefficient_mean` | mean per-solve TC (→ see **Portfolio construction › Diagnostics**) |
| `realized_vs_exante_vol_ratio` | realised gross vol / mean ex-ante vol, forecast at $t$ aligned to the return $t\to t+1$ |
| `n_delistings`, `n_failed_solves` | counts from `diagnostics` |

### References
- [Harvey, Liu & Zhu via CXO Advisory, Taming the Factor Zoo](https://www.cxoadvisory.com/big-ideas/taming-the-factor-zoo/)

## Backtest biases and how this engine guards each

| Bias | What goes wrong | Guard in this system | Residual risk |
|---|---|---|---|
| Look-ahead | a feature, label or parameter uses data after $t$ | every model exposes `as_of(t)`; `MarketData.slice_until`; audit verified alpha, risk, exposures and cost snapshots identical when refit on data truncated at $t$; trainer purge/embargo | same-close signal/execution (above); ADV includes day $t$ volume |
| Survivorship | universe built from today's survivors, so dead losers vanish | universe mask built per date from point-in-time market cap/price/ADV; delisting haircut −30% plus liquidation costs; labels for names delisting inside the horizon use last price × (1 − 30%) | needs a data source that includes delisted names (Norgate/Sharadar per plan §7); synthetic data has none of this problem by construction |
| Universe membership | cross-sectional ranks computed over names that join the universe *later* | `MarketData.eligible` panel; features, labels and scores on eligible names only (CRITICAL audit fix: +110%/yr in the 63 days before entry had leaked) | — |
| Restatement / filing timing | fundamentals joined at their *restated* values or before they were public | `merge_asof` on filing date (HIGH audit fix for asset growth); `sharadar_sf1_to_pit_fundamentals(lag_days=...)` | default `lag_days = 0` makes a filing visible at the close of its filing date, though many are filed after 16:00; use `lag_days = 1` on real data |
| Overfitting / data snooping | trying many configurations and reporting the best | trial ledger keyed by (config hash, data hash); deflated Sharpe with effective trial count; PBO over a config grid; t > 3 hurdle; fold consistency (→ see **Validation & statistics**) | ledger only knows trials run through the pipeline |
| Unrealistic fills | assuming any size trades at the close price | ADV participation cap (5%); per-name cap; square-root impact with auction volume ≈ 9% ADV; persistence multiplier | no partial fills in the daily simulator; trades > auction capacity priced, not refused |
| Ignoring costs/borrow | gross Sharpe reported as if it were net | full cost model in both optimiser and simulator (same function); borrow on shorts with short-interest proxy and HTB cap; cost sensitivity; 2× breakeven gate | borrow proxy is a placeholder until IBKR SLB history exists |
| Capacity | strategy works at $1M, not at $100M | participation caps and impact scale with `nav0` (both in weight units via ADV/NAV) | run the pipeline at several `nav0` values; not automated |
| Regime dependence | one bull market makes everything look good | calendar-year fold Sharpes and the "≥ 2/3 folds" gate; IC by year | plan's VIX-regime and size-tercile splits (Gate 4) are not in the pipeline yet |

### A checklist for reading any backtest critically
Before believing a result, answer these in order. Each question maps to a field you already have.

1. **Is the gross Sharpe plausible?** Above ~2 net (or ~3 gross) on daily equity data, assume a leak first. Check the `suspect` gate.
2. **Where does the money come from?** Run `brinson_like_attribution`. Steady PnL from `market` or an industry is an unintended bet, not alpha.
3. **How much do costs take, and how fragile is that?** Compare gross and net Sharpe, then read the slope of the cost-sensitivity table. If 2× costs kill the strategy, the cost model has to be right to within a factor of two, and it has never been calibrated on your own fills.
4. **Is turnover consistent with the signal?** Effective holding days far below the signal half-life means you are paying to trade noise.
5. **Is the risk real?** `realized_vs_exante_vol_ratio` near 1. Well above 1 means the book is riskier than it was sized to be.
6. **Did the machinery work every day?** `n_failed_solves`, `n_delistings`, `n_out_of_model_name_days` and `forced_exits` should all be small and explainable.
7. **Is it consistent?** Fold Sharpes by year: one great year and three flat ones is a regime bet.
8. **How many things did you try?** The DSR only knows about trials logged in the ledger. Running variants outside the pipeline and reporting the best is exactly the snooping the gates exist to catch.
9. **Does it survive the one-day signal lag?** If the lagged run loses most of the edge, the strategy depends on the same-close assumption.

<callout icon="🤔" color="purple_bg">
Why not use an event-driven / limit-order-book simulator? For a daily closing-auction rebalancer the extra realism buys little. Square-root impact is nearly independent of the execution schedule at modest participation, and auction fills are single prints. BUILD_PLAN §6 recommends a daily-bar backtester with a cvxportfolio-style cost model on realised volume, and keeping an event-driven layer (e.g. NautilusTrader fill models) for when intraday algos are added.
</callout>

## The ResearchPipeline

### What it is
`qts/pipeline.py::ResearchPipeline` is the thin vertical slice that wires every layer together: universe → risk → alpha → cost → portfolio → backtest → validation. One call, `ResearchPipeline(cfg, use_ml=False).run(md)`, produces a `PipelineResult`, and `format_result(res)` renders the report card.

### Stages
1. **`fit(md)`**
   - `build_universe_mask` + `apply_mask`: the point-in-time investable universe (`md.eligible`).
   - `build_risk_model(cfg.risk).fit(md_u)`: fundamental factor model (+ statistical companion).
   - `AlphaModel.fit(md_u, risk_exposures_fn=...)`: labels residualised on the risk model's **risk-only** exposures (market, industries, beta, size, residual vol, liquidity, short-term reversal), never on the alpha themes.
   - `TransactionCostModel(cfg.cost)` and `SinglePeriodOptimizer(cfg.portfolio, cost, horizon_days=H)`.
2. **`backtest_dates()`**: out-of-sample dates start after the risk model has `min_history_days` (252) of factor returns and after the first out-of-sample ML prediction (if ML is on). The last date is dropped because it has no $t+1$ return.
3. **`run`**: `DailyBarSimulator.run` with an eligibility-filtered `alpha_fn`, `risk_fn(d, alpha)` (alpha passed for the alignment term), `neutral_styles` = risk-only styles, `venue="auction"`, `nav0 = 1e6`.
4. **`evaluate`**:
   - `bt.summary()`;
   - IC report of the live score against the residualised target, on **non-overlapping** dates every $H$ days, eligible names only;
   - top-minus-bottom decile spread of raw forward returns (value-weighted, Newey–West t);
   - calendar-year fold net Sharpes (years with ≥ 60 days);
   - cost sensitivity and breakeven (turnover aligned positionally);
   - trial-ledger logging, deflated Sharpe ratio (DSR) from the ledger family, PBO if a grid is configured;
   - hurdle gates; notes.

### Reading the report card
Below is the actual post-audit Phase-1 demo (`runs/demo_phase1_post_audit.txt`): 300 synthetic stocks × 1,512 days, composite alpha only. **Synthetic data has a planted signal; none of these numbers say anything about real markets.**

```text
=== Backtest summary ===
period            2019-12-10 -> 2023-10-17  (1005 days)
Sharpe gross/net  1.24 / 0.74   (NW t = 1.54)
CAGR gross/net    8.20% / 4.76%   vol 6.6%
max drawdown      -6.3%   calmar 0.75
turnover/day      6.90%   effective holding 58 d
cost per side     16.3 bp   gross exp 2.00  net exp +0.03
transfer coeff    0.43   realized/ex-ante vol 1.04
```

- **period**: first NAV date → last; 1,005 net daily returns (~4 years) after warm-up.
- **Sharpe gross/net (NW t)**: net 0.74 is in the plan's success band, but $t = 1.54$ says four years cannot distinguish it from zero at the plan's t > 3 standard.
- **CAGR / vol**: 4.76% net at **6.6%** vol, below the 10% target because gross leverage is at its 2.0 cap and the vol-target projection cannot lever up (→ see **Portfolio construction › Volatility targeting**).
- **max drawdown / calmar**: −6.3%, under one year's vol: benign for this path.
- **turnover / effective holding**: 6.9%/day two-sided; 58 days, above the 50-day monitored target, so no "raise trade_aversion" note.
- **cost per side / exposures**: 16.3 bp per side (auction model, all-in); gross 2.00, net +0.03: a dollar-neutral 100/100 book.
- **transfer coeff / realised vs ex-ante**: 0.43, so constraints and cost penalties leak more than half the forecast; 1.04, so the risk model forecasts the book's vol well.

The run also logged `WARNING ... 230 name-days with alpha/position outside the risk-model universe`. Those names were priced at the 90th-percentile specific variance with zero factor exposure. A few hundred out of ~300,000 name-days is harmless. Thousands would mean the universe and risk-model coverage disagree.

```text
=== Alpha IC (score vs residualised target, sampled every horizon) ===
mean rank IC 0.0369  ICIR 1.51 (annualised at 12.6/yr)  t 3.02  n 50
decile L/S (VW)  mean 0.52%/period  Sharpe 0.38  NW t 0.74
DSR 0.666  (n_trials 4.0, var SR 0.251)  PBO skipped
config 944dd5b41832277e  data 6c473c276d0c29f4
```

- **mean rank IC**: average Spearman correlation of score vs the 20-day residualised target over 50 non-overlapping dates. **ICIR** = mean/sd of IC, annualised at $252/20 = 12.6$ periods per year. $t = 3.02$ is the IC's t-stat.
- **decile L/S (VW)**: value-weighted top-minus-bottom decile of *raw* forward returns per 20-day period. Its NW t (0.74) is the input to the t-stat gate. Note the contrast: a significant IC on the residualised target, but an insignificant raw decile spread. The decile portfolio carries unhedged factor noise that the residualised IC removes.
- **DSR**: deflated Sharpe ratio, the probability that the true Sharpe exceeds the best you would expect by chance from `n_trials` trials (4 effective) with cross-trial Sharpe variance 0.251. It must exceed 0.95. **PBO** is skipped without a config grid.
- **config / data hashes**: identify the trial in the ledger. Identical re-runs update the record instead of inflating the trial count.

```text
=== Fold (calendar-year) net Sharpe ===
2020 0.69 | 2021 0.71 | 2022 0.85 | 2023 0.48
```
These are calendar-year net Sharpes (shown on one line here; printed one per line). Consistency across folds matters more than the average.

The **cost sensitivity** block is described under Metrics above. The **Plan hurdles** block, then:

```text
OVERALL: FAIL (phase 1)
[FAIL] t_stat: decile L/S t = 0.74 <= 3.0
[PASS] ic_sign: IC > 0 in 4/5 folds
[skipped] incremental_ic: not applied (standalone signal)
[PASS] net_sharpe_delta: net Sharpe by fold (vs zero baseline) >= +0.05 in 4/4 folds
[PASS] breakeven_cost: breakeven 46.6 bps vs 2x modeled 32.7 bps
[FAIL] dsr: DSR = 0.666 <= 0.95
[skipped] pbo: PBO not supplied (no configuration grid run; set pipeline_pbo_configs)
[PASS] phase_sharpe: phase-1 net Sharpe 0.74 >= 0.3
[PASS] suspect: net Sharpe within plausible range
```

### Hurdle gates in brief
Each gate is PASS, FAIL or skipped. A skipped gate fails `overall` only if it is in `ValidationConfig.required_gates` (`phase_sharpe`, `suspect`, `dsr`).

| Gate | Test | Default |
|---|---|---|
| `t_stat` | decile L/S Newey–West t | > 3.0 |
| `ic_sign` | IC positive in enough folds | ≥ 2/3 of folds |
| `incremental_ic` | IC after regressing on the existing composite | skipped for standalone runs |
| `net_sharpe_delta` | per-fold net Sharpe improvement (standalone: vs 0) | ≥ +0.05 in ≥ 2/3 of folds |
| `breakeven_cost` | breakeven vs modelled cost per side | ≥ 2× |
| `dsr` | deflated Sharpe | > 0.95 |
| `pbo` | probability of backtest overfitting | < 0.5 |
| `phase_sharpe` | net Sharpe | ≥ 0.3 (phase 1), ≥ 0.5 (phase 2) |
| `suspect` | implausibly high net Sharpe | ≤ 2 |

The demo fails, correctly: four years of a 0.74-Sharpe strategy is not enough evidence. Details, formulas and the trial ledger → see **Validation & statistics › The hurdle gates**.

<callout icon="⚠️" color="yellow_bg">
`runs/demo_phase2.txt` was produced **before** the audit fixes (its report has no decile line, no DSR line and the old PBO "FAIL when missing" behaviour). Do not compare its numbers with the post-audit Phase-1 run.
</callout>

## Known limitations

- **Same-close signal and execution.** Optimistic for features using $P_t$/$r_t$ (see Timeline). No built-in lag option; use the wrapper workaround.
- **No partial fills in the daily simulator.** Every trade fills completely at the close, at the modelled cost. In reality an LOC can go unfilled if the auction prints through its limit, and a large order may exceed auction liquidity. The execution layer's `FillSimulator` models partial ALGO fills and LOC rejections, but it is not in the backtest loop.
- **Permanent impact is a cost, not a price.** The persistence multiplier charges multi-day builds more, but the price path itself is not moved. The next day's return is still computed from unimpacted closes, so the strategy cannot "see" its own footprint in later prices (e.g. a reversal signal reacting to impact the book caused).
- **Adjusted-close dividend assumption.** Returns from adjusted closes assume dividends are reinvested at the close on the ex-date with no withholding or tax drag. For shorts, the adjusted price implicitly charges the dividend, which is economically right (short sellers pay manufactured dividends) but ignores timing and any tax differences.
- **No cash interest, short rebate or margin financing.** The cash and short-proceeds legs earn 0. For a dollar-neutral book this is conservative in high-rate regimes, because short proceeds would earn a rebate net of borrow. A true 150/50 book would owe margin interest, which is not modelled.
- **All trades at the auction cost model.** This is conservative relative to the router's actual mix, since large orders go to cheaper in-session algos (audit tcost #11).
- **Borrow fee proxy.** Without `md.borrow_fee`, shorts are financed by a short-interest proxy (GC below 5% SI, rising to a 25%/yr cap at 20% SI, ×2 for caps under $2B). It is a placeholder until IBKR borrow history exists.
- **Sparse dates under-charge borrow** (one day per step).
- **Out-of-model names** are traded with a conservative but crude risk treatment.
- **Capacity is not reported automatically.** You must re-run at larger `nav0`.

## Roadmap
- `signal_lag_days` (execution delay) option in `DailyBarSimulator.run`, and an intraday ~15:45 snapshot data path so research features match live features exactly.
- Fill-probability model for LOC orders in the daily loop (unfilled fraction carried to the next day or routed to an algo).
- Cash interest and short rebate at a configurable rate, plus margin interest for net-long books.
- Per-name permanent-impact state fed into effective prices (audit tcost #6 option ii).
- Automatic capacity sweep over `nav0` in the pipeline report.
- Regime splits (VIX high/low, size terciles) in `evaluate`, per plan Gate 4.
- Factor/specific PnL attribution printed in the report card (`brinson_like_attribution` exists as a library function).
