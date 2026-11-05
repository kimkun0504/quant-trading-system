# Transaction cost model

<callout icon="🧭" color="gray_bg">
This page explains how the system prices a trade before making it and measures what the trade really cost afterwards. Cost has three parts. **Explicit fees** are the broker, regulatory and exchange charges on the confirmation. **Implicit costs** never appear on a statement: the bid-ask spread, the price move your own order causes (market impact), and the price drift while you wait. **Holding costs** are mainly the fee for borrowing shares you are short. All of it lives behind one typed contract (`qts.core.costs`: a `LiquiditySnapshot` in, a `CostEstimate` out, a `CostModel` protocol), implemented by `qts.tcost.TransactionCostModel`. The optimizer gets a convex function of the weight change: a linear term (half-spread plus fees) plus a power-law impact term, `a|x| + b·σ·|x|^e/(V/nav)^{e−1}` with $e$ = 1.5 (the square-root law) by default. The backtester charges the same model in dollars. The default **routed** venue prices each trade the way the router executes it: the first slice in the closing auction, the rest in the continuous session. A transaction cost analysis (TCA) ledger breaks every real fill down into delay, impact, opportunity and fees, and recalibrates the impact prefactor `b` separately for each venue. Most published equity anomalies lose essentially all their net return once costs are counted honestly, so this layer decides whether an alpha is real. It is not a detail.
</callout>

<callout icon="🆕" color="green_bg">
**What changed (Nov 2026)**
- **Typed contract.** `LiquiditySnapshot`, `CostEstimate` and the `CostModel` protocol in `qts/core/costs.py` replace the dict-of-Series snapshot and free-form DataFrames (§10).
- **Auction prior $b_{\text{auc}}$ = 0.3** (was 0.5). Its effective prefactor per $\sqrt{Q/\text{ADV}}$ is 0.3/√0.09 = 1.0, equal to the continuous session, so the auction saves the half-spread and never costs more impact. The old "auction is dearer than continuous" tension is gone (§6).
- **Routed venue** (simulator and pipeline default): auction up to the router's cap, continuous remainder priced as the increment $C(q) - C(q_a)$, fees as two orders. The optimizer gets an exact convex surrogate whenever $\beta_a \le b_c$ (§6).
- **Configurable impact exponent** `impact_exponent` (default 1.5, validated $1 < e \le 2$) is now read by the cost model, the optimizer terms, the router's LOC offset and the TCA regressor.
- **Exact multi-day persistence:** day $k$ of a streak is charged $C(Q_k) - C(Q_{k-1})$ from the actual cumulative notional. The $k^{1.5} - (k-1)^{1.5}$ multiplier survives only as the equal-size fallback, and it is not a bound (§7).
- **Borrow source per name**, and the small-cap multiplier now also applies to the GC default of data-poor small caps (§8). TCA regresses on $\sigma(F/V)^{e-1}$ with venue-aware $V$, and reversion defaults to the **next close** (§11).
- **Config bounds** are validated on construction. The auction share has **one source** (`CostConfig.auction_volume_share_default`) and a date-aware calendar that includes the 11 December 2026 Russell reconstitution. Fees and ADV use **as-traded** prices, volatility uses adjusted prices (§3, §10, §12).
</callout>

## 1. Why costs decide whether an alpha is real

### What it is
A *backtest* replays a strategy on history. A *gross* backtest ignores trading costs, while a *net* backtest subtracts them. An *anomaly* (or *factor*) is a stock characteristic that predicts returns, such as cheapness or recent winners. A *long-short* portfolio buys the stocks the anomaly likes and shorts the ones it dislikes. The gap between gross and net is not a rounding error. For most published anomalies it is the whole return.

### Why we need it here
The most careful study of the question, Chen and Velikov (2023), takes 204 published anomalies. Their long-short portfolios earn an average of **68 bp per month gross in-sample**. After three adjustments the expected return falls to **4 bp per month**: post-publication decay (about 50%), restricting to the modern post-2005 trading era (decay rises to 72%), and subtracting effective bid-ask spreads with cost-mitigation techniques applied (decay reaches **93%**). The strongest anomalies net at most about 10 bp per month after controlling for data mining. These are out-of-sample, post-publication numbers net of spreads, and they still **exclude price impact**, so they are an upper bound for a real trader. BUILD_PLAN §1 draws the consequence for this system: alpha has to come from combining signals, implementing them cost-aware and matching horizons. The net Sharpe target is 0.5–1.0, not the 1.5–2 that gross backtests suggest.

The system therefore treats cost as a first-class model, priced inside the optimizer for every trade decision (→ see **Portfolio construction › Objective**). It is not a haircut applied at the end.

### How it works: breakeven cost and the 2× gate
The cleanest single number for cost sensitivity is the **breakeven cost**: the cost per side at which the strategy's mean gross return would be exactly zero.

$$
c_{\text{BE}} = \frac{\overline{g}}{\overline{\tau}} \times 10^4
$$

where:
- $\overline{g}$ is the mean daily gross portfolio return (fraction of NAV);
- $\overline{\tau}$ is the mean daily **two-sided** turnover, $\tau_t = \sum_i \lvert x_{i,t}\rvert$, where $x_{i,t}$ is the change in weight of name $i$ on day $t$ (fraction of NAV). Each unit of $\tau$ is one side of a trade, so a per-side cost of $c$ costs $c\,\tau_t$ per day;
- $c_{\text{BE}}$ is in **basis points of traded notional per side** (1 bp = 0.01%).

The validation layer then requires breakeven to be at least **twice** the cost the model actually charged per side (`qts/validation/hurdles.py`, gate `breakeven_cost`, `BREAKEVEN_MULTIPLE = 2`). The factor of two is a safety margin. Impact models have very low explanatory power (R² under 1% in Citigroup's data, about 7% in AQR's), the calibrated `b` can easily be off by 50%, and the live haircut on backtests is large (BUILD_PLAN §1 cites a median 73% Sharpe deterioration for 215 bank indices). A strategy that dies if costs are 2× the model is not robust enough to trade.

### In the code
- `qts/backtest/metrics.py::breakeven_cost_bps(gross_returns, turnover_series)` implements the formula.
- **Which return series.** $\overline g$ must already carry every cost that is *not* a per-side trading cost. Borrow is a holding cost and does not scale with turnover, so starting from plain gross returns leaves it out of the gate entirely. The second-pass audit (finding 1) measured breakeven 26.2 bp from gross against 24.3 bp net of borrow on a real run. The simulator therefore records a third return column, `gross_after_borrow` = gross − (that day's borrow / pre-trade NAV), and `BacktestResult.summary()["breakeven_cost_bps"]`, `BacktestResult.cost_sensitivity` and the pipeline's gate all start from it (→ see **Backtesting › Metrics**).
- Turnover is aligned *positionally* to return dates (`BacktestResult.turnover_on_return_dates()`). An earlier date-join shifted turnover by one day and dropped the initial build; audit #4 found it and it is fixed.
- `qts/backtest/metrics.py::cost_sensitivity` re-computes net Sharpe at 0.5×, 1×, 2× and 3× a per-side cost, so you can see how quickly the strategy dies.
- Tests: `tests/test_audit2_costs_backtest.py::test_gross_after_borrow_and_breakeven_net_of_borrow`, `tests/test_backtest.py::test_breakeven_and_sensitivity`.

### Worked example
A 100/100 book (gross leverage 2.0) targets a 50-day effective holding period. Effective holding days = gross / (two-sided turnover / 2), so daily two-sided turnover is $\tau = 2 \times 2.0 / 50 = 0.08$. Suppose the gross return is 6% a year, or 0.06 / 252 = 2.38 bp a day. Then $c_{\text{BE}} = 2.38\text{ bp} / 0.08 = 29.8$ bp per side. If the cost model charged 12 bp per side, the gate needs $2 \times 12 = 24$ bp and the strategy passes, though not comfortably. Doubling turnover to $\tau = 0.16$ halves breakeven to 14.9 bp and fails the gate. **Turnover is the lever**, which is why the optimizer prices cost directly instead of trading to a target and hoping.

### References
- [Chen & Velikov (2023), Zeroing in on the expected returns of anomalies, JFQA](https://core-prod.cambridgecore.org/core/services/aop-cambridge-core/content/view/945133D5A3ECEEAF466AEE91551FD225/S0022109022000874a.pdf/zeroing-in-on-the-expected-returns-of-anomalies.pdf)
- [Almgren, Thum, Hauptmann & Li (2005), Direct estimation of equity market impact](https://www.cis.upenn.edu/~mkearns/finread/costestim.pdf)
- [Frazzini, Israel & Moskowitz (2018), Trading costs](https://conference.nber.org/conf_papers/f68262/f68262.pdf)

## 2. Taxonomy and typical magnitudes

### What it is
Every dollar lost between deciding to trade and holding the position falls into one of three buckets.

| Bucket | Component | What it is | Who gets the money | Form in the model |
|---|---|---|---|---|
| Explicit | Commission | Broker's fee per share | IBKR | linear in shares |
| Explicit | Clearing (NSCC/DTC) | Settlement utility fee | clearing house via IBKR | linear in shares |
| Explicit | SEC §31 fee | Regulatory fee on **sales** only | SEC (funds its budget) | linear in dollars sold |
| Explicit | FINRA TAF | Trading Activity Fee on **sales** only | FINRA | linear in shares sold, capped per trade |
| Explicit | CAT fee | Consolidated Audit Trail funding | CAT LLC | linear in shares |
| Explicit | Closing-auction exchange fee | NYSE/Nasdaq fee for MOC/LOC executions | exchange | linear in shares, auction only |
| Implicit | Spread | Paying the ask to buy / receiving the bid to sell | liquidity providers | linear in dollars (half-spread) |
| Implicit | Market impact | Your order moves the price against you | other traders | concave per share, total cost $\propto \lvert X\rvert^{e}$ ($e$ = 3/2 by default) |
| Implicit | Delay | Price drift between decision and order arrival | nobody (market move) | measured in TCA only |
| Implicit | Opportunity | Return missed on the part that did not fill | nobody | measured in TCA only |
| Holding | Borrow fee | Rent paid to lenders of shares you are short | securities lenders | linear in short weight, per day |

*Linear* means the cost doubles when the trade doubles. *Concave* means it grows more slowly than the trade size per share but faster in total. Section 5 explains why.

### Typical magnitudes
Numbers are one-way (per side) in bp of traded notional for a single institutional order, unless stated.

| Component | Large cap (≥ \$10B) | Mid cap (\$2–10B) | Small cap (< \$2B) | Source / status |
|---|---|---|---|---|
| Explicit fees, \$50 stock, 1,000 shares | 0.7 (buy) – 1.0 (sell) | same | same (price-driven, not size-driven) | computed in §3 from the 2026 schedule |
| Explicit fees, \$10 stock | ≈ 3.7 (buy) – 4.1 (sell) | same | same | commission is per share, so low prices hurt |
| Half-spread | 1–3 | ≈ 3–8 (rule of thumb) | ≈ 10–30 (rule of thumb) | large cap: BUILD_PLAN §4 / Frazzini et al.; mid/small: unsourced orders of magnitude, measure from your own quotes |
| Impact vs arrival (AQR live fills, 1998–2011) | 11.2 | — | 21.3 | Frazzini, Israel & Moskowitz, live institutional fills, mean |
| Buy-side cost by size (2022) | < 5% ADV: 14.6; 5–10% ADV: 24.6 | | | Bloomberg BTCA, live buy-side data |
| Buy-side cost by volatility (2022) | low vol 9.2, high vol 22.9 | | | Bloomberg BTCA |
| Borrow fee (annual, on short value) | GC ≈ 0.25–0.5%/yr | GC to a few %/yr | 5–50%/yr when hard-to-borrow | code docstring (`qts/tcost/borrow.py`); IBKR publishes no numeric schedule |

Two things stand out. First, explicit fees are small, about **2 bp round trip** on a \$50 stock (BUILD_PLAN §4; verified in `tests/test_tcost.py`). Second, spread plus impact is 5–10× larger and depends heavily on size and volatility. The engineering effort goes into the implicit part.

<callout icon="💡" color="blue_bg">
A useful mental model is buying a used car at a dealer. The commission is the documentation fee: small and printed on the invoice. The spread is the gap between what the dealer pays for cars and what they sell them for. The impact is what happens to the price when you say you want ten of them.
</callout>

### References
- [Frazzini, Israel & Moskowitz (2018), Trading costs](https://conference.nber.org/conf_papers/f68262/f68262.pdf)
- [Bloomberg (2022), Buy-side transaction cost analysis (BTCA)](https://www.bloomberg.com/professional/?p=46500)
- [IBKR (2026), Commissions — stocks](https://ndcdyn.interactivebrokers.com/en/pricing/commissions-stocks.php)

## 3. Explicit fees

### What it is
Explicit fees are what the broker and regulators charge per order. They are deterministic. Given shares, price, side, trade date and venue you can compute them to the cent, so they are the one cost component the system knows exactly.

### Why we need it here
They are small per trade but cannot be avoided, and two features make them matter. (1) Commission is **per share**, so in bp it is inversely proportional to price: a \$10 stock costs 5× more in bp than a \$50 stock. (2) There is a **\$0.35 per-order minimum**. REVIEW_NOTES records that for NAV under about \$1M spread over 300 names, minimums dominate costs. That is why the demo uses NAV = \$5M.

### How it works
For one order of $q$ shares at price $P$ with value $v = qP$:

$$
\begin{aligned}
\text{commission} &= \min\!\big(\max(c_s\, q,\; c_{\min}),\; c_{\max}\, v\big) \\
\text{clearing} &= \min(k_s\, q,\; 0.005\, v) \\
\text{SEC} &= \mathbb{1}[\text{sell}]\; r_{\text{SEC}}(d)\, v \\
\text{TAF} &= \mathbb{1}[\text{sell}]\; \min\!\big(r_{\text{TAF}}(d)\, q,\; \text{cap}_{\text{TAF}}(d)\big) \\
\text{CAT} &= r_{\text{CAT}}\, q \\
\text{exchange} &= \mathbb{1}[\text{auction}]\; r_{\text{auc}}\, q
\end{aligned}
$$

where:
- $c_s$ = `commission_per_share` (\$0.0035), $c_{\min}$ = `commission_min_per_order` (\$0.35), $c_{\max}$ = `commission_max_pct_value` (1% of value);
- $k_s$ = `clearing_per_share` (\$0.0002), capped at 0.5% of value (`CLEARING_MAX_PCT_VALUE`);
- $r_{\text{SEC}}(d)$ is the SEC Section 31 rate **per dollar sold** in force on trade date $d$;
- $r_{\text{TAF}}(d)$, $\text{cap}_{\text{TAF}}(d)$ are the FINRA TAF per share sold and the per-trade cap on date $d$;
- $r_{\text{CAT}}$ = `cat_fee_per_share` (\$0.000003), charged on both sides;
- $r_{\text{auc}}$ = `auction_exchange_fee_per_share` (\$0.001), charged only when the fill is in the closing auction;
- all results are in **dollars**; divide by $v$ and multiply by $10^4$ for bp.

**IBKR Pro tiered.** The model uses the first tier (\$0.0035/share up to 300,000 shares a month). Higher volume tiers (\$0.0020/share from 300k to 3M shares a month, per BUILD_PLAN §4) are not modelled, which is conservative. A daily auction rebalancer must use IBKR **Pro**: IBKR Lite charges for on-close orders above 10% of monthly volume (BUILD_PLAN §6).

**SEC Section 31 history.** The SEC resets this rate once or twice per fiscal year so that collections match its appropriation, so it moves a lot. The dated table `SEC_SECTION31_RATES` in `qts/tcost/fees.py`:

| Effective date | \$ per \$1M sold | bp per sell | Note in code |
|---|---|---|---|
| 2016-02-16 | 21.80 | 0.218 | rate confirmed, date unverified; used for all earlier dates |
| 2017-07-04 | 23.10 | 0.231 | FINRA IN 6/9/17 |
| 2018-05-22 | 13.00 | 0.130 | FINRA IN 5/7/18 |
| 2019-04-16 | 20.70 | 0.207 | FINRA IN 4/3/19 |
| 2020-02-18 | 22.10 | 0.221 | FINRA IN 2/3/20 |
| 2021-02-25 | 5.10 | 0.051 | FINRA IN 2/16/21 |
| 2022-05-14 | 22.90 | 0.229 | SEC press release 2022-60 |
| 2023-02-27 | 8.00 | 0.080 | FINRA IN 2/27/23 |
| 2024-05-22 | 27.80 | 0.278 | SEC fee rate advisory 2024-2 |
| 2025-05-14 | 0.00 | 0.000 | SEC fee rate advisory 2025-2 |
| 2026-04-04 | **20.60** | **0.206** | SEC fee rate advisory 2026-2 (current) |

The lookup takes the last row whose effective date is on or before the trade date. Dates before the first row use the first row's rate, never zero: the rate was never zero before FY2025. Audit #5 found that the earlier code charged zero SEC fee for every pre-2026 date. It is fixed.

**FINRA TAF.** The current rate is **\$0.000195 per share sold, capped at \$9.79 per trade** (SR-FINRA-2024-019 schedule; SR-FINRA-2026-020 keeps it through 2028). There is a **fee holiday from 2026-10-01 to 2026-12-31 inclusive** (SR-FINRA-2026-021), during which TAF is zero. November 2026 is inside the holiday. Table `FINRA_TAF_RATES`:

| Effective date | \$ per share sold | Cap per trade | Status in code |
|---|---|---|---|
| 2012-01-01 | 0.000119 | 5.95 | unverified |
| 2022-01-01 | 0.000130 | 6.49 | unverified |
| 2023-01-01 | 0.000145 | 7.27 | unverified |
| 2024-01-01 | 0.000166 | 8.30 | verified (unchanged in 2025) |
| 2026-01-01 | **0.000195** | **9.79** | verified (current) |
| 2026-10-01 → 2026-12-31 | 0 | 0 | holiday, verified |

The cap binds above \$9.79 / \$0.000195 ≈ 50,200 shares in one trade.

### Why this choice
- **Dated tables rather than a single config number.** A backtest from 2018 must charge 2018's rates. A config constant would charge today's \$20.60 in years when the true rate was \$5.10 or \$27.80. With `date=None` (unknown date) the code falls back to the config values.
- **Exchange add/remove-liquidity rebates are not modelled** in the continuous session. They are small and their sign depends on whether a child order adds or removes liquidity, which the daily model cannot know. The **closing-auction fee is modelled** because the system sends small orders, and the first slice of larger ones, to the auction, and that fee (≈ \$0.0008–0.0015/share per audit #13) is 3–4× the clearing fee.
- **Volume tiers are not modelled.** The first tier is the worst case, and modelling the monthly share count would couple days together.

### In the code
- `qts/tcost/fees.py::FeeSchedule.explicit_cost(shares, price, side, date, venue)` returns a dict `commission, clearing, sec, taf, cat, exchange, total` in dollars. `explicit_bps` and `round_trip_bps` are wrappers.
- `FeeSchedule.explicit_cost_vec(shares, price, is_sell, date, is_auction)` is the vectorised form the cost model uses: same formulas, same dated rates, same per-order minimum and caps, but one numpy pass over all names instead of one Python call per name. `test_vectorised_fees_match_scalar` checks that the two agree.
- `qts/tcost/fees.py::sec_section31_rate(date)` and `finra_taf_rate(date)` do the dated lookups. `FeeSchedule.fee_holiday(date)` returns the overrides active on a date.
- `FeeSchedule.marginal_fraction(price, side, date, venue)` (vectorised: `marginal_fraction_vec`) is the **linear coefficient per dollar traded** used by the optimizer. It ignores the \$0.35 minimum and the caps, because those make the function non-convex. The backtester uses the exact fees, so the two differ only for orders small enough to hit the minimum.
- **Routed trades are two orders.** At the routed venue (§6) the auction slice and the continuous remainder are separate orders, so `estimate` charges `explicit_cost_vec` twice: once on the auction shares with the exchange fee, once on the continuous shares without it. Each order pays its own commission minimum. A zero-share leg costs nothing.
- **Which price.** Fees are per share, so the share count must be the number of shares that actually trade that day. The snapshot's `price` is the **as-traded** close (`MarketData.trade_price()`, i.e. `close_raw` when the loader provides it), not the split-adjusted close. Shares = dollars / as-traded price (→ §10, "Raw vs adjusted prices").
- Config: every fee is a `CostConfig` field (table in §12). To update for a new SEC rate, add a row to `SEC_SECTION31_RATES` **and** update `sec_fee_per_dollar_sold`, which is used when no date is passed.


### Pitfalls & checks
- **Sells only.** SEC and TAF apply only to sales, and short sales are sales. `_norm_side` maps `short`, `sell_short` and `ss` to `sell`.
- **Fee tables go stale.** Rates change by SEC advisory with a few weeks' notice. The rows carry verification notes, and pre-2024 TAF rows are flagged unverified (code TODO dated 2026-11-04). Check the SEC fee-rate advisories page every spring.
- **Minimum-commission trap at small NAV.** At NAV \$500k a 0.1% weight change is \$500. On a \$50 stock that is 10 shares, the commission is \$0.35, and the cost is **7 bp in commission alone**. The optimizer does not see this (linear approximation), so the dust threshold in execution (→ see **Execution › From target weights to orders**) and a realistic NAV matter.
- Tests: `tests/test_tcost.py::test_fee_schedule_1000_shares_50_dollars`, `test_fee_min_and_cap`, `test_fee_holiday_zeroes_taf`, `test_sec_section31_dated_table`.

### Worked example: 1,000 shares at \$50, buy then sell
Trade value $v = 1{,}000 \times 50 = \$50{,}000$. First on **2026-06-15** (SEC \$20.60/M in force, no TAF holiday), continuous venue:

| Line | Buy | Sell | Arithmetic |
|---|---|---|---|
| Commission | \$3.500 | \$3.500 | 1,000 × 0.0035 = 3.50; above the 0.35 minimum; below the 1% cap (\$500) |
| Clearing | \$0.200 | \$0.200 | 1,000 × 0.0002 = 0.20; below 0.5% of value |
| SEC §31 | \$0.000 | \$1.030 | 50,000 × 20.60 / 1,000,000 |
| FINRA TAF | \$0.000 | \$0.195 | 1,000 × 0.000195 = 0.195; below the \$9.79 cap |
| CAT | \$0.003 | \$0.003 | 1,000 × 0.000003 |
| Exchange (auction) | \$0.000 | \$0.000 | continuous venue |
| **Total** | **\$3.703** | **\$4.928** | |
| In bp of \$50,000 | 0.741 | 0.986 | total / 50,000 × 10⁴ |

Round trip: \$3.703 + \$4.928 = \$8.631, or **1.726 bp** of one-way notional. That is the "about 2 bp" in the plan.

The same trade on **2026-11-04** falls inside the TAF holiday, so the sell is \$4.733 and the round trip is 1.687 bp. Routing both legs through the **closing auction** adds 1,000 × \$0.001 = \$1.00 per side: \$4.703 buy, \$5.928 sell, 2.13 bp round trip. The optimizer's linear coefficients for this stock are `marginal_fraction` = 0.741 bp (buy) and 0.986 bp (sell), identical to the exact numbers because nothing hits a minimum or cap.

### References
- [SEC (2026), Fee rate advisory 2026-2](https://www.sec.gov/rules-regulations/fee-rate-advisories/2026-2)
- [FINRA, Schedule A to the By-Laws — Section 1, member regulatory fees](https://www.finra.org/rules-guidance/rulebooks/corporate-organization/section-1-member-regulatory-fees)
- [IBKR (2026), Commissions — stocks](https://ndcdyn.interactivebrokers.com/en/pricing/commissions-stocks.php)

## 4. Bid-ask spread

### What it is
At any moment a stock has a best **bid** (the highest price someone will pay) and a best **ask** (the lowest price someone will sell at). The **mid** is their average, and the market's "fair" price is usually taken to be the mid. A buyer who wants to trade *now* pays the ask, which is half a spread above the mid. A seller receives the bid, half a spread below. The **half-spread** is therefore the cost of immediacy for one side of a trade.

### Why we need it here
For small orders in liquid stocks the half-spread is the largest implicit cost: 1–3 bp in large caps against a few bp of impact. It is linear (the same bp whatever the size), so it enters the optimizer as part of the $a\lvert x\rvert$ term and creates a **no-trade zone**. When a weight change is small enough that the alpha gained is less than the half-spread paid, the optimizer does not trade.

### How it works
$$
\text{spread cost}_i = h_i \,\lvert X_i\rvert, \qquad h_i = \frac{\text{ask}_i - \text{bid}_i}{2\,\text{mid}_i}
$$

where:
- $h_i$ is the half-spread of name $i$ as a **fraction** (the snapshot carries both `half_spread_bps` and `half_spread` = bps / 10⁴);
- $X_i$ is the trade in dollars.

**Auction fills pay no spread.** The closing auction is a *call auction*: all buy and sell orders are collected and crossed at **one** price, the price that maximises executed volume. No one "crosses the spread" because there is no spread at the moment of execution, and every buyer and seller in the cross gets the same print. The model therefore sets the spread leg to zero for the auction venue, and at the routed venue it charges the spread only on the continuous remainder $q_c$ (`estimate`: `spread = h * qc`; `cvx_terms`: `spread * pos(|x| - k)`, §6). The auction still costs money. Your order adds to one side of the auction's imbalance and moves the clearing price, which is impact (§5–6), and the exchange charges an auction fee (§3).

### Why this choice
- A **per-name** half-spread from quote data (`md.spread_bps`), falling back to `default_half_spread_bps` = 2 bp. A single constant would under-charge illiquid names exactly where the optimizer is most tempted to trade.
- **Quoted** rather than effective spread. Effective spread (twice the distance of the fill from the mid) is lower for patient flow and is what TCA measures. The quoted half-spread is conservative for pre-trade use.
- The Citigroup study (Almgren et al. 2005) dropped spread from its impact regression for lack of a consistent effect. The system keeps it as a separate linear term and **subtracts** it from slippage before calibrating `b` (§11), so the two do not double-count.

### In the code
- `qts/tcost/model.py::TransactionCostModel.snapshot(md, date)` reads the last row of `md.spread_bps` on or before the date (forward data never used), fills gaps with `default_half_spread_bps`, and stores the result as `LiquiditySnapshot.half_spread`, a **fraction** of price. `half_spread_bps` is a derived property (×10⁴).
- An old dict snapshot is converted by `LiquiditySnapshot.from_mapping`, which prefers `half_spread_bps`, then `half_spread`, then the default. This chain dates from audit #3, which found real spreads silently replaced by 2 bp in the simulator path.

### Pitfalls & checks
- **Half vs full spread.** `MarketData.spread_bps` is documented as a *half*-spread proxy but named like a full spread (audit #20). Feed it a quoted full spread and every linear cost doubles. That is conservative, but it distorts the no-trade zone. Check your loader.
- Spreads widen sharply at the open, in the last minutes and in stress. A daily average under-states the cost of an arrival-price order at the open.
- Test: `tests/test_tcost.py::test_estimate_prices_requested_venue_and_real_spreads` (continuous venue uses real spreads; auction venue has zero spread but a higher explicit fee).

## 5. Market impact: the square-root law

### What it is
**Market impact** is the price move your own trading causes. A buy order consumes the sell orders resting in the book and signals demand to other participants, who then raise their offers. By the time you have bought 50,000 shares, the price is higher than when you started, and your average fill price is above the arrival price. Impact has two parts:
- **Temporary impact** is the part that decays after you stop: the liquidity you consumed is replenished.
- **Permanent impact** is the part that stays, because the market has updated its view of fair value from your trading. Frazzini, Israel & Moskowitz estimate about **70% permanent** in AQR's live fills.

A **metaorder** is the whole parent order (for example "buy 50,000 shares today"). It is usually split into many small **child orders**. Impact laws describe the metaorder.

### Why we need it here
Impact is the only cost component that grows **faster than linearly** in trade size. It decides how fast the portfolio can move toward its target, how large the book can grow before alpha is eaten (capacity), and how the optimizer splits a trade over days. Without it, an optimizer with linear costs either trades nothing (inside the no-trade zone) or trades everything at once.

### How it works: the square-root law
The empirical law, confirmed across markets, decades and traders, is

$$
\frac{I}{\sigma} = Y \sqrt{\frac{Q}{V}}
$$

where:
- $I$ is the expected **average** price impact of the metaorder, as a fraction of price (return units), measured as fill VWAP versus arrival price, signed so that positive = cost;
- $\sigma$ is the stock's daily return volatility (fraction, e.g. 0.02 = 2%/day);
- $Q$ is the metaorder size (shares or dollars, consistent with $V$);
- $V$ is the volume traded in the market over the same period (shares or dollars). Daily ADV for a one-day order, or the auction's volume for an auction order (§6);
- $Q/V$ is the **participation**, your share of the volume;
- $Y$ is a dimensionless prefactor "of order one". The code calls it $b$.

A trade of 1% of daily volume in a 2%-volatility stock costs about $1 \times 0.02 \times \sqrt{0.01} = 0.2\%$ = **20 bp**. A trade of 4% costs 40 bp, not 80.

**Why concave?** Three complementary explanations from the econophysics literature (Tóth et al. 2011; Bouchaud):
1. **Latent liquidity.** Most supply and demand is not displayed in the book. It is "latent" in traders' heads and revealed only as price moves toward it. Near the current price the latent order book is approximately linear in distance (an empty V-shape). Pushing the price by $\Delta p$ therefore uncovers volume proportional to $\Delta p^2$, so to absorb $Q$ shares you need $\Delta p \propto \sqrt{Q}$.
2. **Information.** Bigger orders are more likely informed, but the market cannot tell a large informed trader from a sequence of small ones, so it updates by less than proportionally.
3. **Schedule independence.** At modest participation the impact of the metaorder depends mostly on its total size, and much less on how it is split across child orders or time. This has a big consequence for execution: scheduling buys little expected cost (→ see **Execution › Scheduling math**).

**Empirical evidence:**

| Study | Data | Finding | Evidence class |
|---|---|---|---|
| Almgren, Thum, Hauptmann & Li (2005) | Citigroup US equity orders | Temporary impact exponent **0.600 ± 0.038** on participation; volatility the main scale factor; spread dropped; R² < 1% | live fills, in-sample fit |
| Tóth, Lempérière, Deremble et al. (2011) | CFM futures metaorders | $I \propto \sigma\sqrt{Q/V}$ over several decades of $Q/V$; latent-liquidity explanation | live fills |
| Frazzini, Israel & Moskowitz (2018) | AQR, 5.3M orders, \$721B, 1998–2011 | Mean impact vs arrival 12.18 bp (large 11.21, small 21.27); ~70% permanent; 3.7 bp vs VWAP; R² ≈ 7% | live fills |
| Sato & Kanazawa (2024) | All liquid Tokyo stocks, 8 years, account-resolved | Exponent 1/2 within error per stock and per trader | live, exchange data |
| Vasaikar (2026) | AAPL metaorders reconstructed from Nasdaq ITCH, Dec 2024–Aug 2025 | $c_{\text{raw}} = 0.69$, $c_{\text{eff}} = 0.34$ after bias correction; square root beats linear (ΔAIC = 22); prefactor stable over 32 weekly walk-forward recalibrations | reconstructed, out-of-sample walk-forward |
| Bloomberg BTCA (2022) | buy-side TCA | Biggest cost step from < 5% ADV (14.6 bp) to 5–10% ADV (24.6 bp) | live |

BUILD_PLAN §4 takes $b \in [0.3, 1.0]$, starting at **$b = 1.0$ for the continuous session** until own fills calibrate it.

### How it works: from impact per share to total cost
The optimizer needs the **dollar cost of the whole trade**, not the impact in bp.

$$
C(X) = a\,\lvert X\rvert \;+\; b\,\sigma\,\frac{\lvert X\rvert^{e}}{V^{\,e-1}}
$$

where:
- $X$ is the trade in dollars, $V$ the venue's volume in dollars per day;
- $a$ is the proportional cost per dollar: half-spread plus linear explicit fees;
- $b, \sigma$ as above;
- $e$ is `CostConfig.impact_exponent`, **1.5 by default**, which is the square-root law: $\lvert X\rvert^{3/2}/\sqrt{V}$. The average impact per dollar is $C/\lvert X\rvert - a = b\,\sigma\,(\lvert X\rvert/V)^{e-1}$, so $e - 1$ is the exponent on participation (0.5 for the square root, 0.6 for Almgren et al.'s fit);
- $C$ is in **dollars**.

This is cvxportfolio's form (with exponent 1.5 and default $b = 1$) and MOSEK's power-cone example. The rest of this page writes the formulas with general $e$ and the numbers with $e$ = 1.5.

<toggle heading="h3">
Derivation: cost = trade size × average impact

The square-root law, as measured in TCA, is a statement about the **average** slippage of the metaorder relative to the arrival price: every share of the order is filled on average $I = b\,\sigma\sqrt{\lvert X\rvert / V}$ (as a fraction of price) worse than arrival. The dollar cost is the dollars traded times the average fractional slippage:

$$
C_{\text{impact}}(X) = \lvert X\rvert \cdot I = \lvert X\rvert \cdot b\,\sigma\sqrt{\frac{\lvert X\rvert}{V}} = b\,\sigma\,\frac{\lvert X\rvert^{3/2}}{\sqrt{V}}
$$

Adding the linear cost $a\lvert X\rvert$ gives $C(X)$. Dividing by $\lvert X\rvert$ recovers the cost per dollar, $a + b\sigma\sqrt{\lvert X\rvert/V}$, which is exactly the point estimate `cost_bps` returns. `total_cost_fraction` and `cost_bps` are the same model; `tests/test_tcost.py::test_impact_sqrt_scaling_and_auction_b` checks it.

A different convention starts from the **marginal** (instantaneous) impact when $q$ of the $Q$ dollars have been executed, $I_{\text{marg}}(q) = c\,\sigma\sqrt{q/V}$, and integrates: $\int_0^{Q} c\,\sigma\sqrt{q/V}\,dq = \tfrac{2}{3}c\,\sigma Q^{3/2}/\sqrt{V}$. Same 3/2 power, with $b = \tfrac{2}{3}c$. The system calibrates $b$ on average slippage (VWAP − arrival), so it uses the first convention, and no 2/3 factor appears anywhere. When you compare $b$ with a paper, check which one the paper reports.

Horizon invariance: over a $k$-day metaorder the relevant volatility is $\sigma\sqrt{k}$ and volume is $kV$, and $\sigma\sqrt{k}/\sqrt{kV} = \sigma/\sqrt{V}$. So the formula with **daily** $\sigma$ and **daily** $V$ prices a multi-day metaorder of total size $Q$ correctly, which is what §7 relies on.
</toggle>

### How it works: units for the optimizer
The optimizer's decision variable is not dollars but the **weight change** $x_i = w^{\text{post}}_i - w^{\text{pre}}_i$, a fraction of NAV. Its objective is also in fractions of NAV (return units). Substituting $X = x \cdot \text{nav}$ and dividing by nav:

$$
\frac{C(x\,\text{nav})}{\text{nav}} = a\,\lvert x\rvert + b\,\sigma\,\frac{\lvert x\rvert^{e}}{(V/\text{nav})^{e-1}}
$$

<toggle heading="h3">
Derivation of the (V/nav)^(e−1) term

$$
\frac{b\,\sigma\,\lvert x\,\text{nav}\rvert^{e}}{V^{e-1}\;\text{nav}} = b\,\sigma\,\lvert x\rvert^{e}\,\frac{\text{nav}^{e-1}}{V^{e-1}} = b\,\sigma\,\frac{\lvert x\rvert^{e}}{(V/\text{nav})^{e-1}}
$$

At $e$ = 1.5 the denominator is $\sqrt{V/\text{nav}}$. $V/\text{nav}$ is the stock's daily volume measured in units of your NAV. A \$5M fund trading a stock with \$50M ADV has $V/\text{nav} = 10$. The bigger the fund relative to the stock's volume, the smaller $V/\text{nav}$ and the larger the impact coefficient: **capacity is built into the cost function.** Doubling NAV multiplies the impact cost per unit weight by $2^{e-1}$ ($\sqrt{2}$ at the default).
</toggle>

Per name the coefficient is `coef_i = b * sigma_i / (V_i / nav)^(e-1)`, computed once per rebalance by `qts/tcost/model.py::TransactionCostModel.cvx_coefficients` and assembled by `cvx_terms`:

```python
# cvx_coefficients
coef = float(b) * sig / (vol / nav) ** (e - 1.0)
# cvx_terms
ax = cp.abs(x)
if c["venue"] == VENUE_ROUTED:
    spread_term = cp.sum(cp.multiply(c["spread"], cp.pos(ax - c["spread_kink"])))
else:
    spread_term = cp.sum(cp.multiply(c["spread"], ax))
pw = cp.power(ax, c["exponent"])
trade = spread_term + cp.sum(cp.multiply(c["impact"], pw))
```

Linear fees enter as `fee_buy · pos(x) + fee_sell · neg(x)`, with different coefficients on buys and sells because SEC and TAF apply only to sells. `cvx_coefficients` returns the numpy arrays separately so the optimizer can turn them into cvxpy parameters and re-solve one compiled problem every day instead of rebuilding it (→ see **Portfolio construction › Solver**).

### How it works: convexity and DCP
The optimizer is solved with cvxpy and Clarabel (→ see **Portfolio construction › Solver**). cvxpy accepts only problems it can **prove** convex using *disciplined convex programming* (DCP), a small set of composition rules. The cost expression passes:
- $\lvert x\rvert$ is convex and non-negative;
- $t \mapsto t^{e}$ is convex and **increasing** for $t \ge 0$ whenever $e \ge 1$, and an increasing convex function of a convex function is convex, so $\lvert x\rvert^{e}$ is convex. This is one reason the config rejects $e \le 1$;
- `pos(x)` and `neg(x)` are convex; non-negative weighted sums of convex functions are convex;
- the routed venue's kink terms `pos(|x| - k)` and `pos(|x|^e - k^e)` are `pos` (convex, non-decreasing) of a convex function minus a constant, so they are convex too (§6).

Clarabel handles `power(·, e)` through the **power cone**. The optimizer asks cvxpy for the exact power-cone atom (one 3-d cone per name) instead of cvxpy's default rational second-order-cone approximation. Convexity guarantees a single global optimum, found reliably, and matches the economics: the marginal cost of trading rises with size. A concave cost such as a fixed per-order fee would make the problem combinatorial, which is why the \$0.35 minimum stays out of the optimizer.

### Why this choice
- **Square root over linear impact.** Linear impact ($I \propto Q$, the Almgren–Chriss default) gives quadratic total cost. It overstates the cost of big trades and understates small ones, and the evidence above rejects it (ΔAIC = 22 in favour of the square root in Vasaikar 2026).
- **Exponent configurable, defaulting to 1.5, and not fitted.** Almgren 2005 found 0.6 on participation ($e$ = 1.6) and Sato & Kanazawa 0.5 ($e$ = 1.5). With R² below 7%, own-fill data cannot distinguish them, so the prefactor $b$ is the only parameter calibrated from fills. `impact_exponent` exists so a sensitivity run at, say, $e$ = 1.6 is a config change rather than a code change. Every consumer reads it: the cost model's estimate and convex terms, `SquareRootImpact`, the router's LOC offset and the TCA regressor. Before the November wave the field existed but 1.5 was hard-coded everywhere. **Changing $e$ changes the meaning of $b$** (units of $b$ depend on $e$), so a $b$ calibrated at one exponent must be recalibrated at another.
- **Propagator models** (transient impact with power-law decay; cross-impact between stocks) slightly beat parametric forms on CFM metaorder data (Hey, Neuman & Tuschmann 2025) but need tens of thousands of metaorders. BUILD_PLAN §4: ML effort is better spent forecasting $\sigma$ and $V$.
- **No credit for reversion.** Temporary impact does revert, but the plan forbids modelling next-day reversion as a benefit (§11).

### In the code
- `qts/tcost/impact.py::SquareRootImpact` (the name is historical; it uses the configured exponent)
  - `cost_bps(notional, adv_usd, sigma_daily, b, venue)` gives $10^4\, b\,\sigma\,(\lvert Q\rvert/V)^{e-1}$, vectorised;
  - `total_cost_fraction(...)` gives the integrated cost in **dollars**, despite the name, which is kept for API compatibility;
  - `cvx_cost_expression(x, nav, ...)` is a stand-alone cvxpy form for one venue (the optimizer uses `TransactionCostModel.cvx_terms` instead);
  - `participation_cap(adv_usd, nav)` gives `max_participation_adv × ADV / nav`, the per-name bound on $\lvert x\rvert$ (default 5% of ADV);
  - `validate_impact_exponent(e)` raises unless $1 < e \le 2$.
- `qts/tcost/model.py::TransactionCostModel` is the production `CostModel` (§10). `estimate(trades_usd, snap, venue, streak, date, prior_streak_usd)` returns a `CostEstimate` (USD per ticker). `cvx_terms(x, w_post, nav, snap, venue)` returns `(trade_cost_expr, holding_cost_expr)` in fractions of NAV. Internally both go through `_impact_usd(q, V, sigma, b)` = $b\,\sigma\,q^e/V^{e-1}$.
- Snapshot inputs, all point-in-time: `adv_usd` = 63-day mean dollar volume at as-traded prices, `sigma_daily` = 63-day standard deviation of daily simple returns of the adjusted close. Missing sigma (new listings) is filled with the cross-sectional median by the `LiquiditySnapshot` itself, never with 0, which would make impact free (audit #10).
- Tuning: `CostConfig.impact_b_continuous` (1.0), `impact_b_auction` (0.3), `impact_exponent` (1.5), `max_participation_adv` (0.05).
- Tests: `test_impact_sqrt_scaling_and_auction_b`, `test_impact_exponent_is_wired_and_validated`, `tests/test_cost_equivalence.py::test_cvx_terms_equal_estimate_non_default_exponent_and_heavy_day`.

### Pitfalls & checks
- **σ in the wrong units.** σ must be **daily**. Feeding annual volatility (≈ 16× larger) multiplies impact by 16.
- **V must match the venue** (§6). Full-day ADV for an auction order under-states auction impact by $1/\sqrt{0.09} = 3.3$×.
- **Outside the validated range.** The law is validated up to roughly 10% participation. The 5% ADV cap keeps the optimizer inside it.
- **Optimizer ⇄ simulator equivalence.** See §13.

### Worked example
A \$500,000 buy, stock ADV \$50M, σ = 2%/day, NAV \$5M, so $x = 0.1$:
- Participation $Q/V = 0.01$; continuous impact = $1.0 \times 0.02 \times \sqrt{0.01} = 0.0020$ = **20 bp** = **\$1,000**.
- Optimizer units: coef = $1.0 \times 0.02 / \sqrt{50/5} = 0.006325$; $\lvert x\rvert^{3/2} = 0.1^{1.5} = 0.03162$; product = 0.000200 of NAV = \$1,000. ✓
- Double the trade to \$1M: impact 28.3 bp (×√2), dollar cost \$2,828 (×2.83).

### References
- [Tóth, Lempérière, Deremble, de Lataillade, Kockelkoren & Bouchaud (2011), Anomalous price impact and the critical nature of liquidity in financial markets](https://arxiv.org/abs/1105.1694)
- [Bouchaud, The square-root law of market impact](https://bouchaud.substack.com/p/the-square-root-law-of-market-impact)
- [Sato & Kanazawa (2024), square-root impact law tested on account-resolved Tokyo Stock Exchange data](https://arxiv.org/abs/2411.13965v2)
- [Vasaikar (2026), square-root impact on reconstructed AAPL metaorders](https://arxiv.org/abs/2606.24019)
- [cvxportfolio, Transaction and holding costs](https://www.cvxportfolio.com/en/stable/costs.html)
- [MOSEK Portfolio Optimization Cookbook, Transaction costs](https://docs.mosek.com/portfolio-cookbook/transaction.html)
- [Hey, Neuman & Tuschmann (2025), propagator and cross-impact models on metaorder data](https://arxiv.org/abs/2510.06879)

## 6. Venue: continuous session, closing auction, and the routed split

### What it is
US stocks trade in two ways each day. In the **continuous session** (09:30–16:00 ET) orders match one at a time against the limit order book. In the **closing auction**, all market-on-close (MOC) and limit-on-close (LOC) orders are collected and crossed at a single price at 16:00. That price is the official close: the price index funds, benchmarks and NAVs use. Mechanics are in → see **Execution › US market microstructure primer**. Here the question is how to price each venue, and how to price a trade that uses both.

### Why we need it here
The system is a **daily rebalancer** whose backtest trades at the close. The closing auction has a very different volume from the whole day, and so a different $V$ in the impact law. The router (→ see **Execution › The routing table**) sends the first slice of each order to the auction and anything above a cap to an in-session algorithm. If the cost model priced every trade as if it went entirely to one venue, the backtest would charge a cost the live system never pays. The cost model therefore knows three venues:

| Venue | Meaning | Who uses it |
|---|---|---|
| `continuous` | whole trade by an in-session algo: $V$ = ADV, pays the half-spread | delisting liquidations; research runs |
| `auction` | whole trade in the closing auction: $V$ = auction share × ADV, no spread | research runs (the pre-November default) |
| `routed` | auction up to the router's cap, continuous remainder | **default** of `TransactionCostModel`, `DailyBarSimulator.run`, `ResearchPipeline` and `SinglePeriodOptimizer.solve` (design decision D3) |

### How it works: auction volume share
The share of a day's volume that trades in the closing auction (BMLL, US equities):

| Day type | Share of daily volume in the close | Code |
|---|---|---|
| Normal day | **≈ 9%** (≈ \$50B notional) | `CostConfig.auction_volume_share_default` = 0.09 |
| Quarterly index rebalance / options expiry ("quad witching": 3rd Friday of Mar/Jun/Sep/Dec) | **≈ 20%** | 0.20 |
| Russell reconstitution day (last Friday of June) | **≈ 34%** | 0.34 |
| Russell second semi-annual reconstitution (from 2026): **Friday 11 December 2026** | placeholder 34% | `RUSSELL_RECON_DATES` |

`qts/core/costs.py::auction_share_for_date(date, share_default)` implements the calendar. Russell dates are checked first, then the June last-Friday rule, then the quad-witching rule (a Friday on day 15–21 of a quarter-end month); every other day gets `share_default`. The heavy-day figures are **floors**: `max(share_default, 0.34)`, so a configured normal-day share above them still wins.

FTSE Russell moved the Russell US indexes to **semi-annual reconstitution from 2026**. The December date is published each year (2026: rank day 30 October, effective after the close on Friday 11 December, per FTSE Russell's 1 September 2026 announcement) and **must be added by hand** to `RUSSELL_RECON_DATES` every year. The 34% for December is a placeholder equal to June's figure. 2026-12-11 is a Friday *before* the third Friday (12-18, quad witching, 20%), so December 2026 has two heavy closes a week apart.

**One source for the normal-day share.** Before November the 0.09 constant lived in five places, and the config field was not read. Now `CostConfig.auction_volume_share_default` is **the** source, read by the cost model (`SquareRootImpact.auction_share_for_date`), the router, the fill simulator, the dry-run broker and the TCA ledger. `ExecutionConfig` no longer has a copy. The module constant `AUCTION_SHARE_NORMAL_DAY` = 0.09 is only a fallback when no config is at hand (`tests/test_tcost.py::test_auction_share_single_source`).

**Which date's share.** The share is a property of the **execution** date. The calendar is published in advance, so the optimizer is allowed to know that tomorrow is Russell day. With a signal lag (→ see **Backtesting › The simulator timeline**) the simulator builds the optimizer's snapshot from the signal date's liquidity but stamps it with the execution date's `auction_share` (`DailyBarSimulator._optimizer_snapshot`; second-pass audit finding 14; `test_optimizer_snapshot_uses_execution_date_auction_share`). Pricing a heavy day as a normal day would only make the optimizer under-use a cheap auction.

### How it works: the auction prefactor (design decision D2)
For an auction fill the cost model uses $V$ = auction volume:

$$
I_{\text{auc}} = b_{\text{auc}}\,\sigma\Big(\frac{Q}{\phi \cdot \text{ADV}}\Big)^{e-1} = \underbrace{\frac{b_{\text{auc}}}{\phi^{\,e-1}}}_{\beta_a}\;\sigma\Big(\frac{Q}{\text{ADV}}\Big)^{e-1}
$$

where:
- $\phi$ is the auction share of daily volume on the execution date (0.09 on a normal day);
- $b_{\text{auc}}$ = `impact_b_auction` = **0.3**;
- $\beta_a = b_{\text{auc}}/\phi^{e-1}$ is the **effective** auction prefactor per $(Q/\text{ADV})^{e-1}$, `TransactionCostModel.effective_auction_b(phi)`. On a normal day at $e$ = 1.5: $0.3/\sqrt{0.09} = 0.3/0.3 = \mathbf{1.0}$, exactly the continuous prefactor $b_c$ = `impact_b_continuous` = 1.0. On quad witching $\beta_a = 0.3/\sqrt{0.20} = 0.67$; on Russell day $0.3/\sqrt{0.34} = 0.51$.

**The reasoning.** An auction fill and a continuous fill of the same size now cost the **same impact** on a normal day, and the auction **saves the half-spread**, because a call auction has no spread to cross. On heavy days the auction is cheaper still. That is exactly the premise of the routing table: send what the auction can absorb to the close. It also matches the qualitative evidence in BUILD_PLAN §4: BMLL reports stable auction prices and no outsized deviation at the next open. There is no published US auction prefactor, so 0.3 is a **prior**, chosen so the cost model and the router agree, and the TCA loop (§11) exists to replace it with a number from own fills.

<callout icon="🤔" color="purple_bg">
**Why not the old 0.5 (or the review's 1.5)?** The senior-quant review first proposed $b_{\text{auc}} = 1.5$. With $V$ = auction volume that is an effective 1.5/0.3 = 5.0 per $\sqrt{Q/\text{ADV}}$: every auction order would cost 5× an in-session algo. The REVIEW_NOTES amendment cut it to 0.5, but that still gave an effective **1.67**. Under 0.5 the auction was the *more expensive* venue for every order above roughly 0.02–0.05% of ADV, which contradicted the router sending orders up to 0.18% of ADV to the close; the previous version of this page carried that as a "known tension" callout. Design decision D2 (`docs/notes/doc_review_findings.md`) resolved it with 0.3. On the golden test market the change cut impact cost by about a third (cost per side 10.7 → 7.2 bp, `docs/notes/golden_changes_2026-11-05.md`, step S5) and let the optimizer trade more (turnover 10.0% → 11.4% a day). That is in-sample, synthetic, 60 days, and only shows the size of the lever, not which value is right.
</callout>

### How it works: venue choice
`SquareRootImpact.cost_for_venue_choice(notional, adv_usd, sigma_daily, half_spread_bps, date, price=..., auction_max_pct=None)` compares the per-dollar costs (bp of $Q$) of three ways to execute:

$$
\text{cont} = h + b_c\,\sigma\,(Q/\text{ADV})^{e-1}, \qquad \text{auc} = f_a + \beta_a\,\sigma\,(Q/\text{ADV})^{e-1},
$$

and **routed**, the router's split (auction up to $q_a = \min(Q, \kappa\phi A)$, the rest continuous, charged the metaorder increment exactly as the routed venue below). $f_a$ is the closing-auction exchange fee, `auction_exchange_fee_per_share / price` (\$0.001/share = 0.2 bp on a \$50 stock); it needs `price`, and without it the fee is left out and the result says `fees_included = False`. Commissions and regulatory fees are the same per share on both venues and do not change the choice. It returns all three costs, `venue` (the cheapest) and `venue_within_cap` (the cheapest the router may use: the pure auction only when the whole order fits under the $\kappa$ = 2% cap, otherwise routed or continuous).

With D2 the impact terms are equal on a normal day, so the uncapped auction wins by $h - f_a$ at **every** size. At 0.18% of ADV (σ = 2%, $h$ = 2 bp, \$50 stock): continuous 10.49 bp, auction 8.69 bp. On Russell day at 1% of ADV: 22.0 bp against 10.5 bp. Only when the fee exceeds the half-spread (a cheap stock with a tight spread: 2 bp of fee at \$5) does the continuous session win. "Cheaper per dollar at every size" does **not** mean "send everything to the auction": the auction is only 9% of the day's volume, so a large order would be a large share of it. That is what `venue_within_cap` and the routed option capture: a \$1M order in a \$50M-ADV name is 11× the cap, and the routed split costs 30.1 bp against 30.3 bp continuous.

### How it works: the routed venue (design decision D3)
The router sends up to `ExecutionConfig.auction_max_pct_of_auction_volume` (κ = 2%) of the expected auction volume to the close and the rest to an algo (→ see **Execution › The routing table**). The routed venue prices a trade the same way. Per name, with $q = \lvert\text{trade}\rvert$ in dollars, $A$ = `adv_usd`, $h$ the half-spread (fraction):

$$
K = \kappa\,\phi\,A, \qquad q_a = \min(q, K), \qquad q_c = q - q_a
$$

$$
\begin{aligned}
\text{spread} &= h\,q_c \\
\text{impact} &= C_{\phi A,\,b_a}(q_a) + \big[\,C_{A,\,b_c}(q) - C_{A,\,b_c}(q_a)\,\big] \\
\text{fees} &= \text{fees}(q_a \text{ shares, auction}) + \text{fees}(q_c \text{ shares, continuous})
\end{aligned}
$$

where:
- $C_{V,b}(Q) = b\,\sigma\,Q^{e}/V^{e-1}$ is the integrated impact of a $Q$-dollar metaorder against volume $V$ (§5);
- $K$ is the **auction cap** in dollars, `TransactionCostModel.auction_cap_usd(snap)`; at a 9% share it is 0.18% of ADV;
- the fees are **two orders**: the auction slice pays the auction exchange fee, and each slice pays its own commission minimum (§3);
- `CostEstimate.auction_fraction` records $q_a/q$ per name.

**Why the continuous leg is an increment, not $C(q_c)$.** Both slices are one day's metaorder in the same name and push the same price. The square-root law describes the *whole* metaorder: the impact of a child order depends on what has already been done. $C$ is convex with $C(0) = 0$, hence **superadditive**: $C(q_a) + C(q_c) < C(q)$. Pricing $q_c$ as an independent order would therefore make splitting look artificially cheap. The model treats the auction fill as the first $q_a$ dollars of the metaorder and charges the continuous remainder the *marginal* cost of extending it from $q_a$ to $q$. This is the same argument as multi-day persistence (§7). The second-pass audit notes that this ordering (auction first) is the conservative one: the other ordering would be cheaper by $(b_c - \beta_a)(q^e - q_a^e - q_c^e) \ge 0$ (dollar units after the common factor).

**What D2 implies.** Under the D2 prior on a normal day $\beta_a = b_c$, so the auction leg equals $C_{A,b_c}(q_a)$ and the routed impact collapses to $C_{A,b_c}(q)$: the same impact as sending everything to the algo. The router saves only the half-spread on $q_a$, and pays the auction exchange fee on it. **It never saves impact on a normal day.** On heavy days $\beta_a < b_c$ and the auction slice is cheaper.

### How it works: the convex surrogate the optimizer sees
The optimizer needs the routed cost as a convex cvxpy expression of the weight change $x$ (fraction of NAV). Write $k = K/\text{nav}$ and $c = \sigma/(A/\text{nav})^{e-1}$. Dividing the routed impact by nav:

$$
\frac{\text{impact}}{\text{nav}} = c\Big[\beta_a \min(\lvert x\rvert,k)^e + b_c\big(\lvert x\rvert^e - \min(\lvert x\rvert,k)^e\big)\Big] = c\Big[\beta_a\lvert x\rvert^e + (b_c - \beta_a)\big(\lvert x\rvert^e - \min(\lvert x\rvert,k)^e\big)\Big]
$$

and $\lvert x\rvert^e - \min(\lvert x\rvert,k)^e = \text{pos}(\lvert x\rvert^e - k^e)$, likewise $\lvert x\rvert - \min(\lvert x\rvert,k) = \text{pos}(\lvert x\rvert - k)$. So `cvx_terms(venue="routed")` builds

$$
h\,\text{pos}(\lvert x\rvert - k) \;+\; c\Big[\beta_a\,\lvert x\rvert^{e} + (b_c - \beta_a)\,\text{pos}\big(\lvert x\rvert^{e} - k^{e}\big)\Big] \;+\; (f_c + f_{\text{auc}})^{\text{buy}}\,\text{pos}(x) + (f_c + f_{\text{auc}})^{\text{sell}}\,\text{neg}(x)
$$

where:
- $h\,\text{pos}(\lvert x\rvert - k)$ is the spread on the continuous part only;
- $f_c$ is the linear continuous fee per dollar (`marginal_fraction`) and $f_{\text{auc}}$ the auction exchange fee per dollar, charged on the whole trade;
- `cvx_coefficients` returns $c\,\beta_a$ as `impact`, $c\,(b_c - \beta_a)$ as `impact_kink` and $k$ as `spread_kink`.

**Validity condition: $\beta_a \le b_c$.** `pos` is convex and non-decreasing, and $\lvert x\rvert^e - k^e$ is convex, so $\text{pos}(\lvert x\rvert^e - k^e)$ is convex. With a **non-negative** coefficient $(b_c - \beta_a)$ the whole expression is convex, and the spread and impact terms are **exact**: they equal `estimate` to rounding. That holds at the defaults on every day: $\beta_a$ = 1.0 = $b_c$ on a normal day (the kink term vanishes) and $\beta_a < b_c$ on heavy days. If $\beta_a > b_c$ (for example `impact_b_auction` = 0.5, or after a TCA update pushes $b_{\text{auc}}$ up), the coefficient would be negative and the problem non-convex. The code then drops the kink and prices the **whole** trade at $\beta_a$, an upper bound. The second-pass audit (finding 7) measured that bound at up to +66% per name and 1.58× in total with $b_{\text{auc}}$ = 0.5: the optimizer would under-trade, though the backtest's booked cost (from `estimate`) would stay exact.

The only other approximation is in fees: the auction exchange fee is charged on the whole trade (exact: on $q_a$ only) and the per-order minimum (possibly twice, two orders) and caps are ignored, as for every venue. The audit measured surrogate/estimate totals of 1.0013–1.0023, all from fees.

<callout icon="💡" color="blue_bg">
**Why the kink is allowed.** A "kink" at $k$ means the marginal cost jumps when the trade outgrows the auction cap: below $k$ you pay auction impact and no spread, above it you also pay the spread and the continuous impact rate. A cost whose marginal *rises* at a kink is still convex (think of a tax bracket). A cost whose marginal *falls* would not be, which is the $\beta_a > b_c$ case. Convexity is exactly "the next dollar never costs less than the previous one".
</callout>

The optimizer caches one compiled problem and re-solves it with new coefficients each day (disciplined parametrised programming, DPP). A term `C * pos(A - K)` with both `C` and `K` as parameters is not DPP, so the optimizer rewrites it as `pos(C * A - C K)` with `C K` a parameter of its own (`qts/portfolio/optimizer/dpp.py::_Parametrizer`; second-pass risk/portfolio finding 6).

### Why this choice
- **Auction volume as V.** BUILD_PLAN §4 is explicit: use closing-auction volume, not full-day ADV. Full ADV would claim the auction can absorb 11× more than it does.
- **Date-aware share.** Heavy closes (quad witching, Russell) can absorb 2–4× more size at the same impact. Ignoring them would over-charge exactly the days on which a rebalancer can trade cheaply.
- **Routed as the default.** Pricing every trade at the auction (the pre-November default) charged large trades auction impact on their full size at an effective prefactor of 1.67. Pricing at the routed venue matches what the live system does. On the golden market the switch moved costs by less than 0.2%, because almost every trade fits under the cap (golden note, last step).
- **A forecast model of auction volume** (per name, using imbalance history) is on the roadmap. The flat 9%/20%/34% shares are cross-sectional averages.

### In the code
- `qts/core/costs.py::auction_share_for_date(date, share_default)` and `RUSSELL_RECON_DATES`; `SquareRootImpact.auction_share_for_date(date)` passes `cfg.auction_volume_share_default`; `auction_volume_estimate(adv_shares, date)`.
- `TransactionCostModel.snapshot` stores the date's share on `LiquiditySnapshot.auction_share`; `estimate` and `cvx_terms` read it from there.
- `TransactionCostModel(cfg, exec_cfg, default_venue="routed")`: `exec_cfg` supplies κ. The pipeline constructs it as `TransactionCostModel(cfg.cost, cfg.execution)`. `auction_cap_usd(snap)`, `effective_auction_b(phi)`, `estimate(..., venue=)`, `cvx_coefficients`, `cvx_terms`.
- Tests: `tests/test_tcost.py::test_participation_cap_and_auction_volume` (06-26 → 0.34, 03-20 → 0.20, venue choice), `test_routed_venue_hand_computed`; `tests/test_cost_equivalence.py::test_routed_spread_and_impact_exact_when_beta_a_le_b_c`, `test_routed_surrogate_is_convex_and_solves`, `test_routed_small_trades_cheaper_and_surrogate_error_bounded`; `tests/test_execution.py::test_router_split_matches_routed_cost_model`.

### Pitfalls & checks
- **Russell calendar.** The June rule ("last Friday of June") approximates the official calendar. December dates must be added yearly, or a December reconstitution is priced as a normal day (or a quad-witching day if it happens to fall on the 15th–21st).
- **Raising $b_{\text{auc}}$ above $b_c\,\phi^{e-1}$** (0.3 at the defaults) switches the optimizer to the loose upper-bound surrogate. Expect lower turnover and a lower transfer coefficient. The backtest's booked costs stay exact.
- **Changing the exponent** changes $\beta_a$ too: at $e$ = 1.6 the normal-day $\beta_a = 0.3/0.09^{0.6} = 1.27 > 1$, so D2's equality no longer holds and the surrogate becomes an upper bound. Recalibrate both prefactors together.
- **The fill simulator prices splits like the model.** `FillSimulator` charges a split's algo child the increment $C(q_a + q_c) - C(q_a)$ on the day's realised volume, with $q_a$ the auction child's filled shares (→ see **Execution › The fill simulator**), so simulated fills and backtest costs use the same metaorder rule.

### Worked example
A \$500,000 buy of a \$50 stock (10,000 shares), ADV \$50M, σ = 2%/day, half-spread 2 bp, NAV \$5M ($x$ = 0.1), on a normal day (2026-06-15, $\phi$ = 0.09). Run through `TransactionCostModel.estimate`:

| Venue | Spread | Impact | Fees | Total | Optimizer surrogate × nav |
|---|---|---|---|---|---|
| continuous | \$100.00 | \$1,000.00 | \$37.03 | \$1,137.03 | \$1,137.03 |
| auction | \$0 | \$1,000.00 | \$47.03 | \$1,047.03 | \$1,047.03 |
| routed | \$82.00 | \$1,000.00 | \$38.83 | \$1,120.83 | \$1,129.03 |

- Cap $K = 0.02 \times 0.09 \times 50\text{M} = \$90{,}000$ (1,800 shares), so $q_a$ = \$90k and $q_c$ = \$410k; `auction_fraction` = 0.18.
- Spread: 2 bp × \$410k = \$82.
- Impact: auction leg $0.3 \times 0.02 \times 90{,}000^{1.5}/\sqrt{4.5\text{M}} = \$76.37$; continuous increment $C(500\text{k}) - C(90\text{k}) = 1{,}000 - 76.37 = \$923.63$; total \$1,000.00. That is exactly the continuous impact, as D2 implies.
- Fees: auction order 1,800 shares = \$6.30 commission + \$0.36 clearing + \$0.005 CAT + \$1.80 auction fee = \$8.47; continuous order 8,200 shares = \$28.70 + \$1.64 + \$0.025 = \$30.36; total \$38.83.
- Surrogate: spread and impact exact; fees \$8.20 higher because the \$0.001/share auction fee is charged on all 10,000 shares instead of 1,800.

The same order on **Russell day 2026-12-11** ($\phi$ = 0.34, $K$ = \$340k, $\beta_a$ = 0.514): continuous \$1,137.03, auction \$561.53, routed \$803.59 (spread \$32, impact \$727.76, fees \$43.83). On quad witching 2026-03-20 ($\phi$ = 0.20): routed \$1,017.75.

### References
- [BMLL, Into the close: unpacking US closing auction dynamics and the impact of the Russell reconstitution](https://www.bmlltech.com/news/market-insight/into-the-close-unpacking-u-s-closing-auction-dynamics-and-the-impact-of-the-russell-reconstitution)
- [FTSE Russell (2026), FTSE Russell announces December 2026 Russell US indexes reconstitution schedule](https://www.lseg.com/en/media-centre/press-releases/ftse-russell/2026/ftse-russell-announces-december-2026-russell-us-indexes-reconstitution-schedule)

## 7. Multi-day impact persistence

### What it is
The optimizer often moves toward a target over several days. Partial trading toward a slowly moving target is the Gârleanu–Pedersen design intent (→ see **Portfolio construction**). Five days of buying 1% of ADV is, to the market, **one metaorder** of 5% of ADV spread over a week, not five independent 1% orders. Impact from day 1 has not fully decayed by day 2, so day 2's buying starts from a higher price.

### Why we need it here
Pricing each day as an isolated metaorder **under-states** cost during multi-day builds. Audit #6 estimated the under-statement of the impact leg at 1.4–2× for 2–4-day builds. The bias is optimistic, which is the dangerous direction for a go/no-go decision.

### How it works: the exact streak increment
Treat a streak of consecutive same-direction trading days in one name as one metaorder. Let $q_j$ be the dollars traded on day $j$ of the streak and $Q_k = \sum_{j \le k} q_j$ the cumulative notional through day $k$. The metaorder's integrated cost after $k$ days is $C(Q_k) = b\,\sigma\,Q_k^{e}/V^{e-1}$, with **daily** $\sigma$ and $V$ thanks to horizon invariance (§5). The cost attributable to day $k$ is the **increment**

$$
\text{impact}_k = C(Q_k) - C(Q_{k-1}) = C(Q_{k-1} + q_k) - C(Q_{k-1})
$$

where:
- $Q_{k-1}$ is the **actual** cumulative notional of the earlier days of the streak (USD), 0 on the first day;
- $q_k$ is today's trade (USD);
- $C$ uses today's $\sigma$ and $V$ (the convention under which $b$ is calibrated).

The sum telescopes: $\sum_{j \le k}\text{impact}_j = C(Q_k)$, so over the streak the backtest charges exactly the cost of one metaorder of the total size, whatever the daily sizes were.

**State the simulator carries.** Per name, `DailyBarSimulator` keeps `streak` (consecutive same-direction trading days including today) and `streak_usd` (dollars traded in the current streak through the last trade, each day's $\lvert x\rvert \cdot \text{nav}_t$). A trade with the same sign as the previous day's continues the streak; a sign flip, or a day without a trade, resets it. The simulator passes $Q_{k-1}$ as `estimate(prior_streak_usd=)` whenever the cost model's signature accepts it (the production model does):

```python
same = (sgn != 0) & (sgn == last_sign)
streak = np.where(sgn != 0, np.where(same, streak + 1.0, 1.0), 0.0)
prior_usd = np.where(same, streak_usd, 0.0)  # Q_{k-1}: earlier days of this streak
streak_usd = np.where(sgn != 0, prior_usd + np.abs(x) * nav, 0.0)
```

**How `estimate` applies it.** It first prices today's trade at the venue as a one-day order, then adds a **streak surcharge**

$$
S = C(Q_{k-1} + q) - C(Q_{k-1}) - C(q) \;\ge\; 0
$$

($S \ge 0$ because $C$ is superadditive). For `continuous` the result is exactly $C(Q_k) - C(Q_{k-1})$. For `auction`, $C$ is $C_{\phi A,b_a}$. For `routed`, $S$ uses $C_{A,b_c}$, because the earlier days and the continuous increment share $V = A$, so the routed impact becomes

$$
\big[C_{A,b_c}(Q_k) - C_{A,b_c}(Q_{k-1})\big] + \big[C_{\phi A,b_a}(q_a) - C_{A,b_c}(q_a)\big]
$$

which is the exact streak increment plus today's auction discount on the first $q_a$ dollars (zero under D2 on a normal day, negative on heavy days). `CostEstimate.persistence_mult` reports (impact + $S$)/impact per name. Only the **impact** leg persists; spread and fees are paid afresh each day.

<toggle heading="h3">
Special case: equal daily trades and the k^e − (k−1)^e multiplier

When only the streak length $k$ is known (`estimate(streak=)` without `prior_streak_usd`, as for a legacy cost model), the code assumes the earlier days were the same size as today: $Q_{k-1} = (k-1)q_k$ and $Q_k = k\,q_k$. Then

$$
C(Q_k) - C(Q_{k-1}) = \frac{b\sigma}{V^{e-1}}\Big[(k q_k)^{e} - ((k-1)q_k)^{e}\Big] = \underbrace{\frac{b\sigma q_k^{e}}{V^{e-1}}}_{\text{one-day cost}} \cdot \underbrace{\big[k^{e} - (k-1)^{e}\big]}_{m_k}
$$

So day $k$'s impact is the one-day impact times $m_k = k^{e} - (k-1)^{e}$, and $\sum_{j=1}^{k} m_j = k^{e}$. For large $k$, $m_k \approx e\,k^{e-1}$ ($1.5\sqrt{k}$ at $e$ = 1.5). `qts/tcost/model.py::persistence_multiplier(streak, exponent)`:

| k | 1 | 2 | 3 | 4 | 5 | 10 |
|---|---|---|---|---|---|---|
| $m_k$ ($e$ = 1.5) | 1.000 | 1.828 | 2.368 | 2.804 | 3.180 | 4.623 |
| $1.5\sqrt{k}$ | 1.50 | 2.12 | 2.60 | 3.00 | 3.35 | 4.74 |

**This is not a bound.** It is exact only when the daily trades really are equal. Partial adjustment toward a target usually produces *shrinking* trades, and then $m_k$ under-states the increment; growing trades are over-stated. The second-pass audit (finding 5) measured 0.785× the exact total on a halving path, 1.256× on a growing path and 1.18× on a real backtest at $e$ = 1.5, so even the sign of the error depends on the trade path. The previous version of this page (and of the module docstring) called it an upper bound; that was wrong, and is why the simulator now tracks the actual cumulative notional.
</toggle>

**Why it is still conservative.** Both forms treat all impact within a streak as **fully permanent**: day 1's price push is still there on day $k$. In reality only part is permanent (≈70% per Frazzini et al.), and the transient part decays as a power law over hours to days. The true increment lies between the one-day cost (full decay overnight) and the full-persistence increment. The system takes the conservative end on purpose. A propagator-style model with a decay kernel is on the roadmap.

**Asymmetry with the optimizer.** `cvx_terms` gives the optimizer the **one-day marginal** cost (no surcharge), because the optimizer does not know how many more days it will keep trading in the same direction. During a multi-day build the backtest's realised cost is therefore ≥ what the optimizer expected. This is a deliberate, conservative inconsistency, documented in the module docstring of `qts/tcost/model.py`.

### Why this choice
Audit #6 listed three options: a $\sqrt{k_{\text{eff}}}$ haircut from trade-sign autocorrelation, a decaying "pending permanent impact" state, or reporting both numbers. The first fix (November wave 1) was the equal-size multiplier, which needs one integer of state per name. The second-pass audit showed its error has a path-dependent sign, and the simulator knows the trades anyway, so wave 2 added one float of state per name (`streak_usd`) and made the increment exact under the model's own assumptions (power-law impact, full persistence). The decay kernel (option two) remains the roadmap item.

### In the code
- `TransactionCostModel.estimate(trades_usd, snap, venue, streak=None, date=None, prior_streak_usd=None)`: with `prior_streak_usd` and `CostConfig.impact_persistence` = True (default) it adds the surcharge $S$; with only `streak` it multiplies by $m_k$; the multiplier is returned in column `persistence_mult`.
- `qts/backtest/simulator.py` maintains `streak`, `streak_usd` and `last_sign` per name (snippet above) and detects `prior_streak_usd` by signature inspection, so a legacy cost model still gets `streak`.
- Delisting liquidations are priced without persistence (`streak=None`).

### Pitfalls & checks
- A one-day pause resets the streak, although real impact does not reset overnight. That is mildly optimistic.
- $C(Q_k)$ uses **today's** $\sigma$ and $V$ for the whole cumulative notional. If liquidity changes during a long streak the increment is priced on the latest liquidity, not on a blend.
- Turning `impact_persistence` off makes every day a fresh metaorder. Use it only to measure how much persistence costs, never for a headline number.
- Tests: `tests/test_tcost.py::test_impact_persistence_multiplier`; `tests/test_audit2_costs_backtest.py::test_streak_increment_is_exact_for_any_path`, `test_streak_increment_routed_and_auction`, `test_simulator_books_exact_streak_increment`; `tests/test_backtest.py::test_run_with_tcost_model_venue_and_persistence`.

### Worked example: a halving build
Same stock as §6 (ADV \$50M, σ = 2%, continuous venue). The optimizer buys \$500k, then \$250k, then \$125k on three consecutive days:

| Day | $q_k$ | $Q_{k-1}$ | One-day impact $C(q_k)$ | Exact increment $C(Q_k) - C(Q_{k-1})$ | `persistence_mult` | Equal-size approximation $m_k\,C(q_k)$ |
|---|---|---|---|---|---|---|
| 1 | \$500k | 0 | \$1,000.00 | \$1,000.00 | 1.000 | \$1,000.00 |
| 2 | \$250k | \$500k | \$353.55 | \$837.12 | 2.368 | \$646.45 |
| 3 | \$125k | \$750k | \$125.00 | \$477.92 | 3.823 | \$295.97 |

Check day 2: $C(750\text{k}) = 1{,}000 \times 1.5^{1.5} = 1{,}837.12$, minus $C(500\text{k})$ = 1,000, gives \$837.12. The three-day total is $C(875\text{k}) = 1{,}000 \times 1.75^{1.5} = \$2{,}315.04$, as the telescoping sum says. The equal-size approximation charges \$1,942.42, i.e. 0.84× the exact cost: on a shrinking build it is optimistic.

## 8. Borrow costs

### What it is
To **short** a stock you sell shares you do not own. To deliver them, your broker must **borrow** them from a lender: a pension fund, an index fund or another broker's client. The mechanics:
- **Locate.** Before a short sale, Reg SHO requires the broker to have reasonable grounds to believe the shares can be borrowed. IBKR does this automatically and shows availability.
- **Fee.** The borrower pays the lender an annualised fee on the value of the shares borrowed, accrued daily. In the cash-collateral market this is expressed as a **rebate**: the short seller posts the sale proceeds as collateral, the lender pays interest on that cash at a rebate rate, and the borrow fee is the benchmark interest rate minus the rebate. A negative rebate means the borrower pays more than the interest earned.
- **GC vs HTB.** Most large caps are **general collateral (GC)**: plentiful supply, fee a fraction of a percent a year. Stocks with heavy short demand relative to lendable supply are **hard to borrow (HTB)**, with fees from 5% to over 50% a year and a risk of recall (the lender demands the shares back).

### Why we need it here
The alpha has a `short_interest` theme, and equity loan fees are themselves the most persistent single predictor in the plan's evidence (Engelberg et al. 2025). Its shorts are, almost by definition, the expensive-to-borrow names. Audit #2 (HIGH) showed the earlier code charged every short a flat 0.5%/yr GC fee. A 20%-of-book short sleeve at 10%/yr actually costs about 2% of NAV a year, roughly 0.2 Sharpe at 10% vol, which is larger than the gap between the Phase-1 (0.3) and Phase-2 (0.5) hurdles.

### How it works
Daily holding cost as a fraction of NAV:

$$
H = \sum_i \max(-w_i, 0)\,\frac{f_i}{252}
$$

where $w_i$ is the post-trade weight (shorts negative), $f_i$ the annual borrow fee (fraction), 252 trading days a year. Interest earned on short proceeds is **ignored**, which is conservative. Longs carry no holding cost here.

The fee $f_i$ is chosen **per name**, in priority order:
1. **data**: the name's latest finite `md.borrow_fee` value on or before $t$ (eventually the logged IBKR indicative rate), used as-is;
2. **proxy**: if the name has no datum but has a short-interest ratio, the short-interest proxy below;
3. **default**: otherwise the GC fee `default_borrow_fee_annual` (0.5%/yr), **times the small-cap multiplier when the name's market cap is known and below \$2B**.

Each name's source is recorded. The snapshot's `borrow_fee_source` summarises them as the sources actually used, in the order data, proxy, default, joined by `+` (for example `"proxy+default"`). `"data"` appears only if at least one name had a finite datum. Before the second-pass audit (finding 8) the label read `"data"` whenever a `borrow_fee` panel existed, even an all-NaN one, and data-poor small caps got plain 0.5% GC: exactly the names that should not get the cheapest borrow. The proxy (`borrow_fee_proxy`):

$$
u = \text{clip}\!\left(\frac{\text{SI} - 0.05}{0.20 - 0.05}, 0, 1\right), \quad u_{\text{DTC}} = \text{clip}\!\left(\frac{\text{DTC} - 2}{8}, 0, 1\right), \quad f = f_{\text{GC}} + (f_{\text{cap}} - f_{\text{GC}})\,\max(u, u_{\text{DTC}})^2
$$

then $f \leftarrow 2f$ if market cap < \$2B, where:
- SI is the **short-interest ratio**: shares sold short / shares outstanding (FINRA bi-monthly data with an 8-business-day publication lag, REVIEW_NOTES item 2);
- DTC is **days to cover**: SI shares / ADV shares, i.e. how many days of normal volume it would take shorts to buy back. DTC 2 maps to the GC end, DTC 10 to the HTB cap;
- the tighter of the two signals wins (`np.fmax` ignores a NaN on either side);
- $f_{\text{GC}}$ = `default_borrow_fee_annual` (0.5%), $f_{\text{cap}}$ = `htb_fee_cap_annual` (25%);
- the quadratic in $u$ makes the rise **convex**: slow just above 5% SI, steep near 20%;
- the small-cap multiplier (`small_cap_borrow_multiplier` = 2.0 below `small_cap_threshold_usd` = \$2B) is applied **after** the curve, so a small-cap HTB name reaches 50%/yr and small-cap GC is 1%/yr.

**Small-cap multiplier semantics, precisely.** It multiplies the proxy curve and the GC default (sources `proxy` and `default`) when the name's market cap is **known** and below the threshold. It never touches a `data` fee: a logged borrow rate already reflects the name's size. A name with unknown market cap gets no multiplier. A data-poor small cap keeps the source label `default` even though its fee is 1%/yr.

Values: SI 5% → 0.5%; 10% → 3.2%; 15% → 11.4%; ≥ 20% → 25% (large cap).

### Why this choice
- **Why a proxy at all?** IBKR publishes indicative borrow rates daily but has no history API and no numeric schedule. The only way to get history is to log it from day one (BUILD_PLAN §4 and §7). Until about 12 months accumulate (REVIEW_NOTES item 2), short interest is the best free tightness measure.
- **Why deliberately pessimistic?** The `short_interest` sleeve shorts exactly the high-SI names, so an optimistic proxy would inflate that sleeve's net return. A 25% cap at 20% SI is at the high end of typical HTB fees.
- **Why not cvxportfolio's 5%/yr default?** cvxportfolio itself calls 5% a rough, *optimistic* figure for liquid names. For GC large caps it is too high, and for HTB names too low. A size- and SI-conditional proxy beats any constant.

### In the code
- `qts/tcost/borrow.py::borrow_fee_proxy(short_interest_ratio, days_to_cover, cfg, market_cap)`, `holding_cost_fraction`, `holding_cost_breakdown`, `cvx_holding_expression` (convex: `fee/252 · neg(w_post)`).
- `qts/tcost/model.py::TransactionCostModel._borrow_fee_detail(md, date, tickers, adv_shares)` applies the per-name chain and returns `(fee, source)` Series. `borrow_fee_sources(md, date)` exposes the per-name sources; `borrow_source_label` builds the summary string stored on `LiquiditySnapshot.borrow_fee_source`. A warning is logged once when the proxy is in use and no name has data.
- Days-to-cover needs `md.shares_outstanding` (SI ratio × shares outstanding / ADV shares); without it the proxy uses short interest alone.
- The optimizer receives the holding cost from `cvx_terms`. The portfolio layer multiplies it by the holding horizon $H$ (→ see **Portfolio construction › Objective units**).
- The backtester charges one day of borrow per step on post-trade shorts through `TransactionCostModel.holding_cost_usd(w_post, nav, snap)` = $\max(-w,0)\cdot\text{nav}\cdot f/252$, on the **execution-date** snapshot.

### Pitfalls & checks
- **Placeholder.** The proxy is a guess with plausible shape, not a calibrated model. Replace it with logged SLB rates as soon as they exist, and check `borrow_fee_source` in every backtest report.
- **Sparse simulation dates.** Borrow is charged at `fee/252` once per step. Running the simulator on a sparse date grid under-charges borrow proportionally (audit #19).
- **Recall and locate failure** are not modelled. An HTB short can be bought in at the worst moment.
- **No short rebate or cash interest** (audit #21), which is conservative in high-rate regimes.
- **Borrow in the gates.** Borrow is not a per-side cost, so breakeven and cost sensitivity start from `gross_after_borrow` (§1). Before the fix the gate ignored borrow entirely.
- Tests: `test_borrow_fee_proxy_shape`, `test_snapshot_uses_borrow_proxy_when_fee_absent`, `test_borrow_fee_source_per_name_and_all_nan_panel`, `test_holding_cost`.

### Worked example
A large-cap short of 1% of NAV with SI = 12%: $u = (0.12 - 0.05)/0.15 = 0.467$, so $f = 0.005 + 0.245 \times 0.218 = 5.84\%$/yr. Daily cost = $0.01 \times 0.0584/252 = 2.3 \times 10^{-6}$ of NAV, or \$11.6 a day on a \$5M book (≈ \$2,900 a year). The same name with market cap \$1.5B: 11.7%/yr. A \$1.5B name with **no** short-interest data: GC 0.5% × 2 = 1.0%/yr, source `default`. The same name with a logged fee of 3%: 3%/yr, source `data`, no multiplier.

### References
- [Engelberg et al. (2025), equity loan fees as a return predictor, Management Science 71(7)](https://ideas.repec.org/a/inm/ormnsc/v71y2025i7p5529-5551.html)
- [IBKR, Short sale cost](https://www.interactivebrokers.com/en/pricing/short-sale-cost.php?menu=A)
- [IBKR, Short securities availability (SLB)](https://gdcdyn.interactivebrokers.com/en/trading/short-securities-availability.php?menu=B)
- [cvxportfolio, Transaction and holding costs](https://www.cvxportfolio.com/en/stable/costs.html)

## 9. Delisting cost

### What it is
A **delisting** happens when a stock stops trading on its exchange: acquisition, bankruptcy, failure to meet listing standards. In price data it shows up as a missing next close. What a holder actually receives ranges from 0 (bankruptcy) to a premium (cash takeover).

### Why we need it here
The earlier simulator gave a delisted name a 0 return at its last price and removed it for free, with no trade, cost or turnover (audit #7). Last-price exits are optimistic for longs in failures, because performance-related delistings with a missing delisting return lose about 30% on average (Shumway 1997, the convention cited in `qts/core/config.py`).

### How it works
On the day a held name's next close is NaN, the backtester:
1. applies `CostConfig.delisting_haircut` (**−30%**) **against the position's sign**: $r = h \cdot \text{sign}(w)$. A long loses 30%. A short also loses: the stock is treated as rising 30%. This is conservative for both sides, since a takeover premium hurts shorts;
2. force-liquidates the remaining value $-w(1+r)$ at **spread + explicit fees** on the continuous venue, with no impact, and books it as a trade on that decision row so it counts in turnover and cost;
3. records the event in `BacktestResult.diagnostics["delistings"]`.

Labels get the matching treatment: a forward return that runs into a delisting is measured to the last close and multiplied by $(1 + h)$ with the same `CostConfig.delisting_haircut`. That field is now the **single source** of the delisting assumption; the former `AlphaConfig.delisting_return` was removed. Labels identify a delisting from `MarketData.delisting_dates` when the vendor supplies them, otherwise from a causal price-gap rule (`AlphaConfig.delisting_gap_days` = 63) (→ see **Alpha model › Targets**).

### Why this choice
Charging −30% to both sides has no theoretical justification beyond conservatism. It removes a bias that otherwise always favours the strategy. When the price source carries real delisting returns (Sharadar and Norgate do), they should replace the haircut. That is on the roadmap.

### In the code
`qts/backtest/simulator.py` (delisting block). Config: `CostConfig.delisting_haircut` = −0.30. Test: `tests/test_backtest.py::test_delisting_is_liquidated_and_flagged`.

### References
- *Shumway (1997), The delisting bias in CRSP data, Journal of Finance* (cited in `qts/core/config.py`).

## 10. The cost model end to end

### What it is
The pieces above combine into one model used in two forms, behind one **typed contract** in `qts/core/costs.py`. Think of it as an interface in the software sense: the portfolio and backtest layers only know three types, and any cost model that satisfies the protocol can be dropped in.

| Type | What it is | Produced by | Consumed by |
|---|---|---|---|
| `LiquiditySnapshot` | per-ticker market inputs as of one date (frozen dataclass) | `CostModel.snapshot(md, date)` | `estimate`, `cvx_terms`, `holding_cost_usd`, the optimizer |
| `CostEstimate` | per-ticker USD cost of one trade vector | `CostModel.estimate(trades_usd, snap, venue, streak)` | the simulator |
| `CostModel` | the protocol: `snapshot`, `estimate`, `cvx_terms`, `holding_cost_usd` | implemented by `qts.tcost.TransactionCostModel` (production) and `qts.portfolio.costs.SimpleCostModel` (test double) | `SinglePeriodOptimizer`, `DailyBarSimulator` |

How each consumer sees the model:

| Consumer | Form | Units | Called as |
|---|---|---|---|
| Optimizer | cvxpy expression: $\sum_i \big(h_i\lvert x_i\rvert + \text{coef}_i\lvert x_i\rvert^{e} + f^{b}_i\,x_i^+ + f^{s}_i\,x_i^-\big)$ (routed: the surrogate of §6) plus holding $\sum_i (f_i/252)\,w_i^-$ | fraction of NAV | `cvx_terms(x, w_post, nav, snap, venue)` |
| Backtester | `CostEstimate`: `spread_usd, impact_usd, fees_usd, total_usd, venue, persistence_mult` | dollars at $\text{nav}_t$ | `estimate(...)`; borrow via `holding_cost_usd(...)` |
| Router / reports | `cost_bps`, `cost_for_venue_choice` | bp of notional | `SquareRootImpact` |
| TCA | measured, not modelled | bp of decision notional | `TCALedger` |

### How it works: `LiquiditySnapshot`
Fields and units:

| Field | Units | Built from (point-in-time, data ≤ $t$) |
|---|---|---|
| `date`, `tickers` | — | the snapshot date; `md.tickers` |
| `price` | USD/share, **as-traded** | `md.trade_price()` on $t$ |
| `adv_usd` | USD/day | mean of as-traded price × actual shares over the last 63 days (includes day $t$, audit #18) |
| `adv_shares` | shares/day | mean share volume over the same window, restated on day $t$'s share basis across splits |
| `sigma_daily` | fraction | sample sd of daily simple returns of the **adjusted** close over 63 days |
| `half_spread` | **fraction** of price | `md.spread_bps` on $t$ / 10⁴, default 2 bp; `half_spread_bps` is a derived property |
| `borrow_fee` | annual fraction | data → proxy → default per name (§8) |
| `auction_share` | fraction of ADV | `auction_share_for_date(date)`; validated in (0, 1] |
| `borrow_fee_source` | string | summary label (§8) |
| `nav` | USD, optional | — |

Construction **normalises** every per-ticker field: reindexed to `tickers`; NaN or ±inf filled with the cross-sectional median (fallbacks if the whole column is missing: price \$100, ADV \$1M, σ 2%, half-spread 2 bp, borrow 0.5%/yr); floored at price \$0.01, ADV \$1,000, σ 10⁻⁴, spread and borrow 0. A missing `adv_shares` becomes `adv_usd / price`. So a snapshot is always aligned and always finite, and the optimizer's convex terms and the simulator's point estimates can never price different inputs. A new listing gets median impact, never zero impact (audit #10). `snap.reindex(tickers)` extends a snapshot to new names with medians; `snap.to_dict()` and read-only mapping access (`snap["adv_usd"]`) keep old dict-based callers working, and `as_snapshot` converts an old dict.

<callout icon="⚠️" color="yellow_bg">
**Raw vs adjusted prices.** Adjusted closes (splits and dividends folded in) are right for **returns** and wrong for **levels**. A 2-for-1 split halves the as-traded price and doubles the shares traded; the adjusted series rescales the past instead. The snapshot therefore uses the **as-traded** close for everything that is a level on one date: the per-share fee count (shares = USD / as-traded price), dollar volume (as-traded price × actual shares) and `adv_shares` (earlier volumes rescaled by the change in `close_raw / close`). It uses the **adjusted** close for $\sigma$, which is a return statistic. When a loader supplies no `close_raw`, `trade_price()` falls back to the adjusted close, which is historically wrong across splits: real-data loaders should provide it (`test_snapshot_adv_and_fees_use_as_traded_prices_across_splits`).
</callout>

### How it works: `CostEstimate`
`CostEstimate.frame` has fixed columns `spread_usd, impact_usd, fees_usd, total_usd, venue, persistence_mult`, plus `trade_usd` (the signed trade) and, for the routed venue, `auction_fraction`. Construction **raises** if a column is missing, any USD value is non-finite, or any USD component is below −10⁻⁶ (a negative fee from a bad config would otherwise be booked as a rebate). `totals()` sums the components; `bps()` gives them per ticker in bp of the trade. Holding cost is deliberately *not* part of a trade estimate: it is `holding_cost_usd(w_post, nav, snap)`, one day of borrow per ticker.

### How it works: the simulator's side of the contract
`DailyBarSimulator` calls `snapshot(md, t)` once per date (cached for the lag window), `estimate` on the execution-date snapshot, and `holding_cost_usd`. A legacy duck-typed model without `snapshot` is wrapped once, at construction, by `_LegacyCostModel`, whose output is normalised by `cost_frame` with the same "raise on unrecognised or non-finite" rule (audit #8). There is no silent fallback between paths.

### Why this choice
Before the contract the snapshot was a dict of Series whose keys drifted (audit #3 lost venue, date and real spreads in `align_snapshot`), `estimate` returned a free-form DataFrame, and two snapshot builders existed. `docs/audit/code_quality.md` refactor #1 asked for one typed contract; the November wave delivered it (WORK_PLAN 1.2). The math did not change; what changed is that inputs are validated once and every consumer reads the same object.

### In the code
- `qts/core/costs.py`: `LiquiditySnapshot`, `CostEstimate`, `CostModel`, `as_snapshot`, `auction_share_for_date`, `VENUE_CONTINUOUS / VENUE_AUCTION / VENUE_ROUTED`.
- `qts/tcost/model.py::TransactionCostModel(cfg, exec_cfg, default_venue="routed")`: `snapshot`, `estimate`, `cvx_coefficients`, `cvx_terms`, `holding_cost_usd`, `participation_cap`, `auction_cap_usd`, `effective_auction_b`. `snapshot_from_market` is a deprecated alias returning `snapshot(...).to_dict()`.
- `qts/portfolio/costs.py`: `snapshot_from_market` (the single builder, delegating to `TransactionCostModel.snapshot`), `align_snapshot`, and `SimpleCostModel` (test double only).
- Tests: `tests/test_tcost.py::test_snapshot_point_in_time_and_estimate`, `test_benchmark_vectorised_estimate_1500_names`; `tests/test_backtest.py::test_simulator_snapshot_is_the_cost_model_snapshot`, `test_cost_frame_raises_on_unrecognised_or_nan`; `tests/test_audit2_costs_backtest.py::test_cost_estimate_rejects_negative_components`.

## 11. Transaction cost analysis (TCA)

### What it is
**TCA** is the after-the-fact accounting of what each order actually cost, broken into components so you can tell bad luck from bad execution and calibrate the pre-trade model. The standard framework is **implementation shortfall** (Perold 1988): the difference between the return of a *paper* portfolio that traded instantly at the decision price and the *real* portfolio.

### Why we need it here
Every impact parameter in §5–7 is a prior from someone else's data. AQR's, Citigroup's and AAPL's metaorders are not this system's orders. The TCA loop is the only way to learn this system's $b$ for each venue. BUILD_PLAN §6 calls it "the part of execution that compounds".

### How it works: the Perold decomposition
For one parent order, let $s = +1$ for a buy and $-1$ for a sell. $F$ is the filled shares, $U$ the unfilled shares, $Q = F + U$. $P_{\text{dec}}$ is the decision price (when the signal said trade), $P_{\text{arr}}$ the arrival price (when the order reached the market), $\overline{P}_{\text{fill}}$ the fill VWAP, $P_{\text{close}}$ the day's close, $P_{\text{next}}$ the next open. Everything is in **bp of decision notional** $N = Q \cdot P_{\text{dec}}$, signed so that **positive = cost**:

$$
\begin{aligned}
\text{delay} &= s\,F\,(P_{\text{arr}} - P_{\text{dec}}) \,/\, N \\
\text{impact} &= s\,F\,(\overline{P}_{\text{fill}} - P_{\text{arr}}) \,/\, N \\
\text{opportunity} &= s\,U\,(P_{\text{close}} - P_{\text{dec}}) \,/\, N \\
\text{fees} &= \text{explicit dollars} \,/\, N \\
\text{IS} &= \text{delay} + \text{impact} + \text{opportunity} + \text{fees} = \frac{s\,[F(\overline{P}_{\text{fill}} - P_{\text{dec}}) + U(P_{\text{close}} - P_{\text{dec}})] + \text{fees}}{N}
\end{aligned}
$$

(each × 10⁴ for bp), where:
- **delay** is the drift between decision and arrival. For a buy, a rising price is a cost. It measures the cost of latency, for example deciding at the 15:00 snapshot and arriving at 15:50;
- **impact** (here: execution cost against arrival) includes spread crossing and market impact. This is the quantity the square-root law predicts;
- **opportunity** is the paper return missed on unfilled shares, valued at the close. If an unfilled buy's stock rose, you lost that gain;
- **reversion** $= s\,F\,(\overline{P}_{\text{fill}} - P_{\text{ref}})/N$ is a **diagnostic, not part of IS**. $P_{\text{ref}}$ is the **next close** by default (`CostConfig.tca_reversion_reference` = `"next_close"`, the BUILD_PLAN convention) or the next open (`"next_open"`). It is positive when the price came back after you traded (a buy followed by a lower reference price), i.e. the part of impact that turned out temporary. A missing reference price gives NaN reversion, never a silent 0, and NaN rows are left out of the bucket averages.

Sign conventions are symmetric: for a sell, $s = -1$, so a falling price before arrival ($P_{\text{arr}} < P_{\text{dec}}$) is a positive delay cost.

<callout icon="🤔" color="purple_bg">
**Why never credit reversion in the optimizer?** If impact reverts overnight, a buy at an inflated close is partly "refunded" by tomorrow's mark-to-market. It is tempting to subtract that from expected cost. BUILD_PLAN §4 forbids it. About 70% of impact is permanent, reversion estimates are noisy, and crediting it would make the optimizer trade more on the strength of a benefit it cannot reliably collect. The plan defines reversion against the **next close**; the code used the next open until November (audit #15, handbook finding #25). It now defaults to the next close and keeps the next open as an option.
</callout>

**Benchmarks matter.** AQR's identical trades measure 12.18 bp against arrival but 3.7 bp against VWAP (Frazzini et al.), because a VWAP benchmark absorbs your own impact. Fix the benchmark before comparing tactics. For auction fills BUILD_PLAN §6 says the relevant arrival benchmark is the 15:50 mid or the pre-imbalance mid. The fill simulator approximates it with the close (see → **Execution › Fill simulator**).

### How it works: calibrating b from own fills
Model: arrival slippage minus half-spread equals impact, $y_i = b\,x_i + \varepsilon_i$, with

$$
y_i = s_i\frac{\overline{P}_{\text{fill},i} - P_{\text{arr},i}}{P_{\text{arr},i}} - h_i, \qquad x_i = \sigma_i\Big(\frac{F_i}{V_i}\Big)^{e-1}
$$

where:
- $y_i$ is in return units (fraction). $h_i$ is the half-spread recorded with the order: **zero for auction fills**, which pay no spread;
- $x_i$ is the impact regressor, `decompose()["impact_x"]`. It is the *average* impact per dollar of the cost model's $C(Q) = b\,\sigma\,Q^e/V^{e-1}$, i.e. $C(Q)/Q = b\,\sigma\,(Q/V)^{e-1}$, with $e$ = `CostConfig.impact_exponent`. At the default $e$ = 1.5 it is the familiar $\sigma\sqrt{F/V}$. The participation $F/V$ is the same in shares or dollars;
- $V_i$ is the **venue volume**: full-day ADV (63-day, shares) for continuous orders; auction share × ADV for auction orders, where the share is the order's own `auction_share` if supplied, else `auction_share_for_date` of the decision date (so a Russell-day fill is regressed on Russell-day volume); or the realised volume the fill was priced on (`venue_volume`), which overrides both.

**Exponent-aware regressor (second-pass finding 2, fixed).** The regressor used to hard-code $\sqrt{F/V}$. With any other exponent in the model, $b$ came out wrong by a large factor: with noise-free fills at $e$ = 1.25 the calibration returned 0.88 / 2.54 for true values 0.3 / 1.0, and at $e$ = 1.8 it returned 0.084 / 0.333. Now `impact_x` uses `cost_cfg.impact_exponent`, and `test_calibrate_b_uses_configured_exponent` checks the recovery.

**Regression through the origin.** There is no intercept, because zero participation must mean zero impact:

$$
\hat b_{\text{OLS}} = \frac{\sum_i x_i y_i}{\sum_i x_i^2}, \qquad \text{se} = \sqrt{\frac{\sum_i (y_i - \hat b x_i)^2}{(n-1)\sum_i x_i^2}}
$$

Orders with no fill, fill rate < 50% (`min_fill_rate`) or non-finite $x$ are dropped.

**Venue-aware V (audit #1, HIGH, fixed).** Regressing auction fills on full-ADV participation recovers $b_{\text{auc}}/\phi^{e-1}$ (the effective $\beta_a$ of §6), which is 3.3× too large at $\phi$ = 0.09. The auditors verified this: noise-free simulated fills with the then-prior $b_{\text{auc}} = 0.5$ calibrated to **1.667**. Feeding that back into `impact_b_auction` would have tripled auction costs on the first monthly run. The fix stores the venue on each order and regresses each venue on its own $V$. `calibrate_b_by_venue` returns one $b$ per venue.

**Shrinkage to the prior.** With a few hundred noisy fills and R² of a few percent, $\hat b_{\text{OLS}}$ is very noisy. The code blends it with the prior:

$$
\hat b = \frac{n\,\hat b_{\text{OLS}} + w\,b_{\text{prior}}}{n + w}
$$

where $n$ is the number of usable orders, $w$ = `prior_weight` (default **20**) and $b_{\text{prior}}$ is the current config value for that venue.

<callout icon="💡" color="blue_bg">
**Bayesian shrinkage, intuitively.** Act as if you had already observed $w = 20$ imaginary orders whose slope was exactly the prior, then pool them with your $n$ real orders. With 5 real fills the estimate is 80% prior. With 200 it is 91% data. With 2,000 the prior barely matters. This is what a Bayesian posterior mean looks like when the prior is "worth" $w$ observations: the data must earn the right to move the parameter. Strictly, the conjugate-normal posterior would weight by information ($\sum x_i^2/\sigma_\varepsilon^2$) rather than by order count. Counting orders is a simpler, slightly cruder version of the same idea.
</callout>

### How it works: the monthly loop
1. Every parent order is logged at submission (`record_order`) with decision price/time, arrival price/time, venue and order type. Fills are logged as they arrive (`record_fill`). After the next close, the order is closed (`close_order`) with close, next open, next close, 63-day ADV, σ, fees, half-spread and the venue volume.
2. Monthly: `TCALedger.decompose()` and `summary()` produce notional-weighted components by participation bucket (< 1%, 1–5%, 5–10%, > 10% ADV) and volatility bucket (< 1.5%, 1.5–3%, > 3% daily σ), as Bloomberg's buy-side reporting does.
3. `calibrate_b_by_venue(prior_weight=20)` gives new `impact_b_auction` and `impact_b_continuous`. Review `n`, `b_ols`, `se` and `r2` before accepting. A move of more than about 2 standard errors from the prior deserves investigation, not automatic adoption. Check the new pair against the surrogate's validity condition $b_{\text{auc}}/\phi^{e-1} \le b_{\text{cont}}$ (§6): if the auction prefactor rises above $b_{\text{cont}} \times 0.3$ the optimizer switches to the upper-bound surrogate.
4. Update `CostConfig`, which changes the config hash. Re-run the backtest and the breakeven gate.

This loop is **not automated** in the code. No scheduler calls `calibrate_b`, and the router, fill simulator and ledger are not wired into `ResearchPipeline`. It is a manual monthly procedure until the live harness exists.

### Why this choice
- **Pooled cross-sectional regression rather than per-stock.** A small fund has tens of fills per stock, not thousands. De Witt & Pakkanen (2026) excluded stocks whose per-stock impact R² was ≤ 0.02 even with institutional data.
- **Expect low R².** Citigroup < 1%, AQR ≈ 7%. A low R² does not mean the model is wrong: daily price noise ($\sigma$) dwarfs impact for small orders.
- **Arrival, not VWAP or close,** as the impact benchmark: VWAP hides your own impact.

### In the code
`qts/tcost/tca.py::TCALedger`: `record_order(order_id, ticker, side, qty, decision_px, decision_ts, arrival_px, arrival_ts, venue=None, **meta)` (venue derived from `meta['order_type']` ∈ {MOC, LOC} if omitted), `record_fill`, `close_order(..., auction_share=None, venue_volume=None, next_close_px=None)`, `decompose`, `summary`, `calibrate_b(prior_b=1.0, prior_weight=20.0, min_fill_rate=0.5, venue=None)`, `calibrate_b_by_venue`, `save`/`load` (parquet or CSV). `TCALedger(cost_cfg=...)` supplies the priors, the exponent, the default auction share and the reversion reference. Order ids must be unique across days. `targets_to_orders` prefixes the date (audit #14).

### Pitfalls & checks
- **Called directly, `calibrate_b` defaults to `prior_b = 1.0`**, the continuous prior. For auction orders, use `calibrate_b_by_venue`, which takes the priors from `CostConfig`.
- **ADV denominator.** Pass the 63-day ADV to `close_order`. The fill simulator falls back to the day's realised volume when the caller gives none (audit #16).
- **Selection bias.** Dropping low-fill-rate orders removes exactly the LOCs that failed because the price ran away, which biases $b$ down. Watch the opportunity-cost column.
- Tests: `test_decomposition_sums_and_signs`, `test_calibrate_b_recovers_truth` (synthetic $b$ = 0.8 recovered within 0.2), `test_calibrate_b_is_venue_aware` (auction and continuous priors recovered per venue; the old full-ADV regressor gives the effective prefactor instead), `test_calibrate_b_uses_configured_exponent`, `test_ledger_persistence_roundtrip`.

### Worked example
A buy of 100 shares. Decision price \$100.00 (10:00), arrival \$100.10 (10:05), fills 60 @ \$100.30 and 20 @ \$100.50, 20 shares unfilled, close \$101.00, next open \$100.80, next close \$100.20, fees \$0.50. Decision notional $N$ = 100 × 100 = \$10,000, so 1 dollar = 1 bp.
- Fill VWAP = (60 × 100.30 + 20 × 100.50) / 80 = \$100.35.
- Delay = +1 × 80 × (100.10 − 100.00) = \$8.00 → **8.0 bp**.
- Impact = 80 × (100.35 − 100.10) = \$20.00 → **20.0 bp**.
- Opportunity = 20 × (101.00 − 100.00) = \$20.00 → **20.0 bp**.
- Fees = **0.5 bp**.
- IS = **48.5 bp**. Check: 80 × 0.35 + 20 × 1.00 + 0.50 = 48.50 ✓.
- Reversion (default, vs the next close) = 80 × (100.35 − 100.20) = +\$12 → **+12 bp**: by the next close the price had come back below the fill, so part of the impact looks temporary.
- Reversion vs the next open (`tca_reversion_reference="next_open"`) = 80 × (100.35 − 100.80) = −\$36 → **−36 bp**: measured at the open, the price was still rising and none of the impact had reverted. The two references can disagree in sign on the same order, which is one more reason reversion is a diagnostic and never a credit.

This is `tests/test_tcost.py::test_decomposition_sums_and_signs`.

### References
- *Perold (1988), The implementation shortfall: paper versus reality, Journal of Portfolio Management.*
- [Frazzini, Israel & Moskowitz (2018), Trading costs](https://conference.nber.org/conf_papers/f68262/f68262.pdf)
- [De Witt & Pakkanen (2026), RL execution with calibrated transient impact](https://arxiv.org/html/2601.22113v1)
- [Bloomberg (2022), BTCA](https://www.bloomberg.com/professional/?p=46500)

## 12. Parameter reference (`CostConfig`)

Defaults match `qts/core/config.py` and `configs/default.yaml`. **Bounds** are the semantic bounds in `COST_CONFIG_BOUNDS`, checked when the config is constructed (`[` / `]` inclusive, `(` / `)` exclusive).

| Field | Default | Units | Bounds | Meaning / where used |
|---|---|---|---|---|
| `commission_per_share` | 0.0035 | \$/share | [0, 1] | IBKR Pro tiered, first tier |
| `commission_min_per_order` | 0.35 | \$/order | [0, 100] | per-order minimum (exact fees only; routed = two orders) |
| `commission_max_pct_value` | 0.01 | fraction of value | [0, 1] | commission cap |
| `clearing_per_share` | 0.0002 | \$/share | [0, 1] | NSCC/DTC, capped at 0.5% of value |
| `sec_fee_per_dollar_sold` | 20.60e-6 | \$/\$ sold | [0, 0.01] | SEC §31 when no date is given (dated table otherwise) |
| `finra_taf_per_share_sold` | 0.000195 | \$/share sold | [0, 1] | TAF when no date is given |
| `finra_taf_max_per_trade` | 9.79 | \$/trade | [0, 1000] | TAF cap when no date is given |
| `cat_fee_per_share` | 0.000003 | \$/share | [0, 1] | CAT, both sides (audit #13: current prospective rate \$0.000001; immaterial) |
| `auction_exchange_fee_per_share` | 0.001 | \$/share | [0, 1] | auction fills / the auction slice of a routed trade |
| `impact_b_continuous` | 1.0 | — | (0, 10] | impact prefactor $b_c$, $V$ = ADV |
| `impact_b_auction` | **0.3** | — | (0, 10] | prefactor $b_a$, $V$ = auction volume; effective $\beta_a$ = 1.0 per $\sqrt{Q/\text{ADV}}$ on a normal day (D2) |
| `auction_volume_share_default` | 0.09 | fraction of ADV | (0, 1] | **single source** of the normal-day auction share (cost model, router, fill simulator, dry-run broker, TCA) |
| `impact_exponent` | 1.5 | — | (1, 2] | $e$ in $C(Q) = b\sigma Q^e/V^{e-1}$; read by every consumer |
| `default_half_spread_bps` | 2.0 | bp | [0, 1000] | fallback half-spread |
| `default_borrow_fee_annual` | 0.005 | fraction/yr | [0, 5] | GC fee; proxy floor |
| `max_participation_adv` | 0.05 | fraction of ADV | (0, 1] | per-name daily trade cap via `participation_cap` |
| `htb_fee_cap_annual` | 0.25 | fraction/yr | [0, 5] | proxy ceiling |
| `borrow_proxy_si_gc` | 0.05 | SI ratio | [0, 1] | proxy GC threshold; must be < `borrow_proxy_si_htb` |
| `borrow_proxy_si_htb` | 0.20 | SI ratio | (0, 1] | proxy HTB threshold |
| `small_cap_borrow_multiplier` | 2.0 | × | [0, 100] | applied to proxy and default fees below the size threshold (§8) |
| `small_cap_threshold_usd` | 2e9 | \$ | [0, ∞) | market cap threshold |
| `delisting_haircut` | −0.30 | return | [−1, 1] | against position sign on delisting; also the label haircut (single source) |
| `impact_persistence` | True | bool | — | exact streak increment (or $m_k$ fallback) in `estimate` |
| `tca_reversion_reference` | `"next_close"` | enum | `next_close` \| `next_open` | TCA reversion diagnostic reference price |

**Validation.** `CostConfig.__post_init__` calls `validate()`, which checks every bound (a value must be a finite real number inside its interval; booleans and strings are rejected), `borrow_proxy_si_gc < borrow_proxy_si_htb` and the reversion enum, and raises one `ValueError` naming every offending field. So `SystemConfig.from_dict`, `from_yaml` and API overrides are all checked; after mutating a field in place, call `validate()` again. The second-pass audit (finding 3) found that negative fees and a zero auction share were accepted by the API and that negative fees were then booked as rebates; the bounds plus `CostEstimate`'s negative-cost check close both. The upper bounds only reject absurd values (for example a typo of 10¹²); they are not plausibility ranges. The API's config schema publishes the same bounds as `minimum` / `exclusiveMinimum` / `maximum`. Tests: `tests/test_audit2_costs_backtest.py::test_cost_config_rejects_out_of_bounds`, `test_config_bounds_edges_and_round_trip`.

Related fields outside `CostConfig`: `ExecutionConfig.auction_max_pct_of_auction_volume` (0.02, bounds (0, 1]) is the routed venue's κ. `PortfolioConfig.max_adv_participation` (0.05) is the optimizer-side copy of the participation cap. `TCALedger.calibrate_b` has its own defaults (`prior_weight` = 20, `min_fill_rate` = 0.5).

## 13. Pitfalls & checks (cross-cutting)

<callout icon="⚠️" color="yellow_bg">
**Units, units, units.** The layer handles four unit systems: **bp** (`cost_bps`, `half_spread_bps`, `explicit_bps`, `CostEstimate.bps()`, TCA output), **fractions of notional or price** (`LiquiditySnapshot.half_spread`, `marginal_fraction`), **dollars** (`explicit_cost`, `total_cost_fraction` despite its name, `CostEstimate` `*_usd`) and **fractions of NAV** (`cvx_terms`, `cvx_cost_expression`, `holding_cost_fraction`). A bp value passed where a fraction is expected is a 10,000× error. The convention: names ending `_bps` are bp; `half_spread` without suffix is a fraction; anything feeding cvxpy is a fraction of NAV. When in doubt, compute one trade by hand as in §5's or §6's worked example.
</callout>

**Optimizer ⇄ simulator consistency.** If the optimizer believes trades are cheaper than the backtester charges, it over-trades and the backtest is internally inconsistent. If they agree, a weak net result is an alpha problem, not a plumbing problem. By construction:
- both read the same `LiquiditySnapshot` type with the same median fills, venue volume, $b$ and spread per name (with a signal lag, the optimizer's snapshot is the signal date's liquidity and the simulator books on the execution date's; → see **Backtesting › The simulator timeline**);
- for `continuous` and `auction`, the spread and impact terms are algebraically identical: $\text{coef}\cdot\lvert x\rvert^{e} = \text{impact\_usd}/\text{nav}$ (§5);
- for `routed`, spread and impact are exact whenever $\beta_a \le b_c$, which holds at the defaults (§6), and an upper bound otherwise;
- explicit fees differ only by the \$0.35 minimum and per-trade caps (linear in the optimizer, exact in the simulator) and, for `routed`, by the auction exchange fee charged on the whole trade;
- the multi-day persistence surcharge is applied only in the simulator (§7), deliberately.

This is now **tested**. `tests/test_cost_equivalence.py` (handbook finding #24) compares `cvx_terms(...).value × nav` with `estimate(...).total` for random trades: `test_cvx_terms_equal_estimate` (default exponent, both single venues), `test_cvx_terms_equal_estimate_non_default_exponent_and_heavy_day`, and the routed tests listed in §6. The second-pass audit measured the per-name impact error of the routed surrogate at ≤ 7.7 × 10⁻¹⁶ and total surrogate/estimate ratios of 1.000045 (continuous), 1.000058 (auction) and 1.0013–1.0023 (routed), all from fees.

Other cross-cutting checks:
- **Silent zero cost.** `CostEstimate` raises on missing columns, non-finite values and negative components; `cost_frame` raises on unrecognised columns for legacy models (audit #8; `test_cost_frame_raises_on_unrecognised_or_nan`, `test_cost_estimate_rejects_negative_components`). Do not weaken this.
- **Snapshot key loss.** Audit #3: an `align_snapshot` that kept only five keys dropped venue, date and real spreads. The typed snapshot makes this impossible for the production path; `test_simulator_snapshot_is_the_cost_model_snapshot` checks that the simulator prices on exactly the cost model's snapshot.
- **Remaining biases.** Optimistic: no recall risk, streak reset on one-day pauses, the fill simulator's independent pricing of a split's continuous child (§6). Conservative: full-permanence persistence, no short rebate or cash interest, first-tier commissions, pessimistic borrow proxy and small-cap multiplier, −30% delisting for both sides, the auction-first ordering of the routed increment. The second-pass audit's verdict: on the default configuration the net backtest numbers are conservative, and the cost, timing and NAV accounting reproduce an independent hand computation exactly.

## 14. Roadmap
- **Auction volume forecast** per name (imbalance history, special-day calendar) to replace the flat 9/20/34% shares; automate the yearly Russell December date.
- **Impact decay kernel** (propagator with ~70% permanent and power-law transient decay) to replace full persistence within a streak.
- **Log IBKR SLB indicative rates daily** and splice them in after ~12 months. Model short rebate and cash interest.
- **Real delisting returns** from Sharadar/Norgate in place of the −30% haircut.
- **Automate the monthly TCA job** once live fills exist. Revisit `impact_b_auction` once 200+ auction fills exist, check it against the surrogate condition (§6), and revisit the router's 2% auction cap with it.
- Make `FillSimulator` price a split's continuous child as the increment $C(q) - C(q_a)$, as the routed cost model does.
- Verify the pre-2024 FINRA TAF rows (code TODO).
