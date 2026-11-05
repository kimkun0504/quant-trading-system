# Transaction cost model

<callout icon="🧭" color="gray_bg">
This page explains how the system prices a trade before making it and measures what the trade really cost afterwards. Cost has three parts. **Explicit fees** are the broker, regulatory and exchange charges on the confirmation. **Implicit costs** never appear on a statement: the bid-ask spread, the price move your own order causes (market impact), and the price drift while you wait. **Holding costs** are mainly the fee for borrowing shares you are short. All of it lives in the cost model (`qts.tcost`). The optimizer gets one convex function of the weight change: a linear term (half-spread plus fees) plus a 3/2-power impact term, `a|x| + b·σ·|x|^{3/2}/√(V/nav)`. The backtester charges the same function in dollars. A transaction cost analysis (TCA) ledger breaks every real fill down into delay, impact, opportunity and fees, and recalibrates the impact prefactor `b` separately for each venue. Most published equity anomalies lose essentially all their net return once costs are counted honestly, so this layer decides whether an alpha is real. It is not a detail.
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
- `BacktestResult.summary()["breakeven_cost_bps"]` and the pipeline both use turnover aligned *positionally* to return dates. An earlier date-join shifted turnover by one day and dropped the initial build; audit #4 found it and it is fixed.
- `qts/backtest/metrics.py::cost_sensitivity` re-computes net Sharpe at 0.5×, 1×, 2× and 3× a per-side cost, so you can see how quickly the strategy dies.

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
| Implicit | Market impact | Your order moves the price against you | other traders | concave in size, 3/2 power in the optimizer |
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

**FINRA TAF.** The current rate is **\$0.000195 per share sold, capped at \$9.79 per trade** (SR-FINRA-2024-019 schedule; SR-FINRA-2026-020 keeps it through 2028). There is a **fee holiday from 2026-10-01 to 2026-12-31 inclusive** (SR-FINRA-2026-021), during which TAF is zero. Today (2026-11-04) is inside the holiday. Table `FINRA_TAF_RATES`:

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
- **Exchange add/remove-liquidity rebates are not modelled** in the continuous session. They are small and their sign depends on whether a child order adds or removes liquidity, which the daily model cannot know. The **closing-auction fee is modelled** because the system's default venue is the auction, and that fee (≈ \$0.0008–0.0015/share per audit #13) is 3–4× the clearing fee.
- **Volume tiers are not modelled.** The first tier is the worst case, and modelling the monthly share count would couple days together.

### In the code
- `qts/tcost/fees.py::FeeSchedule.explicit_cost(shares, price, side, date, venue)` returns a dict `commission, clearing, sec, taf, cat, exchange, total` in dollars. `explicit_bps` and `round_trip_bps` are wrappers.
- `qts/tcost/fees.py::sec_section31_rate(date)` and `finra_taf_rate(date)` do the dated lookups. `FeeSchedule.fee_holiday(date)` returns the overrides active on a date.
- `FeeSchedule.marginal_fraction(price, side, date, venue)` is the **linear coefficient per dollar traded** used by the optimizer. It ignores the \$0.35 minimum and the caps, because those make the function non-convex. The backtester uses the exact `explicit_cost`, so the two differ only for orders small enough to hit the minimum.
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

The same trade **today (2026-11-04)** falls inside the TAF holiday, so the sell is \$4.733 and the round trip is 1.687 bp. Routing both legs through the **closing auction** adds 1,000 × \$0.001 = \$1.00 per side: \$4.703 buy, \$5.928 sell, 2.13 bp round trip. The optimizer's linear coefficients for this stock are `marginal_fraction` = 0.741 bp (buy) and 0.986 bp (sell), identical to the exact numbers because nothing hits a minimum or cap.

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

**Auction fills pay no spread.** The closing auction is a *call auction*: all buy and sell orders are collected and crossed at **one** price, the price that maximises executed volume. No one "crosses the spread" because there is no spread at the moment of execution, and every buyer and seller in the cross gets the same print. The model therefore sets the spread leg to zero for auction-venue fills (`estimate`: `spread_bps = hs_bps.where(vname != VENUE_AUCTION, 0.0)`; `cvx_terms` does the same). The auction still costs money. Your order adds to one side of the auction's imbalance and moves the clearing price, which is impact (§5–6), and the exchange charges an auction fee (§3).

### Why this choice
- A **per-name** half-spread from quote data (`md.spread_bps`), falling back to `default_half_spread_bps` = 2 bp. A single constant would under-charge illiquid names exactly where the optimizer is most tempted to trade.
- **Quoted** rather than effective spread. Effective spread (twice the distance of the fill from the mid) is lower for patient flow and is what TCA measures. The quoted half-spread is conservative for pre-trade use.
- The Citigroup study (Almgren et al. 2005) dropped spread from its impact regression for lack of a consistent effect. The system keeps it as a separate linear term and **subtracts** it from slippage before calibrating `b` (§11), so the two do not double-count.

### In the code
- `qts/tcost/model.py::TransactionCostModel.snapshot_from_market` builds `half_spread_bps` from `md.spread_bps` as of the date (forward data never used), filling gaps with the default.
- `_half_spread_bps` prefers `half_spread_bps`, then `half_spread` × 10⁴, then the default. This fallback chain was added after audit #3 found real spreads silently replaced by 2 bp in the simulator path.

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
C(X) = a\,\lvert X\rvert \;+\; b\,\sigma\,\frac{\lvert X\rvert^{3/2}}{\sqrt{V}}
$$

where:
- $X$ is the trade in dollars, $V$ the venue's volume in dollars per day;
- $a$ is the proportional cost per dollar: half-spread plus linear explicit fees;
- $b, \sigma$ as above;
- $C$ is in **dollars**.

This is cvxportfolio's form (with exponent 1.5 and default $b = 1$) and MOSEK's power-cone example.

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
\frac{C(x\,\text{nav})}{\text{nav}} = a\,\lvert x\rvert + b\,\sigma\,\frac{\lvert x\rvert^{3/2}}{\sqrt{V/\text{nav}}}
$$

<toggle heading="h3">
Derivation of the √(V/nav) term

$$
\frac{b\,\sigma\,\lvert x\,\text{nav}\rvert^{3/2}}{\sqrt{V}\;\text{nav}} = b\,\sigma\,\lvert x\rvert^{3/2}\,\frac{\text{nav}^{3/2}}{\text{nav}\,\sqrt{V}} = b\,\sigma\,\lvert x\rvert^{3/2}\,\frac{\sqrt{\text{nav}}}{\sqrt{V}} = b\,\sigma\,\frac{\lvert x\rvert^{3/2}}{\sqrt{V/\text{nav}}}
$$

$V/\text{nav}$ is the stock's daily volume measured in units of your NAV. A \$5M fund trading a stock with \$50M ADV has $V/\text{nav} = 10$. The bigger the fund relative to the stock's volume, the smaller $V/\text{nav}$ and the larger the impact coefficient: **capacity is built into the cost function.** Doubling NAV multiplies the impact cost per unit weight by $\sqrt{2}$.
</toggle>

Per name the coefficient is `coef_i = b_i * sigma_i / sqrt(V_i / nav)`, computed once per rebalance (`qts/tcost/model.py::cvx_terms`):

```python
vol = np.maximum(vol.fillna(vol.median() if vol.notna().any() else 1.0).values, 1e-9)
coef = b.values * sigma / np.sqrt(vol / float(nav))
ax = cp.abs(x)
trade = cp.sum(cp.multiply(hs, ax)) + cp.sum(cp.multiply(coef, cp.power(ax, 1.5)))
```

Linear fees enter as `fee_buy · pos(x) + fee_sell · neg(x)`, with different coefficients on buys and sells because SEC and TAF apply only to sells.

### How it works: convexity and DCP
The optimizer is solved with cvxpy and Clarabel (→ see **Portfolio construction › Solver**). cvxpy accepts only problems it can **prove** convex using *disciplined convex programming* (DCP), a small set of composition rules. The cost expression passes:
- $\lvert x\rvert$ is convex and non-negative;
- $t \mapsto t^{3/2}$ is convex and **increasing** for $t \ge 0$, and an increasing convex function of a convex function is convex, so $\lvert x\rvert^{3/2}$ is convex;
- `pos(x)` and `neg(x)` are convex; non-negative weighted sums of convex functions are convex.

Clarabel handles `power(·, 1.5)` through the **power cone**. Convexity guarantees a single global optimum, found reliably, and matches the economics: the marginal cost of trading rises with size. A concave cost such as a fixed per-order fee would make the problem combinatorial, which is why the \$0.35 minimum stays out of the optimizer.

### Why this choice
- **Square root over linear impact.** Linear impact ($I \propto Q$, the Almgren–Chriss default) gives quadratic total cost. It overstates the cost of big trades and understates small ones, and the evidence above rejects it (ΔAIC = 22 in favour of the square root in Vasaikar 2026).
- **Exponent fixed at 1.5, not fitted.** Almgren 2005 found 0.6 and Sato & Kanazawa 0.5. With R² below 7%, own-fill data cannot distinguish them, so the prefactor $b$ is the only calibrated parameter.
- **Propagator models** (transient impact with power-law decay; cross-impact between stocks) slightly beat parametric forms on CFM metaorder data (Hey, Neuman & Tuschmann 2025) but need tens of thousands of metaorders. BUILD_PLAN §4: ML effort is better spent forecasting $\sigma$ and $V$.
- **No credit for reversion.** Temporary impact does revert, but the plan forbids modelling next-day reversion as a benefit (§11).

### In the code
- `qts/tcost/impact.py::SquareRootImpact`
  - `cost_bps(notional, adv_usd, sigma_daily, b, venue)` gives $10^4\, b\,\sigma\sqrt{\lvert Q\rvert/V}$, vectorised;
  - `total_cost_fraction(...)` gives the integrated cost in **dollars**, despite the name, which is kept for API compatibility;
  - `cvx_cost_expression(x, nav, ...)` is the stand-alone cvxpy form;
  - `participation_cap(adv_usd, nav)` gives `max_participation_adv × ADV / nav`, the per-name bound on $\lvert x\rvert$ (default 5% of ADV).
- `qts/tcost/model.py::TransactionCostModel` is the facade the optimizer and backtester call. `estimate(...)` returns point estimates per ticker (columns `spread_bps, impact_bps, explicit_bps, total_bps` and the `_usd` equivalents). `cvx_terms(...)` returns `(trade_cost_expr, holding_cost_expr)` in fractions of NAV.
- Snapshot inputs, all point-in-time: `adv_usd` = 63-day mean dollar volume, `sigma_daily` = 63-day standard deviation of daily simple returns. Missing sigma (new listings) is filled with the cross-sectional median in both `estimate` and `cvx_terms`, never with 0, which would make impact free (audit #10).
- Tuning: `CostConfig.impact_b_continuous` (1.0), `impact_b_auction` (0.5), `max_participation_adv` (0.05). `impact_exponent` (1.5) exists in the config but **is not read**: 1.5 is hard-coded.

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

## 6. Venue: continuous session vs closing auction

### What it is
US stocks trade in two ways each day. In the **continuous session** (09:30–16:00 ET) orders match one at a time against the limit order book. In the **closing auction**, all market-on-close (MOC) and limit-on-close (LOC) orders are collected and crossed at a single price at 16:00. That price is the official close: the price index funds, benchmarks and NAVs use. Mechanics are in → see **Execution › US market microstructure primer**. Here the question is how to price each venue.

### Why we need it here
The system is a **daily rebalancer** whose backtest assumes trades at the close, so the auction is the default venue (`snapshot["venue"] = "auction"`). The closing auction has a very different volume, and so a different $V$ in the square-root law.

### How it works: auction volume share
The share of a day's volume that trades in the closing auction (BMLL, US equities):

| Day type | Share of daily volume in the close | Code |
|---|---|---|
| Normal day | **≈ 9%** (≈ \$50B notional) | `auction_share_for_date` default 0.09 |
| Quarterly index rebalance / options expiry ("quad witching": 3rd Friday of Mar/Jun/Sep/Dec) | **≈ 20%** | 0.20 |
| Russell reconstitution day (last Friday of June) | **≈ 34%** | 0.34 |
| Russell second semi-annual reconstitution (from 2026): **Friday 11 December 2026** | placeholder 34% | `RUSSELL_RECON_DATES` |

FTSE Russell moved the Russell US indexes to **semi-annual reconstitution from 2026**. The December date is published each year (2026: rank day 30 October, effective after the close on 11 December), and **must be added by hand** to `RUSSELL_RECON_DATES` every year. The 34% for December is a placeholder equal to June's figure. Note that 2026-12-11 is a Friday *before* the third Friday (12-18, quad witching, 20%), so December 2026 has two heavy closes a week apart.

### How it works: the auction prefactor
For an auction fill the cost model uses

$$
I_{\text{auc}} = b_{\text{auc}}\,\sigma\sqrt{\frac{Q}{s \cdot \text{ADV}}} = \frac{b_{\text{auc}}}{\sqrt{s}}\;\sigma\sqrt{\frac{Q}{\text{ADV}}}
$$

where:
- $s$ is the auction share of daily volume (0.09 on a normal day);
- $b_{\text{auc}}$ = `impact_b_auction` = **0.5**;
- $b_{\text{auc}}/\sqrt{s}$ is the **effective** coefficient per $\sqrt{Q/\text{ADV}}$: $0.5/\sqrt{0.09} = 0.5/0.3 = \mathbf{1.67}$, against 1.0 for the continuous session.

**The review reasoning (REVIEW_NOTES item 9).** The senior-quant review first proposed $b_{\text{auc}} = 1.5$. With $V$ = auction volume that is an effective 1.5 / 0.3 = 5.0 per $\sqrt{Q/\text{ADV}}$: every auction order would cost **5× an in-session algo**. That contradicts the routing table, which sends small orders to the auction because it is believed to be the cheapest venue. The amendment set $b_{\text{auc}} = 0.5$ as a **prior** (effective 1.67) to be replaced by a per-venue calibration from own fills (§11). The BUILD_PLAN evidence for auctions is qualitative. BMLL reports stable auction prices and no outsized deviation at the next open. There is no published US auction prefactor. So 0.5 is a judgment call, and the TCA loop exists to correct it.

### How it works: venue choice
`SquareRootImpact.cost_for_venue_choice(notional, adv_usd, sigma_daily, half_spread_bps, date)` compares

$$
\text{cont} = h + b_{\text{cont}}\,\sigma\sqrt{Q/\text{ADV}}, \qquad \text{auc} = b_{\text{auc}}\,\sigma\sqrt{Q/(s\,\text{ADV})}
$$

and returns the cheaper venue. Setting them equal, the auction is cheaper only when

$$
\sqrt{p}\,\Big(\frac{b_{\text{auc}}}{\sqrt{s}} - b_{\text{cont}}\Big) < \frac{h}{\sigma}, \qquad p = Q/\text{ADV}
$$

With the defaults (effective 1.67 vs 1.0), $h$ = 2 bp and σ = 2%, that is $\sqrt{p} < 0.0002/(0.667 \times 0.02) = 0.015$, i.e. **$p < 0.0225\%$ of ADV**. With $h$ = 3 bp the crossover is $p < 0.05\%$ of ADV.

<callout icon="⚠️" color="yellow_bg">
**Known tension between the cost model and the router.** The router (→ see **Execution › The routing table**) sends orders up to **2% of auction volume** to the auction as LOC, which is 0.18% of ADV at a 9% share. Under the cost model's own priors the auction is the *more expensive* venue for most of that range. At 0.18% ADV, σ = 2%, h = 2 bp: auction 14.1 bp against continuous 10.5 bp, before the auction exchange fee. The config comment ("0.5 keeps small auction orders cheaper than crossing the spread") holds only for orders below about 0.02–0.05% of ADV. The router does not call `cost_for_venue_choice`. It routes on the plan's rule of thumb. Until the TCA loop calibrates $b_{\text{auc}}$, read the auction default as a **conservative** costing choice: the backtest prices every trade at the auction model, which audit #11 also notes. Do not read it as a proven cost saving.
</callout>

### Why this choice
- **Auction volume as V.** BUILD_PLAN §4 is explicit: use closing-auction volume, not full-day ADV. Full ADV would claim the auction can absorb 11× more than it does.
- **Date-aware share.** Heavy closes (quad witching, Russell) can absorb 2–4× more size at the same impact. Ignoring them would over-charge exactly the days on which a rebalancer can trade cheaply.
- **A forecast model of auction volume** (per name, using imbalance history) is on the roadmap. The flat 9%/20%/34% shares are cross-sectional averages.

### In the code
- `qts/tcost/impact.py::SquareRootImpact.auction_share_for_date(date)` and `auction_volume_estimate(adv_shares, date)`.
- `qts/tcost/model.py::TransactionCostModel._venue_volume` uses `V = auction_share × ADV` and $b = b_{\text{auc}}$ for auction names, and `V = ADV`, $b = b_{\text{cont}}$ otherwise. The share comes from `snapshot["auction_share"]` if present, else `auction_share_for_date(snapshot["date"])`.
- The `venue` passed to `estimate` overrides the per-name venue. The backtester passes the venue it was asked to run, a fix from audit #3.

### Pitfalls & checks
- **The 0.09 constant is duplicated** in `CostConfig.auction_volume_share_default`, `ExecutionConfig.auction_volume_share_default`, `SquareRootImpact.cfg_share_default()` (hard-coded), `DryRunBroker` (hard-coded) and `FillSimulator(auction_share=0.09)`. `CostConfig.auction_volume_share_default` is **not read** by the cost model. Change the share in one place and the others will not follow.
- **Russell calendar.** The June rule ("last Friday of June") approximates the official calendar. December dates must be added yearly.
- Tests: `test_participation_cap_and_auction_volume` (06-26 → 0.34, 03-20 → 0.20, venue choice formula).

### Worked example
The same \$500k order (1% of \$50M ADV, σ = 2%) on a normal day: continuous = 2 bp spread + 20 bp impact = 22 bp; auction = $0.5 \times 0.02 \times \sqrt{0.01/0.09} = 33.3$ bp. `cost_for_venue_choice` returns `continuous`. On quad witching ($s$ = 0.20): auction = $0.5 \times 0.02 \times \sqrt{0.05} = 22.4$ bp, roughly a tie. On Russell day ($s$ = 0.34): 17.1 bp, so the auction wins.

### References
- [BMLL, Into the close: unpacking US closing auction dynamics and the impact of the Russell reconstitution](https://www.bmlltech.com/news/market-insight/into-the-close-unpacking-u-s-closing-auction-dynamics-and-the-impact-of-the-russell-reconstitution)

## 7. Multi-day impact persistence

### What it is
The optimizer often moves toward a target over several days. Partial trading toward a slowly moving target is the Gârleanu–Pedersen design intent (→ see **Portfolio construction**). Five days of buying 1% of ADV is, to the market, **one metaorder** of 5% of ADV spread over a week, not five independent 1% orders. Impact from day 1 has not fully decayed by day 2, so day 2's buying starts from a higher price.

### Why we need it here
Pricing each day as an isolated metaorder **under-states** cost during multi-day builds. Audit #6 estimated the under-statement of the impact leg at 1.4–2× for 2–4-day builds. The bias is optimistic, which is the dangerous direction for a go/no-go decision.

### How it works
Treat a streak of $k$ consecutive same-direction trading days with daily notional $q_j$ as one metaorder of size $Q_k = \sum_{j \le k} q_j$. Its integrated cost is $C(Q_k) = b\sigma Q_k^{3/2}/\sqrt{V}$, with daily $V$ thanks to horizon invariance (§5). The cost attributable to day $k$ is the **increment** $C(Q_k) - C(Q_{k-1})$.

<toggle heading="h3">
Derivation of the k^1.5 − (k−1)^1.5 multiplier

`estimate` only sees today's notional $q_k$ and the streak length $k$. It assumes the earlier days were the same size, $Q_{k-1} = (k-1)q_k$ and $Q_k = k\,q_k$. Then

$$
C(Q_k) - C(Q_{k-1}) = \frac{b\sigma}{\sqrt{V}}\Big[(k q_k)^{3/2} - ((k-1)q_k)^{3/2}\Big] = \underbrace{\frac{b\sigma q_k^{3/2}}{\sqrt{V}}}_{\text{one-day cost}} \cdot \underbrace{\big[k^{3/2} - (k-1)^{3/2}\big]}_{m_k}
$$

So day $k$'s impact is the one-day impact times $m_k = k^{3/2} - (k-1)^{3/2}$. The sum telescopes: $\sum_{j=1}^{k} m_j = k^{3/2}$, so the total charged over the streak equals the cost of one metaorder of size $k q$, as it should (`test_impact_persistence_multiplier` checks this to 1e-9). For large $k$, $m_k \approx \tfrac{d}{dk}k^{3/2} = 1.5\sqrt{k}$.

| k | 1 | 2 | 3 | 4 | 5 | 10 |
|---|---|---|---|---|---|---|
| $m_k$ | 1.000 | 1.828 | 2.368 | 2.804 | 3.180 | 4.623 |
| $1.5\sqrt{k}$ | 1.50 | 2.12 | 2.60 | 3.00 | 3.35 | 4.74 |
</toggle>

The multiplier applies to the **impact leg only**. Spread and fees are paid afresh each day.

**Why it is an upper bound.** It treats all impact within a streak as **fully permanent**: day 1's price push is still there on day $k$. In reality only part is permanent (≈70% per Frazzini et al.), and the transient part decays as a power law over hours to days. The true multiplier lies between 1 (full decay overnight) and $m_k$ (no decay). The system takes the conservative end on purpose. A propagator-style model with a decay kernel is on the roadmap.

**Asymmetry with the optimizer.** `cvx_terms` gives the optimizer the **one-day marginal** cost ($m = 1$), because the optimizer does not know how many more days it will keep trading in the same direction. During a multi-day build the backtest's realised cost is therefore ≥ what the optimizer expected. This is a deliberate, conservative inconsistency, documented in the module docstring of `qts/tcost/model.py`.

### Why this choice
Audit #6 listed three options: a $\sqrt{k_{\text{eff}}}$ haircut from trade-sign autocorrelation, a decaying "pending permanent impact" state, or reporting both numbers. The incremental-metaorder multiplier is the simplest one that is *exactly right* under the model's own assumptions (square root, full persistence). It needs only one integer of state per name.

### In the code
- `qts/tcost/model.py::persistence_multiplier(streak)`:
```python
def persistence_multiplier(streak) -> np.ndarray:
    """``m_k = k^{1.5} - (k-1)^{1.5}`` for streak length ``k >= 1`` (1 for ``k <= 1``)."""
    k = np.asarray(streak, dtype=float)
    k = np.where(np.isfinite(k), np.maximum(k, 1.0), 1.0)
    return k ** 1.5 - (k - 1.0) ** 1.5
```
- `TransactionCostModel.estimate(..., streak=)` multiplies `impact_bps` by $m_k$ when `CostConfig.impact_persistence` is True (default) and returns the multiplier in column `persistence_mult`.
- `qts/backtest/simulator.py` maintains `streak` per name: +1 if today's trade has the same sign as yesterday's, reset to 1 on a sign change or after a day without trading, 0 if no trade today.

### Pitfalls & checks
- The equal-size assumption over-charges a streak whose later days are smaller than its first (the usual shape of a partial-adjustment build) and under-charges the reverse.
- A one-day pause resets the streak, although real impact does not reset overnight. That is mildly optimistic.
- The docstring in `qts/tcost/model.py` says $m_3 = 2.40$. The exact value is 2.368.
- Tests: `tests/test_tcost.py::test_impact_persistence_multiplier`, `tests/test_backtest.py::test_run_with_tcost_model_venue_and_persistence`.

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

The fee $f_i$ comes from, in priority order: (1) data (`md.borrow_fee`, eventually the logged IBKR indicative rate); (2) a **proxy** from short interest; (3) the GC default 0.5%/yr. The proxy (`borrow_fee_proxy`):

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

Values: SI 5% → 0.5%; 10% → 3.2%; 15% → 11.4%; ≥ 20% → 25% (large cap).

### Why this choice
- **Why a proxy at all?** IBKR publishes indicative borrow rates daily but has no history API and no numeric schedule. The only way to get history is to log it from day one (BUILD_PLAN §4 and §7). Until about 12 months accumulate (REVIEW_NOTES item 2), short interest is the best free tightness measure.
- **Why deliberately pessimistic?** The `short_interest` sleeve shorts exactly the high-SI names, so an optimistic proxy would inflate that sleeve's net return. A 25% cap at 20% SI is at the high end of typical HTB fees.
- **Why not cvxportfolio's 5%/yr default?** cvxportfolio itself calls 5% a rough, *optimistic* figure for liquid names. For GC large caps it is too high, and for HTB names too low. A size- and SI-conditional proxy beats any constant.

### In the code
- `qts/tcost/borrow.py::borrow_fee_proxy(short_interest_ratio, days_to_cover, cfg, market_cap)`, `holding_cost_fraction`, `holding_cost_breakdown`, `cvx_holding_expression` (convex: `fee/252 · neg(w_post)`).
- `qts/tcost/model.py::TransactionCostModel._borrow_fee_asof` applies the priority chain and records `snapshot["borrow_fee_source"]` ∈ {`data`, `proxy`, `default`}. It logs a warning once when the proxy is in use.
- The optimizer receives the holding cost from `cvx_terms`. The portfolio layer multiplies it by the holding horizon $H$ (→ see **Portfolio construction › Objective units**).
- The backtester charges one day of borrow per step on post-trade shorts.

### Pitfalls & checks
- **Placeholder.** The proxy is a guess with plausible shape, not a calibrated model. Replace it with logged SLB rates as soon as they exist, and check `borrow_fee_source` in every backtest report.
- **Sparse simulation dates.** Borrow is charged at `fee/252` once per step. Running the simulator on a sparse date grid under-charges borrow proportionally (audit #19).
- **Recall and locate failure** are not modelled. An HTB short can be bought in at the worst moment.
- **No short rebate or cash interest** (audit #21), which is conservative in high-rate regimes.
- Tests: `test_borrow_fee_proxy_shape`, `test_snapshot_uses_borrow_proxy_when_fee_absent`, `test_holding_cost`.

### Worked example
A large-cap short of 1% of NAV with SI = 12%: $u = (0.12 - 0.05)/0.15 = 0.467$, so $f = 0.005 + 0.245 \times 0.218 = 5.84\%$/yr. Daily cost = $0.01 \times 0.0584/252 = 2.3 \times 10^{-6}$ of NAV, or \$11.6 a day on a \$5M book (≈ \$2,900 a year). The same name with market cap \$1.5B: 11.7%/yr.

### References
- [Engelberg et al. (2025), equity loan fees as a return predictor, Management Science 71(7)](https://ideas.repec.org/a/inm/ormnsc/v71y2025i7p5529-5551.html)
- [IBKR, Short sale cost](https://www.interactivebrokers.com/en/pricing/short-sale-cost.php?menu=A)
- [IBKR, Short securities availability (SLB)](https://gdcdyn.interactivebrokers.com/en/trading/short-securities-availability.php?menu=B)
- [cvxportfolio, Transaction and holding costs](https://www.cvxportfolio.com/en/stable/costs.html)

## 9. Delisting cost

### What it is
A **delisting** happens when a stock stops trading on its exchange: acquisition, bankruptcy, failure to meet listing standards. In price data it shows up as a missing next close. What a holder actually receives ranges from 0 (bankruptcy) to a premium (cash takeover).

### Why we need it here
The earlier simulator gave a delisted name a 0 return at its last price and removed it for free, with no trade, cost or turnover (audit #7). Last-price exits are optimistic for longs in failures, because performance-related delistings with a missing delisting return lose about 30% on average (Shumway 1997, the convention cited in `AlphaConfig`).

### How it works
On the day a held name's next close is NaN, the backtester:
1. applies `CostConfig.delisting_haircut` (**−30%**) **against the position's sign**: $r = h \cdot \text{sign}(w)$. A long loses 30%. A short also loses: the stock is treated as rising 30%. This is conservative for both sides, since a takeover premium hurts shorts;
2. force-liquidates the remaining value $-w(1+r)$ at **spread + explicit fees** on the continuous venue, with no impact, and books it as a trade on that decision row so it counts in turnover and cost;
3. records the event in `BacktestResult.diagnostics["delistings"]`.

Labels get the matching treatment: a forward return that runs into a delisting is measured to the last close and multiplied by $(1 + \text{delisting\_return})$, with `AlphaConfig.delisting_return` = −0.30 (→ see **Alpha model › Targets**).

### Why this choice
Charging −30% to both sides has no theoretical justification beyond conservatism. It removes a bias that otherwise always favours the strategy. When the price source carries real delisting returns (Sharadar and Norgate do), they should replace the haircut. That is on the roadmap.

### In the code
`qts/backtest/simulator.py` (delisting block). Config: `CostConfig.delisting_haircut` = −0.30. Test: `tests/test_backtest.py::test_delisting_is_liquidated_and_flagged`.

### References
- *Shumway (1997), The delisting bias in CRSP data, Journal of Finance* (cited in `qts/core/config.py`).

## 10. The cost model end to end

### What it is
The pieces above combine into one function used in two forms:

| Consumer | Form | Units | Called as |
|---|---|---|---|
| Optimizer | cvxpy expression: $\sum_i \big(h_i\lvert x_i\rvert + \text{coef}_i\lvert x_i\rvert^{3/2} + f^{b}_i\,x_i^+ + f^{s}_i\,x_i^-\big)$ plus holding $\sum_i (f_i/252)\,w_i^-$ | fraction of NAV | the cost model's convex-terms method |
| Backtester | per-ticker DataFrame: `spread_usd, impact_usd, explicit_usd, total_usd` (+ `borrow_usd`) | dollars at $\text{nav}_t$ | the cost model's estimate method |
| Router / reports | `cost_bps`, `cost_for_venue_choice` | bp of notional | `SquareRootImpact` |
| TCA | measured, not modelled | bp of decision notional | `TCALedger` |

The cost interface is being consolidated into a single typed contract (a liquidity snapshot in, a cost estimate out, behind a cost-model protocol in `qts.core`). This page describes the math, which does not change. Check `qts.tcost` for the current call signatures.

### Snapshot inputs (point-in-time, data ≤ $t$)
`adv_usd` (63-day mean dollar volume), `adv_shares`, `sigma_daily` (63-day std of daily simple returns), `half_spread_bps` / `half_spread`, `price`, `borrow_fee`, per-ticker `venue` (default `auction`), `date`, `borrow_fee_source`. The audit confirmed snapshots are identical when rebuilt on data truncated at $t$. One small exception: ADV includes day $t$'s own volume (audit #18).

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
- **reversion** $= s\,F\,(\overline{P}_{\text{fill}} - P_{\text{next}})/N$ is a **diagnostic, not part of IS**. It is positive when the price came back after you traded (a buy followed by a lower next open), i.e. the part of impact that turned out temporary.

Sign conventions are symmetric: for a sell, $s = -1$, so a falling price before arrival ($P_{\text{arr}} < P_{\text{dec}}$) is a positive delay cost.

<callout icon="🤔" color="purple_bg">
**Why never credit reversion in the optimizer?** If impact reverts overnight, a buy at an inflated close is partly "refunded" by tomorrow's mark-to-market. It is tempting to subtract that from expected cost. BUILD_PLAN §4 forbids it. About 70% of impact is permanent, reversion estimates are noisy, and crediting it would make the optimizer trade more on the strength of a benefit it cannot reliably collect. The plan defines reversion against the **next close**; the code uses the **next open** (audit #15, still open). Next close is the more common convention.
</callout>

**Benchmarks matter.** AQR's identical trades measure 12.18 bp against arrival but 3.7 bp against VWAP (Frazzini et al.), because a VWAP benchmark absorbs your own impact. Fix the benchmark before comparing tactics. For auction fills BUILD_PLAN §6 says the relevant arrival benchmark is the 15:50 mid or the pre-imbalance mid. The fill simulator approximates it with the close (see → **Execution › Fill simulator**).

### How it works: calibrating b from own fills
Model: arrival slippage minus half-spread equals impact, $y_i = b\,x_i + \varepsilon_i$, with

$$
y_i = s_i\frac{\overline{P}_{\text{fill},i} - P_{\text{arr},i}}{P_{\text{arr},i}} - h_i, \qquad x_i = \sigma_i\sqrt{\frac{F_i}{V_i}}
$$

where:
- $y_i$ is in return units (fraction). $h_i$ is the half-spread recorded with the order: **zero for auction fills**, which pay no spread;
- $x_i$ is the square-root regressor. $V_i$ is the **venue volume**: full-day ADV (63-day, shares) for continuous orders, auction share × ADV for auction orders, or the realised auction print if logged (`venue_volume`).

**Regression through the origin.** There is no intercept, because zero participation must mean zero impact:

$$
\hat b_{\text{OLS}} = \frac{\sum_i x_i y_i}{\sum_i x_i^2}, \qquad \text{se} = \sqrt{\frac{\sum_i (y_i - \hat b x_i)^2}{(n-1)\sum_i x_i^2}}
$$

Orders with no fill, fill rate < 50% (`min_fill_rate`) or non-finite $x$ are dropped.

**Venue-aware V (audit #1, HIGH, fixed).** Regressing auction fills on full-ADV participation recovers $b_{\text{auc}}/\sqrt{s}$, which is 3.3× too large at $s$ = 0.09. The auditors verified this: noise-free simulated fills with $b_{\text{auc}} = 0.5$ calibrated to **1.667**. Feeding that back into `impact_b_auction` would have tripled auction costs on the first monthly run. The fix stores the venue on each order and regresses each venue on its own $V$. `calibrate_b_by_venue` returns one $b$ per venue.

**Shrinkage to the prior.** With a few hundred noisy fills and R² of a few percent, $\hat b_{\text{OLS}}$ is very noisy. The code blends it with the prior:

$$
\hat b = \frac{n\,\hat b_{\text{OLS}} + w\,b_{\text{prior}}}{n + w}
$$

where $n$ is the number of usable orders, $w$ = `prior_weight` (default **20**) and $b_{\text{prior}}$ is the current config value for that venue.

<callout icon="💡" color="blue_bg">
**Bayesian shrinkage, intuitively.** Act as if you had already observed $w = 20$ imaginary orders whose slope was exactly the prior, then pool them with your $n$ real orders. With 5 real fills the estimate is 80% prior. With 200 it is 91% data. With 2,000 the prior barely matters. This is what a Bayesian posterior mean looks like when the prior is "worth" $w$ observations: the data must earn the right to move the parameter. Strictly, the conjugate-normal posterior would weight by information ($\sum x_i^2/\sigma_\varepsilon^2$) rather than by order count. Counting orders is a simpler, slightly cruder version of the same idea.
</callout>

### How it works: the monthly loop
1. Every parent order is logged at submission (`record_order`) with decision price/time, arrival price/time, venue and order type. Fills are logged as they arrive (`record_fill`). After the close and the next open, the order is closed (`close_order`) with close, next open, 63-day ADV, σ, fees, half-spread and the venue volume.
2. Monthly: `TCALedger.decompose()` and `summary()` produce notional-weighted components by participation bucket (< 1%, 1–5%, 5–10%, > 10% ADV) and volatility bucket (< 1.5%, 1.5–3%, > 3% daily σ), as Bloomberg's buy-side reporting does.
3. `calibrate_b_by_venue(prior_weight=20)` gives new `impact_b_auction` and `impact_b_continuous`. Review `n`, `b_ols`, `se` and `r2` before accepting. A move of more than about 2 standard errors from the prior deserves investigation, not automatic adoption.
4. Update `CostConfig`, which changes the config hash. Re-run the backtest and the breakeven gate.

This loop is **not automated** in the code. No scheduler calls `calibrate_b`, and the router, fill simulator and ledger are not wired into `ResearchPipeline`. It is a manual monthly procedure until the live harness exists.

### Why this choice
- **Pooled cross-sectional regression rather than per-stock.** A small fund has tens of fills per stock, not thousands. De Witt & Pakkanen (2026) excluded stocks whose per-stock impact R² was ≤ 0.02 even with institutional data.
- **Expect low R².** Citigroup < 1%, AQR ≈ 7%. A low R² does not mean the model is wrong: daily price noise ($\sigma$) dwarfs impact for small orders.
- **Arrival, not VWAP or close,** as the impact benchmark: VWAP hides your own impact.

### In the code
`qts/tcost/tca.py::TCALedger`: `record_order(order_id, ticker, side, qty, decision_px, decision_ts, arrival_px, arrival_ts, venue=None, **meta)` (venue derived from `meta['order_type']` ∈ {MOC, LOC} if omitted), `record_fill`, `close_order(..., auction_share=None, venue_volume=None)`, `decompose`, `summary`, `calibrate_b(prior_b=1.0, prior_weight=20.0, min_fill_rate=0.5, venue=None)`, `calibrate_b_by_venue`, `save`/`load` (parquet or CSV). Order ids must be unique across days. `targets_to_orders` prefixes the date (audit #14).

### Pitfalls & checks
- **Called directly, `calibrate_b` defaults to `prior_b = 1.0`**, the continuous prior. For auction orders, use `calibrate_b_by_venue`, which takes the priors from `CostConfig`.
- **ADV denominator.** Pass the 63-day ADV to `close_order`. The fill simulator falls back to the day's realised volume when the caller gives none (audit #16).
- **Selection bias.** Dropping low-fill-rate orders removes exactly the LOCs that failed because the price ran away, which biases $b$ down. Watch the opportunity-cost column.
- Tests: `test_decomposition_sums_and_signs`, `test_calibrate_b_recovers_truth` (synthetic $b$ = 0.8 recovered within 0.2), `test_calibrate_b_is_venue_aware` (0.5 / 1.0 recovered per venue; the old full-ADV regressor gives ≈ 1.67), `test_ledger_persistence_roundtrip`.

### Worked example
A buy of 100 shares. Decision price \$100.00 (10:00), arrival \$100.10 (10:05), fills 60 @ \$100.30 and 20 @ \$100.50, 20 shares unfilled, close \$101.00, next open \$100.80, fees \$0.50. Decision notional $N$ = 100 × 100 = \$10,000, so 1 dollar = 1 bp.
- Fill VWAP = (60 × 100.30 + 20 × 100.50) / 80 = \$100.35.
- Delay = +1 × 80 × (100.10 − 100.00) = \$8.00 → **8.0 bp**.
- Impact = 80 × (100.35 − 100.10) = \$20.00 → **20.0 bp**.
- Opportunity = 20 × (101.00 − 100.00) = \$20.00 → **20.0 bp**.
- Fees = **0.5 bp**.
- IS = **48.5 bp**. Check: 80 × 0.35 + 20 × 1.00 + 0.50 = 48.50 ✓.
- Reversion = 80 × (100.35 − 100.80) = −\$36 → **−36 bp**. The price kept rising after the fill, so none of the impact reverted. Either the impact was permanent or the move was information.

This is `tests/test_tcost.py::test_decomposition_sums_and_signs`.

### References
- *Perold (1988), The implementation shortfall: paper versus reality, Journal of Portfolio Management.*
- [Frazzini, Israel & Moskowitz (2018), Trading costs](https://conference.nber.org/conf_papers/f68262/f68262.pdf)
- [De Witt & Pakkanen (2026), RL execution with calibrated transient impact](https://arxiv.org/html/2601.22113v1)
- [Bloomberg (2022), BTCA](https://www.bloomberg.com/professional/?p=46500)

## 12. Parameter reference (`CostConfig`)

| Field | Default | Units | Meaning / where used |
|---|---|---|---|
| `commission_per_share` | 0.0035 | \$/share | IBKR Pro tiered, first tier |
| `commission_min_per_order` | 0.35 | \$/order | per-order minimum (exact `explicit_cost` only) |
| `commission_max_pct_value` | 0.01 | fraction of value | commission cap |
| `clearing_per_share` | 0.0002 | \$/share | NSCC/DTC, capped at 0.5% of value |
| `sec_fee_per_dollar_sold` | 20.60e-6 | \$/\$ sold | SEC §31 when no date is given (dated table otherwise) |
| `finra_taf_per_share_sold` | 0.000195 | \$/share sold | TAF when no date is given |
| `finra_taf_max_per_trade` | 9.79 | \$/trade | TAF cap when no date is given |
| `cat_fee_per_share` | 0.000003 | \$/share | CAT, both sides (audit #13: current prospective rate \$0.000001; immaterial) |
| `auction_exchange_fee_per_share` | 0.001 | \$/share | MOC/LOC fills only |
| `impact_b_continuous` | 1.0 | — | square-root prefactor, $V$ = ADV |
| `impact_b_auction` | 0.5 | — | prefactor, $V$ = auction volume (effective 1.67 per √(Q/ADV)) |
| `auction_volume_share_default` | 0.09 | fraction of ADV | **not read** by the cost model (date-aware function used instead) |
| `impact_exponent` | 1.5 | — | **not read**: 1.5 is hard-coded |
| `default_half_spread_bps` | 2.0 | bp | fallback half-spread |
| `default_borrow_fee_annual` | 0.005 | fraction/yr | GC fee; proxy floor |
| `max_participation_adv` | 0.05 | fraction of ADV | per-name daily trade cap via `participation_cap` |
| `htb_fee_cap_annual` | 0.25 | fraction/yr | proxy ceiling |
| `borrow_proxy_si_gc` | 0.05 | SI ratio | proxy GC threshold |
| `borrow_proxy_si_htb` | 0.20 | SI ratio | proxy HTB threshold |
| `small_cap_borrow_multiplier` | 2.0 | × | applied below the size threshold |
| `small_cap_threshold_usd` | 2e9 | \$ | market cap threshold |
| `delisting_haircut` | −0.30 | return | applied against position sign on delisting |
| `impact_persistence` | True | bool | multi-day $m_k$ multiplier in `estimate` |

`TCALedger.calibrate_b` has its own defaults (`prior_weight` = 20, `min_fill_rate` = 0.5). `PortfolioConfig.max_adv_participation` (0.05) is the optimizer-side copy of the participation cap.

## 13. Pitfalls & checks (cross-cutting)

<callout icon="⚠️" color="yellow_bg">
**Units, units, units.** The layer handles four unit systems: **bp** (`cost_bps`, `half_spread_bps`, `explicit_bps`, TCA output), **fractions of notional** (`half_spread`, `marginal_fraction`), **dollars** (`explicit_cost`, `total_cost_fraction` despite its name, `estimate()[*_usd]`) and **fractions of NAV** (`cvx_terms`, `cvx_cost_expression`, `holding_cost_fraction`). A bp value passed where a fraction is expected is a 10,000× error. The convention: names ending `_bps` are bp; `half_spread` without suffix is a fraction; anything feeding cvxpy is a fraction of NAV. When in doubt, compute one trade by hand as in §5's worked example.
</callout>

**Optimizer ⇄ simulator consistency.** If the optimizer believes trades are cheaper than the backtester charges, it over-trades and the backtest is internally inconsistent. If they agree, a weak net result is an alpha problem, not a plumbing problem. By construction:
- both use the same snapshot (same σ with the same median fill, same venue volume and $b$ per name, same zero spread for auction names);
- the impact terms are algebraically identical: $\text{coef}\cdot\lvert x\rvert^{3/2} = \text{impact\_usd}/\text{nav}$ (derivation in §5);
- explicit fees differ only by the \$0.35 minimum and per-trade caps (linear in the optimizer, exact in the simulator);
- the multi-day persistence multiplier is applied only in the simulator (§7), deliberately.

The audit verified the impact equivalence **to machine precision** for random trades at both venues (`docs/audit/tcost_execution_backtest.md`, "cvx_terms ≡ estimate"; script `/tmp/audit/cost_check.py`), and `docs/audit/RESOLUTION.md` lists it under "verified correct". **The repository's test suite has no test that compares the two numerically.** The tests check convexity and solvability of `cvx_terms`, the NaN-σ median fill on both paths, and that the simulator prices the requested venue (`tests/test_tcost.py::test_estimate_prices_requested_venue_and_real_spreads`, `tests/test_backtest.py::test_simulator_snapshot_keeps_venue_date_and_spreads`). A one-line regression test should be added, especially during the cost-contract refactor:

```python
x = cp.Variable(n); x.value = xv
trade, _ = tcm.cvx_terms(x, x, nav, snap, include_explicit=False)
est = tcm.estimate(pd.Series(xv * nav, index=idx), snap)
assert trade.value == pytest.approx((est.impact_usd.sum() + est.spread_usd.sum()) / nav)
```

This was run while writing this page and matches to all printed digits.

Other cross-cutting checks:
- **Silent zero cost.** The backtester raises if `estimate()` returns no recognised cost column or NaN costs (audit #8 fix; `tests/test_backtest.py::test_cost_frame_raises_on_unrecognised_or_nan`). Do not weaken this during refactors.
- **Snapshot key loss.** Audit #3: an `align_snapshot` that kept only five keys dropped venue, date and real spreads. Fixed by pass-through; guarded by `test_simulator_snapshot_keeps_venue_date_and_spreads`.
- **Remaining biases.** Optimistic: no recall risk, streak reset on one-day pauses. Conservative: auction model for every backtest trade, full-permanence persistence, no short rebate, first-tier commissions, pessimistic borrow proxy, −30% delisting for both sides.

## 14. Roadmap
- **Single typed cost contract** (snapshot / estimate / protocol in `qts.core`), in progress, per `docs/audit/code_quality.md`. Add the numerical optimizer ⇄ simulator equivalence test as part of it.
- **Auction volume forecast** per name (imbalance history, special-day calendar) to replace the flat 9/20/34% shares, plus a single source of truth for the default share.
- **Impact decay kernel** (propagator with ~70% permanent and power-law transient decay) to replace the full-persistence upper bound.
- **Log IBKR SLB indicative rates daily** and splice them in after ~12 months. Model short rebate and cash interest.
- **Real delisting returns** from Sharadar/Norgate in place of the −30% haircut.
- **Automate the monthly TCA job** once live fills exist. Fix reversion to next close (audit #15).
- **Wire `impact_exponent`** or remove it from the config. Revisit `b_auction` once 200+ auction fills exist, and with it the router's auction threshold (§6 callout).
- Verify the pre-2024 FINRA TAF rows (code TODO).
