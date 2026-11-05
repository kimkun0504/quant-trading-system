# Start here

<callout icon="🧭" color="gray_bg">
This page is the front door to the handbook. It explains in one picture what the `qts` system does (it turns daily market data into a dollar-neutral long/short stock portfolio, traded in the closing auction), teaches the trading vocabulary a software engineer usually lacks (long and short, gross and net exposure, liquidity, spreads, borrow fees, survivorship, alpha, risk, return conventions and the √252 rule), sets a realistic expectation for performance (a net Sharpe ratio of 0.5–1.0 counts as success, and anything above 2 is treated as a bug), walks through one trading day step by step, and gives the reading order for the rest of the handbook.
</callout>

<callout icon="🆕" color="green_bg">
<b>What changed (Nov 2026)</b>
- Trades now execute one day after the signal by default (`ExecutionConfig.signal_lag_days = 1`): orders for the closing auction are due by 15:50 ET, before the official close prints, so the close cannot be both the signal and the fill price. The day walkthrough below follows the new timeline.
- The optimiser no longer uses a fixed risk aversion of 10. By default (`PortfolioConfig.risk_aversion_mode = "target_vol"`) it searches, every day, for the risk aversion whose optimal book has 10% ex-ante annual volatility.
- The simulator prices trades at the `"routed"` venue (auction up to a participation cap, the rest in the continuous session), not the pure auction.
- The universe now screens on the as-traded (unadjusted) price and refreshes on the first trading day of each calendar month.
- Every run writes a run directory (`runs/<run_id>/`) and is logged to an append-only trial ledger; the `qts` command and a web workbench (API + UI) sit on top. → see **Operating guide**.
- The alignment-factor variance scale was recalibrated from 0.25 to 0.05.
</callout>

## What the system does in one picture

`qts` is a research-to-production skeleton for a **daily-rebalanced, systematic, US equity long/short strategy**. Every trading day it reads end-of-day market data, decides which stocks it is allowed to trade, forecasts which of them will beat or lag the others over the next few weeks, estimates how risky any combination of them would be and what trading would cost, solves an optimisation problem for the best portfolio given all of that, and sends the resulting trades to the closing auction. Research runs use the same chain, ending in a backtest and a set of statistical checks.

```text
                 ┌────────────────────────────────────────────────────────────┐
  market data    │  qts.data     prices, volume, market cap, fundamentals,    │
  (close of t)   │               short interest, industry  →  MarketData     │
                 └───────────────────────────┬────────────────────────────────┘
                                             ▼
                 ┌────────────────────────────────────────────────────────────┐
                 │  universe     which names are investable at t (PIT mask)   │
                 └───────────────────────────┬────────────────────────────────┘
                                             ▼
        ┌────────────────────┬───────────────┴──────┬────────────────────────┐
        ▼                    ▼                      ▼                        │
 ┌──────────────┐    ┌───────────────┐     ┌──────────────────┐              │
 │ ALPHA        │    │ RISK          │     │ TRANSACTION COST │              │
 │ qts.alpha    │───▶│ qts.risk      │     │ qts.tcost        │              │
 │ expected     │    │ covariance    │     │ spread, impact,  │              │
 │ residual     │    │ B F Bᵀ + D    │     │ fees, borrow     │              │
 │ return       │    │ (+ alignment) │     │                  │              │
 └──────┬───────┘    └───────┬───────┘     └────────┬─────────┘              │
        └────────────────────┼──────────────────────┘                        │
                             ▼                                               │
                 ┌────────────────────────────────────────┐                  │
                 │ PORTFOLIO CONSTRUCTION  qts.portfolio  │                  │
                 │ max alpha − risk − cost s.t. limits    │                  │
                 └───────────────────┬────────────────────┘                  │
                                     ▼                                       │
                 ┌────────────────────────────────────────┐                  │
                 │ EXECUTION  qts.execution               │                  │
                 │ orders → LOC/MOC in the closing auction│                  │
                 └───────────────────┬────────────────────┘                  │
                                     ▼                                       │
                 ┌────────────────────────────────────────┐                  │
                 │ BACKTEST  qts.backtest  (earn t → t+1) │◀─────────────────┘
                 │ VALIDATION qts.validation (DSR, PBO,   │
                 │ purged CV, leakage tests, hurdles)     │
                 └────────────────────────────────────────┘
```

Reading the picture top to bottom:

- **Data** (`qts.data`) assembles a `MarketData` object: aligned date × ticker tables of adjusted closing prices, share volume, market capitalisation and optional fundamentals, short interest and borrow fees, plus an industry label per stock. → see **Data & universe**.
- **Universe** (`qts.data.universe`) decides, for every date, which stocks are investable: an as-traded price of at least \$5, enough daily dollar trading volume, not in the smallest 20% by size, and within the top 1,000 by market capitalisation. The answer is a boolean table stored on `MarketData.eligible` and read through `MarketData.eligibility()`. → see **Data & universe › Universe rules**.
- **Alpha** (`qts.alpha`) forecasts each eligible stock's *residual* return over the next 20 trading days (the part of the return not explained by the market, the stock's industry and common risk styles). The output is an `AlphaOutput`. → see **Alpha model**.
- **Risk** (`qts.risk`) estimates the covariance matrix of stock returns with a factor model, so the optimiser knows which combinations of positions are risky. The output is a `RiskModelOutput`. → see **Risk model**.
- **Transaction cost** (`qts.tcost`) prices a proposed trade: commissions and regulatory fees, half the bid-ask spread, market impact that grows with trade size, and the daily fee for borrowing shares to short. → see **Transaction cost model**.
- **Portfolio construction** (`qts.portfolio`) solves a convex optimisation problem: maximise expected alpha minus a risk penalty minus expected costs, subject to limits on leverage, position size, market beta, industry and style exposure. The output is an `OptimizationResult` with target weights. → see **Portfolio construction**.
- **Execution** (`qts.execution`) turns weight changes into share orders and routes them: small orders as limit-on-close (LOC) orders into the closing auction, larger ones partly to a broker algorithm in the continuous session. → see **Execution**.
- **Backtest and validation** (`qts.backtest`, `qts.validation`) replay the whole chain day by day on history, book every cost explicitly, and then ask whether the result is statistically credible after accounting for how many variants were tried. → see **Backtesting** and **Validation & statistics**.

The arrow from alpha into risk is real: the risk model receives the alpha vector so it can add an "alignment factor" that covers the part of the alpha the risk factors cannot see (→ see **Risk model › The alpha alignment problem (Saxena–Stubbs)**). The whole chain is wired together in `qts/pipeline.py::ResearchPipeline`, and the `qts` command-line tool runs it (`qts demo` on a synthetic market, `qts run --store PATH` on real data; → see **Operating guide**).

## What "systematic equity long/short" means

Each word in the phrase carries a design decision.

**Systematic** means every decision is made by code from data, with no human discretion on individual stocks. The benefit is that the strategy can be tested on history exactly as it would have run, and that every choice can be audited. The cost is that a systematic strategy is only as good as its data and its validation: a human analyst who reads tomorrow's newspaper by accident knows they did; a backtest that reads tomorrow's data by accident does not. Most of the engineering in this repository exists to stop that from happening (→ see **Architecture & data flow › Point-in-time discipline**).

**Equity** means the instruments are common stocks, here US-listed ones. No options, futures, bonds or currencies are traded.

**Long/short** means the portfolio holds some stocks (long positions, which profit when the price rises) and sells others it does not own (short positions, which profit when the price falls). With roughly equal dollars on each side, the market's overall direction mostly cancels out, and what remains is the *relative* performance of the longs against the shorts. That relative performance is what the alpha model tries to predict. The default book in this system is **dollar-neutral**: long and short dollar amounts are equal, within a ±5% band (`PortfolioConfig.net_exposure_target = 0.0`, `net_exposure_band = 0.05`).

**Daily-rebalanced** means the optimiser runs once per trading day after the close data is in. It does not mean the portfolio turns over every day. The forecasts look three to four weeks ahead and the optimiser charges for trading, so the realised average holding period is targeted at about 50 trading days (`PortfolioConfig.effective_holding_days_target = 50`).

<callout icon="🤔" color="purple_bg">
Why long/short and not long-only? A long-only portfolio can express a negative view on a stock only by not owning it, which caps how much of the forecast reaches the positions. The build plan cites a transfer coefficient (the correlation between forecasts and positions) of about 0.3 for a constrained long-only mandate against 0.7–0.8 for a long/short book. The cost of long/short is the need to borrow shares (a fee) and the operational complexity of shorting. `PortfolioConfig.book` also accepts `"long_only"` and `"130_30"` for comparison runs.
</callout>

### References

- [Clarke, de Silva & Thorley via Institutional Investor, The Cost of Constraint](https://www.institutionalinvestor.com/article/2btgijdgl1kk3akuoutc0/home/the-cost-of-constraint)

## Trading basics for engineers

This section is a primer. Each term is defined once here and then used freely in the rest of the handbook. → see **Concepts glossary** for the alphabetical list.

### Stocks, shares and market capitalisation

A **stock** (or *equity*, *share*) is a unit of ownership in a company. A company has some number of **shares outstanding**; multiply by the share price and you get its **market capitalisation** ("market cap"), the market's valuation of the whole company in dollars:

$$
\text{MarketCap}_{i,t} = P_{i,t} \times \text{SharesOutstanding}_{i,t}
$$

where:
- $P_{i,t}$ is the price of stock $i$ at the close of day $t$, in dollars;
- $\text{SharesOutstanding}_{i,t}$ is the number of shares in issue on that day.

Market cap spans six orders of magnitude in the US market, from a few million dollars to several trillion. It matters for three reasons in this system. First, large companies are cheap to trade and small ones are expensive (→ see **Liquidity: ADV, the bid-ask spread and market impact** below). Second, many published "anomalies" live almost entirely in tiny companies where they cannot be captured after costs, which is why the universe excludes them (→ see **Data & universe › Universe rules**). Third, size itself is a risk factor: small stocks tend to move together. In the code, market cap is the `MarketData.market_cap` panel, in US dollars.

A **ticker** is the short exchange symbol for a stock (AAPL, MSFT). Tickers are reused and changed over time, which is a data-engineering headache covered on the data page.

### Long and short positions

A **long** position is the ordinary case: you buy shares and profit if the price rises. If you buy \$10,000 of a stock and it rises 3%, you gain \$300.

A **short** position reverses the sign. You **borrow** shares from someone who owns them (through your broker), sell them immediately, and must buy them back later to return them. If the price falls, you buy back cheaper and keep the difference; if it rises, you lose. Shorting \$10,000 of a stock that then rises 3% loses \$300. Two facts make shorts different from longs:

1. **You pay to borrow.** The lender charges an annual fee on the value borrowed (→ see **Borrow fees and short-selling mechanics** below).
2. **Losses are unbounded in principle.** A long can lose at most 100%; a short loses more the higher the price goes. Position limits (`PortfolioConfig.max_position_weight = 0.02`, i.e. 2% of capital per name) contain this.

In the code, a position is a **weight**: the position's dollar value divided by the portfolio's total capital (NAV, below). Long weights are positive, short weights negative. A weight of $-0.015$ means a short position worth 1.5% of capital.

### NAV, leverage, gross and net exposure, dollar-neutral

**NAV** (net asset value) is the portfolio's capital: cash plus the value of long positions minus the value of short positions (what you would have if you closed everything at current prices). All weights in `qts` are fractions of NAV (`qts/core/types.py` conventions).

Two numbers summarise how a book is positioned:

$$
\text{Gross} = \sum_i |w_i|, \qquad \text{Net} = \sum_i w_i
$$

where:
- $w_i$ is the weight of stock $i$ (fraction of NAV, signed);
- **gross exposure** measures the total size of the bets: how many dollars of stock you control per dollar of capital;
- **net exposure** measures the directional bet on the market as a whole.

**Leverage** usually means gross exposure above 1.0. A book with \$1M of NAV, \$1M long and \$1M short has gross 2.0 and net 0.0. That book is **dollar-neutral**: a 1% move of every stock in the same direction leaves it unchanged. The system's default is gross at most 2.0 (`PortfolioConfig.gross_leverage_max = 2.0`) with net held at $0 \pm 0.05$, i.e. up to "100/100" (100% of NAV long, 100% short).

<callout icon="⚠️" color="yellow_bg">
Naming trap. The review notes (`docs/REVIEW_NOTES.md` item 7) and a comment in `qts/core/config.py` call the default book "150/50". A 150/50 book (150% long, 50% short) has gross 2.0 but net +1.0, i.e. it is fully exposed to the market. The code's defaults (gross ≤ 2.0, net target 0 ± 0.05) describe a dollar-neutral 100/100 book. Read "150/50" there as "gross 2.0", not as the long/short split.
</callout>

**Dollar-neutral is not the same as market-neutral.** Stocks have different sensitivities to the market, measured by **beta** ($\beta$): a stock with $\beta = 1.5$ tends to move 1.5% when the market moves 1%. A dollar-neutral book that is long high-beta stocks and short low-beta ones still loses money in a market sell-off. That is why the optimiser also constrains the portfolio's beta exposure (`PortfolioConfig.beta_neutral = True`, band `beta_band = 0.05`; → see **Portfolio construction › Constraints and why each exists**).

Why gross 2.0 and not more? US retail margin rules (Regulation T) allow roughly 2:1 gross on a standard margin account; more needs a portfolio-margin account. The review notes adopted 2.0 as the realistic default for a small account.

### Liquidity: ADV, the bid-ask spread and market impact

**Liquidity** is how much you can trade without moving the price against yourself. Three quantities describe it.

**Average daily volume (ADV)** is how much of a stock trades on a typical day. It can be counted in shares or in dollars; `qts` mostly uses **average daily dollar volume**, the mean over a trailing window of price × shares traded:

$$
\text{ADV}^{\$}_{i,t} = \frac{1}{L} \sum_{s=t-L+1}^{t} P_{i,s} \, V_{i,s}
$$

where:
- $V_{i,s}$ is the number of shares of stock $i$ traded on day $s$;
- $L$ is the window length in trading days (63, about three months, in `UniverseConfig.adv_window`).

ADV is the natural yardstick for trade size. Buying 1% of a stock's ADV is easy; buying 50% of it moves the price a lot. The system caps trades at 5% of ADV per name per day (`PortfolioConfig.max_adv_participation = 0.05`).

The **bid-ask spread** is the gap between the highest price anyone is currently willing to pay (the **bid**) and the lowest price anyone is willing to sell at (the **ask** or *offer*). The midpoint between them is the fair price; if you need to trade immediately, you buy at the ask or sell at the bid, paying about **half the spread** relative to the midpoint. Spreads are measured in **basis points** (bp): 1 bp = 0.01% = 0.0001. For liquid large caps the half-spread is typically 1–3 bp; for small caps it is far wider. `MarketData.spread_bps` holds a half-spread proxy in bp.

**Market impact** is the price move your own order causes. Large orders consume the available liquidity at the best prices and push the price away. Empirically, impact grows with the square root of order size relative to volume (the "square-root law"), scaled by the stock's volatility. → see **Transaction cost model › Market impact: the square-root law**.

Liquidity is why the system has a universe at all: trading costs, not forecasting skill, decide whether a small book makes money. The build plan's evidence puts half-spread plus impact at roughly 5–10 bp per side for liquid large caps traded under 1% of ADV, and 20 bp or more for small caps.

### The closing auction

US stock exchanges run continuous trading from 9:30 to 16:00 Eastern time, then settle the official **closing price** in a **closing auction**: a single batch match at 16:00 in which all orders flagged "on close" execute at one price. Two order types matter:

- **MOC (market-on-close)**: buy or sell at whatever the closing price turns out to be.
- **LOC (limit-on-close)**: the same, but only if the closing price is no worse than a limit you set.

On NYSE and Nasdaq, on-close orders must be entered (or cancelled) by **15:50 ET**; after that only orders that offset a published imbalance are accepted (`qts/execution/ibkr.py::auction_deadline`). The closing auction carries about 9% of total US daily volume on a typical day and much more on index-rebalance days, all at a single price that is also the price most benchmarks and fund NAVs use. For a strategy that rebalances once a day on end-of-day data, it is the natural venue: trading at the closing price means the backtest's "trade at the close" assumption is close to what happens live. The 15:50 cutoff has one consequence that shapes the whole backtest: the close you trade at cannot also be an input to the decision, because it does not exist yet when the order must be in. The system therefore forms each day's decision from the previous close by default (→ see **The life of one trading day** below). The backtest prices trades at the `"routed"` venue (`ResearchPipeline(venue="routed")`): each name's trade up to 2% of the expected auction volume is priced in the auction, any remainder in the continuous session (→ see **Execution › US market microstructure primer**).

### Borrow fees and short-selling mechanics

To short a stock you must locate shares to borrow. Brokers lend from a pool and charge an annualised **borrow fee** on the value borrowed. Most large, widely held stocks are **general collateral (GC)**: easy to borrow at a low fee, on the order of 0.25–0.5% per year. Stocks that many people want to short, or that few lenders hold, become **hard to borrow (HTB)**, with fees that can reach tens of percent per year, and the lender can recall the shares.

The fee accrues daily. The simulator charges one day of fee on every short position each day:

$$
\text{BorrowCost}_{t} = \sum_{i: w_i < 0} |w_i| \cdot \text{NAV}_t \cdot \frac{f_i}{252}
$$

where:
- $f_i$ is the annualised borrow fee of stock $i$ as a fraction (0.005 = 0.5% per year);
- 252 is the number of trading days per year (`SystemConfig.trading_days_per_year`).

Worked example: a \$500,000 short at a GC fee of 0.5% per year costs $500{,}000 \times 0.005 / 252 \approx \$9.92$ per day, about \$2,500 a year. The same short in an HTB name at 25% per year costs \$125,000 a year, which would swamp any plausible alpha. Borrow fees are also informative: heavily shorted, expensive-to-borrow stocks tend to underperform, which is why short interest and borrow fee appear as an alpha theme (`qts/alpha/composite.py::THEMES["short_interest"]`). Without a borrow-fee history, the cost model uses a proxy from short interest (`qts/tcost/borrow.py::borrow_fee_proxy`, capped at `CostConfig.htb_fee_cap_annual = 0.25`).

### Survivorship and other data biases (preview)

**Survivorship bias** arises when a historical dataset contains only the companies that still exist today. Companies that went bankrupt, were delisted or were acquired vanish, and a backtest on the survivors silently avoids every past disaster. A related trap is using the *current* list of index members (say, today's S&P 500) to test a strategy in 2010: membership is itself a function of past performance. These biases inflate backtests and are invisible unless you look for them. → see **Data & universe › Biases in historical data**, which also covers look-ahead from restated fundamentals and the "ever in universe" leak that the 2026-11-04 audit found in this code.

### Benchmarks

A **benchmark** is the yardstick a strategy is compared against. A long-only fund is compared against an index such as the S&P 500, and its "active return" is the difference. For a dollar-neutral long/short book the natural benchmark is **cash**: the book has no net market exposure, so its return should be judged against what the capital would earn doing nothing. The simulator does not credit interest on cash or on short-sale proceeds, so the returns it reports are best read as excess returns over a cash rate, and the Sharpe ratio is computed with `rf = 0` (`qts/backtest/metrics.py::annualized_sharpe`). The risk model's factors (market, industries, styles) act as a finer-grained benchmark: performance that a few known factors explain is cheap to obtain and is not counted as alpha.

### What "alpha" and "risk" mean here

Both words are overloaded in finance. In this handbook they mean the following.

**Alpha** is a forecast of the part of a stock's future return that is *not* explained by its exposure to known risk factors (the market, its industry, and styles such as size, beta, volatility and liquidity). That leftover is the **residual return**. The alpha model's output has two forms (`qts/core/types.py::AlphaOutput`):

- `score`: a cross-sectional z-score per stock (mean 0, standard deviation 1 across the eligible names on that date), meaning "how attractive relative to the others";
- `expected_return`: the score converted into an expected residual return over the forecast horizon (20 trading days) by Grinold's rule, $\alpha_i = \text{IC} \times \sigma_i \times z_i$, where IC is the forecast's measured skill and $\sigma_i$ the stock's residual volatility. → see **Alpha model › Grinold's rule**.

The **information coefficient (IC)** is the cross-sectional correlation between forecasts and realised returns on a date (here the Spearman rank correlation). It is small: an IC of 0.03–0.05 on a 1,000-stock universe is the build plan's realistic working target. A tiny correlation is enough because it is applied to hundreds of stocks at once and repeated many times.

**Risk** is the forecast variability of the portfolio's return, measured as variance (or its square root, volatility). The risk model provides a covariance matrix for all stocks in factor form:

$$
\Sigma = B F B^\top + D
$$

where:
- $\Sigma$ is the $N \times N$ covariance matrix of daily stock returns;
- $B$ is the $N \times K$ matrix of factor exposures (`RiskModelOutput.exposures`);
- $F$ is the $K \times K$ factor covariance matrix (`factor_cov`);
- $D$ is a diagonal matrix of stock-specific variances (`specific_var`), daily units.

Portfolio variance is then $w^\top \Sigma w$. Risk in this sense is about *forecasting* variance, not about the chance of losing money, though the two are related. → see **Risk model**.

The **Sharpe ratio** ties the two together: annualised mean return divided by annualised volatility. It is the main yardstick for a long/short strategy because it is unchanged by leverage (doubling every position doubles both mean and volatility).

### Returns: simple versus log

A **simple return** (arithmetic return) over one day is

$$
r_{i,t} = \frac{P_{i,t}}{P_{i,t-1}} - 1
$$

and a **log return** (continuously compounded return) is

$$
\ell_{i,t} = \ln\!\left(\frac{P_{i,t}}{P_{i,t-1}}\right) = \ln(1 + r_{i,t})
$$

where $P_{i,t}$ is the adjusted close price (adjusted for splits and dividends, → see **Data & universe › Corporate actions and adjusted prices**).

They have complementary algebra:

| Property | Simple return $r$ | Log return $\ell$ |
|---|---|---|
| Aggregating across stocks on one day | Exact: portfolio return $= \sum_i w_i r_i$ | Not additive across stocks |
| Aggregating across days for one asset | Multiply: $\prod_t (1 + r_t) - 1$ | Exact: add, $\sum_t \ell_t$ |
| Lower bound | $-100\%$ | $-\infty$ |
| Approximation for small moves | $r \approx \ell$ when $\lvert r \rvert$ is a few percent | |

`qts` uses **simple returns everywhere** (`MarketData.returns` is `close.pct_change()`), because the core operation, computing a portfolio's return from its weights, is exact only for simple returns. Multi-day returns are compounded, and labels are $P_{t+h} / P_t - 1$.

Worked example (cross-section). Weights $w = (+0.5, +0.5, -0.5, -0.5)$, one-day returns $r = (+2\%, -1\%, +1\%, -3\%)$. The portfolio return is $0.5(0.02) + 0.5(-0.01) - 0.5(0.01) - 0.5(-0.03) = 0.010 - 0.005 - 0.005 + 0.015 = +1.5\%$. Gross exposure is 2.0, net is 0.

Worked example (time). A stock rises 10% and then falls 10%. Simple: $1.10 \times 0.90 - 1 = -1\%$, not 0%. Log: $\ln 1.1 + \ln 0.9 = 0.09531 - 0.10536 = -0.01005$, and $e^{-0.01005} - 1 = -1\%$. The sum of simple returns ($0\%$) is wrong; the sum of log returns is right.

### Daily versus annualised: the √252 rule

Research papers and the hurdle gates quote annual numbers; the simulator produces daily ones. The conversion uses 252 trading days per year:

$$
\mu_{\text{ann}} = 252\,\mu_{d}, \qquad \sigma_{\text{ann}} = \sqrt{252}\,\sigma_{d}, \qquad \text{SR}_{\text{ann}} = \frac{\mu_{\text{ann}}}{\sigma_{\text{ann}}} = \sqrt{252}\,\text{SR}_{d}
$$

where:
- $\mu_d$, $\sigma_d$ are the mean and standard deviation of daily returns;
- $\text{SR}_d = \mu_d / \sigma_d$ is the daily Sharpe ratio;
- $\sqrt{252} \approx 15.87$.

Worked example. A strategy averages +0.02% per day with a daily standard deviation of 0.63%. Annual volatility is $0.63\% \times 15.87 \approx 10.0\%$ (the system's default target, `PortfolioConfig.target_vol_annual = 0.10`). Daily Sharpe is $0.0002 / 0.0063 = 0.0317$; annual Sharpe is $0.0317 \times 15.87 \approx 0.50$. The annual mean is $252 \times 0.02\% = 5.04\%$, and $5.04\% / 10.0\% \approx 0.50$ agrees.

<toggle heading="h3">
Why volatility scales with √T, and when it does not

Let the annual return be the sum of $T = 252$ daily returns, $R = \sum_{t=1}^{T} r_t$ (for small daily returns, the sum of simple returns is close to the sum of log returns, which is exact). Then

$$
\mathbb{E}[R] = \sum_{t} \mathbb{E}[r_t] = T\mu_d
$$

$$
\operatorname{Var}[R] = \sum_{t}\operatorname{Var}[r_t] + 2\sum_{s<t}\operatorname{Cov}[r_s, r_t]
$$

If daily returns are **uncorrelated** across days and have a **constant variance** $\sigma_d^2$, the covariance terms vanish and $\operatorname{Var}[R] = T\sigma_d^2$, so $\sigma_{\text{ann}} = \sqrt{T}\,\sigma_d$. The mean scales with $T$ and the volatility with $\sqrt{T}$, so the Sharpe ratio scales with $T / \sqrt{T} = \sqrt{T}$.

The assumptions behind the rule:

1. **No serial correlation.** If returns are positively autocorrelated (common for strategies that trade slowly or hold illiquid positions), the covariance terms are positive, true annual volatility is larger than $\sqrt{252}\,\sigma_d$, and the √252 Sharpe is overstated. Negative autocorrelation does the opposite.
2. **Constant variance.** Volatility clusters in reality. The rule then gives an average, not a forecast for any particular year.
3. **Stable mean.** The Sharpe ratio assumes the same expected return throughout; alpha decay breaks this.
4. **Additivity.** The exact form holds for log returns; for simple returns it is an approximation that is excellent for daily data.

Lo (2002) derived the corrected annualisation for autocorrelated returns:

$$
\text{SR}_{q} = \text{SR}_{1}\,\frac{q}{\sqrt{q + 2\sum_{k=1}^{q-1}(q-k)\rho_k}}
$$

where $q = 252$, $\text{SR}_1$ is the daily Sharpe ratio and $\rho_k$ is the lag-$k$ autocorrelation of daily returns. With all $\rho_k = 0$ this collapses to $\sqrt{q}\,\text{SR}_1$. The code implements it as `qts/backtest/metrics.py::annualized_sharpe(..., lo_adjust=True)` (lags truncated at 20) and reports it as `sharpe_net_lo` in the backtest summary. A large gap between `sharpe_net` and `sharpe_net_lo` is a warning sign.
</toggle>

The same logic explains why a modest Sharpe ratio takes years to prove. The t-statistic of a mean return is roughly the annual Sharpe ratio times the square root of the number of years: $t \approx \text{SR}_{\text{ann}}\sqrt{Y}$. A true Sharpe of 0.6 needs $Y \approx (2 / 0.6)^2 \approx 11$ years of out-of-sample data to reach $t \approx 2$, which is the build plan's argument for judging a new strategy by process controls rather than a short live record.

### References

- [Lo (2002), The Statistics of Sharpe Ratios](https://ideas.repec.org/a/taf/ufajxx/v58y2002i4p36-52.html)
- [Frazzini, Israel & Moskowitz, Trading Costs](https://conference.nber.org/conf_papers/f68262/f68262.pdf)
- [BMLL, Into the close: unpacking U.S. closing auction dynamics](https://www.bmlltech.com/news/market-insight/into-the-close-unpacking-u-s-closing-auction-dynamics-and-the-impact-of-the-russell-reconstitution)
- [NYSE, Closing auction process fact sheet](https://www.nyse.com/publicdocs/nyse/NYSE_Auctions_Closing_Process_Fact_Sheet.pdf)
- [IBKR, Short sale cost](https://www.interactivebrokers.com/en/pricing/short-sale-cost.php?menu=A)
- [Gençay (2026)](https://arxiv.org/pdf/2608.27734)

## What performance to expect

### The number to keep in mind: net Sharpe 0.5–1.0

The build plan (`docs/BUILD_PLAN.md` §1) reviewed three independent bodies of evidence and they agree: a well-built, cost-aware US equity long/short strategy run by a small team should expect a **net Sharpe ratio of about 0.5–1.0**. "Net" means after every trading cost and borrow fee. The table records where the number comes from and what kind of evidence each source is.

| Finding | Evidence type | Number | Source |
|---|---|---|---|
| Published anomalies, average across 204, after publication and after 2005 | Out-of-sample, post-publication, net of costs | ~4 bp per month net; value-weighted ≈ −2 bp per month | Chen & Velikov |
| Anomaly alphas with weights fixed from pre-publication data | Out-of-sample | 9.9% per year in-sample → 1.1% per year out-of-sample (t = 0.15) | Gonçalves, Loudis & Ogden |
| Neural-net long/short decile portfolio, value-weighted | Pseudo-out-of-sample, gross of costs | Sharpe 1.35 (vs 0.61 for a 3-variable OLS) | Gu, Kelly & Xiu |
| Best machine-learning long/short strategy restricted to published anomalies, post-2005, realistic costs | Pseudo-out-of-sample, net of costs | Net Sharpe 0.84 (an LSTM); about 57% lower than gross | Azevedo, Hoegner & Velikov |
| 215 bank-sponsored strategy indices, backtest vs live | Live | Median Sharpe deterioration of 73% | Suhonen, Lennkh & Perez |
| Headline LLM news signal | Post-training-cutoff out-of-sample | Sharpe 6.54 (2021 Q4) → 1.22 (Jan–May 2024); unprofitable at 20 bp costs | Lopez-Lira & Tang |
| This project's target | Synthesis | Net Sharpe 0.5–1.0 | `docs/BUILD_PLAN.md` |

"Pseudo-out-of-sample" means the model was fitted on early data and tested on later historical data, but the researcher chose the method knowing roughly how the later period went; "live" means real money or real-time tracking after the strategy was fixed.

The review notes then **lowered the phase-1 bar** to a net Sharpe of 0.3 for the rank composite on its own (`ValidationConfig.phase1_min_net_sharpe = 0.3`), keeping 0.5 for the composite plus machine learning (`phase2_min_net_sharpe = 0.5`). Their reasoning: Chen and Velikov's post-2005 net numbers make 0.5 a ceiling, not a floor, for a six-theme composite. The plan also says to expect live performance at one-quarter to one-half of the backtest Sharpe, given the 73% median haircut.

### Why "too good" backtests are suspect

A net Sharpe above 2 on daily US equity data is, in this project, **treated as a defect until proven otherwise** (`docs/BUILD_PLAN.md` §2.4; `qts/validation/leakage.py::SUSPECT_SHARPE = 2.0`; the pipeline adds a note to any run above it). The reasons:

1. **The cost-inclusive literature has no such numbers.** The most careful studies land below 1. A result four times better than the best published cost-aware evidence is far more likely to be a bug than a discovery.
2. **Leakage produces exactly this signature.** A backtest that sees a fraction of tomorrow's return produces spectacular, smooth equity curves. The validation layer's planted-oracle test makes this concrete: feed the future 20-day return in as a "feature" and the toy backtest reports a Sharpe far above 2 (`tests/test_validation.py::test_planted_oracle_is_flagged`).
3. **Statistics cannot catch leakage.** The deflated Sharpe ratio corrects for having tried many variants, but a leaked oracle scores a deflated Sharpe of 1.00, the best possible (Gençay 2026). Leakage has to be prevented by construction (→ see **Architecture & data flow › Point-in-time discipline**).
4. **Selection inflates the best result.** If you try 100 variants of pure noise, the best one will look good. The trial ledger records every run so the deflated Sharpe can account for this (→ see **Validation & statistics › Probabilistic Sharpe, expected maximum Sharpe, Deflated Sharpe**).
5. **Survivorship and missing costs inflate everything.** Dropping delisted stocks or forgetting borrow fees can add several Sharpe points on small-cap-heavy strategies.

<callout icon="💡" color="blue_bg">
A useful reflex: when a change improves the backtest dramatically, look for the bug before celebrating. Typical culprits are a feature that uses `shift(-1)` or a full-sample statistic, a universe or label that knows the future, fundamentals keyed on the period-end date instead of the filing date, a missing cost term, and a price source without delisted stocks. Each has a guard in this code; the guards are listed in **Architecture & data flow › Point-in-time discipline**.
</callout>

The synthetic market in `qts.data.synthetic` gives the opposite calibration point. When the backtest is handed the *true* planted signal (which nobody has in reality), the review notes report a gross Sharpe of about 1.6 and a net of about 1.3, measured on the earlier version of the generator whose planted score drove returns in the same month it was revealed. That number is in-sample on a market built to contain exactly that signal; it shows the plumbing transmits a known signal, nothing more. The real composite on the same synthetic market earns much less and usually fails the phase-1 gate on merit (`README.md`).

### References

- [Chen & Velikov, Zeroing in on the Expected Returns of Anomalies](https://core-prod.cambridgecore.org/core/services/aop-cambridge-core/content/view/945133D5A3ECEEAF466AEE91551FD225/S0022109022000874a.pdf/zeroing-in-on-the-expected-returns-of-anomalies.pdf)
- [Gonçalves, Loudis & Ogden (2025), Out-of-Sample Alphas](https://finance-conference.wpcarey.asu.edu/sites/default/files/2025-02/Out-of-Sample_Alphas_%282025-02-12%29_with_link.pdf)
- [Gu, Kelly & Xiu, Empirical Asset Pricing via Machine Learning](https://www.nber.org/papers/w25398.pdf)
- [Azevedo, Hoegner & Velikov (2025), The Expected Returns of Machine-Learning Strategies](https://www.aeaweb.org/conference/2025/program/paper/TihddefB)
- [Suhonen, Lennkh & Perez via Alpha Architect](https://alphaarchitect.com/?p=36892)
- [Lopez-Lira & Tang, Can ChatGPT Forecast Stock Price Movements?](https://arxiv.org/html/2304.07619v6)
- [Gençay (2026)](https://arxiv.org/pdf/2608.27734)

## The life of one trading day

This walkthrough follows one execution date $t$ through the system as the backtest simulator runs it (`qts/backtest/simulator.py::DailyBarSimulator.run`), with notes on what a live deployment would do differently. Two dates matter. The **execution date** $t$ is the close at which trades fill. The **signal date** $s$ is the trading day `ExecutionConfig.signal_lag_days` $= L$ days earlier ($L = 1$ by default, so $s = t - 1$). Every input to the decision (alpha, risk, the optimiser's cost snapshot, universe membership) is "as of $s$": it may only use data stamped on or before $s$. Labels (future returns) are the single exception and are never used as inputs.

<callout icon="🤔" color="purple_bg">
Why a one-day delay and not "trade at the close you just saw"? Closing-auction orders must be entered by 15:50 ET and the official close prints around 16:00, so a live system cannot use the close it trades at. Setting $L = 0$ (close-as-signal) is kept as an optimistic bound. The lag investigation of 2026-11-05 (`docs/notes/lag_investigation_2026-11-05.md`) found that, on the synthetic market, $L = 1$ costs nothing measurable for this system's slow signals: the composite keeps 93–98% of its lag-0 rank IC, and across five seeds the lag-1 Sharpe was on average 0.16 *higher* than lag 0 (path noise; synthetic, out-of-sample backtest window). An earlier report of a 0.84 Sharpe drop came from an uncommitted code state and did not reproduce.
</callout>

**Step 0: data arrives.** After the 16:00 ET close of each day, that day's row is appended to every panel in `MarketData`: adjusted close $P_{i,t}$ (for returns), as-traded close (for price levels), shares traded $V_{i,t}$, shares outstanding, market cap, half-spread. Slower data becomes visible only once it would really have been public: a quarterly report becomes visible on its **filing date** (Sharadar `datekey`) plus one day by default, not on the quarter-end date it describes; a FINRA short-interest figure for settlement date $S$ becomes visible 8 NYSE sessions after $S$ (→ see **Data & universe**). The return realised on day $t$ is $r_{i,t} = P_{i,t}/P_{i,t-1} - 1$.

**Step 1: eligibility.** The simulator reads `MarketData.eligibility()` (the point-in-time universe mask built by `qts/data/universe.py::build_universe_mask`, combined with "has a close") as of the signal date $s$, and whether a name has a price at all as of $t$. Membership is refreshed on the first trading day of each calendar month and held in between; a member leaves only after failing the screens at three consecutive refreshes. Three groups of names come out of this step:

- eligible and priced: can be traded freely;
- priced but ineligible: cannot be opened; if currently held, the name enters a forced exit that shrinks its allowed weight linearly to zero over `PortfolioConfig.exit_ineligible_days = 5` days;
- no price on $t$ (not yet listed, halted or delisted): no trade is possible (`no_trade` list).

**Step 2: alpha.** `alpha_fn(s)` calls `AlphaModel.as_of(s)`. The 18 Tier-1 features (value, momentum, quality, investment, low-risk, liquidity, size, short interest, short-term reversal) are computed with trailing windows only, then per date over eligible names only: winsorised at the 1st/99th percentiles, industry-demeaned for the value/quality/investment features, and rank-gaussianised. The theme composite averages features into six themes with dynamic weights (factor-momentum tilt capped at ±25%, momentum crash control). If the ML sleeve is on, its prediction for $s$ comes from a model trained only on labels that were fully realised before $s$. Labels are aligned to what is traded: with lag $L$ the label of signal date $s$ is the return from the close of $s + L$ to the close of $s + L + 20$. The combined score is converted by Grinold's rule into `expected_return` over the 20-day horizon. Names that are not eligible on $s$ are absent from the output, not zero-filled (→ see **Alpha model**).

**Step 3: risk.** `risk_fn(s, alpha)` calls the risk model's `as_of(s, alpha=...)`. Factor exposures are measured at $s$, standardised over the names eligible at $s$ only. The factor covariance matrix is built from the history of daily factor returns up to and including $s$ (exponentially weighted, Newey–West corrected, with eigenfactor and volatility-regime adjustments). Specific variances come from stock-specific return history up to $s$. Given the alpha vector, the model adds the alignment factor: the direction of the alpha that the factors cannot explain, with a variance $\nu$ (scale `RiskConfig.alignment_nu_scale = 0.05`), so that the optimiser cannot load up on risk the model does not see. The result is a `RiskModelOutput`, including `factor_groups`, which tells the optimiser which columns are the market, the industries and the styles (→ see **Risk model**).

**Step 4: cost snapshot.** The cost model builds a `LiquiditySnapshot` (`qts/core/costs.py`) from data up to $s$ via `TransactionCostModel.snapshot(md, s)`: 63-day ADV in dollars and shares, daily volatility, half-spread, as-traded price and annual borrow fee per name, every field filled and floored so that it is always finite. The expected share of daily volume in the closing auction is taken for the execution date $t$, because the auction calendar (index rebalances, Russell reconstitution) is published in advance. The optimiser prices every candidate trade from this one snapshot (→ see **Transaction cost model**).

**Step 5: optimisation.** `SinglePeriodOptimizer.solve` takes the expected returns, the risk model, the current (drifted) weights $w^{\text{pre}}$ and NAV marked at $t$, and the snapshot, and solves a convex problem whose objective, in horizon units, is roughly

$$
\max_{w}\; \alpha^\top w \;-\; \gamma_{\text{risk}}\, H\, w^\top \Sigma w \;-\; \gamma_{\text{trade}}\, \text{TC}(w - w^{\text{pre}}) \;-\; \gamma_{\text{hold}}\, H\, \text{Borrow}(w)
$$

where:
- $\alpha$ is the vector of expected residual returns over the horizon $H = 20$ days;
- $\Sigma$ is the daily covariance from step 3 (multiplied by $H$ to match the horizon);
- $\text{TC}(\cdot)$ is the one-off trading cost of the trade vector, a fraction of NAV;
- $\text{Borrow}(w)$ is one day of borrow cost on the shorts;
- $\gamma_{\text{risk}}$ is an annual risk aversion; $\gamma_{\text{trade}} = 1$ and $\gamma_{\text{hold}} = 1$ are `PortfolioConfig.trade_aversion` and `hold_aversion`.

The risk aversion is not a fixed number. With the default `PortfolioConfig.risk_aversion_mode = "target_vol"`, the optimiser starts from the closed-form value $\gamma_0 = \text{IR} / (2 \times 0.10)$ (the risk aversion at which an unconstrained, cost-free book would have exactly the 10% annual volatility target, `target_vol_annual = 0.10`; IR is the forecast's information ratio) and re-solves the same problem with a root search on $\log \gamma$ (at most 8 solves) until the optimal book's ex-ante volatility is within 2% of the target, or until the gross, position or ADV caps stop it growing. Sizing therefore happens inside the alpha-aware objective, so costs act at the size actually traded. (`risk_aversion_mode = "fixed"` uses `risk_aversion = 2.0` instead.) The problem is subject to the constraint set: gross ≤ 2.0, net within ±0.05, |position| ≤ 2% of NAV, trades ≤ 5% of ADV, beta style within ±0.05, industry exposures within ±0.03, risk-only style exposures within ±0.10 standard deviations, and the no-trade and forced-exit lists. The output is an `OptimizationResult` with target weights $w^{\text{post}}$, the trade vector $x = w^{\text{post}} - w^{\text{pre}}$, expected alpha, risk and cost, and solver status. If the solver fails, the current book is held unchanged and the failure is counted (→ see **Portfolio construction**).

**Step 6: orders into the auction.** Live, `qts/execution/orders.py::targets_to_orders` turns weight changes into share quantities using NAV and prices, and `qts/execution/router.py::ExecutionRouter.route` assigns a tactic per order: small orders (≤ 2% of the expected closing-auction volume) become LOC orders priced a few basis points through the indicative closing price and submitted before the 15:50 ET cutoff; larger ones go to a broker algorithm (close-price or VWAP) with participation capped at 5% of ADV; fast signals can use an arrival-price algorithm. In the backtest, the trades fill at the close of $t$ and are charged by `TransactionCostModel.estimate` at the `"routed"` venue, using the liquidity snapshot of the execution date $t$ (the liquidity that was actually there): the part of each name's trade up to 2% of auction volume pays square-root impact against auction volume and the auction exchange fee, the remainder pays half-spread plus impact against ADV; explicit fees on top, plus a multi-day impact increment when the same name is traded in the same direction on consecutive days. The result is a `CostEstimate` with fixed columns `spread_usd`, `impact_usd`, `fees_usd`, `total_usd`; a non-finite cost raises rather than making a trade silently free (→ see **Execution**).

**Step 7: overnight and day $t+1$.** The post-trade positions are held from the close of $t$ to the close of $t+1$ and earn $r_{i,t+1}$. Shorts pay one day of borrow. The NAV identity, exact in the simulator and checked by `tests/test_backtest.py::test_nav_identity`, is

$$
\text{NAV}_{t+1} = \text{NAV}_t \left(1 + \sum_i w^{\text{post}}_{i,t}\, r_{i,t+1}\right) - C_t
$$

where $C_t$ is the dollar total of spread, impact, fees and borrow booked on execution date $t$. If a held name has no close on $t+1$ (a delisting, or a halt, which the simulator cannot tell apart on that day), it earns `CostConfig.delisting_haircut = -0.30` against the position's sign, is liquidated at spread plus fees, and the event is logged in `BacktestResult.diagnostics["delistings"]`. The weights drift with prices to become $w^{\text{pre}}$ for $t+1$, and the loop repeats.

Recorded outputs are indexed carefully: weights, trades and costs by the **execution (decision) date** $t$; returns and NAV by the **realisation date** $t+1$. Mixing the two is a classic off-by-one bug; the audit found one in an earlier version of the pipeline's turnover join.

<callout icon="⚠️" color="yellow_bg">
The one-day delay is slightly pessimistic, not exact. A live system would form signals from a snapshot around 15:30–15:45 on day $t$ and price LOC limits off the indicative close, so its information is a few minutes old, not a full day. For the slow signals this system uses (20-day horizon, roughly 60–100-day signal half-life) the difference is about one day of decay. `qts demo --compare-lag` runs both settings and prints the Sharpe difference; set `signal_lag_days = 0` only to reproduce the optimistic bound, and never for a short-horizon signal such as one-day reversal, where it would book a bid-ask bounce the strategy cannot earn.
</callout>

### Worked example: one name, one day

Take a stock at $P_t = \$50.00$, NAV $= \$5{,}000{,}000$, current weight $w^{\text{pre}} = +0.010$ (a \$50,000 long, 1,000 shares). The optimiser returns $w^{\text{post}} = +0.014$, so $x = +0.004$, a \$20,000 buy, 400 shares. Suppose the cost model charges a half-spread of 2 bp (\$4.00), impact of 1.5 bp (\$3.00, an illustrative figure; the real number comes from the square-root model) and explicit fees of about \$1.88 (commission $400 \times \$0.0035 = \$1.40$, which clears the \$0.35 minimum; clearing $400 \times \$0.0002 = \$0.08$; closing-auction exchange fee $400 \times \$0.001 = \$0.40$; no SEC or FINRA fee because it is a buy): $C \approx \$8.88$ for this name. The next day the stock closes at \$51.00, so $r_{t+1} = +2\%$ and the position contributes $0.014 \times 0.02 = 0.028\%$ of NAV, i.e. $\$1{,}400$, gross. Net of the trade cost the contribution is $\$1{,}391.12$. The post-trade dollar position is \$70,000; after the move it is worth \$71,400, which is the drifted pre-trade weight for $t+1$ once divided by the new NAV.

## How to read this handbook

The pages are written to be read in order the first time, but each stands on its own with cross-references.

| # | Page | Read it to learn | Prerequisites |
|---|---|---|---|
| 1 | Start here (this page) | Vocabulary, the pipeline, expectations, one trading day | None |
| 2 | Architecture & data flow | Package responsibilities, data contracts, point-in-time discipline, run lifecycle, reproducibility, testing, extension points | 1 |
| 3 | Data & universe | Data sources and costs, biases, the PIT store, corporate actions, universe rules, the synthetic market | 1, 2 |
| 4 | Alpha model | Every feature, residualised targets, the rank composite, ML models, walk-forward training, Grinold scaling, signal decay | 1–3 |
| 5 | Risk model | Factor models, exposures, cross-sectional regression, covariance estimation and its adjustments, specific risk, shrinkage, the statistical companion, the alignment factor | 1–3 |
| 6 | Transaction cost model | Fees, spreads, square-root impact, auction vs continuous trading, borrow, impact persistence, TCA and calibration | 1, 2 |
| 7 | Portfolio construction | Mean-variance optimisation, convex constraints, transfer coefficient, Gârleanu–Pedersen, volatility targeting | 4–6 |
| 8 | Execution | Closing auction mechanics, order types, broker algorithms, schedules, routing | 6, 7 |
| 9 | Backtesting | Simulator timeline, accounting identity, costs and delistings in the backtest, common backtest biases | 2, 7 |
| 10 | Validation & statistics | Sharpe and its corrections, IC/ICIR, t-statistics, drawdowns, deflated Sharpe, PBO, purged and combinatorial cross-validation, hurdle gates | 1, 9 |
| 11 | Concepts glossary | One paragraph per term, alphabetical | Any |
| 12 | Operating guide | Install, the `qts` command, the run registry and trial ledger, reading the report card, the Workbench UI, loading real data, recalibration, tests and CI, troubleshooting | 2 |
| 13 | Audit history & known limitations | What the independent audits found and what is still open | 2, 10 |

Three suggested paths:

- **"I want to run it."** Pages 1, 2, then 12. Run `qts demo` (or `make demo`), read its report card alongside page 12's line-by-line guide and page 10, then open the run in the Workbench (`make dev`).
- **"I want to change a model."** Pages 1–3, then the model's page (4–8), then 10, then the extension-point section of page 2.
- **"I want to judge whether a result is real."** Pages 1, 3 (biases), 9 and 10, and the audit page 13.

Conventions used throughout: returns are simple and daily unless marked annual; weights are fractions of NAV; "bp" is a basis point (0.01%); a "name" is a stock; "PIT" means point-in-time; "$t$" is the decision (execution) date, a close, and a quantity "as of $t$" uses only data stamped on or before $t$; when the execution delay matters, the signal date is written $s = t - L$. Parameter defaults quoted in the handbook come from `qts/core/config.py`; if a page and the code ever disagree, the code wins.

## Quick glossary

| Term | Meaning in this system |
|---|---|
| ADV | Average daily dollar volume over a trailing window (63 days by default) |
| Alpha | Forecast residual return; `AlphaOutput.expected_return` over 20 trading days |
| Basis point (bp) | 0.01% = 0.0001 |
| Beta | Sensitivity of a stock's return to the market's return |
| Borrow fee | Annual fee for borrowing shares to short; GC ≈ 0.25–0.5% per year, HTB much higher |
| Closing auction | 16:00 ET batch match setting the official close; MOC/LOC orders due by 15:50 ET |
| Dollar-neutral | Long dollars equal short dollars (net exposure ≈ 0) |
| Eligible | In the point-in-time investable universe on that date and priced (`MarketData.eligibility()`) |
| Execution delay (signal lag) | Days between the close the decision uses and the close it trades at; 1 by default |
| Gross / net exposure | $\sum_i \lvert w_i \rvert$ / $\sum_i w_i$ |
| IC | Cross-sectional rank correlation between forecasts and realised returns |
| NAV | Net asset value: the portfolio's capital |
| PIT | Point-in-time: using only information available at that moment |
| Residual return | Return left after removing market, industry and style factor effects |
| Sharpe ratio | Annualised mean return / annualised volatility |
| Survivorship bias | Inflation from testing only on companies that survived |
| Weight | Position value / NAV, signed |
