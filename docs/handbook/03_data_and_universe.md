# Data & universe

<callout icon="🧭" color="gray_bg">
This page covers everything that happens before a model sees a number: which data a long/short equity strategy needs and why, which vendors supply it and what they cost, the biases hidden in historical data (survivorship, delisting returns, restatements, filing dates) and how large they are, the point-in-time Parquet/DuckDB store, corporate actions and adjusted prices, and the rules that define the investable universe (price at least \$5, ADV at least \$2M, no bottom-20% size names, top 1,000 by market cap, refreshed monthly with three-strike hysteresis), with the reasoning behind each rule. It explains the eligibility panel and why an "ever in the universe" stock list leaks, the FINRA short-interest publication lag, Sharadar's filing-date key, Fama–French 48 industries from SIC codes, the Ken French factor files, and the synthetic market generator that the test suite is built on.
</callout>

## The data a strategy needs

### What it is

Every model in the system reads one object, `qts/core/types.py::MarketData`: a set of aligned date × ticker tables ("wide panels", → see **Architecture & data flow › Data layout: wide and long panels**) plus one industry label per ticker. The table lists each panel, why it is needed, and when a value for date $t$ may become visible.

| Panel (`MarketData` field) | Units | Why it is needed | Visible at $t$ when… |
|---|---|---|---|
| `close` (required) | USD, split- and dividend-adjusted | Returns, momentum, volatility, beta, labels | the close of $t$ prints |
| `volume` (required) | shares | ADV screen, liquidity features, impact | the close of $t$ |
| `market_cap` (required) | USD | Size screen, size factor, cap weights | shares known at $t$ × close of $t$ |
| `industry` (required) | Fama–French 48 label | Industry demeaning, factors, bands | quasi-static (see pitfalls) |
| `spread_bps` | half-spread, bp | Spread cost; liquidity feature | the close of $t$ |
| `short_interest_ratio` | short interest ÷ shares | Short-interest theme and style; borrow proxy | settlement + 8 business days |
| `borrow_fee` | annual fraction | Cost of shorts; loan-fee signal | the day the rate is logged |
| `book_to_price`, `earnings_yield` | ratio | Value theme and style | the report's filing date |
| `gross_profitability` | gross profit ÷ assets | Quality theme | the filing date |
| `asset_growth` | year-over-year fraction | Investment theme | the later report's filing date |
| `eligible` | boolean | The point-in-time universe | built PIT from the above |
| `extra` | any wide panel | Additional data; synthetic ground truth | per panel |

### Why each kind is needed

Three points the table compresses. Prices must be adjusted, or a 2-for-1 split looks like a 50% crash. Fundamentals must be dated by when they became public, not by the period they describe. Short interest and borrow fees serve both the cost model (hard-to-borrow shorts are expensive) and the alpha model (equity loan fees were the most persistent non-text predictor in the build plan's evidence).

### In the code

`ParquetStore.as_market_data` assembles a `MarketData` from stored datasets (`REQUIRED = ("prices_close", "prices_volume", "market_cap", "industry")`, plus `OPTIONAL_PANELS`); `qts/data/synthetic.py::generate` builds one from the synthetic model; `qts/data/loaders.py::yfinance_to_market_data` builds a prototyping one from Yahoo Finance. `MarketData.validate()` checks alignment and industry coverage. `scripts/run_real.py --store PATH` runs the full pipeline on a store.

## Vendors and costs

### What it is

The build plan (`docs/BUILD_PLAN.md` §7) priced the data a single engineer can buy in 2026. Its conclusion: **delisting-complete prices and point-in-time fundamentals are the two non-negotiables**, and among individually accessible vendors only Sharadar documents true point-in-time fundamentals.

| Tier | Monthly cost | Components | What it buys |
|---|---|---|---|
| Free | \$0 | Alpaca IEX data; `openassetpricing`; JKP factors; Ken French library; EODHD sandbox (20 calls/day); SEC EDGAR | Prototyping and factor benchmarks; no PIT fundamentals, no consolidated tape |
| Under \$100 | ~\$85–150 | Norgate Platinum (\$630/yr ≈ \$52.50/mo); Tiingo Power (\$30) or EODHD All-In-One (\$99.99); Databento pay-as-you-go against a \$125 credit | Delisting-complete daily prices and index constituents; current-view fundamentals |
| Under \$1,000 (recommended from Phase 1) | ~\$400–600 | The above plus a Sharadar bundle (low hundreds per month, from a secondary source); Databento Standard (\$199) or Massive Stocks Advanced (\$199, 20+ years of minute history) | True PIT fundamentals; consolidated real-time quotes; closing-auction imbalance feed |
| Options and borrow add-ons | \$30–99 | EODHD US options EOD (\$29.99); Alpaca Algo Trader Plus (\$99); IBKR indicative borrow rates (free with an account, logged daily) | Implied-volatility features; borrow-fee history |

Key facts behind the recommendation:

- **Norgate Platinum** has prices, delisted securities and index constituents from 1990, but current-view fundamentals only: it is the price source.
- **Sharadar SF1** carries each report's filing date (`datekey`), keeps as-reported and most-recent versions, retains delisted companies and has S&P 500 constituents since 1957. "Low hundreds per month" is from a secondary source.
- **Databento**'s US equity history starts in 2018; long minute histories need Massive.
- **Borrow-fee history is not available at individual budgets,** so the plan starts a daily IBKR rate logger in Phase 0, and the loan-fee feature uses FINRA short interest meanwhile (→ see **Short interest: FINRA data and the publication lag** below).
- **Free factor libraries** cover benchmarking: `openassetpricing` (212 predictors), JKP (153 factors, through December 2025) and the Ken French library (daily through September 2026).

### Why this choice

Cheaper paths fail a non-negotiable. Yahoo Finance is free but survivorship-biased, revises adjusted prices retroactively, and has only current shares outstanding and industry; `yfinance_to_market_data` exists for plumbing tests and its docstring says "Never report a Sharpe from this source." Institutional sources (CRSP/Compustat, Bloomberg) are better but priced for institutions.

### Pitfalls & checks

- Run `qts/data/universe.py::survivorship_check(md)` on any new source: if no ticker's prices end before the panel's last date, the source has no delistings and it warns.
- Prices above are the build plan's mid-2026 research; check before buying.

### References

- [Norgate Data, stock market packages](https://norgatedata.com/stockmarketpackages.php)
- [QuantRocket, Sharadar fundamentals documentation](https://www.quantrocket.com/docs/data/fundamental/sharadar/)
- [pickuma, FMP vs Sharadar](https://dev.to/pickuma/financial-modeling-prep-vs-sharadar-fundamental-data-apis-for-quant-backtests-la2)
- [Databento pricing](https://databento.com/pricing)
- [Databento equities coverage](https://databento.com/equities)
- [Massive pricing](https://massive.com/pricing)
- [openassetpricing on PyPI](https://pypi.org/project/openassetpricing/)
- [JKP factors](https://jkpfactors.com/)
- [Ken French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_factors.html)
- [IBKR short securities availability](https://gdcdyn.interactivebrokers.com/en/trading/short-securities-availability.php?menu=B)

## Biases in historical data

### What it is

A **bias** is a systematic difference between what a backtest saw and what a trader at the time could have seen or done. The four below all flatter backtests, silently.

### Survivorship bias

**What it is.** A dataset built from companies that exist *today* omits every bankruptcy, delisting and acquisition in between.

**Why it matters here.** Without the dead companies, the longs never hold a blow-up they really would have held, and the short book's best and worst outcomes are both missing. Small, distressed and volatile names are both the likeliest to vanish and where anomalies look strongest, so survivorship and the microcap effect compound. The build plan requires delisted names and point-in-time constituents for any honest test (citing FINSABER).

**Guards in the code.** `survivorship_check` on the source; `ParquetStore` keeps delisted names (they just have no later rows); the universe is rebuilt point-in-time from data, not from a current index list.

### Delisting returns

**What it is.** When a stock is delisted, its final return (from the last close to what holders eventually receive) is often missing from price databases. Treated as zero or dropped, every bankruptcy becomes a free exit at the last traded price.

**The numbers.** Shumway (1997) showed that CRSP's missing performance-related delisting returns bias returns upward and proposed substituting −30% for NYSE/AMEX stocks; Shumway & Warther (1999) found the corresponding figure for Nasdaq stocks to be about −55%. In this code base, the 2026-11-04 audit found that the label builder dropped every delisting outcome: **0 of 630** pre-delisting labels survived on a constructed test, so ML training, IC and decile statistics never saw a single delisting.

**Guards in the code.** Two knobs, both defaulting to −30%:

- `AlphaConfig.delisting_return = -0.30` in `qts/alpha/targets.py::forward_returns`: if a name's last ever close falls inside the label window $(t, t+h]$, the label is the return to that last close, compounded with a −30% haircut. A temporary gap (a halt followed by more prices) is not a delisting.
- `CostConfig.delisting_haircut = -0.30` in the simulator: a held name with no close on $t+1$ earns −30% against the position's sign (a loss for longs and shorts alike, conservatively) and is liquidated at spread plus fees.

<callout icon="💡" color="blue_bg">
Why charge −30% to shorts too? A short in a company delisted for cause would profit, but one in a company acquired at a premium would lose, and price data cannot tell them apart, so the haircut goes against the position in both cases. Replace it with real delisting returns when the source has them.
</callout>

### Look-ahead from restatements

**What it is.** Companies sometimes restate published figures (a 10-K/A, or restated comparatives in a later filing). Databases that keep only the latest value silently replace the original, so a backtest uses numbers not public until later. Restatements cluster in companies with accounting problems, exactly the ones a quality signal is trying to identify.

**Guards in the code.** Use as-reported data (Sharadar dimensions `ARQ`, `ARY`, `ART`), not most-recent-reported (`MRQ`, `MRY`, `MRT`); `sharadar_sf1_to_pit_fundamentals` warns if an `MR*` dimension is requested. Derived year-over-year fields join the prior period by filing date with `merge_asof` (→ see **Fundamentals: Sharadar SF1 and the datekey** below).

### Filing date versus period end

**What it is.** A quarterly report describes a period (say, the quarter ending 31 March) but becomes public on its filing date: US companies have about 40–45 days after quarter end to file a 10-Q and 60–90 days after year end for a 10-K, depending on size. A database keyed on the period end shows the numbers 30–90 days early, which is worth a lot in a backtest, since prices react to earnings, and nothing live.

**Guards in the code.** Sharadar's `datekey` is the filing date; the loader keys every panel on `datekey + lag_days`. For sources that carry only period dates, the build plan prescribes a conservative 90-day lag (`lag_days ≈ 90`).

A fifth bias, universe-membership look-ahead, was the audit's most severe finding; → see **The eligibility panel and the "ever in universe" leak** below.

### Why the microcap tail is excluded rather than modelled

Anomalies concentrate in the smallest stocks and fade once those are removed or costs are charged:

| Evidence | Number | Source |
|---|---|---|
| Microcaps' share of total market value | ≈ 3% | UCLA Anderson Review |
| Post-earnings-announcement drift, 2001–2024 | t = 2.18 on all stocks; t = 1.43 excluding microcaps | UCLA Anderson Review; Martineau |
| Machine-learning predictability | Concentrated in hard-to-arbitrage stocks; excluding microcaps, distressed firms or high-volatility periods considerably weakens it | Avramov, Cheng & Metzker |
| Average of 204 anomalies, post-2005, net of costs | ≈ 4 bp per month equal-weighted; ≈ −2 bp per month value-weighted | Chen & Velikov |
| Market impact versus arrival, AQR live orders 1998–2011 | 11.21 bp large caps vs 21.27 bp small caps | Frazzini, Israel & Moskowitz |

All but the first row are out-of-sample or live. The universe rules below act on this: do not chase what cannot be captured net of costs.

### References

- [Shumway (1997), The Delisting Bias in CRSP Data](https://ideas.repec.org/a/bla/jfinan/v52y1997i1p327-40.html)
- [Shumway & Warther (1999), The Delisting Bias in CRSP's Nasdaq Data and Its Implications for the Size Effect](https://ideas.repec.org/a/bla/jfinan/v54y1999i6p2361-2379.html)
- [FINSABER](https://arxiv.org/html/2505.07078v3)
- [UCLA Anderson Review, Is post-earnings-announcement drift a thing again?](https://anderson-review.ucla.edu/is-post-earnings-announcement-drift-a-thing-again/)
- [Martineau, Rest in Peace Post-Earnings Announcement Drift](https://nowpublishers.com/article/Details/CFR-0122)
- [Avramov, Cheng & Metzker, Machine Learning vs. Economic Restrictions: Evidence from Stock Return Predictability](https://papers.ssrn.com/abstract=3450322)
- [Chen & Velikov, Zeroing in on the Expected Returns of Anomalies](https://core-prod.cambridgecore.org/core/services/aop-cambridge-core/content/view/945133D5A3ECEEAF466AEE91551FD225/S0022109022000874a.pdf/zeroing-in-on-the-expected-returns-of-anomalies.pdf)
- [Frazzini, Israel & Moskowitz, Trading Costs](https://conference.nber.org/conf_papers/f68262/f68262.pdf)

## The point-in-time store

### What it is

`qts/data/store.py::ParquetStore` is a local, file-based store for every dataset the system reads: a directory per dataset, Parquet files partitioned by year, and a DuckDB connection that exposes each dataset as a SQL view. Think of it as a tiny data warehouse on a laptop disk.

```text
root/
  prices_close/_meta.json              {"kind": "wide", "columns": [...], "partitioned": true}
  prices_close/year=2006/data.parquet  long rows: date, ticker, value   (zstd)
  prices_close/year=2007/data.parquet
  ...
  industry/_meta.json
  industry/all.parquet                 long table without a date: ticker, industry
```

### Why we need it here

Research reads the same panels thousands of times and must read the same bytes each time. A local store decouples ingest (a nightly writer) from research (many readers), makes runs reproducible through content hashes, and lets SQL answer ad-hoc questions ("which tickers closed below \$5 in 2009?") without loading everything into pandas.

### How it works

1. **Wide panels are stored long.** `write_wide(name, df)` stacks a date × ticker frame into `(date, ticker, value)` rows and **drops NaNs**. The ticker set can then change between writes, delisted names are simply missing rows, and every dataset shares one three-column schema.
2. **Partitioned by calendar year** into `year=YYYY/data.parquet` (zstd); reads with `start`/`end` open only partitions in range. Tables without dates (the industry map) go to `all.parquet`; `_meta.json` records kind, columns and partitioning.
3. **Reading.** `read_wide(name, start, end, tickers)` reads the partitions, filters and pivots back to the wide convention. `as_market_data(universe_tickers, start, end, extra_names)` reads the four required datasets and any optional panels, reindexes every panel to the close panel's dates and tickers, maps industry (missing → `"UNKNOWN"`) and validates the result. `write_market_data(md)` is the inverse.
4. **SQL.** `duckdb_query(sql)` opens an in-memory DuckDB connection with one view per dataset (`read_parquet` over the partition glob) and returns a DataFrame.
5. **Hashing.** `snapshot_hash(name)` is a SHA-256 over the relative paths and bytes of a dataset's files; `snapshot_hash_all()` combines them. These are the data identities for the trial ledger (→ see **Architecture & data flow › Reproducibility**).

### Why this choice

The build plan recommends Polars and DuckDB over local Parquet as the PIT store. Parquet is columnar, compressed and readable by every tool in the stack; DuckDB queries it in place with no server. Postgres or a cloud warehouse would add operational weight and cost to a once-a-day batch workload.

### In the code

Dataset names understood by `as_market_data`: `prices_close`, `prices_volume`, `market_cap`, `industry` (required); `spread_bps`, `short_interest_ratio`, `borrow_fee`, `book_to_price`, `earnings_yield`, `gross_profitability`, `asset_growth` (optional); any further wide dataset can be loaded into `MarketData.extra` via `extra_names`.

### Pitfalls & checks

<callout icon="⚠️" color="yellow_bg">
In the default mode `"overwrite_partitions"`, `write_long` / `write_wide` replace **every year partition the new data touches**. Writing one new day of 2026 prices overwrites the whole `year=2026` file with that single day. A nightly ingest must read the current year, append and write the full year back.
</callout>

- **The store is not bitemporal.** It holds one value per `(date, ticker)`; point-in-time correctness comes from writing panels already keyed on availability date (the loaders below do this). Keep raw vendor drops immutable and rebuild derived panels from them, or a vendor revision will silently change history.
- **Sparse panels are not forward-filled on read.** `as_market_data` reindexes without filling, so write `pit_forward_fill` output (already on the trading calendar), not raw filing-date rows. `pit_forward_fill` has no staleness cap: a company that stops filing keeps its last value (audit finding 16).

## Corporate actions and adjusted prices

### What it is

A **corporate action** changes a stock's price or share count for reasons other than trading: a **split** (each share becomes several and the price falls proportionally), a **cash dividend** (the price drops by about the dividend on the ex-dividend date), a spin-off, a merger, a ticker change. An **adjusted price** series rewrites history so that the ratio of consecutive prices equals the return a holder earned.

### Why we need it here

On raw prices a 2-for-1 split is a −50% day and every dividend a small loss, corrupting momentum, volatility, beta and labels. The system expects `close` to be a **total-return adjusted close** (splits and dividends).

### How it works

Adjustment walks backwards from the present. On each event date, all **earlier** prices are multiplied by a factor:

$$
P^{\text{adj}}_{s} = P_{s} \prod_{e \,:\, s < d_e \le T} k_e
$$

where:
- $P_s$ is the raw close on day $s$;
- $d_e$ is the effective (ex-) date of corporate action $e$, and $T$ the last date in the data;
- $k_e = 1/m$ for an $m$-for-1 split;
- $k_e = 1 - D_e / P_{d_e - 1}$ for a cash dividend $D_e$, using the raw close the day before the ex-date.

**Worked example: a split.** A stock closes at \$100.00 on Monday. A 2-for-1 split takes effect Tuesday and the stock closes at \$51.00. Raw return: $51/100 - 1 = -49\%$. Adjusted: Monday's price becomes $100 \times \tfrac{1}{2} = \$50.00$, so the return is $51/50 - 1 = +2\%$, which is what a holder of 1 old share (now 2 new shares worth \$102) actually earned.

**Worked example: a dividend.** A stock closes at \$50.00 the day before its ex-date and pays a \$0.50 dividend. On the ex-date it closes at \$49.70. Raw return: $49.70/50 - 1 = -0.6\%$. The holder's true return is $(49.70 + 0.50)/50 - 1 = +0.40\%$. With the adjustment factor $k = 1 - 0.50/50 = 0.99$, the prior close becomes \$49.50 and the adjusted return is $49.70/49.50 - 1 = +0.404\%$. The small difference from $0.400\%$ is the standard approximation of multiplicative adjustment (it treats the dividend as reinvested at the ex-date close).

Mergers and spin-offs are messier and vendors record them differently. Ticker changes and reused tickers need a permanent security identifier behind the ticker string (Sharadar's `permaticker`, for example); the build plan's Phase 0 includes a symbol-mapping layer.

### Why this choice

Total-return adjustment makes simple returns on `close` equal to holder returns. Raw prices plus a separate dividend stream would be more faithful, but every consumer would have to remember to add dividends back.

### In the code

The code assumes the price source has already adjusted `close`; there is no adjustment logic in `qts`. `yfinance_to_market_data` downloads with `auto_adjust=True`. `MarketData.returns` is `close.pct_change(fill_method=None)`, so gaps stay NaN instead of being bridged.

### Pitfalls & checks

<callout icon="⚠️" color="yellow_bg">
Adjusted prices are right for returns and wrong for levels. The \$5 screen (`qts/data/universe.py::raw_eligibility` compares `md.close` with `min_price`) and dollar volume (`MarketData.dollar_volume = close × volume`) both use the adjusted close, which lies below the prices that actually traded because every later split and dividend is divided out. A stock that traded at \$40 in 2006 and has since split 2-for-1 three times shows an adjusted 2006 price of \$5 or less, and can fail the price and ADV screens in a year when it was a liquid mid-cap. On real data, add an unadjusted close panel (for example in `MarketData.extra`) and use it for the price screen and for dollar volume.
</callout>

- **Per-share fundamentals need the matching price.** As-reported book value per share and EPS are not split-adjusted; dividing them by a split-adjusted close inflates book-to-price and earnings yield before every later split. Pass the unadjusted close to `sharadar_sf1_to_pit_fundamentals(close=...)`.
- **Retroactive revision.** Every new dividend rescales the adjusted history before it; never mix series downloaded on different days.
- **Shares outstanding from market cap.** `to_short_interest_ratio` needs shares outstanding. Deriving it as `market_cap / close` is only correct with the unadjusted close.

## Universe rules

### What it is

The **universe** is the set of stocks the strategy may trade on a given date. `qts/data/universe.py::build_universe_mask` returns a boolean wide panel that is `True` where a name is investable as of that date, using only information available on or before it. The rules, with defaults from `UniverseConfig`:

| Rule | Default | Parameter |
|---|---|---|
| Price at least | \$5 | `min_price = 5.0` |
| Average daily dollar volume at least | \$2,000,000 over 63 trading days | `min_adv_usd = 2_000_000.0`, `adv_window = 63` |
| Not in the bottom size quantile of names passing the screens above | bottom 20% excluded | `exclude_bottom_size_quantile = 0.20` |
| Within the top N by market cap among the survivors | N = 1,000 | `top_n_by_mcap = 1000` |
| Refresh frequency | every 21 trading days (≈ monthly) | `build_universe_mask(refresh_days=21)` |
| Hysteresis | a member leaves only after failing at 3 consecutive refreshes; a non-member enters as soon as it passes | `build_universe_mask(exit_after_n_fails=3)` |
| Must be priced | a name with no close at $t$ is never in the universe | always |

### Why we need it here

The universe is the first and cheapest risk control: it decides whether the strategy competes among liquid large companies, where costs are low and data clean, or among small illiquid ones, where anomalies look strongest but cannot be captured after costs. The build plan's specification (top ~1,000–1,500 by cap and ADV, no bottom NYSE size quintile, price ≥ \$5, PIT membership including delisted names) follows from the evidence above.

### How it works, rule by rule

**1. Price at least \$5.** Low-priced stocks are expensive per dollar traded. The \$0.01 tick is 0.2% of a \$5 stock but 1% of a \$1 stock, so percentage spreads are mechanically wider, and a \$0.0035 per-share commission is 7 bp of a \$5 trade but 0.35 bp of a \$100 trade. Many brokers restrict margin and shorting below \$5, and sub-\$5 stocks are disproportionately distressed.

**2. ADV at least \$2M.** A capacity screen. Trades are capped at 5% of ADV (`PortfolioConfig.max_adv_participation = 0.05`), so a \$2M ADV name allows \$100,000 of trading a day. With positions capped at 2% of NAV, a \$5M book (cap \$100,000) can build a full position in a day; a \$50M book (cap \$1M) needs ten days, while the forecast decays and impact accumulates. ADV is the trailing 63-day mean of `close × volume` with at least 31 observations (`min_periods = max(5, adv_window // 2)`), so a new listing becomes eligible after about six weeks at the earliest.

**3. Exclude the bottom 20% by size.** Among names passing the price and ADV screens, those below the per-date 20th percentile of market cap are dropped. This proxies the academic convention of excluding stocks below the 20th percentile of **NYSE** market caps ("NYSE breakpoints"); the proxy uses the eligible cross-section because exchange history is not in `MarketData`. It backs up the liquidity screen: liquid microcaps still carry the anomalies that vanish net of costs.

**4. Top N by market cap.** The 1,000 largest survivors are kept (ties broken by column order). This bounds problem size for the optimiser and the risk regressions, concentrates the book where data is best and costs lowest, and matches the build plan's 1,000–1,500.

**5. Monthly refresh.** Membership is recomputed every 21 trading days (rows 0, 21, 42, … of the date index) and held in between, because every exit forces a sale.

**6. Three-strike hysteresis.** At each refresh, a name that passes all screens is in. A current member that fails is *not* removed immediately: a failure counter increments, and the name leaves only when the counter reaches 3, i.e. at the third consecutive failed refresh. Passing resets the counter. In code:

```python
for i in range(R.shape[0]):
    passed = R[i]
    fails = np.where(passed, 0, fails + 1)
    in_u = passed | (in_u & (fails < exit_after_n_fails))
    out[i] = in_u
```

The point is turnover: a stock hovering at the boundary would otherwise enter and leave every month, each time being bought and then liquidated (ineligible holdings are unwound over `PortfolioConfig.exit_ineligible_days = 5`). Hysteresis applies to every screen, so membership can briefly exceed `top_n_by_mcap` and a grandfathered name can sit below \$5 for up to two refreshes. A name with no price is out immediately (`mask &= md.close.notna()`).

### Why this choice

An index's constituent list (S&P 500, Russell 1000) would need a point-in-time history and brings index rules irrelevant here; a rules-based universe is reproducible from data alone. Dropping the size floor and trusting the optimiser's cost model would extrapolate a model calibrated on liquid names into the microcap tail, where costs are large and uncertain. A 3,000-name universe adds breadth, but the extra names are small, expensive and where spurious alpha concentrates. Daily refresh without hysteresis generates turnover with no alpha behind it.

### In the code

- `raw_eligibility(md, cfg)`: per-day eligibility before refresh and hysteresis.
- `build_universe_mask(md, cfg=None, refresh_days=21, exit_after_n_fails=3)`: the PIT mask. `refresh_days` and `exit_after_n_fails` are function arguments, not `UniverseConfig` fields, so `ResearchPipeline` always uses the defaults.
- `apply_mask(md, mask)` sets `eligible` (next section); `universe_turnover(mask)` is the fraction of names entering or leaving per date.

To tune: scale `min_adv_usd` with NAV (the capacity arithmetic above); raise `exclude_bottom_size_quantile` for a more conservative book; lower `top_n_by_mcap` for faster research sweeps.

### Pitfalls & checks

- **Adjusted prices in the screens** (see the warning under **Corporate actions and adjusted prices**).
- **The size quantile is relative to the eligible set,** not NYSE breakpoints. In a period when many small names pass the ADV screen, the cut-off is lower than the NYSE convention would give.
- **Universe churn is a cost.** Check `universe_turnover` and the `forced_exits` diagnostics after any rule change.
- **The default synthetic market never exercises the universe** (300 names, `top_n_by_mcap = 1000`). Test eligibility logic with `top_n_by_mcap` well below the number of names (the audit used 120 of 300).

### Worked example: three toy stocks

Use a 5-day ADV window to keep the arithmetic short (the default is 63 days). Closes and share volumes over days 1–5:

| Stock | Closes | Volumes (thousand shares) | Market cap at $t$ |
|---|---|---|---|
| A | 40.00, 40.50, 41.00, 40.00, 40.50 | 100, 120, 80, 110, 90 | \$1.2B |
| B | 4.80, 4.90, 5.10, 5.00, 4.95 | 900, 900, 900, 900, 900 | \$0.3B |
| C | 22.00, 22.40, 21.80, 22.20, 22.00 | 70, 90, 60, 80, 75 | \$0.8B |

Daily dollar volume is close × volume:

- A: \$4.000M, \$4.860M, \$3.280M, \$4.400M, \$3.645M → sum \$20.185M → **ADV \$4.037M**.
- B: \$4.320M, \$4.410M, \$4.590M, \$4.500M, \$4.455M → sum \$22.275M → **ADV \$4.455M**.
- C: \$1.540M, \$2.016M, \$1.308M, \$1.776M, \$1.650M → sum \$8.290M → **ADV \$1.658M**.

Screens on day 5:

- A: price \$40.50 ≥ \$5 ✓; ADV \$4.04M ≥ \$2M ✓ → passes the base screens.
- B: ADV \$4.46M ✓, but price \$4.95 < \$5 ✗ → fails. Liquidity alone is not enough.
- C: price \$22.00 ✓, but ADV \$1.66M < \$2M ✗ → fails: a \$0.8B company can still be too thinly traded.

Now suppose ten names pass the base screens on day 5, with market caps (in \$B) of 0.6, 0.7, 0.9, **1.2 (A)**, 1.5, 2, 3, 5, 8, 20. The 20% quantile (pandas linear interpolation at position $0.2 \times (10-1) = 1.8$ in the sorted list) is $0.7 + 0.8 \times (0.9 - 0.7) = \$0.86\text{B}$. The 0.6 and 0.7 names are dropped; A survives. With the default `top_n_by_mcap = 1000`, A is eligible. With `top_n_by_mcap = 5`, the five largest survivors are 20, 8, 5, 3 and 2; A ranks seventh and is not eligible.

Hysteresis, from the constructed test `tests/test_data.py::test_universe_hysteresis_constructed` (refreshes on days 0, 21, 42, 63, 84, …; its stocks are unrelated to the three above). One stock falls below \$5 on days 42–83 and recovers: it fails at the refreshes on days 42 and 63 (counter 1, then 2), passes on day 84 (counter reset), and never leaves. A second falls below \$5 from day 42 for good: it fails on days 42, 63 and 84 and leaves on day 84, the third consecutive failure. A third stops trading on day 100 and leaves that day, whatever its counter.

### References

- [UCLA Anderson Review, Is post-earnings-announcement drift a thing again?](https://anderson-review.ucla.edu/is-post-earnings-announcement-drift-a-thing-again/)
- [Avramov, Cheng & Metzker, Machine Learning vs. Economic Restrictions: Evidence from Stock Return Predictability](https://papers.ssrn.com/abstract=3450322)
- [IBKR commissions](https://ndcdyn.interactivebrokers.com/en/pricing/commissions-stocks.php)

## The eligibility panel and the "ever in universe" leak

### What it is

`qts/data/universe.py::apply_mask(md, mask)` returns a new `MarketData` whose `eligible` field holds the per-date membership (`mask & close.notna()`). Volume, market cap, short interest, borrow fee, fundamentals and `extra` panels are set to NaN wherever the name is out of the universe. `close` is kept intact, so returns and labels stay computable for names that have just left and for held names being unwound. Tickers that are never in the universe are dropped from the columns, purely to save space. Every cross-sectional operation downstream must go through `MarketData.eligibility()` (in the universe at $t$ **and** priced at $t$).

### Why "ever in universe" leaks

After `apply_mask`, the column set is "every ticker that is ever in the universe during the sample", a list chosen with knowledge of the whole sample. Any cross-section computed over the column set, rather than over the names eligible at that date, contains stocks that are present *because of what they do later*. Names enter a cap-ranked universe by rising in price, so their pre-entry period is one of strong returns.

The 2026-11-04 audit found exactly this. An earlier version zero-filled features of out-of-universe names instead of dropping them, and since `close` is unmasked, every ever-member had price features, a label, an ML training row and an emitted score on every date. On a synthetic market of 300 names with `top_n_by_mcap = 120`, 21% of feature rows were outside the universe at their date, and **future entrants earned +110% annualised in the 63 days before entry against −6% for the universe average**. Mean IC was 0.0137 over all kept names versus 0.0234 over in-universe names only, so the leak also distorted the measurement of skill. It was rated CRITICAL and fixed by the eligibility panel: features, labels, IC and scores are computed over eligible names only and ineligible rows are dropped, and the simulator refuses to open ineligible names and unwinds held ones over 5 days. → see **Architecture & data flow › Point-in-time discipline** for the full list of guards.

### Pitfalls & checks

Any new cross-sectional computation must use `md.eligibility()` (or receive it as `valid`); `md.close.notna()` or the column set reintroduces the leak. `tests/test_pipeline.py::test_pipeline_held_names_are_eligible` checks the backtest's holdings. The leak is invisible whenever every name is always eligible, so test with a universe smaller than the ticker set.

## Short interest: FINRA data and the publication lag

### What it is

**Short interest** is the number of shares currently sold short. FINRA collects it twice a month, as of fixed **settlement dates**, and publishes it some days later. Short interest ÷ shares outstanding is the **short-interest ratio** (`MarketData.short_interest_ratio`); short interest ÷ average daily volume is **days to cover**.

### Why we need it here

It stands in for borrow-fee history until the IBKR logger has a year of data (review amendment 2), and feeds the `short_interest` alpha theme (negatively signed), the risk model's `short_interest` style, and the cost model's borrow-fee proxy (general collateral below a 5% ratio, rising convexly to a 25% annual cap at 20%; `CostConfig.borrow_proxy_si_gc`, `borrow_proxy_si_htb`, `htb_fee_cap_annual`).

### How it works

The value for settlement date $S$ is disseminated about 8 business days later, so a PIT panel reveals it only from then:

$$
\text{SIR}_{i,t} = \frac{\text{SI}_{i,S^*(t)}}{\text{SharesOut}_{i,t}}, \qquad S^*(t) = \max\{S : S + 8\ \text{business days} \le t\}
$$

where:
- $\text{SI}_{i,S}$ is the short interest reported for settlement date $S$;
- $S^*(t)$ is the latest settlement date already published at $t$;
- $\text{SharesOut}_{i,t}$ is shares outstanding as of $t$.

`parse_finra_short_interest` reads FINRA's pipe-delimited files (column aliases accepted, delimiter sniffed) into a long table; `finra_publication_date(S, lag_business_days=8)` gives the availability date; `to_short_interest_ratio` applies the lag, forward-fills with `pit_forward_fill` and divides by shares outstanding as of each date; `short_interest_long_to_wide` does the same for the raw count.

### Pitfalls & checks

- The lag uses pandas business days (no exchange holidays), and FINRA publishes after the close; the audit (finding 14) recommends FINRA's dissemination calendar plus one trading day.
- Shares outstanding must be the actual share count (see **Corporate actions and adjusted prices**).
- Twice-monthly data is stale by up to three weeks plus the lag; do not read it as a daily signal.

### References

- [FINRA, Equity Short Interest data](https://www.finra.org/finra-data/browse-catalog/equity-short-interest)
- [Engelberg et al. (2025), The Loan Fee Anomaly: A Short Seller's Best Ideas](https://ideas.repec.org/a/inm/ormnsc/v71y2025i7p5529-5551.html)

## Fundamentals: Sharadar SF1 and the datekey

### What it is

Sharadar's SF1 table has one row per company, reporting period and **dimension**. Three dates matter: `calendardate` (period end normalised to a calendar quarter), `reportperiod` (actual fiscal period end) and **`datekey`** (the SEC filing date, when the figures became public). As-reported dimensions (`ARQ` quarterly, `ARY` annual, `ART` trailing twelve months) keep what was filed; most-recent-reported ones (`MRQ`, `MRY`, `MRT`) overwrite history with restatements. Value, quality and investment, three of the six alpha themes, are built from it.

### How it works

`qts/data/loaders.py::sharadar_sf1_to_pit_fundamentals` (defaults `dimension="ARQ"`, `lag_days=0`):

1. Filters to the dimension (warning on `MR*`), sorts by `(ticker, datekey, calendardate)` and keeps the latest period among same-day filings.
2. Computes **asset growth** with a filing-date-aware join (below), so a restatement of last year's assets filed after the current report cannot leak into it.
3. Computes gross profitability as `gp / assets`.
4. Makes every field visible from `datekey + lag_days` via `pit_forward_fill` on the trading calendar.
5. Derives `book_to_price = bvps / close` and `earnings_yield = eps / close` daily if `close` is passed, otherwise `1 / pb` and `1 / pe` as of the filing date.

```python
merged = pd.merge_asof(cur, prev, left_on="datekey", right_on="_prev_dk", by=["ticker", "_prev_cd"],
                       direction="backward", allow_exact_matches=True)
```

`cur` holds the current rows with `_prev_cd` = their `calendardate` minus one year; `prev` holds every filing's assets keyed by its own filing date `_prev_dk`; `direction="backward"` picks the latest prior-year filing made on or before the current filing date.

### Why this choice

SF1's `datekey` is the only filing-date field at individual budgets (build plan §7). For sources with period dates only, the plan's fallback is a 90-day lag: conservative for large companies, roughly right for small companies' 10-Ks.

### Pitfalls & checks

- **After-hours filings.** With `lag_days = 0`, a filing dated $d$ is visible at the close of $d$, though many are filed after 16:00. Use `lag_days = 1` on real data.
- **Quarterly EPS.** With `ARQ`, `eps / close` is a quarterly yield; use `ART` for trailing-twelve-month ratios.
- **Unadjusted close for per-share ratios** (see **Corporate actions and adjusted prices**).

### References

- [QuantRocket, Sharadar fundamentals documentation](https://www.quantrocket.com/docs/data/fundamental/sharadar/)

## Industries: Fama–French 48 from SIC codes

### What it is

**SIC** (Standard Industrial Classification) codes are four-digit US industry codes carried by every SEC registrant. Fama and French group SIC ranges into **48 industries** (Food Products, Banking, Computers, Pharmaceutical Products, …); codes outside the 47 named groups fall into "Other" ("Almost Nothing"). Industries are used to demean value, quality and investment within industry, as risk factors, and for the optimiser's ±3% industry bands (`PortfolioConfig.industry_band`, resolved through `factor_groups["industry"]`).

### How it works

`FF48_DEFINITIONS` in `qts/data/loaders.py` lists `(short name, long name, [inclusive SIC ranges])`, taken from the French library's `Siccodes48` table. At import a 10,000-entry lookup array is built; `sic_to_ff48(code)` is one array lookup (invalid or unmapped → `"Other"`) and `ff48_from_sic_series` vectorises it. The labels go into the store's `industry` dataset.

### Why this choice

Review amendment 3 chose FF48 from SIC over **GICS** (the S&P/MSCI classification behind Barra's USE4). GICS history is proprietary and costly point-in-time; SIC is free in SEC filings and Sharadar's metadata; FF48 is the academic standard, comparable with French's industry portfolios. The build plan's Phase 0 still lists a "GICS mapping"; the code follows the amendment. The trade-off: SIC dates from 1987, lumps much of software and internet into Business Services, and is rarely updated by companies.

### Pitfalls & checks

- **One label per ticker for all history** (`slice_until` keeps `industry` whole): today's SIC applied to 2006 is a mild look-ahead.
- **"Other" is a catch-all;** check its size.
- **Labels no longer start with `IND`,** which is why the optimiser uses `factor_groups` instead of name prefixes.

### References

- [Ken French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)

## Factor benchmarks: Ken French data

### What it is

The Ken French library publishes daily and monthly returns of the standard academic factors: the five Fama–French factors (`Mkt-RF`, `SMB` size, `HML` value, `RMW` profitability, `CMA` investment, plus the risk-free rate `RF`), momentum and short- and long-term reversal. They are an independent yardstick: correlating the system's own momentum or value factor with `Mom` or `HML` checks data and construction end to end; regressing strategy returns on them shows how much of a result is a known premium; `RF` is the cash rate.

### How it works

`KEN_FRENCH_DATASETS` names five daily files. `parse_ken_french_csv` handles the layout (preamble, a header row starting with a comma, `YYYYMMDD` / `YYYYMM` / `YYYY` dates, several blocks in monthly files), converts percent to decimals and maps the `-99.99` / `-999` sentinels to NaN. `parse_ken_french_zip` opens the zip; `download_ken_french` fetches it (run locally; the build sandbox could not reach the site); `ken_french_factor_panel` joins five factors and momentum.

### Pitfalls & checks

French factors use all NYSE/AMEX/Nasdaq stocks with NYSE breakpoints, so a 1,000-name universe's factors correlate strongly but not perfectly. The files are revised occasionally; store a snapshot.

### References

- [Ken French Data Library, Fama/French factors](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_factors.html)

## The synthetic market

### What it is

`qts/data/synthetic.py::generate(SyntheticSpec(...))` builds a complete, seeded `MarketData` for $N$ stocks over $T$ days from a model whose every ingredient is known: factor returns, exposures, a planted predictable component of known strength, and noise. It is the ground truth for the test suite and the default input of `scripts/run_demo.py`.

### Why we need it here

Real data has no answer key. Planted truth lets tests check that the alpha layer recovers about the planted skill (less means lost signal, more means leakage), that the risk model recovers true betas and factor returns, and that the chain turns a known signal into net returns. → see **Architecture & data flow › Testing strategy**.

### How it works

Daily simple returns follow

$$
r_{i,t} = \beta_i\, m_t + \sum_{k=1}^{K} x_{i,k,t}\, f_{k,t} + a_{i,t} + \varepsilon_{i,t}
$$

clipped to $[-50\%, +100\%]$, where:

- $m_t$ is the **market return**, with GARCH-style volatility clustering: $h_t = 0.05\,\bar\sigma^2 + 0.10\, m_{t-1}^2 + 0.85\, h_{t-1}$ and $m_t = \sqrt{h_t}\, z_t + 0.06/252$, where $\bar\sigma = 0.16/\sqrt{252}$ (16% annual) and $z_t$ is standard normal;
- $\beta_i = \operatorname{clip}(1 + 0.35\,u_i - 0.1\, s_i,\ 0.2,\ 2.5)$ with $u_i$ standard normal and $s_i \sim N(0, 1.2^2)$ the stock's log-size draw (small stocks get slightly higher beta);
- $f_{k,t}$ are $K = 4$ **style factor returns** for size, value, quality and investment, each normal with 4% annual volatility (`style_vol_annual`) and a 2% annual premium (`style_premium_annual`);
- $x_{i,k,t}$ are the **exposures**: AR(1) paths in monthly steps with persistence 0.9, held constant within each 21-day block; the size exposure is fixed at $s_i / 1.2$;
- $a_{i,t}$ is the **planted alpha**, defined below;
- $\varepsilon_{i,t} \sim N(0, \sigma_i^2)$ is idiosyncratic noise, with annual volatility $0.30 \cdot \exp(0.35\,v_i - 0.15\, s_i)$ clipped to $[10\%, 120\%]$ ($v_i$ standard normal; small stocks are more volatile), converted to daily by $\sqrt{252}$.

About one **crash episode** per 600 days (after the first year) takes 0.4% a day from the market and 0.3% a day from value for 10 days, giving the risk model fat tails.

**The planted alpha.** A latent score $z_{i,t}$ follows a daily AR(1) whose 21-day autocorrelation is 0.80 (daily coefficient $0.8^{1/21}$), drawn on its own random stream. The score revealed at the close of $t$ is $z_{i,t}$; the alpha in the return of day $t$ loads on the score from `alpha_lag_days` earlier ($L = 1$ by default):

$$
a_{i,t} = \frac{\text{IC}_{\text{true}}\; \sigma_i\; z_{i,t-L}}{\sqrt{H}}
$$

where $\text{IC}_{\text{true}} = 0.04$ (`ic_true`), $\sigma_i$ is the stock's daily idiosyncratic volatility, $L$ is `alpha_lag_days` and $H = 20$ (`alpha_horizon_days`). So $z_t$ predicts $r_{t+1}, \dots, r_{t+H}$ and not $r_t$: anything observed at $t$ predicts strictly after $t$, as in reality. Summed over $H$ days with a constant score, the planted component is $\text{IC}_{\text{true}} \cdot \sigma_i\sqrt{H} \cdot z_i$, which is Grinold's rule ($\alpha = \text{IC} \times \text{volatility} \times \text{score}$) at the 20-day horizon; since $H$-day idiosyncratic noise has standard deviation $\sigma_i\sqrt{H}$, the correlation of $z$ with the 20-day residual return is close to $\text{IC}_{\text{true}}$. Score decay within the window and factor noise bring the realised residual rank IC to about 0.85–0.9 × `ic_true` ($0.036 \pm 0.006$ over seeds 0–4, per the generator's docstring; synthetic, in-sample). `tests/test_synthetic_oracle.py` checks the timing and that IC lands in a band below 0.04. `alpha_lag_days = 0` reproduces the older same-month generator.

**Observables.** The models never see $z$ directly; it leaks into characteristics with noise, as real alpha leaks into accounting ratios. For style $k$, the observable at $t$ is $o_{i,k,t} = x_{i,k,t} + 0.4\, z_{i,t} + 0.5\, \eta_{i,t}$ (the score revealed at $t$), with $\eta$ a persistent (monthly) measurement noise scaled by `characteristic_noise = 1.0`, and:

| Panel | Built from | What it is for |
|---|---|---|
| `close` | $100 \times$ cumulative product of $(1 + r)$ | Returns, momentum, volatility, labels |
| `market_cap` | close × fixed share count $e^{s_i + 17.2}$ (median ≈ \$3B) | Size screen, size factor, cap weights |
| `volume` | turnover ≈ 0.8% of shares a day ($e^{-4.8}$) with log-normal noise | ADV, liquidity features, impact |
| `spread_bps` | $(1 + 12(1 - \text{size rank})^2) \times$ noise: ≈ 1 bp (largest) to ≈ 13 bp (smallest) | Spread cost |
| `short_interest_ratio` | $0.03 + 0.03\, e^{0.8\, q}$, $q$ a persistent monthly AR(1) | Short-interest theme, borrow proxy |
| `borrow_fee` | $0.003 + 2\max(\text{SIR} - 0.08, 0)^{1.5}$, clipped to $[0.003, 1]$ | Holding cost of shorts |
| `book_to_price` | $\exp(0.5\, o_{\text{value}})$ | Value theme and style |
| `earnings_yield` | $0.05 + 0.03\, o_{\text{value}}$ | Value theme |
| `gross_profitability` | $0.3 + 0.1\, o_{\text{quality}}$ | Quality theme and style |
| `asset_growth` | $0.1 - 0.1\, o_{\text{investment}}$ | Investment theme (low growth is good) |
| `industry` | 12 random labels `IND00`–`IND11` | Industry demeaning and factors |
| `extra["true_latent_score"]` | $z$ | Oracle for IC recovery and planted backtests |
| `extra["true_beta"]`, `["true_market_return"]` | $\beta_i$, $m_t$ | Risk-model recovery tests |
| `extra["true_style_returns"]` | the first style factor's return (size) | Factor-return recovery tests |

### Why this choice

Every latent style is observable through a characteristic: an earlier version with unobservable "momentum" and "low-risk" factors made optimised books run at 1.8 times ex-ante risk (`docs/REVIEW_NOTES.md`), a real lesson that motivates the statistical companion model but a bad property for a fixture.

### In the code

`SyntheticSpec` defaults: `n_stocks = 300`, `n_days = 1512` (six years), `start = "2018-01-02"`, `n_industries = 12`, `market_vol_annual = 0.16`, `style_vol_annual = 0.04`, `style_premium_annual = 0.02`, `idio_vol_annual_median = 0.30`, `ic_true = 0.04`, `alpha_horizon_days = 20`, `characteristic_noise = 1.0`, `seed = 0`, `alpha_lag_days = 1`. Dates are business days (`pd.bdate_range`). `STYLE_FACTORS = ["size", "value", "quality", "investment"]`.

### Pitfalls & checks: what the synthetic market cannot test

- **No delistings or IPOs, and a static universe by default** (300 names, `top_n_by_mcap = 1000`). `survivorship_check` always warns on it, and the "ever in universe" leak was invisible for this reason. Delisting and eligibility paths need hand-built fixtures.
- **No reporting lags or restatements;** PIT loaders are tested separately on fixtures.
- **Planted-score timing.** With the default `alpha_lag_days = 1` the score is lagged and fair. The legacy setting `alpha_lag_days = 0` lets a monthly score drive returns in the same month it is revealed: PIT-fair but optimistic (audit finding 8); use it only to reproduce old results.
- **No industry return factor.** Industry labels are random, so industry risk factors have nothing to find.
- **No corporate actions, constant share counts, volume unrelated to returns,** and no momentum, low-risk or short-interest premium beyond what the planted score's persistence induces.

Results on the synthetic market are plumbing checks, not evidence about alpha.

## Roadmap

Not yet in the code (from `README.md`, `docs/REVIEW_NOTES.md`, `docs/WORK_PLAN.md`): Phase 0 ingest of Norgate prices and Sharadar SF1 into a `ParquetStore` with a symbol-mapping layer; the daily IBKR borrow-rate logger, spliced into `borrow_fee` after 12 months; reproducing five `openassetpricing` factor series from own data as a PIT audit; property-based PIT tests over the universe mask and loaders (item 1.1).
