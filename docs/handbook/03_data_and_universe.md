# Data & universe

<callout icon="🧭" color="gray_bg">
This page covers everything that happens before a model sees a number: which data a long/short equity strategy needs and why, which vendors supply it and what they cost, the biases hidden in historical data (survivorship, delisting returns, restatements, filing dates) and how large they are, the point-in-time Parquet/DuckDB store, corporate actions and adjusted prices, and the rules that define the investable universe (price at least \$5, ADV at least \$2M, no bottom-20% size names, top 1,000 by market cap, refreshed monthly with three-strike hysteresis), with the reasoning behind each rule. It explains the eligibility panel and why an "ever in the universe" stock list leaks, the FINRA short-interest publication lag, Sharadar's filing-date key, Fama–French 48 industries from SIC codes, the Ken French factor files, and the synthetic market generator that the test suite is built on.
</callout>

<callout icon="🆕" color="green_bg">
<b>What changed (Nov 2026)</b>
- `MarketData` gained `close_raw` (as-traded close), `shares_outstanding`, `industry_history` with `industry_as_of(t)`, `delisting_dates`, and `trade_price()`; the \$5 screen, dollar volume and ADV now use as-traded prices.
- The universe refreshes on the first trading day of each calendar month (`UniverseConfig.refresh_anchor = "month_start"`), so membership no longer depends on the load start date.
- `apply_mask` no longer blanks volume, market cap and fundamentals outside the universe; membership acts only through `md.eligibility()`, and `masked_levels` gives the old view to the risk model.
- `ParquetStore` writes are upserts by default (the old mode silently dropped the rest of each year), with per-dataset file locks and an explicit NaN policy.
- FINRA's publication lag is counted in NYSE sessions (`qts/data/calendar.py`); the short-interest ratio uses shares as of the settlement date.
- Sharadar defaults are now `dimension="ART"`, `lag_days=1`; B/P and E/P are split-consistent; late restatements of older periods no longer replace the current period.
- Labels tell halts from delistings causally (`delisting_dates`, else a 63-day gap rule); the synthetic generator can plant splits, halts and delistings, and its alpha timing and realised IC band are documented.
- First real source: `qts/data/sharadar.py` loads the Sharadar Core US Equities Bundle (prices, caps, PIT fundamentals, corporate actions, S&P 500 membership) into the store, keyed on permanent identifiers, with `qts data sharadar sync` / `validate` (→ see **Real data: Sharadar**).
</callout>

## The data a strategy needs

### What it is

Every model in the system reads one object, `qts/core/types.py::MarketData`: a set of aligned date × ticker tables ("wide panels", → see **Architecture & data flow › Data layout: wide and long panels**) plus one industry label per ticker. The table lists each panel, why it is needed, and when a value for date $t$ may become visible.

| Panel (`MarketData` field) | Units | Why it is needed | Visible at $t$ when… |
|---|---|---|---|
| `close` (required) | USD, split- and dividend-adjusted | Returns, momentum, volatility, beta, labels | the close of $t$ prints |
| `close_raw` | USD, as traded | Price screen, dollar volume, per-share fees and fundamentals (via `trade_price()`) | the close of $t$ |
| `volume` (required) | actual shares traded | ADV screen, liquidity features, impact | the close of $t$ |
| `shares_outstanding` | actual share count | Short-interest ratio; split-consistent B/P and E/P | the date the count is effective |
| `market_cap` (required) | USD | Size screen, size factor, cap weights | shares known at $t$ × close of $t$ |
| `industry` (required) | Fama–French 48 label | Static fallback label | not dated (see pitfalls) |
| `industry_history` | dated labels | Industry demeaning, factors, bands via `industry_as_of(t)` | the reclassification date |
| `delisting_dates` | ticker → last trading date | Telling delistings from halts in labels | the delisting date |
| `delisting_reasons` | ticker → `acquisition` / `performance` / `voluntary` / `unknown` | The final return of a delisted name in labels and backtest | the delisting date (see **Sharadar › Delisting reasons** for the caveat) |
| `spread_bps` | half-spread, bp | Spread cost; liquidity feature | the close of $t$ |
| `short_interest_ratio` | short interest ÷ shares | Short-interest theme and style; borrow proxy | settlement + 8 NYSE sessions |
| `borrow_fee` | annual fraction | Cost of shorts; loan-fee signal | the day the rate is logged |
| `book_to_price`, `earnings_yield` | ratio | Value theme and style | the report's filing date + 1 day |
| `gross_profitability` | gross profit ÷ assets | Quality theme | the filing date + 1 day |
| `asset_growth` | year-over-year fraction | Investment theme | the later report's filing date + 1 day |
| `eligible` | boolean | The point-in-time universe | built PIT from the above |
| `extra` | any wide panel | Additional data; synthetic ground truth | per panel |

### Why each kind is needed

Four points the table compresses. Prices must be adjusted for anything compared across time, or a 2-for-1 split looks like a 50% crash, but *unadjusted* for anything that is a level on one date, or a stock that later split looks like a penny stock in its past (→ see **Corporate actions and adjusted prices**). Fundamentals must be dated by when they became public, not by the period they describe. Short interest and borrow fees serve both the cost model (hard-to-borrow shorts are expensive) and the alpha model (equity loan fees were the most persistent non-text predictor in the build plan's evidence).

### In the code

`ParquetStore.as_market_data` assembles a `MarketData` from stored datasets (`REQUIRED = ("prices_close", "prices_volume", "market_cap", "industry")`, plus `OPTIONAL_PANELS`, the raw-price panels `prices_close_raw` and `shares_outstanding`, `industry_history` and `delisting_dates`); `qts/data/synthetic.py::generate` builds one from the synthetic model; `qts/data/loaders.py::yfinance_to_market_data` builds a prototyping one from Yahoo Finance. `MarketData.validate()` checks that `volume`, `market_cap`, `close_raw` and `shares_outstanding` align with `close`, that dates ascend and that every ticker has an industry label. `qts run --store PATH` (or `scripts/run_real.py --store PATH`) runs the full pipeline on a store. → see **Operating guide › Loading real data into the ParquetStore**.

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

**Guards in the code.** The delisting assumption is one map, reason → return on the stock, in `CostConfig` (`delisting_return_acquisition = 0.0`, `delisting_return_performance = -0.30`, `delisting_return_voluntary = -0.10`, `delisting_return_unknown = -0.30`), used by both the labels and the backtest (→ see **Backtesting › Delistings by reason**). The reason of each name is `MarketData.delisting_reasons`, derived from the vendor's corporate actions (→ see **Sharadar › Delisting reasons**); a name without one is `unknown`:

- In the labels (`qts/alpha/targets.py::forward_returns`): if a name has no close at the label window's end, the label is the return to its last close before the end, compounded with its reason's return **if that is a delisting** (so an acquisition's label is simply the return to the last price), and without it if it is a temporary halt.
- In the simulator: a held name with no close on $t+1$ is classified with the same rule below. If it is a delisting it earns its reason's return on the stock (a long loses on a negative return, a short gains) and is liquidated at spread plus fees; if it is a halt the position is frozen at its last price until prices return.

**Halts versus delistings, without looking ahead.** A trading halt (pending news, a regulatory halt, a long suspension) also produces missing closes, then prices resume. Deciding "halt or delisting" by peeking at whether prices come back later is a look-ahead: the second audit found that the earlier rule (a backward fill over the whole panel) made labels "realised" at $t$ depend on prices after $t$, verified on 4,228 label cells (HIGH finding 1). The rule is now causal:

1. If `MarketData.delisting_dates` covers the name, it is a delisting if and only if its last trading date is on or before the window end; decided at the window end.
2. Otherwise, with $G$ = `AlphaConfig.delisting_gap_days = 63` trading days after the last close: it is a halt if a close reappears within $G$ days (decided on the day it reappears), and a delisting if none does (decided $G$ days after the last close). Until one of those happens the label is NaN and the row is not used for training or IC.

`qts/alpha/targets.py::label_realisation` gives, per decision date, the first date at which every label of that row is decided, so training and the trailing IC use a row only from then on. Without a delisting table, a delisting delays its rows by up to 63 days; supply `delisting_dates` for real data.

<callout icon="⚠️" color="yellow_bg">
The simulator applies the same rule on the day it has to act. A held name with no close is frozen at its last price (no return, no trading) while it is a halt; the delisting return and liquidation are booked when the vendor delisting date has passed or, without a vendor table, when the gap reaches 63 trading days. Without `delisting_dates` a real delisting therefore sits frozen in the book for up to 63 days before its loss is booked (→ see **Backtesting › Halts and delistings**).
</callout>

<callout icon="🤔" color="purple_bg">
Why not one −30% haircut against every position, as before 2026-11-12? Because the reason is in the data, and it changes the answer. In a large-cap universe most delistings are acquisitions that complete at about the last traded price; charging those 30% on both sides cost the first real-data run 3.53% a year, more than its gross alpha. Shumway's −30% is a return on the stock for performance delistings: a long loses it and a short gains it. `CostConfig.delisting_against_position = True` restores the old rule; `delisting_short_gain_haircut` scales how much of a short's gain is booked.
</callout>

### Look-ahead from restatements

**What it is.** Companies sometimes restate published figures (a 10-K/A, or restated comparatives in a later filing). Databases that keep only the latest value silently replace the original, so a backtest uses numbers not public until later. Restatements cluster in companies with accounting problems, exactly the ones a quality signal is trying to identify.

**Guards in the code.** Use as-reported data (Sharadar dimensions `ARQ`, `ARY`, `ART`), not most-recent-reported (`MRQ`, `MRY`, `MRT`); `sharadar_sf1_to_pit_fundamentals` warns if an `MR*` dimension is requested. Derived year-over-year fields join the prior period by filing date with `merge_asof` (→ see **Fundamentals: Sharadar SF1 and the datekey** below).

### Filing date versus period end

**What it is.** A quarterly report describes a period (say, the quarter ending 31 March) but becomes public on its filing date: US companies have about 40–45 days after quarter end to file a 10-Q and 60–90 days after year end for a 10-K, depending on size. A database keyed on the period end shows the numbers 30–90 days early, which is worth a lot in a backtest, since prices react to earnings, and nothing live.

**Guards in the code.** Sharadar's `datekey` is the filing date; the loader keys every panel on `datekey + lag_days`, with `lag_days = 1` by default because a filing dated $d$ may be released after that day's close. For sources that carry only period dates, the build plan prescribes a conservative 90-day lag (`lag_days ≈ 90`).

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
  .locks/prices_close.lock             one advisory lock file per dataset
  prices_close/_meta.json              {"kind": "wide", "columns": [...], "partitioned": true}
  prices_close/year=2006/data.parquet  long rows: date, ticker, value   (zstd)
  prices_close/year=2007/data.parquet
  ...
  industry/_meta.json
  industry/all.parquet                 long table without a date: ticker, industry
  delisting_dates/all.parquet          long table: ticker, delisting_date
  delisting_reasons/all.parquet        long table: id, delisting_date, reason, source_action
  earnings_dates/year=2019/...         long table: date (announcement), ticker, source, datekey, ...
  earnings_fundamentals/year=2019/...  long table: date (datekey), ticker, dimension, eps, split_cum, ...
```

### Why we need it here

Research reads the same panels thousands of times and must read the same bytes each time. A local store decouples ingest (a nightly writer) from research (many readers), makes runs reproducible through content hashes, and lets SQL answer ad-hoc questions ("which tickers closed below \$5 in 2009?") without loading everything into pandas.

### How it works

1. **Wide panels are stored long.** `write_wide(name, df)` stacks a date × ticker frame into `(date, ticker, value)` rows and stores no NaN cells. The ticker set can then change between writes, delisted names are simply missing rows, and every dataset shares one three-column schema.
2. **Partitioned by calendar year** into `year=YYYY/data.parquet` (zstd); reads with `start`/`end` open only partitions in range. Tables without dates (the industry map, delisting dates) go to `all.parquet`; `_meta.json` records kind, columns and partitioning.
3. **Writes merge by default.** Three modes (`mode=` on `write_long` / `write_wide`):

| Mode | What happens to a year partition the new data touches | Use it for |
|---|---|---|
| `"upsert"` (default) | stored rows whose key (`date, ticker`) appears in the new data are replaced; every other stored row is kept | nightly appends, vendor corrections |
| `"replace_partition"` | rewritten with only the new rows | an intentional full rewrite of those years |
| `"replace"` | the whole dataset is dropped and rewritten | rebuilding a derived panel from raw drops |

For wide panels under upsert, `nan_policy` decides what a NaN cell means: `"keep_existing"` (default) treats it as "no information" and leaves any stored value in place, so a partial re-delivery of a sparse panel (short interest, fundamentals) cannot delete data; `"delete"` makes a NaN cell delete the stored value. Tables without a date column are upserted on `ticker` if present, otherwise replaced.
4. **Atomic, locked writes.** Every Parquet file is written to a temporary file in the same directory and moved into place with `os.replace`, so a crash never leaves a truncated partition. Every write of a dataset holds an exclusive `fcntl.flock` on `<root>/.locks/<dataset>.lock` and every read the shared lock, so a reader never sees a half-rewritten dataset and concurrent upserts from several processes never lose rows (`tests/test_store_concurrency.py`).
5. **Reading.** `read_wide(name, start, end, tickers)` reads the partitions, filters and pivots back to the wide convention. `as_market_data(universe_tickers, start, end, extra_names)` reads the four required datasets and any optional panels, reindexes every panel to the close panel's dates and tickers, maps the static industry (missing → `"UNKNOWN"`), attaches `industry_history` (with an `"UNKNOWN"` row for every ticker not yet classified at the panel start, never the static map) and `delisting_dates`, and validates the result. `write_market_data(md)` is the inverse.
6. **SQL.** `duckdb_query(sql)` opens an in-memory DuckDB connection with one view per dataset (`read_parquet` over the partition glob) and returns a DataFrame.
7. **Hashing.** `snapshot_hash(name)` is a SHA-256 over the relative paths and bytes of a dataset's files; `snapshot_hash_all()` combines them. The pipeline's own data identity is the in-memory hash of the assembled `MarketData` (→ see **Architecture & data flow › Reproducibility**).

### Why this choice

The build plan recommends Polars and DuckDB over local Parquet as the PIT store. Parquet is columnar, compressed and readable by every tool in the stack; DuckDB queries it in place with no server. Postgres or a cloud warehouse would add operational weight and cost to a once-a-day batch workload.

### In the code

Dataset names understood by `as_market_data`: `prices_close`, `prices_volume`, `market_cap`, `industry` (required); `prices_close_raw`, `shares_outstanding`, `industry_history` (wide, string labels), `delisting_dates` (long: `ticker, delisting_date`), `delisting_reasons` (long: `id, delisting_date, reason, source_action` → `MarketData.delisting_reasons`; absent means every delisting is `unknown`), `spread_bps`, `short_interest_ratio`, `borrow_fee`, `book_to_price`, `earnings_yield`, `gross_profitability`, `asset_growth` (optional); any further wide dataset can be loaded into `MarketData.extra` via `extra_names`. The full ingest recipe is in **Operating guide › Loading real data into the ParquetStore**.

### Pitfalls & checks

<callout icon="⚠️" color="yellow_bg">
The old default mode `"overwrite_partitions"` replaced **every year partition the new data touched**, so writing one new day of 2026 prices left the `year=2026` file holding that single day. The mode is deprecated: it still works, maps to `"replace_partition"` and warns. Use the default `"upsert"` for appends.
</callout>

- **Locks are local.** `flock` is advisory and not reliable on NFS; keep the store on a local disk.
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

The source should also supply the **as-traded** close as `close_raw` (store dataset `prices_close_raw`) and the actual share count as `shares_outstanding`. Then `MarketData.trade_price()` returns `close_raw` and `MarketData.dollar_volume` is `close_raw × volume`, with `volume` in actual shares traded that day. Everything that is a *level* on one date reads these: the \$5 screen and the ADV screen in `qts/data/universe.py::raw_eligibility`, the cost model's ADV and per-share fees (`TransactionCostModel.snapshot`), the liquidity features, and the risk model's turnover descriptor. When `close_raw` is missing, `trade_price()` falls back to the adjusted close, which is historically wrong across splits.

### Pitfalls & checks

<callout icon="⚠️" color="yellow_bg">
Adjusted prices are right for returns and wrong for levels. A stock that traded at \$40 in 2006 and has since split 2-for-1 three times shows an adjusted 2006 price of \$5, and on the adjusted close it fails the price screen (and, with share volume left unadjusted, understates its dollar volume eightfold) in a year when it was a liquid mid-cap. The code now screens on `trade_price()`, so this only bites when a source provides no `close_raw`. Load one for real data.
</callout>

- **Per-share fundamentals need one share basis.** As-reported book value per share and EPS are on the share count *of the filing*. Dividing them by a split-adjusted close inflates the ratio before every later split; dividing them by today's raw close is still wrong by the split factor between a split and the next filing (second audit, finding 7: B/P read 0.8 instead of 0.4 after a 2:1 split). `sharadar_sf1_to_pit_fundamentals` now forms totals: book value = BVPS × shares as of the filing date, divided by today's market cap, so a split leaves the ratio unchanged (→ see **Fundamentals: Sharadar SF1 and the datekey**).
- **Retroactive revision.** Every new dividend rescales the adjusted history before it; never mix series downloaded on different days.
- **Shares outstanding from market cap.** `qts/data/loaders.py::shares_outstanding_panel(md)` returns `md.shares_outstanding`, else `market_cap / close_raw`, and raises if neither exists rather than dividing by the split-adjusted close.
- **The synthetic market can plant splits.** `SyntheticSpec(n_splits=k)` plants $k$ forward splits in `close_raw`, `shares_outstanding` and `volume` while leaving `close`, `market_cap` and dollar volume bit-identical, so tests can check that nothing level-based reads the adjusted close.

### What the loader does **not** clean, and why cleaning lives downstream

The Sharadar loader (and every other loader) stores the vendor's prices as delivered: it does not drop, cap or repair a daily return. Real panels nevertheless contain a handful of absurd returns — a restated adjusted close that moved one day and not the next, a sub-penny print where the tick size alone is a ±50 % "return", a split applied to one side of a date — and the first real-data run showed what one of them does to an unprotected estimator: a single return on 2018-03-05 drove the risk model's ex-ante vol from 5 % to 5,110,263 % and emptied the book for four years (`docs/notes/risk_outlier_fix_2026-11-10.md`).

The fix is deliberately *not* in the loader, for three reasons. First, what a bad return means depends on the consumer: for the risk regression it is a cell to drop (and genuine fat tails are something to winsorise), for the label builder it is a missing price, for the simulator it is a halt day, and for TCA it is a price that must stay as traded. Second, a loader that silently edits prices makes the store irreproducible against the vendor's files and hides the data problem from `validate`; the store should be a faithful copy with a hash. Third, the thresholds are a modelling decision that belongs in `SystemConfig` (hashed, in the ledger) rather than in a one-off at sync time. So the store keeps the raw cells, `qts/core/returns.py` defines the rules once (data-error caps, estimation-only price floor and cross-sectional winsorisation, with flags), and each consumer applies the subset that is right for it (→ see **Risk model › Return cleaning and the estimation universe**, **Backtesting › Data-error prices**). `python scripts/find_outlier_returns.py --store data/store ...` lists the worst cells in a store with a likely-cause tag, so a vendor problem is reported rather than quietly absorbed.

## Universe rules

### What it is

The **universe** is the set of stocks the strategy may trade on a given date. `qts/data/universe.py::build_universe_mask` returns a boolean wide panel that is `True` where a name is investable as of that date, using only information available on or before it. The rules, with defaults from `UniverseConfig`:

| Rule | Default | Parameter |
|---|---|---|
| As-traded price at least | \$5 | `min_price = 5.0` |
| Average daily dollar volume (as-traded price × shares) at least | \$2,000,000 over 63 trading days | `min_adv_usd = 2_000_000.0`, `adv_window = 63` |
| Not in the bottom size quantile of names passing the screens above | bottom 20% excluded | `exclude_bottom_size_quantile = 0.20` |
| Within the top N by market cap among the survivors | N = 1,000 | `top_n_by_mcap = 1000` |
| Refresh dates | the first trading day of each calendar month, plus the panel's first date | `refresh_anchor = "month_start"` (`"panel_start"`: every `refresh_days = 21` from the first date) |
| Hysteresis | a member leaves only after failing at 3 consecutive refreshes; a non-member enters as soon as it passes | `exit_after_n_fails = 3` |
| Must be priced | a name with no close at $t$ is never in the universe | always |

### Why we need it here

The universe is the first and cheapest risk control: it decides whether the strategy competes among liquid large companies, where costs are low and data clean, or among small illiquid ones, where anomalies look strongest but cannot be captured after costs. The build plan's specification (top ~1,000–1,500 by cap and ADV, no bottom NYSE size quintile, price ≥ \$5, PIT membership including delisted names) follows from the evidence above.

### How it works, rule by rule

**1. Price at least \$5.** The price is the as-traded close (`md.trade_price()`), the price a trader actually saw that day. Low-priced stocks are expensive per dollar traded. The \$0.01 tick is 0.2% of a \$5 stock but 1% of a \$1 stock, so percentage spreads are mechanically wider, and a \$0.0035 per-share commission is 7 bp of a \$5 trade but 0.35 bp of a \$100 trade. Many brokers restrict margin and shorting below \$5, and sub-\$5 stocks are disproportionately distressed.

**2. ADV at least \$2M.** A capacity screen. Trades are capped at 5% of ADV (`PortfolioConfig.max_adv_participation = 0.05`), so a \$2M ADV name allows \$100,000 of trading a day. With positions capped at 2% of NAV, a \$5M book (cap \$100,000) can build a full position in a day; a \$50M book (cap \$1M) needs ten days, while the forecast decays and impact accumulates. ADV is the trailing 63-day mean of `md.dollar_volume` (as-traded price × actual shares traded) with at least 31 observations (`min_periods = max(5, adv_window // 2)`), so a new listing becomes eligible after about six weeks at the earliest.

**3. Exclude the bottom 20% by size.** Among names passing the price and ADV screens, those below the per-date 20th percentile of market cap are dropped. This proxies the academic convention of excluding stocks below the 20th percentile of **NYSE** market caps ("NYSE breakpoints"); the proxy uses the eligible cross-section because exchange history is not in `MarketData`. It backs up the liquidity screen: liquid microcaps still carry the anomalies that vanish net of costs.

**4. Top N by market cap.** The 1,000 largest survivors are kept (ties broken by column order). This bounds problem size for the optimiser and the risk regressions, concentrates the book where data is best and costs lowest, and matches the build plan's 1,000–1,500.

**5. Monthly refresh.** Membership is recomputed on the first trading day of every calendar month (and on the panel's first date, so the mask is defined from the start) and held in between, because every exit forces a sale. Anchoring the grid to the calendar matters for reproducibility: the old grid (rows 0, 21, 42, … of the date index, still available as `refresh_anchor = "panel_start"`) depended on where the data happened to start, so loading from 2006-01-03 or from 2006-01-10 gave different universes for the same date (second audit, finding 18). The hysteresis state still needs `exit_after_n_fails` refreshes to forget the start.

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
- `refresh_positions(dates, refresh_days, anchor)`: the refresh grid.
- `build_universe_mask(md, cfg=None, refresh_days=None, exit_after_n_fails=None)`: the PIT mask. The two keyword arguments default to `UniverseConfig.refresh_days` (used only with the `"panel_start"` anchor) and `UniverseConfig.exit_after_n_fails`, so `ResearchPipeline` honours the config.
- `apply_mask(md, mask)` sets `eligible` (next section); `masked_levels(md)` gives the risk model its "NaN outside the universe" view; `universe_turnover(mask)` is the fraction of names entering or leaving per date.

To tune: scale `min_adv_usd` with NAV (the capacity arithmetic above); raise `exclude_bottom_size_quantile` for a more conservative book; lower `top_n_by_mcap` for faster research sweeps.

### Pitfalls & checks

- **No `close_raw`, adjusted screens.** Without an as-traded close the screens fall back to the adjusted close (see the warning under **Corporate actions and adjusted prices**).
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

Hysteresis, from the constructed test `tests/test_data.py::test_universe_hysteresis_constructed`, which uses the legacy `refresh_anchor = "panel_start"` grid so that refreshes fall on days 0, 21, 42, 63, 84, … (its stocks are unrelated to the three above; with the default month-start grid the logic is identical, only the refresh dates differ). One stock falls below \$5 on days 42–83 and recovers: it fails at the refreshes on days 42 and 63 (counter 1, then 2), passes on day 84 (counter reset), and never leaves. A second falls below \$5 from day 42 for good: it fails on days 42, 63 and 84 and leaves on day 84, the third consecutive failure. A third stops trading on day 100 and leaves that day, whatever its counter.

### References

- [UCLA Anderson Review, Is post-earnings-announcement drift a thing again?](https://anderson-review.ucla.edu/is-post-earnings-announcement-drift-a-thing-again/)
- [Avramov, Cheng & Metzker, Machine Learning vs. Economic Restrictions: Evidence from Stock Return Predictability](https://papers.ssrn.com/abstract=3450322)
- [IBKR commissions](https://ndcdyn.interactivebrokers.com/en/pricing/commissions-stocks.php)

## The eligibility panel and the "ever in universe" leak

### What it is

`qts/data/universe.py::apply_mask(md, mask)` returns a new `MarketData` whose `eligible` field holds the per-date membership (`mask & close.notna()`). **No panel is masked**: prices, volume, market cap, shares outstanding, fundamentals, short interest and `extra` keep their full history. Tickers that are never in the universe are dropped from the columns, purely to save space (`industry_history` is column-restricted the same way; `extra` panels whose columns are not tickers pass through). Universe membership acts only through `MarketData.eligibility()` (in the universe at $t$ **and** priced at $t$), which every cross-sectional operation downstream must use.

Why not blank the levels outside the universe, as an earlier version did? Because the history before a name enters the universe was knowable at the time. Blanking it gave every new entrant NaN 63-day turnover, Amihud illiquidity and dollar volume (and risk liquidity) for about 40 days, which were then filled with the cross-sectional median, and computed its cost-snapshot ADV from post-entry days only (second audit, finding 6; `tests/test_apply_mask_history.py`). A consumer that still wants the old view, such as the risk model whose estimation universe is "has a market cap", calls `masked_levels(md)`, which blanks `market_cap`, `volume` and `shares_outstanding` where the name is not eligible.

### Why "ever in universe" leaks

After `apply_mask`, the column set is "every ticker that is ever in the universe during the sample", a list chosen with knowledge of the whole sample. Any cross-section computed over the column set, rather than over the names eligible at that date, contains stocks that are present *because of what they do later*. Names enter a cap-ranked universe by rising in price, so their pre-entry period is one of strong returns.

The 2026-11-04 audit found exactly this. An earlier version zero-filled features of out-of-universe names instead of dropping them, and since `close` was unmasked, every ever-member had price features, a label, an ML training row and an emitted score on every date. On a synthetic market of 300 names with `top_n_by_mcap = 120`, 21% of feature rows were outside the universe at their date, and **future entrants earned +110% annualised in the 63 days before entry against −6% for the universe average**. Mean IC was 0.0137 over all kept names versus 0.0234 over in-universe names only, so the leak also distorted the measurement of skill. It was rated CRITICAL and fixed by the eligibility panel: features, labels, IC and scores are computed over eligible names only and ineligible rows are dropped, and the simulator refuses to open ineligible names and unwinds held ones over 5 days. → see **Architecture & data flow › Point-in-time discipline** for the full list of guards.

### Pitfalls & checks

Any new cross-sectional computation must use `md.eligibility()` (or receive it as `valid`); `md.close.notna()`, `md.market_cap.notna()` or the column set reintroduces the leak, and since `apply_mask` no longer blanks levels, "has a market cap" no longer implies "in the universe". The lag investigation found exactly this kind of slip in the risk model: style exposures were standardised (and beta estimated against a market proxy) over names that join the universe later; they are now restricted to names eligible at each date. `tests/test_pipeline.py::test_pipeline_held_names_are_eligible` checks the backtest's holdings. The leak is invisible whenever every name is always eligible, so test with a universe smaller than the ticker set.

## Short interest: FINRA data and the publication lag

### What it is

**Short interest** is the number of shares currently sold short. FINRA collects it twice a month, as of fixed **settlement dates**, and publishes it some days later. Short interest ÷ shares outstanding is the **short-interest ratio** (`MarketData.short_interest_ratio`); short interest ÷ average daily volume is **days to cover**.

### Why we need it here

It stands in for borrow-fee history until the IBKR logger has a year of data (review amendment 2), and feeds the `short_interest` alpha theme (negatively signed), the risk model's `short_interest` style, and the cost model's borrow-fee proxy (general collateral below a 5% ratio, rising convexly to a 25% annual cap at 20%; `CostConfig.borrow_proxy_si_gc`, `borrow_proxy_si_htb`, `htb_fee_cap_annual`).

### How it works

The value for settlement date $S$ is disseminated about 8 trading sessions later, so a PIT panel reveals it only from then. The ratio is formed per report, with the share count *as of the settlement date* (the basis FINRA's short position is reported on), and the ratio is what is carried forward:

$$
\text{SIR}_{i,t} = \frac{\text{SI}_{i,S^*(t)}}{\text{SharesOut}_{i,S^*(t)}}, \qquad S^*(t) = \max\{S : S + 8\ \text{NYSE sessions} \le t\}
$$

where:
- $\text{SI}_{i,S}$ is the short interest reported for settlement date $S$, in shares;
- $S^*(t)$ is the latest settlement date already published at $t$;
- $\text{SharesOut}_{i,S}$ is the actual share count on (or last before) $S$.

Forming the ratio at the settlement date keeps it split-consistent: if the stock splits 2-for-1 between two reports, the share count doubles but the reported short position does not until the next report, so dividing by today's shares would halve the ratio for up to a month (second audit, finding 8).

**Why NYSE sessions and not business days.** `pandas.offsets.BDay` skips weekends but not exchange holidays, so it is early by one session for every holiday the lag spans. Settlement on 2024-12-31 plus `BDay(8)` gives 2025-01-10; on the NYSE calendar, which was closed on 1 January and 9 January 2025 (the national day of mourning for President Carter), it is 2025-01-14, two sessions later. `qts/data/calendar.py` is a dependency-free NYSE calendar for 2000–2035: the ten full-day holidays with their observed-day rules (New Year's Day on a Saturday is *not* observed on the Friday), Good Friday from the Easter computus, Juneteenth from 2022, and the unscheduled closures (11–14 September 2001, 11 June 2004, 2 January 2007, 29–30 October 2012, 5 December 2018, 9 January 2025). Early 13:00 closes count as sessions. `add_sessions(dates, n)` does the arithmetic; dates outside the range raise.

`parse_finra_short_interest` reads FINRA's pipe-delimited files (column aliases accepted, delimiter sniffed) into a long table; `finra_publication_date(S, lag_business_days=8)` gives the availability date via `add_sessions`; `to_short_interest_ratio(long_df, shares_outstanding, lag_business_days=8)` forms the ratio per report, applies the lag and forward-fills with `pit_forward_fill`; `short_interest_long_to_wide` does the same for the raw count.

### Pitfalls & checks

- FINRA publishes after the close; the first audit (finding 14) recommends FINRA's dissemination calendar plus one session. The 8-session default approximates it; raise `lag_business_days` to 9 to be conservative.
- `shares_outstanding` must be a wide panel of actual share counts (`MarketData.shares_outstanding`, or `shares_outstanding_panel(md)`); the function refuses anything else.
- Twice-monthly data is stale by up to three weeks plus the lag; do not read it as a daily signal.
- FINRA short interest is **not** part of feature set 2 of the alpha model: the history available would cover only part of 2003–2026 (deferred; → see **Alpha model › Feature set 2**).

### References

- [FINRA, Equity Short Interest data](https://www.finra.org/finra-data/browse-catalog/equity-short-interest)
- [Engelberg et al. (2025), The Loan Fee Anomaly: A Short Seller's Best Ideas](https://ideas.repec.org/a/inm/ormnsc/v71y2025i7p5529-5551.html)

## Fundamentals: Sharadar SF1 and the datekey

### What it is

Sharadar's SF1 table has one row per company, reporting period and **dimension**. Three dates matter: `calendardate` (period end normalised to a calendar quarter), `reportperiod` (actual fiscal period end) and **`datekey`** (the SEC filing date, when the figures became public). As-reported dimensions (`ARQ` quarterly, `ARY` annual, `ART` trailing twelve months) keep what was filed; most-recent-reported ones (`MRQ`, `MRY`, `MRT`) overwrite history with restatements. Value, quality and investment, three of the six alpha themes, are built from it.

### How it works

`qts/data/loaders.py::sharadar_sf1_to_pit_fundamentals(sf1_df, dimension="ART", lag_days=1, dates=None, fields=None, close=None, close_raw=None, price_is_adjusted=True, shares_outstanding=None, market_cap=None)`:

1. Filters to the dimension (warning on `MR*`), sorts by `(ticker, datekey, calendardate)` and keeps the latest period among same-day filings.
2. **Restatements of older periods.** Per ticker, a row whose `calendardate` is older than the latest period already filed (as of its `datekey`), such as a 10-K/A for last year filed after this year's first 10-Q, is dropped from the level panels: otherwise the older period would show as "current" from its filing date (second audit, finding 10: gross profitability fell back from 0.20 to the restated prior-year 0.11). It still feeds the prior-year lookup. A restatement of the *current* period replaces it.
3. Computes **asset growth** with a filing-date-aware join (below), so a restatement of last year's assets filed after the current report cannot leak into it.
4. Computes gross profitability as `gp / assets`.
5. Makes every field visible from `datekey + lag_days` (calendar days, snapped to the next panel date) via `pit_forward_fill`.
6. Derives the valuation ratios, preferring **totals on one share basis**:

$$
\text{B/P}_{i,t} = \frac{\text{BVPS}_{i,f} \times \text{Shares}_{i,f}}{\text{MarketCap}_{i,t}}, \qquad \text{E/P}_{i,t} = \frac{\text{EPS}_{i,f} \times \text{Shares}_{i,f}}{\text{MarketCap}_{i,t}}
$$

where:
- $f$ is the latest filing visible at $t$, and $\text{BVPS}_{i,f}$, $\text{EPS}_{i,f}$ are its as-reported per-share values;
- $\text{Shares}_{i,f}$ is the actual share count as of the filing's `datekey` (the `shares_outstanding` panel, else SF1's `sharesbas` of that row);
- $\text{MarketCap}_{i,t}$ is the point-in-time market cap (the `market_cap` panel, else `close_raw` × `shares_outstanding`).

BVPS × shares is the total book value the filing reports, and market cap is unaffected by splits, so a split after the filing leaves the ratio unchanged. Without share data the legacy path divides per-share values by the as-traded close (`close_raw`), still wrong by the split factor between a split and the next filing; passing an adjusted `close` raises unless you assert `price_is_adjusted=False`. With no price at all the ratios fall back to the vendor's `1 / pb` and `1 / pe`.

**Worked example.** A filing on 5 May reports BVPS \$20 with 100M shares (book value \$2B). On 20 May the stock trades at \$50, so market cap is \$5B and B/P $= 2/5 = 0.40$. On 1 June a 2-for-1 split takes effect: price \$25, 200M shares, market cap still \$5B. The totals path still gives $20 \times 100\text{M} / \$5\text{B} = 0.40$. The legacy per-share path gives $20 / 25 = 0.80$ until the next filing reports BVPS \$10.

```python
merged = pd.merge_asof(cur, prev, left_on="datekey", right_on="_prev_dk", by=["ticker", "_prev_cd"],
                       direction="backward", allow_exact_matches=True)
```

`cur` holds the current rows with `_prev_cd` = their `calendardate` minus one year; `prev` holds every filing's assets keyed by its own filing date `_prev_dk`; `direction="backward"` picks the latest prior-year filing made on or before the current filing date.

### Why this choice

SF1's `datekey` is the only filing-date field at individual budgets (build plan §7). For sources with period dates only, the plan's fallback is a 90-day lag: conservative for large companies, roughly right for small companies' 10-Ks.

### Pitfalls & checks

- **After-hours filings.** The default `lag_days = 1` makes a filing dated $d$ usable at the close of $d+1$, because many 10-Qs and 8-Ks are filed after 16:00. `lag_days = 0` assumes every filing was public before the close of its date (optimistic).
- **Dimension.** The default `ART` (as-reported trailing twelve months) makes `eps`-based earnings yield and `gp/assets` annual and free of seasonality. With `ARQ`, flow items are single quarters, so E/P is about a quarter of the annual figure and seasonal; use it only when you need quarter-on-quarter changes.
- **Pass share data.** Supply `shares_outstanding` and `market_cap` (or `close_raw`) so the split-consistent path is used.

### References

- [QuantRocket, Sharadar fundamentals documentation](https://www.quantrocket.com/docs/data/fundamental/sharadar/)

## Industries: Fama–French 48 from SIC codes

### What it is

**SIC** (Standard Industrial Classification) codes are four-digit US industry codes carried by every SEC registrant. Fama and French group SIC ranges into **48 industries** (Food Products, Banking, Computers, Pharmaceutical Products, …); codes outside the 47 named groups fall into "Other" ("Almost Nothing"). Industries are used to demean value, quality and investment within industry, as risk factors, and for the optimiser's ±3% industry bands (`PortfolioConfig.industry_band`, resolved through `factor_groups["industry"]`).

### How it works

`FF48_DEFINITIONS` in `qts/data/loaders.py` lists `(short name, long name, [inclusive SIC ranges])`, taken from the French library's `Siccodes48` table. At import a 10,000-entry lookup array is built; `sic_to_ff48(code)` is one array lookup (invalid or unmapped → `"Other"`) and `ff48_from_sic_series` vectorises it. The labels go into the store's `industry` dataset.

**Parent sectors: Fama–French 12.** On a top-1,000 universe many of the 48 industries are thin — on the Sharadar run 22,983 (date, industry) cells had exactly one eligible member (Ships, Smoke, RlEst, Rubbr, Aero, Mines, Other, …) — and a one-member industry breaks the risk model's regression (→ see **Risk model › 4b. Thin industries**). The risk model therefore shrinks a thin industry's factor return toward its parent sector, French's **12-industry** classification (`Siccodes12`). French defines both on SIC ranges directly; `FF48_TO_FF12` in `qts/data/loaders.py` is the crosswalk that assigns each of the 48 to the sector holding the bulk of its SIC codes, following French's own sector descriptions. Five industries straddle two sectors (Hshld, ElcEq, LabEq, PerSv, Toys) and go to the majority sector.

| FF12 sector | French's description | FF48 members |
|---|---|---|
| `NoDur` | Consumer non-durables: food, tobacco, textiles, apparel, leather, toys | Agric, Food, Soda, Beer, Smoke, Toys, Books, Clths, Txtls |
| `Durbl` | Consumer durables: cars, TVs, furniture, household appliances | Hshld, Autos |
| `Manuf` | Manufacturing: machinery, trucks, planes, office furniture, paper, commercial printing | Rubbr, Steel, FabPr, Mach, ElcEq, Aero, Ships, Guns, Paper, Boxes |
| `Enrgy` | Oil, gas and coal extraction and products | Coal, Oil |
| `Chems` | Chemicals and allied products | Chems |
| `BusEq` | Business equipment: computers, software, electronic equipment | Comps, Chips, LabEq |
| `Telcm` | Telephone and television transmission | Telcm |
| `Utils` | Utilities | Util |
| `Shops` | Wholesale, retail and some services (laundries, repair shops) | Whlsl, Rtail, PerSv |
| `Hlth` | Healthcare, medical equipment and drugs | Hlth, MedEq, Drugs |
| `Money` | Finance | Banks, Insur, RlEst, Fin |
| `Other` | Mines, construction, building materials, transport, hotels, business services, entertainment | BldMt, Cnstr, Gold, Mines, Trans, Meals, BusSv, Fun, Other |

A label that is not in the table (the synthetic market's `IND00..IND11`) has the whole industry block as its parent. `tests/test_risk_thin_industry.py::test_ff48_to_ff12_is_a_complete_crosswalk` pins the table to the 48 names and 12 sectors.

### Why this choice

Review amendment 3 chose FF48 from SIC over **GICS** (the S&P/MSCI classification behind Barra's USE4). GICS history is proprietary and costly point-in-time; SIC is free in SEC filings and Sharadar's metadata; FF48 is the academic standard, comparable with French's industry portfolios. The build plan's Phase 0 still lists a "GICS mapping"; the code follows the amendment. The trade-off: SIC dates from 1987, lumps much of software and internet into Business Services, and is rarely updated by companies.

### Pitfalls & checks

- **Dated labels, if you have them.** `MarketData.industry` is one label per ticker for all history, so today's SIC applied to 2006 is a mild look-ahead. Supply `industry_history` (dated labels, store dataset `industry_history`) and every consumer reads `md.industry_as_of(t)`: the alpha layer's industry demeaning (`qts/alpha/features.py::industry_segments`), the label residualisation and the risk model's industry factors (`qts/risk/exposures.py::industry_panel`). The risk model also carries, at each date, only the industry columns already used by an eligible name by then, so a label introduced later cannot appear as an all-zero factor before it exists.
- **"Other" is a catch-all;** check its size.
- **Labels no longer start with `IND`,** which is why the optimiser uses `factor_groups` instead of name prefixes.

### References

- [Ken French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
- [Ken French, Detail for 48 industry portfolios (Siccodes48) and 12 industry portfolios (Siccodes12)](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_48_ind_port.html)

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

where $\text{IC}_{\text{true}} = 0.04$ (`ic_true`), $\sigma_i$ is the stock's daily idiosyncratic volatility, $L$ is `alpha_lag_days` and $H = 20$ (`alpha_horizon_days`). So $z_t$ predicts $r_{t+1}, \dots, r_{t+H}$ and not $r_t$: anything observed at $t$ predicts strictly after $t$, as in reality (the return on $t$ loads on $z_{t-1}$, and $z_t$ correlates with it only through the score's persistence). Summed over $H$ days with a constant score, the planted component is $\text{IC}_{\text{true}} \cdot \sigma_i\sqrt{H} \cdot z_i$, which is Grinold's rule ($\alpha = \text{IC} \times \text{volatility} \times \text{score}$) at the 20-day horizon; since $H$-day idiosyncratic noise has standard deviation $\sigma_i\sqrt{H}$, the correlation of $z$ with the 20-day residual return is close to $\text{IC}_{\text{true}}$. `alpha_lag_days = 0` reproduces the older generator bit for bit (a monthly-step score driving the returns of the same month it is revealed in).

**The realised IC band.** Score decay within the window (mean 20-day autocorrelation about 0.94) and the residualisation and factor noise mean that the realised Spearman IC of $z_t$ against the residualised 20-day forward return is about 0.85–0.9 × `ic_true`. The generator's docstring records the band on the default spec (300 names, 1,512 days, seeds 0–4; synthetic, full-sample):

| `ic_true` | Residual rank IC, mean ± sd (range) | Raw forward rank IC |
|---|---|---|
| 0.04 | 0.036 ± 0.006 (0.029–0.044) | 0.037 |
| 0.08 | 0.066 ± 0.007 (0.059–0.075) | 0.069 |

`tests/test_synthetic_oracle.py` checks the timing and this band. The alpha model should recover at most this, never more: an estimated IC above the band is a leak until proven otherwise.

**Observables.** The models never see $z$ directly; it leaks into characteristics with noise, as real alpha leaks into accounting ratios. For style $k$, the observable at $t$ is $o_{i,k,t} = x_{i,k,t} + 0.4\, z_{i,t} + 0.5\, \eta_{i,t}$ (the score revealed at $t$), with $\eta$ a persistent (monthly) measurement noise scaled by `characteristic_noise = 1.0`, and:

| Panel | Built from | What it is for |
|---|---|---|
| `close` | $100 \times$ cumulative product of $(1 + r)$ | Returns, momentum, volatility, labels |
| `close_raw`, `shares_outstanding` | equal to `close` and the fixed share count unless splits are planted | Level-based screens, split tests |
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
| `extra["true_style_factor_returns"]` | all four style factor returns, date × factor (not a ticker panel) | Factor-return recovery tests |
| `extra["true_style_returns"]` | legacy key: the size factor's return tiled across tickers | Older tests |
| `extra["true_splits"]`, `["true_halts"]`; `delisting_dates` | planted events (when enabled) | Split, halt and delisting tests |

### Why this choice

Every latent style is observable through a characteristic: an earlier version with unobservable "momentum" and "low-risk" factors made optimised books run at 1.8 times ex-ante risk (`docs/REVIEW_NOTES.md`), a real lesson that motivates the statistical companion model but a bad property for a fixture.

### In the code

`SyntheticSpec` defaults: `n_stocks = 300`, `n_days = 1512` (six years), `start = "2018-01-02"`, `n_industries = 12`, `market_vol_annual = 0.16`, `style_vol_annual = 0.04`, `style_premium_annual = 0.02`, `idio_vol_annual_median = 0.30`, `ic_true = 0.04`, `alpha_horizon_days = 20`, `characteristic_noise = 1.0`, `seed = 0`, `alpha_lag_days = 1`, `n_splits = 0`, `n_halts = 0`, `halt_days = 10`, `n_delistings = 0`. Dates are business days (`pd.bdate_range`, no exchange holidays). `STYLE_FACTORS = ["size", "value", "quality", "investment"]`.

**Planted corporate events** (each on its own random stream, so with all three at 0 the output is bit-identical to the generator without them):

- `n_splits` forward splits (ratios cycling 2:1, 3:1) on random names and dates: before a $k$:1 split the as-traded price is $k \times$ `close` and the share count and shares traded are divided by $k$, so dollar volume and market cap do not move. Recorded in `extra["true_splits"]`.
- `n_halts` names lose `halt_days` consecutive days of every price-level panel; the price path continues underneath, so the first close after the halt carries the whole halt-period return. Recorded in `extra["true_halts"]`.
- `n_delistings` other names lose every price-level panel from a random day to the end, and `md.delisting_dates` is set (last trading date for those names, `NaT` for every other name).

### Pitfalls & checks: what the synthetic market cannot test

- **No IPOs, and a static universe by default** (300 names, `top_n_by_mcap = 1000`). Without `n_delistings`, `survivorship_check` warns on it; the "ever in universe" leak was invisible for this reason. Eligibility paths need a universe smaller than the ticker set, and halts and delistings must be switched on explicitly.
- **No reporting lags or restatements;** PIT loaders are tested separately on fixtures.
- **Planted-score timing.** With the default `alpha_lag_days = 1` the score is lagged and fair. The legacy setting `alpha_lag_days = 0` lets a monthly score drive returns in the same month it is revealed: PIT-fair but optimistic (audit finding 8); use it only to reproduce old results.
- **No industry return factor.** Industry labels are random, so industry risk factors have nothing to find.
- **No dividends, constant share counts apart from planted splits, volume unrelated to returns,** characteristics that refresh daily with $z_t$ (optimistic against quarterly fundamentals), and no momentum, low-risk or short-interest premium beyond what the planted score's persistence induces.

Results on the synthetic market are plumbing checks, not evidence about alpha.

## Real data: Sharadar

### What it is

The **Sharadar Core US Equities Bundle** (sold through Nasdaq Data Link) is the first real data source wired into the system. One subscription carries delisting-complete daily prices for every US-listed stock since 1998, point-in-time fundamentals keyed on the SEC filing date, a security master with a permanent identifier, daily market capitalisations, corporate actions and the history of S&P 500 membership. `qts/data/sharadar.py` downloads its tables, maps them onto the store's dataset names and checks the result; `qts data sharadar sync` and `qts data sharadar validate` are the commands (→ see **Operating guide › Loading Sharadar**).

| Table | Rows | What the loader makes of it | Store datasets |
|---|---|---|---|
| `TICKERS` | ~30k | security master, kept universe, panel ids, industry labels, delisting dates | `security_master`, `industry`, `delisting_dates` |
| `SEP` | ~20M | adjusted and as-traded closes, actual shares traded | `prices_close`, `prices_close_raw`, `prices_volume` |
| `DAILY` | ~15M | market cap and valuation ratios (USD millions) | `sharadar_marketcap`, `sharadar_ev`, `sharadar_pb`, `sharadar_pe`, `sharadar_ps`; derived `market_cap`, `shares_outstanding` |
| `SF1` | ~3M | as-reported fundamentals (`ART` and `ARQ` kept) | `sharadar_sf1`; derived `book_to_price`, `earnings_yield`, `gross_profitability`, `asset_growth` |
| `ACTIONS` | ~300k | splits, dividends, listings, delistings, mergers (reference); the reason of each delisting | `corporate_actions`; derived `delisting_reasons` |
| `SP500` | ~2k | index additions and removals since 1957 | `sp500_membership` (+ `sp500_membership_panel`) |
| `METRICS`, `EVENTS`, `SFP`, `INDICATORS` | optional | stored verbatim (`--tables`) | `sharadar_<table>` |

Row counts are order-of-magnitude expectations for the full history; the first real download fixes them.

### The kept universe

`TICKERS` has one row per security *and per table it appears in*, so the loader keeps the `SEP` rows and then filters on two columns (both configurable through `MasterFilters`):

- `category` in {`Domestic Common Stock`, `Domestic Common Stock Primary Class`}. ADRs are foreign issuers with different filings, currencies and settlement; Canadian listings, preferred stock, ETFs, ETNs, closed-end funds, units and warrants are not the common-equity cross-section the alpha and risk models describe; a **secondary share class** (`... Secondary Class`) would put one company's capitalisation into the panel twice, so only the primary class is kept.
- `exchange` in {`NYSE`, `NASDAQ`, `NYSEMKT`, `BATS`}. OTC names have no closing auction and unreliable prints; `NYSEARCA` is where ETFs list; `INDEX` rows are not stocks.

What survives is roughly 8,000–10,000 names alive at any date and perhaps 20,000 over the history, delisted companies included.

### Identifiers: one id per company, not per ticker string

Sharadar writes a company's whole history under its **latest** ticker, and a ticker can later be reused by a different company (a different `permaticker`, with a disjoint `[firstpricedate, lastpricedate]`). The build plan's "symbol-mapping layer on permanent identifiers" is therefore needed from day one, and the loader implements it as follows:

- The panel **id** is the Sharadar ticker when exactly one kept `permaticker` ever used it, and `TICKER.PERMATICKER` (for example `DDD.100004` and `DDD.100005`) for **every** holder when several did. Most names keep their plain ticker; a reused ticker is split into as many ids as it had holders, and no price series ever merges two companies.
- Price and valuation rows (`SEP`, `DAILY`) are resolved by `(ticker, date)` against each holder's price-date range; the alive holder's range is open-ended. A print dated while nobody held the ticker is a data error and is dropped (counted in the sync report).
- Filings (`SF1`) are resolved by `(ticker, datekey)`. A company files before its first price date (IPO filings), so a filing in a gap is assigned to the **next** holder; the sync report counts these too.
- Rows whose ticker is not in the kept master are dropped and counted (ETFs, ADRs, OTC names with prices in `SEP`).

<callout icon="⚠️" color="yellow_bg">
Sharadar tickers may themselves contain a dot (`BRK.A`). An id's suffix is a permaticker only when it is all digits; `security_master` holds the full mapping (`id`, `permaticker`, `ticker`, `name`, exchange, category, both industry labels, price-date range, delisting flag, CUSIPs, location).
</callout>

### Adjustments and units

The three price fields are not interchangeable, and the loader uses each for one thing:

- `closeadj` is split- **and** dividend-adjusted → `prices_close` (returns, momentum, volatility, labels).
- `closeunadj` is the price on the tape → `prices_close_raw` (the \$5 screen, dollar volume, per-share fees).
- `open`, `high`, `low`, `close`, `volume` are **split-adjusted but not dividend-adjusted**. The store wants actual shares traded, so the loader undoes the split adjustment:

$$
V^{\text{actual}}_t = V^{\text{SEP}}_t \times \frac{\text{close}_t}{\text{closeunadj}_t}
$$

where:
- $V^{\text{SEP}}_t$ is the vendor's split-adjusted volume;
- $\text{close}_t$ is the vendor's split-adjusted close and $\text{closeunadj}_t$ the as-traded close.

Before an $m$-for-1 split, $\text{close}_t = \text{closeunadj}_t / m$ and the adjusted volume is $m$ times the shares that actually changed hands, so the ratio undoes the adjustment exactly. **Worked example:** a stock trades 1,000 shares at \$100 the day before a 2-for-1 split. Sharadar shows `close = 50`, `closeunadj = 100`, `volume = 2000`; the loader stores $2000 \times 50 / 100 = 1000$ shares. Where `closeunadj` is missing or zero the vendor volume is kept and the row counted (`validate` reports the fraction).

`DAILY.marketcap` and `DAILY.ev` are in **USD millions** (`DAILY_USD_SCALE = 1e6`): `market_cap = marketcap × 10⁶` and `shares_outstanding = market_cap / closeunadj` on the same day, so cap, raw price and share count are consistent across splits. On days without a `DAILY` row the latest SF1 `sharesbas` filed on or before the date is carried forward and `market_cap = closeunadj × shares`. `validate` checks the unit two ways: the largest market cap in the store must be in the $10^{11}$–$2 \times 10^{13}$ USD range, and the implied share count must match SF1's `sharesbas` at filing dates (median ratio near 1).

SF1 amounts are plain USD and shares. The fundamentals panels are built by `sharadar_sf1_to_pit_fundamentals` with the defaults described above (`ART`, `lag_days = 1`, totals on one share basis), so a value is visible from `datekey + 1` calendar day snapped to the next trading day, never earlier. When a filing predates the first price in the shares panel (a company's first filings, or filings before 1998) the filing's own `sharesbas` supplies the share count, which is known at its `datekey`. `validate` re-derives `gross_profitability` for a sample of 50 names from the stored SF1 rows and confirms nothing appears before `datekey + 1`.

### Industries

`TICKERS.famaindustry` already carries the Fama–French 48 industry as a long name ("Banking", "Business Services", "Pharmaceutical Products"). `FF48_NAMES` uses the short names ("Banks", "BusSv", "Drugs"), so `map_famaindustry` normalises the vendor label and looks it up in `FAMAINDUSTRY_TABLE` (built from `FF48_DEFINITIONS` plus Sharadar's own spellings, notably "Restaraunts Hotels Motels" → `Meals`); a name without a usable label falls back to `sic_to_ff48(siccode)` and then to `"Other"`. The sync report prints how many names took each path and warns when more than 10% land in "Other".

TICKERS carries only the **current** classification, so no `industry_history` is written and the static map applies for the whole history. FF48 reclassifications are rare, so this is a mild look-ahead; `build_industry_history` is the hook for a dated SIC source (EDGAR filing headers carry the SIC at each filing date).

### Delistings, corporate actions and the S&P 500

- `delisting_dates`: `lastpricedate` for `isdelisted = "Y"`, `NaT` for alive names, so **every** kept id is covered and the label builder and simulator use the vendor rule rather than the 63-day gap rule. `validate` cross-checks the dates against `ACTIONS` rows of type `delisted` (within 5 days) without overriding TICKERS.
- `corporate_actions` keeps every `ACTIONS` row (date, action, ticker, id, value, contra ticker) for reference; `validate` confirms that raw-close jumps of more than 40% with a continuous adjusted close line up with `split` actions.
- `delisting_reasons`: one row per delisted id with its reason, derived from `corporate_actions` (next subsection).
- `earnings_dates` and `earnings_fundamentals`: earnings announcements and the as-reported quarterly record, derived from `sharadar_sf1`, the optional `EVENTS` table and the splits in `corporate_actions` (→ see **Earnings dates and the quarterly record** below).
- `sp500_membership` stores the `SP500` events (`added`, `removed`, plus the vendor's `current` and `historical` listings). `sp500_membership_panel(store)` turns the add/remove events into a date × id boolean panel (member on `[added, removed)`, events before the panel start carried in). It is stored and exposed only: **it is not a universe rule** and nothing in the pipeline reads it yet.

### Delisting reasons
**What it is.** A derived reference table, `delisting_reasons`: one row per delisted id with `id`, `delisting_date` (TICKERS `lastpricedate`), `reason` (`acquisition`, `performance`, `voluntary` or `unknown`) and `source_action` (the `ACTIONS` code that decided it). `ParquetStore.as_market_data` loads it into `MarketData.delisting_reasons` (ticker → reason).

**Why we need it here.** The final return of a delisted name depends on why it left (→ see **Backtesting › Delistings by reason**). TICKERS says *that* and *when* a name delisted; only `ACTIONS` says why.

**How it works** (`qts/data/sharadar.py::derive_delisting_reasons`). For every delisted id, take the `ACTIONS` rows of its ticker dated within `DELISTING_REASON_WINDOW_DAYS = 5` business days (weekdays; holidays are not excluded) of its delisting date, on either side, and map their action codes through the module constant `DELISTING_REASON_ACTIONS`:

| `ACTIONS` code | Reason | Reading of the row |
|---|---|---|
| `acquisitionby` | `acquisition` | the company in `ticker` was acquired by `contraticker` |
| `mergerto` | `acquisition` | the company in `ticker` merged into `contraticker` |
| `bankruptcyliquidation` | `performance` | bankruptcy or liquidation |
| `regulatorydelisting` | `performance` | removed by the exchange or regulator |
| `voluntarydelisting` | `voluntary` | the issuer chose to delist |
| `delisted` alone, or no row | `unknown` | the vendor recorded the delisting without a cause |

If several mapped codes sit on one delisting the strongest wins: `performance` > `acquisition` > `voluntary` > `unknown`. The mirror codes `acquisitionof` and `mergerfrom` describe the surviving company and are not mapped. A row whose id was resolved to a *different* holder of a reused ticker is ignored; a row dated just after the last price date (its id unresolved, because it is outside the holder's price range) is matched by ticker.

<callout icon="⚠️" color="yellow_bg">
The reading of `acquisitionby` / `mergerto` follows the action names; it was not checked against the vendor's field documentation when this was written. The derive report therefore prints every action code found next to a delisting. An acquirer does not delist in the week it buys something, so the code that dominates there is the "was acquired" one: if that is `acquisitionof` rather than `acquisitionby`, the mapping is inverted and `DELISTING_REASON_ACTIONS` must be corrected (it is one constant; `derive_delisting_reasons(..., mapping=...)` takes an override).
</callout>

**In the code.** The table needs no download: it is rebuilt from the stored `corporate_actions` and `security_master`.

```bash
qts data sharadar derive --store data/store            # rebuild delisting_reasons, print the counts
qts data sharadar derive --store data/store --window 10 --json reasons.json
```

`sync` writes it too, after the raw tables. The command prints counts by reason, a year × reason table, the action codes seen near all delistings, and for the ids left `unknown` the codes seen in the window plus any mapped code found within ±63 business days (a large count there means the 5-day window is too tight).

**Pitfalls & checks.**
- `unknown` is booked at −30% like `performance`. A large unknown share distorts the delisting P&L; read the report before a run.
- The action can be dated a few days after the last price date, while the simulator books the delisting the day after it. The reason therefore sets the size of an accounting entry with information up to 5 business days ahead; it never reaches a signal or a trade.
- The stage cache keys the alpha fit (the labels read the reasons) on this table and the risk fit not at all, so deriving or re-deriving it refits the alpha model only.
- Tests: `tests/test_sharadar.py::test_derive_delisting_reasons_mapping_precedence_and_window`, `::test_sync_writes_delisting_reasons_and_market_data_loads_them`, `::test_derive_from_an_existing_store_without_download`.

### Earnings dates and the quarterly record

**What it is.** Two derived long tables that feed feature set 2 of the alpha model (→ see **Alpha model › Feature set 2**): `earnings_dates`, one row per earnings announcement, and `earnings_fundamentals`, the as-reported quarterly numbers with a split factor. Both are plain functions of tables already in the store (`sharadar_sf1`, `sharadar_events`, `corporate_actions`), so they are derived, never downloaded.

**Why we need it here.** The SF1 `datekey` is the date of the full quarterly report. The market usually learns the result earlier, from a press release filed as a form 8-K under item 2.02 ("Results of Operations and Financial Condition"). A feature that measures the price reaction to earnings needs that earlier date; measured around the 10-Q date it would miss the reaction.

**`earnings_dates`.**

| Column | Meaning |
|---|---|
| `date` | the announcement date (`announce_date` on `MarketData`) |
| `ticker` | the store id |
| `source` | `events` (an 8-K item 2.02 from Sharadar `EVENTS`) or `datekey` (the filing date, the fallback) |
| `datekey`, `calendardate`, `reportperiod` | the quarterly filing the announcement precedes, if one follows within 45 days. Informational: attached with hindsight and never read by a feature |

The announcement series of a name is the union of two sets:

1. every `EVENTS` row whose `eventcodes` contain the earnings code (`qts/data/earnings.py::EARNINGS_EVENT_CODES`, default `22`);
2. every SF1 `ARQ` filing date that has **no** such event in the 45 calendar days up to and including the filing date (and after the previous filing date).

Each row is public on its own date: an 8-K is public the day it is filed, and on a filing date it is known whether an 8-K came before it. No row is chosen or dropped using anything dated later, so the series as of $t$ is simply the rows dated $\le t$. That is why the rule keeps *every* item 2.02 event, including a pre-announcement, instead of picking "the" announcement of each quarter, which would need to know which filing follows.

<callout icon="⚠️" color="yellow_bg">
The event code is an assumption. Sharadar's `eventcodes` are taken to be the 8-K item numbers without the dot (`22` = item 2.02); this was not checked against the vendor's documentation. `derive-earnings` prints the frequency of every code with two tell-tales: events per name-year (an earnings code shows about four) and the share of events dated in the 45 days before a quarterly filing (high for earnings). If another code fits, pass `--codes`.
</callout>

**`earnings_fundamentals`.** One row per SF1 row of the `ARQ` and `ART` dimensions: `date` (= `datekey`), `ticker`, `dimension`, `calendardate`, `eps`, `netinc`, `ncfo`, `assets`, and `split_cum`. `split_cum` is the product of the ratios of every split of the name dated on or before the row's `datekey` (1.0 without splits), so `eps × split_cum` is EPS per original share, continuous across splits, built from past splits only. Split dates and ratios come from the `corporate_actions` rows with `action = split`. Whether the vendor's `value` is the ratio (2 for a 2-for-1) or its inverse is checked row by row against the prices: on the split date $(P^{\text{raw}}_{t-1}/P^{\text{raw}}_t)\cdot(P^{\text{adj}}_t/P^{\text{adj}}_{t-1})$ is the ratio the two closes imply; the command reports how many rows the prices confirm each way.

**In the code.**

```bash
qts data sharadar derive-earnings --store data/store          # build both tables, print the report
qts data sharadar sync --tables EVENTS --from-files ~/Downloads/sharadar    # add EVENTS, then derive again
qts data sharadar sync --tables SF1 --from-files ~/Downloads/sharadar       # re-read SF1 (adds ncfo)
```

- `qts/data/earnings.py`: `derive_earnings_dates`, `build_earnings_fundamentals`, `confirm_split_ratios`, `cumulative_split_factor` (pure functions); `qts/data/sharadar.py::derive_earnings_from_store`. `sync` rebuilds both tables at the end of every run.
- `ParquetStore.as_market_data` loads them into `MarketData.earnings_dates` / `earnings_fundamentals` for the loaded ids, rows up to `end` (no `start` cut: the earnings surprise needs three years of quarters before the first panel date). `MarketData.slice_until(t)` keeps the rows dated $\le t$.
- Without `EVENTS` in the store the command says so, uses the filing date for every announcement and prints the `sync` line that adds the table.
- `ncfo` (operating cash flow) was added to the kept SF1 fields on 2026-11-13. A store synced earlier lacks it; the command warns, and `sync --tables SF1` reads the table again even if the raw file is unchanged, because the kept field list is part of the step's resume state.

**Pitfalls & checks.**
- A filing date as announcement date is late by the gap between press release and report (days for a 10-Q, often weeks for a 10-K). The reaction window then misses the real reaction; the feature is noisier, not forward-looking.
- 8-K item 2.02 exists since August 2004, so earlier years rest on filing dates. The report's by-year table shows the split.
- Neither table enters the risk fit's cache key, and they enter the alpha fit's key only under `alpha.feature_set = "v2"`.
- Tests: `tests/test_feature_set_2.py::test_announcement_is_the_event_and_the_filing_date_is_the_fallback`, `::test_the_series_as_of_t_is_the_series_restricted_to_t`, `::test_sync_derives_the_earnings_tables_from_sf1_events_and_actions`, `::test_split_ratio_is_confirmed_against_the_stored_prices`, `::test_derive_earnings_cli_on_an_existing_store`.

### Incremental refresh and memory

A full load drops and rebuilds every dataset a table feeds (`mode="replace_partition"`). With `--since YYYY-MM-DD` the loader exports only rows with `lastupdated >= since` for `SEP`, `DAILY` and `SF1` (TICKERS, ACTIONS and SP500 are always whole) and upserts them; the derived `market_cap`, `shares_outstanding` and fundamentals panels are rebuilt from the store for every year the delta touched. Sharadar restates `closeadj` for the whole history whenever a dividend is paid, so a full reload every month or two keeps adjusted prices current. Each table is resumable: a raw file already processed in the same mode is skipped (`<store>/_sharadar/state.json`, `--force` ignores it).

Nothing is pivoted whole. CSVs are streamed in chunks of 500,000 rows, rows are resolved to ids and spilled to per-year staging files, and each year is written once; the fundamentals are computed in batches of 500 ids. Peak memory for a full load is a few GB; the numbers are in the Operating guide.

### What the bundle does not cover

Half-spreads (`spread_bps`), borrow fees, short interest and dated industry history are not in the bundle: the cost model's spread falls back to its proxy, the borrow signal waits for the IBKR logger, and FINRA short interest is loaded separately (→ see **Short interest: FINRA data and the publication lag**). Delisting *returns* are not provided either: the delisting *reason* is (previous subsection), and the return per reason is an assumption in `CostConfig`.

### References

- [Nasdaq Data Link, Sharadar Core US Equities Bundle](https://data.nasdaq.com/databases/SFA)
- [Nasdaq Data Link, Tables API bulk export](https://docs.data.nasdaq.com/docs/tables-1)

## Roadmap

Not yet in the code (from `README.md`, `docs/REVIEW_NOTES.md`, `docs/WORK_PLAN.md`): the Norgate price path as a second source (`qts/data/loaders.py::norgate_to_store` is a stub); the daily IBKR borrow-rate logger, spliced into `borrow_fee` after 12 months; real delisting returns from a vendor in place of the per-reason assumptions; dated industry history (`build_industry_history` hook); reproducing five `openassetpricing` factor series from own data as a PIT audit. Phase 0 (Sharadar prices, caps and fundamentals into a `ParquetStore` keyed on permanent identifiers) is in the code: → see **Real data: Sharadar** above.
