# Architecture & data flow

<callout icon="🧭" color="gray_bg">
This page describes how `qts` is put together: the layers and what each package is responsible for, the typed contracts that pass between them (`MarketData`, `AlphaOutput`, `RiskModelOutput`, the cost snapshot, `OptimizationResult`, `BacktestResult`), the data-layout and unit conventions every module follows, and the point-in-time discipline that keeps future information out of every decision, with the four kinds of leakage and the specific guard for each. It then follows one research run from `fit` to the trial ledger, explains how runs are made reproducible (config hash, data hash, run manifest, run registry, append-only trial ledger), why the test suite is built around a synthetic market with a planted signal, and how to extend the system with a new feature, risk factor, cost component or constraint.
</callout>

<callout icon="🆕" color="green_bg">
<b>What changed (Nov 2026)</b>
- One typed cost contract in `qts/core/costs.py` (`LiquiditySnapshot`, `CostEstimate`, the `CostModel` protocol) replaces the dict snapshot and the `CostTerms` protocol; `TransactionCostModel` is the only production implementation.
- New single sources of truth in `qts/core`: `roles.py` (risk-only styles vs alpha themes), `stats.py` (winsorise, rank-gauss, rank IC, Newey–West, rolling beta, industry dummies), `manifest.py` (`RunManifest`, the all-panel data hash).
- `SystemConfig` serialises strictly to and from YAML and has a stable hash; `configs/default.yaml` is generated from it.
- Every run writes `runs/<run_id>/` (manifest, config, summary, Parquet frames, plus UI artifacts from `qts/runs/artifacts.py`); a `qts` command, a FastAPI service (`api/`) and a React workbench (`ui/`) sit on top.
- The trial ledger is strictly append-only under a file lock; re-runs collapse at read time on a key that includes the code version and the run spec; DSR families are per phase and per data hash.
- Execution is delayed one day by default (`signal_lag_days = 1`); `apply_mask` no longer blanks level panels outside the universe (eligibility acts only through `md.eligibility()`); a future-universe leak in risk style standardisation was fixed.
</callout>

## The layers

### What it is

`qts` is a single Python package organised as a stack of layers. Lower layers know nothing about higher ones. Every layer talks to its neighbours through a small number of typed containers defined in `qts/core/types.py`, so any layer can be developed, tested or replaced on its own.

```text
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ clients         ui/ (React workbench) ──HTTP──► api/ (FastAPI, qts_api)   │
 │                 qts/cli.py  (`qts demo | run | config | runs | ledger`)  │
 ├──────────────────────────────────────────────────────────────────────────┤
 │ orchestration   qts/pipeline.py  ResearchPipeline · save_run · load_run  │
 │                 qts/runs         artifacts (extra per-run files for UI)  │
 ├──────────────────────────────────────────────────────────────────────────┤
 │ evaluation      qts.validation   splits · leakage · ledger · DSR · PBO   │
 │                 qts.backtest     DailyBarSimulator · metrics             │
 ├──────────────────────────────────────────────────────────────────────────┤
 │ decision        qts.portfolio    SinglePeriodOptimizer · alpha_scaling   │
 │                 qts.execution    orders · router · scheduler · ibkr      │
 ├──────────────────────────────────────────────────────────────────────────┤
 │ models          qts.alpha        features · targets · composite · ML     │
 │                 qts.risk         exposures · factor returns · covariance │
 │                 qts.tcost        fees · impact · borrow · TCA            │
 ├──────────────────────────────────────────────────────────────────────────┤
 │ data            qts.data         synthetic · store · universe · loaders  │
 │                                  calendar (NYSE sessions)                │
 ├──────────────────────────────────────────────────────────────────────────┤
 │ core            qts.core         types · config · costs · roles · stats  │
 │                                  manifest · crosssection · risk_fill     │
 └──────────────────────────────────────────────────────────────────────────┘
        each layer imports only from layers below it (one exception, noted)
```

The cost contract now lives in the core layer, so the portfolio and backtest layers depend only on `qts.core.costs`, never on `qts.tcost`. The one remaining exception to strict layering is deliberate and lazy: `qts/portfolio/costs.py` keeps backward-compatible helpers (`snapshot_from_market`, `SimpleCostModel`, a test double) that import `qts.tcost.TransactionCostModel` only at call time. The optimiser and simulator see a cost model only through the `CostModel` protocol (structural typing), so any object with its four methods plugs in. The API and UI are separate installable folders outside the `qts` package: the API imports `qts`; nothing in `qts` imports the API.

### Why we need it here

Five models must agree on units, dates and tickers. The classic failure is a mismatch between two of them: a 20-day alpha fed to an optimiser assuming daily units, a cost in basis points read as a fraction. The README calls the build a **thin vertical slice of all five models**, wired end to end before any is deepened so such failures appear early; clear contracts let each layer deepen without rewiring.

### Package-by-package responsibilities

| Package | Responsibility | Main entry points | Produces |
|---|---|---|---|
| `qts.core` | Shared contracts, configuration, the cost contract, factor roles, shared statistics, run identity | `types.py`, `config.py::SystemConfig`, `costs.py` (`LiquiditySnapshot`, `CostEstimate`, `CostModel`), `roles.py`, `stats.py`, `manifest.py::RunManifest`, `risk_fill.py`, `crosssection.py` (thin wrappers over `stats`) | `MarketData`, `AlphaOutput`, `RiskModelOutput`; config dataclasses; snapshots and cost estimates; manifests |
| `qts.data` | Get data into a `MarketData`; decide the investable universe | `synthetic.generate`, `store.ParquetStore`, `universe.build_universe_mask`, `universe.apply_mask`, `universe.masked_levels`, `loaders.*`, `calendar.add_sessions` | `MarketData` with `eligible` set |
| `qts.alpha` | Forecast residual returns | `AlphaModel.fit` / `as_of`, `FeatureBuilder`, `ThemeComposite`, `WalkForwardTrainer`, `evaluation.ic_report` | `AlphaOutput` |
| `qts.risk` | Forecast the covariance of returns | `build_risk_model` → `BarraStyleRiskModel` or `HybridRiskModel`; `.fit` / `.as_of` | `RiskModelOutput` |
| `qts.tcost` | Price trades and holdings; calibrate from fills | `TransactionCostModel` (`snapshot`, `estimate`, `cvx_terms`, `holding_cost_usd`), `tca.TCALedger` | `LiquiditySnapshot`; `CostEstimate` in USD; convex cost expressions |
| `qts.portfolio` | Turn alpha, risk and cost into target weights | `SinglePeriodOptimizer.solve`, `alpha_scaling`, `vol_target`, `diagnostics` | `OptimizationResult` |
| `qts.execution` | Turn weight changes into orders and route them | `targets_to_orders`, `ExecutionRouter.route`, schedulers, `to_ib_order_dict`, `FillSimulator` | `Order` lists, broker payloads, simulated fills |
| `qts.backtest` | Replay the chain day by day with explicit costs | `DailyBarSimulator.run`, `metrics` | `BacktestResult` |
| `qts.validation` | Decide whether a result is credible | `PurgedWalkForward`, `CombinatorialPurgedCV`, `pit_shift_test`, `planted_oracle_test`, `TrialLedger`, `deflated_sharpe_ratio_annual`, `probability_of_backtest_overfitting`, `HurdleReport` | gate verdicts, DSR, PBO |
| `qts.pipeline` | Wire everything for a research run; write and read run directories | `ResearchPipeline.fit` / `run` / `evaluate`, `format_result`, `save_run`, `load_run`, `list_runs` | `PipelineResult`; `runs/<run_id>/`; `RunRecord` |
| `qts.runs` | Extra per-run files the UI needs (ex-ante vol, factor exposures, month-end holdings, IC by year, DSR moments) | `artifacts.write_extra_artifacts`, `read_extras` | `extras.json`, `exante_vol.parquet`, `exposures.parquet`, `holdings.parquet` |
| `qts.cli` | The `qts` command | `main`, `build_parser` | run directories, ledger records, printed reports |
| `api/` (`qts_api`) | HTTP service over the run registry, ledger, config schema and background jobs | `create_app`, `RunRegistry`, `JobManager` | JSON (OpenAPI contract) |
| `ui/` | Browser workbench (React + TypeScript) | runs list, run detail tabs, new-run form, ledger | screens |

### Why this choice

The build plan considered an off-the-shelf framework (Qlib) and an event-driven engine (NautilusTrader, LEAN) from the start. It recommends a vectorised custom daily-bar simulator for research sweeps and reserves an event-driven engine for paper and live trading: a daily auction rebalancer needs no tick simulation, and event-driven engines make thousands of research runs slow. Qlib is mined for feature definitions only; its author calls it a foundation, not a production system. Contracts are plain dataclasses of pandas objects, not a database schema or message bus, so any intermediate can be inspected in a notebook.

### In the code

The layering is visible in the imports: `qts/core/types.py` imports only numpy and pandas; `qts/data/universe.py` imports `qts.core`; `qts/pipeline.py` imports every layer. Configuration for every layer lives in one frozen-by-convention tree, `qts/core/config.py::SystemConfig`, with one dataclass per layer (`UniverseConfig`, `AlphaConfig`, `RiskConfig`, `CostConfig`, `PortfolioConfig`, `ExecutionConfig`, `ValidationConfig`). A generated mirror of the defaults is in `configs/default.yaml` (`qts config dump > configs/default.yaml`; `tests/test_config.py` fails when it drifts); the dataclasses are the source of truth. → see **Reproducibility › Configuration as data** below.

### Pitfalls & checks

- **Hidden coupling through string conventions.** An earlier version of the optimiser found the industry block by the `IND` prefix and the beta factor by name; with Fama–French 48 industry labels the industry bands silently switched off, and "beta-neutral" constrained the market column instead of the beta style. The fix was the `factor_groups` field of `RiskModelOutput` (below), which states each column's role explicitly (`docs/audit/RESOLUTION.md`).
- **Two implementations of one thing.** The cost model briefly existed twice (`SimpleCostModel` and `TransactionCostModel`), and helpers such as winsorisation, rank IC, Newey–West t-statistics and rolling beta existed once per layer. The refactor (`docs/WORK_PLAN.md` items 1.2 and 1.3) made one implementation of each: the cost contract in `qts/core/costs.py` with `TransactionCostModel` as the only production model (`SimpleCostModel` is a test double); the statistics in `qts/core/stats.py`, each checked against a brute-force reference in `tests/test_core_stats.py`; the risk-only/alpha-theme split in `qts/core/roles.py`. `tests/test_cost_equivalence.py` pins optimiser ⇄ simulator cost agreement.

### References

- [Qlib benchmarks](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md)
- [NautilusTrader v1.231.0 release notes](https://github.com/nautechsystems/nautilus_trader/releases/tag/v1.231.0)

## Data layout: wide and long panels

### What it is

Two table shapes are used everywhere, defined in the module docstring of `qts/core/types.py`:

- A **wide panel** is a `pd.DataFrame` indexed by a `pd.DatetimeIndex` of trading days (ascending, timezone-naive) with **one column per ticker**. Prices, returns, volume, market cap, spreads, short interest, fundamentals and the eligibility mask are wide. Type alias `Wide`.
- A **long panel** is a `pd.DataFrame` with a two-level `MultiIndex` `(date, ticker)` and **one column per feature**. Processed feature matrices, ML training sets and alpha scores are long. Type alias `Long`.

```text
 wide: close                         long: processed features
             AAPL    MSFT    XOM                          value_bp  mom_12_1  ...
 2024-01-02  185.6   370.9   99.8    (2024-01-02, AAPL)     -0.81      0.44
 2024-01-03  184.3   370.6  100.7    (2024-01-02, MSFT)     -0.52      0.91
 ...                                 (2024-01-02, XOM)       1.10     -0.37
                                     (2024-01-03, AAPL)     -0.80      0.41
```

### Why we need it here

Per-stock time-series operations and per-date cross-sectional ones are single vectorised calls on a wide panel (`close.shift(21)`, `ret.rolling(63).std()`, `x.rank(axis=1)`). ML models want one row per observation and one column per feature: the long shape. Storage uses long `date, ticker, value` triples (→ see **Data & universe › The point-in-time store**).

### How it works

Conversion is one function each way, `qts/core/types.py::to_long` and `to_wide`:

```python
def to_long(wide: Wide, name: str) -> pd.Series:
    """Wide (date x ticker) -> long Series indexed by (date, ticker)."""
    s = wide.stack(future_stack=True)
    s.index.names = ["date", "ticker"]
    s.name = name
    return s


def to_wide(long: pd.Series) -> Wide:
    return long.unstack("ticker")
```

`MarketData.validate` checks that every required wide panel shares the close panel's index and columns, that dates ascend, and that every ticker has an industry label. `assert_wide_aligned` does the same for arbitrary frames.

### Pitfalls & checks

- **Silent misalignment.** pandas aligns on labels, so adding two wide frames with different ticker sets produces NaNs rather than an error. Reindex explicitly (`reindex(index=..., columns=...)`) at layer boundaries; the risk model's `covariance()` does this and documents its fill policy.
- **NaN means "absent", not zero.** A missing close means the stock is not trading (not yet listed, halted, delisted). The alpha layer drops ineligible rows instead of zero-filling them; zero-filling was the root of the critical "ever in universe" audit finding.

## The contracts

Six containers carry everything between layers. Field lists below follow the current dataclasses; check the dataclass before relying on a keyword.

### MarketData

**What it is.** The bundle of wide panels for one universe (`qts/core/types.py::MarketData`). Required: `close` (split- and dividend-adjusted close), `volume` (actual shares traded that day), `market_cap` (USD), `industry` (a Series mapping ticker to a static fallback industry label). Optional: `spread_bps` (half-spread proxy in bp), `short_interest_ratio` (short interest / shares outstanding), `borrow_fee` (annual fraction), `book_to_price`, `earnings_yield`, `gross_profitability`, `asset_growth`, an `extra` dict of further wide panels, `eligible` (the per-date boolean universe mask), and four fields added in the refactor:

| Field | What it holds | Who uses it |
|---|---|---|
| `close_raw` | As-traded (unadjusted) close, aligned with `close` | price screen, dollar volume, per-share fees, share counts, per-share fundamentals, via `trade_price()` |
| `shares_outstanding` | Actual share count per date ($\text{market cap} \approx$ `close_raw` × shares) | FINRA short-interest ratio, split-consistent B/P and E/P |
| `industry_history` | Dated classification: date × ticker labels, a row holds the label in force from that date | `industry_as_of(t)`: risk industry factors, alpha industry demeaning, label residualisation |
| `delisting_dates` | Series ticker → last trading date (`NaT` = known alive) | the label builder's halt-versus-delisting rule |

The rule of thumb in the class docstring: use `close` for anything that compares prices across time (returns, momentum, volatility) and `trade_price()` for anything that is a *level* on one date.

**Derived views.** `returns` (simple close-to-close on the adjusted close, `pct_change(fill_method=None)`, so a gap stays NaN), `trade_price()` (`close_raw` if present, else `close`), `dollar_volume` (`trade_price()` × `volume`), `industry_as_of(date)` (the label in force at `date`; static labels fill names not yet classified), `dates`, `tickers`, and two methods that carry the point-in-time discipline:

```python
def eligibility(self) -> Wide:
    alive = self.close.notna()
    if self.eligible is None:
        return alive
    e = self.eligible.reindex(index=alive.index, columns=alive.columns)
    return e.fillna(False).astype(bool) & alive
```

`eligibility()` is the only function any layer uses to decide which names enter a cross-section: in the universe at $t$ **and** priced at $t$. Since the second audit, the universe acts *only* through it: `apply_mask` attaches `eligible` but leaves every level panel (volume, market cap, shares, fundamentals) with its full history, so a name that just entered the universe has true 63-day trailing statistics rather than a truncated window. A consumer that still wants "NaN outside the universe" levels (the risk model's estimation universe) calls `qts/data/universe.py::masked_levels(md)`. `slice_until(date)` returns a copy with every panel cut at `date` (delisting events after the cut are blanked), the point-in-time view used by the leakage tests.

### AlphaOutput

**What it is.** The alpha layer's answer for one date (`AlphaOutput`): `date`; `score`, a cross-sectional z-score per eligible ticker (mean 0, standard deviation 1); `expected_return`, the Grinold-scaled expected residual return over the forecast horizon ($\text{IC} \times \sigma^{\text{resid}} \times \text{score}$); `horizon_days` (20 by default) and `half_life_days` (the signal's estimated decay half-life, used for Gârleanu–Pedersen shrinkage); optional per-sleeve `components` for attribution; and `ic_estimate`.

**Why both score and expected return.** The score is unitless and good for diagnostics (IC, deciles). The optimiser needs a quantity in return units that is comparable with risk and cost; feeding raw scores to an optimiser lets it treat a z-score of 2 as a 200% expected return. → see **Alpha model › Grinold's rule**.

**Invariant.** Names that are not eligible on `date` are **absent** from `score` and `expected_return`. The pipeline enforces this a second time in `ResearchPipeline._eligible_alpha_fn`, and the simulator sets the expected return of any absent name to zero.

### RiskModelOutput

**What it is.** The risk model as of one date (`RiskModelOutput`): `exposures` $B$ (tickers × factors), `factor_cov` $F$ (factors × factors, daily units), `specific_var` (the diagonal of $D$, daily), `factor_returns` (history up to and including `date`), the optional Saxena–Stubbs alignment term (`alignment_vector` $y$, unit norm, and `alignment_var` $\nu$), and **`factor_groups`**. The implied covariance is

$$
\Sigma = B F B^\top + D + \nu\, y y^\top
$$

where:
- $B$, $F$, $D$ are as above;
- $y$ is the unit-norm direction of the alpha that the factors do not explain;
- $\nu$ is the variance assigned to that direction (0 when alignment is off).

`covariance(tickers)` materialises $\Sigma$ for a ticker set (tickers outside the model get zero exposure and the 90th percentile of the modelled specific variances, the conservative fill shared with the optimiser in `qts/core/risk_fill.py`); `portfolio_variance(w)` computes $w^\top \Sigma w$ in factor form without building the $N \times N$ matrix.

**`factor_groups`.** A dict from role to column names, with keys `"market"`, `"industry"`, `"style"`, `"statistical"` and `"alignment"`. The portfolio layer uses it to find the beta style for the beta-neutral constraint, the industry block for industry bands, and the style block for style bands, instead of guessing from column names. It exists because of two audit findings: "beta-neutral" had been constraining the unit-exposure `market` (country) column, leaving the true beta style exposure near 2, and industry bands silently turned off when industry labels stopped starting with `IND`. `qts/risk/model.py::factor_groups_for` fills it; an empty dict means "unknown" and triggers a logged name-based fallback in the optimiser.

### The cost contract: LiquiditySnapshot, CostEstimate, CostModel

**What it is.** Three typed pieces in `qts/core/costs.py`, shared by the cost, portfolio and backtest layers.

- **`LiquiditySnapshot`** (frozen dataclass): the per-ticker market inputs a cost model needs *as of one date*. Fields: `date`, `tickers`, `price` (USD per share, as traded), `adv_usd` (USD per day), `adv_shares`, `sigma_daily` (daily return standard deviation, a fraction), `half_spread` (a **fraction** of price; `half_spread_bps` is a derived property), `borrow_fee` (**annual** fraction), `auction_share` (expected closing-auction volume ÷ ADV for that date), `borrow_fee_source` and an optional `nav`. Construction reindexes every field to `tickers` and makes it finite: NaN or infinite values are filled with the cross-sectional median (fallbacks: price \$100, ADV \$1M, σ 2%, half-spread 2 bp, borrow 0.5% a year) and floored (price ≥ \$0.01, ADV ≥ \$1,000, σ ≥ 0.0001). A brand-new listing therefore gets median impact, never zero impact.
- **`CostEstimate`**: the USD cost of one trade vector, with fixed columns `spread_usd`, `impact_usd`, `fees_usd`, `total_usd`, plus `venue` and `persistence_mult`. It refuses to exist if a column is missing, a value is non-finite, or a component is negative beyond rounding noise.
- **`CostModel`** (a `typing.Protocol`): `snapshot(md, date)` → `LiquiditySnapshot`; `estimate(trades_usd, snap, venue, streak)` → `CostEstimate`; `cvx_terms(x, w_post, nav, snap, venue)` → (one-off trade cost, one-day holding cost) as cvxpy expressions in **fractions of NAV**; `holding_cost_usd(w_post, nav, snap)` → one day of borrow in USD.

Venues are named constants: `"continuous"` (in-session algorithm, volume = ADV, pays the half-spread), `"auction"` (closing auction, volume = `auction_share` × ADV, no spread) and `"routed"` (up to `ExecutionConfig.auction_max_pct_of_auction_volume` of the auction at the auction, the rest continuous). `auction_share_for_date` supplies the heavy-day shares (about 20% on quarterly index-rebalance Fridays, 34% on Russell reconstitution days; `CostConfig.auction_volume_share_default = 0.09` otherwise).

**Why one snapshot for both.** If the optimiser's forecast cost and the simulator's charged cost differ, the optimiser systematically over- or under-trades. Before the refactor the snapshot was a dict of Series whose keys could be renamed silently, and an unrecognised cost column was once zero-filled. With a typed snapshot that is always finite and an estimate that raises on anything odd, the optimiser's convex terms and the simulator's point estimates provably price the same inputs (`tests/test_cost_equivalence.py`).

<callout icon="💡" color="blue_bg">
Backward compatibility: `LiquiditySnapshot` still answers read-only mapping access with the old key names (`snap["adv_usd"]`, `"date" in snap`, `dict(snap)`), and `LiquiditySnapshot.from_mapping` converts an old dict. New code should use the attributes.
</callout>

### OptimizationResult

**What it is.** The portfolio layer's answer for one date (`qts/portfolio/optimizer.py::OptimizationResult`): `w_target` (post-trade weights), `x` (trades, $w^{\text{target}} - w^{\text{current}}$), `expected_alpha` ($\alpha^\top w$ over the horizon), `expected_risk` (annualised ex-ante volatility), `expected_cost` (trade cost plus horizon × holding cost, fraction of NAV), `turnover` ($\sum_i |x_i|$), `transfer_coefficient`, solver `status` and `solve_time`, counts of `binding_constraints`, the `vol_scalar` used by the volatility-target projection (fixed-γ mode only), `exante_var_daily`, `n_out_of_model` (material names outside the risk model's universe), `alignment_share` (the share of ex-ante variance coming from the alignment term), and the sizing diagnostics added with the γ search: `risk_aversion` (the annual γ of the returned book), `vol_target_binding` (`"target"` when the vol target was hit, `"caps"` when gross / position / ADV caps stopped the book first), `n_solves` and `gp_shrink`. `ok` is true for `"optimal"` and `"optimal_inaccurate"`.

**Failure policy.** A failed solve is reported, not hidden: the simulator holds the current book, logs a warning and increments `n_failed_solves` in the summary. Relaxing the ADV participation caps on infeasibility is opt-in (`PortfolioConfig.relax_participation_on_infeasible = False`).

### BacktestResult

**What it is.** The simulator's record of a run (`qts/backtest/simulator.py::BacktestResult`): `nav`; `returns` (columns `gross`, `net`, and `gross_after_borrow`, the starting point for per-side cost gates); `weights` (post-trade) and `trades` (weight changes); `costs` in USD with columns `spread`, `impact`, `fees`, `borrow`, `total`; `exante_var`; `turnover`; `transfer_coefficient`; and `diagnostics` (delistings, forced exits, solve log, failed-solve count). `summary()` computes the report card: gross/net CAGR, volatility and Sharpe, Lo-adjusted Sharpe, Newey–West t, drawdown, Calmar, Sortino, exposures, turnover, effective holding days, cost per side, realised versus ex-ante volatility, breakeven cost, delisting and failure counts.

**Indexing rule.** Decision-date panels (`weights`, `trades`, `costs`, `exante_var`, `turnover`) are indexed by $t$; `returns` and `nav` by the realisation date $t+1$. `turnover_on_return_dates()` re-indexes turnover positionally onto the return dates. The audit found and fixed an earlier date-join that was off by one day.

`ResearchPipeline.evaluate` wraps a `BacktestResult` into a `PipelineResult` with the IC report, decile spread, fold Sharpes, cost sensitivity, hurdle verdicts, deflated Sharpe and its inputs, PBO, and the config and data hashes.

## Conventions

These rules hold in every module. They are written at the top of `qts/core/types.py`, and code review should reject anything that breaks them.

| Convention | Rule | Why |
|---|---|---|
| Returns | Simple close-to-close, $r_t = P_t / P_{t-1} - 1$; multi-day returns compounded | Portfolio return $= \sum_i w_i r_i$ exactly (→ see **Start here › Returns: simple versus log**) |
| Return "as of $t$" | May use $r_s$ only for $s \le t$ | Point-in-time rule |
| Weights | Fractions of NAV; long positive, short negative | Scale-free; leverage is $\sum \lvert w \rvert$ |
| Trades | $x = w^{\text{post}} - w^{\text{pre}}$, a weight change; dollars $= x \cdot \text{NAV}$ | One variable for optimiser and cost model |
| Pre-trade weights | Yesterday's post-trade weights drifted by returns and re-based to today's NAV | Costs and price moves change weights without trading |
| Variance units | Daily in the risk model; the optimiser multiplies by the horizon $H$ | One horizon for alpha, risk and holding cost in the objective |
| Expected return | Over `AlphaOutput.horizon_days` (20) | Matches the label the model was trained on |
| Costs | Fractions of NAV inside the optimiser; USD in the simulator; bp in reports | Each is the natural unit at its point of use |
| Annualisation | 252 trading days (`SystemConfig.trading_days_per_year`) | → see **Start here › Daily versus annualised: the √252 rule** |
| Dates | `pd.DatetimeIndex` of trading days, ascending, tz-naive; trading-day positions for windows and embargoes; lags stated in exchange sessions use the NYSE calendar (`qts/data/calendar.py`) | Calendar arithmetic on trading days avoids weekend and holiday off-by-ones |
| Signal date vs execution date | The decision at execution date $t$ uses data as of $s = t - L$, $L$ = `ExecutionConfig.signal_lag_days` (default 1); labels of $s$ run from the close of $s+L$ | A live order is due at 15:50 ET, before the close it trades at |
| Prices | `close` (adjusted) for anything across time; `trade_price()` (as traded) for levels on one date | Adjusted prices understate historical levels across splits |

<callout icon="⚠️" color="yellow_bg">
Unit errors are the most common silent bug in this kind of system. The audit's verified list includes "objective units (alpha per H, risk ×H, trade cost once, holding ×H)" for a reason: if the optimiser compares a 20-day alpha with a daily variance, it takes roughly 20 times too much risk. When you add a term to any objective or metric, write its unit next to it in the docstring.
</callout>

## Point-in-time discipline

### What it is

A computation is **point-in-time (PIT)** at date $t$ if its output depends only on information that was actually available at $t$. In this system that means: any object produced "as of $t$" (a feature, a universe decision, a covariance matrix, a cost snapshot, an alpha score) is a function of data with timestamp $\le t$, where the timestamp is **when the data became public**, not the period it describes. Labels, the forward returns the models learn to predict, are the only things allowed to look past $t$, and they are never used as inputs.

Breaking the rule is **leakage** (look-ahead bias): like grading a forecaster who has glimpsed tomorrow's newspaper, the track record is worthless however good it looks.

### Why we need it here

Leakage is the main reason backtests fail live, and it is invisible in the output: a leaked backtest does not look broken, it looks excellent. Statistical defences do not help. The build plan cites a planted look-ahead oracle that scored a deflated Sharpe of 1.00 (Gençay 2026), and draws the conclusion this code is built around: leakage must be made **inexpressible upstream** by construction, with tests that would fail if it crept in, rather than detected statistically afterwards.

### How it works: the four kinds of leakage and their guards

**Leakage kind 1: look-ahead in features.**

**Example.** A volatility feature computed as `returns.rolling(21, center=True).std()` uses ten days of future returns. Subtler: z-scoring a feature with the full-sample mean and standard deviation (`(x - x.mean()) / x.std()`), which makes every early value depend on later data; or back-filling a missing fundamental with `bfill()`, which copies a future value backwards. Each of these makes a feature at $t$ a function of rows after $t$.

**Guards.**
- **Construction rule.** All time-series operations in `qts/alpha/features.py` and `qts/risk/exposures.py` are `shift(k)` with $k \ge 0$ or trailing `rolling` windows; cross-sectional operations are per date. The only `ffill`/`shift(-h)` in the alpha layer is inside the label builder `qts/alpha/targets.py::forward_returns`, documented as label-only.
- **`MarketData.slice_until(t)`** produces the true information set at $t$: every panel cut at $t$.
- **`qts/validation/leakage.py::pit_shift_test(build_fn, md, cut_date)`** runs `build_fn` on the full history and on `md.slice_until(cut_date)` and asserts that the two agree on every date up to `cut_date` (same NaN pattern, values equal within `1e-10`). Any use of future rows changes the full-history values at or before the cut and fails the test. `tests/test_validation.py::test_pit_shift_test_catches_lookahead` shows it catching both `shift(-1)` and full-sample demeaning. The 2026-11-04 audit independently re-ran this check across `AlphaModel.as_of`, every feature, dynamic theme weights, risk `as_of`, exposures, factor returns and cost snapshots, and found them identical when refit on truncated data.
- **`feature_registry`**, a decorator that inspects a feature function's source and refuses literal negative shifts (`shift(-1)`, `shift(periods=-5)`). It is a cheap tripwire with documented limits (it cannot see helpers, variable shifts, `center=True` or `bfill`); `pit_shift_test` is the real gate.
- **Property-based PIT tests.** `tests/test_pit_properties.py` uses `hypothesis` to draw random cut dates and checks the truncated-equals-full property over registered features, the universe mask, cost snapshots and risk `as_of`.
- **Cross-sections over eligible names only, in every layer.** The lag investigation of 2026-11-05 found that the risk model's style exposures (beta, residual volatility, and the standardisation of every style) had been computed using names that join the universe only *after* $t$. That is a future-membership leak of the same family as kind 3 below; it also leaked slightly into the label residualisation. Exposures are now restricted to names eligible at each date, with a regression test in `tests/test_risk.py`.

**Leakage kind 2: label overlap between training and testing.**

**Example.** The model is trained to predict 20-day forward returns. A training row dated 10 March has a label that covers 11 March to 7 April. If the test period starts on 25 March, that training label contains returns from inside the test period: the model has been shown part of the answer. With daily rows and 20-day labels, consecutive labels overlap by 19 days, so a naive train/test split leaks heavily at every boundary.

**Guards.**
- **Purge** removes every training row whose label window overlaps the test window. **Embargo** additionally drops rows just after the test window, guarding against slow serial correlation. In the walk-forward trainer, a label at date $s$ is realised over $(s, s+h]$, and a model that predicts from position $t$ onward may train only on labels with $\text{pos}(s) + h < \text{pos}(t)$:

```python
def embargoed_train_mask(label_pos: np.ndarray, pred_start_pos: int, horizon: int) -> np.ndarray:
    """Boolean mask over label positions that may be used to train a model that
    predicts from ``pred_start_pos`` onward: ``pos(s) + h < pred_start_pos``."""
    return label_pos + horizon < pred_start_pos
```

- `qts/alpha/trainer.py::WalkForwardTrainer` applies this with the horizon as the embargo (`AlphaConfig.embargo_days = None` defaults to `horizon_days = 20`), purges the trailing 12-month validation block from the fit rows the same way, and re-creates each model from its factory at every refit so no parameters carry over. The audit verified the purge is exact (one day stricter than strictly needed).
- **The execution delay lengthens every wait.** With `signal_lag_days = L`, the label of signal date $s$ runs from the close of $s+L$ to the close of $s+L+h$, so it is realised only at $s+L+h$; the walk-forward purge and the trailing-IC window grow by $L$ (`qts/alpha/targets.py`).
- **A label is used only once it is decided.** When a name has no close at the label's window end, whether that is a delisting or a halt is decided causally (→ see **Leakage kind 3** below), and `qts/alpha/targets.py::label_realisation` gives the first date at which a row's labels are all decided; training and the trailing IC consume a row only after that date.
- `qts/validation/splits.py` provides `PurgedWalkForward` (expanding train, strictly forward test blocks), `PurgedKFold` and `CombinatorialPurgedCV`, all purging label windows that touch the test block and embargoing `ValidationConfig.embargo_days = 20` days after it; `label_overlap_check` and `check_no_overlap` assert the property.

**Leakage kind 3: survivorship and universe membership.**

**Example A (survivorship).** Testing on today's listed stocks omits every company that went bankrupt or was delisted since 2006, so the longs never hold a collapse.

**Example B (universe membership).** The universe at each date is "top 1,000 by market cap". If the code builds the stock list as "every name that is *ever* in the universe" and computes 2010 cross-sectional ranks over that list, a small company that will enter the top 1,000 in 2015 sits in the 2010 cross-section. It is there *because* it later grew, so its 2010–2015 returns are high by selection. The 2026-11-04 audit found exactly this bug in an earlier `apply_mask`: future entrants earned **+110% annualised in the 63 days before entry** versus −6% for the universe average, and 21% of training rows were out of the universe at their date. It was invisible on the default synthetic market, where every name is always in the universe.

**Guards.**
- **The eligibility panel.** `qts/data/universe.py::apply_mask` stores the per-date membership on `MarketData.eligible`. Every cross-sectional statistic (winsorisation, ranks, labels, IC, emitted scores, risk-style standardisation) is computed over `md.eligibility()` **at that date only**, and ineligible rows are dropped, not zero-filled. The simulator reads the same panel: ineligible names cannot be opened, and held names that drop out are unwound over 5 days. `tests/test_pipeline.py::test_pipeline_held_names_are_eligible` checks the result.
- **The universe itself is PIT.** `build_universe_mask` uses only trailing ADV, the current market cap and the current as-traded price, refreshed on the first trading day of each month with hysteresis that looks only at past refreshes. → see **Data & universe › Universe rules**.
- **Delistings stay in.** `qts/data/universe.py::survivorship_check` warns if no ticker's price series ends before the panel's last date, the signature of a survivorship-biased source. Labels of names that delist inside the forward window use the return to the last close times $1 + $ `CostConfig.delisting_haircut` (−30%, the single source of the delisting assumption for labels and the backtest) instead of being dropped, and the simulator applies the same haircut to held names that vanish.
- **Halts are not delistings, decided without the future.** A missing close at the label's window end is a delisting if `MarketData.delisting_dates` says so; without a delisting table, it is a halt if a close reappears within `AlphaConfig.delisting_gap_days = 63` trading days and a delisting otherwise, decided on the day that becomes known. The earlier rule used a backward fill over the whole panel, so a halt that resumed after $t$ changed labels "realised" at $t$ (second audit, HIGH finding 1); `tests/test_label_halts.py` checks that labels seen at $t$ are identical whether the data end at $t$ or later.

**Leakage kind 4: restated fundamentals and period-end dating.**

**Example A (period-end dating).** A company's fiscal quarter ends 31 March; it files its 10-Q on 5 May. A dataset keyed on the period end shows the March figures from 31 March, five weeks before anyone could have seen them, and around earnings that is exactly when prices move most.

**Example B (restatement).** The company later files a 10-K/A in November restating March's total assets. A "most recent" dataset overwrites the original value, so a 2019 backtest uses numbers that were not published until 2020. Restatements are not random: they cluster in troubled firms.

**Guards.**
- **Key on the filing date.** `qts/data/loaders.py::sharadar_sf1_to_pit_fundamentals` makes every field visible from Sharadar's `datekey` (the filing date) plus `lag_days` (default 1, because many filings arrive after the close), never from `calendardate` (the normalised period end), and warns if a restating `MR*` dimension is requested. The default dimension is `ART` (as-reported trailing twelve months).
- **A late restatement of an older period does not become "current".** A 10-K/A for last year filed after this year's first 10-Q feeds only the prior-year lookup for asset growth; it does not replace the current period in the level panels.
- **`merge_asof` on filing date for derived fields.** Year-over-year asset growth needs last year's assets. The loader takes the prior-year value from the latest filing for that period whose filing date is **on or before the current row's filing date** (`pd.merge_asof(..., direction="backward")`), so a restatement filed later cannot leak into an earlier row. An earlier version used `drop_duplicates(keep="last")`, which picked up restated values; the audit flagged it and this is the fix.
- **`pit_forward_fill`** makes any `(ticker, available_date, value)` row visible from the first trading day on or after its availability date and forward-fills until superseded; nothing is visible earlier. The same helper applies FINRA's publication lag to short interest.

### The backstop: the planted-oracle test and the suspect gate

The four guards prevent known kinds of leakage. The backstop catches the unknown ones by their symptom. `qts/validation/leakage.py::planted_oracle_test` feeds the *future* 20-day return in as a feature and checks that the backtest reports a Sharpe above `SUSPECT_SHARPE = 2.0`, i.e. that the "too good to be true" gate fires; it also fails if the backtest ignores the feature entirely (a sign of broken wiring). `ResearchPipeline.evaluate` adds a note to any run with net Sharpe above 2, and the hurdle report includes a required `suspect` gate.

<callout icon="⚠️" color="yellow_bg">
Known limitation: `planted_oracle_test` currently exercises the toy `simple_rank_backtest`, not the full `ResearchPipeline`, because there is no hook for injecting an extra feature panel into the pipeline (audit `alpha_validation_data.md` finding 11). The pipeline's suspect gate is the same threshold, but it has not been shown end to end to flag an oracle.
</callout>

### Why this choice

Statistical detection (deflated Sharpe, PBO, holdouts) cannot see leakage, and reviewer vigilance does not scale to dozens of features. The plan follows Gençay (2026) instead: compute features through functions that cannot see future rows, and test the property ("same answer on truncated data") rather than inspecting code.

### In the code

| Leakage kind | Guard | Location |
|---|---|---|
| Look-ahead in features | trailing-only construction; `slice_until`; `pit_shift_test`; `feature_registry` | `qts/alpha/features.py`, `qts/core/types.py`, `qts/validation/leakage.py` |
| Label overlap | exact purge + embargo in walk-forward training; purged/embargoed splitters | `qts/alpha/trainer.py::embargoed_train_mask`, `qts/validation/splits.py` |
| Survivorship / membership | `eligible` panel and `eligibility()`; PIT universe; `survivorship_check`; causal halt/delisting rule for labels; delisting haircut in the simulator | `qts/data/universe.py`, `qts/core/types.py`, `qts/alpha/targets.py`, `qts/backtest/simulator.py` |
| Restated / period-end fundamentals | `datekey` + 1 day keying; `merge_asof` on filing date; older-period restatements kept out of the level panels; `MR*` warning; `pit_forward_fill` | `qts/data/loaders.py` |
| Publication lags in sessions | FINRA settlement + 8 NYSE sessions (holidays skipped) | `qts/data/calendar.py`, `qts/data/loaders.py` |
| Unknown leakage | `planted_oracle_test`; suspect gate (net Sharpe > 2) | `qts/validation/leakage.py`, `qts/pipeline.py` |

Tuning knobs: `AlphaConfig.embargo_days` (default = horizon), `AlphaConfig.delisting_gap_days = 63`, `ValidationConfig.embargo_days = 20`, `ExecutionConfig.signal_lag_days = 1`, the `lag_days` argument of the Sharadar loader (default 1; about 90 for sources with period dates only), the `lag_business_days` argument of the FINRA loaders (8 NYSE sessions).

### Pitfalls & checks

- **Dated industries need data from the panel start.** With `industry_history` supplied, `industry_as_of(t)` uses the label in force at $t$. A store fills dates before a ticker's first dated label with `"UNKNOWN"`, never with the static map (which in a store is the *current* classification, a look-ahead). Without `industry_history`, the static label applies to the whole history, a mild look-ahead in industry demeaning and the industry factors.
- **The close as both information and price.** With `signal_lag_days = 0` the simulator forms signals and trades on the same close. That is not leakage of future data, but it is not executable either (the order is due at 15:50 ET), so it is kept only as an optimistic bound (→ see **Start here › The life of one trading day**).
- **New panels need new tests.** Any new feature or data panel needs a `pit_shift_test` entry and, if it is cross-sectional, must go through `md.eligibility()`; add it to the hypothesis properties in `tests/test_pit_properties.py`.
- **The simulator cannot tell a halt from a delisting on the day it happens.** A held name with no close on $t+1$ is haircut and liquidated, whatever happens later. The label rule above is causal and distinguishes the two; the simulator's rule is conservative.

### Worked example: catching full-sample demeaning

Take one stock with daily returns $(+1\%, -1\%, +3\%, +5\%)$ on days 1–4, and a "feature" equal to the return minus the full-sample mean. On the full history the mean is $(1 - 1 + 3 + 5)/4 = 2\%$, so the feature on day 2 is $-1\% - 2\% = -3\%$. Cut the data at day 2: the mean of $(+1\%, -1\%)$ is $0\%$ and the feature on day 2 is $-1\%$. The two answers for day 2 differ, so `pit_shift_test` returns `False`. A trailing feature, such as the return minus the mean of the previous 20 days, gives the same day-2 value on both inputs, because it never reads day 3 or 4.

### References

- [Gençay (2026)](https://arxiv.org/pdf/2608.27734)
- [AlgoXpert protocol](https://arxiv.org/pdf/2603.09219)
- [QuantRocket, Sharadar fundamentals documentation](https://www.quantrocket.com/docs/data/fundamental/sharadar/)

## The run lifecycle

### What it is

A research run is one call to `ResearchPipeline.run(md)`, which goes through seven stages: **fit → as_of per date → solve → simulate → evaluate → ledger → run directory**.

```text
 ResearchPipeline.fit(md)
   ├─ data_hash_of(md)                           every panel + eligibility (RunManifest)
   ├─ build_universe_mask(md, cfg.universe) ──► apply_mask ──► md_u (eligible attached)
   ├─ build_risk_model(cfg.risk).fit(masked_levels(md_u))
   │                                             exposures, factor & specific returns
   ├─ AlphaModel(cfg.alpha, signal_lag_days=L).fit(md_u, risk_exposures_fn)
   │                                             features, risk-only residualised labels,
   │                                             composite, walk-forward ML predictions
   └─ TransactionCostModel(cfg.cost, cfg.execution); SinglePeriodOptimizer(cfg.portfolio, cost)
 ResearchPipeline.run(md)
   ├─ backtest_dates()                           after risk warm-up, first OOS ML date, + L days
   └─ DailyBarSimulator.run(md_u, dates, alpha_fn, risk_fn, optimizer, signal_lag_days=L, ...)
        for t in dates:  s = t - L
                         snapshot(s) → alpha.as_of(s) → risk.as_of(s, alpha) → optimizer.solve
                         → cost_model.estimate(snapshot(t), "routed") → earn t→t+1 → NAV identity
 ResearchPipeline.evaluate(bt)
   ├─ IC report and value-weighted decile spread (every H days, eligible names, lag-aligned)
   ├─ purged walk-forward fold Sharpes, cost sensitivity, breakeven cost
   ├─ TrialLedger.log(... code_version, run_spec)   family pipeline_phase{p}:{data_hash[:12]}
   ├─ deflated Sharpe from the ledger family; PBO if a config grid is set
   └─ HurdleReport.evaluate_signal(...) ──► PipelineResult
 save_run(res, cfg, runs_root) + qts.runs.artifacts.write_extra_artifacts ──► runs/<run_id>/
```

### Why we need it here

The split between `fit` and `as_of` is what makes a fast backtest compatible with point-in-time correctness. `fit` precomputes whole histories once with vectorised, causal operations (rolling windows, per-date regressions, walk-forward ML that only predicts out of sample); `as_of(t)` then reads only rows up to $t$ from those histories. The alternative, refitting every model from scratch on `md.slice_until(t)` for each of several thousand dates, is PIT by construction but hundreds of times slower. The audit's verification is what licenses the shortcut: refitting on truncated data gives identical `as_of` outputs.

### How it works

1. **Fit.** `ResearchPipeline.fit` hashes the input data, builds the universe mask over the whole sample (each row uses only data up to that row) and attaches it. The risk model is fitted on `masked_levels(md_u)`, a view with market cap and volume blanked outside the universe, so its estimation universe is unchanged by the second audit's `apply_mask` change. It is built by `qts/risk/model.py::build_risk_model`, which returns a `HybridRiskModel` (fundamental model plus a statistical companion on its residuals) when `RiskConfig.statistical_companion = True`, otherwise a `BarraStyleRiskModel`. The alpha model is fitted with a hook, `ResearchPipeline._exposures_fn`, that supplies the risk model's **risk-only** exposures (market, industries and the styles in `qts/core/roles.py::RISK_ONLY_STYLES = ("beta", "size", "residual_vol", "liquidity", "short_term_reversal")`, the same constant that tells the optimiser which styles to band) for label residualisation; residualising on the alpha themes would define the alpha away. Before the risk model's warm-up ends the hook returns `None` and the label builder uses its own risk-only style set; after it, any failure raises rather than silently changing the label.
2. **as_of per date.** For execution date $t$ and signal date $s = t - L$, the simulator calls `alpha_fn(s)` (the eligibility-filtered `AlphaModel.as_of`) and then `risk_fn(s, alpha)` (`risk.as_of(s, alpha=...)`, passing the alpha for the alignment factor when `RiskConfig.alignment` is on).
3. **Solve.** `SinglePeriodOptimizer.solve` receives alpha, risk, current weights and NAV marked at $t$, the `LiquiditySnapshot` as of $s$ (with $t$'s auction share), the no-trade list, forced-exit caps and `neutral_styles` (the risk-only styles to band), and searches the risk aversion that puts the book at the vol target.
4. **Simulate.** The simulator books trades at the close of $t$ with costs from `estimate` on the snapshot of $t$ at the routed venue, charges borrow, applies returns $t \to t+1$, handles delistings and records everything.
5. **Evaluate.** `ResearchPipeline.evaluate` scores, every $H = 20$ execution days (non-overlapping), the score that was traded there (`alpha.as_of(t - L)`) against that signal date's label, for the IC report and the value-weighted top-minus-bottom decile spread. Fold Sharpes are the contiguous test blocks of `PurgedWalkForward(n_splits=6, embargo_days=20)` over the backtest's return dates (blocks shorter than `min_fold_days = 20` merged away); the same blocks drive the `ic_sign` gate. It also computes cost sensitivity and breakeven cost.
6. **Ledger.** The run is logged to the trial ledger (if `ledger_path` is set) under family `pipeline_phase{p}:{data_hash[:12]}`, with its code version and run spec; any PBO grid configurations are logged as trials of the same family; then the deflated Sharpe ratio is computed from the family's effective trial count and cross-trial Sharpe variance, PBO is computed if `SystemConfig.pipeline_pbo_configs` lists a grid, and `HurdleReport` issues PASS / FAIL / skipped per gate. Required gates (`phase_sharpe`, `suspect`, `dsr`) fail `overall` when missing.
7. **Run directory.** With `runs_root` set, `save_run` writes `runs/<run_id>/` and `qts.runs.artifacts.write_extra_artifacts` adds the UI files (→ see **Reproducibility › The run manifest and the run registry**).

```python
from qts.data.synthetic import generate, SyntheticSpec
from qts.pipeline import ResearchPipeline, format_result, load_run

md  = generate(SyntheticSpec(n_stocks=300, n_days=1512))
res = ResearchPipeline(use_ml=False, ledger_path="runs/trial_ledger.jsonl", runs_root="runs").run(md)
print(format_result(res))
rec = load_run(res.run_dir)          # typed re-load: manifest, config, summary, frames
```

### Why this choice

A single-period optimiser re-run daily, with a turnover-aware cost term and half-life shrinkage of alpha, is the build plan's recommendation over multi-period optimisation (MPO), because MPO adds cost roughly linear in the planning horizon and the evidence that it beats a well-tuned single-period rule on single stocks was not found. The pipeline orders alpha before risk on each date because the risk model's alignment factor needs the alpha vector.

### In the code

`ResearchPipeline(cfg=None, use_ml=True, phase=None, ledger_path=None, composite_weight=None, ml_factory=None, rebalance_every=1, venue="routed", nav0=1_000_000.0, warmup_days=None, runs_root=None)`; `use_ml=False` is phase 1 (rank composite only), and `composite_weight=None` means `AlphaConfig.composite_weight = 0.5`. The CLI and the API use NAV \$5M so that fixed per-order fee minimums do not dominate costs (`docs/REVIEW_NOTES.md`). `run(md, dates=None, max_days=None)` restricts the backtest window. A changed `cfg.execution.signal_lag_days` triggers a full `fit` again inside `run` (labels, purge and IC windows depend on it), so `qts demo --compare-lag` fits twice. Fitted components are exposed for inspection: `universe_mask`, `md_u`, `risk`, `alpha`, `cost`, `optimizer`, `timings_`, `data_hash_`.

### Pitfalls & checks

- **Warm-up.** The backtest starts only after the risk model has `RiskConfig.min_history_days = 252` days of factor returns and, if ML is on, after the first out-of-sample ML prediction (`AlphaConfig.train_min_days = 756` days of training data), plus $L$ days so that the first signal date is itself out of sample (a lag of 2 or more used to crash on the first date). A five-year sample therefore yields only a couple of years of backtest when ML is on.
- **Label definition across warm-up.** Labels before the risk model's first exposure date use the internal risk-only style set; after it, the risk model's exposures. Both are risk-only, so the label means the same thing, but the exposures differ in detail. The pipeline logs this once.
- **PBO skipped by default.** A single configuration has no PBO. Set `pipeline_pbo_configs` to a list of `{"section.field": value}` overrides to run the CSCV grid; each entry costs one full backtest and is logged to the ledger as a trial.

## Reproducibility

### What it is

Two runs with the same code, configuration and data must produce the same numbers, and the record of every run must say exactly which code, configuration and data produced it.

### Why we need it here

The deflated Sharpe ratio is only honest if trials are counted honestly, which requires telling a re-run (same experiment) from a new trial. Determinism also exposes bugs (unseeded draws, order dependence) that would make live behaviour unpredictable.

### How it works

Five mechanisms work together: seeds, configuration as data, a data hash, a run manifest with a run directory, and the trial ledger.

**Seeds.** `SystemConfig.seed = 0`; the synthetic generator takes `SyntheticSpec.seed` (default 0) and draws splits, halts and delistings from separate random streams so that turning them on leaves every other panel bit-identical; ML models are seed-bagged with `AlphaConfig.model_seeds = 3`. `tests/test_determinism.py` and `tests/test_pipeline.py::test_pipeline_is_deterministic` re-run and require identical output.

### Configuration as data

Every config dataclass inherits from a small mixin (`qts/core/config.py::_ConfigBase`) that gives it a strict, round-trippable serialisation:

- `to_dict()` returns plain nested dicts (tuples become lists, numpy scalars become Python scalars), so the result is JSON- and YAML-safe;
- `from_dict(d)` is **strict**: an unknown key raises `KeyError` naming the known keys (a typo in a YAML file must never be silently ignored), a missing key takes the dataclass default, nested sections are rebuilt recursively, and a `schema_version` key is accepted and ignored (`CONFIG_SCHEMA_VERSION = 1`);
- `to_yaml()` / `from_yaml(path)` use PyYAML's safe dumper and loader;
- `hash()` is the first 16 hex characters of the SHA-256 of the canonical JSON (sorted keys, compact separators), stable across processes and machines;
- `SystemConfig.with_overrides({"portfolio.risk_aversion": 5, "seed": 1})` returns a deep copy with dotted-path overrides, also strict;
- `SystemConfig.validate()` checks that `from_dict(to_dict())` reproduces the hash (catching non-serialisable values set in code); `CostConfig` and `ExecutionConfig` additionally check semantic bounds on construction (non-negative fees, $1 < e \le 2$ for the impact exponent, an integer `signal_lag_days` between 0 and 10).

```python
def hash(self) -> str:
    """Short SHA-256 of the canonical JSON: two configs with identical fields
    share the hash, so the trial ledger can de-duplicate identical re-runs."""
    return hashlib.sha256(self.to_json().encode()).hexdigest()[:16]
```

`configs/default.yaml` is not hand-written: it is `qts config dump` of `SystemConfig()` with a "GENERATED FILE" header, and `tests/test_config.py` fails when the file differs from the dump. `qts config validate my.yaml` strict-loads a file and prints its hash.

### The data hash

`qts/core/manifest.py::hash_market_data(md)` (called through `qts/pipeline.py::data_hash_of`) is a SHA-256 over **every** `MarketData` panel: required and optional fields, each `extra` entry (sorted by key), the industry map, and `md.eligibility()`, including index and column labels, so a relabelled panel changes the hash. Absent optional panels contribute a `<name:None>` marker so that "missing" differs from "empty". (`delisting_dates` is skipped when absent, so hashes of older data did not change when it was added.) The earlier hash covered only the close panel, so two datasets with identical prices and different fundamentals looked like the same data. For a stored dataset, `ParquetStore.snapshot_hash(name)` and `snapshot_hash_all()` hash the Parquet files themselves.

### The run manifest and the run registry

`qts/core/manifest.py::RunManifest` answers "what exactly produced these numbers?":

| Field | Content |
|---|---|
| `run_id` | `YYYYmmddTHHMMSSffffffZ_<config_hash[:8]>`, sortable by creation time |
| `created_at` | ISO-8601 UTC |
| `config_hash`, `data_hash` | as above |
| `git_sha`, `git_dirty` | `git rev-parse HEAD`; dirty if any tracked file is modified **or** any untracked, non-ignored file exists (a new module the run imports changes the code) |
| `versions` | numpy, pandas, scipy, scikit-learn, LightGBM, cvxpy, Clarabel, pyarrow, PyYAML |
| `python_version`, `platform`, `timings`, `seeds`, `command`, `notes` | environment and provenance |

With `runs_root` set, the pipeline writes one directory per run:

```text
runs/<run_id>/
  manifest.json     RunManifest
  config.yaml       SystemConfig.to_yaml() with a header naming the run and hash
  summary.json      backtest summary, IC scalars, decile, DSR and PBO, folds, IC decay,
                    notes, flags (use_ml, phase, signal_lag_days)
  nav.parquet  weights.parquet  trades.parquet  costs.parquet  returns.parquet  ic.parquet
  hurdles.json      HurdleResult.to_dict()
  extras.json  exante_vol.parquet  exposures.parquet  holdings.parquet   (qts/runs/artifacts.py)
```

The last row is written by `qts.runs.artifacts.write_extra_artifacts` right after `save_run`, because only the in-memory pipeline knows them: run facts (data kind, universe size, seed, venue), DSR inputs (skew, kurtosis, observation count), mean IC by calendar year, factor roles, the annualised ex-ante volatility of each day's book, the book's net exposure to the market and every style factor, and month-end holdings (the last 60) with the alpha the optimiser saw, specific volatility and industry. It is best effort: a failure is logged and never fails the run. `load_run(path)` re-loads a directory as a typed `RunRecord`; `list_runs(root)` lists directories newest first. The `qts runs` subcommands and the API's run registry are readers of this layout (→ see **Operating guide**).

### The trial ledger

`qts/validation/ledger.py::TrialLedger` is the input to the deflated Sharpe ratio, so it is treated like an accounting ledger.

- **Strictly append-only.** Every `log` appends one complete JSON line under an exclusive `fcntl.flock` on `<path>.lock`; readers take the shared lock. Nothing ever rewrites the file. The previous version rewrote the file to de-duplicate, and the second audit showed two concurrent writers losing 105 of 401 trials.
- **De-duplication at read time.** `log(..., replace_identical=True)` stamps the record with a `dedup_key`, the SHA-256 of `(name, config_hash, data_hash, code_version, run_spec)`. When the file is read, records sharing a key collapse into the latest one, carrying `n_reruns` and `ts_first`. `run_spec` (`ResearchPipeline.run_spec`) holds everything outside `SystemConfig` that changes the numbers: `use_ml`, phase, the identity of the ML factory (module, name and a hash of its code), `composite_weight`, `rebalance_every`, `venue`, `nav0`, `warmup_days`, the backtest dates and the signal lag.
- **Dirty code never de-duplicates.** `code_version` is the git sha for a clean checkout and `<sha>+dirty` otherwise (`unknown` outside git). A record with a dirty or unknown code version, or an empty data hash, is always its own trial, because the code may differ from the last run in ways nobody recorded.
- **Families per phase and data set.** The pipeline logs under `pipeline_phase{p}:{data_hash[:12]}`, so trials on a synthetic seed and on real data never pool, while signal-lag and config variants on the same data do.
- **Effective trials.** Nyholt's eigenvalue formula on the correlation matrix of the stored monthly return streams, used unrounded, so ten near-identical variants count as fewer than ten independent trials.

### Why this choice

A JSON-lines file is greppable, diffable and shareable, and `flock` makes it safe for several processes on one machine (the API's job workers, a CLI run, a notebook) to append at once. The docstring states the rule: it must never be reset, since deleting trials lets selection bias go unpunished. A run directory of plain JSON, YAML and Parquet can be read by pandas, DuckDB or the browser API without the code that wrote it.

### Pitfalls & checks

- **Runs from a dirty tree are not reproducible from the sha.** The lag investigation traced a 0.84 Sharpe "effect" to a demo run made from an uncommitted intermediate tree that no longer existed. Stored reference outputs should come from clean trees (`git_dirty: false` in the manifest).
- **`flock` is local.** Locks are advisory and not reliable on NFS; keep the ledger and the store on a local disk.
- **Library versions matter.** Solver and LightGBM versions can change results at the fourth decimal; the manifest records them, but pin them.
- **Old ledger records.** Records written before the family change carry the bare family `pipeline_phase1`; they no longer pool with new runs.

### References

- [Gençay (2026)](https://arxiv.org/pdf/2608.27734)

## Testing strategy

### What it is

The test suite (about 300 test functions under `tests/`, plus about 40 in `api/tests/` and the UI's own Vitest and Playwright suites) runs entirely offline. It is split into a fast tier (`-m "not slow"`, every push) and a slow tier (`-m slow`: walk-forward ML and full pipelines, nightly); → see **Operating guide › Tests and CI**. It combines ordinary unit tests (fee arithmetic, parser fixtures, solver constraints) with **oracle tests** on a synthetic market whose true structure is known: which stocks have which factor exposures, what the factor returns were, and above all which latent signal drives returns and how strong it is.

### Why a known-truth generator is valuable

On real data there is no answer key. If a new alpha model reports an IC of 0.03, you cannot tell whether the model is weak, the data is noisy, or a bug halves the signal; if it reports 0.30, you cannot immediately tell whether it is brilliant or leaking. A generator with a **planted signal of known strength** turns these into checkable assertions:

- **Recovery.** The planted latent score has a known strength against the residualised 20-day target (`SyntheticSpec.ic_true = 0.04`; realised about 0.036 because the score decays within the window). `tests/test_synthetic_oracle.py::test_planted_residual_ic_band` asserts the realised IC lands in a band below 0.04 across seeds, and `tests/test_alpha.py::test_planted_truth_ic` checks the same on the alpha fixtures. If the label builder, residualisation or IC code were wrong, they would fail.
- **Timing.** `test_score_at_t_predicts_only_after_t` checks that the score revealed at $t$ predicts returns after $t$ and not the return of $t$ itself.
- **No more than the truth.** A model that recovers *more* skill than was planted is leaking. The oracle band gives an upper bound that real data never provides.
- **End-to-end transmission.** Feeding the true score through the whole chain (Grinold scaling, risk model, optimiser, costs, simulator) should produce a positive net Sharpe and a realised-to-ex-ante volatility ratio near 1. The review notes record gross 1.6, net 1.3 and a ratio of about 0.9 (in-sample, on the earlier same-month-timing generator), and `tests/test_backtest.py::test_planted_alpha_positive_net_sharpe` guards it. This separated "the composite is weak" (an alpha question) from "the plumbing loses signal" (an engineering question) during the shakedown.
- **Risk model truth.** The generator emits the true betas and market returns (`md.extra["true_beta"]`, `["true_market_return"]`), so estimated exposures and factor returns can be compared with the truth. The shakedown found that a latent factor with no observable footprint made optimised books run at 1.8× their ex-ante risk; the generator was changed so every latent style is observable, and the finding motivated the statistical companion risk model.
- **Determinism.** A seeded generator makes every failure reproducible.

### How it works

| Test family | What it pins down | Examples |
|---|---|---|
| Oracle recovery | Planted IC recovered within a band, never above `ic_true`; score predicts only after $t$; IC decay accumulates for a persistent score | `test_synthetic_oracle.py`, `test_alpha.py::test_planted_truth_ic`, `test_half_life_and_decay` |
| End-to-end transmission | Planted alpha yields positive net Sharpe; NAV identity exact | `test_backtest.py::test_planted_alpha_positive_net_sharpe`, `test_nav_identity` |
| Leakage tripwires | `pit_shift_test` catches `shift(-1)` and full-sample statistics; oracle feature flagged; negative shifts refused | `test_validation.py::test_pit_shift_test_catches_lookahead`, `test_planted_oracle_is_flagged`, `test_feature_registry_refuses_negative_shift` |
| Constructed fixtures | Hand-built panels with known answers: universe hysteresis, delisting liquidation, survivorship warning | `test_data.py::test_universe_hysteresis_constructed`, `test_backtest.py::test_delisting_is_liquidated_and_flagged`, `test_data.py::test_survivorship_check` |
| Contracts | Held names are eligible; snapshot keeps venue/date/spread; unknown cost columns raise | `test_pipeline.py::test_pipeline_held_names_are_eligible`, `test_backtest.py::test_cost_frame_raises_on_unrecognised_or_nan` |
| Reproducibility | Deterministic pipeline; ledger de-duplicates identical clean re-runs; concurrent ledger and store writers lose nothing; manifests round-trip; config round-trips and `default.yaml` matches the dump | `test_determinism.py`, `test_pipeline.py::test_pipeline_ledger_dedupes_reruns`, `test_ledger_concurrency.py`, `test_store_concurrency.py`, `test_manifest.py`, `test_config.py` |
| Hand-computed fixtures | A 3-stock × 5-day simulator path checked against arithmetic done by hand | `test_simulator_handcomputed.py` |
| Golden file | `ResearchPipeline(use_ml=False).run(md, max_days=60)` on a fixed synthetic market compared field by field with `tests/golden/pipeline_summary.json` | `test_golden.py` |
| PIT properties | hypothesis-drawn cut dates over features, universe, cost snapshots, risk `as_of`; halts that resume after $t$; `apply_mask` keeps level history | `test_pit_properties.py`, `test_label_halts.py`, `test_apply_mask_history.py` |
| Execution delay | Lag 0 vs 1 through the real simulator; lag ≥ 2 does not crash | `test_signal_lag_real.py` |
| Shared statistics | Every `qts/core/stats.py` function vs a brute-force reference | `test_core_stats.py` |
| Calendar | NYSE holidays and session arithmetic | `test_calendar.py` |
| Parsers | Ken French, FINRA, Sharadar (split-consistent ratios, restatements), FF48 mapping on inline fixtures (no network) | `test_data.py` |

### Pitfalls & checks

The synthetic market is only as good as what it contains, and the audits' sharpest lesson was that **green tests on a synthetic market are not evidence about bugs the synthetic market cannot express**. The default market has a static universe (with 300 names and `top_n_by_mcap = 1000`, every name is always eligible), no IPOs, no restatements and no filing lags. The critical "ever in universe" leak and the dropped delisting labels were invisible to every test until the first audit built a smaller universe and planted delistings by hand, and the second audit's halt finding was invisible because the generator had no halts. The generator now can plant splits, halts and delistings (`SyntheticSpec.n_splits`, `n_halts`, `n_delistings`; off by default so the default market is bit-identical), but the general rule stands: when you add a guard for a real-data hazard, add a constructed fixture that exhibits the hazard. → see **Data & universe › The synthetic market** for the generator's equation and limitations.

<callout icon="⚠️" color="yellow_bg">
A golden-file diff is a question, not an obstacle. Regenerate it (`pytest tests/test_golden.py --update-golden`) only after attributing every moved field to an intended change, as `docs/notes/golden_changes_2026-11-05.md` does step by step. A 60-day Sharpe has a standard error of about $\sqrt{252/60} \approx 2$, so the golden file guards against unintended drift and says nothing about strategy quality.
</callout>

## Extension points

Each recipe lists the files to touch, the contract to respect and the test to add. Any new behavioural knob goes into a `SystemConfig` section (then regenerate `configs/default.yaml` with `qts config dump > configs/default.yaml`), never into a module-level constant, so it reaches the config hash, the run directory and the UI's launch form.

### Adding an alpha feature

1. **Compute it** in `qts/alpha/features.py::FeatureBuilder.build` as a wide panel from `MarketData`, using only `shift(k >= 0)` and trailing windows. If it needs a new data panel, add the panel to `MarketData` (and to `slice_until`, `apply_mask` and `ParquetStore` dataset names; the data hash picks up new fields automatically). If it is a price *level*, read `md.trade_price()`, not `md.close`.
2. **Register its position** in `FEATURE_ORDER`. The order is part of the train/live parity hash (`feature_signature`), so appending changes the hash deliberately.
3. **Decide its role.** If it belongs to an alpha theme, add it to `qts/alpha/composite.py::THEMES` with a sign, and to `qts/core/roles.py::ALPHA_THEME_FEATURES` so it can never be used for label residualisation (an import-time assertion checks that the risk-only and alpha-theme sets do not overlap). If it should be industry-demeaned, add it to `AlphaConfig.industry_demean`.
4. **Test it.** Add a `pit_shift_test` case, and if it encodes a hazard the synthetic market lacks, a constructed fixture.
5. **Gate it.** A new signal must pass the hurdles in plan §2.4: incremental rank IC over the existing composite > 0.005 in ≥ 2/3 of folds and net Sharpe improvement ≥ 0.05; t > 3 applies only to standalone signals (`HurdleReport.evaluate_signal(standalone=...)`). Every attempt goes in the ledger. → see **Validation & statistics › The hurdle gates**.

### Adding a risk factor

1. **Add the style** to `qts/risk/exposures.py::SUPPORTED_STYLES` and a branch in `raw_style_panels` computing the raw descriptor causally; standardisation (cap-weighted mean 0, unit standard deviation, trimmed at ±3 standard deviations and re-standardised, missing values set to 0) is applied by `standardize_style`.
2. **Enable it** in `RiskConfig.styles`. `factor_groups_for` places it in the `"style"` group automatically, so the optimiser can band it.
3. **Decide whether it is risk-only.** If labels should be residualised on it and the optimiser should hold it near zero, add it to `qts/core/roles.py::RISK_ONLY_STYLES` together with an alpha-layer proxy feature in `RISK_ONLY_FEATURE_PROXIES` (used during the risk model's warm-up). If it is an alpha theme (value, momentum, quality, investment, short interest), add it to `ALPHA_THEME_STYLES` instead: neutralising an alpha theme removes the alpha. The label, the optimiser's style bands and the UI's factor roles all read these two tuples (design decision D1).
4. **Check it.** Run the risk evaluation (bias statistics within [0.88, 1.12] on factor and random portfolios, QLIKE) before and after (`qts/risk/evaluation.py`). → see **Risk model › Evaluating a risk model**.

### Adding a cost component

1. **Implement both views** in `qts/tcost/model.py::TransactionCostModel`, the one production `CostModel`: the USD point estimate in `estimate` (folded into `spread_usd`, `impact_usd` or `fees_usd`; the `CostEstimate` columns are fixed and the total must equal their sum) and the convex fraction-of-NAV expression in `cvx_terms`. The component must be convex in the trade (linear, power ≥ 1 of absolute value) or the optimiser cannot use it, and it must keep the DPP structure the optimiser caches (array coefficients, not data-dependent expression shapes).
2. **Keep them equal.** The simulator charges `estimate`; the optimiser plans with `cvx_terms`. Extend `tests/test_cost_equivalence.py` so the two agree on random trade vectors at every venue.
3. **Feed it from the snapshot.** Any new input (for example, a per-name auction volume forecast) is a new `LiquiditySnapshot` field built PIT in `TransactionCostModel.snapshot`, with a fill default and a floor so it is always finite.
4. **Put its parameters in `CostConfig`** with bounds in `COST_CONFIG_BOUNDS`, so a typo such as a negative fee fails on construction.

### Adding a portfolio constraint

1. **For a one-off,** pass `constraints_extra` to `SinglePeriodOptimizer.solve`: a list of callables `fn(w, x, tickers) -> list[cp.Constraint]`, where `w` is the post-trade weight variable, `x` the trade expression and `tickers` the problem's ticker index. They are applied to the alpha solve and to the volatility-target projection alike.
2. **For a permanent constraint,** add a field to `PortfolioConfig` and a named constraint in the optimiser's constraint builder, so it appears in `binding_constraints` diagnostics.
3. **Keep it convex** (linear equalities, linear or norm inequalities, second-order cones). Non-convex rules such as "at most 200 names" need a heuristic outside the solve.
4. **Watch feasibility.** A constraint that conflicts with the participation caps or forced exits makes the problem infeasible, and the default policy is to hold the book and report the failure. Check `n_failed_solves` in the summary after any change.

For example, a cap of 15% of NAV on the net weight of a hand-picked list of technology names is one callable that builds a 0/1 indicator vector over `tickers` and returns the single linear constraint "indicator · `w` ≤ 0.15"; passing it in `constraints_extra` applies it on every rebalance without touching the optimiser's code.

### References

- [Boyd et al., Multi-Period Trading via Convex Optimization](https://arxiv.org/pdf/1705.00109)
- [cvxportfolio, Optimization policies](https://www.cvxportfolio.com/en/stable/optimization_policies.html)
