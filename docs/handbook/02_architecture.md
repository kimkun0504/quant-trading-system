# Architecture & data flow

<callout icon="🧭" color="gray_bg">
This page describes how `qts` is put together: the layers and what each package is responsible for, the typed contracts that pass between them (`MarketData`, `AlphaOutput`, `RiskModelOutput`, the cost snapshot, `OptimizationResult`, `BacktestResult`), the data-layout and unit conventions every module follows, and the point-in-time discipline that keeps future information out of every decision, with the four kinds of leakage and the specific guard for each. It then follows one research run from `fit` to the trial ledger, explains how runs are made reproducible, why the test suite is built around a synthetic market with a planted signal, and how to extend the system with a new feature, risk factor, cost component or constraint.
</callout>

## The layers

### What it is

`qts` is a single Python package organised as a stack of layers. Lower layers know nothing about higher ones. Every layer talks to its neighbours through a small number of typed containers defined in `qts/core/types.py`, so any layer can be developed, tested or replaced on its own.

```text
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ orchestration   qts/pipeline.py  ResearchPipeline · scripts/run_*.py     │
 ├──────────────────────────────────────────────────────────────────────────┤
 │ evaluation      qts.validation   splits · leakage · ledger · DSR · PBO   │
 │                 qts.backtest     DailyBarSimulator · metrics             │
 ├──────────────────────────────────────────────────────────────────────────┤
 │ decision        qts.portfolio    SinglePeriodOptimizer · alpha_scaling   │
 │                                  costs (CostTerms protocol, snapshot)    │
 │                 qts.execution    orders · router · scheduler · ibkr      │
 ├──────────────────────────────────────────────────────────────────────────┤
 │ models          qts.alpha        features · targets · composite · ML     │
 │                 qts.risk         exposures · factor returns · covariance │
 │                 qts.tcost        fees · impact · borrow · TCA            │
 ├──────────────────────────────────────────────────────────────────────────┤
 │ data            qts.data         synthetic · store · universe · loaders  │
 ├──────────────────────────────────────────────────────────────────────────┤
 │ core            qts.core         types (contracts) · config · crosssection│
 └──────────────────────────────────────────────────────────────────────────┘
        each layer imports only from layers below it (one exception, noted)
```

The single exception to strict layering is deliberate and lazy: `qts.portfolio.costs` delegates to `qts.tcost.TransactionCostModel` when that package is importable, but imports it only at call time, so `qts.portfolio` never depends on `qts.tcost` at import time. The optimiser and simulator see a cost model only through the `CostTerms` protocol (two methods, structural typing), so any object with those methods plugs in.

### Why we need it here

Five models must agree on units, dates and tickers. The classic failure is a mismatch between two of them: a 20-day alpha fed to an optimiser assuming daily units, a cost in basis points read as a fraction. The README calls the build a **thin vertical slice of all five models**, wired end to end before any is deepened so such failures appear early; clear contracts let each layer deepen without rewiring.

### Package-by-package responsibilities

| Package | Responsibility | Main entry points | Produces |
|---|---|---|---|
| `qts.core` | Shared contracts, configuration, cross-sectional helpers (winsorise, z-score, rank-gauss, neutralise, rank IC) | `types.py`, `config.py::SystemConfig`, `crosssection.py` | `MarketData`, `AlphaOutput`, `RiskModelOutput`; config dataclasses |
| `qts.data` | Get data into a `MarketData`; decide the investable universe | `synthetic.generate`, `store.ParquetStore`, `universe.build_universe_mask`, `universe.apply_mask`, `loaders.*` | `MarketData` with `eligible` set |
| `qts.alpha` | Forecast residual returns | `AlphaModel.fit` / `as_of`, `FeatureBuilder`, `ThemeComposite`, `WalkForwardTrainer`, `evaluation.ic_report` | `AlphaOutput` |
| `qts.risk` | Forecast the covariance of returns | `build_risk_model` → `BarraStyleRiskModel` or `HybridRiskModel`; `.fit` / `.as_of` | `RiskModelOutput` |
| `qts.tcost` | Price trades and holdings; calibrate from fills | `TransactionCostModel` (`snapshot_from_market`, `estimate`, `cvx_terms`, `holding_cost_usd`), `tca.TCALedger` | cost snapshot; cost estimates in USD; convex cost expressions |
| `qts.portfolio` | Turn alpha, risk and cost into target weights | `SinglePeriodOptimizer.solve`, `alpha_scaling`, `vol_target`, `diagnostics` | `OptimizationResult` |
| `qts.execution` | Turn weight changes into orders and route them | `targets_to_orders`, `ExecutionRouter.route`, schedulers, `to_ib_order_dict`, `FillSimulator` | `Order` lists, broker payloads, simulated fills |
| `qts.backtest` | Replay the chain day by day with explicit costs | `DailyBarSimulator.run`, `metrics` | `BacktestResult` |
| `qts.validation` | Decide whether a result is credible | `PurgedWalkForward`, `CombinatorialPurgedCV`, `pit_shift_test`, `planted_oracle_test`, `TrialLedger`, `deflated_sharpe_ratio_annual`, `probability_of_backtest_overfitting`, `HurdleReport` | gate verdicts, DSR, PBO |
| `qts.pipeline` | Wire everything for a research run | `ResearchPipeline.fit` / `run` / `evaluate`, `format_result` | `PipelineResult` |

### Why this choice

The build plan considered an off-the-shelf framework (Qlib) and an event-driven engine (NautilusTrader, LEAN) from the start. It recommends a vectorised custom daily-bar simulator for research sweeps and reserves an event-driven engine for paper and live trading: a daily auction rebalancer needs no tick simulation, and event-driven engines make thousands of research runs slow. Qlib is mined for feature definitions only; its author calls it a foundation, not a production system. Contracts are plain dataclasses of pandas objects, not a database schema or message bus, so any intermediate can be inspected in a notebook.

### In the code

The layering is visible in the imports: `qts/core/types.py` imports only numpy and pandas; `qts/data/universe.py` imports `qts.core`; `qts/pipeline.py` imports every layer. Configuration for every layer lives in one frozen-by-convention tree, `qts/core/config.py::SystemConfig`, with one dataclass per layer (`UniverseConfig`, `AlphaConfig`, `RiskConfig`, `CostConfig`, `PortfolioConfig`, `ExecutionConfig`, `ValidationConfig`). A generated mirror of the defaults is in `configs/default.yaml`; the dataclasses are the source of truth. Configuration serialisation (`from_yaml` / `to_yaml`) is being added as part of the hardening work in `docs/WORK_PLAN.md` item 1.4.

### Pitfalls & checks

- **Hidden coupling through string conventions.** An earlier version of the optimiser found the industry block by the `IND` prefix and the beta factor by name; with Fama–French 48 industry labels the industry bands silently switched off, and "beta-neutral" constrained the market column instead of the beta style. The fix was the `factor_groups` field of `RiskModelOutput` (below), which states each column's role explicitly (`docs/audit/RESOLUTION.md`).
- **Two implementations of one thing.** The cost model briefly existed twice (`SimpleCostModel` and `TransactionCostModel`). The current code makes the former delegate to the latter so the optimiser and the simulator price the same model; the audit verified optimiser ⇄ simulator cost equivalence to machine precision. The planned refactor (`docs/WORK_PLAN.md` item 1.2) replaces this with one `CostModel` protocol in `qts/core`.

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

Six containers carry everything between layers. Field lists below are the stable concepts; exact signatures are being tidied in the concurrent refactor, so check the dataclass before relying on a keyword.

### MarketData

**What it is.** The bundle of wide panels for one universe (`qts/core/types.py::MarketData`). Required: `close` (adjusted close), `volume` (shares traded), `market_cap` (USD), `industry` (a Series mapping ticker to a quasi-static industry label). Optional: `spread_bps` (half-spread proxy in bp), `short_interest_ratio` (short interest / shares outstanding), `borrow_fee` (annual fraction), `book_to_price`, `earnings_yield`, `gross_profitability`, `asset_growth`, an `extra` dict of further wide panels, and `eligible`, the per-date boolean universe mask.

**Derived views.** `returns` (simple close-to-close, `pct_change(fill_method=None)`, so a gap stays NaN), `dollar_volume` (close × volume), `dates`, `tickers`, and two methods that carry the point-in-time discipline:

```python
def eligibility(self) -> Wide:
    alive = self.close.notna()
    if self.eligible is None:
        return alive
    e = self.eligible.reindex(index=alive.index, columns=alive.columns)
    return e.fillna(False).astype(bool) & alive
```

`eligibility()` is the only function the alpha layer uses to decide which names enter a cross-section: in the universe at $t$ **and** priced at $t$. `slice_until(date)` returns a copy with every panel cut at `date`, the point-in-time view used by the leakage tests.

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

`covariance(tickers)` materialises $\Sigma$ for a ticker set (tickers outside the model get zero exposure and the median specific variance, a conservative fill); `portfolio_variance(w)` computes $w^\top \Sigma w$ in factor form without building the $N \times N$ matrix.

**`factor_groups`.** A dict from role to column names, with keys `"market"`, `"industry"`, `"style"`, `"statistical"` and `"alignment"`. The portfolio layer uses it to find the beta style for the beta-neutral constraint, the industry block for industry bands, and the style block for style bands, instead of guessing from column names. It exists because of two audit findings: "beta-neutral" had been constraining the unit-exposure `market` (country) column, leaving the true beta style exposure near 2, and industry bands silently turned off when industry labels stopped starting with `IND`. `qts/risk/model.py::factor_groups_for` fills it; an empty dict means "unknown" and triggers a logged name-based fallback in the optimiser.

### The cost snapshot and the cost protocol

**What it is.** A per-date dictionary of per-ticker Series describing liquidity, built only from data up to the date by `qts/portfolio/costs.py::snapshot_from_market` (which delegates to `TransactionCostModel.snapshot_from_market`). Required keys are `adv_usd`, `sigma_daily`, `half_spread` (fraction), `price` and `borrow_fee` (annual fraction); the tcost model adds pass-through keys such as `half_spread_bps`, `adv_shares`, `venue` and a scalar `date`. Missing values are filled by documented policy (cross-sectional medians and floors).

The cost model is seen by the optimiser and simulator through the `CostTerms` protocol:

- `cvx_terms(x, w_post, nav, snapshot, venue)` returns two convex cvxpy expressions in fractions of NAV: the one-off cost of trade vector $x$ and the cost of holding $w^{\text{post}}$ for one day;
- `estimate(trades_usd, snapshot)` returns a per-ticker frame of `spread`, `impact`, `fees` in USD for the simulator.

**Why one snapshot for both.** If the optimiser's forecast cost and the simulator's charged cost differ, the optimiser systematically over- or under-trades. The audit verified they agree to machine precision.

**Being refactored.** `docs/WORK_PLAN.md` item 1.2 replaces the dict with `LiquiditySnapshot` and `CostEstimate` dataclasses and a `CostModel` protocol in `qts/core`, with `TransactionCostModel` as the only implementation. The concepts (one PIT snapshot, convex terms for the optimiser, USD estimates for the simulator) do not change.

### OptimizationResult

**What it is.** The portfolio layer's answer for one date (`qts/portfolio/optimizer.py::OptimizationResult`): `w_target` (post-trade weights), `x` (trades, $w^{\text{target}} - w^{\text{current}}$), `expected_alpha` ($\alpha^\top w$ over the horizon), `expected_risk` (annualised ex-ante volatility), `expected_cost` (trade cost plus horizon × holding cost, fraction of NAV), `turnover` ($\sum_i |x_i|$), `transfer_coefficient`, solver `status` and `solve_time`, counts of `binding_constraints`, the `vol_scalar` used by the volatility-target projection, `exante_var_daily`, `n_out_of_model` (material names outside the risk model's universe) and `alignment_share` (the share of ex-ante variance coming from the alignment term). `ok` is true for `"optimal"` and `"optimal_inaccurate"`.

**Failure policy.** A failed solve is reported, not hidden: the simulator holds the current book, logs a warning and increments `n_failed_solves` in the summary. Relaxing the ADV participation caps on infeasibility is opt-in (`PortfolioConfig.relax_participation_on_infeasible = False`).

### BacktestResult

**What it is.** The simulator's record of a run (`qts/backtest/simulator.py::BacktestResult`): `nav`; `returns` (columns `gross`, `net`); `weights` (post-trade) and `trades` (weight changes); `costs` in USD with columns `spread`, `impact`, `fees`, `borrow`, `total`; `exante_var`; `turnover`; `transfer_coefficient`; and `diagnostics` (delistings, forced exits, solve log, failed-solve count). `summary()` computes the report card: gross/net CAGR, volatility and Sharpe, Lo-adjusted Sharpe, Newey–West t, drawdown, Calmar, Sortino, exposures, turnover, effective holding days, cost per side, realised versus ex-ante volatility, breakeven cost, delisting and failure counts.

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
| Dates | `pd.DatetimeIndex` of trading days, ascending, tz-naive; trading-day positions for windows and embargoes | Calendar arithmetic on trading days avoids weekend off-by-ones |

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
- `qts/validation/splits.py` provides `PurgedWalkForward` (expanding train, strictly forward test blocks), `PurgedKFold` and `CombinatorialPurgedCV`, all purging label windows that touch the test block and embargoing `ValidationConfig.embargo_days = 20` days after it; `label_overlap_check` and `check_no_overlap` assert the property.

**Leakage kind 3: survivorship and universe membership.**

**Example A (survivorship).** Testing on today's listed stocks omits every company that went bankrupt or was delisted since 2006, so the longs never hold a collapse.

**Example B (universe membership).** The universe at each date is "top 1,000 by market cap". If the code builds the stock list as "every name that is *ever* in the universe" and computes 2010 cross-sectional ranks over that list, a small company that will enter the top 1,000 in 2015 sits in the 2010 cross-section. It is there *because* it later grew, so its 2010–2015 returns are high by selection. The 2026-11-04 audit found exactly this bug in an earlier `apply_mask`: future entrants earned **+110% annualised in the 63 days before entry** versus −6% for the universe average, and 21% of training rows were out of the universe at their date. It was invisible on the default synthetic market, where every name is always in the universe.

**Guards.**
- **The eligibility panel.** `qts/data/universe.py::apply_mask` stores the per-date membership on `MarketData.eligible`. Every cross-sectional statistic in the alpha layer (winsorisation, ranks, labels, IC, emitted scores) is computed over `md.eligibility()` **at that date only**, and ineligible rows are dropped, not zero-filled. The simulator reads the same panel: ineligible names cannot be opened, and held names that drop out are unwound over 5 days. `tests/test_pipeline.py::test_pipeline_held_names_are_eligible` checks the result.
- **The universe itself is PIT.** `build_universe_mask` uses only trailing ADV, the current market cap and the current price, refreshed every 21 days with hysteresis that looks only at past refreshes. → see **Data & universe › Universe rules**.
- **Delistings stay in.** `qts/data/universe.py::survivorship_check` warns if no ticker's price series ends before the panel's last date, the signature of a survivorship-biased source. Labels of names that delist inside the forward window use the return to the last close times $1 + $ `AlphaConfig.delisting_return` (−30%) instead of being dropped, and the simulator applies `CostConfig.delisting_haircut` (−30%) to held names that vanish.

**Leakage kind 4: restated fundamentals and period-end dating.**

**Example A (period-end dating).** A company's fiscal quarter ends 31 March; it files its 10-Q on 5 May. A dataset keyed on the period end shows the March figures from 31 March, five weeks before anyone could have seen them, and around earnings that is exactly when prices move most.

**Example B (restatement).** The company later files a 10-K/A in November restating March's total assets. A "most recent" dataset overwrites the original value, so a 2019 backtest uses numbers that were not published until 2020. Restatements are not random: they cluster in troubled firms.

**Guards.**
- **Key on the filing date.** `qts/data/loaders.py::sharadar_sf1_to_pit_fundamentals` makes every field visible from Sharadar's `datekey` (the filing date) plus an optional `lag_days`, never from `calendardate` (the normalised period end), and warns if a restating `MR*` dimension is requested (the as-reported `ARQ` dimension is the recommended default).
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
| Survivorship / membership | `eligible` panel and `eligibility()`; PIT universe; `survivorship_check`; delisting-aware labels and simulator | `qts/data/universe.py`, `qts/core/types.py`, `qts/alpha/targets.py`, `qts/backtest/simulator.py` |
| Restated / period-end fundamentals | `datekey` keying; `merge_asof` on filing date; `MR*` warning; `pit_forward_fill` | `qts/data/loaders.py` |
| Unknown leakage | `planted_oracle_test`; suspect gate (net Sharpe > 2) | `qts/validation/leakage.py`, `qts/pipeline.py` |

Tuning knobs: `AlphaConfig.embargo_days` (default = horizon), `ValidationConfig.embargo_days = 20`, the `lag_days` argument of the Sharadar loader (0 for SF1, about 90 for sources with period dates only), the `lag_business_days` argument of the FINRA loaders (8).

### Pitfalls & checks

- **Same-day filings after the close.** A filing on date $d$ becomes visible at the close of $d$ with the default `lag_days = 0`. Many 10-Qs and 8-Ks are filed after 16:00, so a signal formed at the close of $d$ could not have seen them. A one-day lag (`lag_days = 1`) is the conservative choice for real data.
- **Static industry labels.** `MarketData.industry` is a single label per ticker for the whole history, so a company reclassified in 2018 carries its 2018 industry back to 2006. This is a mild look-ahead in industry demeaning and the risk model's industry factors.
- **The close as both information and price.** The simulator forms signals and trades on the same close (→ see **Start here › The life of one trading day**). This is a timing simplification rather than leakage of future data, but it is optimistic for fast signals.
- **New panels need new tests.** Any new feature or data panel needs a `pit_shift_test` entry; the work plan adds property-based PIT tests over every registered feature, universe mask, cost snapshot and risk `as_of` (`docs/WORK_PLAN.md` item 1.1).

### Worked example: catching full-sample demeaning

Take one stock with daily returns $(+1\%, -1\%, +3\%, +5\%)$ on days 1–4, and a "feature" equal to the return minus the full-sample mean. On the full history the mean is $(1 - 1 + 3 + 5)/4 = 2\%$, so the feature on day 2 is $-1\% - 2\% = -3\%$. Cut the data at day 2: the mean of $(+1\%, -1\%)$ is $0\%$ and the feature on day 2 is $-1\%$. The two answers for day 2 differ, so `pit_shift_test` returns `False`. A trailing feature, such as the return minus the mean of the previous 20 days, gives the same day-2 value on both inputs, because it never reads day 3 or 4.

### References

- [Gençay (2026)](https://arxiv.org/pdf/2608.27734)
- [AlgoXpert protocol](https://arxiv.org/pdf/2603.09219)
- [QuantRocket, Sharadar fundamentals documentation](https://www.quantrocket.com/docs/data/fundamental/sharadar/)

## The run lifecycle

### What it is

A research run is one call to `ResearchPipeline.run(md)`, which goes through six stages: **fit → as_of per date → solve → simulate → evaluate → ledger**.

```text
 ResearchPipeline.fit(md)
   ├─ build_universe_mask(md, cfg.universe) ──► apply_mask ──► md_u (eligible set)
   ├─ build_risk_model(cfg.risk).fit(md_u)       precompute exposures, factor & specific returns
   ├─ AlphaModel(cfg.alpha).fit(md_u, risk_exposures_fn)
   │                                             features, risk-only residualised labels,
   │                                             composite, walk-forward ML predictions
   └─ TransactionCostModel(cfg.cost); SinglePeriodOptimizer(cfg.portfolio, cost)
 ResearchPipeline.run(md)
   ├─ backtest_dates()                           after risk warm-up and first OOS ML date
   └─ DailyBarSimulator.run(md_u, dates, alpha_fn, risk_fn, optimizer, ...)
        for t in dates:  snapshot(t) → alpha.as_of(t) → risk.as_of(t, alpha) → optimizer.solve
                         → cost_model.estimate → earn t→t+1 → NAV identity
 ResearchPipeline.evaluate(bt)
   ├─ IC report and value-weighted decile spread (every H days, eligible names)
   ├─ calendar-year fold Sharpes, cost sensitivity, breakeven cost
   ├─ TrialLedger.log(config_hash, data_hash, metrics, monthly returns)
   ├─ deflated Sharpe from the ledger family; PBO if a config grid is set
   └─ HurdleReport.evaluate_signal(...) ──► PipelineResult
```

### Why we need it here

The split between `fit` and `as_of` is what makes a fast backtest compatible with point-in-time correctness. `fit` precomputes whole histories once with vectorised, causal operations (rolling windows, per-date regressions, walk-forward ML that only predicts out of sample); `as_of(t)` then reads only rows up to $t$ from those histories. The alternative, refitting every model from scratch on `md.slice_until(t)` for each of several thousand dates, is PIT by construction but hundreds of times slower. The audit's verification is what licenses the shortcut: refitting on truncated data gives identical `as_of` outputs.

### How it works

1. **Fit.** `ResearchPipeline.fit` builds the universe mask over the whole sample (each row uses only data up to that row) and applies it. The risk model is built by `qts/risk/model.py::build_risk_model`, which returns a `HybridRiskModel` (fundamental model plus a statistical companion on its residuals) when `RiskConfig.statistical_companion = True`, otherwise a `BarraStyleRiskModel`. The alpha model is fitted with a hook, `ResearchPipeline._exposures_fn`, that supplies the risk model's **risk-only** exposures (market, industries and the styles in `RISK_ONLY_STYLES = ("beta", "size", "residual_vol", "liquidity", "short_term_reversal")`) for label residualisation; residualising on the alpha themes would define the alpha away. Before the risk model's warm-up ends the hook returns `None` and the label builder uses its own risk-only style set; after it, any failure raises rather than silently changing the label.
2. **as_of per date.** The simulator calls `alpha_fn(t)` (the eligibility-filtered `AlphaModel.as_of`) and then `risk_fn(t, alpha)` (`risk.as_of(t, alpha=...)`, passing the alpha for the alignment factor when `RiskConfig.alignment` is on).
3. **Solve.** `SinglePeriodOptimizer.solve` receives alpha, risk, current weights, NAV, the cost snapshot, the no-trade list, forced-exit caps and `neutral_styles` (the risk-only styles to band).
4. **Simulate.** The simulator books trades at the close of $t$ with costs from `estimate`, charges borrow, applies returns $t \to t+1$, handles delistings and records everything.
5. **Evaluate.** `ResearchPipeline.evaluate` samples the live score every $H = 20$ days (non-overlapping) for the IC report and the value-weighted top-minus-bottom decile spread, computes calendar-year fold Sharpes (years with at least 60 days), cost sensitivity and breakeven cost.
6. **Ledger.** The run is logged to the trial ledger (if `ledger_path` is set) under family `pipeline_phase{1|2}`, then the deflated Sharpe ratio is computed from that family's trial count and cross-trial Sharpe variance, PBO is computed if `SystemConfig.pipeline_pbo_configs` lists a grid, and `HurdleReport` issues PASS / FAIL / skipped per gate. Required gates (`phase_sharpe`, `suspect`, `dsr`) fail `overall` when missing.

```python
from qts.core.config import SystemConfig
from qts.data.synthetic import generate, SyntheticSpec
from qts.pipeline import ResearchPipeline, format_result

md  = generate(SyntheticSpec(n_stocks=300, n_days=1512))
res = ResearchPipeline(SystemConfig(), use_ml=False, ledger_path="runs/trial_ledger.jsonl").run(md)
print(format_result(res))
```

### Why this choice

A single-period optimiser re-run daily, with a turnover-aware cost term and half-life shrinkage of alpha, is the build plan's recommendation over multi-period optimisation (MPO), because MPO adds cost roughly linear in the planning horizon and the evidence that it beats a well-tuned single-period rule on single stocks was not found. The pipeline orders alpha before risk on each date because the risk model's alignment factor needs the alpha vector.

### In the code

`ResearchPipeline(cfg, use_ml=True, phase=None, ledger_path=None, composite_weight=0.5, rebalance_every=1, venue="auction", nav0=1_000_000.0, warmup_days=None)`; `use_ml=False` is phase 1 (rank composite only). The demo script uses NAV \$5M so that fixed per-order fee minimums do not dominate costs (`docs/REVIEW_NOTES.md`). `run(md, dates=None, max_days=None)` restricts the backtest window. Fitted components are exposed for inspection: `universe_mask`, `md_u`, `risk`, `alpha`, `cost`, `optimizer`, `timings_`.

### Pitfalls & checks

- **Warm-up.** The backtest starts only after the risk model has `RiskConfig.min_history_days = 252` days of factor returns and, if ML is on, after the first out-of-sample ML prediction (`AlphaConfig.train_min_days = 756` days of training data). A five-year sample therefore yields only a couple of years of backtest when ML is on.
- **Label definition across warm-up.** Labels before the risk model's first exposure date use the internal risk-only style set; after it, the risk model's exposures. Both are risk-only, so the label means the same thing, but the exposures differ in detail. The pipeline logs this once.
- **PBO skipped by default.** A single configuration has no PBO. Set `pipeline_pbo_configs` to a list of `{"section.field": value}` overrides to run the CSCV grid; each entry costs one full backtest.

## Reproducibility

### What it is

Two runs with the same code, configuration and data must produce the same numbers, and the record of every run must say exactly which code, configuration and data produced it.

### Why we need it here

The deflated Sharpe ratio is only honest if trials are counted honestly, which requires telling a re-run (same experiment) from a new trial. Determinism also exposes bugs (unseeded draws, order dependence) that would make live behaviour unpredictable.

### How it works

- **Seeds.** `SystemConfig.seed = 0`; the synthetic generator takes `SyntheticSpec.seed` (default 0) and uses one `np.random.default_rng(seed)` stream; ML models are seed-bagged with `AlphaConfig.model_seeds = 3`. `tests/test_pipeline.py::test_pipeline_is_deterministic` re-runs the pipeline and requires identical output.
- **Config hash.** `SystemConfig.hash()` is the first 16 hex characters of the SHA-256 of the canonical JSON of the entire configuration tree (sorted keys, fixed separators). Changing any field, however small, changes the hash.
- **Data hash.** `qts/pipeline.py::data_hash_of(md)` hashes the bytes of the close panel plus its shape and date range. For a stored dataset, `ParquetStore.snapshot_hash(name)` and `snapshot_hash_all()` hash the Parquet files themselves.
- **Trial ledger.** `qts/validation/ledger.py::TrialLedger` is an append-only JSON-lines file. Each record holds a timestamp, name, family, the full config, its hash, the data hash, metrics (Sharpe, observation count, skew, kurtosis, mean IC, decile t), monthly returns and notes. `log(..., replace_identical=True)` *updates* the record with the same `(name, config_hash, data_hash)` instead of appending, so an identical re-run is not a new trial; anything else is appended (`tests/test_pipeline.py::test_pipeline_ledger_dedupes_reruns`). The ledger's effective number of trials uses Nyholt's eigenvalue formula on the correlation matrix of the stored monthly return streams, so ten near-identical variants count as fewer than ten independent trials.

### Why this choice

A JSON-lines file is greppable, diffable and shareable. Its docstring states the rule: it is the input to the deflated Sharpe and must never be reset, since deleting trials lets selection bias go unpunished.

### Pitfalls & checks

- **The data hash covers only the close panel.** Datasets with identical prices but different fundamentals or short interest share a `data_hash`, so the ledger treats the second run as a re-run. Change the run `name` when only non-price data changes.
- **No code version yet.** The pipeline does not record a git commit in the ledger (`code_version` is empty), so a code change with the same config and data also counts as a re-run.
- **Library versions matter.** Solver and LightGBM versions can change results at the fourth decimal. Pin them.

### Roadmap

`docs/WORK_PLAN.md` item 1.4 adds a `RunManifest` saved with every run (git SHA, data hash, config hash, library versions, timings), YAML config round-tripping, and determinism tests; item 1.1 adds a golden-file regression of the pipeline summary.

### References

- [Gençay (2026)](https://arxiv.org/pdf/2608.27734)

## Testing strategy

### What it is

The test suite (about 140 tests under `tests/`, one file per layer plus `test_pipeline.py`) runs entirely offline. It combines ordinary unit tests (fee arithmetic, parser fixtures, solver constraints) with **oracle tests** on a synthetic market whose true structure is known: which stocks have which factor exposures, what the factor returns were, and above all which latent signal drives returns and how strong it is.

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
| Reproducibility | Deterministic pipeline; ledger de-duplicates re-runs | `test_pipeline.py::test_pipeline_is_deterministic`, `test_pipeline_ledger_dedupes_reruns` |
| Parsers | Ken French, FINRA, Sharadar, FF48 mapping on inline fixtures (no network) | `test_data.py` |

### Pitfalls & checks

The synthetic market is only as good as what it contains, and the audit's sharpest lesson was that **green tests on a synthetic market are not evidence about bugs the synthetic market cannot express**. The default market has a static universe (with 300 names and `top_n_by_mcap = 1000`, every name is always eligible), no delistings, no IPOs, no restatements and no filing lags. The critical "ever in universe" leak and the dropped delisting labels were invisible to every test until the audit built a smaller universe and planted delistings by hand. When you add a guard for a real-data hazard, add a constructed fixture that exhibits the hazard. → see **Data & universe › The synthetic market** for the generator's equation and limitations.

### Roadmap

`docs/WORK_PLAN.md` item 1.1: a hand-computed 3-stock × 5-day simulator fixture, golden-file regression of the pipeline summary, property-based PIT tests (hypothesis) over every registered feature, universe mask, cost snapshot and risk `as_of`, removal of wall-clock assertions and seed-robust thresholds.

## Extension points

Each recipe lists the files to touch, the contract to respect and the test to add. The code is being refactored (cost contract, config), so treat these as the shape of the change and check current signatures.

### Adding an alpha feature

1. **Compute it** in `qts/alpha/features.py::FeatureBuilder.build` as a wide panel from `MarketData`, using only `shift(k >= 0)` and trailing windows. If it needs a new data panel, add the panel to `MarketData` (and to `slice_until`, `apply_mask` and `ParquetStore` dataset names).
2. **Register its position** in `FEATURE_ORDER`. The order is part of the train/live parity hash (`feature_signature`), so appending changes the hash deliberately.
3. **Decide its role.** If it belongs to an alpha theme, add it to `qts/alpha/composite.py::THEMES` with a sign, and to `qts/alpha/targets.py::ALPHA_THEME_FEATURES` so it can never be used for label residualisation. If it should be industry-demeaned, add it to `AlphaConfig.industry_demean`.
4. **Test it.** Add a `pit_shift_test` case, and if it encodes a hazard the synthetic market lacks, a constructed fixture.
5. **Gate it.** A new signal must pass the hurdles in plan §2.4: incremental rank IC over the existing composite > 0.005 in ≥ 2/3 of folds and net Sharpe improvement ≥ 0.05; t > 3 applies only to standalone signals (`HurdleReport.evaluate_signal(standalone=...)`). Every attempt goes in the ledger. → see **Validation & statistics › The hurdle gates**.

### Adding a risk factor

1. **Add the style** to `qts/risk/exposures.py::SUPPORTED_STYLES` and a branch in `raw_style_panels` computing the raw descriptor causally; standardisation (cap-weighted mean 0, unit standard deviation, trimmed at ±3 standard deviations and re-standardised, missing values set to 0) is applied by `standardize_style`.
2. **Enable it** in `RiskConfig.styles`. `factor_groups_for` places it in the `"style"` group automatically, so the optimiser can band it.
3. **Decide whether it is risk-only.** If labels should be residualised on it and the optimiser should hold it near zero, add it to `ResearchPipeline.RISK_ONLY_STYLES`. If it is also an alpha theme (value, momentum, quality, investment, short interest), leave it out: neutralising an alpha theme removes the alpha.
4. **Check it.** Run the risk evaluation (bias statistics within [0.88, 1.12] on factor and random portfolios, QLIKE) before and after (`qts/risk/evaluation.py`). → see **Risk model › Evaluating a risk model**.

### Adding a cost component

1. **Implement both views** in `qts/tcost/model.py::TransactionCostModel`: the USD point estimate in `estimate` (a new column or an addition to `spread` / `impact` / `fees`) and the convex fraction-of-NAV expression in `cvx_terms`. The component must be convex in the trade (linear, power ≥ 1 of absolute value) or the optimiser cannot use it.
2. **Keep them equal.** The simulator charges `estimate`; the optimiser plans with `cvx_terms`. Add a test that the two agree on a random trade vector, as the audit did.
3. **Feed it from the snapshot.** Any new input (for example, a per-name auction volume forecast) is a new snapshot key built PIT in `snapshot_from_market`.
4. **Watch the refactor.** After `docs/WORK_PLAN.md` item 1.2, the component will be added to the single `CostModel` implementation behind the `CostEstimate` dataclass; the two-view rule stays.

### Adding a portfolio constraint

1. **For a one-off,** pass `constraints_extra` to `SinglePeriodOptimizer.solve`: a list of callables `fn(w, x, tickers) -> list[cp.Constraint]`, where `w` is the post-trade weight variable, `x` the trade expression and `tickers` the problem's ticker index. They are applied to the alpha solve and to the volatility-target projection alike.
2. **For a permanent constraint,** add a field to `PortfolioConfig` and a named constraint in the optimiser's constraint builder, so it appears in `binding_constraints` diagnostics.
3. **Keep it convex** (linear equalities, linear or norm inequalities, second-order cones). Non-convex rules such as "at most 200 names" need a heuristic outside the solve.
4. **Watch feasibility.** A constraint that conflicts with the participation caps or forced exits makes the problem infeasible, and the default policy is to hold the book and report the failure. Check `n_failed_solves` in the summary after any change.

For example, a cap of 15% of NAV on the net weight of a hand-picked list of technology names is one callable that builds a 0/1 indicator vector over `tickers` and returns the single linear constraint "indicator · `w` ≤ 0.15"; passing it in `constraints_extra` applies it on every rebalance without touching the optimiser's code.

### References

- [Boyd et al., Multi-Period Trading via Convex Optimization](https://arxiv.org/pdf/1705.00109)
- [cvxportfolio, Optimization policies](https://www.cvxportfolio.com/en/stable/optimization_policies.html)
