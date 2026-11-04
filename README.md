# Quant Trading System (`qts`)

A research-to-production skeleton for a daily-rebalanced US equity long/short
strategy, built as a **thin vertical slice of all five models** so that
integration problems surface before any one layer is deepened:

| Layer | Package | What is implemented |
|---|---|---|
| Alpha | `qts.alpha` | PIT feature builder (18 panels), residualised multi-week targets, theme rank composite (value / momentum / quality / investment / low-risk / short-interest) with capped factor-momentum tilts, momentum crash control, walk-forward LightGBM / MLP / ridge ensemble with exact label embargo, IC / decile / half-life diagnostics, Grinold scaling |
| Risk | `qts.risk` | Barra USE4-style fundamental model: √cap-weighted cross-sectional WLS with cap-weighted industry constraint, EWMA + Newey–West factor covariance (84d vol / 504d corr half-lives), volatility-regime adjustment, Monte-Carlo eigenfactor adjustment, USE4 specific-risk pipeline (structural blend + Bayesian shrinkage), Ledoit–Wolf QIS nonlinear shrinkage, Saxena–Stubbs alpha-alignment factor, bias-statistic / QLIKE / min-variance evaluation |
| Transaction cost | `qts.tcost` | 2026 IBKR Pro / SEC §31 / FINRA TAF / CAT fee schedule, square-root impact in the cvxportfolio convex form `a·|x| + b·σ·|x|^1.5/√V`, auction-vs-continuous venue costing, borrow holding cost, TCA ledger with implementation-shortfall decomposition and monthly recalibration of `b` |
| Portfolio | `qts.portfolio` | Boyd single-period convex optimizer (cvxpy + Clarabel) in factor form, full constraint menu (book type, position/ADV caps, beta & style neutrality, industry bands, no-trade, turnover penalty), Gârleanu–Pedersen half-life shrinkage in alpha space, vol targeting within hard caps, transfer coefficient, holding-period and factor attribution diagnostics |
| Execution | `qts.execution` | Order generation, routing table (LOC/MOC in the closing auction, IBKR ClosePx / VWAP / ArrivalPx / Adaptive algos by size and signal urgency, passive limits for reversal legs), ib_async-style payload builders, VWAP / POV / Almgren–Chriss schedules, fill simulator feeding the TCA ledger |
| Backtest | `qts.backtest` | Daily-bar simulator (signals at close *t*, trades at the close, returns *t→t+1*, costs and borrow booked explicitly, delistings handled), performance metrics incl. Newey–West t, Lo-adjusted Sharpe, breakeven cost, cost sensitivity |
| Validation | `qts.validation` | Purged walk-forward / k-fold / combinatorial purged CV with label-horizon purging and embargo, deflated Sharpe ratio, PBO (CSCV), append-only trial ledger, plan hurdle gates, leakage tripwires (PIT shift test, planted-oracle test, AST `shift(-k)` registry), Nagel "collapse to momentum" audit |
| Data | `qts.data` | Seeded synthetic market with known factor structure and a planted alpha of known IC; Parquet/DuckDB PIT store; PIT universe builder with hysteresis; parsers for Ken French, FINRA short interest (with publication lag), Sharadar SF1 (keyed on filing date), Fama-French-48 from SIC; yfinance prototyping loader |

The design rationale, the evidence behind every choice, and the phased
roadmap are in `docs/BUILD_PLAN.md`. `docs/REVIEW_NOTES.md` records the
review amendments applied in code.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # add ".[data]" for yfinance / requests / polars
pytest                             # ~140 tests, ~4 min (LightGBM walk-forward and pipeline smoke are the slow part)

python scripts/run_demo.py         # phase 1: rank composite on a synthetic market
python scripts/run_demo.py --ml    # phase 2: + walk-forward LightGBM / ridge sleeve
python scripts/run_real.py --help  # same pipeline on a ParquetStore of real data
```

`run_demo.py` prints the backtest summary, alpha IC report (ICIR annualised at
the sampled frequency), the value-weighted decile long-short t-stat, DSR with its
ledger-derived trial count, calendar-year fold Sharpes, cost sensitivity and the
plan's hurdle gates. Each gate reports `PASS` / `FAIL` / `skipped`; a gate whose
input is unavailable (e.g. PBO without a configuration grid, see
`SystemConfig.pipeline_pbo_configs`) is *skipped* and only the required gates
(`phase_sharpe`, `suspect`, `dsr`) fail `overall` when missing. On the synthetic
market the composite earns a small net Sharpe and generally **fails the phase-1
gate** on merit (t-stat / phase Sharpe), not by construction: the demo validates
plumbing.

## Programmatic use

```python
from qts.core.config import SystemConfig
from qts.data.synthetic import generate, SyntheticSpec
from qts.pipeline import ResearchPipeline, format_result

md  = generate(SyntheticSpec(n_stocks=300, n_days=1512))
res = ResearchPipeline(SystemConfig(), use_ml=False, ledger_path="runs/trial_ledger.jsonl").run(md)
print(format_result(res))
```

Every layer can be used on its own against the contracts in
`qts/core/types.py` (`MarketData`, `RiskModelOutput`, `AlphaOutput`):

```python
from qts.risk import BarraStyleRiskModel
rm = BarraStyleRiskModel().fit(md).as_of(md.dates[-1])      # exposures, factor_cov, specific_var
from qts.alpha import AlphaModel
ao = AlphaModel().fit(md).as_of(md.dates[-1])                # z-scores + Grinold expected returns
from qts.tcost import TransactionCostModel
tc = TransactionCostModel(); snap = tc.snapshot_from_market(md, md.dates[-1])
from qts.portfolio import SinglePeriodOptimizer
w  = SinglePeriodOptimizer(SystemConfig().portfolio, tc).solve(ao.expected_return, rm, w_current, nav=5e6, snapshot=snap)
```

## Point-in-time discipline (enforced, not assumed)

* Every feature is a function of data with timestamp ≤ *t*; `qts.validation.leakage.pit_shift_test`
  rebuilds features on `md.slice_until(t)` and asserts equality.
* The investable universe is point-in-time: `apply_mask` stores the per-date membership on
  `MarketData.eligible`, and every cross-sectional statistic in the alpha layer (winsorisation,
  ranks, labels, IC, emitted scores) uses `md.eligibility()` at that date only — a name that enters
  the universe in 2015 is not in the 2010 cross-section, and dead names are never scored.
* Labels (forward returns) are residualised on **risk-only** styles (beta, size, residual vol,
  liquidity, short-term reversal) plus market and industries — never on the alpha themes, which
  would define the alpha away. This holds for the library default (`qts.alpha.targets.DEFAULT_STYLE_EXPOSURES`)
  and for the pipeline; the pipeline's exposures hook raises on any failure other than the risk
  model's warm-up, so the label definition cannot change silently mid-sample.
* Labels of names that delist inside the forward window use the return to the last close times
  `1 + AlphaConfig.delisting_return` (default −30 %, conservative CRSP-style) instead of NaN.
* Walk-forward training purges any label whose forward window overlaps the prediction
  date and applies an embargo ≥ horizon; the trailing validation block is purged the same way.
* The trial ledger is append-only; the deflated Sharpe ratio is computed from its trial count,
  the cross-trial Sharpe variance and Nyholt effective trials of the pipeline's family. Identical
  re-runs (same config hash + data hash) update the record instead of adding a trial.

## Status and what is next

Done: the thin slice end-to-end on synthetic data; 100 unit/integration tests.

Phase 0 (next): point `scripts/run_real.py` at a `ParquetStore` filled from
Norgate / Sharadar (see `qts/data/loaders.py` and `docs/BUILD_PLAN.md` §7),
start the daily IBKR borrow-rate logger, download the Ken French files, and
re-run the hurdle gates on 2006–2025 data. Known deepening items are listed
in `docs/REVIEW_NOTES.md`.
