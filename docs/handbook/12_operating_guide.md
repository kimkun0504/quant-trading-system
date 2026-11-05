# Operating guide

<callout icon="🧭" color="gray_bg">
This page is the hands-on manual. It shows how to install the research package, the API and the UI; what every `make` target and every `qts` subcommand does, with examples; how to read the report card a run prints, line by line; how to start the API and the QTS Workbench and what each screen shows; how to launch a run from the browser; how to load real data (Norgate/Sharadar prices and fundamentals, FINRA short interest, Ken French factors) into the `ParquetStore`; how to recalibrate the two parameters that must be re-estimated on real data (the alignment variance scale ν and the impact prefactors b); how the test tiers, the golden file and CI work; and what to do when something goes wrong.
</callout>

<callout icon="🆕" color="green_bg">
<b>What is new (Nov 2026)</b>
- This page is new. It covers the `qts` command (`demo`, `run`, `config`, `runs`, `ledger`), the run registry under `runs/<run_id>/`, the append-only trial ledger, the FastAPI service in `api/` and the React workbench in `ui/`.
- Runs now trade one day after the signal by default (`ExecutionConfig.signal_lag_days = 1`); `qts demo --compare-lag` prints the lag-0 versus lag-1 comparison.
- The defaults quoted here are those of `qts/core/config.py` and the generated `configs/default.yaml` as of 2026-11-05.
</callout>

## Install

### What you need

- **Python 3.10 or newer** (CI tests 3.10 and 3.12; `pyproject.toml` says `requires-python = ">=3.10"`).
- **Node.js and npm** for the UI only (Vite 8, React 19). The Python side never needs Node.
- **git**, because every run records the commit it ran on (→ see **Architecture & data flow › Reproducibility**).
- A few GB of RAM. The default demo (300 names × 1,512 days) takes several minutes, almost all in the daily optimiser solves.

### The research package (`qts`)

From the repository root:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,data]"
pre-commit install --install-hooks      # optional: lint on commit, fast tests on push
```

The install is editable (`-e`). The two extras:

| Extra | Contents | Needed for |
|---|---|---|
| `dev` | pytest, pytest-cov, hypothesis, ruff, pyright, pandas-stubs, scipy-stubs, pre-commit | tests, lint, type checks |
| `data` | yfinance, requests, polars | the yfinance prototype loader, downloads, Polars queries |

`make install` is a shortcut for `pip install -e ".[dev]"` plus the pre-commit hooks; it does **not** install the `data` extra.

The install also creates a console script: `qts` is the same as `python -m qts.cli`. Check it with `qts --help`.

### The API (`api/`, package `qts_api`)

```bash
pip install -e "api[dev]"       # or: make api-install
```

The API is a separate installable package that imports `qts`, so install `qts` first. `pip install -e api` without the extra installs the service alone; `[dev]` adds pytest, httpx and ruff for its tests.

### The UI (`ui/`)

```bash
cd ui && npm ci && cd ..
```

`npm ci` installs exactly the versions in `ui/package-lock.json`. Use it rather than `npm install`, which may update the lock file. Playwright's browser is needed only for the end-to-end tests (`npm run test:e2e`).

## Make targets

`make help` prints every target with its one-line description. The table groups them.

| Target | What it runs | When to use it |
|---|---|---|
| `make install` | `pip install -e ".[dev]"` + pre-commit hooks | first setup |
| `make lint` | `ruff check qts tests scripts` | before a commit |
| `make format` | `ruff check --fix` + `ruff format` (rewrites files) | before a commit |
| `make typecheck` | pyright, strict on `qts/core`, basic on `qts` | after touching types |
| `make test` | the whole suite | before merging |
| `make test-fast` | `pytest -m "not slow"` | every change |
| `make test-slow` | `pytest -m slow` (walk-forward ML, full pipelines) | before merging changes to ML, the pipeline or the simulator |
| `make coverage` | fast tier with term, XML and HTML coverage | coverage review |
| `make demo` | `python scripts/run_demo.py` (= `qts demo`) | first run |
| `make demo-ml` | `python scripts/run_demo.py --ml` | phase-2 run with the ML sleeve |
| `make precommit` | every pre-commit hook on every file | after changing hook config |
| `make clean` | removes caches, build and coverage artefacts | housekeeping |
| `make api-install` | `pip install -e "api[dev]"` | API setup |
| `make api-dev` | uvicorn on port 8000 with auto-reload | API development |
| `make api-test` | `cd api && pytest -q` (about 45 s; builds one tiny real run) | after API changes |
| `make ui-dev` | Vite dev server on port 5173, proxying `/api` to port 8000 | UI development |
| `make dev` | `api-dev` and `ui-dev` together; Ctrl-C stops both | using the workbench |

## The `qts` command

`qts` has five subcommands. Every pipeline command (`demo`, `run`) fits the full pipeline, backtests, evaluates, prints the report card, writes a run directory under `--runs-root` (default `runs`) and appends a trial to the ledger at `--ledger` (default `runs/trial_ledger.jsonl`).

### Flags shared by `demo` and `run`

| Flag | Default | Meaning |
|---|---|---|
| `--config PATH` | built-in defaults | a `SystemConfig` YAML file, loaded strictly |
| `--ml` | off | add the walk-forward ML sleeve (phase 2; slower) |
| `--signal-lag N` | config (1) | execution delay in days (`ExecutionConfig.signal_lag_days`) |
| `--compare-lag` | off | run lag 0 and lag 1 and print both |
| `--rebalance-every N` | 1 | rebalance every N days |
| `--nav USD` | 5,000,000 | starting NAV (large enough that per-order fee minimums do not dominate) |
| `--ledger PATH` | `runs/trial_ledger.jsonl` | trial ledger; `''` disables logging |
| `--runs-root DIR` | `runs` | where run directories go |
| `--no-save` | off | do not write a run directory |
| `--max-days N` | all | backtest only the last N out-of-sample days (quick checks) |
| `-v` | off | log progress (every 50 backtest days) |

The CLI clamps `universe.top_n_by_mcap` to the number of tickers in the data, so a 300-name synthetic market runs with a 300-name universe cap.

### `qts demo`: the synthetic market

```bash
qts demo                                   # phase 1, 300 names x 1512 days, seed 0
qts demo --ml                              # phase 2: composite + LightGBM/MLP/ridge sleeve
qts demo --stocks 500 --days 2520 --seed 3
qts demo --max-days 120 -v                 # quick smoke run: last 120 out-of-sample days
qts demo --signal-lag 0                    # optimistic close-as-signal bound
```

`demo` generates a market with `qts.data.synthetic.generate(SyntheticSpec(n_stocks, n_days, seed))` and sets `SystemConfig.seed` to the same seed. The market has a planted signal of known strength, so a demo run tests plumbing, not alpha (→ see **Data & universe › The synthetic market**).

### `qts demo --compare-lag`: the execution-delay check

```bash
qts demo --compare-lag
```

This runs the pipeline twice, at `signal_lag_days = 0` and `1`, and prints a comparison block after the report card:

```text
=== Execution-delay check (D4: signal lag 0 vs 1 day) ===
                       lag 0     lag 1      diff
sharpe_gross           1.328     0.490    -0.839
sharpe_net             0.956     0.111    -0.846
cagr_net               0.064     0.005    -0.059
avg_daily_turnover     0.080     0.082    +0.001
```

That block is from `runs/demo_post_refactor.txt`, and it is a cautionary example rather than a result. The lag investigation (`docs/notes/lag_investigation_2026-11-05.md`) showed that this −0.84 Sharpe gap came from an uncommitted code state and did not reproduce: on the committed code the same pair gave a difference of −0.08 ± 0.15 (paired block-bootstrap standard error), and across five seeds lag 1 was on average 0.16 Sharpe *better*. Single-path Sharpe differences of 0.5 or more arise from small code changes alone on this market, because cost-aware books are path dependent. Judge a lag effect by paired, multi-seed comparisons, never by one pair of runs.

Both runs are full trials (two run directories, two ledger records), each refits the pipeline (labels depend on the lag), and the report card printed above the block is the **lag-0** run unless you also pass `--signal-lag 1`.

### `qts run`: real data

```bash
qts run --store data/store --start 2006-01-01                # phase 1 on a ParquetStore
qts run --store data/store --start 2006-01-01 --ml --config configs/real.yaml
qts run --yfinance AAPL MSFT NVDA JPM XOM --start 2015-01-01  # plumbing only
```

`--store` assembles `MarketData` with `ParquetStore(path).as_market_data(start, end)`; `--yfinance` downloads with `qts.data.loaders.yfinance_to_market_data` (default start 2015-01-01). Both then call `qts.data.universe.survivorship_check`, which warns when no ticker's prices end before the last date. yfinance data always triggers it: it has no delisted companies, so **never report a Sharpe from it**. `scripts/run_real.py` is a thin wrapper with the same flags.

### `qts config dump` and `qts config validate`

```bash
qts config dump                         # the defaults as YAML (with the GENERATED header)
qts config dump > configs/default.yaml  # regenerate the committed defaults file
qts config dump --config my.yaml        # a file, normalised: every field, defaults filled in
qts config validate my.yaml             # strict load; prints "OK my.yaml  hash 42b11162..."
```

A config file only needs the fields you change; missing fields take the defaults. Loading is strict: an unknown key fails with the list of known keys, and `CostConfig` and `ExecutionConfig` check bounds. `validate` exits with status 1 and prints `INVALID path: reason` on any error. For example:

```yaml
# configs/real.yaml
universe:
  top_n_by_mcap: 1000
  min_adv_usd: 5000000.0
portfolio:
  target_vol_annual: 0.08
risk:
  alignment_nu_scale: 0.05
```

The printed hash is `SystemConfig.hash()`; two files that load to the same values share it, whatever their formatting.

### `qts runs list` and `qts runs show`

```bash
qts runs list                        # newest first
qts runs list --root /data/runs --limit 20
qts runs show 20261105T072821098301Z_2d9f80cd
qts runs show runs/20261105T072821098301Z_2d9f80cd   # a path works too
```

`runs list` prints one line per run directory:

```text
run_id                                   sharpe_net  overall  ml lag  config / data / git
20261105T073352232658Z_cc651202                0.11     FAIL   n   1  cc6512021600d08f / acc387f0eca55816 / 63ce32bf1f
20261105T072821098301Z_2d9f80cd                0.96     FAIL   n   0  2d9f80cdc0532202 / acc387f0eca55816 / 63ce32bf1f
```

The run id is `<UTC timestamp>_<first 8 characters of the config hash>`, so ids sort by creation time and two runs of the same config share a suffix. `overall` is the hurdle verdict, `ml` whether the ML sleeve was on, `lag` the execution delay. `runs show` prints the manifest (created, config/data hashes, git sha with `(dirty)` if the tree had uncommitted changes, library versions), the headline backtest numbers, the IC summary and the full hurdle table.

### `qts ledger show`

```bash
qts ledger show
qts ledger show --family "pipeline_phase1:acc387f0eca5"
qts ledger show --path /shared/trial_ledger.jsonl
```

It prints one row per trial after de-duplication: timestamp, family, annualised Sharpe, observation count, mean IC, config hash, data hash, code version and `reruns` (how many identical re-runs collapsed into this row). Remember the rules from **Architecture & data flow › The trial ledger**: identical re-runs collapse only when the name, config hash, data hash, a *clean* code version and the run spec all match; a run from a dirty tree is always a new trial. Families are `pipeline_phase{1|2}:{first 12 characters of the data hash}`. Records written before November 2026 carry the bare family `pipeline_phase1` and empty hashes.

<callout icon="⚠️" color="yellow_bg">
Never delete or edit `runs/trial_ledger.jsonl`. It is the input to the deflated Sharpe ratio: every variant you tried raises the bar for the best one. Deleting failed trials makes every later result look more significant than it is. Back it up, and share one ledger across everyone who runs experiments on the same data.
</callout>

## What a run leaves behind

```text
runs/
  trial_ledger.jsonl            append-only, one JSON line per trial (+ .lock file)
  <run_id>/
    manifest.json               run id, created, git sha/dirty, config & data hashes, versions, timings
    config.yaml                 the exact SystemConfig (reload: SystemConfig.from_yaml)
    summary.json                every number in the report card
    hurdles.json                gate verdicts
    nav.parquet  returns.parquet  weights.parquet  trades.parquet  costs.parquet  ic.parquet
    extras.json  exante_vol.parquet  exposures.parquet  holdings.parquet
  _staging/  _jobs/             API job workspace and job state (API only)
```

`qts.pipeline.load_run(path)` re-loads a run as a typed `RunRecord` (manifest, config, summary, hurdles and every frame). To reproduce a run, check out `manifest.git_sha` (only meaningful when `git_dirty` is false), install the library versions in `manifest.versions`, and run with `--config runs/<id>/config.yaml` on data with the same `data_hash`.

## Reading the report card

This walkthrough uses `runs/demo_post_refactor.txt`, the most recent stored demo report when this page was written (`runs/demo_audit2.txt` did not exist yet). It was produced by `qts demo --compare-lag` on the default synthetic market (300 names × 1,512 days, seed 0), so the report card shown is the **lag-0** run. Its numbers come from an uncommitted working tree at commit 63ce32b and do not reproduce on the current code; read it for the *format*, not as evidence. All figures are synthetic and from the out-of-sample backtest window.

### The warnings above the card

```text
WARNING qts.backtest.simulator: backtest: 220 name-days with alpha/position outside the
risk-model universe (priced at 90th-pct specific variance, zero factor exposure)
```

A handful of names had an alpha or a position but no row in the risk model on some days (typically held names being exited after they left the universe). The simulator prices their risk conservatively: zero factor exposure and the 90th percentile of specific variance (`qts/core/risk_fill.py`). A few hundred name-days out of about 300,000 is harmless.

### Backtest summary

```text
period            2019-12-10 -> 2023-10-17  (1005 days)
```

The out-of-sample window: it starts after the risk model's 252-day warm-up (and, with `--ml`, after the first walk-forward model), plus the signal lag. A signal-lag suffix (`signal lag 1d`) appears when the lag is not zero.

```text
Sharpe gross/net  1.33 / 0.96   (NW t = 2.06)
```

Annualised Sharpe ratio before and after all costs (spread, impact, fees, borrow), with `rf = 0`, so read the returns as excess over cash. NW t is the Newey–West t-statistic of the mean daily net return, which corrects for autocorrelation; roughly Sharpe × √years. Above 2 the mean is unlikely to be zero; the plan asks for more than that, through the gates below. A net Sharpe above 2 is flagged as suspect (→ see **Start here › Why "too good" backtests are suspect**).

```text
CAGR gross/net    9.08% / 6.40%   vol 6.7%
```

Compound annual growth rate before and after costs, and annualised volatility of net returns. The vol target is 10% (`target_vol_annual`); realised vol below it means the caps (gross ≤ 2, 2% per name, 5% of ADV) or the risk model's over-forecast kept the book smaller.

```text
max drawdown      -5.3%   calmar 1.21
```

The worst peak-to-trough fall of net NAV, and Calmar = net CAGR ÷ |max drawdown|.

```text
turnover/day      8.03%   effective holding 50 d
```

Average two-sided daily turnover, $\sum_i |x_i|$ as a fraction of NAV, and the implied holding period, roughly 2 × average gross ÷ daily turnover. The target is about 50 days (`effective_holding_days_target`); much shorter means costs are eating the alpha.

```text
cost per side     10.5 bp   gross exp 2.00  net exp +0.04
```

Average cost per dollar traded, one side, in basis points; average gross exposure (cap 2.0) and net exposure (target 0 ± 0.05).

```text
transfer coeff    0.53   realized/ex-ante vol 1.04
```

The transfer coefficient is the mean correlation between the alpha and the positions; 1 would mean the book is the forecast, and constraints and costs pull it down. Realised ÷ ex-ante volatility checks the risk model: near 1 is good, well above 1 means it under-forecasts.

### Alpha IC

```text
mean rank IC 0.0392  ICIR 1.45 (annualised at 12.6/yr)  t 2.89  NW t 3.16 (L=3)  n 50
```

The score that was traded is compared with the risk-residualised 20-day forward return every 20 days (non-overlapping), 50 times here. Mean rank IC is the average Spearman correlation; 0.03–0.05 is the realistic target, and the synthetic market's planted band is about 0.036 for `ic_true = 0.04`. ICIR = mean ÷ standard deviation of the IC, annualised with 252 ÷ 20 = 12.6 samples a year. `t` and `NW t` are plain and Newey–West t-statistics of the mean IC, with L lags.

```text
IC decay (horizon d: mean IC)  5d: 0.0180  20d: 0.0392  60d: 0.0590
```

Mean IC against 5-, 20- and 60-day labels (`AlphaConfig.secondary_horizons`). IC rising with horizon means a slow signal; a fast-decaying signal would show the 5-day IC highest.

```text
decile L/S (VW)  mean 1.25%/period  Sharpe 0.92  NW t 1.67 (L=3)
```

A value-weighted top-minus-bottom decile portfolio of raw forward returns, formed every 20 days: the mean return per 20-day period, its annualised Sharpe and its Newey–West t. This t-statistic feeds the `t_stat` gate (threshold 3).

```text
DSR 0.704  (n_trials 5.0, var SR 0.332)  PBO skipped
```

The deflated Sharpe ratio: the probability that the true Sharpe is above what the best of `n_trials` trials would show by luck, given the cross-trial variance of Sharpe ratios (`var SR`, annual units) and the return distribution's skew and kurtosis. `n_trials` is the effective number of trials in the ledger family, by Nyholt's formula (5 here). The gate needs DSR > 0.95. PBO (probability of backtest overfitting) needs a configuration grid (`pipeline_pbo_configs`), so it is skipped by default.

```text
config 2d9f80cdc0532202  data acc387f0eca55816  git 63ce32bf1f8c  run 20261105T072821098301Z_2d9f80cd
```

The config hash, data hash, commit and run id: everything needed to find the run directory and ledger record.

### Folds, cost sensitivity and hurdles

```text
=== Fold (purged walk-forward test block) net Sharpe ===
     start        end  n_days  sharpe
2019-12-11 2020-07-30     167    0.76
...
```

The backtest window cut into `ValidationConfig.n_splits = 6` contiguous test blocks of `PurgedWalkForward` (blocks shorter than 20 days are merged away), with the net Sharpe of each. Consistency across blocks matters more than the average: the `net_sharpe_delta` gate needs at least +0.05 in two thirds of them.

```text
=== Cost sensitivity (net Sharpe at cost multiples) ===
            cost_bps_per_side  net_sharpe  net_cagr  mean_daily_cost_bps
multiplier
0.5                     5.233       1.170     0.079                 0.42
1.0                    10.467       1.013     0.068                 0.84
2.0                    20.933       0.697     0.046                 1.68
3.0                    31.400       0.382     0.024                 2.52
```

The same book re-costed at half, one, two and three times the average modelled cost per side, charged on each day's turnover (positions are held fixed, so this is an approximation). If the strategy dies at 2×, it is fragile: real costs are uncertain by that much. The 1.0 row is an approximation of the simulated net Sharpe, not a copy of it: it charges one average cost rate on every day. In this older report it is also visibly higher (1.01 against 0.96) because the code then started from gross returns before borrow; since the second audit, cost sensitivity and breakeven start from `returns["gross_after_borrow"]`, because borrow is a holding cost, not a per-side trading cost.

```text
=== Plan hurdles ===
OVERALL: FAIL (phase 1)
[FAIL] t_stat: decile L/S t = 1.67 <= 3.0
[PASS] ic_sign: IC > 0 in 4/5 folds (need >= 0.67)
[skipped] incremental_ic: not applied (standalone signal)
[PASS] net_sharpe_delta: net Sharpe by fold (vs zero baseline) >= +0.05 in 6/6 folds (need >= 0.67)
[PASS] breakeven_cost: breakeven 44.1 bps >= 2x modeled 20.9 bps
[FAIL] dsr: DSR = 0.704 <= 0.95
[skipped] pbo: PBO not supplied (no configuration grid run; set pipeline_pbo_configs)
[PASS] phase_sharpe: phase-1 net Sharpe 0.96 >= 0.3
[PASS] suspect: net Sharpe within plausible range
skipped (not required): incremental_ic, pbo
```

The gates were fixed before any result was seen (`docs/BUILD_PLAN.md` §2.4). Each is PASS, FAIL or skipped (input missing). `overall` fails if any non-skipped gate fails, or if a **required** gate (`phase_sharpe`, `suspect`, `dsr`) is missing. In order:

| Gate | Passes when | Here |
|---|---|---|
| `t_stat` | decile long/short Newey–West t > 3 | 1.67: four years is too short for this Sharpe |
| `ic_sign` | mean IC > 0 in ≥ 2/3 of the fold blocks | 4 of 5 blocks with IC samples |
| `incremental_ic` | a new signal adds > 0.005 rank IC over the existing composite | skipped for a standalone run |
| `net_sharpe_delta` | fold net Sharpe ≥ +0.05 (versus a zero baseline) in ≥ 2/3 of folds | 6 of 6 |
| `breakeven_cost` | the cost per side at which net return reaches zero is at least twice the modelled cost per side | 44.1 bp against 2 × 10.5 bp (the "20.9 bps" in the message is already the doubled figure) |
| `dsr` | deflated Sharpe > 0.95 | 0.704 with 5 trials |
| `pbo` | PBO < 0.5 | skipped |
| `phase_sharpe` | net Sharpe ≥ 0.3 (phase 1) or ≥ 0.5 (phase 2) | 0.96 |
| `suspect` | net Sharpe ≤ 2 | yes |

So this run "passes the cost and Sharpe gates but is not yet distinguishable from luck": the typical verdict for an honest four-year backtest. → see **Validation & statistics › The hurdle gates**.

```text
NOTES: PBO skipped: set SystemConfig.pipeline_pbo_configs to run a CSCV grid
timings (s): data_hash=0.3, universe=0.3, risk_fit=8.6, alpha_fit=8.3, backtest=375.0, evaluate=1.6, total_run=394.1
run directory: runs/20261105T072821098301Z_2d9f80cd
```

Pipeline notes (any warning the evaluation raised, including the suspect-Sharpe note), seconds per stage, and where the run was written. The backtest dominates: one optimiser solve per day, several when the γ search runs.

## The QTS Workbench: API and UI

### Starting it

For everyday use, from the repository root:

```bash
make dev            # API on http://127.0.0.1:8000 (auto-reload), UI on http://localhost:5173
```

Open http://localhost:5173. The Vite dev server proxies `/api` to port 8000. The interactive API documentation is at http://localhost:8000/api/docs. Ctrl-C stops both servers.

**Single-origin mode** serves the built UI from the API process, with no Node server at run time:

```bash
npm --prefix ui run build
QTS_SERVE_UI=1 uvicorn qts_api.main:app      # UI at http://127.0.0.1:8000/, API under /api
```

The API reads its settings from the environment. The defaults match the CLI, so CLI runs and UI runs share one registry and one ledger.

| Variable | Meaning | Default |
|---|---|---|
| `QTS_RUNS_ROOT` | run directories | `./runs` |
| `QTS_LEDGER_PATH` | trial ledger | `<runs root>/trial_ledger.jsonl` |
| `QTS_STORE_PATH` | `ParquetStore` root for "Data store" launches | unset: store launches are rejected (422) |
| `QTS_MAX_WORKERS` | concurrent pipeline jobs | 1 |
| `QTS_MAX_QUEUED` | jobs waiting for a worker; more get a 429 | 4 |
| `QTS_SERVE_UI`, `QTS_UI_DIST` | single-origin mode and the built UI's folder | off, `ui/dist` |
| `QTS_CORS_ORIGINS` | allowed browser origins | `http://localhost:5173` |
| `QTS_ALLOWED_HOSTS` | extra accepted `Host` headers | none (localhost, 127.0.0.1, ::1 always) |
| `QTS_JOB_START_METHOD` | multiprocessing start method | `spawn` |

<callout icon="⚠️" color="yellow_bg">
The API has no authentication. It binds to 127.0.0.1 by default and rejects requests whose `Host` header is not on the allow-list (a defence against DNS rebinding, which CORS does not stop). Keep it on localhost; do not expose port 8000 on a network.
</callout>

To put something in the registry without waiting six minutes, `python -m qts_api.fixture runs` builds a tiny real run (about 30 seconds).

### A tour of the screens

The workbench is designed as an annotated lab notebook (`ui/DESIGN.md`): a calm reading column, and beside every figure a margin note that says what the number means *for this run* and links to the handbook section that defines it. A navigation rail on the left holds Runs, New run, Models, Ledger and Handbook, plus a Light / Dark / System theme switch. Screenshots of every screen in both themes at 1440 px and 390 px are in `ui/docs/screenshots/` (taken on the mock data, so their numbers are illustrative).

**Runs** (`/runs`). Every run directory, newest first, as a sortable table: run label and description (phase, sleeve, data kind, time), gate verdict (Pass / Fail, or Running / Run failed for jobs), net and gross Sharpe, net CAGR, maximum drawdown, cost per side and deflated Sharpe. A net Sharpe above 2 is tagged "suspect". The margin suggests sorting by deflated Sharpe rather than net Sharpe, because it discounts each result by the number of trials it took.

**Run detail** (`/runs/<id>`). The header gives the run's facts (phase, sleeve, data, stocks, seed, signal lag) and a manifest box (run id, created, period, config hash, data hash, git sha). Below it is a one-sentence **verdict** generated from the gates, for example "Passes the IC-sign, fold-Sharpe, cost and Sharpe gates but fails the decile t-stat gate and deflated Sharpe with 4 trials — not yet distinguishable from luck", with a reminder when the data is synthetic. A **gate strip** shows each gate as passed, failed or skipped, marking the required ones. Then six tabs:

| Tab | What it shows |
|---|---|
| Performance | Net NAV with drawdown underneath; a key-metrics table (net and gross Sharpe, Newey–West t, CAGR, volatility, drawdown, Calmar, Sortino, hit rate, turnover, holding period, cost per side, breakeven, transfer coefficient, realised/ex-ante vol), each with an interpretation; a monthly-returns heatmap; the 126-day rolling net Sharpe against the phase bar |
| Risk | Net book exposure to the market and each style factor over time; realised 63-day volatility against the risk model's ex-ante forecast; month-end positions with weight, alpha, specific volatility and industry |
| Alpha | Rank IC per formation date; the IC decay curve across horizons; signal statistics (mean IC, ICIR, t-statistics, decile spread) |
| Costs | Where the costs go (spread, impact, fees, borrow per day); cost metrics; the cost-sensitivity table; breakeven cost |
| Execution | The routing used to price trades (venue description, e.g. auction up to the participation cap, remainder continuous) |
| Validation | The hurdle table with values, thresholds and the pipeline's reasons; signal statistics; mean IC by calendar year; net Sharpe by walk-forward fold; a worked explanation of the deflated Sharpe with this run's trial count, Sharpe variance, skew, kurtosis and observation count; pipeline notes |

**Models** (`/models`). One short section per model, saying where its diagnostics appear on a run page, with a handbook link.

**Ledger** (`/ledger`). The families (trial count, effective number of trials and cross-trial Sharpe variance for each) and every de-duplicated trial with config hash, data hash, net Sharpe, mean IC, days, re-runs and creation time.

### Launching a run from the UI

**New run** (`/runs/new`) is a form over `POST /api/runs`:

1. **Market data.** *Synthetic market* (seeded, with a planted signal; validates plumbing) or *Data store* (the `ParquetStore` at `QTS_STORE_PATH`).
2. **Phase.** *Phase 1: composite* (net Sharpe bar 0.3) or *Phase 2: composite + ML* (bar 0.5); the "Train the ML sleeve" checkbox turns on the walk-forward LightGBM / MLP / ridge sleeve.
3. **Stocks, days, seed.** 50–3,000 names; 764–6,000 days (the API's 512-day warm-up allowance plus at least a year); the seed fixes the market and the models.
4. **Signal lag.** *Trade at the next close (1 day)*, the realistic default, or *Trade at the same close (0 days)*, the optimistic bound, which the run page flags.
5. **Config overrides** (collapsed). Generated from `GET /api/config/schema`: every scalar field of every `SystemConfig` section, grouped by section, with enums as selects, booleans as checkboxes and bounds checked in the browser and again, strictly, by the API. `seed` and `execution.signal_lag_days` come from the form's own fields.
6. **Launch run.** The API validates the request (unknown fields, wrong types such as `"stocks": "60"`, out-of-range values and unknown override paths all return a 422 with one error per field), fixes the run id, queues the job and answers 202. The page then polls the job every 750 ms and shows stage-based progress: market data, universe, risk model, alpha model, the backtest ("day k of n"), evaluation, writing. When the job finishes, the new run opens.

Each job runs in its own process (a crash cannot take the API down), follows the same code path as `qts demo` / `qts run` with NAV \$5M, writes into `runs/_staging/` and renames the directory into place, so readers never see a half-written run. Job state is kept in `runs/_jobs/<job_id>.json`; a job that was running when the API stopped is reported as failed ("Interrupted"). A queued or running job can be cancelled (`DELETE /api/jobs/<id>`). Every launch is a trial in the ledger, which raises the deflated-Sharpe bar for every run in its family; re-running an identical config on identical data from a clean checkout does not.

## Loading real data into the ParquetStore

`qts` has parsers and transforms for each source but no vendor downloaders for Norgate or Sharadar: you export from the vendor, then write the panels into a `ParquetStore` with the dataset names below. The store keeps one value per `(date, ticker)`; point-in-time correctness comes from writing panels that are already keyed on availability dates.

### The dataset names

| Dataset | Kind | Content | Typical source |
|---|---|---|---|
| `prices_close` (required) | wide | split- and dividend-adjusted close | Norgate (total-return adjusted) or Sharadar SEP `closeadj` |
| `prices_close_raw` | wide | as-traded close | Norgate unadjusted, Sharadar SEP `closeunadj` |
| `prices_volume` (required) | wide | actual shares traded that day (not split-adjusted) | Norgate unadjusted volume; check the vendor's adjustment convention |
| `shares_outstanding` | wide | actual share count | Sharadar SF1 `sharesbas` (PIT via `pit_forward_fill`) or DAILY |
| `market_cap` (required) | wide | USD | `close_raw × shares_outstanding`, or Sharadar DAILY `marketcap` |
| `industry` (required) | long `ticker, industry` | static fallback label | SIC → FF48 via `ff48_from_sic_series` |
| `industry_history` | wide (strings) | dated labels | SIC history → FF48 |
| `delisting_dates` | long `ticker, delisting_date` | last trading date (NaT = alive) | Norgate delisted date, Sharadar TICKERS / ACTIONS |
| `spread_bps` | wide | half-spread proxy, bp | quotes, or a high-low estimator |
| `short_interest_ratio` | wide | short interest ÷ shares | FINRA via `to_short_interest_ratio` |
| `borrow_fee` | wide | annual fraction | own IBKR logger (none historically) |
| `book_to_price`, `earnings_yield`, `gross_profitability`, `asset_growth` | wide | PIT fundamentals | Sharadar SF1 via `sharadar_sf1_to_pit_fundamentals` |

Wide datasets are date × ticker DataFrames with a `DatetimeIndex`; `write_wide` stores them as `(date, ticker, value)` rows.

### Prices and corporate events (Norgate or Sharadar SEP)

Norgate is the build plan's price source: delisting-complete history from 1990 with index constituents. Use the total-return adjusted series for `prices_close` and the unadjusted series for `prices_close_raw`, keep volume unadjusted (actual shares), and take the delisting date of every dead security. Key everything on a permanent identifier mapped to a ticker string that is unique across the whole history (a reused ticker must not merge two companies).

```python
from qts.data.store import ParquetStore
store = ParquetStore("data/store")
store.write_wide("prices_close", close_adj)            # date x ticker, adjusted
store.write_wide("prices_close_raw", close_raw)        # as traded
store.write_wide("prices_volume", volume_shares)
store.write_wide("shares_outstanding", shares)
store.write_wide("market_cap", close_raw * shares)
store.write_long("delisting_dates", dd, mode="replace")  # columns: ticker, delisting_date
```

Nightly appends need nothing special: the default `mode="upsert"` replaces only the rows whose `(date, ticker)` appear in the new data and keeps the rest of the year. A NaN in a wide upsert means "no information" and leaves the stored value alone; pass `nan_policy="delete"` only when a vendor correction really removes a value. Use `mode="replace"` to rebuild a derived dataset from raw drops.

### Industries

```python
from qts.data.loaders import ff48_from_sic_series
ind = ff48_from_sic_series(sic_by_ticker)               # Series ticker -> FF48 short name
store.write_long("industry", ind.rename("industry").rename_axis("ticker").reset_index(), mode="replace")
store.write_wide("industry_history", dated_labels.astype(object))   # date x ticker, from the panel start
```

Supply `industry_history` from the first panel date: before a ticker's first dated label the store reports `"UNKNOWN"`, never the static map, because the static map is today's classification.

### Fundamentals (Sharadar SF1)

```python
from qts.data.loaders import sharadar_sf1_to_pit_fundamentals
panels = sharadar_sf1_to_pit_fundamentals(
    sf1,                       # raw SF1 rows: ticker, dimension, calendardate, datekey, fields...
    dimension="ART", lag_days=1, dates=close_adj.index,
    shares_outstanding=shares, market_cap=close_raw * shares,
)
for name in ("book_to_price", "earnings_yield", "gross_profitability", "asset_growth"):
    store.write_wide(name, panels[name])
```

The defaults are the safe ones: `ART` (as-reported trailing twelve months, so ratios are annual and free of seasonality; `MR*` dimensions restate history and trigger a warning), `lag_days = 1` (a filing is usable from the close after its filing date), share-basis totals for B/P and E/P (split-consistent), and older-period restatements kept out of the level panels. → see **Data & universe › Fundamentals: Sharadar SF1 and the datekey**.

### Short interest (FINRA)

```python
from qts.data.loaders import parse_finra_short_interest, to_short_interest_ratio
si_long = pd.concat(parse_finra_short_interest(f) for f in finra_files)
store.write_wide("short_interest_ratio", to_short_interest_ratio(si_long, shares, lag_business_days=8,
                                                                 dates=close_adj.index))
```

Each settlement-date report becomes visible 8 NYSE sessions later (holidays skipped by `qts/data/calendar.py`), and the ratio uses the share count as of the settlement date.

### Factor benchmarks (Ken French)

`download_ken_french("F-F_Research_Data_5_Factors_2x3_daily")` and `download_ken_french("F-F_Momentum_Factor_daily")`, joined by `ken_french_factor_panel(ff5, mom)`, give daily factor returns in decimals (or fetch the zips by hand and use `parse_ken_french_zip(path)`). They are not pipeline inputs: use them to check your own factor returns and to regress strategy returns on known premia. Store a snapshot; the files are revised.

### Run on the store

```bash
qts run --store data/store --start 2006-01-01
```

Before trusting the result, check: no survivorship warning; `md.delisting_dates` present; `close_raw` present (otherwise the price screens fall back to adjusted prices); the universe size over time (`build_universe_mask(md).sum(axis=1)`); and that the data hash in the manifest changes whenever you re-ingest.

## Recalibrating ν and b

Two parameters were set on synthetic data and **must be re-estimated on real data** before results are trusted.

### The alignment variance scale ν

The risk model adds an alignment factor along the part of the alpha the factors cannot explain, with variance $\nu = \text{alignment\_nu\_scale} \times N \times \text{median style variance}$ (→ see **Risk model › The alpha alignment problem (Saxena–Stubbs)**). The scale is chosen so that the optimised book's **bias statistic** (realised ÷ forecast volatility of standardised returns) is close to 1. On the synthetic market with the production settings (γ search to the vol target, lag 1, only risk-model names held) the calibration landed on the search floor, 0.05, for all three seeds: the risk model already slightly over-forecasts the optimised book (bias 0.86–0.90 at ν = 0), so the alignment term can only make it more conservative (`docs/notes/alignment_nu_calibration.md`). The default is therefore `RiskConfig.alignment_nu_scale = 0.05`. On real data, unexplained alpha may carry real unpriced risk and the answer may differ.

```bash
python scripts/calibrate_alignment_nu.py                         # synthetic, seeds 0 1 2
python scripts/calibrate_alignment_nu.py --seeds 0 --n-dates 120 --burn-in 20 --n-iter 5 \
       --grid 0 0.25 0.5 1 2 --json runs/nu_cal.json
```

The script fits the production pipeline, takes the last `--n-dates` out-of-sample dates, and runs `qts.risk.calibrate_alignment_nu` with the real optimiser and the production execution lag, carrying the previous day's book so turnover terms act as in the simulator. For real data, replace the `generate(...)` call with your loaded `MarketData` (the script has no `--store` flag), use the longest window you can afford (the bias band is $1 \pm 2/\sqrt{2T}$, so 120 dates give only ±0.13), and set the median calibrated value in your config:

```yaml
risk:
  alignment_nu_scale: 0.05      # replace with the real-data calibration
```

### The impact prefactors b

The square-root impact model charges $b \, \sigma \, Q^{e} / V^{e-1}$ (average cost per dollar $b\,\sigma\,(Q/V)^{e-1}$, $e = 1.5$), with separate prefactors per venue: `CostConfig.impact_b_continuous = 1.0` (V = ADV) and `impact_b_auction = 0.3` (V = auction volume, about 9% of ADV, which makes the effective coefficient per $\sqrt{Q/\text{ADV}}$ equal to $0.3/\sqrt{0.09} = 1.0$). Both are priors from the literature (→ see **Transaction cost model**). They can only be measured from your own fills.

`qts/tcost/tca.py::TCALedger` is the record: `record_order(...)` at decision time (decision price, arrival price, ADV, σ, venue), `record_fill(order_id, qty, px, ts, venue)` per fill, `close_order(...)` with the day's close and the next close. After a month or more of paper or live trading:

```python
from qts.tcost.tca import TCALedger
led = TCALedger.load("tca/2027-01")
fit = led.calibrate_b_by_venue(prior_weight=20.0)
b_auction, diag_a = fit["auction"]; b_cont, diag_c = fit["continuous"]
print(diag_a["n"], diag_a["b_ols"], diag_a["se"], b_auction)
```

Each venue's fit is a through-the-origin regression of arrival slippage minus half-spread on $\sigma (F/V)^{e-1}$, with V the *venue's* volume, shrunk toward the current prior with a weight of 20 orders: $b = (n\, b_{\text{OLS}} + 20\, b_{\text{prior}})/(n + 20)$. With few orders the answer stays near the prior. Write the results into your config (`cost.impact_b_auction`, `cost.impact_b_continuous`) and re-run; the change moves every backtest, so it is a new trial in a new config hash.

## Tests and CI

### The tiers

| Tier | Command | What is in it | Budget |
|---|---|---|---|
| fast | `make test-fast` (`pytest -m "not slow"`) | unit tests, constructed fixtures, hypothesis PIT properties, the golden file, short pipeline runs | a few minutes |
| slow | `make test-slow` (`pytest -m slow`) | walk-forward ML, full pipeline runs, statistical seed sweeps, optimiser performance | minutes to an hour |
| API | `make api-test` | every endpoint against a tiny real run, job lifecycle and cancellation with real worker processes | about 45 s |
| UI | `npm --prefix ui test`; `npm --prefix ui run test:e2e`; `npm --prefix ui run test:e2e:real` | Vitest unit/component tests; Playwright smoke + axe accessibility on mocks; one smoke against the real API | about a minute each |

Markers are strict (`--strict-markers`): mark long tests `@pytest.mark.slow` and cross-layer ones `@pytest.mark.integration`. Before merging a change to ML training, the pipeline or the simulator, run the slow tier locally. For UI–API contract changes, run `npm --prefix ui run api:types` (regenerates the OpenAPI document and the TypeScript types) and then `npm --prefix ui run typecheck`, which fails if the hand-written schema and the API disagree.

### The golden file

`tests/test_golden.py` runs `ResearchPipeline(use_ml=False).run(md, max_days=60)` on a fixed synthetic market (100 names × 700 days, seed 11, `ic_true = 0.08`, top-80 universe) and compares the summary, IC report, decile, DSR, hurdle statuses and backtest aggregates with `tests/golden/pipeline_summary.json` at a relative tolerance of 1e-6. Any change to the numbers fails it, by design.

When a change is **meant** to move the numbers:

1. Run the golden test and read the full diff, not just the first failure (a changed data hash can hide every other field).
2. Attribute every moved field to an intended change. The method in `docs/notes/golden_changes_2026-11-05.md` is the model: revert the intended changes one at a time through config overrides and record which step moved which field.
3. Regenerate: `pytest tests/test_golden.py --update-golden`.
4. Inspect `git diff tests/golden/` and explain the moves in the commit body.

Never regenerate the golden file to make an unexplained diff go away. And never read meaning into its Sharpe ratios: a 60-day Sharpe has a standard error of about 2.

### CI

`.github/workflows/ci.yml` runs on pushes to `main` and on pull requests, on Python 3.10 and 3.12: `pip install -e ".[dev]"`, ruff lint and format check (blocking), pyright strict on `qts/core` (blocking) and basic on `qts` (non-blocking), and the fast tier with coverage (uploaded as an artifact). A nightly job at 06:17 UTC (and on manual dispatch) runs the slow tier on Python 3.12 with a 90-minute limit. The pre-commit hooks run ruff, whitespace and YAML/TOML checks, private-key detection and a 500 kB file guard on every commit, and the fast tier on `git push`.

<callout icon="⚠️" color="yellow_bg">
CI covers the `qts` package only. The API tests (`make api-test`) and the UI checks (`npm run typecheck`, `lint`, `test`, `test:e2e`) are not in the workflow, so run them locally after touching `api/` or `ui/`.
</callout>

## Troubleshooting

**`qts: command not found`.** The virtual environment is not active or the package is not installed: `source .venv/bin/activate && pip install -e ".[dev,data]"`, or use `python -m qts.cli`.

**`KeyError: SystemConfig.portfolio: unknown config key(s) [...]`.** A typo or an outdated field in your YAML; the message lists the known keys. Fields are renamed rarely (`CONFIG_SCHEMA_VERSION` is bumped when they are). `AlphaConfig.delisting_return` no longer exists: the delisting assumption is `cost.delisting_haircut`.

**`ValueError: CostConfig: ...` or `ExecutionConfig: ...`.** A value outside its bounds, such as a negative fee, an impact exponent outside $(1, 2]$, or a non-integer signal lag. The message names every offending field.

**`tests/test_config.py` fails.** `configs/default.yaml` no longer matches the code defaults. Regenerate it with `qts config dump > configs/default.yaml` and commit it with the change that moved the default.

**The golden test fails after an unrelated-looking change.** Something moved the numbers. Follow the golden-file procedure above; do not regenerate first.

**The report card after `--compare-lag` shows lag 0.** By design of the current CLI, the card printed above the comparison block is the lag-0 run unless you pass `--signal-lag 1` as well. Both runs are in `runs/`; `qts runs list` shows their lags.

**My lag-1 Sharpe is much lower than lag 0.** Before concluding that the signal is fast, rerun on several seeds and compute a paired comparison. On the synthetic market, single-path differences of 0.5–0.8 Sharpe come from path noise alone. If the drop is real on real data, check `strev_21` and other short-horizon features: they are the ones that lose IC within a day.

**DSR is low although the Sharpe looks fine.** Look at `n_trials` and `var SR` in the report and at the family in `qts ledger show`. Many variants on the same data set raise the bar; that is the point. A run from a dirty tree is never collapsed with a re-run, so commit before re-running.

**Two identical runs counted as two trials.** They differ in code version (dirty or new commit), data hash, or run spec (`--max-days`, NAV, venue, rebalance frequency, ML factory, signal lag).

**`survivorship_check` warns.** The source has no delisted names. With yfinance that is expected and the result is not reportable. With a store, the delisted securities were not exported.

**Few or no eligible names.** Check that `prices_close_raw` exists (adjusted prices understate old price levels), that volume is in shares (not dollars or split-adjusted), and the universe size over time. The first refresh needs at least 31 days of volume history for ADV, so the start of the panel has no universe.

**The backtest is slow.** It is solver-bound (one convex solve per day, up to 8 during the γ search). Use `--max-days`, a smaller `universe.top_n_by_mcap`, or `--rebalance-every 5`. Leave `portfolio.reuse_problem` off: its compile needs memory growing with the square of the number of names (about 3 GB at 600).

**Many "failed solves" in the summary.** Constraints conflict: usually the ADV participation cap against forced exits of illiquid names, or bands that the universe cannot satisfy. The book is held on a failed day. `portfolio.relax_participation_on_infeasible: true` re-solves without the ADV caps (opt-in).

**Warnings about "name-days outside the risk-model universe".** A few hundred are normal (exiting names). Many mean the alpha universe and the risk model's estimation universe disagree, for example missing market caps.

**The UI says it cannot reach the API.** Start the API (`make api-dev`) and check http://localhost:8000/api/health. If the UI is served from another host or port, set `QTS_CORS_ORIGINS` (and `QTS_ALLOWED_HOSTS` for a non-local host name), or use single-origin mode.

**A launch returns 429.** The job queue is full (`QTS_MAX_QUEUED`, default 4). Wait, or raise `QTS_MAX_WORKERS` if the machine has the memory for parallel backtests.

**Lock files in `runs/` and `data/store/.locks/`.** Empty `flock` targets for the ledger and the store; harmless. Keep both on a local disk (no NFS).

## References

- [Saxena & Stubbs (2013), Alpha alignment factor: a solution to the underestimation of risk for optimized active portfolios](https://doi.org/10.3905/jor.2013.15.3.116)
- [Nyholt (2004), A simple correction for multiple testing for SNPs in linkage disequilibrium with each other](https://doi.org/10.1086/383251)
- [Ken French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
- [FINRA, Equity Short Interest data](https://www.finra.org/finra-data/browse-catalog/equity-short-interest)
- [QuantRocket, Sharadar fundamentals documentation](https://www.quantrocket.com/docs/data/fundamental/sharadar/)
- [Norgate Data, stock market packages](https://norgatedata.com/stockmarketpackages.php)
