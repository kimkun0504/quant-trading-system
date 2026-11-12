# qts-api

FastAPI service over the `qts` research system: the run registry (`runs/<run_id>/`), the trial
ledger, the `SystemConfig` schema, and background pipeline jobs. It is the backend of the
workbench in `ui/`; the contract is `ui/src/shared/api/schema.ts`, mirrored name for name by the
Pydantic models in `qts_api/schemas.py`.

## Running locally

```bash
pip install -e .            # qts (repo root), once
pip install -e "api[dev]"   # qts-api + pytest/httpx/ruff   (or: make api-install)

uvicorn qts_api.main:app --reload          # API on :8000 (or: make api-dev)
npm --prefix ui run dev                    # UI on :5173, proxies /api to :8000 (or: make ui-dev)
make dev                                   # both
```

Open http://localhost:5173. Interactive API docs: http://localhost:8000/api/docs; the OpenAPI
document is `/api/openapi.json` (also exported to `api/openapi.json` by `npm --prefix ui run api:types`).

One origin, no dev server: `npm --prefix ui run build && QTS_SERVE_UI=1 uvicorn qts_api.main:app`
serves the built UI at `/` (SPA fallback to `index.html`) and the API under `/api`.

To fill the registry: `python -m qts.cli demo` (writes `runs/<run_id>/`), or launch from the
UI's *New run* page, or `python -m qts_api.fixture runs` for a tiny run (~30 s).

## Settings (environment)

| Variable               | Meaning                                    | Default                           |
| ---------------------- | ------------------------------------------ | --------------------------------- |
| `QTS_RUNS_ROOT`        | run directories                            | `./runs`                          |
| `QTS_LEDGER_PATH`      | trial ledger JSONL                         | `<runs root>/trial_ledger.jsonl`  |
| `QTS_CORS_ORIGINS`     | comma-separated allowed origins            | `http://localhost:5173`           |
| `QTS_SERVE_UI`         | `1`/`true`: serve the built UI at `/`      | off                               |
| `QTS_UI_DIST`          | the built UI                               | `<repo>/ui/dist`                  |
| `QTS_MAX_WORKERS`      | concurrent pipeline jobs                   | `1`                               |
| `QTS_STORE_PATH`       | `ParquetStore` root for `kind: "store"`    | unset (store launches get a 422)  |
| `QTS_JOB_START_METHOD` | `multiprocessing` start method for jobs    | `spawn`                           |
| `QTS_ALLOWED_HOSTS`    | extra accepted `Host` names (comma-sep.)   | none (`localhost`, `127.0.0.1`, `::1` always) |
| `QTS_MAX_QUEUED`       | jobs waiting for a worker; more get a 429  | `4`                               |

Requests whose `Host` header is not on the allow-list get a 400 (`TrustedHostMiddleware`):
this is the DNS-rebinding defence, since CORS does not stop a rebound page from reading a
local server. Keep the default bind of `127.0.0.1`.

The defaults match `python -m qts.cli` (`runs/`, `runs/trial_ledger.jsonl`), so CLI runs and
API runs share one registry and one ledger.

## Endpoints

| Method | Path                               | Response (`schemas.py`)  | Errors        |
| ------ | ---------------------------------- | ------------------------ | ------------- |
| GET    | `/api/runs`                        | `RunSummary[]`           |               |
| GET    | `/api/runs/compare?ids=a,b,c`      | `RunCompare`             | 404, 409, 422 |
| GET    | `/api/runs/{id}`                   | `RunDetail`              | 404, 409      |
| GET    | `/api/runs/{id}/series?names=`     | `ColumnarSeries`         | 404, 409, 422 |
| GET    | `/api/runs/{id}/costs`             | `RunCosts`               | 404, 409      |
| GET    | `/api/runs/{id}/exposures`         | `RunExposures`           | 404, 409      |
| GET    | `/api/runs/{id}/holdings?date=`    | `RunHoldings`            | 404, 409      |
| GET    | `/api/runs/{id}/ic`                | `RunIc`                  | 404, 409      |
| GET    | `/api/runs/{id}/alpha/refits`      | `AlphaRefits`            | 404, 409      |
| GET    | `/api/runs/{id}/alpha/theme-weights` | `AlphaThemeWeights`    | 404, 409      |
| GET    | `/api/runs/{id}/alpha/ic-by-year`  | `AlphaIcByYear`          | 404, 409      |
| GET    | `/api/runs/{id}/constraints`       | `RunConstraints`         | 404, 409      |
| GET    | `/api/runs/{id}/files`             | `RunFiles`               | 404, 409      |
| GET    | `/api/runs/{id}/files/{name}`      | the file (download)      | 404, 409      |
| POST   | `/api/runs`                        | `LaunchRunResponse` 202  | 422           |
| GET    | `/api/jobs/{job_id}`               | `Job`                    | 404           |
| DELETE | `/api/jobs/{job_id}`               | `Job` (cancel)           | 404           |
| GET    | `/api/ledger`                      | `Ledger`                 |               |
| GET    | `/api/config/schema`               | `JsonSchemaObject`       |               |
| GET    | `/api/config/warmup?ml=&signal_lag_days=` | `Warmup`          | 422           |
| GET    | `/api/health`                      | `Health`                 |               |

**Errors** are RFC 7807 problem documents (`application/problem+json`):
`{type, title, status, detail, errors?}`. `detail` is always a human-readable string (the UI shows
it verbatim); a 422 adds `errors: [{loc, msg, type}]`, one per bad field (for config overrides
`loc` is `["body", "config_overrides", "<dotted.path>"]`). A run that is still running or whose
job failed returns **409** on every `/api/runs/{id}*` endpoint (it is listed by `/api/runs` with
`status: "running" | "failed"` and `headline: null`); an unknown id is **404**.

**Numbers:** a value that is not finite in Python (NaN/inf) is serialised as `null`.

**Compare.** `GET /api/runs/compare?ids=` takes 1 to 8 run ids (duplicates dropped, order
kept) and returns their NAV on one shared axis — the union of the runs' realisation dates,
thinned to at most 2,000 evenly spaced points (first and last always kept; `downsampled`
says when) — plus, per run, the headline numbers, the DSR trial count, fold Sharpes, hurdles
and `config_leaves`. `config_leaves` (also on every `RunSummary`) is the run's `config.yaml`
flattened to dotted paths → scalars (`portfolio.trade_aversion: 1.0`); lists are compact JSON
strings so a differing list still diffs as one key. The UI clusters runs into families and
builds the config diff from it. `RunHeadline` also carries `invested_day_frac`. The run detail's
`BacktestSummary` carries the diversification numbers `effective_n_names`, `specific_var_share` and
`exante_vol_at_gross2` (null for runs recorded before 2026-11-12).

## Architecture

```
qts_api/
  main.py            create_app(settings): routers, CORS, problem handlers, optional UI mount
  settings.py        Settings.from_env()
  deps.py            app.state singletons as FastAPI dependencies
  schemas.py         Pydantic v2 models = the contract (mirrors ui/src/shared/api/schema.ts)
  problems.py        RFC 7807 responses for Problem, validation and HTTP errors
  routers/           runs, jobs, ledger, config, health
  services/
    registry.py      run registry: parse + cache run directories, build every run payload
    series.py        derived series (drawdown, turnover, positional alignment, cost sensitivity)
    jobs.py          JobManager (process-based queue) + the worker entry point run_job
    config_schema.py JSON Schema of SystemConfig + strict override validation
    ledger.py        trial ledger -> Ledger (mtime-cached)
  fixture.py         a tiny real run (tests, e2e)
  openapi.py         python -m qts_api.openapi [out.json]
scripts/e2e_server.py  serve built UI + API over a fixture runs root (ui/playwright.real.config.ts)
```

### Run registry

A run directory is written by `qts.pipeline.save_run` (manifest, config, summary, NAV / weights /
trades / costs / returns / IC Parquet, hurdles) plus `qts.runs.artifacts.write_extra_artifacts`,
which the pipeline calls right after `save_run` and which adds what only the in-memory pipeline
knows:

| File                 | Content                                                                    |
| -------------------- | -------------------------------------------------------------------------- |
| `extras.json`        | data kind, universe size, market days, seed, venue, label; DSR skew / kurtosis / n_obs; IC by year; factor roles |
| `exante_vol.parquet` | the risk model's annualised ex-ante vol of the post-trade book, by decision date |
| `exposures.parquet`  | decision date x factor: net book exposure to the market and each style   |
| `holdings.parquet`   | month-end positions (last 60) with alpha, specific vol, industry           |
| `alpha_refits.parquet` | long: refit_date, sleeve (lgbm / ridge / mlp / ensemble), feature, raw importance, one row per refit × sleeve × feature (ML runs only) |
| `alpha_refits.json`  | the refits' windows (train_start, train_end, valid_start, pred_end, n_train, n_valid, blend weights), embargo, notes, and the full-sample feature IC table |
| `theme_weights.parquet` | decision date × theme: the composite's (dynamic or static) theme weights |
| `alpha_ic_by_sleeve.parquet` | date × sleeve: daily OOS rank IC of combined, composite, ml and every theme score |
| `constraints.parquet` | one row per rebalance solve: status, vol_target_binding, transfer coefficient, vol scalar, n_solves, n_traded and `binding_<name>` counts |

`RunRegistry` discovers runs with `qts.pipeline.list_runs`, parses the JSON files once per
*signature* (name, mtime, size of every file in the directory) and loads the Parquet side with
`qts.pipeline.load_run` lazily into an 8-run LRU; a changed file re-parses on the next request.
Unreadable directories are skipped with a warning.

Derived on request (cheap): `drawdown` (`qts.backtest.metrics.drawdown_series` of net returns, so
its minimum equals `max_drawdown_net`), `turnover` (sum of |trades|), the positional alignment of
decision-date values onto realisation dates, per-day cost components with pre-trade NAV, fold
labels; IC by year and DSR moments are recomputed from `ic.parquet` / `returns.parquet` when
`extras.json` is absent. The UI computes rolling Sharpe and monthly returns from `/series`.

**Labs.** `/alpha/refits` pivots `alpha_refits.parquet` to sleeve → feature → one raw value per refit
(the UI normalises each refit to sum 1) and carries the windows and the feature IC table from
`alpha_refits.json` (`recorded: false` when the file is missing). `/alpha/theme-weights` serves
`theme_weights.parquet` thinned to 2,000 points. `/alpha/ic-by-year` groups `alpha_ic_by_sleeve.parquet`
by calendar year per sleeve with a Newey–West t (bandwidth floor = horizon + signal lag, the label
overlap); without that file it falls back to the sampled `ic.parquet` (combined row only, plain t,
`source: "sampled_combined"`). `/constraints` widens `constraints.parquet` into per-name binding-count
arrays plus the whole-run share of solves on which each constraint bound; older runs get empty arrays.

**Older runs** (no `extras.json`) still load: `execution_venue: null`; `config.stocks/days` from
the recorded `qts.cli demo` command when there was one, else `null`; `exante_vol` all `null`;
exposures empty with `latest: null`; holdings from `weights.parquet` with
`alpha/specific_vol/industry: null`. A `config.yaml` that no longer strict-loads (an older
`SystemConfig`) falls back to reading the frames directly.

### Jobs

`POST /api/runs` validates the request (Pydantic: ranges, enums, unknown fields), then the
overrides: each dotted path must name a scalar leaf of `SystemConfig` with a value of the right
type, inside the curated bounds and enums; the nested dict then goes through
`SystemConfig.from_dict` (strict) and `validate()`. `seed` and `execution.signal_lag_days` are set
by the request's own fields and rejected as overrides. For synthetic runs `days` must cover the
run's effective warm-up plus one year, computed from its own config (overrides included) by
`qts.pipeline.estimate_warmup_days`; `GET /api/config/warmup?ml=&signal_lag_days=` returns the
same numbers for the default config (the launch form's floor). The run id
(`<UTC timestamp>_<config_hash[:8]>`) is fixed at submission and returned with the job id.

`JobManager` keeps a FIFO queue and runs at most `QTS_MAX_WORKERS` (default 1) jobs, each in its
own `spawn`ed process (backtests are CPU- and memory-heavy; a crash cannot take the API down).
The worker follows the `qts.cli demo` / `run` code path (synthetic `generate(...)` or
`ParquetStore` + `survivorship_check`, `cfg.seed`, `top_n_by_mcap` clamped to the universe,
`ResearchPipeline(..., ledger_path, nav0=5e6).run(md, max_days)`), sets the manifest's run id,
writes with `save_run` + `write_extra_artifacts` into `runs/_staging/` and renames the directory
into place, so readers never see a half-written run.

Progress is stage based: market data 2%, universe 5%, risk model 8%, alpha model 20%, backtest
30-90% (50-90% with the ML sleeve) with "day k of n" from the simulator's every-50-days progress
log line, evaluation 91%, writing 96%, done 100%. State is persisted in `runs/_jobs/<job_id>.json`
(written by the worker, read by the API), so a restarted API still reports finished jobs; jobs
that were queued or running when the API stopped are marked failed ("Interrupted").
`DELETE /api/jobs/{id}` cancels a queued job, or terminates a running worker (best effort; a job
that finished first stays done).

Optional request fields beyond the UI form: `label`, `max_days` (backtest only the last N
out-of-sample days), `venue` (`routed` | `auction` | `continuous`).

### Config schema

`/api/config/schema` is generated from the config dataclasses: one `object` per section (nested
sections such as `alpha.lgbm` included) with a title; one leaf per scalar field with its type,
default, a title from the field name, a description from the field's comments in
`qts/core/config.py` (a few hand-written where the comment does not stand alone), enums
(`portfolio.book`, `portfolio.vol_target_mode`, `portfolio.solver`,
`cost.tca_reversion_reference`, `risk.exposure_dtype`) and curated bounds. Lists and dicts are
left out (not form-editable). The pricing venue is not a `SystemConfig` field; it is the request's
`venue`.

## Tests

```bash
cd api && pytest -q        # or: make api-test   (~45 s)
```

A session fixture builds one tiny real run with the pipeline (60 stocks x 764 days, last 120
out-of-sample days) plus an "old" copy without the extra artifacts. Every endpoint is validated
against the Pydantic models and spot-checked against the run's own files (`summary.json`,
`hurdles.json`, Parquet); 404/409/422 paths, cache invalidation, NaN -> null, CORS, serving the
UI, restart behaviour, and a full job lifecycle and a cancellation with real worker processes
(~15 s and ~2 s).

The TypeScript side: `npm --prefix ui run api:types` exports the OpenAPI document and regenerates
`ui/src/shared/api/generated.ts`; `ui/src/shared/api/contract.check.ts` makes `npm run typecheck`
fail when `schema.ts` and the API drift. `npm --prefix ui run test:e2e:real` runs a Playwright
smoke against this API serving the built UI.
