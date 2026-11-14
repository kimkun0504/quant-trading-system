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
UI's *Run builder*, or `python -m qts_api.fixture runs` for a tiny run (~30 s).

## Settings (environment)

| Variable               | Meaning                                    | Default                           |
| ---------------------- | ------------------------------------------ | --------------------------------- |
| `QTS_RUNS_ROOT`        | run directories                            | `./runs`                          |
| `QTS_LEDGER_PATH`      | trial ledger JSONL                         | `<runs root>/trial_ledger.jsonl`  |
| `QTS_CORS_ORIGINS`     | comma-separated allowed origins            | `http://localhost:5173`           |
| `QTS_SERVE_UI`         | `1`/`true`: serve the built UI at `/`      | off                               |
| `QTS_UI_DIST`          | the built UI                               | `<repo>/ui/dist`                  |
| `QTS_MAX_WORKERS`      | concurrent pipeline jobs                   | `1`                               |
| `QTS_STORE_PATH`       | `ParquetStore` root for `kind: "store"` (also reported by `/api/health`) | unset (store launches get a 422)  |
| `QTS_JOB_START_METHOD` | `multiprocessing` start method for jobs    | `spawn`                           |
| `QTS_ALLOWED_HOSTS`    | extra accepted `Host` names (comma-sep.)   | none (`localhost`, `127.0.0.1`, `::1` always) |
| `QTS_MAX_QUEUED`       | jobs waiting for a worker; more get a 429  | `20`                              |
| `QTS_PAPER_ROOT`       | paper books (`qts paper --paper-root`)     | `./paper`                         |

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
| POST   | `/api/runs`                        | `LaunchRunResponse` 202 (queued) | 422, 429 |
| GET    | `/api/jobs?recent=`                | `JobQueue`               |               |
| GET    | `/api/jobs/{job_id}`               | `Job`                    | 404           |
| GET    | `/api/jobs/{job_id}/log?tail=`     | `JobLog`                 | 404, 422      |
| POST   | `/api/jobs/{job_id}/move`          | `Job` (body `JobMove`)   | 404, 409, 422 |
| DELETE | `/api/jobs/{job_id}`               | `Job` (cancel)           | 404           |
| GET    | `/api/ledger`                      | `Ledger`                 |               |
| GET    | `/api/config/schema`               | `JsonSchemaObject`       |               |
| GET    | `/api/config/warmup?ml=&signal_lag_days=` | `Warmup`          | 422           |
| POST   | `/api/config/validate`             | `ConfigValidation` (body: `LaunchRunRequest`) | 422 |
| GET    | `/api/health`                      | `Health`                 |               |

Research tools and paper books (`routers/research.py`, `routers/paper.py`; the library's own JSON
through permissive models): `/api/seal` (+ `/unseals`), `/api/ledger/impact` (DSR haircut of N new
trials), `/api/features`, `POST /api/families/resolve` (+ `/jobs`: compute it as a job), `/api/signals`,
`/api/signals/tests` (+ `/preview`, `POST` launch,
`/{test_id}`), `/api/sweeps` (+ `/plan`, `POST` create, `/{sweep_id}`, `/{sweep_id}/resume`),
`/api/capacity`, `/api/risk-calibrations`,
`/api/checks/midcap` (+ `/preregistration`, `/results/{run_id}`, `POST` launch: once-only, `force`
needs the typed phrase), `/api/paper/books` (+ `/{name}/status|history|orders|holdings|schedule`,
`POST /{name}/ack|run|reconcile`). Long work runs as **task jobs** in the same queue (`task`,
`subject`, `command`, `result`, `error_kind`: `stale_data` exit 3, `book_locked` exit 4, `coverage`,
`seal`, `refused`); every launch records the exact `qts ...` command. See `/docs` for the schemas.

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
    jobs.py          JobManager (persisted FIFO queue of worker processes) + the worker entry point run_job
    launch.py        prepare_launch: request -> SystemConfig or issues (shared by validate and launch)
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

### Launch, validate, queue

`POST /api/runs` and `POST /api/config/validate` take the same body (`LaunchRunRequest`) and prepare it
with the same function (`services/launch.py: prepare_launch`), so what the dry run reports is what a
launch does:

1. Pydantic (strict): ranges, enums, unknown fields; a malformed body is a 422 on both routes.
2. Base config: `SystemConfig()`, or the `config.yaml` of `base_run_id` (the run builder's "start from a
   run"; a config written by an older `SystemConfig` that no longer strict-loads is an error on
   `base_run_id`). A base run's `cache` section is not inherited.
3. Overrides: each dotted path must name a scalar leaf with a value of the right type inside the curated
   bounds and enums (`null` only for nullable leaves); `seed`, `execution.signal_lag_days` and `cache.*`
   are set by the request's own fields. They are then applied one key at a time through
   `SystemConfig.with_overrides`, so a section's own `__post_init__` validation names the key that broke
   it, and the result is round-tripped through `from_dict` / `validate()`.
4. The request's `seed` and `signal_lag_days`; for a synthetic market `universe.top_n_by_mcap` is clamped to
   `stocks` (what the pipeline does on load, so the hash shown is the hash of the run) and `days` must
   cover the run's own warm-up plus one year (`qts.pipeline.estimate_warmup_days`).

`ConfigValidation` = `{valid, errors, warnings, config_hash, base_config_hash, warmup_days, min_days,
overrides_applied}`. Config problems are `errors` with a 200 (`valid: false`, `config_hash: null`), each
with the `loc` a launch's 422 would carry (`["body", "config_overrides", "<path>"]`, `["body", "days"]`,
...). Warnings (`type: "warning.*"`) do not block: `lag0` (same-close execution), `ml_off` (an
`alpha.lgbm.*` / `alpha.mlp.*` override with the ML sleeve off), `same_config` (the resulting hash equals
an existing run's: on the same data and clean code a re-run, not a new trial).
`GET /api/config/warmup?ml=&signal_lag_days=` still returns the warm-up of the default config.

A launch then fixes the run id (`<UTC timestamp>_<config_hash[:8]>`), puts the job on the queue and
answers 202 with the job and run ids (429 when `QTS_MAX_QUEUED` jobs are already waiting).

Store runs (`kind: "store"`) read `QTS_STORE_PATH` the way `qts run --store` does: `start` / `end` slice
the store and `max_rank` (screened market-cap rank, `--max-rank`) or `max_names` / `min_mcap_ever` (peak
market cap) prefilter before anything is pivoted; the library's load-coverage guard refuses a prefilter
that misses universe names unless `allow_partial_coverage` (`--allow-partial-coverage`, flagged). Store
runs load under the research seal (`unseal_reason`, `force`: once per configuration).
`cache: true` is `qts --cache` (stage cache under `<runs root>/_cache`; outside the config hash).

**The queue.** `JobManager` keeps a FIFO queue and runs at most `QTS_MAX_WORKERS` (default 1) jobs, each
in its own `spawn`ed process (backtests are CPU- and memory-heavy; a crash cannot take the API down).
Statuses: `queued` → `running` → `done` | `failed`, or `cancelled` from either. `<runs root>/_jobs/` holds

| File             | Content                                                                              |
| ---------------- | ------------------------------------------------------------------------------------ |
| `<job_id>.json`  | state, the request and the worker payload (written by the API and by the worker)     |
| `<job_id>.log`   | the worker's log: one line per stage plus the pipeline's INFO log (`qts.*`)          |
| `queue.json`     | the ids waiting, in order                                                            |

so the queue **survives an API restart**: finished jobs are still reported, `queued` jobs go back on the
queue in their order and run, and a job that was `running` is marked `failed` with a note (its worker died
with the API and wrote nothing; queue it again). A clean shutdown terminates the running worker (failed,
"the API shut down") and leaves the queue on disk.

`GET /api/jobs` lists running jobs, then the queue in order, then the newest finished / failed / cancelled
jobs (`recent`, default 20), with `max_workers`, `max_queued` and counts by status. A `Job` carries
`label`, `kind`, `ml`, `phase`, `config_hash`, `base_run_id`, `n_overrides`, the timestamps, `position`
(0-based place in the queue), `stage` (`data`, `universe`, `risk`, `alpha`, `backtest`, `evaluate`,
`write`), `last_log` (last line of the log), `estimated_seconds` (median duration of finished jobs of
similar size: same data kind and ML flag, and the same slice or a synthetic market within a factor of two,
scaled by names × days) and `eta_seconds` (that estimate minus the time already spent; without history,
extrapolated from the backtest's own day counter once it is 5% through; else `null`).
`GET /api/jobs/{id}/log?tail=` returns the last lines (default 200, at most 2,000).
`POST /api/jobs/{id}/move {position}` re-orders a queued job (409 otherwise).
`DELETE /api/jobs/{id}` takes a queued job off the queue, or terminates a running worker (best effort; a
job that finished first stays done); either way the job ends `cancelled`. `/api/runs` lists queued and
running jobs as `running` runs and failed jobs as `failed`; a cancelled job is not listed (direct access
to its run id is a 409).

**The worker** follows the `qts demo` / `qts run` code path (synthetic `generate(...)` or `ParquetStore` +
prefilter + `survivorship_check`, `top_n_by_mcap` clamped to the universe,
`ResearchPipeline(..., ledger_path, nav0=5e6).run(md, max_days)`), sets the manifest's run id, writes with
`save_run` + `write_extra_artifacts` into `runs/_staging/`, adds `launch.json` (the request, the store
path and the equivalent `qts` command line) and renames the directory into place, so readers never see a
half-written run. Progress is stage based: market data 2%, universe 5%, risk model 8%, alpha model 20%,
backtest 30-90% (50-90% with the ML sleeve) with "day k of n" from the simulator's every-50-days progress
log line, evaluation 91%, writing 96%, done 100%.

**Run source.** `RunDetail.config.source` (`RunSource`) says where a run's data came from and how it was
invoked, which is what the run builder needs to start from it: from `launch.json` for API jobs
(`origin: "api"`), else parsed from the CLI command in the manifest (`qts demo --stocks/--days/--seed`,
`qts run --store/--start/--end/--max-names/--min-mcap-ever`, `--max-days`, `--cache`; `origin: "cli"`),
else `recorded: false` with nulls. `RunDetail.manifest.ledger_family` is the trial family the manifest
records (null on older manifests).

Optional request fields beyond the builder's: `venue` (`routed` | `auction` | `continuous`).

### Config schema

`/api/config/schema` is generated from the live config dataclasses on every start (nothing is listed by
hand, so a field added to `SystemConfig` is served, and editable in the run builder, as soon as the code
has it): one `object` per section (nested sections such as `alpha.lgbm` included) with a title; one leaf
per scalar field with its type, default, `nullable: true` for optional scalars (`float | None`), a title
from the field name, a description from the field's comments in `qts/core/config.py` (falling back to
its ````field``: text`` entry in the class docstring, and a few hand-written where neither stands alone), enums
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
UI, and a full job lifecycle and a cancellation with real worker processes (~15 s and ~2 s; the
lifecycle also checks the log tail, `launch.json` / the run source, cloning the finished run and the ETA).
`test_queue.py` drives the queue with fake worker processes: FIFO one at a time, statuses, cancel and
reorder, restart recovery (queued jobs re-queued in order, a running job failed with a note, shutdown keeps
the queue), the log tail, the ETA estimate, `POST /api/config/validate` (errors per key, warnings, nullable
overrides, a base run, hash equal to the launch's) and the run-source parsing.

The TypeScript side: `npm --prefix ui run api:types` exports the OpenAPI document and regenerates
`ui/src/shared/api/generated.ts`; `ui/src/shared/api/contract.check.ts` makes `npm run typecheck`
fail when `schema.ts` and the API drift. `npm --prefix ui run test:e2e:real` runs a Playwright
smoke against this API serving the built UI.
