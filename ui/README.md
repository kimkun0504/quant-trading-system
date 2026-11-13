# QTS Workbench (UI)

A research terminal for the `qts` pipeline: runs grouped into experiment families, selected
into a tray and compared side by side (NAV, fold Sharpe, gates, config diff); each run as a
one-page dossier (verdict, gates, performance, signal, portfolio, costs, validation, config,
artifacts); the run builder; the trial ledger. Design system and
tokens: [`DESIGN.md`](./DESIGN.md); the agreed redesign plan and build order:
[`REDESIGN.md`](./REDESIGN.md) (shipped: all six steps: design system, shell and Experiments; Compare; the run dossier; the three labs; the run builder with diff and queue).

Stack: Vite 8 · React 19 · TypeScript (strict) · React Router 8 (data router, lazy routes) ·
TanStack Query 5 · CSS Modules over custom-property tokens · uPlot (long time series) ·
d3-scale / d3-shape / d3-array (small SVG charts) · Vitest + Testing Library + MSW 3 ·
Playwright + axe · ESLint (typescript-eslint strict, react-hooks, jsx-a11y) + Prettier.

## Scripts

| Script                            | What it does                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `npm run dev`                     | Dev server on :5173 against the real API: `/api` is proxied to http://localhost:8000 (`QTS_API_PROXY` overrides).                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `npm run dev:mock`                | Dev server on the in-browser mock API (MSW fixtures; `VITE_USE_MOCKS=true`).                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `npm run build`                   | Typecheck (`tsc -b`) and production build to `dist/`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `npm run build:analyze`           | Build plus `dist/bundle-report.html` (treemap, gzip sizes).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `npm run preview`                 | Serve `dist/`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `npm run typecheck`               | `tsc -b` over app, tests, configs and e2e.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `npm run lint`                    | ESLint, zero warnings allowed.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `npm run format` / `format:check` | Prettier.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `npm test`                        | Vitest: unit, component and integration tests (MSW in Node).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `npm run test:e2e`                | Playwright smoke tests and axe scans (starts the dev server with mocks).                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `npm run test:e2e:real`           | Build, then one Playwright smoke against the real API: `api/scripts/e2e_server.py` serves `dist/` and the API over a runs root holding a real pipeline run (built on first use, ~30 s).                                                                                                                                                                                                                                                                                                                                                           |
| `npm run api:types`               | Export the API's OpenAPI document (`api/openapi.json`) and regenerate `src/shared/api/generated.ts` (`openapi-typescript`).                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `npm run contrast`                | WCAG contrast of every token pair in both themes; non-zero exit on failure.                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `npm run screenshots`             | Light/dark × 1440/390 px screenshots of every page into `docs/screenshots/` (Experiments with and without a selection, Compare, the three labs with the 24-year run (`signal-lab`, `portfolio-lab`, `costs-lab`), with three runs (`*-lab-3`) and the picker (`signal-lab-picker`), the dossier (`dossier-*`: top, Portfolio section, Config section, a suspicious run), the run builder (`run-builder`: defaults; `run-builder-diff`: cloned from a run with two edits; `run-builder-queue`: two variants queued, one running), ledger, models). |

Playwright uses its own Chromium if present, otherwise the newest `chromium-*` under
`PLAYWRIGHT_BROWSERS_PATH` (default `/opt/pw-browsers`), or `PW_CHROMIUM_PATH` if set
(`scripts/chromium.mjs`).

## Architecture

Feature-sliced: a layer may import only from layers below it.

```
src/
  app/        App, QueryClient defaults, route table (every page is a lazy chunk), route error page
  pages/      experiments (home: families table + inline compare), compare (the tray's target),
              signal-lab, portfolio-lab, costs-lab (each: <Name>LabPage + lib/shape.ts, the pure data
              shaping with its tests; labs.test.tsx renders all three through the route tree),
              run-dossier (DossierHeader, SectionNav + useScrollSpy, DossierSection, sections/: Performance,
              Signal, Portfolio, Costs, Validation, Config, Artifacts; RunTabRedirect for the old tab paths),
              run-builder (RunBuilderPage: `?from=` / `?set=` → base, the builder, the queue), models, ledger, not-found
  widgets/    AppShell (sidebar + top bar + main + tray), SelectionTray, MetricTable
  features/   selection (store + URL sync), compare (ComparePanel), run-search (global combobox),
              lab-runs (LabPage frame: the tray's runs as lanes, RunPicker when the tray is empty, RunLanes),
              run-builder (RunBuilder; StartFrom, RunSettingsPanel, CommonSettings + ConfigEditor, FieldControl,
              LaunchSlip; lib/: fields (schema → form), draft (base, values, diff, request), cli, reducer, common,
              trials), run-queue (QueuePanel, QueueIndicator, lib: queue reducer and wording), theme-toggle
  entities/   run (query hooks, query keys, selectors, families: clustering + config diff),
              metric (definitions + interpretations), gate (definitions, verdict)
  shared/     api (schema, client, endpoints, generated types, contract check),
              ui (Button, IconButton, Badge, Tooltip, Definition, Panel, Table/Th/Td, SortableTable, TabNav,
              Kbd, About (+ MarginNote alias), Page/Section/Annotated, SegmentedControl, Skeleton,
              Empty/ErrorState, VisuallyHidden, icons),
              charts (ChartFrame; NavDrawdownChart, TimeSeriesChart and OverlayChart on uPlot; BarChart,
              GroupedBarChart, StackedBarChart, PointLineChart, MultiPointLineChart, SmallMultiples,
              MonthlyHeatmap, Heatmap (diverging / sequential cells + HeatLegend), StackedAreaChart
              (d3 stack, diverging offset) in SVG/HTML),
              lib (format, stats, series (rolling mean/sd, period grouping, histogram, thinning), theme,
              shortcuts, motion, handbook links), styles (tokens.css, global.css)
  mocks/      MSW handlers, in-memory registry, deterministic fixture generator
  test/       Vitest setup and render helpers
e2e/          Playwright smoke, shell (nav, search, tray, compare, theme) + axe specs
scripts/      contrast check, screenshots, Chromium resolver
```

## Routes

| Path                              | Page                                                                                                                                                    |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `/`                               | Experiments: runs grouped into families; inline side-by-side when ≥ 2 are selected                                                                      |
| `/compare`                        | The selection as a full-page comparison (`?sel=a,b` seeds it)                                                                                           |
| `/signal`, `/portfolio`, `/costs` | The labs, on the runs in the selection tray (one run: the deep view; 2–8: one lane per run, overlays on one chart); a run picker when the tray is empty |
| `/builder`                        | Run builder: `?from=<run_id>` starts from that run, `?set=key=value` (repeatable) seeds edits, `#queue` is the queue                                    |
| `/runs/:id`                       | Run dossier: one page, sections `#performance` `#signal` `#portfolio` `#costs` `#validation` `#config` `#artifacts`                                     |
| `/runs/:id/<tab>`                 | Old tab paths redirect to the section (`risk` → `#portfolio`, `alpha` → `#signal`, `execution` → `#costs`)                                              |
| `/ledger`, `/models`              | Trial ledger, model explorer                                                                                                                            |
| `/runs`, `/runs/new`              | Redirect to `/` and `/builder`                                                                                                                          |

**Selection and tray.** `features/selection` keeps the run ids chosen for comparison (max 8, the
API's limit) in a module store persisted to `localStorage` (`qts-selection`);
`SelectionUrlSync` in the shell mirrors it into `?sel=` on every page (a URL with `sel=` wins
on first load, so a link shows what it was meant to show). The tray at the bottom lists the
runs as chips in series-palette order, removes them, and opens `/compare`. Shortcuts: `/`
focuses search, `c` opens compare (ignored while typing in a field).

**Labs.** `features/lab-runs` reads the tray (`useLabRuns`: finished runs in selection order, unfinished
ones reported) and frames every lab (`LabPage`: title, run strip with series swatches, `RunPicker` when the
tray is empty, `RunLanes` = one panel per run, three across at most). _Signal lab_: IC by theme by year
heatmap (diverging, t in the readout, `/alpha/ic-by-year`), IC decay per horizon (all runs on one
`MultiPointLineChart`), the IC series with its trailing one-year mean (one run) or the rolling means
overlaid, learned weights over time (features × refit heatmap per sleeve from `/alpha/refits`, each refit
normalised to sum 1 in `refitGrid`), theme weights over time (`/alpha/theme-weights`, stacked area) and
the feature IC table. _Portfolio anatomy_: gross/net/turnover with empty-book spans, exposure by industry
(long stacked up, short down, from `/holdings` on ≤ 36 month ends), exposure by style (theme stack +
small multiples with the ±0.05 band on risk-only styles), effective holding period (rolling, overlaid),
realised 63-day vol against ex-ante, and the holdings snapshot with a date scrubber, top longs/shorts and
the full table. _Cost & constraint lab_: binding constraints over time (constraint × month/quarter share
heatmap + whole-run bars, `/constraints`), transfer coefficient and ADV-cap usage overlaid, the vol-target
outcome shares (0–1 stacked area), costs by month and kind, the cost-per-side distribution (histogram of
daily trading cost per dollar traded) and the cost-sensitivity table with breakeven. Every chart sits in a
`ChartFrame` (data table) with an `About` line; older runs without the recordings get "Not recorded" states.

**Families.** `entities/run/families.ts`: inside one ledger family (`pipeline_phase{p}:{data_hash[:12]}`),
runs whose flattened configs (`RunSummary.config_leaves`) differ in ≤ 3 leaf keys are
single-linkage clustered; the family label is the members' common label prefix, else the
shortest label; the header lists the keys that vary. `configDiff()` gives the keys that differ
across the compared runs with one value per run.

Conventions:

- One component per file, a CSS Module beside it. No colour literals outside `tokens.css`;
  SVG uses `var(--color-*)`, canvas (uPlot) reads resolved tokens via `useChartTokens()`.
- Numbers use proportional figures with `tabular-nums`; negatives use the true minus sign;
  signed formats always show `+`. All formatting goes through `shared/lib/format.ts`.
- Charts never use two y-axes. Every chart sits in a `ChartFrame` with a caption, legend
  (≥ 2 series), an `aria-label` summary and a "Show data table" fallback.
- Loading states are skeletons with the final footprint, so nothing shifts when data lands.

## Data flow

```
component → entities/run/queries.ts hook → shared/api/endpoints.ts → shared/api/client.ts → fetch
                         ↑ TanStack Query cache, keyed by runKeys / queueKeys
```

- **Query keys** are hierarchical (`entities/run/queries.ts`):
  `['runs','list']`, `['runs','detail',id]`, `['runs','detail',id,'series','nav,drawdown,net']`,
  `…,'costs'`, `…,'exposures'`, `…,'holdings',date`, `…,'ic'`, `…,'alpha','refits'`,
  `…,'alpha','theme-weights'`, `…,'alpha','ic-by-year'`, `…,'constraints'`, `['jobs','list']`, `['jobs','log',jobId]`, `['config-validate', body]`, `['health']`, `['ledger']`,
  `['config-schema']`. Invalidating `['runs']` refreshes everything about runs.
- A finished run never changes, so its series, costs, exposures and IC use `staleTime: Infinity`.
- `useQueue` polls `/api/jobs` every second while a job runs or waits (every 15 s when idle) and
  invalidates `['runs']` and `['ledger']` when a job changes state; `useJobLog` refreshes an open log every
  2 s while its job runs. Cancel and move update the cached listing first (`queueReducer`), then refetch.
- `useValidateConfig` / the builder's `useQueries` key `POST /api/config/validate` by the request body, so
  each distinct edit is validated once (after a 300 ms debounce).
- 4xx answers are not retried (missing run, run still running → 409); transport and 5xx errors
  are retried twice. `ErrorState` turns `ApiError`s into direction ("Check that the qts API is
  running and that VITE_API_URL points at it").
- The dossier fetches the run once (`useRun`) and passes it to its sections as a prop; each
  section fetches only the extra data it charts (series, IC, costs, exposures, files, and the
  compare endpoint for the config leaves / diff).

## API contract and the mock layer

`src/shared/api/schema.ts` is the contract (field names follow `BacktestResult.summary()`,
`HurdleResult.to_dict()` and `RunManifest` verbatim). The API (`api/`, see `api/README.md`)
implements it with Pydantic models of the same names; `src/shared/api/generated.ts` is
`openapi-typescript` output of its OpenAPI document and `src/shared/api/contract.check.ts`
asserts, at type level, that every `schema.ts` type and every endpoint's response type agrees
with the generated one. After changing either side run `npm run api:types`; `npm run
typecheck` then fails on any drift. (`schema.ts` stays hand-written rather than re-exporting the
generated types because the UI narrows generic shapes such as `ColumnarSeries<N>`.)

`api.*` paths are relative to `API_BASE_URL` = `VITE_API_URL` or `/api` (same origin).
Errors are RFC 7807 problem documents; `ApiError.detail` is their `detail` string.

| Method | Path                                                                                                 | Returns                                                                                                                                                                                                                                                                      |
| ------ | ---------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| GET    | `/api/runs`                                                                                          | `RunSummary[]` (headline `null` and `config_leaves` empty while running / failed)                                                                                                                                                                                            |
| GET    | `/api/runs/compare?ids=a,b,c`                                                                        | `RunCompare`: NAV on one date axis (≤ 2,000 points), folds, gates, flattened configs (1–8 ids)                                                                                                                                                                               |
| GET    | `/api/runs/{id}`                                                                                     | `RunDetail` (409 while running/failed, 404 if unknown)                                                                                                                                                                                                                       |
| GET    | `/api/runs/{id}/series?names=nav,gross,net,drawdown,turnover,exante_vol,gross_exposure,net_exposure` | columnar `{dates, series}` (`*_exposure`: Σ\|w\| and Σw of the post-trade book)                                                                                                                                                                                              |
| GET    | `/api/runs/{id}/files` · `/files/{name}`                                                             | the run directory's files with sizes and descriptions; `/files/{name}` downloads one                                                                                                                                                                                         |
| GET    | `/api/runs/{id}/costs`                                                                               | daily `spread/impact/fees/borrow` USD, daily NAV, `cost_sensitivity`                                                                                                                                                                                                         |
| GET    | `/api/runs/{id}/exposures`                                                                           | factor list, roles, columnar history, latest snapshot (or empty)                                                                                                                                                                                                             |
| GET    | `/api/runs/{id}/holdings?date=`                                                                      | positions on a month-end decision date + available dates                                                                                                                                                                                                                     |
| GET    | `/api/runs/{id}/ic`                                                                                  | per-date rank IC + decay curve                                                                                                                                                                                                                                               |
| GET    | `/api/runs/{id}/alpha/refits`                                                                        | `AlphaRefits`: per-refit feature importance by sleeve (raw), refit windows, feature IC table                                                                                                                                                                                 |
| GET    | `/api/runs/{id}/alpha/theme-weights`                                                                 | `AlphaThemeWeights`: the composite's theme weights by date (≤ 2,000 points)                                                                                                                                                                                                  |
| GET    | `/api/runs/{id}/alpha/ic-by-year`                                                                    | `AlphaIcByYear`: mean rank IC and t per sleeve per calendar year                                                                                                                                                                                                             |
| GET    | `/api/runs/{id}/constraints`                                                                         | `RunConstraints`: per-rebalance binding counts, vol-target reason, transfer coefficient                                                                                                                                                                                      |
| POST   | `/api/runs`                                                                                          | adds the run to the queue: `{job_id, run_id}` (202; 422 with per-field `errors`; 429 when the queue is full). Body: data kind and size or store slice + prefilter, `ml`, `seed`, `signal_lag_days`, `config_overrides`, optional `base_run_id`, `label`, `max_days`, `cache` |
| POST   | `/api/config/validate`                                                                               | `ConfigValidation`: the same body, nothing queued: `errors` (per key), `warnings`, the resulting `config_hash`, `warmup_days` / `min_days`                                                                                                                                   |
| GET    | `/api/jobs`                                                                                          | `JobQueue`: running, then queued in order, then recent finished / failed / cancelled jobs (`?recent=`)                                                                                                                                                                       |
| GET    | `/api/jobs/{job_id}`                                                                                 | `Job`: status (`queued`, `running`, `done`, `failed`, `cancelled`), progress 0..1, message, stage, queue position, last log line, ETA                                                                                                                                        |
| GET    | `/api/jobs/{job_id}/log?tail=`                                                                       | `JobLog`: the last lines of the job's log                                                                                                                                                                                                                                    |
| POST   | `/api/jobs/{job_id}/move`                                                                            | `{position}` → the moved `Job` (409 unless queued)                                                                                                                                                                                                                           |
| DELETE | `/api/jobs/{job_id}`                                                                                 | cancel: a queued job leaves the queue, a running one is terminated; returns the job                                                                                                                                                                                          |
| GET    | `/api/ledger`                                                                                        | trials + per-family trial counts                                                                                                                                                                                                                                             |
| GET    | `/api/config/schema`                                                                                 | JSON Schema of `SystemConfig`, generated from the live dataclasses (type, default, bounds, enum, `nullable`, description per field): drives the builder's editor                                                                                                             |
| GET    | `/api/health`                                                                                        | liveness, runs root, the configured store path, job counts                                                                                                                                                                                                                   |

**Older runs.** Run directories written before the pipeline stored its extra artifacts have
`null` for `execution_venue`, `config.stocks/days` (unless recoverable from the recorded CLI
command), per-holding `alpha/specific_vol/industry`, an all-`null` `exante_vol` series and an
empty exposures payload (`latest: null`). The dossier's Portfolio section shows empty states or
"not recorded" text for these.

**Run builder** (`/builder`, `features/run-builder`). _Start from_ the SystemConfig defaults or a finished
run: a searchable picker, the deep link `/builder?from=<run_id>`, and "Clone & edit" on the dossier header
and on every finished row of Experiments. A run base loads its config (`RunSummary.config_leaves`) and its
recorded data source (`RunDetail.config.source`: synthetic size, or store path + date range + prefilter,
ML, seed, lag, `--max-days`, `--cache`); a run that did not record its source says so. _Editor_: every
field comes from `/api/config/schema` (`lib/fields.ts`), so a field added to `SystemConfig` appears with its
type, default, bounds and description and no UI change: grouped by section, with search, a "changed only"
filter, per-field reset, and an input per type (number with bounds, switch, enum select, text; a "No value"
switch on nullable fields). `lib/common.ts` curates the _Common settings_ panel on top (plain-language
labels and help; a path the schema lacks is skipped, e.g. `alpha.sample_weight_halflife_years` on an older
config). _Data and model_ holds what is not a config key (market, date range, universe prefilter, ML, seed,
signal lag, backtest days, stage cache, label). _Validation_: types and bounds are checked as you type
(`parseRaw`), and the whole request is sent, debounced, to `POST /api/config/validate`, whose per-key errors
land on their fields and whose warnings, config hash and warm-up go to the slip. _Before launch_ (the
sticky slip): only the keys that differ from the base, one column per variant; the config hash; the
equivalent `qts demo` / `qts run … --set k=v` command with a copy button (`lib/cli.ts`, the same text the
API records in `launch.json`); a warning when more than three strategy keys differ ("a new strategy, not a
variant"); and the trial line from `/api/ledger` ("This will be trial N+1 in family F", or why the family
cannot be known before the data is loaded). _Variants_: "Add variant" duplicates the current edit; "Queue
all N variants" posts them in order. State is one reducer (`lib/reducer.ts`): a base plus drafts that store
only what differs from it.

**Queue** (`features/run-queue`). `QueuePanel` (on the builder, under the slip) shows the running job
(progress, the pipeline stage rail, the message, the last log line, time left from finished jobs of
similar size, a log tail on demand), the waiting jobs in order (move up / down, cancel) and the recently
finished ones (link to the dossier, "Add to tray"). `QueueIndicator` in the top bar of every page shows
"1 running, 2 queued" with a small meter and links to `/builder#queue`.

**Mocks.** `src/main.tsx` starts the MSW service worker only when `VITE_USE_MOCKS=true`
(`npm run dev:mock`, the Vitest setup and the mock Playwright suite); in any other build the mock
code is never loaded (it is not in the production bundle). Handlers
(`src/mocks/handlers.ts`) match `*/api/...`, so they work whatever `VITE_API_URL` says. The
fixtures (`src/mocks/fixtures/generate.ts`) are deterministic and internally consistent: every
summary metric, gate and DSR is computed from the generated series (DSR with the same formula
as the handbook's worked example; a unit test checks it reproduces 0.666); each run carries a
config tree (`RunSpec.overrides` sets dotted keys) that feeds `config_yaml`, `config_leaves`
and the config hash. Seed runs: the post-audit phase-1 baseline plus two one-knob siblings
(`portfolio.trade_aversion: 0`, `alpha.horizon_days: 10`) that form a family with it, a
phase-2 ML run below its bar, a lag-0 run with a suspicious Sharpe, a passing data-store run, a 24-year
ML store run (2001–2025) for the labs' by-year views, one running and one failed run. The lab fixtures
(`fixtures/labs.ts`) tell one story keyed on the calendar year, so every run agrees on the years it
covers: value fades after 2010, quality rises from 2012, momentum crashes in 2009; refit importance follows
the same drift; caps bind more in 2008–09 and 2020 and with trade aversion off. `/api/runs/compare` is served from the same fixtures.
The mock queue (`mocks/db.ts`) behaves like the API's: FIFO, one job at a time, each taking `jobDurationMs`
(4 s; `localStorage['qts-mock-job-ms']` overrides it in the browser, which the screenshot script uses), with
stage, message, log lines and ETA derived from the elapsed time; a finished job becomes a run built from its
base run's fixture plus the overrides (so it joins the base's family in Experiments) and a ledger trial.
`POST /api/config/validate` checks overrides against the mock schema and hashes the resulting config tree.

**Full stack locally:** `make dev` from the repo root (API on :8000 with `--reload`, Vite on
:5173), or the two halves: `uvicorn qts_api.main:app --reload` and `npm --prefix ui run dev`.
Production-like: `npm --prefix ui run build && QTS_SERVE_UI=1 uvicorn qts_api.main:app`
serves the UI and the API from one origin.

## How to add…

**A metric.** Add the key to `MetricKey` and an entry to `defs` in
`entities/metric/definitions.ts`: label (handbook vocabulary), one-sentence definition, plain
formula, handbook section, `format`, and `interpret(value, ctx)` — a sentence led by the value
saying what it means _for this run_, honest about weak or suspicious values. Add any run facts
it needs to `MetricContext` and `metricContext()` (`entities/run/selectors.ts`). Add test cases
to `definitions.test.ts` (the suite already checks every metric handles missing values). Show it
by adding `{ key, value }` to a `MetricTable`.

**A gate.** Add the name to `GateName` (schema), then a `GateDef` and its position in
`GATE_ORDER` (`entities/gate/definitions.ts`). If it changes the conclusion of the verdict line,
add a branch to `conclusion()` in `verdict.ts` and a test in `verdict.test.ts`.

**A page.** Create `pages/<name>/<Name>Page.tsx` rendering `<Page docTitle title>`; put content
in `Panel`s, explanations in an `About` line (with its handbook ref), titled parts in
`Section`. Register it in `app/routes.tsx` with
`lazy: () => import(...).then(m => ({ Component: m.X }))` and add it to `WORKSPACES` or
`REGISTRY_NAV` in `widgets/AppShell/nav.ts` if it belongs in the sidebar. Add a line to
`e2e/a11y.spec.ts` and `scripts/screenshots.mjs`.

**A column on Experiments.** Add the key to `SortKey`, its accessor to `SORT_VALUE` and its
header to `COLUMNS` in `pages/experiments/ExperimentsPage.tsx`, and the cell to `RunRow`
(`<Td numeric>` for figures). If the number is not yet in `RunHeadline`, add it on both sides
of the contract (`schemas.py`, `schema.ts`, the fixture generator) and run `npm run api:types`.

**A dossier section.** Add the id to `SectionId` and an entry (title, lab link) to `SECTIONS` in
`pages/run-dossier/sections.ts` (reading order = array order), write
`pages/run-dossier/sections/<Name>Section.tsx` rendering `<DossierSection id=…>` with `Panel`s
and `About` lines, and mount it in `RunDossierPage.tsx`. The side nav, scrollspy, `j`/`k` and
`#anchor` handling pick it up from `SECTIONS`.

**A common setting (run builder).** Add `{ path, label, help }` to `COMMON_SETTINGS` in
`features/run-builder/lib/common.ts`. The control, its default, bounds and validation come from the
schema; a path the schema does not have is skipped. A new `SystemConfig` field needs nothing here: it is in
"All settings" as soon as the API serves it (regenerate the mock schema entry in
`mocks/fixtures/configSchema.ts` if tests should cover it).

**A chart.** Small, few-point charts: SVG with d3-scale in `shared/charts/`, sized with
`useMeasure()`, colours from `var(--color-*)`, a hover readout, an `aria-label` summary.
Long series (hundreds of points or more): build on `UplotChart` (see `TimeSeriesChart`) and
import it from its own file, not the `shared/charts` barrel, so uPlot stays in lazy chunks.
Wrap it in `ChartFrame` with a `table` (a function, so the data table is built only when opened).

## Quality gates (current)

- `npm run typecheck`, `npm run lint`: clean (strict TS, no `any`, `strictTypeChecked` rules).
- `npm test`: 166 tests (format, stats/DSR, metric interpretations, verdict generation,
  selectors, family clustering + config diff, selection store, primitives,
  API client, MetricTable with MSW, Experiments and the dossier through the real
  route tree: header facts and gates, seven sections, section nav + scrollspy, `j`/`k`, add to
  tray, config search + diff, artifacts, old-tab redirects; the labs' data shaping and all three labs
  through the route tree with the picker, one run and three runs; the run builder's library: schema → form
  mapping (types, bounds, nullable, a field the schema gains, the curated common settings), the validation
  states of an input and of the launch settings, base / diff / request, the CLI command builder, the builder
  reducer (variants), the trial forecast; the queue reducer and the mock queue progressing over time; and the
  builder through the route tree: defaults, inline validation and reset, search / changed-only / nullable,
  clone → two edits → diff → add variant → queue both → running + queued → finished, the new-strategy
  warning, switching the base, an unusable base).
- `npm run test:e2e`: smoke (Experiments → dossier: header, every section, section nav click +
  `j`/`k`, data-table toggle; old tab redirects, add to tray → Compare, config search/diff,
  artifact download link; suspicious run, skip link, theme toggle + persistence), shell (sidebar to every
  workspace and the old redirects, `/` search → Enter opens the run, tray fill / URL / reload /
  `c` → compare page / chip removal / clear, `?sel=` seeding), labs (each with one and three
  selected runs: picker → deep view, sleeve switch + heatmap readout, holdings scrubber, data
  tables, dossier "Open in" links), builder (dossier "Clone & edit" → two fields changed → the diff lists
  them → add variant → queue both → one running, one queued, log tail, top-bar indicator → both finished →
  add to tray → the runs in Experiments and their dossier; clone from an Experiments row, reorder and cancel
  queued runs; inline validation) and axe on 18 pages × 2 themes plus the builder with a running and a
  queued job in both themes, with zero serious or critical violations.
- `npm run contrast`: all 70 pairs pass (table in `DESIGN.md`).
