# QTS Workbench (UI)

A research workbench for the `qts` pipeline: browse backtest runs, read each run's verdict
and hurdle gates, inspect the five models' diagnostics, launch new runs and read the trial
ledger. Design brief and token decisions: [`DESIGN.md`](./DESIGN.md).

Stack: Vite 8 · React 19 · TypeScript (strict) · React Router 8 (data router, lazy routes) ·
TanStack Query 5 · CSS Modules over custom-property tokens · uPlot (long time series) ·
d3-scale / d3-shape / d3-array (small SVG charts) · Vitest + Testing Library + MSW 3 ·
Playwright + axe · ESLint (typescript-eslint strict, react-hooks, jsx-a11y) + Prettier.

## Scripts

| Script                            | What it does                                                                                                                                                                            |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `npm run dev`                     | Dev server on :5173 against the real API: `/api` is proxied to http://localhost:8000 (`QTS_API_PROXY` overrides).                                                                       |
| `npm run dev:mock`                | Dev server on the in-browser mock API (MSW fixtures; `VITE_USE_MOCKS=true`).                                                                                                            |
| `npm run build`                   | Typecheck (`tsc -b`) and production build to `dist/`.                                                                                                                                   |
| `npm run build:analyze`           | Build plus `dist/bundle-report.html` (treemap, gzip sizes).                                                                                                                             |
| `npm run preview`                 | Serve `dist/`.                                                                                                                                                                          |
| `npm run typecheck`               | `tsc -b` over app, tests, configs and e2e.                                                                                                                                              |
| `npm run lint`                    | ESLint, zero warnings allowed.                                                                                                                                                          |
| `npm run format` / `format:check` | Prettier.                                                                                                                                                                               |
| `npm test`                        | Vitest: unit, component and integration tests (MSW in Node).                                                                                                                            |
| `npm run test:e2e`                | Playwright smoke tests and axe scans (starts the dev server with mocks).                                                                                                                |
| `npm run test:e2e:real`           | Build, then one Playwright smoke against the real API: `api/scripts/e2e_server.py` serves `dist/` and the API over a runs root holding a real pipeline run (built on first use, ~30 s). |
| `npm run api:types`               | Export the API's OpenAPI document (`api/openapi.json`) and regenerate `src/shared/api/generated.ts` (`openapi-typescript`).                                                             |
| `npm run contrast`                | WCAG contrast of every token pair in both themes; non-zero exit on failure.                                                                                                             |
| `npm run screenshots`             | Light/dark × 1440/390 px screenshots of every page into `docs/screenshots/`.                                                                                                            |

Playwright uses its own Chromium if present, otherwise the newest `chromium-*` under
`PLAYWRIGHT_BROWSERS_PATH` (default `/opt/pw-browsers`), or `PW_CHROMIUM_PATH` if set
(`scripts/chromium.mjs`).

## Architecture

Feature-sliced: a layer may import only from layers below it.

```
src/
  app/        App, QueryClient defaults, route table (every page and run tab is a lazy chunk), route error page
  pages/      RunsPage, RunDetailPage (+ tabs/: Performance, Risk, Alpha, Costs, Execution, Validation),
              NewRunPage, ModelsPage, LedgerPage, NotFoundPage
  widgets/    AppShell (nav rail + main), RunHeader (+ GateStrip), MetricTable
  features/   launch-run (form, validation, schema-driven overrides, job progress), theme-toggle
  entities/   run (query hooks, query keys, selectors), metric (definitions + interpretations), gate (definitions, verdict)
  shared/     api (schema, client, endpoints), ui (Page grid, Section, Annotated, MarginNote, Definition,
              SortableTable, TabNav, Button, SegmentedControl, Skeleton, Empty/ErrorState, VisuallyHidden),
              charts (ChartFrame, NavDrawdownChart and TimeSeriesChart on uPlot; BarChart, StackedBarChart,
              PointLineChart, SmallMultiples, MonthlyHeatmap in SVG/HTML), lib (format, stats, theme, motion,
              handbook links), styles (tokens.css, global.css)
  mocks/      MSW handlers, in-memory registry, deterministic fixture generator
  test/       Vitest setup and render helpers
e2e/          Playwright smoke + axe specs
scripts/      contrast check, screenshots, Chromium resolver
```

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
                         ↑ TanStack Query cache, keyed by runKeys / jobKeys
```

- **Query keys** are hierarchical (`entities/run/queries.ts`):
  `['runs','list']`, `['runs','detail',id]`, `['runs','detail',id,'series','nav,drawdown,net']`,
  `…,'costs'`, `…,'exposures'`, `…,'holdings',date`, `…,'ic'`, `['jobs',jobId]`, `['ledger']`,
  `['config-schema']`. Invalidating `['runs']` refreshes everything about runs.
- A finished run never changes, so its series, costs, exposures and IC use `staleTime: Infinity`.
- `useJob` polls `/api/jobs/{id}` every 750 ms until `done`/`failed`, then invalidates `['runs']`.
- 4xx answers are not retried (missing run, run still running → 409); transport and 5xx errors
  are retried twice. `ErrorState` turns `ApiError`s into direction ("Check that the qts API is
  running and that VITE_API_URL points at it").
- The run page fetches the run once and passes it to its tabs through the outlet context
  (`pages/run-detail/context.ts`); tabs fetch only the extra series they chart.

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

| Method | Path                                                                     | Returns                                                              |
| ------ | ------------------------------------------------------------------------ | -------------------------------------------------------------------- |
| GET    | `/api/runs`                                                              | `RunSummary[]` (headline is `null` while running or after a failure) |
| GET    | `/api/runs/{id}`                                                         | `RunDetail` (409 while running/failed, 404 if unknown)               |
| GET    | `/api/runs/{id}/series?names=nav,gross,net,drawdown,turnover,exante_vol` | columnar `{dates, series}`                                           |
| GET    | `/api/runs/{id}/costs`                                                   | daily `spread/impact/fees/borrow` USD, daily NAV, `cost_sensitivity` |
| GET    | `/api/runs/{id}/exposures`                                               | factor list, roles, columnar history, latest snapshot (or empty)     |
| GET    | `/api/runs/{id}/holdings?date=`                                          | positions on a month-end decision date + available dates             |
| GET    | `/api/runs/{id}/ic`                                                      | per-date rank IC + decay curve                                       |
| POST   | `/api/runs`                                                              | `{job_id, run_id}` (202; 422 with per-field `errors`)                |
| GET    | `/api/jobs/{job_id}`                                                     | `{status, progress 0..1, message}`                                   |
| DELETE | `/api/jobs/{job_id}`                                                     | cancel (best effort), returns the job                                |
| GET    | `/api/ledger`                                                            | trials + per-family trial counts                                     |
| GET    | `/api/config/schema`                                                     | JSON Schema of `SystemConfig` (drives the override fields)           |
| GET    | `/api/health`                                                            | liveness, runs root, job counts                                      |

**Older runs.** Run directories written before the pipeline stored its extra artifacts have
`null` for `execution_venue`, `config.stocks/days` (unless recoverable from the recorded CLI
command), per-holding `alpha/specific_vol/industry`, an all-`null` `exante_vol` series and an
empty exposures payload (`latest: null`). The Risk and Execution tabs show empty states or
"not recorded" text for these.

**Launch form.** The override fields are generated from `/api/config/schema`: every scalar of
every section (nested sections such as `alpha.lgbm` included), one collapsible group per
section, `enum` strings as selects, booleans as checkboxes, bounds validated client-side and
again (strictly) by the API.

**Mocks.** `src/main.tsx` starts the MSW service worker only when `VITE_USE_MOCKS=true`
(`npm run dev:mock`, the Vitest setup and the mock Playwright suite); in any other build the mock
code is never loaded (it is not in the production bundle). Handlers
(`src/mocks/handlers.ts`) match `*/api/...`, so they work whatever `VITE_API_URL` says. The
fixtures (`src/mocks/fixtures/generate.ts`) are deterministic and internally consistent: every
summary metric, gate and DSR is computed from the generated series (DSR with the same formula
as the handbook's worked example; a unit test checks it reproduces 0.666). Seed runs: the
post-audit phase-1 demo, a phase-2 ML run below its bar, a lag-0 run with a suspicious Sharpe,
a passing data-store run, one running and one failed run. Launched runs complete after 4 s.

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

**A page.** Create `pages/<name>/<Name>Page.tsx` rendering `<Page docTitle title>`; use
`Annotated` for content with a margin note, `Section` for titled parts, `Wide` for wide tables.
Register it in `app/routes.tsx` with `lazy: () => import(...).then(m => ({ Component: m.X }))`
and add it to `NAV` in `widgets/AppShell/AppShell.tsx` if it belongs in the rail. Add a line
to `e2e/a11y.spec.ts` and `scripts/screenshots.mjs`.

**A run tab.** Add `pages/run-detail/tabs/<Name>Tab.tsx` (read the run with `useRunOutlet()`),
a child route in `app/routes.tsx` and an entry in `TABS` in `RunDetailPage.tsx`.

**A chart.** Small, few-point charts: SVG with d3-scale in `shared/charts/`, sized with
`useMeasure()`, colours from `var(--color-*)`, a hover readout, an `aria-label` summary.
Long series (hundreds of points or more): build on `UplotChart` (see `TimeSeriesChart`) and
import it from its own file, not the `shared/charts` barrel, so uPlot stays in lazy chunks.
Wrap it in `ChartFrame` with a `table` (a function, so the data table is built only when opened).

## Quality gates (current)

- `npm run typecheck`, `npm run lint`: clean (strict TS, no `any`, `strictTypeChecked` rules).
- `npm test`: 57 tests (format, stats/DSR, metric interpretations, verdict generation,
  selectors, form validation, API client, RunHeader/GateStrip/MetricTable with MSW, Runs and
  New-run pages through the real route tree).
- `npm run test:e2e`: smoke (runs list → run → every tab, data-table toggle, suspicious run,
  launch → progress → run, skip link, theme persistence) and axe on 11 pages × 2 themes with
  zero serious or critical violations.
- `npm run contrast`: all 44 pairs pass (table in `DESIGN.md`).
