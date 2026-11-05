# Work plan — hardening, documentation, UI (started 2026-11-04)

Owner request: keep improving the models (optimise, verify the math), write detailed
Notion documentation aimed at a CS graduate who is not a quant, then — once the
models are solid — build a UI in a separate folder following strong frontend
architecture. Work runs unattended; this file is the source of truth for scope
and order. Every milestone ends with: full test suite green → commit (no AI
co-author trailers) → push to `main-clean` → status line appended below.

## Ground rules
- Commits authored by Kun Kim with **no `Co-Authored-By`/`Claude-Session` trailers**.
- Branch `main-clean` carries the rewritten history; `main` on GitHub still has the
  old trailers until the owner runs the swap command (force-push was blocked by the
  safety classifier and is the owner's call).
- No change merges without tests; math changes need a numerical check against a
  hand computation or reference implementation.
- Don't widen scope: no new alpha signals, no real-data purchases, no live trading.

## Phase 1 — Correctness and engineering hardening
Exit criteria: suite green; second independent math audit finds no CRITICAL/HIGH;
code-quality score ≥ 8/10 on the same rubric as `docs/audit/code_quality.md`.

| # | Work item | Detail |
|---|---|---|
| 1.1 | Test foundation | hand-computed 3-stock × 5-day simulator fixture; golden-file regression of pipeline summary (tolerances); determinism tests; property-based PIT tests (hypothesis) over every registered feature, universe mask, cost snapshot, risk `as_of`; remove wall-clock asserts; seed-robust thresholds |
| 1.2 | One cost contract | `LiquiditySnapshot`, `CostEstimate` dataclasses + `CostModel` protocol in `qts/core`; `TransactionCostModel` is the only implementation (`SimpleCostModel` → thin test double); single snapshot builder |
| 1.3 | De-duplication | one winsorize / rank-IC / rolling beta / industry dummies / Newey–West t / auction share / suspect-Sharpe constant, all in `qts/core` |
| 1.4 | Config + reproducibility | every behavioural knob in `SystemConfig` (frozen, `from_dict`/`from_yaml`/`to_yaml`, hash); generated `configs/default.yaml`; `RunManifest` (git sha, data hash, config hash, library versions, timings) saved with every run |
| 1.5 | Typed results | dataclasses for `ic_report`, hurdle results, factor-return fits, calibration diagnostics |
| 1.6 | Performance | DPP-parametrised cvxpy problem reused across days; exposures as a `(T,N,K)` tensor; vectorised fee/impact loops; vectorised LightGBM rank-IC metric; benchmark script; target composite run at 1,000 names × 15 yrs < 30 min on a laptop |
| 1.7 | Tooling | ruff + pyright config (strict on `qts/core`), pre-commit, GitHub Actions CI (lint, type, tests), Makefile, `py.typed` |
| 1.8 | Open math items | lag the synthetic planted score (oracle fairness); run `calibrate_alignment_nu` on synthetic and set a default; re-check every docstring formula against code |
| 1.9 | Second audit | fresh reviewers re-audit math + leakage + accounting after the refactor; fix findings |

## Phase 2 — Notion documentation
Audience: CS degree, not a quant. Each page covers *what it is, why this choice over
alternatives, how the algorithm works step by step, the math with every symbol
defined, parameters and their defaults, failure modes, and references*. A repo
mirror lives in `docs/handbook/`.

Page tree (hub: "Quant Trading System — Handbook"):
1. Start here — what the system does, the five-model pipeline, glossary of trading basics
2. Architecture & data flow — layers, contracts, point-in-time discipline, run lifecycle
3. Data & universe — sources, survivorship/look-ahead bias, PIT store, universe rules, corporate actions
4. Alpha model — features (each indicator explained), targets & residualisation, rank composite, ML models (GBDT, MLP, ridge), walk-forward training & embargo, Grinold scaling, decay & half-life
5. Risk model — factor models, Barra-style exposures, cross-sectional regression, EWMA/Newey–West covariance, VRA, eigenfactor adjustment, specific risk, shrinkage (Ledoit–Wolf/QIS), APCA hybrid, alignment factor
6. Transaction cost model — explicit fees, spread, square-root impact, auction vs continuous, borrow, persistence, TCA and calibration
7. Portfolio construction — mean-variance, convex optimisation, constraints, transfer coefficient, Gârleanu–Pedersen, vol targeting
8. Execution — closing auction, order types, broker algos, VWAP/POV/Almgren–Chriss, routing table
9. Backtesting — simulator timeline, accounting identity, costs, delistings, common backtest biases
10. Validation & statistics — Sharpe (and Lo adjustment), IC/ICIR, t-stats & Newey–West, drawdown, Calmar/Sortino, deflated Sharpe, PBO, purged CV/CPCV, multiple testing, hurdle gates
11. Concepts glossary — every mathematical/financial term used, alphabetical, one-paragraph each
12. Operating guide — install, run demo, load real data, read the report card, extend a model
13. Audit history & known limitations

## Phase 3 — UI (only after Phase 1 exit criteria)
Separate folders: `api/` (Python FastAPI service over `qts`) and `ui/` (web app).

| # | Work item | Detail |
|---|---|---|
| 3.1 | Design direction | load the frontend-design skill; define tokens (type scale, colour, spacing, motion), dark/light, data-dense "research terminal" aesthetic; written design brief in `ui/DESIGN.md` |
| 3.2 | API | FastAPI + Pydantic schemas; run registry backed by `runs/` (manifest + Parquet artifacts); async job runner for backtests (synthetic or store); endpoints for runs, NAV/drawdown, exposures, costs, alpha snapshot (top/bottom names, theme scores), risk snapshot (factor vols, correlations, missing-factor monitor), hurdles, ledger, config schema; OpenAPI spec; pytest |
| 3.3 | Frontend architecture | React + TypeScript (strict) + Vite; TanStack Query for server state; React Router; generated typed API client from OpenAPI; feature-sliced folders; charting library chosen for large time series; CSS via design tokens; ESLint + Prettier; Vitest + Testing Library; Playwright smoke |
| 3.4 | Screens | Overview dashboard; Runs list; Run detail (Performance, Risk, Alpha, Costs, Execution, Validation tabs); Model explorer (risk factors, alpha themes); New run (config form generated from schema, live job progress); Trial ledger |
| 3.5 | Quality | accessibility (keyboard, contrast, ARIA), responsive layout, empty/loading/error states, performance budget, screenshots reviewed visually |
| 3.6 | Docs | `ui/README.md`, `api/README.md`, Notion page for the UI |

## Status log
- 2026-11-04 22:30 — history rewritten without AI trailers, pushed to `main-clean`; plan written.
- 2026-11-05 00:45 — Wave 1 done: test foundation (hand-computed simulator fixture, golden file, determinism, hypothesis PIT properties; 168 tests), tooling (ruff/pyright/pre-commit/CI/Makefile), synthetic oracle lagged (alpha_lag_days=1), alignment nu calibrated (default 0.25), 11 handbook pages drafted, 46 findings logged in docs/notes/doc_review_findings.md.
- 2026-11-05 — Wave 2 (refactor) mostly landed: typed cost contract (qts/core/costs.py), routed venue, b_auction 0.3, signal_lag_days, vectorised costs (15–30×); store upsert (data-loss fix), raw prices/shares/dated industry on MarketData, qts/core/stats.py de-dup; eigen-cache and APCA factor-count look-ahead fixed, eigen inflation a=1.2, projection with cost term, ExposureHistory tensor (~6× less memory); optimizer daily-solve speed-up only 1.2–1.5× (solver-bound; DPP compile O(N²) memory) — documented. Handbook 01–11 published to Notion. Session restart interrupted the alpha/pipeline/config agent; resumed.
- Decision: start the UI frontend foundation now against a mock API (ui/ is isolated from Python); the real API (api/) is built after Phase 1 exit so it matches the final run registry.
- 2026-11-05 — Lag investigation: the 0.96→0.11 lag gap was path noise from an uncommitted intermediate tree; found and fixed a future-universe look-ahead in risk style standardisation and a lag≥2 crash; default `signal_lag_days=1` (MOC cutoff 15:50). API (api/) + UI wired end to end. Second independent audit (1.9): no CRITICAL; HIGH items = sizing via alpha-blind projection (γ=10 too high), halt/delisting label rule, ledger key/concurrency. Fix wave 3 running.
- 2026-11-05 — Code-quality refactor wave (behaviour-preserving; golden file unchanged, full backtest panels bitwise identical on three configurations): `DailyBarSimulator.run` split into a per-day step pipeline (`_day` → `_universe_step` → `_target_step` → `_cost_step` → `_mark_to_market_step` → `_accounting_step`) with a frozen `qts/backtest/spec.py::BacktestSpec` (`run(...)` kwargs still accepted; new `run_spec`); `BacktestResult` → `qts/backtest/result.py`, cost plumbing → `qts/backtest/cost_adapter.py`. `qts/portfolio/optimizer.py` (1,658 lines) → package `qts/portfolio/optimizer/` (`constants`, `result`, `problem`, `dpp`, `constraints`, `compiled`, `sizing`, `solvers`, `spo`; old imports still work). Solver chain catches only `SolverError` / numerical errors (others propagate), records them in `OptimizationResult.solver_errors`; simulator lists failures in `diagnostics["failed_solves"]` and counts `n_rebalance_solves`; new `failed_solves` hurdle gate (`ValidationConfig.max_failed_solve_frac = 0.02`; config hash changes, `configs/default.yaml`, `api/openapi.json`, UI contract regenerated). Frozen config dataclasses (`with_overrides` rebuilds via `dataclasses.replace` and re-validates; all in-place mutations in `qts/`, `api/`, `scripts/`, `tests/` converted), frozen `RiskModelOutput` / `AlphaOutput` (with `__post_init__` checks), `OptimizationResult`, `GateResult` / `HurdleResult`, `ICReport`; TypedDicts for `BacktestResult.summary()`, `mincer_zarnowitz`, `survivorship_check`, NamedTuple `TCALedger.calibrate_b -> ImpactCalibration`; run registry / report card moved from `pipeline.py` to `qts/runs/registry.py` / `report.py` (re-exported) and `ResearchPipeline.evaluate` / `HurdleReport.evaluate_signal` split into steps; uncalled shims removed (`crosssection.winsorize/rank_ic/ic_series`, both `industry_dummies` wrappers, `_parametrize`, duplicate `SUSPECT_SHARPE`, the pipeline's signature-sniffing of `sim.run`). Re-score: `docs/audit/code_quality_rescore_2.md`.
