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
