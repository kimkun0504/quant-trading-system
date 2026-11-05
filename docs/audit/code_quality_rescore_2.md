# `qts` engineering-quality re-score #2

Scope: same rubric as `docs/audit/code_quality.md` and `docs/audit/code_quality_rescore.md` (7 areas, 1–10 each, engineering only, not alpha quality), applied on 2026-11-05 to the working tree after the behaviour-preserving refactor wave logged in `docs/WORK_PLAN.md` (HEAD `570303d` plus uncommitted edits; nothing committed). Covers `qts/`, `api/`, `ui/`, `tests/`, CI and docs.

Machine: 2 vCPU Linux box, Python 3.13, pyright 1.1.414, ruff 0.16.9, Node 22. Unlike the first re-score, no other agent was editing the tree, so wall times are lower and are **not** directly comparable with that document's load-average-5 numbers.

---

## What changed (summary)

| Rescore #1 item | Change | Where |
|---|---|---|
| (d)1 split `DailyBarSimulator.run` | `run` packs its keyword arguments into a frozen `BacktestSpec` and calls `run_spec`, a 19-line loop over a fixed step pipeline: `_day` → `_universe_step` → `_target_step` → `_cost_step` → `_mark_to_market_step` (+ `_classify_missing`, `_close_halt_episodes`, `_liquidate_delisted`) → `_accounting_step` → `_result`. State in `_BookState`, events in `_EventLog`, arrays in `_Recorder`, run constants in a frozen `_RunContext`. `BacktestResult` moved to `result.py`, cost plumbing to `cost_adapter.py`. The pipeline builds a `BacktestSpec` instead of sniffing `inspect.signature(sim.run)`. | `qts/backtest/{simulator,spec,result,cost_adapter}.py`, `qts/pipeline.py` |
| (d)2 break up `optimizer.py`, narrow the catch, gate failed solves | 1,658-line module → package of 9 modules (`constants`, `result`, `problem`, `dpp`, `constraints`, `compiled`, `sizing`, `solvers`, `spo`); every old import path still works. `solve` is now a 57-line orchestration of named steps. The solver chain catches only `cvxpy.error.SolverError`, `ArithmeticError` and `numpy.linalg.LinAlgError`; anything else propagates. Caught errors go to `OptimizationResult.solver_errors`; the simulator lists failed solves in `diagnostics["failed_solves"]` and counts `n_rebalance_solves`; the new `failed_solves` hurdle gate FAILs above `ValidationConfig.max_failed_solve_frac` (default 2 %). | `qts/portfolio/optimizer/`, `qts/validation/hurdles.py`, `tests/test_solver_failures.py` |
| (d)4 freeze / validate / type | `frozen=True` on every config dataclass (`with_overrides` rebuilds via `dataclasses.replace`, so section validation now also runs on overrides; all in-place mutations in `qts/`, `api/`, `scripts/`, `tests/` converted), on `RiskModelOutput` and `AlphaOutput` (with `__post_init__` structural checks), `OptimizationResult`, `GateResult`, `HurdleResult`, `ICReport`. TypedDicts `BacktestSummary`, `SizingSummary`, `MZResult`, `SurvivorshipReport`, `CalibrationDiagnostics`; `calibrate_b` returns the NamedTuple `ImpactCalibration` (still unpacks as `(b, diag)`). | `qts/core/{config,types}.py`, `qts/backtest/result.py`, `qts/risk/evaluation.py`, `qts/data/universe.py`, `qts/tcost/tca.py` |
| (d)4 delete compat layers without callers | Removed `crosssection.winsorize` / `rank_ic` / `ic_series`, `alpha/targets.py:industry_dummies`, `risk/exposures.py:industry_dummies`, `optimizer._parametrize`, the duplicate `SUSPECT_SHARPE` in `hurdles.py`, and the pipeline's `eligibility=` / no-`signal_lag_days` fallbacks. Kept (they still have callers): `_LegacyCostModel` (the hand-computed fixture's cost double), `LiquiditySnapshot` dict access, `ExposureHistory(dict)`, dict-style access on `HurdleResult` / `ICReport`. | as listed |
| (beyond the brief) long functions | `HurdleReport.evaluate_signal` (207 lines) → one method per gate (34-line orchestrator); `ResearchPipeline.evaluate` (175) → `_signal_quality`, `_fold_sharpes`, `_cost_gates`, `_selection_stats`, `_notes` (64-line orchestrator); run registry and report card moved from `pipeline.py` to `qts/runs/registry.py` and `qts/runs/report.py` (re-exported). | `qts/validation/hurdles.py`, `qts/pipeline.py`, `qts/runs/` |

Contract changes that follow from the gate: `ValidationConfig` gained a field, so **every config hash changes** (`configs/default.yaml` regenerated); the API `GateName` literal, `api/openapi.json`, `ui/src/shared/api/{generated,schema}.ts`, the UI gate definitions and mock fixtures gained `failed_solves`. `tests/test_golden.py` compares the golden file's gate set and pins the new gate separately (`GATES_NOT_IN_GOLDEN`); `tests/golden/pipeline_summary.json` is unchanged.

### How "behaviour-preserving" was checked
1. The golden test (rtol 1e-6) and the hand-computed simulator fixture (1e-10) stayed green after every step, with the golden file untouched.
2. A stronger check outside the repo: full `BacktestResult` panels (NAV, returns, weights, trades, costs, ex-ante variance, turnover, transfer coefficient), the complete diagnostics (solve log with notes and binding counts) and the hurdle result were pickled for three configurations (golden; `signal_lag_days=0` + fixed γ + projection; fixed γ + constraint mode + `rebalance_every=2`) before the refactor and compared after each step with `DataFrame.equals` / exact `repr` equality. Result after the final step: **bitwise identical**, apart from the two new diagnostics keys and the new gate.
3. The rewritten `evaluate_signal` was compared with the HEAD version on 4,000 random input combinations (NaN / inf / boundary values, skipped and required gates): identical `to_dict()` output.

---

## (c) Tool counts (re-run)

| Check | Rescore #1 | Now | Notes |
|---|---|---|---|
| `ruff check qts tests api scripts` | 0 | **0** | same rule set |
| `ruff format --check` | 121 files | **134 files already formatted** | |
| `pyright qts/core` (strict) | 0 / 0 | **0 errors, 0 warnings** | CI-blocking |
| `pyright qts/backtest` (basic) | `simulator.py` alone did not finish in 200 s | **finishes in 8 s; 1 error** (pre-existing, `metrics.py:22`) | loop-carried state is gone from one 365-line function |
| `pyright qts/portfolio/optimizer` | 29 errors in `optimizer.py` | **0** | the rest of `qts/portfolio` keeps 12 pre-existing errors (`diagnostics.py`, `alpha_scaling.py`) |
| `pyright qts` (basic, whole package) | did not finish (killed at 580–600 s) | **finishes in 58 s: 275 errors, 1 warning** (rescore #1 measured 346 errors package by package with `simulator.py` excluded) | still a non-blocking CI step |
| `pytest --collect-only` | 373 (344 fast + 29 slow) | **387 (358 fast + 29 slow)** | +14: `tests/test_solver_failures.py` |
| Fast tier `pytest -m "not slow"` | 344 passed, 444 s under load | **358 passed in 238 s** (unloaded) | |
| Slow tier, file by file | not run | **all 29 passed, every file exit 0** (alpha 285 s, backtest 69 s, determinism 45 s, optimizer_perf 196 s, pipeline 442 s, risk 9 s, risk_statistical 33 s, tcost 2 s) | |
| API tests | 90 collected | **90 passed (48 s)** | OpenAPI regenerated; `python -m qts_api.openapi` output equals `api/openapi.json` |
| UI `typecheck` / `lint` / `format:check` / `vitest` | pass / pass / pass / 61 | **pass / pass / pass / 61 passed** | after the `failed_solves` contract change |
| Functions > 100 lines in `qts/` | 12 | **6** | longest now: `sharadar_sf1_to_pit_fundamentals` 179, `FillSimulator.simulate` 175, `synthetic.generate` 162, `calibrate_alignment_nu` 131, `router.route` 125, `sizing.gamma_search` 107 (32 of them docstring) |
| Public docstring coverage (`qts/`) | ~60 % (368/615) | **61 % (397/653)** | |

Key function sizes, before → after: `DailyBarSimulator.run` 365 → 44 (`run_spec` 28, largest step `_target_step` 66); `SinglePeriodOptimizer.solve` 224 → 90 (57 without the docstring); `_Compiled.__init__` 128 → 54; `_gamma_search` 148 → 107; `HurdleReport.evaluate_signal` 207 → 34; `ResearchPipeline.evaluate` 175 → 64. Module sizes: `optimizer.py` 1,658 → largest module `spo.py` 718 (others ≤ 400); `pipeline.py` 1,193 → 1,013; `simulator.py` 800 → 902 lines (longer module docstring and step methods) plus `result.py` 213, `spec.py` 105, `cost_adapter.py` 153.

---

## (a) Scorecard

| Area | #1 | Now | One-line justification |
|---|---|---|---|
| 1. Architecture & boundaries | 7 | **8** | The two named monoliths are broken up along real seams: the optimizer is a package whose modules form an acyclic chain (`constants` ← `problem`/`dpp`/`result` ← `constraints` ← `compiled` ← `sizing` ← `spo`), the simulator separates spec, result, cost plumbing and the step pipeline, and the pipeline lost its run registry and report card to `qts.runs` and no longer sniffs the simulator's signature. Not 9: `pipeline.py` is still ~1,000 lines of orchestration plus evaluation, and four compat layers survive because they still have callers (`_LegacyCostModel`, `LiquiditySnapshot` dict access, `ExposureHistory(dict)`, dict access on `HurdleResult`/`ICReport`). |
| 2. PIT safety by construction | 7 | **7** | Unchanged: no `LabelPanel`/`AsOf` type, `FEATURE_REGISTRY` empty at runtime, the simulator still passes bare timestamps to `alpha_fn`/`risk_fn`. |
| 3. API design | 6 | **8** | Every item rescore #1 listed under this area is addressed: `RiskModelOutput`/`AlphaOutput` frozen with structural `__post_init__` checks; configs frozen (overrides rebuild and re-validate, so the old "mutate in place to bypass validation" path is gone); `run` takes a validated `BacktestSpec` instead of 14 loose parameters; `summary()`, `mincer_zarnowitz`, `survivorship_check` and `calibrate_b` have typed results; the solver catch is narrow and failures are surfaced (`solver_errors`, `diagnostics["failed_solves"]`, a gate). Remaining: `BacktestResult` is mutable and its `diagnostics` is `dict[str, object]`; the summary is a TypedDict rather than a frozen object; dict-style legacy access persists; `factor_groups` is not `Literal`-keyed and is still read with `getattr(..., None)`. |
| 4. Performance & scalability | 6 | **6** | Unchanged by design (behaviour-preserving wave): same solves per day, no stage cache, the 1,000 × 15 y < 30 min target is still not demonstrated. |
| 5. Testing | 8 | **8** | +14 tests for failure handling (narrow catch, error recording, propagation of programming errors, simulator diagnostics, gate thresholds and round-trip); the refactor was verified bitwise, not only within tolerances. Short of 9 for the same reasons as before: no coverage gate, no `pytest-xdist`, `integration` marker unused, seeded-threshold tests dominate the slowest list. |
| 6. Reproducibility & ops | 8 | **8** | Config immutability closed one listed gap; `pyright` can now analyse the backtest package in seconds. Still: no Python lock file, no stage cache, the basic `pyright qts` step is non-blocking with no baseline. |
| 7. Readability | 6 | **8** | The regression called out in rescore #1 is reversed: the three central long functions are now short orchestrators over named steps (`run` 365 → 44, `solve` 224 → 90, `evaluate_signal` 207 → 34) and functions over 100 lines went from 12 to 6. Not 9: `_Parametrizer`'s expression rewriting is still clever and hard to maintain, six long functions remain in loaders / fills / synthetic / routing, and docstring coverage is 61 %. |

**Overall: 8/10** (mean 7.57: 8 / 7 / 8 / 6 / 8 / 8 / 8). Rescore #1 rounded a mean of 6.9 to 7; by the same convention 7.57 rounds to 8. It is a narrow 8: one point less in any area would round down to 7, and areas 2 and 4 are where the remaining weight sits.

---

## (b) Status of the rescore #1 "top 5 to reach 9/10"

| # | Item | Status |
|---|---|---|
| 1 | Split `DailyBarSimulator.run` into a step function, `BacktestSpec` | **Done** (methods on the simulator rather than free functions in `step.py`; `BacktestSpec` is not yet written into `RunManifest`, only `describe()` exists) |
| 2 | Break up `optimizer.py`, narrow the catch, failed-solve gate | **Done** (`max_failed_solve_frac` in `ValidationConfig`, default 0.02) |
| 3 | Structural PIT (`LabelPanel`, registry, `md.as_of`) | **Not done** |
| 4 | Freeze and validate contracts, type dict returns, remove compat layers | **Mostly done** (configs, contracts, results frozen; typed returns; shims without callers removed; four compat layers with callers remain; `BacktestResult` mutable) |
| 5 | Stage cache, solve-count reduction, blocking pyright with baseline, xdist, lock file, UI budget | **Not done** (only side effect: `pyright` on the backtest package now finishes) |

---

## (d) Remaining gaps, in priority order

1. **PIT by construction** (area 2): `qts/core/pit.py` with a frozen `LabelPanel(values, horizon, lag)`; trainer purge/embargo derived from it; one `@feature_registry` function per feature panel with a non-empty-registry assertion in `tests/test_pit_properties.py`; `md.as_of(t)` views passed to `alpha_fn` / `risk_fn`.
2. **Performance at target scale** (area 4): stage cache keyed by data hash + config sub-hash; γ warm start and a solve cap once bracketed; `scripts/bench_backtest.py` at 1,000 names with a regression check in the nightly job.
3. **Type gate** (area 6): with the simulator no longer blocking analysis, record a `pyright qts` baseline and make the CI step fail on new errors; burn down `alpha/models.py` and `validation/`; add a lock file.
4. **Remaining compat layers** (areas 1 and 3): migrate the hand-computed fixture's cost double to the `CostModel` protocol and delete `_LegacyCostModel` / `snapshot_from_market` on `TransactionCostModel`; drop dict access on `LiquiditySnapshot`, `HurdleResult`, `ICReport` once the API and tests use attributes; make `ExposureHistory` a plain class; type `BacktestResult.diagnostics` (TypedDict) and freeze `BacktestResult`.
5. **`pipeline.py`** (area 1): move evaluation (`evaluate` and its helpers, ~300 lines) into `qts/validation/` or a `qts/pipeline/` package so orchestration stands alone.

## (e) Verdict

**Yes, narrowly. On the same rubric the codebase now scores 8/10 (mean 7.57), meeting the WORK_PLAN Phase-1 exit criterion of ≥ 8.** The gain comes from architecture (+1), API design (+2) and readability (+2), all through refactors verified to leave every backtest number bitwise unchanged. PIT-by-construction and performance at scale are untouched and are what stands between this and 9. The other Phase-1 exit conditions (second audit, suite green) were not re-checked beyond the test runs recorded above.
