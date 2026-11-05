# `qts` engineering-quality re-score

Scope: same rubric as `docs/audit/code_quality.md` (7 areas, 1–10 each, engineering only, not alpha quality), applied to the working tree on 2026-11-05 between 09:45 and 10:35 EST (HEAD `570303d` plus uncommitted edits). Covers `qts/`, `api/`, `ui/`, `tests/`, CI and docs. No source files were changed. Another agent was making small polish edits at the same time (CLI, simulator, fills, CI, UI copy). Where that affected a measurement, the note says so.

Machine: 2 vCPU Linux box, Python 3.13, pyright 1.1.414, ruff 0.16.9, Node 22. The first review ran pytest on 4 threads, so wall times are **not** directly comparable.

---

## (c) Tool counts (re-run)

| Check | First review | Now | Notes |
|---|---|---|---|
| `ruff check qts tests api scripts --statistics` | 1,223 findings (narrower rule set: E,F,W,B,I,UP) | **0** | Wider rule set now (`E,F,W,B,I,UP,SIM,NPY,PD,C4,RUF`). Ignores are justified inline in `pyproject.toml`. E501 is ignored because the formatter owns line length. `api/` uses its own `api/pyproject.toml` ruff config. |
| `ruff format --check qts tests api scripts` | not run | **121 files already formatted** | |
| `pyright qts/core` (strict) | not configured | **0 errors, 0 warnings** (13 s) | Strict mode is set via `[tool.pyright] strict = ["qts/core"]`. CI blocks on it. |
| `pyright qts` (basic), 10-min box | 323 errors / 55 files | **Did not finish.** Killed at 600 s with pytest also running, and again at 580 s when run alone. | Root cause: `pyright qts/backtest/simulator.py` alone does not finish in 200 s, while every other package takes 10–40 s. See below. |
| `pyright` per package, excluding `simulator.py` | — | **346 errors** | alpha 83, risk 86, validation 49, portfolio 41, data 28, pipeline+cli 24, tcost 20, execution 14, backtest/metrics 1, runs 0, core 0. By rule: `reportArgumentType` 230, `reportCallIssue` 31, `reportOperatorIssue` 29, `reportAttributeAccessIssue` 28, `reportOptionalMemberAccess` 14, `reportReturnType` 9. Hot files: `alpha/models.py` 46, `validation/hurdles.py` 40, `portfolio/optimizer.py` 29, `pipeline.py` 24, `risk/statistical.py` 21. |
| `pytest --collect-only` | 100 | **373** (344 fast + 29 `slow`; 1 `statistical`; the `integration` marker is registered but used by 0 tests) | |
| Fast tier `pytest -m "not slow"` | (full suite 79 s on 4 threads) | **344 passed in 444 s** on 2 vCPU with load average ~5 from the other agent. A first run at 323 s showed 1 failure, `test_feature_registry_refuses_negative_shift`. The other agent rewrote `tests/test_validation.py` 24 s into that run (`inspect.getsource` then read the new file), and the test passes in isolation, so this was not a real flake. | Slowest: `test_random_portfolio_bias_statistic` 31–45 s, golden fixture setup 29–35 s, `test_run_with_tcost_model_venue_and_persistence` 17–24 s, `test_cli_demo_writes_loadable_run_dir` 12–22 s. No `pytest-xdist`. |
| API tests (`api/`) | n/a | **90 collected** | Run in a new CI `api` job. |
| UI `npm run typecheck` (`tsc -b`) | n/a | **pass** (7 s) | |
| UI `npm run lint` (`eslint --max-warnings 0`) | n/a | **pass** on the final run. At 10:05 it briefly failed with 2 `react-hooks/preserve-manual-memoization` errors in `LaunchRunForm.tsx:33-34`, while the other agent was editing that file. It is clean as of 10:32. | `format:check` passes too. |
| UI `vitest run` | n/a | **61 passed / 13 files** (44 s; 53% of the time is jsdom set-up) | Plus Playwright smoke and axe specs in `ui/e2e/` (not run here). |
| UI `npm run build` | n/a | **pass**. JS 553 kB raw / **182 kB gzip** in total; CSS 6.8 kB gzip | Largest chunks: `react` 219 kB (68 kB gz), `router` 95 kB (31 kB gz), `charts` 67 kB (22 kB gz), `TimeSeriesChart` 55 kB (24 kB gz). Routes are code-split (each tab is 2–12 kB). No budget is written down or enforced. |

---

## (a) Scorecard

| Area | Old | New | One-line justification |
|---|---|---|---|
| 1. Architecture & boundaries | 6 | **7** | There is now one typed cost contract (`qts/core/costs.py`), factor roles live in the contract (`RiskModelOutput.factor_groups`, `qts/core/roles.py`), the helpers are de-duplicated into `qts/core/stats.py`, and the API is a separate package over `runs/`. But `portfolio/optimizer.py` (1,634 lines) and `pipeline.py` (1,116 lines: orchestration, evaluation, persistence and formatting) are monoliths, and compatibility shims keep second code paths alive (`_LegacyCostModel`, dict-access on `LiquiditySnapshot`, `ExposureHistory(dict)`). |
| 2. PIT safety by construction | 5 | **7** | Hypothesis PIT properties now cover features, the universe mask, the cost snapshot, risk `as_of`, alpha `as_of` and labels (`tests/test_pit_properties.py`). The label fallback is a hard fail with risk-only fallback styles. Two look-aheads were found and fixed (eigen-cache order, Bai–Ng `k`, future-universe standardisation). Still convention, not construction: there is no `LabelPanel`/`AsOf` type, `FEATURE_REGISTRY` is empty in production, and models are still fitted on the full panel and indexed by date. |
| 3. API design | 5 | **6** | `LiquiditySnapshot`/`CostEstimate` are frozen and validated, and `IcReport`, `IncrementalICReport`, `HurdleReport`, `RunManifest` and `RunRecord` are typed. But `RiskModelOutput`, `AlphaOutput` and `MarketData` are still mutable with no `__post_init__` checks. `BacktestResult.summary()` is still a 30-key dict. `calibrate_b` still returns a tuple; `mincer_zarnowitz` and `survivorship_check` still return dicts. `DailyBarSimulator.run` takes 14 parameters. `_try_solvers` still catches `(cp.SolverError, Exception)`. |
| 4. Performance & scalability | 5 | **6** | Costs are vectorised (15–30×), the LightGBM rank-IC metric is vectorised (`DateGroupedRankIC`), exposures are a `(T,N,K)` tensor with O(log T) `as_of`, and half-life is cached per block. But the daily loop is slower: about 0.5–0.57 s/day at 300 names in today's `runs/*/manifest.json` (backtest 507–572 s for ~1,005 days), against 0.26 s/day in the first review. This is caused by the γ-search's extra solves; machines differ. DPP reuse is off above 300 names, there is no stage cache, and the 1,000 × 15y < 30 min target is not demonstrated (`docs/handbook/13_audit_history.md:189`). |
| 5. Testing | 6 | **8** | 100 → 373 Python tests, plus 90 API, 61 Vitest and Playwright/axe. There is a golden file (`tests/golden/pipeline_summary.json`, rtol 1e-6), a hand-computed simulator fixture with borrow and delisting, a determinism suite including call-order independence, hypothesis PIT tests, cost-equivalence tests, ledger and store concurrency tests, wall-clock asserts turned into printed benchmarks, and a multi-seed majority rule for statistical thresholds. Short of 9 because the fast tier takes 5–7 min single-process, coverage is not gated, `integration` is unused, and seeded-threshold tests still dominate the slowest list. |
| 6. Reproducibility & ops | 3 | **8** | Strict `from_dict`/`to_yaml`/`hash()` config with a generated `configs/default.yaml` and a drift test. `RunManifest` records git sha and dirty flag, data hash, config hash, versions, seeds and timings. Run directories hold Parquet artefacts. The ledger keys on config hash and run spec, with a file lock. Also added: a `qts` CLI entry point, Makefile, pre-commit, `py.typed`, and CI for Python 3.10/3.12, nightly slow, `api` (with an OpenAPI drift check) and `ui` (typecheck, lint, format, vitest, e2e+axe). Gaps: config is mutable (`apply_overrides` uses `setattr`), there is no Python lock file, there is no caching, and the basic pyright step is non-blocking and cannot finish. |
| 7. Readability | 6 | **6** | Fixed: semicolon chains, `if True:` and dead `NotImplementedError` are gone; ruff is clean under a strict rule set; constants are named; docs and comments are candid. But the long functions got **longer**: `DailyBarSimulator.run` went from 133 to **365 lines (307 code lines)**, `SinglePeriodOptimizer.solve` from 201 to 224, `HurdleReport.evaluate_signal` from 93 to 207. `_gamma_search` is 148 lines and `_Compiled.__init__` 128. Twelve functions exceed 100 lines. `_Parametrizer` rewrites cvxpy expression trees into Parameters, which is clever but hard to maintain. Public docstring coverage is about 60% (368/615). |

**Overall: 7/10** (mean 6.9). Up from 5/10. The production path is now tested, reproducible and typed at the cost boundary. It is not yet at 8: the two central modules (simulator loop, optimizer) are larger and harder to read than at the first review, basic type-checking of `qts` cannot complete, and the scale target is unmet.

---

## (b) Status of the original top-10 refactors

| # | Refactor | Status | Evidence |
|---|---|---|---|
| 1 | One cost contract, typed, single snapshot | **Done (with residue)** | `qts/core/costs.py` has `LiquiditySnapshot` (frozen, finite, aligned), `CostEstimate` (fixed columns; non-finite raises) and the `CostModel` protocol. `portfolio/costs.py:snapshot_from_market` delegates to `TransactionCostModel.snapshot`, and `SimpleCostModel` is documented as a test double. The simulator's old silent fallbacks are gone (`backtest/simulator.py:270-293`): an unrecognised column now raises. Residue: `_LegacyCostModel` (`simulator.py:297`) and dict-compat on `LiquiditySnapshot` (`__getitem__`/`keys`/`from_mapping`) keep a second input shape alive. `SimpleCostModel` is still exported from `qts.portfolio`. `portfolio/costs.py:34 SNAPSHOT_KEYS` survives for dict snapshots. |
| 2 | Factor roles in `RiskModelOutput`; no string matching | **Done** | `core/types.py:212` `factor_groups: dict[str, list[str]]`; `risk/model.py:47 factor_groups_for`; `optimizer.py:708 _resolve_factor_roles` uses the groups, picks beta by exact name (`cfg.beta_factor_name`), and raises on unknown `neutral_styles`. The prefix fallback is only used when `factor_groups` is empty, and then logs a WARNING and adds a note. Not a `Literal`-typed mapping, and it is read via `getattr(risk, "factor_groups", None)`. |
| 3 | Hard-fail label residualiser; risk-only fallback | **Done** | `pipeline.py:365 _exposures_fn` returns `None` only before the risk model's first exposure date; otherwise it re-raises. Alpha-theme styles are dropped. `qts/core/roles.py` is the single source for `RISK_ONLY_STYLES`/`RISK_ONLY_FEATURE_PROXIES`/`ALPHA_THEME_FEATURES`. Covered by `tests/test_pipeline.py::test_pipeline_neutralises_exactly_the_label_set`. |
| 4 | `LabelPanel` + `AsOf` view; register every feature; PIT property tests | **Partial** | Done: `tests/test_pit_properties.py` (hypothesis over features, universe, cost snapshot, risk `as_of`, alpha `as_of`, labels-vs-features) and `tests/test_label_halts.py`. Not done: no `qts/core/pit.py`, no `LabelPanel`/`AsOf` (grep finds 0 hits), `WalkForwardTrainer.run` still takes a plain frame plus a horizon from config, and `@feature_registry` is applied only in `tests/test_validation.py`, so `FEATURE_REGISTRY` is empty at runtime. The simulator still passes bare timestamps to `alpha_fn`/`risk_fn`. |
| 5 | Every knob in config, serialised; `RunManifest` | **Done (mostly)** | `core/config.py:26 _ConfigBase` (strict `from_dict`, `to_yaml`, `hash()`, `CONFIG_SCHEMA_VERSION`); `tests/test_config.py:102` checks `configs/default.yaml` for drift; `core/manifest.py:171 RunManifest`; `pipeline.py:974 save_run`. Example: `runs/20261105T143113426781Z_fab1b3b6/{manifest.json, config.yaml, summary.json, *.parquet}`. The ledger keys on `config_hash` + `run_spec` (`pipeline.py:567`). Gaps: config dataclasses are not frozen (`apply_overrides` mutates a deepcopy). `BarraStyleRiskModel(eigen_refresh_days, seed, constraint_weights, weight_cap_quantile)`, `DailyBarSimulator(adv_window, vol_window)` and `FillSimulator(default_max_pct_vol)` are still constructor kwargs outside config. `ResearchPipeline(rebalance_every, venue, nav0)` are kwargs, but they are captured in `run_spec`. |
| 6 | Decompose `solve`; parametrised cvxpy problem | **Partial** | DPP machinery exists (`optimizer.py:359 _Parametrizer`, `:471 _Compiled`, `reuse_problem`), and the γ-search re-parametrises one compiled problem. But reuse is **off by default** because the DPP compile needs O(N²) memory (0.8 GB at N=300, >7 GB at N=1,000; handbook §9), and the measured speed-up is 1.2–1.5×. `solve` is not decomposed: still one 224-line method in a 1,634-line file, with no `optimizer/` subpackage. `_try_solvers` (`optimizer.py:1625`) still catches `(cp.SolverError, Exception)`, so a programming error becomes "try next solver / hold". |
| 7 | Exposures as a `(T,N,K)` tensor | **Done** | `risk/exposures.py:217-360 ExposureHistory`: static block plus `(T,N,S)` styles, `(T,N)` valid mask, dated industry codes, `as_of` via `searchsorted`, and `memory_report()`. It subclasses `dict` for back-compat, which is a smell. |
| 8 | On-disk stage cache keyed by data+config hash | **Not done** | No `core/cache.py` or stage cache anywhere (`grep stage_cache` finds nothing). Every run refits features, risk and ML. |
| 9 | Vectorise daily-loop hot spots and the LGBM metric | **Done** | `tcost/model.py:356 estimate` ("Fully vectorised (numpy)"); `tcost/fees.py:234 explicit_cost_vec`, `:264 marginal_fraction_vec`; `alpha/models.py:90 DateGroupedRankIC`; `alpha/alpha_model.py:226 _half_life` cached per `refit_frequency_days` block. Equivalence is checked against loop references (`tests/test_tcost.py:~760`). |
| 10 | Test the production path (pipeline, golden, hand-computed, determinism) | **Done** | `tests/test_pipeline.py` (19 tests, including a CLI round-trip), `tests/test_golden.py` + `tests/golden/pipeline_summary.json`, `tests/test_simulator_handcomputed.py` (NAV, costs, borrow, delisting to 1e-10), `tests/test_determinism.py` (including call-order independence and a slow full-pipeline equality check). Wall-clock asserts are gone (`tests/test_risk.py:291` and `tests/test_optimizer_perf.py` print `[benchmark]` only). `@pytest.mark.statistical` is applied with a multi-seed majority rule (`tests/test_statistical_seeds.py`). |

Score: 6 done (1, 2, 3, 7, 9, 10), 3 partial or mostly done (4, 5, 6), 1 not done (8).

### Silent-fallback table from the first review: status of the four "dangerous" entries
- `simulator.py:146` (`_snapshot` fallback) and `simulator.py:180` (`_borrow_usd`): **removed**. The snapshot and borrow now come from the `CostModel` with no `except`.
- `pipeline.py:127` (`_exposures_fn`): **fixed** (#3).
- `optimizer.py:272` (`part = np.inf`): **removed**. No `np.inf` participation fallback remains.
- `optimizer.py:311` `_try_solvers` broad catch: **still present** (`optimizer.py:1625`). `n_failed_solves` is reported in `BacktestResult.summary()` but is **not** a hurdle gate (`grep n_failed qts/validation/hurdles.py` finds nothing).

The remaining broad catches on the research path are `optimizer.py:1061` and `:1259` (diagnostics, annotated) and `runs/artifacts.py:79,214,220,225,300` (best-effort artefact writing, by design). These are acceptable, though `artifacts.py` should log what it swallowed.

---

## (d) Remaining top 5 improvements to reach 9/10

### 1. Split `DailyBarSimulator.run` into a step function (readability 6→8, also unblocks pyright)
This is the single largest drag. 307 code lines of loop-carried state make pyright's flow analysis not terminate: `pyright qts/backtest/simulator.py` exceeds 200 s, and that alone is why `pyright qts` cannot finish.
```python
# qts/backtest/state.py
@dataclass
class BookState:             # everything carried day to day
    nav: float; w: pd.Series; streak: pd.Series; prior_streak_usd: pd.Series
    pending_exit: dict[str, int]; delisted: set[str]

@dataclass(frozen=True)
class DayInputs:             # signal-date and exec-date views; built once per day
    t_sig: pd.Timestamp; t_exec: pd.Timestamp; t_next: pd.Timestamp
    alpha: AlphaOutput; risk: RiskModelOutput; snap_sig: LiquiditySnapshot; snap_exec: LiquiditySnapshot
    eligible: pd.Index

# qts/backtest/step.py: each one a pure function, < 60 lines, unit-tested on the 3x5 fixture
def plan_trades(state, day, optimizer, cfg) -> TradePlan: ...
def book_costs(plan, day, cost_model) -> CostEstimate: ...
def apply_returns(state, plan, md, day, cfg) -> tuple[BookState, DayRecord]: ...   # drift, delisting, borrow
# simulator.py
def run(self, md, spec: BacktestSpec, alpha_fn, risk_fn, optimizer) -> BacktestResult:
    for day in self._days(md, spec):              # ~40 lines
        plan = plan_trades(state, day, optimizer, self.cfg)
        ...
```
`BacktestSpec` (a frozen dataclass: dates, nav0, rebalance_every, venue, neutral_styles, signal_lag_days, exit_ineligible_days) replaces 10 of the 14 parameters and goes into `RunManifest`. The hand-computed and golden tests already pin behaviour, so this is a safe refactor.

### 2. Break up `portfolio/optimizer.py` and narrow the solver catch (architecture and API → 8)
```
qts/portfolio/optimizer/
  __init__.py        SinglePeriodOptimizer (thin: solve() <= 60 lines orchestrating the parts below)
  data.py            _ProblemData, _prepare, factor-role resolution
  dpp.py             _Parametrizer, _Compiled (move as is; add a module docstring on recipe semantics)
  constraints.py     build_constraints(pdat, cfg, roles) -> dict[str, cp.Constraint]
  sizing.py          closed_form_gamma, _gamma_search, _vol_target_projection (pure, unit-tested)
  result.py          OptimizationResult (frozen), binding diagnostics
```
In `_try_solvers`, catch `cp.SolverError` and `ArithmeticError` only and re-raise the rest. Add an `n_failed_solves == 0` gate (or a `max_failed_solve_frac` setting in `ValidationConfig`) to `validation/hurdles.py`, so a broken optimizer cannot produce a passing "hold everything" backtest.

### 3. Make the PIT guarantee structural (PIT 7→9)
- New `qts/core/pit.py` with `LabelPanel(values, horizon, lag)`, frozen. `alpha/targets.py:residualize_forward_returns` returns it. `alpha/trainer.py:WalkForwardTrainer.run(X, y: LabelPanel)` derives purge and embargo from `y.horizon + y.lag` instead of config. `ThemeComposite.score_all` and `to_long` reject it, by type and with a runtime `isinstance` check.
- Split `alpha/features.py:FeatureBuilder.build` into one `@feature_registry`-decorated function per panel (18 small functions). Then make `tests/test_pit_properties.py::test_features_pit` parametrise over `FEATURE_REGISTRY` and assert the registry is non-empty, so a new unregistered feature fails CI.
- `simulator.run` passes `md.as_of(t)` (a read-only truncated view) to `alpha_fn`/`risk_fn`. A `QTS_PIT_STRICT=1` CI leg refits on `md.slice_until(t)` for 1% of dates on the composite path.

### 4. Freeze and validate the contracts; type the remaining dict returns (API 6→8)
- `core/types.py`: `@dataclass(frozen=True)` on `RiskModelOutput` and `AlphaOutput`, with `__post_init__` checks (index alignment of `score`/`expected_return`, finite values, `horizon_days > 0`, `factor_groups` keys ⊆ the role literal and values ⊆ `exposures.columns`). Type it as `Mapping[FactorRole, tuple[str, ...]]`.
- `core/config.py`: `frozen=True` on every config. `apply_overrides` rebuilds via `dataclasses.replace` instead of `setattr`.
- Replace `BacktestResult.summary() -> dict[str, float]` with a frozen `BacktestSummary` (keeping `to_dict()` for JSON and API). Do the same for `TCALedger.calibrate_b -> ImpactCalibration`, `risk/evaluation.py:mincer_zarnowitz -> MZResult` and `data/universe.py:survivorship_check -> SurvivorshipReport`.
- Remove the compat layers once their callers are migrated: `_LegacyCostModel`, `LiquiditySnapshot.__getitem__/keys/from_mapping`, `ExposureHistory`'s `dict` base class, and the thin delegates in `core/crosssection.py` (`winsorize`, `rank_ic`, `ic_series`), `alpha/targets.py:industry_dummies` and `risk/exposures.py:industry_dummies`. Move `SUSPECT_SHARPE` (still defined in both `validation/leakage.py:41` and `validation/hurdles.py:44`) into `ValidationConfig`.

### 5. Performance at target scale plus a gated type check (performance 6→8, ops 8→9)
- Stage cache (`qts/core/cache.py`): `@stage_cache("features" | "exposures" | "factor_returns" | "wf_fold")` keyed by `hash_market_data(md)` and the stage's config sub-hash, stored as Parquet or `.npy` under `runs/_cache/`, and recorded in `RunManifest.cache_hits`. With it, a portfolio-layer sweep no longer refits alpha and risk.
- Daily solve: the solver dominates, so the gain has to come from fewer solves. Warm-start γ from the previous day's γ, cap the search at 3 solves once it is bracketed, and log per-day solve counts in `BacktestResult.diagnostics`. Add `scripts/bench_backtest.py`, which runs 1,000 names × 3 years and records s/day and peak RSS to `docs/notes/benchmarks.md`. Run it in the nightly job and fail on a regression of more than 25%. Either meet WORK_PLAN 1.6 (< 30 min) or formally re-baseline it.
- Tooling: with item 1 done, make `pyright qts` (basic) **blocking** in CI with a baseline file (`pyright --outputjson` → `tests/pyright_baseline.json`; fail on any new error). Burn down `alpha/models.py` (46) and `validation/hurdles.py` (40) first. Add `pytest-xdist` (`-n auto`) to the fast tier, and add `--cov-fail-under` at the current level. Pin a lock file (`uv.lock` or `requirements-lock.txt`), because LightGBM, cvxpy and Clarabel versions move numbers. Write down a UI bundle budget (for example ≤ 200 kB gzip JS in total, ≤ 120 kB for the initial route; currently 182 kB total) and check it in the `ui` CI job.

---

## (e) Verdict

**No. The codebase does not yet meet the WORK_PLAN Phase-1 exit criterion of a code-quality score ≥ 8/10 on this rubric. It scores 7/10** (7 / 7 / 6 / 6 / 8 / 8 / 6).

Two areas, testing and repro/ops, cleared 8, and the largest problems from the first review are fixed: one typed cost contract, factor roles in the contract, the hard-fail label residualiser, config serialisation with a manifest, tests on the production path, and CI across Python, API and UI. The three refactors that remain partial or undone (#4 PIT types, #6 optimizer decomposition, #8 caching) are the ones that drive API design, readability and performance. On readability the code has regressed since the first review: the simulator loop and the optimizer both grew.

The shortest path to ≥ 8: items 1, 2 and 4 of (d). They are behaviour-preserving under the existing golden, hand-computed and determinism tests, they would lift readability to about 8 and API to about 7–8, and item 1 also unblocks a blocking `pyright qts` gate. Item 3 (PIT types) and item 5 (perf and caching) are what separate 8 from 9.

The other Phase-1 exit conditions were not re-checked here, except for one observation: the suite is green, with 344/344 fast tests passing on the final run and the one transient failure caused by the other agent's edit.
