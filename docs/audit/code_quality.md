# `qts` engineering-quality review

Scope: engineering quality only (architecture, PIT safety, API, performance, tests, ops, readability) — not alpha quality. Reviewed at commit `e63aab0` plus the working-tree edits present on 2026-11-04. Nothing in `qts/` was modified.

Note on the snapshot: `qts/risk/statistical.py`, `qts/risk/hybrid.py`, `build_risk_model` and the `statistical_*` fields of `RiskConfig` appeared in the working tree *during* this review (mtime 11:32–11:33). They were skimmed, not reviewed in depth; they are not wired into `ResearchPipeline` yet and have zero tests. Where they are mentioned below it is flagged.

Tooling run for this review:

| Tool | Result |
|---|---|
| `ruff check qts tests --select E,F,W,B,I,UP --statistics` | **1,223 findings**: 762 E501 (line > 88; only 27 at 120 cols), 164 UP045 + 148 UP006 + 79 UP035 + 9 UP007 (pre-PEP 604/585 typing — `Optional[...]`, `Dict[...]`, `typing.List`), 16 I001 unsorted imports, 10 E702 (semicolon-chained statements in `simulator.py`), 6 F401 unused imports, 6 B007 unused loop variables, 5 B905 `zip()` without `strict=`, 3 F841 unused locals, 3 E741, 3 E731, 1 B009. 367 auto-fixable. Nothing in F/B is a latent bug, but F841 in `leakage.py:98` (`te_lo, te_hi` computed and discarded) hints at an unfinished refactor. |
| `pyright qts` (default/basic mode) | **323 errors** in 55 files, 0 warnings. Top rules: `reportArgumentType` 174, `reportAttributeAccessIssue` 54, `reportCallIssue` 30, `reportOperatorIssue` 28, `reportOptionalMemberAccess` 13, `reportReturnType` 13. Hot files: `alpha/models.py` 46, `pipeline.py` 31, `backtest/simulator.py` 26, `data/loaders.py` 26, `portfolio/optimizer.py` 24. Most are pandas-stubs noise, but ~20 are real: `Optional` attributes dereferenced without a guard (`pipeline.py:126-180`, `risk/model.py:113-134`, `alpha/models.py:167,170`), `reportPossiblyUnboundVariable` (`optimizer.py:248 prob`, `exposures.py:104,117 beta_hat/resid_var`), and `Timestamp | NaTType` assigned to `Timestamp` (`alpha_model.py:116`, `exposures.py:204`, `risk/model.py:87,112`). |
| `pytest` | 100 passed in 79 s (4 threads). README's "~4 min" is stale. Slowest: 44 s session fixture for `test_backtest.py` (full `DailyBarSimulator` run), 16 s `test_alpha_model_as_of`, 15 s `test_walk_forward_oos_ic`. |

---

## (a) Scorecard

| Area | Score | One-line justification |
|---|---|---|
| 1. Architecture & boundaries | **6/10** | Layer contracts in `core/types.py` are clear and documented, but the cost boundary is duplicated (two cost models, two snapshot builders, two `SNAPSHOT_KEYS`), the optimizer discovers factor roles by string matching, and tuning knobs are split between `SystemConfig` and ~40 constructor kwargs. |
| 2. PIT safety by construction | **5/10** | Enforced by convention (`shift(k>=0)`, `.loc[:t]`) plus a good `pit_shift_test` and an exact embargo mask; nothing in the type system distinguishes a label from a feature, models are fitted on full-history panels and `as_of` indexes into them, and one silent fallback changes the label definition across time. |
| 3. API design | **5/10** | `fit/as_of/solve/run` verbs are consistent and `OptimizationResult`/`BacktestResult`/`FoldInfo` are typed, but a dozen public functions return untyped dicts/tuples, three functions change return type on a boolean flag, outputs are mutable dataclasses, and 4 broad `except Exception` blocks sit on the trading path. |
| 4. Performance & scalability | **5/10** | Feature/risk computations are genuinely vectorised; the daily loop (cvxpy re-canonicalisation, per-name Python fee loops, dict-of-DataFrames exposures, daily half-life recomputation) and the walk-forward ML (Python eval metric per boosting round) will make a 1,500 × 15y run a 4–12 h job. |
| 5. Testing | **6/10** | 100 fast, meaningful tests with real PIT, NAV-identity and embargo checks; but no pipeline test, no test of the production cost model inside the simulator, no golden/regression file, no property tests, no determinism test, several seed-tied thresholds and two wall-clock assertions. |
| 6. Reproducibility & ops | **3/10** | Dataclass config with no serialisation/versioning, trial ledger logs only `{use_ml, phase}`, no code/data/config hash is written, no CLI entry points, Makefile, CI, lint config, mypy/pyright config or `py.typed`; `configs/default.yaml` is a hand-maintained mirror that is already out of date. |
| 7. Readability | **6/10** | Module docstrings are excellent (methodology + references); but `solve()` is 201 lines, `run()` 133, `simulator.py` uses semicolon-chained statements, bps/fraction suffixes are inconsistent, `0.09` and `252` are hard-coded in 5 and ~70 places, and there is dead scaffolding (`if True:`, `NotImplementedError`). |

Overall: **5/10** — a strong research prototype written by someone who knows the literature and the failure modes, organised in the right layers, but not yet hardened to the point where a second engineer could run, reproduce, or extend it without reading every module.

---

## 1. Architecture & boundaries

### What is good
- `qts/core/types.py` states the panel conventions once (wide/long, simple returns, PIT rule, weights as NAV fractions) and every module follows them. `RiskModelOutput` is a proper factor-form contract with `covariance()`/`portfolio_variance()` helpers so consumers never rebuild `BFB'+D` themselves; the alignment term is carried as data (`alignment_vector`, `alignment_var`) rather than behaviour.
- Dependency direction is sane: `core` ← `data`/`alpha`/`risk`/`tcost` ← `portfolio` ← `backtest`/`execution` ← `pipeline`. No import cycles exist except the `risk/model.py` ↔ `risk/hybrid.py` pair, which is dodged with a local import inside `build_risk_model` (`risk/model.py:148`). That pattern is fine once; it should not spread.
- `CostTerms` (`portfolio/costs.py:40`) is a `Protocol`, so the heavy `qts.tcost` package is not an import-time dependency of the optimizer.

### Duplication across layers (each one is a divergence waiting to happen)

| Concept | Copy 1 | Copy 2 | Divergence already visible |
|---|---|---|---|
| Cost model | `portfolio/costs.py:SimpleCostModel` | `tcost/model.py:TransactionCostModel` | `SimpleCostModel.estimate` returns columns `spread/impact/fees`; `TransactionCostModel.estimate` returns `spread_usd/impact_usd/explicit_usd` + an alias `fees_usd` added "for the simulator". `SimpleCostModel` applies the per-order minimum commission; `TransactionCostModel.cvx_terms` uses `marginal_fraction` which ignores it. `SimpleCostModel.fee_fraction` caps at `commission_max_pct_value`; `FeeSchedule.marginal_fraction` does not. |
| Snapshot builder | `portfolio/costs.py:snapshot_from_market` (`half_spread` fraction; fills NaN ADV with median, price with 100) | `tcost/model.py:snapshot_from_market` (`half_spread_bps` *and* `half_spread`, plus `adv_shares`, `venue`, scalar `date`) | Different keys, different fill policy, different window arithmetic (`tail(cfg_adv_window+1)` vs `tail(adv_window)`). `DailyBarSimulator._snapshot` tries one and silently falls back to the other. |
| `SNAPSHOT_KEYS` | `portfolio/costs.py:37` (5 keys) | `tcost/model.py:32` (8 keys) | `align_snapshot` only knows the first set, so `venue`/`adv_shares`/`date` pass through un-aligned. |
| Winsorise | `core/crosssection.py:winsorize` (row-wise `apply`) | `alpha/features.py:_winsorize_rows` (numpy `nanpercentile`) | `targets.py` imports the *private* `_winsorize_rows` across modules; `crosssection.winsorize` is dead. |
| Rank IC | `core/crosssection.py:rank_ic`/`ic_series` (per-date Python loop) | `alpha/evaluation.py:rank_ic_series` (vectorised wide), `alpha/models.py:rank_ic_by_date` (vectorised long), `alpha/models.py:make_rank_ic_metric` (numpy bincount) | Four implementations, three min-N conventions (`< 20` in all but one is `where(n >= 20)` vs `r[n < min_n]`), one is O(T) Python. |
| Rolling market beta | `alpha/features.py:rolling_market_beta` | `risk/exposures.py:_rolling_market_regression` | Identical maths; `features` uses `min_periods=0.8*window`, `exposures` uses `window//4`. Feature beta and risk beta therefore disagree in the first year, and the alpha's residualiser (`targets.default_exposures`) uses the former while the risk model uses the latter. |
| Industry dummies | `alpha/targets.py:industry_dummies` (drops first column) | `risk/exposures.py:industry_dummies` (keeps all, sorts) | Two functions, same name, different semantics. |
| Newey–West t | `alpha/evaluation.py:_nw_tstat` | `backtest/metrics.py:newey_west_tstat` | Same kernel, two lag conventions. |
| `SUSPECT_SHARPE = 2.0` | `validation/leakage.py:51` | `validation/hurdles.py:30` | Not in `ValidationConfig`. |
| Auction volume share 0.09 | `CostConfig.auction_volume_share_default` | `ExecutionConfig.auction_volume_share_default`, `SquareRootImpact.cfg_share_default()` (hard-coded "to avoid a cross-layer import" — but config lives in `core`, so no cycle would occur), `FillSimulator(auction_share=0.09)`, `DryRunBroker.simulate_fill` literal `0.09` | Five sources of truth. Changing the config changes two of them. |

### Config vs constructor kwargs
`SystemConfig` is the declared source of truth, but these behaviour-changing knobs live only in constructor/method signatures and are therefore never serialised, never logged to the trial ledger, and never visible in `configs/default.yaml`:

- `AlphaModel(composite_weight=0.5, dynamic_weights=True, ic_window=252)`, `default_ml_factory` weights `[0.45, 0.35, 0.20]`, `MLPModel(max_train_rows=400_000)`, every LightGBM/MLP hyper-parameter, `WalkForwardTrainer(valid_days=252)`.
- `BarraStyleRiskModel(warmup_days, eigen_refresh_days=21, seed=0, constraint_weights="cap", weight_cap_quantile=None)`.
- `SinglePeriodOptimizer(horizon_days=20, beta_band=0.05, solver_opts)`, `solve(venue, neutral_styles, industry_columns, benchmark_weights)`, `transfer_coefficient(exact_max_n=500)`.
- `DailyBarSimulator(adv_window=63, vol_window=63)`, `TransactionCostModel.snapshot_from_market(cfg_adv_window=63)`, `build_universe_mask(refresh_days=21, exit_after_n_fails=3)`, `ExecutionRouter(loc_offset_impact_mult=1.5)`, `FillSimulator(auction_share, default_max_pct_vol)`, `ResearchPipeline(composite_weight, rebalance_every, venue, nav0)`.

`ResearchPipeline.evaluate` logs `config={"use_ml": ..., "phase": ...}` to the ledger (`pipeline.py:207`). Two runs with different `risk_aversion`, `composite_weight` or `impact_b_auction` are indistinguishable in the ledger, which defeats the DSR bookkeeping the ledger exists for.

### Name-based coupling that will break on real data (correctness, not style)
- `optimizer.py:186`: industry columns default to `[c for c in factors if str(c).upper().startswith("IND")]`. This matches the synthetic `IND00..IND11` labels only. With Fama-French-48 names (`Banks`, `Softw`, …) from `qts.data.loaders`, **the industry band constraint is silently dropped** on real data; neither `pipeline.py` nor `simulator.py` passes `industry_columns`. The risk model already knows the split (`exposures.split_factor_columns`); the optimizer should receive it from `RiskModelOutput` rather than guess.
- `optimizer.py:44,95`: `BETA_FACTOR_NAMES = ("beta", "market", "mkt")` and `_find_beta_column` returns the *first* match in column order. `BarraStyleRiskModel` orders factors `[market] + industries + styles`, so `beta_col == "market"`: the "beta-neutral" constraint is `|sum(w)| <= 0.05` (a second net-exposure constraint), while the actual `beta` style is only bounded by the ±0.10 sd style band. The constraint is not what its name and `PortfolioConfig.beta_neutral` docstring say it is.
- `simulator.py:_cost_frame`: cost columns are matched by alias list; a cost model returning any other naming (`spread_cost_usd`, `impact_dollars`) yields **zero costs with no warning**.

Fix for all three: put the roles in the contract (`RiskModelOutput.factor_roles: Mapping[str, Literal["market","industry","style"]]`; a typed `CostEstimate` dataclass) instead of inferring from strings.

---

## 2. Point-in-time safety by construction

### What is enforced today
- `MarketData.slice_until(t)` + `validation.leakage.pit_shift_test` is the right universal test and `tests/test_alpha.py::test_features_point_in_time` and `tests/test_risk.py::test_point_in_time_as_of` use it.
- `trainer.embargoed_train_mask` is an exact positional rule (`pos(s) + h < pos(t)`), unit-tested with a constructed case (`test_embargo_excludes_overlapping_labels`).
- `AlphaModel.trailing_ic` only uses IC observations whose label window has closed (`end = pos - h`).
- Feature construction is restricted to `shift(k>=0)`/trailing `rolling`, and `feature_registry` adds a cheap AST tripwire for literal `shift(-k)`.

### Where it is convention only
1. **Labels and features share a type.** `forward_returns` returns a `Wide` exactly like `mom_12_1`. Nothing stops `self.target_` (a `Wide` on `AlphaModel`) from being passed to `FeatureBuilder.process`, `ThemeComposite`, or `to_long`; only the comment "never used as features" does.
2. **Models fit on the full panel and `as_of` indexes into it.** `AlphaModel.fit(md)` stores `combined_score_`, `oos_ic_`, `resid_vol_h_`, `theme_scores_` for all dates; `as_of(t)` is a set of `.loc[date]`/`.iloc[...: end+1]` lookups whose PIT-correctness rests on each index expression being right. The same is true of `BarraStyleRiskModel` (`factor_returns.loc[:date]`, `specific_returns.loc[:date]`). There is no `as_of`-vs-`slice_until` equivalence test for `AlphaModel.as_of`, `ThemeComposite.dynamic_weights`, `trailing_ic` or `signal_half_life` — only for `FeatureBuilder.build` and the risk model.
3. **A silent fallback changes the label definition over time.** `pipeline._exposures_fn` (`pipeline.py:124-130`) returns `None` on *any* exception. `residualize_forward_returns` then uses `default_exposures`, whose style set is `("beta_252", "log_mcap", "mom_12_1", "value_bp")` (`targets.py:30`) — i.e. **momentum and value, the alpha themes the pipeline docstring promises never to residualise on**. For the risk warm-up window (the first `min_history_days` = 252 dates) and for any date where `exposures_as_of` raises, the label is defined differently than for the rest of the sample. Both the labels and `AlphaModel.oos_ic_` are therefore a mixture of two targets.
4. **`HybridRiskModel` picks `k` by Bai–Ng on end-of-sample residuals** (documented in its docstring; in-flight code). With `statistical_companion=True` as the new default in `RiskConfig`, the default configuration is no longer fully PIT.
5. `AlphaModel.as_of` and `BarraStyleRiskModel.exposures_as_of` silently fall back to the latest earlier date when `date` is missing from the index — fine for holidays, but it also masks a data gap or a mis-aligned calendar.

### Concrete proposal: make the wrong thing hard to type

```python
# qts/core/pit.py
from typing import NewType, Generic, TypeVar
import pandas as pd

@dataclass(frozen=True)
class AsOf:
    """A read-only view of MarketData truncated at `date`.  Constructed only by
    MarketData.as_of(); every panel accessor returns .iloc[:pos+1] views (no copy)."""
    _md: MarketData
    date: pd.Timestamp
    _pos: int
    @property
    def close(self) -> Wide: return self._md.close.iloc[: self._pos + 1]
    # ... one property per panel; no __setattr__; no access to the parent.

@dataclass(frozen=True)
class LabelPanel:
    """Forward-looking target.  Deliberately NOT a Wide: FeatureBuilder.process,
    ThemeComposite.score_all, to_long and EnsembleModel.predict refuse it."""
    values: Wide
    horizon: int            # the purge/embargo horizon is derived from here, not cfg
    name: str = "fwd_resid"

FeatureFn = Callable[[AsOf], pd.Series]          # one cross-section, data <= t only
PanelFeatureFn = Callable[[MarketData], Wide]    # vectorised; MUST be registered and pass pit_shift_test
```

Then:
- `WalkForwardTrainer.run(X: Long, y: LabelPanel, ...)` reads `h = y.horizon` — removes the possibility of `embargo_days < horizon` when `secondary_horizons` are used, and makes "label" visible at every call site.
- `AlphaModel.fit` keeps `self.target_: LabelPanel`; `ThemeComposite.score_all(processed: Long)` gains an `assert not isinstance(processed, LabelPanel)` (cheap runtime guard; pyright does the rest).
- `DailyBarSimulator.run` calls `alpha_fn(md.as_of(t))` / `risk_fn(md.as_of(t), alpha)` instead of passing a bare timestamp. Precomputed models keep their caches but receive an `AsOf` whose `.date` they look up — and in CI a flag `QTS_PIT_STRICT=1` makes the simulator compare `alpha_fn(md.as_of(t))` against `AlphaModel().fit(md.slice_until(t)).as_of(t)` on a random 1% of dates for the non-ML path.
- Every `PanelFeatureFn` must be decorated with `feature_registry`, and a single parametrised test runs `pit_shift_test` over `FEATURE_REGISTRY` with three random cut dates (hypothesis strategy). Today `FEATURE_REGISTRY` is empty — `FeatureBuilder.build` is not registered.
- Replace `pipeline._exposures_fn`'s `except Exception: return None` with: explicit `KeyError` handling for warm-up, raise on anything else, and **do not fall back to a different exposure set** — residualise warm-up dates on market + industries + `beta`/`size` from the internal builder only (i.e. drop `mom_12_1`/`value_bp` from `DEFAULT_STYLE_EXPOSURES` or pass `styles=RISK_ONLY_STYLES`).

---

## 3. API design

### Naming
- Verbs are mostly consistent: `fit()` (alpha, risk, models, pipeline) / `as_of()` (alpha, risk) / `solve()` (optimizer) / `run()` (simulator, trainer, pipeline) / `estimate()` (cost). Exceptions: `FillSimulator.simulate`, `HurdleReport.evaluate_signal` vs `ResearchPipeline.evaluate`, `ThemeComposite.score` vs `score_all`, `FeatureBuilder.build`/`process`/`build_and_process`.
- sklearn-style trailing underscores are used in `AlphaModel` (`fitted_`, `X_`, `target_`) and `WalkForwardTrainer` (`folds_`, `models_`) but not in `BarraStyleRiskModel` (`self.exposures`, `self.factor_returns`) or `ResearchPipeline` (`self.risk`, `self.alpha` but `self.timings_`). Pick one.
- Units in names are inconsistent: `MarketData.spread_bps` (bps) → `portfolio` snapshot `half_spread` (fraction) → `tcost` snapshot `half_spread_bps` *and* `half_spread`; `borrow_fee` (annual fraction, no suffix); `sec_fee_per_dollar_sold` (fraction); `impact_bps`; `specific_var` (daily) vs `expected_risk` (annualised vol) vs `exante_var_daily`; `turnover` is two-sided without saying so. `SquareRootImpact.total_cost_fraction` **returns dollars** and the docstring apologises for it (`impact.py:107`).

### Return types
Typed: `OptimizationResult`, `BacktestResult`, `FoldInfo`, `CrossSectionalFit` (NamedTuple), `DecileSpreadResult`, `FactorCovarianceResult`, `GateResult`, `Order`. Good.

Untyped dict/tuple returns that a caller must read the source to use:

| Function | Returns |
|---|---|
| `risk/factor_returns.py:estimate_factor_return_history` | `Tuple[DataFrame, DataFrame, Series]` |
| `tcost/tca.py:TCALedger.calibrate_b` | `Tuple[float, Dict[str, float]]` |
| `alpha/evaluation.py:ic_report`, `incremental_ic` | `Dict` mixing floats, a `DataFrame` and a `Series` |
| `validation/hurdles.py:HurdleReport.evaluate_signal` | nested `Dict[str, object]` (`PipelineResult.hurdles` is typed `object`) |
| `risk/alignment.py:alpha_alignment` | `Tuple[Series, float]` |
| `risk/evaluation.py:mincer_zarnowitz` | `Dict[str, float]` |
| `portfolio/diagnostics.py:realized_vs_exante_risk` | `Dict[str, float]` |
| `tcost/impact.py:cost_for_venue_choice` | `Dict[str, float]` (contains a `str`; pyright flags it) |
| `validation/leakage.py:simple_rank_backtest`, `planted_oracle_test` | `Dict` |
| `data/universe.py:survivorship_check` | `Dict[str, object]` |
| `backtest/simulator.py:BacktestResult.summary` | `Dict[str, float]` with 30 string keys consumed by `format_result` via `s['sharpe_net']` |
| `execution/scheduler.py:almgren_chriss_schedule` | `Dict` |
| `tcost/model.py` snapshot | `Dict[str, pd.Series]` that also carries scalars `date`, `auction_share`, `nav` |

Return type that depends on a flag (pyright cannot type these; callers unpack blindly): `specific_risk(return_details=True)` → `(Series, dict)` else `Series`; `factor_covariance(return_details=True)`; `ThemeComposite.score_all(return_components=True)`.

### Mutability and hidden state
- `AlphaOutput`, `RiskModelOutput`, `MarketData` are plain mutable dataclasses. `risk/model.py:128` (and `hybrid.py:184`) mutate a freshly built `RiskModelOutput` to add alignment. The simulator reindexes `expected_return` into a copy, so no caller currently corrupts a shared output — but nothing prevents it, and `AlphaOutput` performs no validation (index alignment of `score` vs `expected_return`, finite values, `horizon_days > 0`). Make them `frozen=True` with `__post_init__` checks; build the alignment term before construction.
- `BarraStyleRiskModel._eigen_cache` is keyed on `len(F_hist) // eigen_refresh_days` and populated by whichever `as_of` call first hits a bucket. If `as_of` is called out of date order (e.g. `portfolio_bias_test` on a shuffled date list) the cached bias vector differs from the in-order run — results depend on call history.
- `AlphaModel.fit` retains `md_`, `raw_features_`, `X_`, `target_`, `theme_returns_`, `theme_scores_` — at target scale several GB kept alive for the life of the object.
- `SinglePeriodOptimizer.solve` has 13 parameters; `venue`, `neutral_styles`, `industry_columns`, `benchmark_weights`, `horizon_days` are per-strategy constants that belong on the constructor or config.

### Error handling — every broad `except` on the trading path
See section (c) for the full table; the four dangerous ones are `simulator.py:146`, `simulator.py:180`, `pipeline.py:127`, `optimizer.py:272`.

---

## 4. Performance & scalability (1,000–1,500 names × 15 years ≈ 3,780 days)

Measured reference (from `runs/demo_phase2.txt`, 300 names × 1,512 days, ML on, 4 threads): `risk_fit=5 s`, `alpha_fit=786 s`, `backtest=196 s` for ~750 simulated days (**0.26 s/day**).

### Per-date Python loops (inventory)
| Location | Loop | At target scale |
|---|---|---|
| `backtest/simulator.py:228` `for i in range(T)` | the daily loop (unavoidable), but inside it: `optimizer.solve` rebuilds and re-canonicalises a fresh `cp.Problem` every day; `TransactionCostModel.estimate` runs two Python per-name loops (`impact.cost_bps` and `fees.explicit_bps` per ticker, `tcost/model.py:127-135`, each `explicit_bps` building a dict); `cvx_terms` builds `fb`/`fs` with two more per-name Python loops (`model.py:168-169`); `AlphaModel.as_of` recomputes `signal_half_life` over 504 × N ranks for 13 lags *every day*; `risk.portfolio_variance` is called twice. | ~1.5–3 s/day → 1.5–3 h for the loop alone |
| `risk/exposures.py:188` `for d in dates` | builds one `pd.DataFrame` per date and stores a `Dict[Timestamp, DataFrame]` | 3,780 frames × (1,500 × ~60) → ~2.7 GB plus index objects; `exposures_as_of` on a miss does a linear scan of all keys (`risk/model.py:90`) |
| `risk/factor_returns.py:146` `for d in candidates` | one `lstsq` per day | fine (~minutes) but builds `pd.Series` per day then `DataFrame(dict).T` — memory churn |
| `alpha/targets.py:114,131` | per-date `lstsq`, and the external-exposures path does a pandas `reindex`+`neutralize` per date | fine (~30–60 s) |
| `pipeline.py:180` | `alpha.as_of(d)` for every H-th date in `evaluate()` and `residualize_forward_returns` **recomputed** from scratch although `AlphaModel.fit` already holds `target_` | duplicate ~1 min |
| `core/crosssection.py:78 ic_series` | per-date Python loop (dead code in practice) | — |
| `portfolio/diagnostics.py:136 brinson_like_attribution` | per-date with `xs()`/`reindex` | diagnostics only |
| `risk/evaluation.py:76 portfolio_bias_test` | per-date `model.as_of` | evaluation only |
| `alpha/models.py:make_rank_ic_metric._ranks` | **pandas `groupby().rank()` inside the LightGBM eval metric, called every boosting round × seed × fold** | the single biggest ML cost (see below) |

### pandas anti-patterns
- `demean_by_group` (`crosssection.py:46`) and `cap_weighted_group_mean` (`features.py:95`) loop over groups assigning `out[cols] = ...` into a wide frame — O(groups) column assignments with copies; a `groupby(axis=1)`/matrix multiply with the dummy matrix is one BLAS call.
- `rank_gauss` → `norm.ppf` on a full panel then `.where()` — fine, but `process()` then `to_long` + `pd.concat(cols, axis=1)` over 18 panels creates 18 stacked copies of a 5.7 M-row index.
- `ThemeComposite.score_all` sums `themes[th].mul(W[th], axis=0).fillna(0.0)` per theme — several full-panel temporaries per theme.
- `BacktestResult.summary` / `cost_bps_of_nav` recompute from scratch on every call; `format_result` calls `summary()` indirectly multiple times.
- `pd.DataFrame(fr).T` / `pd.DataFrame(sr).T` from dicts of Series (`factor_returns.py:166-167`) — slow for 3,780 keys; preallocate numpy.

### Model fit time
- Walk-forward: `train_min_days=756`, `refit=252` → 12 folds on 15 years. LightGBM: 3 seeds × ≤600 rounds × 2 (early-stop fit + refit) on up to 5.7 M rows × 18 features with `n_jobs=4`. Rough: 20–40 min per fold on 4 cores → **4–8 h**. The custom `rank_ic` eval metric runs pandas `groupby.rank` on ~380 k validation rows per round: ~0.1–0.3 s × 600 × 3 × 12 ≈ 1–2 h of that is the metric alone; pre-computing the `y_true` ranks once (already cached) and ranking `y_pred` with a numpy argsort-per-group would make it ~10×.
- MLP: capped at 400 k rows, 200 epochs × 3 seeds, sklearn (single-threaded, no GPU) → 5–10 min/fold → 1–2 h. `ConvergenceWarning` already appears at 300 names.
- Scaling the 786 s demo by rows (×12.5) and folds (×4) gives a naive 11 h; realistic after trivial fixes: 3–5 h.

### Memory
- Each wide float64 panel at 3,780 × 1,500 is 45 MB. `FeatureBuilder.build` holds 18, `process` another 18 + the long frame (5.7 M × 18 × 8 B ≈ 0.8 GB + MultiIndex), `AlphaModel` keeps all of it plus `target_`, `theme_scores_` (6), `combined_score_`, `ml_score_`. Expect **8–16 GB peak** in the ML path; `exposure_history`'s dict-of-DataFrames adds ~3 GB. A single `(T, N, K)` float32 ndarray (3,780 × 1,500 × 60 × 4 B = 1.4 GB) with a ticker-mask array replaces the dict and makes `exposures_as_of` O(1).

### Optimizer
- Clarabel on N=1,500 with 1,500 power cones + K≈60 factor variables solves in ~0.3–1 s; cvxpy canonicalisation of a fresh problem of that size is 0.5–1.5 s. Use a `cp.Parameter`-ised DPP problem built once per universe change (alpha, `w0`, `L`, `sd`, `coef`, `part_cap` as Parameters) → canonicalise once, ~0.3 s/day. The participation-relaxed second solve on infeasible days doubles the cost on exactly the days that are already slow.
- `transfer_coefficient(exact_max_n=500)` silently switches from the exact Σ-correlation to the approximation when N > 500 — the reported TC changes definition at target scale.

### Rough full-sample runtime at target scale (single 8-core box, current code)
- Composite-only (phase 1): features+targets ~5 min, risk fit ~5 min, backtest loop ~2–3 h → **≈3 h**.
- With ML (phase 2): add 4–8 h → **≈7–11 h**, 8–16 GB RAM.
- After the refactors in (b): ≈1 h / ≈3 h.

Where Polars/numba/caching would pay: not in features (already vectorised); in (i) the daily loop's cost/fees vectorisation (numpy), (ii) a parametrised cvxpy problem, (iii) a numpy rank-IC metric, (iv) a `(T,N,K)` exposure tensor, (v) on-disk caching of `FeatureBuilder.build`, `exposure_history` and `estimate_factor_return_history` keyed by `data_hash + config_hash` (Parquet/Zarr) so a config change in the portfolio layer does not recompute the alpha layer. Polars would help `ParquetStore.read_wide` (pivot of a 5.7 M-row long table) and `TCALedger` aggregations, not the numerics.

---

## 5. Testing

### Shape
100 tests, 79 s. Roughly: 55 unit (fees, impact, splits, DSR/PBO, metrics, store, parsers), 35 integration on synthetic panels (features→composite→trainer; exposures→factor returns→covariance→`as_of`; optimizer on a stand-in risk model; simulator NAV identity), 10 property-like (PIT equality for features and risk `as_of`, embargo mask on a constructed case, constraint monotonicity, zero-alpha ⇒ zero trade, QIS beats sample covariance). No hypothesis-style generative tests, no golden files, no determinism tests.

### What is good
- NAV identity (`test_nav_identity`) and "zero alpha, zero cost ⇒ zero trade" are exactly the kind of invariants a simulator needs.
- `test_embargo_excludes_overlapping_labels` and `test_no_train_label_overlaps_test` (parametrised over all three splitters) are constructed, not statistical.
- `test_planted_truth_ic` checks the synthetic generator plants what it claims (0.025–0.055 vs `ic_true=0.04`).
- Parsers (Ken French, FINRA, Sharadar) are tested with inline fixtures offline; `test_pit_forward_fill_never_leaks` and `test_sharadar_sf1_pit_uses_datekey` test the PIT property of the data layer explicitly.

### What is missing (priority order)
1. **No test executes the production path.** `ResearchPipeline` appears in zero tests; `DailyBarSimulator` is tested only with `SimpleCostModel` (`tests/test_backtest.py:43,98,118,138`), so `TransactionCostModel.estimate` → `simulator._cost_frame` alias matching, the `_snapshot` try/except fallback, `_borrow_usd`, and `pipeline._exposures_fn` are exercised only by `scripts/run_demo.py`. A 60-name × 700-day `ResearchPipeline(use_ml=False).run()` smoke test (~20 s) would have caught the industry-column and label-fallback issues in §1/§2.
2. **No golden/regression file.** `BacktestResult.summary()` on the session fixture should be pinned (`tests/golden/backtest_summary_seed3.json`, tolerance 1e-9) so that any refactor in §(b) is provably behaviour-preserving. Same for `ic_report` and `HurdleReport` output.
3. **No hand-computed end-to-end fixture.** A 3-ticker × 5-day panel with integer prices, one short, one delisting, a fixed cost model and a stub optimizer returning known weights, with the expected `nav`, `costs`, `weights`, `trades` written by hand. Today the NAV identity is checked against the simulator's *own* cost totals, so a sign error in both would pass.
4. **Determinism.** No test runs `generate → fit → run` twice and asserts bitwise-equal NAV. LightGBM with `bagging` + `n_jobs=4` and the MLP with `early_stopping=True` are the risks; `_eigen_cache` order-dependence (§3) is another.
5. **PIT property tests.** `pit_shift_test` is applied to `FeatureBuilder.build` at one cut date (`md.dates[-300]`) and to the risk `as_of` at one date. Hypothesis over cut positions, plus coverage for `ThemeComposite.score_all(dynamic=True)`, `theme_portfolio_returns`, `AlphaModel.as_of` (composite path), `TransactionCostModel.snapshot_from_market`, `build_universe_mask` and `residualize_forward_returns`-with-`exposures_fn` are absent.
6. **Fragile thresholds tied to synthetic seeds / environment:** `test_alpha.py:118` `0.025 <= m <= 0.055` (seed 2), `:131` `ic_static > 0.01`, `:158` `rep["mean_ic"] > 0.01` from a 1-seed LightGBM (depends on LightGBM version and thread scheduling), `test_risk.py:217` `0.7 < bias < 1.4`, `:76` `corr > 0.6`, `test_backtest.py:70` `0.5 < realized/exante < 2.0`, `:74` `ann_vol_net <= 0.15`; wall-clock asserts `test_portfolio.py:69` (`solve_time < 2.0`) and `test_risk.py:271` (`< 5.0`) will flake on a loaded CI runner. Mark the statistical ones with the seed they were tuned on and widen or convert to `pytest.approx` against a golden value; move timing checks to a benchmark job.
7. **Untested modules / branches:** `qts/risk/statistical.py`, `qts/risk/hybrid.py` (in-flight, 0 tests), `scripts/*`, `TCALedger.summary`, `ExecutionRouter.routing_table`, `DryRunBroker` LOC-rejection branch, `simulator._cost_frame` with a `Series`/`dict`/mis-named frame, `optimizer` solver-chain fallback to SCS/OSQP, `BacktestResult` NAV ≤ 0 early-stop branch.
8. **Speed:** the 44 s session-scoped backtest fixture is the right idea; `test_alpha_model_as_of` (16 s) and `test_walk_forward_oos_ic` (15 s) should share one fitted model fixture (they fit LightGBM twice on the same data).

---

## 6. Reproducibility & ops

- **Config**: dataclasses only. No `to_dict`/`from_dict`/YAML/JSON round-trip, no schema version, no hash. `configs/default.yaml` is a hand-typed mirror and is already stale (`RiskConfig` gained six `statistical_*`/`missing_factor_*` fields today; the YAML has none). `SystemConfig` instances are mutated in place by the scripts (`cfg.universe.top_n_by_mcap = min(...)`).
- **Experiment identity**: `TrialLedger.log` accepts `data_hash` and `code_version` but `ResearchPipeline` passes neither; `ParquetStore.snapshot_hash_all` exists and is unused by the pipeline. There is no `git rev-parse HEAD` capture, no config hash, no package-version capture (LightGBM/cvxpy/Clarabel versions all change numerics). A `RunManifest` dataclass (code sha, dirty flag, config JSON + sha256, data hash, library versions, seed, host, wall time) written next to every run artefact is the minimum.
- **Artefacts**: `BacktestResult`/`PipelineResult` have no `save()`; `runs/demo_phase2.txt` is a redirected stdout. No run directory convention.
- **Logging**: `logging` is used in `optimizer`, `simulator`, `pipeline`; `trainer` uses `print` (`trainer.py:120`); `warnings.warn` in `universe`, `pbo`. No structured fields (date, fold, solver status) and the two most important silent paths log at `DEBUG` (`simulator.py:147`) or not at all (`simulator.py:181`, `pipeline.py:128`).
- **CLI/automation**: two argparse scripts, no `[project.scripts]` entry points, no Makefile/`justfile`, no CI workflow, no `pre-commit`, no `ruff.toml`/`[tool.ruff]`, no `[tool.pyright]`/`mypy.ini`, no `py.typed`, `dev` extra is `pytest` + `ruff` only (no `pytest-xdist`, `hypothesis`, `pyright`).
- **Packaging**: `pyproject.toml` is minimal and correct; `requires-python >= 3.10` but the code uses `X | None` annotations only in `crosssection.py`/`conftest.py` (fine with `from __future__ import annotations`). No pinned lock file for reproducible numerics.
- **Docstrings**: 173 of 462 public defs/classes lack a docstring (37%), concentrated in `models.py`, `metrics.py`, `tca.py`, `splits.py`. Module docstrings are consistently excellent — that is the right place for methodology — but function-level units/contract docs are uneven (e.g. `grinold_alpha` documents units; `holding_cost_usd` and `_borrow_usd` do not say per-day vs per-horizon).
- Stale docs inside code: `tcost/model.py:150` ("auction -> b=1.5") and `impact.py:19` ("1.5, review item 9") vs `CostConfig.impact_b_auction = 0.5`; README "~4 min" vs 79 s.

---

## 7. Readability

- **Long functions**: `SinglePeriodOptimizer.solve` 201 lines (universe alignment, objective, 8 constraint families, solve-with-fallback, dual extraction, vol targeting with scale-up bounds, expected-cost recomputation, TC); `DailyBarSimulator.run` 133 lines with eleven semicolon-chained assignments (`simulator.py:218-221, 280, 284`); `HurdleReport.evaluate_signal` 93; `synthetic.generate` 91; `sharadar_sf1_to_pit_fundamentals` 85; `FillSimulator.simulate` 78; `ExecutionRouter.route` 75.
- **Dead code / scaffolding**: `synthetic.true_factor_returns` raises `NotImplementedError` and is exported; `targets.py:113` `if True:` guarding the "fast path"; `crosssection.winsorize`, `rank_ic`, `ic_series` superseded; `leakage.py:98` `te_lo, te_hi` unused; `costs.py:159` `n` unused; `statistical.py:452` `ref` unused; `tcost/model.py:SNAPSHOT_KEYS` unused; `pipeline.py:28` unused import `rank_ic_series`; `factor_returns.py:33` unused `MARKET_FACTOR`.
- **Magic numbers**: `252` appears ~70 times despite `SystemConfig.trading_days_per_year` (e.g. `simulator.py` uses the config, `vol_target.py`, `borrow.py`, `diagnostics.py`, `evaluation.py`, `hurdles` do not); `0.09` ×5; `1e-6` solver-noise thresholds ×3 in `optimizer.py`; `60` min IC observations (`alpha_model.py:98`); `504` half-life window (`alpha_model.py:111`); `0.5` crash multiplier (`composite.py:129`); `20` min cross-section in four places; `400_000` MLP cap; `500` exact-TC cap; `1e6`/`0.02`/`100.0` snapshot defaults (`costs.py:88-92`); `0.25`/`1.5` momentum vol-scalar clip; "15:50" auction cutoff.
- **Comment quality**: high where it matters (every estimator cites its source and states units), but several comments describe a previous design ("Until the Barra-style risk model is wired in…" in `targets.py` — it is wired in; "stand-ins exist so the portfolio/backtest layers can be tested before `qts.risk`/`qts.alpha` land" in `conftest.py` — they have landed and the stand-ins are now the *only* thing the simulator tests run against).
- **Typing style**: `from __future__ import annotations` everywhere but `typing.Dict/List/Optional` used throughout (312 UP findings); consistent, just dated.

---

## (b) Top 10 refactors, in priority order

### 1. One cost contract, one snapshot, typed — delete `SimpleCostModel`'s production role
**Files**: `portfolio/costs.py`, `tcost/model.py`, `backtest/simulator.py`, `portfolio/optimizer.py`, `core/types.py` (new `core/cost.py`).
**Before**: two cost models with different column names, two `snapshot_from_market`, two `SNAPSHOT_KEYS`, `simulator._cost_frame` guessing columns, `simulator._snapshot` trying one builder and silently falling back to the other.
**After**:
```python
# qts/core/cost.py
@dataclass(frozen=True)
class LiquiditySnapshot:
    date: pd.Timestamp
    adv_usd: pd.Series; adv_shares: pd.Series; sigma_daily: pd.Series
    half_spread_frac: pd.Series; price: pd.Series; borrow_fee_annual: pd.Series
    venue: pd.Series                      # Categorical: auction | continuous
    auction_share: float
    def aligned(self, tickers: pd.Index, cfg: CostConfig) -> "LiquiditySnapshot": ...

@dataclass(frozen=True)
class CostEstimate:                       # USD per ticker
    spread: pd.Series; impact: pd.Series; fees: pd.Series; borrow: pd.Series
    @property
    def total(self) -> pd.Series: ...

class CostModel(Protocol):
    def snapshot(self, md: AsOf) -> LiquiditySnapshot: ...
    def cvx_terms(self, x, w_post, nav, snap: LiquiditySnapshot) -> tuple[cp.Expression, cp.Expression]: ...
    def estimate(self, trades_usd: pd.Series, w_post: pd.Series, nav: float, snap: LiquiditySnapshot) -> CostEstimate: ...
```
`SimpleCostModel` becomes `tests/stubs.py:ZeroCostModel` / `LinearCostModel`. `_cost_frame`, `_snapshot`, `_borrow_usd` and their three `except` blocks disappear.
**Why**: today the simulator's cost numbers depend on which of two code paths happened not to raise; a renamed column zeroes costs silently. Costs are the difference between the phase-1 gate passing and failing (`runs/demo_phase2.txt`: Sharpe 0.36 → 0.26 → 0.06 across 0.5×/1×/2× cost).

### 2. Put factor roles in `RiskModelOutput`; stop string-matching in the optimizer
**Files**: `core/types.py`, `risk/model.py`, `risk/hybrid.py`, `portfolio/optimizer.py`, `tests/conftest.py`.
**Before**: `optimizer.py:186` `startswith("IND")`, `optimizer.py:95` first-match on `("beta","market","mkt")`.
**After**:
```python
@dataclass(frozen=True)
class RiskModelOutput:
    ...
    roles: Mapping[str, Literal["market", "industry", "style", "statistical"]]
    @property
    def industry_factors(self) -> list[str]: ...
    @property
    def beta_factor(self) -> str | None: return next((f for f, r in self.roles.items() if f == "beta"), None)
```
Optimizer: `ind_cols = risk.industry_factors`; `beta_neutral` uses `"beta"` if present else raises a configuration error (do not silently pick `market`).
**Why**: on FF48-labelled real data the industry band is currently not applied at all, and "beta neutral" currently constrains the country factor. Both are invisible in the backtest output.

### 3. Hard-fail the label residualiser; make the fallback exposure set risk-only
**Files**: `pipeline.py:_exposures_fn`, `alpha/targets.py`.
**Before**: `except Exception: return None` → `default_exposures` with `mom_12_1`, `value_bp`.
**After**:
```python
def _exposures_fn(self, date):
    try:
        X = self.risk.exposures_as_of(date)
    except KeyError:                      # risk warm-up only
        return None
    return X.loc[:, self._label_neutral_columns(X.columns)]

# targets.py
RISK_ONLY_FALLBACK = ("beta_252", "log_mcap")   # never alpha themes
def residualize_forward_returns(..., fallback_styles=RISK_ONLY_FALLBACK, on_missing: Literal["fallback","nan"]="nan")
```
and a test asserting `target` built with `exposures_fn` equals the one built with the risk model's columns on every date where the risk model exists, and is NaN (or risk-only) elsewhere.
**Why**: the label is currently defined two different ways in one sample, and the "never residualise on alpha themes" invariant in the README is violated on warm-up dates.

### 4. `LabelPanel` + `AsOf` view (section 2 proposal) and register every panel feature
**Files**: new `core/pit.py`, `alpha/targets.py`, `alpha/trainer.py`, `alpha/features.py`, `backtest/simulator.py`, `validation/leakage.py`, `tests/test_pit_property.py`.
**Before**: labels are `Wide`; `FEATURE_REGISTRY` is empty; PIT tests cover two functions at one cut date.
**After**: as sketched in §2; `WalkForwardTrainer.run(X, y: LabelPanel)`; `@feature_registry` on each builder in `FeatureBuilder.build` (split the 18 panels into registered functions); one hypothesis test over `(feature, cut_pos)`; `AlphaModel.as_of` equivalence test on the composite path.
**Why**: the current guarantee is "the author was careful"; this makes the common leakage mistakes a type error or a CI failure.

### 5. Move every tuning knob into config and serialise it; write a `RunManifest`
**Files**: `core/config.py`, `alpha/alpha_model.py`, `risk/model.py`, `portfolio/optimizer.py`, `backtest/simulator.py`, `pipeline.py`, `validation/ledger.py`, `scripts/*`.
**Before**: ~40 kwargs outside config; ledger `config={"use_ml","phase"}`; YAML mirror hand-maintained.
**After**:
```python
@dataclass(frozen=True)
class SystemConfig:
    ...
    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, d) -> "SystemConfig": ...   # strict: unknown keys raise
    def sha(self) -> str: return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:12]

@dataclass(frozen=True)
class RunManifest:
    config: dict; config_sha: str; data_sha: str; code_sha: str; dirty: bool
    versions: dict[str, str]   # numpy, pandas, lightgbm, cvxpy, clarabel, sklearn
    seed: int; started: str; host: str
```
`ResearchPipeline.run` writes `runs/<ts>_<config_sha>/{manifest.json, summary.json, nav.parquet, weights.parquet, solve_log.parquet}` and logs `config_sha`/`data_sha`/`code_sha` to the ledger. `configs/default.yaml` is generated by `python -m qts.config dump` and checked in CI for drift.
**Why**: the DSR machinery is only honest if trials are distinguishable; today they are not. Also enables the caching in #8.

### 6. Decompose `SinglePeriodOptimizer.solve` and parametrise the cvxpy problem
**Files**: `portfolio/optimizer.py` (→ `optimizer/problem.py`, `optimizer/constraints.py`, `optimizer/post.py`).
**Before**: 201-line method; fresh `cp.Problem` per day; scale-up bound computation wrapped in `except Exception: part = np.inf`.
**After**:
```python
class SPOProblem:                      # built once per (tickers, factors) signature
    alpha = cp.Parameter(n); w0 = cp.Parameter(n); sd = cp.Parameter(n, nonneg=True)
    L = cp.Parameter((K, K)); lin = cp.Parameter(n, nonneg=True); kimp = cp.Parameter(n, nonneg=True)
    part_cap = cp.Parameter(n, nonneg=True)
    def solve(self, inputs: SPOInputs) -> SPORaw: ...          # DPP, warm-started

def build_constraints(cfg, B, roles, tickers, w, x) -> dict[str, cp.Constraint]: ...
def vol_target_post(w, w0, risk, snap, cfg) -> tuple[pd.Series, float, list[str]]: ...  # pure, tested, no try/except
```
**Why**: ~5× faster daily loop at N=1,500 (canonicalise once), each piece unit-testable, and the scale-up participation bound can no longer silently become `inf`.

### 7. Replace the exposures dict with a `(T, N, K)` tensor + mask; O(1) `as_of`
**Files**: `risk/exposures.py`, `risk/model.py`, `risk/factor_returns.py`, `risk/hybrid.py`, `portfolio/diagnostics.py`.
**Before**: `Dict[Timestamp, DataFrame]`, linear scan on missing dates, ~3 GB at scale, pandas frame per day in `estimate_factor_return_history`.
**After**:
```python
@dataclass(frozen=True)
class ExposureHistory:
    dates: pd.DatetimeIndex; tickers: pd.Index; factors: pd.Index; roles: Mapping[str, str]
    values: np.ndarray     # (T, N, K) float32
    valid: np.ndarray      # (T, N) bool
    def as_of(self, date) -> pd.DataFrame: i = self.dates.searchsorted(date, "right") - 1; ...
```
`estimate_factor_return_history` writes into preallocated `(T, K)` / `(T, N)` arrays.
**Why**: memory ~2×, `exposures_as_of` from O(T) to O(log T), and the hybrid model can concatenate tensors instead of frames.

### 8. Cache the expensive stages on disk, keyed by `data_sha + stage_config_sha`
**Files**: new `core/cache.py`, `alpha/features.py`, `alpha/alpha_model.py`, `risk/model.py`, `pipeline.py`.
**Before**: every `ResearchPipeline.run` recomputes features, targets, walk-forward ML and risk from scratch; `evaluate()` recomputes `residualize_forward_returns` a second time.
**After**: `@stage_cache("features", key=lambda md, cfg: (md.sha(), cfg.alpha.features_sha()))` on `FeatureBuilder.build_and_process`, `exposure_history`, `estimate_factor_return_history`, `WalkForwardTrainer.run` (per fold, keyed additionally by `fold.train_end`). Parquet for long frames, `.npy` for tensors. `evaluate()` reuses `self.alpha.target_`.
**Why**: a portfolio-layer parameter sweep at target scale should cost minutes, not hours; fold-level caching makes "append one more year" incremental.

### 9. Vectorise the daily-loop hot spots and the LightGBM metric
**Files**: `tcost/model.py:estimate,cvx_terms`, `tcost/fees.py`, `alpha/models.py:make_rank_ic_metric`, `alpha/alpha_model.py:as_of`.
**Before**: four per-name Python loops per day in `tcost/model.py:127-135,168-169`; pandas `groupby.rank` per boosting round; `signal_half_life` over 504×N each day.
**After**: `FeeSchedule.explicit_bps_vec(notional: np.ndarray, price: np.ndarray, side: np.ndarray, date) -> np.ndarray` (all components are affine in shares/dollars with `min`/`max` caps — pure numpy); `impact.cost_bps` is already vectorised, call it once; rank-IC metric via `np.lexsort((y_pred, codes))` + per-group rank assignment, `y_true` ranks cached; half-life recomputed every `cfg.alpha.half_life_refresh_days` (21) or precomputed in `fit` as a series.
**Why**: ~0.1 s/day saved on costs, 1–2 h saved per full ML fit, no behaviour change — ideal first PR because the golden file from #10 proves it.

### 10. Test the production path: pipeline smoke test, golden file, hand-computed simulator fixture, determinism
**Files**: `tests/test_pipeline.py`, `tests/golden/*.json`, `tests/test_simulator_tiny.py`, `tests/test_determinism.py`, `conftest.py`.
**Before**: `ResearchPipeline` untested; simulator tested only with the stub cost model; thresholds tied to seeds; timing asserts.
**After**: (i) `ResearchPipeline(use_ml=False)` on 60 × 700 synthetic → assert summary keys finite, `n_failed_solves == 0`, industry band binding at least once, and `summary == golden ± 1e-9`; (ii) a 3 × 5 hand-computed fixture with one short and one delisting asserting `nav`, `costs`, `weights`, `trades` exactly; (iii) run the session backtest twice and `np.testing.assert_array_equal(nav1, nav2)`; (iv) replace `solve_time < 2.0` / `< 5.0` with `pytest-benchmark` in a separate job; (v) tag seed-tuned thresholds `@pytest.mark.statistical` with the seed in the message.
**Why**: refactors #1–#9 are only safe with (i)–(iii) in place; do this first in practice (it is #10 only because it is not itself a design change).

---

## (c) Every silent fallback / broad `except` (file:line)

### Broad `except Exception`
| Location | What it swallows | Verdict |
|---|---|---|
| `qts/backtest/simulator.py:146` `except Exception as e` in `_snapshot` | any failure of `cost_model.snapshot_from_market` → `log.debug` → silently uses `portfolio.costs.snapshot_from_market` (different keys, fills, windows) | **Dangerous.** The backtest's cost basis changes without a visible signal; `n_failed_solves`-style counter does not exist for this. |
| `qts/backtest/simulator.py:180` `except Exception: pass` in `_borrow_usd` | any failure of `cost_model.holding_cost_usd` → generic borrow formula | **Dangerous.** Borrow is a first-order cost for the short book; a bug in the real model yields a different number with no log line at all. |
| `qts/pipeline.py:127` `except Exception: return None` in `_exposures_fn` | any error in `risk.exposures_as_of` (intended: warm-up `KeyError`) → label residualised on the internal exposure set that includes momentum and value | **Dangerous / correctness.** Changes the definition of the training target per date (see §2 item 3). |
| `qts/portfolio/optimizer.py:272` `except Exception: part = np.inf` | failure computing the ADV-participation bound for vol-target scale-up → **no participation bound on scale-up** | **Dangerous.** Can scale a book into trades above `max_adv_participation` with only a `notes` entry about the scalar, not about the dropped bound. |
| `qts/portfolio/optimizer.py:292` `except Exception as e` (expected-cost recomputation) | → `expected_cost = NaN`, note appended | Acceptable (diagnostic only), but should be narrowed. |
| `qts/portfolio/optimizer.py:311` `except (cp.SolverError, Exception)` in `_try_solvers` | solver crash **or any programming error raised during `prob.solve`** → status `error:<Type>`, next solver tried, then `w_current` returned | **Risky.** A `TypeError` from a mis-shaped constraint produces a plausible "hold everything" backtest with `n_failed_solves > 0` buried in `summary()`. Catch `cp.SolverError` only; re-raise others; make `n_failed_solves > 0` fail the hurdle report. |
| `qts/data/loaders.py:321` `except Exception as e` (yfinance per-ticker metadata) | network/API errors → NaN shares/industry | Acceptable (prototype loader, `pragma: no cover`), but logs via `warnings` only. |

### Narrow `except` (reviewed, acceptable)
`qts/tcost/tca.py:246` `(TypeError, ValueError)` on JSON meta; `qts/portfolio/diagnostics.py:41` `LinAlgError` → approximate TC (note the silent definition switch); `qts/validation/leakage.py:217` `(OSError, TypeError, SyntaxError)` → static check disabled, recorded in the registry entry; `qts/execution/router.py:63`, `qts/data/loaders.py:449` `(TypeError, ValueError)` on scalar parsing; `qts/tcost/impact.py:48`, `qts/tcost/borrow.py:22`, `qts/data/loaders.py:291` `ImportError` for optional deps.

### Silent fallbacks without `except`
| Location | Behaviour |
|---|---|
| `portfolio/optimizer.py:186` | industry constraint silently not applied when factor names do not start with `IND` |
| `portfolio/optimizer.py:95-97,175-180` | `beta_neutral` binds to `market` when both `market` and `beta` exist; if neither, a note is appended and the constraint is skipped |
| `portfolio/optimizer.py:230-235` | infeasible with participation bounds → silently re-solved without them (note only) |
| `portfolio/optimizer.py:246-256` | infeasible/failed → returns `w_current` with `status` set; simulator counts it but continues |
| `backtest/simulator.py:155-171` `_cost_frame` | unrecognised columns → zero spread/fees, `total` or nothing booked as impact |
| `portfolio/costs.py:align_snapshot` | missing snapshot series → constants (`adv 1e6`, `sigma 0.02`, `price 100`); NaNs → cross-sectional median |
| `core/types.py:RiskModelOutput.covariance/portfolio_variance` | tickers outside the model get zero exposure + median specific variance |
| `portfolio/diagnostics.py:transfer_coefficient` | exact → approximate when `N > 500` or `LinAlgError` |
| `alpha/alpha_model.py:112-114` | `as_of(date)` not in index → previous date's score (stale alpha) |
| `alpha/alpha_model.py:120-122` | trailing IC NaN → `default_ic`; negative → alpha clipped to **zero** (whole book flattens) with no log |
| `alpha/alpha_model.py:128-130` | half-life non-finite → `3 × horizon` |
| `alpha/composite.py:191` | dynamic weights missing → `ffill` then static weights |
| `alpha/trainer.py:95-97` | validation block empty → trains without validation (no early stopping) silently |
| `alpha/models.py:220` | `best_iteration_` falsy → full `n_estimators` |
| `risk/model.py:88-92` | `exposures_as_of` → latest earlier date |
| `risk/covariance.py:factor_vra_multiplier:136`, `specific_vra_multiplier:323,335` | too little history → multiplier 1.0 |
| `risk/covariance.py:structural_specific_vol:345-348` | too few full-history names → fit on all; fewer than 5 → median |
| `risk/alignment.py:62-64` | alpha fully spanned → `nu = 0` (alignment off) |
| `tcost/model.py:_venue_volume` / `cvx_terms:163` | missing ADV → median ADV, floor `1e-9` |
| `tcost/impact.py:cfg_share_default` | hard-coded `0.09` regardless of `ExecutionConfig` |
| `data/store.py:as_market_data:227` | unknown industry → `"UNKNOWN"` (becomes a real industry factor downstream) |
| `validation/pbo.py` | combinations capped to 5,000 random subsamples with a `RuntimeWarning` (shows in the test run) |
| `pipeline.py:evaluate:195` | `pbo=None` → gate fails with "not supplied" (explicit — good) |

---

## (d) Verdict

**Is this organised like a top-1% quant engineer would organise it?** Partly.

What is already at that bar:
- The layer decomposition and the single statement of data conventions in `core/types.py`; the factor-form `RiskModelOutput` contract; the decision to run the optimizer in factor form and never materialise Σ.
- Vectorised feature and exposure construction with no per-date loops; the exact positional embargo mask; `pit_shift_test` as a universal, cheap PIT oracle; the planted-oracle test and the explicit "DSR cannot detect leakage" stance; the NAV-identity test; a synthetic market with *known* truth used to test the estimators.
- The write-up quality: every estimator cites its source, states units, and documents its known limitations (QIS, USE4 specific risk, Saxena–Stubbs, Perold IS decomposition, CSCV cap). That is rarer than clean code.

What is not:
- **Correctness hinges on string matching and silent fallbacks.** Industry bands evaporate on real industry names, "beta neutral" constrains the country factor, the training label is defined two ways in one sample, and the simulator can switch cost models or zero costs without a log line. A top-1% codebase makes these impossible by construction (roles in the contract, typed cost outputs, `LabelPanel`), not by remembering.
- **Reproducibility is nominal.** The trial ledger exists but cannot distinguish two runs; there is no manifest, no config serialisation, no hashing, no CI, no type-check gate; 323 pyright errors and 1,223 ruff findings have never been looked at. A fund's research platform treats "can I regenerate this number from the ledger row" as a hard requirement.
- **The production path is untested.** All 100 tests pass while `ResearchPipeline` and the `TransactionCostModel`-inside-`DailyBarSimulator` path have zero coverage; the only evidence they work is a stdout file in `runs/`.
- **Two of everything** (cost model, snapshot, winsorise, rank IC, rolling beta, industry dummies, NW t-stat, auction share) is the signature of a fast solo build, not a maintained platform; each pair has already drifted.
- **Scale**: the design is right (vectorised panels, factor-form optimizer), but the daily loop and the walk-forward ML as written are a 7–11 h job at 1,500 names × 15 years with 8–16 GB RAM, with no caching between runs.

Priority if only three things get done: #10 (golden + pipeline test), then #1/#2/#3 together (they are the correctness bugs), then #5 (manifest + config serialisation). Everything else is velocity.
