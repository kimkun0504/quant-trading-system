# Performance profile, stage cache and scale benchmark (2026-11-05)

Scope: the open performance gap of `docs/audit/code_quality_rescore_2.md` (area 4:
no stage cache; WORK_PLAN 1.6 target "composite run at 1,000 names × 15 years
< 30 min" not demonstrated; optimizer solver-bound). Every change below is
behaviour-preserving: the golden file is unchanged and full backtest outputs are
bitwise identical (method in §4).

Machine: the 2-vCPU Linux development box (7.7 GB RAM, Python 3.13, numpy 2.5.3,
pandas 3.0.5, cvxpy 1.9.3 with Clarabel). Phase 1 configuration (`use_ml=False`,
defaults: `risk_aversion_mode="target_vol"`, routed venue, `signal_lag_days=1`),
synthetic market `SyntheticSpec(n_stocks=N, n_days=D, seed=0)`, universe cap = N.

## 1. Profile (cProfile, `ResearchPipeline.run`)

Driver: generate the market, `ResearchPipeline(cfg, use_ml=False).run(md, max_days=K)`
under `cProfile`. Times are cumulative seconds *under the profiler* (about 1.3–1.6×
the unprofiled wall time); "/day" divides by the number of backtest days.

### Before (HEAD `0cd2875`)

| Stage / function | 300 names × 6 y, 120 days | 1,000 names × 6 y, 30 days ¹ |
|---|---|---|
| `fit`: risk model (`HybridRiskModel.fit`) | 11.0 s | 17.0 s |
| `fit`: alpha model (`AlphaModel.fit`) | 13.0 s (residualise labels 10.4) | 25.9 s (residualise 16.3, feature processing 7.0) |
| backtest total | 98.0 s (0.82 s/day) | 44.1 s (1.43 s/day) |
| · optimizer `solve` | 54.5 s (0.45 s/day; 3.05 cvxpy solves/day) | 12.6 s (0.42 s/day; 1 solve/day) |
| · · cvxpy compile (`get_problem_data`, `ignore_dpp`) | 27.3 s | 2.5 s |
| · · Clarabel (`solve_via_data`) | 19.3 s | 8.1 s |
| · · γ search (extra solves, included above) | 32.2 s | — |
| · risk `as_of` (`risk_fn`) | 36.7 s (0.31 s/day) | 28.3 s (0.94 s/day) |
| · · `specific_risk` | 30.6 s | 26.1 s |
| · · · `specific_vra_multiplier` | **22.9 s** (0.19 s/call) | **20.5 s** (0.69 s/call) |
| · · · `DataFrame.quantile` ×2 | 5.5 s | 4.9 s |
| · alpha `as_of` | 1.5 s | 1.3 s |

¹ measured while another process shared the CPU; the per-function shares, not the
absolute seconds, are the point.

**Hot spot #1** was not the solver: `specific_vra_multiplier` built its cap
weights with `market_cap.reindex(...).ffill().iloc[-1]` over the **whole** cap
history (O(T·N) per day, so O(T²·N) per backtest) and then `fillna(Series)` /
`fillna(0)` / `clip(lower=0)` on a frame that `fillna(Series)` had split into
one pandas block per column (pandas `where`/`clip` then loop over N blocks).
**Hot spot #2** was `DataFrame.quantile` in the specific-risk outlier damping:
with any NaN in the window pandas calls `np.quantile` once per column.
**Hot spot #3** is the optimizer: at 300 names cvxpy's per-solve compile
(`ignore_dpp=True`, every γ-search step recompiles) costs more than Clarabel;
at 1,000 names Clarabel dominates.

### After

| Stage / function | 300 names, 120 days | 1,000 names, 30 days |
|---|---|---|
| backtest total | 73.3 s (0.61 s/day) | 20.0 s (0.67 s/day) |
| · optimizer `solve` | 55.1 s (unchanged) | 12.1 s (unchanged) |
| · risk `as_of` | 11.3 s (0.094 s/day, **3.2×**) | 5.1 s (0.17 s/day, **5.6×**) |
| · · `specific_vra_multiplier` | 1.7 s (14 ms/call, **14×**) | 1.2 s (41 ms/call, **17×**) |
| · · column quantiles ×2 | 1.2 s (**4.6×**) | 0.7 s (**6.9×**) |

What is left per backtest day at 1,000 names (unprofiled ≈ 0.6 s): Clarabel
≈ 0.26 s, cvxpy compile ≈ 0.08 s per solve (1–8 solves per day, γ search), risk
`as_of` ≈ 0.13 s, problem preparation / alpha `as_of` / simulator bookkeeping
≈ 0.1 s. The fit stages (risk ≈ 11 s, alpha ≈ 17 s at 1,000 × 6 y unprofiled)
are linear in the history and run once per run.

## 2. Speed-ups applied (bitwise identical)

| Change | Where | Effect |
|---|---|---|
| Cap weights of the specific VRA in NumPy: one `get_indexer` for the shifted rows, last valid cap per column by `argmax` on the reversed validity mask, `np.where` for fill / floor. Same memory layout as `DataFrame.values` (Fortran order) because the row sums that follow are layout-sensitive in the last bit (a C-ordered first version moved NAV in the 7th digit; caught by the pickle comparison). Falls back to the pandas expression for an empty or non-float64 panel or non-unique labels. | `qts/risk/covariance.py::_vra_cap_weights` (reference: `_vra_cap_weights_pandas`) | VRA 14–17× faster; removes the O(T²·N) term |
| Column quantiles: sort once (NaN last), then one `np.quantile(axis=0)` per group of columns with the same number of valid values (same order statistics, same interpolation arithmetic as pandas' per-column calls). | `qts/risk/covariance.py::_column_quantiles` | 4.6–6.9× |

Rejected (not bitwise):

* **Re-using the compiled DPP problem** (`portfolio.reuse_problem: true`, already in
  the code, off by default): 1.4–1.7× faster backtest at 100 names, but the
  parametrised canonicalisation feeds Clarabel different (equivalent) data, so
  results move: max |ΔNAV/NAV| 5e-7 to 1.1e-6, max |Δw| 4.5e-6 to 1.0e-5 on the
  four fingerprint configurations; it also changes the config hash. Its compile
  memory grows with N² (≈ 3 GB at 600 names). Left as an opt-in.
* **Warm-starting the solver**: Clarabel has no warm start through cvxpy; any warm
  start changes the iterates and therefore the last digits.
* **Capping or re-ordering γ-search solves**: changes which book is chosen.
* Replacing pandas' EWM in the VRA (`ewm(...).mean()`, ≈ 28 ms/day at 1,000 names)
  or the fit-stage loops (`estimate_factor_returns`, `residualize_forward_returns`)
  with NumPy re-implementations: possible, but bitwise equality with pandas' and
  LAPACK's summation order is not guaranteed; the stage cache removes the fit cost
  on re-runs instead.

## 3. Stage cache

`qts/runs/cache.py::StageCache`, wired into `ResearchPipeline.fit`; config section
`cache` (`enabled: false`, `root: ''` → `<runs_root>/_cache`, `stages:
[risk_fit, alpha_fit]`); CLI `--cache`. Operating details: handbook page 12,
"The stage cache".

* **Stages.** `risk_fit` (fitted `HybridRiskModel`) and `alpha_fit` (fitted
  `AlphaModel`). The daily risk output is not cached: it depends on the alpha
  (alignment factor) and would need one entry per day; the optimizer depends on the
  previous book.
* **Key** = SHA-256 of: stage, schema version, code identity (SHA-256 of every
  `qts/**/*.py` + numerical library versions), the **full** data hash of the input
  `MarketData` (`hash_market_data(md, 64)`; its 16-char prefix is the run's
  `data_hash`), the data span (first / last date, #dates, #tickers), and the
  hashes of the config sections the stage reads (`universe`, `risk`, `warmup_days`;
  `alpha_fit` adds `alpha`, `cost.delisting_haircut`, `execution.signal_lag_days`,
  `use_ml`, `composite_weight`, ML-factory identity, and chains on the `risk_fit` key).
* **PIT safety.** An entry is only served for the identical snapshot: a different
  as-of span (data through an earlier / later date), a revised value or another
  universe changes the key; the JSON sidecar repeats data hash + span and is
  re-checked on load (mismatch = miss, logged). Tested with
  `md.slice_until(dates[-2])` (new entry, equal to an uncached run on that
  snapshot) and with one revised historical price (new entry).
* **Hashes and golden.** `cache` is excluded from `SystemConfig.hash()`
  (`HASH_EXCLUDED_SECTIONS`), so config hashes, ledger keys and the golden file are
  unchanged (`70ca0cd52263a3fc` for the defaults before and after);
  `configs/default.yaml` gained the section.
* **Storage.** `<root>/<stage>/<key[:2]>/<key>.pkl` (pickle 5) + `.json` sidecar,
  atomic writes (`mkstemp` + `os.replace`). The input `MarketData`, the ML factory
  and the pipeline are written as pickle *references* (`persistent_id`) and
  re-attached from the current run on load, so entries do not duplicate the data
  and the loaded model points at this run's objects. A failed write is logged and
  never fails the run. Entry sizes at
  1,000 names × 6 years: `risk_fit` 304 MB, `alpha_fit` 452 MB (linear in the
  history: ≈ 1.9 GB together at 15 years); `fit` 31.7 s cold (incl. ≈ 1.5 s to
  write) → 1.8 s warm, of which 0.4 s loading and 1.4 s the data hash and
  universe mask that always run.
* **Tests** (`tests/test_stage_cache.py`, 13): off by default and outside the hash;
  cold run (store) and warm run (hit) bitwise equal to an uncached run (NAV,
  weights, trades, costs, ex-ante variance, diagnostics, summary, hurdles, factor
  and specific returns, score and feature panels); invalidation per config section
  and constructor knob; stage subset; PIT (earlier as-of span, revised price);
  sidecar mismatch is a miss; key depends on code; references re-attached, missing
  reference = miss; corrupt / partial entry = miss; unpicklable object does not
  fail the run; the two NumPy speed-ups pinned to the pandas expressions (values,
  sign of zero, memory layout) on random panels.

## 4. Verification that nothing moved

1. `tests/test_golden.py` green with `tests/golden/pipeline_summary.json`
   untouched (`git diff` empty).
2. Full outputs pickled before the change (HEAD) and after, for four
   configurations, compared with `DataFrame.equals` / exact `repr` equality: NAV,
   returns, weights, trades, costs, ex-ante variance, turnover, transfer
   coefficient, the complete diagnostics (solve log with notes and binding counts,
   minus wall-clock fields), summary, hurdle result, `risk.as_of` (exposures,
   factor covariance, specific variance) on four dates, factor and specific return
   histories, alpha score and expected return on four dates, data and config hash.
   Configurations: golden (100 names × 700 d, top 80, 40 days); `signal_lag_days=0`
   + fixed γ; fixed γ + vol-target constraint mode + `rebalance_every=2`; 150 names
   × 900 d with 3 halts and 3 delistings and the ML sleeve (ridge-only, 1 seed, 30
   days). Result: **identical** for (a) the new code without cache, (b) the new code
   with the cache cold (fit + store), (c) the new code with the cache warm (load).
   The baseline itself was reproduced bitwise by a second run of HEAD.
3. The default ML ensemble (LightGBM + MLP + ridge, 60 names × 760 days, 1 seed):
   uncached, cold and warm runs give equal NAV, weights and `ml_score_` panels
   (the fitted sklearn / LightGBM models pickle; `AlphaModel.as_of` reads the
   precomputed score panels).
4. Suites: fast tier 371 passed; slow tier file by file, all 9 files exit 0;
   API 90 passed (OpenAPI output unchanged); `ruff check` / `ruff format --check`
   clean; `pyright qts/core` (strict) 0 errors; basic `pyright` on the touched
   modules has the same 32 pre-existing errors as HEAD.

## 5. Benchmark (`scripts/bench_scale.py`, unprofiled wall time)

Command: `python scripts/bench_scale.py --names N --years Y [--max-days K] [--cache]`.
Nothing else ran on the box during these measurements. "OOS days" = out-of-sample
backtest days (history minus the 505-day risk warm-up and the 1-day lag).

| Run | risk fit | alpha fit | backtest | s / day | evaluate | total `run` |
|---|---|---|---|---|---|---|
| 300 × 6 y (1,512 d), all 1,005 OOS days | 6.9 s | 6.9 s | 378 s | 0.376 | 1.1 s | **394 s (6.6 min)** |
| 1,000 × 6 y (1,512 d), all 1,005 OOS days | 11.3 s | 16.6 s | 617 s | 0.614 | 3.1 s | **650 s (10.8 min)** |
| **1,000 × 15 y (3,780 d), all 3,273 OOS days** | 33.6 s | 46.4 s | 1,620 s | 0.495 | 7.2 s | **1,710 s (28.5 min)** |

(`total run` also includes the data hash and universe mask, 1–3 s; generating
the synthetic market adds 0.2–1.1 s.)

Stage cache, last 60 OOS days of the 6-year market (`--max-days 60 --cache`):

| Run | cold (fit + store) | warm (load) | fits cold → warm |
|---|---|---|---|
| 300 × 6 y | 41.0 s | 24.5 s | 16.1 s → 0.13 s |
| 1,000 × 6 y | 62.0 s | 34.0 s | 28.9 s → 0.30 s |

Before vs after on the same window (1,000 × 15 y, last 60 OOS days, no cache):
backtest 67.6 s → 49.7 s (1.108 → 0.814 s/day), total 153 s → 137 s. The saving
per day grows with the history length, because the removed `ffill` over the
whole cap history was O(T) per day.

The cost per day is not constant along a run: it is driven by the number of
convex solves per day (1 when the closed-form γ already hits the vol target or
the caps bind, up to 8 in the γ search). On the 15-year market the last 60 days
cost 0.81 s/day against a 0.495 s/day average; the 6-year market averages 0.614
s/day. Extrapolations from a short window are therefore unreliable in both
directions; `bench_scale.py` prints one (fit linear in days, backtest = per-day
cost × OOS days), and the 6-year full-run average extrapolated to 15 years gives
34.9 min (1,000 names) and 21.2 min (300 names).

## 6. Verdict on the WORK_PLAN target

**Met on the measurement, narrowly: the full phase-1 composite run at 1,000
names × 15 years (3,780 trading days of data, 3,273 out-of-sample backtest days)
took 1,710 s = 28.5 min** on this 2-vCPU box (fit 1.3 min, backtest 27.0 min,
evaluation 0.1 min), with the stage cache off. With a warm cache a re-run saves
the 1.3 min of fitting (≈ 27.2 min).

How much weight to put on it:

* The margin is 5 %, from one seed. The per-day cost depends on the path (solves
  per day): the 6-year market's average (0.614 s/day) applied to 3,273 days would
  give 34.9 min. A different seed or a configuration that triggers more γ-search
  steps can exceed 30 min on this box; a recent laptop (faster single core) has
  more headroom.
* Before this change the same run would have taken roughly 37–45 min (estimated
  from the measured per-day saving, 0.29 s/day at the end of the 15-year sample
  and smaller earlier; not run in full).
* Only phase 1 was measured. Phase 2 adds the walk-forward ML training
  (LightGBM/MLP/ridge × 3 seeds per refit), which is not covered by this target.
* What remains is solver-bound: Clarabel ≈ 0.26 s per solve at 1,000 names plus
  ≈ 0.08 s cvxpy compile per solve. The next speed-ups that would move the needle
  all change the last digits and so need an explicit decision and a golden update:
  DPP re-use (`portfolio.reuse_problem`, measured 1.4–1.7×), a cap on γ-search
  solves once bracketed, or a looser solver tolerance.
