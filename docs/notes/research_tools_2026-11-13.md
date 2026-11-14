# Research tools: mid-cap check, holdout seal, signal tests, feature toggles, sweeps, capacity, risk calibration (2026-11-13)

Owner decisions of 2026-11-13: (1) one out-of-sample check of the frozen baseline on mid
caps; (2) seal the last three years for all new research; (3) researcher tools with
honest trial accounting. This note records what was built, the design choices that were
not in the brief, the library entry points for the API / UI, the owner's commands, and
what could not be verified without the real store. Statistics: handbook 10 §15.
Commands: handbook 12 "Research tools". Pre-registration of the mid-cap check:
`docs/notes/midcap_check_2026-11-13.md`.

## What shipped

| # | Item | Code | Tests |
|---|---|---|---|
| 1 | `universe.rank_offset` (skip the N largest before `top_n_by_mcap`; same PIT daily ranks, refresh grid, hysteresis); `qts check midcap` (pre-registered, once per config, own family `confirm:midcap:<hash12>`, coverage check V2 before the run, verdict, side-by-side vs the reference run); screened-rank prefilter `--max-rank` | `qts/data/universe.py`, `qts/data/store.py::screened_rank_sql / tickers_by_screened_rank`, `qts/research/midcap.py` | `tests/test_midcap.py` (14) |
| 2 | Holdout seal `research/seal.yaml` (`holdout_start: 2023-11-09`), outside `SystemConfig`; sealed loads for `qts run --store`, the diagnostics loader, signal tests, sweeps, capacity, calibration; `--unseal "<reason>"` once per configuration (`unseal` record first, banner, `--force` flagged "holdout already used"); `qts seal status`; full-history family for sealed runs; automatic `reference: true` for a sealed re-run of a counted configuration; `sealed_end` / `unsealed` / … in `summary.json` and the manifest | `qts/research/seal.py`, `data.py`, `store_run.py`, `qts/runs/context.py`, `qts/cli.py::cmd_run`, `scripts/diagnose_signal.py::load_market` | `tests/test_seal.py` (8), `tests/test_ledger_kinds.py` (5) |
| 3 | `qts signal list|test`: coverage, IC by horizon (NW lag = overlap), by year / fold, decay, neutral paper book, correlations, incremental IC and the fixed w = 0.25 blend, family `signal:<hash12>`, HLZ hurdle, Holm; custom-signal registry | `qts/research/signals.py`, `custom_signals.py` | `tests/test_signals.py` (9) |
| 4 | `alpha.disabled_features` (validated, normalised, dropped before processing); `qts features list`; one-line registry of all 26 features; field metadata `choices` for the API | `qts/core/config.py`, `qts/core/roles.py`, `qts/alpha/alpha_model.py`, `qts/alpha/feature_registry.py` | `tests/test_feature_toggles.py` (5) |
| 5 | `qts sweep` (grid / zip, cap 12, trial impact + DSR haircut, confirmation, cached fits, resumable manifest, PBO S = 10, metric vs parameter) | `qts/research/sweep.py` | `tests/test_sweep.py` (5) |
| 6 | `qts capacity` (backtest only per NAV, ADV-cap share, cost split, half-Sharpe NAV, `kind: capacity`) | `qts/research/capacity.py`, `baseline.py` | `tests/test_capacity.py` (7, shared with 7) |
| 7 | `qts risk calibrate` (realised / ex-ante vol by year per ν, backtest re-runs on the cached fit, recommendation rule, `kind: calibration`) | `qts/research/risk_calibration.py`, `baseline.py::NuScaledRisk` | `tests/test_capacity.py` |

Ledger (`qts/validation/ledger.py`): record `kind` (absent = trial; `unseal`, `capacity`,
`calibration`) and `reference`; only ordinary trials enter `n_trials`, `records`,
`trial_sharpes_*`, `returns_matrix`, `dsr_inputs`; `records(include_non_trials=True)`;
`families()` gains `n_not_counted`; `log(..., extra=...)` (standard keys protected; a
non-trial kind / reference is part of the de-duplication key);
`normalized_config_hash(rec)` (stored config re-hashed by the current `SystemConfig`: the
identity the once-only rules use, stable when a field is added).

## Choices not in the brief

1. **Mid-cap prefilter by screened rank, not by peak cap.** The brief proposed
   `--max-names` 4,000–5,000 by peak cap. A peak-cap cut can miss names that were mid caps
   once and never grew (ranked 1,800 in 2003 at $400M, acquired in 2005), and
   `--min-mcap-ever 3e8` drops names ranked ≤ 2,000 in early 2009 at about $250M.
   `--max-rank 2750` loads every id that ranks ≤ 2,750 by market cap *among the names
   passing the universe's price / ADV screens* on at least one date (DuckDB over the long
   tables): every name the universe can pick is loaded, and on every date at least 2,750
   passing names are, so the bottom-20 % size cut (a quantile of the loaded passing
   names) stays below rank 2,000. The coverage check (V2) measures exactly that before the
   run, for any prefilter; the pre-registration was written to it before anything ran.
2. **Sealed prefilter window.** The sealed load's peak-cap prefilter runs over
   `[start, load_end]`, so not even the *choice of ids* uses post-seal market caps. The
   family is still the full-history one: its hash is computed separately from the close
   panel of the full-window prefilter (what an unsealed `qts run` with the same flags
   hashes). Cost: the sealed run's id set can differ slightly from the reference run's
   (names that only became large after 2023-11 are not loaded), which is the point.
3. **Reference re-runs are detected, not declared.** A sealed run whose configuration is
   already a counted trial of its family is logged `reference: true` automatically (no
   flag to forget, none to abuse). Within `qts sweep` the same rule marks members.
4. **int → float in configs.** `--set portfolio.trade_aversion=1` used to give an int and a
   different config hash from the baseline's `1.0`, so `qts sweep --param
   portfolio.trade_aversion=0.5,1,2` would have counted the baseline as a new trial. An int
   for a `float` field is now a float (`from_dict`, `with_overrides`). Defaults and YAML
   files with floats are unchanged (default and baseline hashes unchanged by this); a config
   that set a float field with an int (`--set x=5`) now hashes like `x=5.0`, so such an
   earlier run's re-run gets a new config hash string (the normalised hash matches it).
5. **Overrides of one section are applied together** (one `dataclasses.replace` per
   section): fields validated against each other (`alpha.disabled_features` vs
   `alpha.feature_set`) can be given in any order. Results identical to sequential
   application whenever that succeeded.
6. **The mid-cap verdict is INCONCLUSIVE when a criterion cannot be computed** (no fold),
   and V1 is relative (`0.9 x top_n_by_mcap`, 900 for the check), so scaled-down tests
   use the same code.
7. **Capacity: impact is not monotone in NAV in a backtest, and fees fall.** The optimizer
   sees costs in NAV units, so a larger book trades less and in more liquid names; the
   realised impact per side can fall between two large NAVs while turnover collapses. Fees
   per side fall with NAV (the $0.35 per-order minimum). The cost model itself is monotone
   for a fixed book (square-root law, tested); the curve reports the split.
8. **Signal tests sample every 5th date** (the baseline's rebalance cadence) and floor the
   Newey–West lag at `ceil(h / 5)` sampled observations, which keeps a test at minutes
   (labels at the extra horizons on about 1,000 dates instead of 5,000).

## Mid-cap load

`scripts/bench_store_load.py` (synthetic store of 5,000 names × 1,000 days written to a
temporary ParquetStore, each load in a fresh process, peak RSS = `VmHWM`; this sandbox:
2 vCPU, 7 GB):

| ids loaded | days | prefilter s | load s | universe s | peak RSS |
|---|---|---|---|---|---|
| 1,000 | 1,000 | 0.2 | 5.7 | 0.3 | 0.48 GB |
| 3,000 | 1,000 | 0.2 | 10.3 | 0.9 | 0.85 GB |
| 5,000 | 1,000 | 0.2 | 13.8 | 1.4 | 1.23 GB |

Fit: peak RSS ≈ 0.28 GB + 197 bytes × ids × days. At ~6,700 days from 2000-01-01: about
1.3 GB per 1,000 ids for the load and the universe mask, before the fits
(`qts.research.data.estimated_load_gb`; the dry run prints it). How many ids
`--max-rank 2750` loads on the real store is not known here (every US company that was
about a Russell-2500 member since 2000: several thousand); `qts check midcap --dry-run`
prints it. After `apply_mask` only the names ever in the mid-cap universe reach the fits.

## Library entry points (API / UI)

Every function returns a dataclass with `to_dict()` (JSON-safe: non-finite floats as
`null`); none prints unless given `log=print`.

| Function | Returns | Written to | Top-level JSON keys |
|---|---|---|---|
| `qts.research.midcap.run_midcap_check(spec, config_path=, ledger_path=, runs_root=, force=, dry_run=, log=)` | `MidcapCheckResult` | `runs/_checks/midcap_<run_id>.json`, `<run>/midcap_check.json` | `config_hash, store, coverage{rank, prefilter, n_loaded_ids, share_dates_full, min_loaded, needed_loaded, est_load_gb, ok, …}, preregistration_sha256, previous_runs, dry_run, forced, family, run_dir, verdict{outcome, criteria[{id,name,value,threshold,passed,mark}], validity[…], secondary{…}, side_by_side[{metric, midcap, largecap, …}]}` |
| `qts.research.midcap.prefilter_coverage(spec, rank=, needed_loaded=)` | `CoverageReport` | – | as `coverage` above |
| `qts.research.midcap.criteria_table()` | `list[dict]` | – | the pre-registered rules |
| `qts.research.seal.load_seal()` / `is_sealed(date)` / `seal_status(ledger_path)` | `Seal` / `bool` / `SealStatus` | – | `seal{holdout_start, load_end, reason, sealed_by, sealed_at, store_end, inherit_family}, unseals[…], runs_under_seal, by_family{family: {kind: n}}, inherited_families, exempt_runs` |
| `qts.research.data.load_research_market(spec, unseal=, exempt=)` | `ResearchLoad(md, seal, sealed, unsealed, market_hash_full)`; `.context(phase)` -> `RunContext` | – | – |
| `qts.research.store_run.prepare_store_run(spec, cfg, use_ml=, ledger_path=, unseal_reason=, force=)` then `.finalize(cfg_final, ledger_path)` | `StoreRun`, `RunContext` | the unseal record | – (pass the context to `ResearchPipeline(context=...)`) |
| `qts.research.signals.list_signals()` / `run_signal_test(feature, spec, horizons=, baseline=, ...)` | `list[SignalDef]` / `SignalTestResult` | `runs/_signals/<id>.json` | `id, signal{name, theme, sign, role, feature_set, description, source}, horizons, primary_horizon, step, window, baseline{config_hash, cache, sealed_end, …}, coverage_by_year, ic_by_horizon{h: {mean_ic, sd, t_nw, nw_lags, n, hit}}, ic_by_year, ic_by_fold[…], decay{ic_by_horizon, lag_ic, half_life_days}, paper{sharpe_gross, sharpe_net, turnover_per_day, cost_bps_per_side, breakeven_cost_bps, …}, correlations{feature: rho}, incremental{weight, ic_by_horizon, resid_var_share, baseline_book, blend_book, delta_sharpe_net, delta_t_nw}, trial{family, n_tests, t, p, p_holm, hurdle, passes_hurdle, tests[…]}, timings, notes` |
| `qts.alpha.feature_registry.feature_info(feature_set)` | `list[FeatureInfo]` | – | `name, theme, sign, role, feature_set, description` |
| `qts.research.sweep.plan_sweep(params, base_config=, mode=, spec=, ...)`, `create_sweep(plan)`, `run_sweep_member(sweep_id, i, ...)`, `finalize_sweep(sweep_id)`, `run_sweep(plan)` | `SweepPlan`, `SweepMember`, `SweepResult` | `runs/_sweeps/<id>.json` | `id, base_config, base_config_hash, params[{key, values}], mode, store, use_ml, nav, max_days, members[{index, overrides, config_hash, status, run_id, run_dir, reference, error, metrics{sharpe_net, …, cache, seconds}}], family, impact{n_trials_raw, n_eff_before, n_new, n_eff_after, var_sr_annual, exp_max_sr_before/after, best_sharpe, dsr_best_before/after}, pbo{pbo, n_blocks, n_combinations, n_days, n_configs, prob_oos_loss, degradation_slope, logits, partial}, status, n_members` |
| `qts.research.capacity.run_capacity(spec, navs=, ...)` | `CapacityResult` | `runs/_capacity/<id>.json` | `id, config_hash, family, points[{nav, sharpe_gross, sharpe_net, cagr_net, cost_bps_per_side, impact/spread/fees_bps_per_side, turnover_per_day, traded_at_adv_cap, solves_with_adv_cap, breakeven_cost_bps, run_id, seconds}], half_sharpe_nav, reference_sharpe_net, cache, sealed_end, notes` |
| `qts.research.risk_calibration.run_risk_calibration(spec, nu_grid=, ...)` | `RiskCalibrationResult` | `runs/_risk/<id>.json` | `id, config_hash, current_nu_scale, results[{nu_scale, overall{realised_vol, exante_vol, ratio, bias_stat}, by_year{year: {…, n_days}}, sharpe_net, run_id}], recommended_nu_scale, rule, family, sealed_end, cache` |

**Sweeps as queued jobs.** The API should call `plan_sweep` (show `plan.format()` /
`plan.to_dict()["impact"]` and ask for confirmation), `create_sweep(plan)`, then enqueue
one job per pending member calling `run_sweep_member(plan.id, i, runs_root=...,
ledger_path=...)` (each job loads the store itself under the seal; members can run in any
order and a crashed job is re-run by enqueueing it again), and a final job
`finalize_sweep(plan.id)` (safe to call early: PBO over the members done, `partial: true`).
Members must run one at a time on one machine if the owner wants the fits shared: two
concurrent members that both miss the cache fit twice (both results identical).

**What the API / UI agent must do.**
- `api/qts_api/services/jobs.py::_load_market` loads the store directly
  (`ParquetStore.as_market_data`): **it bypasses the seal.** Store jobs must load through
  `qts.research.data.load_research_market` (or `prepare_store_run` + `finalize`, which also
  handles `--unseal`) and pass the returned context to `ResearchPipeline(context=...)`.
- Regenerate `api/openapi.json` / the config-schema fixtures / generated types for
  `universe.rank_offset` (integer) and `alpha.disabled_features`; the schema generator
  skips tuple fields, so it needs a branch that renders the field's metadata (`choices`
  keyed by `alpha.feature_set`, `multiple: true`) as a multi-select.
- `summary.json` gained `holdout_start, sealed_end, unsealed, unseal_reason,
  holdout_already_used, seal_exempt, record_kind, reference, market_hash_full`; the
  manifest `sealed_end, unsealed, holdout_start`. The UI should badge sealed / unsealed /
  reference runs and show "holdout already used" prominently. Ledger views should hide or
  grey out records with `kind != "trial"` or `reference: true` (they are not trials).

## Owner commands, in order

Same load flags everywhere (the stage cache is keyed on the data, so the flags must match
for the fits to be reused). Runtimes are estimates for the Mac (36 GB) from the last full
run (≈ 50 min with fits rebuilt, ≈ 25 min backtest alone), not measurements.

```bash
# (a) mid-cap check: coverage first (1-3 min), then the one run (1-1.5 h)
qts check midcap --store data/store --start 2000-01-01 --dry-run
qts check midcap --store data/store --start 2000-01-01 --cache -v 2>&1 | tee runs/midcap_check.log

# (b) the baseline once under the seal: the pre-seal benchmark (~45 min; logged as a reference)
qts run --store data/store --start 2000-01-01 --max-names 3000 --min-mcap-ever 3e8 \
    --config configs/baseline.yaml --ml --cache -v 2>&1 | tee runs/baseline_sealed.log
qts seal status

# (c) one signal test on the cached fit (5-10 min, mostly the load)
qts signal test high_52w --store data/store --start 2000-01-01 --max-names 3000 --min-mcap-ever 3e8

# (d) capacity curve on the cached fit (~1.5 h: one ~22-min backtest per NAV)
qts capacity --store data/store --start 2000-01-01 --max-names 3000 --min-mcap-ever 3e8 \
    --navs 5e6,5e7,2.5e8,1e9 -v 2>&1 | tee runs/capacity.log

# (e) optional: risk forecast report (~1.5 h)
qts risk calibrate --store data/store --start 2000-01-01 --max-names 3000 --min-mcap-ever 3e8 \
    --nu 0,0.05,0.25,1 -v 2>&1 | tee runs/risk_calibrate.log
```

Order matters for (c)-(e): they reuse the fits that (b) stores. Run (a) whenever, it has
its own data. After (b): `qts ledger list` should show the run under
`pipeline_phase2:c1da477b63d2` as not counted; if (b) prints a warning that its family is
a different `pipeline_phase2:…`, the store or the load flags differ from the reference
run's: check before any new research (merge the families if they are one search).

## Not verified without the real store

- That the full-history market hash of the reference flags is still `c1da477b63d2` (the
  store must be unchanged since the reference run) and that the reference run's ledger
  record re-hashes to the baseline's config hash (then (b) is a reference; otherwise it is
  counted as a new trial and says nothing).
- The number of ids `--max-rank 2750` loads, its memory, and the DuckDB time of the
  screened-rank query on ~20M rows (the dry run measures all three).
- The SQL screens' agreement with the pandas universe on names with gaps (equal on the
  gap-free synthetic market, tested; the 10 % rank margin covers the difference).
- Runtimes on the Mac (all estimates above are scaled, not measured).
- The cost per side the signal tests charge comes from the newest run directory with the
  baseline's configuration; before (b) there may be none under the seal, and the
  reference run's directory is not in this sandbox.
