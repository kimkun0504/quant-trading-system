# Second-pass adversarial audit: alpha, data, validation (2026-11-05)

Independent re-verification after the first audit (`alpha_validation_data.md`) and two refactor
waves. Prior conclusions were not trusted; everything below was re-derived from the code and,
where stated, checked numerically. Scripts: `/tmp/audit2_a/*.py` (outputs `out_*.txt`). No source
file was modified.

Scope read: `README.md`, `CONTRIBUTING.md`, `qts/core/*`, `qts/alpha/*`, `qts/data/*`,
`qts/validation/*`, `qts/pipeline.py`, `qts/runs/artifacts.py`, the simulator's signal-lag path,
risk exposures/hybrid `as_of`, `tests/test_pit_properties.py`, `tests/test_signal_lag_real.py`.

## Severity-ranked summary

| # | Sev | Where | Finding |
|---|---|---|---|
| 1 | HIGH | `qts/alpha/targets.py:97-99` | Delisting vs temporary gap is decided with `c.bfill()` over the **whole panel**. A halt that later resumes is a "gap" in the full-sample fit and a "delisting" (-30 %) in a live fit. Training labels realised by `t`, `oos_ic_`, `ic_used` and therefore `expected_return` at `t` depend on prices **after** `t`. Verified: 4,228 residualised label cells realised by `t` differ (up to 3.7 z), `ic_estimate` 0.01248 vs 0.01237, expected return differs by 0.9 % of its max. Introduced by the first audit's delisting fix; invisible on the synthetic market (no halts). |
| 2 | HIGH | `qts/validation/ledger.py:112-128`; `qts/pipeline.py:682-703` | Ledger de-dup key `(name, config_hash, data_hash)` omits the code version and every pipeline knob outside `SystemConfig` (`use_ml` when `phase` is passed, `ml_factory`, `composite_weight` override, `rebalance_every`, `venue`, `nav0`, `warmup_days`, `dates`/`max_days`). Different experiments overwrite each other. Verified: three runs with code versions `sha_A`, `sha_B+dirty`, `unknown` → `n_trials = 1`, the surviving record is the last one (SR 1.6). The DSR is deflated by too few trials. |
| 3 | HIGH | `qts/validation/ledger.py:113-131` | Ledger is not concurrency-safe: `replace_identical` reads the file, rewrites it to a fixed `.tmp`, and `os.replace`s it, discarding appends made in between. Verified: one appender (400 trials) and one re-runner in parallel → 296 of 401 records survive (105 trials lost). The docstring says the file is shared "across every researcher/agent". |
| 4 | MEDIUM | `qts/alpha/targets.py:79-100`; `qts/alpha/trainer.py:67,107`; `qts/alpha/alpha_model.py:186`; `qts/pipeline.py:627-644` | With `signal_lag_days=1` the label, the Grinold IC and the hurdle statistics are all **lag-0**: the target is `close_s→close_{s+h}` but a trade decided on `s` earns from `close_{s+1}`; `evaluate()` scores `as_of(t)` on execution dates instead of the `as_of(t-1)` that was traded. Not leakage (purge is sufficient), but optimistic. Verified on seed 0: raw IC 0.0566 (trained/Grinold) vs 0.0537 (tradeable); reported residual IC 0.0425 vs 0.0408 for the score actually traded. |
| 5 | MEDIUM | `qts/risk/exposures.py:205-213` | The factor set includes every industry label that **ever** appears in `industry_history`. A label introduced after `t` adds an all-zero factor at `t`, which changes the eigenfactor Monte-Carlo adjustment. Verified: new label `IND99` dated at pos 1200, compare at pos 1000: factor variances differ by up to **1.2 %** (`eigenfactor_adjust=True`); 4e-15 with it off. So the risk decision input is not PIT under dated industries. |
| 6 | MEDIUM | `qts/data/universe.py:108-121` | `apply_mask` NaNs `volume`/`market_cap`/`shares` outside the universe, which throws away **knowable pre-entry history**. Every new entrant has NaN `turnover_63`/`amihud_63`/`log_dollar_vol_63` (and risk `liquidity`) for about 40 days, which are then zero-filled to the median. Its cost-snapshot ADV is computed from the post-entry days only. Verified: 2,745 eligible rows NaN→median, 8,126 eligible `turnover_63` values differ from the true trailing value, and snapshot ADV is off by up to 22 %. PIT but biased. |
| 7 | MEDIUM | `qts/data/loaders.py:614-621` | SF1 `book_to_price = bvps / close_raw` and `earnings_yield = eps / close_raw` divide per-share figures on the **filing's** share basis by the **current** raw price. After a split between filings the ratio is wrong by the split factor. Verified: 2:1 split → B/P 0.8 instead of 0.4 until the next filing. |
| 8 | MEDIUM | `qts/data/loaders.py:269-274` | FINRA short-interest ratio = SI (shares as of the settlement date) / shares outstanding **as of today**. After a split it is wrong by the split factor until the next report. Verified 0.025 vs 0.05. |
| 9 | MEDIUM | `qts/data/store.py:98-110,188-201` | ParquetStore upsert has no lock (lost updates), and a NaN cell in `write_wide` **deletes** the stored value. Verified: two concurrent writers → 40 of 80 rows stored, plus a reader crash `OSError: Couldn't deserialize thrift`. A partial re-delivery with the full ticker set deleted the stored `Y` value. Sparse panels (SI, fundamentals) are the exposed ones. |
| 10 | MEDIUM | `qts/data/loaders.py:567-569,595-604` | A late restatement of an **older** period (10-K/A for FY-1 filed after the Q1 10-Q) becomes the "latest" row at its `datekey`, and `pit_forward_fill` shows the older period as current. Verified: GP/A falls back from 0.20 (Q1-24) to 0.11 (restated FY23). Stale, not look-ahead. |
| 11 | MEDIUM | `qts/pipeline.py:553-583,715` | The PBO grid configurations are run but **never logged** in the trial ledger. A grid search adds no trials to the DSR family. |
| 12 | LOW | `qts/data/loaders.py:245,269,296` | FINRA availability = settlement + `BDay(8)`, which ignores exchange holidays. Still open from audit 1, finding 14. Verified: 2024-12-31 settlement → 2025-01-10 vs 2025-01-14 on the NYSE calendar (closed 1 Jan and 9 Jan 2025), i.e. 2 trading days early. |
| 13 | LOW | `qts/core/types.py:151`; `qts/data/store.py:313-316` | `industry_as_of(fallback=True)` fills dates before a name's first dated classification with the static map. In the store, that map is the *current* classification, so it is a look-ahead on real data. |
| 14 | LOW | `qts/core/manifest.py:117` | `git_dirty` uses `--untracked-files=no`. A new, untracked module that the run imports leaves `git_dirty=False` and `code_version=<sha>`. Verified: `docs/audit/second_pass_costs_backtest_api.md` is untracked and `git_info()` reports dirty False. |
| 15 | LOW | `qts/pipeline.py:644` vs `qts/alpha/evaluation.py:290-300` | Decile Sharpe is annualised with `round(252/H)=13` while the IC report infers 12.6, so the decile Sharpe is 1.6 % high. |
| 16 | LOW | `qts/pipeline.py:717-719`; `qts/pipeline.py:196-210` | `ic_sign` "folds" are calendar years of about 12 sampled ICs, including partial first and last years, not the purged blocks used by `net_sharpe_delta`. `ValidationConfig.embargo_days` has no effect on the fold gate (test blocks only, `min_train_days=0`). |
| 17 | LOW | `qts/pipeline.py:707`; `qts/validation/ledger.py:190-204` | `n_trials = round(n_eff)` rounds down (2.4 → 2, favourable). The DSR family `pipeline_phase{p}` pools trials across data hashes (synthetic seeds and real data) and signal lags. |
| 18 | LOW | `qts/data/universe.py:71` | The universe refresh grid is anchored at the first panel date, so membership depends on the load `start`. This is a reproducibility issue, not a PIT one. |
| 19 | LOW | `qts/data/synthetic.py` | The generator has no halts, delistings or IPOs, so findings 1 and 6 cannot show up in any synthetic test. Characteristics refresh daily with `z_t`, which is optimistic vs quarterly fundamentals. |

No CRITICAL finding: on the synthetic market, every decision input of the full pipeline is
point-in-time to ≤ 5e-14 (section "Verified correct").

---

## 1 (HIGH) — future-dependent delisting test in the labels

```python
# qts/alpha/targets.py:97
dead = c.isna() & last.notna() & c.bfill().isna()      # "no later close anywhere in the panel"
dead_by_end = dead.shift(-horizon)...
```
`c.bfill()` looks to the end of the panel. A name halted over `[H0, H1)` that resumes later is
not "dead" in the full-sample run. In the sliced (live) run it has no later close, so it gets the
−30 % haircut. Labels at `s` with `s+h ≤ t` are realised by `t`, enter ML training and `oos_ic_`,
and so reach `ic_used(t)` and `expected_return(t)`.

Verification (`/tmp/audit2_a/halt_label_leak.py`: 15 names halted for 60 days from pos 1100, live date pos 1130):
```
label rows realised by t that differ (halted names): 300
example S0081 at 2022-03-15: full=-0.0190 live=-0.3133
residualised labels realised by t: max |full-live| = 3.6755  n cells >1e-12: 4228
ic_estimate full vs live: 0.012481323535500676 0.012368775421139481
expected_return max |diff|: 5.19e-05  rel to |er| max: 0.0090
```
**Fix.** The delisting classification must use only information available when the label is
realised:
```python
# preferred: explicit, dated delisting events (CRSP dlstdt / Sharadar ACTIONS "delisted")
#   md.delist_date: pd.Series ticker -> Timestamp (NaT if alive)
dead_at = pd.DataFrame(md.dates.values[:, None] >= md.delist_date.reindex(c.columns).values[None, :],
                       index=c.index, columns=c.columns)
# fallback without an events table: causal rule evaluated at the realisation date only
dead = c.isna() & last.notna()            # no close at d (halt or delisting: unknown at d -> treat alike)
```
Add a regression test that injects a resuming halt and asserts that
`residualize_forward_returns(md)` and `residualize_forward_returns(md.slice_until(t))` agree on
every row `s` with `pos(s)+h ≤ pos(t)`. Also add `n_halts`/`n_delistings` to `SyntheticSpec`.

## 2 (HIGH) — ledger key collisions across code versions and run knobs

`TrialLedger.log(..., replace_identical=True)` replaces on `(name, config_hash, data_hash)`.
`code_version` is stored but not part of the key. The pipeline's constructor knobs are not in
`SystemConfig.hash()` (`repro.py`: `['use_ml','ml_factory','rebalance_every','venue','nav0','warmup_days','max_days']`;
the `composite_weight=` override also bypasses `cfg.alpha.composite_weight`).
```
three runs with different code versions -> n_trials = 1 ; surviving record code_version/SR: [('unknown', 1.6, 2)]
```
Iterating on code with an unchanged config and data never increases `n_trials`. Each iteration
also overwrites the previous Sharpe, which removes it from the family variance. This is the
selection bias the DSR exists to punish.

**Fix.**
```python
run_spec = {"use_ml": self.use_ml, "phase": self.phase, "ml_factory": _qualname(self.ml_factory),
            "composite_weight": self.composite_weight, "rebalance_every": self.rebalance_every,
            "venue": self.venue, "nav0": self.nav0, "warmup_days": self.warmup_days,
            "dates": [str(dates[0]), str(dates[-1]), len(dates)]}
key = (name, config_hash, data_hash, code_version, sha256(json.dumps(run_spec, sort_keys=True))[:16])
# and never replace when code_version endswith "+dirty" or == "unknown" (append instead)
```

## 3 (HIGH) — ledger loses trials under concurrent writers

`ledger_race.py`: `expected 401 records, found 296`. **Fix:** hold an exclusive
`fcntl.flock` on `<path>.lock` around both the append and the read-modify-replace, and use a
unique temp name. Better still, never rewrite the file: append a `{"rerun_of": key}` record and
de-duplicate at read time (the ledger then really is append-only).

## 4 (MEDIUM) — lag-0 label, IC and gates under `signal_lag_days = 1`

Timeline with L = 1: the signal is formed at close `s`, the trade executes at close `t = s+1`, and
returns accrue from `close_t`. The label is `close_s→close_{s+h}` (`targets.py:79-100`), so it
includes `r_{s+1}`, which the book cannot earn. The same lag-0 quantity drives `oos_ic_`, and
through it Grinold `ic_used`. `evaluate()` (`pipeline.py:627-638`) builds the IC, decile and
hurdle inputs from `alpha.as_of(t)` on **execution** dates, a score that was never traded.

Leakage check for the purge: the trainer allows `pos(s')+h < pos(p0)` (`trainer.py:107`) with
`p0 ≤ s`, so every training label is realised by close `s-1`. `trailing_ic` uses `end = pos-h`
(label realised at close `s`). Both are leak-free for decisions dated `s`. The extra execution day
needs no extra purge **as long as the label stays lag-0**. If the label is re-aligned (fix below),
the purge and the IC window must grow by `L`.

Verification (`lag_alignment.py`, seed 0, every 20th backtest date):
```
raw fwd IC  score_s vs label(s)       [trained/Grinold IC]: 0.05661
raw fwd IC  score_s vs ret(s+1..s+1+h) [tradeable, lag 1] : 0.05367
resid IC    score_t   vs target_t  [what evaluate()/hurdles report]: 0.04248
resid IC    score_t-1 vs target_t  [score actually traded at t]   : 0.04084
```
The synthetic alpha is slow (AR 0.8 per month). On real data, where short-term reversal and news
decay within days, the gap is larger.

**Fix.** Let `L = cfg.execution.signal_lag_days`:
```python
# targets.forward_returns / residualize_forward_returns: label starts at the execution close
fwd = (last.shift(-(horizon + L)) / c.shift(-L) - 1.0)                 # aligned at signal date s
# trainer: embargo must cover the whole label window
embargo = int(cfg.embargo_days or cfg.horizon_days) + L
# alpha_model.trailing_ic
end = pos - h - L
# pipeline.evaluate: score what was traded
sig = md.dates[md.dates.get_indexer(sample_dates) - L]
scores = pd.DataFrame({t: self.alpha.as_of(s).score for t, s in zip(sample_dates, sig)}).T
```

## 5 (MEDIUM) — future industry labels change the risk model at t

`_industry_labels(md)` (`exposures.py:205-211`) unions every label in `industry_history`, future
rows included. A label first used after `t` becomes an all-zero factor column at `t`. The
eigenfactor Monte-Carlo (dimension-dependent) then changes the other factors' variances.
```
industry_pit.py  : factor columns full-only: ['IND99'];  factor_cov common maxabs: 6.85e-07 ; expected_return 8.7e-19
industry_pit2.py : eigenfactor_adjust=True : max rel diff of factor variances 1.215e-02
                   eigenfactor_adjust=False: max rel diff of factor variances 3.997e-15
```
**Fix:** in `BarraStyleRiskModel.as_of(date)` (and therefore in the hybrid), drop factors whose
exposure column is identically zero at `date` before computing the covariance and the eigenfactor
adjustment. Alternatively, build the industry list from `industry_history.loc[:date]` ∪ static.
Add a PIT test with a new label after the cut.

## 6 (MEDIUM) — `apply_mask` destroys pre-entry liquidity history

`cut()` (`universe.py:108-111`) masks `volume`, `market_cap` and `shares_outstanding` wherever
the name is not in the universe. Since the eligibility refactor, the cross-section is already
gated by `md.eligibility()`, so this masking no longer protects anything. It does make every
rolling liquidity statistic of a fresh entrant NaN for about 40 days (→ zero-filled to the
cross-sectional median) and truncates its ADV window.
```
turnover_63 / amihud_63 / log_dollar_vol_63: eligible rows NaN (-> median): 2745 of 266271
turnover_63 eligible rows that differ from the unmasked (true trailing) value: 8126
cost snapshot ADV of eligible names: max rel diff masked vs true trailing history: 0.2246
```
**Fix:** `apply_mask` should set `eligible=` and leave the price, volume, cap and share panels
intact (keep the column drop only for names never eligible). Everything that forms a
cross-section already uses `eligibility()`.

## 7 / 8 (MEDIUM) — per-share fundamentals and SI ratio across splits

```
B/P  before split: 0.4  split->next filing: 0.8  after 2nd filing: 0.4     (true 0.4)
SI ratio after split, before post-split report: 0.025                      (true 0.05)
```
**Fix 7:** use totals on a consistent basis: `book_to_price = equity / market_cap`,
`earnings_yield = netinc / market_cap` (ART), with `market_cap` the PIT panel. Alternatively,
rescale `bvps`/`eps` by `split_factor(t)/split_factor(datekey)` from `close_raw/close`.
**Fix 8:** divide SI by shares outstanding **as of the settlement date**
(`so.asof(settlement_date)`, computed in the long frame before `pit_forward_fill`), or apply
FINRA's `stockSplitFlag`.

## 9 (MEDIUM) — ParquetStore: lost updates and NaN-deletes

```
store after partial upsert: Y on 2024-01-02 -> NaN (stored 20.0 deleted)
concurrent upserts: expected 80 rows, stored 40   (+ reader OSError "Couldn't deserialize thrift")
```
**Fix:** take a per-dataset `fcntl.flock` around the read-merge-replace. Make deletion explicit:
`write_wide(..., nan_policy="keep")` as the default (NaN = "no information"), with
`nan_policy="delete"` opt-in. Write `_meta.json` under the same lock.

## 10 (MEDIUM) — SF1 restatement of an older period overrides the current one

```
GP/A BBB 2024-05-15 (Q1-24): 0.2   2024-06-20 (after FY23 10-K/A): 0.11
```
**Fix:** for the level panels, per ticker, discard rows whose `calendardate` is older than the
latest `calendardate` already visible at their `datekey` (running max). Restated older periods
should only feed the YoY merge used for asset growth.

## 11 (MEDIUM) — PBO grid not counted as trials

`_pbo_from_grid` runs `len(pipeline_pbo_configs)` extra full backtests (`pipeline.py:562-574`)
without a ledger. **Fix:** pass `ledger_path=self.ledger_path` to the grid pipelines (same
family). They are trials of the same search.

## 12–19 (LOW)

Fixes, by number:
* 12: replace `BDay(n)` with `n` sessions of an exchange calendar (`pandas_market_calendars`
  XNYS), plus one session if the file is used close-to-close.
* 13: change the default to `fallback=False` and map missing labels to `"UNKNOWN"` in the store
  loader. Require `industry_history` coverage from the panel start for real data.
* 14: use `git status --porcelain` with untracked files included, or hash the imported `qts/**/*.py`
  contents into `code_version`.
* 15: pass `periods_per_year=infer_periods_per_year(scores.index)` (the same 12.6 the IC report
  uses) to `decile_spread`, and widen its annotation from `int` to `float`.
* 16: build `ic_by_fold` from the same `purged_fold_blocks` as the Sharpe gate. Drop partial years.
* 17: use `math.ceil(n_eff)`. Add `data_hash`/`signal_lag_days` to the family name, or document
  the pooling.
* 18: anchor refreshes at a fixed calendar (e.g. the first trading day of each month).
* 19: add `n_halts`, `n_delistings` and `n_ipos` to `SyntheticSpec`, and add the full-pipeline
  PIT test below to CI with them switched on.

---

## Verified correct (re-derived, with numbers)

**Full-pipeline PIT (`pit_pipeline.py`).** `ResearchPipeline(SystemConfig())` was fitted on the
full synthetic market (300 × 1512, Hybrid APCA risk active) and refitted on `md.slice_until(t)`.
The decision inputs at `t` were compared: `AlphaOutput` (score, expected_return, ic_estimate,
half_life, all components), `risk.as_of(t, alpha)` (exposures, factor_cov, specific_var,
alignment vector/variance), `cost.snapshot(md_u, t)` (price, adv_usd, adv_shares, sigma,
half_spread, borrow), the eligible set, labels realised by `t` and the OOS IC:

| run | dates (pos) | worst abs diff |
|---|---|---|
| composite (`use_ml=False`) | 530, 800, 1000, 1150, 1300, 1450 | 4.9e-14 |
| ML sleeve (ridge factory; trainer, purge, refit schedule) | 800, 850, 1040, 1300, 1450 | 8.4e-15 |
| 30 planted splits | 800, 1150, 1450 | 1.1e-14 |

`ic_estimate` and `half_life` match exactly (0.0). The eligible-set symmetric difference is 0.
The only structural differences are *extra rows* in the full fit for names that enter the
universe later (cost snapshot, label columns, ML rows). Those names are ineligible at `t`, get
`exp_ret = 0`, and are on the optimizer's `no_trade` list. At pos 800 the full fit carries an
all-NaN `ml` component that the slice omits (no fold yet): equivalent.

This covers features, label residualisation on risk exposures (warm-up fallback included),
dynamic theme weights (factor-momentum tilt, crash scalar, vol scalar), the trailing IC window
`end = pos - h`, the half-life block cache, universe hysteresis and raw-price screens, and the
APCA companion (statistical block available at every test date). It does not cover data with
halts or a dated industry introducing new labels (findings 1 and 5).

**Split invariance (`split_invariance.py`).** With 30 planted splits (only `close_raw`, `volume`
and shares change), the universe is identical and all 18 raw features match (≤ 3.6e-15).
`expected_return` matches exactly. `adv_usd` matches. `price` and `adv_shares` are consistently
on the as-traded basis (`adv_shares*price/adv_usd` invariant to 2e-16). The universe `$5`
screen, ADV, turnover, Amihud and fees all use raw price × actual shares.

**Trainer purge.** `pos(s)+h < pos(p0)`, with the validation block purged the same way. It is
leak-free for decisions dated at the signal date with L = 1 (and one day stricter than needed).

**Statistics by hand (`stats_check.py`).** All match to ≤ 1e-12:
* ICReport n/mean/NW-t/ICIR (lags 3 = the rule; ppy 12.6).
* `newey_west_lags` = `min(max(h, ⌊4(T/100)^{2/9}⌋), T-1)` for (100, 20) → 20, (252, 1) → 4,
  (3000, 5) → 8, (10, 20) → 9.
* Value-weighted decile mean and Sharpe.
* `_dsr_inputs` from a 3-trial ledger: Nyholt `n_eff` = 1.667; variance floor `252/n_obs` = 0.336
  applied; DSR 0.957186 by hand.
* CSCV PBO = 0.5857 against brute force.
* DSR SE uses raw kurtosis `(κ-1)/4`. `E[max SR]` follows Bailey–LdP eq. 8. Nyholt gives
  `M_eff = 1` for identical streams and `M` for independent ones.

**Reproducibility (`repro.py`).** `SystemConfig.hash()` is identical under
`PYTHONHASHSEED` = 0 / 1 / random and after a YAML round trip (`c1e4882749e62547`), and is
insensitive to dict insertion order. `hash_market_data` is deterministic across processes and
covers every field plus `eligibility()`. `signal_lag_days` is part of the config hash.

**Synthetic generator fairness.** `r_t` loads on `z_full[t] = z_{t-1}` and characteristics at `t`
reveal `z_t` (`synthetic.py`: `z = z_full[lag:]`, `z_drive = z_full[:T]`). The planted alpha is
known only at `t-1`. The style exposures step monthly and are revealed the same day they load
(contemporaneous, fair).

## Verdict

**On the synthetic market, the decision inputs are point-in-time.** The full
`ResearchPipeline` (alpha incl. ML, hybrid risk with APCA, cost snapshot, eligibility) reproduces
to ≤ 5e-14 when refitted on truncated data, at 14 date/configuration pairs.

**On real data they are not yet guaranteed PIT**, because of two leaks the synthetic market
cannot show:

1. Labels (and through them ML training, the Grinold IC and `expected_return`) see whether a
   halted name resumes after `t` (finding 1, HIGH).
2. Risk covariances depend on industry labels first used after `t` (finding 5, MEDIUM).

Separately, the validation layer **under-counts trials**: ledger key collisions (finding 2),
concurrent loss (finding 3) and unlogged PBO grids (finding 11). The gates are also evaluated on
the lag-0 signal (finding 4). The DSR/hurdle verdicts are therefore optimistic until those are
fixed.
