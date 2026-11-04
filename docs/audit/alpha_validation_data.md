# Adversarial audit: alpha, validation and data layers

Scope: `qts/alpha/*`, `qts/validation/*`, `qts/data/*`, `qts/pipeline.py` (read against
`README.md`, `qts/core/types.py`, `docs/BUILD_PLAN.md` §2.4 / §8, `docs/REVIEW_NOTES.md`).
Method: line-by-line read, then toy-data scripts in `/tmp/audit/` (outputs quoted below).
`pytest tests/test_alpha.py tests/test_validation.py tests/test_data.py -q`: 36 passed.
No source file was modified.

Severity key: CRITICAL = look-ahead, or wrong sign/scale that would mislead a go/no-go;
HIGH = wrong number but direction-preserving, or affects a reported diagnostic/gate input;
MEDIUM = fragility / silent fallback; LOW = style or minor approximation.

## Severity-ranked summary

| # | Sev | Where | Finding |
|---|---|---|---|
| 1 | CRITICAL | `qts/data/universe.py:85,94`; `qts/alpha/features.py:181`; `qts/alpha/targets.py:35,130`; `qts/alpha/alpha_model.py:119` | `apply_mask` keeps every ticker that is *ever* in the universe (`mask.any(axis=0)`) and leaves `close` unmasked, so the alpha cross-section at `t` (ranks, z-scores, labels, ML training rows, IC, emitted scores) contains names selected on **future** membership. Verified: 21 % of feature rows are out-of-universe at their date; future entrants earn +110 % annualised in the 63 days before entry vs −6 % for the universe. Invisible on the default synthetic (300 names, `top_n=1000`) so no test catches it. |
| 2 | CRITICAL | `qts/alpha/targets.py:32,97`; `qts/alpha/alpha_model.py:67`; `qts/pipeline.py:124-130` | The library-default label residualises on **momentum and value** (`DEFAULT_STYLE_EXPOSURES`), contradicting README/REVIEW_NOTES item 4. It is what `AlphaModel().fit(md)` (README example), `ic_decay_curve`, the test fixtures and the pipeline's warm-up rows use, and it is the *silent* fallback whenever `_exposures_fn` raises (bare `except Exception`). Verified: rank IC of log B/P vs the default target = −0.001, vs a risk-only target = +0.019 — the value theme is defined away. |
| 3 | HIGH | `qts/pipeline.py:196-202`; `qts/validation/hurdles.py:57-120` | Hurdle wiring does not implement plan §2.4: `decile_tstat` receives the *IC* t-stat (not the decile long-short t), `net_sharpe_delta_by_fold` receives absolute fold Sharpes (gate becomes "fold SR ≥ 0.05"), `pbo=None` makes the PBO gate fail unconditionally so `overall` can never be PASS, and `ic_by_fold` is built from ~12 observations per year. |
| 4 | HIGH | `qts/pipeline.py:180-182`; `qts/alpha/evaluation.py:64` | ICIR is annualised with `periods_per_year=252` while the IC series is sampled every `H=20` days → ICIR inflated by √20 ≈ 4.5×. (`t_stat` is correct.) |
| 5 | HIGH | `qts/alpha/targets.py:35-37` | Label survivorship: `forward_returns` is NaN whenever `close_{t+h}` is NaN, so the last `h` labels of every delisted name (the delisting return) are dropped from ML training and from every IC/decile statistic. Verified: 0 of 630 pre-delisting labels survive. |
| 6 | HIGH | `qts/data/loaders.py:521-528` | Sharadar `asset_growth` joins the prior-year row with `keep="last"` after sorting by `datekey`, i.e. the **latest-filed** (possibly restated) value even when that filing is after the current row's `datekey`. Verified: a restatement filed 2021-06-01 changes the growth visible on 2021-02-16 from +20 % to −20 %. |
| 7 | HIGH | `qts/pipeline.py:192-195` | DSR inputs: `var_trials_sr_annual` hard-coded to 0.25; the ledger's own cross-trial variance (`TrialLedger.dsr_for`) and `effective_trials` are never used; `n_trials` counts all families. 0.25 happens to equal the pure sampling-noise variance of an annual SR over ~1 000 days (252/n) and is arbitrary for any other sample length or trial correlation. |
| 8 | MEDIUM | `qts/data/synthetic.py:97-99,119,146` | Planted alpha is **not** lagged (docstring says it is): `z_m` starts driving returns on the first day of month `m` and is revealed in the characteristics on the same day. PIT-fair (verified `as_of` invariance), but optimistic vs real data (no publication lag; characteristics explain the *past* month's return as well as the next: IC 0.025 vs 0.021), it creates an exploitable day-of-month structure, and the realised IC of the true score vs the target is ≈0.02, not `ic_true=0.04`, so the "recovers ic_true and not more" oracle is loose. |
| 9 | MEDIUM | `qts/alpha/features.py:181`; `qts/alpha/composite.py`; `qts/alpha/alpha_model.py:119` | Delisted / no-price names are zero-filled, ranked and z-scored with the live cross-section and receive scores from `as_of` (verified 30/100 dead names scored). Dilutes dispersion; on real data hundreds of dead names sit at the median of every rank. |
| 10 | MEDIUM | `qts/validation/pbo.py:113-116` | `PBO = mean(logit <= 0)` counts an exact-median OOS rank as overfit; for small odd `N` this biases PBO up (N=3 pure noise → 0.66, not 0.5); identical/constant configurations give PBO = 1.0; NaNs are silently zeroed. |
| 11 | MEDIUM | `qts/validation/leakage.py:148`; `qts/pipeline.py` | `planted_oracle_test` only exercises the toy `simple_rank_backtest`; the actual `ResearchPipeline` suspect gate has never been shown to flag an oracle (there is no `extra`-panel feature hook to inject one). |
| 12 | MEDIUM | `qts/pipeline.py:124-130`, `qts/alpha/trainer.py` | Target definition changes at the risk-model warm-up boundary (first 252 dates residualised on alpha themes, later dates on risk-only styles); the first ML fold trains on a 1/3 : 2/3 mixture of two different labels. |
| 13 | LOW | `qts/alpha/models.py:273` | MLP early stopping uses sklearn's random 10 % holdout drawn from the (overlapping-label) training block — optimistic stopping, no OOS leak. |
| 14 | LOW | `qts/data/loaders.py:242,258` | FINRA availability = settlement + `BDay(8)` ignores exchange holidays and the 4 pm publication time; use FINRA's dissemination calendar (+1 trading day for close-to-close use). |
| 15 | LOW | `qts/alpha/evaluation.py:109` | Newey–West lags = `h−1` for `h`-day overlapping spreads under-covers the MA(h−1) structure with a Bartlett kernel; use ≥ `h` (1.5·h is common). |
| 16 | LOW | `qts/data/loaders.py:62`; `qts/data/store.py:212` | `pit_forward_fill` carries a value forever (no staleness cap); `as_market_data` reindexes sparse panels without ffill. |
| 17 | LOW | `qts/pipeline.py:204-209` | Ledger records have empty `data_hash`/`code_version`; identical re-runs inflate `n_trials` (conservative, but makes the DSR path-dependent on how often the demo is run). |

---

## Finding 1 (CRITICAL) — universe membership leaks into the alpha cross-section

**Files**: `qts/data/universe.py:85` (`keep = mask.columns[mask.any(axis=0)]`), `:94`
(`close=md.close[keep]` unmasked), `qts/alpha/features.py:181` (`fillna(0.0)` after rank-gauss),
`qts/alpha/targets.py:35` (`forward_returns` from unmasked close), `:130`
(`X.reindex(tickers).fillna(0.0)` gives out-of-model names a label), `qts/alpha/alpha_model.py:119`
(`as_of` returns every non-NaN combined score).

**What is wrong.** `apply_mask` masks volume/mcap/fundamentals but (by design) not `close`. Because
momentum, beta, idio-vol, max-ret and reversal are functions of `close` alone, every ticker in
`keep` has a full set of price features on every date, and the remaining features are zero-filled
by `process`. The processed frame, the per-date ranks/z-scores, the theme decile portfolios, the
labels and hence the ML training set, the IC series and the emitted `AlphaOutput.score` therefore
all contain names that are not in the universe at `t`. The set `keep` is chosen with knowledge of
the whole sample (`mask.any(axis=0)`): a microcap is in the cross-section in 2010 because it
entered the top-1000 in 2015. Names enter the universe by rising in market cap, so pre-entry rows
carry strongly positive realised returns; conversely names that have left (and delisted) remain
with zero features. The optimizer takes the union of alpha tickers and risk tickers
(`qts/portfolio/optimizer.py:123`), so the backtest can also hold out-of-universe names.

**Verification** (`/tmp/audit/v4_universe.py`, synthetic 300 names, `top_n_by_mcap=120`):

```
tickers total / kept (ever in universe) / avg in-universe per date: 300 151 119.07
fraction of X_ rows (date,ticker) OUTSIDE the universe at that date: 0.2114
as_of score count / in-universe count at t: 151 125
target non-NaN for out-of-universe rows: 31381
mean IC all kept names 0.0137 vs in-universe only 0.0234
mean annualised return of future entrants over the 63d BEFORE entry: 1.102  (universe avg -0.059)
```

With the default `SyntheticSpec` (300 names) and `top_n_by_mcap=1000` every name is always in the
universe, which is why `tests/` never see this.

**Fix.** Carry the mask into the alpha layer and drop rows instead of zero-filling:

```python
# qts/alpha/features.py  FeatureBuilder.process(self, features, md, valid: Wide | None = None)
valid = md.market_cap.notna() if valid is None else valid          # in-universe & alive
for name in FEATURE_ORDER:
    ...
    w = _winsorize_rows(features[name].where(valid), self.cfg.winsor_pct)   # rank only members
    ...
out = pd.concat(cols, axis=1)
out = out[to_long(valid, "v").reindex(out.index).fillna(False).astype(bool)].fillna(0.0)

# qts/alpha/targets.py  residualize_forward_returns(...)
fwd = _winsorize_rows(forward_returns(md, horizon).where(md.market_cap.notna()), winsor_pct)

# qts/alpha/alpha_model.py  as_of()
score = self.combined_score_.loc[date].where(self.md_.market_cap.loc[date].notna()).dropna()
```

and in `apply_mask` store the mask on the object (`md.extra["universe_mask"] = m`) so
`theme_portfolio_returns`, `momentum_crash_scalar_series` (`md.returns.mean(axis=1)`) and
`rolling_market_beta`'s market proxy can be restricted to members as of `t`.

## Finding 2 (CRITICAL) — default labels residualise on the alpha themes; silent fallback

**Files**: `qts/alpha/targets.py:32` (`DEFAULT_STYLE_EXPOSURES = ("beta_252","log_mcap","mom_12_1","value_bp")`),
`:97` (`styles = default_exposures(md, features)` always built and used for any row the external
hook does not cover), `qts/alpha/alpha_model.py:67` (`risk_exposures_fn=None` default),
`qts/pipeline.py:124-130` (`except Exception: return None`).

**What is wrong.** README ("Labels … are residualised on risk-only styles … never on the alpha
themes") and REVIEW_NOTES item 4 describe `ResearchPipeline` only. The library default — the
README's own `AlphaModel().fit(md)` example, `evaluation.ic_decay_curve(residualized=True)`,
`tests/test_alpha.py`'s `target` fixture — neutralises the 20-day forward return against 12-1
momentum and log(B/P). Inside the pipeline the same default is used (a) for every date before the
risk model's warm-up (`exposures_as_of` raises `KeyError` → `None`), i.e. the first 252 dates, a
third of the first ML fold's training window, and (b) for *every* date if `exposures_as_of` ever
raises for another reason, because the bare `except Exception` hides it. A model whose target has
value and momentum projected out will learn that value and momentum carry no information.

**Verification** (`/tmp/audit/v5_synth.py`):

```
IC(log B/P, default target) = -0.0011 ; (risk-only target) = 0.0190
IC(z, target residualised on beta+size+mom+value) = 0.0163 ; on beta+size only = 0.0210
```

**Fix.**

```python
# targets.py
DEFAULT_STYLE_EXPOSURES = ("beta_252", "log_mcap", "idio_vol_252", "log_dollar_vol_63", "strev_21")  # risk-only
# residualize_forward_returns: make the fallback explicit and loud
if exposures_fn is not None and not external:
    raise ValueError("exposures_fn returned nothing for any date; refusing to fall back silently")
if exposures_fn is not None and len(internal_rows):
    warnings.warn(f"{len(internal_rows)} dates use the internal risk-only exposures (risk-model warm-up)")

# pipeline.py
def _exposures_fn(self, date):
    try:
        X = self.risk.exposures_as_of(date)
    except KeyError:            # warm-up only; anything else must surface
        return None
    ...
```

and either start labels at the risk warm-up (`target_.loc[:warmup] = NaN`) or fit the risk model
with `warmup_days=0` for exposure purposes so a single target definition is used throughout.

## Finding 3 (HIGH) — hurdle gates are fed the wrong quantities

**File**: `qts/pipeline.py:196-202`.

* `decile_tstat=float(icr.get("t_stat"))` — the plan's Gate 2 is the **decile long-short t-stat
  (gross, VW, ex-microcap) > 3**; what is passed is the t-stat of the mean rank IC over
  `n_backtest_days / 20` observations (≈ 50 for a 4-year run). `evaluation.decile_spread` exists and
  is never called by the pipeline.
* `net_sharpe_delta_by_fold=fold_sharpes` — absolute calendar-year Sharpes are compared with the
  +0.05 *improvement* threshold. For a standalone run this gate is meaningless; it should be
  `applied=False` or computed as delta vs. the previous ledger baseline.
* `pbo=None` — `hurdles.py:120` then appends a failing gate, so `overall` is always `FAIL` for any
  pipeline run regardless of performance. The README presents the demo's FAIL as "the intended
  behaviour of the gates"; it is structurally guaranteed, not a verdict.
* `ic_by_fold` = yearly means of ~12 ICs; the 2/3-of-folds rule on such small folds is noise.

**Fix.**

```python
from qts.alpha.evaluation import decile_spread
from qts.alpha.targets import forward_returns
ds = decile_spread(scores, forward_returns(self.md_u, H).reindex(scores.index), horizon=1,   # non-overlapping sample
                   weights=self.md_u.market_cap.reindex(scores.index), periods_per_year=252 // H)
hurdles = HurdleReport(cfg.validation).evaluate_signal(
    ic_by_fold=..., decile_tstat=ds.t_stat,
    net_sharpe_delta_by_fold=None,                       # not defined for a standalone run
    breakeven_cost_bps=breakeven, modeled_cost_bps=..., dsr=dsr,
    pbo=pbo_from_cpcv_or_None, phase=self.phase, net_sharpe=summ["sharpe_net"], standalone=True)
```

and in `hurdles.py` make the PBO gate `applied=False` (with a reason) when `pbo is None`, as is
already done for `phase_sharpe`.

## Finding 4 (HIGH) — ICIR annualised with the wrong frequency

**Files**: `qts/pipeline.py:180-182` (`bt.weights.index[::H]` → one IC per 20 trading days),
`qts/alpha/evaluation.py:64` (`icir = mean/sd*sqrt(periods_per_year)` with default 252).

The reported ICIR is `mean/sd·√252` on a series that has 252/20 ≈ 12.6 observations per year, so
it is overstated by √20 ≈ 4.47. `t_stat` (uses `len(ic)`) is unaffected. **Fix**:
`ic_report(scores, target.reindex(scores.index), periods_per_year=252 // H)`.

## Finding 5 (HIGH) — delisting outcomes never reach the labels

**File**: `qts/alpha/targets.py:35-37` (`md.close.shift(-h) / md.close - 1`).

If a stock's last close is at `d`, labels for `d-h+1 … d` are NaN and are removed by `y.notna()`
in `trainer.py` and by `rank_ic_series`. The alpha layer is therefore trained and scored on a panel
that omits exactly the observations where shorts pay (and where longs blow up). On the synthetic
market there are no delistings so this is invisible.

**Verification** (`/tmp/audit/v8_dead.py`, 30 names delisted at date 400):
`labels for dead names in the 20 days before delisting: 0 non-NaN of 630`.

**Fix.** Compute the forward return to the last available close within the window and treat the
remaining days as cash (or apply a delisting-return panel when available):

```python
def forward_returns(md, horizon, delist_return: Optional[pd.Series] = None):
    c = md.close
    last = c.ffill()                                   # last traded price carried forward
    fwd = last.shift(-horizon) / c - 1.0               # NaN only where c itself is NaN at t
    if delist_return is not None:                      # e.g. CRSP dlret / Sharadar
        ...apply at the delisting date...
    return fwd.where(c.notna())
```

Note `ffill` of `close` must not be reused for features (it is a label-only helper).

## Finding 6 (HIGH) — Sharadar asset growth can use a restatement filed after `datekey`

**File**: `qts/data/loaders.py:521-528`.

`prev` is de-duplicated on `(ticker, calendardate)` with `keep="last"` after sorting by `datekey`,
so the prior-year assets used for a row filed on `datekey_i` may come from a later filing (10-K/A,
restated comparatives in a later 10-K) with the same `calendardate`.

**Verification** (`/tmp/audit/v6_sharadar.py`):

```
asset_growth visible on 2021-02-16: -0.2  (PIT-correct would be 0.20)
```

**Fix.** Use an as-of merge on the filing date:

```python
prev = df[["ticker", "calendardate", "datekey", "assets"]].rename(
    columns={"calendardate": "_prev_cd", "datekey": "_prev_dk", "assets": "_assets_prev"})
cur = df.assign(_prev_cd=df["calendardate"] - pd.DateOffset(years=1)).sort_values("datekey")
prev = prev.sort_values("_prev_dk")
df = pd.merge_asof(cur, prev, left_on="datekey", right_on="_prev_dk",
                   by=["ticker", "_prev_cd"], direction="backward")   # latest prior-year value filed <= datekey
```

The same `keep="last"` pattern on line 517 is correct (same-day filings), and `pit_forward_fill`
itself is PIT (verified by `test_pit_forward_fill_never_leaks`).

## Finding 7 (HIGH) — DSR inputs are a constant, not the ledger

**File**: `qts/pipeline.py:192-195`.

`var_trials_sr_annual=0.25` (sd 0.5 annual) is a prior unrelated to the ledger. Bailey & López de
Prado's `V` is the variance of the trials' SR *estimates*; under the null it is ≈ 252/n annualised
(0.25 for n ≈ 1 000 days, 0.05 for 20 years) plus the dispersion of true SRs across trials, less
whatever is removed by correlation between trials. `TrialLedger.dsr_for` already computes the
empirical variance and `effective_trials` the Nyholt count; neither is used. `n_trials` counts every
record of every family (conservative) but every demo re-run of the same config adds one, so the
DSR of an unchanged strategy drifts down with the number of times the script was run.

**Hand check** (`/tmp/audit/v3_dsr_pbo.py`): the DSR/PSR/E[max]/MinTRL formulas themselves match
Bailey–LdP (se, PSR and MinTRL reproduce hand computations to machine precision; E[max SR] for
N=100, V=1 = 2.53 vs the paper's ≈2.5; `kurt=net.kurt()+3` is correctly raw).

**Fix.**

```python
srs = self.ledger.trial_sharpes_daily() if self.ledger else pd.Series(dtype=float)
n_eff = max(len(srs) + 1, 1)
floor = 1.0 / len(net)                                   # sampling-noise variance of a daily SR
var_d = max(float(srs.var(ddof=1)) if len(srs) >= 3 else 0.0, floor) if n_eff > 1 else 0.0
dsr = deflated_sharpe_ratio(deannualize_sharpe(summ["sharpe_net"]), n_eff, var_d, len(net), skew, kurt)
```

and log `data_hash=store.snapshot_hash_all()` / `code_version=git sha` so identical re-runs can be
de-duplicated (`family` + `data_hash` + `config`).

## Finding 8 (MEDIUM) — synthetic oracle: fair in the PIT sense, optimistic in structure

**File**: `qts/data/synthetic.py:97-99` (`z = z_m[month_of_day]`), `:119` (`per_day_alpha … * z`,
same `t`), `:146` (`base = expo + w_alpha * z + noise`).

The comment on line 117 says the score is "lagged so z at t predicts the next H days"; it is not
lagged. `z_m` is constant over the 21 days of month `m`, drives `r_t` from the first day of the
month and is simultaneously visible in B/P, E/P, GP/A and asset growth. Consequences:

* No look-ahead: features at `t` use only `z` as of `t`, and `AlphaModel.as_of(t)` is bit-identical
  under `md.slice_until(t)` (`/tmp/audit/v1_asof_pit.py`, all diffs 0.0, with and without the ridge
  ML sleeve). The oracle is **fair**.
* It is optimistic relative to real fundamentals (which arrive after the period they describe):
  `IC(z_t, past 20d return) = 0.025 ≥ IC(z_t, next 20d) = 0.021`.
* The forward IC depends on the day of the month (bucket 0: 0.0247; buckets 1–2: ≈0.019), a
  calendar regularity that a flexible model could latch onto and that does not exist in real data.
* `ic_true=0.04` is not the achievable IC: raw and residual IC of the *true* score is ≈0.021 here
  (0.03 in the test fixture); the "recovers ic_true and not more" check is implemented as a wide
  band (0.025–0.055).
* REVIEW_NOTES' "planted-alpha backtest (gross 1.6 / net 1.3 with the true latent score)" used a
  score that knows the current month's state from its first day — a plumbing check, not an
  achievable number.

**Fix.** Lag the observable (`obs(k, w_alpha)` should use `z_m[month_of_day - 1]` or a
`characteristic_lag_days` shift), let `z` evolve daily (AR(1) with daily phi), and compute the
planted IC analytically from the generated panel (`rank_ic_series(z, resid_target).mean()`) as the
test oracle instead of `ic_true`.

## Finding 9 (MEDIUM) — dead names are ranked and scored

**Files**: `qts/alpha/features.py:181`, `qts/alpha/composite.py:187-203`, `qts/alpha/alpha_model.py:119`.

Names with NaN close get NaN price features → zero after `fillna(0.0)`, but `industry_momentum`
(broadcast of the group mean) and `strev_21`'s group-demean give them non-zero values, so they are
not even at an exact median. `as_of` returns a score for them (verified: 30 of 100 dead names
scored, mean 0.02, sd 0.10). The simulator zeroes their expected return, but the cross-sectional
z-score and the theme decile portfolios have already been computed with them in the sample.
Fix as in Finding 1 (drop rows where `close`/`market_cap` is NaN before ranking).

## Finding 10 (MEDIUM) — PBO degenerate cases

**File**: `qts/validation/pbo.py:113-116`.

* `logits <= 0.0` classifies the exact OOS median (`w = 0.5`) as overfit. For N odd the IS-best
  lands on the median configuration with probability ≈1/N under the null, so PBO ≈ 0.5 + 0.5/N
  instead of 0.5. Verified: N=3 pure noise → PBO 0.657; identical columns → logits all 0 → PBO 1.0;
  all-zero returns → PBO 1.0.
* `np.nan_to_num(M, nan=0.0)` silently converts missing days into zero returns (changes Sharpes).

**Fix**: `pbo = float(np.mean(logits < 0.0))` (strict, as in Bailey et al. — "underperforms the
median"), `raise`/`warn` when `N` configs have fewer than, say, 2 distinct columns, and reject NaNs
(`raise ValueError("perf_matrix has NaNs; align the trials first")`).

## Finding 11 (MEDIUM) — the pipeline's suspect gate is untested against an oracle

**File**: `qts/validation/leakage.py:148`; `tests/test_validation.py::test_planted_oracle_is_flagged`.

The test passes `backtest_fn=None`, so the only thing shown is that a toy quantile backtest of the
future return has Sharpe > 2. Nothing demonstrates that an oracle injected into `FeatureBuilder`
(e.g. via `md.extra`) propagates through `process → composite → optimizer → simulator` and trips
`notes`/`suspect`. Add an `extra_features` hook to `FeatureBuilder.build` and a slow test that runs
`ResearchPipeline(use_ml=False)` with `md.extra["oracle"] = forward_return(md, 20)` added to the
value theme, asserting `hurdles["gates"]["suspect"] is False`.

## Finding 12 (MEDIUM) — mixed target definitions inside the first training window

Covered by Finding 2: with `train_min_days=756` and `min_history_days=252`, the first ML fold's
labels are 1/3 theme-residualised and 2/3 risk-only-residualised; the per-date z-scoring hides the
regime change from the model. Fix as in Finding 2.

## Findings 13–17 (LOW)

13. `qts/alpha/models.py:273` — `MLPRegressor(early_stopping=True, validation_fraction=0.1)` draws a
    random 10 % holdout from rows whose 20-day labels overlap the fit rows; stopping is optimistic
    but the trainer's purge keeps OOS predictions clean. Prefer `max_iter` chosen on the trainer's
    `valid_mask` block (as `LGBMModel` does).
14. `qts/data/loaders.py:242,258` — `BDay(8)` ignores NYSE holidays and FINRA's 4 pm ET release; for a
    close-to-close strategy the first *usable* close is the next trading day. Parametrise with the
    published dissemination calendar and add `+1` trading day.
15. `qts/alpha/evaluation.py:109` — `_nw_tstat(lags=h-1)`; with a Bartlett kernel the MA(h−1)
    autocovariance is under-weighted. Use `lags=int(1.5*h)` or report both.
16. `qts/data/loaders.py:62` / `qts/data/store.py:212` — add a `max_staleness_days` to
    `pit_forward_fill` (fundamentals older than ~15 months → NaN) and document that
    `as_market_data` expects already-filled panels.
17. `qts/pipeline.py:204-209` — empty `data_hash`/`code_version`; see Finding 7.

---

## Checked and found correct

Alpha / features (`qts/alpha/features.py`)
* All 18 panels use `shift(k ≥ 0)` / trailing `rolling` only; `_trailing_return(start, end)` is
  `close_{t-end}/close_{t-start}`; `mom_change = mom_12_1 − mom_12_1.shift(63)`;
  `industry_momentum` is a cap-weighted mean of the (lagged) 6-1 momentum, not of contemporaneous
  returns; `rolling_market_beta` is a trailing OLS; `max_ret_21`, `turnover_63`, `amihud_63`,
  `log_dollar_vol_63` trailing; `strev_21` industry-demeaned per date.
* Winsorisation (`_winsorize_rows`), industry demean, `rank_gauss` and `zscore` are all per-row
  (per-date) — no full-sample or expanding statistics anywhere in `process`; `fillna(0.0)` is a
  constant, not a sample median. `tests/test_features_point_in_time` and the `pit_shift_test`
  tripwire cover raw and processed frames at one cut date; my own cut at a different date with
  `AlphaModel` (composite + ridge) gave zero difference.
* `momentum_crash_scalar_series` uses an *expanding* 80th percentile (`composite.py:121`) and
  trailing windows; `factor_momentum_tilt_series` rolling sums/sds; `momentum_vol_scalar_series`
  trailing 63-day vol.
* `theme_portfolio_returns` forms the portfolio on the score at `t`, earns `r_{t+1}` and indexes
  the result at `t+1` (verified against a hand computation); `dynamic_weights` therefore uses only
  theme returns realised on or before `t`, and `score()`'s single-date path slices both the
  processed frame and `theme_returns` at `date`.

Targets (`qts/alpha/targets.py`)
* `forward_returns` is `close_{t+h}/close_t − 1` at row `t`; winsorised and z-scored per date;
  exposures are those of date `t`; the external hook is called with `dates[i]` and handles
  collinear columns; WLS weights per date.

Trainer (`qts/alpha/trainer.py`)
* Purge is exact and one day stricter than required: for prediction start `p0` the last training
  label position is `p0 − h − 1` (gap 1); the validation block is purged from the fit rows the
  same way (`/tmp/audit/v2_purge.py`). Refit every 252 dates, expanding window, fresh model each
  fold, predictions only for `[p0, p1)`.
* `AlphaModel.trailing_ic` uses `oos_ic_` up to position `pos − h` inclusive (label realised at
  `pos`), `resid_vol_h_` is the trailing idio vol, `signal_half_life` sees scores `≤ date`, and
  `_combine_history` / `zscore(ml)` are row-wise. `as_of(t)` is identical when fitted on
  `md.slice_until(t)` (scores, expected returns, `ic_estimate`, half-life, ML panel, OOS IC series).

Evaluation (`qts/alpha/evaluation.py`)
* `rank_ic_series` aligns score at `t` with the label at `t`; `decile_spread` masks scores where the
  forward return is NaN, annualises by `periods_per_year/horizon` and uses NW; `ic_decay_curve`
  recomputes the target per horizon; `rank_autocorrelation` correlates `score_t` with
  `score_{t−lag}`; `signal_half_life` fits through the origin on `rho > 0.02`; `residualize_on` is a
  per-date OLS with intercept.

Validation (`qts/validation/*`)
* `purge_train`: left bound `t0 − h` (any `s` with `s + h ≥ t0` dropped), right bound
  `t1 + h + embargo`; `PurgedWalkForward` is strictly forward-only; `PurgedKFold` and CPCV purge both
  sides; `check_no_overlap` passes for all three; `backtest_paths` greedy fill gives
  `C(N,k)·k/N` complete paths.
* DSR: `se(SR) = sqrt((1 − γ₃SR + (γ₄−1)/4·SR²)/(n−1))` with raw kurtosis; E[max SR] with
  Euler–Mascheroni weights; `deflated_sharpe_ratio_annual` divides the annual variance by 252;
  MinTRL and MinBTL match Bailey et al.; `moments_from_returns` uses `kurtosis(fisher=False)`.
* `TrialLedger.effective_trials` is Nyholt's `1 + (M−1)(1 − Var(λ)/M)` with sample variance
  (identity → M, ρ=0.9 on 5 → 1.76); ledger is append-only; `dsr_for` restricts to a family.
* `label_overlap_check`, `pit_shift_test` (catches `shift(−k)`, centred windows, full-sample
  statistics at the boundary), `feature_registry` AST check (documented limits),
  `nagel_collapse_test` (train/test gap of `horizon`, IC t-stat on non-overlapping dates).
* Hurdle thresholds (t > 3, incremental IC 0.005, Δ-Sharpe 0.05, 2× breakeven, DSR 0.95,
  PBO 0.5, phase Sharpes 0.3/0.5, suspect > 2) match plan §2.4; only the pipeline's inputs are wrong
  (Finding 3).

Data (`qts/data/*`)
* `raw_eligibility`: trailing 63-day ADV, per-date size quantile on already-eligible names,
  per-date rank; hysteresis state machine uses only past refreshes and is forward-filled;
  `mask &= close.notna()`.
* `pit_forward_fill` snaps to the first panel date `≥ available_date` and never shows a value
  earlier; FINRA `to_short_interest_ratio` applies the lag *before* forward-filling and divides by
  shares outstanding as of each date; Sharadar panels are keyed on `datekey (+lag_days)`, the
  `MR*` warning is present, and `book_to_price`/`earnings_yield` use the PIT close when supplied.
* Ken French parser: `/100`, `−99.99/−999` sentinels → NaN, 8/6/4-digit date handling, block
  selection.
* `ParquetStore` round-trips wide panels sparsely and reindexes every panel to the close index.
* `pipeline.backtest_dates` starts after both the risk warm-up and the first OOS ML date and drops
  the last day; `RISK_ONLY_STYLES` correctly excludes momentum/value/quality/investment/short-interest
  from the hook; the IC report samples scores every `H` days so its t-stat is on non-overlapping
  observations.

## Overall verdict on leakage risk

The time-series mechanics of the alpha layer are clean: every feature, the dynamic theme weights,
the walk-forward purge/embargo and `AlphaModel.as_of` are point-in-time, and I could not make the
model's output at `t` depend on data after `t` on a full-ticker panel. The leakage that remains is
**cross-sectional and structural** rather than temporal: (i) the alpha cross-section at `t` is
assembled from tickers chosen with knowledge of future universe membership and keeps dead names,
which on real data biases training, IC and the tradable set in a return-correlated way; (ii) the
labels drop every delisting outcome; (iii) the library-default target projects the value and
momentum themes out, and the pipeline falls back to it silently; and (iv) one PIT loader
(Sharadar asset growth) can read restated history. None of these is visible on the shipped
synthetic market, which has a static universe, no delistings and an un-lagged planted score, so
the 36 green tests are not evidence against them. Until Findings 1, 2, 5 and 6 are fixed and the
hurdle inputs (Findings 3, 4, 7) are corrected, numbers from `run_real.py` should not be used for a
go/no-go decision; the synthetic numbers remain a plumbing check only.
