# Audit: risk model (`qts/risk`) and portfolio construction (`qts/portfolio`)

Date: 2026-11-04. Scope: `qts/risk/*`, `qts/portfolio/*`, `qts/core/types.py`,
`qts/core/config.py`, cost-model consistency with `qts/tcost/model.py`, and the
wiring in `qts/pipeline.py` / `qts/backtest/simulator.py` that decides which code
paths actually run. No source file was modified. Verification scripts live in
`/tmp/audit/v1..v10_*.py` (outputs reproduced below).

`python -m pytest tests/test_risk.py tests/test_portfolio.py -q -p no:warnings`
-> **37 passed in 12.8 s**. Note that `tests/test_portfolio.py` runs against the
`StandInRiskModel` in `tests/conftest.py` (factors `beta, size, IND..`, no
`market` column), which is why findings 1 and 4 below are invisible to the suite.

Severity definitions: CRITICAL = wrong sign/scale/look-ahead that would mislead
decisions; HIGH = wrong number, direction-preserving; MEDIUM = fragility or
silent fallback; LOW = style / documentation.

## Summary table

| # | Sev | File:line | Finding |
|---|---|---|---|
| 1 | CRITICAL | `qts/portfolio/optimizer.py:44,95-99,175-179` | `beta_neutral` binds the **`market` (country) factor** – i.e. `|1'w| <= 0.05`, a duplicate of the net band – not the `beta` style. With the real risk model the "beta-neutral" book carried a beta-style exposure of 2.2 in a direct test. |
| 2 | HIGH | `qts/portfolio/optimizer.py:253-281` | Vol-target scale-up multiplies the book by `s` after checking only gross, per-name cap and (incorrectly) ADV participation. Net band, beta/style bands, industry bands and the participation cap for sign-flipping names are all violated after scaling (verified: net 0.085 vs band 0.05, beta style −0.13 vs band 0.10, 7 names at up to 1.8× the ADV cap). |
| 3 | HIGH | `qts/portfolio/optimizer.py:186` | Industry-band columns are auto-detected by the prefix `"IND"`. Synthetic labels are `IND00..`, but the real-data path (Fama-French-48 names such as `Banks`, `Other`) matches nothing; neither `pipeline.py` nor `simulator.py` passes `industry_columns`, so industry bands will silently switch off on real data. |
| 4 | MEDIUM | `qts/risk/covariance.py:412-416` | USE4 thin-history blend: robust sd is `1.35*IQR` (should be `IQR/1.35`) and `Z = sd_eq/sd_robust` (should be `|sd_eq/sd_robust - 1|`). The two errors partly cancel; outlier damping is weaker than USE4 for fat-tailed names. |
| 5 | MEDIUM | `qts/risk/alignment.py:213-218`, `qts/portfolio/optimizer.py:255-256,297` | `nu` is an uncalibrated heuristic (`N * median style var`) and enters `expected_risk` and the vol-target scalar. On an orthogonal alpha it made the augmented variance 4.7× the model variance; on the pipeline alpha its share was ~0 %. There is no bias-statistic calibration loop as Saxena–Stubbs prescribe. |
| 6 | MEDIUM | `qts/portfolio/optimizer.py:206-231` | Solver fallback: (a) if the ADV-participation problem is infeasible the caps are dropped entirely (not relaxed), (b) a failed/infeasible solve returns `w_current` with `expected_cost=0`; the simulator only counts it. Both are logged, not raised. |
| 7 | MEDIUM | `qts/portfolio/optimizer.py:123-136` | Universe = union(alpha, w_current, risk). Names absent from the risk model get zero factor exposure and median specific variance, so they look purely idiosyncratic to the optimizer. Harmless in the pipeline (same universe), dangerous for API users. |
| 8 | MEDIUM | `qts/portfolio/alpha_scaling.py:403-423` | GP shrinkage renormalises sleeve weights to sum to 1, so the *level* of alpha is never shrunk as holding period rises (GP shrinks magnitude). Direction is right; not used by `pipeline.py` today. |
| 9 | LOW | `qts/risk/factor_returns.py:229-230`, `exposures.py:11-16` | Docstring says the market factor "equals the cap-weighted market return". With √cap regression weights, cap constraint weights and cap-weighted style standardisation it is only approximate (0.00492 vs 0.00461 on the test day; the √cap-weighted `size` mean was −0.70). Beta is regressed on the equal-weighted market (USE4: cap-weighted). |
| 10 | LOW | `qts/portfolio/costs.py:150-170` vs `qts/tcost/model.py:160-190` | `SimpleCostModel` "auction" = full ADV + half-spread; `TransactionCostModel` "auction" = 9 % of ADV + no spread (1.75× apart on the same trade). The pipeline uses `TransactionCostModel` for both optimizer and simulator, so production is consistent; tests/README snippets mixing the two are not. |
| 11 | LOW | `qts/risk/covariance.py:114-121,198-203` | VRA `B_t` averages over factor columns that are identically 0 (empty industries), biasing `lambda_F` slightly down; eigenfactor simulation truncates history at `4*corr_half_life` and omits USE4's `a=1.2` inflation, so it under-corrects (v ≈ 1.03–1.05 where the true small-eigenvalue bias was 1.1–1.4). |

## Findings in detail

### 1. CRITICAL — `beta_neutral` constrains the country factor, not beta

`qts/portfolio/optimizer.py:44` defines `BETA_FACTOR_NAMES = ("beta", "market", "mkt")`
and `_find_beta_column` (lines 95-99) returns the *first factor in column order*
whose lowercase name is in that tuple. `qts/risk/exposures.py:179` lays out
columns as `[market] + industries + styles`, so for every `BarraStyleRiskModel`
output the function returns `"market"`. Because the market factor has unit
exposure for every stock, line 179 becomes `|1'w| <= beta_band`, which is just
the net-exposure band already imposed at lines 168-169. The `beta` style is
left unconstrained. Worse, line 193 then removes `beta_col` (= `market`) from
`neutral_styles`, so a caller who passes `neutral_styles=["beta"]` gets the
`style_band_sd = 0.10` band on beta (the pipeline path) while a caller using the
README snippet gets nothing.

Verification (`/tmp/audit/v1_beta_col.py`, 150-stock synthetic market, alpha
tilted toward high beta, `beta_neutral=True`, `beta_band=0.05`):

```
factor order: ['market', 'IND00', 'IND01'] ... ['quality', 'investment', 'short_interest']
beta col detected: market
status optimal gross 2.000 net -0.0500
beta-style exposure B_beta'w = 2.2073   (band would be 0.05 if constraint worked)
market exposure 1'w = -0.0500
binding: {'gross': 1, 'net_lo': 1, 'max_position': 12, 'beta_neutral': 1, 'industry_band': 9}
```

And in the pipeline configuration (`/tmp/audit/v9_align_pipeline.py`, real
`AlphaModel` composite, `neutral_styles` = risk-only styles): beta-style exposure
−0.130 after vol scaling (band 0.10; see finding 2), `1'w = 0.085`.

Impact: the spec (BUILD_PLAN §5 hard constraints: "Beta-neutral") is not met
anywhere; in the pipeline the beta tilt is bounded only by the generic 0.10 sd
style band and in direct use not at all. `test_beta_neutral_skipped_without_beta_factor`
passes only because the stand-in risk model has no `market` column.

Fix (`optimizer.py:44,95-99`):

```python
BETA_FACTOR_NAMES = ("beta", "market_beta", "mkt_beta")   # never the country factor

@staticmethod
def _find_beta_column(factors):
    low = {str(c).lower(): c for c in factors}
    for name in BETA_FACTOR_NAMES:
        if name in low:
            return low[name]
    return None
```

and add a regression test that runs the optimizer against
`BarraStyleRiskModel` output and asserts `|B_beta' w| <= beta_band`. Also
consider expressing beta neutrality in *raw* beta units: the standardised style
has cap-weighted mean 0 / sd 1, so `B_beta'w` is in sd units and the raw beta is
`beta_bar * 1'w + sd_beta * B_beta'w`; with a dollar-neutral book the style
constraint is adequate, but document it.

### 2. HIGH — vol-target scale-up breaks constraints the optimizer enforced

`optimizer.py:259-275` computes `max_up` from gross, per-name cap and an ADV
bound, then line 278 multiplies `w_t` by `scalar` up to `max_up`. Three problems:

* Every *linear* band that scales with `w` (net exposure lines 168-169, beta
  line 179, industry line 192, style line 197, extra constraints) is simply
  multiplied by `s`. The optimizer legitimately sits on those bounds; scaling
  then exceeds them.
* The participation bound at line 271, `s <= (cap + |w0|)/|w|`, is only valid
  when `w` and `w0` have the same sign. If the trade flips sign the correct
  bound is `(cap - |w0|)/|w|`.
* `pos` at line 269 excludes names with `w0 != 0, w == 0`; fine, but names with
  `|w| > 0` and `dw ≈ 0` (untouched) are also excluded although scaling makes
  `dw = (s-1) w` non-zero.

Verification (`/tmp/audit/v5_scaleup.py`, ADV cap 1 % NAV, current book with
opposite sign to alpha; `/tmp/audit/v9_align_pipeline.py`, pipeline config):

```
status optimal scalar 3.601
participation violations after scale-up: 7 of 150, max |x|/cap = 1.800
---
status optimal scalar 2.722
beta style exposure -0.130 (style band 0.10), 1'w 0.0847 (net band 0.05)
```

Scale-down (`s < 1`) still works: `portfolio_vol_scalar` clips to `[0, max_up]`
with `max_up >= 1` (`vol_target.py:454-455`, `test_vol_target_scales_down`).

Fix: do not scale post hoc. Either (a) implement the target as a second solve
that *projects* the levered book onto the feasible set —
`minimize ||w - s*w_star||^2` subject to the identical constraint list including
`participation` — or (b) include `max_up` bounds for every linear constraint
(`s <= band / |c'w_star|` for each row of each band, and the sign-correct
participation bound `s <= (cap ∓ |w0|)/|w|`). Option (a) is ~10 lines with the
existing `cons`/`participation` objects and is exact.

### 3. HIGH — industry bands silently disabled on real data

`optimizer.py:186`: `ind_cols = industry_columns or [c for c in factors if str(c).upper().startswith("IND")]`.
`qts/data/synthetic.py:86` names industries `IND00..IND11`, so tests and the demo
exercise the band. `qts/data/loaders.py:441ff` maps SIC to Fama-French-48 short
names (`Agric`, `Banks`, `Other`, ...); none starts with `IND`, and the
`UNKNOWN` label from the yfinance loader does not either. `grep` shows no caller
passes `industry_columns` (`pipeline.py`, `simulator.py`, `scripts/`). On the
Phase-0 real-data run the "|industry exposure| ≤ 0.03" hard constraint from
BUILD_PLAN §5 will vanish with no note.

Fix: derive industry columns from the risk model, not from a name prefix:
`from qts.risk.exposures import split_factor_columns; _, ind_cols, _ = split_factor_columns(factors, risk_cfg)`
(requires passing `RiskConfig` or storing the split on `RiskModelOutput`, e.g.
a `factor_groups: Dict[str, List[str]]` attribute set in `model.as_of`). At
minimum, append a note to `notes` when `cfg.industry_band > 0` and
`ind_cols == []`.

### 4. MEDIUM — USE4 thin-history blend uses the wrong robust sd and Z

`covariance.py:412-416`:

```python
sd_robust = (1.35 * (q75 - q25)).replace(0.0, np.nan)   # USE4: (Q3-Q1)/1.35
Z = (sd_eq / sd_robust).fillna(1.0)                       # USE4: |sd_eq/sd_robust - 1|
gamma = ((n_obs - 60) / 120.0).clip(0.0, 1.0) * np.minimum(1.0, np.exp(1.0 - Z))
```

For a Gaussian name the code gives `Z ≈ 1/1.82 = 0.55`, `exp(0.45) -> 1` (no
damping, correct by accident). For a name whose equal-weight sd is 4× its robust
sd USE4 gives `exp(1-3) = 0.135`; the code gives `Z = 4/1.82 = 2.2`,
`exp(-1.2) = 0.30` — the structural model gets ~half the weight it should.
Direction preserved, magnitude wrong; the structural blend is a second-order
input, hence MEDIUM.

Fix:

```python
sd_robust = ((q75 - q25) / 1.35).replace(0.0, np.nan)
Z = (sd_eq / sd_robust - 1.0).abs().fillna(0.0)
```

### 5. MEDIUM — alignment variance `nu` is uncalibrated and feeds the vol target

`alignment.py:216`: `nu = nu_scale * N * median_style_var`. The √N argument in
the docstring is coherent (a unit-sd style has `||x||² ≈ N`), but Saxena–Stubbs
calibrate `nu` so that the bias statistic of the alignment portfolio is ≈1;
nothing here does that, and `cfg.alignment=True` is the default. Because
`RiskModelOutput.portfolio_variance` includes the term by default,
`expected_risk` (`optimizer.py:256`) and therefore the vol-target scalar are
driven by it.

Verification (`/tmp/audit/v4_voltarget_align.py`): `||y|| = 1`,
`max|B'y| = 3e-14` (orthogonality exact). With a random alpha (fully
orthogonal to the factors) the un-aligned optimum had model variance 1.58e-5
but augmented variance 7.46e-5 (4.7×); the aligned solve cut `y'w` from 0.215 to
0.071 and `alpha'w` by 65 %. So the mechanism works as designed (augmented
variance of the optimal portfolio rises; the optimizer backs off the invisible
direction). In the pipeline configuration (`v9`) the alignment share of the
optimized book's variance was ~0 % with `1-R² = 0.097`, so today the term is
nearly inert — which also means it is untested on the book it is meant to fix.

Fix: add a calibration hook — run `portfolio_bias_test` on the *optimized* book
with `include_alignment=False` vs `True` over the warm-up window and choose
`nu_scale` so the aligned bias statistic is inside `bias_band(T)`; log the
alignment share of ex-ante variance in `OptimizationResult.notes` every day.

### 6. MEDIUM — silent fallbacks in the solve path

`optimizer.py:208-218`: if the problem is infeasible with the ADV participation
constraint, it is re-solved with the constraint *removed*. That can produce
trades of arbitrary size relative to ADV and is reported only through
`notes`. Lines 221-231: any non-optimal status returns `w_current` with
`expected_cost = 0.0`, `turnover = 0`, `transfer_coefficient = nan`; the
simulator (`simulator.py:246`) increments `n_failed` and keeps going. Also
`_try_solvers` (line 318) returns on the first solver that says `infeasible`
rather than trying the next, which is fine for genuine infeasibility but not
for solver-specific numerical infeasibility.

Fix: relax participation progressively (`2x, 4x` the cap) before dropping it;
raise or at least surface `status != optimal` in `BacktestResult.summary()`
(count and dates); treat `optimal_inaccurate` as a warning.

### 7. MEDIUM — out-of-model tickers are modelled as pure idiosyncratic risk

`optimizer.py:123-136`: `B.reindex(tickers).fillna(0.0)` and
`d.fillna(d.median())`. A name present in `alpha` but not in `risk.tickers`
(delisted, newly listed, or any API misuse) has zero systematic risk in the
objective and therefore attracts weight. Prefer `tickers = risk.tickers`
intersect (`alpha` ∪ `w_current`), zero the alpha of out-of-model names, and
add a note when any were dropped.

### 8. MEDIUM — GP shrinkage direction correct, magnitude not shrunk

`alpha_scaling.py:376-381,403-423`: weights `1/(1 + H/hl_k)` are the right
first-order GP shape (faster signals, i.e. smaller `hl_k`, are shrunk more as the
cost-implied holding period `H` rises). But the weights are renormalised to sum
to one, so a single-sleeve alpha is never shrunk and a multi-sleeve alpha keeps
the scale of "one sleeve" regardless of `H`. GP's aim portfolio shrinks the
level of each signal; the level matters because it sets the alpha/risk/cost
trade-off. `grep` shows `garleanu_pedersen_shrink` is not called by
`pipeline.py` or `alpha_model.py`, so this is currently latent.

Fix: return `combine_in_forecast_space(alphas, gp_raw * prior / prior.sum())`
(no renormalisation of the GP factor), and document that the IC used in
`grinold_alpha` is then the per-sleeve IC.

### 9. LOW — documentation overstates the market-factor identity

`factor_returns.py:229-230` and `exposures.py:11-16`. The normal equation for
the unit column gives `f_market = Σ w_i r_i` only when the regression weights,
the constraint weights and the style-standardisation weights coincide. Here they
are √cap / cap / cap. Verified (`/tmp/audit/v7_market_factor.py`):

```
market f = 0.004920 ; cap-weighted mkt = 0.004607 ; sqrt-cap-weighted mkt = 0.005252
constraint sum: 0.0
sqrt-cap-weighted style means: [0.418 -0.696 -0.138 0.062 0.135 0.049 -0.104 -0.054 0.015 -0.226]
max |B' W u| = 5.7e-17
```

The constraint holds exactly and residuals are W-orthogonal to all exposures
(regression correct); only the interpretation line is wrong. Fitted systematic
returns `B f` are identification-invariant, so forecasts are unaffected.
Change the docstring to "approximately", or standardise styles with the same
weights as the regression if an exact country factor is wanted. Separately,
`_rolling_market_regression` uses the equal-weighted market; USE4 uses the
cap-weighted market. Either is defensible; say which and why.

### 10. LOW — two cost models disagree for the auction venue

`/tmp/audit/v10_costs.py`, identical trade vector, `nav = 5e6`:

```
auction:    optimizer trade cost  tcost=0.000266 simple=0.000152 | estimate() tcost=0.000266 simple=0.000153
continuous: optimizer trade cost  tcost=0.000229 simple=0.000230 | estimate() tcost=0.000229 simple=0.000231
hold: identical (2.902e-06)
```

Each model is internally consistent between its convex term and its point
estimate (same units: fraction of NAV, `x` in weights, `|x|^{1.5}/sqrt(V/nav)`);
`SimpleCostModel` ignores the auction-volume share and charges the half-spread
in the auction, `TransactionCostModel` does the opposite. `pipeline.py:114`
passes the same `TransactionCostModel` to the optimizer and the simulator, so
production sees one cost function. Make `SimpleCostModel` use
`auction_volume_share_default` and zero spread for the auction, or state in its
docstring that it is a test stub.

### 11. LOW — VRA and eigenfactor details

`factor_vra_multiplier` (`covariance.py:103-121`) matched a hand-rolled loop to
1e-11 and responded correctly to regime shifts (`/tmp/audit/v8_vra_eigen.py`:
constant vol 1.03, last-60-day vol ×2 → 1.42, ×0.5 → 0.77); `lambda_F²` scales
the covariance (line 209). Columns that are identically zero (industries with no
members) contribute `0/σ²` terms to the cross-sectional mean and bias `B_t`
down. Eigenfactor adjustment: rotation back is correct (`U diag(v² D) U'`), the
simulation length `min(T, 4*corr_half_life)` is reasonable, but on a 20-factor
test the raw bias of the five smallest eigenvariances was 1.1–1.4 while the
estimated `v` was 1.02–1.05, i.e. it under-corrects; USE4's empirical `a = 1.2`
inflation exists for this reason. Consider `v = 1 + a (v - 1)` with `a = 1.2`
or 1.4 and a parabola fit across `k`.

## Verified as correct

* **Exposures** (`exposures.py`): all descriptors are rolling/lag-only; the
  `exposure_history` slice at `t` uses only rows `<= t` (test
  `test_exposure_history_matches_single_date_and_is_point_in_time`); no
  full-sample statistics are used anywhere (`zscore` is row-wise). Beta at `t`
  uses returns through `t` inclusive; exposures dated `t` are then paired with
  returns at `t+1` in `estimate_factor_return_history` (`B_{t-1}`,
  `cap_{t-1}`, `r_t`). Standardisation is cap-weighted mean 0 / equal-weighted
  sd 1, clip ±3, re-standardise, fill 0 — matching USE4. Industry dummies are
  0/1 rows summing to 1. Reversal/momentum/size/liquidity signs are conventional
  and sign is irrelevant for a risk factor.
* **Factor returns** (`factor_returns.py`): WLS weights ∝ √cap of `t-1`;
  null-space reparameterisation enforces `Σ c_j f_j = 0` exactly (1e-17);
  residuals are W-orthogonal to `B`; empty industry columns are dropped and
  reported as 0; weighted R² is standard.
* **`ewma_cov`**: EWMA weights normalised; Newey–West applies `(1 - l/(L+1))`
  to `Γ_l + Γ_l'`; brute-force loop agrees to 2e-16; `ewma_nw_variance`
  agrees with the diagonal to 7e-16; PSD clipping applied.
* **`factor_covariance`**: `D R D` combination with 84d/504d half-lives;
  eigen-adjust before VRA; VRA scales by `λ²`; daily units throughout.
* **`nonlinear_shrinkage`**: matches the published QIS.m step by step
  (centering with `n-1`, `h = min(c², 1/c²)^0.35 / p^0.35`, `Lj[i,j] = 1/λ_j`,
  `θ`, `Hθ`, `δ`, singular-case `δ₀`, trace preservation). On simulated data
  with known Σ: Frobenius error vs sample covariance 7.4 vs 10.8 (N=100,T=200),
  10.2 vs 24.3 (p>n: N=120,T=60), 15.2 vs 28.8 (N=300,T=252); min eigenvalue
  > 0 in all cases. (A deliberately transposed kernel was *worse* than the
  sample covariance — confirming the orientation in the code is the right
  one.)
* **`specific_risk`**: EWMA+NW per name with renormalised weights over observed
  days; structural model (log σ on exposures, √cap weights, `E0` level
  correction); Bayesian shrinkage `v = q|σ-σ̄|/(Δ + q|σ-σ̄|)`,
  `σ_SH = v σ̄ + (1-v) σ` with cap-weighted mean and equal-weighted sd per size
  decile — hand computation agrees to 0.0; specific VRA cap-weighted and applied
  as `(λ_S σ_SH)²`.
* **Alignment**: `y` is the OLS-orthogonal component of exactly the alpha the
  optimizer receives (`simulator.py:239-243` passes the same `exp_ret` to
  `risk_fn` and to `solve`), unit norm, `B'y = 0`; augmented variance of the
  un-aligned optimum rises and the aligned optimum loads less on `y`.
* **`model.as_of`**: factor returns `.loc[:date]` (return realised at the
  close of `date`, known then), exposures as of `date`, market cap `.loc[:date]`,
  specific returns `.loc[:date]`; eigen-bias cache keyed on history length,
  refreshed every 21 observations; no `self.md` use beyond `date`.
* **Evaluation**: bias statistic = sd of `r_t/σ_{t-1}`; band `1 ± 2/√(2T)` with
  `T` = number of forecasts; forecast at `t`, realisation at `t+1`; QLIKE
  `r²/σ² - log(r²/σ²) - 1`; Mincer–Zarnowitz regresses realised `r²` on
  predicted `σ²`.
* **Optimizer objective units**: `alpha` is the H-day expected return
  (`AlphaModel.resid_vol_h_ = idio_vol_252 * sqrt(H)`, `alpha = IC σ_H z`);
  risk `γ H (f'Ff + Σ d w² + ν (y'w)²)` with daily variances; trade cost once;
  holding cost `H × (borrow/252)·neg(w)`. `||L'f||² + Σ d_i w_i²` equals
  `w'(BFB'+D)w` to 3e-21 with `L L' = F` to 7e-19; `F` is eigenvalue-clipped;
  `expected_cost` recomputes `tc + H·hc` on the final (scaled) trade.
* **Constraints as coded**: long/short gross `||w||₁ ≤ 2`, net band, per-name
  cap, ADV cap `0.05·ADV/nav` in weight units, 130/30 and long-only books,
  style bands relative to benchmark when given, no-trade equality, extras.
  Duals are read for binding-constraint reporting.
* **Transfer coefficient**: exact `a'w / sqrt(w'Σw · a'Σ⁻¹a)` for N ≤ 500,
  else the Clarke–de Silva–Thorley approximation; `effective_holding_days =
  gross / (two-sided turnover / 2)` consistent with its test;
  `realized_vs_exante_risk` and `brinson_like_attribution` (`t` weights × `t+1`
  factor returns) are correct.
* **Costs**: both models use fraction-of-NAV units with `x` in weights and the
  `|x|^{1.5}/sqrt(V/nav)` scaling; convex term and point estimate agree within
  each model.

## Verdict

The estimation core is sound: exposures, the constrained cross-sectional
regression, EWMA/Newey–West, the USE4 covariance pipeline, QIS shrinkage, the
specific-risk chain and the evaluation statistics all reproduce their reference
formulas numerically and respect point-in-time discipline. The portfolio layer,
however, does not currently deliver the constraints the plan and the config
promise: `beta_neutral=True` constrains dollar neutrality instead of beta
(finding 1), the vol-target lever-up step pushes the book outside the net,
style and ADV bounds the optimizer just enforced (finding 2), and industry bands
will silently disappear the moment real industry labels are used (finding 3).
None of these is a risk-forecast error, but each makes the realised book differ
from the one the risk model was asked to size, which is exactly the failure a
paper-trading phase is meant to catch early rather than late. Recommendation:
fix 1–3 (all small, localised changes), add the three regression tests against
`BarraStyleRiskModel` output, then the risk model + optimizer are fit for a
paper-trading phase; findings 4–8 can be scheduled behind it, with `nu`
calibration (5) the first to address once optimized-book bias statistics are
being collected.
