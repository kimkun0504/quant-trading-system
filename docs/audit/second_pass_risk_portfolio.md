# Second-pass audit: risk model and portfolio construction

Date: 2026-11-05. Scope: `qts/core/{stats,crosssection,types,config,roles}.py`, `qts/risk/*`,
`qts/portfolio/*`, `docs/notes/alignment_nu_calibration.md`, `scripts/calibrate_alignment_nu.py`,
and the wiring in `qts/pipeline.py` / `qts/backtest/simulator.py`. This is an independent
re-verification after the first audit (`risk_portfolio.md`) and two refactor waves. **No source
file was modified.** Scripts: `/tmp/audit2_b/v*.py` (shared fixture `common.py`: synthetic market
with 20 late universe entrants at day 400 and a dated reclassification of 10 names at day 350;
`fixture_pipe.py`: production `ResearchPipeline` on synthetic seed 0).

`pytest tests/test_risk.py tests/test_portfolio.py tests/test_risk_statistical.py tests/test_optimizer_perf.py`: all pass.

Severity: CRITICAL = wrong sign/scale/look-ahead that misleads decisions; HIGH = wrong number or
design defect on the default path; MEDIUM = fragility, silent hazard or invariant broken off the
default path; LOW = documentation / minor inconsistency.

## Summary

| # | Sev | File:line | Finding |
|---|---|---|---|
| 1 | HIGH | `qts/core/config.py:389`, `qts/portfolio/optimizer.py:1009-1062` | Default `risk_aversion = 10` is about 4x too high for the 10 % vol target. The alpha stage solves at 1.3–3.5 % ex-ante vol. The alpha-blind projection then levers the book by `s` ≈ 3–5 (median 3.05 in the sweep; the `MAX_SCALE_UP = 5` clamp is hit). `"constraint"` mode runs at about 1/4 of target. |
| 2 | MEDIUM | `qts/portfolio/optimizer.py:386-390`, `config.py:417` | The projection objective `‖w − s w*‖² + 1.0·TC` adds squared weights to a fraction of NAV. It uses the same cost model and venue as the main objective, but at an arbitrary scale: cost is about 5 % of the objective, and the projection ignores alpha. |
| 3 | MEDIUM | `qts/portfolio/optimizer.py:565-591`, `scripts/calibrate_alignment_nu.py:94-107` | The optimizer opens positions in names that are outside the risk model and have zero alpha. In the test these were 47 names, net −0.032, and acted as "free" hedges for the net band, so the modelled market exposure was 0.082 > 0.05. `n_out_of_model` reported 0 because it is counted before the solve. The simulator is safe. The ν calibration script is not. |
| 4 | MEDIUM | `qts/risk/exposures.py:205-210`, `qts/risk/model.py:178-196` | An industry label that first appears in a *future* `industry_history` row is a column at every earlier date. As a result, `as_of(t)` on a full fit ≠ a fit truncated at `t`: F differs by up to 1.1 % relative, through the eigen-simulation dimension. This breaks the PIT-reproducibility invariant and reveals the future label. |
| 5 | MEDIUM | `qts/portfolio/optimizer.py:780-810, 856-858` | The GP alpha shrink (off by default) is undone by the vol-target projection: gross 1.689 → 1.702 and vol 6.95 % → 6.94 % with and without GP. |
| 6 | LOW | `qts/portfolio/optimizer.py:394`, `qts/tcost` routed `cvx_terms` | The DPP reuse path never runs in production. The `routed` cost expression is not DPP, so `is_dpp=False`, giving `n_reuses = 0` over a 12-day simulator run. The path is equal to rebuild only trivially. |
| 7 | LOW | `qts/portfolio/optimizer.py:821` | `solve(venue="auction")` default differs from the cost model's and pipeline's `routed`. Direct callers get auction pricing. |
| 8 | LOW | `optimizer.py:582` vs `qts/core/types.py:233,245` | Out-of-model specific variance is filled with the 90th percentile in the optimizer and the median in `covariance`/`portfolio_variance` (simulator EXV). Measured gap: 0.07 % of ex-ante variance. |
| 9 | LOW | `qts/risk/covariance.py:461, 398-400, 176` | Bayesian Δ is taken about the equal-weighted mean (USE4: about the cap-weighted σ̄). Specific VRA uses EWMA(u²) forecasts and the *current* cap. The raw eigen bias is clipped before inflation. |
| 10 | LOW | `qts/risk/alignment.py:216-227`, `docs/notes/alignment_nu_calibration.md` | ν was calibrated at γ = 10 with the projection and with lag 0 (production lag 1), on books that include finding-3 phantom hedges. It must be recalibrated after fixes 1 and 3. |

No CRITICAL defects found. Every first-audit CRITICAL/HIGH fix in scope was re-verified (beta on the exact `beta` style, industry/style bands by role, vol targeting now keeps every band).

---

## Findings in detail

### 1. HIGH: default γ = 10 is inconsistent with the vol target; the projection does the sizing

**Units.** The objective is `α_H'w − γ·H·w'Σ_d w − TC − H·HC` (`optimizer.py:379, 737-739`), i.e.
`(H/252)·[α_ann'w − γ·w'Σ_ann w] − TC − …`, so **γ is an annual risk aversion**. The objective
recomputed from its parts matches the solver to 1.1e-6 relative (`v7f_obj_gp.py`). For the
unconstrained optimum, `σ*_ann = IR_ann / (2γ)` with `IR_ann = √252/H · √(α_H'Σ_d⁻¹α_H)`.

**Measured** (`v8b_gamma_units.py`, production pipeline, seed 0):

```
2023-08-24: unconstrained ex-ante IR_ann = 0.45 -> gamma for 10% = 2.3; at gamma=10 sigma* = 0.023
   gamma=0.5: alpha-stage vol 0.069 | after projection vol 0.078 s=1.45
   gamma=  2: alpha-stage vol 0.039 | after projection vol 0.065 s=2.55
   gamma= 10: alpha-stage vol 0.013 | after projection vol 0.045 s=5.00 (clamped)  alpha_H 0.00131 (alpha stage 0.00039)
2023-10-16: IR_ann = 0.54 -> gamma for 10% = 2.7; at gamma=10 sigma* = 0.027
   gamma= 10: alpha-stage vol 0.016 | after projection s=5.00 (clamped)
```

With γ = 10 the alpha/cost/risk trade-off is solved for a book 3–5× smaller than the one traded.
Impact is convex, so the cost per unit traded is understated by about `s^0.5` at the 3/2 law. The book is
then levered by a Euclidean projection that has no alpha term (finding 2), so γ has no risk
interpretation. The transfer coefficient falls (0.69 at γ=10 vs 0.74 at γ≤2, sweep below). In
`vol_target_mode="constraint"` the target is only a ceiling, so the default book runs at 2.2 % vol
against a 10 % target (`v7_optimizer.py`: `constraint mode: vol 0.02219 <= target 0.1`).

**Sweep** (`v8_gamma_sweep.py`; 3 synthetic seeds; last 400 OOS days; production pipeline with lag 1,
routed venue, default constraints). The ex-ante IR shown is per seed and per γ. SE(net Sharpe) per run is about 0.79
(Lo iid), and about 0.46 for the 3-seed mean.

| γ | gross SR | net SR | realised vol | alpha-stage vol (median) | projection s (median) | turnover/day | cost drag/yr | TC (transfer coeff.) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.99 | 0.61 | 8.7 % | 9.8 % | 1.02 | 0.127 | 3.3 % | 0.73 |
| 1 | 1.05 | 0.67 | 8.5 % | 9.3 % | 1.09 | 0.124 | 3.3 % | 0.74 |
| 2 | 1.02 | 0.64 | 8.2 % | 8.2 % | 1.25 | 0.123 | 3.1 % | 0.74 |
| 5 | 0.95 | 0.50 | 7.3 % | 5.7 % | 1.89 | 0.135 | 3.2 % | 0.72 |
| 10 | 1.05 | 0.55 | 6.8 % | 3.5 % | 3.05 | 0.144 | 3.4 % | 0.69 |

Paired net-Sharpe differences vs γ = 10 (same seed, same dates):

```
gamma    0.5    1.0    2.0    5.0
seed 0 -0.104 -0.011 -0.045 -0.007
seed 1  0.510  0.420  0.308 -0.004
seed 2 -0.228 -0.060  0.004 -0.138
```

**Reading.** The refactor agent saw a better net Sharpe at γ = 0.5. That result comes from one seed (seed 1) and is
**not robust**: the signs are mixed across seeds and every difference is well inside 1 SE. What is
robust:
- the gross cap (2.0) is the binding sizing constraint at γ ≤ 1 on seeds 1 and 2 (bound on 99–100 % of days);
- the 10 % target is unreachable under gross 2 / 2 % caps / bands (realised vol tops out at 8–9 %);
- higher γ only buys a more diversified book at the same gross (lower realised vol, slightly higher
  turnover, lower transfer coefficient).

The ex-ante/realised vol ratio is closer to 1 at γ ≥ 5 (0.95–1.01 vs 0.89–0.98). This is synthetic data.

**Recommended default: γ = 2.** This is principled: γ = IR_ann/(2σ_target) = 0.45–0.54/0.2 ≈ 2.3–2.7 on the measured
alpha, rounded down because constraints reduce the reachable vol. At γ = 2 the alpha stage sizes the book near target
(median s = 1.25), so the projection is a trim and not a 3–5× lever. The empirical cost is none: net SR 0.64 vs 0.55, within noise.
Better still, set γ per date: `γ_t = clip(IR_t / (2·σ_target), 0.5, 10)` from the closed form above
(one extra solve with Σ⁻¹α in factor form). Alternatively, bisect γ on the alpha problem until the alpha-stage vol hits the target,
which puts the book on the cost-aware efficient frontier. Recalibrate `alignment_nu_scale` afterwards
(finding 10). Do not change `MAX_SCALE_UP` as a substitute.

### 2. MEDIUM: projection cost term is in different units and ignores alpha

`_Compiled` (`optimizer.py:386-390`): `minimize ‖w − s·w*‖² + c·TC(w − w0)`, with `c = 1.0`. The cost term
uses the same `cost_model.cvx_terms`, venue and parametrisation as the alpha solve (verified: same
call, `tc_p`). The units differ, though: weight² against fraction of NAV. At the fixture solution
(`v7_optimizer.py`): `‖w − s w*‖² = 1.14e-2`, `TC = 5.6e-4`. The gradient of the distance term is about 2·0.01 per unit
weight, while the marginal cost is about 10 bp = 0.001, so cost carries about 5 % of the trade-off where the main objective
trades it 1:1 against alpha. The projected book also keeps only 64 % of the scaled alpha
(`alpha_H post 0.00199` vs `s·pre 0.00313`). **Fix:** remove the separate projection. Size with γ (finding 1)
and keep the SOC cap as a ceiling, i.e. use `"constraint"` mode with the calibrated γ. If a projection is kept, make it
`maximize α'w − TC − γ'·H·w'Σw` with γ' found by bisection so that vol = target. That keeps alpha, cost and
risk in the main objective's units.

### 3. MEDIUM: out-of-model names used as phantom hedges; `n_out_of_model` under-counts

`_prepare` (`optimizer.py:565-591`) builds the universe as alpha ∪ w_current ∪ risk tickers. Names outside the
risk model get zero exposure and the 90th-percentile specific variance. `material` is computed from `a`
and `w0` *before* the solve. In the test (`v7b_debug.py`: day-1 solve from flat, no `no_trade` list) the optimizer shorted **47 names outside the model with zero alpha** (gross 0.032, net −0.032) to satisfy the net band
while the modelled book had market-factor exposure **0.082** (band 0.05). `n_out_of_model = 0`.
In reality those names carry market beta of about 1, so the true net exposure is outside the band.

- `DailyBarSimulator` puts ineligible names that are not held on `no_trade` (`simulator.py`
  `dead = …tickers[inel & ~held]`). Verified: 0 position-days outside the model over 12 days
  (`v7c_dpp_sim.py`).
- `scripts/calibrate_alignment_nu.py:94-107` passes `no_trade=dead` only (names without a price), so the
  ν calibration books include such phantom hedges. Their realised returns include market moves, while their
  forecast has no factor risk, which biases the calibration bias statistic upward.
- Real data can have eligible names with a missing market cap. Those names are in the universe but outside the
  model, so the simulator does not protect against this either.

**Fix:** in `_prepare`, set `no_trade` for every name outside `risk.tickers` with `w0 == 0`, and force
held out-of-model names toward zero (or raise). Count `n_out_of_model` on the *returned* `w`. In the
calibration script add `tickers[alive & ~elig & ~held]` to `no_trade`.

### 4. MEDIUM: future industry labels enter earlier column sets (PIT reproducibility)

`_industry_labels` (`exposures.py:205-210`) takes the union of every label in `industry_history` over the
whole sample. Test (`v3b_newlabel.py`): a label `ZZZ_NEW` assigned at day 560, queried at day ~300:

```
full-fit factors (24, 24) | truncated-fit factors (23, 23)
future label present in as_of(t) of full fit: True  exposure col sum: 0.0
max |F_full - F_trunc| on common factors: 1.56e-06  (1.1 % of max |F|)
```

The extra all-zero column changes the eigen-simulation draw shape (`rng.standard_normal((T, K))`) and the
eigen ordering, so F at `t` depends on data after `t`. The effect on portfolio vol is small
(4.886 % vs 4.887 %), but the guarantee "`as_of(t)` = fit on data ≤ t", which the audit trail
relies on, is broken, and the column set itself reveals the future label. With the static
classification and the late-entrant mask this does **not** happen (verified: 0.0 difference).
**Fix:** in `factor_covariance`, run the eigen adjustment and VRA on the sub-block of columns that are
not identically zero in `F_hist`, re-insert zero rows/columns, and key the eigen cache on the active
columns. Optionally, drop columns whose label is not in force at any date ≤ `t` from `as_of(t)`.

### 5. MEDIUM: GP shrink is neutralised by the vol-target projection

The formula and wiring are correct. The note `g = 0.868 (holding 15.0d vs half-life 98.6d)` matches the hand computation
`h = 2·TC(unit-gross α-book)/(α'w_dir/H) = 15.02`, `g = 1/(1 + h/hl) = 0.8678` (`v7f_obj_gp.py`).
The shrink only lowers the alpha *level*, though, and the projection then re-levers to the same vol:

```
use_gp=False: pre-target gross 0.464 vol 0.0222 | after projection gross 1.689 vol 0.0695
use_gp=True:  pre-target gross 0.426 vol 0.0202 | after projection gross 1.702 vol 0.0694
```

**Fix:** this resolves with finding 1 (γ-based sizing or constraint mode). Otherwise document that GP
has no effect in projection mode. A further LOW point: `h` uses the cost of a *unit-gross* book traded from flat.
At the real gross of 2 the 3/2-law cost per unit is about √2 higher.

### 6. LOW: DPP reuse never happens on the production path

Simulator with `reuse_problem=True`: `builds 24 reuses 0`. The structure keys match and the cost signature matches,
but `is_dpp: False` (`v7d_dpp_why.py`). `v7e_dpp_term.py`: `auction` and `continuous` cost terms are
DPP, while `routed` is not: a `multiply` in the routed TC is not DPP. Weights are identical to rebuild (0.0), but the
cached path is dead code with the pipeline's venue. **Fix:** parametrise the routed venue blend (both
factors as parameters, not parameter × parameter), or document the limitation.

### 7. LOW: `solve()` default venue is `"auction"`

`optimizer.py:821`. The pipeline passes `routed`, but direct callers (tests, notebooks, scripts) price at the auction venue.
This explains a 3.5 % GP holding-day gap and a 0.3 % objective gap in my first check. **Fix:**
`venue: str = VENUE_ROUTED`.

### 8. LOW: two different fills for out-of-model specific variance

The optimizer uses the 90th percentile (`optimizer.py:582`). `RiskModelOutput.covariance/portfolio_variance` use the median
(`types.py:233,245`), and the simulator EXV comes from the latter. Measured: `reported 1.92818e-05` vs
`full N×N 1.92682e-05`. **Fix:** one fill rule in `RiskModelOutput`, used by both.

### 9. LOW: USE4 details in specific risk / eigen post-processing

- `bayesian_shrink_specific` (`covariance.py:461`): `Δ = sd_.std()` is about the equal-weighted mean. USE4
  defines Δ about the cap-weighted σ̄: `sqrt(mean((σ − σ̄)²))`.
- `specific_vra_multiplier` (`covariance.py:398-400`): the bias uses the per-stock EWMA(u²) forecast (not the
  final specific forecast) and today's cap for every historical day. This is documented, but it is not USE4.
- `eigenfactor_bias` (`covariance.py:176`) clips the raw `v` to [0.5, 2] before the parabola/inflation
  step. This is harmless at the observed v ≈ 1.01–1.05.

### 10. LOW: ν calibration conditions

ν was calibrated with γ = 10 plus the projection (finding 1). The bias pairs `as_of(d)` with `r_{d+1}`, while production
trades with lag 1. The calibration books contain finding-3 phantom hedges. ν is correct in *units*: it is a daily variance,
`nu_scale·N·median(F_style,daily)`, equivalent to one more unit-sd style with vol `√0.25 × median style vol`
(verified below). It is used identically in the optimizer risk term, the vol-target cap, `expected_risk` and the
simulator EXV. **Fix:** re-run `scripts/calibrate_alignment_nu.py` after findings 1 and 3, with the
production lag.

---

## Verified correct (with numbers)

**Exposures** (`v1_exposures.py`, brute-force per date, 6 dates around the entry date):
- The tensor equals the independent per-date recomputation: beta 1.8e-15, size 1.0e-14, momentum 8.9e-16,
  residual_vol 9.2e-15, reversal 2.2e-16. The rolling beta goes through `qts.core.stats.rolling_beta`, regressed on
  the eligible equal-weighted market.
- Standardisation is restricted to eligible names. The cap-weighted mean is 0.000 and the equal-weighted sd is 1.000 for all 10 styles.
  Perturbing the pre-entry prices of the 20 future entrants changes pre-entry exposures by **0.0**.
- The exposure frame universe equals `eligibility()` on every test date (symmetric difference 0).
- Dated industries match `industry_as_of(t)`, with one dummy per row, including across the day-350 reclassification.
- The float32 tensor has a relative error of 2.4e-7.

**Factor returns** (`v2_factor_returns.py`):
- Constrained WLS vs a direct KKT solve: 1.5e-16. `Σ c_j f_j` = 6.5e-19 (cap-weighted constraint).
- Residuals equal `r_t − B_{t−1} f_t` exactly. The test is sensitive: using `B_t` changes f by 4.8e-3.
- Market factor vs the cap-weighted market return: corr 0.9996, mean |diff| 2.3 bp. Median R² 0.29.

**Covariance** (`v3_covariance.py`):
- EWMA+NW vs a loop implementation: 5e-20. VRA λ matches brute force (1.343785).
- Eigen bias vs an independent re-implementation with the same RNG stream: 2.2e-16. It is the USE4 *volatility* bias
  `v_k = sqrt(mean_m (U_m'F0U_m)_kk / D_mk)`. The inflation is `v_S = a(v − 1) + 1` with a = 1.2, the USE4 form, applied to `v`.
  Eigenvalues are scaled by `v_S²`, which is checked.
- The adjustment de-biases out of sample: true/forecast eigen-variance went from 1.003–1.055 to 0.98–1.02 at a = 1.
- Eigen-cache bucket keying: order independence 0.0. Fit truncated at t vs full fit 0.0 on 3 dates, with a static
  classification. The bucket lengths are 252/273/294 (first row count of the bucket).

**Specific risk** (`v4_specific.py`, at a date where 20 names have thin history):
- σ_TS 1e-16, γ blend 0.0 (USE4: `IQR/1.35`, `Z = |σ_eq/σ_rob − 1|`, `clip((h−60)/120)·min(1, e^{1−Z})`).
- Structural WLS with √cap weights and E0 = 1.0047: 0.0. Bayesian shrinkage 0.0. λ_S 0.0. Final variance 8.9e-16.

**Statistical / hybrid** (`v5_hybrid.py`, `v5b_stat_varying_k.py`, panel with 2 → 4 factors):
- Per-refit Bai–Ng k matches IC_p2 on the refit's own window at 31/31 refits. k goes 2,…,2,3,4,…,4.
- Inactive slots have exactly 0 loadings, factor returns and variances.
- Procrustes rotates only the active slots: `F B'` is unchanged at 2.5e-16. Slot-wise correlation with the previous refit is ≥ 0.997.
- PIT: `as_of` on a full fit equals a fit truncated at t (dF = dB = 0) across refit boundaries, for both the
  statistical and the hybrid model.
- Hybrid cross-block = 0 (block-diagonal F). The diagonal is preserved stock by stock (2.2e-16).
- Statistical exposures are row-proportional to the total-return OLS loadings (1e-18). Inactive-slot exposure is 0.

**Alignment** (`v6_alignment.py`):
- `‖y‖ = 1`, `max|B'y| = 6.4e-13`. `ν = 0.25·N·median(style var)` exactly (3.2476e-4, N = 212).
- The augmented variance is identical in the optimizer (`exante_var_daily`), `RiskModelOutput.portfolio_variance`
  (= simulator EXV) and the vol-target norm: 7.9148e-6 in all three. The alignment share is 4.7 %.

**Optimizer** (`v7_optimizer.py`; hybrid risk model, 212 names, no-trade and forced exits active):
- Status optimal. Gross 1.715 ≤ 2. Net 0.042 ≤ 0.05. max|w| = 0.0200000005. `max |x|/(0.05·ADV/NAV)` = 0.058.
- **beta style** −0.0500 (binding, exact `beta` via `factor_groups`; the market column is not used).
- Industries max 0.0300. Risk-only styles ≤ 0.10 (`residual_vol` −0.100 binding). Alpha-theme styles are unbanded,
  as intended (D1).
- no-trade max|x| = 0. Forced exits at their caps.
- Factor-form risk vs full N×N Σ: equal. `expected_cost` = TC + H·HC exactly.
- Infeasible problem: status `infeasible`, `w = w0`, `expected_cost = nan`, `ok = False`. Constraint-mode cap holds.
- Exact power cone vs SOC approximation: max|Δw| 1.3e-6, identical objective (0.32 s vs 0.22 s).
- DPP reuse vs rebuild: identical weights. On the production venue, though, the cached path is never used (finding 6).

## Verdict

**Risk model: trustworthy for paper trading.** Every numerical component re-verified to machine
precision against independent references. PIT holds except for the narrow future-label case
(finding 4), which should still be fixed before real FF48 reclassification data is loaded.

**Portfolio construction: constraints are enforced and the units are right, but the default sizing is not.**
Before paper trading:
1. Set `risk_aversion` ≈ 2, or make it adaptive or bisected to the target, and drop or replace the alpha-blind projection
   (findings 1, 2, 5).
2. Block positions in out-of-model names in `_prepare` (finding 3).
3. Re-run the ν calibration with the production lag (finding 10).

None of these changes the sign of any decision. The synthetic sweep shows no significant net-Sharpe cost
from γ ∈ [0.5, 10], but the current default makes γ meaningless as a risk parameter and makes the traded book an
alpha-blind 3–5× rescale of the optimised one, which would be hard to diagnose on live fills.
