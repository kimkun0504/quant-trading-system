# Audit resolution log — 2026-11-04

Four independent reviews (`alpha_validation_data.md`, `risk_portfolio.md`,
`tcost_execution_backtest.md`, `code_quality.md`) were run by reviewers who had
not written the code. This file records what was found and what was done.

## Verified correct by the auditors (numerically, with scripts)
- Temporal point-in-time: `AlphaModel.as_of(t)`, every feature, dynamic weights,
  risk `as_of`, exposures, factor returns and cost snapshots are identical when
  refit on data truncated at *t*. Trainer purge is exact (one day stricter than needed).
- Backtest accounting: trade at close *t*, earn *t→t+1*, NAV identity to 1e-10,
  borrow on post-trade shorts, ex-ante variance alignment, turnover definitions
  consistent across cost-per-side, breakeven, sensitivity and holding days.
- Optimizer ⇄ simulator cost equivalence to machine precision; objective units
  (alpha per H, risk ×H, trade cost once, holding ×H).
- EWMA/Newey–West, factor-form risk vs full covariance, QIS vs Ledoit–Wolf's
  reference, USE4 Bayesian shrinkage, VRA, constraint sum-to-zero, bias bands,
  QLIKE/MZ orientation, DSR/PSR/MinTRL formulas, CPCV path assembly, fee arithmetic.

## Defects found and fixed
| Sev | Finding | Fix |
|---|---|---|
| CRITICAL | `apply_mask` kept every name that *ever* entered the universe → cross-sectional ranks, labels, ML rows and emitted scores included future entrants (+110 %/yr in the 63 days before entry) | `MarketData.eligible` panel; features/labels/scores computed on eligible names only; ineligible names dropped (not zero-filled); simulator eligibility hook with 5-day forced exit |
| CRITICAL | Default label residualisation used `mom_12_1` and `value_bp` (alpha themes) and the pipeline fell back to it silently | Risk-only default styles with an assertion guard; `_exposures_fn` hard-fails after warm-up |
| CRITICAL | "beta-neutral" constrained the `market` (country) column, leaving beta style exposure ≈ 2 | `RiskModelOutput.factor_groups`; optimizer resolves beta/industry/style by role, not string prefix (FF48 labels now work) |
| HIGH | Vol-target scale-up multiplied the book post-hoc, breaking net/beta/industry/participation bands | Replaced by a projection re-solve over the full constraint set plus an SOC risk cap |
| HIGH | Industry bands silently off for non-`IND*` labels | via `factor_groups["industry"]` |
| HIGH | `calibrate_b` ignored venue → auction prefactor recovered 3.3× too high | venue-aware V, per-venue calibration |
| HIGH | Borrow fee defaulted to GC for every short when data missing | short-interest / days-to-cover proxy with HTB cap and small-cap multiplier; source recorded |
| HIGH | Hurdle inputs mismatched the plan (IC t for decile t; PBO=None forced FAIL) | decile spread t-stat; gates PASS/FAIL/skipped with a required set; PBO from a config grid when supplied |
| HIGH | ICIR annualised ×4.5 too high for 20-day sampling | sampling frequency inferred |
| HIGH | Labels NaN when a name delists inside the horizon | last price × (1 + `delisting_return`, default −30 %) |
| HIGH | Sharadar asset growth joined restated values by `keep="last"` | `merge_asof` on filing date |
| HIGH | DSR used a hard-coded trial variance and double-counted re-runs | ledger keyed by (config hash, data hash); family variance / effective trials |
| HIGH | Pipeline turnover date-joined one day off | positional alignment |
| MEDIUM | Snapshot lost venue/date/spread on the way to the simulator; `_cost_frame` zeroed unknown columns | pass-through; raise on unknown/NaN |
| MEDIUM | SEC §31 zero for all pre-2026 dates; TAF at 2024 rate | dated SEC table 2017–2026 (verified); TAF $0.000195 / $9.79 (SR-FINRA-2026-021, verified) |
| MEDIUM | No multi-day impact persistence; delistings exited free at 0 return | k-day metaorder increment `k^1.5 − (k−1)^1.5`; −30 % haircut + liquidation costs |
| MEDIUM | LOC offset rounded to zero under $10; order ids collided across days; USE4 thin-history formula; VRA zero columns; GP shrink never shrank level; solver failures silent; participation relaxed silently | all fixed; participation relaxation now opt-in |
| MEDIUM | Alignment `nu` uncalibrated | `calibrate_alignment_nu` added (Saxena–Stubbs bias loop), off the default path |
| BUILD | `.gitignore` pattern `data/` excluded `qts/data/` from the first commit | pattern anchored to `/data/` |

## Still open (tracked)
- ~~`nu` uncalibrated~~ — calibrated on three synthetic seeds, default `alignment_nu_scale=0.25` (docs/notes/alignment_nu_calibration.md); recalibrate on real data.
- ~~Synthetic planted score contemporaneous~~ — fixed 2026-11-04: `SyntheticSpec.alpha_lag_days=1` (daily AR(1) score known at t−1).
- Code-quality refactors from `code_quality.md` (single cost contract, `AsOf` view, config serialisation, run manifest, lint/type CI, golden tests) — scheduled, not done.
- Pyright: remaining errors are pandas-stub noise; real Optional-deref items fixed.
