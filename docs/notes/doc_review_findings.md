# Findings from writing the handbook (2026-11-04)

Writing the handbook forced a line-by-line read of every module; these are the
code/doc inconsistencies the writers found. Status is tracked in WORK_PLAN.md.

## Design decisions (orchestrator)
- **D1 Label residualisation = optimizer neutralisation set.** The label is residualised on
  exactly the styles the optimizer neutralises (market, industries, beta, size,
  residual_vol, liquidity, short_term_reversal). Predicting a premium the book is then
  forced to hedge away is useless, so the low-risk theme's linear premium is
  intentionally removed from the ML target. One constant (`qts/core/roles.py`
  or equivalent) must define this set for both targets and optimizer, and docs must say so.
- **D2 Auction impact prior b_auction = 0.3.** With V = 9 % of ADV the effective impact per
  √(Q/ADV) is 0.3/√0.09 = 1.0, equal to the continuous session; the auction then saves the
  spread, matching the routing table (small orders → auction). 0.5 made every auction trade
  dearer than continuous above ~0.02 % ADV, contradicting the router. Still a prior —
  recalibrate from fills.
- **D3 Routed venue in the simulator.** The simulator should price each trade the way the
  router would execute it: the part up to `auction_max_pct_of_auction_volume` × auction
  volume at the auction, the remainder in the continuous session.
- **D4 Execution-delay option.** Add `signal_lag_days` (default 0, research setting 1) so
  signals/risk/snapshot use data as of t−lag while trades execute at close t. Report the
  Sharpe difference between 0 and 1 in the demo to bound the close-as-signal bias.
- **D5 One IC-shrink convention.** `ic_shrink` = fraction kept (0.5 ⇒ half); apply it to the
  fallback `default_ic` too; delete the inverse convention in `alpha_scaling.shrink_ic`.

## Fix list
| # | File:line (at time of finding) | Issue |
|---|---|---|
| 1 | qts/core/config.py:2 | docstring "auction prefactor 1.5" stale |
| 2 | qts/core/config.py:133, docs/REVIEW_NOTES.md:20 | "150/50" label; constraints give dollar-neutral 100/100 — fix wording (or support net target) |
| 3 | qts/alpha/targets.py:46-48, qts/pipeline.py:189 | low-risk overlap undocumented; ALPHA_THEME_FEATURES guard incomplete (→ D1) |
| 4 | qts/alpha/alpha_model.py:155, qts/portfolio/alpha_scaling.py:29-34 | ic_shrink two conventions; default_ic unshrunk (→ D5) |
| 5 | qts/alpha/alpha_model.py:160, simulator.py:394 | half_life_days computed but unused; GP shrink not wired (wire or document) |
| 6 | qts/core/crosssection.py:34 | rank_gauss bound docstring (±3.72 actual) |
| 7 | qts/alpha/evaluation.py:132 | Newey–West lags h−1 (audit #15) — use Newey–West automatic lag ≥ h |
| 8 | qts/alpha/models.py:273-277 | MLP early stopping random holdout (audit #13) — use time-contiguous validation |
| 9 | qts/core/config.py:25 | `AlphaConfig.secondary_horizons` unused |
| 10 | qts/alpha/composite.py:143 | momentum vol scalar clipped [0.5, 1.5] vs "vol-targeted" claim — document |
| 11 | qts/alpha/features.py:163 | `strev_21` "timing feature" comment; no timing use |
| 12 | qts/alpha/trainer.py:57, alpha_model.py | `valid_days`, `composite_weight`, `ic_window` outside config |
| 13 | qts/data/loaders.py:478 | Sharadar default ARQ (quarterly) — default ART for TTM ratios; `lag_days` default 1 |
| 14 | qts/risk/factor_returns.py:10-11, tests/test_risk.py:88 | "market factor equals cap-weighted market" overstated |
| 15 | qts/risk/covariance.py:148-150 | eigenfactor a=1.2 inflation omitted; add as config (default 1.2 per USE4) |
| 16 | qts/core/config.py:64 | `use_nonlinear_shrinkage` has no effect on production risk — wire or remove |
| 17 | qts/risk/hybrid.py:19 | docstring vs code (1−keep_i) |
| 18 | qts/risk/model.py:24 | eigen refresh counts observations, not calendar days |
| 19 | qts/risk/hybrid.py:166-167 | statistical factor count chosen on end-of-sample window (look-ahead) — choose per refit date |
| 20 | qts/core/config.py:102,106,107; impact.py:249; ibkr.py:140 | `impact_exponent`, `auction_volume_share_default` unused; 9 % hard-coded in 3 places |
| 21 | qts/execution/router.py | router ignores date-aware auction share; `urgency_from_half_life` unused |
| 22 | qts/execution/router.py:179 vs ibkr.py:18 | `noTakeLiq` on ClosePx not in documented params — verify / remove |
| 23 | qts/tcost/model.py:32 | m₃ = 2.368 not 2.40 |
| 24 | tests | no optimizer-vs-simulator cost-equivalence test |
| 25 | qts/tcost/tca.py:17 | reversion to next open vs plan's next close — make configurable, default next close |
| 26 | qts/validation/hurdles.py:1,9,66,121,127; pipeline.py:426 | ">" in text vs ">=" in code; "BUILD_PLAN s.8" → §2.4 |
| 27 | qts/core/config.py:174-177 | ValidationConfig split fields unused by pipeline — wire into fold construction or document |
| 28 | qts/portfolio/optimizer.py `_vol_target_projection` | projection ignores trading cost — add cost term |
| 29 | PortfolioConfig.max_adv_participation 5 % vs auction cap (→ D3) |
| 30 | docs/BUILD_PLAN.md:169 | "3:55 indicative … submitted before 3:50" impossible — fix text |
| 31 | docs/BUILD_PLAN.md:130 | TAF 2024 rate — update |
| 32 | docs/audit/RESOLUTION.md:45 | synthetic lag item stale once fixed |
| 33 | qts/data/store.py:106-110 | **data loss**: `overwrite_partitions` rewrites a whole year with only the new rows — appending one day wipes the year. Must merge (upsert by key) |
| 34 | qts/data/universe.py:43-44, types.py:78-79 | price ≥ $5 and dollar ADV screens use the split/dividend-*adjusted* close — historically wrong; add an unadjusted `close_raw` (or price adjustment factor) panel to MarketData and use it for screens/dollar volume/share counts |
| 35 | qts/data/loaders.py:562-565 | B/P and E/P divide as-reported per-share values by the adjusted close — use the raw close |
| 36 | qts/data/loaders.py:546 | filing visible at close of filing date (`lag_days=0`); default 1 |
| 37 | qts/pipeline.py:103-109, 409 | data hash covers close only; no code version in ledger — hash all panels + git sha |
| 38 | qts/data/synthetic.py:14-15, 212 | docstring lists 6 style factors (now 4); `true_style_returns` holds only size |
| 39 | README.md | inconsistent test counts |
| 40 | qts/data/universe.py:55 | `refresh_days`, `exit_after_n_fails` not in UniverseConfig |
| 41 | docs/BUILD_PLAN.md:206 | Phase 0 lists GICS mapping (now FF48) |
| 42 | config.py:48,123 | two separate −30 % delisting settings — single source |
| 43 | types.py | `MarketData.industry` static; support dated classification (wide panel) |
| 44 | loaders.py `to_short_interest_ratio` | needs shares outstanding; add `shares_outstanding` panel to MarketData |
| 45 | validation | planted_oracle_test not run through ResearchPipeline |
| 46 | qts/risk/model.py:114-124 | **look-ahead**: eigenfactor bias cache filled by whichever as_of hits the bucket first; key on bucket start date and estimate from history up to it (xfail test_determinism::test_risk_as_of_independent_of_call_order_within_bucket) |
