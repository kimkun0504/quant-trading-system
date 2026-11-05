# Golden-file changes — 2026-11-05 (Wave 2 integration)

`tests/golden/pipeline_summary.json` pins `ResearchPipeline(use_ml=False).run(md, max_days=60)` on the
golden synthetic market (`tests/conftest.py`: 100 names x 700 days, seed 11, top-80 universe).
After the Wave 2 refactor the test failed first on the data hash, which hid every other diff, so the
full field-by-field comparison below was computed separately. The golden file was regenerated with
`python -m pytest tests/test_golden.py --update-golden` only after every moved field was attributed.

## How each change was attributed

The synthetic market is **bit-identical** to the pre-refactor generator (HEAD `63ce32b`): `close`,
`volume`, `market_cap`, spreads, short interest, borrow, all fundamentals, `industry` and every
`extra["true_*"]` panel compare `DataFrame.equals` → True; the refactor only *adds* panels
(`close_raw` == `close` with `n_splits=0`, `shares_outstanding`, `true_splits`,
`true_style_factor_returns`).

The current code was run with the intended changes reverted one at a time through config/venue (plus a
monkeypatched HEAD eigen cache for S1), giving a chain from the old behaviour to the new one:

| step | what is reverted relative to the current code | change introduced by the next step |
|---|---|---|
| before | HEAD `63ce32b` (the stored golden file) | — |
| S1 | venue `auction`, `impact_b_auction=0.5`, `eigen_inflation_a=1.0`, `vol_target_projection_cost_weight=0`, HEAD eigen-bias cache | residual: optimizer reformulation, NW lag rule, all-panel data hash |
| S2 | as S1 but the current eigen cache | **eigen-bias cache keyed on bucket start, estimated from history up to it (finding #46, look-ahead fix)** |
| S3 | venue `auction`, b 0.5, a 1.0 | **projection cost term** (`vol_target_projection_cost_weight=1`, finding #28) |
| S4 | venue `auction`, b 0.5 | **eigenfactor inflation a = 1.2** (USE4, finding #15) |
| S5 | venue `auction` | **b_auction 0.5 → 0.3** (design decision D2) |
| after | nothing (current defaults) | **routed venue** (design decision D3) |

Component checks with the same reverts: the transaction-cost model (spread/impact/fees of one random
trade list at one date) and the risk model (`factor_cov`, `specific_var`, `exposures` from `as_of`) are
**bit-identical** to HEAD. Alpha outputs (`score`, `expected_return`, `ic_estimate`, `half_life_days`) are
bit-identical. The optimizer, given identical alpha, risk and snapshot, returns target weights within
5e-6 (from flat) and 1.1e-5 (from a held book) of HEAD's — solver-tolerance differences from the
reformulated problem (DPP-parametrised objective, vectorised cost terms). That is the S1 residual. It
compounds through 60 path-dependent days to the ~0.1–1 % moves in S1 (e.g. net Sharpe 0.1372 → 0.1390).

## Before / after (all fields that moved; rtol 1e-6)

| metric | before (golden) | S1 | S2 | S3 | S4 | S5 | after |
|---|---:|---:|---:|---:|---:|---:|---:|
| `data_hash` | `198fb24b43e31b23` | `19e097672173ad8f` | `19e097672173ad8f` | `19e097672173ad8f` | `19e097672173ad8f` | `19e097672173ad8f` | `19e097672173ad8f` |
| `summary.final_nav` | 1,001,389.0 | 1,001,415.5 | 1,002,205.7 | 1,002,550.9 | 992,522.3 | 992,828.0 | 992,807.4 |
| `summary.cagr_gross` | 0.0359 | 0.0360 | 0.0396 | 0.0406 | -0.00165 | -0.00689 | -0.00693 |
| `summary.cagr_net` | 0.00585 | 0.00596 | 0.0093 | 0.0108 | -0.0310 | -0.0298 | -0.0299 |
| `summary.ann_vol_gross` | 0.0523 | 0.0525 | 0.0534 | 0.0531 | 0.0519 | 0.0529 | 0.0529 |
| `summary.ann_vol_net` | 0.0523 | 0.0525 | 0.0533 | 0.0530 | 0.0518 | 0.0528 | 0.0528 |
| `summary.sharpe_gross` | 0.6990 | 0.6998 | 0.7525 | 0.7746 | -0.00636 | -0.1047 | -0.1055 |
| `summary.sharpe_net` | 0.1372 | 0.1390 | 0.1998 | 0.2279 | -0.5825 | -0.5463 | -0.5479 |
| `summary.sharpe_net_lo` | 0.1372 | 0.1390 | 0.1998 | 0.2279 | -0.5825 | -1.1208 | -1.1248 |
| `summary.sharpe_t_stat_net` | 0.0669 | 0.0678 | 0.0975 | 0.1112 | -0.2841 | -0.2665 | -0.2673 |
| `summary.newey_west_t_net` | 0.0697 | 0.0704 | 0.0989 | 0.1138 | -0.2837 | -0.2812 | -0.2819 |
| `summary.max_drawdown_net` | -0.0283 | -0.0284 | -0.0299 | -0.0289 | -0.0309 | -0.0326 | -0.0326 |
| `summary.calmar_net` | 0.2064 | 0.2096 | 0.3110 | 0.3726 | -1.0049 | -0.9146 | -0.9171 |
| `summary.sortino_net` | 0.1892 | 0.1915 | 0.2739 | 0.3115 | -0.7666 | -0.7340 | -0.7358 |
| `summary.hit_rate_net` | 0.5500 | 0.5500 | 0.5333 | 0.5167 | 0.5500 | 0.4833 | 0.4667 |
| `summary.avg_gross_exposure` | 1.2225 | 1.2208 | 1.2105 | 1.2008 | 1.1976 | 1.1064 | 1.1068 |
| `summary.avg_net_exposure` | 0.0415 | 0.0415 | 0.0438 | 0.0429 | 0.0411 | 0.0448 | 0.0448 |
| `summary.avg_daily_turnover` | 0.1011 | 0.1013 | 0.1015 | 0.0983 | 0.0998 | 0.1137 | 0.1137 |
| `summary.effective_holding_days` | 24.1776 | 24.1101 | 23.8603 | 24.4422 | 24.0106 | 19.4618 | 19.4707 |
| `summary.avg_cost_bps_per_side` | 10.3763 | 10.3783 | 10.4102 | 10.5583 | 10.7002 | 7.2319 | 7.2451 |
| `summary.avg_daily_cost_bps_of_nav` | 1.1672 | 1.1688 | 1.1732 | 1.1533 | 1.1853 | 0.9254 | 0.9272 |
| `summary.total_costs_usd` | 7,041.8 | 7,052.1 | 7,077.2 | 6,956.2 | 7,123.9 | 5,577.9 | 5,588.8 |
| `summary.transfer_coefficient_mean` | 0.7798 | 0.7806 | 0.7770 | 0.7767 | 0.7787 | 0.8276 | 0.8278 |
| `summary.realized_vs_exante_vol_ratio` | 0.9440 | 0.9473 | 0.9689 | 0.9671 | 0.9409 | 0.9690 | 0.9687 |
| `summary.breakeven_cost_bps` | 14.3584 | 14.4011 | 15.7292 | 16.6180 | -0.1313 | -1.9321 | -1.9466 |
| `decile.t_stat` | 1.2381 | 1.2451 | 1.2451 | 1.2451 | 1.2451 | 1.2451 | 1.2451 |
| `dsr` | 0.5264 | 0.5268 | 0.5384 | 0.5438 | 0.3884 | 0.3954 | 0.3951 |
| `backtest.nav_quartiles[0]` | 1,002,011.3 | 1,002,015.3 | 1,002,759.6 | 1,003,645.0 | 1,002,839.9 | 1,002,553.9 | 1,002,542.1 |
| `backtest.nav_quartiles[1]` | 1,002,276.6 | 1,002,340.6 | 1,003,319.2 | 1,002,280.3 | 998,726.0 | 1,002,578.9 | 1,002,576.4 |
| `backtest.nav_quartiles[2]` | 1,014,460.2 | 1,014,540.5 | 1,011,691.2 | 1,010,822.3 | 1,004,096.7 | 1,011,773.9 | 1,011,752.4 |
| `backtest.sum_abs_weights` | 73.3505 | 73.2482 | 72.6309 | 72.0502 | 71.8533 | 66.3844 | 66.4106 |
| `backtest.sum_costs_by_kind.spread` | 0 | 0 | 0 | 0 | 0 | 0 | 0.0456 |
| `backtest.sum_costs_by_kind.impact` | 4,604.6 | 4,616.6 | 4,639.8 | 4,546.9 | 4,703.0 | 3,210.7 | 3,219.9 |
| `backtest.sum_costs_by_kind.fees` | 1,723.0 | 1,721.6 | 1,728.1 | 1,706.5 | 1,708.5 | 1,743.5 | 1,743.0 |
| `backtest.sum_costs_by_kind.borrow` | 714.1898 | 713.9350 | 709.3761 | 702.8600 | 712.5070 | 623.7060 | 625.8630 |


Unchanged (exactly or within rtol 1e-6): `summary.start`, `summary.end`, `summary.n_days`, `summary.n_delistings`, `summary.n_failed_solves`, `ic_report.n`, `ic_report.mean_ic`, `ic_report.ic_sd`, `ic_report.icir`, `ic_report.t_stat`, `ic_report.hit_rate`, `ic_report.periods_per_year`, `decile.mean`, `decile.sharpe`, `hurdles.overall`, `hurdles.status.breakeven_cost`, `hurdles.status.dsr`, `hurdles.status.ic_sign`, `hurdles.status.incremental_ic`, `hurdles.status.net_sharpe_delta`, `hurdles.status.pbo`, `hurdles.status.phase_sharpe`, `hurdles.status.suspect`, `hurdles.status.t_stat`, `backtest.n_nav`.

## Explanation per metric

* **`data_hash`** `198fb24b43e31b23` → `19e097672173ad8f`: the hash now covers every panel
  (`qts.core.manifest.hash_market_data`, finding #37), not only `close`. The data itself did not move
  (see above).
* **`decile.t_stat`** 1.2381 → 1.2451: the Newey–West lag rule (finding #7). It was `h − 1 = 19` lags;
  now `max(h, floor(4 (T/100)^(2/9)))`, capped at `T − 1` (`qts.alpha.evaluation.newey_west_lags`). `decile.mean`/`decile.sharpe` and the whole `ic_report` are
  unchanged, which confirms that the alpha layer did not move. Identical in S1…after.
* **Backtest metrics** (`summary.*`, `dsr`, `backtest.*`): every one of them is a function of the
  simulated book, so each step in the chain moves all of them together. Main moves:
  * *Eigen cache #46 (S1 → S2)*: the bias of the first backtest bucket was estimated on the history
    up to the first date the backtest asked for (a mid-bucket date); now it is estimated up to the
    bucket start. Different eigen bias → different factor covariance → different books: net Sharpe
    0.139 → 0.200, gross exposure 1.221 → 1.211.
  * *Projection cost term (S2 → S3)*: the vol-target re-solve now penalises trading cost
    (`+ 1.0 x TC(w − w0)`): turnover 0.1015 → 0.0983/day, holding period 23.9 → 24.4 d.
  * *Eigen inflation a = 1.2 (S3 → S4)*: inflates the simulated eigenfactor bias (USE4), so ex-ante factor
    risk rises on the low-eigenvalue directions and the book changes. Over this 60-day window gross Sharpe
    goes from 0.77 to −0.01. A 60-day annualised Sharpe has a standard error of about √(252/60) ≈ 2, so this
    is a path effect, not evidence that a = 1.2 is worse. `realized_vs_exante_vol_ratio` 0.967 → 0.941.
  * *b_auction 0.3 (S4 → S5)*: auction impact per √(Q/ADV) falls from 1.67 to 1.0 (D2), so impact costs
    drop 4,703 → 3,211 USD, cost/side 10.70 → 7.23 bps. The cheaper auction lets the optimizer trade
    more: turnover 0.0998 → 0.1137/day, transfer coefficient 0.779 → 0.828, gross exposure 1.198 → 1.106.
  * *Routed venue (S5 → after)*: trades above `auction_max_pct_of_auction_volume` of auction volume go
    partly to the continuous session, so a small spread cost appears (`sum_costs_by_kind.spread`
    0 → 0.046 USD) and costs move by < 0.2 %. In this universe almost every trade fits within the
    auction cap.
  * `sharpe_net_lo` (Lo 2002 autocorrelation-adjusted Sharpe) separates from `sharpe_net` in S5/after
    (−1.125 vs −0.548). `metrics.annualized_sharpe(lo_adjust=True)` falls back to the plain Sharpe when
    `q + 2 Σ (q−k) ρ_k ≤ 0` (strongly negative autocorrelation of the 60 daily returns). That held for the
    golden path and S1–S4; on the b = 0.3 paths the denominator is positive, so the correction applies.
* **`hurdles.*`**: every status and `overall` is unchanged.

## Notes

* The golden window is only 60 days, so the headline Sharpe signs are noise. The file guards against
  *unintended* drift; it says nothing about strategy quality. `test_golden_run_is_not_degenerate` still
  holds (60 days, 0 failed solves, gross exposure 1.11, turnover > 0, costs > 0).
* The eigen-cache fix (#46) was not on the list of expected golden moves handed to the integration
  step, but it is an intended Wave 2 change (look-ahead fix, `docs/notes/doc_review_findings.md` #46).
* Reproduce the attribution with the throwaway scripts described above: run
  `ResearchPipeline(golden_config(), use_ml=False, venue=...)` with the config reverts listed per step.

---

# Audit-2 fix wave (second-pass audit, three fix agents + integration)

Same golden run (`ResearchPipeline(golden_config(), use_ml=False).run(md, max_days=60)`, 100 x 700
synthetic market, seed 11, top-80 universe). The synthetic market is unchanged (`data_hash` identical),
so every move below comes from the model code / defaults. Two summary keys were added to the golden
file (`vol_target_caps_share`, `mean_risk_aversion`, from the simulator's `solve_log`).

## Attribution chain

The config-switchable changes were reverted one at a time on the current code (throwaway script that
runs the golden pipeline with the overrides below; nothing committed):

| step | reverted relative to the current code | change introduced by the next step |
|---|---|---|
| before | stored golden file (pre-wave code) | -- |
| R3 | `risk_aversion_mode="fixed"`, `risk_aversion=10`, `risk.alignment_nu_scale=0.25`, `universe.refresh_anchor="panel_start"` | -- (R3 vs before = all *code-only* changes, see below) |
| R2 | as R3 but current refresh anchor | **month-start universe refresh** (`refresh_anchor="month_start"`) |
| R1 | `risk_aversion_mode="fixed"`, `risk_aversion=10` | **alignment nu 0.25 -> 0.05** |
| after | nothing (current defaults) | **target-vol gamma search** (finding 1) |

*before -> R3* bundles the changes that have no config switch: alpha/data (lag-1 label alignment,
full pre-entry feature history, purged-fold `ic_sign`, inferred IC periods-per-year, point-in-time
future-industry fix), costs/backtest (breakeven on gross-after-borrow returns, exact cumulative impact
persistence, execution-date auction share, borrow source), and the risk/portfolio code (out-of-model
names excluded from the variable set, one 90th-percentile out-of-model specific variance fill in
`qts.core.risk_fill`, also used by `RiskModelOutput.covariance` / `portfolio_variance`).

## Before / after (all fields that moved; rtol 1e-6)

| metric | before (golden) | R3 | R2 | R1 | after |
|---|---|---|---|---|---|
| `summary.final_nav` | 987,902 | 1,006,829 | 1,004,907 | 1,006,154 | 1,035,510 |
| `summary.cagr_gross` | -0.02727 | 0.0524 | 0.0441 | 0.04974 | 0.1834 |
| `summary.cagr_net` | -0.04984 | 0.029 | 0.02077 | 0.0261 | 0.1578 |
| `summary.ann_vol_gross` | 0.05283 | 0.05849 | 0.06043 | 0.06051 | 0.06909 |
| `summary.ann_vol_net` | 0.05268 | 0.05837 | 0.06032 | 0.06041 | 0.06892 |
| `summary.sharpe_gross` | -0.4973 | 0.902 | 0.744 | 0.832 | 2.472 |
| `summary.sharpe_net` | -0.9443 | 0.5184 | 0.3705 | 0.4563 | 2.161 |
| `summary.sharpe_net_lo` | -1.753 | 1.54 | 1.357 | 0.4563 | 8.33 |
| `summary.sharpe_t_stat_net` | -0.4604 | 0.2529 | 0.1808 | 0.2226 | 1.05 |
| `summary.newey_west_t_net` | -0.4872 | 0.276 | 0.1976 | 0.2451 | 1.183 |
| `summary.max_drawdown_net` | -0.03386 | -0.03002 | -0.02914 | -0.02751 | -0.01873 |
| `summary.calmar_net` | -1.472 | 0.9659 | 0.713 | 0.9487 | 8.427 |
| `summary.sortino_net` | -1.234 | 0.7569 | 0.5317 | 0.6604 | 3.34 |
| `summary.hit_rate_net` | 0.5167 | 0.5 | 0.5333 | 0.5167 | 0.6 |
| `summary.avg_gross_exposure` | 1.101 | 1.223 | 1.217 | 1.215 | 1.263 |
| `summary.avg_net_exposure` | 0.04387 | 0.04172 | 0.04122 | 0.04109 | 0.05 |
| `summary.avg_daily_turnover` | 0.1128 | 0.11 | 0.1115 | 0.1121 | 0.1056 |
| `summary.effective_holding_days` | 19.53 | 22.24 | 21.84 | 21.68 | 23.91 |
| `summary.avg_cost_bps_per_side` | 7.35 | 6.508 | 6.428 | 6.413 | 6.807 |
| `summary.avg_daily_cost_bps_of_nav` | 0.9316 | 0.8927 | 0.897 | 0.904 | 0.8686 |
| `summary.total_costs_usd` | 5605 | 5406 | 5420 | 5460 | 5298 |
| `summary.transfer_coefficient_mean` | 0.8264 | 0.8483 | 0.8562 | 0.8378 | 0.7933 |
| `summary.realized_vs_exante_vol_ratio` | 0.9714 | 0.9109 | 0.925 | 0.9327 | 0.971 |
| `summary.breakeven_cost_bps` | -9.247 | 17.43 | 14.38 | 16.17 | 62.77 |
| `summary.vol_target_caps_share` | (new) | nan | nan | nan | 1 |
| `summary.mean_risk_aversion` | (new) | 10 | 10 | 10 | 0.2994 |
| `ic_report.mean_ic` | 0.06313 | 0.05289 | 0.05063 | 0.05063 | 0.05063 |
| `ic_report.ic_sd` | 0.1491 | 0.137 | 0.1348 | 0.1348 | 0.1348 |
| `ic_report.icir` | 1.503 | 1.37 | 1.333 | 1.333 | 1.333 |
| `ic_report.t_stat` | 0.7334 | 0.6686 | 0.6505 | 0.6505 | 0.6505 |
| `ic_report.hit_rate` | 0.6667 | 0.6667 | 0.3333 | 0.3333 | 0.3333 |
| `decile.mean` | 0.03071 | 0.0299 | 0.0321 | 0.0321 | 0.0321 |
| `decile.sharpe` | 2.104 | 1.847 | 1.869 | 1.869 | 1.869 |
| `decile.t_stat` | 1.245 | 1.108 | 1.124 | 1.124 | 1.124 |
| `dsr` | 0.3227 | 0.5991 | 0.5711 | 0.5873 | 0.8495 |
| `hurdles.status.breakeven_cost` | FAIL | PASS | PASS | PASS | PASS |
| `hurdles.status.ic_sign` | PASS | FAIL | FAIL | FAIL | FAIL |
| `hurdles.status.phase_sharpe` | FAIL | PASS | PASS | PASS | PASS |
| `hurdles.status.suspect` | PASS | PASS | PASS | PASS | FAIL |
| `backtest.nav_quartiles[0]` | 1,003,866 | 1,005,809 | 1,002,811 | 1,002,830 | 1,001,193 |
| `backtest.nav_quartiles[1]` | 1,001,975 | 1,008,809 | 1,006,185 | 1,004,657 | 1,017,636 |
| `backtest.nav_quartiles[2]` | 1,006,358 | 1,017,541 | 1,015,242 | 1,014,420 | 1,025,854 |
| `backtest.sum_abs_weights` | 66.07 | 73.36 | 73.03 | 72.91 | 75.76 |
| `backtest.sum_costs_by_kind.spread` | 0 | 6.433 | 9.834 | 9.596 | 14.58 |
| `backtest.sum_costs_by_kind.impact` | 3249 | 2632 | 2626 | 2637 | 2728 |
| `backtest.sum_costs_by_kind.fees` | 1737 | 1692 | 1693 | 1694 | 1640 |
| `backtest.sum_costs_by_kind.borrow` | 619.4 | 1075 | 1091 | 1120 | 916 |

## Explanation

* **Code-only changes (before -> R3)**: net Sharpe -0.94 -> 0.52, gross exposure 1.10 -> 1.22.
  * `sum_costs_by_kind.spread` 0 -> 6.4 USD: the auction share is now measured on the *execution*
    date's auction volume, so a few trades spill into the continuous session.
  * `impact` 3,249 -> 2,632 USD and cost/side 7.35 -> 6.51 bp: exact cumulative impact persistence
    (prior-streak USD instead of the streak-count approximation) plus the different book.
  * `borrow` 619 -> 1,075 USD: borrow-source fix (second pass #8: names with NaN short interest now
    get the small-cap borrow multiplier too, and `borrow_fee_source` reports the real source) on a
    larger short book.
  * `breakeven_cost_bps` -9.2 -> 17.4 and the `breakeven_cost` gate FAIL -> PASS: breakeven is now
    computed from gross-after-borrow returns (borrow is not a per-side trading cost) and the return path
    changed.
  * IC (`mean_ic` 0.063 -> 0.053, `icir` 1.50 -> 1.37): lag-1 label alignment and the full pre-entry
    history change the scored panel; the IC is measured on 3 non-overlapping 20-day periods in this
    window, so these numbers are noise-dominated.
  * `ic_sign` PASS -> FAIL: the gate's folds are now the purged walk-forward test blocks (as for
    `net_sharpe_delta`) instead of calendar years (second pass #16); with 3 IC observations in 60 days
    most blocks hold 0-1 of them.
* **Month-start refresh (R3 -> R2)**: different universe refresh dates -> different eligible set:
  `ic_report.hit_rate` 0.67 -> 0.33 (1 of 3 periods flips), net Sharpe 0.52 -> 0.37.
* **nu 0.25 -> 0.05 (R2 -> R1)**: smaller alignment variance (calibrated in
  `docs/notes/alignment_nu_calibration.md`); small book changes, net Sharpe 0.37 -> 0.46.
  `sharpe_net_lo` falls back to the plain Sharpe on this path (negative autocorrelation denominator).
* **Target-vol gamma search (R1 -> after)**: gamma 10 (fixed) -> mean 0.30 searched per solve. The book
  is sized on the alpha-aware objective instead of a 10x-risk-averse book levered by projection:
  realised vol 6.0 % -> 6.9 %, gross 1.22 -> 1.26, turnover 0.112 -> 0.106/day, holding 21.7 -> 23.9 d.
  `vol_target_caps_share` = 1.0: on every solve the per-name / ADV caps bind before the 10 % target
  (80-name universe, 2 % position cap), so the search stops at the capped book; that is why gamma
  ends far below the closed form. Net Sharpe 0.46 -> 2.16, DSR 0.59 -> 0.85.
* **`suspect` PASS -> FAIL**: net Sharpe 2.16 > 2 trips the suspect gate. Over 60 days the annualised
  Sharpe has a standard error of about sqrt(252/60) ~ 2 (t-stat 1.05, NW t 1.18), so this is a property
  of the window, not evidence of leakage. Checked: the planted-oracle tests
  (`tests/test_validation.py::test_planted_oracle_is_flagged`,
  `tests/test_pipeline.py::test_planted_oracle_through_pipeline_is_flagged_suspect`),
  `tests/test_synthetic_oracle.py`, `tests/test_pit_properties.py` and `tests/test_label_halts.py` all
  pass; the full-length demo (300 x 1512) is the meaningful check (see the integration report).
* `test_golden_run_is_not_degenerate` still holds (60 days, 0 failed solves, gross 1.26, costs > 0).
