# Audit: transaction cost, execution and backtest layers

Scope: `qts/tcost/*`, `qts/execution/*`, `qts/backtest/*`, `qts/portfolio/costs.py`,
`qts/pipeline.py` against `docs/BUILD_PLAN.md` §4 and §6. Method: line-by-line read,
then numerical verification scripts (`/tmp/audit/sim_check.py`, `cost_check.py`,
`exec_check.py`, metrics one-liners). `pytest tests/test_tcost.py tests/test_execution.py
tests/test_backtest.py -q -p no:warnings` → 27 passed. No source files were modified.

Severity key: CRITICAL = wrong sign/scale/timing that would mislead a go/no-go decision;
HIGH = wrong number, direction-preserving; MEDIUM = fragility / silent fallback / material
unmodelled optimism; LOW = style or immaterial.

## Summary table

| # | Sev | Where | Finding |
|---|-----|-------|---------|
| 1 | HIGH | `qts/tcost/tca.py:142-144`, `qts/execution/simulator_fills.py:316` | `calibrate_b` regresses on `sigma*sqrt(F/ADV_full)` regardless of venue. For auction fills (the default tactic) the model's `V` is `auction_share*ADV`, so the recovered `b_ols` is `b_auction/sqrt(0.09) = 3.33x` the parameter it is meant to update. Feeding it back into `impact_b_auction` would inflate auction impact 3.3x (verified: simulated fills with `b_auction=0.5` calibrate to 1.6667). |
| 2 | HIGH | `qts/core/config.py` `default_borrow_fee_annual=0.005`, `qts/backtest/simulator.py:182`, `qts/tcost/borrow.py:28-34` | Every short with no borrow datum is charged 0.5%/yr GC. The plan (§4 table, "Borrow") calls for a conservative haircut for small caps in backtests; the alpha has a `short_interest` theme whose shorts are structurally hard-to-borrow (5-50%/yr). Net Sharpe of any strategy with that sleeve is optimistic until `md.borrow_fee` is populated or a SI-/size-conditional haircut is applied. |
| 3 | MEDIUM | `qts/backtest/simulator.py:145` → `qts/portfolio/costs.py:199,253-272` | `DailyBarSimulator._snapshot` passes the `TransactionCostModel` snapshot through `align_snapshot`, whose `SNAPSHOT_KEYS` drop `venue`, `date`, `half_spread_bps`, `adv_shares`. Consequences in the simulator path: (a) `estimate()` always prices the **auction** venue even when `run(..., venue="continuous")` makes the optimizer price continuous (verified: 3.57e-4 vs 4.55e-4 of NAV on the same trade); (b) `estimate()` reads `half_spread_bps`, not `half_spread`, so real spreads from `md.spread_bps` are replaced by the 2 bp default for continuous costing (verified: `[0.9, 12.3, 2.8, 14.3, 7.4]` bp → `[2,2,2,2,2]`); (c) `date=None` → special auction days, SEC effective date and TAF holiday never apply in a backtest. |
| 4 | MEDIUM | `qts/pipeline.py:188-190` | `cost_sensitivity` and the hurdle-gate `breakeven_cost_bps` use `bt.turnover.reindex(gross.index).fillna(0)`. Turnover is indexed by decision date `t`, returns by `t+1`, so this date-join shifts turnover one day late, drops the first (largest, initial-build) turnover and pads the last day with 0. `BacktestResult.summary()` uses the correct positional `turnover_on_return_dates()`, so the pipeline and the summary report two different breakeven numbers. |
| 5 | MEDIUM | `qts/tcost/fees.py:460-461,494-495` | `fee_holiday` sets the SEC §31 rate to 0 for **every** date before 2026-04-04. Correct for 2025-10-01..2026-04-03 (rate advisory 2026-2), wrong for 2006-2025 when the rate was $5-$278 per $1M (0.05-2.8 bp per sell). Affects any caller that passes a date (`FillSimulator`, `estimate(..., date=)`), not the daily simulator (which, per #3, has no date and charges $20.60/M throughout). |
| 6 | MEDIUM | `qts/backtest/simulator.py:268-289` (design) | Impact is charged per day and never persisted into the next day's price. Under the square-root law a position built over `k` days at participation `p/day` costs `sqrt(k)` times what `k` independent daily trades are charged (and ~70% of impact is permanent per Frazzini et al.). With Gârleanu-Pedersen-style partial trading toward the target (the design intent), multi-day same-direction trading is the norm. At the plan's caps (≤5% ADV, ≤2% of auction volume) and a $1-5M NAV in top-1000 names, typical trades are ≪1% ADV so the absolute impact is small (~5-15 bp), but the understatement factor for 2-4 day builds is 1.4-2x on the impact component. |
| 7 | MEDIUM | `qts/backtest/simulator.py:269-274,288` | Delistings: a held name whose `t+1` return is NaN earns 0 and is removed with no trade, no cost, no turnover. (a) Last-close exit is optimistic for longs in bankruptcies/halts (true recovery often 0) and no delisting return hook exists; (b) the liquidation is free (no spread/impact/fees) and absent from `trades`, so turnover and cost-per-side are understated on those days. |
| 8 | MEDIUM | `qts/backtest/simulator.py:151-173` | `_cost_frame` silently books **zero** cost when `estimate()` returns a DataFrame with no recognised column (no `spread/impact/fees[_usd|_cost]` and no `total`), and `fillna(0)` turns any NaN cost into 0. A cost-model refactor that renames a column would make the backtest free without an error. |
| 9 | MEDIUM | `qts/execution/router.py:84-86,91-92` | `loc_limit`/`passive_limit` use `round(px, 2)`. For stocks ≤ $10 a 3 bp offset rounds to 0, i.e. the LOC sits exactly at the indicative and is unfilled whenever the auction prints one tick against (verified: $5 and $10 → 0.0 bp offset; $20 → 5 bp). Buys should ceil to tick, sells floor. |
| 10 | MEDIUM | `qts/tcost/model.py:173`, `qts/tcost/model.py:127-129` | Direct (non-simulator) use: `cvx_terms` does `sigma.fillna(0.0)` → zero impact coefficient for names with NaN sigma (new listings), while `estimate` returns NaN impact/total for the same name (verified). Optimizer and realised cost disagree and the optimizer sees free impact. (The simulator path median-fills first, so it is only exposed via the README-style direct API.) |
| 11 | MEDIUM | `qts/core/config.py` (`impact_b_auction=0.5`) vs `qts/tcost/impact.py:209-211`, `qts/tcost/model.py:166` | Docstrings still say `b_auction = 1.5 (review item 9)`; config and REVIEW_NOTES say 0.5. Prior is user-set; but note that at 0.5 the effective auction coefficient is 1.67 per `sqrt(Q/ADV)` vs 1.0 continuous, so auction is cheaper than continuous only for trades < ~0.02% of ADV (`0.667*sqrt(p) < hs/sigma` with hs=2 bp, sigma=2%). The backtest costs **all** trades at the auction model, which is therefore conservative relative to the router's actual mix (large orders → cheaper in-session algos). |
| 12 | LOW | `qts/core/config.py` `finra_taf_per_share_sold=0.000166`, `finra_taf_max_per_trade=8.30` | These are the 2024 rates. FINRA's current (2026) TAF for covered equities is $0.000195/share, max $9.79/trade (SR-FINRA-2026-021, Exhibit 5A); the Oct-Dec 2026 holiday is correct. Impact ≈ 0.01 bp. |
| 13 | LOW | `qts/tcost/fees.py:443-444` | Exchange pass-through fees are ignored, as documented, but for an auction-centric system the NYSE/Nasdaq closing-auction fee (~$0.0008-0.0015/share for MOC/LOC) is 3-4x the clearing fee and ~0.2 bp/side on a $50 stock. Worth a `cfg.auction_exchange_fee_per_share` line. CAT fee: current prospective rate is $0.000001/share (CAT Fee 2026-1); config's $0.000003 is immaterial. |
| 14 | LOW | `qts/execution/orders.py:405`, `qts/execution/simulator_fills.py:307` | `order_id = f"{ticker}-{side}-{shares}"` collides across days; a persistent `TCALedger` raises `KeyError` on the second day (verified). Include the date in the id. |
| 15 | LOW | `qts/tcost/tca.py:17-18` vs `docs/BUILD_PLAN.md` §6 | Plan defines reversion vs **next close**; code uses **next open**. Pick one (next close is the more common TCA convention and captures the overnight-plus-day decay) and document. |
| 16 | LOW | `qts/execution/simulator_fills.py:316`, `qts/tcost/tca.py:142` | `adv_shares` passed to the ledger defaults to the **day's realised volume**, not the 63-day ADV, so `participation`/`adv_bucket`/`impact_x` are computed on the wrong denominator unless the caller supplies `adv_shares`. |
| 17 | LOW | `qts/backtest/simulator.py:139-148,175-181` | Broad `except Exception` fallbacks (snapshot → built-in; borrow hook → default fee) hide a broken cost model behind `log.debug`/silence. Should at least `log.warning` once. |
| 18 | LOW | `qts/tcost/impact.py:382-390` | `auction_share_for_date`: third-Friday (`15<=day<=21`) and last-Friday-of-June logic verified correct for 2024-2026. Note FTSE Russell moves to semi-annual reconstitution (June + November) from 2026 — November recon Fridays are not flagged. Also "ADV" in the snapshot includes day `t`'s own volume (tiny look-ahead on the auction size). |
| 19 | LOW | `qts/backtest/simulator.py:183`, `qts/tcost/borrow.py:43` | Borrow is charged once per simulation step at `fee/252`. Correct for daily steps; if `run` is given a sparse `dates` subset (multi-day returns) borrow is under-charged proportionally. |
| 20 | LOW | `qts/core/types.py:45` | `MarketData.spread_bps` is documented as a *half*-spread proxy but named like a full spread; a loader feeding quoted spread would double the linear cost (conservative), one feeding effective half-spread is right. Rename to `half_spread_bps`. |
| 21 | LOW | `qts/backtest/simulator.py` (design) | No cash interest / short rebate / margin financing. For the default 100/100 book (gross ≤ 2, net 0 ± 0.05) this omits positive carry on short proceeds in high-rate regimes (conservative); the config comment "150/50 -> 2.0 gross" is misleading — the constraints produce a dollar-neutral 100/100 book, and a true 150/50 would need ~0.5 NAV of margin interest that is not modelled. |

No CRITICAL findings: timing, NAV identity, units and signs in the daily simulator are correct (see "Verified correct").

## Findings in detail

### 1. `calibrate_b` ignores venue volume (HIGH)

`qts/tcost/tca.py:142-144`:

```python
participation = filled / o["adv_shares"].astype(float)
impact_x = o["sigma_daily"].astype(float) * np.sqrt(participation)
```

The regressor is `sigma*sqrt(F/ADV)`. The cost model (and `FillSimulator`) prices auction fills
with `V = auction_share*ADV` and `b = impact_b_auction`. Regressing auction slippage on the
full-ADV regressor returns `b_auction/sqrt(share)`.

Verification (`/tmp/audit/exec_check.py`): 300 MOC orders through `FillSimulator(auction_share=0.09)`
with `impact_b_auction=0.5`, no noise, `calibrate_b(prior_weight=0)`:

```
calibrate_b recovers b_ols = 1.6667  (= b_auction/sqrt(0.09) = 1.6667)
```

The monthly recalibration job described in README/BUILD_PLAN §6 ("recalibrated from own fills via
TCALedger.calibrate_b") would therefore push `impact_b_auction` from 0.5 to ~1.67 on the first run
and triple auction cost estimates (or, if the user calibrates `b_continuous` from a mixed ledger,
pollute it with auction fills).

Fix: store the venue on the order (`record_order(..., venue=)` or derive from `meta['order_type']`
in `{MOC, LOC}`) and in `decompose()` compute
`impact_x = sigma*sqrt(F / (ADV*auction_share))` for auction orders (auction_share from
`SquareRootImpact.auction_share_for_date(decision_ts)` or a logged realised auction volume).
Better: `calibrate_b(venue=...)` returning one `b` per venue, each regressed on its own `V`.

### 2. Default borrow fee 0.5% on every short (HIGH)

`CostConfig.default_borrow_fee_annual = 0.005` is used whenever `md.borrow_fee` is None
(`snapshot_from_market`, `borrow_fee_vector`, `_borrow_usd`). BUILD_PLAN §4 ("conservative haircut
for small caps in backtests; 5%/yr only as a placeholder") is not implemented, and the
`short_interest` theme explicitly shorts high-SI names where fees of 5-50%/yr are typical.
A 20%-of-book sleeve of shorts at 10%/yr ≈ 2%/yr of NAV ≈ 0.2 Sharpe at 10% vol — larger than
the gap between the phase-1 (0.3) and phase-2 (0.5) hurdles.

Fix: in `snapshot_from_market`, when `md.borrow_fee` is None, derive a proxy from
`short_interest_ratio` and size (e.g. GC 0.5% for SI<2%, 2-5% for 2-10%, 10-30% above, +haircut for
bottom size quintile) and log that a proxy is in use; expose `borrow_fee_proxy_fn` in `CostConfig`.
Also reject shorts in names with no proxy and no datum (treat as unborrowable) as a stress run.

### 3. `align_snapshot` strips keys the cost model needs (MEDIUM)

`qts/backtest/simulator.py:143-145`:

```python
snap = fn(md, t)
if isinstance(snap, dict) and "adv_usd" in snap:
    return align_snapshot(snap, md.tickers, self.cfg.cost)
```

`align_snapshot` (`qts/portfolio/costs.py:253-272`) rebuilds the dict from `SNAPSHOT_KEYS =
("adv_usd","sigma_daily","half_spread","price","borrow_fee")` only. Verification:

```
raw snapshot keys:       ['adv_shares','adv_usd','borrow_fee','date','half_spread','half_spread_bps','price','sigma_daily','venue']
simulator snapshot keys: ['adv_usd','borrow_fee','half_spread','price','sigma_daily']
sim-path estimate venue values: ['auction']   date in snap: None
continuous spread bps raw-snapshot (first 5): [ 0.90 12.30  2.77 14.29  7.38]
continuous spread bps sim-snapshot  (first 5): [2. 2. 2. 2. 2.]
optimizer(venue=continuous) cost frac: 0.000357   simulator charge frac: 0.000455
```

With the default `venue="auction"` the simulator and optimizer agree (both zero spread, same `b`,
same `V`), so the default backtest is internally consistent. But: `run(venue="continuous")` is
silently costed as auction; real spreads never reach `estimate()`; and `date` is lost so
`auction_share_for_date` (special days), the SEC effective date and the TAF holiday are inert in
backtests.

Fix: (a) in `simulator.run` call `self.cost_model.estimate(trades, snap, venue=venue, date=t)`;
(b) make `align_snapshot` preserve unknown keys (`out.update({k: v for k, v in snapshot.items()
if k not in SNAPSHOT_KEYS})`) and reindex `half_spread_bps`/`adv_shares`/`venue` too;
(c) in `TransactionCostModel.estimate` fall back to `half_spread*1e4` when `half_spread_bps` is
absent.

### 4. Pipeline turnover/return misalignment (MEDIUM)

`qts/pipeline.py:188-190` date-joins decision-indexed turnover onto realisation-indexed returns.
Verification on the 5-day hand case (`/tmp/audit/sim_check.py`):

```
turnover (decision idx):   [0.      0.0476 0.1118 0.0863 0.0959]
turnover_on_return_dates:  [0.      0.0476 0.1118 0.0863 0.0959]   <- summary() (correct)
pipeline reindex:          [0.0476 0.1118 0.0863 0.0959 0.    ]   <- pipeline (shifted, first dropped)
```

Each day's cost is deducted from the wrong day's gross return and the initial build (turnover ≈ gross
leverage, the single largest day) is dropped, so `cost_sensitivity` and the hurdle `breakeven` are
slightly optimistic and inconsistent with `summary()["breakeven_cost_bps"]`.

Fix: replace both with `bt.turnover_on_return_dates().reindex(gross.index)`.

### 5. SEC §31 is zero for all pre-2026 dates (MEDIUM)

`fees.py:494-495`: `if d < SEC_FEE_EFFECTIVE: out["sec_fee_per_dollar_sold"] = 0.0`. True only
from 2025-10-01 (FY2026 start) to 2026-04-03. Any historical fill simulation with a date gets zero
SEC fee on sells (actual: FY2021 $5.10 then $92.70/M, FY2023 $8.00/M, FY2024-25 $27.80/M, etc.).

Fix: replace the single switch with a dated rate table (`[(effective_date, rate_per_dollar), ...]`)
and pick the last row ≤ date; default to the config rate when `date is None`.

### 6. No cross-day impact persistence (MEDIUM, design)

Each day's `estimate()` prices the trade as an isolated metaorder in that day's volume. For a
signal with a 20-60 day horizon and GP shrinkage the optimizer moves toward the target over
several days in the same direction; the square-root law applied to the aggregate says cost
scales like `(kQ)^{1.5}` not `k*Q^{1.5}` — a `sqrt(k)` understatement on the impact leg for a
`k`-day build if impact is fully persistent (≈70% per Frazzini et al.). Materiality at the plan's
caps: small in absolute bp (auction-model impact at 0.1% ADV, σ=2% ≈ 10 bp) but 1.4-2x on that
component for 2-4 day builds, and it grows with NAV.

Fix options (cheapest first): (i) in the backtest, apply a `sqrt(k_eff)` multiplier to impact
where `k_eff` is estimated from the autocorrelation of per-name trade sign (a diagnostic already
computable from `res.trades`); (ii) carry a per-name "pending permanent impact" state
(`0.7*impact_bps` decaying with a power law) and add it to the next day's effective fill price;
(iii) report both numbers.

### 7. Delisting handling (MEDIUM)

`simulator.py:271-274,288`: NaN return at `t+1` → `r=0`, position removed from `w_pre` with no entry
in `trades`/`costs`. (a) Last-price exit is optimistic for longs in bankruptcies; (b) the exit is
free and invisible to turnover. Fix: accept an optional `md.extra["delist_return"]` panel (Sharadar/
Norgate provide it) and use it as `r` on the delisting row; book the liquidation as a trade
through `cost_model.estimate` (at least spread + fees) and include it in `X`/`TO`.

### 8. `_cost_frame` silent zero (MEDIUM)

`simulator.py:161-172` returns zeros when no alias matches and no `total` exists, and `fillna(0)`
masks NaN costs. Fix: raise `ValueError(f"estimate() returned unrecognised columns {list(est.columns)}")`
when `matched` is False and there is no `total`; and assert `np.isfinite(...)` on traded names.

### 9. LOC limit rounding (MEDIUM)

Verified: `loc_limit("BUY", 5.0, 3.0) == 5.0`, `loc_limit("BUY", 10.0, 3.0) == 10.0`. Fix:
`px = math.ceil(px*100)/100` for buys, `math.floor` for sells (and tick=0.0001 below $1).
Same for `passive_limit` but in the opposite direction (buys floor, sells ceil) to stay passive.

### 10. NaN sigma → free impact in `cvx_terms` (MEDIUM, direct API only)

Verified: with `sigma_daily=NaN` on the only traded name, `cvx_terms` returns explicit fees only
(2.0e-6) while `estimate` returns NaN. Fix: `fillna(sigma.median())` (as `align_snapshot` does)
and make `estimate` raise/median-fill rather than propagate NaN.

### 11-21. See table; fixes are one-liners (rate updates, docstring sync, `warning` instead of
`debug`, `order_id` with date, rename `spread_bps`, note on financing).

## Verified as correct

* **Timeline / look-ahead** (`simulator.run`): snapshot, alpha and risk at `t` use data ≤ `t`;
  trades at close `t` priced on `nav_t`; returns `R.iloc[i+1] = close[t+1]/close[t]-1` earned by
  `w_post` decided at `t`; no return at `t` accrues to a position decided at `t`. Hand case: A
  +10%/−10%/0/+21% with `w_A=0.5` reproduces NAV `1.00→1.05→0.945→0.98795→1.0927M` exactly.
* **NAV identity** `nav_{t+1} = nav_t(1+w_post·r) − costs_t` exact; `w_pre(t+1) = w_post(1+r)·nav_t/nav_{t+1}`
  verified (trades on day 2 equal `0.5 − 0.5·1.1·1e6/nav_1` etc.).
* **Costs in USD at `nav_t`**: `estimate(x·nav_t)` matches an independent recomputation to 1e-12;
  `cost_bps_of_nav` and `traded_usd` use the pre-trade NAV consistently.
* **Borrow**: charged on post-trade shorts, `|w|·nav_t·fee/252`, one day per step (500 USD on
  0.5 NAV short at 25.2%/yr, as expected).
* **`exante_var` alignment**: `exante_var.shift(1).reindex(returns.index)` maps `EXV[k]` (position
  decided at `dates[k]`) onto the return realised at `dates[k+1]` — correct.
* **Turnover definitions** are mutually consistent: `turnover = Σ|x|` (two-sided),
  `avg_cost_bps_per_side = trading_cost/Σ|x|·nav` (cost per dollar traded),
  `breakeven = mean(g)/mean(Σ|x|)·1e4`, `cost_sensitivity` charges `c·Σ|x|`, and
  `effective_holding_days = gross/(Σ|x|/2)`. Borrow is excluded from all three cost-per-side
  quantities (documented; keep in mind when reading cost multiples).
* **cvx_terms ≡ estimate**: for random weight trades, `b·σ·|x|^{1.5}/sqrt(V/nav)` (fraction of NAV)
  equals `estimate().impact_usd/nav` to machine precision for auction (`V=0.09·ADV`, `b=0.5`) and
  continuous (`V=ADV`, `b=1`). Derivation: `C_$ = b σ Q^{1.5}/√V`, `Q=x·nav` ⇒
  `C_$/nav = b σ x^{1.5} nav^{0.5}/√V = b σ |x|^{1.5}/√(V/nav)` — code matches. Explicit fees differ
  only by the per-order $0.35 minimum (linear approximation in the optimizer, exact in `estimate`).
  `SimpleCostModel` likewise consistent.
* **Auction venue** `V = share·ADV`; `auction_share_for_date`: 2026-03-20/06-19/09-18/12-18 → 0.20,
  2026-06-26, 2025-06-27, 2024-06-28 → 0.34, ordinary Friday → 0.09, non-Friday → 0.09.
* **fees.py vs IBKR Pro tiered**: commission `min(max(0.0035·q, 0.35), 1%·value)`, clearing
  `min(0.0002·q, 0.5%·value)`, SEC on sells only in dollars, TAF per share sold with per-trade cap,
  TAF holiday 2026-10-01..12-31 inclusive, CAT both sides. SEC $20.60/M from 2026-04-04 confirmed
  against the SEC fee rate advisory 2026-2 and FINRA Information Notice 3/17/26 (charge date = trade
  date). The hand-computed test (`tests/test_tcost.py:14-28`) is arithmetically right given the
  config rates: buy $3.703, sell $4.899, 1.72 bp round trip.
* **TCA decomposition**: `delay+impact+opportunity+fees = total_is` (positive = cost for both sides),
  reversion separate, through-origin OLS with `se = sqrt(RSS/(n−1)/Σx²)`, shrinkage
  `(n·b_ols + w·prior)/(n+w)`, `right=False` buckets. Auction orders pass `half_spread_bps=0` so no
  spread is subtracted from auction slippage. Synthetic test recovers `b=0.8`.
* **Router / fills**: LOC buy limit above / sell below indicative; offset floored at 1.5× expected
  auction impact (so an LOC fills when the auction prints at the indicative, is rejected at +20 bp);
  `maxPctVol` capped at 0.5; `ArrivalPx` for half-life < 5 d; MOC/LOC fill at `close·(1+s·impact)`
  with the correct sign; ALGO partial fills `min(qty, maxPctVol·volume)` with spread + impact on
  the right side; fees on filled shares with the sim date; ledger records decision vs arrival
  (close for auction, open for continuous).
* **Scheduler**: U-curve sums to 1 (13 and 26 buckets, with `close_share`); VWAP/POV sums; AC
  closed form `x_j = X·sinh(κ(T−t_j))/sinh(κT)` with `cosh(κτ) = 1 + κ̃²τ²/2`, `κ̃² = λσ²/η`
  (κ=0.916 vs κ̃=0.949 for the test inputs, front-loaded, `λ→0` ⇒ TWAP); `urgency_from_half_life(5)
  = sqrt(0.01·0.25) = 0.05`.
* **Metrics**: Sharpe `mean/std(ddof=1)·√252`; Lo (2002) `SR·q/√(q+2Σ(q−k)ρ_k)` (shrinks AR(1)
  returns toward the theoretical value); NW with Bartlett weights and `⌊4(T/100)^{2/9}⌋` lags
  (lags=0 reproduces the iid t); max drawdown on compounded net returns ≡ NAV drawdown
  (identical to 1e-16); Calmar, Sortino (full-sample downside RMS) standard.
* **format_result** reads the right fields and labels turnover as two-sided and cost as per side.

## Verdict

As an *accounting* engine the daily simulator is sound: no look-ahead, the NAV identity holds to
machine precision, costs and borrow are booked in dollars on the right NAV, the optimizer's convex
cost and the simulator's realised cost are the same function of the same inputs, and the
turnover/cost-per-side/breakeven definitions are mutually consistent. Nothing here would flip the
sign or scale of a net Sharpe. But the net numbers should **not** be read as conservative by
construction. Three optimistic biases remain and all push the same way: shorts are financed at a
flat 0.5%/yr unless borrow data are supplied (material for the short-interest sleeve), impact is
priced one day at a time with no permanent component carried across a multi-day build, and
delistings exit free at the last print. Against these sit two conservative biases — every trade is
costed at the auction model (which is dearer than in-session execution for anything above ~0.02%
ADV) and no interest is earned on cash or short proceeds — plus the simulator's deliberately
all-in explicit fees. The pipeline's turnover misalignment (#4) and the `align_snapshot` key loss
(#3) are plumbing bugs that make the default `venue="auction"` run self-consistent but any other
configuration subtly wrong. Recommendation before the phase-1 gate is trusted: fix #3, #4 and #8
(mechanical), populate or proxy borrow fees with a HTB haircut (#2), add a multi-day impact
haircut or persistence state (#6), add a delisting-return hook (#7), and fix `calibrate_b`'s venue
volume (#1) before the first live recalibration. With #2 and #6 addressed, a net Sharpe from this
engine at the plan's participation caps can reasonably be called conservative; without them it is
an unbiased-to-optimistic estimate with roughly 0.1-0.3 of annual Sharpe at stake.
