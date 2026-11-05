# Second-pass audit: costs, execution, backtest engine, API

Date 2026-11-05. Independent re-review after two refactor waves. Scope: `qts/core/costs.py`,
`qts/tcost/*`, `qts/execution/*`, `qts/backtest/*`, `qts/portfolio/costs.py` (+ the optimizer's
cost plumbing), `qts/pipeline.py`, `qts/runs/artifacts.py`, `api/qts_api/**`, the UI selectors
that derive series. No source file was modified. Scripts: `/tmp/audit2_c/*.py`.

Test suites: `tests/test_tcost.py test_execution.py test_backtest.py test_cost_equivalence.py
test_simulator_handcomputed.py test_signal_lag_real.py` all pass; `api/` suite 51 passed (118 s).

Severity: CRITICAL = wrong sign/scale/timing that would mislead a go/no-go decision; HIGH = wrong
number, direction-preserving; MEDIUM = fragility, silent optimism, or a safety gap; LOW = minor.

## Summary

No CRITICAL or HIGH findings. On the default config the backtest's cost, timing and NAV
accounting reproduce an independent hand computation **exactly** (real `TransactionCostModel`, real
`SinglePeriodOptimizer`, lag 1, split, delisting, forced exit, dated fees). The remaining issues are
in derived gates (borrow left out of breakeven), TCA with a non-default exponent, config-override
semantics, and local-server hardening.

| # | Sev | Where | Finding |
|---|-----|-------|---------|
| 1 | MEDIUM | `qts/pipeline.py:668-670`, `qts/backtest/metrics.py:139-176`, `qts/backtest/simulator.py:160` | Breakeven cost and the cost-sensitivity table use **gross** returns, which leave out borrow. The `breakeven_cost` hurdle (>= 2x modelled cost) is therefore optimistic for short-heavy books. On the real run: sensitivity at 1x gives net Sharpe **0.528** against the actual **0.463**; breakeven **26.2 bp** against **24.3 bp** once borrow is deducted. |
| 2 | MEDIUM | `qts/tcost/tca.py:280` | `impact_x = sigma*sqrt(F/V)` hard-codes e = 1.5, but `CostConfig.impact_exponent` can be set anywhere in (1, 2] (API bound 1.0001-2). With noise-free fills at e = 1.25, `calibrate_b_by_venue` returns 0.88 / 2.54 for true 0.3 / 1.0 (about 2.9x). At e = 1.8 it returns 0.084 / 0.333. |
| 3 | MEDIUM | `api/qts_api/services/config_schema.py:61-108,363-402`; `qts/core/config.py:504` | "Strict" override validation only covers types plus a curated bounds list. `SystemConfig.validate()` only checks the round trip. Accepted with 202: negative `cost.clearing_per_share`, `cat_fee_per_share`, `finra_taf_per_share_sold`, `auction_exchange_fee_per_share` (negative fees are booked silently, because `CostEstimate` only checks that values are finite); `cost.auction_volume_share_default = 0` (the job crashes later in `LiquiditySnapshot`); `cost.impact_b_continuous = 1e12`; `alpha.lgbm.n_estimators = 1e9`. |
| 4 | MEDIUM | `api/qts_api/main.py:60-66` | No Host-header check. A request with `Host: attacker.example` gets 200, so a DNS-rebinding page can read every run, launch jobs up to 3000 x 6000, and cancel jobs. CORS is set correctly (an evil-origin preflight gets 400 and no ACAO header; a `text/plain` POST gets 422), but CORS does not protect against rebinding. |
| 5 | LOW | `qts/tcost/model.py:73-91,363-367` | The persistence multiplier `k^e-(k-1)^e` assumes the earlier days of a streak had the same notional. The docstring says it is an upper bound; it is not. When daily trades halve (GP-style convergence) the approximation gives 0.785x the exact streak increment `C(Q_k)-C(Q_{k-1})` (0.66x at e = 1.8). It overstates when trades grow (1.26x). On the real run it came out **1.18x** (conservative), so the effect is small, but the sign depends on the trade path. |
| 6 | LOW | `qts/execution/router.py:223-279` vs `qts/tcost/model.py:21-29` | The routed venue says it prices "the way the router executes", but the router never splits an order. An order above the auction cap goes **entirely** to ClosePx/Vwap (continuous, pays the full spread). The model saves `h*min(q,K)` on those orders. On the real run that is 5% of traded notional, and spread is 0.9% of costs, so the effect is immaterial. Separately, slow signals get `maxPctVol` of 1-2.6% while the optimizer allows 5% ADV in one fill. That is also immaterial at $5M NAV (0.2% of notional above 1% ADV). |
| 7 | LOW | `qts/tcost/model.py:423-427` | When `beta_a = b_a/phi^(e-1) > b_c` (e.g. `impact_b_auction = 0.5`, or after a TCA update), the routed surrogate prices the whole trade at `beta_a`. That is a valid upper bound but loose: per-name impact is up to **+66%** above `estimate`, and the total is 1.58x. The optimizer under-trades; the backtest itself is unaffected. |
| 8 | LOW | `qts/tcost/model.py:231-266` | `borrow_fee_source` reads `"data"` whenever an `md.borrow_fee` panel exists, even if the row is all NaN and every fee comes from the proxy or the default (verified). Names with NaN short interest get GC 0.5% **without** the small-cap multiplier (`proxy.where(si.notna())`, l.253), so the data-poor small caps are the ones that miss the conservative haircut. |
| 9 | LOW | `qts/validation/ledger.py:113-131` | `log()` does read-modify-write with a fixed `path + ".tmp"`. With `QTS_MAX_WORKERS > 1`, or the CLI and API running at the same time, concurrent jobs can drop each other's trial records, which understates the DSR trial count (optimistic). The default `max_workers = 1` is safe. |
| 10 | LOW | `api/qts_api/services/jobs.py:423-442` | Any exception in `Process.start()` (seen in this audit: spawn bootstrapping error; in production: EAGAIN or a pickling error) kills the dispatcher thread with no handler. The popped job and every later job stay `"queued"` forever, and nothing is reported. |
| 11 | LOW | `api/qts_api/services/registry.py:250-256` | A run id longer than about 255 characters gives **500** (`OSError ENAMETOOLONG`) instead of 404. Traversal is otherwise blocked: `..`, `%2e%2e`, `..%2F..`, `_jobs`, `_staging`, `.hidden`, `\`, NUL and `trial_ledger.jsonl` all return problem+json 404. |
| 12 | LOW | `api/qts_api/services/registry.py:211,265-277` | Entries take their id from `manifest.run_id`, not the directory name. A copied run directory shows up as **two list rows with the same id** (duplicate React keys), and both share one frames-cache slot. |
| 13 | LOW | `api/qts_api/schemas.py:310-325` | `LaunchRunRequest` runs in pydantic lax mode: `"stocks": "60"` and `"ml": "true"` are accepted. Labels may contain NUL and control characters. |
| 14 | LOW | `qts/backtest/simulator.py:467` | The optimizer prices costs with the signal date's auction share. On a quad-witching or Russell day `t` it prices a normal day, although the calendar is known in advance. Realised cost is lower than expected (suboptimal, not optimistic). |
| 15 | LOW | `qts/runs/artifacts.py:186` | `holdings.parquet` shows `alpha.as_of(d)` on decision date `d`; with lag 1 the book was formed on `as_of(d-1)`. |
| 16 | LOW | `qts/backtest/simulator.py:4` | The module docstring still says `signal_lag_days` "default 0"; the config default is 1. |
| 17 | LOW | (still open from the first audit) | #19: borrow is charged as one day per step when `dates` is sparse. #20: `MarketData.spread_bps` holds a *half*-spread. #21: no short rebate or cash interest (conservative). Also `PUT /api/runs` returns 404 "No API endpoint" rather than 405. |

## Details and fixes

### 1. Borrow missing from breakeven and cost sensitivity (MEDIUM)

`evaluate()` calls `M.cost_sensitivity(gross, to, …)` and `M.breakeven_cost_bps(gross, to)`.
`returns["gross"] = w·r` comes before every cost, including borrow, and `avg_cost_bps_per_side`
leaves borrow out too. Borrow is not a per-side cost, so it simply disappears from the gate.
`/tmp/audit2_c/real_run.py` (120 names, 250 OOS days, lag 1, routed):

```
cost shares: spread 0.009  impact 0.763  fees 0.088  borrow 0.140
cost_sensitivity net_sharpe @1x: 0.528   actual net: 0.4626
breakeven (pipeline, ignores borrow): 26.159   breakeven net of borrow: 24.319
```

With the HTB proxy (up to 25%/yr, 50% for small caps) a `short_interest` sleeve could flip the gate.
**Fix:** let `r_hb = gross - borrow_usd/nav_pre` (positionally aligned) and use it in place of
`gross` in `breakeven_cost_bps`, `cost_sensitivity`, and `BacktestResult.summary()["breakeven_cost_bps"]`.
Equivalently, add a `borrow_ret` column to `returns`.

### 2. TCA regressor ignores `impact_exponent` (MEDIUM)

`/tmp/audit2_c/tca.py`: 300 orders through `FillSimulator` with no noise, `calibrate_b_by_venue(prior_weight=0)`:

```
e=1.5 : truth auction=0.3 cont=1.0 -> 0.3000 / 1.0000   (exact; venue-aware V works)
e=1.25: truth auction=0.3 cont=1.0 -> 0.8828 / 2.5381
e=1.8 : truth auction=0.3 cont=1.0 -> 0.0838 / 0.3334
```

**Fix:** `tca.py:280` → `impact_x = sigma * (filled / venue_vol) ** (e - 1.0)` with
`e = float(self.cost_cfg.impact_exponent)`. Update the docstrings at l.28-31 and l.355-356.

### 3. Override validation is type-strict but not semantically strict (MEDIUM)

`/tmp/audit2_c/api_launch.py`. Correctly rejected: unknown keys/sections, `__class__`,
`cost.__dict__`, list fields, bool/str for numbers, 5.5 for int, NaN/Infinity tokens, out-of-range
curated bounds, enum violations, reserved `seed`/`signal_lag_days`, dict values, extra top-level keys,
stocks > 3000, days > 6000, labels over 120 chars, bad venue. **Accepted with 202:** the cases in the
table. `explicit_cost_vec` then books negative clearing/CAT/TAF/exchange fees, and `CostEstimate`
accepts them because it only checks that values are finite, despite its docstring saying ">= 0".
**Fix:** add `(0, None)` bounds for every `cost.*` fee, spread, borrow and multiplier field;
`(1e-3, 1)` for `auction_volume_share_default`; upper bounds for `impact_b_*` (e.g. 10) and the
LightGBM/MLP sizes. Better still, put a semantic `SystemConfig.validate()` in `qts/core/config.py`
so the CLI benefits too. Make `CostEstimate.__post_init__` reject negative USD columns.

### 4. DNS rebinding (MEDIUM)

`TestClient.get("/api/runs", headers={"Host": "attacker.example"})` → 200. **Fix:**
`app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)` with a default of
`["localhost", "127.0.0.1", "[::1]"]` (`QTS_ALLOWED_HOSTS`), plus a cap on queued jobs (for example
429 above 10). uvicorn's default bind of 127.0.0.1 is kept (Makefile, README).

### 5. Persistence approximation (LOW)

`/tmp/audit2_c/persistence.py` (`approx = m_k*C(q_k)`, `exact = C(Q_k)-C(Q_{k-1})`):

```
e=1.5 halving (GP-style)   day-by-day approx/exact=[1. 0.772 0.619 0.491 0.384]  total 0.785
e=1.5 growing              total 1.256
real run (exact-streak recomputation with each day's snapshot): approx/exact = 1.184
```

**Fix:** the simulator knows the trades. Carry `Q_prev` (the cumulative same-direction USD of the
current streak; reset when the sign changes or a day has no trade) and pass it to `estimate`. The
impact leg becomes `C(Q_prev+q) - C(Q_prev)` at today's V (for routed, apply it to the continuous
increment term). Then drop the "upper bound" sentence.

### 6-8

See the table. For 6, either make the router split (LOC for `min(q,K)`, algo for the rest), or price
over-cap orders as continuous in `estimate(venue="routed")`. For 8, return the per-name source
(`data`/`proxy`/`default` Series or counts) and apply the small-cap multiplier to the GC default as
well.

## Verified correct (numerically)

* **Routed venue** (`/tmp/audit2_c/routed.py`, 400 random names, buys from 1e-5 to 10% ADV, normal,
  quad-witching and Russell days, b_a in {0.2, 0.3, 0.5}). Hand derivation and `estimate()` agree
  exactly (max abs diff 0). The hand derivation: `q_a = min(q, κ·φ·A)`; spread `h·q_c`; impact
  `C_{φA,b_a}(q_a) + C_{A,b_c}(q) − C_{A,b_c}(q_a)`; fees as two orders (auction leg with exchange
  fee, separate commission minimums). The increment ordering (auction first) is the conservative
  one: the alternative ordering is cheaper by `(b_c−β_a)(q^e−q_a^e−q_c^e) ≥ 0`.
* **Surrogate convexity and exactness.** `|x|^e` is convex for e ≥ 1, so `pos(|x|^e − k^e)` is convex
  (pos is convex and nondecreasing). `|x|^e − min(|x|,k)^e ≡ pos(|x|^e − k^e)` and
  `|x| − min(|x|,k) ≡ pos(|x|−k)`, so for `β_a ≤ b_c` the spread and impact terms are exact.
  Measured per-name impact error is at most 7.7e-16 in all six such cases; the total surrogate/estimate
  ratio is 1.0013-1.0023, from fees only. At the defaults `effective_auction_b(0.09) = 1.0` exactly,
  with no kink term. Continuous / auction surrogate/estimate = 1.000045 / 1.000058 (min-commission
  only).
* **Simulator, lag 1, real `TransactionCostModel` + real optimizer** (`/tmp/audit2_c/sim_fixture.py`:
  12 names, 2-for-1 split inside the window with `close_raw`/actual volume, delisting, eligibility
  exit, HTB small cap with SI 25%, spreads panel, dated fees across 2026-04-04). Results:
  * alpha and risk are called on `t−1` on every day; costs are recomputed independently by hand from
    raw panels on `t`'s snapshot. Max relative error: spread 0, impact 0, fees 3e-16, borrow 0.
    This covers ADV from raw price x actual shares, `adv_shares` restated to day-t basis, fees from
    shares = USD/raw price, the SEC rate 0 before and $20.60/M after 2026-04-04 (so the trade date
    **is** passed), the date-aware auction share, the streak multiplier, and the SI/DTC/small-cap
    borrow proxy.
  * The NAV identity holds with max relative error 0, including the delisting haircut.
  * The delisting is liquidated on its last decision row: haircut −30% against the position sign,
    spread + fees, impact zeroed, booked in trades/turnover; weight is 0 afterwards.
  * The forced exit follows `w0·(1−k/5)` exactly (0.0373, 0.0280, 0.0186, 0.0093, 0).
  * A run starting at `md.dates[0]` holds on the first day (`n_no_signal_days = 1`).
  * `avg_cost_bps_per_side` and `breakeven_cost_bps` match hand recomputation exactly; turnover is
    positional (decision t to realisation t+1).
* **TCA:** venue-aware calibration recovers 0.3000 / 1.0000 exactly at e = 1.5; the decomposition
  sums to `total_is` (max error 0); reversion is measured against `next_close` (diff 3e-14).
* **Router and IBKR payloads:**
  * Participation is `urgency_from_half_life` clipped to [1%, 5%] (h = 5/10/20/40 → 5.0/2.59/1.32/1.0%).
  * The auction share is date-aware: a 2,500-share order is ALGO on 2026-03-10 and LOC on
    2026-03-20 and on Russell days 2026-06-26 / 2026-12-11.
  * LOC limits round through the indicative: buys ceil, sells floor (e.g. $7 → 7.01/6.99; $100 → 100.03).
  * The ClosePx payload has only `maxPctVol` (no `noTakeLiq`); Vwap keeps `noTakeLiq = "0"`.
  * Two dated external facts checked against the primary sources:
    [SEC Fee Rate Advisory 2026-2](https://www.sec.gov/rules-regulations/fee-rate-advisories/2026-2)
    ($20.60/M effective 2026-04-04, $0 before) and the
    [FTSE Russell December 2026 schedule](https://www.lseg.com/en/media-centre/press-releases/ftse-russell/2026/ftse-russell-announces-december-2026-russell-us-indexes-reconstitution-schedule)
    (effective after the close on 11 Dec 2026).
* **API** (`/tmp/audit2_c/api_check.py`, `api_jobs.py`, against a real pipeline run). Series values
  agree with recomputation from parquet:
  * `nav` = nav.parquet on realisation dates; drawdown equals `(1+net).cumprod()/cummax−1`, and its
    minimum equals `max_drawdown_net`.
  * Turnover and ex-ante vol are aligned positionally.
  * Costs equal costs.parquet, components sum to the total, and `nav` is the pre-trade NAV.
  * The UI's monthly compounding and rolling Sharpe (`ui/src/entities/run/selectors.ts`) are correct
    as read.
  * NaN/inf in summary or returns become `null` (no raw NaN in any body).
  * The cache invalidates on JSON and parquet rewrites and on deletion.

  Error handling and jobs:
  * Errors return problem+json (404/409/422/500).
  * A job runs in a spawned process; the staging directory is renamed atomically, the manifest
    `run_id` matches, and staging is cleaned up.
  * Cancelling a job mid-run kills the worker, leaves no run directory, gives a 409 "run-failed",
    and the worker cannot overwrite the cancelled status.
  * Shutdown marks interrupted jobs; a new app instance recovers the job state.
  * CORS allows only `http://localhost:5173`.

## Verdict

**Are net backtest numbers conservative?** Yes, for the default configuration. Execution-cost
accounting is exact against a hand computation, timing is honest (lag 1, costs on the execution
day's liquidity), and the deliberate biases all point toward overstating cost:
* delisting −30% for longs **and** shorts;
* no short rebate or cash interest;
* the HTB borrow proxy;
* persistence that measured +18% on a real run;
* `b_a` chosen so the auction never beats continuous on impact.

The optimistic pieces are small: the routed spread saving on over-cap orders (6), the path-dependent
persistence approximation (5), and data-poor small caps missing the borrow haircut (8). The one
place a reported gate is optimistic is **breakeven / cost sensitivity, which leave out borrow (1)**;
fix that before relying on the `breakeven_cost` gate for short-heavy sleeves. Do not change
`impact_exponent` until finding 2 is fixed.

**Is the API safe to run locally?** Yes, bound to 127.0.0.1 as the Makefile and README do:
* no path traversal into files;
* run and job ids are confined;
* overrides cannot reach anything but scalar config leaves;
* work runs in an isolated process.

Before using it on a machine that browses the web while the server runs, add a Host allow-list
(finding 4) and semantic bounds on cost overrides (finding 3). Keep `QTS_MAX_WORKERS = 1` until the
ledger write is locked (finding 9).
