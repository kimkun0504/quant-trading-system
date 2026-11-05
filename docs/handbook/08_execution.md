# Execution

<callout icon="🧭" color="gray_bg">
Execution turns the optimizer's target weights into real orders at Interactive Brokers, and makes their cost measurable. The pipeline has five steps. Weight changes become whole-share orders, and dust is dropped. A rule-based router gives each order a tactic. Small orders go into the closing auction as limit-on-close (LOC) orders priced a little through the indicative close. Larger ones go to IBKR's Close Price or VWAP algorithms, capped at 5% of volume. Fast-decaying signals use an Arrival Price algo at the next open. Short-term-reversal trades rest as passive limits. Each order is then serialised into an IBKR API payload. Before live trading it is filled by a daily-bar fill simulator or a dry-run broker. Every fill lands in the TCA ledger that recalibrates the cost model. Scheduling math (VWAP, POV, Almgren–Chriss) is included as reference. Under the square-root impact law, how you slice an order matters much less than how big it is. No network code exists yet; live connectivity through `ib_async` is a later phase.
</callout>

## 1. From target weights to orders

### What it is
The optimizer outputs **target weights**: fractions of NAV per stock, with shorts negative. A broker accepts **orders**: buy or sell an integer number of shares of a ticker. `targets_to_orders` bridges the two.

### Why we need it here
Three practical problems sit between a weight and an order. Shares are integers. Tiny weight changes produce orders whose fixed costs exceed any benefit. And every order needs an identifier that stays unique across days so its fills can be matched in the TCA ledger.

### How it works
For each ticker in the union of target and current holdings (missing = weight 0):

$$
q_i = \left\lfloor \frac{\lvert w^{\text{tgt}}_i - w^{\text{cur}}_i\rvert \cdot \text{nav}}{P_i \cdot L} \right\rfloor L
$$

where:
- $w^{\text{tgt}}_i, w^{\text{cur}}_i$ are target and current weights (fraction of NAV);
- $\text{nav}$ is the portfolio value in dollars;
- $P_i$ is the decision price, usually the latest price when the decision is made;
- $L$ is the lot size (default 1 share);
- $q_i$ is the order quantity in shares, **rounded down** so the order never overshoots the target. Side = BUY if the dollar change is positive, else SELL.

Orders are skipped when $q_i = 0$, when $q_i P_i$ < `min_notional` (default **\$200**, the "dust" threshold), or when the price is missing or non-positive.

Each order gets:
- `order_type="MOC"` and `venue_hint="auction"` as **defaults** that the router then overrides;
- `order_id = "{YYYYMMDD}-{ticker}-{side}-{shares}"` when a decision timestamp is supplied. Without the date prefix, the same ticker/side/size on two days collided and a persistent TCA ledger raised `KeyError` on day two (audit #14);
- `meta` with `decision_px`, `target_w`, `current_w`, `decision_ts` and anything passed per ticker, e.g. `{"sleeve": "reversal", "half_life_days": 3}`. The router reads these.

### Why this choice
- **Round down, not to nearest.** Overshooting a short-side target by one share is harmless, but systematically rounding up inflates gross exposure on hundreds of names.
- **A \$200 dust floor** rather than a percentage. Below about 100 shares, IBKR's \$0.35 minimum commission is a larger fraction of the trade (→ see **Transaction costs › Explicit fees**). A \$200 trade pays at least 17.5 bp in commission alone. The optimizer's linear fee model does not see this, so execution filters it.

### In the code
`qts/execution/orders.py::Order` is a dataclass with `ticker, side, qty, order_type ∈ {MOC, LOC, LMT, MKT, ALGO}, limit_px, algo, algo_params, tif, venue_hint, order_id, meta`. It validates side and type and rejects negative quantities (direction lives in `side`). `targets_to_orders(w_target, w_current, nav, prices, lot_size=1, min_notional=200.0, decision_ts=None, meta=None)`.

### Pitfalls & checks
- **Stale decision prices.** If `prices` is yesterday's close and the stock gapped, share counts are off by the gap. The TCA ledger records it as delay cost.
- **Dust accumulates.** A name whose target keeps moving by \$150 a day is never traded, so its weight drifts. The optimizer sees the drifted weight the next day, so this self-corrects.
- Tests: `tests/test_execution.py::test_targets_to_orders_rounding_and_dust`, `test_order_ids_unique_across_days`.

## 2. US market microstructure primer

### What it is
*Market microstructure* is the study of how orders become trades. This section covers the minimum a software engineer needs to read the router.

**The limit order book (LOB).** Each exchange keeps, per stock, two priority queues of resting **limit orders**: bids (buy at ≤ price) sorted highest first, and asks (sell at ≥ price) sorted lowest first. Within a price level, earlier orders have priority (price-time priority). The top of the book is the best bid and best ask, and the gap between them is the **spread**. US equities trade on more than a dozen exchanges plus off-exchange venues. Reg NMS requires routing to the best displayed price, and a broker's "SMART" router chooses among venues.

**Market vs limit orders.**
- A **market order** executes immediately against the best available prices. It is certain to fill, but the price is uncertain, and it pays the half-spread plus any book it walks through. It *takes* liquidity.
- A **limit order** specifies the worst acceptable price. If the price is *marketable* (a buy at or above the ask) it executes at once like a market order with a cap. Otherwise it rests in the book, *provides* liquidity, and fills only if someone trades against it. The price is certain but the fill is not.
- **Tick size.** Under Reg NMS Rule 612, prices of stocks at \$1 or more are quoted in \$0.01 increments, and below \$1 in \$0.0001. A 2024 SEC amendment introduces a half-penny tick for some tightly quoted stocks. The code assumes \$0.01 (see Pitfalls).

**The closing auction.** At 16:00 ET each listing exchange runs a **call auction** that sets the official closing price. Instead of matching continuously, it collects orders and crosses them all at the single price that maximises executed volume.
- **MOC** (market-on-close): buy or sell at whatever the closing price is. Guaranteed to participate, no price protection.
- **LOC** (limit-on-close): participate only if the closing price is at or better than the limit.
- **NYSE**: MOC/LOC orders can be entered, modified or cancelled until **15:50 ET**. After that, only orders on the contra side of a published significant imbalance are accepted. Imbalance information is published from 15:50 and then every second when it changes. Floor-broker "D-orders" can still be inserted late, and BMLL finds no outsized next-open deviation despite this.
- **Nasdaq Closing Cross**: MOC/LOC entry also closes at 15:50 in the code's convention (`auction_deadline`), with imbalance-only orders accepted afterwards. Nasdaq publishes a Net Order Imbalance Indicator with *near* and *far* indicative clearing prices. The far price considers only auction orders; the near price also includes the continuous book. The two converge as 16:00 approaches, and their convergence is the market's real-time estimate of the close.
- An **imbalance** is the excess of buy over sell interest (or vice versa) at the indicative price. A large published buy imbalance attracts contra-side sellers, which is how the auction finds a price.

**Why a daily rebalancer defaults to the close.**
1. **Liquidity.** The close is the single deepest moment of the day: about 9% of US daily volume on a normal day, about 20% on quarterly index/expiry days, about 34% on Russell reconstitution day (BMLL).
2. **Benchmark consistency.** The backtest marks positions and computes returns close to close (→ see **Backtesting**). Trading at the close makes the realised fill price the backtest's price, which minimises live-vs-backtest tracking error.
3. **No spread.** A call auction has one price for everyone, so there is no half-spread to cross (→ see **Transaction costs › Bid-ask spread**).
4. **Momentum accrues overnight.** Momentum returns accrue mostly overnight, while most other anomalies earn intraday, and the pattern is stronger in large caps (Lou, Polk & Skouras). Entering momentum longs at the close captures that.
5. **Simplicity.** One decision time, one order per name, one fill.

The trade-off is that auction volume is only about 9% of the day. Under the cost model's priors that makes the auction *more* expensive per dollar for anything but very small orders (→ see **Transaction costs › Venue**). That tension is why larger orders are routed to in-session algorithms.

### References
- [NYSE, Closing auction process fact sheet](https://www.nyse.com/publicdocs/nyse/NYSE_Auctions_Closing_Process_Fact_Sheet.pdf)
- [BMLL, Into the close: unpacking US closing auction dynamics](https://www.bmlltech.com/news/market-insight/into-the-close-unpacking-u-s-closing-auction-dynamics-and-the-impact-of-the-russell-reconstitution)
- [Lou, Polk & Skouras, A tug of war: overnight versus intraday expected returns](https://www.fmg.ac.uk/cm/publications/discussion-papers/tug-war-overnight-versus-intraday-expected-returns)

## 3. Broker algorithms (IBKR)

### What it is
A **broker algorithm** ("algo") is a parent-order service: you hand IBKR a parent order with a strategy name and parameters, and its servers slice it into child orders over time. Each algo optimises a different **benchmark**, the price its performance is judged against.

| Algo (`algoStrategy`) | Optimises | Typical use | Key parameters used here |
|---|---|---|---|
| VWAP (`Vwap`) | Fill price close to the **volume-weighted average price** over a window: trades in proportion to expected volume | Larger orders with moderate urgency | `maxPctVol`, `startTime`, `endTime`, `noTakeLiq` |
| TWAP (`Twap`) | Fill close to the **time-weighted** average: equal slices per time unit | Illiquid names where volume curves are unreliable | `startTime`, `endTime` (not used by the router) |
| POV / Percent of Volume (`PctVol`) | Participate at a **fixed share of market volume** (1–50%) until done | Capping footprint with uncertain completion time | `pctVol`, `noTakeLiq` (not used by the router) |
| Arrival Price (`ArrivalPx`) | Minimise **implementation shortfall vs the arrival mid**; front-loads according to risk aversion | Urgent, fast-decaying signals | `maxPctVol` (0.1–0.5), `riskAversion` ∈ {Get Done, Aggressive, Neutral, Passive}, `forceCompletion`, `allowPastEndTime` |
| Close Price (`ClosePx`) | Minimise slippage vs the **closing price**: works the order into the close, trading off impact against the risk of the close moving | Low-urgency orders too big for the auction alone | `maxPctVol`, `riskAversion` |
| Adaptive (`Adaptive`) | Not a scheduler: a **smart limit order** that works between bid and ask to improve on the spread | Single orders that should not cross the full spread | `adaptivePriority` ∈ {Urgent, Normal, Patient} |

Value ranges come from the TWS API reference (`qts/execution/ibkr.py` docstring): `maxPctVol` 0.01–0.5, times as `"hh:mm:ss TMZ"`. That GitHub reference is **marked deprecated in favour of IBKR Campus**. BUILD_PLAN §6 and the code both require verifying every tag and value range against the current IBKR Campus documentation before live use.

### Why we need it here
IBKR's algos are free to use on IBKR Pro, battle-tested, and run server-side with direct venue access. A part-time system should not build its own slicing engine. BUILD_PLAN §6 sets a high bar for any in-house tactic: it must win randomised A/B tests against the broker algo on arrival slippage.

### References
- [IBKR TWS API, IB Algos](https://interactivebrokers.github.io/tws-api/ibalgos.html)

## 4. The routing table

### What it is
`ExecutionRouter.route` assigns each order a tactic from a fixed decision table (BUILD_PLAN §6), writes a human-readable `meta["routing_reason"]`, and returns new `Order` objects. Rules are checked **in this order**, and the first match wins:

| # | Condition | Tactic | Venue hint | Parameters |
|---|---|---|---|---|
| 1 | Reversal-sleeve order (`meta["sleeve"]` ∈ {reversal, str, short_term_reversal} or `meta["reversal"]`) | **LMT**, passive: buy at indicative × (1 − half-spread) rounded down, sell at indicative × (1 + half-spread) rounded up | continuous | TIF DAY, never marketable |
| 2 | Signal half-life < `urgency_half_life_days_threshold` (5 days) | **ALGO ArrivalPx** at next open | open | `maxPctVol` = min(`algo_max_pct_adv`, 0.5) = 0.05, `riskAversion` Aggressive, `forceCompletion` False, `allowPastEndTime` True |
| 3a | qty ≤ `auction_max_pct_of_auction_volume` (2%) × expected auction volume, indicative price known | **LOC** through the indicative by max(`loc_offset_bps`, 1.5 × expected auction impact) | auction | submit before 15:50 ET |
| 3b | Same, but no indicative price | **MOC** | auction | |
| 4 | Otherwise; half-life unknown or ≥ 2 × threshold (10 days) | **ALGO ClosePx** | continuous | `maxPctVol` 0.05, `noTakeLiq` True |
| 4′ | Otherwise; half-life in [5, 10) days | **ALGO Vwap**, 13:00–16:00 ET | continuous | `maxPctVol` 0.05, `noTakeLiq` False |
| 5 | Rule 4/4′ order with qty > `algo_max_pct_adv` (5%) × ADV | as 4/4′, plus `meta["multi_day"] = True` and a WARNING in the reason | continuous | expect partial fill |

Expected auction volume is `snapshot["auction_volume_est"]` if supplied, else `adv_shares × auction_volume_share_default` (0.09). The router does **not** call the date-aware `auction_share_for_date`, so the caller must supply `auction_volume_est` on heavy days.

### Why we need it here
Different orders have different cost structures (size relative to liquidity), different urgency (how fast the alpha decays) and different economics (some trades *earn* the spread). One tactic for all would be wrong for most.

### How it works, and why each row
**Row 1: passive limits for reversal legs.** Short-term reversal is a **liquidity-provision** return. Stocks that fell sharply over days tend to bounce because someone was paid to absorb the selling (Dimensional; BUILD_PLAN §2). A reversal trade that crosses the spread pays away exactly the premium it is trying to earn. So the order rests one half-spread away from the indicative, on the passive side. The limit is rounded **away** from the indicative (buys floor, sells ceil) so tick rounding can never turn it marketable. Reversal is checked first because it is a property of *why* we trade, which overrides size and urgency. The cost is fill uncertainty. Unfilled reversal orders show up as opportunity cost in TCA.

**Row 2: urgency from signal half-life.** A signal's **half-life** is the time for its predictive power (IC) to halve. A signal with half-life $h$ days loses about $1 - 2^{-1/h}$ of its value per day of delay: 29% a day at $h$ = 2, 13% at $h$ = 5, 3.4% at $h$ = 20. One period of execution delay flips event-driven text-signal returns negative (Li, Qiao & Zheng). Waiting for tomorrow's close is too slow for such signals, so they go to Arrival Price at the next open with Aggressive risk aversion, which front-loads. The per-order `meta["half_life_days"]` overrides the batch-level `signal_half_life_days`. The continuous mapping `qts/execution/scheduler.py::urgency_from_half_life` (participation from 1% to 25%, 5% at a 5-day half-life) exists but the router does not use it yet. The router uses a single threshold.

**Row 3: LOC for orders small relative to the auction.** If an order is ≤ 2% of expected closing-auction volume, the auction can absorb it with modest impact and the fill is the benchmark close. LOC rather than MOC gives **price protection** against a dislocated print, such as a large late imbalance. Two details matter:
- **The limit sits *through* the indicative**: a buy above, a sell below. Being marketable means the order participates at any reasonable clearing price. A limit *at* the indicative would be unfilled whenever the auction prints one tick against you.
- **The offset floor at 1.5× expected impact.** The offset is

$$
\text{off} = \max\big(\text{loc\_offset\_bps},\; 1.5 \times 10^4\, b_{\text{auc}}\, \sigma \sqrt{q / V_{\text{auc}}}\big)
$$

where `loc_offset_bps` = 3 bp, $b_{\text{auc}}$ = `impact_b_auction` = 0.5, $\sigma$ the daily volatility, $q$ the order in shares and $V_{\text{auc}}$ the expected auction volume in shares. Without the floor a fixed 3 bp offset is inconsistent with the cost model. A 1%-of-auction order in a 2%-vol stock has expected impact of 10 bp, so its *own* expected impact would push the clearing price through a 3 bp limit, and the order would usually go unfilled. 1.5× leaves headroom for impact plus noise. REVIEW_NOTES item 9 documents the rule. `loc_offset_impact_mult=0` restores the fixed offset.
- **Tick rounding fix (audit #9).** The earlier code used `round(px, 2)`. A 3 bp offset on a \$7 stock is \$0.0021, which rounds to zero, putting the LOC exactly at the indicative. The fix rounds in the **favourable direction**: buys `ceil` to the tick, sells `floor`, using the Reg NMS tick (\$0.01 at or above \$1, \$0.0001 below), with a `round(n, 6)` guard against float noise. A \$7 buy becomes \$7.01, never \$7.00.

**Row 3b: MOC when there is no indicative.** Without a reference price there is nothing to set a limit against. Participating at the close beats not trading.

**Row 4: Close Price or VWAP for larger orders.** Above 2% of the auction, the auction alone would move the close too much. BUILD_PLAN §6 sends these to in-session algos over the last 1–3 hours with participation ≤ 5–10%, because the buy-side cost step is largest above 5% of ADV (Bloomberg BTCA). Low urgency (half-life ≥ 10 days or unknown) gets **ClosePx with `noTakeLiq`**: track the close, which is the backtest's benchmark, and avoid crossing the spread. Medium urgency (5–10 days) gets **VWAP from 13:00**, starting earlier and allowed to take liquidity.

**Row 5: the multi-day flag.** An order above 5% of ADV cannot complete at a 5% participation cap in one day. The router does not split it. It flags it, and the fill simulator models the partial fill. The optimizer's participation constraint (`max_adv_participation` = 5%) should keep this rare.

### Why this choice
- **A transparent rule table rather than a cost-minimising venue optimizer.** Every routing decision carries a reason string a human can audit. `SquareRootImpact.cost_for_venue_choice` exists, but its answer depends on the uncalibrated $b_{\text{auc}}$. Under the current priors it prefers the continuous session for almost every order above about 0.02–0.05% of ADV (→ see **Transaction costs › Venue**), which contradicts the plan's auction-first premise. Rules plus TCA let data decide later.
- **Not implemented from the plan table:** "Published imbalance on the same side: do not add; consider contra-side only." It needs imbalance data (Databento publishes a worked example for auction-imbalance messages) and is on the roadmap.

### In the code
`qts/execution/router.py::ExecutionRouter(cfg: ExecutionConfig, cost_cfg: CostConfig, loc_offset_impact_mult=1.5)`; `route(orders, snapshot, signal_half_life_days=None, date=None)`; helpers `loc_offset_bps`, `loc_limit`, `passive_limit`, `tick_size`, `round_to_tick`, `is_reversal`; `routing_table(routed)` gives a DataFrame for review. Snapshot keys: `adv_shares, auction_volume_est, sigma_daily, half_spread_bps, indicative_close_px`. The indicative falls back to `meta["decision_px"]` and the half-spread to 2 bp.

```python
def loc_offset_bps(self, qty: float, auction_volume: float, sigma_daily: float) -> float:
    """Offset in bps through the indicative: see class docstring."""
    off = float(self.cfg.loc_offset_bps)
    if self.loc_offset_impact_mult > 0 and np.isfinite(auction_volume) and auction_volume > 0 and np.isfinite(sigma_daily):
        exp_imp = 1e4 * self.cost_cfg.impact_b_auction * sigma_daily * np.sqrt(qty / auction_volume)
        off = max(off, self.loc_offset_impact_mult * exp_imp)
    return off
```

`ExecutionConfig` parameters:

| Field | Default | Meaning |
|---|---|---|
| `auction_max_pct_of_auction_volume` | 0.02 | LOC/MOC if qty ≤ 2% of expected auction volume |
| `algo_max_pct_adv` | 0.05 | `maxPctVol` for algos (capped at 0.5) and the multi-day flag threshold |
| `loc_offset_bps` | 3.0 | minimum LOC offset through the indicative |
| `auction_volume_share_default` | 0.09 | auction share used when `auction_volume_est` is missing |
| `urgency_half_life_days_threshold` | 5.0 | half-life below which ArrivalPx is used; 2× it separates ClosePx from VWAP |

Router constructor: `loc_offset_impact_mult` = 1.5 (not in `ExecutionConfig`).

### Pitfalls & checks
- **The indicative price timing problem.** BUILD_PLAN §6 says "LOC a few bp through the **3:55** indicative price, submitted before **3:50**". That is impossible as written: the order must be in before the official NYSE imbalance feed starts at 15:50. In practice `indicative_close_px` must be the last trade or mid at roughly 15:45–15:49, and the offset must absorb the move from there to the close. Treat the plan's wording as a known inconsistency until imbalance-aware logic exists.
- **`noTakeLiq` on ClosePx.** The router sends `noTakeLiq` to ClosePx, but the `qts/execution/ibkr.py` docstring lists only `maxPctVol` and `riskAversion` for ClosePx. Verify against IBKR Campus. IBKR may reject or ignore an unsupported tag.
- **ArrivalPx "at next open"** is a `venue_hint` only. The payload carries no `startTime`, so a live harness must submit it at or just after 09:30.
- **Half-penny ticks.** If the 2024 Rule 612 amendment applies to a name, `tick_size` returns \$0.01 where \$0.005 is allowed. That is harmless for LOC (slightly wider) and slightly conservative for passive limits.
- **Heavy days.** Without `auction_volume_est`, the router assumes 9% on Russell day too, so it under-uses the auction exactly when it is deepest.
- Tests: `test_router_tactics`, `test_loc_limit_rounds_to_tick_in_favourable_direction`.

### References
- [Dimensional, Q&A on short-run reversals](https://www.dimensional.com/us-en/insights/q-and-a-on-short-run-reversals-with-mamdouh-medhat-and-robert-novy-marx)
- [Li, Qiao & Zheng (2025), LLM event extraction and execution delay](https://arxiv.org/pdf/2512.19484)
- [Bloomberg (2022), BTCA](https://www.bloomberg.com/professional/?p=46500)
- [Databento, Auction imbalance example](https://databento.com/docs/examples/equities/auction-imbalance)

## 5. Scheduling math

### What it is
**Scheduling** decides how a parent order is spread over the trading day. The broker algos do this server-side. `qts/execution/scheduler.py` implements the same math for simulation, intuition and future in-house tactics.

### How it works: the intraday volume curve
US volume is **U-shaped**: heavy at the open (overnight news, price discovery), light at midday, heaviest at the close (index funds, the auction). The default 13 half-hour buckets from 09:30 to 16:00:

| Bucket | 09:30 | 10:00 | 10:30 | 11:00 | 11:30 | 12:00 | 12:30 | 13:00 | 13:30 | 14:00 | 14:30 | 15:00 | 15:30 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Share of day | 11.0% | 7.5% | 6.0% | 5.5% | 5.0% | 4.8% | 4.8% | 5.0% | 5.5% | 6.5% | 8.0% | 10.4% | 20.0% |

The last bucket includes the closing auction. `intraday_volume_curve(n_buckets, close_share)` interpolates the shape for other bucket counts and can rescale the last bucket, for example `close_share=0.30` on an index-rebalance day, with the rest scaled to sum to 1. The shape is a stylised default, not estimated from data. Forecasting it per name is one of the two jobs where BUILD_PLAN §6 says ML earns its keep.

### How it works: VWAP and POV schedules
**VWAP schedule.** To match the day's VWAP, trade in proportion to expected volume:

$$
n_j = Q\,\frac{u_j}{\sum_k u_k}
$$

where $Q$ is the parent quantity, $u_j$ the expected volume share of bucket $j$ and $n_j$ the child quantity. If you trade the same fraction of every bucket's volume, your average price is the market's VWAP. `vwap_schedule(qty, curve)`.

**POV schedule.** Trade a fixed fraction $p$ of each bucket's expected volume $E_j$ (shares) until done: $n_j = \min(p\,E_j,\; Q - \sum_{k<j} n_k)$. Completion time is uncertain. Anything left after the last bucket stays unfilled, and the function returns a sum < $Q$. `pov_schedule(qty, expected_volume_curve, pct)`.

### How it works: Almgren–Chriss optimal execution
**The trade-off.** Trading fast costs more impact. Trading slowly leaves you exposed to price moves while you still hold inventory (**timing risk**). Almgren and Chriss (2000) make this a mean-variance problem: choose the holdings trajectory $x(t)$, from $x(0) = X$ to $x(T) = 0$, minimising

$$
\mathbb{E}[\text{cost}] + \lambda\,\mathrm{Var}[\text{cost}] = \eta \int_0^T \dot{x}(t)^2\,dt \;+\; \lambda\,\sigma^2 \int_0^T x(t)^2\,dt
$$

where:
- $\dot{x} = dx/dt$ is minus the trading rate (shares per unit time);
- $\eta$ is the **linear temporary impact** coefficient: trading at rate $v$ moves the price by $\eta v$, so cost per unit time is $\eta v^2$. Permanent impact adds a constant that does not depend on the path and drops out;
- $\sigma$ is price volatility per $\sqrt{\text{time}}$ (price units), so holding $x$ shares for $dt$ adds variance $\sigma^2 x^2 dt$;
- $\lambda$ is risk aversion. $\lambda = 0$ cares only about expected cost.

<toggle heading="h3">
Derivation of the sinh trajectory

The integrand is $L(x, \dot{x}) = \eta\dot{x}^2 + \lambda\sigma^2 x^2$. The Euler–Lagrange equation $\frac{d}{dt}\frac{\partial L}{\partial \dot{x}} = \frac{\partial L}{\partial x}$ gives $2\eta\ddot{x} = 2\lambda\sigma^2 x$, i.e.

$$
\ddot{x} = \kappa^2 x, \qquad \kappa^2 = \frac{\lambda\sigma^2}{\eta}
$$

The general solution is $x(t) = A\sinh(\kappa t) + B\cosh(\kappa t)$. Imposing $x(T) = 0$ and $x(0) = X$:

$$
x(t) = X\,\frac{\sinh(\kappa(T - t))}{\sinh(\kappa T)}
$$

As $\kappa \to 0$ ($\lambda \to 0$), $\sinh(\kappa(T-t))/\sinh(\kappa T) \to (T-t)/T$: a straight line, which is TWAP. Larger $\kappa$ (more risk-averse, more volatile, or cheaper impact) front-loads the trading. $1/\kappa$ is the order's natural "half-life" in time units. In discrete time with step $\tau$ the same algebra on the difference equation gives $\cosh(\kappa\tau) = 1 + \tilde\kappa^2\tau^2/2$, where $\tilde\kappa^2 = \lambda\sigma^2/\eta$. That is what the code uses (`kappa = arccosh(kt2*tau**2/2 + 1)/tau`).
</toggle>

`almgren_chriss_schedule(qty, T_buckets, sigma, eta, lam, tau=1.0)` returns holdings $x_0..x_T$, trades $n_j = x_{j-1} - x_j$ and $\kappa$. Units must be consistent per bucket.

### Why this choice: the schedule-independence caveat
Almgren–Chriss assumes **linear** temporary impact. The empirical evidence says impact is **square root** in metaorder size, and approximately **independent of how the metaorder is split** across child orders and time at modest participation (Bouchaud; BUILD_PLAN §4). If total impact depends mainly on $Q/V$, then for a given order the choice between VWAP, TWAP or a front-loaded schedule changes expected cost only at the margin. Scheduling's real value is in three things:
1. **controlling participation**, staying under 5% of volume where the impact model is validated;
2. **managing timing risk** for urgent signals;
3. for an auction-centric book, **simply getting to the close**.

That is why the scheduler is reference code. The Almgren–Chriss trajectory gives intuition for urgency, and the broker's algos do the real slicing. An in-house scheduler would have to win A/B tests against them (BUILD_PLAN §6).

### Pitfalls & checks
- Do not feed Almgren–Chriss a daily σ with half-hour buckets without rescaling ($\sigma_{\text{bucket}} = \sigma_{\text{day}}/\sqrt{13}$ under i.i.d. returns).
- POV's returned sum can be below the order. The caller must handle the residual.
- Test: `tests/test_execution.py::test_scheduler` (curve sums to 1, VWAP and POV sums, AC endpoints, front-loading, $\lambda = 0$ gives TWAP).

### References
- *Almgren & Chriss (2000), Optimal execution of portfolio transactions, Journal of Risk.*
- [Bouchaud, The square-root law of market impact](https://bouchaud.substack.com/p/the-square-root-law-of-market-impact)

## 6. Reinforcement learning for execution

### What it is
**Reinforcement learning (RL)** trains an agent to choose actions (here: how much to trade each minute) to maximise a reward (here: minus slippage) by trial and error in an environment, which in practice is a market simulator.

### What the 2024–26 evidence shows
| Study | Setting | Result | Evidence class |
|---|---|---|---|
| De Witt & Pakkanen (2026) | PPO agents adjusting participation minute by minute | 2.13 bp arrival slippage vs 7.01 bp for TWAP across 4,900 out-of-sample orders | **simulation** with calibrated transient impact; authors state results are not predictions of live execution |
| Hafsi & Vittori (2024) | DQN in the ABIDES agent-based simulator | improvements reported | simulation; does **not** test against Almgren–Chriss or VWAP |
| RL-Exec (2025) | BTC liquidation | improvements reported | one out-of-sample month |

### Why it is not used here
1. **Every result is simulation-only.** None has been validated on live fills against a broker algo.
2. **Bounded upside.** Under the square-root law, impact is nearly schedule-independent at modest participation, so the room for clever scheduling is small. The gains an RL agent finds are likely to come from exploiting the simulator's structure, which is exactly the part least likely to transfer to real markets.
3. **Wrong problem for this book.** An auction-centric daily rebalancer does most of its volume in a single call auction, where there is nothing to schedule.
4. **Where ML does help** (BUILD_PLAN §6): choosing urgency per order from signal decay, and forecasting intraday volume curves and closing-auction volume. Both are on the roadmap. Neither needs RL.

### References
- [De Witt & Pakkanen (2026)](https://arxiv.org/html/2601.22113v1)
- [Hafsi & Vittori (2024)](https://export.arxiv.org/pdf/2411.06389)
- [RL-Exec (2025)](https://arxiv.org/html/2511.07434v1)

## 7. Broker payloads, dry run, fill simulation and TCA

### IBKR payloads
`qts/execution/ibkr.py::to_ib_order_dict(order, account=None, transmit=True, outside_rth=False)` builds a dict whose keys mirror `ib_async.Order` attribute names, so the future adapter is `Order(**payload)` plus `TagValue` wrapping:

| `Order.order_type` | `orderType` | Extra fields |
|---|---|---|
| MOC | `MOC` | none |
| LOC | `LOC` | `lmtPrice` (required) |
| LMT | `LMT` | `lmtPrice` (required) |
| MKT | `MKT` | none |
| ALGO | `LMT` if a limit is set, else `MKT` | `algoStrategy` (must be in `KNOWN_ALGOS`), `algoParams` = list of `{"tag", "value"}` in a fixed order; booleans as "1"/"0" |

Every payload carries `action`, `totalQuantity`, `tif`, `outsideRth`, `transmit`, `orderRef` = the internal order id (this links broker reports back to the TCA ledger), and a `contract` sub-dict `{symbol, secType: STK, exchange: SMART, currency: USD}`.

`auction_deadline(date, exchange)` returns the tz-aware cutoff: NYSE 15:50, NASDAQ 15:50, ARCA 15:59 ET. `is_before_auction_deadline(now, date, exchange)` answers whether an MOC/LOC can still be sent. Naive times are read as ET. DST is handled through `America/New_York`.

### The dry-run broker
`DryRunBroker` accepts orders (`place_order` builds the payload, so serialisation errors surface early, and returns an integer id) and fills them on request at `ref_px × (1 + s × impact)`. Impact comes from the square-root model with the auction volume hard-coded at 9% of ADV for MOC/LOC. LOC/LMT orders whose fill price would violate the limit get zero fill and status "Cancelled". It exists for **plumbing tests**, not cost realism.

### The fill simulator
`qts/execution/simulator_fills.py::FillSimulator(cfg_cost, auction_share=0.09, default_max_pct_vol=0.05)` fills routed orders against one day's realised bar per ticker (`open, close, next_open, volume, sigma_daily`, optionally `auction_volume`, `half_spread_bps`, `adv_shares`). Rules:

| Order type | Fill price | Fill quantity | Spread | Impact |
|---|---|---|---|---|
| MOC | close × (1 + s·impact) | all | 0 | $b_{\text{auc}}$, V = auction volume (realised if given, else share × day volume) |
| LOC | same | all **if the fill price does not violate the limit** (buy: fill ≤ limit; sell: fill ≥ limit), else 0 | 0 | same |
| ALGO Vwap / ClosePx | (open + close)/2 × (1 + s·(impact + half-spread)) | **min(qty, maxPctVol × day volume)**: partial fill | yes | $b_{\text{cont}}$, V = day volume, on the filled quantity |
| ALGO ArrivalPx / Adaptive | open × (1 + s·(impact + half-spread)) | same partial-fill rule | yes | same |
| LMT (passive) | the limit | all if the close crossed the limit (buy: close ≤ limit), else 0 | 0 | 0 |
| MKT | open × (1 + s·(impact + half-spread)) | all | yes | $b_{\text{cont}}$, V = day volume |

Here $s = +1$ for buys and $-1$ for sells. Fees come from the exact fee schedule on filled shares, dated with the simulation date, with the auction exchange fee for auction fills (→ see **Transaction costs › Explicit fees**).

**Why the LOC rule matters.** The LOC's *own* impact can push the clearing price through its limit, which is exactly what the 1.5×-impact offset floor (§4) prevents. The simulator makes the failure visible: an LOC with too small an offset shows up as unfilled, and its missed return appears as opportunity cost.

**Feeding the TCA ledger.** With `ledger=` supplied, each order is recorded with:
- decision price = `meta["decision_px"]`, else the open;
- arrival = the **close** for auction orders (approximating the 15:50 mid, the plan's auction benchmark) or the **open** for continuous orders, timestamped 15:50 or 09:30;
- venue derived from the order type;
- fills at 16:00;
- the order closed with close, next open, ADV (63-day `adv_shares` if given, else the day's volume, audit #16), σ, fees, half-spread, and the **realised venue volume** the fill was priced on, so `calibrate_b_by_venue` regresses each venue on the right $V$.

A repeated order id on a later day gets a date prefix instead of raising.

**Simplifications to keep in mind:**
- The ALGO VWAP proxy (open + close)/2 is crude. A ClosePx order is really benchmarked to the close, but the simulator fills it at the mid-of-day proxy.
- Passive LMT fills are binary on the close. A probabilistic fill model (NautilusTrader-style) is a later step.
- **No look-ahead safeguard is needed** because the simulator is used *after* the day. It is a fill model, not a predictor.
- The daily **backtester does not use the router or the fill simulator.** It charges every trade through the cost model at one venue (default auction) (→ see **Backtesting**). The fill simulator is for execution research and for paper-trading comparison.

### Path to live trading
1. **Now:** payload builders, router, fill simulator, dry-run broker and TCA ledger. No network code.
2. **Phase 0 → 4 plumbing:** an `ib_async` adapter (`IB().placeOrder(contract, order)`), connecting first to an **IBKR paper account**, plus a daily SLB borrow logger from day one.
3. **Phase 4, paper trading (4–6 months, BUILD_PLAN §9):** an event-driven layer (NautilusTrader or LEAN), LOC/MOC default with imbalance-aware limits, full TCA capture, monthly $b$ recalibration, daily live-vs-model fill comparison. Acceptance: live-vs-backtest tracking error inside a pre-registered band; realised arrival slippage within **1.5×** modelled cost for ≥ 80% of orders; $b$ stable across three monthly recalibrations; no point-in-time or parity defects in a code audit. Paper trading validates *plumbing, costs and parity*, not alpha: four years of out-of-sample data cannot certify a moderate Sharpe.
4. **Phase 5, live:** smallest viable book on IBKR Pro. Capital grows only while 3-month realised Sharpe and slippage stay inside bands.

**Kill switches and circuit breakers** must exist before the first live order. They are in the plan, not yet in code. Minimum set: stop submitting if the 15:50 deadline has passed (`is_before_auction_deadline`); cap per-order notional and %ADV; cap daily gross traded; halt on a fill-vs-model slippage breach or a P&L drawdown breach; halt on stale data (snapshot date ≠ today); manual flatten and cancel-all. Any breach returns the system to Phase 4.

### References
- [NautilusTrader, Fill models](https://nautilustrader.io/docs/nightly/concepts/backtesting/fill-models/)
- [IBKR TWS API, IB Algos](https://interactivebrokers.github.io/tws-api/ibalgos.html)

## 8. Pitfalls and a worked example

### Pitfalls & checks (summary)
<callout icon="⚠️" color="yellow_bg">
- **Deadlines are hard.** An MOC/LOC sent at 15:50:01 to NYSE is rejected unless it offsets a published imbalance. Build the batch by 15:45.
- **Verify algo tags on IBKR Campus** before live use. The TWS reference the code follows is deprecated. `noTakeLiq` on ClosePx is the most likely mismatch.
- **The plan's "3:55 indicative" cannot be used for an order due by 3:50** (§4).
- **Router vs cost model.** The router's auction-first rule and the cost model's venue comparison disagree under the current $b_{\text{auc}}$ prior. TCA will settle it. Do not hand-tune the threshold without fills.
- **Auction share is not date-aware in the router** unless the caller supplies `auction_volume_est`.
- **Unit traps:** `qty` in shares, `half_spread_bps` in bp, `sigma_daily` as a daily fraction, auction volume in **shares** for the router but **dollars** inside the cost model.
- **IBKR Pro, not Lite.** Lite charges for on-close orders beyond 10% of monthly volume.
</callout>

### Worked example: routing three orders by hand
Snapshot for each of three stocks: ADV = 1,000,000 shares, expected auction volume = 90,000 shares (9%), σ = 2%/day, half-spread = 3 bp, indicative close = \$100.00. Defaults: 2% auction threshold, 5% algo cap, 3 bp LOC floor, 5-day urgency threshold. Fill-simulator day: open = close = next open = \$100.00, volume = 1,000,000, trade date 2026-11-04.

**Order A: BUY 1,000 shares, half-life 20 days.**
- Not reversal. Half-life 20 ≥ 5, so not urgent.
- % of auction = 1,000 / 90,000 = 1.11% ≤ 2%, so **LOC**.
- Expected auction impact = $10^4 \times 0.5 \times 0.02 \times \sqrt{1{,}000/90{,}000}$ = 100 × 0.1054 = 10.54 bp. Floor = 1.5 × 10.54 = 15.81 bp > 3 bp, so the offset is 15.81 bp.
- Limit = 100 × 1.001581 = 100.158, ceil to tick, **\$100.16**.
- Payload: `{"action": "BUY", "totalQuantity": 1000, "orderType": "LOC", "lmtPrice": 100.16, ...}`.
- Simulator: auction impact 10.54 bp, fill = \$100.105 ≤ \$100.16, so filled. Fees: \$3.50 commission + \$0.20 clearing + \$0.003 CAT + \$1.00 auction fee = **\$4.703**.
- With the old fixed 3 bp offset the limit would be \$100.03, and the order's own 10.5 bp impact would have left it **unfilled**.

**Order B: SELL 10,000 shares, half-life 20 days.**
- % of auction = 11.1% > 2%, so not auction. % of ADV = 1.0% ≤ 5%, no flag.
- Half-life 20 ≥ 10 means low urgency, so **ALGO ClosePx**, `maxPctVol` 0.05, `noTakeLiq` True; payload `orderType` MKT with `algoParams` `[maxPctVol=0.05, noTakeLiq=1]`.
- Simulator: cap = 0.05 × 1,000,000 = 50,000 ≥ 10,000, so full fill. Impact = $10^4 \times 1.0 \times 0.02 \times \sqrt{0.01}$ = 20 bp, plus 3 bp spread = 23 bp, fill = \$99.77. Fees on \$997,700: commission \$35.00, clearing \$2.00, SEC 997,700 × 20.60/10⁶ = \$20.55, TAF **\$0** (2026 Q4 holiday), CAT \$0.03, total **\$57.58**.

**Order C: BUY 100,000 shares, half-life 6 days.**
- % of auction = 111% > 2%, so not auction. Half-life 6 is ≥ 5 (not urgent) but < 10, so **ALGO Vwap** 13:00–16:00 ET, `maxPctVol` 0.05, `noTakeLiq` False.
- % of ADV = 10% > 5%, so `multi_day = True`; the reason reads "WARNING order is 10.0% of ADV > 5% cap: expect partial fill / multi-day working".
- Simulator: cap = 50,000 shares, so a **partial fill of 50,000**. Impact on the filled part = $10^4 \times 0.02 \times \sqrt{0.05}$ = 44.7 bp, plus 3 bp spread, fill = \$100.477. Fees = \$185.15 (commission \$175, clearing \$10, CAT \$0.15). In TCA the unfilled 50,000 shares become opportunity cost at the close. The next day the optimizer sees the half-built position, and if it continues buying the cost model's persistence multiplier ($m_2 = 1.83$) prices the second day as the continuation of one metaorder (→ see **Transaction costs › Multi-day impact persistence**).

If Order C had a half-life of 3 days instead, rule 2 would fire first: **ArrivalPx at the next open**, Aggressive, `maxPctVol` 0.05, still partially filled at 50,000 by the simulator. If it were tagged `sleeve="reversal"`, rule 1 would make it a passive **LMT at \$99.97** (100 × (1 − 0.0003), floored), filled only if the close is ≤ \$99.97.

These numbers come from running `ExecutionRouter().route(...)` and `FillSimulator().simulate(..., date="2026-11-04")` on the snapshot above. `tests/test_execution.py::test_router_tactics` and `test_fill_simulator_roundtrip_with_ledger` cover the same paths.

## Roadmap
- `ib_async` adapter and paper-trading harness (Phase 4), with kill switches and a cancel-all path.
- Imbalance-aware LOC logic from auction-imbalance feeds: don't add to a same-side imbalance; use the 15:50+ indicative for contra-side orders.
- Use `urgency_from_half_life` for continuous participation instead of the single 5-day threshold.
- Date-aware auction volume in the router via `auction_share_for_date` or a per-name auction-volume forecast. One source of truth for the 9% default (now in five places).
- Probabilistic passive-limit fills and an intraday event-driven simulator (NautilusTrader) if intraday tactics are added.
- Randomised A/B tests of tactics on arrival slippage once live fills exist. Revisit the 2% auction threshold after $b_{\text{auc}}$ is calibrated.
- Verify algo tags (especially ClosePx `noTakeLiq`) and tick sizes (Rule 612 half-penny) against current IBKR Campus and SEC rules.
