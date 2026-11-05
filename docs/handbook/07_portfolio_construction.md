# Portfolio construction

<callout icon="🧭" color="gray_bg">
Portfolio construction turns three inputs into one decision. The inputs are the alpha model's expected returns, the risk model's covariance, and the cost model's trading and borrow costs. The decision is the set of target weights to hold after today's closing auction. The system solves one convex optimisation problem per day (Boyd et al.'s single-period optimiser, built in cvxpy and solved with Clarabel). It maximises horizon-matched expected return minus penalties for risk, trading cost and holding cost, subject to a list of hard constraints: leverage, net exposure, position and liquidity caps, beta, industry and style neutrality, no-trade lists and forced exits. Volatility targeting is then a second, projection solve over the same constraints. This page derives the maths from mean–variance first principles, explains every term and constraint and why it exists, shows where each lives in `qts/portfolio/`, and lists the diagnostics that tell you how much of the forecast actually reached the portfolio.
</callout>

## The problem: from forecasts to positions

### What it is
The alpha model gives every stock a number: "I expect this name to beat its risk-model-predicted return by 0.3% over the next 20 trading days." The risk model says how those names move together. The cost model says what it costs to change a position and to hold a short. Portfolio construction answers the only question that makes money: **how many dollars of each stock should we hold after today's close?**

Think of a chef with a recipe score for every ingredient (alpha), a nutritionist's warning about which ingredients overlap (risk), and grocery prices that rise the more of any one item you buy in a day (costs). The best meal is not "the ten highest-scoring ingredients". It is the best balance of all three, given what is already in the fridge (the current portfolio).

### Why we need it here
The obvious shortcut is "buy the top decile, short the bottom decile, equal-weight". Academic factor papers do exactly this, which is why it feels natural. In a real book it fails in three ways:

1. **Risk concentration.** Scores cluster. On a given day the top value decile may be 40% banks, and the top momentum decile may be the same handful of AI-related semis. Equal weights inside a decile ignore that these names move together, so the book is really a few large bets dressed up as 100 small ones.
2. **Costs.** Deciles are recomputed daily, and names near the decile boundary flip in and out. A decile book can turn over its whole gross several times a month. At 10–15 bp per side that alone can consume a 3–5%/yr gross edge (→ see **Transaction cost model › Why costs decide whether an alpha is real**). A decile rule has no way to say "this name moved from rank 95 to rank 89; keep it, because selling and rebuying costs more than the forecast difference".
3. **Unintended factor bets.** Sorting on one characteristic loads on others. The top momentum decile has high market beta after rallies and low beta after crashes; the low-volatility decile is a short-beta bet; a value decile is often short the "growth" industries wholesale. These are exposures nobody chose and the alpha model does not forecast. They add risk with no expected return.

An optimiser fixes all three at once. Risk enters through the covariance, costs enter as a penalty on the trade, and unwanted exposures are removed with explicit constraints. Its weakness is that it believes its inputs, which is the subject of the next section.

## Mean–variance from first principles

### What it is
Mean–variance optimisation (Markowitz, 1952) chooses weights to balance expected return against variance. With an $N$-vector of expected (excess) returns $\alpha$, an $N\times N$ covariance $\Sigma$ and weights $w$ (fractions of NAV, the fund's net asset value), the investor maximises a quadratic utility:

$$
U(w) = \alpha^\top w - \gamma\, w^\top \Sigma w
$$

where:
- $\alpha^\top w$ is the portfolio's expected return over the period;
- $w^\top \Sigma w$ is its variance over the same period;
- $\gamma > 0$ is the **risk aversion**, the rate at which the investor trades a unit of variance for a unit of expected return. Some texts write $\tfrac{\gamma}{2}$. The code uses $\gamma$ without the half, and so does this page.

### How it works
With no constraints, set the gradient to zero:

$$
\nabla U = \alpha - 2\gamma \Sigma w = 0 \quad\Longrightarrow\quad w^\star = \frac{1}{2\gamma}\,\Sigma^{-1}\alpha
$$

Three facts follow, and all of them are useful later.

- **Direction and size separate.** The direction of $w^\star$ is $\Sigma^{-1}\alpha$. The risk aversion $\gamma$ only scales it. So "how aggressive" and "which names" are separate decisions, which is why the system can target volatility in a separate step.
- **Optimal risk is proportional to the Sharpe ratio.** Substituting back, the optimal utility is $\tfrac{1}{4\gamma}\alpha^\top\Sigma^{-1}\alpha$ and the optimal volatility is $\sqrt{\alpha^\top\Sigma^{-1}\alpha}/(2\gamma)$. The quantity $\sqrt{\alpha^\top\Sigma^{-1}\alpha}$ is the ex-ante Sharpe ratio (expected return per unit of risk) of the best portfolio. A model that believes it has a big edge will take big risk.
- **$\Sigma^{-1}$ rewards differences.** For two highly correlated stocks, $\Sigma^{-1}$ has large positive diagonal and large negative off-diagonal entries. A small difference in their alphas becomes a large long-one/short-the-other position.

### Why it is unusable raw
That third fact is the problem. Michaud (1989) called mean–variance an "error maximiser": $\alpha$ is estimated with large noise, and the optimiser puts the most weight where the estimated alpha differences are largest *relative to* the risk model's view of how similar the names are. Those are exactly the places where noise dominates. The worked example at the end of this page shows a one-percentage-point change in one asset's expected return doubling that asset's weight.

The empirical verdict is harsh. DeMiguel, Garlappi and Uppal (2009) tested 14 sample-based mean–variance variants against the naive $1/N$ portfolio across seven datasets. None consistently beat $1/N$ out of sample, and they estimated that about 3,000 months of data would be needed for a 25-asset sample-mean optimiser to win reliably (BUILD_PLAN §5). Their study uses *historical sample means* as $\alpha$, which is the worst possible forecast. It does not show that optimisation is useless. It shows that **the forecast must be shrunk hard toward zero and the optimiser must be constrained and regularised**.

<callout icon="🤔" color="purple_bg">
Why not give up and use 1/N, risk parity or HRP? Those methods have no input for "this name is expected to outperform", so they cannot use an alpha model at all. The system keeps the optimiser and defuses error maximisation in four ways: (1) alphas are Grinold-scaled with a shrunk IC, so their dispersion is realistic (§Alpha scaling); (2) the covariance is a shrunk factor model, so $\Sigma^{-1}$ is well conditioned (→ see **Risk model › Shrinkage estimators**); (3) hard per-name and exposure caps bound the damage any single error can do; (4) the trade-cost penalty makes the optimiser move only partway toward its noisy target each day.
</callout>

### References
- [DeMiguel, Garlappi & Uppal (2009), Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?](https://ideas.repec.org/a/oup/rfinst/v22y2009i5p1915-1953.html)
- [Boyd et al. (2017), Multi-Period Trading via Convex Optimization](https://arxiv.org/pdf/1705.00109)

## Convex optimisation primer for engineers

### What it is
A **convex optimisation problem** minimises a convex function over a convex set (equivalently, maximises a concave function).

- A set $C$ is **convex** if the straight line between any two of its points stays inside it: $x, y \in C \Rightarrow \theta x + (1-\theta) y \in C$ for $\theta\in[0,1]$. Boxes ($\lvert w_i\rvert \le 0.02$), half-spaces ($\mathbf{1}^\top w \le 0.05$), balls and cones are convex. "At most 50 names held" is not: the midpoint of two 50-name portfolios can hold 100 names.
- A function $f$ is **convex** if its graph lies below every chord: $f(\theta x + (1-\theta) y) \le \theta f(x) + (1-\theta) f(y)$. Examples: $\lvert x\rvert$, $x^2$, $\lVert x\rVert_2$, $w^\top\Sigma w$ for positive semidefinite (PSD) $\Sigma$, $\lvert x\rvert^{3/2}$.

### Why it matters here
**In a convex problem every local optimum is a global optimum.** If you are at a point where no small move improves the objective, no big move does either. Two engineering consequences follow:

1. **Reliability.** Interior-point solvers converge to the global optimum in a predictable number of iterations (typically 10–50), each a sparse linear solve. No random restarts, no learning rates. A daily production job needs this.
2. **Certificates.** The solver returns *dual variables* (Lagrange multipliers), one per constraint. A non-zero dual means the constraint is binding, and its value is the marginal objective gain from relaxing it by one unit. The optimiser uses this to report which constraints bind (`OptimizationResult.binding_constraints`).

### The problem classes
| Class | Objective | Constraints | Example in this system |
|---|---|---|---|
| LP (linear program) | linear | linear | alpha-only objective with box and leverage limits |
| QP (quadratic program) | convex quadratic | linear | mean–variance with linear constraints |
| SOCP (second-order cone program) | linear or SOC-representable | second-order cones | the 3/2-power impact term; the volatility cap |
| Power / exponential cone | — | power cones | how solvers represent $\lvert x\rvert^{p}$ exactly |

A **second-order cone** (SOC, also called the "ice-cream cone") is the set $\{(z, t): \lVert z\rVert_2 \le t\}$. Many useful constraints can be written as "some vector's Euclidean norm is at most some scalar". The volatility cap $\sqrt{w^\top\Sigma w} \le \sigma_{\max}$ is one: it says $\lVert L^\top w\rVert_2 \le \sigma_{\max}$ for any $L$ with $LL^\top = \Sigma$.

### Disciplined convex programming (DCP) and cvxpy
You should not have to prove convexity by hand for each new constraint. **cvxpy** (Diamond & Boyd, 2016) uses *disciplined convex programming*. You build expressions from atoms with known curvature (convex, concave, affine) and monotonicity, and cvxpy checks a small set of composition rules. For example, "a convex non-decreasing function of a convex expression is convex". If every expression passes, cvxpy rewrites the problem into a cone program and hands it to a solver. If one fails, you get a `DCPError` at build time instead of a wrong answer at run time. In practice, DCP is why the cost model can be swapped without touching the optimiser: any cost written in DCP-valid atoms just works.

### Solver: Clarabel
**Clarabel** is an open-source interior-point solver for cone programs: linear, SOC, exponential, power and PSD cones. It has been cvxpy's default conic solver since CVXPY 1.5. Its paper reports a normalised solve time of about 1.0 versus 15.5 for ECOS and 32.7 for MOSEK on the Maros–Mészáros QP test set (BUILD_PLAN §5; benchmark, not this system's timing). Because it handles QPs and cones natively, the whole problem (quadratic risk, 3/2-power impact, SOC volatility cap) goes to one solver.

### Why the 3/2-power cost term is convex
Market impact follows the empirical square-root law: the price moves by roughly $b\,\sigma\sqrt{Q/V}$ when you trade $Q$ against daily volume $V$ (→ see **Transaction cost model › Market impact: the square-root law**). The *dollar cost* is that move times the traded amount, so it scales as $Q\cdot\sqrt{Q} = Q^{3/2}$. In weight units the impact cost of trade $x_i$ is $k_i\lvert x_i\rvert^{3/2}$ for a positive coefficient $k_i$.

$g(u)=u^{3/2}$ on $u\ge 0$ has $g''(u) = \tfrac{3}{4}u^{-1/2} \ge 0$, so it is convex. It is also non-decreasing. $\lvert x\rvert$ is convex and non-negative. By the DCP composition rule (convex non-decreasing of convex), $\lvert x\rvert^{3/2}$ is convex, and a sum with positive coefficients $k_i$ stays convex. In cvxpy this is `cp.power(cp.abs(x), 1.5)`, which cvxpy represents exactly with SOC/power cones. Linear costs (half-spread, fees, borrow on shorts via `cp.neg(w)`) are convex too. So the full objective is concave (we maximise alpha minus convex penalties) and the problem stays convex.

<callout icon="⚠️" color="yellow_bg">
Things that would break convexity and are therefore *not* in the optimiser: fixed per-trade costs (the $0.35 minimum commission is linearised; see the cost page), "hold at most K names" cardinality limits, minimum position sizes ("either 0 or at least 0.5%"), and lot-size rounding. All of these need mixed-integer solvers that are slower and less reliable. The system handles them outside the optimiser: tiny trades are dropped below a numerical tolerance, and the minimum commission is charged exactly in the simulator.
</callout>

### References
- [Diamond & Boyd (2016), CVXPY: A Python-Embedded Modeling Language for Convex Optimization](https://arxiv.org/abs/1603.00943)
- [Goulart & Chen (2024), Clarabel: An interior-point solver for conic programs with quadratic objectives](https://arxiv.org/pdf/2405.12762)

## Objective: Boyd single-period optimisation

### What it is
The optimiser is the single-period optimisation (SPO) policy from Boyd et al. (2017), *Multi-Period Trading via Convex Optimization*. Each day it chooses the post-trade weights $w$ that maximise:

$$
\alpha^\top w \;-\; \gamma_{\text{risk}}\, H \Big( \lVert L^\top B^\top w\rVert_2^2 + \sum_i d_i w_i^2 + \nu\,(y^\top w)^2 \Big) \;-\; \gamma_{\text{trade}}\,\mathrm{TC}(x) \;-\; \gamma_{\text{hold}}\, H\, \mathrm{HC}(w) \;-\; \kappa \lVert x\rVert_1
$$

where:
- $w$ = post-trade weights (fraction of NAV, long positive, short negative); $w_0$ = current (pre-trade, drifted) weights; $x = w - w_0$ = today's trade;
- $\alpha$ = expected residual return **over the alpha horizon** $H$ days (default $H = 20$, `AlphaConfig.horizon_days`), in fractions (0.003 = 30 bp over 20 days);
- $B$ = $N\times K$ factor exposure matrix; $F$ = $K\times K$ factor covariance (**daily** variance units); $L$ = a matrix with $LL^\top = F$ after PSD clipping (below);
- $d_i$ = specific (idiosyncratic) **daily** variance of stock $i$;
- $y$, $\nu$ = the Saxena–Stubbs alignment direction (unit vector) and its variance (below);
- $\mathrm{TC}(x)$ = one-off trading cost of $x$ as a fraction of NAV: linear half-spread and fees plus the 3/2-power impact term;
- $\mathrm{HC}(w)$ = **one day** of holding cost (borrow fees on shorts), fraction of NAV;
- $\gamma_{\text{risk}}$ = `risk_aversion` (10.0), $\gamma_{\text{trade}}$ = `trade_aversion` (1.0), $\gamma_{\text{hold}}$ = `hold_aversion` (1.0), $\kappa$ = `turnover_penalty` (0.0, off).

### Horizon units: why the $H$ factors are there
Every term must be in the same units, or the trade-offs are wrong by an order of magnitude. The common unit is "fraction of NAV over the next $H$ days".

| Term | Native unit | Conversion | Why |
|---|---|---|---|
| $\alpha^\top w$ | return over $H$ days | none | Grinold alpha uses residual vol over $H$ days: `resid_vol_h_ = idio_vol_252 * sqrt(H)` |
| risk | daily variance | $\times H$ | under i.i.d. daily returns, $H$-day variance $= H\times$ daily variance |
| $\mathrm{TC}(x)$ | paid once, today | none | you pay to trade once, then hold for (roughly) the horizon |
| $\mathrm{HC}(w)$ | per day | $\times H$ | borrow accrues every day the short is held |

Multiply the whole objective by $252/H$ and the risk term becomes $\gamma_{\text{risk}}\,\sigma^2_{\text{annual}}$ while alpha becomes annual alpha. So **`risk_aversion = 10` is an annual risk aversion**, which is easier to reason about: a book at 10% annual vol pays a $10 \times 0.01 = 0.10$ "utility tax". From the closed form above, an unconstrained optimiser with risk aversion $\gamma$ chooses vol $=\text{SR}_{\text{ex-ante}}/(2\gamma)$. At $\gamma=10$, a forecast ex-ante Sharpe of 2 gives 10% vol. The audit verified these units numerically (`docs/audit/risk_portfolio.md`, "Optimizer objective units").

<callout icon="💡" color="blue_bg">
Intuition for the cost term. Trading cost is charged once but alpha is earned over $H$ days, so a trade is worth making only if the extra $H$-day alpha it buys exceeds the cost it pays today. A longer horizon makes alpha "bigger" relative to a fixed cost and the optimiser trades more readily. A shorter horizon makes it trade less. If you change `horizon_days` without re-scaling alpha, you silently change the cost trade-off, which is why the alpha model, not the optimiser, owns $H$.
</callout>

### The factor-form risk term
The risk model says total covariance is $\Sigma = BFB^\top + D$, with $D = \mathrm{diag}(d)$ (→ see **Risk model › The risk model at a glance**). The naive way to write portfolio variance is `cp.quad_form(w, Sigma)`, which needs the dense $N\times N$ matrix. The optimiser instead writes:

$$
w^\top\Sigma w = (B^\top w)^\top F (B^\top w) + \sum_i d_i w_i^2 = \lVert L^\top f\rVert_2^2 + \lVert \sqrt{d}\circ w\rVert_2^2, \qquad f = B^\top w
$$

where:
- $f = B^\top w$ is the $K$-vector of portfolio factor exposures;
- $\sqrt{d}\circ w$ is the elementwise product of specific vols and weights.

**Complexity.** For $N = 1{,}000$ stocks and $K\approx 65$ factors (1 market, up to 48 industries, 10 styles, up to 5 statistical), the dense covariance has $10^6$ entries and a dense interior-point step costs $O(N^3)$. The factor form has $NK + K^2 \approx 70{,}000$ numbers, and introducing $f$ as an intermediate gives the solver a sparse system whose cost grows like $O(NK^2)$. Boyd et al. report under half a second per solve for 1,500 assets and 50 factors on one core (BUILD_PLAN §5). The audit confirmed the factor form equals $w^\top(BFB^\top + D)w$ to $3\times10^{-21}$.

**PSD clipping of $F$.** The factor covariance goes through shrinkage, eigenfactor adjustment and volatility-regime scaling. Floating-point error can leave it with tiny negative eigenvalues, and a "variance" that can be negative makes the problem non-convex. `_psd_factor` symmetrises $F$, eigen-decomposes $F = V\Lambda V^\top$, clips $\Lambda$ at zero and returns $L = V\Lambda^{1/2}$. Writing risk as a sum of squares of $L^\top f$ is DCP-valid by construction, so cvxpy never has to check PSD-ness itself.

```python
def _psd_factor(F: np.ndarray, floor: float = 0.0) -> np.ndarray:
    """Return L with L L' = PSD-clipped symmetric part of F."""
    Fs = 0.5 * (F + F.T)
    vals, vecs = np.linalg.eigh(Fs)
    vals = np.clip(vals, floor, None)
    return vecs * np.sqrt(vals)[None, :]
```

### The alignment term
A factor risk model can only see risk in directions spanned by its factors and in specific risk. An optimiser chasing alpha will find directions the model *underestimates*, such as a hidden "AI capex" factor that the alpha loads on but $B$ does not contain, and lever into them. Saxena and Stubbs observed an optimiser on a misaligned model realise 4.11% active risk against a 3% target (BUILD_PLAN §1).

The fix is the alpha alignment factor. Take the part of $\alpha$ orthogonal to the factor exposures, normalise it to a unit vector $y$ (so $B^\top y = 0$), and add $\nu(y^\top w)^2$ to the risk. Positions along "alpha the factors can't explain" now cost extra variance, and the optimiser backs off them. The risk model builds $y$ from the same alpha the optimiser receives (`risk_fn(t, alpha)` in the simulator). The magnitude $\nu$ is a heuristic ($\nu = $ `nu_scale` $\times N \times$ median style variance). A Saxena–Stubbs bias-statistic calibration exists (`calibrate_alignment_nu`) but is off the default path (audit finding 5, still open in `docs/audit/RESOLUTION.md`). Every solve logs `alignment share of ex-ante variance` in `OptimizationResult.notes`. If that share is near zero the term is inert; if it is large, check $\nu$ before trusting the risk number.

### Why this choice
- **SPO, not multi-period optimisation (MPO).** Boyd's MPO plans $H$ trades ahead and executes the first, at a cost roughly linear in $H$. The cvxportfolio documentation offers no guidance on when MPO beats SPO, and no credible out-of-sample single-stock test of the multi-period "aim" rule was found (BUILD_PLAN §5). SPO plus a trade-cost penalty plus half-life-aware alpha produces nearly the same trajectory. MPO is reserved for known term structure (earnings dates, index rebalances).
- **Penalties, not turnover caps.** A hard cap ("turnover ≤ 10%/day") binds on the days you most need to trade, such as after a large signal change. A cost penalty trades exactly where marginal alpha exceeds marginal cost. The linear `turnover_penalty` is kept as an extra knob but defaults to 0.
- **Factor risk, not sample covariance.** For $N = 1{,}000$ and a few years of daily data the sample covariance is singular or nearly so; see the error-maximisation discussion above.

### In the code
- `qts/portfolio/optimizer.py::SinglePeriodOptimizer.solve`: builds the objective in about five lines:

```python
w = cp.Variable(n, name="w")
x = w - w0
trade_cost, hold_cost = self.cost_model.cvx_terms(x, w, nav, snap, venue)
obj = a @ w - cfg.risk_aversion * H * pdat.risk_expr(w) \
    - cfg.trade_aversion * trade_cost - cfg.hold_aversion * H * hold_cost
if cfg.turnover_penalty > 0:
    obj = obj - cfg.turnover_penalty * cp.norm1(x)
```

- `_ProblemData.risk_expr` (factor form), `_ProblemData.risk_norm` (SOC form for the vol cap), `_ProblemData.variance` (numpy check value).
- The cost model plugs in through a structural interface that returns convex expressions in fraction of NAV. That interface is being refactored, so see the **Transaction cost model** page for the current contract. The optimiser relies only on "convex trade cost, convex one-day holding cost".
- $H$ comes from `AlphaOutput.horizon_days` per call (falls back to the constructor's `horizon_days`, set by the pipeline to `AlphaConfig.horizon_days = 20`).

### Pitfalls & checks
- **Unit mismatch** (daily risk with $H$-day alpha without the $\times H$) makes the book 20× too risky or too timid. Guard: the audit's numerical check; `test_resolve_at_optimum_is_fixed_point` (re-solving at the optimum produces no trade).
- **Out-of-model names.** A name with an alpha but no risk-model row gets zero factor exposure, so it looks purely idiosyncratic and attracts weight. The optimiser gives such names the **90th-percentile** specific variance (not the median), counts them in `n_out_of_model`, and the simulator warns with a total of name-days. The post-audit demo shows 230 such name-days.
- **Alpha scale.** If alphas are raw z-scores (dispersion ≈ 1) instead of Grinold-scaled (dispersion ≈ 0.001–0.01), the risk and cost terms are negligible and every constraint binds. Check `expected_alpha` in the solve log: annualised, it should be a few percent to low tens of percent, not hundreds.

### References
- [Boyd et al. (2017), Multi-Period Trading via Convex Optimization](https://arxiv.org/pdf/1705.00109)
- [Saxena & Stubbs, Alpha Alignment Factor: A Solution to the Underestimation of Risk for Optimized Active Portfolios](https://www.math.ttu.edu/conferences/past/mcmhr/docs/saxena.pdf)
- [cvxportfolio documentation, Optimization policies](https://www.cvxportfolio.com/en/stable/optimization_policies.html)

## Constraints and why each exists

Hard constraints encode things the objective cannot express or that we refuse to leave to estimated parameters: legal and broker limits, liquidity, and exposures we have decided not to bet on. `_build_constraints` assembles them once, and the *same* list is reused by the alpha solve and the volatility projection.

### Book types
| `book` | Constraints | When to use |
|---|---|---|
| `"long_short"` (default) | $\lVert w\rVert_1 \le$ `gross_leverage_max`; net $\mathbf{1}^\top w \in$ `net_exposure_target` ± `net_exposure_band` | the plan's default: dollar-neutral market-neutral book |
| `"130_30"` | long leg $\sum \max(w,0) \le 1.3$; short leg $\sum \max(-w,0) \le 0.3$; net $\in 1 \pm$ `net_exposure_band` | if borrow is constrained; keeps market beta near 1 |
| `"long_only"` | $w \ge 0$, $\mathbf{1}^\top w = 1$ | fallback only |

All books also get the per-name cap $\lvert w_i\rvert \le$ `max_position_weight`.

**Why long/short.** The transfer coefficient (see Diagnostics) of a constrained long-only book is around 0.3, versus 0.7–0.8 for long/short (Clarke, de Silva & Thorley via BUILD_PLAN §5). A long-only manager cannot express a negative view on a name beyond not owning it, and most names in a cap-weighted universe have tiny weights, so most negative forecasts are wasted.

### Gross leverage: the Reg T reality
Gross leverage $\lVert w\rVert_1 = \sum_i\lvert w_i\rvert$ is the total long plus short notional as a multiple of NAV. Under the Federal Reserve's Regulation T, a US retail margin account must post 50% initial margin on both long purchases and short sales. That caps overnight gross exposure at about **2× NAV**. Portfolio-margin accounts can go higher, but a part-time book at Interactive Brokers should assume Reg T. Hence `gross_leverage_max = 2.0`.

<callout icon="⚠️" color="yellow_bg">
The config comment says "150/50 → 2.0 gross". That is misleading. A 150/50 book (150% long, 50% short) has gross 2.0 but **net +1.0**. With the default `net_exposure_target = 0.0` and `net_exposure_band = 0.05`, the constraints produce a dollar-neutral **≈100/100** book: the post-audit demo reports gross 2.00, net +0.03. A true 150/50 would need `net_exposure_target = 1.0` and would borrow ~0.5 NAV on margin, whose interest is not modelled (audit tcost #21). Treat "gross 2.0, dollar-neutral" as the actual default.
</callout>

### Net exposure band
Net exposure $\mathbf{1}^\top w$ is held within `net_exposure_target ± net_exposure_band` (0 ± 0.05). Dollar neutrality removes most market risk mechanically. Beta neutrality (below) removes the rest. The band rather than an equality gives the optimiser slack, so a cost-free 1% net drift does not force trades.

### Per-name cap
$\lvert w_i\rvert \le 0.02$ (2% of NAV). This is the most important protection against error maximisation: however wrong one alpha is, one name can cost at most about 2% × (its move). With gross 2.0 and a 2% cap the book holds at least 100 names. In practice it holds many more, because risk aversion spreads weight further.

### ADV participation cap (in weight units)
ADV is average daily dollar volume (63-day mean). A trade larger than a few percent of ADV moves the price far more than the square-root model's calibration range and may simply not fill. The constraint is on the **trade**, not the position:

$$
\lvert x_i\rvert \le \text{part\_cap}_i = \frac{\texttt{max\_adv\_participation}\times \text{ADV}_i}{\text{NAV}}
$$

where:
- `max_adv_participation` = 0.05 (5% of ADV per day);
- $\text{ADV}_i$ in USD from the point-in-time cost snapshot; NAV in USD.

Dividing by NAV converts dollars into weight units, so the constraint is linear in $x$. Example: ADV $20M, NAV $1M → cap = 0.05 × 20/1 = 1.0, i.e. 100% of NAV, never binding at this size. At NAV $100M the same name is capped at 1% of NAV per day. **The constraint scales with NAV automatically**, which is why capacity tests should re-run the backtest at a larger `nav0`.

If the problem is infeasible *with* participation caps (e.g. a forced exit larger than 5% ADV), the default is to report the failure and hold the book. Setting `relax_participation_on_infeasible = True` re-solves without the caps and logs a warning. That is opt-in because silently dropping the caps was audit finding 6.

<callout icon="⚠️" color="yellow_bg">
Design tension, not a bug: the closing auction is only about 9% of daily volume, so a trade at 5% of ADV executed entirely in the auction would be ~55% of auction volume. That is far above `ExecutionConfig.auction_max_pct_of_auction_volume = 0.02` (≈0.18% of ADV). Live, the router sends such orders to in-session algos. The backtest charges them at the auction impact model, which is the more expensive one for large trades (→ see **Execution model**).
</callout>

### Beta neutrality and factor roles
The risk model's exposure matrix has columns that play different **roles**, recorded in `RiskModelOutput.factor_groups`:

| Role | Columns | Exposure of a stock | What $B_k^\top w$ means |
|---|---|---|---|
| `market` | one country factor | 1 for every stock | net exposure $\mathbf{1}^\top w$ |
| `industry` | Fama–French-48 dummies (or `IND00..` on synthetic data) | 1 if in industry, else 0 | net weight in that industry |
| `style` | `beta`, `size`, `momentum`, `value`, `residual_vol`, `liquidity`, `short_term_reversal`, `quality`, `investment`, `short_interest` | standardised: cap-weighted mean 0, sd 1, clipped ±3 | portfolio tilt in sd units |
| `statistical` | APCA companion factors | loadings | residual statistical exposure |
| `alignment` | (not a column) | — | handled by $y$, $\nu$ |

Beta neutrality must constrain the **`beta` style**: $\lvert B_{\text{beta}}^\top w\rvert \le$ `beta_band` (0.05 sd units).

<callout icon="⚠️" color="yellow_bg">
The CRITICAL audit bug (risk_portfolio #1, fixed). The old code found "the beta column" by searching for the first factor named `beta`, `market` or `mkt`. The real risk model lays columns out as `[market] + industries + styles`, so it always picked **`market`**. Every stock has exposure 1 to `market`, so the "beta-neutral" constraint became $\lvert\mathbf{1}^\top w\rvert \le 0.05$, a duplicate of the net band, and the beta style was left free. In a direct test the "beta-neutral" book carried a beta-style exposure of **2.2**. The tests did not catch it because the stand-in risk model in `tests/conftest.py` had no `market` column. The fix resolves roles through `factor_groups`, never by name prefix. The beta column is `cfg.beta_factor_name` *only if* it is in `factor_groups["style"]`, and a regression test now runs against the real `BarraStyleRiskModel` output (`test_factor_groups_drive_beta_and_ff48_industry_bands`).
</callout>

**From sd units to raw beta.** Write a stock's raw beta as $\beta_i = \bar\beta + s_\beta z_i$, where $z_i$ is the standardised style exposure, $\bar\beta$ the cap-weighted mean raw beta (≈1) and $s_\beta$ its cross-sectional sd. Then portfolio raw beta is

$$
\beta_p = \bar\beta\,\mathbf{1}^\top w + s_\beta\, B_{\text{beta}}^\top w
$$

With $\mathbf{1}^\top w \approx 0$ (dollar-neutral) the second term dominates, so bounding $B_{\text{beta}}^\top w$ bounds raw beta at roughly $\pm 0.05\,s_\beta$. For a 130/30 or long-only book the first term dominates and the style band is not a market-beta constraint.

If `beta_neutral=True` but no `beta` style exists, a note is added and the constraint is skipped. If `factor_groups` is empty (a custom risk model), the optimiser falls back to exact-name matching for beta and the `IND*` prefix for industries, with a logged warning.

### Industry bands
$\lvert B_{\text{ind}}^\top (w - w_b)\rvert \le$ `industry_band` (0.03) for every industry column, where $w_b$ is an optional benchmark (zero for the default book). The book may be at most 3% of NAV net long or short any industry. Industry returns are large, persistent and mostly unforecast by a stock-selection model; a 10% net bet on semis is a macro view.

The columns come from `factor_groups["industry"]`, so Fama–French-48 short names (`Banks`, `Chips`, `Other`) work. Before the fix, industry detection used the `"IND"` prefix and would have **silently switched off** on real data (audit #3). If `industry_band > 0` and no industry columns are found, a note says the band was not applied.

### Style bands on risk-only styles
$\lvert B_{\text{sty}}^\top (w - w_b)\rvert \le$ `style_band_sd` (0.10 sd) for each style in `neutral_styles`. The pipeline passes the **risk-only** styles (`ResearchPipeline.RISK_ONLY_STYLES`): `beta`, `size`, `residual_vol`, `liquidity`, `short_term_reversal`. `beta` is removed from the list when `beta_neutral` is on, because it has its own tighter band.

The **alpha-theme styles** (`momentum`, `value`, `quality`, `investment`, `short_interest`) are deliberately *not* banded. They are where the alpha lives. Neutralising them would remove the bet the alpha model is making and drive the transfer coefficient toward zero. The plan's wording is "|style exposure| to non-alpha styles ≤ 0.1 SD" (BUILD_PLAN §5). When `factor_groups` is present, unknown style names in `neutral_styles` raise `ValueError` instead of being silently ignored.

### No-trade lists
`no_trade` names get $x_i = 0$. The simulator puts two kinds of names there automatically:
- names with **no price** at $t$ (halted or not yet listed), since you cannot trade what has no price;
- names **ineligible and not held**, so nothing can be opened outside today's universe.

The plan's no-trade list for hard-to-borrow names is not wired automatically. A caller can pass `no_trade=[...]` directly to `solve`. A name in both `no_trade` and `force_to_zero` is treated as no-trade.

### Eligibility forced exits over 5 days
When a held name leaves the investable universe (it falls below the market-cap rank, price or ADV filter), the system must sell it. Dumping it in one auction is expensive, and its eligibility may bounce back tomorrow. The simulator therefore passes `force_to_zero = {ticker: cap}`, where the allowed $\lvert w\rvert$ shrinks linearly over `exit_ineligible_days = 5`:

$$
\text{cap}_k = \lvert w_{\text{entry}}\rvert\,\big(1 - k/5\big),\quad k = 1,\dots,5
$$

The optimiser enforces $\lvert w_i\rvert \le \max(\text{cap}_k,\ \lvert w_{0,i}\rvert - \text{part\_cap}_i)$, so it never demands a sale larger than the ADV cap allows in one day. That keeps the problem feasible. The ineligible name's expected return is set to 0, so the optimiser has no reason to resist the exit. If the name becomes eligible again, the forced exit is cancelled. Exits are logged in `diagnostics["forced_exits"]`.

### Extra constraints
`constraints_extra` takes callables `fn(w, x, tickers) -> [cvxpy constraints]` for anything bespoke, such as a sector cap or a hard turnover limit. They must be DCP-valid, and they are included in the volatility projection too.

### Pitfalls & checks
- **Over-constraining.** Every binding constraint costs alpha (see Diagnostics). Read `binding_constraints` in the solve log: if `max_position` binds on hundreds of names every day, your alphas are too large or the cap too tight.
- **Infeasibility.** Contradictory constraints (130/30 with a 0.5% cap and a 50-name universe) make the problem infeasible. The result is a held book and a counted failed solve, not an exception (see Solver failure semantics).
- Regression tests: `test_long_short_constraints_and_speed`, `test_participation_bounds`, `test_long_only_and_130_30`, `test_no_trade_and_style_band`, `test_beta_band_from_config_and_unknown_neutral_style_raises`, `test_force_to_zero_and_out_of_model_names`, `test_simulator_eligibility_hook_unwinds_and_blocks`.

### References
- [Clarke, de Silva & Thorley via Institutional Investor, The Cost of Constraint](https://www.institutionalinvestor.com/article/2btgijdgl1kk3akuoutc0/home/the-cost-of-constraint)
- [Electronic Code of Federal Regulations, 12 CFR Part 220 (Regulation T)](https://www.ecfr.gov/current/title-12/chapter-II/subchapter-A/part-220)

## Volatility targeting

### What it is
Volatility targeting scales the book so that its predicted annual volatility equals a target, `target_vol_annual = 0.10` (10%). It is like cruise control for risk: the optimiser decides *what* to hold, and the vol target sets *how much*.

### Why we need it here
The optimiser's natural risk level is $\text{SR}_{\text{ex-ante}}/(2\gamma)$. That swings with the alpha model's confidence, which is itself noisy. Without a target, the book's risk would rise when the model is (over)confident and fall when it is not. A fixed risk budget makes PnL comparable across time, sets drawdown expectations, and gives risk control that does not depend on alpha calibration. For long equity, Harvey et al. found vol targeting cut vol-of-vol from 4.6% to 1.8% and raised Sharpe from 0.40 to about 0.48–0.51. For a dollar-neutral book the plan expects mostly the vol-of-vol reduction (BUILD_PLAN §5).

**Why 10%.** The plan treats a net Sharpe of 0.5–1.0 as success. At 10% vol that is 5–10% expected annual return, and a 2-sigma bad year at Sharpe 0.6 is about $6\% - 2\times10\% = -14\%$. That is a level a part-time operator can sit through. The number is a convention for market-neutral books, not derived in the plan. Treat it as a tunable risk budget.

### How it works: projection, not multiplication
After the alpha solve returns $w^\star$, compute predicted vol $\hat\sigma = \sqrt{252\,w^{\star\top}\Sigma w^\star}$ and the scale $s = \sigma_{\text{target}}/\hat\sigma$.

- $\lvert s - 1\rvert \le 10^{-4}$: nothing to do.
- $s < 1$ (too risky): always scale down.
- $s > 1$ (too timid): scale up only if `vol_target_scale_up=True`, the book is `long_short` (long-only and 130/30 are fully invested), and gross is below the cap. $s$ is capped at `MAX_SCALE_UP = 5`.

Then solve a **second** problem:

$$
\min_w \ \lVert w - s\,w^\star\rVert_2^2 \quad\text{s.t.}\quad \text{every constraint of the alpha solve},\quad \big\lVert \big[L^\top B^\top w;\ \sqrt d\circ w;\ \sqrt\nu\, y^\top w\big]\big\rVert_2 \le \frac{\sigma_{\text{target}}}{\sqrt{252}}
$$

where:
- $s\,w^\star$ is the "ideal" scaled book, used as an anchor;
- the norm constraint is the SOC form of $\sqrt{w^\top\Sigma_{\text{aug}}w} \le$ daily target vol ($\Sigma_{\text{aug}}$ includes the alignment term);
- the constraint list is *the same Python objects* as in the alpha solve, including participation if it was active.

So the result is the feasible book closest to $s w^\star$ with predicted vol at or below target. Names at their per-name or ADV caps stay clipped while the rest of the book is levered. If the projection fails, $w^\star$ is kept with a note.

<callout icon="⚠️" color="yellow_bg">
Why post-hoc scaling broke the constraints (audit risk_portfolio #2, HIGH, fixed). The old code computed $s$, checked it only against gross, the per-name cap and an ADV bound, and returned $s\,w^\star$. But the optimiser *legitimately sits on* its linear bands. If net exposure was at its 0.05 limit, $s = 1.7$ put it at 0.085. If the beta exposure sat at the style band, scaling broke it (−0.13 vs 0.10 in the pipeline config). The ADV bound was wrong when a trade flipped a position's sign: 7 of 150 names ended up at up to 1.8× the participation cap. Any constraint of the form $\lvert c^\top w\rvert \le b$ is violated by scaling $w$ by $s>1$ when it binds. The projection keeps the whole feasible set, so this cannot happen. Regression tests: `test_vol_target_scale_up_is_a_projection`, `test_vol_target_scale_down_respects_no_trade_and_participation`.
</callout>

### Why this choice
- *Multiply and check every band* (audit option b) works, but it is fragile: every new constraint needs its own scale bound.
- *Put the risk cap inside the alpha solve* is cleaner in theory, but a cap only binds from above. It cannot lever *up* a timid book, and it changes the alpha solve's duals, which are used for reporting.
- *Realised-vol targeting* (scaling by trailing realised vol) is available as `qts/portfolio/vol_target.py::realized_vol_scalar` for overlays and reporting. The optimiser uses the ex-ante risk model because it reacts the same day.

### In the code
`SinglePeriodOptimizer._vol_target_projection`; `OptimizationResult.vol_scalar` (the anchor $s$) and the note `vol-target projection s=…: vol a -> b`. `qts/portfolio/vol_target.py::portfolio_vol_scalar` is the plain $\text{target}/\hat\sigma$ clip, used for reporting only.

### Pitfalls & checks
- **The target is effectively a cap when leverage binds.** In the post-audit demo, realised net vol was **6.6%** against a 10% target, because gross was at the 2.0 cap (`vol-target scale-up blocked: gross leverage already at cap`). To reach 10% you would need more gross (not allowed under Reg T) or more concentrated bets (a larger `max_position_weight`, looser bands). Low realised vol here is a feature of the constraints, not a bug.
- **The projection ignores costs and alpha.** Its objective is pure distance to $s w^\star$, so scaling trades are not weighed against their cost. When $s$ changes a lot day to day (for example after a vol-regime shift in the risk model), the projection can add turnover the cost model never approved. Watch `vol_scalar` in the solve log. A smoother approach is on the roadmap.
- **Ex-ante vs realised.** The target is only as good as the risk forecast. Check `realized_vs_exante_vol_ratio` (1.04 in the demo). Persistently above 1.2 means the model under-forecasts and the book is riskier than targeted.
- `binding_constraints` is read from the **alpha** solve's duals, before the projection overwrites them.

### References
- [Harvey et al. (2018), The Impact of Volatility Targeting](https://people.duke.edu/~charvey/Research/Published_Papers/P135_The_impact_of.pdf)

## Alpha scaling and combination

### What it is
The optimiser needs **expected returns**. The alpha model produces **scores**: z-scores with mean 0 and sd 1. Converting one into the other with the right *magnitude* matters as much as the ranking, because the magnitude decides how alpha trades off against risk and costs. Raw z-scores fed into the objective would look like a 100% expected return per name and "get eaten by the optimiser" (MSCI, BUILD_PLAN §5).

### How it works: Grinold's rule
Grinold's "alpha = volatility × IC × score" rule follows from a regression of returns on scores. If a score $z_i$ (sd 1) has correlation IC with the residual return $r_i$ (sd $\sigma_i$), the best linear forecast is

$$
\alpha_i = \mathbb{E}[r_i \mid z_i] = \text{IC}\cdot\sigma_i\cdot z_i
$$

where:
- IC = information coefficient, the cross-sectional correlation between today's score and the realised residual return over the horizon (typically 0.02–0.05 for good equity signals);
- $\sigma_i$ = residual volatility of stock $i$ **over the same horizon** ($H$ days): daily residual vol × $\sqrt H$;
- $z_i$ = cross-sectionally re-standardised score; missing scores map to $\alpha_i = 0$ ("no opinion").

Two things follow. A volatile stock gets a larger alpha for the same score, because it moves more. And the alpha *dispersion* is IC × typical vol, a few tenths of a percent over 20 days, which is the right order of magnitude to compete with risk and costs.

**IC shrinkage.** The IC itself is estimated, with selection bias (we keep signals whose backtest IC looked good) and decay (published anomalies keep roughly half their in-sample return; Chen & Velikov via BUILD_PLAN §1). The plan therefore shrinks the walk-forward IC by 50%. In `AlphaModel.as_of`:

```python
ic_tr = self.trailing_ic(date)
ic_shrunk = self.cfg.ic_shrink * ic_tr if np.isfinite(ic_tr) else self.cfg.default_ic
ic_shrunk = max(ic_shrunk, 0.0)
vol = self.resid_vol_h_.loc[date].reindex(score.index)
vol = vol.fillna(vol.median()).fillna(0.0)
expected = (ic_shrunk * vol * score).rename("expected_return")
```

`ic_shrink = 0.5`, `default_ic = 0.03` (used when there are fewer than 60 IC observations), floored at 0 so a negative trailing IC switches the alpha off rather than flipping it. `qts/portfolio/alpha_scaling.py::grinold_alpha` and `shrink_ic` are the stand-alone versions of the same rule.

<callout icon="⚠️" color="yellow_bg">
Parameter semantics differ between the two IC-shrink paths. `AlphaModel` multiplies by `ic_shrink` (it *keeps* that fraction), while `shrink_ic(ic, shrink)` multiplies by `1 - shrink` and caps at 0.1. At the default 0.5 they agree. At `ic_shrink = 0.3` the alpha model keeps 30% of the IC, while `shrink_ic(shrink=0.3)` keeps 70%. Also, `default_ic` is used without shrinkage. Check which path you are tuning.
</callout>

### Gârleanu–Pedersen: the "aim portfolio" intuition
Gârleanu and Pedersen (2013) solved the multi-period problem with quadratic costs and mean-reverting signals in closed form. Two results matter here:

1. **Trade partially toward an aim.** The optimal trade moves a fixed fraction of the gap between the current portfolio and an *aim* portfolio each period. You never jump to the target, because costs are convex: two half-trades cost less than one full trade.
2. **The aim looks ahead.** The aim is a weighted average of today's Markowitz portfolio and the *expected future* Markowitz portfolios. A signal that decays fast (short half-life) will mostly be gone by the time you have finished trading into it, so the aim gives it less weight. Each signal is effectively shrunk by a factor that depends on its decay speed relative to the trading speed.

In their in-sample commodity-futures test the dynamic rule earned a net Sharpe of 0.58, versus 0.46 for the best static rule and −9.84 for naive Markowitz rebalancing every period (BUILD_PLAN §5, in-sample).

**How this system gets GP behaviour from SPO.** The trade-cost penalty already makes the single-period solve move partway toward its target each day. That is property 1. Property 2 can be applied *in alpha space*, before the optimiser. With signal $k$ mean-reverting at speed $\phi_k = \ln 2/\text{hl}_k$ and the cost-implied trading speed $\rho = \ln 2/H_{\text{hold}}$, the GP weight $1/(1+\phi_k/\rho)$ becomes

$$
g_k = \frac{1}{1 + H_{\text{hold}}/\text{hl}_k}, \qquad \alpha = \sum_k \pi_k\, g_k\, \alpha_k
$$

where:
- $\text{hl}_k$ = half-life of sleeve $k$'s signal in days (from score autocorrelation);
- $H_{\text{hold}}$ = cost-implied holding period in days; `cost_implied_holding_days` gives the crude proxy "round-trip cost ÷ alpha per day", clipped to [1, 252];
- $\pi_k$ = prior sleeve weights (normalised to sum to 1);
- $\alpha_k$ = Grinold-scaled alpha of sleeve $k$.

A signal with half-life equal to the holding period is halved. One with a half-life 4× the holding period keeps 80%.

**The `renormalize` flag.** By default (`renormalize=False`) the $g_k$ are applied as is, so the *level* of the combined alpha falls as holding periods lengthen. This is GP's behaviour, and it matters because the level sets the alpha/risk/cost balance. `renormalize=True` rescales the weights to sum to one: only the *mix* changes and a single sleeve is never shrunk. That was the pre-audit behaviour (audit #8). Use it only if the alpha level is calibrated elsewhere.

<callout icon="💡" color="blue_bg">
Combine forecasts, never portfolios. Optimising each sleeve separately and averaging the portfolios ("mixed portfolios") double-counts shared risk and wastes netting between sleeves. One sleeve's buy is another's sell, and trading both costs twice. `combine_in_forecast_space` sums alphas first, so one optimiser sees one alpha vector (BUILD_PLAN §5).
</callout>

### Current wiring
The pipeline's `AlphaModel` already combines its sleeves (rank composite + ML) into **one** score and applies Grinold with a single shrunk IC. `garleanu_pedersen_shrink` is a library function that **is not called by `pipeline.py` today**. `AlphaOutput.half_life_days` is computed and carried, but the optimiser does not use it. In the default system, GP-style partial trading comes entirely from `trade_aversion` and the convex cost. GP half-life shrinkage across sleeves is available for multi-sleeve research (`gp_sleeve_shrink_factors`, `gp_sleeve_weights`, `combine_in_forecast_space`).

### Pitfalls & checks
- **Horizon mismatch.** $\sigma_i$ must be over the alpha horizon. Daily vol with a 20-day IC under-scales alpha by $\sqrt{20}\approx4.5\times$.
- **Missing vol.** It is filled with the cross-sectional median. Missing score means alpha 0.
- **Score re-standardisation** happens on the eligible cross-section only, so out-of-universe names cannot shift the mean (the CRITICAL eligibility fix in `RESOLUTION.md`).
- Tests: `test_grinold_shapes_and_missing`, `test_gp_weights`, `test_combine_and_shrink_ic`.

### Worked example
IC estimate 0.04 → shrunk to 0.02. Daily residual vol 1.5% → 20-day $\sigma = 0.015\sqrt{20} = 6.7\%$. Score $z = +2$. Then $\alpha = 0.02 \times 0.067 \times 2 = 0.27\%$ over 20 days (about 3.4%/yr annualised). Check against risk: with $\gamma=10$, $H=20$, $d = 0.015^2$, the unconstrained single-name optimum is $w = \alpha/(2\gamma H d) = 0.00268/(2\cdot10\cdot20\cdot0.000225) = 2.98\%$. The 2% cap binds. A $z=+1$ name would want 1.49% and stay inside the cap. This is the right order of magnitude: only extreme scores reach the cap.

### References
- [MSCI, Converting Scores into Alphas](https://www.msci.com/documents/10199/1645561/PI_Converting_Scores_Into_Alphas.pdf/7adf1f42-10aa-40eb-9e8c-ecc11eeba2d4)
- [Gârleanu & Pedersen (2013), Dynamic Trading with Predictable Returns and Transaction Costs](https://research-api.cbs.dk/ws/portalfiles/portal/60191441/lasse_heje_pedersen_et_al_dynamic_trading_acceptedmanuscript.pdf)
- [Chen & Velikov (2023), Zeroing In on the Expected Returns of Anomalies](https://core-prod.cambridgecore.org/core/services/aop-cambridge-core/content/view/945133D5A3ECEEAF466AEE91551FD225/S0022109022000874a.pdf/zeroing-in-on-the-expected-returns-of-anomalies.pdf)

## Diagnostics

The optimiser always returns *something*. Diagnostics tell you whether that something resembles what the alpha model asked for, and whether the risk it was sized on was real.

### Transfer coefficient
**What it is.** The transfer coefficient (TC; Clarke, de Silva & Thorley, 2002) is the correlation between the positions you hold and the positions an unconstrained optimiser would hold. It measures how much of the forecast survives constraints and costs. The "fundamental law of active management" with constraints reads

$$
\text{IR} \approx \text{TC}\times\text{IC}\times\sqrt{\text{BR}}
$$

where:
- IR = information ratio (expected active return / active risk);
- IC = forecast skill; BR = breadth (independent bets per year);
- TC ∈ [−1, 1] = transfer coefficient.

**What 0.3 vs 0.8 means.** At TC = 0.8 you capture 80% of the IR your forecasts could deliver; at 0.3, under a third. The same alpha model is worth more than twice as much in a long/short book (TC ≈ 0.7–0.8) as in a constrained long-only one (≈ 0.3). **Constraints are the largest alpha leak in most real portfolios**: larger than costs and often larger than the gap between a good and a mediocre signal.

**How it is computed** (`qts/portfolio/diagnostics.py::transfer_coefficient`):
- *Exact*, for $N \le 500$ with a covariance: the $\Sigma$-correlation between $w$ and the unconstrained Markowitz direction $\Sigma^{-1}\alpha$:

$$
\text{TC} = \frac{\alpha^\top w}{\sqrt{w^\top\Sigma w}\,\sqrt{\alpha^\top\Sigma^{-1}\alpha}}
$$

- *Approximate*, otherwise: $\mathrm{corr}(w_i\sigma_i,\ \alpha_i/\sigma_i)$ with $\sigma_i$ the specific vol (Clarke–de Silva–Thorley's risk-adjusted form).

Each solve records TC, and `BacktestResult.summary()["transfer_coefficient_mean"]` averages it. The post-audit Phase-1 demo shows **0.43**. Typical culprits are the 2% per-name cap, the gross cap, and the trade-cost penalty, which deliberately keeps yesterday's positions. A low TC from the cost penalty is fine (it is GP behaviour). A low TC from a constraint is a design decision worth revisiting.

### Effective holding period
$$
\text{EHD} = \frac{\overline{\text{gross}}}{\overline{\text{two-sided turnover}}/2}
$$

where:
- $\overline{\text{gross}}$ = mean $\sum_i\lvert w_{it}\rvert$;
- two-sided turnover $= \sum_i\lvert x_{it}\rvert$ per day (selling 1% and buying 1% is 2%).

This is "how many days a dollar of position lives on average". Demo: gross 2.00, turnover 6.90%/day → $2.00/0.0345 = 58$ days. `effective_holding_days_target = 50` is monitored, not enforced. The pipeline adds a note ("raise trade_aversion") when EHD < 0.6 × target. EHD should be comparable to the signal's half-life. Much shorter, and you are paying to trade noise.

### Factor exposure report
`factor_exposure_report(w, risk)` returns, per factor: exposure $f_k = (B^\top w)_k$, annualised factor vol, variance contribution $f_k (Ff)_k$ and share of total variance, plus `__specific__` and `__total__` rows. For a healthy market-neutral stock picker, the specific share should be large (often >50%), `market` and industry contributions near zero, and alpha-theme styles carrying the deliberate tilts. A large contribution from a *risk-only* style means a band is not working.

### Realised vs ex-ante risk
`realized_vs_exante_risk(returns, exante_var)` compares realised vol with the average forecast and computes the bias statistic $\mathrm{sd}(r_t/\hat\sigma_{t-1})$, which should be ≈1. The forecast made at $t$ is aligned to the return earned $t\to t+1$ (`exante_var.shift(1)` in `summary()`, verified in the audit). Demo ratio: 1.04.

### Attribution: factor vs specific PnL
`brinson_like_attribution` splits each day's PnL into factor PnL $\sum_k (B_t^\top w_t)_k f_{k,t+1}$ and specific PnL $w_t^\top u_{t+1}$. It uses weights held at $t$, exposures as of $t$, and factor and specific returns realised at $t+1$. A stock-selection strategy should make most of its money in the specific column and in the alpha-theme factors it intends to hold. Steady PnL from `market` or an industry means an unintended bet. This function is a library tool and is not printed by the pipeline report.

### References
- [Clarke, de Silva & Thorley (2002) via Institutional Investor, The Cost of Constraint](https://www.institutionalinvestor.com/article/2btgijdgl1kk3akuoutc0/home/the-cost-of-constraint)

## What we don't use and why

<callout icon="🤔" color="purple_bg">
End-to-end / decision-focused learning (training a network to output weights that maximise realised Sharpe). The positive results come from tiny universes: 7 ETFs, Sharpe 1.16 vs 0.79 for risk parity, but with no transaction costs and one test window (Uysal, Li & Mulvey); 1.33 vs 1.24 at 20 bp on 7 ETFs, single split (IPMO). The most practitioner-grounded study, on S&P 500 universes, found raw end-to-end Sharpe maximisation unstable and worse than predict-then-optimise (Kim et al.). If decision-focused calibration is ever used here, it is limited to a few scalars: risk aversion, shrinkage intensity (BUILD_PLAN §5).
</callout>

<callout icon="🤔" color="purple_bg">
Reinforcement-learning allocation. The robustness record of RL portfolio agents is poor (Velay et al.): sensitive to seeds, reward shaping and regime. The plan keeps RL out of allocation entirely. A convex optimiser has a global optimum, interpretable duals and no training instability.
</callout>

<callout icon="🤔" color="purple_bg">
Hierarchical risk parity (HRP) and variants (NCO, Schur complement). As of 2026 the evidence is synthetic only. CRISP reached 80–94% of oracle Sharpe in simulation, but the Schur-complement allocator was unstable at intermediate settings (Wuebben). HRP also has no alpha input. These methods are useful as sanity **baselines** (e.g. from skfolio), not as the production allocator.
</callout>

Also avoided: optimising each signal separately and averaging portfolios (see "Combine forecasts, never portfolios"), and naive full Markowitz rebalancing every day (Gârleanu–Pedersen's −9.84 net Sharpe).

### References
- [Uysal, Li & Mulvey (2021), End-to-End Risk Budgeting Portfolio Optimization with Neural Networks](https://ar5iv.labs.arxiv.org/html/2107.04636)
- [Velay et al. (2023), Benchmarking Robustness of Deep Reinforcement Learning approaches to Online Portfolio Management](https://www.arxiv.org/abs/2306.10950)
- Kim et al. (2025) and Wuebben (2026): see the links in `docs/BUILD_PLAN.md` §5.

## Parameters, tuning and failure semantics

### Every `PortfolioConfig` field
Defaults from `qts/core/config.py::PortfolioConfig`.

| Field | Default | Meaning | Notes |
|---|---|---|---|
| `book` | `"long_short"` | `"long_short"`, `"130_30"`, `"long_only"` | selects the leverage/net constraint block |
| `gross_leverage_max` | 2.0 | $\lVert w\rVert_1$ cap (long_short only) | Reg T; the comment "150/50" is misleading (see above) |
| `net_exposure_target` | 0.0 | centre of the net band (long_short) | 130/30 uses 1.0 internally |
| `net_exposure_band` | 0.05 | ± band on net exposure | also used by 130/30 |
| `max_position_weight` | 0.02 | per-name absolute weight cap | all books |
| `max_adv_participation` | 0.05 | per-name trade cap as share of ADV | converted to weights via ADV/NAV |
| `beta_neutral` | True | bind the beta *style* | resolved via `factor_groups["style"]` |
| `industry_band` | 0.03 | ± net weight per industry | via `factor_groups["industry"]`; 0 disables |
| `style_band_sd` | 0.10 | ± sd exposure per style in `neutral_styles` | pipeline passes risk-only styles |
| `risk_aversion` | 10.0 | $\gamma_{\text{risk}}$ (annual-equivalent) | multiplies $H\times$ daily variance |
| `trade_aversion` | 1.0 | $\gamma_{\text{trade}}$ | main turnover knob |
| `hold_aversion` | 1.0 | $\gamma_{\text{hold}}$ | multiplies $H\times$ daily borrow |
| `turnover_penalty` | 0.0 | extra linear penalty $\kappa\lVert x\rVert_1$ | off by default |
| `target_vol_annual` | 0.10 | vol target for the projection | `None` disables |
| `vol_target_scale_up` | True | allow levering up toward target | long_short only, within gross cap, $s\le5$ |
| `effective_holding_days_target` | 50 | monitored holding period | pipeline note if EHD < 30 |
| `solver` | `"CLARABEL"` | first solver tried | chain: this, CLARABEL, SCS, OSQP (installed only) |
| `beta_factor_name` | `"beta"` | name of the beta style | must be in `factor_groups["style"]` |
| `beta_band` | 0.05 | ± beta-style exposure (sd units) | constructor arg overrides |
| `relax_participation_on_infeasible` | False | drop ADV caps if infeasible | opt-in; logged |
| `exit_ineligible_days` | 5 | linear unwind length for ineligible holdings | used by the simulator |

### Tuning guide
- **Start from turnover, not from Sharpe.** The goal is an effective holding period comparable to the alpha's half-life. If the signal half-life is ~60 days and EHD is 15, raise `trade_aversion` (try 2, 5). If EHD is far longer than the half-life, alpha decays before you trade into it, so lower `trade_aversion`. The plan's target is a penalty that produces 20–40% of the frictionless trade per day. Lopez-Lira & Tang found 25% partial rebalancing cut turnover 4× with no Sharpe loss (BUILD_PLAN §5).
- **Turnover vs alpha decay.** A trade costs $c$ once and earns roughly the alpha over the signal's remaining life. With half-life hl, a forecast's expected remaining alpha is about $\alpha\cdot\text{hl}/\ln 2$ in daily-alpha units. Fast signals (short hl) justify less trading per unit of alpha. That is GP shrinkage expressed as a tuning rule.
- **`risk_aversion` matters less than you think** while the vol target is active, because the projection rescales the book anyway. It still shapes the *direction* of $w^\star$ (more diversification at higher $\gamma$). Tune it for diversification and TC, not for the risk level.
- **`hold_aversion`** scales borrow. Raise it if the short book drifts into expensive names. Check `borrow` costs in the backtest.
- **Bands.** Loosen one band at a time and watch TC and the factor exposure report. A band that never binds (no entry in `binding_constraints`) is free insurance. One that binds daily is costing alpha.
- **Tune out of sample only.** Every change is a trial for the deflated Sharpe (→ see **Validation & statistics › The trial ledger**). Parameter grids for PBO go in `SystemConfig.pipeline_pbo_configs`, e.g. `[{"portfolio.trade_aversion": 2}, {"portfolio.trade_aversion": 5}]`.

### Solver failure semantics
- Solvers are tried in order: `cfg.solver`, `CLARABEL`, `SCS`, `OSQP` (deduplicated, installed only). A solver exception moves on to the next one. A status of `infeasible` or `unbounded` is returned immediately, since genuine infeasibility will not be fixed by another solver. OSQP accepts only QPs, so it can solve the alpha problem only when no 3/2-power cost term is present.
- `optimal` and `optimal_inaccurate` count as success (`OptimizationResult.ok`).
- **On failure:** `w_target = w_current` (hold the book), `x = 0`, `expected_cost = NaN`, `transfer_coefficient = NaN`, `status` = the solver's status, and a warning is logged. The simulator counts it in `n_failed_solves`, which `summary()` reports, and keeps going.
- If `relax_participation_on_infeasible=True`, a second attempt without ADV caps runs first and a note records it.
- After a successful solve, weights below $10^{-6}$ are zeroed and untouched names keep their exact current weight (`_clean`), so solver noise does not create micro-trades.
- `expected_cost` is recomputed on the **final** (post-projection) trade as trade cost + $H\times$ holding cost.

<callout icon="⚠️" color="yellow_bg">
A failed solve means "hold yesterday's book", not "go flat". That is the right default for a daily rebalancer, because one missed day costs little. But a run of failures leaves forced exits unexecuted and risk unmanaged. Treat `n_failed_solves > 0` in a backtest as a defect to investigate, and alert on it in production.
</callout>

### Worked example: 2-asset mean–variance with a cap
Two stocks over one period: expected returns $\alpha = (4\%, 2\%)$, vols $\sigma = (20\%, 10\%)$, correlation 0.5, risk aversion $\gamma = 2$.

$$
\Sigma = \begin{pmatrix} 0.04 & 0.01\\ 0.01 & 0.01\end{pmatrix},\quad \det\Sigma = 0.0003,\quad \Sigma^{-1} = \frac{1}{0.0003}\begin{pmatrix} 0.01 & -0.01\\ -0.01 & 0.04\end{pmatrix}
$$

**Unconstrained.** $\Sigma^{-1}\alpha = \frac{1}{0.0003}(0.0004-0.0002,\ -0.0004+0.0008) = (0.667,\ 1.333)$, so

$$
w^\star = \frac{1}{2\gamma}\Sigma^{-1}\alpha = \tfrac14(0.667,\ 1.333) = (16.7\%,\ 33.3\%)
$$

Expected return $\alpha^\top w^\star = 1.333\%$; variance $w^{\star\top}\Sigma w^\star = 0.00333$ (vol 5.77%); utility $=1.333\% - 2\times0.00333 = 0.667\%$.

**Add a per-name cap $w_i \le 25\%$.** B wants 33.3%, so the cap binds at $w_B = 25\%$. Re-optimise A given $w_B$. The first-order condition for A is $\alpha_A - 2\gamma(\sigma_A^2 w_A + \sigma_{AB} w_B) = 0$:

$$
0.04 - 4\,(0.04\,w_A + 0.01\times 0.25) = 0 \;\Rightarrow\; w_A = \frac{0.04 - 0.01}{0.16} = 18.75\%
$$

The cap on **B** *raised* **A** from 16.7% to 18.75%. Because the two are positively correlated, holding less B means less overlapping risk, so the optimiser can afford more A. A cap is never a local edit: it re-prices risk for every correlated name. That is why `binding_constraints` and TC matter.

Check the KKT condition for B (the gradient must push against the cap): $\alpha_B - 2\gamma(\sigma_{AB}w_A + \sigma_B^2 w_B) = 0.02 - 4(0.001875 + 0.0025) = 0.0025 > 0$. So B "wants" more, and the shadow price of the cap is 0.25% of utility per unit of weight. This is the dual the optimiser reads for `binding_constraints`.

Constrained portfolio: return $0.04(0.1875)+0.02(0.25) = 1.25\%$; variance $0.04(0.1875^2) + 2(0.01)(0.1875)(0.25) + 0.01(0.25^2) = 0.00297$ (vol 5.45%); utility $1.25\% - 2(0.00297) = 0.656\%$, a 1.6% utility loss. Transfer coefficient: $\alpha^\top\Sigma^{-1}\alpha = 0.04(0.667)+0.02(1.333) = 0.0533$, so TC $= 0.0125/\sqrt{0.00297\times0.0533} = 0.993$. A single mild cap costs little. Dozens of binding caps and bands compound to the 0.43 seen in the demo.

**Error maximisation in the same example.** Suppose B's forecast is 3% instead of 2%, a one-point error. Then $\Sigma^{-1}\alpha = \frac{1}{0.0003}(0.0001,\ 0.0008) = (0.333,\ 2.667)$ and $w^\star = (8.3\%,\ 66.7\%)$. A 1-point change in one input **doubled** B and **halved** A. The cap limits B to 25% whatever the forecast says. That is the practical case for per-name caps.

## Roadmap
- **Cost-aware vol projection.** Add the trade-cost term to the projection objective (or smooth $s$ over days) so scaling does not create unpriced turnover.
- **Alignment calibration on the default path.** Run `calibrate_alignment_nu` once optimised-book bias statistics exist (open item in `RESOLUTION.md`).
- **Hard-to-borrow no-trade list** wired from borrow data (plan §5 hard constraint), and per-name short caps by borrow availability.
- **Robust covariance option:** worst case over two covariance scenarios (plan §5, Phase-2 option).
- **Multi-period optimisation** around known events (earnings, index rebalances) only.
- **GP shrinkage wired into the pipeline** once there are multiple sleeves with distinct half-lives, using `garleanu_pedersen_shrink(renormalize=False)`.
- **Out-of-model policy.** Option to zero the alpha of names outside the risk universe instead of pricing them at the 90th-percentile specific variance.
- **Unify IC-shrink semantics** between `AlphaConfig.ic_shrink` and `shrink_ic`.
