# Risk model

<callout icon="🧭" color="gray_bg">
The risk model forecasts how much any portfolio of our ~1,000 stocks can move tomorrow. It writes the stock covariance matrix as $\Sigma = BFB^\top + D$: a small number of common factors (market, industries, ten styles) with exposures $B$ and factor covariance $F$, plus a diagonal of stock-specific variances $D$. Factor returns come from a daily weighted cross-sectional regression. $F$ is an exponentially weighted, Newey–West-corrected covariance with an eigenfactor adjustment and a volatility-regime adjustment. $D$ is per-stock EWMA risk, blended with a structural model for stocks with short histories and shrunk toward size-decile means. On top of that sit a statistical (APCA) companion fitted on the residuals, which prices common moves the fundamental model cannot see, and a Saxena–Stubbs alpha-alignment term that stops the optimizer from piling into directions the model is blind to. Everything is point-in-time, in daily units, and lives in `qts/risk/`.
</callout>

## The risk model at a glance

The optimizer (→ see **Portfolio construction**) trades alpha (→ see **Alpha model**) against risk from this layer. The contract is `qts/core/types.py::RiskModelOutput`, one object per date:

| Field | Shape | Meaning |
|---|---|---|
| `exposures` | tickers × factors | $B$: how much each stock loads on each factor |
| `factor_cov` | factors × factors | $F$: daily covariance of factor returns |
| `specific_var` | tickers | diagonal of $D$: daily idiosyncratic variance |
| `factor_returns` | dates × factors | estimated factor-return history up to and including `date` |
| `alignment_vector`, `alignment_var` | tickers, scalar | $y$ and $\nu$ of the alignment term $\nu y y^\top$ |
| `factor_groups` | dict | which columns are `market`, `industry`, `style`, `statistical`, `alignment` |

`RiskModelOutput.covariance()` returns $BFB^\top + D + \nu y y^\top$. `RiskModelOutput.portfolio_variance(w)` returns $w^\top \Sigma w$ without building the $N \times N$ matrix. The optimizer uses `factor_groups` to find the beta style and the industry block by role, not by name. An audit found the old name-matching approach constrained the wrong column (`docs/audit/RESOLUTION.md`, CRITICAL row 3).

The pipeline, in the order the code runs it:

| Stage | Module | Output |
|---|---|---|
| 1. Exposures | `qts/risk/exposures.py` | $B_t$ for every date |
| 2. Factor returns | `qts/risk/factor_returns.py` | $f_t$, specific returns $u_t$, $R^2_t$ |
| 3. Factor covariance | `qts/risk/covariance.py::factor_covariance` | $F$ |
| 4. Specific risk | `qts/risk/covariance.py::specific_risk` | $D$ |
| 5. Fundamental model | `qts/risk/model.py::BarraStyleRiskModel` | `RiskModelOutput` |
| 6. Statistical companion | `qts/risk/statistical.py`, `qts/risk/hybrid.py` | extra residual factors |
| 7. Alpha alignment | `qts/risk/alignment.py` | $y$, $\nu$ |
| 8. Evaluation | `qts/risk/evaluation.py` | bias statistics, QLIKE, MZ, GMV tests |

`qts/risk/model.py::build_risk_model` is the factory the pipeline calls (`qts/pipeline.py`). With the default `RiskConfig.statistical_companion = True` it returns a `HybridRiskModel` (stages 1–7). Otherwise it returns a plain `BarraStyleRiskModel` (stages 1–5 and 7).

**Units.** All variances are **daily** and all returns are simple daily returns stored as fractions (0.01 = 1%). To annualise a variance, multiply by 252. To annualise a volatility, multiply by $\sqrt{252} \approx 15.9$.

## 1. Why a risk model

**What it is.** A risk model is a forecast of the covariance matrix $\Sigma$ of next-period stock returns. Given portfolio weights $w$ (fractions of NAV, positive for longs and negative for shorts), the forecast variance of the portfolio return $r_p = w^\top r$ is

$$
\operatorname{Var}(r_p) = w^\top \Sigma w = \sum_i \sum_j w_i w_j \Sigma_{ij}
$$

where:
- $w \in \mathbb{R}^N$ is the vector of portfolio weights;
- $r \in \mathbb{R}^N$ is the vector of next-day stock returns;
- $\Sigma_{ij} = \operatorname{Cov}(r_i, r_j)$, daily units.

Think of it as a weather forecast for a portfolio: it says how far the portfolio is likely to move, not which way.

**Why we need it here.** The system uses $\Sigma$ in three places:
1. The optimizer's objective penalises $\gamma\, w^\top \Sigma w$, which sizes positions and hedges common exposures.
2. The volatility target scales the book so that $\sqrt{252\, w^\top \Sigma w}$ is about 10% a year (`PortfolioConfig.target_vol_annual`).
3. Residualisation: alpha labels are regressed on the risk exposures so the alpha model predicts what the risk factors don't already explain (→ see **Alpha model › Labels**).

If $\Sigma$ underestimates risk in some direction, the optimizer piles into that direction because it looks free. Almost every failure mode on this page comes back to that.

**How it works: why the sample covariance fails.** The obvious estimator is the sample covariance of the last $T$ days of returns, $S = \frac{1}{T}\sum_t (r_t - \bar r)(r_t - \bar r)^\top$. For our universe it is useless:

- **Parameter count.** A symmetric $N \times N$ matrix has $N(N+1)/2$ free parameters. For $N = 1{,}000$ that is **500,500** numbers. Three years of daily data gives $T \approx 750$ observations per stock. Each covariance entry is estimated from 750 noisy products, so each one carries a lot of noise, and there are half a million of them.
- **Rank deficiency.** $S$ is a sum of $T$ rank-one matrices, so $\operatorname{rank}(S) \le T$. When $N > T$, at least $N - T$ eigenvalues are exactly zero. Some long/short portfolio then has a forecast risk of **zero**. A mean–variance optimizer will find it and lever it without limit.
- **Eigenvalue dispersion (noise eigenvalues).** Even when $N < T$, sample eigenvalues are spread out much more than the true ones. The large ones come out too large and the small ones too small. The optimizer wants exactly the small-eigenvalue directions, because they look like cheap risk.

<callout icon="💡" color="blue_bg">
Marchenko–Pastur intuition. Suppose every stock were independent noise with variance $\sigma^2$, so the true $\Sigma = \sigma^2 I$ and every true eigenvalue equals $\sigma^2$. As $N, T \to \infty$ with $q = N/T$ fixed, the eigenvalues of the sample covariance do not pile up at $\sigma^2$. They spread over the interval $[\sigma^2(1-\sqrt{q})^2,\ \sigma^2(1+\sqrt{q})^2]$ (the Marchenko–Pastur law). With $N = 1000$, $T = 750$ ($q = 1.33$) the top noise eigenvalue is about $4.6\,\sigma^2$, and 250 eigenvalues are exactly zero. Pure noise looks like a strong "factor" plus a large set of riskless portfolios. An eigenvalue is believable only if it sits clearly above the upper edge $\lambda_+ = \sigma^2(1+\sqrt{q})^2$. We use this threshold again in the statistical model (§8).
</callout>

**Factor models as structured covariance.** A factor model assumes stock returns are driven by $K \ll N$ common factors plus independent noise:

$$
r_{i,t} = \sum_{k=1}^{K} B_{ik} f_{k,t} + u_{i,t}, \qquad \Sigma = B F B^\top + D
$$

where:
- $B_{ik}$ is stock $i$'s exposure (loading) to factor $k$ (dimensionless, or a standardised score);
- $f_{k,t}$ is factor $k$'s return on day $t$ (fraction per day);
- $u_{i,t}$ is the specific (idiosyncratic) return, assumed uncorrelated across stocks and with the factors;
- $F = \operatorname{Cov}(f)$ is $K \times K$, daily;
- $D = \operatorname{diag}(\sigma^2_{u,1}, \dots, \sigma^2_{u,N})$, daily.

The covariance of stocks $i$ and $j$ is then $\sum_{k,l} B_{ik}F_{kl}B_{jl}$. Two stocks co-move only because they share factors. This structure fixes all three problems:

| Estimator | Parameters estimated from returns ($N = 1000$) | Rank |
|---|---|---|
| Sample covariance | $N(N+1)/2 = 500{,}500$ | $\le T$ (singular if $N > T$) |
| Fundamental factor model, $K = 59$ (1 market + 48 industries + 10 styles) | $K(K+1)/2 + N = 1{,}770 + 1{,}000 = 2{,}770$ ($B$ comes from characteristics, not from return covariances) | full ($D > 0$) |
| Statistical factor model, $K = 5$ | $NK + K(K+1)/2 + N \approx 6{,}000$ | full |

$BFB^\top + D$ is positive definite whenever $F \succeq 0$ and every $\sigma^2_{u,i} > 0$. No portfolio has zero forecast risk, so the optimizer can't find a free lunch in the noise.

**Why this choice.** Every serious equity risk system uses a structured estimator. The only question is which structure (§2). Shrinkage, the best unstructured fix, is covered in §7.

**References**
- [Menchero, Orr & Wang (2011), The Barra US Equity Model (USE4) Methodology Notes](https://www.top1000funds.com/wp-content/uploads/2011/09/USE4_Methodology_Notes_August_2011.pdf)
- Marchenko & Pastur (1967), *Distribution of eigenvalues for some sets of random matrices*, Mathematics of the USSR-Sbornik.

## 2. Types of factor models

**What it is.** There are three ways to choose the factors, and they differ in what is observed and what is estimated:

| Type | Observed | Estimated | Examples |
|---|---|---|---|
| **Fundamental** (cross-sectional) | exposures $B$ from stock characteristics (industry, size, momentum, …) | factor returns $f_t$, by a regression across stocks each day | MSCI Barra USE4, Axioma fundamental models |
| **Statistical** | nothing | both $B$ and $f_t$, by PCA of the return panel | APCA (Connor–Korajczyk), Axioma statistical models |
| **Macroeconomic** (time-series) | factor returns $f_t$ (market excess return, rate changes, oil, credit spreads) | exposures $B$, by a time-series regression of each stock on the factors | Chen–Roll–Ross, Fama–French time-series regressions |

**Pros and cons.**

| Type | Strengths | Weaknesses |
|---|---|---|
| Fundamental | Interpretable factors (you can say "we are long momentum"). Exposures update daily as characteristics change. Handles new listings immediately. Gives the optimizer named constraints (beta-neutral, industry bands). | Can only price factors you thought of. A common driver with no observable characteristic stays in the residuals as "specific" risk, which is wrong. |
| Statistical | Finds whatever common structure is in the data, including transient factors (a meme-stock cluster, a crowded-trade unwind). Needs only returns. | Factors have no names. They are identified only up to rotation and sign and can flip between refits. Needs a long history per stock. Fits noise if $K$ is too large. |
| Macro | Directly answers "what happens if rates jump". | Macro series are low-frequency and noisy. Time-series betas are estimated with error and are slow to change. Explanatory power for daily stock returns is weak. |

**Why we need it here, and why this choice.** We run a **fundamental model as the primary** and a **statistical model on its residuals as a companion**. This follows the plan (`docs/BUILD_PLAN.md` §3). Axioma calls fundamental models "by far the dominant class" and recommends a statistical model at the same horizon alongside it to catch transient factors (SimCorp/Axioma). Our own synthetic-market shakedown showed what is at stake. Books aligned with a latent factor that had no observable footprint realised 1.3–1.8× their ex-ante risk (`docs/REVIEW_NOTES.md`, "Integration findings"; `qts/risk/statistical.py` docstring). The fundamental model gives the optimizer named handles. The statistical companion covers its blind spots (§8). A macro model would add little at a daily horizon and is not planned.

**References**
- [SimCorp/Axioma, Statistical versus fundamental models](https://www.simcorp.com/resources/insights/white-papers-and-reports/statistical-versus-fundamental-models)
- [Menchero, Orr & Wang (2011), USE4 Methodology Notes](https://www.top1000funds.com/wp-content/uploads/2011/09/USE4_Methodology_Notes_August_2011.pdf)

## 3. Exposures

**What it is.** The exposure matrix $B_t$ (tickers × factors) describes each stock on date $t$ by three blocks of columns, in this order (`qts/risk/exposures.py::exposure_history`):

1. **`market`**: USE4's *Country* factor. Every stock has exposure 1.
2. **Industry dummies**: one 0/1 column per industry label in `MarketData.industry`, and each row sums to 1. On real data the labels are the Fama–French 48 industries mapped from SIC codes (`qts/data/loaders.py`). The synthetic generator uses 12 (`IND00`–`IND11`).
3. **Style factors**: the names in `RiskConfig.styles`, each standardised as described below.

A stock enters $B_t$ on date $t$ if it has a finite close and a positive market cap on $t$.

**Why we need it here.** Exposures are the "observed" half of a fundamental model. If they are wrong or contain look-ahead, the factor returns, the covariance and every residual-based alpha label inherit the error.

**How it works: the styles.** Each raw descriptor is computed on wide date × ticker panels using rolling windows and lags only (`raw_style_panels`), so the slice at date $t$ uses data $\le t$:

| Style | Raw descriptor | Window / constants | Intuition |
|---|---|---|---|
| `beta` | slope of a rolling regression of the stock's return on the **equal-weighted** market return, shrunk toward 1: $\beta = 1 + (1 - \tfrac13)(\hat\beta - 1)$ | 252 days (`BETA_WINDOW`), min 63 obs, `BETA_SHRINK = 1/3` | sensitivity to the market. The shrinkage (Blume/Vasicek style) pulls noisy extremes toward the cross-sectional norm |
| `size` | $\log(\text{market cap})$ | — | large vs small companies |
| `momentum` | $P_{t-21}/P_{t-252} - 1$ (12-1 month return) | `MOMENTUM_LONG = 252`, `MOMENTUM_SKIP = 21` | past-year winners vs losers, skipping the last month, which belongs to reversal |
| `value` | $\log(\text{book-to-price})$, or earnings yield if B/P is missing | — | cheap vs expensive |
| `residual_vol` | $\sqrt{\operatorname{var}(r) - \operatorname{cov}(r,m)^2/\operatorname{var}(m)}$, the residual volatility from the beta regression | 252 days | idiosyncratically volatile stocks co-move with each other |
| `liquidity` | $\log$ of the 63-day mean of dollar volume / market cap (share turnover) | 63 days, min 21 | heavily traded vs thinly traded |
| `short_term_reversal` | $P_t / P_{t-21} - 1$ | 21 days | last month's winners/losers. A big short-horizon risk factor for a daily book |
| `quality` | gross profitability | — | profitable vs unprofitable |
| `investment` | $-$asset growth | — | conservative investors load positively |
| `short_interest` | short interest / shares outstanding | — | heavily shorted names move together (squeezes) |

Several styles (momentum, value, quality, investment, short interest) are also alpha themes. That is deliberate. The plan says to include the alpha themes as risk styles so the optimizer sees their risk (`docs/BUILD_PLAN.md` §3). For the separate question of which styles alpha labels are residualised on, see **Alpha model › Labels**. There a "risk-only" subset is used, after an audit found that residualising on alpha themes removed the signal (`docs/audit/RESOLUTION.md`, CRITICAL row 2).

**How it works: Barra standardisation.** Raw descriptors have arbitrary units: log dollars, percentages, ratios. Each one is turned into a z-score row by row (per date) in `standardize_style`:

$$
x_{i} = \frac{d_i - \mu^{\text{cap}}}{s^{\text{eq}}}, \qquad \mu^{\text{cap}} = \frac{\sum_j c_j d_j}{\sum_j c_j}, \qquad s^{\text{eq}} = \sqrt{\frac{1}{n}\sum_j (d_j - \mu^{\text{cap}})^2}
$$

where:
- $d_i$ is stock $i$'s raw descriptor on the date;
- $c_j$ is stock $j$'s market cap (the weight in the mean);
- $s^{\text{eq}}$ is the **equal-weighted** standard deviation around the cap-weighted mean (`qts/core/crosssection.py::zscore`);
- $n$ is the number of stocks with a valid descriptor.

The steps are: z-score, then clip to $[-3, 3]$ (`WINSOR_SD`), then z-score again so the sd is about 1 after trimming, then set missing values to 0.

<callout icon="💡" color="blue_bg">
Why a cap-weighted mean? Because then the cap-weighted market portfolio $w^m_i = c_i/\sum_j c_j$ has zero exposure to every style: $\sum_i w^m_i x_i = 0$ by construction. "The market" is style-neutral, so a long-only index fund's risk sits in the `market` factor and not spread across ten styles. A "+1 momentum" exposure then means "momentum relative to the market". Why an equal-weighted sd? A cap-weighted sd would be dominated by a handful of mega-caps. Equal weighting makes one unit of exposure equal to one cross-sectional standard deviation for a typical stock. This is the USE4 convention.
</callout>

Setting a missing style to 0 treats the stock as average (cap-weighted mean) on that style, which is also the USE4 convention. Industry dummies and the market column are never standardised.

**Why this choice.** USE4 uses a Country factor, about 60 GICS industries and 12 styles (Beta, Non-linear Beta, Liquidity, Residual Volatility, Momentum, Size, Non-linear Size, Book-to-Price, Growth, Leverage, Dividend Yield, Earnings Yield), standardised as above (`docs/BUILD_PLAN.md` §3). For a daily book, MSCI's Trading model adds short-horizon styles: short-term reversal, 1-day reversal, short interest. Our list keeps the USE4 core that public data supports, adds short-term reversal and short interest from the trading-model list, and adds the alpha themes quality and investment.

Differences from the plan, all noted in the code or the audit:
- Non-linear beta, non-linear size, growth, leverage and dividend yield are not implemented.
- The plan's 1-day reversal is not implemented (→ Roadmap).
- Beta is regressed on the equal-weighted market. USE4 uses the cap-weighted market (`docs/audit/risk_portfolio.md` finding 9). Either is defensible. Equal weighting stops two mega-caps from defining "the market" in a concentrated universe.

**In the code.**
- `qts/risk/exposures.py::exposure_history(md, dates, cfg)` computes every rolling statistic once on the wide panels, stacks them, and slices per date. This is the fast path `BarraStyleRiskModel.fit` uses.
- `build_exposures(md, date, cfg)` gives one date by truncating `md` at `date` first.
- `split_factor_columns(columns, cfg)` returns `(market, industries, styles)`.
- To add a style, implement it in `raw_style_panels`, add the name to `SUPPORTED_STYLES`, and add it to `RiskConfig.styles`.
- Window lengths are module constants, not config fields.

**Pitfalls & checks.**
- *Look-ahead.* Fundamentals (`book_to_price`, `asset_growth`, …) must be point-in-time at the data layer. A restated value joined by report period leaks the future. An audit caught this for Sharadar asset growth (`docs/audit/RESOLUTION.md`, HIGH row "`merge_asof` on filing date").
  - Checks: `tests/test_risk.py::test_exposure_history_matches_single_date_and_is_point_in_time` rebuilds exposures on data truncated at $t$ and requires identical output. `test_point_in_time_as_of` does the same for the whole model.
- *Standardisation drift.* `test_exposures_shapes_and_standardization` checks that on a test date every style has a cap-weighted mean within 0.05 of 0, an equal-weighted sd in $(0.85, 1.1)$ and $|x| \le 3.5$, that the market column is all ones, and that industry rows sum to 1.
- *Signal sanity.* `test_beta_exposure_tracks_true_beta` requires correlation > 0.6 with the synthetic generator's true betas.
- *Signs don't matter for risk.* Flipping a style's sign flips its factor return, and $BFB^\top$ is unchanged. Signs matter only for interpretation and constraints.

**References**
- [Menchero, Orr & Wang (2011), USE4 Methodology Notes](https://www.top1000funds.com/wp-content/uploads/2011/09/USE4_Methodology_Notes_August_2011.pdf)
- [MSCI, Barra Market Equity Models factsheet](https://www.msci.com/documents/1296102/1636401/MSCI_Barra_Market+Equity+Models_Factsheet+.pdf)

## 4. Factor returns: the cross-sectional regression

**What it is.** Each day we know every stock's exposures (from yesterday) and every stock's return (today). We ask which set of factor returns best explains today's cross-section of returns. That is one linear regression per day, with ~1,000 observations (stocks) and $K$ unknowns (factor returns). The slope coefficients *are* the factor returns. The residuals are the specific returns.

**Why we need it here.** A fundamental model observes $B$ but not $f$. The regression turns the characteristic-based exposures into a daily time series of factor returns, which is what the factor covariance (§5) is estimated from. The residuals feed specific risk (§6), the statistical companion (§8) and the alpha labels.

**How it works.** For each day $t$ (`qts/risk/factor_returns.py::estimate_factor_returns`):

$$
\min_{f_t} \sum_i w_i \left(r_{i,t} - \sum_k B_{ik,t-1} f_{k,t}\right)^2 \quad \text{s.t.} \quad \sum_{j \in \text{ind}} c_j f_{j,t} = 0
$$

where:
- $r_{i,t}$ is stock $i$'s simple return on day $t$;
- $B_{t-1}$ are the exposures **built from data up to $t-1$**, so they were known before the return was realised;
- $w_i \propto \sqrt{\text{cap}_{i,t-1}}$ are the regression weights, normalised to sum to 1;
- $c_j = \text{(total cap of industry } j) / \text{(total cap)}$ at $t-1$ are the constraint weights;
- $f_t$ is the vector of factor returns (fraction per day).

The specific return is $u_{i,t} = r_{i,t} - B_{i,t-1}^\top f_t$. The regression $R^2$ is weighted: $R^2 = 1 - \sum_i w_i u_i^2 / \sum_i w_i (r_i - \bar r_w)^2$, with $\bar r_w = \sum_i w_i r_i$.

**Why $\sqrt{\text{cap}}$ weights.** Ordinary least squares assumes every observation is equally noisy. Stock returns are not: small caps have much larger idiosyncratic variance than large caps. Empirically, specific variance falls roughly like $1/\sqrt{\text{cap}}$. Generalised least squares weights each observation by the inverse of its noise variance, so $w_i \propto 1/\sigma^2_{u,i} \propto \sqrt{\text{cap}_i}$. That gives more efficient factor-return estimates than OLS. It also sits between equal weights (which let microcaps dominate) and cap weights (which let five mega-caps dominate).

**The identification problem.** The market column is a vector of ones, and every stock is in exactly one industry, so the industry dummies also sum to a vector of ones:

$$
B_{\cdot,\text{market}} = \sum_{j \in \text{ind}} B_{\cdot,j}
$$

The columns of $B$ are linearly dependent, $B^\top W B$ is singular, and the regression has infinitely many solutions. You can add any constant $a$ to every industry return and subtract it from the market return, and the fitted values don't change. Some normalisation must pick one solution. USE4 requires the **cap-weighted industry returns to sum to zero**, $\sum_j c_j f_j = 0$. Industry returns are then *relative* to the market, and the market factor captures the overall level.

<callout icon="💡" color="blue_bg">
An analogy: you can measure how high each person's head is above the floor, but they are standing on a stage of unknown height. "Stage + person" is all you observe, so you must pin one thing down, for example by defining the people's heights to average zero. The constraint does that for the industries.
</callout>

**Null-space reparameterisation.** The constraint $c^\top f = 0$ is imposed exactly by solving the problem in a lower-dimensional space (`_constraint_nullspace`, `restricted_wls`):
1. Pick a pivot industry $p$ with the largest $|c_p|$ (the largest-cap industry).
2. Build $R$ ($K \times (K-1)$). Every factor except $p$ is free. The pivot is determined by the others: $f_p = -\sum_{k \ne p} (c_k/c_p) f_k$.
3. Write $f = R g$. Any $g$ then satisfies $c^\top R g = 0$.
4. Solve the unconstrained WLS of $r$ on $BR$ for $g$, and map back $f = R g$.

```python
def restricted_wls(y: np.ndarray, X: np.ndarray, w: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Minimise ``sum w_i (y_i - x_i'f)^2`` subject to ``c'f = 0``."""
    R = _constraint_nullspace(c)
    sw = np.sqrt(w)[:, None]
    XR = X @ R
    g, *_ = np.linalg.lstsq(XR * sw, y * sw[:, 0], rcond=None)
    return R @ g
```

$c$ is non-zero only on industry columns, so the market and style returns are unconstrained. Industries with no members on the day are dropped before the solve so that $X$ keeps full rank, and their return is reported as 0.0.

**Interpretation of the market factor.** The module docstring says the market factor "equals the cap-weighted market return". That is exact only when the regression weights, the constraint weights and the style-standardisation weights coincide. Here they are √cap, cap and cap. The audit measured 0.00492 for the market factor against 0.00461 for the cap-weighted market on a test day (`docs/audit/risk_portfolio.md` finding 9). The test requires correlation > 0.98, not equality. Fitted systematic returns $Bf$ do not depend on the identification, so risk forecasts are unaffected.

**Why this choice.** WLS with √cap weights and a cap-weighted sum-to-zero constraint is the USE4 recipe (`docs/BUILD_PLAN.md` §3 table, "Regression weights"). The plan notes this layer is "a solved engineering problem" (open-source toraniko reproduces Barra-like factor returns). Alternatives exposed as options:
- `constraint_weights="regression"` makes industries sum to zero under the √cap weights instead, so the market factor tracks a √cap-weighted market. It is less dominated by mega-caps in a concentrated universe.
- `weight_cap_quantile` winsorises caps before forming the weights (Axioma-style mega-cap damping).

Both default to pure USE4. Dropping an industry instead of constraining (the usual dummy-variable trick) would make one arbitrary industry the baseline, and the other industries' returns would change meaning whenever membership changes.

**In the code.**
- `estimate_factor_returns(returns_t, exposures_tminus1, market_cap_tminus1, cfg, constraint_weights="cap", weight_cap_quantile=None)` runs one day and returns a `CrossSectionalFit(factor_returns, specific_returns, r2)`.
- `estimate_factor_return_history(md, exposures_by_date, cfg, ...)` runs every day $t$ whose previous trading day has exposures. It pairs $B_{t-1}$ and $\text{cap}_{t-1}$ with $r_t$.
- `BarraStyleRiskModel(constraint_weights=..., weight_cap_quantile=...)` forwards the options.

**Pitfalls & checks.**
- *Look-ahead.* Using $B_t$ with $r_t$ would let today's return leak into today's exposures through momentum, reversal, beta and size, which all contain $P_t$. The pairing $(B_{t-1}, r_t)$ is verified by the audit ("Verified as correct") and by the point-in-time tests.
- *Collinearity beyond the market/industry identity.* Two styles that are almost the same (for example size and liquidity in some universes) inflate each other's factor-return variance. They don't break the fit, because `lstsq` is rank-revealing, but they make $F$ noisy. Watch the condition number of $B^\top W B$ when adding styles.
- *Constraint check.* `cap_weighted_industry_sum` is a diagnostic. `tests/test_risk.py::test_factor_returns_market_and_constraint` requires $|\sum_j c_j f_j| < 10^{-10}$ on sampled days, mean $R^2$ in $(0.05, 0.8)$ and every daily $R^2$ in $(0, 1)$.
- `test_single_day_regression_returns_specific_returns` checks the identity $r = Bf + u$ exactly.

**References**
- [Menchero, Orr & Wang (2011), USE4 Methodology Notes](https://www.top1000funds.com/wp-content/uploads/2011/09/USE4_Methodology_Notes_August_2011.pdf)
- [toraniko (fork)](https://github.com/Pleometric/toraniko)

## 5. Factor covariance

**What it is.** With a daily history of $K$ factor returns, $F$ is "just" their covariance matrix. $K \approx 60$ is small enough that the sample-size problem of §1 is mild. The work goes into making $F$ *responsive* (recent data matters more), *correct at the forecast horizon* (daily returns are slightly autocorrelated) and *unbiased for optimised portfolios*. Those are the jobs of the four sub-steps below. `qts/risk/covariance.py::factor_covariance` runs them in this order:
1. EWMA + Newey–West with the volatility half-life, giving variances.
2. EWMA + Newey–West with the correlation half-life, giving correlations.
3. Combine the two: $F_0 = \operatorname{diag}(\sigma)\, C\, \operatorname{diag}(\sigma)$.
4. Eigenfactor risk adjustment.
5. Volatility regime adjustment: multiply by $\lambda_F^2$.

### Exponentially weighted covariance (EWMA)

**What it is.** A weighted covariance in which yesterday counts most and older days fade geometrically. It is like a moving average that never fully forgets but steadily discounts.

**How it works.**

$$
\hat\Sigma = \sum_{t=1}^{T} w_t (x_t - \mu)(x_t - \mu)^\top, \qquad w_t = \frac{\lambda^{T-t}}{\sum_{s=1}^T \lambda^{T-s}}, \qquad \lambda = 0.5^{1/h}
$$

where:
- $x_t \in \mathbb{R}^K$ is the factor-return vector on day $t$ (day $T$ is the most recent);
- $\mu = \sum_t w_t x_t$ is the weighted mean;
- $h$ is the half-life in trading days: an observation $h$ days old gets half the weight of today's;
- $\lambda$ is the daily decay factor.

The half-life and decay are two ways of stating the same thing. $h = 84$ gives $\lambda = 0.99178$ (each day keeps 99.18% of the previous day's weight). $h = 504$ gives $\lambda = 0.99863$. $h = 42$ gives $\lambda = 0.98363$. The "effective sample size" of an EWMA is roughly $(1+\lambda)/(1-\lambda) \approx 2.9h$ days.

### Separate volatility and correlation half-lives

Volatilities change fast. Market volatility can double in a week. Correlations are harder to estimate and change more slowly. A $K \times K$ correlation matrix with a short half-life has a small effective sample and comes out noisy and nearly singular. USE4 therefore estimates:
- **variances** with $h_{\text{vol}} = 84$ days (`RiskConfig.vol_half_life`), so they react quickly;
- **correlations** with $h_{\text{corr}} = 504$ days (`corr_half_life`), so they are stable;

and combines them (`combine_vol_corr`):

$$
F_0 = \operatorname{diag}(\sigma)\, C\, \operatorname{diag}(\sigma), \qquad \sigma_k = \sqrt{\hat\Sigma^{(84)}_{kk}}, \qquad C = \text{corr}(\hat\Sigma^{(504)})
$$

The result is projected to the nearest positive semi-definite (PSD) matrix by clipping negative eigenvalues (`nearest_psd`).

### Newey–West adjustment for autocorrelation

**Why.** If daily returns were independent, the variance of an $H$-day return would be $H$ times the daily variance. Daily factor returns are slightly autocorrelated: reversal factors are negatively autocorrelated, trend-following and illiquid factors positively. For a series with lag-$l$ autocorrelation $\rho_l$:

$$
\operatorname{Var}\left(\sum_{s=1}^{H} x_s\right) = H\sigma^2\left(1 + 2\sum_{l=1}^{H-1}\left(1 - \tfrac{l}{H}\right)\rho_l\right)
$$

With positive $\rho$, scaling daily variance by $H$ underestimates multi-day risk. With negative $\rho$, it overestimates. The optimizer holds positions for many days (the alpha horizon is 20 days and the cost-implied holding period is longer), so the variance that matters is the per-day variance of multi-day sums.

**How it works.** Newey–West adds weighted autocovariances (`ewma_cov`):

$$
\hat\Sigma_{\text{NW}} = \Gamma_0 + \sum_{l=1}^{L}\left(1 - \frac{l}{L+1}\right)\left(\Gamma_l + \Gamma_l^\top\right), \qquad \Gamma_l = \sum_t w_t (x_t - \mu)(x_{t-l} - \mu)^\top
$$

where:
- $\Gamma_l$ is the EWMA-weighted lag-$l$ autocovariance matrix;
- $L$ is the number of lags: 5 for the volatility estimate (`newey_west_lags_vol`) and 2 for the correlation estimate (`newey_west_lags_corr`);
- $1 - l/(L+1)$ are the **Bartlett weights**. They taper the higher lags, and they are what makes the plain Newey–West estimator PSD. Combined with EWMA weights it is PSD only asymptotically, so the code clips eigenvalues as a safeguard.

The output is still a **daily** covariance, but one that already includes the serial correlation.

```python
S = (A * w[:, None]).T @ A
L = int(newey_west_lags)
for l in range(1, min(L, T - 1) + 1):
    G = (A[l:] * w[l:, None]).T @ A[:-l]
    S = S + (1.0 - l / (L + 1.0)) * (G + G.T)
S = (S + S.T) / 2.0
return nearest_psd(S) if ensure_psd else S
```

### Eigenfactor risk adjustment

**Why.** §1 showed that sample eigenvalues are too dispersed. The smallest ones are biased down. An optimizer is a machine for finding low-variance directions, so it concentrates in the eigen-directions whose risk is most underestimated. USE4 documents this directly: the bias statistic of eigenfactor portfolios (§10) rises well above 1 for the smallest eigenvalues, even when the plain factor portfolios look fine. With $K \approx 60$ and a 504-day correlation half-life the effect is modest but real.

**How it works.** A Monte-Carlo estimate of how much the estimator itself distorts each eigenvalue, then an undo (`eigenfactor_bias`, `apply_eigen_adjustment`):

1. Diagonalise $F_0 = U D U^\top$. The columns of $U$ are *eigenfactors*: uncorrelated portfolios of factors with variances $D_k$.
2. Treat $F_0$ as the truth. For $m = 1, \dots, M$ ($M = 200$, `n_eigen_sims`):
   - simulate $T$ days of eigenfactor returns $b_t \sim N(0, D)$ and rotate them to factor space, $f_t = U b_t$;
   - re-estimate the covariance with the **same** vol/corr EWMA estimator, giving $F_m$;
   - diagonalise $F_m = U_m D_m U_m^\top$;
   - compute the *true* variance of each simulated eigenfactor, $\tilde d_{mk} = (U_m^\top F_0 U_m)_{kk}$.
3. The bias of eigenfactor $k$ is $v_k^2 = \frac{1}{M}\sum_m \tilde d_{mk} / D_{mk}$. This is the ratio of true to estimated variance, and it is greater than 1 for the small eigenvalues.
4. Adjust: $F_{\text{eig}} = U \operatorname{diag}(v_k^2 D_k) U^\top$.

where:
- $T_{\text{sim}} = \min(T, 4 h_{\text{corr}}) = \min(T, 2016)$ days;
- $v_k$ is clipped to $[0.5, 2]$;
- the simulation estimator uses no Newey–West lags and no demeaning, for speed.

<callout icon="🤔" color="purple_bg">
Why not shrink $F$ with Ledoit–Wolf instead? Shrinkage pulls every eigenvalue toward a common target based on a statistical loss function. The eigenfactor adjustment measures the bias of *our own* estimator (EWMA with these half-lives on this spectrum) and undoes exactly that. It is also the documented USE4 step, with published evidence that it fixes optimised-portfolio bias. Shrinkage is used where there is no structure to exploit (§7).
</callout>

**Simplifications versus USE4.** USE4 also fits a parabola to $v_k$ across $k$ and inflates the result by an empirical factor $a = 1.2$, because the simulation under-corrects. Our code returns the raw simulated $v_k$. On a 20-factor test the audit found the raw bias of the five smallest eigenvariances was 1.1–1.4, while the estimated $v$ was 1.02–1.05. The adjustment points in the right direction but corrects too little (`docs/audit/risk_portfolio.md` finding 11). Adding $v \leftarrow 1 + a(v-1)$ with $a = 1.2$ is on the Roadmap.

**Speed.** The Monte-Carlo is the only expensive step in `as_of`. `BarraStyleRiskModel._eigen_bias` caches $v$, keyed on `len(history) // eigen_refresh_days` (default 21 observations). It is recomputed about once a month, because the bias ratios depend on the estimator and the slowly drifting spectrum, not on daily noise.

### Volatility regime adjustment (VRA)

**Why.** An 84-day EWMA needs weeks to catch up when volatility jumps (March 2020) and weeks to relax afterwards. In the meantime every portfolio's risk is mis-stated in the same direction. VRA measures that common bias from recent forecast errors and rescales the whole matrix.

**How it works** (`factor_vra_multiplier`):

$$
B_t^F = \sqrt{\frac{1}{K_t}\sum_k \left(\frac{f_{k,t}}{\sigma_{k,t-1}}\right)^2}, \qquad \lambda_F = \sqrt{\sum_t w^{\text{VRA}}_t \left(B_t^F\right)^2}, \qquad F \leftarrow \lambda_F^2 F
$$

where:
- $f_{k,t}/\sigma_{k,t-1}$ is factor $k$'s return on day $t$ standardised by the volatility forecast made the day before. Here $\sigma^2_{k,t-1}$ is an EWMA of $f_k^2$ with the 84-day vol half-life, shifted one day;
- $B_t^F$ is the **cross-sectional bias statistic**. It is about 1 if forecasts are right on average across factors that day, above 1 if the day was more volatile than forecast, below 1 if calmer;
- $K_t$ counts factors with a non-zero return that day. Columns that are identically zero, and exact-zero days (an empty industry), are excluded, so they don't pull $B_t$ down (audit finding 11, fixed);
- $w^{\text{VRA}}$ are EWMA weights with half-life 42 days (`vra_half_life`);
- with fewer than 20 usable days, $\lambda_F = 1$.

The VRA half-life (42) is shorter than the vol half-life (84). The point is to react to a regime shift faster than the base estimator does. In audit tests (`docs/audit/risk_portfolio.md` finding 11), constant volatility gave $\lambda_F = 1.03$. Doubling volatility over the last 60 days gave 1.42. Halving it gave 0.77.

**Why this order.** Eigen-adjust first, VRA second, so the eigen bias is measured on the shape of the matrix and VRA then sets its overall level. That is the order in the plan.

**Why this choice (factor covariance overall).** Every parameter here is the USE4 short-horizon recipe in `docs/BUILD_PLAN.md` §3. The plan's long-horizon variant (252-day vol, 168-day VRA) is not a preset; set the fields by hand. DCC-GARCH is on the Roadmap.

**In the code.** `factor_covariance(factor_returns, cfg, eigen_bias=None, seed=0, return_details=False)`. With `return_details=True` it returns a `FactorCovarianceResult(cov, vra_multiplier, eigen_bias, raw_cov)`, which is useful for monitoring $\lambda_F$ and $v$. Building blocks: `ewma_cov`, `combine_vol_corr`, `eigenfactor_bias`, `apply_eigen_adjustment`, `factor_vra_multiplier`, `half_life_to_lambda`.

**Pitfalls & checks.**
- *Units.* Everything is daily. An annual vol of the market factor in $(5\%, 50\%)$ is asserted in `tests/test_risk.py::test_factor_covariance_psd`, which also checks that $F$ is PSD with and without the eigen adjustment.
- `test_ewma_cov_psd_and_scale` checks that with an effectively infinite half-life and no lags `ewma_cov` reproduces `np.cov`.
- The audit verified `ewma_cov` against a brute-force loop to $2 \times 10^{-16}$.
- *Look-ahead.* `as_of(date)` uses factor returns `.loc[:date]`: the return realised at the close of `date`, which is known by then.
- *VRA overreaction.* A single extreme day moves $\lambda_F$ through a 42-day EWMA of $B_t^2$. Watch `vra_multiplier` in `FactorCovarianceResult`. Persistent values far from 1 mean the base half-life is wrong for the regime.
- `test_vra_ignores_identically_zero_and_empty_industry_days` guards the empty-industry fix.

**References**
- [Menchero, Orr & Wang (2011), USE4 Methodology Notes](https://www.top1000funds.com/wp-content/uploads/2011/09/USE4_Methodology_Notes_August_2011.pdf)
- [Newey & West (1987), A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix](https://doi.org/10.2307/1913610)

## 6. Specific risk

**What it is.** The diagonal $D$: each stock's idiosyncratic variance, the part of its return no factor explains. In a diversified long/short book the factor exposures largely net out, and specific risk is often half or more of total risk. In the worked example below it is just over half.

**Why we need it here.** $D$ sets how much the optimizer is willing to put in any single name. If one stock's specific risk is underestimated, the optimizer overweights it. If it is overestimated, a good alpha is wasted. Stocks with short or erratic histories (recent IPOs, names returning from a halt, names with one huge earnings jump) are where naive estimates go wrong.

**How it works.** Four steps (`qts/risk/covariance.py::specific_risk`), on the last $\max(6 \times 84, 252) = 504$ days of specific returns $u_{i,t}$ from §4:

**Step 1: time-series estimate.** EWMA variance of each stock's specific returns with half-life 84 (`specific_half_life`), plus Newey–West on its own autocovariances with 5 lags (`ewma_nw_variance`). Weights are renormalised over the days the stock was actually observed. The NW-adjusted variance is floored at 25% of the lag-0 variance so that a strongly negative autocorrelation cannot drive it to zero.

**Step 2: structural model for thin histories.** For a stock with few observations or a fat-tailed history the time-series estimate is unreliable. USE4 blends it with a *structural* forecast from the stock's characteristics:

$$
\sigma_i = \gamma_i\, \sigma_i^{\text{TS}} + (1 - \gamma_i)\, \sigma_i^{\text{STR}}, \qquad \gamma_i = \operatorname{clip}\!\left(\frac{h_i - 60}{120}, 0, 1\right)\cdot \min\!\left(1, e^{1 - Z_i}\right)
$$

where:
- $h_i$ is the number of observed days in the window. Below 60 the stock gets no time-series weight, and from 180 it can get full weight;
- $Z_i = \left|\sigma_i^{\text{eq}} / \sigma_i^{\text{rob}} - 1\right|$ compares the plain sd with a robust sd, $\sigma^{\text{rob}} = \text{IQR}/1.35$ (for a normal distribution the interquartile range is $1.35\sigma$). For a well-behaved stock $Z \approx 0$. For a stock with a few huge outliers $\sigma^{\text{eq}} \gg \sigma^{\text{rob}}$, so $Z$ is large and the time-series weight drops;
- $\sigma_i^{\text{STR}}$ is the structural model (`structural_specific_vol`): regress $\ln \sigma^{\text{TS}}$ on the stock's factor exposures by WLS with √cap weights, using only stocks with $\gamma = 1$. Then $\sigma^{\text{STR}}_i = E_0 \exp(B_i^\top b)$, where $E_0 = \sum w\sigma^{\text{TS}} / \sum w \exp(Bb)$ corrects the downward bias of exponentiating a log regression.

The intuition: a 3-week-old IPO in a high-volatility industry, small and illiquid, probably has a specific risk like other stocks with those characteristics.

**Step 3: Bayesian shrinkage to size-decile means.** Sort stocks into 10 size deciles. Within each decile (`bayesian_shrink_specific`):

$$
\sigma_i^{\text{SH}} = v_i\, \bar\sigma_{s(i)} + (1 - v_i)\, \sigma_i, \qquad v_i = \frac{q\,\lvert \sigma_i - \bar\sigma_{s(i)} \rvert}{\Delta_{s(i)} + q\,\lvert \sigma_i - \bar\sigma_{s(i)} \rvert}
$$

where:
- $s(i)$ is stock $i$'s size decile;
- $\bar\sigma_s$ is the **cap-weighted** mean of $\sigma$ in the decile;
- $\Delta_s$ is the **equal-weighted** standard deviation of $\sigma$ in the decile;
- $q = 0.1$ (`specific_shrink_q`) sets the shrinkage strength;
- $v_i \in [0, 1)$ is the shrinkage intensity.

<callout icon="💡" color="blue_bg">
Why shrink extremes more? $v_i$ grows with the distance from the decile mean. A stock whose estimated risk sits right at its peer average is barely touched. One whose estimate is three decile-sds away is pulled back noticeably. The most extreme estimates are the ones most likely to contain estimation noise: being at the top of a noisy ranking is partly bad luck in the sample. This is regression to the mean, and it is the same reason a batting average after ten at-bats should be shrunk toward the league average. Example: $\sigma = 3\%$, $\bar\sigma = 1.5\%$, $\Delta = 0.6\%$ gives $v = 0.1 \cdot 1.5 / (0.6 + 0.15) = 0.2$ and $\sigma^{\text{SH}} = 0.2 \cdot 1.5\% + 0.8 \cdot 3\% = 2.7\%$.
</callout>

**Step 4: specific VRA.** The same idea as factor VRA, applied to stocks with cap weights (`specific_vra_multiplier`):

$$
\left(B_t^S\right)^2 = \sum_i \frac{c_i}{\sum_j c_j}\left(\frac{u_{i,t}}{\sigma_{i,t-1}}\right)^2, \qquad \lambda_S = \sqrt{\text{EWMA}_{42}\left((B_t^S)^2\right)}
$$

$\sigma_{i,t-1}$ is an 84-day EWMA of $u_i^2$ shifted one day. USE4 uses the structural forecasts there, and we use the EWMA forecast. The final output is the daily variance $D_{ii} = (\lambda_S\, \sigma_i^{\text{SH}})^2$.

**Why this choice.** Every step and constant is the USE4 specific-risk recipe listed in `docs/BUILD_PLAN.md` §3. The plan's alternative, a **full** residual covariance by nonlinear shrinkage (§7), is not used for $D$. A diagonal keeps the optimizer's factor-form risk $\|L^\top B^\top w\|^2 + \sum_i d_i w_i^2$ fast, and the statistical companion (§8) picks up the residual co-movement that matters.

**In the code.** `specific_risk(specific_returns, exposures_latest, market_cap, cfg, return_details=False)`. With `return_details=True` it also returns `sigma_ts`, `sigma_str`, `gamma`, `sigma_shrunk`, `vra_multiplier` and `n_obs`, which are useful to see why a particular name's risk looks odd. Tuning: `specific_half_life` (shorter reacts faster but is noisier) and `specific_shrink_q` (larger means more shrinkage). The `min(1, e^{1-Z})` blend and the 60/120 constants are not configurable.

**Pitfalls & checks.**
- *The audit bug.* The original blend used $1.35 \times$IQR and $Z = \sigma^{\text{eq}}/\sigma^{\text{rob}}$. The two errors partly cancelled, and fat-tailed names got about half the structural weight they should have. Both are fixed (`docs/audit/RESOLUTION.md`, MEDIUM row). `tests/test_risk.py::test_thin_history_blend_use4_formula` plants a 100σ outlier and checks $\gamma = \min(1, e^{1-Z})$ exactly and $\gamma < 0.6$, while a Gaussian name gets $\gamma \approx 1$.
- `test_specific_risk_pipeline` checks positivity, a median annualised specific vol in $(5\%, 80\%)$, $\lambda_S \in (0.5, 2)$, and that more than 90% of names get full time-series weight on a long synthetic history.
- *Out-of-model tickers.* `RiskModelOutput.covariance()` gives a ticker absent from the model zero exposure and the **median** specific variance, so it looks purely idiosyncratic to the optimizer. Callers should intersect universes first (`docs/audit/risk_portfolio.md` finding 7).

**References**
- [Menchero, Orr & Wang (2011), USE4 Methodology Notes](https://www.top1000funds.com/wp-content/uploads/2011/09/USE4_Methodology_Notes_August_2011.pdf)

## 7. Shrinkage estimators

**What it is.** Shrinkage blends a noisy, unbiased estimate with a stable, biased one. The sample covariance has low bias and huge variance. A structured target (say, "all stocks have the same variance and no correlation") has big bias and almost no variance. Somewhere between the two lies a lower total error. It is the matrix version of regression to the mean.

**Why we need it here.** Two uses:
1. Estimating a **full residual covariance** (`shrunk_residual_covariance`, `BarraStyleRiskModel.residual_covariance_as_of`). After removing the factors, the residuals still form an $N \times N$ problem with $N$ comparable to $T$. This is exactly the regime of §1.
2. Baselines in the **out-of-sample minimum-variance test** (§10), the benchmark any covariance estimator must beat.

`RiskModelOutput` itself keeps only the diagonal $D$. The shrunk residual matrix is used for evaluation and is a candidate for a future residual block (→ Roadmap).

**How it works: Ledoit–Wolf linear shrinkage.**

$$
\hat\Sigma_{\text{LW}} = \delta^*\, \mu I + (1 - \delta^*)\, S, \qquad \mu = \frac{\operatorname{tr}(S)}{N}
$$

where:
- $S$ is the sample covariance;
- $\mu I$ is the target: a scaled identity (every variance equal to the average, no correlations);
- $\delta^* \in [0, 1]$ is the shrinkage intensity that minimises the expected Frobenius error $\mathbb{E}\|\hat\Sigma - \Sigma\|_F^2$. Ledoit & Wolf (2004) give a closed-form consistent estimate of it from the data: roughly, the noisier $S$ is relative to its distance from the target, the larger $\delta^*$.

In eigenvalue terms, linear shrinkage keeps the eigenvectors of $S$ and moves every eigenvalue the same fraction of the way toward $\mu$: $\lambda_i \mapsto \delta^*\mu + (1-\delta^*)\lambda_i$. The result is always well-conditioned. We call scikit-learn's implementation (`linear_shrinkage`).

**How it works: nonlinear (quadratic-inverse) shrinkage, QIS.** Linear shrinkage uses one intensity for every eigenvalue. Random-matrix theory says the optimal correction depends on *where* an eigenvalue sits in the spectrum: eigenvalues in the noise bulk need heavy shrinkage, and isolated spikes (true factors) need little. Nonlinear shrinkage keeps the sample eigenvectors $u_i$ and replaces each sample eigenvalue $\lambda_i$ by its own $\delta_i$:

$$
\hat\Sigma_{\text{QIS}} = \sum_i \delta_i\, u_i u_i^\top, \qquad \delta_i = \left(\frac{(1-c)^2}{\lambda_i} + \frac{2c(1-c)\,\theta_i}{\lambda_i} + \frac{c^2\left(\theta_i^2 + \mathcal{H}\theta_i^2\right)}{\lambda_i}\right)^{-1}
$$

where:
- $c = N/T$ is the concentration ratio (with $T$ reduced by 1 after demeaning);
- $\theta_i$ and $\mathcal{H}\theta_i$ come from a kernel estimate of the limiting spectral density of the inverse eigenvalues and its Hilbert transform, with bandwidth $h = \min(c^2, 1/c^2)^{0.35} / N^{0.35}$;
- the formula above is the case $N \le T$. For $N > T$ the $N - T$ null eigenvalues get a common value $\delta_0 = 1/\big((c-1)\,\overline{1/\lambda}\big)$, and the rest follow a variant of the same formula;
- the $\delta_i$ are finally rescaled so $\sum_i \delta_i = \operatorname{tr}(S)$ (trace preserved).

What matters is that $\delta_i$ is a smooth, data-driven, *nonlinear* function of $\lambda_i$ that lifts small eigenvalues and lowers large ones more cleverly than a straight line. QIS is the asymptotically optimal rotation-equivariant estimator under a loss on the *inverse* covariance. That loss is what matters for portfolio weights, since $w \propto \Sigma^{-1}\alpha$.

<toggle heading="h3">
Where the QIS formula comes from (sketch)

For a rotation-equivariant estimator $\sum_i \delta_i u_i u_i^\top$, the best possible $\delta_i$ under Frobenius loss is the "oracle" $u_i^\top \Sigma u_i$: the true variance in the sample eigen-direction. Random-matrix theory (Ledoit & Péché) relates the oracle to the Stieltjes transform of the limiting spectral distribution of $S$. The transform can be estimated from the observed eigenvalues with a kernel. QIS applies this to the *inverse* eigenvalues, which yields the closed form above with a single $O(N^2)$ kernel evaluation and no numerical inversion. Hence "quadratic-inverse". The code is a line-by-line port of Ledoit & Wolf's MATLAB routine `QIS.m`.
</toggle>

**Why this choice.** Ledoit & Wolf's 2022 review (`docs/BUILD_PLAN.md` §3) finds that nonlinear shrinkage clearly beats linear once $N$ and $T$ both exceed about 50 and eigenvalues are dispersed. That describes any equity universe. The analytical formula scales to $N = 10{,}000+$. Their best overall estimator is a factor model with DCC-nonlinear-shrunk residuals (AFM-DCC-NL), which works about as well with only the market factor as with five Fama–French factors, as long as the market factor is included. Deep-learning covariance models have not displaced it. In a daily-rebalanced minimum-variance test DCC-NL had lower realised variance (0.002048) than a CNN-BiLSTM-attention model (0.002293) (Reis, Serra & Gama, out-of-sample). The audit verified our QIS port on simulated data with known $\Sigma$. Frobenius error was 7.4 vs 10.8 for the sample covariance ($N=100$, $T=200$), 10.2 vs 24.3 ($N=120$, $T=60$) and 15.2 vs 28.8 ($N=300$, $T=252$), with a positive minimum eigenvalue in every case (`docs/audit/risk_portfolio.md`, "Verified as correct"). These are synthetic-data checks, not market evidence.

**In the code.**
- `nonlinear_shrinkage(X, assume_centered=False)` takes a **data matrix** ($T \times N$), not a covariance.
- `linear_shrinkage(X)`.
- `shrunk_residual_covariance(specific_returns, cfg, window=None)` uses the last $\max(4 \times 84, 252) = 336$ days, sets missing residuals to 0, and picks QIS when `RiskConfig.use_nonlinear_shrinkage = True` (default) or Ledoit–Wolf otherwise.

**Pitfalls & checks.**
- *Passing a covariance instead of data* silently gives nonsense. Both functions expect observations × variables.
- *Zero-filled missing residuals* bias variances down for names with gaps. Restrict to names with good coverage before relying on the matrix.
- `tests/test_risk.py::test_nonlinear_shrinkage_beats_sample_cov` requires QIS error < 0.8× sample error, Ledoit–Wolf error < sample error, PSD, and trace preservation.
- `test_nonlinear_shrinkage_singular_case` checks $N > T$ gives a strictly positive-definite matrix.
- `test_min_variance_oos_test` requires QIS out-of-sample GMV variance ≤ the sample covariance's.

**References**
- Ledoit & Wolf (2004), *A well-conditioned estimator for large-dimensional covariance matrices*, Journal of Multivariate Analysis.
- [Ledoit & Wolf (2022), The Power of (Non-)Linear Shrinking: A Review and Guide to Covariance Matrix Estimation](https://www.econ.uzh.ch/dam/jcr:e946b1e3-35e8-4c4f-894f-5f4306bf28a5/jfec_2022.pdf)
- Ledoit & Wolf (2022), *Quadratic Shrinkage for Large Covariance Matrices*, Bernoulli.
- [Reis, Serra & Gama (2025), arXiv:2503.01581](https://arxiv.org/html/2503.01581v1)

## 8. Statistical companion (APCA) and the hybrid model

**What it is.** A second factor model that sees only returns. It runs principal component analysis on the **residuals** of the fundamental model and looks for common movement the fundamental factors missed. Its factors are appended to the fundamental ones. A dashboard alarm fires when the residuals contain a strong common factor.

**Why we need it here.** A fundamental model prices only the factors it can name. In the synthetic shakedown, books aligned with an unnamed latent factor ran at 1.3–1.8× their ex-ante risk (`docs/REVIEW_NOTES.md`). In real markets the equivalents are things like a crowded-trade unwind, a theme ("AI beneficiaries") that cuts across industries, or a new macro sensitivity. The plan requires a statistical model at the same horizon "to catch transient factors" (`docs/BUILD_PLAN.md` §3).

### PCA refresher

Given a demeaned $T \times N$ panel $R$ (rows are days, columns are stocks), the sample covariance $S = R^\top R / T$ has the eigen-decomposition $S = W \Lambda W^\top$, with eigenvalues $\lambda_1 \ge \lambda_2 \ge \dots$. The $k$-th principal component is the portfolio $W_{\cdot k}$. Its return series is $R W_{\cdot k}$ and its variance is $\lambda_k$. The share of total variance explained by the first $k$ components is $\sum_{j \le k}\lambda_j / \sum_j \lambda_j$. Keeping the top $k$ gives a factor model with factor returns $F = RW_{\cdot,1:k}$ and loadings from regressing each stock on $F$.

### Why asymptotic PCA works on the $T \times T$ matrix

With $N = 1{,}000$ stocks and a 252-day window, $S$ is $1000 \times 1000$ with rank at most 252. Connor & Korajczyk's **asymptotic PCA (APCA)** uses the $T \times T$ cross-product instead:

$$
\Omega = \frac{1}{N} R R^\top \quad (T \times T)
$$

$RR^\top$ and $R^\top R$ have the same non-zero eigenvalues. The top-$k$ eigenvectors of $\Omega$, scaled so $F^\top F / T = I_k$, are consistent estimates (as $N \to \infty$) of the factor **returns**. Loadings follow by time-series regression, $B = R^\top F / T$ when nothing is missing. Eigen-decomposing a $252 \times 252$ matrix is far cheaper than a $1000 \times 1000$ one, and the large-$N$ asymptotics are the right ones for a wide equity panel. When $N \le T$ the code decomposes $N \times N$ instead. It gives the same answer more cheaply (`_extract_factors`).

### Heteroskedasticity iterations

Plain PCA is dominated by high-variance stocks: one volatile biotech can define a "factor". APCA corrects this (`apca`, `n_iter = 3`):
1. Extract factors from $R$.
2. Estimate each stock's residual variance $s_i$.
3. Rescale columns, $R_s = R\, \operatorname{diag}(s)^{-1/2}$, and re-extract the factors from $R_s$.
4. Re-estimate loadings by regressing the **original** returns on the new factors.
5. Repeat.

Missing returns are demeaned and set to zero for the extraction, and skipped in the per-stock regressions. A stock needs at least `min_stock_obs` observations (default $\max(20, \text{window}/6) = 42$) to get loadings. Otherwise its loadings are zero and its specific variance is the cross-sectional median.

### Choosing the number of factors

Too few factors and the model misses structure. Too many and it fits noise. Two criteria are implemented (`select_n_factors`). Both work on the spectrum after standardising each column by its own sd, so the noise variance is comparable across stocks.

**Bai–Ng $IC_{p2}$** (`bai_ng_ic2`) picks $k$ minimising

$$
IC_{p2}(k) = \ln V(k) + k\,\frac{N+T}{NT}\,\ln\min(N, T), \qquad V(k) = \frac{1}{N}\sum_{j > k}\lambda_j
$$

where $V(k)$ is the average residual variance after removing $k$ factors and the second term penalises complexity. It trades fit against parameters in the same way as an information criterion such as BIC.

**Marchenko–Pastur edge** (`marchenko_pastur_edge`) counts eigenvalues above the noise edge $\lambda_+ = \sigma^2(1 + \sqrt{N/T})^2$ from §1. Here $\sigma^2$ starts as the average eigenvalue and is re-estimated from the bulk (the eigenvalues below the edge) until the count is stable. The dimension is the number of non-zero eigenvalues, because residuals of a $p$-factor regression live in an $(N-p)$-dimensional subspace.

In the hybrid, $k$ is chosen **once** in `fit` by Bai–Ng on the last `statistical_window` (252) days of residuals, then floored at 1 and capped at `max_statistical_factors` (5). Setting `statistical_n_factors` fixes $k$ instead.

### Sign and rotation alignment (Procrustes)

Principal components are defined only up to sign, and nearly equal eigenvalues can swap or rotate between refits. Factor 2 this month could be "minus factor 3" last month, which makes the factor-return history meaningless. After each refit the new factor returns on the days that overlap with the previous window are rotated onto the old ones by **orthogonal Procrustes** (`_procrustes`):

$$
Q^* = \arg\min_{Q^\top Q = I}\|F_{\text{new}}Q - F_{\text{prev}}\|_F = UV^\top, \quad \text{where } F_{\text{new}}^\top F_{\text{prev}} = U\Sigma V^\top \text{ (SVD)}
$$

Then $F \leftarrow FQ^*$ and $B \leftarrow BQ^*$. This leaves $FB^\top$, and therefore the covariance, unchanged. The first solution is signed so that each factor's loadings sum to a positive number.

### Point-in-time schedule

`StatisticalRiskModel.fit` refits APCA every `statistical_refit_every` (21) days on the trailing 252-day window. The refit grid is anchored at the **first** eligible date, not the last, so a long fit and a fit truncated at any date agree exactly. Between refits:
- daily factor returns come from a cross-sectional OLS of the day's returns on the last refit's loadings;
- exposures are refreshed daily by time-series regression over the trailing window;
- the factor covariance is the (equal-weighted by default) covariance of the trailing window.

`tests/test_risk_statistical.py::test_statistical_model_is_point_in_time` checks a refit date, a mid-cycle date and the last date against truncated fits, to $10^{-12}$.

### The hybrid construction

`qts/risk/hybrid.py::HybridRiskModel.as_of` combines the two models:

$$
B_{\text{hyb}} = \big[\, B_{\text{fund}} \;\big|\; \tilde B_{\text{stat}} \,\big], \qquad F_{\text{hyb}} = \begin{pmatrix} F_{\text{fund}} & 0 \\ 0 & F_{\text{stat}} \end{pmatrix}, \qquad D_{\text{hyb},i} = D_{\text{fund},i}\cdot \text{keep}_i
$$

where:
- $F_{\text{stat}}$ is the covariance of the residual factor returns, which are extracted from the fundamental **residuals**;
- $b_i$ are the loadings of stock $i$ on those factors (see "the leak" below);
- $\text{share}_i = b_i^\top F_{\text{stat}} b_i / (b_i^\top F_{\text{stat}} b_i + s_i)$ is the fraction of the stock's residual variance that the residual factors explain, with $s_i$ the APCA specific variance;
- $\text{keep}_i = \operatorname{clip}(1 - \text{share}_i,\ 0.1,\ 1)$. The 10% floor (`SPECIFIC_FLOOR`) keeps every stock's idiosyncratic variance positive;
- $\tilde b_i = b_i \sqrt{(1 - \text{keep}_i) D_{\text{fund},i} / (b_i^\top F_{\text{stat}} b_i)}$ re-expresses the loadings in the fundamental model's specific-variance units.

**Keeping per-stock variance unchanged.** With this scaling, $(\tilde B F_{\text{stat}} \tilde B^\top)_{ii} = (1 - \text{keep}_i) D_{\text{fund},i}$. Adding it to $D_{\text{hyb},i} = \text{keep}_i D_{\text{fund},i}$ gives back $D_{\text{fund},i}$. **The hybrid diagonal equals the fundamental diagonal stock by stock.** Each stock's own variance is unchanged, and the hybrid only *moves* part of each stock's specific variance into shared factors. A diversified random book keeps essentially the same predicted variance, because the new off-diagonal terms average out. A book aligned with a residual factor picks up the covariance it was missing.

```python
keep = np.clip(1.0 - share, SPECIFIC_FLOOR, 1.0)
D_fund = fund.specific_var.reindex(tickers).values
D_h = D_fund * keep
# express the statistical block in the fundamental specific-variance units so
# that (B~ F B~')_ii = (1 - keep_i) D_fund_i: the hybrid diagonal equals the
# fundamental's stock by stock, only the off-diagonal structure is added
scale = np.where(sys_var > 0, np.sqrt((1.0 - keep) * D_fund / np.where(sys_var > 0, sys_var, 1.0)), 0.0)
Bs_scaled = B_use * scale[:, None]
```

**Where the loadings come from: the leak argument.** The residual factor *returns* come from the residuals. The *loadings* used in the hybrid come, by default (`loadings="total"`), from regressing each stock's **total** return on those factor returns. The reason is a leak in the cross-sectional regression:
- Suppose a latent factor has true loadings $g$ (an $N$-vector) and return $f_t$.
- Each day the regression of §4 applies its hat matrix $H_t$ (the projection onto the exposure columns). The part $H_t g f_t$ **leaks into the fundamental factor returns**, and only $(I - H_t) g f_t$ stays in the residuals.
- APCA on the residuals therefore recovers only $(I - H)g$.
- A book aligned with $g$ really earns $w^\top g f_t$. A block model built from residual loadings loses the cross term $2(w^\top Hg)(w^\top(I-H)g)$. That is roughly $2p/N$ of the aligned variance with $p$ fundamental factors on $N$ stocks: about 35% in the $N=150$ test market and a few percent at $N = 1000$.
- The leak can't be recovered from factor-return time series either, because $H_t$ moves with the style exposures and the leaked component changes sign over time.
- The latent return is in the *total* return in full, so regressing total returns on the residual factor returns recovers $g$ directly, and the aligned book is priced correctly.

The explained share is computed from the same loadings, so the diagonal stays pinned to the fundamental model either way.

**Block-diagonal $F$.** With total-return loadings the cross block between fundamental and residual factors is set to zero. The residual factors are cross-sectionally orthogonal to the exposures by construction, their time-series correlation with the fundamental factors is only the sign-varying leak, and the total-return loadings already carry the leaked part, so a cross block would double count. `cross_block=True` estimates one anyway (weighted correlation × the two blocks' vols, then projected to PSD). It is meant for `loadings="residual"`, where it recovers the time-stable part of the leak.

**Alignment and fallback.** The alpha-alignment term (§9) is computed against the *hybrid* exposures. Alpha that a residual factor spans is no longer "unseen", so the alignment term shrinks accordingly. Before the statistical block has enough residual history (`min_obs` = 201 days with the defaults), `as_of` returns the fundamental output unchanged. `statistical_available(date)` tells you which case you are in.

### Missing-factor monitor

`missing_factor_monitor(md, model, lookback=252, ...)` runs per date on the trailing 252 days of fundamental residuals:
1. Keep stocks observed on at least 80% of the days and standardise each.
2. Eigen-decompose the sample covariance.
3. Report: PC1's share of residual variance, the top-$k$ share, $\lambda_1$, the MP edge, and the MP factor count.
4. **Flag** when PC1's share exceeds `missing_factor_alarm_share` (10%) **and** $\lambda_1$ is above the MP edge.

In words: there is a common factor in the residuals that the fundamental model does not price. A flag is a research prompt. Find the characteristic behind PC1 and, if it persists, promote it to a fundamental style. Once promoted, it disappears from the residual block automatically.

### The test result

`tests/test_risk_statistical.py::test_hybrid_prices_unobserved_factor` is the acceptance test. Setup:
- the synthetic market has 150 stocks and 900 days;
- from day 500 a latent factor with 1% daily vol and random loadings $g$ is added to every return, with no observable footprint;
- the "aligned" book is $g$, demeaned and scaled to gross 2;
- bias statistics (§10) are computed over the last 150 days, where the 95% band for $T = 150$ is $[0.885, 1.115]$.

| Book | Fundamental model | Hybrid model |
|---|---|---|
| aligned with the latent factor | **3.28** | **1.06** |
| random long/short 0 | 0.97 | 0.98 |
| random long/short 1 | 0.86 | 0.87 |
| random long/short 2 | 0.92 | 0.90 |

The fundamental model understates the aligned book's volatility by a factor of 3.3. The hybrid brings it inside the band and leaves random books essentially unchanged. The hybrid/fundamental ratio of predicted variance for 200 random books is 0.997 (test bound ≤ 1.05). The test also asserts the two diagonals agree to $10^{-8}$. These are synthetic, in-sample-by-construction numbers from a deterministic test run, not market evidence.

**Why this choice.** APCA over plain PCA because of heteroskedasticity. Residuals over total returns so the statistical factors add to the named ones instead of duplicating the market. Total-return loadings because of the leak. Bai–Ng because it picks 0 on pure noise and 3 ± 1 on a planted 3-factor panel (`test_apca_recovers_planted_factor_structure`).

**In the code.**
- `StatisticalRiskModel(n_factors, window, half_life, min_obs, method, apca_iters, refit_every, min_stock_obs)`.
- `apca`, `select_n_factors`, `bai_ng_ic2`, `marchenko_pastur_edge`.
- `HybridRiskModel(fundamental, cfg, half_life=None, loadings="total", cross_block=False)`.
- `missing_factor_monitor`.
- Config: `statistical_companion`, `max_statistical_factors`, `statistical_n_factors`, `statistical_window`, `statistical_refit_every`, `missing_factor_alarm_share`.

**Pitfalls & checks.**
- *End-of-sample factor count.* In a backtest, `fit` picks $k$ from the **last** 252 days, which is a mild look-ahead in one integer. Set `statistical_n_factors` for a fully point-in-time backtest, or refit periodically in production.
- *Rotation instability.* `test_factor_sign_and_rotation_stability` requires every factor's returns to correlate positively across consecutive refits, and the real-time chain to correlate > 0.5 with the refit solution.
- *Missing data.* `test_apca_handles_missing_returns` drops 5% of entries at random and gives one stock a short history.
- *Over-fitting noise.* The MP edge check in the monitor prevents alarms on eigenvalues that pure noise would produce.

**References**
- Connor & Korajczyk (1986), *Performance measurement with the arbitrage pricing theory: A new framework for analysis*, Journal of Financial Economics.
- Connor & Korajczyk (1988), *Risk and return in an equilibrium APT: Application of a new test methodology*, Journal of Financial Economics.
- [Bai & Ng (2002), Determining the Number of Factors in Approximate Factor Models](https://doi.org/10.1111/1468-0262.00273)
- [SimCorp/Axioma, Statistical versus fundamental models](https://www.simcorp.com/resources/insights/white-papers-and-reports/statistical-versus-fundamental-models)

## 9. The alpha alignment problem (Saxena–Stubbs)

**What it is.** A rank-one correction to $\Sigma$ that adds risk in the one direction the optimizer is most tempted to exploit: the part of our alpha that the risk factors cannot explain.

**Why we need it here.** Split the alpha vector into a part spanned by the exposures and an orthogonal remainder:

$$
\alpha = B\,b + \alpha_\perp, \qquad B^\top \alpha_\perp = 0, \qquad y = \frac{\alpha_\perp}{\|\alpha_\perp\|}
$$

where:
- $b$ are the OLS coefficients of $\alpha$ on the columns of $B$;
- $\alpha_\perp$ is the residual, i.e. the alpha the risk model "can't see";
- $y$ is its unit-norm direction.

A portfolio that loads on $y$ has zero factor exposure, so the model prices its risk as purely specific: $\sum_i w_i^2 \sigma^2_{u,i}$, which diversifies away fast. But alpha signals are themselves common factors. Every stock with a high score tends to move together, because the score is built from characteristics that drive returns. So the $y$ direction carries real systematic risk that the model is blind to. The optimizer sees high return and low risk, and loads up. Realised risk then exceeds forecast risk *specifically for the optimised book*, even if the model is unbiased for random portfolios.

Saxena & Stubbs quantify this (plan §3). At a 3% active-risk target, optimised portfolios realised 4.11%, a bias statistic of 1.37 against a 95% band of $[0.88, 1.12]$. Adding the alignment term brought the bias to 0.93, kept the problem convex, and shifted the realised efficient frontier up by 60–80 bp. Those results are from the paper's backtests.

**How it works.** Augment the covariance:

$$
\Sigma_{\text{aug}} = B F B^\top + D + \nu\, y y^\top, \qquad w^\top \Sigma_{\text{aug}} w = w^\top(BFB^\top + D)w + \nu\,(y^\top w)^2
$$

where:
- $\nu$ is the variance assigned to the alignment factor (daily units);
- $(y^\top w)$ is the portfolio's exposure to the hidden direction.

The optimizer now pays for loading on $y$ and backs off. The term is a sum of squares, so the problem stays convex.

**Choosing $\nu$.** Saxena–Stubbs treat $\nu$ as a calibration parameter: tune it until the optimised book's bias statistic is about 1. The code has two levels:

1. **Heuristic starting point** (`alpha_alignment`). Give the alignment factor the variance of a typical style factor. A standardised style with exposures $x$ has $\|x\|^2 \approx N$ and contributes $F_{kk}(x^\top w)^2$. The alignment factor has $\|y\| = 1$. To make the two comparable:

$$
\nu = \text{nu\_scale} \times N \times \operatorname{median}_k F_{kk}^{\text{style}}
$$

If no factor covariance is passed, the fallback is $\nu = \text{nu\_scale} \times \operatorname{median}(D)$.

```python
y = pd.Series(resid / norm, index=idx)
N = len(idx)
if factor_cov is not None:
    sv = style_variances(factor_cov, cfg)
    base = float(np.median(sv.values)) if len(sv) else float(np.median(np.diag(factor_cov.values)))
    nu = nu_scale * N * base
else:
    nu = nu_scale * float(specific_var.reindex(idx).median())
```

2. **Calibration** (`calibrate_alignment_nu`). For each candidate `nu_scale`:
   - build the optimised book on every date in a calibration window;
   - forecast its volatility with the alignment term included;
   - compare the forecast with the next day's realised return;
   - form the bias statistic.

   A larger `nu_scale` both raises the forecast and pushes the optimizer off $y$, so in expectation the bias falls monotonically in `nu_scale`. The function bisects on $\log(\text{nu\_scale})$ in $[0.05, 50]$ until the bias is within 0.02 of 1. `model.as_of` is evaluated once per date, because $\nu$ is linear in `nu_scale` and is rescaled in place. The default book is a dollar-neutral $\Sigma_{\text{aug}}^{-1}\alpha$ scaled to gross 2. Pass `solve_fn` wrapping the production optimizer to calibrate on the real constraint set. The result dict reports `nu_scale`, `bias`, `bias_unaligned`, the band, `in_band`, `converged` and the search history.

The `nu_scale` used day to day is the `as_of(date, alpha, nu_scale=...)` argument. The pipeline's `risk_fn` (`qts/pipeline.py`) relies on its default. Calibration is deliberately **not** on the default `as_of` path, because it costs one optimisation per date per bisection step. Run it offline on the warm-up window once optimised-book bias statistics exist. Record the calibrated value, and the procedure used, in `docs/notes/alignment_nu_calibration.md`. Until that note exists, treat the default as the heuristic above. The open item is tracked in `docs/audit/RESOLUTION.md`, "Still open".

**Misalignment gauge.** `misalignment_gauge(alpha, exposures)` returns $1 - R^2$ of alpha regressed on the exposures: the fraction of alpha variance the risk model can't see. The plan asks for it as a standing monitor.

**Numbers from the audit** (`docs/audit/risk_portfolio.md` finding 5; synthetic market):
- *Mechanism.* Orthogonality is exact ($\max|B^\top y| = 3 \times 10^{-14}$, $\|y\| = 1$). With a random alpha (fully orthogonal to the factors), the un-aligned optimum had model variance $1.58 \times 10^{-5}$ and augmented variance $7.46 \times 10^{-5}$, i.e. $4.7\times$. The aligned solve cut $y^\top w$ from 0.215 to 0.071 and $\alpha^\top w$ by 65%. The term works as designed: the optimizer backs off the invisible direction.
- *On the pipeline's real composite alpha*, $1 - R^2 = 0.097$ and the alignment share of the optimised book's variance was about 0%. Today the term is nearly inert. That is fine, but it also means it hasn't been tested on the book it is meant to protect, which is why calibration is the first follow-up once live bias statistics accumulate.

**Why this choice.** The plan calls the alignment fix "non-negotiable because the problem is created by the optimizer, not the estimator". The alternatives are to add alpha factors as risk factors (done in part: momentum, value, quality, investment and short interest are styles, §3) or to cap exposure to the alpha direction with a hard constraint, which is cruder and non-adaptive. The alignment factor handles whatever part of the alpha remains orthogonal, day by day. OLS projection (`weighted=False`) is the default because it matches the paper and makes $B^\top y = 0$ exact. `weighted=True` uses $1/D$ weights instead.

**In the code.**
- `alpha_alignment(alpha, exposures, specific_var, nu_scale=1.0, factor_cov=None, cfg=None, weighted=False)` returns `(y, nu)`.
- `BarraStyleRiskModel.as_of` and `HybridRiskModel.as_of` call it when `alpha` is given and `RiskConfig.alignment` is on.
- `RiskModelOutput.covariance(include_alignment=True)` and `portfolio_variance(w, include_alignment=True)` include the term by default, so it also enters `expected_risk` and the vol target.

**Pitfalls & checks.**
- *Uncalibrated $\nu$ feeds the vol target.* An oversized $\nu$ inflates `expected_risk` and shrinks the book. Log the alignment share of ex-ante variance.
- *Alpha mismatch.* $y$ must be built from exactly the alpha the optimizer receives. The audit verified that the simulator passes the same `exp_ret` to `risk_fn` and to `solve`.
- `tests/test_risk.py::test_alignment_orthogonal_unit_norm` checks $\|y\| = 1$, $|B^\top y| < 10^{-8}$, $\nu > 0$, and that the augmented variance of $y$ exceeds the un-augmented.
- `test_calibrate_alignment_nu_smoke` checks the search stays in bounds, that bias at `nu_scale=100` < bias at 0.01, and that the unaligned bias is the largest.

**References**
- [Saxena & Stubbs, Alpha Alignment Factor: A Solution to the Underestimation of Risk for Optimized Active Portfolios](https://www.math.ttu.edu/conferences/past/mcmhr/docs/saxena.pdf)
- [SEI (2026), Risk Modelling of Optimized Factor Portfolios](https://www.seic.com/sites/default/files/2026-03/Risk%20Modelling%20of%20Optimized%20Factor%20Portfolios.pdf)

## 10. Evaluating a risk model

**What it is.** Daily forecasts checked against what happened: is the *level* right (bias statistic), is it right in *every regime* (rolling bias, MRAD), and does it *rank* risks well (QLIKE, Mincer–Zarnowitz, minimum-variance test)?

**Why we need it here.** A risk model can't be judged by eye, and a model that is right "on average" can still be badly wrong for the one portfolio that matters, the optimised book. The plan requires tests on pure-factor portfolios, on random long/short portfolios, and "decisively, the actual optimized book" (`docs/BUILD_PLAN.md` §3).

### Bias statistic

**How it works** (`qts/risk/evaluation.py::bias_statistic`). For a portfolio with forecast volatility $\sigma_{t}$ made at the close of day $t$ and realised return $r_{t+1}$ the next day:

$$
z_{t+1} = \frac{r_{t+1}}{\sigma_t}, \qquad b = \operatorname{sd}(z) = \sqrt{\frac{1}{T}\sum_{t}(z_t - \bar z)^2}
$$

where:
- $z_t$ is the standardised return, which should have sd 1 if the forecast is right;
- $T$ is the number of forecast/realisation pairs;
- $b$ is the bias statistic (computed with `ddof=0`).

**Reading it.**
- $b > 1$: realised moves are larger than forecast, so the model **underestimates** risk. This is the dangerous direction, because the optimizer will over-lever.
- $b < 1$: the model **overestimates** risk. The book is too small and alpha is left on the table.

**The 95% band.** If $z$ were exactly standard normal, the sample sd of $T$ draws would have standard error $\approx 1/\sqrt{2T}$, so

$$
b \in \left[1 - \frac{2}{\sqrt{2T}},\ 1 + \frac{2}{\sqrt{2T}}\right] \text{ with about 95\% probability}
$$

(`bias_band`). For $T = 250$ that is $1 \pm 0.089$. Fat tails widen the true band, which is why USE4 also reports rolling percentiles.

### Rolling bias and MRAD

`rolling_bias_stats(predicted_vol, realized_returns, window=252)` computes $b$ over trailing 12-month windows. Under realistic kurtosis USE4 reports 5th/95th percentiles of about 0.66/1.34 for a correctly specified model. The **mean rolling absolute deviation** (`mean_rolling_absolute_deviation`) is

$$
\text{MRAD} = \frac{1}{W}\sum_{w} \lvert b_w - 1 \rvert
$$

It summarises how far the model drifts from unbiased across regimes. The ideal is about 0.17 for 12-month windows, rising to about 0.19 under kurtosis 3.5–4 (USE4, via plan §3). A model can have $b \approx 1$ overall while being too high in calm years and too low in crises. MRAD exposes that, and it is the failure that VRA (§5) targets.

### QLIKE

For per-stock variance forecasts $\hat\sigma^2_t$ and realised squared returns $r_t^2$ (`qlike`):

$$
\text{QLIKE} = \frac{1}{T}\sum_t \left(\frac{r_t^2}{\hat\sigma_t^2} - \ln\frac{r_t^2}{\hat\sigma_t^2} - 1\right)
$$

Each term is $\ge 0$, with equality when $r_t^2 = \hat\sigma^2_t$. Lower is better. Two properties make it the primary loss:
- **Robust to a noisy proxy.** We never observe the true variance, only $r_t^2$, which is unbiased but extremely noisy. Patton (2011) shows that QLIKE and MSE are among the few losses for which ranking forecasts against the noisy proxy gives the same ranking as against the true variance.
- **Asymmetric.** It penalises under-prediction (ratio ≫ 1) more than over-prediction, which matches the cost asymmetry for a levered optimiser.

Zero squared returns are floored at $10^{-18}$ to keep the log finite.

### Mincer–Zarnowitz regression

$$
r_t^2 = a + \beta\,\hat\sigma_t^2 + e_t
$$

(`mincer_zarnowitz`). An unbiased, efficient forecast has $a = 0$, $\beta = 1$. $\beta < 1$ means forecasts are too dispersed: high forecasts too high, low ones too low. The $R^2$ is always small for daily data, because $r^2$ is a very noisy proxy. Compare $R^2$ across models, never against an absolute bar. The plan asks for several losses side by side because rankings can differ by loss (Pollok).

### Out-of-sample minimum-variance test

`min_variance_oos_test(cov_estimators, returns, rebalance_every=21, window=252)`:
1. Every 21 days, estimate $\Sigma$ from the trailing window with each candidate estimator.
2. Form the global minimum-variance portfolio $w \propto \Sigma^{-1}\mathbf{1}$.
3. Hold it until the next rebalance and record realised returns.

The estimator whose GMV portfolio has the lowest realised variance wins. Average turnover is reported too. This tests exactly what an optimizer does with $\Sigma$, namely invert it, so it punishes underestimated small eigenvalues directly. Ledoit–Wolf and DCC-NL use it as the benchmark (plan §3).

### How the tests use them

| Test | What it checks |
|---|---|
| `tests/test_risk.py::test_random_portfolio_bias_statistic` | 3 random dollar-neutral books, 250 days, via `portfolio_bias_test`: each $b \in (0.7, 1.4)$, upper band $= 1 + 2/\sqrt{500}$ |
| `test_evaluation_metrics` | on simulated returns with known variance: $b \in (0.9, 1.1)$, MZ $\beta \in (0.7, 1.3)$, QLIKE of the true variance < QLIKE of 3× it |
| `test_min_variance_oos_test` | QIS GMV variance ≤ sample-covariance GMV variance on a one-factor panel |
| `tests/test_risk_statistical.py::test_hybrid_prices_unobserved_factor` | aligned book bias > 1.3 under the fundamental model, in $[0.8, 1.25]$ under the hybrid. Random books in $[0.8, 1.25]$ (§8) |
| `test_calibrate_alignment_nu_smoke` | bias statistic of the optimised book falls as `nu_scale` rises (§9) |

`portfolio_bias_test(model, md, portfolios, dates, include_alignment=True, alpha_fn=None)` is the general harness. It evaluates `model.as_of(t)`, forecasts $\sqrt{w^\top\Sigma_t w}$, realises the return on $t+1$, and returns `bias, lower, upper, in_band, T` per portfolio. Portfolios can be static weights or callables `f(date, tickers)`. Use `random_long_short_weights(rng, gross=2.0)` for random books.

**Pitfalls & checks.**
- *Off-by-one.* The forecast must be made at $t$ and compared with $r_{t+1}$. `portfolio_bias_test` does this. A hand-rolled loop that pairs $\sigma_t$ with $r_t$ makes any model look better than it is.
- *Small $T$.* With 4 observations the band is $[0.29, 1.71]$ (see the worked example), so short tests prove nothing.
- *Testing the wrong portfolio.* Random books are close to the diagonal and hide factor and alignment problems. Always include the optimised book.

**References**
- [Menchero, Orr & Wang (2011), USE4 Methodology Notes](https://www.top1000funds.com/wp-content/uploads/2011/09/USE4_Methodology_Notes_August_2011.pdf)
- Patton (2011), *Volatility forecast comparison using imperfect volatility proxies*, Journal of Econometrics.
- [Pollok (2025), arXiv:2506.07928](https://arxiv.org/pdf/2506.07928)
- [Reis, Serra & Gama (2025), arXiv:2503.01581](https://arxiv.org/html/2503.01581v1)

## 11. Parameters, tuning and pitfalls

### `RiskConfig` reference

All defaults from `qts/core/config.py::RiskConfig`.

| Field | Default | Meaning |
|---|---|---|
| `vol_half_life` | 84 | EWMA half-life (trading days) for factor **variances** (§5) |
| `corr_half_life` | 504 | EWMA half-life for factor **correlations**. Also sets the eigen-simulation length $\min(T, 4 \times 504)$ |
| `vra_half_life` | 42 | half-life of the EWMA of squared bias statistics in factor and specific VRA |
| `newey_west_lags_vol` | 5 | Newey–West lags for factor variances and specific variances |
| `newey_west_lags_corr` | 2 | Newey–West lags for factor correlations |
| `specific_half_life` | 84 | EWMA half-life for specific variance. Also sets the specific window $\max(6h, 252)$ and the shrunk-residual window $\max(4h, 252)$ |
| `specific_shrink_q` | 0.1 | Bayesian shrinkage strength $q$ toward size-decile means (§6) |
| `min_history_days` | 252 | warm-up before exposures are built (default `warmup_days`), and minimum factor-return history for `as_of` |
| `styles` | `beta, size, momentum, value, residual_vol, liquidity, short_term_reversal, quality, investment, short_interest` | style factors (§3). Must be a subset of `SUPPORTED_STYLES` |
| `use_nonlinear_shrinkage` | True | QIS (True) or Ledoit–Wolf (False) in `shrunk_residual_covariance`. Does **not** affect `specific_var` |
| `alignment` | True | add the Saxena–Stubbs term when an alpha is passed to `as_of` |
| `eigenfactor_adjust` | True | run the eigenfactor risk adjustment |
| `n_eigen_sims` | 200 | Monte-Carlo draws for the eigenfactor bias |
| `statistical_companion` | True | `build_risk_model` returns a `HybridRiskModel` |
| `max_statistical_factors` | 5 | cap on the Bai–Ng residual factor count |
| `statistical_n_factors` | None | fix $k$ (fully point-in-time). `None` selects by Bai–Ng in `fit` |
| `statistical_window` | 252 | trailing window for APCA, factor count and total-return loadings |
| `statistical_refit_every` | 21 | APCA re-extraction frequency (trading days) |
| `missing_factor_alarm_share` | 0.10 | PC1 share of residual variance that, together with the MP edge, raises the monitor flag |

Constructor arguments that are not in `RiskConfig`:
- `BarraStyleRiskModel(warmup_days=None, eigen_refresh_days=21, seed=0, constraint_weights="cap", weight_cap_quantile=None)`.
- `HybridRiskModel(half_life=None, loadings="total", cross_block=False)`.
- `as_of(..., nu_scale=...)`.

Exposure window lengths and `BETA_SHRINK` are module constants in `qts/risk/exposures.py`.

### Tuning guide

| Symptom | Likely cause | Knob |
|---|---|---|
| Rolling bias of factor portfolios swings above 1 in sell-offs and below 1 afterwards | factor vols react too slowly | shorten `vra_half_life` (e.g. 21) or `vol_half_life` (e.g. 63). For a slower book, use the plan's long-horizon variant (252 / 168) |
| Optimised book's bias > 1 while random books are fine | alpha misalignment or unseen factor | run `calibrate_alignment_nu`. Check `misalignment_gauge` and `missing_factor_monitor`. Raise `max_statistical_factors` if the monitor flags persistently |
| Random books' bias < 1 (risk overstated) | too much NW, too long half-lives after a calm spell, or alignment $\nu$ too large | check `vra_multiplier`. Reduce `nu_scale` |
| A few names carry huge specific risk | outliers, thin histories | inspect `specific_risk(..., return_details=True)`: `gamma`, `sigma_ts` vs `sigma_str`. Raise `specific_shrink_q` |
| Factor returns noisy, $F$ ill-conditioned | collinear styles, empty industries | drop or merge styles. Merge tiny industries in the data layer |
| `as_of` slow | eigen simulation | lower `n_eigen_sims` (e.g. 100) or raise `eigen_refresh_days`. Set `eigenfactor_adjust=False` for quick research |

Tune on bias statistics, not backtest Sharpe. Every parameter change is a new configuration in the trial ledger (→ see **Validation › Deflated Sharpe**).

### Pitfalls (summary)

<callout icon="⚠️" color="yellow_bg">
Look-ahead in exposures. Every descriptor must use data ≤ t, fundamentals must be joined on filing date, and the regression for day t must use exposures from t−1. Guarded by `test_exposure_history_matches_single_date_and_is_point_in_time`, `test_point_in_time_as_of` and `test_statistical_model_is_point_in_time`. The audit verified all of them. One known end-of-sample choice remains: the hybrid's factor count. Fix it with `statistical_n_factors` for a strict backtest.
</callout>

<callout icon="⚠️" color="yellow_bg">
Collinearity. Market + industries are exactly collinear, and the sum-to-zero constraint handles that. Near-collinear styles are not handled. They inflate factor variances and make the optimizer's style constraints fight each other. An industry with no members is dropped for the day and reported as 0. VRA ignores those zeros.
</callout>

<callout icon="⚠️" color="yellow_bg">
Thin histories. New listings get a structural specific risk until about 180 observations. APCA gives zero loadings below `min_stock_obs`. Tickers outside the model get the median specific variance and zero factor exposure, so they look idiosyncratic. Intersect universes before optimising.
</callout>

<callout icon="⚠️" color="yellow_bg">
Regime shifts. EWMA lags. VRA corrects the overall level within weeks but not the correlation structure, which moves on a 504-day half-life. In a correlation regime break (everything going to correlation 1), expect the bias of diversified books to exceed 1 for a while. Watch rolling bias, not just the full-sample number.
</callout>

## 12. Worked example

### A three-stock, one-factor model by hand

Three stocks, one factor (the market). Daily units.

| Stock | Exposure $B_i$ | Specific vol $\sigma_{u,i}$ | Specific variance $D_{ii}$ |
|---|---|---|---|
| A | 1.2 | 2.0% | 0.000400 |
| B | 0.8 | 1.0% | 0.000100 |
| C | 1.0 | 1.5% | 0.000225 |

The market factor has a daily vol of 1%, so $F = 0.0001$.

**Step 1: $BFB^\top$.** Each entry is $B_i B_j F$:

$$
BFB^\top = 0.0001 \begin{pmatrix} 1.44 & 0.96 & 1.20 \\ 0.96 & 0.64 & 0.80 \\ 1.20 & 0.80 & 1.00 \end{pmatrix} = \begin{pmatrix} 0.000144 & 0.000096 & 0.000120 \\ 0.000096 & 0.000064 & 0.000080 \\ 0.000120 & 0.000080 & 0.000100 \end{pmatrix}
$$

**Step 2: add $D$ to the diagonal.**

$$
\Sigma = BFB^\top + D = \begin{pmatrix} 0.000544 & 0.000096 & 0.000120 \\ 0.000096 & 0.000164 & 0.000080 \\ 0.000120 & 0.000080 & 0.000325 \end{pmatrix}
$$

Check one entry: stock A's total daily vol is $\sqrt{0.000544} = 2.33\%$. The implied correlation of A and B is $0.000096/\sqrt{0.000544 \times 0.000164} = 0.32$. They are correlated only through the market.

**Step 3: portfolio variance.** Take $w = (0.5, 0.3, 0.2)$. Use the factor form, which is how `portfolio_variance` computes it:
- portfolio exposure: $B^\top w = 1.2(0.5) + 0.8(0.3) + 1.0(0.2) = 1.04$;
- factor variance: $1.04^2 \times 0.0001 = 0.00010816$;
- specific variance: $0.5^2(0.0004) + 0.3^2(0.0001) + 0.2^2(0.000225) = 0.0001 + 0.000009 + 0.000009 = 0.000118$;
- total: $w^\top \Sigma w = 0.00010816 + 0.000118 = 0.00022616$.

The daily vol is $\sqrt{0.00022616} = 1.504\%$, and annualised it is $1.504\% \times \sqrt{252} = 23.9\%$. Factor risk is 48% of the variance and specific risk 52%. Multiplying out $w^\top\Sigma w$ with the full matrix from Step 2 gives the same 0.00022616, which is a good self-check.

You can verify in Python:

```python
import numpy as np
B = np.array([[1.2], [0.8], [1.0]]); F = np.array([[1e-4]])
D = np.diag([4e-4, 1e-4, 2.25e-4]); w = np.array([0.5, 0.3, 0.2])
S = B @ F @ B.T + D
print(w @ S @ w, np.sqrt(w @ S @ w * 252))   # 0.00022616  0.2387
```

### A bias statistic for four days

Made-up forecasts (daily vol made at the close of day $t$) and realised returns on day $t+1$:

| Day | Forecast $\sigma_t$ | Realised $r_{t+1}$ | $z = r/\sigma$ | $z^2$ |
|---|---|---|---|---|
| 1 | 1.0% | +1.5% | +1.5 | 2.25 |
| 2 | 1.2% | −0.6% | −0.5 | 0.25 |
| 3 | 0.8% | +1.2% | +1.5 | 2.25 |
| 4 | 1.0% | −2.0% | −2.0 | 4.00 |

- Mean: $\bar z = (1.5 - 0.5 + 1.5 - 2.0)/4 = 0.125$.
- Mean of squares: $(2.25 + 0.25 + 2.25 + 4.00)/4 = 2.1875$.
- Population variance: $2.1875 - 0.125^2 = 2.171875$.
- Bias statistic: $b = \sqrt{2.171875} = 1.474$.

The realised moves were about 47% larger than forecast, so on its face the model underestimates risk. The 95% band for $T = 4$, though, is $1 \pm 2/\sqrt{8} = [0.29, 1.71]$, so 1.47 is **inside** the band. Four days can't distinguish a 47% underestimate from luck. With $T = 250$ the band is $[0.91, 1.09]$, and a persistent 1.47 would be decisive. `bias_statistic(pd.Series([.010, .012, .008, .010]), pd.Series([.015, -.006, .012, -.020]))` returns 1.474.

## Roadmap

Planned in `docs/BUILD_PLAN.md` §3 or the audit, not implemented:
- **Eigenfactor inflation**: $v \leftarrow 1 + a(v - 1)$ with $a = 1.2$ and a parabola fit across $k$, to fix the measured under-correction (audit finding 11).
- **Residual block**: a full QIS-shrunk (optionally DCC-GARCH) residual covariance in `RiskModelOutput`, as in Ledoit–Wolf's AFM-DCC-NL. Today it is used only for evaluation.
- **More styles**: 1-day reversal, industry momentum, non-linear beta/size, leverage, growth, dividend yield. Horizon-matched half-lives per alpha theme (plan §3, SEI evidence).
- **Long-horizon preset** (252-day vol, 168-day VRA) as a named configuration.
- **Alignment `nu` calibration on the optimised book**, once bias statistics from the production optimizer are collected (`docs/audit/RESOLUTION.md`, "Still open").
- **Factor-crowding monitor** (MSCI design) and an skfolio-based evaluation harness with Diebold–Mariano tests by regime.
- **Realised-covariance (HAR) methods**, only if intraday TAQ-grade data is bought (Phase 2+).
