# Concepts glossary

<callout icon="🧭" color="gray_bg">
Every financial, statistical, mathematical and machine-learning term used in the handbook and in the `qts` code, in alphabetical order. Each entry gives a plain-English explanation first, then the precise definition or formula where one helps, the repo's default value where one exists (from `qts/core/config.py`), and an arrow to the page where the concept is used. Units: "bp" is a basis point (0.01%); "daily" and "annual" are stated explicitly; returns are simple returns as fractions unless noted. Demo figures quoted in entries come from `runs/demo_post_refactor.txt` (synthetic market, 300 names, lag 0) unless another file is named.
</callout>

<callout icon="🆕" color="green_bg">
<b>What changed (Nov 2026)</b>
- New entries for the refactor, the lag investigation and the second audit: code version, DPP, execution delay, gross after borrow, halt vs delisting, label realisation, metaorder streak, NYSE session, out-of-model name, power cone, property-based test, routed venue, run manifest, run registry, slice equivalence, split-consistent ratio, target-vol risk aversion, trial family, upsert, vol-target binding.
- Corrected defaults: auction impact prefactor $b_a$ = 0.3 (was 0.5), alignment $\nu$ scale 0.05 (recalibrated), risk aversion $\gamma$ = 2 and searched per date by default (was a fixed 10), execution delay 1 day (was 0).
- Newey–West lags now $\min(\max(h, \lfloor 4(T/100)^{2/9}\rfloor), T-1)$, not $h-1$; gate folds are purged walk-forward blocks, not calendar years.
- `RiskConfig.use_nonlinear_shrinkage` no longer exists (QIS is an evaluation tool, not on the production path); `AlphaConfig.delisting_return` was folded into `CostConfig` (now `delisting_return_<reason>`; `delisting_haircut` is deprecated).
- The trial ledger is append-only with read-time de-duplication; demo numbers now refer to the post-refactor demo.
</callout>

## A

**130/30 and long-only books** — Alternative book types to the default long/short. A 130/30 book is 130% long and 30% short (net 100%); a long-only book holds no shorts. `PortfolioConfig.book` accepts `"long_short"` (default), `"long_only"` or `"130_30"`. Long-only books transfer far less of a signal into positions (transfer coefficient ≈ 0.3 vs 0.7–0.8 for long/short, `BUILD_PLAN.md` §1). → see **Portfolio construction**

**Active return** — The return of a portfolio minus the return of its benchmark. For a market-neutral long/short book the benchmark is cash (zero), so active return is just the return. Its standard deviation is called tracking error, and its Sharpe ratio the information ratio. → see **Validation & statistics › The information coefficient**

**ADV (average daily volume)** — How much of a stock trades on a typical day, used to size orders so they do not move the price too much. In this repo it is the trailing 63-day mean of daily dollar volume (`adv_usd`) or share volume (`adv_shares`), computed from data up to the decision date. Position changes are capped at 5% of ADV (`PortfolioConfig.max_adv_participation = 0.05`), and the universe requires ADV ≥ $2M (`UniverseConfig.min_adv_usd`). → see **Transaction cost model**, **Data & universe**

**Alpha** — The part of a stock's expected return that is not explained by the risk factors it is exposed to; informally, "edge". In this system the alpha model outputs a score per stock, which Grinold's rule converts into an expected *residual* return over the 20-day horizon. Alpha is distinct from beta (market exposure), which can be bought cheaply. → see **Alpha model**

**Alpha alignment factor** — A correction to the risk model for the fact that an optimizer loves exactly the part of the alpha that the risk model cannot see. Saxena & Stubbs decompose the alpha into a component spanned by the factor exposures and an orthogonal remainder $\alpha_\perp$, and add a synthetic factor along $y = \alpha_\perp/\|\alpha_\perp\|$ with variance $\nu$: $\Sigma_{\text{aug}} = BFB^\top + D + \nu yy^\top$. Without it an optimized book realized 4.11% active risk against a 3% target in their study. `RiskConfig.alignment = True`; $\nu = $ `alignment_nu_scale` $\times N \times$ the median style-factor variance, with `alignment_nu_scale = 0.05`. That value was recalibrated on 2026-11-05 with production settings (searched γ, one-day delay, no out-of-model names): on three synthetic seeds the risk model already slightly over-forecast the optimized book, so the calibration hit the search floor. It must be recalibrated on real data (`scripts/calibrate_alignment_nu.py`). → see **Risk model**

**Almgren–Chriss** — A classic model for splitting a large order over time that trades off market impact (trade slowly) against price risk (trade quickly), assuming impact is linear in trading speed. Because empirical impact follows a square-root law instead, the repo keeps it only as a reference trajectory and for intuition about urgency (`qts/execution/scheduler.py::almgren_chriss_schedule`). → see **Execution**

**Amihud illiquidity** — A measure of how much a stock's price moves per dollar traded: the average of $|r_t| / \text{dollar volume}_t$. High values mean an illiquid stock where trading is expensive. The feature `amihud_63` is the log of the 63-day mean (scaled by $10^6$) and belongs to the liquidity features. → see **Alpha model**

**Annualisation** — Converting a per-period statistic into yearly units. Under the iid assumption the mean scales with $q$ periods per year and the standard deviation with $\sqrt q$, so a Sharpe ratio scales with $\sqrt q$ ($q = 252$ for daily data). If returns are autocorrelated or the series is sampled less often (e.g. every 20 days, $q = 12.6$), using $\sqrt{252}$ overstates the result. → see **Validation & statistics › Return statistics**

**APCA (asymptotic principal components)** — A way to extract statistical factors when there are more stocks than days: eigen-decompose the $T\times T$ matrix $RR^\top/N$ instead of the singular $N\times N$ covariance (Connor & Korajczyk). The repo fits APCA on the fundamental model's residuals as a "companion" that catches common factors the fundamental model cannot observe (`qts/risk/statistical.py`, `RiskConfig.statistical_companion = True`, up to 5 factors, 252-day window, refit every 21 days). The number of factors is chosen by Bai–Ng on each refit's own window; it used to be chosen once on the end-of-sample window, a look-ahead fixed in the refactor. → see **Risk model**

**Arrival price** — The market price at the moment an order is submitted. It is the benchmark for measuring the impact component of execution cost: impact = signed difference between the average fill price and the arrival price. IBKR's `ArrivalPx` algorithm targets it; the router uses that algorithm for fast-decaying signals (half-life < 5 days). → see **Execution**, **Transaction cost model › TCA**

**AST (abstract syntax tree)** — The tree representation of source code that Python's `ast` module produces. The `@feature_registry` decorator parses each feature function's AST and refuses any literal `.shift(-k)`, which would read future rows. → see **Validation & statistics › Leakage tests**

**Auction (closing)** — At 4:00 pm ET, NYSE and Nasdaq match all market-on-close and limit-on-close orders at a single price, the official close. About 9% of a day's volume trades there (`auction_volume_share_default = 0.09`), and it is the price every backtest uses, so executing in the auction makes the live book match the backtest. Orders are capped at 2% of expected auction volume; the simulator prices each trade as the router would split it between the auction and the continuous session (**routed venue**). → see **Execution**

**Autocorrelation** — The correlation of a series with its own past values: $\rho_k = \operatorname{corr}(r_t, r_{t-k})$. Positive autocorrelation makes the √252 annualisation and the naive t-statistic too optimistic; overlapping multi-day returns are strongly autocorrelated by construction. → see **Validation & statistics**

## B

**Backtest** — A simulation of a trading strategy on historical data, as if it had been run live. In this repo the simulator trades at the close of day $t$ using only data up to $t - L$ (the **execution delay**, default $L = 1$), earns the return from $t$ to $t+1$, and books spread, impact, fees and borrow explicitly. A backtest is a hypothesis, not evidence, until it passes validation. → see **Backtesting**

**Backtest path** — In combinatorial purged CV, a complete out-of-sample history stitched together from the test predictions of different splits, each time group appearing once. With $N$ groups and $k$ test groups per split there are $\binom{N}{k}k/N$ paths (7 for the defaults 8 and 2). → see **Validation & statistics › Cross-validation**

**Bai–Ng criterion** — A rule for choosing how many statistical factors a return panel contains, balancing fit against model size: pick $k$ minimizing $IC_{p2}(k) = \ln V(k) + k\,\frac{N+T}{NT}\ln\min(N,T)$, where $V(k)$ is the average residual variance after $k$ factors. It is the default method in `qts/risk/statistical.py::select_n_factors`. → see **Risk model**

**Barra model** — The industry-standard family of *fundamental* factor risk models (named after the firm Barra, now MSCI). Each stock's return is explained by observable exposures — a market factor, industries and style characteristics such as size or momentum — whose factor returns are estimated each day by cross-sectional regression. This repo implements a USE4-style version (`qts/risk/model.py::BarraStyleRiskModel`). → see **Risk model**

**Bartlett kernel** — Linearly declining weights $1 - k/(L+1)$ applied to autocovariances at lags $k = 1..L$. Used by Newey–West estimators because it guarantees a non-negative variance estimate. → see **Validation & statistics › Newey–West**, **Risk model**

**Basis point (bp)** — One hundredth of a percent: 1 bp = 0.01% = 0.0001. Costs are quoted in bp of traded notional (e.g. "16.3 bp per side"); 100 bp = 1%. → see **Transaction cost model**

**Benjamini–Hochberg (BH)** — A procedure that controls the false discovery rate when running many tests: sort p-values, find the largest $k$ with $p_{(k)} \le kq/m$, and reject the first $k$. Less conservative than Bonferroni. → see **Validation & statistics › Multiple testing**

**Beta** — How much a stock moves with the market: the slope of a regression of its returns on market returns. A beta of 1.2 means it moves 1.2% on a 1% market day on average. The risk model's beta style is a 252-day regression on the equal-weighted return of the names eligible on each date, shrunk one third toward 1 (Blume/Vasicek-style). Before the lag investigation the "market" averaged every column, including names that join the universe later, a future-membership look-ahead. → see **Risk model**

**Beta neutrality** — Constraining the portfolio's net exposure to the market beta factor to (near) zero so market moves do not drive its P&L. The optimizer holds the *style* factor `beta` within ±`beta_band` = 0.05 standard deviations (`PortfolioConfig.beta_neutral = True`); before the audit it constrained the wrong column (the country factor), leaving beta exposure near 2. → see **Portfolio construction**

**Bias statistic** — The standard deviation of returns divided by their *forecast* volatility, $b = \operatorname{sd}_t(r_t/\hat\sigma_{t-1})$. An unbiased risk model scores 1; above 1 means risk was under-forecast. The plan's acceptance band is [0.88, 1.12]; the code's statistical band is $1 \pm 2/\sqrt{2T}$ (`qts/risk/evaluation.py`). → see **Risk model**

**Bid–ask spread** — The gap between the highest price a buyer offers (bid) and the lowest a seller asks (ask). Crossing it costs half the spread per trade (the **half-spread**), the most basic trading cost. Default half-spread when no data: 2 bp (`CostConfig.default_half_spread_bps`). → see **Transaction cost model**

**Bonferroni correction** — Testing each of $m$ hypotheses at level $\alpha/m$ so the chance of any false positive stays below $\alpha$. Valid under any dependence but overly strict when tests are correlated. → see **Validation & statistics › Multiple testing**

**Book-to-price (B/P)** — Accounting book value of equity divided by market capitalisation; the classic "value" measure (high B/P = cheap). Used as `value_bp` (log B/P, industry-demeaned) in the value theme and as the risk model's value style. → see **Alpha model**, **Data & universe**

**Borrow fee** — The annual fee a short seller pays to borrow shares. Most stocks are "general collateral" at well under 1%/yr; hard-to-borrow names can cost 5–50%/yr. The repo charges it daily on short positions; until logged IBKR rates exist it uses a short-interest proxy rising convexly from 0.5% (GC) to a 25% cap, doubled for caps under $2B. Loan fees are also a predictor (the short-interest theme). → see **Transaction cost model**

**Breadth** — In the fundamental law, the number of independent bets per year. Trading 1,000 stocks monthly is *not* 12,000 independent bets because stocks share factors, so effective breadth is far smaller. → see **Validation & statistics › The information coefficient**

**Breakeven cost** — The cost per side, in bp of traded notional, that would wipe out the mean return before trading costs: $c_{BE} = \overline{g^b_t} / \overline{\text{turnover}_t} \times 10^4$, where $g^b_t$ is the **gross-after-borrow** return. The hurdle requires breakeven ≥ 2× modeled cost (demo: 44.1 bp vs 2 × 10.5 bp; that demo still used plain gross returns). → see **Validation & statistics › The hurdle gates**, **Backtesting**

## C

**Calmar ratio** — Compound annual return divided by the absolute maximum drawdown. Easy to read ("I earn 0.75 of my worst loss per year") but grows less favourable with sample length, so compare only like-length backtests. → see **Validation & statistics › Drawdown, Calmar, Sortino**

**CAT fee** — The fee funding the SEC's Consolidated Audit Trail, passed through by brokers per share on both buys and sells; $0.000003 per share in `CostConfig.cat_fee_per_share`. → see **Transaction cost model**

**CatBoost** — A gradient-boosted decision tree library (Yandex) with strong defaults on tabular data. The build plan lists it alongside LightGBM; the current code implements LightGBM, MLP and ridge only. → see **Alpha model**

**Clarabel** — An open-source interior-point solver for convex conic problems, used through cvxpy as the default optimizer backend (`PortfolioConfig.solver = "CLARABEL"`). It supports the 3-dimensional **power cone** natively, which the optimizer uses for the 3/2-power impact term. Solve time, not model building, dominates the daily backtest loop. → see **Portfolio construction**

**Code version** — The identity of the code that produced a result, written to the run manifest and the trial ledger: the git commit SHA for a clean checkout, `<sha>+dirty` when tracked files are modified or untracked (non-ignored) files exist, and `unknown` outside git (`qts/core/manifest.py::code_version`). A dirty or unknown code version is never de-duplicated in the ledger, because the SHA alone cannot reproduce the run. → see **Validation & statistics › The trial ledger**, **Architecture & data flow**

**Combinatorial purged cross-validation (CPCV)** — Cut the history into $N$ groups and use every combination of $k$ groups as a test set, training on the rest after purging and embargo. It yields many out-of-sample backtest paths instead of one, i.e. a distribution of OOS performance. `qts/validation/splits.py::CombinatorialPurgedCV(n_groups=8, k_test=2)`. → see **Validation & statistics › Cross-validation**

**Commission** — The broker's charge per trade. Modeled on IBKR Pro tiered pricing: $0.0035/share, minimum $0.35 per order, capped at 1% of trade value, plus $0.0002/share clearing. → see **Transaction cost model**

**Composite (rank composite)** — The Phase-1 alpha ("Baseline A"): features are rank-gaussianised per date, averaged within each theme (value, momentum, quality, investment, low-risk, short interest), z-scored, then averaged across themes with fixed weights plus capped dynamic tilts. Simple, hard to overfit and the baseline every ML model must beat. → see **Alpha model**

**Convex optimisation** — Optimisation problems whose objective is convex and whose feasible set is convex, so any local optimum is the global one and solvers are reliable and fast. The portfolio optimizer is written so that every term — expected return, factor risk, square-root impact, borrow cost, constraints — is convex. → see **Portfolio construction**

**Correlation** — A scale-free measure of how two variables move together, from −1 to +1: covariance divided by the product of standard deviations. Pearson correlation measures linear association; Spearman correlation is Pearson on ranks. → see **Validation & statistics**

**Country factor (market factor)** — In USE4-style models, a factor to which every stock has exposure 1; its return is (by construction) the cap-weighted market return, and industry factor returns are measured relative to it. Not to be confused with the beta style factor. → see **Risk model**

**Covariance (matrix)** — How two returns vary together: $\operatorname{cov}(X,Y) = \mathbb{E}[(X-\mu_X)(Y-\mu_Y)]$. The $N\times N$ covariance matrix of stock returns determines portfolio risk ($w^\top\Sigma w$); estimating it reliably for 1,000 stocks is the risk model's whole job. → see **Risk model**

**Cross-sectional** — Across stocks on one date, as opposed to time-series (across dates for one stock). Ranks, z-scores, ICs and factor-return regressions in this system are all computed cross-sectionally, date by date. → see **Alpha model**, **Risk model**

**Cross-sectional regression** — Each day, regress all stocks' returns on their factor exposures to estimate that day's factor returns: $r_t = B_{t-1}f_t + u_t$. The repo uses weighted least squares with weights ∝ √market cap and a constraint that cap-weighted industry returns sum to zero; residuals $u_t$ are the specific returns. → see **Risk model**

**Cross-validation (CV)** — Estimating out-of-sample performance by repeatedly fitting on part of the data and testing on another part. Standard shuffled k-fold leaks with time-series labels; this repo uses purged walk-forward, purged k-fold and CPCV. → see **Validation & statistics › Cross-validation**

**CSCV (combinatorially symmetric cross-validation)** — The resampling scheme behind PBO: split the history into $S$ groups, and for every choice of half the groups as in-sample, compare the in-sample winner's out-of-sample rank. → see **Validation & statistics › PBO**

**cvxpy** — A Python modelling language for convex optimisation that checks problems follow DCP rules and passes them to a solver (here Clarabel). → see **Portfolio construction**

## D

**Data snooping** — Finding patterns that exist only in the specific sample because many hypotheses were tried on it. The cure is counting trials (ledger), deflating statistics (DSR, t > 3) and keeping a locked holdout. → see **Validation & statistics**

**Days to cover** — Short interest in shares divided by average daily share volume: how many days of normal trading it would take short sellers to buy back. Used by the borrow-fee proxy as an alternative tightness measure (2 days ↔ GC, 10 days ↔ hard-to-borrow). → see **Transaction cost model**

**DCP (disciplined convex programming)** — A set of composition rules (e.g. "a convex increasing function of a convex expression is convex") that cvxpy uses to *verify* a problem is convex. Writing impact as `cp.power(cp.abs(x), 1.5)` is DCP-compliant. → see **Portfolio construction**, **Transaction cost model**

**Decile spread** — Return of the top 10% of stocks by signal minus the bottom 10% over the next horizon; a direct "would a simple long/short bet have made money?" test. The pipeline's version is value-weighted, on eligible names, scored on the traded (lagged) signal every 20 execution days; its Newey–West t-stat is the `t_stat` gate (> 3). → see **Validation & statistics › Decile portfolios**

**Decision price** — The price at the moment the trading decision was made (signal time). Implementation shortfall measures all execution costs relative to it. → see **Transaction cost model › TCA**

**Deflated Sharpe ratio (DSR)** — The probability that a strategy's true Sharpe exceeds the Sharpe you would expect from the best of $N$ useless strategies, accounting for sample length, skew and kurtosis (Bailey & López de Prado 2014). It is a probabilistic Sharpe ratio evaluated at the expected maximum Sharpe. Gate: DSR > 0.95. It cannot detect leakage. → see **Validation & statistics › Deflated Sharpe**

**Delisting return** — The return on a stock's final days when it leaves the exchange (bankruptcy, takeover, failing listing rules). Omitting it is a survivorship bias that flatters short legs. One map, reason → return on the stock in `CostConfig` (`delisting_return_acquisition` 0, `_performance` −0.30 following the CRSP-style convention for performance delistings, `_voluntary` −0.10, `_unknown` −0.30), is used by both the labels and the backtester; a long loses on a negative return and a short gains. The reason is `MarketData.delisting_reasons`. The former single `delisting_haircut`, charged against the position on both sides, is deprecated (`delisting_against_position = True` reproduces it). A temporary halt is not a delisting (see **Halt vs delisting**). → see **Data & universe**, **Backtesting**

**Dollar-neutral** — Long and short dollar amounts are equal, so net exposure is zero. The default book targets net exposure 0 within ±5% of NAV (`net_exposure_target = 0.0`, `net_exposure_band = 0.05`). → see **Portfolio construction**

**Downside deviation** — The root mean square of returns below a target, $\sqrt{\tfrac1T\sum\min(r_t-\tau,0)^2}$; the denominator of the Sortino ratio. → see **Validation & statistics**

**DPP (disciplined parametrized programming)** — A cvxpy feature: write a problem with `Parameter` objects in place of data, and cvxpy compiles ("canonicalises") it once and re-solves it with new parameter values without recompiling. The optimizer builds its problem in DPP form and rewrites the routed venue's kink terms so they stay DPP; it re-parametrises one compiled problem during the γ search. Reusing it across days is off by default (`PortfolioConfig.reuse_problem = False`) because cvxpy's DPP compile of this problem needs memory that grows like $N^2$ (about 0.8 GB at 300 names, more than 7 GB at 1,000). → see **Portfolio construction**

**Drawdown** — The percentage fall of NAV from its previous peak: $DD_t = NAV_t/\max_{s\le t}NAV_s - 1$. Always ≤ 0; the most negative value is the maximum drawdown. → see **Validation & statistics › Drawdown**

## E

**Earnings yield (E/P)** — Earnings divided by price; a value measure, the inverse of the P/E ratio. Used as `value_ep` (industry-demeaned) and as the value style fallback when B/P is missing. → see **Alpha model**

**Effective holding period** — Average gross book divided by average one-sided daily trading: how many days a position lives on average. Target 50 days (`effective_holding_days_target`); runs below 60% of target are flagged (review item 5). Demo: 50 days. → see **Portfolio construction**, **Backtesting**

**Effective number of trials** — The number of independent tests that a set of correlated trials is worth. Nyholt's estimate from the eigenvalues of the trials' return correlation matrix: $M_{\text{eff}} = 1 + (M-1)(1 - \operatorname{Var}(\lambda)/M)$. Used as $N$ in the DSR when the ledger family has ≥ 3 trials with monthly return streams. It is usually fractional (e.g. 1.67) and is used unrounded: rounding down made the DSR more lenient (second audit). → see **Validation & statistics › Multiple testing**

**Eigenfactor adjustment** — A USE4 correction for the fact that sample covariance matrices systematically underestimate the risk of their own smallest-variance directions (eigenportfolios). Monte Carlo simulation (`n_eigen_sims = 200`) measures that bias per eigenvalue and scales the eigenvalues up accordingly (`RiskConfig.eigenfactor_adjust = True`), inflated by USE4's empirical factor $a = 1.2$ (`eigen_inflation_a`). The simulated bias is cached per refresh bucket and estimated only from history up to the bucket's start; it used to depend on which date filled the cache first (an order-dependent look-ahead fixed in the refactor). → see **Risk model**

**Eigenvalue / eigenvector** — For a covariance matrix, the eigenvectors are uncorrelated "directions" of portfolios and the eigenvalues their variances. PCA, the eigenfactor adjustment, Marchenko–Pastur, Bai–Ng and Nyholt's $M_{\text{eff}}$ all work with eigenvalues. → see **Risk model**

**Embargo** — Extra training observations dropped *after* a test block (beyond purging) so that serially correlated information cannot flow from the test period into training. The ML trainer's embargo is the label horizon plus the execution delay: 20 + 1 = 21 trading days by default. → see **Validation & statistics › Cross-validation**, **Alpha model**

**Ensemble** — Averaging the predictions of several models (different seeds or algorithms) to reduce variance. The ML sleeve averages seeds within a model family (`model_seeds = 3`) and combines families by rank. → see **Alpha model**

**Euler–Mascheroni constant (γ ≈ 0.5772)** — The limit of $\sum_{k=1}^n 1/k - \ln n$, and the mean of the standard Gumbel distribution. It weights the two quantiles in the expected-maximum-Sharpe formula. → see **Validation & statistics › Deflated Sharpe**

**EWMA (exponentially weighted moving average)** — An average that weights recent observations more, with weights decaying by a factor $\lambda$ per day; the half-life is the lag at which the weight halves ($\lambda = 0.5^{1/\text{hl}}$). The risk model uses EWMA covariances with half-lives 84 days (volatility), 504 days (correlation) and 42 days (VRA). → see **Risk model**

**Execution delay (signal lag)** — The number of trading days $L$ between the data a decision uses and the close at which it is executed (`ExecutionConfig.signal_lag_days`, default 1). With $L = 1$ the book traded at the close of $t$ is built from alpha, risk and cost data as of $t-1$, because real market-on-close orders must be submitted by 15:50 ET, before the close they trade at. Labels, the trainer's embargo and the validation statistics are aligned to the same timeline. $L = 0$ (close-as-signal) remains available as an optimistic bound. → see **Backtesting**, **Validation & statistics › The information coefficient**

**Expected maximum Sharpe** — The Sharpe ratio the best of $N$ zero-skill strategies is expected to show by luck: $\sqrt V\big((1-\gamma)\Phi^{-1}(1-1/N) + \gamma\Phi^{-1}(1-1/(Ne))\big)$. It is the benchmark the DSR deflates against. → see **Validation & statistics › Deflated Sharpe**

**Expected return** — The mean return a model forecasts for a stock over a horizon. Here, the alpha model's expected *residual* return over 20 days, $\alpha_i = IC\cdot\sigma_i\cdot z_i$ (Grinold's rule), which the optimizer trades against risk and cost. → see **Alpha model**, **Portfolio construction**

**Exposure** — How strongly a stock or portfolio responds to a factor: a stock's exposure to the size factor is its standardised log market cap; a portfolio's exposure is the weighted sum $B^\top w$. Constraining exposures keeps the book from making unintended factor bets. → see **Risk model**, **Portfolio construction**

## F

**Factor** — A common driver of many stocks' returns: the market, an industry, or a characteristic such as size, value or momentum. Factor models describe each stock's return as exposures × factor returns + a stock-specific part. → see **Risk model**

**Factor covariance** — The $K\times K$ covariance matrix $F$ of factor returns. Portfolio risk in a factor model is $w^\top(BFB^\top + D)w$. Estimated with EWMA plus Newey–West, VRA and the eigenfactor adjustment. → see **Risk model**

**Factor momentum** — The tendency of factors that did well recently to keep doing well. The composite tilts theme weights by trailing one-month theme performance, capped at ±25% (`factor_momentum_max_tilt = 0.25`, review item 6). Evidence: Gupta & Kelly report Sharpe 0.84 for a combined factor-momentum strategy (in-sample, via `BUILD_PLAN.md` §2.1); no post-publication test was found. → see **Alpha model**

**Factor return** — The return attributed to one unit of exposure to a factor on a given day, estimated by the cross-sectional regression. → see **Risk model**

**False discovery rate (FDR)** — The expected fraction of "discoveries" that are false. Controlled by Benjamini–Hochberg; a more useful criterion than family-wise error for research pipelines that admit many signals. → see **Validation & statistics › Multiple testing**

**Fama–French 48 (FF48)** — A classification of US firms into 48 industries from their SIC codes, published by Kenneth French. This repo uses FF48 as the industry layer (`qts.data.loaders.sic_to_ff48`) instead of GICS, whose history is not freely available point-in-time (review item 3). → see **Data & universe**, **Risk model**

**Family-wise error rate (FWER)** — The probability of at least one false positive among $m$ tests: $1-(1-\alpha)^m$ for independent tests (64% for 20 tests at 5%). → see **Validation & statistics › Multiple testing**

**Feature** — An input variable to the alpha model, computed for each stock and date from data available at that date (e.g. 12-1 momentum, Amihud illiquidity). The repo builds 18 point-in-time panels in `qts/alpha/features.py`. → see **Alpha model**

**FINRA short interest** — Twice-monthly reports of the number of shares sold short per stock, published by FINRA. The repo uses it, with a publication lag of 8 **NYSE sessions** after the settlement date, as the history for the short-interest theme and borrow proxy until ≥ 12 months of IBKR borrow rates are logged (review item 2). → see **Data & universe**

**FINRA TAF (Trading Activity Fee)** — A per-share regulatory fee on sales: $0.000195 per share sold, capped at $9.79 per trade (`CostConfig`), with a dated table in `qts/tcost/fees.py` including a $0 fee holiday for 2026-10-01 to 2026-12-31. → see **Transaction cost model**

**Fold** — One train/test partition in cross-validation, or one evaluation period. The pipeline's gate folds are the contiguous test blocks of a `PurgedWalkForward` splitter over the backtest dates (`ValidationConfig.n_splits = 6`, blocks of at least `min_fold_days = 20`); they used to be calendar years. → see **Validation & statistics › The hurdle gates**

**Fundamental law of active management** — Grinold's approximation $IR \approx IC\sqrt{BR}$: skill per bet times the square root of the number of independent bets; with constraints, $IR \approx TC\cdot IC\sqrt{BR}$ (Clarke, de Silva & Thorley). An upper bound and reasoning tool, not a forecast. → see **Validation & statistics › The information coefficient**

## G

**Garden of forking paths** — Gelman & Loken's term for the many data-dependent choices a researcher makes without consciously running multiple tests, each of which silently multiplies the effective number of trials. → see **Validation & statistics › Why validation is the core**

**Gârleanu–Pedersen (GP)** — A dynamic trading theory: with quadratic costs, trade partially toward an "aim" portfolio and weight each signal by how slowly it decays relative to how slowly you can trade. The repo uses its shrink factor $g_k = 1/(1 + H/\text{hl}_k)$, where $H$ is the cost-implied holding period and $\text{hl}_k$ the signal's half-life, to down-weight fast signals. → see **Portfolio construction**

**GBDT (gradient-boosted decision trees)** — An ensemble of shallow decision trees, each fitted to the errors of the previous ones. The default winner on tabular data in competitions and in the build plan's evidence; implemented here with LightGBM. → see **Alpha model**

**General collateral (GC)** — Stocks that are easy to borrow for shorting, at a low fee. The proxy charges 0.5%/yr below 5% short-interest ratio (`default_borrow_fee_annual = 0.005`). → see **Transaction cost model**

**Global minimum variance (GMV) portfolio** — The fully invested portfolio with the lowest predicted variance, $w \propto \Sigma^{-1}\mathbf 1$. Its realised out-of-sample variance is a standard test of covariance estimators (`min_variance_oos_test`). → see **Risk model**

**Grinold's rule** — Converts a standardised score into an expected return: $\alpha_i = IC\cdot\sigma_i\cdot z_i$, where $\sigma_i$ is residual volatility over the alpha's horizon and $z_i$ the score's z-score. The estimated IC is shrunk 50% toward zero (`ic_shrink = 0.5`, fallback `default_ic = 0.03`). → see **Alpha model**, **Portfolio construction**

**Gross after borrow** — The daily gross return minus that day's borrow fee as a fraction of NAV, $g_t - \text{borrow}_t/NAV_t$ (`BacktestResult.returns["gross_after_borrow"]`). Borrow is a holding cost that does not scale with turnover, so breakeven cost and the cost-sensitivity table start from this series; starting from plain gross left borrow out of the breakeven gate (second audit). → see **Backtesting**, **Validation & statistics › The hurdle gates**

**Gross exposure / net exposure** — Gross exposure is the sum of absolute position sizes as a fraction of NAV, $\sum|w_i|$; net exposure is the signed sum $\sum w_i$. The default book allows gross up to 2.0 (`gross_leverage_max`) with net near 0; demo: gross 2.00, net +0.04. → see **Portfolio construction**

**Gross profitability (GP/A)** — Gross profit divided by total assets; a quality measure (Novy-Marx). Used as `quality_gpa` (industry-demeaned) and as the risk model's quality style. → see **Alpha model**

**Gross vs net return** — Gross return is before trading costs and borrow; net return is after. Every hurdle except the IC and decile tests uses net. → see **Backtesting**

## H

**Half-life** — The time for something to decay to half its size. For EWMA weights, the lag at which weight halves; for a signal, the lag at which its rank autocorrelation halves (`signal_half_life`), estimated by fitting $\rho(\ell)=e^{-\lambda\ell}$ and returning $\ln2/\lambda$. → see **Alpha model**, **Risk model**

**Halt vs delisting** — A name with no closing price may be *halted* (trading suspended, resumes later) or *delisted* (gone for good). Only a delisting gets the delisting return of its reason in the label. The decision may only use information available when it is made: with a vendor `delisting_dates` table, a name is delisted at its last trading date; otherwise the causal gap rule calls it a halt if a close reappears within `AlphaConfig.delisting_gap_days = 63` days and a delisting if none does. The backtest simulator uses the same rule: a halted position is frozen at its last price, and the delisting return and liquidation are booked only once the rule says delisted. The old rule looked to the end of the panel, so a halt that later resumed was a "gap" in a backtest and a "delisting" live (second audit finding 1, HIGH). → see **Alpha model › Targets**, **Data & universe**, **Backtesting › Halts and delistings**

**Hard-to-borrow (HTB)** — Stocks with scarce lendable supply and high borrow fees. The proxy caps fees at 25%/yr (`htb_fee_cap_annual`) at or above 20% short-interest ratio, doubled for small caps. → see **Transaction cost model**

**Hit rate** — The fraction of non-zero periods with positive return (for P&L) or positive IC (for signals). A 52% daily hit rate can be a very good strategy; it says nothing about the size of wins and losses. → see **Validation & statistics**

**Holding cost** — The cost of *keeping* a position, as opposed to trading it: in this system, the borrow fee on shorts, charged per day and scaled by the horizon in the optimizer (`hold_aversion = 1.0`). → see **Transaction cost model**, **Portfolio construction**

**Holdout** — Data set aside and never looked at during research, used once for a final check. The build plan calls for one locked holdout; it is a process control, not code. → see **Validation & statistics**

**Hurdle gate** — A pre-registered pass/fail test a candidate signal or model must clear: decile t > 3, IC sign, incremental IC, net Sharpe delta, breakeven ≥ 2× cost, DSR > 0.95, PBO < 0.5, phase Sharpe, suspect flag. Each reports PASS, FAIL or skipped. → see **Validation & statistics › The hurdle gates**

**Hysteresis** — Using different thresholds to enter and leave a state so membership does not flicker. A stock enters the universe as soon as it qualifies but leaves only after failing 3 consecutive monthly refreshes, reducing forced trading. → see **Data & universe**

## I

**IBKR (Interactive Brokers)** — The broker the system targets: its Pro tiered commission schedule is the fee model, its algorithms (ClosePx, Vwap, ArrivalPx, Adaptive) are the execution tactics, and its indicative borrow rates are the future borrow-fee source. → see **Execution**, **Transaction cost model**

**IC (information coefficient)** — The cross-sectional correlation, on one date, between a signal's scores and the subsequent returns; this repo uses the Spearman (rank) version against residualised 20-day returns. An IC of 0.03–0.05 is typical of useful equity signals. → see **Validation & statistics › The information coefficient**

**IC decay** — How a signal's IC changes as the forecast horizon lengthens; slow-decaying signals tolerate slow, cheap trading. Computed by `ic_decay_curve` for horizons 1–120 days. → see **Alpha model**, **Validation & statistics**

**ICIR** — Mean IC divided by the standard deviation of the IC series, annualised by the square root of the number of IC observations per year: the "Sharpe ratio" of a signal's skill. Annualising a 20-day-sampled series with √252 instead of √12.6 inflates it 4.47× (audit finding 4, fixed). The pipeline infers the sampling frequency once and uses it for both the ICIR and the decile Sharpe. → see **Validation & statistics › The information coefficient**

**Idiosyncratic volatility** — The volatility of the part of a stock's return not explained by the market (or by the factor model). The risk model's `residual_vol` style is $\sqrt{\operatorname{var}(r) - \operatorname{cov}(r,m)^2/\operatorname{var}(m)}$ over 252 days; the low-risk theme favours low idiosyncratic volatility. → see **Risk model**, **Alpha model**

**iid (independent and identically distributed)** — Each observation is drawn from the same distribution, independently of the others. Most textbook formulas (√T standard errors, √252 annualisation) assume it; financial returns violate it through volatility clustering and, for overlapping returns, autocorrelation. → see **Validation & statistics**

**Implementation shortfall** — The total cost of executing a decision, measured against the decision price: delay (price drift before submission) + impact (fill price vs arrival) + opportunity cost (unfilled shares) + explicit fees (Perold decomposition). Logged per order by the TCA ledger. → see **Transaction cost model › TCA**

**In-sample (IS)** — Data used to build, fit or select a model. Performance measured in-sample is optimistic by construction. → see **Validation & statistics**

**Incremental IC** — The IC of a new signal after regressing out (per date) the existing composite: what the new signal adds that the system does not already have. Gate: > 0.005 in ≥ 2/3 of folds. → see **Validation & statistics › The hurdle gates**

**Industry neutrality** — Limiting the portfolio's net exposure to each industry so P&L does not hinge on sector bets. The optimizer keeps each industry's net weight within ±3% of NAV (`industry_band = 0.03`), resolved via `RiskModelOutput.factor_groups["industry"]`. → see **Portfolio construction**

**Information ratio (IR)** — Mean active return divided by tracking error, annualised; for a market-neutral book it equals the Sharpe ratio. → see **Validation & statistics**

**Investment (asset growth)** — Firms that grow assets aggressively tend to earn lower subsequent returns. The investment theme uses negative asset growth (`investment_ag`, industry-demeaned). → see **Alpha model**

## K

**Kill switch** — A pre-defined rule that halts trading when live behaviour breaches a band (e.g. tracking error or slippage). Planned for Phases 4–5 in the build plan; not implemented in code. → see **Execution**

**Kurtosis** — A measure of tail heaviness: $\gamma_4 = \mathbb{E}[(r-\mu)^4]/\sigma^4$, equal to 3 for a normal distribution ("raw"); **excess kurtosis** is $\gamma_4 - 3$. pandas returns excess; the DSR functions expect raw. → see **Validation & statistics › Return statistics**

## L

**Label (target)** — The quantity a model is trained to predict. Here, the 20-day forward return residualised against risk-only exposures known at $t$ (beta, size, residual volatility, liquidity, short-term reversal, industries), then z-scored per date. With an execution delay $L$ the window is $\text{close}_{t+L} \to \text{close}_{t+L+20}$, the return the traded book can actually earn. Labels are never features. → see **Alpha model**

**Label realisation** — The first date on which a label row is fully known and may be used: at least $s + L + h$ for signal date $s$, later if a halted name's halt-or-delisting status is still unresolved (`qts/alpha/targets.py::label_realisation`). The walk-forward trainer and the trailing IC used for Grinold scaling only consume rows realised by their own date. → see **Alpha model**, **Validation & statistics › Leakage tests**

**Ledoit–Wolf shrinkage** — Estimators that pull a noisy sample covariance toward a structured target to reduce estimation error. *Linear* Ledoit–Wolf (2004) blends with a scaled identity using an optimal weight; *nonlinear* shrinkage (QIS) adjusts each eigenvalue separately. → see **Risk model**

**Leverage** — Holding positions whose total size exceeds capital. A dollar-neutral book with gross exposure 2.0 is "2× levered" in gross terms even though its net market exposure is zero. → see **Portfolio construction**

**LightGBM** — Microsoft's fast gradient-boosted decision tree library, the ML sleeve's main model in Phase 2 (`qts/alpha/models.py`), trained walk-forward with early stopping on a purged validation block. On the synthetic market it sat at the noise floor (review notes). → see **Alpha model**

**Limit order** — An order to buy at or below (sell at or above) a stated price. It controls price but may not fill. → see **Execution**

**Liquidity** — How easily a stock can be traded without moving its price. Proxied by turnover, dollar volume, Amihud illiquidity and spread; the risk model's liquidity style is log 63-day turnover. → see **Alpha model**, **Transaction cost model**

**LOC / MOC (limit-on-close / market-on-close)** — Orders that execute in the closing auction. MOC fills at the closing price whatever it is; LOC fills only if the close is at or better than a limit. The router sends small orders as LOC with a limit 3 bp through the indicative price (`loc_offset_bps = 3.0`, floored at 1.5× expected auction impact) before the 15:50 ET cutoff. → see **Execution**

**Logit** — The log-odds transform $\ln\frac{w}{1-w}$, mapping (0, 1) to the real line. In PBO it turns the IS-winner's OOS relative rank into a number that is negative when the winner is below the OOS median. → see **Validation & statistics › PBO**

**Long/short** — A book that buys (goes long) stocks expected to outperform and sells short stocks expected to underperform. Profit comes from the spread between them rather than from the market's direction. → see **Portfolio construction**

**Long-run variance** — The variance of a sample mean scaled by $T$ when observations are autocorrelated: $\gamma_0 + 2\sum_k\gamma_k$. Newey–West estimates it. → see **Validation & statistics › Newey–West**

**Look-ahead bias** — Using information in a backtest that would not have been available at decision time (future prices, restated fundamentals, future index membership). It produces spectacular, fictitious results and is invisible to DSR; the repo prevents it structurally and tests for it with PIT checks. → see **Validation & statistics › Leakage tests**, **Data & universe**

**Low risk (low volatility)** — The anomaly that low-beta, low-volatility stocks earn higher risk-adjusted returns than high-risk ones. The low-risk theme uses beta, idiosyncratic volatility and maximum daily return. → see **Alpha model**

## M

**Marchenko–Pastur** — The distribution of eigenvalues of a sample correlation matrix built from pure noise: they fall within $[(1-\sqrt{N/T})^2, (1+\sqrt{N/T})^2]\sigma^2$. Eigenvalues above the upper edge are evidence of real factors; `select_n_factors(method="marchenko_pastur")` counts them. → see **Risk model**

**Market capitalisation** — Share price × shares outstanding: a company's equity market value. Used for size, value weighting, universe screens and regression weights. → see **Data & universe**

**Market impact** — The adverse price move caused by your own trading: buying pushes the price up before you finish. It grows with the square root of order size relative to volume and is usually the largest cost for institutional-size orders. → see **Transaction cost model**

**Max drawdown** — The largest peak-to-trough fall of NAV over the sample; demo −5.3%. It grows with sample length. → see **Validation & statistics › Drawdown**

**Mean–variance optimisation** — Choosing weights to maximise expected return minus a penalty on variance, $\alpha^\top w - \gamma\,w^\top\Sigma w$ (Markowitz). The optimizer is a mean–variance problem extended with cost terms and constraints; by default its risk aversion $\gamma$ is searched per date so the book hits the 10% volatility target (**target-vol risk aversion**). → see **Portfolio construction**

**Metaorder** — A large parent order executed as many child orders, possibly over several days. The square-root law applies to the metaorder as a whole, not to each child. → see **Transaction cost model**

**Metaorder streak** — Consecutive days of same-direction trading in one name, treated as one metaorder for impact (`CostConfig.impact_persistence = True`). The simulator tracks the cumulative dollars $Q_{k-1}$ already traded in the streak and charges day $k$ the exact increment $C(Q_{k-1} + q_k) - C(Q_{k-1})$ of the integrated cost $C(Q) = b\sigma Q^{1.5}/\sqrt V$. The older shortcut $k^{1.5}-(k-1)^{1.5}$ times the one-day impact assumes equal daily trades; it is only a fallback and is not a bound. The streak resets when the sign flips or a day has no trade. → see **Transaction cost model**

**Microcap** — Very small companies (the bottom NYSE size quintile or below). Anomaly returns concentrate there but cannot be captured after costs; the universe excludes the bottom 20% by size (`exclude_bottom_size_quantile = 0.20`). → see **Data & universe**

**Mincer–Zarnowitz regression** — Regressing realised squared returns on predicted variance, $r^2 = a + b\hat\sigma^2 + e$; a good forecast has $a = 0$, $b = 1$. A risk-model diagnostic in `qts/risk/evaluation.py`. → see **Risk model**

**Minimum backtest length (MinBTL)** — How long a backtest must be before the best of $N$ pure-noise configurations is no longer expected to reach a target Sharpe: $n_{\min} = (z_N/SR_{\text{target}})^2$. For target annual Sharpe 1.0: 2.5 years for 10 trials, 6.4 for 100. → see **Validation & statistics › Deflated Sharpe**

**Minimum track record length (MinTRL)** — How many observations are needed to be 95% confident the true Sharpe exceeds a benchmark, given the observed Sharpe, skew and kurtosis. Annual Sharpe 1.0 vs 0 needs about 2.7 years; 0.5 needs about 10.8. → see **Validation & statistics › Deflated Sharpe**

**MLP (multi-layer perceptron)** — A feed-forward neural network. The ML sleeve includes scikit-learn's `MLPRegressor` as one model family. → see **Alpha model**

**Momentum** — Stocks that rose over the past year tend to keep outperforming for months. The standard measure is 12-1 momentum, $P_{t-21}/P_{t-252} - 1$, which skips the most recent month (that belongs to reversal). The momentum theme averages 12-1, 6-1 and industry momentum; it is the highest-Sharpe single theme in the evidence but crashes periodically. → see **Alpha model**

**Momentum crash control** — Scaling the momentum theme down (to as low as 0.25×) when the trailing 24-month equal-weight market return is negative and 21-day market volatility is above its 80th percentile, plus vol-targeting the momentum sleeve to 12% (review item 8). → see **Alpha model**

**Multiple testing** — Running many hypothesis tests on the same data, which inflates the chance of false positives unless thresholds are adjusted (Bonferroni, BH, t > 3, DSR). → see **Validation & statistics › Multiple testing**

## N

**Nagel critique / collapse test** — Nagel (2025) shows that very high-capacity models fitted on short windows effectively become recency-weighted momentum strategies regardless of the data. The repo's `nagel_collapse_test` fits a model on synthetic noise and fails it if it shows IC (|t| ≥ 3) or its predictions correlate > 0.5 with a recency-weighted mean of past returns. → see **Validation & statistics › Leakage tests**

**NAV (net asset value)** — The value of the portfolio's capital: cash plus positions minus liabilities. Weights are fractions of NAV; the demo uses NAV = $5M so costs reflect impact rather than order minimums. → see **Backtesting**

**Newey–West standard error** — A t-statistic denominator robust to autocorrelation and heteroskedasticity, using Bartlett-weighted autocovariances up to $L$ lags. Essential for overlapping multi-day returns, whose naive t-stats are inflated by about $\sqrt h$. The repo's lag rule is $L = \min(\max(h, \lfloor 4(T/100)^{2/9}\rfloor), T-1)$ with $h$ the overlap; the old $L = h-1$ under-covered the overlap because the Bartlett weights shrink exactly those lags. → see **Validation & statistics › Newey–West**

**No-trade list** — Names the optimizer is told not to trade on a given day (e.g. halted stocks, or ineligible names that are not held), enforced as $x_i = 0$ constraints. → see **Portfolio construction**

**Null hypothesis** — The default assumption a test tries to reject, typically "this strategy has zero true edge". → see **Validation & statistics › Hypothesis testing**

**NYSE session** — A day on which the New York Stock Exchange is open. Lags stated "in sessions" skip weekends *and* exchange holidays (`qts/data/calendar.py`, a built-in NYSE calendar for 2000–2035). `pandas`' business-day offset ignores holidays and was early by one day per holiday spanned (second audit: a FINRA publication date two trading days early). → see **Data & universe**

## O

**OLS / WLS (ordinary / weighted least squares)** — Fitting a linear model by minimising the (weighted) sum of squared residuals. Factor returns use WLS with √cap weights; residualisation of labels and incremental IC use per-date OLS. → see **Risk model**, **Alpha model**

**Opportunity cost** — In implementation shortfall, the cost of shares you intended to trade but did not, measured as the price move from decision to close on the unfilled quantity. → see **Transaction cost model › TCA**

**Out-of-model name** — A stock with an expected return or a position but no row in the risk model (no factor exposures, no specific variance), e.g. a fresh listing without a market cap. The optimizer only holds or trades names in the risk model: held out-of-model names are forced toward zero with a conservative specific variance, and the rest are dropped. Before the second audit the optimizer used 47 such names with zero alpha as "free" hedges for the net band, because their market risk was invisible to it. → see **Portfolio construction**

**Out-of-sample (OOS)** — Data not used to build or select the model. Only OOS performance is evidence of skill, and only if no information from it leaked into the choices. → see **Validation & statistics**

**Overfitting** — Fitting noise in the sample rather than the underlying relationship, so performance collapses out of sample. In finance it mostly happens through selection among many variants (backtest overfitting), which PBO and DSR measure. → see **Validation & statistics**

**Overlapping returns** — Multi-day returns sampled more often than their length (e.g. 20-day returns every day), so consecutive observations share most of their days. They are strongly autocorrelated and inflate naive t-stats by about $\sqrt h$. → see **Validation & statistics › Newey–West**

## P

**p-value** — The probability, if the null hypothesis were true, of a result at least as extreme as the one observed. It is not the probability that the strategy is fake, and it is invalid when the test was chosen after many tries. → see **Validation & statistics › Hypothesis testing**

**Parquet / DuckDB store** — The local point-in-time data store: datasets saved as partitioned Parquet files (columnar on disk) and queried with DuckDB (an embedded SQL engine). → see **Data & universe**

**Participation rate** — Your share of the market's traded volume while executing, e.g. 5% of ADV. Higher participation means faster execution and more impact. → see **Execution**, **Transaction cost model**

**PBO (probability of backtest overfitting)** — The estimated probability that the configuration that looked best in-sample ranks below the median out-of-sample, computed by CSCV. Gate: PBO < 0.5; skipped unless a configuration grid is supplied. → see **Validation & statistics › PBO**

**PBO grid** — The set of configuration overrides (`SystemConfig.pipeline_pbo_configs`) the pipeline re-runs to compute PBO. Every grid run is logged to the trial ledger as a trial of the same family. → see **Validation & statistics › PBO**

**PCA (principal component analysis)** — Finding the orthogonal directions of maximum variance in data via the eigen-decomposition of its covariance matrix. Statistical risk models are PCA on returns; APCA is the variant for $N > T$. → see **Risk model**

**Phase (roadmap phase)** — The build plan's stages: 0 data, 1 factor composite with risk and cost models, 2 ML alpha, 3 text/alt data, 4 paper trading, 5 live. The phase sets the net Sharpe bar: 0.3 for phase 1, 0.5 for phase 2 and later. → see **Start here**, **Validation & statistics › The hurdle gates**

**Point-in-time (PIT)** — Data as it was known on each historical date, including original (unrestated) values, filing dates and index membership at the time. Every feature, label, risk estimate and cost snapshot at $t$ must use only data with timestamp ≤ $t$; `pit_shift_test` checks it. → see **Data & universe**, **Architecture & data flow**

**Portfolio margin** — A risk-based margin regime that allows more leverage than Reg T for hedged books; the review notes it would be needed for more than 2× gross. → see **Portfolio construction**

**POV (percentage of volume)** — An execution schedule that trades a fixed fraction of market volume as it occurs (`pov_schedule`). → see **Execution**

**Power cone** — The convex set $\{(x, y, z): x^{p} y^{1-p} \ge |z|,\ x, y \ge 0\}$ for $0 < p < 1$. It expresses terms like $|x|^{3/2}$ exactly, which is how the optimizer writes the square-root-law impact cost; cvxpy's default instead approximates rational powers with second-order cones. The second audit found the two agree to 1.3e-6 in weights. → see **Portfolio construction**, **Transaction cost model**

**Power (statistical)** — The probability that a test detects a real effect. Low for moderate Sharpe ratios over a few years: a true Sharpe of 0.5 over four years has $t \approx 1$. → see **Validation & statistics › Hypothesis testing**

**Probabilistic Sharpe ratio (PSR)** — The probability that the true Sharpe exceeds a benchmark, $\Phi((\widehat{SR} - SR^*)/\operatorname{se})$ with the skew- and kurtosis-aware standard error. DSR is PSR with the benchmark set to the expected maximum of $N$ trials. → see **Validation & statistics › Deflated Sharpe**

**Property-based test** — A test that states a property ("the risk model at $t$ is the same whether fitted on all data or on data up to $t$") and lets a library (here **hypothesis**) generate the inputs, such as cut dates. The repo derandomises it so CI runs the same examples every time (`tests/test_pit_properties.py`). → see **Validation & statistics › Leakage tests**

**Purging** — Removing from the training set every observation whose label window overlaps a test observation's label window, so the model cannot learn the test answer. → see **Validation & statistics › Cross-validation**

## Q

**QIS (quadratic-inverse shrinkage)** — Ledoit & Wolf's analytical nonlinear shrinkage of covariance eigenvalues, which corrects each eigenvalue by an amount that depends on its position in the spectrum. Available for evaluation (`shrunk_residual_covariance(method="qis")`) but not on the production path: the former `RiskConfig.use_nonlinear_shrinkage` flag never affected the specific variances and was removed, and a QIS-based prior lost to the USE4 structural model out of sample on synthetic seeds (QLIKE 0.07 vs 0.15–0.47). → see **Risk model**

**QLIKE** — A loss function for variance forecasts, $\text{mean}(r^2/\hat\sigma^2 - \ln(r^2/\hat\sigma^2) - 1)$, robust to noise in the realised-variance proxy; the risk model's primary reported loss (Patton 2011). → see **Risk model**

**Quality** — Profitable, stable, conservatively financed firms tend to outperform. In code the quality theme currently has one feature, gross profitability (`quality_gpa`, industry-demeaned); the build plan also lists ROE and accruals. → see **Alpha model**

**Quantile portfolio** — Stocks sorted by a signal and split into equal-sized groups (quintiles, deciles); decile spreads are one case. → see **Validation & statistics › Decile portfolios**

## R

**Rank-gauss** — A transform that replaces each value by its cross-sectional rank, then maps ranks to normal quantiles: $\Phi^{-1}((\text{rank}\cdot n - 0.5)/n)$, clipped to about ±3.7. It removes outliers and makes every feature comparable (`qts/core/crosssection.py::rank_gauss`). → see **Alpha model**

**Rebalance** — Trading the portfolio back to target weights. The simulator considers rebalancing at every close; costs and the optimizer's trade aversion determine how much it actually trades. → see **Backtesting**

**Reg T** — The Federal Reserve's Regulation T, which sets initial margin at 50% for purchases in a standard margin account, effectively limiting a retail long/short book to about 2× gross. The default `gross_leverage_max = 2.0` reflects it (review item 7). → see **Portfolio construction**

**Regularisation** — Penalising model complexity (e.g. large coefficients) to reduce overfitting; ridge uses an L2 penalty, early stopping limits boosting rounds. → see **Alpha model**

**Residual return** — The part of a stock's return not explained by the factor model: $u_t = r_t - B_{t-1}f_t$; also called specific return. The alpha model predicts residual returns because the factor part is cheaply available or deliberately hedged. → see **Risk model**, **Alpha model**

**Residualisation** — Removing the part of a variable explained by other variables via regression and keeping the residual. Labels are residualised on risk-only exposures (never on alpha themes — audit finding 2); incremental IC residualises a new signal on the composite. → see **Alpha model**

**Reversal (short-term)** — Stocks that rose sharply over the past month tend to fall back over the next. Used as a risk style (`short_term_reversal`, 21-day return) and as an industry-demeaned timing feature (`strev_21`), not as a standalone alpha. → see **Alpha model**, **Risk model**

**Reversion (TCA)** — Price movement back against a fill after the trade (e.g. a buy followed by a lower next open): the temporary part of impact. Reported but never credited as a benefit. → see **Transaction cost model › TCA**

**Ridge regression** — Linear regression with an L2 penalty on coefficients, $\min\|y - X\beta\|^2 + \lambda\|\beta\|^2$, which stabilises estimates with many correlated features. One ML-sleeve family ("Baseline B" in the plan). → see **Alpha model**

**Risk aversion** — The weight $\gamma$ on predicted variance in the optimizer's objective; higher values give smaller, more diversified books. It is an *annual* risk aversion: the unconstrained optimum has annual volatility $IR/(2\gamma)$. By default it is searched per date (`risk_aversion_mode = "target_vol"`, see **Target-vol risk aversion**); `risk_aversion = 2.0` is used in `"fixed"` mode and as a fallback. The former fixed default of 10 sized the book 3–5× too small and let an alpha-blind projection lever it up (second audit finding 1, HIGH). → see **Portfolio construction**

**Risk model** — The component that forecasts the covariance of stock returns, here as $BFB^\top + D$ (exposures, factor covariance, specific variances) plus optional statistical and alignment terms. → see **Risk model**

**Rolling window** — A fixed-length window of the most recent observations that moves forward each day (e.g. a 63-day ADV). Contrast with an expanding window, which keeps all history (used by the ML trainer). → see **Alpha model**, **Data & universe**

**Routed venue** — The default way the simulator and optimizer price a trade (design decision D3): the part up to the auction cap ($2\%$ of expected auction volume) is priced in the closing auction, the remainder in the continuous session, as the execution router would send it. Impact is charged on the whole day's metaorder, so the split saves the half-spread on the auction part but no impact under the default priors. The router itself splits orders the same way (second audit). → see **Transaction cost model**, **Execution**

**Run manifest** — `manifest.json` written with every pipeline run (`qts/core/manifest.py::RunManifest`): run id, creation time, git SHA and dirty flag, config hash, data hash, library versions, Python version, platform, stage timings and seeds. It answers "what exactly produced these numbers?". → see **Architecture & data flow**

**Run registry** — The directory of saved runs, `runs/<run_id>/` with `run_id = <UTC timestamp>_<config hash[:8]>`, each holding the manifest, `config.yaml`, `summary.json`, `hurdles.json` and Parquet frames (NAV, weights, trades, costs, returns, IC). `load_run` reloads one as a typed `RunRecord`, `list_runs` lists them newest first, and the API serves them to the UI. → see **Architecture & data flow**, **Operating guide**

## S

**SEC Section 31 fee** — A transaction fee the SEC levies on stock sales, passed through by brokers: $20.60 per $1M sold from 2026-04-04 in `CostConfig.sec_fee_per_dollar_sold`, with a dated table in `qts/tcost/fees.py`. → see **Transaction cost model**

**Seed bagging** — Training the same model with several random seeds and averaging predictions to reduce variance from random initialisation and sampling (`AlphaConfig.model_seeds = 3`). → see **Alpha model**

**Sharpe ratio** — Mean excess return divided by its standard deviation, usually annualised: $SR_A = \sqrt{252}\,\hat\mu_d/s_d$. It is the system's main performance unit: phase bars 0.3/0.5 net, suspect above 2 net. Its standard error over $T$ observations is about $\sqrt{(1+SR^2/2)/T}$. → see **Validation & statistics › The Sharpe ratio**

**Short interest** — The number of shares currently sold short, often divided by shares outstanding (the short-interest ratio). Heavily shorted stocks tend to underperform and are expensive to borrow; the short-interest theme (negative short interest and negative borrow fee) and the borrow-fee proxy use it. → see **Alpha model**, **Transaction cost model**

**Short selling** — Selling borrowed shares, hoping to buy them back cheaper. It requires locating a lender, pays a borrow fee and has unbounded loss potential. → see **Transaction cost model**, **Portfolio construction**

**Shrinkage** — Pulling a noisy estimate toward a more stable target to reduce estimation error at the cost of a little bias. Used throughout: IC shrunk 50% toward 0, beta toward 1, specific risk toward size-group means (`specific_shrink_q = 0.1`), covariance eigenvalues (Ledoit–Wolf, QIS). → see **Risk model**, **Alpha model**

**Signal lag** — See **Execution delay**.

**Size** — The size factor: log market capitalisation. Small stocks behave differently from large ones; size is a risk style, not an alpha theme, in this system. → see **Risk model**

**Skewness** — Asymmetry of a distribution: $\gamma_3 = \mathbb{E}[(r-\mu)^3]/\sigma^3$. Negative skew (frequent small gains, rare large losses) widens the Sharpe ratio's standard error. → see **Validation & statistics › Return statistics**

**Slice equivalence** — The point-in-time property that every object computed "as of $t$" is identical whether the model was fitted on the full history or on `md.slice_until(t)`. The second audit verified it for every decision input of the full pipeline on the synthetic market to about $10^{-14}$. → see **Validation & statistics › Leakage tests**

**SOC constraint (second-order cone)** — A constraint of the form $\|Ax + b\|_2 \le c^\top x + d$; a portfolio volatility cap $\sqrt{w^\top\Sigma w} \le \sigma_{\max}$ is one. In `"fixed"` risk-aversion mode, vol targeting adds an SOC risk cap to the problem rather than scaling the book after the fact (first audit fix). → see **Portfolio construction**

**Sortino ratio** — Like the Sharpe ratio but dividing by downside deviation, so upside volatility is not penalised. For a symmetric distribution it is about √2 × Sharpe. → see **Validation & statistics › Drawdown, Calmar, Sortino**

**Spearman correlation** — Pearson correlation computed on ranks; robust to outliers and invariant to monotone transformations. Used for rank IC. → see **Validation & statistics › The information coefficient**

**Specific risk** — The variance of a stock's residual (specific) return, the diagonal $D$ of the factor-model covariance. Estimated by EWMA/Newey–West of specific returns (half-life 84), blended with a structural model and shrunk toward size-group averages. → see **Risk model**

**Split-consistent ratio** — A ratio whose numerator and denominator are on the same share basis, so a stock split does not change it. Per-share fundamentals (book value per share, EPS) are on the filing's share count; dividing them by today's price is wrong by the split factor after a split. The loaders instead form totals over market cap (`bvps × shares(filing) / market cap(t)`), and the short-interest ratio uses shares outstanding as of the settlement date (second audit findings 7–8). → see **Data & universe**

**Square-root law** — The empirical regularity that the price impact of a metaorder of size $Q$ is proportional to daily volatility times $\sqrt{Q/V}$: $I/\sigma = b\sqrt{Q/V}$. The integrated cost is $a|x| + b\sigma|x|^{3/2}/\sqrt V$, convex in $x$. Defaults $b = 1.0$ continuous and $b_a = 0.3$ auction (`impact_b_auction`, with $V$ = auction volume ≈ 9% of ADV, so $0.3/\sqrt{0.09} = 1.0$ per $\sqrt{Q/\text{ADV}}$: the same impact as the continuous session; design decision D2). Both are priors until calibrated on the system's own fills. → see **Transaction cost model**

**Standard error** — The standard deviation of an estimator across hypothetical repeated samples; the "± noise" around a reported number. → see **Validation & statistics**

**Style factor** — A risk factor defined by a stock characteristic (beta, size, momentum, value, residual volatility, liquidity, short-term reversal, quality, investment, short interest), standardised cross-sectionally (cap-weighted mean 0, equal-weighted sd 1). The optimizer keeps style exposures within ±0.1 sd (`style_band_sd`). → see **Risk model**, **Portfolio construction**

**Survivorship bias** — Testing only on companies that still exist today, which silently drops the failures and inflates returns. The universe is built from point-in-time membership including delisted names, and `survivorship_check` flags price sources with no delistings. → see **Data & universe**

**Suspect gate** — The hurdle that fails any run with net annual Sharpe above 2: such a result is treated as a bug (leakage, survivorship, cost or selection) until shown otherwise. A planted oracle pushed through the full pipeline must trip it. → see **Validation & statistics › The hurdle gates**

**Synthetic market** — A generated price panel with known factor structure and a planted alpha of known strength (`qts/data/synthetic.py`), used to test that each layer recovers what is there and nothing more. → see **Data & universe**, **Operating guide**

## T

**t-statistic** — An estimate divided by its standard error; under the null roughly standard normal, so $|t| > 1.96$ is "5% significant" for a single test. For a mean return, $t \approx$ annual Sharpe × √years. The plan requires t > 3 for new standalone signals (Harvey–Liu–Zhu). → see **Validation & statistics › Hypothesis testing**

**Target-vol risk aversion** — The default sizing (`PortfolioConfig.risk_aversion_mode = "target_vol"`): on each date the optimizer searches the risk aversion γ so that the optimal book of the full objective (alpha, risk, costs, constraints) has ex-ante annual volatility equal to `target_vol_annual = 0.10`. It starts at the closed form $\gamma_0 = IR/(2\sigma_{\text{target}})$ and refines on $\log\gamma$ in [0.05, 200] with at most 8 solves, stopping within 2% of target. Costs and the alpha trade-off are thus solved at the size actually traded. → see **Portfolio construction**

**TCA (transaction cost analysis)** — Measuring realised execution costs order by order and comparing them with the model, then recalibrating. The TCA ledger logs decision, arrival and fill prices and re-estimates the impact coefficient $b$ monthly, per venue. → see **Transaction cost model**

**Tracking error** — The standard deviation of active return; for a market-neutral book, simply the volatility. → see **Portfolio construction**

**Transfer coefficient (TC)** — The correlation between the weights you actually hold and the weights an unconstrained optimizer would hold given the same alphas; how much of the forecast survives constraints and costs. Phase-1 acceptance asks TC ≥ 0.6; demo 0.53. → see **Portfolio construction**, **Validation & statistics**

**Trial family** — The group of ledger trials that belong to one search and are deflated together. The pipeline uses `pipeline_phase{p}:{data_hash[:12]}`: one family per phase and data set, so runs on another synthetic seed or on real data do not mix into the cross-trial variance. → see **Validation & statistics › The trial ledger**

**Trial ledger** — The append-only JSON-lines record of every experiment (`runs/trial_ledger.jsonl`), supplying the trial count and cross-trial Sharpe variance for the DSR. It must never be reset. Every log appends one line under a file lock; nothing is rewritten. When reading, records with the same key (name, config hash, data hash, clean code version, run spec) count as one trial; runs from a dirty or unknown code version always count. → see **Validation & statistics › The trial ledger**

**Turnover** — How much is traded, as a fraction of NAV. Two-sided daily turnover is $\sum_i|x_{it}|$ (buying 1% and selling 1% is 2%); demo 8.0% per day. Costs scale with it; breakeven cost divides gross return by it. → see **Backtesting**, **Portfolio construction**

**Type I / Type II error** — A type I error is a false positive (accepting a useless strategy); a type II error is a false negative (rejecting a good one). Multiple-testing corrections trade fewer type I errors for more type II errors. → see **Validation & statistics › Hypothesis testing**

## U

**Universe** — The set of stocks eligible to trade on each date. Here: top 1,000 by market cap, as-traded price ≥ $5, 63-day ADV ≥ $2M, bottom 20% by size excluded, refreshed on the first trading day of each month (`refresh_anchor = "month_start"`) with exit hysteresis, all point-in-time (`UniverseConfig`). Membership is stored as the `MarketData.eligible` panel; price and volume history outside membership is kept so trailing statistics stay correct. → see **Data & universe**

**Upsert** — "Update or insert": write new rows into a stored table, replacing rows with the same key and keeping all others. The Parquet store's default write mode; a NaN cell in a wide panel means "no information" and keeps the stored value. The pre-refactor store rewrote a whole year partition with only the new rows, so appending one day deleted the rest of the year. → see **Data & universe**

**USE4** — MSCI Barra's fourth-generation US equity risk model, whose published methodology (cap-weighted industry constraint, separate volatility and correlation half-lives, VRA, eigenfactor adjustment, Bayesian specific-risk shrinkage) the risk model replicates. → see **Risk model**

## V

**Value** — Cheap stocks (high book-to-price, earnings yield) tend to outperform expensive ones over long horizons. The value theme averages industry-demeaned log B/P (`value_bp`) and E/P (`value_ep`); the plan's sales- and cash-flow-based measures are not yet implemented. → see **Alpha model**

**VaR / CVaR (value at risk / conditional value at risk)** — VaR is the loss threshold exceeded with a given probability (e.g. 5%) over a horizon; CVaR (expected shortfall) is the average loss beyond it. They are **not used** in this system, which measures risk by forecast volatility, bias statistics and drawdown. → see **Risk model**

**Variance** — The average squared deviation from the mean, $\sigma^2$; its square root is volatility. Portfolio variance in a factor model is $w^\top(BFB^\top + D)w$. → see **Validation & statistics**, **Risk model**

**Vol-target binding** — The diagnostic `OptimizationResult.vol_target_binding`: `"caps"` when the gross, position or ADV limits stop the γ search below the target volatility (the capped book is accepted), empty when the target was reached. → see **Portfolio construction**

**Volatility** — The standard deviation of returns, usually annualised ($\sigma_A = \sqrt{252}\sigma_d$). Demo net volatility 6.7% a year. → see **Validation & statistics**

**Volatility targeting** — Sizing the book so its predicted volatility matches a target (10% annual, `target_vol_annual = 0.10`). By default this is done by choosing the risk aversion (**target-vol risk aversion**), so every constraint holds and the trade-off with alpha and cost is solved at the traded size. In `"fixed"` mode the older options remain: a projection re-solve or an SOC risk cap. On the synthetic market the 10% target is not reachable under gross 2.0 and the 2% name caps; realised volatility tops out around 8–9%. → see **Portfolio construction**

**VRA (volatility regime adjustment)** — A USE4 correction that scales the factor covariance by $\lambda_F^2$, where $\lambda_F$ is an EWMA (half-life 42) of the cross-sectional bias statistic of factor returns, so forecasts catch up quickly when volatility regimes shift. → see **Risk model**

**VWAP (volume-weighted average price)** — The average price of all trades in a period, weighted by size; a common execution benchmark. A VWAP schedule trades in proportion to the expected intraday volume curve (`vwap_schedule`); IBKR's `Vwap` algo is used for larger orders. → see **Execution**

## W

**Walk-forward** — Train on data up to a date, predict the next block, move forward, repeat, never training on the future. The ML trainer uses an expanding walk-forward with annual refits and an exact purge; `PurgedWalkForward` is the general splitter. → see **Validation & statistics › Cross-validation**, **Alpha model**

**Winsorisation** — Clipping extreme values to chosen percentiles (here the 1st and 99th per date, `winsor_pct = 0.01`) so a handful of outliers cannot dominate a cross-sectional statistic. → see **Alpha model**

## Z

**z-score** — A value's distance from the mean in standard deviations, $(x - \bar x)/s$. Computed per date across stocks; the risk model's style z-scores use a cap-weighted mean and equal-weighted standard deviation (Barra convention). → see **Alpha model**, **Risk model**
