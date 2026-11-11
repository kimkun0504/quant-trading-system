# Validation & statistics

<callout icon="🧭" color="gray_bg">
This page explains the statistics the system uses to decide whether a backtest is evidence or noise, and the code in `qts/validation/`, `qts/backtest/metrics.py`, `qts/core/stats.py` and `qts/alpha/evaluation.py` that computes them. It starts with why validation is the core of quant research (most attractive backtests are false), then works through return statistics, the Sharpe ratio and its standard error, t-statistics with Newey–West corrections, the information coefficient and the fundamental law, decile spreads, drawdown-based metrics, multiple testing, the probabilistic and deflated Sharpe ratios, the probability of backtest overfitting, purged cross-validation, the repo's leakage tests, the trial ledger, and finally every hurdle gate with a line-by-line reading of the demo's report card. Every formula has its symbols defined and a worked example you can check by hand.
</callout>

<callout icon="🆕" color="green_bg">
<b>What changed (Nov 2026)</b>
- `ic_report` returns a typed `ICReport` (with a Newey–West t) instead of a dict; the ICIR and the decile Sharpe now share one inferred sampling frequency (12.6/yr, not 13).
- Newey–West lags follow one rule everywhere: $L = \min\big(\max(h, \lfloor 4(T/100)^{2/9}\rfloor),\, T-1\big)$. The old $L = h-1$ is gone (audit finding 15).
- The `net_sharpe_delta` and `ic_sign` gates are scored on the same purged walk-forward blocks (`ValidationConfig.n_splits`, `embargo_days`, `min_fold_days`), not calendar years.
- Labels, IC, decile spread and gates are scored on the signal that was actually traded under the default one-day execution delay (`signal_lag_days = 1`).
- The trial ledger is strictly append-only under a file lock. Re-runs are de-duplicated at read time on name + config hash + data hash + clean code version + run spec. The DSR family is per data set, PBO grid runs are logged as trials, and the effective trial count is used unrounded.
- A planted oracle is now pushed through the full `ResearchPipeline` and must trip the `suspect` gate. New PIT suites cover hypothesis properties, label halts and `apply_mask` history. The second audit found full-pipeline slice equivalence to about 1e-14.
- Gate comparison operators match their reason strings (`ic_sign` and `incremental_ic` are strict).
</callout>

## 1. Why validation is the core of quant research

### What it is
A backtest is a simulation of how a trading rule would have done on historical data. Validation is the set of procedures that estimate how much of that simulated performance will survive on data the researcher has not seen. In most engineering, a passing test is strong evidence. In quant research a great backtest is weak evidence, because the researcher picks the rule *after* looking at the same data the test uses.

### Why we need it here
The expected Sharpe ratio of a real, cost-aware US equity long/short book is small: the build plan sets a net Sharpe of 0.5–1.0 as success and treats anything above 2 net as a bug until proven otherwise (`docs/BUILD_PLAN.md` §1 and §2.4). At that signal-to-noise level, the statistical noise in a four-year backtest is as large as the effect being measured. Without validation, the system would deploy whichever of its many variants happened to get lucky.

### How it goes wrong: the garden of forking paths
Gelman and Loken's phrase "the garden of forking paths" describes how a researcher who never consciously runs more than one test still makes dozens of data-dependent choices: which universe, which lookback, whether to winsorise at 1% or 2.5%, whether to neutralise by industry, which years to show. Each fork is harmless alone. Together they mean the reported result is the best of an unknown, large number of implicit trials, and the reported p-value is meaningless.

Three pieces of evidence calibrate how bad this is in finance:

| Evidence | What it found | Evidence class |
|---|---|---|
| Harvey, Liu & Zhu (2016), 316 published factors | Of 296 published "significant" factors, roughly one third to one half are likely false once data snooping is corrected for; a new factor should clear **t > 3.0** (p < 0.0027) rather than 1.96 | Meta-analysis of published in-sample results |
| Chen & Velikov (via `BUILD_PLAN.md` §1), 204 anomalies | Only 14 of 204 clear \|t\| > 2 net of costs post-2005, against about 10 expected by chance | Out-of-sample, post-publication, net |
| Gençay (2026), leakage-safe audit of LLM-mined alphas on US large caps | Best in-sample Sharpe **1.69 → 0.18 out of sample**; its DSR of 0.86 was below the 0.95 bar; an agentic Claude search went from IS 0.44 to OOS −0.33 | In-sample vs out-of-sample, with DSR/PBO |
| Suhonen, Lennkh & Perez (via `BUILD_PLAN.md` §1), 215 bank indices | Median **73% Sharpe deterioration** from backtest to live | Live |

The Gençay audit adds the single most important fact for this page: a **planted look-ahead oracle scored DSR = 1.00**. The deflated Sharpe ratio (section 9) corrects for *selection* among honest trials. It cannot tell an honest great strategy from one that peeks at the future, because peeking produces exactly the high, stable Sharpe that DSR rewards. Leakage therefore has to be prevented structurally (point-in-time data, purged splits, code tripwires) and caught by plausibility gates ("net Sharpe > 2 is suspect"), never detected statistically.

<callout icon="💡" color="blue_bg">
Think of validation as two independent defences. Structural defences (sections 11–12) make it impossible to use future information. Statistical defences (sections 3–10, 13) make it expensive to fool yourself by trying many honest variants and keeping the best. Each covers the other's blind spot; neither is optional.
</callout>

### Why this choice
The repo follows the build plan's §8 protocol: structural leakage prevention, purged walk-forward evaluation, an append-only trial ledger feeding a deflated Sharpe ratio, PBO when a configuration grid exists, and fixed hurdle gates. López de Prado's broader causal protocol (`BUILD_PLAN.md` §8) is acknowledged as too heavy for one engineer; the "actionable subset" — resampled backtests and deflated Sharpe ratios plus explicit confounder reasoning — is what is implemented.

### References
- [Gelman & Loken (2013), The garden of forking paths](http://www.stat.columbia.edu/~gelman/research/unpublished/p_hacking.pdf)
- [Harvey, Liu & Zhu (2016), …and the Cross-Section of Expected Returns (NBER w20592)](https://nber.org/papers/w20592); summary: [CXO Advisory, Taming the factor zoo](https://www.cxoadvisory.com/big-ideas/taming-the-factor-zoo/)
- [Gençay (2026), arXiv 2608.27734 (as cited in BUILD_PLAN §2.2, §8)](https://arxiv.org/pdf/2608.27734)
- [Suhonen, Lennkh & Perez via Alpha Architect](https://alphaarchitect.com/?p=36892)

## 2. Return statistics

Everything downstream is built from a series of daily portfolio returns. The repo's convention (`qts/core/types.py`) is **simple** close-to-close returns, $r_t = P_t / P_{t-1} - 1$, as fractions (0.01 = 1%), with 252 trading days per year (`SystemConfig.trading_days_per_year`).

### Mean, variance, standard deviation
$$
\hat\mu = \frac{1}{T}\sum_{t=1}^{T} r_t, \qquad s^2 = \frac{1}{T-1}\sum_{t=1}^{T}(r_t-\hat\mu)^2, \qquad s = \sqrt{s^2}
$$
where:
- $T$ is the number of return observations (days);
- $r_t$ is the return on day $t$;
- $\hat\mu$ is the sample mean (the "hat" marks an estimate of the unknown true mean $\mu$);
- $s^2$ is the unbiased sample variance (dividing by $T-1$, pandas' default `ddof=1`), and $s$ the sample standard deviation, which finance calls **volatility**.

### Annualisation and its iid assumption
If daily returns are independent and identically distributed (**iid**), the variance of a sum of $q$ of them is $q$ times the daily variance. So
$$
\mu_A = q\,\mu_d, \qquad \sigma_A = \sqrt{q}\,\sigma_d, \qquad SR_A = \frac{\mu_A}{\sigma_A} = \sqrt{q}\,SR_d
$$
where:
- $q$ is the number of periods per year (252 for daily data, 12 for monthly, 252/20 = 12.6 for a series sampled every 20 trading days);
- subscripts $d$ and $A$ denote per-period and annual quantities.

The $\sqrt{252}$ rule is only as good as the iid assumption. If returns are positively autocorrelated (today's gain makes tomorrow's more likely — common for illiquid assets, smoothed marks, or overlapping holding periods), the variance of the annual sum is *larger* than $252\,\sigma_d^2$ and $\sqrt{252}$ overstates the Sharpe ratio. Section 3 gives Lo's correction. `qts/backtest/metrics.py::annualized_return` uses the geometric (compounded) version $\prod(1+r_t)^{q/T}-1$ by default; `annualized_vol` uses $s\sqrt{q}$.

### Skewness and kurtosis
$$
\gamma_3 = \frac{\mathbb{E}[(r-\mu)^3]}{\sigma^3}, \qquad \gamma_4 = \frac{\mathbb{E}[(r-\mu)^4]}{\sigma^4}, \qquad \text{excess kurtosis} = \gamma_4 - 3
$$
where:
- $\gamma_3$ (**skewness**) measures asymmetry: negative skew means occasional large losses and many small gains (selling insurance, momentum crashes);
- $\gamma_4$ (**raw kurtosis**) measures tail heaviness; a Gaussian has $\gamma_4 = 3$;
- **excess kurtosis** subtracts 3 so a Gaussian scores 0.

<callout icon="⚠️" color="yellow_bg">
Raw vs excess kurtosis is a classic unit bug. `pandas.Series.kurt()` returns **excess** kurtosis; `scipy.stats.kurtosis(..., fisher=False)` returns **raw**. Every function in `qts/validation/deflated_sharpe.py` expects **raw** kurtosis (3 for Gaussian). That is why `qts/pipeline.py` writes `kurt = float(net.kurt() + 3)` and `moments_from_returns` calls `kurtosis(r, fisher=False)`. The audit checked this explicitly (`docs/audit/alpha_validation_data.md`, "Checked and found correct").
</callout>

### Autocorrelation
$$
\hat\rho_k = \frac{\sum_{t=k+1}^{T}(r_t-\hat\mu)(r_{t-k}-\hat\mu)}{\sum_{t=1}^{T}(r_t-\hat\mu)^2}
$$
where $\hat\rho_k$ is the lag-$k$ sample autocorrelation, between −1 and 1. This is exactly `qts/backtest/metrics.py::_autocorr`. Daily returns of a diversified long/short book usually have $|\rho_1| < 0.1$; returns built from *overlapping* multi-day windows have very large autocorrelations by construction (section 4).

### Worked example
Five daily returns: +1%, −0.5%, +0.2%, +0.8%, −0.3%. Mean $\hat\mu = 1.2\%/5 = 0.24\%$. Deviations: 0.76, −0.74, −0.04, 0.56, −0.54 (in %). Squares sum to 0.5776 + 0.5476 + 0.0016 + 0.3136 + 0.2916 = 1.732; $s^2 = 1.732/4 = 0.433\ (\%)^2$, $s = 0.658\%$. Daily Sharpe $0.24/0.658 = 0.365$; annualised under iid, $0.365\times\sqrt{252} = 5.8$ — an absurd number that shows why five observations prove nothing (section 3 gives its standard error).

## 3. The Sharpe ratio

### What it is
The Sharpe ratio is return per unit of risk: how many standard deviations of noise the average return sits above zero (or above the risk-free rate). It is the single number most quant decisions hinge on, because with leverage any positive-Sharpe strategy can be scaled to any target volatility; what cannot be changed is the ratio.

### Why we need it here
Every hurdle in the plan is expressed in Sharpe units: the phase bars (0.3 and 0.5 net), the suspect flag (> 2 net), the DSR, the fold rule, the cost-sensitivity table. Its standard error decides how long a backtest must be before the number means anything.

### How it works
$$
SR = \frac{\mathbb{E}[r - r_f]}{\sigma(r - r_f)}, \qquad \widehat{SR} = \frac{\hat\mu - r_f}{s}
$$
where:
- $r$ is the strategy return per period and $r_f$ the risk-free rate per period;
- $\sigma(\cdot)$ is the standard deviation.

`qts/backtest/metrics.py::annualized_sharpe` uses `rf=0.0` by default. For a dollar-neutral long/short book that is a reasonable approximation: the book is (roughly) self-financing, so its return is already an excess return. The function returns $\widehat{SR}_d\sqrt{q}$.

**Standard error under iid normal returns (Lo 2002).**
$$
\operatorname{se}(\widehat{SR}) \approx \sqrt{\frac{1 + \tfrac12 SR^2}{T}}
$$
where $SR$ is the per-period Sharpe and $T$ the number of periods. The "1" is the noise in the mean; the $\tfrac12 SR^2$ term is the noise in the standard deviation. This is `qts/backtest/metrics.py::sharpe_t_stat`, which reports $\widehat{SR}/\operatorname{se}$.

**Standard error for non-normal returns (Mertens 2002; Opdyke 2007).**
$$
\operatorname{se}(\widehat{SR}) = \sqrt{\frac{1 - \gamma_3\,SR + \frac{\gamma_4 - 1}{4}\,SR^2}{T-1}}
$$
where $\gamma_3$ is skewness and $\gamma_4$ is **raw** kurtosis of per-period returns. For a Gaussian ($\gamma_3=0$, $\gamma_4=3$) it reduces to Lo's formula. Negative skew and fat tails both *increase* the standard error: a strategy that earns its Sharpe by occasionally blowing up is less certain than its headline suggests. This is `qts/validation/deflated_sharpe.py::sharpe_ratio_se`, used by PSR, DSR and MinTRL.

**Lo's autocorrelation adjustment.**
When returns are serially correlated, the correct annualisation factor replaces $\sqrt{q}$:
$$
SR_q = SR_1 \cdot \frac{q}{\sqrt{q + 2\sum_{k=1}^{q-1}(q-k)\,\rho_k}}
$$
where:
- $SR_1$ is the per-period Sharpe;
- $q$ is the number of periods aggregated (252 for daily → annual);
- $\rho_k$ is the lag-$k$ autocorrelation of per-period returns.

With all $\rho_k = 0$ the denominator is $\sqrt q$ and the factor is $\sqrt q$. Positive autocorrelation shrinks the factor. `annualized_sharpe(..., lo_adjust=True)` implements it, truncating the sum after `lags` terms (default $\min(q-1, 20)$) because high-lag sample autocorrelations are pure noise; the backtest summary reports it as `sharpe_net_lo`.

**What a Sharpe of 0.5 / 1 / 2 means.**
If annual returns are normal with mean $\mu_A$ and volatility $\sigma_A$, the probability of a losing year is
$$
P(R_A < 0) = \Phi\!\left(-\frac{\mu_A}{\sigma_A}\right) = \Phi(-SR_A)
$$
where $\Phi$ is the standard normal cumulative distribution function.

| Annual Sharpe | P(losing year) | Roughly |
|---|---|---|
| 0.5 | $\Phi(-0.5) = 30.9\%$ | 3 losing years per decade |
| 1.0 | $\Phi(-1) = 15.9\%$ | 1–2 per decade |
| 2.0 | $\Phi(-2) = 2.3\%$ | 1 per 44 years |

The plan's realistic target (0.5–1.0) therefore *expects* losing years. A paper-trading year with a loss is not evidence the strategy is broken, which is why the plan rests go-live on process controls rather than a short live window.

### Why this choice
The Sharpe ratio ignores higher moments and path (drawdowns), so the repo reports Sortino, Calmar and max drawdown alongside it (section 7) and uses the skew/kurtosis-aware standard error wherever a probability is computed. The information ratio (Sharpe of active return vs a benchmark) is the same formula; for a market-neutral book with a zero benchmark they coincide.

### In the code
- `qts/backtest/metrics.py::annualized_sharpe(returns, periods_per_year=252, rf=0.0, lo_adjust=False, lags=None)`
- `qts/backtest/metrics.py::sharpe_t_stat` (Lo iid)
- `qts/validation/deflated_sharpe.py::sharpe_ratio_se(sr, n, skew=0, kurt=3)` (Mertens, raw kurtosis), `annualize_sharpe`, `deannualize_sharpe`, `sharpe_from_returns`, `moments_from_returns`
- Backtest summary keys (`qts/backtest/result.py::BacktestResult.summary`, typed as the `BacktestSummary` TypedDict): `sharpe_gross`, `sharpe_net`, `sharpe_net_lo`, `sharpe_t_stat_net`, `newey_west_t_net`.

### Pitfalls & checks
- **Per-period vs annual.** The core DSR functions take per-observation Sharpe ratios; passing an annual Sharpe where a daily one is expected inflates the z-score by $\sqrt{252}\approx 15.9$. Functions that take annual inputs say `annual` in their name.
- **Short samples.** The standard error shrinks with $\sqrt T$ only.
- **Overlapping returns.** Annualising a Sharpe computed on overlapping 20-day returns sampled daily by $\sqrt{252}$ double-counts; `decile_spread` annualises by `periods_per_year / horizon` for this reason.

### Worked example
A strategy with annual Sharpe 1.0 over 5 years of daily data ($T = 1260$). Daily $SR_d = 1/\sqrt{252} = 0.0630$. Lo iid: $\operatorname{se} = \sqrt{(1 + 0.5\times0.00397)/1260} = 0.0282$ daily, or $0.0282\times\sqrt{252} = 0.448$ annual. So a "1.0" is really $1.0 \pm 0.45$ (one standard error) — a 95% interval of roughly 0.1 to 1.9. With monthly data (60 months, $SR_m = 0.289$), skew −1 and raw kurtosis 6, Mertens gives $\operatorname{se} = \sqrt{(1 + 0.289 + 1.25\times0.0833)/59} = 0.154$ monthly (0.53 annual) against 0.133 (0.46 annual) under normality: crash-prone returns widen the band by about 15%.

Lo adjustment: monthly returns with $\rho_1 = 0.2$ and no other autocorrelation, $q = 12$: denominator $\sqrt{12 + 2\times11\times0.2} = \sqrt{16.4} = 4.05$, factor $12/4.05 = 2.96$ instead of $\sqrt{12} = 3.46$ — a 14% haircut on the annual Sharpe.

### References
- [Lo (2002), The Statistics of Sharpe Ratios, Financial Analysts Journal](https://doi.org/10.2469/faj.v58.n4.2453)
- [Opdyke (2007), Comparing Sharpe ratios: So where are the p-values?, Journal of Asset Management](https://doi.org/10.1057/palgrave.jam.2250084)
- Mertens (2002), "Comments on variance of the IID estimator in Lo (2002)", working note (cited via Bailey & López de Prado 2012).

## 4. Hypothesis testing refresher, t-statistics and Newey–West

### What it is
A hypothesis test asks: if the strategy truly had no edge, how surprising would the observed result be? It formalises "could this be luck?".

### How it works
- **Null hypothesis** $H_0$: the true mean return (or IC, or Sharpe) is zero.
- **Test statistic**: a number computed from the data whose distribution under $H_0$ is known — here usually a t-statistic, approximately standard normal for large samples.
- **p-value**: the probability, under $H_0$, of a statistic at least as extreme as the one observed. It is *not* the probability that $H_0$ is true.
- **Type I error** (false positive): rejecting $H_0$ when it is true; its rate is the significance level $\alpha$ (conventionally 5%, i.e. $|t| > 1.96$ two-sided).
- **Type II error** (false negative): failing to reject when the edge is real; its rate is $\beta$.
- **Power** $= 1-\beta$: the probability of detecting a real effect. Power rises with effect size and sample length.

**The t-statistic of a mean return and its relation to Sharpe.**
$$
t = \frac{\hat\mu}{s/\sqrt{T}} = \widehat{SR}_d\sqrt{T} = \widehat{SR}_A\sqrt{\frac{T}{252}} = \widehat{SR}_A\sqrt{\text{years}}
$$
where $\widehat{SR}_d$ is the daily and $\widehat{SR}_A$ the annual Sharpe. This identity is the most useful mental shortcut in the field: **t ≈ annual Sharpe × √years**. A true Sharpe of 0.6 needs $(2/0.6)^2 \approx 11$ years to reach $t \approx 2$ — the figure Gençay (2026) quotes and the reason the build plan says a short paper-trading window cannot certify a moderate Sharpe. Reaching the Harvey–Liu–Zhu bar of 3 with Sharpe 0.6 needs 25 years.

**Newey–West HAC standard errors.**
The simple t-statistic assumes observations are uncorrelated. When they are not, the variance of the sample mean is the **long-run variance** divided by $T$, not $s^2/T$. Newey and West (1987) estimate it with a weighted sum of autocovariances that is guaranteed non-negative:
$$
\hat S = \hat\gamma_0 + 2\sum_{k=1}^{L}\left(1 - \frac{k}{L+1}\right)\hat\gamma_k, \qquad t_{NW} = \frac{\hat\mu}{\sqrt{\hat S / T}}
$$
where:
- $\hat\gamma_k = \frac{1}{T}\sum_{t=k+1}^{T}(r_t-\hat\mu)(r_{t-k}-\hat\mu)$ is the lag-$k$ sample autocovariance ($\hat\gamma_0$ is the variance, with divisor $T$);
- $L$ is the number of lags included (the bandwidth);
- $1 - k/(L+1)$ is the **Bartlett kernel** weight, declining linearly to zero; it is what keeps $\hat S \ge 0$;
- "HAC" stands for heteroskedasticity- and autocorrelation-consistent.

**Lag choice.** Two pieces of knowledge feed the bandwidth. The first is generic: with no structural knowledge, the Newey–West (1994) automatic rule $\lfloor 4(T/100)^{2/9}\rfloor$ grows slowly with the sample (3 lags for $T = 50$, 6 for $T = 1005$). The second is structural: an $h$-day return sampled daily shares $h-1$ days with its neighbour, so its autocorrelations are exactly $\rho_k = (h-k)/h$ for $k < h$ and zero beyond (a moving average of order $h-1$, written MA($h-1$)). The repo combines them in one rule, used for every IC and decile t-statistic:
$$
L = \min\Big(\max\big(h,\ \lfloor 4(T/100)^{2/9}\rfloor\big),\ T-1\Big)
$$
where:
- $T$ is the number of observations in the series being tested;
- $h$ is the **overlap**: how many consecutive observations share label days ($h$ = the label horizon for a daily series of $h$-day returns, $h = 1$ for non-overlapping samples);
- the cap $T-1$ is the largest lag that has any data behind it.

Why the floor is $h$ and not $h-1$: the Bartlett weight on lag $k$ is $1 - k/(L+1)$, so with $L = h-1$ the last overlapping lag gets weight $1/h$ and the earlier ones are also shrunk. The kernel then under-counts exactly the autocorrelation the overlap creates. Flooring at $h$ keeps every overlapping lag with positive weight, and the automatic term still covers ordinary serial correlation when there is no overlap ($h = 1$). The old code used $L = h-1$. The first audit flagged it (finding 15, LOW) and the handbook review listed it as fix #7. The second audit re-derived the function by hand: $(T, h) = (100, 20) \to 20$, $(252, 1) \to 4$, $(3000, 5) \to 8$, $(10, 20) \to 9$.

**Why overlapping 20-day returns inflate naive t-stats.**
With $\rho_k = (h-k)/h$, the long-run variance is
$$
\sigma_h^2\left(1 + 2\sum_{k=1}^{h-1}\frac{h-k}{h}\right) = h\,\sigma_h^2
$$
where $\sigma_h^2$ is the variance of one $h$-day return. The naive t uses $\sigma_h^2$, so it is too large by a factor $\sqrt h$: **4.47 for $h = 20$**. Intuitively, 2,520 daily-sampled 20-day returns contain only 126 independent 20-day periods.

### Why this choice
Two fixes exist: correct the standard error (Newey–West) or remove the overlap by sampling every $h$ days. The pipeline does the second for its gate inputs: `ResearchPipeline.evaluate` samples execution dates every `H = 20` days and calls `decile_spread(..., horizon=1)` and `ic_report(..., overlap=1)`, so consecutive observations do not overlap and the Newey–West correction only mops up residual serial correlation (3 lags for 50 observations). Non-overlapping sampling throws away information (only 1/20 of formation dates) but is almost immune to bandwidth choice. On a 4-year backtest it leaves about 50 observations, which is honest about how little the data can say.

### In the code
- `qts/core/stats.py::newey_west_se(x, lags=None, min_obs=3)` and `newey_west_tstat` — the single implementation (Bartlett kernel, autocovariances with divisor $T$, non-finite values dropped). `default_nw_lags(n)` is $\lfloor 4(n/100)^{2/9}\rfloor$.
- `qts/alpha/evaluation.py::newey_west_lags(n_obs, overlap=1)` — the rule above. `ic_report` (field `t_stat_nw`, `nw_lags`) and `decile_spread` (field `t_stat`, `nw_lags`) both use it.
- `qts/backtest/metrics.py::newey_west_tstat(returns, lags=None)` — daily net returns with the automatic lag (6 for 1,005 days), at least 5 observations; reported as `newey_west_t_net` ("NW t" in the demo).
- The risk model uses the same Bartlett weighting for its factor covariance (`RiskConfig.newey_west_lags_vol = 5`, `newey_west_lags_corr = 2`) → see **Risk model**.

### Pitfalls & checks
- A p-value of 0.04 from one of 50 attempted variants is not significant (section 8).
- Non-significance is not evidence of no edge when power is low: a true Sharpe of 0.5 over 4 years has $t \approx 1$, so it fails $t>2$ most of the time.

### Worked example
Simulated iid daily returns (mean 2 bp, vol 1%) over 2,520 days, then rolled into overlapping 20-day sums: naive t on the overlapping series = 2.13; the daily series' own t = 0.44; non-overlapping 20-day sample t = 0.44; Newey–West with $L = 19$ (the old $h-1$ rule) gives 0.59 and with $L = 30$ gives 0.55. The current rule picks $L = \max(20, \lfloor 4(25.2)^{2/9}\rfloor) = \max(20, 8) = 20$, between the two. The naive statistic would have "passed" at 5%; the honest ones are nowhere near. (These numbers are from an illustrative simulation, not from the repo's test suite.)

### References
- [Newey & West (1987), A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix, Econometrica](https://doi.org/10.2307/1913610)

## 5. The information coefficient (IC)

### What it is
The IC measures how well a signal *ranks* stocks on a given day: the cross-sectional correlation between today's scores and the returns that follow. An IC of 0.05 sounds tiny, but applied across 1,000 stocks every month it is a respectable signal. Think of a weather forecaster who is right 52.5% of the time on whether tomorrow is warmer than average — individually useless, valuable over thousands of cities.

### Why we need it here
IC isolates forecasting skill from portfolio construction, costs and risk. A signal can have a good IC and still lose money after costs (Gate 3), but a signal with no IC cannot be rescued by an optimizer. The alpha layer also uses IC to scale scores into expected returns (Grinold's rule, → see **Alpha model** and **Portfolio construction**).

### How it works
**Pearson IC** on date $t$:
$$
IC_t = \operatorname{corr}\big(s_{\cdot,t},\; y_{\cdot,t}\big) = \frac{\sum_i (s_{it}-\bar s_t)(y_{it}-\bar y_t)}{\sqrt{\sum_i (s_{it}-\bar s_t)^2\sum_i (y_{it}-\bar y_t)^2}}
$$
where:
- $i$ indexes the stocks eligible on date $t$;
- $s_{it}$ is the signal score known at the close of $t$;
- $y_{it}$ is the label: the forward return over $(t, t+h]$ (raw, or residualised against risk factors — the repo's default target). With an execution delay of $L$ days the window moves to $(t+L, t+L+h]$; see "Scoring the traded signal" below;
- bars are cross-sectional means on date $t$.

**Spearman (rank) IC** is the Pearson correlation of the *ranks* of $s$ and $y$. It is robust to outliers (one stock that triples does not dominate) and invariant to monotone transforms of the score. The repo uses rank IC throughout (`rank_ic_series`), requiring at least 20 valid names per date.

The **IC series** $\{IC_t\}$ is one number per formation date. Summaries:
$$
\overline{IC} = \frac{1}{n}\sum_t IC_t, \qquad \text{ICIR}_{\text{period}} = \frac{\overline{IC}}{\operatorname{sd}(IC_t)}, \qquad \text{ICIR}_A = \text{ICIR}_{\text{period}}\sqrt{p}, \qquad t_{IC} = \text{ICIR}_{\text{period}}\sqrt{n}
$$
where:
- $n$ is the number of dates in the IC series;
- $p$ is the number of IC observations **per year** — the sampling frequency of the series, *not* always 252;
- ICIR ("IC information ratio") is the Sharpe ratio of the IC series: consistency of skill.

**The ×4.5 bug.**
If the IC series is sampled every 20 trading days, $p = 252/20 = 12.6$. Annualising with $\sqrt{252}$ instead inflates ICIR by $\sqrt{20} = 4.47$. This was audit finding 4 (HIGH). The fix: `ic_report` infers $p$ from the index spacing (`infer_periods_per_year`: 252 divided by the median gap in business days, the gap converted from calendar days as $\text{round}(\text{gap}\times 5/7)$). The t-statistic uses $n$ directly and was never affected — the tell-tale sign of the bug is an ICIR far larger than the t-stat would allow. `runs/demo_phase2.txt` predates the fix and shows it: ICIR 3.51 with t = 1.33. Working back, per-period ICIR $= 3.51/\sqrt{252} = 0.221$, implying $n = (1.33/0.221)^2 \approx 36$; the correct annual ICIR is $0.221\times\sqrt{12.6} = 0.78$.

**One sampling frequency for IC and decile statistics.** After the first fix the pipeline still annualised the decile Sharpe with `round(252/20) = 13` periods a year while the IC report used the inferred 12.6, so the decile Sharpe was $\sqrt{13/12.6} - 1 = 1.6\%$ too high (second audit, finding 15, LOW). `ResearchPipeline.evaluate` now calls `infer_periods_per_year` once on the sampled execution dates and passes that $p$ to both `ic_report` and `decile_spread`.

**IC decay.**
A signal's IC at horizon $h$ (forward return over $h$ days) traces its **decay curve**: how long the information lasts. `ic_decay_curve(score, md, horizons=(1,5,10,20,40,60,120), residualized=True)` computes cumulative (not marginal) mean rank IC at each horizon. A separate, cheaper measure is the **signal half-life**: the rank autocorrelation of the score itself, $\rho(\ell) = \operatorname{corr}(\text{rank}\,s_t, \text{rank}\,s_{t-\ell})$, fitted as $\rho(\ell) = e^{-\lambda\ell}$ through the origin; half-life $= \ln 2/\lambda$ (`signal_half_life`). Slow signals (value: years) tolerate slow trading; fast ones (reversal: days) die under costs.

**Fundamental Law of Active Management.**
Grinold (1989) connected IC to the achievable information ratio:
$$
IR \approx IC \cdot \sqrt{BR}, \qquad \text{with constraints (Clarke, de Silva \& Thorley 2002):}\quad IR \approx TC \cdot IC \cdot \sqrt{BR}
$$
where:
- $IR$ is the annual information ratio (Sharpe of active return);
- $BR$ (**breadth**) is the number of *independent* bets per year;
- $TC$ (**transfer coefficient**, between 0 and 1) is the correlation between the weights you would hold without constraints and the weights you actually hold; constraints, costs and caps reduce it.

Assumptions that make it an approximation: bets are independent (stocks share factors, so effective breadth is far below $N \times$ rebalances); IC is constant over time; forecasts are optimally scaled; and IC is small. Treat it as an upper bound and a reasoning tool, not a forecast.

### Scoring the traded signal (execution delay)
Since November 2026 the system's default is a one-day **execution delay**: `ExecutionConfig.signal_lag_days = 1`. A signal formed from data up to the close of day $s$ is traded at the close of $t = s + 1$, because real market-on-close orders must be submitted before the 15:50 ET cutoff, ten minutes before the close they execute at (→ see **Backtesting** and **Execution**). The book then earns from $\text{close}_t$, not from $\text{close}_s$. Every statistic on this page that measures "how good was the signal" must be measured on the same timeline, or it describes a trade nobody made.

The second audit found that it was not (finding 4, MEDIUM): the label was $\text{close}_s \to \text{close}_{s+h}$, which includes the day $s \to s+1$ the book cannot earn, and `evaluate()` scored `alpha.as_of(t)` on execution dates although the book traded at $t$ held `alpha.as_of(t-1)`. On seed 0 the reported residual IC was 0.0425 against 0.0408 for the score actually traded: not leakage, but optimistic. With $L$ = `signal_lag_days`, the repo now aligns everything to the trade:

| Quantity | Definition with lag $L$ | Where |
|---|---|---|
| Label of signal date $s$ | $\text{close}_{s+L} \to \text{close}_{s+L+h}$, residualised and z-scored, aligned at $s$ | `qts/alpha/targets.py` (`label_lag_days=L`) |
| When the label is usable | not before $s + L + h$ (later if a halt is still unresolved, see **label realisation** in §12) | `label_realisation`, trainer purge, `trailing_ic` |
| Score evaluated at execution date $t$ | `alpha.as_of(t − L)`, eligible names at $t - L$ only | `ResearchPipeline.evaluate` |
| IC, decile spread, hurdle frames | indexed by the execution date $t$; label and raw forward return both start at $\text{close}_t$ | `ResearchPipeline.evaluate` |

```python
exec_dates = pd.DatetimeIndex(bt.weights.index[::H])        # every H-th execution date
sig_dates = self.md_u.dates[self.md_u.dates.get_indexer(exec_dates) - L]
scores = pd.DataFrame({t: self.alpha.as_of(s).score
                       for t, s in zip(exec_dates, sig_dates, strict=True)}).T
ppy = infer_periods_per_year(scores.index, cfg.trading_days_per_year)
```

(`qts/pipeline.py::ResearchPipeline.evaluate`, abridged.) With $L = 0$ everything reduces to the classic close-as-signal timeline, which remains available as an optimistic bound. The walk-forward trainer's embargo grows from $h$ to $h + L$, and the Grinold IC window ends at $\text{pos} - h - L$. A run with a different `signal_lag_days` refits the alpha model because labels are built at fit time.

### Why this choice
Rank IC over Pearson because return distributions are fat-tailed. IC against the **residualised** target, because the alpha model's job is to predict returns the risk model does not explain (→ see **Alpha model › Targets**). Mean IC and ICIR are reported by calendar year so a reader can see if the effect is concentrated in one period.

### In the code
- `qts/alpha/evaluation.py::ic_report(pred, target, periods_per_year=None, trading_days_per_year=252, overlap=1, decay_targets=None)` returns a typed **`ICReport`** dataclass: `n, mean_ic, ic_sd, icir, t_stat, t_stat_nw, nw_lags, hit_rate, periods_per_year, by_year, ic_series, decay`. `t_stat` is the plain $\overline{IC}/\operatorname{sd}\cdot\sqrt n$ (right for non-overlapping samples); `t_stat_nw` is the Newey–West t with `nw_lags = newey_west_lags(n, overlap)`, the one to read for a daily IC series of multi-day labels. `decay` holds the mean IC against each target in `decay_targets` (the pipeline passes `AlphaConfig.horizon_days = 20` and `secondary_horizons = [5, 60]`). Dict-style access (`rep["mean_ic"]`, `rep.get(...)`) still works during the transition; `scalars()` gives the JSON-safe numbers written to `summary.json`.
- `rank_ic_series` is `qts/core/stats.py::rank_ic_series` (vectorised across dates, at least `IC_MIN_OBS = 20` valid pairs per date).
- `infer_periods_per_year`, `ic_decay_curve`, `rank_autocorrelation`, `signal_half_life`
- `incremental_ic(new_score, existing_composite, target, overlap=1)` returns an `IncrementalICReport` (an `ICReport` of the residualised score plus `standalone_mean_ic` and `composite_mean_ic`): it residualises the new score on the composite per date (`residualize_on`, OLS with intercept) and reports the IC of what is left — Gate 2's "adds something orthogonal" test.
- `qts/core/stats.py::rank_ic_series` is also what the Nagel audit uses (the old `qts/core/crosssection.py::ic_series` wrapper was removed).
- `qts/portfolio/diagnostics.py::transfer_coefficient` (exact $\Sigma$-correlation when $N \le 500$, else an approximation).

### Pitfalls & checks
- Score and label must be aligned at the formation date; with an execution delay, at the *traded* formation date (see above). The second audit re-derived `ICReport` n, mean, NW t and ICIR by hand to 1e-12.
- Overlapping labels make daily IC series autocorrelated: compute the t-stat on non-overlapping dates (the pipeline samples every $H$) or read `t_stat_nw`.
- Including names that are not in the universe at $t$ inflated IC in the past (audit finding 1, CRITICAL, fixed): scores are masked by `MarketData.eligibility()`.

### Worked example
The post-refactor demo (`runs/demo_post_refactor.txt`, lag 0, run before the second audit's fixes): mean rank IC 0.0392, ICIR 1.45 annualised at 12.6/yr, t 2.89, NW t 3.16 with $L = 3$, n 50. Check: per-period ICIR $= 1.45/\sqrt{12.6} = 0.408$; $t = 0.408\times\sqrt{50} = 2.89$ ✓. Lag: $\lfloor 4(50/100)^{2/9}\rfloor = \lfloor 3.43\rfloor = 3$ and $h = 1$, so $L = 3$ ✓. The NW t is *larger* than the plain t because the sampled IC series is slightly negatively autocorrelated. The decay line "5d: 0.0180, 20d: 0.0392, 60d: 0.0590" shows the slow planted signal: IC keeps rising with horizon.

Fundamental law sanity check: IC 0.03, 1,000 names, 12.6 rebalances a year gives $0.03\sqrt{12{,}600} = 3.4$ — obviously too high, because the 12,600 "bets" are far from independent. With the demo's realised TC of 0.53 it would be 1.8; the demo's actual gross Sharpe was 1.33 on 300 names. The law's value is directional: doubling IC doubles IR; quadrupling breadth only doubles it.

### References
- [Grinold (1989), The Fundamental Law of Active Management, Journal of Portfolio Management](https://doi.org/10.3905/jpm.1989.409211)
- [Clarke, de Silva & Thorley (2002), Portfolio Constraints and the Fundamental Law of Active Management, Financial Analysts Journal](https://doi.org/10.2469/faj.v58.n5.2468)

## 6. Decile portfolios and long–short spreads

### What it is
Sort stocks by the signal each formation date, split into ten equal groups (deciles), and track the return of each group over the next horizon. The **decile spread** is top decile minus bottom decile: the return of a simple long–short bet on the signal. It is cruder than IC but closer to money, and it is the plan's Gate 2 statistic.

### How it works
$$
LS_t = \sum_{i\in D_{10,t}} w_{it}\,R_{i,t\to t+h} - \sum_{i\in D_{1,t}} w_{it}\,R_{i,t\to t+h}
$$
where:
- $D_{10,t}$ and $D_{1,t}$ are the top and bottom deciles by score at $t$ (percentile rank $> 0.9$ and $\le 0.1$);
- $w_{it}$ are within-decile weights summing to 1 — equal weights, or market-cap weights for a **value-weighted (VW)** spread;
- $R_{i,t\to t+h}$ is the raw forward return over $h$ days.

The mean of $LS_t$ is the spread return per period; its Sharpe is annualised with $p/h$ periods per year; its t-statistic is Newey–West with $L = \min(\max(h, \lfloor 4(T/100)^{2/9}\rfloor), T-1)$ lags (section 4). Value weighting matters: equal-weighted spreads are dominated by small, illiquid stocks where anomalies are strongest and least tradable (`BUILD_PLAN.md` §2.3 on microcaps).

### In the code
`qts/alpha/evaluation.py::decile_spread(pred, fwd_raw, n=10, weights=None, horizon=1, periods_per_year=252)` returns a `DecileSpreadResult(returns, mean, sharpe, t_stat, horizon, nw_lags)`. The pipeline calls it on the traded scores at every 20th execution date (section 5, "Scoring the traded signal"), eligible names only, value-weighted by the market cap of the signal date, with `horizon=1` (non-overlapping) and `periods_per_year` = the inferred 12.6 shared with the IC report. The raw forward return starts at the execution close and uses the same delisting/halt rule as the labels. Scores are masked where the forward return is missing, so both legs use the same names.

### Pitfalls & checks
- With 300 names a decile is 30 stocks; value-weighting concentrates it further, so the spread is noisy. That is why the demo's decile t (1.67) is well below its IC t (2.89): same signal, noisier statistic.
- Delisting returns must be in the labels or the short leg looks better than it is (first audit finding 5). The haircut is `CostConfig.delisting_haircut = -0.30`, the single source for labels and backtest accounting; `AlphaConfig.delisting_return` was removed. A *temporary halt* must not get the haircut, and deciding which is which may not use future prices (second audit finding 1; see §12).

### Worked example
Post-refactor demo: "decile L/S (VW) mean 1.25%/period Sharpe 0.92 NW t 1.67 (L=3)". That demo predates the shared-frequency fix and annualised with 13 periods a year: Sharpe 0.92 means a period standard deviation of $1.25\%\times\sqrt{13}/0.92 \approx 4.9\%$. With $n = 50$ periods the plain t would be $(0.92/\sqrt{13})\sqrt{50} = 1.80$; the Newey–West t with 3 lags is 1.67. Under the current code the same series would report Sharpe $0.92\times\sqrt{12.6/13} = 0.906$ and the same t-statistic (the t does not depend on annualisation).

## 7. Drawdown, Calmar, Sortino, hit rate

### What they are
Sharpe treats upside and downside volatility alike and ignores the order of returns. Investors experience paths: a 20% peak-to-trough fall is what triggers redemptions or a kill switch. These metrics describe the path.

### How they work
$$
NAV_t = \prod_{s\le t}(1+r_s), \qquad DD_t = \frac{NAV_t}{\max_{s\le t} NAV_s} - 1, \qquad MDD = \min_t DD_t
$$
$$
\text{Calmar} = \frac{\text{CAGR}}{|MDD|}, \qquad \text{Sortino} = \frac{\overline{r - \tau}}{\sqrt{\tfrac1T\sum_t \min(r_t - \tau, 0)^2}}\sqrt{q}, \qquad \text{hit rate} = \frac{\#\{t: r_t > 0\}}{\#\{t: r_t \ne 0\}}
$$
where:
- $NAV_t$ is net asset value per unit invested; $DD_t \le 0$ is the drawdown, and $MDD$ the maximum drawdown (most negative value);
- CAGR is the compound annual growth rate (`annualized_return`, geometric);
- $\tau$ is the per-period target return (default 0);
- the Sortino denominator is the **downside deviation**: the root mean square of shortfalls below target, averaged over *all* periods (not only negative ones);
- the hit rate ignores exactly-zero days (non-trading days in the backtest).

Calmar's numerator and denominator have different time scales: max drawdown grows with sample length (longer samples see worse luck), so Calmar ratios from different-length backtests are not comparable. For a Gaussian strategy, Sortino ≈ Sharpe × $\sqrt 2$; a Sortino much higher than that indicates positive skew.

### In the code
`qts/backtest/metrics.py::drawdown_series`, `max_drawdown`, `calmar`, `sortino(returns, periods_per_year=252, target=0.0)`, `hit_rate`; summary keys `max_drawdown_net`, `calmar_net`, `sortino_net`, `hit_rate_net`. The plan's AlgoXpert protocol "treats a catastrophic drawdown as an immediate fail" (`BUILD_PLAN.md` §8); no drawdown gate exists in `hurdles.py` yet (Roadmap).

### Worked example
Post-refactor demo: net CAGR 6.40%, max drawdown −5.3%, Calmar $6.40/5.3 = 1.21$ ✓. A NAV path 1.00 → 1.10 → 0.99 → 1.05 has drawdowns 0, 0, −10%, −4.5%; MDD = −10%.

## 8. Multiple testing

### What it is
If you test 20 useless signals at the 5% level, you expect one false "discovery". Multiple-testing corrections adjust the bar for the number of tests run.

### How it works
**Family-wise error rate (FWER)** — probability of at least one false positive among $m$ independent tests at level $\alpha$:
$$
\text{FWER} = 1 - (1-\alpha)^m
$$
For $m = 20$, $\alpha = 0.05$: $1 - 0.95^{20} = 64\%$.

**Bonferroni** controls FWER by testing each hypothesis at $\alpha/m$. It is valid under any dependence but very conservative when tests are correlated.

**False discovery rate (FDR)** is the expected *fraction* of rejections that are false. **Benjamini–Hochberg (BH)** controls it at level $q$: sort the p-values $p_{(1)} \le \dots \le p_{(m)}$, find the largest $k$ with
$$
p_{(k)} \le \frac{k}{m}\,q
$$
and reject hypotheses $1, \dots, k$. FDR is the natural criterion for a research pipeline: you accept that some admitted signals are false, as long as most are not.

**Effective number of trials.** Ten variants of one momentum signal with lookbacks 6, 7, …, 15 months are not ten independent tests. Nyholt (2004), originally for correlated genetic markers, estimates the effective number from the eigenvalues $\lambda_j$ of the $M\times M$ correlation matrix of the trials' return streams:
$$
M_{\text{eff}} = 1 + (M-1)\left(1 - \frac{\operatorname{Var}(\lambda)}{M}\right)
$$
where:
- $M$ is the raw number of trials;
- $\operatorname{Var}(\lambda)$ is the sample variance (ddof = 1) of the eigenvalues. Independent trials give all $\lambda_j = 1$, variance 0, $M_{\text{eff}} = M$; perfectly correlated trials give one eigenvalue $M$ and the rest 0, variance $M$, $M_{\text{eff}} = 1$.

### Why trial counting matters
Every correction above needs $m$. Researchers systematically undercount: abandoned ideas, parameter tweaks, "quick checks", re-runs after fixing a bug. The DSR (section 9) is only as honest as its $N$, which is why the repo has an append-only ledger (section 13).

### Why this choice
The repo does not run a BH procedure over signals (there is no signal-zoo search yet); it encodes the multiple-testing lesson in two places: the Harvey–Liu–Zhu threshold $t > 3$ for standalone signals (the gate `t_stat`) and the DSR's expected-maximum deflation with Nyholt's $M_{\text{eff}}$. HLZ's own framework includes Bonferroni, Holm and BHY adjustments; $t > 3$ is their practical summary.

### In the code
`qts/validation/ledger.py::TrialLedger.effective_trials(C)` (static, Nyholt), `effective_trials_for(family)` (builds $C$ from stored `returns_monthly`, needs ≥ 3 trials with ≥ 12 overlapping months, and adds trials without streams as independent). The result is a real number (e.g. 1.67) and the DSR uses it **unrounded**: the pipeline used to pass `round(n_eff)`, which turned 2.4 into 2 and under-deflated (second audit, finding 17).

### Worked example
Ten p-values: 0.001, 0.008, 0.012, 0.030, 0.041, 0.20, 0.35, 0.50, 0.62, 0.90. Naively, five are "significant" at 5%. Bonferroni ($0.05/10 = 0.005$): one. BH with $q = 0.05$: thresholds $k\times0.005$ are 0.005, 0.010, 0.015, 0.020, …; $p_{(3)} = 0.012 \le 0.015$ but $p_{(4)} = 0.030 > 0.020$, and no later $p_{(k)}$ is under its threshold, so BH rejects three. Nyholt: ten trials with pairwise correlation 0.5 have eigenvalues 5.5 and nine of 0.5; sample variance 2.5; $M_{\text{eff}} = 1 + 9(1 - 0.25) = 7.75$. At correlation 0.9, $M_{\text{eff}} = 2.71$. Five trials at 0.9 give 1.76 (the audit's check value).

### References
- [Benjamini & Hochberg (1995), Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing, JRSS B](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x)
- [Nyholt (2004), A Simple Correction for Multiple Testing for Single-Nucleotide Polymorphisms in Linkage Disequilibrium with Each Other, American Journal of Human Genetics](https://doi.org/10.1086/383251)

## 9. Probabilistic Sharpe, expected maximum Sharpe, Deflated Sharpe, MinTRL and MinBTL

### What it is
The **probabilistic Sharpe ratio (PSR)** converts an observed Sharpe into the probability that the *true* Sharpe exceeds a benchmark, accounting for sample length, skew and kurtosis. The **deflated Sharpe ratio (DSR)** is a PSR whose benchmark is the Sharpe you would expect from the *best of N useless strategies* — the luck you bought by searching.

### Why we need it here
The plan's Gate 5 is DSR > 0.95 "with recorded trial count". It is the repo's main defence against selection bias.

### How it works
**PSR** (Bailey & López de Prado 2012):
$$
PSR(SR^*) = \Phi\!\left(\frac{\widehat{SR} - SR^*}{\operatorname{se}(\widehat{SR})}\right)
$$
where:
- $\widehat{SR}$ is the observed **per-observation** Sharpe (daily for daily data);
- $SR^*$ is the benchmark Sharpe, per observation;
- $\operatorname{se}$ is the Mertens standard error of section 3 with $n$ observations, skew $\gamma_3$, raw kurtosis $\gamma_4$.

**Expected maximum Sharpe of N noise trials.** Suppose $N$ independent strategies each have true Sharpe 0, and their *estimated* Sharpes are approximately normal with variance $V$. The best of them is not zero: the expected maximum of $N$ standard normals grows like $\sqrt{2\ln N}$. Bailey & López de Prado (2014) use an accurate closed-form approximation from extreme-value theory:
$$
\mathbb{E}\big[\max_n \widehat{SR}_n\big] \approx \sqrt{V}\left((1-\gamma)\,\Phi^{-1}\!\left(1 - \frac1N\right) + \gamma\,\Phi^{-1}\!\left(1 - \frac{1}{Ne}\right)\right)
$$
where:
- $V$ is the variance of the Sharpe estimates *across trials*;
- $\Phi^{-1}$ is the inverse standard normal CDF;
- $e \approx 2.718$;
- $\gamma \approx 0.5772$ is the **Euler–Mascheroni constant**.

<toggle heading="h3">
Intuition for the expected-maximum formula

The maximum of $N$ iid standard normals lies near the $1 - 1/N$ quantile: of $N$ draws, on average one exceeds $\Phi^{-1}(1-1/N)$. That quantile is the "location" of the maximum's Gumbel limiting distribution, and the spacing between $\Phi^{-1}(1-1/N)$ and $\Phi^{-1}(1-1/(Ne))$ is its "scale" (the Gumbel CDF moves from $e^{-1}$ to $e^{-e^{-1}}$ over one scale unit). The mean of a Gumbel distribution is location + $\gamma$ × scale, where $\gamma$ — the Euler–Mascheroni constant, $\lim_{n\to\infty}(\sum_{k=1}^{n} 1/k - \ln n)$ — appears because it is the mean of the standard Gumbel. Rearranging location + $\gamma$(quantile₂ − quantile₁) gives the $(1-\gamma)$, $\gamma$ weights above. For $N = 100$ and $V = 1$ it gives 2.53, close to the exact simulated value (the audit reproduced the paper's ≈2.5).
</toggle>

**DSR** (Bailey & López de Prado 2014):
$$
DSR = PSR\big(SR^* = \mathbb{E}[\max \widehat{SR}]\big) = \Phi\!\left(\frac{\widehat{SR} - \mathbb{E}[\max\widehat{SR}]}{\sqrt{\dfrac{1 - \gamma_3\widehat{SR} + \frac{\gamma_4-1}{4}\widehat{SR}^2}{n-1}}}\right)
$$
DSR is a probability: "the chance that the selected strategy's true Sharpe is above what pure luck from this search would have produced". With $N = 1$ the expected maximum is 0 and DSR equals PSR against zero.

**Minimum track record length (MinTRL)** — how many observations are needed for $PSR(SR^*) \ge$ `prob`:
$$
\text{MinTRL} = 1 + \left(1 - \gamma_3\widehat{SR} + \frac{\gamma_4 - 1}{4}\widehat{SR}^2\right)\left(\frac{\Phi^{-1}(\text{prob})}{\widehat{SR} - SR^*}\right)^2
$$
It is infinite when $\widehat{SR} \le SR^*$.

**Minimum backtest length (MinBTL)** — Bailey, Borwein, López de Prado & Zhu (2014): with $N$ trials of pure noise, how long must the backtest be before their expected best Sharpe drops below a target? Under the null each daily Sharpe estimate has variance about $1/n$, so $\mathbb{E}[\max] \approx z_N/\sqrt n$ with $z_N$ the unit-variance expected maximum, giving
$$
n_{\min} = \left(\frac{z_N}{SR_{\text{target}}}\right)^2
$$
in observations ($SR_{\text{target}}$ per observation). Read in reverse: on a backtest of $n$ days, trying more than a handful of configurations *guarantees* a best Sharpe at target even with zero skill.

### Why this choice
DSR is the standard correction in the literature the plan cites (Gençay applies it; Bailey & López de Prado define it) and it needs only quantities the ledger records. Its known weaknesses are handled elsewhere: it cannot detect leakage (suspect gate, PIT tests), and it is sensitive to $N$ and $V$ (ledger, Nyholt, variance floor).

### In the code
`qts/validation/deflated_sharpe.py`:

```python
def expected_max_sharpe(n_trials: float, var_sr: float) -> float:
    n_trials = float(n_trials)          # fractional N (Nyholt) used as is
    if n_trials <= 1:
        return 0.0
    sd = math.sqrt(max(var_sr, 0.0))
    if n_trials < 2.0:                  # closed form is invalid here (see below)
        return float((n_trials - 1.0) * _closed_form_unit(2.0) * sd)
    return float(sd * _closed_form_unit(n_trials))


def _closed_form_unit(n_trials: float) -> float:   # eq. 8 with V = 1, N >= 2
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)
```

**Small effective N.** Equation 8 is a large-$N$ approximation and it misbehaves just above $N = 1$: $\Phi^{-1}(1 - 1/N)$ goes to $-\infty$ as $N \to 1^+$, so the formula is *negative* for $N$ below about 1.28 (at $N = 1.01$ it gives $-0.79\sqrt{V}$). The true expected maximum of $N$ zero-mean draws is never negative and is exactly 0 at $N = 1$. A negative benchmark Sharpe would *inflate* the DSR: a family of five near-identical re-runs has Nyholt $N_{\text{eff}} \approx 1.01$, and the uncorrected formula turned a DSR of 0.864 (plain PSR against zero) into 0.958, flipping the gate from FAIL to PASS. Found on 2026-11-09 when a fresh ledger (one trial, $N = 1$) reported 0.864 for a run whose stored report said 0.958. The code now interpolates linearly between 0 at $N = 1$ and the closed form at $N = 2$ ($0.52\sqrt{V}$; the exact value for two standard normals is $0.564\sqrt{V}$), which is non-negative, continuous and non-decreasing; above $N = 2$ the closed form is used unchanged. `tests/test_ledger_concurrency.py::test_expected_max_sharpe_never_negative_for_small_effective_n` pins this.

**Fractional effective trials.** The formula is smooth in $N$, so nothing requires an integer. The code used to truncate with `int(n_trials)`. A Nyholt count of 2.4 then became 2, the expected maximum fell, and the DSR rose: the rounding always favoured the strategy. Now `n_trials` is a float everywhere (`deflated_sharpe_ratio`, `deflated_sharpe_ratio_annual`, the pipeline), and `tests/test_ledger_concurrency.py::test_expected_max_sharpe_uses_fractional_trials` pins the closed form at a non-integer $N$. `min_backtest_length` still takes an integer trial count; it is a planning tool, not a gate input.

Other functions: `probabilistic_sharpe(sr, sr_benchmark, n, skew, kurt)`, `deflated_sharpe_ratio(sr_observed, n_trials, var_trials_sr, n_obs, skew, kurt)` (per-observation inputs), `deflated_sharpe_ratio_annual(..., periods_per_year=252)` (annual Sharpe and annual variance, divides the variance by 252), `min_track_record_length(sr, sr_benchmark, skew, kurt, prob=0.95)`, `min_backtest_length(n_trials, target_sr_annual)` and `..._years`, and `dsr_from_trial_returns(T×N matrix)` which picks the best column and uses the empirical cross-trial variance.

How the pipeline feeds it (`qts/pipeline.py::ResearchPipeline._dsr_inputs`), reading the ledger family `pipeline_phase{p}:{data_hash[:12]}` (section 13):
- no ledger, or one trial in the family → $N = 1$, $V = 0$ (plain PSR vs 0);
- two trials → raw count and their sample variance;
- three or more → `effective_trials_for(family)` (Nyholt, unrounded) and `family_sharpe_variance(family)`;
- whenever more than one trial exists, $V$ is **floored at $252/n$** (annual units), the pure sampling-noise variance of an annual Sharpe over $n$ days. Before the first audit $V$ was hard-coded to 0.25 (finding 7, HIGH).

The order matters: `evaluate()` logs this run first, then runs and logs any PBO grid configurations, and only then reads the family. The DSR therefore always counts the run being judged and every configuration searched alongside it. The second audit checked `_dsr_inputs` by hand on a three-trial ledger (Nyholt $N_{\text{eff}} = 1.667$, floor $252/n = 0.336$ applied, DSR 0.957186 with the closed form at $N = 1.667$; with the small-$N$ interpolation above the expected maximum at that $N$ is slightly higher, $0.347\sqrt{V}$ instead of $0.337\sqrt{V}$, so the DSR is slightly lower).

The DSR threshold is `ValidationConfig.dsr_threshold = 0.95`.

### Pitfalls & checks
- **Annual vs per-observation inputs**: use the `_annual` wrapper with annual numbers.
- **Raw kurtosis**: see section 2.
- **Undercounted N** makes DSR too lenient; **overcounted N** (identical re-runs) makes it too harsh. The ledger de-duplicates only re-runs that are provably identical (section 13); when in doubt it counts.
- **$N_{\text{eff}}$ just above 1** (re-runs of one configuration) must deflate at least as hard as $N = 1$; the raw closed form does the opposite (see *Small effective N*).
- **$V$ is not a free parameter**: under the null it is about $252/n$ annualised (0.25 for $n \approx 1{,}000$ days, 0.05 for 20 years). Dispersion of true Sharpes across trials adds to it.

### Worked example: N = 1, 10, 100 trials
Observed annual Sharpe 1.0 over 5 years ($n = 1260$), Gaussian returns, cross-trial annual Sharpe variance $V_A = 0.25$ (standard deviation 0.5 across trials). Per day: $SR_d = 0.0630$, $V_d = 0.25/252$; $\operatorname{se}_d = 0.0282$ (0.448 annual).

| N | $\mathbb{E}[\max SR]$, annual | z = (1.0 − E[max]) / 0.448 | DSR |
|---|---|---|---|
| 1 | 0 | 2.23 | **0.987** |
| 10 | $0.5\times1.575 = 0.787$ | 0.48 | **0.683** |
| 100 | $0.5\times2.531 = 1.265$ | −0.59 | **0.277** |
| 1000 | $0.5\times3.255 = 1.628$ | −1.40 | **0.081** |

The same backtest passes the 0.95 bar if it was the only thing tried, and is more likely luck than skill if it was the best of 100. With skew −0.5 and raw kurtosis 6 the standard error rises slightly (0.0287 daily) and the DSRs move toward 0.5: 0.986, 0.680, 0.280.

MinTRL (95%, Gaussian, vs $SR^* = 0$): annual Sharpe 1.0 needs 2.7 years; 0.6 needs 7.5 years; 0.5 needs 10.8 years. MinBTL (target annual Sharpe 1.0): 10 trials need 2.5 years; 100 trials 6.4 years; 1,000 trials 10.6 years of data before the best of pure noise is expected to stay below 1.0.

### References
- [Bailey & López de Prado (2012), The Sharpe Ratio Efficient Frontier, Journal of Risk (SSRN 1821643)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1821643)
- [Bailey & López de Prado (2014), The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality, Journal of Portfolio Management (SSRN 2460551)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)
- [Bailey, Borwein, López de Prado & Zhu (2014), Pseudo-Mathematics and Financial Charlatanism, Notices of the AMS (SSRN 2308659)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2308659)

## 10. Probability of Backtest Overfitting (CSCV)

### What it is
PBO asks a different question from DSR. Given a grid of $N$ configurations (say, ten risk-aversion values), if you pick the configuration that looked best in one half of the history, how often does it land in the *bottom half* of the configurations on the other half? If the answer is "about half the time", your selection procedure has no skill: it is picking noise.

### Why we need it here
DSR deflates a single number by a theoretical maximum; PBO measures, empirically, whether *your own* selection procedure generalises across time. The plan's Gate 5 requires PBO < 0.5 (`ValidationConfig.pbo_threshold`).

### How it works: Combinatorially Symmetric Cross-Validation (Bailey, Borwein, López de Prado & Zhu 2017)
1. Build a $T\times N$ matrix $M$ of per-period returns: one column per configuration, all on the same $T$ dates.
2. Cut the rows into $S$ contiguous groups of (nearly) equal size; $S$ must be even.
3. For each of the $\binom{S}{S/2}$ ways to choose $S/2$ groups as **in-sample (IS)**, the other $S/2$ groups are **out-of-sample (OOS)**. The split is symmetric: every IS set is someone else's OOS set.
4. For each split, compute the performance metric (Sharpe by default) of every configuration IS and OOS. Let $n^*$ be the IS-best configuration.
5. Find $n^*$'s OOS relative rank $\omega = \text{rank}(n^*)/(N+1) \in (0, 1)$, where rank 1 is the worst (ties get the average rank).
6. Compute the **logit** $\lambda = \ln\frac{\omega}{1-\omega}$. $\lambda > 0$ means the IS winner is above the OOS median; $\lambda < 0$, below.
7. $PBO = \frac{1}{C}\sum_c \mathbb{1}[\lambda_c < 0]$ over the $C$ splits.

Also reported:
- **degradation slope**: regress the IS-best's OOS metric on its IS metric across splits. A slope near zero or negative means higher IS performance does not predict higher OOS performance — the signature of overfitting.
- **prob_oos_loss**: fraction of splits in which the IS-best configuration loses money OOS.

### Why this choice
CSCV uses all the data for both selection and evaluation without a single arbitrary split, and the symmetric design means every period is judged OOS equally often. Alternatives (a single holdout; walk-forward reselection) give one draw of the same quantity.

### In the code
`qts/validation/pbo.py::probability_of_backtest_overfitting(perf_matrix, n_groups=16, metric="sharpe", max_combinations=5000, seed=0)` returns `pbo, logits, n_combinations, n_configs, degradation_slope, degradation_intercept, degradation_r2, prob_oos_loss, is_best_metric, oos_best_metric, mean_oos_rank_pct`. It is vectorised with per-group sums, sums of squares and counts, so cost is $O(\binom{S}{S/2}\cdot N)$. $\binom{16}{8} = 12{,}870$ splits exceed the cap, so 5,000 are subsampled uniformly with a warning.

The pipeline computes PBO only when `SystemConfig.pipeline_pbo_configs` lists overrides, e.g. `[{"portfolio.trade_aversion": 0.5}, {"portfolio.trade_aversion": 2.0}]`: it re-runs the full pipeline per configuration on the same backtest dates, stacks the net return streams (plus the base run), uses $S = 8$ groups when fewer than 320 days are common (else 16), and returns `None` (gate skipped) for a degenerate grid. Choose knobs that the default sizing actually reads: with `risk_aversion_mode = "target_vol"` the risk aversion is searched per date, so a grid over `portfolio.risk_aversion` (the example in the `SystemConfig` comment) changes nothing (it is only the fallback when the closed-form starting value is not finite) unless the grid also sets `risk_aversion_mode = "fixed"`.

Every grid configuration is itself a trial. Each grid pipeline is constructed with the parent's `ledger_path`, so its run is logged to the same family before the parent reads the DSR inputs. Before the second audit (finding 11, MEDIUM) the grid ran without a ledger: a 10-point grid search added nothing to $N$, which is precisely the selection the DSR exists to punish. One consequence is worth knowing: running a PBO grid lowers the base run's DSR, because $N$ grows by the grid size. That is the intended price of the search.

### Pitfalls & checks
Audit finding 10 (fixed): the comparison is now strict (`logits < 0`), so an IS-best that lands exactly on the OOS median is not counted as overfit (the old `<=` gave PBO ≈ 0.66 for three pure-noise configurations); NaNs raise instead of becoming zero returns; fewer than two distinct columns raise. Ties in the IS metric are broken by the first column (`np.argmax`).

### Worked example
Four configurations A–D, four groups of equal length, using the mean daily return (bp) as the metric so you can verify by hand. D is steady; A–C are volatile.

| Group | A | B | C | D |
|---|---|---|---|---|
| g0 | 4 | 1 | 2 | 2 |
| g1 | −1 | 3 | 0 | 1 |
| g2 | 2 | −2 | 3 | 2 |
| g3 | 0 | 2 | −1 | 1 |

$\binom{4}{2} = 6$ splits:

| IS groups | IS means (A,B,C,D) | IS-best | OOS means | OOS rank of best | ω | λ |
|---|---|---|---|---|---|---|
| g0,g1 | 1.5, 2, 1, 1.5 | B | 1, 0, 1, 1.5 | 1 | 0.2 | −1.39 |
| g0,g2 | 3, −0.5, 2.5, 2 | A | −0.5, 2.5, −0.5, 1 | 1.5 | 0.3 | −0.85 |
| g0,g3 | 2, 1.5, 0.5, 1.5 | A | 0.5, 0.5, 1.5, 1.5 | 1.5 | 0.3 | −0.85 |
| g1,g2 | 0.5, 0.5, 1.5, 1.5 | C (tie, first) | 2, 1.5, 0.5, 1.5 | 1 | 0.2 | −1.39 |
| g1,g3 | −0.5, 2.5, −0.5, 1 | B | 3, −0.5, 2.5, 2 | 1 | 0.2 | −1.39 |
| g2,g3 | 1, 0, 1, 1.5 | D | 1.5, 2, 1, 1.5 | 2.5 | 0.5 | 0 |

Five of six logits are negative and one is exactly zero (not counted), so $PBO = 5/6 = 0.83$. The degradation slope is −1.10: the better a configuration looked IS, the worse it did OOS. Selecting by backtest here is worse than useless, even though one configuration (D) is genuinely steady — it only wins IS once.

### References
- Bailey, Borwein, López de Prado & Zhu (2017), "The Probability of Backtest Overfitting", Journal of Computational Finance (SSRN 2326253): [link](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253)

## 11. Cross-validation for time series

### What it is
Cross-validation (CV) estimates out-of-sample performance by repeatedly training on part of the data and testing on the rest. Standard k-fold CV shuffles observations into random folds. For financial time series with multi-day labels, that leaks.

### Why k-fold shuffling leaks
A label at date $s$ is the 20-day forward return over $(s, s+20]$. A label at $s+1$ covers $(s+1, s+21]$: 19 of its 20 days are the same. If $s$ is in training and $s+1$ in test, the model has effectively seen the test answer. Features leak the same way: a 252-day momentum computed at adjacent dates is nearly identical. Shuffled CV on such data reports spectacular, fictitious accuracy.

### How it works: purging and embargo
- **Purge**: remove from training every observation whose label window overlaps any test label window.
- **Embargo**: additionally remove training observations in a buffer just *after* the test block, because serial correlation in features and returns can carry test information forward into training data that follows it.

In trading-day positions, a training observation $s$ uses $[s, s+h]$. For a test block $[t_0, t_1]$, whose labels reach $t_1 + h$, `purge_train` drops $s$ if
$$
t_0 - h \;\le\; s \;\le\; t_1 + h + \text{embargo}
$$
where $h$ is the label horizon and "embargo" the number of extra days dropped after the test block's forward reach.

```text
positions →   0 ......... t0-h ... t0 ======== t1 ... t1+h ... t1+h+E ......... n
train          ##########|purged  |  TEST BLOCK  | purged    | embargo |#########
               (kept)     (label reaches test)    (test labels   (serial
                                                   still running) correlation)
```

**PurgedWalkForward** (expanding window, forward only). The region after `min_train_days` is cut into `n_splits` contiguous test blocks. For each block, training is everything *before* it, purged; nothing after the block is ever used. This mimics live trading exactly and yields one OOS path.

```text
split 1:  [train.........]--[T1]
split 2:  [train....................]--[T2]
split 3:  [train...............................]--[T3]
          -- = purged gap of h days
```

**PurgedKFold**. $K$ contiguous folds; training is all other folds, purged and embargoed on both sides of the test fold. It uses more data than walk-forward but trains on the future to predict the past, which is acceptable for estimating a stable relationship and unacceptable for anything path-dependent.

**Combinatorial Purged CV (CPCV)** (López de Prado 2018, ch. 12). Cut the sample into $N$ groups; every combination of $k$ groups is a test set, the rest (purged and embargoed around each test group) is training:
$$
\#\text{splits} = \binom{N}{k}, \qquad \#\text{paths} = \binom{N}{k}\frac{k}{N}
$$
Each group appears in $\binom{N-1}{k-1}$ test sets. Stitching one OOS prediction per group, taken from different splits, gives a complete OOS **backtest path** covering the whole history; there are $\binom{N}{k}k/N$ such disjoint paths. Instead of one OOS Sharpe you get a distribution, which is what the plan wants from skfolio-style CPCV (`BUILD_PLAN.md` §8).

```text
N = 6 groups, k = 2  ->  15 splits, 5 paths
split:   1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
g0       T  T  T  T  T  .  .  .  .  .  .  .  .  .  .
g1       T  .  .  .  .  T  T  T  T  .  .  .  .  .  .
g2       .  T  .  .  .  T  .  .  .  T  T  T  .  .  .
...      (each group is tested in 5 splits -> one per path)
```

With the defaults (`ValidationConfig.cpcv_n_groups = 8`, `cpcv_k_test = 2`): 28 splits and 7 paths.

### Why this choice
Walk-forward is the honest default and is what the ML trainer actually uses; CPCV is the tool for estimating the *distribution* of OOS performance when deciding between configurations. Plain k-fold is never used on labels.

### Purged walk-forward folds for the gates
The hurdle gates that ask "does it hold up across time?" (`net_sharpe_delta` and `ic_sign`, section 14) need folds. They used to be calendar years. That had three problems, listed by the handbook review (fix #27) and the second audit (finding 16): partial first and last years with a few weeks of data counted as whole folds; the IC gate and the Sharpe gate used different folds; and `ValidationConfig.n_splits` / `embargo_days` were dead settings. Now both gates use the test blocks of a `PurgedWalkForward` splitter laid over the backtest's return dates:

```python
def purged_fold_blocks(dates, n_splits, embargo_days, label_horizon, min_fold_days=20):
    n = len(dates)
    k = min(int(n_splits), n // max(int(min_fold_days), 1))
    if k < 1:
        return []
    pwf = PurgedWalkForward(n_splits=k, embargo_days=int(embargo_days),
                            label_horizon=int(label_horizon), min_train_days=0)
    return [dates[t0 : t1 + 1] for t0, t1 in pwf.test_blocks(dates)]
```

(`qts/pipeline.py::purged_fold_blocks`.) With the defaults (`n_splits = 6`, `embargo_days = 20`, `min_fold_days = 20`) a 1,005-day backtest gives six contiguous blocks of 167–168 days. If the backtest is short, the number of blocks is reduced until each has at least `min_fold_days` dates; below that the fold gates are skipped and a note says so. Only the test blocks are used. Every backtest date is already out-of-sample (the ML sleeve is trained walk-forward with its own embargo), so the splitter's purge and embargo of the *training* side have nothing to act on here. `embargo_days` is passed for consistency but does not change the blocks; the second audit pointed this out and it is accepted as is. The blocks (start, end, days, net Sharpe) are written to `summary.json` (`folds`) and to `PipelineResult.folds`; the per-block mean IC is computed by `ResearchPipeline._ic_by_fold` and appears only through the `ic_sign` gate.

### In the code
- `qts/validation/splits.py::purge_train`, `label_windows_overlap`, `PurgedWalkForward(n_splits=6, embargo_days=20, label_horizon=20, min_train_days=252)` with `test_blocks(dates)`, `PurgedKFold(n_splits=5, embargo_days=20, label_horizon=20)`, `CombinatorialPurgedCV(n_groups=8, k_test=2, embargo=20, label_horizon=20)` with `backtest_paths()` (greedy assignment: each split's test groups fill the lowest-numbered free path) and `assemble_paths(dates, oos_predictions)`, `check_no_overlap(splitter, dates)`.
- All splitters are scikit-learn compatible (`split(X, y=None, groups=None)`, `get_n_splits`).
- The walk-forward ML trainer has its own exact purge (`qts/alpha/trainer.py`). With label horizon $h$ and execution delay $L$, a label at $s$ is realised over $(s+L, s+L+h]$, so the embargo is `WalkForwardTrainer.embargo = (AlphaConfig.embargo_days or horizon_days) + L` (21 with the defaults). For a prediction block starting at position $p$ a row is used only if it is outside the embargo *and* its `label_realisation` date is before $p$ (a halted name can delay it; §12). Refit every `AlphaConfig.refit_frequency_days = 252`, first fit after `train_min_days = 756`. → see **Alpha model › Walk-forward training**.
- `ValidationConfig.n_splits`, `embargo_days` and `min_fold_days` feed `purged_fold_blocks`; `cpcv_n_groups` / `cpcv_k_test` hold the plan's CPCV defaults for research use and are not consumed by `ResearchPipeline`.

### Pitfalls & checks
- Embargo must be at least the label horizon plus the execution delay; the trainer adds $L$ automatically.
- `tests/test_validation.py::test_no_train_label_overlaps_test` runs `check_no_overlap` on all three splitters; `test_walk_forward_is_forward_only` and `test_cpcv_counts_and_paths` check structure.
- Purging protects labels; it does not protect against features computed with future data (section 12).

### References
- López de Prado (2018), *Advances in Financial Machine Learning*, Wiley, ch. 7 and 12.
- [skfolio CPCV (as cited in BUILD_PLAN §8)](https://arxiv.org/pdf/2507.04176)

## 12. Leakage tests in this repo

Leakage (**look-ahead bias**) is any use of information not available at decision time. Because DSR cannot detect it, the repo makes it hard to express and cheap to catch.

### pit_shift_test — the universal point-in-time test
**What**: builds a feature panel on the full history and again on history truncated at a cut date, then requires the two to agree on every date up to the cut (same index, same NaN pattern, values within `atol = 1e-10`).
**Why it works**: any use of future rows — `shift(-k)`, centred rolling windows, full-sample normalisation, backward fills — changes values before the cut when the future is removed.
**Code**: `qts/validation/leakage.py::pit_shift_test(build_fn, md, cut_date, atol=1e-10, raise_on_fail=False)`. The audit ran equivalent truncation checks on `AlphaModel.as_of`, risk `as_of`, exposures, factor returns and cost snapshots: all bit-identical.

### Planted oracle test
**What**: feeds a backtest a "feature" equal to the future 20-day return and requires the result to be flagged as suspect (annual Sharpe > 2).
**Why**: it operationalises "DSR cannot detect leakage". An oracle produces DSR ≈ 1.00; only the plausibility gate catches it. If the oracle is *not* flagged, either the suspect gate is broken or the backtest is not using the feature.
**Code, toy version**: `planted_oracle_test(backtest_fn, md, horizon=20, suspect_sharpe=2.0)`; with `backtest_fn=None` it uses `simple_rank_backtest` (daily top/bottom 20% equal-weight, positions at $t$ earn $r_{t+1}$). `tests/test_validation.py::test_planted_oracle_is_flagged`.
**Code, through the real pipeline** (first audit finding 11, handbook review fix #45; now closed): `tests/test_pipeline.py::test_planted_oracle_through_pipeline_is_flagged_suspect` fits a `ResearchPipeline` on a 60-name synthetic market, then overwrites the alpha model's ML-sleeve slot with the z-scored *label itself* (the future residualised 20-day return), sets the composite weight to 0 and backtests the last 80 dates on which that label exists. Everything downstream is real: eligibility, risk model, optimizer with costs and constraints, simulator with the one-day execution delay, evaluation and gates. The test requires net Sharpe > 2, the `suspect` gate at FAIL, `overall` FAIL and mean IC > 0.5. It is marked `slow`.

<callout icon="💡" color="blue_bg">
Why inject the label rather than build an oracle feature? The point is to prove that the *wiring* from alpha to gates carries information through intact. If a perfect forecast entered the optimizer and the backtest came out at Sharpe 0.5, something between the alpha model and the P&L (alignment, lag, eligibility masking, units) would be destroying signal. If it came out above 2 but `suspect` passed, the gate would be broken. Either failure would be invisible on an honest, weak signal.
</callout>

### AST shift(−k) registry
**What**: the `@feature_registry` decorator parses the feature function's source into an abstract syntax tree (AST) and refuses, at import time, any literal `.shift(-k)` with $k>0$ (positional or `periods=`). It also records which `MarketData` panels the function reads and a hash of its source.

```python
@feature_registry
def mom_12_1(md: MarketData) -> Wide: ...
# a body containing md.close.shift(-5) raises LeakageError at decoration time
```

**Limits (documented)**: only the decorated function's own source; only literal arguments; it cannot see `shift(k)` with variable `k`, `.iloc` tricks, `rolling(center=True)` or `bfill`. It is a tripwire; `pit_shift_test` is the gate.

### Nagel collapse audit
**What Nagel (2025) says**: in "Seemingly Virtuous Complexity in Return Prediction", Nagel shows that random-feature models with far more parameters than training observations, fitted on short windows, produce forecasts that are effectively a recency-weighted average of past returns — a volatility-timed momentum strategy — regardless of the data. Their historical success comes from that implicit momentum, and on synthetic data with reversals they still build momentum and lose.
**The audit**: `qts/validation/synthetic_audit.py::nagel_collapse_test(model_predict_fn, n_assets=100, n_days=1000, n_lags=60, horizon=20, half_life=10, max_abs_ic_t=3.0, max_abs_corr=0.5)` generates a panel with GARCH-style volatility clustering and a common volatility factor but **no** predictability in the mean (`generate_noise_panel`), fits the caller's model on the first half (60 lagged returns → 20-day forward return), leaves a gap of `horizon` days, and predicts the second half. It passes if the IC t-stat (on non-overlapping dates) has $|t| < 3$ **and** predictions correlate with a recency-weighted mean of past returns (half-life 10 days) at $|\rho| \le 0.5$, both pooled and cross-sectionally. A model that fails the correlation check has "collapsed" onto a momentum/reversal rule.

### Label overlap check
`label_overlap_check(train_dates, test_dates, horizon)` is the date-based counterpart of the splitter checks, for ad-hoc splits.

### Point-in-time test suites
`pit_shift_test` checks one feature at one cut. Since the refactor, the test suite applies the same idea, "computing it on data truncated at $t$ must give the same answer at $t$", to every decision input. Equality is exact where the arithmetic is identical. Where only the order of a floating-point sum differs (the truncated fit emits names in a different order), the tolerance is about $10^{-14}$ absolute or $10^{-12}$ relative.

| Suite | What it pins | Notes |
|---|---|---|
| `tests/test_pit_properties.py` | Every raw feature panel and the processed long frame of `FeatureBuilder`, `build_universe_mask`, `TransactionCostModel.snapshot_from_market`, `BarraStyleRiskModel.as_of`, and `AlphaModel(use_ml=False).as_of` (score, expected return, IC estimate, half-life, components) on full vs `md.slice_until(t)` | Property-based with **hypothesis**: the library generates the cut dates. Runs are derandomised (fixed examples) so CI is stable. Market: 60 names × 420 days, top-45 universe, so eligibility changes enter every cross-section |
| same file, `test_labels_depend_on_future_features_do_not` | Multiplies every close after $t$ by a random per-name factor: the label *at* $t$ must change, labels whose window ended by $t$ and every feature up to $t$ must not | The inverse check: a label that ignores its own future is broken too |
| `tests/test_label_halts.py` | 15 names halted for 10 days and then resumed. Labels realised by $t$, the trailing IC and the expected return at $t$ are identical between a full fit and a fit on `slice_until(t)`, at lag 0 and lag 1. The causal delisting rule and vendor `delisting_dates` haircut true delistings only. The label with lag $L$ is $\text{close}_{s+L}\to\text{close}_{s+L+h}$, and the trainer's embargo grows with $L$ | Written for second audit finding 1 (HIGH) |
| `tests/test_apply_mask_history.py` | A name entering the universe keeps its true trailing liquidity features and cost-model ADV, and features stay PIT after masking | Second audit finding 6 |
| `tests/test_risk.py::test_style_exposures_ignore_future_universe_members` and the dated-industry tests | Risk exposures at $t$ do not depend on which names join the universe, or which industry labels appear, after $t$ | Lag investigation; second audit finding 5 / risk finding 4 |
| `tests/test_signal_lag_real.py` | Through the real optimizer and cost model: every optimizer input at execution date $t$ is dated $t - 1$ except the book and NAV | Lag investigation |

**Full-pipeline slice equivalence (second audit).** The strongest evidence is an audit script rather than a CI test: the second audit fitted the whole `ResearchPipeline(SystemConfig())` (alpha including the ML sleeve, hybrid APCA risk model, cost snapshot, eligibility) on the 300 × 1,512 synthetic market and refitted it on `md.slice_until(t)` at 14 date/configuration pairs. Every decision input at $t$ agreed: worst absolute difference 4.9e-14 for the composite, 8.4e-15 with the ML sleeve, 1.1e-14 with 30 planted stock splits; `ic_estimate`, `half_life` and the eligible set exactly. The verdict was "on the synthetic market the decision inputs are point-in-time", with two exceptions the synthetic market could not show: halts that later resume (label rule) and industry labels first used after $t$ (risk factor set). Both are fixed and now have tests.

<callout icon="⚠️" color="yellow_bg">
Equivalence to $10^{-14}$ on synthetic data is necessary, not sufficient. The synthetic generator can now plant halts, delistings and stock splits (`SyntheticSpec.n_halts`, `n_delistings`, `n_splits`; all 0 by default, so the demo has none), but it has no IPOs, no restatements, no index reconstitutions and no vendor quirks. Real-data look-ahead (restated fundamentals, a "current" industry map, survivorship in the price source) can only be caught when real data is loaded. → see **Audit history & known limitations**.
</callout>

### Pitfalls & checks
The structural fixes from the audits matter as much as the tests. Each of these was a leakage or survivorship bug that none of the tripwires above would have caught when it was written:
- universe membership leaking into the cross-section (first audit finding 1, CRITICAL);
- alpha themes in the label residualisation (finding 2, CRITICAL);
- restated fundamentals (finding 6) and delisting returns missing from labels (finding 5);
- risk style exposures standardised over names that join the universe after $t$ (lag investigation);
- a halt classified as a delisting using prices after $t$ (second audit finding 1, HIGH).

→ see **Data & universe** and **Audit history & known limitations**.

### References
- [Nagel (2025), Seemingly Virtuous Complexity in Return Prediction, NBER w34104](https://ideas.repec.org/p/nbr/nberwo/34104.html)

## 13. The trial ledger

### What it is
An append-only JSON-lines file (`runs/trial_ledger.jsonl`) with one record per experiment: timestamp, name, family, full configuration, config hash, metrics (annual and/or daily Sharpe, number of observations, skew, raw kurtosis, mean IC, decile t…), data hash, code version, notes and optionally monthly returns. Think of it as the lab notebook that an auditor can count.

### Why it must never be reset
The DSR deflates by the expected maximum of the $N$ trials actually run. Deleting the ledger — or not logging the "quick check" that failed — shrinks $N$, weakens the deflation and lets selection bias through unpunished. The ledger is to research what double-entry books are to accounting: its value comes from being complete. The module docstring says so in bold: "This file is the input to the deflated Sharpe ratio and must never be reset."

### How it works
The design went through two versions. The first (after the first audit) rewrote the file in place to merge identical re-runs. The second audit showed that this lost trials under concurrent writers and merged runs that were not identical (findings 2 and 3, both HIGH). The current design has four rules.

**1. Strictly append-only, under a lock.** `log(...)` serialises one complete JSON line and appends it with `O_APPEND` while holding an exclusive `fcntl.flock` on `<path>.lock`, then `fsync`s. Readers take the shared lock. Nothing ever rewrites the file. Several researchers, agents, the CLI and the API job runner can log to the same ledger at once without losing each other's trials; `tests/test_ledger_concurrency.py::test_concurrent_appends_all_present` checks it with parallel processes. (Before: one appender writing 400 trials next to one re-runner left 296 of 401 records.)

**2. Identity is a key, and de-duplication happens at read time.** With `replace_identical=True` the record carries a `dedup_key`, the SHA-256 of
$$
(\text{name},\ \text{config hash},\ \text{data hash},\ \text{code version},\ \text{hash of the run spec})
$$
where:
- **config hash** is `SystemConfig.hash()` (SHA-256 of the canonical JSON of every behavioural knob, including `signal_lag_days`);
- **data hash** is `qts.core.manifest.hash_market_data`: every `MarketData` panel, the `extra` dict, the industry map and the eligibility panel (the pre-refactor hash covered `close` only);
- **code version** is `qts.core.manifest.code_version()`: the git SHA of a clean checkout, `<sha>+dirty` when tracked files are modified *or* untracked non-ignored files exist (an untracked module the run imports changes the code; second audit finding 14), and `unknown` outside git;
- the **run spec** (`ResearchPipeline.run_spec`) holds every setting outside `SystemConfig` that changes the numbers: `use_ml`, `phase`, the ML factory's identity (module, qualified name and a hash of its code object and closure, so two lambdas differ), `composite_weight`, `rebalance_every`, `venue`, `nav0`, `warmup_days`, the backtest date range and the effective signal lag.

When the ledger is read, records sharing a `dedup_key` collapse into the latest one (placed where the first appeared), with `n_reruns` and `ts_first`. `raw_records()` returns every line untouched.

**3. When in doubt, count.** A record gets no key, and is therefore always its own trial, when `replace_identical` is False, the data hash is empty, or the code version is empty, `unknown` or dirty. A dirty tree may differ from the last run in ways nobody recorded, so two runs from dirty trees are two trials even if every hash matches. `tests/test_pipeline.py::test_pipeline_ledger_key_and_family` runs five pipelines (a changed `rebalance_every`, a new code version, two dirty-tree runs) and requires five trials.

**4. One family per phase and data set.** The pipeline logs under `pipeline_phase{p}:{data_hash[:12]}`. Trials on another data set (another synthetic seed, real data) are a different search and do not pool into the variance $V$; signal-lag and configuration variants on the same data do pool. (Before: one family per phase across all data, finding 17.)

- `log(name, config, metrics, data_hash="", code_version="", notes="", family=None, config_hash=None, returns_monthly=None, replace_identical=False, run_spec=None)`. `family` defaults to `name.split(":")[0]`, so `"mom:lb=12"` and `"mom:lb=6"` share family `mom`. The pipeline logs `sharpe_annual`, `n_obs`, skew, raw kurtosis, mean IC, decile t, `use_ml`, `phase`, `signal_lag_days` and the monthly net returns.
- **Queries** (all on the de-duplicated view): `n_trials(family)`, `records(family)`, `trial_sharpes_daily/annual`, `family_sharpe_variance` (the $V$ of the expected maximum), `returns_matrix`, `effective_trials_for` (Nyholt, fractional), `dsr_for(name, family, n_trials)`, `report()`.

### Why this choice
JSONL is human-readable, diff-able and appendable with one system call; an advisory file lock is enough for one machine. Read-time de-duplication keeps the file a faithful log (nothing is ever lost or overwritten) while the DSR still sees "an identical re-run is not a new trial". Keying on code version and run spec closes the failure the second audit demonstrated: three runs with different code versions collapsed to $n_{\text{trials}} = 1$, keeping only the last Sharpe. Iterating on code with a fixed config is the most common research loop, so that bug hid exactly the trials that matter most.

### Pitfalls & checks
- Share one ledger across every researcher or agent touching the same data; separate ledgers hide trials from each other.
- `flock` is local and advisory: it does not protect a ledger on NFS or one edited by hand.
- Records without `data_hash` (pre-audit) cannot be de-duplicated; they stay, and they count.
- Commit before a run you want de-duplicated. Runs from a dirty tree always add a trial, which is conservative but makes the DSR depend on how often you run uncommitted code.
- The family change means records logged under the old family name `pipeline_phase1` are not read by new runs. They are still in the file (and still in `raw_records()`), but they no longer deflate anything. For the synthetic demo this is harmless; for real research, never change the family convention mid-search.

### Worked example
`runs/trial_ledger.jsonl` holds seven raw records. Six are in the legacy family `pipeline_phase1`: three pre-audit runs (annual Sharpe −0.006, −0.466, 0.135; empty data hash), the post-first-audit run (0.742), the post-refactor demo at lag 0 (0.956, data `acc387f0eca55816`, code `63ce32b…+dirty`) and its lag-1 companion run (0.111), logged after the lag-0 run had read its inputs. The remaining record is `pipeline_phase2`.

When the lag-0 demo computed its DSR, its family had $N_{\text{raw}} = 5$ (the four earlier phase-1 records plus itself). Only two of the five carry monthly return streams, below Nyholt's minimum of three, so $N = 5$. The sample variance of the five annual Sharpes is
$$
V = \operatorname{var}(-0.006,\ -0.466,\ 0.135,\ 0.742,\ 0.956) = 0.332
$$
above the floor $252/1005 = 0.251$, so $V = 0.332$ (the demo prints "n_trials 5.0, var SR 0.332"). Under the current family rule a new run on the same data would start the family `pipeline_phase1:acc387f0eca5`, which none of the legacy records belong to, so its first DSR would be a plain PSR ($N = 1$); the lag-1 companion logged after it would see $N = 2$. Runs from a dirty tree, like both demo runs, are never de-duplicated. The pre-audit runs were buggy, but they were trials: the researcher saw their numbers before changing code.

## 14. The hurdle gates

### What they are
A fixed checklist, defined before any result is seen, that a candidate must pass to be admitted. Pre-registration is the point: gates written after the backtest are just another fork in the garden.

### How they work
`qts/validation/hurdles.py::HurdleReport(cfg: ValidationConfig).evaluate_signal(...)` evaluates ten gates. Each `GateResult` has a status:
- **PASS** / **FAIL** — the input was supplied and compared with the threshold;
- **skipped** — the input was not supplied or the gate does not apply. A skipped gate does not affect `overall` **unless it is required**: gates in `ValidationConfig.required_gates` (default `["phase_sharpe", "suspect", "dsr"]`) are converted to FAIL with reason "REQUIRED gate skipped: …".

`overall = all(passed for applied gates)`. The result dict carries `overall, phase, gates, status, details, reasons, failed, skipped, required`; `HurdleReport.format` prints it.

| Gate | Definition | Pass rule (as coded) | Rationale | Skipped when |
|---|---|---|---|---|
| `t_stat` | Decile long–short Newey–West t-stat (pipeline: VW, eligible names, traded score, non-overlapping 20-day periods) | $t > 3.0$ (`hurdle_t_stat`) | Harvey–Liu–Zhu multiple-testing bar | not standalone (`standalone=False`), or value missing |
| `ic_sign` | Fraction of folds whose mean sampled IC is **> 0** | fraction ≥ 2/3 (`min_fold_pass_fraction`) | Consistency across time; added in code (not in the plan table) | `ic_by_fold` missing or empty |
| `incremental_ic` | Fraction of folds whose incremental rank IC is **> 0.005** (after residualising on the existing composite) | fraction ≥ 2/3 | Redundancy against the factor zoo and crowding | not supplied (always for a standalone signal) |
| `net_sharpe_delta` | Fraction of folds whose net Sharpe improvement is **≥ +0.05** | fraction ≥ 2/3 | AlgoXpert plateau-and-majority rule | not supplied (e.g. backtest too short for one fold) |
| `breakeven_cost` | Breakeven cost per side (net of borrow) vs modeled cost per side | breakeven ≥ 2 × modeled | Margin for under-modeled costs (Chen & Velikov) | either input missing; zero turnover (∞) passes |
| `dsr` | Deflated Sharpe ratio | DSR $> 0.95$ (`dsr_threshold`) | Selection bias given the ledger's trial count | missing (**required** → FAIL) |
| `pbo` | Probability of backtest overfitting (CSCV) | PBO $< 0.5$ (`pbo_threshold`) | Selection procedure must beat a coin flip | no configuration grid |
| `phase_sharpe` | Net annual Sharpe of the full pipeline | phase 1: ≥ 0.3; phase 2+: ≥ 0.5 | Realistic net targets (see below) | missing (**required** → FAIL) |
| `suspect` | Net annual Sharpe | must **not** exceed 2.0 | "Any net backtest Sharpe > 2 is a defect until leakage, survivorship, cost and selection are ruled out"; DSR cannot detect leakage | missing (**required** → FAIL) |
| `failed_solves` | Share of rebalance solves that did not reach an optimal status (`BacktestResult.failed_solve_fraction()`; the simulator held the book on those days) | share ≤ 0.02 (`max_failed_solve_frac`) | A broken optimizer holds yesterday's book every day; the backtest then looks like a strategy but is a stale portfolio | no solve was attempted (`failed_solve_frac=None`) |
| `risk_model_sanity` | Share of rebalance solves on which the risk model's annualised ex-ante vol of the incoming book (or of the $\sqrt{\text{cap}}$ reference book when it is empty) exceeded `max_exante_vol` = 2.0 (`BacktestResult.exante_vol_alert_fraction()`) | share ≤ 0.005 (`max_exante_vol_alert_frac`) | A covariance blown up by a data error makes the target-vol optimizer liquidate and the book then stays empty for years (the 2018-03 real-data incident, → see **Risk model › Return cleaning**); the gate turns a quiet flat NAV into a FAIL with the dates and the contributing names/factors in `diagnostics["exante_vol_alerts"]` | the simulator did not record the counter, or no solve was attempted |

The comparison operators in this table are exactly those in `qts/validation/hurdles.py` and in the module docstring. The handbook review found the text and code disagreeing (fix #26: reason strings said ">" where the code used ">="). The code was changed to match the plan: an IC of exactly 0 no longer counts as a positive fold, and an incremental IC of exactly 0.005 no longer passes.

**Breakeven cost** (`qts/backtest/metrics.py::breakeven_cost_bps`) is the per-side cost, in basis points of traded notional, that would drive the mean return to zero:
$$
c_{\text{BE}} = \frac{\overline{g^{b}_t}}{\overline{\sum_i |x_{it}|}}\times 10^4, \qquad g^{b}_t = g_t - \frac{\text{borrow}_t}{NAV_t}
$$
where:
- $g_t$ is the daily gross return (positions times stock returns, before any cost);
- $\text{borrow}_t$ is the day's borrow fee in USD and $NAV_t$ the pre-trade NAV, so $g^{b}_t$ is the **gross-after-borrow** return (`BacktestResult.returns["gross_after_borrow"]`);
- $\sum_i |x_{it}|$ is the two-sided daily turnover (fraction of NAV), aligned positionally with the return it produced.

Borrow is a holding cost, not a per-side trading cost, so it does not scale with turnover. The old formula started from $g_t$ and left borrow out of the gate entirely; the second audit (costs finding 1, MEDIUM) measured breakeven 26.2 bp against 24.3 bp once borrow was deducted, and a cost-sensitivity net Sharpe of 0.528 at 1× against the simulator's actual 0.463. Both the breakeven and the cost-sensitivity table now start from gross-after-borrow. The modeled cost is the backtest's realised average cost per side (spread + impact + fees over traded dollars).

**Standalone vs incremental** (review item 10). The $t>3$ bar applies only to standalone signals. An incremental feature inside an ensemble is judged by incremental IC and net-Sharpe delta instead: a feature can be valuable precisely because it is weak alone but orthogonal.

**Standalone pipeline convention.** For a standalone run there is no baseline, so the pipeline passes the net Sharpe of each purged walk-forward test block (section 11) as `net_sharpe_delta_by_fold`; the gate then reads "net Sharpe by fold ≥ +0.05 in ≥ 2/3 of folds" against a zero baseline. `ic_sign` uses the mean sampled IC of the execution dates inside the *same* blocks (second audit finding 16; before, the IC "folds" were calendar years including partial first and last years).

**Phase-1 0.3 vs phase-2 0.5.** The original plan put 0.5 on Phase 1. The review (`docs/REVIEW_NOTES.md` item 1) lowered Phase 1 to 0.3 because Chen & Velikov's post-2005 net numbers make 0.5 a *ceiling* for a six-theme rank composite, not a baseline. Phase 2 (composite + ML) must reach 0.5: the ML layer has to earn its complexity against a good linear composite. A composite between 0.3 and 0.5 is "in the plausible range for its kind"; the same number from an ML model is "not worth the extra overfitting risk".

**Not gates (yet).** The plan's "expected live Sharpe 0.25–0.5× backtest" is an expectation, not a code check; drawdown and robustness splits (size tercile, VIX regime) are Roadmap items.

### In the code
`HurdleReport.evaluate_signal(...)` returns a typed **`HurdleResult`** (`overall`, `phase`, `required`, and an ordered list of `GateResult(name, passed, value, threshold, reason, applied, required)` with a `status` property). Both are frozen dataclasses. Views `status`, `passed`, `details`, `reasons`, `failed`, `skipped` are derived; `to_dict()` / `from_dict()` round-trip it through `hurdles.json` in the run directory; legacy dict access (`res["overall"]`) still works.

```python
ic_by_fold = self._ic_by_fold(icr.ic_series, blocks)   # same purged blocks as fold_sharpes
hurdles = HurdleReport(cfg.validation).evaluate_signal(
    ic_by_fold=ic_by_fold if len(ic_by_fold) else None,
    decile_tstat=ds.t_stat,
    net_sharpe_delta_by_fold=fold_sharpes,  # standalone: baseline 0
    breakeven_cost_bps=breakeven,           # from gross_after_borrow
    modeled_cost_bps=summ.get("avg_cost_bps_per_side"),
    dsr=dsr, pbo=pbo, phase=self.phase,
    net_sharpe=summ["sharpe_net"], standalone=True)
```

(`qts/pipeline.py::ResearchPipeline.evaluate`.) Constants: `INCREMENTAL_IC_MIN = 0.005`, `NET_SHARPE_DELTA_MIN = 0.05`, `BREAKEVEN_MULTIPLE = 2.0`, `SUSPECT_SHARPE = 2.0`. Thresholds live in `ValidationConfig`; to tighten the phase bars, change `phase1_min_net_sharpe` / `phase2_min_net_sharpe`; to make PBO mandatory, add `"pbo"` to `required_gates`; to change the folds, change `n_splits` and `min_fold_days`.

### Pitfalls & checks
- Before the first audit (finding 3) the pipeline fed the IC t-stat into the decile gate and `pbo=None` failed every run, so `overall` could never be PASS. A gate that cannot pass is not a gate. The fix introduced *skipped* with a required set.
- Gates are only as good as their inputs: six folds make the 2/3 rule coarse (4 of 6 passes), and each fold's Sharpe over ~167 days has a standard error near $\sqrt{252/167}\approx 1.2$.
- Every gate is evaluated on synthetic data so far. On the planted market a FAIL means "the plumbing reports a weak, planted signal honestly", not "the strategy is bad".

### Worked reading of the demo's report card
`runs/demo_audit2.txt` (a demo on the code after the second audit's fixes) did not exist when this page was written, so the reading below uses **`runs/demo_post_refactor.txt`**: synthetic market, 300 stocks, 1,005 backtest days (2019-12-10 to 2023-10-17), phase 1, no ML, run on 2026-11-05 from an uncommitted tree at `63ce32b`. Three things about it differ from the current code, and the reading flags each: it ran at **lag 0** (the default was flipped to 1 afterwards); `ic_sign` still used calendar years while `net_sharpe_delta` already used purged blocks; and breakeven and cost sensitivity started from gross returns, leaving out borrow. The lag investigation also showed that its lag-0/lag-1 pair is not reproducible on committed code (§ "Execution-delay check" below).

| Line | Reading |
|---|---|
| `OVERALL: FAIL (phase 1)` | At least one applied gate failed (two did: `t_stat` and `dsr`). |
| `[FAIL] t_stat: decile L/S t = 1.67 <= 3.0` | The VW decile spread over 50 non-overlapping 20-day periods has a Newey–West t of 1.67 (3 lags). Even the IC t (2.89) is below 3; the 30-name value-weighted deciles are noisier still. A real failure on merit. |
| `[PASS] ic_sign: IC > 0 in 4/5 folds (need >= 0.67)` | Old convention: five calendar years with IC observations, four positive, 80% ≥ 67%. Current code would report $k/6$ over the six purged blocks below. |
| `[skipped] incremental_ic: not applied (standalone signal)` | No existing composite to be incremental to. Not required. |
| `[PASS] net_sharpe_delta: ... >= +0.05 in 6/6 folds` | Six purged walk-forward blocks of 167–168 days with net Sharpes 0.76, 0.86, 1.13, 1.35, 0.94, 0.73. All exceed 0.05. |
| `[PASS] breakeven_cost: breakeven 44.1 bps >= 2x modeled 10.5 bps (= 20.9 bps)` | Modeled cost 10.47 bp per side, so the bar is 20.93 bp; costs could rise 4.2× before gross alpha is gone. The cost-sensitivity table agrees (net Sharpe 1.17 at 0.5×, 1.01 at 1×, 0.70 at 2×, 0.38 at 3×). Its 1× value is above the simulator's 0.96 because, in this demo, the table started from gross returns and omitted borrow; current code starts from gross-after-borrow. |
| `[FAIL] dsr: DSR = 0.704 <= 0.95` | See the calculation below. |
| `[skipped] pbo: PBO not supplied (...)` | No `pipeline_pbo_configs` grid; not required. |
| `[PASS] phase_sharpe: phase-1 net Sharpe 0.96 >= 0.3` | Clears the phase-1 bar comfortably. |
| `[PASS] suspect: net Sharpe within plausible range` | 0.96 < 2. |
| `skipped (not required): incremental_ic, pbo` | Summary line: the two skipped gates are not in `required_gates`. |

**DSR, by hand.** Inputs (from `summary.json` and the ledger record): annual net Sharpe 0.956, $n = 1005$, skew 0.012, raw kurtosis 3.059, $N = 5$, $V_A = 0.332$ (section 13). Per day: $SR_d = 0.956/\sqrt{252} = 0.0602$. Expected maximum: $\sqrt{0.332}\,\big(0.4228\,\Phi^{-1}(0.8) + 0.5772\,\Phi^{-1}(1 - 1/(5e))\big) = 0.5765\times(0.4228\times0.8416 + 0.5772\times1.4497) = 0.688$ annual. Standard error: $\sqrt{(1 - 0.012\times0.0602 + \frac{2.059}{4}\times0.0602^2)/1004} = 0.0316$ daily, 0.501 annual. $z = (0.956 - 0.688)/0.501 = 0.535$; $\Phi(0.535) = 0.704$ ✓.

Two lessons. First, even with $N = 1$ the PSR against zero would be only 0.97: four years of Sharpe 0.96 just clears 95% on its own (MinTRL is 3.0 years). Second, the four earlier phase-1 runs in the family cost about 0.27 of DSR — the price of having looked. The demo therefore fails on merit (decile t and DSR), not by construction, which is what the README says it should do on a synthetic market: the demo validates plumbing, not alpha.

**Execution-delay check.** The demo ends with a lag-0 vs lag-1 comparison: net Sharpe 0.956 → 0.111. That gap is *not* the cost of a one-day delay. The lag investigation (`docs/notes/lag_investigation_2026-11-05.md`) could not reproduce it on committed code (there: gross 0.68 → 0.60, net 0.23 → 0.14; paired block-bootstrap difference −0.08 ± 0.15). Over five seeds lag 1 was not worse than lag 0 (mean gross difference +0.16), and the composite keeps 93–98% of its IC at lag 1. The lesson for reading report cards: a single 1,000-day path moves by 0.5–0.8 Sharpe under small code or setting changes, so compare variants with paired, multi-seed tests and report the bootstrap standard error of the difference.

`runs/demo_phase1_post_audit.txt` (first-audit era: N = 4, DSR 0.666, calendar-year folds) and `runs/demo_phase2.txt` (pre-audit gate format, ICIR annualised with 252) are history, not current reports.

### References
- [AlgoXpert protocol (as cited in BUILD_PLAN §2.4, §8)](https://arxiv.org/pdf/2603.09219)
- `docs/BUILD_PLAN.md` §2.4 (hurdle table), `docs/REVIEW_NOTES.md` items 1 and 10, `docs/audit/RESOLUTION.md`, `docs/audit/second_pass_alpha_data_validation.md` (findings 2–4, 11, 14–17), `docs/audit/second_pass_costs_backtest_api.md` (finding 1), `docs/notes/lag_investigation_2026-11-05.md`.

## Roadmap
- A drawdown gate (catastrophic-drawdown immediate fail, per the AlgoXpert protocol) and robustness splits (EW vs VW, size terciles, VIX regimes) in `hurdles.py`.
- Use CPCV paths (`ValidationConfig.cpcv_*`) to report a distribution of out-of-sample Sharpe for configuration choices, not only the six walk-forward blocks.
- A paired, multi-seed comparison helper (block bootstrap of the difference) for "variant A vs variant B" questions such as lag 0 vs lag 1 (lag investigation, §9).
- A CI version of the full-pipeline slice-equivalence check with halts, delistings and splits switched on in `SyntheticSpec` (second audit, finding 19).
- Re-run every gate on real point-in-time data; until then all verdicts on this page are about the plumbing.

Done since the first draft: purged walk-forward gate folds, the end-to-end planted oracle, the Newey–West lag floor at $h$, and run manifests with `code_version` in the ledger.
