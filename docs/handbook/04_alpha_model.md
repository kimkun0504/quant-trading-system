# Alpha model

<callout icon="🧭" color="gray_bg">
The alpha model turns raw market and accounting data into one number per stock per day: a forecast of how that stock will do over the next 20 trading days *relative to other stocks with the same risk profile*. It builds 18 point-in-time features, standardises them cross-sectionally (winsorise → industry-demean → rank-gauss), combines them two ways — a transparent six-theme rank composite with light dynamic weighting, and a walk-forward-trained LightGBM + MLP + ridge ensemble — blends the two, and converts the blended score into an expected return with Grinold's rule $\alpha = \text{IC}\cdot\sigma\cdot z$. Everything is designed around one sobering fact from the research: a realistic rank IC is 0.02–0.05, so the model's job is to be *honest and slightly right* across a thousand stocks, not spectacularly right about a few. Code lives in `qts/alpha/`; configuration in `qts/core/config.py::AlphaConfig`.
</callout>

<callout icon="🆕" color="green_bg">
**What changed (Nov 2026)**
- **Labels are aligned to the trade.** With the default one-day execution lag $L$ (`ExecutionConfig.signal_lag_days = 1`) the label of signal date $s$ is the return from close $s+L$ to close $s+L+h$; the walk-forward embargo and the trailing-IC window grow by $L$ (§4, §7, §8).
- **Causal halt/delisting rule.** Whether a price gap is a delisting (−30%) or a temporary halt is decided only with data available when the label is realised; `label_realisation` tells every consumer when a label row may be used (§4). The old rule looked at later prices.
- **D1 roles:** the label is residualised on exactly the styles the optimizer neutralises (`qts.core.roles`); the low-risk premium is intentionally excluded from the ML target (§4). **D5:** one IC-shrink convention, applied to `default_ic` too (§8).
- **Evaluation:** Newey–West lags $\max(h, \lfloor 4(T/100)^{2/9}\rfloor)$; IC decay at the secondary horizons (5, 20, 60 days); the pipeline scores the signal that was actually traded (§9).
- **ML:** hyper-parameters live in `AlphaConfig.lgbm` / `AlphaConfig.mlp`; MLP early stopping is time-contiguous; the rank-IC metric is vectorised (§6).
- **Features** use every name's full pre-entry history for rolling statistics, while cross-sections are taken over eligible names only (§3). Per-sleeve half-lives (`components_half_life`) are attached to the output (§8).
</callout>

## 1. What an alpha model is, and what ours predicts

### What it is

An **alpha model** is the forecasting component of a quantitative trading system. "Alpha" is return *not* explained by exposure to broad, compensated risks (the market rising, small caps outperforming). The model scores every stock each day: "this one should beat the average, that one lag it". Downstream, the risk model (→ see **Risk model**) says how stocks co-move, the cost model (→ see **Transaction cost model**) says what trading costs, and the optimizer (→ see **Portfolio construction**) trades off all three.

Analogy: a forecaster who predicts not "will it rain in Boston?" but "which of 1,000 cities will be wetter than the national average this month?" That question maps onto a market-neutral long–short book: buy the top of the ranking, sell the bottom, and the national weather (the market) cancels out.

### Why we need it here

Without a forecast the optimizer can only minimise risk and cost — i.e. hold nothing. The alpha model supplies the expected-return vector $\alpha$ in the optimizer's objective, via `qts.core.types.AlphaOutput`: standardised `score` and `expected_return` (fraction, over `horizon_days` = 20) per ticker, `half_life_days`, `ic_estimate` (the shrunk IC actually used), per-sleeve `components` for attribution and per-sleeve `components_half_life`.

### Cross-sectional versus time-series prediction

| | Time-series prediction | Cross-sectional prediction |
|---|---|---|
| Question | Will stock $i$ (or the market) go up or down tomorrow? | Which stocks will beat which others? |
| Target | $r_{i,t+1}$ itself | $r_{i,t+1}$ minus the cross-sectional average (or minus a risk-model fit) |
| Dominant noise | The market and industry factors | Idiosyncratic noise only |
| Natural portfolio | Long/flat or long/short a single asset over time | Long top-ranked, short bottom-ranked, at every date |
| Evaluation | Hit rate, $R^2$ over time | Per-date correlation between forecast and outcome (the IC, §9) |

We do **cross-sectional** prediction only. Market direction is hard to forecast and irrelevant to a beta-neutral book, while every date's cross-section is a fresh experiment with ~1,000 observations, so a tiny per-stock edge is detectable and diversifiable.

### Why 20-day *residual* returns, not raw next-day returns

The target the model learns (`qts/alpha/targets.py::residualize_forward_returns`) is the **20-trading-day forward return, residualised against risk factors and industries, then z-scored per date**. Three pieces of evidence drive each word of that sentence.

**Horizon: 20 days, not 1.** Robeco's study of the term structure of machine-learning alpha found that one-month-horizon ML models had strong *gross* alphas but net US performance close to zero after 2004, while models trained on 3-, 6- and 12-month targets did considerably better net of costs over 2004–2021, because they trade less and load on slower themes such as quality and value (pseudo-out-of-sample, cost-inclusive). A next-day target would push the model toward exactly the fast, high-turnover signals that costs destroy. Twenty days is the compromise between that evidence and the label-overlap and sample-size cost of longer horizons. `AlphaConfig.secondary_horizons = [5, 60]` adds 5- and 60-day labels to the **IC-decay** section of the evaluation report (§9): diagnostics only, never trained on. Separate 5- and 60-day model heads are still on the Roadmap.

**Residual: strip out what the risk model already explains.** A raw 20-day return is dominated by the market, the industry and loadings on size, beta and volatility. Trained on raw returns, a model spends its capacity learning "high-beta stocks rise when the market rises" — risk, not alpha, and hedged away by the optimizer anyway. In the deep-learning statistical-arbitrage study of Guijarro-Ordonez, Pelger & Zanotti, the *same* CNN+Transformer reached an out-of-sample Sharpe ratio of 4.16 on IPCA-factor residuals, 3.36 on PCA residuals and 3.21 on Fama–French residuals, against only 1.64 on raw returns (roughly 550 large liquid US stocks, OOS 2002–2016, **gross** of costs). That ≈2.5× improvement from changing nothing but the target is the single most transferable lesson in the recent literature.

**Z-scored per date.** Cross-sectional dispersion varies a lot over time (crisis months are far wider than calm ones); z-scoring makes every date count equally so crisis months do not dominate training.

<callout icon="🤔" color="purple_bg">
Why not 60 or 120 days if longer is better net? Each extra day of horizon is another day of label overlap to purge (§7) and another day before a live IC can be measured. The cost-aware optimizer already handles slow signals by holding longer (target ≈50-day holding period, `PortfolioConfig.effective_holding_days_target`), so 20 days is the plan's default and the 60-day head the natural experiment.
</callout>

### What "good" looks like: the realistic IC range

The **information coefficient (IC)** is the correlation, across stocks on one date, between the forecast and the realised outcome. We use the *rank* (Spearman) version (§9). A perfect forecast has IC = 1; random guessing has IC ≈ 0.

Realistic values, from the build plan (all pre-cost):

| Setting | IC | Source and evidence class |
|---|---|---|
| Good daily model, Chinese A-shares | 0.04–0.06 | Qlib benchmarks (pseudo-OOS) |
| NASDAQ-100, OOS 2024–mid-2025 | 0.0162 | R&D-Agent(Q) (OOS) |
| S&P 500, LLM-mined alphas | 0.0056 | AlphaAgent (OOS) |
| Monthly horizon convention | 0.05 good, 0.10 very good | MSCI, *Converting Scores into Alphas* (practitioner rule of thumb) |
| **This project's working target** | **0.03–0.05** rank IC, 20-day residual, 1,000 names | Build plan §2.3 synthesis |

US large caps run 2–4× lower than A-shares (more efficiently priced). An IC of 0.03 explains roughly $0.03^2 \approx 0.1\%$ of cross-sectional return variance — useless-sounding until you account for breadth.

### Why a tiny IC is enough: the fundamental law of active management

Grinold's **fundamental law of active management** says that the information ratio (IR: annualised active return divided by annualised active risk, i.e. the Sharpe ratio of the bets) is approximately

$$
\text{IR} \approx \text{IC}\cdot\sqrt{\text{BR}}
$$

where:
- $\text{IC}$ is the correlation between forecasts and outcomes (dimensionless, per bet);
- $\text{BR}$ ("breadth") is the number of *independent* bets per year.

**Intuition.** Each stock-period is a coin flip biased by a tiny amount: one flip tells you nothing, thousands per month add up. Signal adds linearly in the number of bets while noise adds in quadrature — the same reason a standard error falls like $1/\sqrt{n}$.

**A simulation-style example.** Take 1,000 stocks, a forecast $s_i \sim N(0,1)$ and outcomes $r_i = 0.03\,s_i + \sqrt{1-0.03^2}\,\varepsilon_i$ with independent noise. Simulating 500 dates (numpy, seed 0):

| Scenario | Mean rank IC | SD of IC across dates | IC hit rate | ICIR per period | ICIR annualised (×√12.6) |
|---|---|---|---|---|---|
| Constant true IC 0.03, independent stocks | 0.027 | 0.031 | 81% | 0.85 | **3.0** |
| True IC varies by date: $0.03 + 0.08\cdot N(0,1)$ | 0.023 | 0.082 | 60% | 0.28 | **1.0** |

(The mean rank IC is 0.027 rather than 0.030 because the Spearman correlation of two Gaussians is $\tfrac{6}{\pi}\arcsin(\rho/2) \approx 0.955\rho$.) In the first row the SD of the IC is almost exactly $1/\sqrt{N} = 0.032$: pure sampling noise. The ICIR per period is then $0.03/0.032 \approx \text{IC}\sqrt{N}$ — the fundamental law. Annualised over 12.6 non-overlapping 20-day periods it is ≈3, i.e. $\text{IC}\sqrt{1000 \times 12.6}$.

The second row is what the real world looks like: some months the signal works, some months it reverses (a momentum crash, a value rally), so the IC itself has a time-series standard deviation far above $1/\sqrt{N}$. The achievable IR collapses to ≈1 before costs.

**The assumptions that make the law optimistic** — and why the second row is the right mental model:
1. *Independent bets.* Stocks share factors, so 1,000 stocks are not 1,000 independent bets; residualising the target (§4) pushes the remaining bets toward independence.
2. *Constant IC.* Real IC varies with regime (row 2).
3. *Unconstrained implementation.* The transfer coefficient — the share of the forecast that survives constraints into positions — is ≈0.3 for constrained long-only and 0.7–0.8 for long–short (Clarke, de Silva & Thorley); the full law is $\text{IR} \approx \text{TC}\cdot\text{IC}\cdot\sqrt{\text{BR}}$.
4. *No costs.* The law is gross.

That chain — IC 0.03 → gross IR ≈ 1 → after transfer coefficient and costs → net Sharpe 0.5–1.0 — is exactly the build plan's calibration of success, and why any backtest above ≈2 net is treated as a bug until proven otherwise.

<callout icon="⚠️" color="yellow_bg">
If a new signal shows an IC of 0.10 on US large caps at a 20-day horizon, the overwhelmingly likely explanation is a leak (look-ahead, survivorship, a label used as a feature), not a discovery. The pipeline's "suspect" gate flags any net Sharpe above 2 for exactly this reason (→ see **Validation**).
</callout>

### References
- [Guijarro-Ordonez, Pelger & Zanotti (2021), Deep Learning Statistical Arbitrage](https://ar5iv.labs.arxiv.org/html/2106.04028)
- [Robeco (2023), The term structure of machine learning alpha](https://www.robeco.com/en-uk/insights/2023/07/the-term-structure-of-machine-learning-alpha)
- [Qlib benchmarks](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md)
- [R&D-Agent(Q) (2025)](https://arxiv.org/html/2505.15155v2)
- [AlphaAgent (2025)](https://arxiv.org/html/2502.16789v2)
- [MSCI, Converting Scores into Alphas](https://www.msci.com/documents/10199/1645561/PI_Converting_Scores_Into_Alphas.pdf/7adf1f42-10aa-40eb-9e8c-ecc11eeba2d4)
- [Clarke, de Silva & Thorley, The cost of constraint (via Institutional Investor)](https://www.institutionalinvestor.com/article/2btgijdgl1kk3akuoutc0/home/the-cost-of-constraint)
- [Grinold (1989), The fundamental law of active management](https://doi.org/10.3905/jpm.1989.409211)

## 2. Features

### What they are

A **feature** (quant jargon: *characteristic* or *signal*) is a number computed for every stock on every date from information available at that date. `qts/alpha/features.py::FeatureBuilder.build` produces 18 of them as wide panels (rows = dates, columns = tickers), in the canonical order `FEATURE_ORDER`. The order is part of the train/live parity hash (§7): reordering forces a retrain.

### Why we need them

They are the model's only inputs. The set is the build plan's "Tier 1" list: the themes that survive value-weighting, microcap exclusion and costs in the large replication studies (value, momentum, quality/profitability, investment, low-risk), plus shorting signals with strong evidence and liquidity/size variables that let ML models condition on *where* in the market a signal works.

### Overview

Twelve features feed the six **themes** of the rank composite (§5) with a sign; six are ML-only (some also serve as risk exposures in the label, §4).

| Feature | Theme | Sign | Definition (at date $t$) | Literature status (US large caps, post-2005) |
|---|---|---|---|---|
| `value_bp` | value | + | $\ln(B/P)$ | Replicates gross; volatile; cleaner within industry |
| `value_ep` | value | + | $E/P$ | As above |
| `mom_12_1` | momentum | + | $P_{t-21}/P_{t-252}-1$ | Strongest single theme; crash-prone |
| `mom_6_1` | momentum | + | $P_{t-21}/P_{t-126}-1$ | As above |
| `mom_change` | ML only | n/a | $\text{mom\_12\_1}_t - \text{mom\_12\_1}_{t-63}$ | Conditioning variable |
| `industry_momentum` | momentum | + | cap-weighted industry mean of `mom_6_1` | Documented; explains part of stock momentum |
| `quality_gpa` | quality | + | gross profit / total assets | Robust surviving theme |
| `investment_ag` | investment | + | $-(\text{asset growth})$ | Robust surviving theme |
| `beta_252` | low_risk | − | 252-day beta to EW market | Weak recently |
| `idio_vol_252` | low_risk | − | 252-day residual vol | Lagged in Q4 2025 |
| `max_ret_21` | low_risk | − | max daily return, 21 days | Overlaps idio vol |
| `turnover_63` | ML only | n/a | mean(dollar volume / market cap), 63 d | Conditioning |
| `amihud_63` | ML only | n/a | $\ln(10^6\cdot\overline{\lvert r\rvert/\text{DV}})$, 63 d | Premium mostly small-cap |
| `log_dollar_vol_63` | ML only | n/a | $\ln$ mean dollar volume, 63 d | Conditioning |
| `log_mcap` | ML only | n/a | $\ln(\text{market cap})$ | Size: among weakest JKP themes |
| `short_interest` | short_interest | − | shares short / shares outstanding | Informed shorts |
| `borrow_fee` | short_interest | − | annualised loan fee | Top persistent non-text predictor |
| `strev_21` | ML only (risk-only proxy) | n/a | $-(\text{21-day return} - \text{industry mean})$ | Strong gross, dies standalone |

<callout icon="💡" color="blue_bg">
A **risk-based** story says a return compensates for a disliked risk — it should persist but may crash when you can least afford it. A **behavioural** story says investors misprice something and frictions stop arbitrage — it may decay as capital arrives and is strongest in hard-to-trade names. Each implies a different failure to watch for.
</callout>

### Value: `value_bp`, `value_ep`

**Definition.**

$$
\text{value\_bp}_{i,t} = \ln\!\left(\max\!\left(B_{i,t}/P_{i,t},\,10^{-6}\right)\right), \qquad \text{value\_ep}_{i,t} = E_{i,t}/P_{i,t}
$$

where $B$ and $E$ are the latest *filed* book equity and earnings per share and $P$ the close at $t$. The log tames B/P's skew (negative book is floored and ranks last); E/P stays in levels because it can be negative.

**Rationale.** Cheap stocks have out-earned expensive ones. Risk story: cheap firms are distressed and lose most in bad times. Behavioural story: investors extrapolate growth and over-pay for glamour.

**Status.** Value replicates in Jensen–Kelly–Pedersen (JKP: 85% of 153 factors replicate gross under consistent construction) but is volatile: it led in Q4 2025 (+2.59% vs MSCI ACWI, Russell Investments) after a long drought. A bank's B/P and a software firm's mostly differ by accounting convention, so both are industry-demeaned (§3). **Sign:** +.

**PIT.** Fundamentals are keyed on the **filing date** (`qts/data/loaders.py::sharadar_sf1_to_pit_fundamentals` uses Sharadar's `datekey`; keying on period end would leak each report 30–90 days early; restating `MR*` dimensions trigger a warning), and become usable one trading day after filing (`lag_days=1`). The default dimension is `ART` (as-reported trailing twelve months), the closer match to the academic annual definition (→ see **Data & universe** for the raw-price and split details).

### Momentum: `mom_12_1`, `mom_6_1` — and why skip the last month

**Definition.** $\text{mom\_12\_1}_{i,t} = P_{i,t-21}/P_{i,t-252} - 1$ and $\text{mom\_6\_1}_{i,t} = P_{i,t-21}/P_{i,t-126} - 1$ (`_trailing_return(close, start_lag, end_lag)`; lags in trading days, adjusted closes).

**Why skip the last month.** Over about a month, returns *reverse* (see `strev_21`), largely from liquidity provision and bid–ask bounce; over 3–12 months they *continue*. Including the last month would mix a negative-sign effect into a positive-sign signal. "12-1" is the standard fix since Jegadeesh & Titman.

**Rationale.** Mainly behavioural: under-reaction to gradually diffusing news, later over-reaction.

**Status.** Highest-Sharpe single theme and GKX's most important feature group (price trends), but crash-prone: an equity momentum gauge fell 21% in the four weeks to 10 March 2025, and in July 2026 a widely tracked momentum strategy fell more than 3% in each of two consecutive weeks after positioning hit its highest level since 2002 (Hedgeweek, live reports). Hence crash control and vol scaling on this theme only (§5). **Sign:** +. **PIT:** pure `shift(k ≥ 0)` of closes; NaN for the first 252 days of history.

### `mom_change`

$\text{mom\_12\_1}_t - \text{mom\_12\_1}_{t-63}$: is momentum *accelerating* (fresh winners) or stale? Its standalone sign is ambiguous; it is an interaction variable for the trees. Plan Tier 1, ML only.

### `industry_momentum`

**Definition.** $\sum_{j\in g(i)} m_{j,t}\,\text{mom\_6\_1}_{j,t} \big/ \sum_{j\in g(i)} m_{j,t}$, where $g(i)$ is stock $i$'s Fama–French 48 industry (`qts/data/loaders.py::sic_to_ff48`, review item 3) and $m$ market cap (`features.py::cap_weighted_group_mean`), broadcast to every member.

**Rationale.** Industries trend (slow diffusion of industry news); Moskowitz & Grinblatt found industry momentum explains much of stock momentum. **Sign:** +. **PIT:** uses lagged 6-1 momentum (audit-verified); the cap weights are those of names **eligible** at $t$ (`mcap.where(valid)`), and with a dated classification (`MarketData.industry_history`) each date uses the industry labels in force on that date (`features.py::industry_segments`, test `test_industry_history_is_used_point_in_time`).

### Quality: `quality_gpa`

**Definition.** Gross profit (revenue − cost of goods sold) / total assets, latest filing.

**Rationale.** Novy-Marx's "other side of value": gross profit is the cleanest accounting measure of productivity, high on the income statement and so less distorted by accounting choices. Profitable firms out-earn unprofitable ones despite being pricier, and combining profitability with value improves both. The behavioural reading (under-appreciated persistence of profits) is more natural than a risk one, since profitable firms are if anything safer.

**Status.** One of the five surviving theme families; industry-demeaned (margins differ structurally). **Sign:** +. **PIT:** as value; with the default `ART` dimension the numerator is trailing-twelve-month gross profit.

### Investment: `investment_ag`

**Definition.** $-(A_{i,t}/A_{i,t-1\,\text{yr}} - 1)$: *negative* asset growth (sign flipped at build time, so high = low investment).

**Rationale.** Fast balance-sheet growth predicts underperformance (Cooper, Gulen & Schill). Risk (q-theory): firms invest more when their cost of capital is low. Behavioural: empire-building and extrapolation.

**Status.** Surviving theme; industry-demeaned. **Sign:** +. **PIT:** the audit found the loader took prior-year assets from a possibly *later* restatement (`keep="last"`); fixed with a `merge_asof` on filing date.

### Low risk: `beta_252`, `idio_vol_252`, `max_ret_21`

**Definitions.** `features.py::rolling_market_beta` regresses daily returns $r_i$ on the equal-weighted return $r_m$ of names eligible at each date, over a trailing 252-day window (≥ 80% of days):

$$
\beta_{i,t} = \frac{\widehat{\text{Cov}}(r_i, r_m)}{\widehat{\text{Var}}(r_m)}, \qquad \text{idio\_vol}_{i,t} = \sqrt{\max\!\left(0,\ \widehat{\text{Var}}(r_i) - \beta_{i,t}^2\,\widehat{\text{Var}}(r_m)\right)}
$$

where moments are rolling population moments over $t-251,\dots,t$ and idio vol is in **daily** units. $\text{max\_ret\_21}_{i,t} = \max_{0\le k<21} r_{i,t-k}$ (≥ 15 days).

**Rationale.** *Betting against beta*: leverage-constrained investors buy high-beta stocks instead of borrowing, bidding them up (Frazzini & Pedersen). *Low-vol anomaly*: high idiosyncratic volatility predicts low returns (Ang, Hodrick, Xing & Zhang). *Lottery demand*: extreme recent daily gains predict underperformance (Bali, Cakici & Whitelaw). All behavioural/constraint stories, all **negative** sign.

**Status.** A surviving family that has been weak lately: Low Volatility lagged in Q4 2025 (−1.25%), weakest in the US (Russell). The three are highly correlated, hence one theme with three members rather than three votes. **PIT:** trailing windows; the market proxy uses names eligible *at each date* (audit Finding 1 fix).

<callout icon="⚠️" color="yellow_bg">
`beta_252` and `idio_vol_252` are also risk exposures in the label residualisation (`targets.DEFAULT_STYLE_EXPOSURES`; `beta`/`residual_vol` in `qts.core.roles.RISK_ONLY_STYLES`). The ML target therefore has the low-risk theme's linear effect projected out while the composite still bets on it. This is intentional (design decision D1, §4).
</callout>

### Liquidity and size: `turnover_63`, `amihud_63`, `log_dollar_vol_63`, `log_mcap`

With dollar volume $\text{DV} = P\cdot\text{Volume}$ and market cap $M$, averaged over 63 trading days (≥ 40 valid): $\text{turnover\_63} = \overline{\text{DV}/M}$, $\text{log\_dollar\_vol\_63} = \ln\overline{\text{DV}}$, $\text{log\_mcap} = \ln M_t$.

**Amihud illiquidity** is the average absolute return per dollar traded:

$$
\text{amihud\_63}_{i,t} = \ln\!\left(10^6\cdot\frac{1}{63}\sum_{k=0}^{62}\frac{\lvert r_{i,t-k}\rvert}{\text{DV}_{i,t-k}} + 10^{-12}\right)
$$

where zero-volume days are excluded and $10^6$ rescales to "return per million dollars traded". Read it as "how far does the price move when \$1m changes hands?" — high means illiquid.

**Rationale and status.** Illiquid stocks should pay a premium for their trading costs (Amihud 2002), but in our universe (top ~1,000, bottom NYSE size quintile excluded, ADV ≥ \$2m) that premium is small, and size is among JKP's weakest themes. Their real job is **conditioning**: GKX found liquidity the second most important group after price trends, largely through interactions. ML only; `log_mcap`, `log_dollar_vol_63` and `turnover_63` are also label exposures. **PIT:** trailing windows; market cap uses PIT shares.

### Shorting: `short_interest`, `borrow_fee`

**Definitions.** Shares short / shares outstanding (FINRA bi-monthly reports); annualised stock-loan fee (fraction).

**Rationale and status.** Short sellers are on average informed and pay to express a view. Engelberg et al. found **loan fees** give the highest monthly long–short return (4.01%) and monthly Sharpe (0.66) among 103 anomaly strategies, persistent through the sample, with 42% unexplained by other anomalies — the plan's "single most persistent non-text predictor". Options-implied vol spread and skew predict largely *because* implied vols embed borrow fees (Muravyev, Pearson & Pollet), so they are not added. **Sign:** − for both.

**PIT.** FINRA data become visible 8 NYSE sessions after the settlement date (`loaders.py::to_short_interest_ratio`, review item 2; exchange holidays are skipped via `qts.data.calendar.add_sessions`, which replaced the holiday-blind `BDay` after second-pass finding 12). Borrow-fee history is not freely available: IBKR indicative rates are to be logged daily and spliced in after ≥ 12 months.

### Short-term reversal: `strev_21`

**Definition.** $\text{strev\_21}_{i,t} = -\big(R^{(21)}_{i,t} - \text{mean}_{j\in g(i)} R^{(21)}_{j,t}\big)$ with $R^{(21)}_{i,t} = P_{i,t}/P_{i,t-21}-1$, industry mean over eligible names. Last month's relative losers score high.

**Rationale.** Liquidity providers who absorb order-flow imbalances are paid through subsequent price recovery — a strong *gross* anomaly.

**Why not standalone.** It reshuffles almost completely each month, so costs eat it; the build plan lists standalone 1–20-day signals as "execution-timing overlays only" (Dimensional uses reversal to delay selling recent losers; Robeco's horizon result). Here it is (1) an **ML input**, so trees can learn interactions such as "momentum winner with a bad last month", and (2) the alpha-layer **proxy of the risk-only style** `short_term_reversal` (`roles.RISK_ONLY_FEATURE_PROXIES`), so during the risk model's warm-up the 20-day label is not credited with mechanical reversal. It is in no composite theme and is **not** used for trade timing (an old code comment called it a "timing feature"; finding #11). A dedicated execution-timing use is on the Roadmap.

**PIT.** Uses today's close $P_t$. With the default execution lag of one day the signal from close $t$ is traded at close $t+1$, so the close it reads is known well before the order is due (→ see **Backtesting**). The lag investigation found `strev_21` the only feature whose IC changes sign within 1–2 days of lag (h = 20 IC × 1000: +0.17 at lag 0, −2.41 at lag 1, −9.84 at lag 5; `docs/notes/lag_investigation_2026-11-05.md` §4), so rerun the lag-0 vs lag-1 comparison with `use_ml=True` before trusting a fast ML sleeve.

### Why this feature set

The plan specifies ~60–100 Tier-1 features; the code has 18, a deliberate thin slice (review item 11): every extra feature is another ledger trial and leakage surface, and each must pass the incremental-IC gate (§9). Excluded on evidence (plan §2.2): post-earnings drift (t = 1.43 ex-microcaps, 2001–2024), retail order imbalance (no longer predictive 2016–2021), and anything whose profits concentrate in microcaps (Avramov, Cheng & Metzker).

### In the code

`FeatureBuilder.build(md)` → dict of panels; fundamental features appear only if their `MarketData` panel exists. To add one: trailing operations only, append to `FEATURE_ORDER`, optionally add to `composite.THEMES` with a sign, run the hurdle protocol.

### Pitfalls & checks

- **Look-ahead:** `tests/test_alpha.py::test_features_point_in_time` recomputes features on truncated data and requires identical values; `qts/validation/leakage.py::pit_shift_test` catches `shift(-k)`, centred windows and full-sample statistics.
- **Universe look-ahead:** the audit's top finding — ranks included names that were *ever* in the universe (future entrants earned +110% annualised in the 63 days before entry, synthetic data). Fixed via `MarketData.eligibility()`.
- **Restated fundamentals:** as-reported dimensions keyed on filing date only.

### References
- [Alpha Architect on Jensen, Kelly & Pedersen](https://alphaarchitect.com/is-there-a-replication-crisis-in-finance/)
- [Gu, Kelly & Xiu (2020), Empirical Asset Pricing via Machine Learning](https://www.nber.org/papers/w25398.pdf)
- [Engelberg et al. (2025)](https://ideas.repec.org/a/inm/ormnsc/v71y2025i7p5529-5551.html); [Muravyev, Pearson & Pollet (2025)](https://ideas.repec.org/a/eee/jfinec/v172y2025ics0304405x25001618.html)
- [Dimensional, Q&A on short-run reversals](https://www.dimensional.com/us-en/insights/q-and-a-on-short-run-reversals-with-mamdouh-medhat-and-robert-novy-marx)
- [Hedgeweek (March 2025)](https://www.hedgeweek.com/hedge-fund-momentum-trades-hit-hard-by-market-volatility/); [Hedgeweek (July 2026)](https://www.hedgeweek.com/quant-funds-suffer-steepest-momentum-drawdown-since-2023-amid-factor-rotation/)
- [Russell Investments, Factor report Q4 2025](https://russellinvestments.com/content/dam/ri/files/us/en/institutions/articles/Equity-Factor-Investing-Report_2025-Q4.pdf)
- [UCLA Anderson Review on PEAD](https://anderson-review.ucla.edu/is-post-earnings-announcement-drift-a-thing-again/); [Avramov, Cheng & Metzker](https://papers.ssrn.com/abstract=3450322)
- [Jegadeesh & Titman (1993), Returns to Buying Winners and Selling Losers](https://doi.org/10.1111/j.1540-6261.1993.tb04702.x)
- [Novy-Marx (2013), The other side of value](https://doi.org/10.1016/j.jfineco.2013.01.003)
- [Ang, Hodrick, Xing & Zhang (2006), The Cross-Section of Volatility and Expected Returns](https://doi.org/10.1111/j.1540-6261.2006.00836.x)
- [Frazzini & Pedersen (2014), Betting against beta](https://doi.org/10.1016/j.jfineco.2013.10.005)
- [Amihud (2002), Illiquidity and stock returns](https://doi.org/10.1016/S1386-4181(01)00024-6)

## 3. Cross-sectional processing

### What it is

Raw features live on wildly different scales (B/P around 0.5, Amihud around $10^{-9}$, momentum from −90% to +900%) and have fat tails. **Cross-sectional processing** converts each feature, *separately on each date*, into a well-behaved score: same distribution every day, comparable across features, robust to a single bad data point. `qts/alpha/features.py::FeatureBuilder.process` applies four steps, in this order, to each feature:

1. restrict to names **eligible** at the date;
2. **winsorise** at the 1st/99th percentile;
3. **industry-demean** (only `value_bp`, `value_ep`, `quality_gpa`, `investment_ag`);
4. **rank-gauss** transform; then fill missing values with 0 (the median).

The output is a long frame indexed by `(date, ticker)` containing *only* eligible rows.

### Why we need it here

Without per-date standardisation, a market-wide shift (every stock's momentum rose in a rally) would look like a signal and one mis-recorded book value could dominate a regression. Per-date ("era-wise") processing also guarantees train/live parity: live scoring sees one date, so any statistic pooled across dates in training would be a mismatch — the kind that caused inconsistent live results for a top-10 Numerai participant (build plan §2.2).

### Eligibility masking

**What.** `MarketData.eligibility()` returns a boolean date × ticker panel: in the universe at $t$ (top 1,000 by market cap, price ≥ \$5, ADV ≥ \$2m, outside the bottom size quintile — `UniverseConfig`) **and** has a close at $t$. Before winsorising, every feature is set to NaN outside this mask; after processing, ineligible rows are *dropped*, never zero-filled.

**Why.** The fix for audit Findings 1 (CRITICAL) and 9: previously, out-of-universe and dead names were ranked and scored alongside live ones. Now "the 90th percentile" means "of what we could actually trade that day".

**Time series on full history, cross-sections on eligible names.** Two different operations need two different rules:
- *Rolling, per-stock statistics* (63-day turnover, Amihud, dollar volume, 252-day beta, momentum) use the stock's **whole price and volume history**, including the days before it entered the universe. That history was knowable at $t$; it is just not part of the cross-section yet. Since the second-pass audit `apply_mask` attaches the universe as `md.eligible` and **masks no panel** (alpha/data finding 6): it used to blank volume, market cap and shares outside the universe, so every new entrant had NaN liquidity features for about 40 days (zero-filled to the median: 2,745 eligible rows) and 8,126 eligible `turnover_63` values differed from their true trailing value. Tests: `tests/test_apply_mask_history.py`.
- *Cross-sectional statistics* (winsorisation quantiles, industry means, ranks, z-scores, the equal-weighted market proxy for beta, industry-momentum cap weights) use **only names eligible at $t$**. A name that joins the universe next month must not shift today's ranks.

Both rules are point-in-time: the first uses past data of one stock, the second today's membership.

### Winsorisation

**What.** Clip each value to the date's 1st and 99th percentiles:

$$
\tilde{x}_{i,t} = \min\!\left(\max\!\left(x_{i,t},\ q_{0.01,t}\right),\ q_{0.99,t}\right)
$$

where $q_{p,t}$ is the $p$-quantile of the feature across eligible names at $t$ (`AlphaConfig.winsor_pct = 0.01`). Dates with fewer than 10 valid values are left unclipped.

**Why.** Real data has errors (an unadjusted split, a book value off by 1,000×) and genuine monsters (a biotech up 900%). Because a rank transform follows, winsorising matters *only* through the industry mean: it stops one extreme value shifting its whole industry's mean.

**In the code.** `features.py::_winsorize_rows` delegates to `qts/core/stats.py::winsorize_rows`, the single implementation (vectorised, bit-for-bit numpy's linear-interpolation percentile; rows with fewer than `WINSOR_MIN_OBS` = 10 finite values unclipped).

### Industry demeaning

**What.** For the features in `AlphaConfig.industry_demean` (default `["value_bp", "value_ep", "quality_gpa", "investment_ag"]`), subtract the date's industry mean:

$$
x^{\text{dm}}_{i,t} = \tilde{x}_{i,t} - \frac{1}{\lvert g(i)\rvert_t}\sum_{j \in g(i)} \tilde{x}_{j,t}
$$

where $g(i)$ is stock $i$'s Fama–French 48 industry and the mean is over eligible members at $t$.

**Why for these four.** Accounting ratios have industry structure unrelated to mispricing: banks have high B/P because their assets sit near book; software firms low B/P because code and people are off-balance-sheet; utilities low GP/A because they are capital-intensive. Within-industry comparison asks "is this bank cheap *for a bank*?" (AQR's within-industry value; JKP's consistent construction). Price-based features are *not* demeaned: industry momentum is itself a signal (a separate feature), and industry volatility differences are a risk-model matter.

**In the code.** `qts/core/crosssection.py::demean_by_group`, applied per block of constant industry classification (`features.py::industry_segments`) when the classification is dated.

### Rank-gauss transform

**What.** Replace each value by its cross-sectional rank, then map the rank through the inverse normal CDF so the result looks like a standard normal sample:

$$
u_{i,t} = \frac{\text{rank}_t(x_{i,t}) - 0.5}{n_t}, \qquad z_{i,t} = \Phi^{-1}\!\left(\text{clip}(u_{i,t},\ 10^{-4},\ 1-10^{-4})\right)
$$

where:
- $\text{rank}_t(x_{i,t}) \in \{1,\dots,n_t\}$ is the rank among the $n_t$ non-missing eligible values at $t$ (ties get the average rank);
- $u_{i,t} \in (0,1)$ is the "mid-rank" percentile — the $-0.5$ keeps the extremes away from 0 and 1;
- $\Phi^{-1}$ is the standard normal quantile function (`scipy.stats.norm.ppf`);
- the clip bounds $z$ at $\pm 3.72$; with $n_t = 1{,}000$ the actual extremes are $\Phi^{-1}(0.9995) = \pm 3.29$.

Missing values (e.g. a stock with no book value) are then filled with 0, the cross-sectional median.

**Why ranks.** Three properties matter:
1. *Robust to outliers.* The largest value becomes "rank $n$" no matter whether it is 2× or 2,000× the next one.
2. *Scale-free.* Features measured in dollars, ratios or logs all end up identically distributed, so averaging them (the composite) or feeding them to a neural net needs no further scaling.
3. *Stationary.* The distribution is the same every date, so a model trained in 2010 sees inputs on the same scale in 2026.

**Why Gaussianise rather than keep uniform ranks.** Uniform ranks treat the 50th→60th percentile gap like the 90th→100th; the Gaussian mapping stretches the tails, where most anomaly signal lives, and suits ridge and the MLP. **The cost:** magnitude is lost ("very cheap" = "absurdly cheap") — harmless for trees, usually a good trade for linear models.

**In the code.** `qts/core/stats.py::rank_gauss` (clip `RANK_GAUSS_CLIP` = $10^{-4}$). Its docstring now states the bounds correctly: $\pm\Phi^{-1}(1 - 0.5/n)$, e.g. $\pm 2.58$ at 100 names and $\pm 3.29$ at 1,000, and never beyond $\pm 3.72$ (finding #6).

### Why this choice

A plain z-score is outlier-sensitive (one extreme value compresses everyone else toward zero); uniform ranks under-weight the tails; full-sample or expanding standardisation either uses the future or breaks train/live parity. Per-date rank-gauss is PIT, parity-safe and robust.

### Pitfalls & checks

- **Zero-filling ineligible names** was the original bug; `tests/test_alpha.py::test_cross_section_uses_only_eligible_names` guards it.
- **Coverage.** A feature missing for 40% of names puts 40% at exactly 0; monitor coverage per date.
- `tests/test_alpha.py::test_processed_frame_standardised` checks there are no NaNs and that, after warm-up, each date's processed features have |mean| < 0.05 and SD within 0.12 of 1 (`borrow_fee` excluded because the synthetic generator floors it, creating massive ties).

### Worked example

Five eligible stocks, one feature, values $[0.2,\ 0.5,\ 0.5,\ 1.1,\ 40.0]$ (the last is a data error). Ranks with ties averaged: $[1, 2.5, 2.5, 4, 5]$. Mid-rank percentiles $(r-0.5)/5$: $[0.1, 0.4, 0.4, 0.7, 0.9]$. $\Phi^{-1}$: $[-1.28, -0.25, -0.25, 0.52, 1.28]$. The data error gets the same score it would have had at 1.2 — that is the robustness property.

### References
- [Numerai 2025 write-up (era-wise processing, process parity)](https://kenfus.github.io/machine%20learning/quant/competition/2025/06/20/numerai-2025.html)
- [AQR, Are value stocks cheap for a fundamental reason?](https://www.aqr.com/insights/perspectives/are-value-stocks-cheap-for-a-fundamental-reason)

## 4. Targets (labels)

### What it is

The **target** (or *label*) is what the ML models are trained to predict and what every IC in §9 is measured against: the 20-day forward return of each stock, cleaned of risk-factor effects, standardised per date. `qts/alpha/targets.py::residualize_forward_returns` builds it.

<callout icon="⚠️" color="yellow_bg">
Labels use the future by definition. They are **training targets only**: nothing in `targets.py` may ever be used as a feature, and a label at date $s$ may be used for training only after its forward window has fully elapsed and its row is decided (`label_realisation`, §7). This rule is the single most important leakage guard in the alpha layer.
</callout>

### Why we need it here

Section 1 explained *why* a residual 20-day target; this section is *how*. A careful target definition is half the model: train on the wrong label and the best learner in the world learns the wrong thing (the audit's Finding 2 is a vivid example below).

### How it works

**Step 1 — forward return, aligned with the trade.** A signal formed from data at the close of day $s$ cannot be traded at that close: closing-auction orders are due by 15:50 ET, before the official close prints. With the execution lag $L$ = `ExecutionConfig.signal_lag_days` (default **1**) it is traded at the close of $s+L$ and earns from there. The label must measure what that trade earns:

$$
R^{(h)}_{i,s} = \frac{P_{i,s+L+h}}{P_{i,s+L}} - 1
$$

where $h$ = `AlphaConfig.horizon_days` = 20 trading days, $L$ the execution lag, and $P$ the adjusted close; the label is stored at the **signal date** $s$ (`targets.py::forward_returns(..., label_lag_days=L)`; `AlphaModel(signal_lag_days=L)` passes it). With $L = 0$ this is the classic $P_{t+h}/P_t - 1$. The label of $s$ is complete only at the close of $s + L + h$, so every consumer must wait $h + L$ days, not $h$ (§7, §8).

<callout icon="💡" color="blue_bg">
Why this matters. Before the fix the label was $P_{s+h}/P_s - 1$ while the book traded at $s+1$: the label contained day $s+1$'s return, which the book never earns. It was not a leak (the purge covered it), but every IC, the Grinold IC and the hurdle statistics described a slightly better signal than the one traded. On synthetic seed 0 the raw IC was 0.0566 for the label the model was trained on and 0.0537 for the return the traded book could earn (second-pass audit, finding 4). On real data, where reversal and news decay within days, the gap would be larger.
</callout>

**Step 2 — halts and delistings, decided causally.** If the stock has no close at the window end $e = s+L+h$, the return runs to its last close $P_{\text{last}}$ before $e$, and the cell is either a **delisting** (the stock is gone: apply a haircut) or a **temporary halt** (it will trade again: no haircut):

$$
R^{(h)}_{i,s} = \frac{P_{i,\text{last}}}{P_{i,s+L}}(1 + d) - 1 \quad\text{(delisting)}, \qquad d = \texttt{CostConfig.delisting\_haircut} = -0.30
$$

The decision never uses prices after the date at which it is made:
- if `md.delisting_dates` (vendor delisting dates) covers the name: delisted iff its last trading date is $\le$ date($e$), decided at $e$;
- otherwise the **causal gap rule** with $G$ = `AlphaConfig.delisting_gap_days` = 63: a *halt* if a close reappears within $(\text{lp}, \text{lp} + G]$ trading days of the last close lp (decided on the day it reappears), a *delisting* if none does (decided at $\max(e, \text{lp} + G)$). Until one of the two has happened the cell is **unresolved** and NaN.

`targets.py::label_realisation(md, h, label_lag_days=L, delisting_gap_days=G)` returns, per signal date $s$, the first date on which **every** eligible name's label in row $s$ is decided (at least $s + L + h$; later while a halt is unresolved). Because the next step residualises across the whole row, one unresolved cell holds back the row. The trainer and the trailing IC use row $s$ at date $t$ only if `label_realisation[s] <= t`.

<callout icon="⚠️" color="yellow_bg">
The look-ahead this replaced (second-pass audit, finding 1, HIGH). The first audit's delisting fix decided "delisted" with `c.bfill()` over the whole panel: "no later close *anywhere in the data*". A stock halted for 60 days that later resumed was therefore a halt in a full-sample fit but a −30% delisting in a live fit at a date inside the halt. Labels realised by $t$, and through them the ML fit, the out-of-sample IC, the Grinold IC and `expected_return` at $t$, depended on prices after $t$: in the audit's test 4,228 residualised label cells differed by up to 3.7 z and the IC estimate moved from 0.01248 to 0.01237. The synthetic market has no halts, so no earlier test could see it. Now `tests/test_label_halts.py::test_halt_labels_ic_and_expected_return_are_pit` injects resuming halts and requires identical labels, IC and expected returns whether the data end at $t$ or later. The cost: without a vendor delisting table a real delisting delays its row by up to $G$ = 63 days. Supply `delisting_dates` for real data.
</callout>

The −30% haircut is the conservative CRSP-style assumption for performance delistings with a missing delisting return (Shumway 1997); it is wrong for cash takeovers, so use actual delisting returns when available. Before the first audit these labels were simply NaN ("0 of 630 pre-delisting labels survive", Finding 5), so training and scoring omitted exactly the observations where shorts pay and longs blow up. The haircut has one source, `CostConfig.delisting_haircut`, shared with the backtest's accounting (the old `AlphaConfig.delisting_return` was removed, finding #42); pass `None`/`0.0` to disable it in the label.

**Step 3 — restrict and winsorise.** Keep only names eligible at $t$; clip at the 1st/99th percentile per date.

**Step 4 — residualise.** On each date $t$, run a cross-sectional least-squares regression of the forward return on risk exposures known at $t$:

$$
R^{(h)}_{i,t} = a_t + \sum_{g=2}^{G} \delta_{g,t}\,D_{i,g} + \sum_{k=1}^{K} b_{k,t}\,X_{i,k,t} + \epsilon_{i,t}
$$

where:
- $a_t$ is an intercept (the cross-sectional average return that day);
- $D_{i,g}$ is 1 if stock $i$ is in industry $g$ (one industry dropped to avoid collinearity with the intercept), so $\delta_{g,t}$ is industry $g$'s average excess return over the window;
- $X_{i,k,t}$ are $K$ **risk-only** style exposures, each z-scored per date over eligible names;
- $b_{k,t}$ is the return to style $k$ over the window;
- $\epsilon_{i,t}$ — the residual — is the target before z-scoring.

A stock that rose 8% while its industry rose 7% and its beta explained another 0.5% has a residual of about 0.5%.

The exposures come from one of two sources, and both are defined by the single roles module `qts/core/roles.py`:
- **inside `ResearchPipeline`**, the fitted risk model's exposures (`ResearchPipeline._exposures_fn` → `risk.exposures_as_of`) restricted to market, industries and `RISK_ONLY_STYLES = ("beta", "size", "residual_vol", "liquidity", "short_term_reversal")`;
- **otherwise, and during the risk model's 252-day warm-up**, the internal set `DEFAULT_STYLE_EXPOSURES = roles.RISK_ONLY_FEATURES = ("beta_252", "log_mcap", "idio_vol_252", "log_dollar_vol_63", "turnover_63", "strev_21")` plus industry dummies (each the alpha-layer proxy of one risk-only style, `roles.RISK_ONLY_FEATURE_PROXIES`).

Both sets cover the same concepts, so the label definition is consistent across the warm-up boundary. The exposures are as of $s$, the signal date (they are known then; the label's returns are future by definition). Industry dummies follow the dated classification when `md.industry_history` exists. An optional `weights` panel (e.g. $\sqrt{\text{market cap}}$) turns the regression into weighted least squares; the default is unweighted.

**Design decision D1: residualise on exactly what the optimizer neutralises.** The same constant, `RISK_ONLY_STYLES`, is the optimizer's default set of banded styles (→ see **Portfolio construction › Style bands on risk-only styles**). The reasoning: a premium the book is forced to hedge is a premium it cannot earn, so predicting it wastes model capacity and leaks a hedged exposure into the score. Keeping one constant means the label and the hedge cannot drift apart (test `test_label_and_optimizer_sets_come_from_roles`).

**Step 5 — z-score per date.**

$$
y_{i,t} = \frac{\epsilon_{i,t} - \bar{\epsilon}_t}{\text{sd}_t(\epsilon)}
$$

Dates with fewer than 20 valid forward returns, the last $h + L$ dates of the panel (whose window has not elapsed), and cells whose halt/delisting question is unresolved are NaN.

### Why we never residualise on alpha themes

Audit Finding 2 (CRITICAL) is an easy mistake to repeat. The original default exposures were `("beta_252", "log_mcap", "mom_12_1", "value_bp")`: momentum and value treated as risks. Regression residuals are *uncorrelated with the regressors by construction*, so a value-residualised target contains no value premium and a model trained on it learns — correctly — that value predicts nothing. On synthetic data with a planted value effect:

| Target | Rank IC of log(B/P) |
|---|---|
| Residualised on beta, size, momentum, value (old default) | −0.001 |
| Residualised on risk-only styles (current) | +0.019 |

Worse, the pipeline fell back to that default *silently* whenever the risk-model hook raised. The fix (`docs/audit/RESOLUTION.md`): risk-only defaults, assertions in `qts/core/roles.py` that the risk-only and alpha-theme sets are disjoint (and in `default_exposures` that no theme feature is used), and a hook that returns `None` only during warm-up and raises otherwise.

The general rule: **residualise the label on what you want the model to ignore (risks you will hedge), never on what you want it to learn.** Momentum, value, quality, investment and short interest are alpha themes; beta, size, residual volatility, liquidity and short-term reversal are treated as risks.

<callout icon="🤔" color="purple_bg">
Why is the low-risk premium removed from the ML target on purpose? The composite bets on low beta and low idiosyncratic vol, but beta and residual vol are risk-only styles, so the ML label has their linear effect projected out. This is design decision D1, not an oversight: the optimizer holds the book beta-neutral (±0.05 sd) and residual-vol-banded (±0.10 sd), so a forecast that loads on the low-risk premium could barely be expressed in positions. The ML sleeve therefore learns what the book can trade; only the composite carries the linear low-risk tilt, and the optimizer's bands limit how much of it survives into positions. If you ever un-band `residual_vol` in the optimizer, move it from `RISK_ONLY_STYLES` to `ALPHA_THEME_STYLES` too, and the label follows automatically.
</callout>

### Why this choice

- *Residualised vs raw:* ≈2.5× Sharpe in DLSA (gross, pseudo-OOS), §1.
- *Exposures at $t$:* keeps the regression's right-hand side PIT (the returns are future by definition — it is a label).
- *Per-date z-score:* equal weight per date, as in MASTER's labels and Numerai's eras.
- *Loss:* the plan's primary is pointwise MSE on this target; the rank-correlation head is not implemented (Roadmap).

### In the code

- `targets.py::forward_returns(md, horizon, delisting_return=-0.30, label_lag_days=0, delisting_gap_days=63)` (defaults read from `CostConfig().delisting_haircut` and `AlphaConfig().delisting_gap_days`).
- `targets.py::label_realisation(md, horizon, label_lag_days=0, delisting_gap_days=63, valid=None)` → date each label row is decided.
- `targets.py::default_exposures(md, features)` — the internal risk-only set, reusing raw feature panels when given.
- `targets.py::residualize_forward_returns(md, horizon, exposures_fn=None, features=None, winsor_pct=0.01, weights=None, delisting_return=-0.30, only_dates=None, label_lag_days=0, delisting_gap_days=63)` → wide panel. `only_dates` computes only the listed rows (identical values, used for cheap IC-decay labels).
- `AlphaModel(cfg, ..., delisting_return=None, signal_lag_days=0).fit(md, risk_exposures_fn=None)`; `ResearchPipeline` passes `cfg.cost.delisting_haircut` and `cfg.execution.signal_lag_days`, and refits when the lag changes.
- Tuning knobs: `AlphaConfig.horizon_days` (changes everything downstream, including the purge), `CostConfig.delisting_haircut`, `AlphaConfig.delisting_gap_days`, `ExecutionConfig.signal_lag_days`, and the role sets in `qts/core/roles.py`.

### Pitfalls & checks

- **Using a label as a feature.** Enforced by the module split (features never import from `targets`) and the feature registry check (→ see **Validation**).
- **Residualising on a theme.** Guarded by the assertion in `targets.py`; extend `ALPHA_THEME_FEATURES` if you add a theme.
- **Label survivorship.** Fixed by the delisting rule; verify on real data that the count of non-NaN labels in the 20 days before each delisting is not zero.
- **Halts.** `tests/test_label_halts.py` covers the causal rule (`test_halt_is_not_a_delisting_and_rule_is_causal`, `test_causal_delisting_rule_and_vendor_dates`), the lag shift (`test_label_lag_shifts_window_by_L`) and full PIT of labels, IC and expected returns with halts (`test_halt_labels_ic_and_expected_return_are_pit`). `SyntheticSpec` can now plant halts and delistings (`test_synthetic_halts_and_delistings_flags`).
- `tests/test_alpha.py::test_residualized_target_properties` checks that every date with a complete window has a label, that labels have mean 0 and SD 1 per date, that they are rank-uncorrelated with beta and log market cap, uncorrelated with the forward market return, and that an external exposures hook is honoured.

### References
- [Guijarro-Ordonez, Pelger & Zanotti, DLSA](https://ar5iv.labs.arxiv.org/html/2106.04028)
- [MASTER (2024)](https://arxiv.org/pdf/2312.15235)
- [Shumway (1997), The Delisting Bias in CRSP Data](https://doi.org/10.1111/j.1540-6261.1997.tb03818.x)

## 5. The theme rank composite (Baseline A)

### What it is

The **composite** is a transparent, almost parameter-free forecast: group features into six economic themes, average each theme's signed members, standardise, and average the themes — a committee of six specialists, each with one vote, slightly re-weighted by recent track record (*dynamic weighting*), with momentum de-risked in dangerous markets. It is the build plan's "Baseline A" and the Phase-1 alpha.

### Why we need it here

1. It already captures most of the net alpha the literature supports (build plan §1, Phase 1).
2. It is the benchmark the ML sleeve must beat net of costs.
3. It has no fitted coefficients, so nothing to overfit beyond the choice of themes (which comes from published research — itself data-mined, a caveat to remember).

### How it works

**Themes and members** (`composite.py::THEMES`):

| Theme | Members (sign) |
|---|---|
| value | `value_bp` (+), `value_ep` (+) |
| momentum | `mom_12_1` (+), `mom_6_1` (+), `industry_momentum` (+) |
| quality | `quality_gpa` (+) |
| investment | `investment_ag` (+) |
| low_risk | `beta_252` (−), `idio_vol_252` (−), `max_ret_21` (−) |
| short_interest | `short_interest` (−), `borrow_fee` (−) |

**Step 1 — theme score.** For theme $\tau$ with available members $F_\tau$ and signs $s_f$:

$$
\Theta_{\tau,i,t} = \text{zscore}_t\!\left(\frac{1}{\lvert F_\tau\rvert}\sum_{f\in F_\tau} s_f\, z_{f,i,t}\right)
$$

where $z_{f,i,t}$ is the processed (rank-gauss) feature and $\text{zscore}_t$ standardises across stocks at date $t$ (mean 0, population SD 1).

**Step 2 — composite.** With theme weights $w_{\tau,t}$:

$$
C_{i,t} = \text{zscore}_t\!\left(\frac{\sum_\tau w_{\tau,t}\,\Theta_{\tau,i,t}}{\sum_\tau \lvert w_{\tau,t}\rvert\,\mathbb{1}[\Theta_{\tau,i,t}\ \text{available}]}\right)
$$

The denominator renormalises when a theme is missing for a stock (e.g. no short-interest data), so missing themes neither add nor subtract.

**Step 3 — weights.** Static weights are equal: `AlphaConfig.theme_weights` = 1.0 for all six. With `dynamic_weights=True` (the `AlphaModel` default):

$$
w_{\tau,t} = w^{\text{static}}_\tau \times (1 + \text{tilt}_{\tau,t}) \times \underbrace{\kappa^{\text{crash}}_t \times \kappa^{\text{vol}}_t}_{\text{momentum only}}
$$

each factor described below.

<callout icon="🤔" color="purple_bg">
Why equal weights rather than weights optimised on history? With six themes and noisy returns, optimised weights mostly fit noise, and the best-backtesting theme is the one most likely over-mined. Equal weights ("1/N") are hard to beat out of sample; the plan fixes weights ex ante "until Phase 2 evidence justifies stacking".
</callout>

### Theme portfolio returns (input to the dynamic weights)

To know which themes have "worked recently" we need a return series per theme. `composite.py::theme_portfolio_returns` builds, each day, an equal-weighted long-top-decile / short-bottom-decile portfolio on each theme score among eligible names and records its *next-day* return. The series is indexed at the realisation date $t+1$, so `.loc[:t]` contains only returns known at $t$ (audit-verified against a hand computation).

### Factor-momentum tilt

**What factor momentum is.** Stock momentum says stocks that went up keep going up. **Factor momentum** says the same about *factors*: a factor that earned positive returns recently tends to keep earning them. Gupta & Kelly found that 59 of 65 US factors have positive one-month autocorrelation, and a combined time-series factor-momentum strategy reaches a Sharpe ratio of 0.84 with a one-month lookback, robust to controls for stock momentum, industry momentum, short-term reversal and transaction costs. The build plan notes no post-publication test through 2026 was found, "so treat this as the best-supported tilt rule, not a certainty".

**How.** `composite.py::factor_momentum_tilt_series`:

$$
\text{tilt}_{\tau,t} = \text{clip}\!\left(m\cdot \frac{\sum_{k=0}^{L-1} r_{\tau,t-k}}{\hat\sigma_{\tau,t}\sqrt{L}},\ -m,\ +m\right)
$$

where:
- $r_{\tau,t}$ is the daily theme long–short return;
- $L$ = `factor_momentum_lookback_months` × 21 = 21 trading days;
- $\hat\sigma_{\tau,t}$ is the trailing 252-day (≥ 63 required) standard deviation of daily theme returns;
- the fraction is therefore a t-statistic-like quantity (the trailing sum divided by its standard error under i.i.d. returns);
- $m$ = `factor_momentum_max_tilt` = 0.25.

The multiplier $1+\text{tilt}$ ranges from 0.75 to 1.25 and saturates once the trailing month's t-stat exceeds ±1.

**Why capped at ±25% (review item 6).** It is a *tilt*, not a timing strategy. A one-month t-stat of 1 is a coin flip with a slight bias; uncapped, the composite could swing to mostly-momentum right before a crash — the crowded positioning that hurt quant funds in March 2025 and July 2026. The cap keeps the composite diversified while harvesting some autocorrelation.

### Momentum crash control

**Intuition (Daniel & Moskowitz).** Momentum crashes are not random. They cluster in *rebounds after bear markets*: after a long decline, the momentum portfolio is long low-beta defensive winners and short high-beta beaten-down losers. When the market snaps back, the losers rebound violently and the short side gets squeezed. Daniel & Moskowitz show that these crashes are partly forecastable from two state variables: a negative trailing market return (bear market) and high market volatility.

**The rule.** `composite.py::momentum_crash_scalar_series`, with $r_{m,t}$ the equal-weighted return of eligible names:

$$
\kappa^{\text{crash}}_t = \begin{cases} \text{clip}\!\left(0.5\cdot \dfrac{v^{80}_t}{v_t},\ 0.25,\ 1\right) & \text{if } R^{\text{mkt}}_{t,504} < 0 \ \text{and}\ v_t > v^{80}_t \\ 1 & \text{otherwise} \end{cases}
$$

where:
- $R^{\text{mkt}}_{t,504}$ is the compounded market return over the trailing 504 trading days (≈ 24 months; `momentum_crash_market_lookback_days`), requiring ≥ 252 days;
- $v_t$ is the trailing 21-day standard deviation of daily market returns;
- $v^{80}_t$ is the **expanding** 80th percentile of $v$ up to $t$ (≥ 126 observations), so the threshold uses only the past.

Because the state requires $v_t > v^{80}_t$, the ratio $v^{80}_t/v_t$ is below 1 and the scalar lies in $[0.25, 0.5]$: in a bear + high-vol state the momentum theme's weight is at least halved, and cut to a quarter when volatility is double its 80th percentile.

### Volatility scaling of the momentum sleeve

**What.** `composite.py::momentum_vol_scalar_series`:

$$
\kappa^{\text{vol}}_t = \text{clip}\!\left(\frac{\sigma^{\star}}{\hat\sigma^{\text{mom}}_{t}},\ 0.5,\ 1.5\right)
$$

where $\sigma^\star$ = `momentum_vol_target_annual` = 0.12 and $\hat\sigma^{\text{mom}}_t$ is the trailing 63-day annualised volatility of the momentum theme's daily long–short return.

**Why only momentum.** Across 103 strategies, real-time volatility-managed versions generally earned *lower* Sharpe and certainty-equivalent returns than unmanaged ones, with only 8 consistently better (Cederburg et al.). Momentum is the documented exception: its volatility is forecastable and its crashes happen in high-vol states, so scaling down when its own volatility is high improves its risk-adjusted return. Applying vol-scaling everywhere would, on the evidence, hurt.

The clip to $[0.5, 1.5]$ means this is a bounded *tilt* toward 12% rather than true vol targeting (review item 8 describes it as "vol-targeted to 12%"; the code deliberately limits how far it can move).

### In the code

- `composite.py::ThemeComposite(cfg).score_all(processed, md, dynamic=True, theme_returns=..., return_components=True)` → composite panel and per-theme panels.
- `ThemeComposite.score(processed, date, ...)` — single-date version used for live scoring; slices everything at `date`.
- `models.py::RankCompositeModel` wraps the static composite in the common model interface.
- Config (`AlphaConfig`): `theme_weights` (all 1.0), `dynamic_weights` (True), `factor_momentum_lookback_months` (1), `factor_momentum_max_tilt` (0.25), `momentum_vol_target_annual` (0.12), `momentum_vol_scalar_bounds` ((0.5, 1.5)), `momentum_vol_window` (63), `momentum_crash_market_lookback_days` (504).
- To disable a theme, set its weight to 0. To add one, add an entry to `THEMES` and a weight in `theme_weights`.

### Pitfalls & checks

- **Look-ahead in dynamic weights.** Trailing or expanding statistics only; theme returns indexed at realisation. The audit verified `as_of(t)` is bit-identical when refit on data truncated at $t$.
- **Theme returns are daily, equal-weighted, cost-free** decile spreads — a weighting signal, not a performance report.
- **Missing themes** disappear and the rest renormalise; check `AlphaOutput.components`.
- `tests/test_alpha.py::test_composite_and_dynamic_weights` checks the static composite's IC exceeds 0.01 on the synthetic market, the dynamic composite is z-scored per date, the tilt multipliers lie in [0.75, 1.25], the crash scalar lies in [0.25, 1], and the single-date `score()` equals the corresponding row of `score_all()`.

### References
- [Gupta & Kelly, Factor momentum everywhere (via Alpha Architect)](https://alphaarchitect.com/is-factor-momentum-really-everywhere/)
- [Cederburg et al., The performance of volatility-managed portfolios (via Alpha Architect)](https://alphaarchitect.com/the-performance-of-volatility-managed-portfolios/)
- [Daniel & Moskowitz (2016), Momentum crashes](https://doi.org/10.1016/j.jfineco.2015.12.002)

## 6. Machine-learning models

### What it is

The ML sleeve learns $\hat y = f(z_{1},\dots,z_{18})$ from one stock-date's processed features to its residual label, pooling all stocks and training dates (≈1,000 × 750+ ≈ 750k+ rows). Unlike the composite it can learn **non-linearities** ("value works only among profitable firms") and **interactions** ("momentum is weaker in illiquid names"). The default (`alpha_model.py::default_ml_factory`) is a fixed-weight ensemble of LightGBM (0.45), an MLP (0.35) and ridge (0.20).

### Why we need it here

The gross case is real: in Gu, Kelly & Xiu's US panel (OOS 1987–2016), monthly OOS $R^2$ rises from 0.16% (three-characteristic OLS) to 0.40% (three-layer net) and 0.52–0.70% for trees and nets on the top 1,000 stocks (pseudo-OOS, gross). The net case is modest — best cost-inclusive ML long–short net Sharpe 0.84 (Azevedo, Hoegner & Velikov) — so ML is Phase 2 and must beat the composite *net*.

### Ridge regression (Baseline B)

**What.** Linear regression with an L2 penalty:

$$
\hat\beta = \arg\min_\beta \sum_{n}\left(y_n - x_n^\top\beta\right)^2 + \lambda\lVert\beta\rVert_2^2 = (X^\top X + \lambda I)^{-1}X^\top y
$$

where $x_n$ is the 18-vector of processed features for row $n$, $y_n$ its label, and $\lambda \ge 0$ the penalty. The penalty shrinks coefficients toward zero; geometrically it adds $\lambda$ to every eigenvalue of $X^\top X$, which stabilises the estimate when features are correlated (the three momentum features, the three low-risk features). In Bayesian terms it is a Gaussian prior $\beta \sim N(0, \sigma^2/\lambda\, I)$: "assume effects are small unless the data insist".

**In the code.** `models.py::RidgeModel(alphas=(1, 10, 100, 1e3, 1e4, 1e5))`. Each $\lambda$ is fit on the training rows and scored by **mean per-date rank IC** on the validation block; the best is refit on train + validation. It is the diagnostic member: if LightGBM cannot beat ridge, its non-linearity is not paying.

### LightGBM: gradient-boosted decision trees from first principles

**A regression tree** partitions feature space with axis-aligned splits ("is `mom_12_1` > 0.7?") and predicts a constant in each leaf. It is grown greedily: at each step, choose the feature and threshold whose split most reduces squared error. Trees capture interactions naturally (a split on value under a split on quality is a value-among-quality effect) and are invariant to monotone transforms of features — rank-gauss inputs are harmless.

**Boosting** builds an additive model of many small trees, each correcting the previous ones:

$$
F_m(x) = F_{m-1}(x) + \nu\, f_m(x), \qquad f_m \approx \arg\min_f \sum_n \left(g_{n} - f(x_n)\right)^2,\quad g_n = -\left.\frac{\partial L(y_n, F)}{\partial F}\right|_{F=F_{m-1}(x_n)}
$$

where:
- $F_m$ is the ensemble after $m$ trees, $F_0$ a constant;
- $L$ is the loss; for squared error $g_n = y_n - F_{m-1}(x_n)$, the residual;
- $f_m$ is a tree fitted to the negative gradient — this is **functional gradient descent**: each tree is one gradient step in the space of functions;
- $\nu$ is the learning rate (step size).

With L2 leaf regularisation $\lambda_2$, a leaf's value is $w = \sum_{n\in\text{leaf}} g_n / (n_{\text{leaf}} + \lambda_2)$: a shrunk average residual.

**Parameters** (`AlphaConfig.lgbm`, a `qts.core.config.LGBMParams`; `LGBMModel(params=..., **overrides)` accepts per-field overrides, e.g. `LGBMModel(n_estimators=150)`; sklearn-API names in brackets):

| Parameter | Default | Meaning and why |
|---|---|---|
| `learning_rate` | 0.03 | Small steps → many weak trees → less overfit per tree |
| `n_estimators` | 600 | Upper bound; early stopping chooses the actual number |
| `num_leaves` | 31 | LightGBM grows leaf-wise (always splits the best leaf); caps tree complexity |
| `min_data_in_leaf` (`min_child_samples`) | 200 | Each leaf must contain ≥ 200 stock-dates; essential when $R^2 \approx 0.5\%$ — small leaves fit noise |
| `feature_fraction` (`colsample_bytree`) | 0.7 | Each tree sees a random 70% of features; decorrelates trees (like random forests) |
| `bagging_fraction` / `bagging_freq` (`subsample` / `subsample_freq`) | 0.8 / 1 | Each iteration trains on a random 80% of rows |
| `lambda_l2` (`reg_lambda`) | 1.0 | Shrinks leaf values |
| `early_stopping_rounds` | 100 | Stop if the validation metric hasn't improved for 100 trees |
| `refit_on_full` | True | Refit on train + validation with the best tree count (min 10) |
| `n_jobs` | 4 | OpenMP threads; kept below the core count on shared machines (LightGBM's barriers stall ~15× when another process competes); set to the core count on a dedicated box |

**Early stopping on rank IC, and why L2 failed.** The natural validation metric is the training loss itself, mean squared error. On this target it fails: the code comment records that validation L2 "keeps *rising* with boosting rounds… even while the ranking improves, so l2 stops at ~0 trees". The reason: with a z-scored target whose predictable share is ~0.1–0.5%, the MSE-optimal prediction is almost exactly zero. Each tree adds prediction *variance* (scale) faster than it adds correlation, so MSE gets worse even when the *ordering* of stocks improves. We only use the ordering — every downstream step re-ranks or re-z-scores — so the stopping metric is the per-date mean Spearman IC on the validation block (`models.py::make_rank_ic_metric`), "higher is better".

**A vectorised metric.** That metric runs after every boosting round, so its speed matters. `models.py::DateGroupedRankIC` lays the validation rows out once as a padded `(n_dates, max_names)` array (pads `+inf`, so they rank last and are masked out), ranks each row with `scipy.stats.rankdata(axis=1)` (average ties, identical to a per-date `groupby().rank()`), and caches the label ranks; each call then ranks only the predictions. Dates with fewer than `IC_MIN_OBS` = 20 names are skipped. Test: `test_vectorised_rank_ic_metric_matches_pandas`. Together with other changes the integration benchmark measured a composite + LightGBM fit (300 names × 1,512 days) at 101 s against 161 s for the old code (1.58×; `tests/test_alpha.py::test_benchmark_alpha_fit_composite_lgbm`, slow tier).

### MLP (neural network)

**Architecture.** `models.py::MLPModel`: inputs standardised by `StandardScaler` (fit on the training rows), then 18 → 64 → 32 → 1 with ReLU activations (scikit-learn `MLPRegressor`), Adam optimiser. Settings live in `AlphaConfig.mlp` (`MLPParams`):

| Field | Default | Meaning |
|---|---|---|
| `hidden` | (64, 32) | hidden layer sizes |
| `alpha` | 1e-3 | L2 weight penalty |
| `max_iter` | 200 | maximum epochs |
| `batch_size` | 2,048 | mini-batch size |
| `learning_rate_init` | 1e-3 | Adam step size |
| `max_train_rows` | 400,000 | random row subsample cap (memory and time) |
| `n_iter_no_change` | 10 | early-stopping patience in epochs |
| `valid_fraction` | 0.1 | fallback validation block: the last 10% of dates |
| `refit_on_full` | True | retrain on train + validation for the best epoch count |

**Early stopping, time-contiguous.** scikit-learn's built-in `early_stopping` holds out a *random* 10% of rows. On a panel with overlapping 20-day labels a random validation row's neighbours (same stock, adjacent dates, 19 of 20 label days shared) sit in the training set, so the stopping rule sees almost the same labels it trains on and stops late (audit Finding 13, handbook fix #8). The model now trains epoch by epoch with `partial_fit` on the training rows and scores, after every epoch, the mean per-date rank IC on a **time-contiguous** validation block: the walk-forward trainer's purged trailing block when it is passed (`valid_mask`), otherwise the last `valid_fraction` of dates. It stops after `n_iter_no_change` epochs without improvement and keeps the best epoch count $k^\star$; with `refit_on_full` a fresh net with the same seed is trained on train + validation for $k^\star$ epochs (mirroring LightGBM), otherwise the best-epoch weights are restored. `best_epochs_` and `valid_ic_` record the choice per seed. Test: `test_mlp_early_stopping_is_time_contiguous`.

**Plan difference.** The plan specifies a 3-layer MLP with L1 penalty, batch norm and 5 seeds (GKX recipe); the code has 2 hidden layers, L2, no batch norm, and `model_seeds` = 3.

### Seed bagging and the rank-average ensemble

**Seed bagging.** LightGBM and the MLP are each trained `AlphaConfig.model_seeds` = 3 times with different random seeds (different row/feature subsamples, different initial weights) and their predictions averaged. Averaging reduces variance; in Kim et al.'s experiments, going from 1 to 64 ensemble members raised the median Sharpe and narrowed its interquartile range.

**Rank-average ensemble.** `models.py::EnsembleModel`:

$$
\hat y^{\text{ens}}_{i,t} = \text{zscore}_t\!\left(\sum_k w_k\left(\text{pctrank}_t(\hat y^{(k)}_{i,t}) - 0.5\right)\right), \qquad w = (0.45, 0.35, 0.20)
$$

Ranking first puts members on a common scale (an MLP's raw output may be 10× wider than ridge's) and stops one member's outliers dominating. Weights are fixed *ex ante* (`AlphaConfig.ensemble_weights`; a zero or missing weight drops the member), not fitted — fitting them is another layer of overfitting the plan defers until Phase 2 evidence justifies stacking.

### Why GBDT + NN, and not transformers, foundation models, RL or LLM agents

| Approach | Evidence (from build plan §2.1–2.2) | Verdict |
|---|---|---|
| Gradient-boosted trees | Strongest baseline in the largest recent US benchmark (CatBoost; ~10,000 securities, 2001–2023; long–short Sharpe 6.79 *before costs* — a ceiling, not a target) (Rahimikia, Ni & Wang); default tabular winner in 2025 competitions (ML Contests) | Core |
| GBDT + MLP blend, seed-averaged | Top Jane Street 2021 solution: MLP + XGBoost, 3 seeds each, purged CV | Core |
| Cross-sectional transformers (MASTER) | IC 0.064 on CSI300 in its own split, but 0.0215 vs LightGBM 0.0277 in an independent re-run; Qlib IR SD across 20 seeds ±0.2–0.6 → rankings not separable | Not worth the fragility |
| Time-series foundation models | Zero-shot Chronos-large $R^2$ −1.37%, TimesFM-500M −2.80%; fine-tuning mostly made them worse | Excluded |
| Deep RL for weights | Most algorithms not robust, generalise poorly; policy instability across seeds | Rejected |
| End-to-end Sharpe-maximising nets | Trailed predict-then-optimise in all six S&P 500 configurations (Kim et al., ICAIF'25) | Rejected |
| LLM trading agents | TradingAgents Sharpe 0.43 gross → 0.22 net post-cutoff, below buy-and-hold on 4 of 5 stocks | Rejected as PMs |
| LLM-mined formulaic alphas | Best in-sample Sharpe 1.69 → OOS 0.18; DSR 0.86 < 0.95 (Gençay 2026) | At most a feature generator |

CatBoost is in the plan alongside LightGBM but not implemented (Roadmap).

### Pitfalls & checks

- **Parity.** Every model records `feature_names_`; `predict` raises if columns or order differ.
- **Synthetic data ceiling.** REVIEW_NOTES: "LightGBM was at the noise floor on synthetic data" — the synthetic market's planted signal enters the characteristics linearly (`qts/data/synthetic.py`), so there is nothing non-linear to find; do not judge the ML sleeve there.
- **Overfitting signs.** Validation IC ≫ OOS IC; best iteration at the 600 cap; gain importance (`feature_importance()`) dominated by a liquidity variable.

### References
- [Gu, Kelly & Xiu (2020)](https://www.nber.org/papers/w25398.pdf)
- [Azevedo, Hoegner & Velikov (2025)](https://www.aeaweb.org/conference/2025/program/paper/TihddefB)
- [Rahimikia, Ni & Wang (2025)](https://arxiv.org/html/2511.18578v1)
- [ML Contests, State of ML competitions 2025](https://mlcontests.com/state-of-machine-learning-competitions-2025)
- [Jane Street 2021 summary](https://www.besthub.dev/articles/kaggle-jane-street-market-prediction-competition-summary-and-model-insights-accb0dc2927c)
- [MASTER](https://arxiv.org/pdf/2312.15235); [R&D-Agent(Q)](https://arxiv.org/html/2505.15155v2); [Qlib benchmarks](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md)
- [Velay et al., DRL robustness](https://www.arxiv.org/abs/2306.10950); [Kim et al., ICAIF'25](https://arxiv.org/html/2503.13544v3)
- [Alpha Illusion](https://www.alphaxiv.org/abs/2605.16895v1); [Gençay (2026)](https://arxiv.org/pdf/2608.27734)
- [Friedman (2001), Greedy function approximation: a gradient boosting machine](https://doi.org/10.1214/aos/1013203451)

## 7. Walk-forward training

### What it is

**Walk-forward training** simulates how the model would have been built in real time: at each refit date, train only on data that would have been available, predict the next year, then move forward. `qts/alpha/trainer.py::WalkForwardTrainer.run(X, y, model_factory)` returns a panel of strictly out-of-sample (OOS) predictions.

### Why we need it here

A model evaluated on data it was trained on (or on data that *overlaps* it) looks far better than it is. With 20-day labels, overlap is subtle: the label at $s$ uses returns through $s+20$, so a training row 10 days before the test period "knows" half of the test period's returns.

### How it works

- **Expanding window:** every fold trains on all usable history.
- **Refit every** `refit_frequency_days` = 252 trading days; first fit once `train_min_days` = 756 label dates are usable. A fresh model is created at each refit (no parameter carry-over).
- **Purge/embargo:** a label at signal position $s$ covers returns over $(s+L, s+L+h]$, so the embargo is $e = h + L$ with $h$ = `embargo_days or horizon_days` = 20 and $L$ the execution lag the labels were built with (`WalkForwardTrainer(label_lag_days=L)`; default pipeline $L = 1$, so $e = 21$). A label at position $s$ may train a model predicting from position $p_0$ only if

$$
s + h + L < p_0 \quad\text{and}\quad \text{label\_realisation}(s) < p_0
$$

i.e. the label's last return day is strictly before the first prediction date (one day stricter than necessary, audit-verified), and any halt/delisting question in that row has been decided before $p_0$ (§4).

- **Trailing validation block:** the last `AlphaConfig.valid_days` = 252 usable label dates are the validation block (early stopping, ridge $\lambda$), and the same purge (including the realisation check) separates fit rows from validation rows.
- **First fit** once `train_min_days + e` positions exist (`schedule`).
- **OOS only:** dates before the first fit get NaN; the combined score falls back to the composite there.

**Timeline of the first fold** with $h = 20$, $L = 1$ (positions in trading days; with $L = 0$ every gap is 20 and the boundaries shift by one):

```text
pos:   0 ........ 482 | 483 .. 503 | 504 ........ 755 | 756 .. 776 | 777 ........ 1028
       [ fit rows     ] [ purge 21 ] [ validation      ] [ purge 21 ] [ predict (OOS)  ]
label at s uses returns over (s+1, s+21]:
  last fit label   s=482 -> ends 503  <  504 first validation date
  last valid label s=755 -> ends 776  <  777 first prediction date
next refit at 1029 trains on labels s <= 1007, predicts 1029..1280, ...
```

### Why this choice

This is the GKX protocol adopted by the plan. Annual refits keep trials few and the model stable (quarterly allowed in Phase 2 if evidence supports it); re-initialising every window avoids parameter leakage (Li, Qiao & Zheng).

### Train/live parity check

`trainer.py::train_live_parity_check(model, X_date)` hashes the live feature columns *and order* (`features.py::feature_signature`) and asserts it equals the hash recorded at fit time. Combined with the single `FeatureBuilder.process` code path for training and scoring, this is the defence against the Numerai-style preprocessing mismatch.

### Pitfalls & checks

- `tests/test_alpha.py::test_embargo_excludes_overlapping_labels` asserts, with a spy model, that the gap between last training label and first prediction date is exactly $h+1$ and that the first fit happens at `train_min_days + h` ($L = 0$); `tests/test_label_halts.py::test_trainer_embargo_grows_with_lag` checks the $h + L$ embargo.
- The validation length is `AlphaConfig.valid_days` (252); `WalkForwardTrainer(cfg, valid_days=...)` can override it per instance (finding #12 moved this and `composite_weight`, `ic_window` into the config).
- The validation block ends where the training data ends, so the most recent year is used for model selection; `refit_on_full` (LightGBM) and ridge's refit then train on it too.

### References
- [Gu, Kelly & Xiu (2020)](https://www.nber.org/papers/w25398.pdf); [Li, Qiao & Zheng](https://arxiv.org/pdf/2512.19484)

## 8. Combining and scaling into expected returns

### Composite/ML blend

`AlphaModel._combine_history` blends the two z-scored sleeves with `composite_weight` = 0.5:

$$
S_{i,t} = \text{zscore}_t\!\left(\frac{0.5\,C_{i,t} + 0.5\,M_{i,t}}{0.5\cdot\mathbb{1}[C] + 0.5\cdot\mathbb{1}[M]}\right)
$$

where $M$ is the z-scored ML ensemble; where ML has no OOS prediction yet, $S$ is the composite alone. Only eligible names are kept. Fifty-fifty is a prior, not a fit: the composite is the robust anchor, ML the incremental bet.

### Grinold's rule: alpha = IC × volatility × score

**What.** Convert a dimensionless score into an expected return the optimizer can trade off against risk and cost:

$$
\alpha_{i,t} = \text{IC}\cdot\sigma_{i}\cdot z_{i,t}
$$

where:
- $z_{i,t}$ is the score re-standardised over eligible names at $t$ (mean 0, SD 1);
- $\sigma_i$ is the stock's **residual** volatility over the **forecast horizon**;
- $\text{IC}$ is the (shrunk) information coefficient;
- $\alpha_{i,t}$ is an expected *residual* return over 20 days (fraction).

<toggle heading="h3">
Derivation from a regression of returns on scores

Suppose standardised residual returns $u_i = \epsilon_i/\sigma_i$ and scores $z_i$ are jointly roughly normal across stocks, both with mean 0 and SD 1, and correlation $\rho = \text{IC}$. The best linear predictor of $u$ given $z$ is the regression line

$$
\mathbb{E}[u_i \mid z_i] = \frac{\text{Cov}(u,z)}{\text{Var}(z)}\,z_i = \rho\, z_i = \text{IC}\cdot z_i
$$

Multiplying back by $\sigma_i$ gives $\mathbb{E}[\epsilon_i \mid z_i] = \text{IC}\cdot\sigma_i\cdot z_i$. Two consequences: a stock with a score of +1 is expected to beat its peers by IC standard deviations of its own residual risk (0.02 × 6.7% ≈ 13 bp over 20 days for a typical name); and with IC 0, every alpha is 0 — the formula "refines" scores toward zero in proportion to how little they are trusted.
</toggle>

### IC shrinkage

The IC used (design decision D5, one convention everywhere) is

$$
\text{IC}_t = \max\!\left(0,\ \texttt{ic\_shrink}\cdot \widehat{\text{IC}}_{t}\right), \qquad \widehat{\text{IC}}_t = \begin{cases}\overline{\text{IC}}^{\text{OOS}}_{t} & \text{if at least } \texttt{ic\_min\_obs} \text{ realised dates}\\ \texttt{default\_ic} & \text{otherwise}\end{cases}
$$

where:
- $\overline{\text{IC}}^{\text{OOS}}_t$ (`AlphaModel.trailing_ic`) is the mean daily rank IC of the combined score against its label over the trailing `ic_window` = 252 signal dates whose labels are **realised by $t$**: the window ends at position $t - h - L$ (label of $s$ complete at $s + h + L$) and each row must also satisfy `label_realisation[s] <= t` (halts resolved);
- `ic_min_obs` = 60 realised dates;
- `ic_shrink` = 0.5 is the fraction of the estimate **kept**;
- `default_ic` = 0.03 is a **pre-shrink** prior, shrunk the same way: $0.5 \times 0.03 = 0.015$.

`AlphaOutput.ic_estimate` is $\text{IC}_t$, the value actually used; the portfolio layer must not shrink again.

**Why 50%.** Backtest-to-live haircuts are severe (median 73% Sharpe deterioration across 215 bank alternative-beta indices, live; the plan expects live Sharpe 0.25–0.5× backtest), and a trailing IC carries selection bias (we kept signals that looked good). Halving it stops the optimizer trading aggressively on an overstated forecast. A negative trailing IC floors alpha at zero rather than flipping its sign.

<callout icon="⚠️" color="yellow_bg">
What D5 fixed (finding #4). There used to be two conventions. `AlphaModel` multiplied by `ic_shrink` (fraction kept) but used `default_ic` *unshrunk*, so before 60 IC observations existed the model used 0.03, twice the shrunk value of a measured 0.03. Meanwhile `qts/portfolio/alpha_scaling.py::shrink_ic(ic, shrink)` multiplied by `1 - shrink` (fraction removed): at 0.3 the alpha model kept 30% and `shrink_ic` kept 70%. Now `shrink_ic(ic_est, keep=0.5, floor=0.0, cap=0.1)` uses "fraction kept" too, the inverse argument is deleted, and `default_ic` is shrunk. Test: `tests/test_alpha.py::test_default_ic_is_shrunk_like_estimated_ic`.
</callout>

### Residual volatility and horizon scaling

$\sigma_i$ = `idio_vol_252` × $\sqrt{h}$ (`AlphaModel.resid_vol_h_`): the daily residual volatility from the 252-day market regression, scaled to 20 days assuming independent daily residuals ($\text{Var}$ adds over days, so SD scales with $\sqrt{h}$). Missing values take the cross-sectional median. Units matter: the optimizer's objective is in per-$H$ units (→ see **Portfolio construction**), so a daily vol here would understate every alpha by $\sqrt{20} \approx 4.5\times$.

### Signal half-life

**What.** How fast the score's cross-sectional ordering decays. `evaluation.py::signal_half_life` computes the mean rank autocorrelation $\rho(\ell)$ between $S_t$ and $S_{t-\ell}$ for lags $\ell \in \{1,2,3,5,\dots,120\}$, keeps lags with $\rho > 0.02$, fits $\rho(\ell) \approx e^{-\lambda \ell}$ by least squares through the origin on $-\ln\rho$, and returns $\ln 2/\lambda$ trading days (+inf if it does not decay).

**Per component, point-in-time.** `AlphaModel` estimates a half-life for every sleeve: `combined`, `composite`, `ml` (when fitted) and each `theme_<name>` (`component_half_lives_as_of`). Each is estimated once per `refit_frequency_days` = 252 block from the trailing `half_life_lookback_days` = 504 dates of that sleeve's scores **up to the block's first date**, and cached, so every date in a block gets the same PIT value regardless of the order of `as_of` calls. A non-finite estimate falls back to $3h$ = 60 days. `AlphaOutput.half_life_days` is the combined score's value; `AlphaOutput.components_half_life` holds the whole dict (`tests/test_alpha.py::test_default_ic_is_shrunk_like_estimated_ic` also checks that `components_half_life["combined"] == half_life_days` and that the per-sleeve values are cached). Estimating once per block instead of every day also removed a cost the code-quality audit flagged: 504 × N ranks for 13 lags recomputed on every `as_of` (`docs/audit/code_quality.md`).

**Why it matters.** In Gârleanu & Pedersen's dynamic-trading solution, a signal should be weighted by $1/(1 + \text{holding days}/\text{half-life})$: fast-decaying signals deserve less weight when costs force slow trading (GP's estimates: half-life ≈2.4 days for a 5-day-return signal, ≈206 days for a 1-year one).

**What reads it today.** The simulator forwards `half_life_days` (the combined value) to `SinglePeriodOptimizer.solve(alpha_half_life_days=...)`, which applies a single GP shrink $g$ to the whole alpha vector when `PortfolioConfig.use_gp_shrink` is on (default **off**; → see **Portfolio construction › Current wiring**). The per-sleeve `components_half_life` is carried for attribution and for multi-sleeve research (`qts/portfolio/alpha_scaling.py::garleanu_pedersen_shrink`); it is not forwarded to the optimiser. The alpha layer itself never applies GP shrinkage, so there is no double counting. `qts/execution/router.py` can use half-life for urgency.

### In the code

`AlphaModel(cfg, composite_weight=None, ml_factory=None, use_ml=True, dynamic_weights=None, ic_window=None, delisting_return=None, signal_lag_days=0)`; `None` means "take it from `AlphaConfig`" (`composite_weight` 0.5, `dynamic_weights` True, `ic_window` 252; the delisting haircut from `CostConfig`). `.fit(md, risk_exposures_fn)`; `.as_of(date)` → `AlphaOutput`; `.ic_used(date)`, `.trailing_ic(date)`, `.half_life_as_of(date)`, `.component_half_lives_as_of(date)`. Config: `ic_shrink` (0.5), `default_ic` (0.03), `ic_window` (252), `ic_min_obs` (60), `horizon_days` (20), `half_life_lookback_days` (504).

### Pitfalls & checks

- **Unit errors:** alpha, $\sigma$ and the optimizer's risk term must share the 20-day horizon (→ see **Portfolio construction**).
- **IC look-ahead:** `trailing_ic` stops at position $t-h-L$ and drops rows not yet decided, so only realised labels count; the second-pass audit verified `ic_estimate` and `half_life` match a refit on truncated data exactly, and `test_halt_labels_ic_and_expected_return_are_pit` covers halts.
- `tests/test_alpha.py::test_alpha_model_as_of` checks scores are standardised, `expected_return / (ic_estimate · score)` is a positive volatility, the half-life is finite, and the ML component is absent before the first OOS fold.

### References
- [MSCI, Converting Scores into Alphas](https://www.msci.com/documents/10199/1645561/PI_Converting_Scores_Into_Alphas.pdf/7adf1f42-10aa-40eb-9e8c-ecc11eeba2d4)
- [Suhonen, Lennkh & Perez (via Alpha Architect)](https://alphaarchitect.com/?p=36892)
- [Gârleanu & Pedersen, Dynamic trading with predictable returns](https://research-api.cbs.dk/ws/portalfiles/portal/60191441/lasse_heje_pedersen_et_al_dynamic_trading_acceptedmanuscript.pdf)

## 9. Evaluation

All in `qts/alpha/evaluation.py`; the per-date correlations live in `qts/core/stats.py`.

**Rank IC** (`rank_ic_series` = `stats.rank_ic_series`): for each date, the Pearson correlation of the cross-sectional *ranks* of score and label (Spearman), over names with both, NaN if fewer than `IC_MIN_OBS` = 20. It is fully vectorised: both panels are ranked row-wise once and `stats.row_corr` takes every date's correlation in one array operation (no per-date loop), exactly `scipy.stats.spearmanr` on each paired cross-section. Ranks make it robust to outlier returns and match how the score is used. `ic_report` also returns the hit rate (share of dates with IC > 0; ≈60% in the realistic §1 simulation) and a by-year table with per-year t-stats — an IC earned in two lucky years is not a signal.

**ICIR and its annualisation** (`ic_report` → a typed `ICReport`; dict-style access still works during the transition):

$$
\text{ICIR}_{\text{ann}} = \frac{\overline{\text{IC}}}{\text{sd}(\text{IC})}\sqrt{P}, \qquad t = \frac{\overline{\text{IC}}}{\text{sd}(\text{IC})}\sqrt{n}
$$

where $P$ is the number of IC observations per year and $n$ the number of observations. The audit's Finding 4: the pipeline sampled one IC every 20 days ($P$ = 12.6) but annualised with $P$ = 252, inflating ICIR by $\sqrt{20} \approx 4.5\times$. Now $P$ is inferred from the index spacing (`infer_periods_per_year`), and the pipeline passes the same $P$ to the decile Sharpe (it used `round(252/20) = 13`, 1.6% high; second-pass finding 15). The plain `t_stat` is right for the pipeline's non-overlapping samples; `ICReport.t_stat_nw` is the Newey–West t (lags below) to read for a daily IC series of multi-day labels, which is autocorrelated.

**Newey–West lag rule.** Overlapping $h$-day returns make consecutive observations correlated, so t-stats use a Newey–West (HAC) variance, which adds weighted autocovariances:

$$
\hat V = \hat\gamma_0 + 2\sum_{\ell=1}^{L}\left(1-\frac{\ell}{L+1}\right)\hat\gamma_\ell, \qquad t = \frac{\bar x}{\sqrt{\hat V/n}}, \qquad L = \min\!\Big(\max\big(h,\ \lfloor 4\,(T/100)^{2/9}\rfloor\big),\ T-1\Big)
$$

where $h$ is the number of consecutive observations that share label days (`overlap`; $h$ for daily observations of $h$-day labels, 1 for non-overlapping samples), $T$ the number of observations, and $\lfloor 4(T/100)^{2/9}\rfloor$ the Newey–West (1994) automatic bandwidth (`evaluation.py::newey_west_lags`, `stats.default_nw_lags`). Worked values: $(T, h) = (100, 20) \to 20$; $(252, 1) \to 4$; $(3000, 5) \to 8$; $(10, 20) \to 9$ (capped at $T-1$).

*Why floor at $h$, not $h-1$.* The old code used $L = h - 1$ (audit Finding 15). An MA($h-1$) process, which is what overlapping $h$-day sums produce, has non-zero autocovariances up to lag $h-1$, but the Bartlett weights $1 - \ell/(L+1)$ down-weight exactly those lags; with $L = h-1$ the last one gets weight $1/h$ and the variance is under-stated. Flooring at $h$ keeps every overlapping lag with a weight above $1/(h+1)$ and lets the automatic term cover residual serial correlation when there is no overlap. On the golden file this moved only the decile t-stat (1.2381 → 1.2451; `docs/notes/golden_changes_2026-11-05.md`).

**Decile spreads** (`decile_spread`): top-minus-bottom decile forward return per formation date (optionally value-weighted), t-stat with the lag rule above, Sharpe annualised by `periods_per_year / horizon`. The standalone hurdle is a t-stat above `ValidationConfig.hurdle_t_stat` = 3 (Harvey–Liu–Zhu multiple-testing threshold).

**What the pipeline scores.** `ResearchPipeline.evaluate` samples every $H$-th execution date $t$ and scores the signal that was **actually traded** there, `alpha.as_of(t - L)`, against that signal date's label and raw forward return (which start at close $t$). Frames are indexed by the execution date. Before the fix it scored `as_of(t)` on execution dates, a score that was never traded (second-pass finding 4; on seed 0, residual IC 0.0425 reported vs 0.0408 for the score actually traded).

**IC decay.** Two tools:
- `ic_report(..., decay_targets={h: label_h})` → `ICReport.decay`: the mean rank IC of the sampled score against risk-only residualised labels at `horizon_days` and every `AlphaConfig.secondary_horizons` entry (default 5, 20, 60 days), aligned to the execution lag like the training label. The pipeline computes the secondary labels on the sampled dates only (`residualize_forward_returns(..., only_dates=...)`), so they cost little. Diagnostics only, never trained on.
- `ic_decay_curve(score, md, horizons=(1, 5, 10, 20, 40, 60, 120))` for research: mean IC at each cumulative horizon. A signal whose IC peaks at 5 days is a timing signal; one still rising at 60 belongs in a slow sleeve.

**Incremental IC** (`incremental_ic`): regress the new signal on the existing composite per date, take the residual, and measure its IC. Gate: > 0.005 and positive in ≥ 2/3 of folds. A new signal that is just a noisy copy of momentum has high standalone IC and zero incremental IC.

`tests/test_alpha.py::test_half_life_and_decay` exercises half-life, decay, incremental IC and decile spreads on the synthetic market; `test_newey_west_lag_rule_and_typed_ic_report` the lag rule. → see **Validation** for the full five-gate protocol.

### References
- [Harvey, Liu & Zhu (via CXO Advisory)](https://www.cxoadvisory.com/big-ideas/taming-the-factor-zoo/)
- [Newey & West (1987)](https://doi.org/10.2307/1913610)

## 10. What is deliberately not in the model

| Item | Status | Evidence |
|---|---|---|
| Text / LLM news overlay | Phase 3 | Headline drift decayed from Sharpe 6.54 to 1.22 (2021Q4 → Jan–May 2024), unprofitable at 20 bp round trip (Lopez-Lira & Tang, post-cutoff OOS). Look-ahead contamination: Llama-3.3-70B marginal effect 0.30% where it memorised the outcome vs 0.14% where not (t = 3.64); DatedGPT lookahead premium 26.4 bp per SD (t = 10.65). Best use: momentum-conditioning overlay (S&P 500 Sharpe 0.79 → 1.06 net of 2 bp over 15 months, 10% significance; Anic et al.). Plan: open-weight embeddings with known cutoff, post-cutoff tests only, LAP diagnostic |
| Alternative data (web traffic etc.) | Phase 3, budget-dependent | Similarweb traffic predicts for 1,000+ large firms (magnitudes unavailable); loan fees, already a feature, are the best-documented alt signal |
| Reinforcement learning | Rejected for allocation | Not robust, unstable across seeds (Velay et al.; FinRL contests) |
| Standalone short-horizon alphas, PEAD, retail flow | Rejected | §2 |

### References
- [Lopez-Lira & Tang](https://arxiv.org/html/2304.07619v6); [LAP test](https://huggingface.co/papers/2512.23847.md); [DatedGPT](https://arxiv.org/pdf/2603.11838); [Anic et al.](https://arxiv.org/html/2510.26228v1); [Berkeley Haas on Armstrong et al.](https://newsroom.haas.berkeley.edu/companies-website-traffic-proves-powerful-predictor-of-financial-performance-and-stock-returns/)

## 11. Tuning guide, pitfalls and a worked example

### Tuning guide

| Knob | Default | Turn it when |
|---|---|---|
| `AlphaConfig.horizon_days` | 20 | Testing the 60-day head; changes purge, vol scaling, optimizer horizon |
| `theme_weights` | all 1.0 | Never on backtest results alone; only with an ex-ante economic argument |
| `factor_momentum_max_tilt` | 0.25 | 0 to switch off; above 0.25 reintroduces crowding risk |
| `composite_weight` | 0.5 | Raise if ML incremental IC fails the gate in a fold majority |
| `ic_shrink` | 0.5 | Fraction of the IC kept; lower (more shrink) as more trials are logged |
| `default_ic` | 0.03 | Pre-shrink prior before 60 realised IC dates; lower it for an untested signal |
| `delisting_gap_days` | 63 | Days without a close before a gap counts as a delisting (no vendor dates); shorter resolves labels sooner but calls long halts delistings |
| `secondary_horizons` | [5, 60] | Horizons reported in the IC-decay section; never trained on |
| `lgbm.min_data_in_leaf` | 200 | Raise if OOS IC ≪ validation IC |
| `lgbm.n_jobs` | 4 | Set to the core count on a dedicated machine |
| `mlp.n_iter_no_change` | 10 | Early-stopping patience (time-contiguous validation) |
| `ExecutionConfig.signal_lag_days` | 1 | Changing it changes the labels; the pipeline refits automatically |

Log every change as a trial so the deflated Sharpe stays honest.

### Pitfalls

- **Data snooping.** Mining ~29,000 accounting ratios and keeping |t| > 2 yields "predictors" as good as published ones, and both decay ~50% post-sample (Chen, Lopez-Lira & Zimmermann). Write the hypothesis first; require t > 3.
- **Feature leakage.** Any `shift(-k)`, centred window, full-sample statistic, restated fundamental, or future universe membership. Guards: PIT tests, eligibility mask, registry.
- **Crowding.** Off-the-shelf features are what everyone runs (Numerai); momentum positioning at a 2002-high preceded the July 2026 drawdown. Guards: capped tilts, crash control.
- **Decay.** Expect published signals to lose a third to a half post-publication; monitor trailing OOS IC by theme.

### Worked example: three stocks by hand

Stocks A, B, C, same industry, two themes for brevity (value with two members, momentum with one), static equal weights.

**Raw features:** B/P = (0.8, 0.3, 0.5); E/P = (0.06, 0.02, 0.07); 12-1 momentum = (−10%, +25%, +5%).

**Ranks → rank-gauss** ($n=3$, mid-rank percentiles 1/6, 1/2, 5/6 → $\Phi^{-1}$ = −0.967, 0, +0.967). Industry demeaning within one industry shifts all values equally and leaves ranks unchanged.

| | log B/P rank → z | E/P rank → z | mom rank → z |
|---|---|---|---|
| A | 3 → +0.967 | 2 → 0 | 1 → −0.967 |
| B | 1 → −0.967 | 1 → −0.967 | 3 → +0.967 |
| C | 2 → 0 | 3 → +0.967 | 2 → 0 |

**Theme scores.** Value average: A +0.484, B −0.967, C +0.484; z-score (mean 0, population SD 0.684) → A **+0.707**, B **−1.414**, C **+0.707**. Momentum (one member): SD of (−0.967, 0.967, 0) is 0.790 → A **−1.225**, B **+1.225**, C **0**.

**Composite.** Equal-weight average: A −0.259, B −0.095, C +0.354; z-score (SD 0.259) → A **−1.000**, B **−0.366**, C **+1.366**.

**Grinold alpha.** Trailing OOS IC 0.04 → shrunk IC 0.02. Daily idio vols 1.5%, 2.0%, 1.0% → 20-day: ×√20 = 6.71%, 8.94%, 4.47%.

| | z | σ (20d) | α = 0.02·σ·z |
|---|---|---|---|
| A | −1.000 | 6.71% | −13.4 bp |
| B | −0.366 | 8.94% | −6.5 bp |
| C | +1.366 | 4.47% | +12.2 bp |

C — cheap on earnings, middling elsewhere, low risk — is the long; A — cheap on book but a momentum loser — the short. The alphas are about ±13 bp over 20 days (≈0.6 bp a day); whether that is worth trading is the cost model's call.

### References
- [Chen, Lopez-Lira & Zimmermann](https://arxiv.org/abs/2212.10317); [Numerai 2025](https://kenfus.github.io/machine%20learning/quant/competition/2025/06/20/numerai-2025.html)

## Roadmap and plan differences

| Plan item (BUILD_PLAN §2.3) | Code today |
|---|---|
| 60–100 Tier-1 features (S/EV, CF/P, ROE, accruals, spread) | 18 features |
| Secondary 5- and 60-day heads | `secondary_horizons` used for IC-decay diagnostics only; no model heads |
| Rank-correlation loss head | Not implemented |
| CatBoost; optional GRU over residual sequences | Not implemented |
| MLP: 3 layers, L1, batch norm, 5 seeds | 2 layers (64, 32), L2, 3 seeds |
| Value-spread tilt when spread > 80th pct | Not implemented |
| Reversal as execution-timing overlay | ML input and risk-only proxy only |
| Gârleanu–Pedersen half-life shrink in the default pipeline | Single-signal shrink on the combined alpha wired into the optimiser, **off** by default (`use_gp_shrink`); per-sleeve half-lives (`components_half_life`) computed but not forwarded |
| Vendor delisting dates for real data | `MarketData.delisting_dates` supported; must be loaded, otherwise the 63-day causal rule delays labels |
| Tier-2 Alpha158-style and residual-sequence features | Phase 2 |
| Text and alt-data overlays | Phase 3 |
