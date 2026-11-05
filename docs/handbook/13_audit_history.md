# Audit history & known limitations

<callout icon="🧭" color="gray_bg">
This page records how the system was checked, what each check found and how it was resolved, and what is still not known. Between 4 and 5 November 2026 the code went through four rounds of independent review: a first audit (four reports), a code-quality review that drove a refactor, a "documentation review" (writing this handbook line by line against the code), a focused investigation of the execution delay, and a second audit (three reports) followed by a fix wave. The page gives the timeline, a severity table of the defects that mattered most, and an honest list of known limitations. Read it before trusting any number elsewhere in the handbook: every result so far comes from synthetic data.
</callout>

## 1. Why a quant system needs an audit trail

### What it is
An audit, here, is a review by someone who did not write the code, with the job of finding what is wrong. Reviewers read the modules line by line, checked numbers with small scripts against hand computations or independent implementations, and reported each finding with a severity, a file and line, and a proposed fix. They never modified source files; fixes came afterwards and were verified.

### Why we need it here
In most software, a bug announces itself: a crash, a wrong page, a failed test. In a backtest, the most dangerous bugs make results *better*. A look-ahead leak, a missing cost, a trial count that is too low or a survivorship gap all raise the Sharpe ratio, and nobody complains about a good number. The validation page explains why statistics cannot catch these (→ see **Validation & statistics › Why validation is the core**); the only defence is structural checks plus people actively looking for them. This page is the record of that looking.

<callout icon="💡" color="blue_bg">
The severity scale used by every audit: **CRITICAL** — look-ahead, or a wrong sign or scale that would mislead a go/no-go decision. **HIGH** — a wrong number that preserves direction, or a wrong input to a reported gate. **MEDIUM** — fragility, a silent fallback, or unmodelled optimism. **LOW** — style, documentation, or an immaterial approximation.
</callout>

## 2. Timeline at a glance

| When (2026) | Review | Scope | Headline result | Resolved in |
|---|---|---|---|---|
| 4 Nov | **First audit**, four reports | alpha/validation/data; risk/portfolio; costs/execution/backtest; code quality | 3 CRITICAL, 9 HIGH; code quality 5/10 | commit `3bfcf19` (same day), `docs/audit/RESOLUTION.md` |
| 4–5 Nov | **Documentation review** (writing pages 01–11) | every module, against the code and the plan | 46 findings, 5 design decisions (D1–D5), two hidden data-loss/look-ahead bugs | Wave 1 (`63ce32b`) and Wave 2 refactor (`ab8cea0`) |
| 4–5 Nov | **Code-quality refactor** (driven by the code-quality report) | cost contract, config, manifest, tests, tooling, performance | typed cost contract, serialised config, run manifest, 100 → about 360 tests | `63ce32b`, `ab8cea0`, `a4f2fe1` |
| 5 Nov | **Lag investigation** | a 0.85-Sharpe drop from a one-day execution delay | not a lag effect: path noise from a dirty tree; found a real future-universe leak and a crash | `6572752` |
| 5 Nov | **Second audit**, three reports | alpha/data/validation; risk/portfolio; costs/backtest/API | no CRITICAL; 4 HIGH (sizing, halt labels, ledger key, ledger concurrency) | Fix wave 3 (working tree at the time of writing) |

The file names are `docs/audit/alpha_validation_data.md`, `risk_portfolio.md`, `tcost_execution_backtest.md`, `code_quality.md` (first audit), `docs/notes/doc_review_findings.md` (documentation review), `docs/notes/lag_investigation_2026-11-05.md`, and `docs/audit/second_pass_alpha_data_validation.md`, `second_pass_risk_portfolio.md`, `second_pass_costs_backtest_api.md` (second audit). `docs/WORK_PLAN.md` keeps the status log.

## 3. The first audit (four reports, 4 November)

### Alpha, validation and data
The most serious findings of the whole project came from this report, because they were look-ahead bugs that produced a plausible, not an absurd, backtest.

- **Universe look-ahead (finding 1, CRITICAL).** `apply_mask` kept every ticker that was *ever* in the universe and left its prices unmasked. Cross-sectional ranks, label residualisation, ML training rows and emitted scores therefore included names that would join the universe in the future. Names about to enter the universe had returned **+110% a year in the 63 days before entry** (they entered *because* they had grown), so their presence in the cross-section was pure future information. Fix: a point-in-time `MarketData.eligible` panel; features, labels and scores computed on eligible names only; ineligible names dropped rather than zero-filled; a simulator eligibility hook with a five-day forced exit.
- **Alpha themes in the label (finding 2, CRITICAL).** The default label was residualised on momentum and value, the very themes the composite predicts, and the pipeline silently fell back to that default during the risk model's warm-up. Projecting momentum out of the target defines momentum alpha away. Fix: risk-only styles with an assertion guard; later formalised as design decision D1 (§5).
- **Gate wiring (finding 3, HIGH).** The decile gate received the *IC* t-statistic, and a missing PBO counted as a FAIL, so `overall` could never be PASS. Fix: the real decile spread t-stat, and three gate statuses (PASS / FAIL / skipped) with a required set.
- **ICIR ×4.5 (finding 4, HIGH).** A 20-day-sampled IC series was annualised with $\sqrt{252}$. Fix: infer the sampling frequency.
- **Delisting returns missing from labels (finding 5, HIGH)**, **restated fundamentals joined with `keep="last"` (finding 6, HIGH)** and **a hard-coded DSR trial variance with re-runs double-counted (finding 7, HIGH)** were fixed the same day. The planted synthetic alpha was found to be contemporaneous rather than lagged (finding 8); that was fixed in Wave 1.

### Risk and portfolio construction
- **Beta-neutral on the wrong column (finding 1, CRITICAL).** `beta_neutral` constrained the `market` (country) factor, to which every stock has exposure 1. That constraint is just a duplicate of the net-exposure band. The actual beta style exposure of the "beta-neutral" book was about **2**. The test suite missed it because its stand-in risk model had no `market` column. Fix: `RiskModelOutput.factor_groups`, and the optimizer resolves beta, industry and style factors by role, never by string prefix.
- **Vol-target scale-up breaking constraints (HIGH).** The book was multiplied by a scalar after the solve, which broke the net, beta, industry and participation bands. Fix: a projection re-solve over the full constraint set with a second-order-cone risk cap.
- **Industry bands silently off (HIGH)** for any label not starting with `IND` (all real Fama–French 48 names).
- MEDIUM items included a USE4 formula slip (`1.35·IQR` for `IQR/1.35`), an uncalibrated alignment variance $\nu$, and silent solver failures.

### Transaction costs, execution and backtest
- **Venue-blind impact calibration (HIGH).** `calibrate_b` regressed on $\sigma\sqrt{F/\text{ADV}}$ even for auction fills, whose volume is about 9% of ADV, so the auction prefactor came out **3.3× too high**.
- **Borrow at general-collateral rates for every short (HIGH)** when borrow data was missing. Fix: a short-interest / days-to-cover proxy with a hard-to-borrow cap and a small-cap multiplier.
- **Turnover joined one day off (MEDIUM, listed as HIGH in the resolution)** in breakeven and cost sensitivity. Fix: positional alignment.
- MEDIUM items included a zero SEC fee before 2026, no multi-day impact persistence, and delistings exiting free at zero return.
- LOW items that stayed open (§9): borrow charged once per simulation step, `spread_bps` holding a half-spread under a full-spread name, no short rebate or cash interest.

### Code quality
The engineering review scored the code **5/10** overall (reproducibility lowest, at 3). Its main points: two duplicated cost models, about 40 tuning knobs outside the config, no serialisable config, no record of which code and data produced a run, no CI, and a daily loop that would make a 1,500-name, 15-year run take 4–12 hours. It did not find wrong numbers; it explained why wrong numbers would be hard to find.

### What the first audit verified as correct
The auditors also re-derived, with scripts, what was right: `as_of` matched truncated refits on the dates tried, the trainer's purge was exact, the NAV identity held to 1e-10, optimizer and simulator costs agreed to machine precision, and the risk and DSR formulas matched references. Positive findings tell the next reviewer where not to spend time.

## 4. The code-quality refactor (Waves 1 and 2)

The work plan turned the code-quality report into Phase 1 items 1.1–1.8 and ran them as two waves.

**Wave 1 (`63ce32b`)** built the test foundation before touching behaviour: a hand-computed 3-stock × 5-day simulator fixture, a golden-file regression of the pipeline summary, determinism tests, property-based point-in-time tests with the hypothesis library, and the removal of wall-clock assertions (168 tests). It added ruff, pyright, pre-commit and CI; lagged the synthetic planted score by one day so the oracle was fair; and ran the first calibration of $\nu$.

**Wave 2 (`ab8cea0`)** was the refactor proper: one typed cost contract with `TransactionCostModel` as the only production implementation; the routed venue; the execution-delay option; a serialisable, hashable `SystemConfig`; the `RunManifest`; a data hash over every panel; the store upsert fix; vectorised costs (15–30× faster); an exposure tensor (about 6× less memory). Every moved field in the golden file was attributed to a named change before the file was regenerated (`docs/notes/golden_changes_2026-11-05.md`).

One performance goal was not met: the optimizer's daily solve only got 1.2–1.5× faster. The solver itself dominates, and cvxpy's parametrised (DPP) compile of this problem needs memory that grows like $N^2$, so reusing one compiled problem across days is off by default (§9).

## 5. The documentation review (writing this handbook)

Writing pages 01–11 for a reader who is not a quant forced every formula and default to be checked against the code. The writers logged **46 findings** in `docs/notes/doc_review_findings.md` and the orchestrator took five design decisions:

| Decision | Content |
|---|---|
| D1 | The label is residualised on exactly the set the optimizer neutralises (market, industries, beta, size, residual vol, liquidity, short-term reversal), defined once in `qts/core/roles.py`. Predicting a premium the book must hedge away wastes model capacity. |
| D2 | Auction impact prior $b_a = 0.3$ (was 0.5). With auction volume ≈ 9% of ADV, $0.3/\sqrt{0.09} = 1.0$, the same impact as the continuous session, so the auction saves the spread and never costs more impact — consistent with the router sending small orders to the close. |
| D3 | The simulator prices each trade on the **routed venue**: the part up to the auction cap at the auction, the rest in the continuous session. |
| D4 | An execution-delay option `signal_lag_days`. (Default flipped from 0 to 1 after the lag investigation, §6.) |
| D5 | One IC-shrink convention: `ic_shrink` is the fraction kept. |

Most findings were documentation mismatches (a stale prefactor in a docstring, ">" in text and ">=" in code). Three were real defects that no audit had caught:

- **#33, store data loss.** `ParquetStore` rewrote a whole year partition with only the new rows: appending one day of data deleted the rest of the year. Fixed with a key-based upsert.
- **#46, eigen-cache order dependence.** The eigenfactor bias cache was filled by whichever `as_of` call reached a refresh bucket first. A backtest that happened to query a late date first used a bias estimated from later data at earlier dates: a look-ahead that depended on call order. Fixed by keying the cache on the bucket start and estimating from history up to it.
- **#19, APCA end-of-sample factor count.** The statistical companion chose its number of factors on the end-of-sample window and applied it to every earlier date. Fixed by choosing $k$ per refit (Bai–Ng on the refit's own window).

Others led to features: as-traded price screens (#34), a data hash over every panel and a code version in the ledger (#37), the Newey–West lag floor at $h$ (#7), purged walk-forward gate folds (#27) and the planted oracle through the real pipeline (#45).

<callout icon="🤔" color="purple_bg">
Why count documentation as a review? Because explaining a formula to someone who does not already believe it is the most thorough read a module gets. The review found two look-ahead bugs that two specialist audits had missed, because the auditors checked *whether* `as_of(t)` matched a truncated fit on the dates they tried, while the writers asked *how* the cache was keyed.
</callout>

## 6. The lag investigation (5 November)

The post-refactor demo (`runs/demo_post_refactor.txt`) compared the same strategy at execution delay 0 and 1 and showed net Sharpe falling from **0.96 to 0.11**. The planted signal has a one-day autocorrelation of about 0.989, so a one-day delay should cost about 1% of the edge. Four explanations were tested: a broken lag implementation, same-day leakage at lag 0, a real short-horizon effect, or something else.

The answer was "something else" (`docs/notes/lag_investigation_2026-11-05.md`):

- The gap did not reproduce. The demo had run from an uncommitted working tree that was changed again before it was committed. On committed code the same pair gave gross 0.68 → 0.60 and net 0.23 → 0.14, a difference of −0.08 ± 0.15 (paired block bootstrap).
- Across five synthetic seeds, lag 1 was *not* worse than lag 0 (mean gross difference +0.16). The composite kept 93–98% of its IC at a one-day lag.
- The two demo books were different paths, not lagged copies (daily return correlation 0.75). The cost-aware optimizer is path dependent, and a single 1,000-day path moves by 0.5–0.8 Sharpe under small changes.
- No same-day leakage was found; new tests confirm the simulator's lag path with the real optimizer and cost model.

But the investigation found two genuine bugs:

- **Future-universe standardisation (point-in-time).** The risk model's beta and residual-volatility exposures regressed every name on the mean return of *all columns*, and the style standardisation took its standard deviation over all names. Because `apply_mask` kept the columns of future entrants, exposures at $t$ depended on which names would join the universe after $t$. Refitting on data truncated at $t$ moved the exposure matrix by up to 0.065 and the factor covariance by up to 1e-3; the labels, residualised on those exposures, moved the Grinold IC slightly. After the fix, every decision input matched to ≤ 3e-15.
- **A crash at lag ≥ 2** on the first backtest date (risk model warm-up).

The default `signal_lag_days` was then set to 1 (decision D4): real market-on-close orders must be in by 15:50 ET, so a lag-0 strategy uses a price it cannot have, and on this market the conservative setting costs nothing measurable.

## 7. The second audit (three reports, 5 November) and fix wave 3

Three fresh reviewers re-derived everything, explicitly told not to trust the first audit's conclusions. Their verdicts:

- **Alpha, data, validation:** no CRITICAL. On the synthetic market every decision input of the full pipeline is point-in-time to ≤ 5e-14 at 14 date/configuration pairs. On real data it was not yet guaranteed, because of two leaks synthetic data cannot show, and the validation layer under-counted trials.
- **Risk and portfolio:** no CRITICAL. The risk model was "trustworthy for paper trading", re-verified to machine precision. Portfolio construction enforced its constraints with correct units, "but the default sizing is not" right.
- **Costs, backtest, API:** no CRITICAL or HIGH. The backtest's cost, timing and NAV accounting reproduced an independent hand computation exactly; the deliberate biases all overstate cost. One gate (breakeven) was optimistic, and the local API needed a Host-header check.

The findings that mattered, and how fix wave 3 resolved them:

| Finding | Severity | Resolution |
|---|---|---|
| **Sizing via an alpha-blind projection.** The default risk aversion $\gamma = 10$ sized the optimal book at 1.3–3.5% volatility; a projection with no alpha term then levered it 3–5× to the 10% target (the `MAX_SCALE_UP = 5` clamp was hit). Costs and the alpha trade-off were solved for a book a third to a fifth of the size traded. | HIGH | `risk_aversion_mode = "target_vol"` (default): $\gamma$ is searched per date so the alpha-aware optimum has the target volatility; `risk_aversion = 2.0` for the fixed mode; the projection, kept for fixed mode, now uses a risk-weighted tracking variance in NAV units. |
| **Halts vs delistings in labels.** The rule used `bfill()` over the whole panel: a halt that later resumed was a gap in a backtest and a −30% delisting live. 4,228 residualised label cells realised by $t$ differed, and through them the IC estimate and expected returns. | HIGH | Causal rule: vendor delisting dates when available, otherwise a gap of `delisting_gap_days = 63`; `label_realisation` delays rows until resolved; `tests/test_label_halts.py`. |
| **Ledger key collisions.** The de-duplication key omitted the code version and every pipeline knob outside the config; three runs with different code collapsed into one trial. | HIGH | Key on name + config hash + data hash + clean code version + run spec; dirty or unknown code never de-duplicated. |
| **Ledger lost trials under concurrency.** Read-modify-rewrite with a fixed temp file: 105 of 401 trials lost in a race test. | HIGH | Strictly append-only under `flock`, de-duplication at read time. |
| Labels, IC and gates measured on the lag-0 signal while the book traded the lag-1 signal. | MEDIUM | Labels start at the execution close; the trainer's embargo and IC window grow by the lag; `evaluate()` scores `as_of(t − L)`. |
| Industry labels first used after $t$ changed the risk model at $t$ (eigen-simulation dimension), up to 1.2% in factor variances. | MEDIUM | Industry columns as of each date. |
| `apply_mask` discarded pre-entry history, so new entrants had NaN liquidity features for ~40 days and an ADV off by up to 22%. | MEDIUM | `apply_mask` sets eligibility and keeps every panel. |
| Per-share fundamentals and the short-interest ratio wrong by the split factor after a split. | MEDIUM | Split-consistent ratios (totals over market cap; shares as of settlement). |
| `ParquetStore` lost concurrent updates; a NaN cell deleted a stored value. | MEDIUM | Per-dataset `flock`; NaN keeps the stored value by default. |
| A restated older period displaced the current one in fundamentals. | MEDIUM | Rows older than the latest visible period are dropped from level panels. |
| PBO grid runs never logged as trials. | MEDIUM | Grid pipelines share the ledger. |
| The optimizer used 47 out-of-model names with zero alpha as free hedges for the net band. | MEDIUM | Only risk-model names can be held or traded. |
| Breakeven and cost sensitivity left out borrow. | MEDIUM | Both start from gross-after-borrow returns. |
| The TCA regression hard-coded the 3/2 exponent. | MEDIUM | Uses `impact_exponent`. |
| Weak config-override validation; no Host-header check on the local API. | MEDIUM | Semantic bounds on `CostConfig` / `ExecutionConfig`; `TrustedHostMiddleware`. |
| $\nu$ calibrated at $\gamma = 10$, lag 0, with phantom hedges. | LOW | Recalibrated with production settings: 0.25 → **0.05** (search floor on all three seeds). |
| About 25 LOW items (FINRA lag in business days not NYSE sessions, untracked files not counted as dirty, $N_{\text{eff}}$ rounded down, calendar-year IC folds, and others). | LOW | Mostly fixed in the same wave; the code cites each finding number. The rest are in §9. |

At the time of writing, these fixes are in the working tree but not yet committed. The fast test suite (333 tests) passes on it; `runs/demo_audit2.txt`, a demo on the fixed code, had not yet been produced.

## 8. The defects that mattered most

The table ranks the defects found across all reviews by how much they could have misled a decision. "Effect" is what was measured; "Caught by" is the review that found it.

| # | Defect | Severity | Effect (measured) | Caught by | Fix |
|---|---|---|---|---|---|
| 1 | Universe look-ahead: future entrants in every cross-section | CRITICAL | Entrants returned +110%/yr in the 63 days before entry; their presence leaked that into ranks, labels and scores | First audit (alpha/data #1) | `MarketData.eligible`; eligible-only cross-sections |
| 2 | Alpha themes in the label residualisation | CRITICAL | The target had momentum and value projected out, defining the composite's own alpha away; definition switched at risk warm-up | First audit (alpha/data #2) | Risk-only styles; D1 in `qts/core/roles.py` |
| 3 | Beta neutrality on the country factor | CRITICAL | "Beta-neutral" book carried beta style exposure ≈ 2 | First audit (risk #1) | Factors resolved by role (`factor_groups`) |
| 4 | Future-universe standardisation in risk exposures | HIGH (look-ahead) | Exposures at $t$ moved by up to 0.065 when refitted on data to $t$; Grinold IC moved | Lag investigation | Market proxy and standardisation on eligible names |
| 5 | Eigen-cache order dependence | HIGH (look-ahead) | Bias at early dates could be estimated from later data, depending on call order | Documentation review (#46) | Cache keyed on bucket start, history up to it |
| 6 | APCA factor count chosen on end-of-sample window | MEDIUM (look-ahead) | $k$ at early dates reflected the whole sample | Documentation review (#19) | Bai–Ng per refit |
| 7 | Halts treated as delistings using future prices | HIGH | 4,228 label cells realised by $t$ differed (up to 3.7 z); expected returns moved 0.9% | Second audit (alpha #1) | Causal rule, `label_realisation` |
| 8 | Ledger key omitted code version and run knobs | HIGH | Three code versions → one trial; DSR deflated too little | Second audit (alpha #2) | Full key; dirty never de-duplicated |
| 9 | Ledger rewrite lost concurrent trials | HIGH | 105 of 401 trials lost | Second audit (alpha #3) | Append-only with `flock` |
| 10 | Store rewrote a year with only the new rows | HIGH (data loss) | Appending one day wiped the year | Documentation review (#33) | Key-based upsert |
| 11 | Sizing via an alpha-blind projection ($\gamma = 10$) | HIGH | Alpha stage at 1.3–3.5% vol, levered 3–5×; transfer coefficient 0.69 vs 0.74 | Second audit (risk #1) | Target-vol risk aversion |
| 12 | Vol-target scale-up after the solve | HIGH | Net, beta, industry and participation bands broken | First audit (risk #2) | Constraint-preserving re-solve; later replaced by #11's fix |
| 13 | Venue-blind impact calibration | HIGH | Auction prefactor recovered 3.3× too high | First audit (costs #1) | Venue-aware volume |
| 14 | ICIR annualised with √252 for 20-day samples | HIGH | ICIR 4.47× too high | First audit (alpha #4) | Inferred frequency |
| 15 | DSR trial variance hard-coded, re-runs double-counted | HIGH | DSR depended on how often the demo ran | First audit (alpha #7) | Ledger family variance, effective trials |
| 16 | Breakeven ignored borrow | MEDIUM | 26.2 bp reported vs 24.3 bp; sensitivity 0.528 vs actual 0.463 | Second audit (costs #1) | Gross after borrow |
| 17 | Out-of-model names as phantom hedges | MEDIUM | Modelled market exposure 0.082 vs a 0.05 band | Second audit (risk #3) | Risk-model names only |

<callout icon="⚠️" color="yellow_bg">
Two of the look-ahead defects (rows 4 and 5) coexisted with point-in-time checks that passed: the first audit had verified risk `as_of` against truncated refits, on a market and a call order that did not expose them. They were caught by asking a different question: "what set of names, or what part of the history, does this cross-section or cache actually read?" The lesson is that "the output at $t$ equals a truncated refit at $t$" must be tested on markets where membership, classification and history change over time. The property-based PIT tests now use a time-varying universe, and dedicated tests plant late entrants, dated industries, halts, delistings and splits for exactly that reason.
</callout>

## 9. Known limitations

These are the things this system cannot yet claim, stated as plainly as possible.

### Synthetic data only
Every backtest, IC, Sharpe ratio, gate verdict, calibration and timing in this handbook was produced on the synthetic market generator (`qts/data/synthetic.py`). The generator plants a known factor structure and an alpha of known strength, so it can show that each layer recovers what is there and nothing more. It cannot show that the alpha exists in real markets. The demo is designed to fail its gates on merit (decile t-stat and DSR) and should be read as a test of the plumbing.

### Real-data biases are not yet testable
The second audit's verdict was precise: "on the synthetic market, the decision inputs are point-in-time"; "on real data they are not yet guaranteed". The fixes for the two leaks it found are in place, but the classes of bias that only real data contains have not been exercised:
- survivorship in the price source (`survivorship_check` exists, but no real source has been loaded);
- restatements, late filings and vendor revisions of fundamentals;
- a "current" industry map used as history (the store now reads `UNKNOWN` before a name's first dated classification, but someone has to supply dated classifications);
- IPOs (the generator has none), index reconstitutions and ticker changes;
- the bid–ask bounce and auction-imbalance reversal that make a lag-0 signal optimistic on real closes.

### The optimizer is solver-bound
The daily solve is dominated by the conic solver (Clarabel), not by Python. The refactor sped it up only 1.2–1.5×. Reusing one compiled problem across days (DPP) would save the ~20% spent canonicalising, but cvxpy's DPP compile of this problem needs about 0.8 GB at 300 names, 2.9 GB at 600 and more than 7 GB at 1,000, so `PortfolioConfig.reuse_problem` is off. The γ search adds up to 8 solves per date, but re-parametrises one compiled problem. A 1,000-name, 15-year backtest is a long job; the work plan's target (a composite run at 1,000 names × 15 years in under 30 minutes on a laptop) has not been demonstrated.

### The 10% volatility target is not reachable under gross 2.0
With gross exposure capped at 2.0 (`gross_leverage_max`), names at 2% and the beta, industry and style bands, the synthetic book's realised volatility tops out around 8–9%. Under target-vol sizing the γ search then stops at the caps and reports `vol_target_binding = "caps"`. This is a property of the constraint set, not a bug: either accept a lower realised volatility, or raise the gross cap (which needs portfolio margin rather than Reg T).

### Cost and risk parameters are priors
- The impact prefactors ($b_c = 1.0$, $b_a = 0.3$) and the 3/2 exponent are literature priors (design decision D2). They must be recalibrated from the system's own fills with `TCALedger.calibrate_b_by_venue`, which requires live or paper trading.
- The alignment variance scale $\nu$ = 0.05 was calibrated on three synthetic seeds, where it landed on the search floor. On real data the unexplained part of the alpha may carry real unpriced risk, so $\nu$ may need to be larger.
- The borrow-fee proxy stands in for logged IBKR borrow rates until at least 12 months are collected.
- The risk model over-forecasts the optimized synthetic book by about 12% (bias statistic 0.86–0.90). That is conservative for the volatility target, but it is a calibration question (eigen inflation, VRA) that only real data can settle.

### No live or paper trading
The execution layer (order types, routing, the IBKR algorithms) is implemented and tested against a fill simulator and a dry-run broker, but nothing has traded. The build plan's Phases 4 (paper) and 5 (live) and the kill switch are not started. Every statement about execution quality is a model.

### Single-path noise
On this market, small code or setting changes move a 1,000-day Sharpe by 0.5–0.8, and two books with nearly identical inputs correlate only 0.5–0.75 over four years. Any "A is better than B" claim needs a paired, multi-seed comparison with a bootstrap standard error. There is no helper for that yet.

### Remaining LOW items from the audits
These were listed by the audits and are still open after fix wave 3:
- **Borrow on sparse dates** (first audit costs #19): borrow is charged once per simulation step at `fee/252`; a run on a sparse date list under-charges it.
- **`MarketData.spread_bps` naming** (costs #20): the panel holds a *half*-spread under a name that suggests a full spread; a loader that feeds quoted spreads would double the spread cost (conservative).
- **No short rebate or cash interest** (costs #21): conservative in high-rate regimes.
- **Routed surrogate upper bound** (second audit costs #7): when the auction prefactor exceeds the continuous one (not the default), the optimizer prices the whole trade at the auction rate, up to 66% above the estimate per name; the backtest is unaffected.
- **`embargo_days` has no effect on the gate folds** (second audit alpha #16): only the purged walk-forward *test* blocks are used.
- **No IPOs in the synthetic generator** (alpha #19): halts, delistings and splits are now available, but off by default, and the full-pipeline slice-equivalence check with them on is an audit script, not a CI test.
- **Advisory locks**: the ledger and the store use `flock`, which is not reliable on network file systems.
- **Legacy ledger family**: records logged under the old family name `pipeline_phase1` are no longer read by new runs (the family is now per data set). They remain in the file.
- **Code-quality re-score**: the Phase 1 exit criterion asks for ≥ 8/10 on the code-quality rubric; no re-scored report is in the repo yet.

<callout icon="💡" color="blue_bg">
How to use this list: before loading real data, re-read "Real-data biases"; before trusting a cost number, re-read "Cost and risk parameters are priors"; before believing that one variant beats another, re-read "Single-path noise". Each limitation is also stated on the page that describes the mechanism (→ see **Data & universe**, **Transaction cost model**, **Portfolio construction**, **Validation & statistics**).
</callout>

## 10. What the reviews taught about the process

- **Independent reviewers find different things.** The first audit found the cross-sectional leaks; the documentation review found the cache and window leaks; the lag investigation found the standardisation leak; the second audit found the halt and ledger problems. No single reviewer found more than half of the look-ahead defects.
- **Fixes create new defects.** The halt-vs-delisting leak was introduced by the first audit's delisting fix. The ledger concurrency bug was introduced by the first audit's de-duplication fix. Every fix needs its own regression test and a second look.
- **Never trust a number from a dirty tree.** The 0.96 → 0.11 "lag effect" was an artefact of an uncommitted intermediate state that no longer exists. Stored demo outputs should come from committed code, and the ledger now refuses to de-duplicate dirty runs.
- **Look for optimism.** Most serious defects made results better; "are the net numbers conservative?" is the right default question for every layer.

## References
- `docs/audit/RESOLUTION.md` (first-audit resolution log), `docs/audit/alpha_validation_data.md`, `risk_portfolio.md`, `tcost_execution_backtest.md`, `code_quality.md`.
- `docs/notes/doc_review_findings.md`, `docs/notes/golden_changes_2026-11-05.md`, `docs/notes/lag_investigation_2026-11-05.md`, `docs/notes/alignment_nu_calibration.md`.
- `docs/audit/second_pass_alpha_data_validation.md`, `second_pass_risk_portfolio.md`, `second_pass_costs_backtest_api.md`.
- `docs/WORK_PLAN.md` (Phase 1 items and status log).
