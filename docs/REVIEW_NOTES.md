# Review notes and amendments to the build plan

Amendments adopted in code after the senior-quant review of `BUILD_PLAN.md`
(2026-11-03):

1. **Phase-1 bar lowered** to net Sharpe ≥ 0.3 (`ValidationConfig.phase1_min_net_sharpe`);
   0.5 is the phase-2 (composite + ML) bar. Chen–Velikov's post-2005 net numbers make 0.5 a
   ceiling for a 6-theme composite, not a baseline.
2. **Loan-fee factor history** comes from FINRA bi-monthly short interest with an 8-business-day
   publication lag (`qts.data.loaders.parse_finra_short_interest`, `to_short_interest_ratio`);
   IBKR indicative rates are spliced in once ≥ 12 months accumulate.
3. **Industry layer = Fama-French 48 from SIC** (`qts.data.loaders.sic_to_ff48`), not GICS history.
4. **Label residualisation uses risk-only styles** everywhere (`ResearchPipeline.RISK_ONLY_STYLES`
   for the risk model's exposures, `qts.alpha.targets.DEFAULT_STYLE_EXPOSURES` for the internal
   fallback used only during the risk warm-up); exposures are measured at *t*; targets are never
   features; the exposures hook raises instead of silently falling back.
5. **Effective holding period is a monitored diagnostic** (`effective_holding_days`, target 50 d);
   the pipeline flags runs that fall below 60 % of target.
6. **Factor-momentum tilt capped at ±25 %** (`AlphaConfig.factor_momentum_max_tilt`).
7. **Default book 150/50** (`gross_leverage_max = 2.0`) — Reg T reality for small accounts; portfolio
   margin needed for more.
8. **Momentum crash control** (`qts.alpha.composite.momentum_crash_scalar`): sleeve scaled down when
   the 24-month equal-weight market return is negative and 21-day vol is above its 80th percentile;
   momentum sleeve vol-targeted to 12 %.
9. **Auction impact prefactor**: with V = closing-auction volume (≈ 9 % of ADV) the effective impact
   per dollar is `b_auction / sqrt(0.09)` ≈ 3.3 × `b_auction`. The review's 1.5 made every auction
   order 5× more expensive than an in-session algo, contradicting the routing table, so
   `impact_b_auction = 0.5` is the prior; `TCALedger.calibrate_b` recalibrates from own fills.
   The LOC limit offset floors at 1.5 × expected auction impact (`ExecutionRouter`).
10. **t > 3 applies to standalone signals only**; incremental features are gated on incremental IC and
    net-Sharpe delta (`HurdleReport.evaluate_signal(standalone=...)`).
11. **Build order**: thin vertical slice first (`qts.pipeline.ResearchPipeline`), then deepen.

## Integration findings (synthetic-market shakedown)

* Vol targeting must be allowed to **lever up toward target within the hard caps**
  (`PortfolioConfig.vol_target_scale_up`); scale-down-only left the book at 0.2× gross.
* A risk model built from observables cannot see latent factors with no observable
  footprint; books aligned with such factors ran at 1.8× ex-ante risk. The synthetic
  generator now exposes every latent style through a characteristic. In production this is
  the job of the **APCA companion model** (plan §3) — a statistical factor layer should be
  added to `qts.risk` before live trading.
* Explicit fee minimums ($0.35/order) dominate costs for sub-$1M NAV spread over 300 names;
  the demo uses NAV = $5M so cost figures reflect impact, not order minimums.
* The planted-alpha backtest confirms the chain (gross Sharpe 1.6 / net 1.3 with the true
  latent score; realised/ex-ante vol ≈ 0.9), so a weak composite result is an alpha
  question, not a plumbing one.

## Deepening backlog (ordered)

1. APCA / PCA statistical companion risk model and residual-covariance test (QIS already in place).
2. Real-data loaders wired to a `ParquetStore` (Norgate + Sharadar), Ken French download, FINRA SI.
3. Signal-by-signal IC half-life / turnover tables from OSAP/JKP (plan §10 gap).
4. ML sleeve tuning on the real panel (LightGBM was at the noise floor on synthetic data).
5. Text overlay (Phase 3): post-cutoff only, open-weight embeddings, LAP diagnostic.
6. ib_async live adapter + paper-trading harness around `qts.execution` (Phase 4).
