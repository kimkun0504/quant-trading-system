# Participation alignment: optimizer ADV cap vs router urgency (P16), 2026-11-05

## The mismatch

The optimizer caps each name's daily trade at `PortfolioConfig.max_adv_participation` (5% of
ADV). The router sends the continuous part of an order to an algo with
`maxPctVol = ExecutionRouter.participation(h)`, the urgency-derived rate for the signal
half-life `h`: `urgency_from_half_life(h)` clipped to [1%, `ExecutionConfig.algo_max_pct_adv` =
5%], i.e. 5% for h <= 5 days, 2.6% at 10, 1.3% at 20, 1% from about 27 days. For a slow signal
the optimizer therefore plans same-day trades up to 5% of ADV that the router flags as
multi-day (`meta["multi_day"]`), and the daily backtest books them as filled at the close.

## The switch

`PortfolioConfig.participation_from_urgency: bool = False`. When true,
`SinglePeriodOptimizer.participation_rate(h)` returns
`qts.execution.router.participation_for_half_life(h, algo_max_pct_adv)` (the function the router
itself uses), with `h = AlphaOutput.half_life_days` as passed to `solve(alpha_half_life_days=)`,
and the per-name constraint becomes `|x_i| <= rate * ADV_i / NAV`. `algo_max_pct_adv` comes from
the optimizer's `exec_cfg` (the pipeline passes `cfg.execution`), else the cost model's
`exec_cfg`, else `ExecutionConfig()`. Each solve notes the cap it used. The auction child of a
split (up to 2% of the auction, about 0.18% of ADV) is not added on top: the switched cap is
slightly conservative. Tests: `tests/test_portfolio.py::test_participation_from_urgency`.

## Comparison

Synthetic market, seed 0, 300 names x 1,512 days, composite alpha (no ML), default config
otherwise, last 400 out-of-sample days (`ResearchPipeline.run(md, max_days=400)`), lag 1. The
measured alpha half-life on this market is about 112 days, so the switched cap is the 1% floor
(against 5%). Script: `ResearchPipeline` with `cfg.portfolio.participation_from_urgency` off/on,
no ledger, no run directory.

| NAV | switch | net Sharpe | gross Sharpe | turnover / day | cost bp / side | costs USD | mean TC | cap-binding name-days | failed solves |
|---|---|---|---|---|---|---|---|---|---|
| $5M | off | -0.257 | -0.003 | 5.48% | 10.15 | 127,204 | 0.705 | 0 | 0 |
| $5M | on | -0.259 | -0.006 | 5.47% | 10.14 | 126,959 | 0.705 | 9 | 0 |
| $100M | off | -0.802 | -0.497 | 2.25% | 22.03 | 2,186,189 | 0.619 | 5 | 0 |
| $100M | on | -1.159 | -0.905 | 1.69% | 22.44 | 1,761,090 | 0.605 | 35 | 56 / 400 |

(This 400-day window is a losing stretch of the synthetic path for every setting; the Sharpe
levels are path noise, only the on/off differences are informative, and they come from a
single path.)

## Reading

* **At $5M the switch is immaterial.** The cap binds on 9 name-days in 400; turnover, cost per
  side, TC and Sharpe move in the third decimal. This matches the second audit's judgement that
  the mismatch is immaterial at the demo NAV.
* **At $100M it bites, and not only through the cap.** Turnover falls by a quarter and total
  costs by 19%, but 56 of 400 solves become **infeasible** (the 1% cap cannot reach the beta /
  industry / style bands or a forced exit in one day), and on those days the simulator holds
  the previous book. Gross Sharpe drops from -0.50 to -0.90: the book tracks the signal less
  well (lower TC) and sits still on infeasible days.
* So the switch is the honest setting for large NAV *only together with* a way to stay feasible:
  `PortfolioConfig.relax_participation_on_infeasible = True` (drop the caps on an infeasible
  day), wider bands, or a multi-day schedule in the simulator. The alternative alignment (keep
  the optimizer at 5% and let the simulator spread over-cap trades across days) is not
  implemented.

## Decision

Default stays **off**: at the research NAV the two settings agree, and switched on without a
feasibility fallback it makes large-NAV runs fail solves. Turn it on (with
`relax_participation_on_infeasible`) when testing capacity at NAVs where the router would flag
many multi-day orders; read `participation` in the solve log's binding counts and
`n_failed_solves` in the summary either way.
