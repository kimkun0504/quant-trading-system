# Polish backlog (from the Nov 2026 handbook sync)

| # | Where | Item |
|---|---|---|
| P1 | qts/cli.py:105-108 | `--compare-lag` prints the lag-0 card unless `--signal-lag 1`; with default lag 1 it should print the production (lag-1) card and the lag-0 bound; comment claims models are reused but the pipeline refits on lag change |
| P2 | qts/validation/hurdles.py:340 | breakeven message "2x modeled X bps" shows the already-doubled figure |
| P3 | qts/backtest/simulator.py:636-641 vs qts/alpha/targets.py | simulator treats every missing next close as a delisting (−30 %); labels distinguish halts — use `md.delisting_dates`/causal rule in the simulator too (hold through halts) |
| P4 | ui/src/pages/ledger/LedgerPage.tsx:106 | copy says identical config+data "updates its record"; ledger is append-only with read-time de-dup (clean code version + same run spec) |
| P5 | ui/src/pages/run-detail/tabs/ValidationTab.tsx:133 | "yearly folds" → purged walk-forward blocks |
| P6 | ui/docs/screenshots | regenerate after copy fixes |
| P7 | api/qts_api/schemas.py:320, ui/src/features/launch-run/validation.ts:15 | warm-up hard-coded 512 days; derive from config (risk min history 252, ML train_min_days 756, + lag) and expose via API |
| P8 | scripts/run_real.py:5-8 | dataset list missing prices_close_raw, shares_outstanding, industry_history, delisting_dates |
| P9 | README.md:96-97 | says Norgate loader exists — only Sharadar SF1 transform; add a Norgate loader stub or fix text |
| P10 | qts/pipeline.py:105-120 | run-dir docstring missing artifacts.py files and gross_after_borrow |
| P11 | Makefile install | `.[dev]` → `.[dev,data]` |
| P12 | .github/workflows/ci.yml | add API tests job and UI typecheck/lint/unit job |
| P13 | runs/trial_ledger.jsonl | pre-Nov records have bare family/empty hashes — document or migrate |
| P14 | qts/execution/simulator_fills.py:140-150 | algo child of a split priced as independent order (cost model uses increment); partial fills not floored to shares; ArrivalPx reason text |
| P15 | qts/tcost/impact.py:235 | `cost_for_venue_choice` ignores auction exchange fee, no routed option |
| P16 | router.py:301 vs PortfolioConfig.max_adv_participation | router flags multi-day at 1–1.3 % for slow signals while optimizer trades up to 5 % ADV same-day — align (optimizer participation cap from urgency, or simulator spreads over days) |
