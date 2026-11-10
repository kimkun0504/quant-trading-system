# UI redesign: research terminal

Agreed with the owner on 2026-11-10. This replaces the "annotated lab notebook" direction in
`ui/DESIGN.md` for layout and visual language; the engineering architecture in `ui/README.md`
(React 19 + TS strict + Vite, TanStack Query, uPlot/d3, MSW mocks, Vitest, Playwright + axe)
stays. Every step below ships on its own with tests green and the mock server updated.

## Purpose

An **analytics / research terminal** for testing and comparing strategies, not a trading
terminal. The unit of thought is the *comparison* ("what changed when I changed X, and is it
real or noise?"), not the single run.

## Visual direction

- Modern analytics product (density of Linear / Datadog, typography of a well-made SaaS app).
  Not Bloomberg-dense, not a notebook.
- One clean sans-serif (Inter or IBM Plex Sans; tabular figures for numbers), tight type scale,
  8-px grid.
- Light theme default, real dark mode, follows `prefers-color-scheme`, user override persisted.
- Muted surfaces; colour reserved for data: gain, loss, caution, one accent. Existing semantic
  palette (gain `#1F5F8B`, loss `#9B2F52`, caution `#B8860B`) carried over and re-tuned for dark.
- Explanations survive as hover on metric labels and a collapsible "about this chart" line
  (content from the handbook, same source as today's margin notes). No margin column.
- Dense tables (right-aligned tabular figures, compact rows), generous charts.
- Shell: left sidebar with the five workspaces; persistent **selection tray** at the bottom
  holding the runs currently selected for comparison; keyboard shortcuts for frequent moves.
- Accessibility: axe clean, keyboard reachable, contrast AA in both themes.
- Load `anthropic-skills:frontend-design` before step 1 and follow it for the token system.

## Workspaces

1. **Experiments** (home). Runs as rows, grouped into *families* (same strategy, different
   knob; family = ledger family / config-diff clustering). Select 2+ → side-by-side: NAV
   overlaid, fold Sharpe as paired bars, gates grid, **config diff** showing only differing
   keys (`portfolio.trade_aversion: 1.0 → 0.0`). DSR with trial count front and centre.
2. **Compare** (opened from the tray; same component as the Experiments side-by-side, full
   page).
3. **Dossier** (single run). Replaces the six tabs: one scrollable page, sticky header (name,
   config hash, gates, headline numbers), sections in reading order: performance → signal →
   portfolio → costs → validation → config, each compact with "open in lab" links.
4. **Signal lab**. IC by theme by year heatmap; IC decay curves per horizon; feature IC table;
   **learned weights over time**: per-refit feature importance (gain-based for LightGBM, |coef|
   for ridge, permutation or SHAP for the MLP, normalised per refit) as a heatmap, features ×
   refit date. Needs API: `GET /api/runs/{id}/alpha/refits` served from what the alpha model
   records at fit time (add recording if missing; store in the run directory as parquet).
5. **Portfolio anatomy**. Exposure over time stacked by industry and by style, gross/net,
   turnover, holding period, invested-day gaps drawn explicitly; holdings snapshot on a chosen
   date. API: exposures panel and weights sample already in the run directory.
6. **Cost & constraint lab**. Binding constraints over time and their transfer-coefficient cost,
   cost breakdown (spread / impact / fees / borrow), breakeven, cost sensitivity. API: binding
   counts per rebalance from simulator diagnostics (extend recording if only aggregates exist).
7. **Run builder**. Start from an existing run's config, edit fields, see the diff before
   launch, **queue** several runs (API job queue runs them sequentially; status in Experiments).

## Build order (each step: code + mocks + unit + e2e + axe + screenshots)

1. Design system + shell (tokens, sidebar, tray, dark mode) and the new Experiments screen.
2. Compare view.
3. Dossier replacing the tabbed run page; delete the tabs.
4. Signal lab + refit-importance API.
5. Portfolio anatomy + cost/constraint lab (+ binding-constraint recording).
6. Run builder with diff and queue.

## Out of scope (for now)

Node-and-edge drawing of the MLP (uninformative at 16 inputs; easy add later if wanted); live
trading views; mobile layout beyond "usable".
