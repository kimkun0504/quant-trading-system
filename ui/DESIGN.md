# QTS Workbench — design brief

## Subject, audience, job

- **Subject:** a research workbench for one systematic-equity researcher running the `qts`
  pipeline: backtest runs, the five models' diagnostics, and the validation gates.
- **Audience:** the owner — an experienced software engineer who is learning quant finance.
  He knows what a p99 latency chart is; he does not yet read a bias statistic at a glance.
- **Primary job:** answer "is this run real, and why?" — inspect a run's performance, see
  which hurdle gates passed, understand each number, compare runs, and launch new ones.
- **Secondary job:** teach. Every number on screen can explain itself and link to the
  matching section of the Notion handbook.

## Concept: the annotated lab notebook

A run page reads like a page of a well-kept lab notebook: a calm column of findings with
**margin notes** beside every figure and statistic explaining what it means for _this_ run
("ICIR 1.51 — the IC was positive in 4 of 5 periods; below 2 means the signal is noisy").
The margin is the one bold idea; everything else stays quiet so the notes and the charts
carry the page.

### Revision pass (defaults checked and rejected)

| First instinct                                                | Why it is a default          | Revised to                                                                                             |
| ------------------------------------------------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------ |
| Near-black "terminal" theme with a neon green accent          | the generic trading-UI look  | light cool-paper surface with ink-blue text; dark theme is a respectful inversion, not neon            |
| KPI-card grid at the top (big number, small label, sparkline) | SaaS default                 | a single **verdict line** in prose plus a gate strip; key metrics as a compact table with margin notes |
| Monospace for all numbers                                     | tell of generated dashboards | proportional faces with tabular figures (`font-variant-numeric: tabular-nums`)                         |
| Red/green for P&L                                             | colour-blind unsafe          | deep blue (gain/long) vs wine (loss/short), with sign always shown                                     |
| Uppercase tracked labels over every section                   | template chrome              | sentence-case headings; labels only where they disambiguate                                            |
| Rounded cards with drop shadows everywhere                    | SaaS card kit                | sections separated by whitespace and a single hairline; radius only on interactive controls            |

## Tokens

### Colour (light)

| Name                                                                                   | Hex       | Use                                                      |
| -------------------------------------------------------------------------------------- | --------- | -------------------------------------------------------- |
| paper                                                                                  | `#F2F4F6` | app background (cool, not cream)                         |
| sheet                                                                                  | `#FFFFFF` | reading column / chart background                        |
| ink                                                                                    | `#17212B` | text, axes                                               |
| graphite                                                                               | `#5A6573` | secondary text, margin notes                             |
| rule                                                                                   | `#D3D9E0` | hairlines, gridlines                                     |
| gain                                                                                   | `#1F5F8B` | positive return, long book, PASS                         |
| loss                                                                                   | `#9B2F52` | negative return, short book, FAIL                        |
| caution                                                                                | `#B8860B` | skipped gates, warnings, drawdown fill (at 18 % opacity) |
| focus                                                                                  | `#3B82C4` | keyboard focus ring                                      |
| Dark theme maps paper→`#10161D`, sheet→`#161E27`, ink→`#E6EBF0`, graphite→`#9AA6B4`,   |
| rule→`#2A3542`, gain→`#6FA8D6`, loss→`#D97A9A`, caution→`#D9B25C`. All text pairs meet |
| WCAG AA (checked with a contrast script in the build).                                 |

### Type

- **Source Serif 4** — headings, the verdict line and margin notes (the explanatory voice).
- **Source Sans 3** — UI, tables, chart labels, numbers with tabular figures.
- Scale (1.2 minor third, base 16 px): 13.3 / 16 / 19.2 / 23 / 27.6 / 33.2.
- Line length ≤ 72 ch in the reading column; serif notes at 1.55 line-height, sans at 1.45.

### Layout

```
┌────┬──────────────────────────────────────────────┬───────────────┐
│nav │ Run 2026-11-05 · composite · synthetic seed 0 │               │
│    │ Verdict: passes cost & Sharpe gates, fails    │ margin notes  │
│Runs│ deflated Sharpe (4 trials) — not yet real.    │ (sticky,      │
│New │ [gate strip ■■■■□■■]                          │  aligned to   │
│Mod.│ ────────────────────────────────────────────  │  the figure   │
│Ldgr│ NAV + drawdown (full width of the column)     │  they explain)│
│    │ Metrics table                                 │               │
│    │ Tabs: Performance · Risk · Alpha · Costs ·    │               │
│    │       Execution · Validation                  │               │
└────┴──────────────────────────────────────────────┴───────────────┘
```

- Left-aligned everything; the reading column is 720–840 px; the margin column 260–320 px
  and collapses under each figure below 1100 px viewport width.
- Navigation rail: Runs, New run, Models, Ledger, Handbook (external).
- Charts span the reading column; wide tables scroll horizontally inside their section.

### Motion

One orchestrated moment: on opening a run, the NAV line draws once (≤ 600 ms), then the
gate strip fills left to right. Everything else is instant; `prefers-reduced-motion`
disables both.

## Principles

1. **Every number explains itself.** Hover/focus shows the definition; the margin note gives
   the interpretation for this run; a link opens the handbook section.
2. **Honesty over flattery.** Failing gates are stated plainly in the verdict; suspicious
   results (net Sharpe > 2) are flagged before anything else.
3. **Density without clutter.** Tables over cards; one hairline between sections.
4. **Same words as the system.** Labels use the handbook's vocabulary (net Sharpe, rank IC,
   bias statistic) so the docs and the UI teach each other.
5. **Accessible by default.** Keyboard reachable, visible focus, AA contrast, charts have
   text alternatives (a data table toggle).

## Architecture (summary — details in ui/README.md once built)

- `api/`: FastAPI over the `qts` run registry (`runs/<run_id>/`), background job runner,
  OpenAPI schema as the contract.
- `ui/`: React + TypeScript (strict) + Vite; TanStack Query for server state; React Router;
  types generated from the OpenAPI schema (`openapi-typescript`) with `openapi-fetch`;
  feature-sliced folders (`app/`, `pages/`, `features/`, `entities/`, `shared/`);
  CSS Modules over CSS custom-property tokens; uPlot for long time series, small SVG
  charts (d3-scale/d3-shape) for bars and heatmaps; Vitest + Testing Library + MSW;
  Playwright smoke test; ESLint + Prettier.

## Implementation record (UI foundation, 2026-11-05)

### Token additions

The brief's nine colours are implemented as written in `src/shared/styles/tokens.css`. Four
derived tokens were added because the build needed them; none introduces a new hue:

| Token                               | Light                                    | Dark      | Why                                                                                                                  |
| ----------------------------------- | ---------------------------------------- | --------- | -------------------------------------------------------------------------------------------------------------------- |
| `caution-ink`                       | `#7D5B00`                                | `#D9B25C` | `caution` (#B8860B) is 3.25:1 on white: fine for marks, fails AA as text. Warnings set in text use this darker step. |
| `on-accent`                         | `#FFFFFF`                                | `#10161D` | text on a `gain` fill (primary button, gate-marker glyphs)                                                           |
| `wash`, `caution-wash`, `loss-wash` | `color-mix()` of the accent into `sheet` | same      | row hover, suspicious banner, error summary                                                                          |

Focus in dark mode uses the dark `gain` (#6FA8D6) rather than #3B82C4, which is 4.0:1 on the dark
paper but reads as a different blue from every other accent.

### Contrast check

`npm run contrast` (`scripts/contrast.mjs`) reads `tokens.css`, mixes the `color-mix()` washes the
same way the browser does, and fails the build step if any pair drops below its minimum
(4.5:1 for text, 3:1 for marks, focus rings and input borders). Results for the shipped tokens:

| Theme | Pair                                        | Foreground | Background | Ratio | Needs | Result |
| ----- | ------------------------------------------- | ---------- | ---------- | ----- | ----- | ------ |
| light | ink on paper (body text)                    | `#17212b`  | `#f2f4f6`  | 14.78 | 4.5   | pass   |
| light | ink on sheet (tables, charts)               | `#17212b`  | `#ffffff`  | 16.29 | 4.5   | pass   |
| light | graphite on paper (margin notes)            | `#5a6573`  | `#f2f4f6`  | 5.37  | 4.5   | pass   |
| light | graphite on sheet (axis labels)             | `#5a6573`  | `#ffffff`  | 5.93  | 4.5   | pass   |
| light | gain on paper (links, positive numbers)     | `#1f5f8b`  | `#f2f4f6`  | 6.21  | 4.5   | pass   |
| light | gain on sheet                               | `#1f5f8b`  | `#ffffff`  | 6.84  | 4.5   | pass   |
| light | loss on paper (negative numbers, FAIL)      | `#9b2f52`  | `#f2f4f6`  | 6.52  | 4.5   | pass   |
| light | loss on sheet                               | `#9b2f52`  | `#ffffff`  | 7.19  | 4.5   | pass   |
| light | caution-ink on paper (warnings as text)     | `#7d5b00`  | `#f2f4f6`  | 5.65  | 4.5   | pass   |
| light | caution-ink on sheet                        | `#7d5b00`  | `#ffffff`  | 6.23  | 4.5   | pass   |
| light | on-accent on gain (primary button)          | `#ffffff`  | `#1f5f8b`  | 6.84  | 4.5   | pass   |
| light | paper on ink (tooltip, selected segment)    | `#f2f4f6`  | `#17212b`  | 14.78 | 4.5   | pass   |
| light | ink on caution-wash (suspicious banner)     | `#17212b`  | `#f5eedd`  | 14.09 | 4.5   | pass   |
| light | loss on loss-wash (error summary links)     | `#9b2f52`  | `#f7eef1`  | 6.32  | 4.5   | pass   |
| light | ink on gain wash 50% (heatmap max tint)     | `#17212b`  | `#8fafc5`  | 7.07  | 4.5   | pass   |
| light | ink on loss wash 50% (heatmap max tint)     | `#17212b`  | `#cd97a9`  | 6.67  | 4.5   | pass   |
| light | focus ring on paper (UI, 3:1)               | `#3b82c4`  | `#f2f4f6`  | 3.68  | 3     | pass   |
| light | focus ring on sheet (UI, 3:1)               | `#3b82c4`  | `#ffffff`  | 4.06  | 3     | pass   |
| light | gain marks on sheet (chart lines/bars, 3:1) | `#1f5f8b`  | `#ffffff`  | 6.84  | 3     | pass   |
| light | loss marks on sheet (chart bars, 3:1)       | `#9b2f52`  | `#ffffff`  | 7.19  | 3     | pass   |
| light | caution marks on sheet (drawdown line, 3:1) | `#b8860b`  | `#ffffff`  | 3.25  | 3     | pass   |
| light | graphite input border on sheet (UI, 3:1)    | `#5a6573`  | `#ffffff`  | 5.93  | 3     | pass   |
| dark  | ink on paper (body text)                    | `#e6ebf0`  | `#10161d`  | 15.16 | 4.5   | pass   |
| dark  | ink on sheet (tables, charts)               | `#e6ebf0`  | `#161e27`  | 14.01 | 4.5   | pass   |
| dark  | graphite on paper (margin notes)            | `#9aa6b4`  | `#10161d`  | 7.35  | 4.5   | pass   |
| dark  | graphite on sheet (axis labels)             | `#9aa6b4`  | `#161e27`  | 6.79  | 4.5   | pass   |
| dark  | gain on paper (links, positive numbers)     | `#6fa8d6`  | `#10161d`  | 7.14  | 4.5   | pass   |
| dark  | gain on sheet                               | `#6fa8d6`  | `#161e27`  | 6.60  | 4.5   | pass   |
| dark  | loss on paper (negative numbers, FAIL)      | `#d97a9a`  | `#10161d`  | 6.24  | 4.5   | pass   |
| dark  | loss on sheet                               | `#d97a9a`  | `#161e27`  | 5.76  | 4.5   | pass   |
| dark  | caution-ink on paper (warnings as text)     | `#d9b25c`  | `#10161d`  | 9.07  | 4.5   | pass   |
| dark  | caution-ink on sheet                        | `#d9b25c`  | `#161e27`  | 8.38  | 4.5   | pass   |
| dark  | on-accent on gain (primary button)          | `#10161d`  | `#6fa8d6`  | 7.14  | 4.5   | pass   |
| dark  | paper on ink (tooltip, selected segment)    | `#10161d`  | `#e6ebf0`  | 15.16 | 4.5   | pass   |
| dark  | ink on caution-wash (suspicious banner)     | `#e6ebf0`  | `#2d302d`  | 11.13 | 4.5   | pass   |
| dark  | loss on loss-wash (error summary links)     | `#d97a9a`  | `#2a2733`  | 5.01  | 4.5   | pass   |
| dark  | ink on gain wash 50% (heatmap max tint)     | `#e6ebf0`  | `#43637f`  | 5.26  | 4.5   | pass   |
| dark  | ink on loss wash 50% (heatmap max tint)     | `#e6ebf0`  | `#784c61`  | 5.83  | 4.5   | pass   |
| dark  | focus ring on paper (UI, 3:1)               | `#6fa8d6`  | `#10161d`  | 7.14  | 3     | pass   |
| dark  | focus ring on sheet (UI, 3:1)               | `#6fa8d6`  | `#161e27`  | 6.60  | 3     | pass   |
| dark  | gain marks on sheet (chart lines/bars, 3:1) | `#6fa8d6`  | `#161e27`  | 6.60  | 3     | pass   |
| dark  | loss marks on sheet (chart bars, 3:1)       | `#d97a9a`  | `#161e27`  | 5.76  | 3     | pass   |
| dark  | caution marks on sheet (drawdown line, 3:1) | `#d9b25c`  | `#161e27`  | 8.38  | 3     | pass   |
| dark  | graphite input border on sheet (UI, 3:1)    | `#9aa6b4`  | `#161e27`  | 6.79  | 3     | pass   |

### Decisions made while building

- **Margin notes align by subgrid.** The page is a two-track grid (reading column, margin
  column); each annotated block, each section and each metric-table row is a `subgrid` of it,
  so a note sits beside the exact figure or row it explains and drops beneath it under 1100 px.
  The metric table keeps table semantics through ARIA table roles.
- **NAV and drawdown are two panes, not two axes.** They share the time axis and a synced
  crosshair; drawdown is a caution-tinted area at 18 % (22 % in dark).
- **Cost components** are the one multi-series categorical encoding. Under the brief's
  single-accent palette they are one hue in three strengths (spread, impact, fees: the parts
  of cost per side) plus neutral graphite for borrow (a holding cost). The dataviz palette
  validator passes CVD separation (ΔE 15.4) and the normal-vision floor, and fails its chroma
  floor by design. Relief: legend, 1 px segment gaps, a hover breakdown and a data table.
- **Monthly returns heatmap is an HTML table** (its own text alternative). The tint is capped
  at a 50 % wash so ink text stays above 6.6:1 on every cell.
- **Motion.** The NAV line is drawn by progressively revealing data (≤ 600 ms, ease-out), then
  the gate markers fill left to right (70 ms stagger). It plays once per run per page session;
  `prefers-reduced-motion` turns both off.
- **Status is always in words.** Gate markers differ by shape too (filled with a tick, filled
  with a cross, dashed outline for skipped), and every status is written next to the marker.
