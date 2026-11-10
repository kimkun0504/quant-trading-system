# QTS research terminal — design system

Direction agreed 2026-11-10 (`REDESIGN.md`): an analytics / research terminal for testing
and comparing strategies. The unit of thought is the comparison, not the single run.

**History.** The first design (2026-11-05) was an "annotated lab notebook": a serif reading
column with margin notes beside every figure. Its engineering stack and its data palette
survive; the serif, the margin column and the single-run-centred layout were replaced by
this system in step 1 of the redesign.

## Subject, audience, job

- **Subject:** a research terminal over the `qts` pipeline: backtest runs, their diagnostics
  and the validation gates.
- **Audience:** the owner — an experienced engineer learning quant finance, who reads a
  dashboard faster than a notebook but still wants every number to explain itself.
- **Primary job:** answer "what changed when I changed X, and is it real or noise?" — select
  runs, see them side by side (NAV, folds, gates, config diff), with the trial count in view.
- **Secondary job:** teach, without taking space: definitions on hover, an "About" line that
  opens into the handbook's explanation and links to the section.

## Concept: one quiet surface, colour only where it is data

The chrome is near-monochrome and dense (Linear / Datadog density, not Bloomberg). Colour
appears in exactly two roles: **data** (gain, loss, caution, and a categorical palette for
runs under comparison) and **one interactive accent** (links, selection, focus). The memorable
element is the **selection tray and the comparison it opens**: pick runs anywhere, the tray
keeps them, and the compare panel lays them side by side in the tray's colours.

### Defaults checked and rejected

| First instinct                            | Why it is a default          | Chosen instead                                                                                 |
| ----------------------------------------- | ---------------------------- | ---------------------------------------------------------------------------------------------- |
| Near-black canvas with one acid accent    | the generic terminal look    | neutral cool surfaces in both themes; the dark canvas is `#13161B`, a colour, not tinted black |
| KPI tiles above the runs list             | SaaS dashboard default       | one line of facts in the title row ("8 runs in 5 families", "trials: 8") and a dense table     |
| Identical rounded cards with soft shadows | the SaaS card kit            | bordered panels, no shadows except floating layers (tooltip, search list, tray)                |
| Tracked ALL-CAPS eyebrow labels           | template chrome              | sentence case everywhere; group headers are sentence-case rows                                 |
| Monospace for numbers                     | tell of generated dashboards | Inter with tabular figures (`tnum`); `code` only for literal config values                     |
| Red / green for P&L                       | colour-blind unsafe          | blue (gain) vs wine (loss), sign always printed                                                |
| Gain blue as the interactive accent       | muddles data and chrome      | a separate violet accent for links, selection and focus; blue stays a data colour              |

## Tokens (`src/shared/styles/tokens.css`)

Every colour in the app comes from this file; SVG uses `var(--color-*)`, canvas (uPlot)
reads the resolved values through `useChartTokens()`. The first-generation names
(`paper`, `sheet`, `ink`, `graphite`, `rule`, `step-*`) remain as aliases.

### Colour

| Token                 | Light     | Dark      | Use                                                       |
| --------------------- | --------- | --------- | --------------------------------------------------------- |
| `canvas`              | `#F4F5F7` | `#13161B` | app background                                            |
| `surface`             | `#FFFFFF` | `#1A1E25` | panels, tables, charts, tray                              |
| `surface-2`           | `#EEF0F3` | `#232831` | hover rows, chips, kbd, family group headers              |
| `fg`                  | `#1A1F27` | `#E6E9EF` | text, axes                                                |
| `fg-muted`            | `#5B6574` | `#9AA4B3` | secondary text, captions, table headers                   |
| `border`              | `#E1E4EA` | `#2C323C` | hairlines, gridlines                                      |
| `border-strong`       | `#8690A0` | `#6E7A8B` | input and secondary-button borders (3:1)                  |
| `accent`              | `#4F46C9` | `#A9A0FF` | links, selection, focus ring, primary button              |
| `accent-wash`         | mix 9 %   | mix 14 %  | selected rows and chips                                   |
| `gain`                | `#1F5F8B` | `#6FA8D6` | positive return, long book, pass                          |
| `loss`                | `#9B2F52` | `#D97A9A` | negative return, short book, fail                         |
| `caution`             | `#B8860B` | `#D9B25C` | skipped gates, drawdown fill, thresholds (marks only)     |
| `caution-ink`         | `#7D5B00` | `#D9B25C` | caution as text (`#B8860B` fails AA on light surfaces)    |
| `series-1 … series-8` | see file  | see file  | categorical palette for compared runs; 1 = gain, 2 = loss |

Washes (`wash`, `caution-wash`, `loss-wash`, `accent-wash`) are `color-mix()` of the accent
into `surface`. Light is the default; dark applies with `[data-theme='dark']` or, with no
stored preference, `prefers-color-scheme: dark`. The theme toggle in the top bar stores an
explicit preference in `localStorage` (`qts-theme`); `index.html` applies it before first paint.

### Type

One family: **Inter Variable** (`@fontsource-variable/inter`), `font-feature-settings: "tnum"`
so every column of figures aligns. Scale (px): 11 `xs` · 12 `sm` · 13 `md` · 14 `base`
(body) · 16 `lg` · 18 `xl` · 22 `2xl` (page title) · 28 `3xl`. Line height 1.45 for UI,
1.5 for explanatory text; headings 1.2 with −0.012 em tracking. Line length ≤ 68 ch.

### Space, radii, elevation, focus, motion

- Space: 8-px grid — `space-1..8` = 4 / 8 / 12 / 16 / 24 / 32 / 48 / 64 px. Table rows are
  32 px tall with 12-px cell padding.
- Radii: `sm` 4 px (controls, cells), `md` 6 px (panels, tooltips), `lg` 10 px, `pill` for
  chips and badges.
- Elevation: borders carry structure; `shadow-1` only under floating layers (tooltips, the
  search list), `shadow-2` under the sticky tray.
- Focus: `2px solid accent`, 2-px offset, on every focusable element (`:focus-visible`).
- Motion: 120 ms for hover and disclosure; the run page's NAV draw (600 ms) and gate fill are
  the one orchestrated moment; `prefers-reduced-motion` disables all of it.

### Primitives (`src/shared/ui`)

`Button` (primary / secondary / quiet, md / sm) · `IconButton` (label in a tooltip) · `Badge`
(neutral / pass / fail / skipped / caution / accent; word plus dot) · `Tooltip` (hover + focus,
Escape dismisses, content doubles as the trigger's description) · `Definition` (a term with a
tooltip) · `Panel` (bordered surface with an optional title row) · `Table`, `Th`, `Td`
(dense rows, `numeric` right-aligns) and `SortableTable` · `TabNav` (where a page still has
tabs) · `Kbd` · `About` (a closed `<details>` line that opens into the explanation and the
handbook link; replaces margin notes) · `EmptyState`, `ErrorState`, `Skeleton`.

## Layout

```
┌────────┬──────────────────────────────────────────────────────┐
│ QTS    │ [⌕ search runs            /]                      ☾  │  top bar 48 px
│        ├──────────────────────────────────────────────────────┤
│ Exper. │ Experiments        8 runs in 5 families  trials: 8   │
│ Signal │ ▸ About this table                                   │
│ Portf. │ Comparing 3 runs  (only when ≥ 2 selected)           │
│ Costs  │ ┌ headline table ───────────────────────────────────┐ │
│ Build. │ ┌ NAV overlay (log toggle) ─────────────────────────┐ │
│        │ ┌ fold Sharpe bars ──────┐ ┌ gates grid ───────────┐ │
│ Ledger │ ┌ config diff (keys that differ) ───────────────────┐ │
│ Models │ ┌ runs by family ───────────────────────────────────┐ │
│ Handb. │ │ ▾ Composite baseline  4 runs   varies: horizon…   │ │
│        │ │ ☐ name/date  gates  nSh  gSh  DSR  CAGR  DD  …    │ │
├────────┴──────────────────────────────────────────────────────┤
│ 2 of 8 selected  [■ run ×] [■ run ×]        Clear  Compare c  │  tray
└───────────────────────────────────────────────────────────────┘
```

- Everything left-aligned; numbers right-aligned in their columns. Sidebar 216 px, content up
  to 1600 px. Below 900 px the sidebar becomes an icon row and the top bar stacks under it.
- Keyboard: `/` focuses search, `c` opens Compare; checkboxes, sort buttons, family toggles,
  disclosure lines and tooltips are all reachable by Tab.

## Principles

1. **The comparison is the unit.** Selection is global (tray + URL + storage); every table of
   runs can feed it; the compare panel is one component used inline and as a page.
2. **Every number explains itself, on request.** Hover a label for the definition; open the
   About line for the reading; follow the link to the handbook. Nothing explanatory takes
   space by default.
3. **Honesty over flattery.** Trials are in the header; failing gates are plain words;
   suspicious Sharpes are flagged in the row.
4. **Density without clutter.** Tables over cards; one hairline between rows; panels only
   where they group one thing.
5. **Accessible by default.** AA contrast in both themes (table below), visible focus,
   charts with text alternatives, no colour-only signals.

## Contrast check (`npm run contrast`)

`scripts/contrast.mjs` reads `tokens.css`, mixes the `color-mix()` washes the way the browser
does, and exits non-zero if any pair drops below its minimum (4.5:1 text, 3:1 marks and UI).

| Theme | Pair                                          | Foreground | Background | Ratio | Needs | Result |
| ----- | --------------------------------------------- | ---------- | ---------- | ----- | ----- | ------ |
| light | fg on canvas (body text)                      | `#1a1f27`  | `#f4f5f7`  | 15.17 | 4.5   | pass   |
| light | fg on surface (tables, panels, charts)        | `#1a1f27`  | `#ffffff`  | 16.55 | 4.5   | pass   |
| light | fg on surface-2 (chips, hover rows, kbd)      | `#1a1f27`  | `#eef0f3`  | 14.49 | 4.5   | pass   |
| light | fg-muted on canvas (secondary text)           | `#5b6574`  | `#f4f5f7`  | 5.41  | 4.5   | pass   |
| light | fg-muted on surface (captions, axis labels)   | `#5b6574`  | `#ffffff`  | 5.90  | 4.5   | pass   |
| light | fg-muted on surface-2 (table header)          | `#5b6574`  | `#eef0f3`  | 5.17  | 4.5   | pass   |
| light | accent on canvas (links)                      | `#4f46c9`  | `#f4f5f7`  | 6.31  | 4.5   | pass   |
| light | accent on surface (links, selected text)      | `#4f46c9`  | `#ffffff`  | 6.88  | 4.5   | pass   |
| light | fg on accent-wash (selected rows, chips)      | `#1a1f27`  | `#efeefa`  | 14.40 | 4.5   | pass   |
| light | accent on accent-wash (chip text)             | `#4f46c9`  | `#efeefa`  | 5.99  | 4.5   | pass   |
| light | on-accent on accent (primary button)          | `#ffffff`  | `#4f46c9`  | 6.88  | 4.5   | pass   |
| light | gain on canvas (positive numbers)             | `#1f5f8b`  | `#f4f5f7`  | 6.27  | 4.5   | pass   |
| light | gain on surface                               | `#1f5f8b`  | `#ffffff`  | 6.84  | 4.5   | pass   |
| light | loss on canvas (negative numbers, FAIL)       | `#9b2f52`  | `#f4f5f7`  | 6.59  | 4.5   | pass   |
| light | loss on surface                               | `#9b2f52`  | `#ffffff`  | 7.19  | 4.5   | pass   |
| light | caution-ink on canvas (warnings as text)      | `#7d5b00`  | `#f4f5f7`  | 5.72  | 4.5   | pass   |
| light | caution-ink on surface                        | `#7d5b00`  | `#ffffff`  | 6.23  | 4.5   | pass   |
| light | canvas on fg (tooltip)                        | `#f4f5f7`  | `#1a1f27`  | 15.17 | 4.5   | pass   |
| light | fg on caution-wash (suspicious banner)        | `#1a1f27`  | `#f5eedd`  | 14.31 | 4.5   | pass   |
| light | loss on loss-wash (error summary links)       | `#9b2f52`  | `#f7eef1`  | 6.32  | 4.5   | pass   |
| light | fg on gain wash 50% (heatmap max tint)        | `#1a1f27`  | `#8fafc5`  | 7.17  | 4.5   | pass   |
| light | fg on loss wash 50% (heatmap max tint)        | `#1a1f27`  | `#cd97a9`  | 6.77  | 4.5   | pass   |
| light | focus ring (accent) on canvas (UI, 3:1)       | `#4f46c9`  | `#f4f5f7`  | 6.31  | 3     | pass   |
| light | focus ring (accent) on surface (UI, 3:1)      | `#4f46c9`  | `#ffffff`  | 6.88  | 3     | pass   |
| light | border-strong on surface (inputs, 3:1)        | `#8690a0`  | `#ffffff`  | 3.23  | 3     | pass   |
| light | gain marks on surface (chart lines/bars, 3:1) | `#1f5f8b`  | `#ffffff`  | 6.84  | 3     | pass   |
| light | loss marks on surface (chart bars, 3:1)       | `#9b2f52`  | `#ffffff`  | 7.19  | 3     | pass   |
| light | caution marks on surface (drawdown line, 3:1) | `#b8860b`  | `#ffffff`  | 3.25  | 3     | pass   |
| light | fg-muted input border on surface (UI, 3:1)    | `#5b6574`  | `#ffffff`  | 5.90  | 3     | pass   |
| dark  | fg on canvas (body text)                      | `#e6e9ef`  | `#13161b`  | 14.91 | 4.5   | pass   |
| dark  | fg on surface (tables, panels, charts)        | `#e6e9ef`  | `#1a1e25`  | 13.74 | 4.5   | pass   |
| dark  | fg on surface-2 (chips, hover rows, kbd)      | `#e6e9ef`  | `#232831`  | 12.17 | 4.5   | pass   |
| dark  | fg-muted on canvas (secondary text)           | `#9aa4b3`  | `#13161b`  | 7.20  | 4.5   | pass   |
| dark  | fg-muted on surface (captions, axis labels)   | `#9aa4b3`  | `#1a1e25`  | 6.63  | 4.5   | pass   |
| dark  | fg-muted on surface-2 (table header)          | `#9aa4b3`  | `#232831`  | 5.87  | 4.5   | pass   |
| dark  | accent on canvas (links)                      | `#a9a0ff`  | `#13161b`  | 7.91  | 4.5   | pass   |
| dark  | accent on surface (links, selected text)      | `#a9a0ff`  | `#1a1e25`  | 7.29  | 4.5   | pass   |
| dark  | fg on accent-wash (selected rows, chips)      | `#e6e9ef`  | `#2e3044`  | 10.64 | 4.5   | pass   |
| dark  | accent on accent-wash (chip text)             | `#a9a0ff`  | `#2e3044`  | 5.65  | 4.5   | pass   |
| dark  | on-accent on accent (primary button)          | `#13161b`  | `#a9a0ff`  | 7.91  | 4.5   | pass   |
| dark  | gain on canvas (positive numbers)             | `#6fa8d6`  | `#13161b`  | 7.12  | 4.5   | pass   |
| dark  | gain on surface                               | `#6fa8d6`  | `#1a1e25`  | 6.57  | 4.5   | pass   |
| dark  | loss on canvas (negative numbers, FAIL)       | `#d97a9a`  | `#13161b`  | 6.22  | 4.5   | pass   |
| dark  | loss on surface                               | `#d97a9a`  | `#1a1e25`  | 5.73  | 4.5   | pass   |
| dark  | caution-ink on canvas (warnings as text)      | `#d9b25c`  | `#13161b`  | 9.04  | 4.5   | pass   |
| dark  | caution-ink on surface                        | `#d9b25c`  | `#1a1e25`  | 8.34  | 4.5   | pass   |
| dark  | canvas on fg (tooltip)                        | `#13161b`  | `#e6e9ef`  | 14.91 | 4.5   | pass   |
| dark  | fg on caution-wash (suspicious banner)        | `#e6e9ef`  | `#31302c`  | 10.86 | 4.5   | pass   |
| dark  | loss on loss-wash (error summary links)       | `#d97a9a`  | `#2d2731`  | 4.98  | 4.5   | pass   |
| dark  | fg on gain wash 50% (heatmap max tint)        | `#e6e9ef`  | `#45637e`  | 5.17  | 4.5   | pass   |
| dark  | fg on loss wash 50% (heatmap max tint)        | `#e6e9ef`  | `#7a4c60`  | 5.70  | 4.5   | pass   |
| dark  | focus ring (accent) on canvas (UI, 3:1)       | `#a9a0ff`  | `#13161b`  | 7.91  | 3     | pass   |
| dark  | focus ring (accent) on surface (UI, 3:1)      | `#a9a0ff`  | `#1a1e25`  | 7.29  | 3     | pass   |
| dark  | border-strong on surface (inputs, 3:1)        | `#6e7a8b`  | `#1a1e25`  | 3.84  | 3     | pass   |
| dark  | gain marks on surface (chart lines/bars, 3:1) | `#6fa8d6`  | `#1a1e25`  | 6.57  | 3     | pass   |
| dark  | loss marks on surface (chart bars, 3:1)       | `#d97a9a`  | `#1a1e25`  | 5.73  | 3     | pass   |
| dark  | caution marks on surface (drawdown line, 3:1) | `#d9b25c`  | `#1a1e25`  | 8.34  | 3     | pass   |
| dark  | fg-muted input border on surface (UI, 3:1)    | `#9aa4b3`  | `#1a1e25`  | 6.63  | 3     | pass   |
