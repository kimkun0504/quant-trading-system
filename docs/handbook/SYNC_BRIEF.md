# Handbook sync brief (2026-11-05)

The handbook pages 01–11 were drafted at commit 63ce32b. Since then: the refactor wave
(commit ab8cea0), the lag investigation and default execution delay (6572752), the API
and UI (ddf9317, 4d3117f), and the audit-2 fix wave (uncommitted working tree at the time
of writing; treat the working tree as the source of truth). Every writer:

1. Reads `git log --stat 63ce32b..HEAD`, `git diff 63ce32b -- <the modules your pages cover>`,
   `docs/notes/*.md` (findings, calibration, golden changes, lag investigation) and
   `docs/audit/second_pass_*.md`, then re-reads the modules your pages describe.
2. Updates the repo page(s) in `docs/handbook/` in place, following `docs/handbook/STYLE.md`.
   Keep the existing structure and voice; change what is now wrong; add sections for new
   mechanisms; remove statements about things that no longer exist. Every default and
   formula must match the current code (`qts/core/config.py`, `configs/default.yaml`).
3. Adds a short "What changed (Nov 2026)" callout near the top of each updated page listing
   the material changes in 3–8 bullets.
4. Pushes the updated page to Notion: `notion-update-page` with command `replace_content`
   using the first ≤20 KB chunk (split at `## ` boundaries, never inside `$$`, code fences,
   callouts, toggles or tables), then `insert_content` at end for the remaining chunks.
   The `# Title` line is not part of the body. Verify by fetching (heading count and `$$`
   count match the source). Do not touch any other Notion page.

Notion page ids:
- hub 40f8262e-79e0-8190-a02c-fabeab723272 (do not edit — the orchestrator updates it)
- 01 Start here 40f8262e-79e0-8136-a6f0-f012fa3ad279
- 02 Architecture & data flow 40f8262e-79e0-8120-9aa5-dce482de359c
- 03 Data & universe 40f8262e-79e0-81c2-af1e-e852238618de
- 04 Alpha model 40f8262e-79e0-8106-b0ee-edbc2347444e
- 05 Risk model 40f8262e-79e0-8170-90b7-ef82499a28b4
- 06 Transaction cost model 40f8262e-79e0-8124-bee9-d3f4d8366924
- 07 Portfolio construction 40f8262e-79e0-816e-bfcf-de09cf00838f
- 08 Execution 40f8262e-79e0-8116-86a7-f228aaed53ac
- 09 Backtesting 40f8262e-79e0-81b6-82e7-f49e2520c2b7
- 10 Validation & statistics 40f8262e-79e0-81a3-9129-e966f8fa7c2b
- 11 Glossary 40f8262e-79e0-81c1-a4b7-d4cf357f3d33
- 13 Audit history & known limitations 40f8262e-79e0-81e7-aefa-eddf31cf8570
- 12 Operating guide 40f8262e-79e0-810a-92de-cd8568ffee80

Load the Notion tools with ToolSearch: "select:mcp__Notion__notion-update-page,mcp__Notion__notion-create-pages,mcp__Notion__notion-fetch".
