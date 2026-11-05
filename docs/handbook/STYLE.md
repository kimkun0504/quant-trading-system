# Handbook style guide

These pages are published to Notion as the "Quant Trading System — Handbook" and
mirrored here. Every writer follows this guide.

## Audience
A software engineer with a CS degree: comfortable with linear algebra, probability,
statistics at an undergraduate level, Python, and data structures. **Not** a quant:
has never built a factor model, doesn't know what an IC or a Barra model is, may not
know how a closing auction works. Explain finance vocabulary on first use; never
assume it. Prefer intuition first, then the formula, then the code pointer.

## Format (Notion-flavoured Markdown — must render in Notion)
- File = one page. First line is `# Title` (it becomes the page title and is removed
  from the body on upload). Then a one-paragraph summary in a callout:
  `<callout icon="🧭" color="gray_bg">` … `</callout>`.
- Headings: `##` for sections, `###` for subsections. No `####` deeper than needed.
- Inline math `$\sigma_i$`. Display math: a line containing only `$$`, the equation
  on the next line(s), then a line containing only `$$`. Never `$$...$$` on one line,
  never `\[ \]`. Use standard LaTeX (`\frac`, `\sum`, `\mathbb{E}`, `\top`).
- Callouts for intuition / warnings / "why not X?":
  `<callout icon="💡" color="blue_bg">` (intuition), `<callout icon="⚠️" color="yellow_bg">`
  (pitfall), `<callout icon="🤔" color="purple_bg">` (why this and not the alternative).
  Body text follows the opening tag on the same line or next line; no blank line
  directly after the opening tag or before the closing tag.
- Tables: GFM pipe tables. Escape `|` inside cells as `\|`.
- Code: fenced with a language tag; keep snippets short (≤ 15 lines) and real
  (taken from the repo). Reference code as `qts/risk/covariance.py::ewma_cov`.
- No raw HTML other than the Notion tags above (`<callout>`, `<b>`, `<i>`, `<u>`,
  `<span color>`, `<toggle heading="h3">`). Use `<toggle heading="h3">Derivation</toggle>`
  style toggles for long derivations so the page stays scannable:
  ```
  <toggle heading="h3">
  Derivation of the QIS shrinkage

  …body…
  </toggle>
  ```
- Cross-page references: write `→ see **Risk model › Shrinkage**` (links are added on
  upload).
- External references: `[Author (year), Title](URL)` at the end of each section under
  "References". Only cite sources you can name precisely (paper title, authors, year);
  the build plan `docs/BUILD_PLAN.md` and the research notes contain URLs you may reuse.

## Section template for every model component / algorithm
1. **What it is** — one paragraph, plain English, with an everyday analogy if useful.
2. **Why we need it here** — the problem it solves in this system.
3. **How it works** — step by step; the math with every symbol defined in a
   "where:" list right below the equation; units (daily vs annual, fraction vs bps).
4. **Why this choice** — alternatives considered and why rejected, citing evidence
   (from `docs/BUILD_PLAN.md`, `docs/REVIEW_NOTES.md`, `docs/audit/*`).
5. **In the code** — module/function, key parameters with defaults from
   `qts/core/config.py`, and what to change to tune it.
6. **Pitfalls & checks** — what goes wrong (look-ahead, overfitting, unit errors) and
   how the system guards against it (tests, gates).
7. **Worked example** (where it helps) — small numbers a reader can verify by hand.

## Accuracy rules
- Read the actual code before describing it. Parameters and defaults must match
  `qts/core/config.py` and the module. If code and plan disagree, describe the code
  and note the plan difference.
- Numbers from research (Sharpe ratios, IC values) must carry their source and
  whether they are in-sample, out-of-sample, or live.
- Don't invent features the code doesn't have; list planned items under "Roadmap".
- Length: thorough. A model page is typically 4,000–8,000 words. Clarity beats brevity,
  but no filler.
