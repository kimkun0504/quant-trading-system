// WCAG 2.x contrast of every text/UI token pair, both themes. Reads tokens.css so the
// numbers always describe the shipped values. Exit code 1 if a required pair fails.
// Usage: node scripts/contrast.mjs [--markdown]
import { readFileSync } from 'node:fs';

const css = readFileSync(new URL('../src/shared/styles/tokens.css', import.meta.url), 'utf8');

function block(selector) {
  const i = css.indexOf(selector);
  const start = css.indexOf('{', i);
  let depth = 0;
  for (let j = start; j < css.length; j++) {
    if (css[j] === '{') depth++;
    if (css[j] === '}' && --depth === 0) return css.slice(start + 1, j);
  }
  throw new Error(`no block ${selector}`);
}

function tokens(body) {
  const out = {};
  for (const m of body.matchAll(/--color-([a-z0-9-]+):\s*(#[0-9a-fA-F]{6})/g))
    out[m[1]] = m[2].toLowerCase();
  return out;
}

const light = tokens(block(':root {'));
const dark = { ...light, ...tokens(block(":root[data-theme='dark']")) };

const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
const toHex = (rgb) => '#' + rgb.map((v) => Math.round(v).toString(16).padStart(2, '0')).join('');
/** color-mix(in srgb, a p%, b) */
const mix = (a, p, b) => toHex(hex(a).map((v, i) => v * p + hex(b)[i] * (1 - p)));
const lum = (h) => {
  const [r, g, b] = hex(h).map((v) => {
    const c = v / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
};
const ratio = (a, b) => {
  const [x, y] = [lum(a), lum(b)].sort((m, n) => n - m);
  return (x + 0.05) / (y + 0.05);
};

// [label, fg, bg, minimum] — 4.5 for text, 3 for large text / UI components and focus.
function pairs(t, wash) {
  return [
    ['fg on canvas (body text)', t.fg, t.canvas, 4.5],
    ['fg on surface (tables, panels, charts)', t.fg, t.surface, 4.5],
    ['fg on surface-2 (chips, hover rows, kbd)', t.fg, t['surface-2'], 4.5],
    ['fg-muted on canvas (secondary text)', t['fg-muted'], t.canvas, 4.5],
    ['fg-muted on surface (captions, axis labels)', t['fg-muted'], t.surface, 4.5],
    ['fg-muted on surface-2 (table header)', t['fg-muted'], t['surface-2'], 4.5],
    ['accent on canvas (links)', t.accent, t.canvas, 4.5],
    ['accent on surface (links, selected text)', t.accent, t.surface, 4.5],
    ['fg on accent-wash (selected rows, chips)', t.fg, wash.accent, 4.5],
    ['accent on accent-wash (chip text)', t.accent, wash.accent, 4.5],
    ['on-accent on accent (primary button)', t['on-accent'], t.accent, 4.5],
    ['gain on canvas (positive numbers)', t.gain, t.canvas, 4.5],
    ['gain on surface', t.gain, t.surface, 4.5],
    ['loss on canvas (negative numbers, FAIL)', t.loss, t.canvas, 4.5],
    ['loss on surface', t.loss, t.surface, 4.5],
    ['caution-ink on canvas (warnings as text)', t['caution-ink'], t.canvas, 4.5],
    ['caution-ink on surface', t['caution-ink'], t.surface, 4.5],
    ['canvas on fg (tooltip)', t.canvas, t.fg, 4.5],
    ['fg on caution-wash (suspicious banner)', t.fg, wash.caution, 4.5],
    ['loss on loss-wash (error summary links)', t.loss, wash.loss, 4.5],
    ['fg on gain wash 50% (heatmap max tint)', t.fg, mix(t.gain, 0.5, t.surface), 4.5],
    ['fg on loss wash 50% (heatmap max tint)', t.fg, mix(t.loss, 0.5, t.surface), 4.5],
    ['focus ring (accent) on canvas (UI, 3:1)', t.accent, t.canvas, 3],
    ['focus ring (accent) on surface (UI, 3:1)', t.accent, t.surface, 3],
    ['border-strong on surface (inputs, 3:1)', t['border-strong'], t.surface, 3],
    ['gain marks on surface (chart lines/bars, 3:1)', t.gain, t.surface, 3],
    ['loss marks on surface (chart bars, 3:1)', t.loss, t.surface, 3],
    ['caution marks on surface (drawdown line, 3:1)', t.caution, t.surface, 3],
    ['fg-muted input border on surface (UI, 3:1)', t['fg-muted'], t.surface, 3],
    // run builder and queue
    ['accent on surface-2 (changed count in a section header)', t.accent, t['surface-2'], 4.5],
    ['loss on surface-2 (errors-to-fix count in a section header)', t.loss, t['surface-2'], 4.5],
    ['fg-muted on accent-wash (current variant chip)', t['fg-muted'], wash.accent, 4.5],
    ['fg-muted on caution-wash (discard-edits prompt)', t['fg-muted'], wash.caution, 4.5],
    ['accent marks on surface-2 (queue progress, 3:1)', t.accent, t['surface-2'], 3],
    ['caution marks on surface (launch-slip caution mark, 3:1)', t.caution, t.surface, 3],
  ];
}

const wash = (t, isDark) => ({
  caution: mix(t.caution, isDark ? 0.12 : 0.14, t.surface),
  loss: mix(t.loss, isDark ? 0.1 : 0.08, t.surface),
  accent: mix(t.accent, isDark ? 0.14 : 0.09, t.surface),
});

let failed = 0;
const rows = [];
for (const [name, t, isDark] of [
  ['light', light, false],
  ['dark', dark, true],
]) {
  for (const [label, fg, bg, min] of pairs(t, wash(t, isDark))) {
    const r = ratio(fg, bg);
    const ok = r >= min;
    if (!ok) failed++;
    rows.push({
      theme: name,
      pair: label,
      fg,
      bg,
      ratio: r.toFixed(2),
      min,
      result: ok ? 'pass' : 'FAIL',
    });
  }
}

if (process.argv.includes('--markdown')) {
  console.log('| Theme | Pair | Foreground | Background | Ratio | Needs | Result |');
  console.log('|---|---|---|---|---|---|---|');
  for (const r of rows)
    console.log(
      `| ${r.theme} | ${r.pair} | \`${r.fg}\` | \`${r.bg}\` | ${r.ratio} | ${r.min} | ${r.result} |`,
    );
} else {
  console.table(rows);
}
if (failed) {
  console.error(`${failed} pair(s) below their minimum`);
  process.exit(1);
}
