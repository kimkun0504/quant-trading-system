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
  for (const m of body.matchAll(/--color-([a-z-]+):\s*(#[0-9a-fA-F]{6})/g))
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
    ['ink on paper (body text)', t.ink, t.paper, 4.5],
    ['ink on sheet (tables, charts)', t.ink, t.sheet, 4.5],
    ['graphite on paper (margin notes)', t.graphite, t.paper, 4.5],
    ['graphite on sheet (axis labels)', t.graphite, t.sheet, 4.5],
    ['gain on paper (links, positive numbers)', t.gain, t.paper, 4.5],
    ['gain on sheet', t.gain, t.sheet, 4.5],
    ['loss on paper (negative numbers, FAIL)', t.loss, t.paper, 4.5],
    ['loss on sheet', t.loss, t.sheet, 4.5],
    ['caution-ink on paper (warnings as text)', t['caution-ink'], t.paper, 4.5],
    ['caution-ink on sheet', t['caution-ink'], t.sheet, 4.5],
    ['on-accent on gain (primary button)', t['on-accent'], t.gain, 4.5],
    ['paper on ink (tooltip, selected segment)', t.paper, t.ink, 4.5],
    ['ink on caution-wash (suspicious banner)', t.ink, wash.caution, 4.5],
    ['loss on loss-wash (error summary links)', t.loss, wash.loss, 4.5],
    ['ink on gain wash 50% (heatmap max tint)', t.ink, mix(t.gain, 0.5, t.sheet), 4.5],
    ['ink on loss wash 50% (heatmap max tint)', t.ink, mix(t.loss, 0.5, t.sheet), 4.5],
    ['focus ring on paper (UI, 3:1)', t.focus, t.paper, 3],
    ['focus ring on sheet (UI, 3:1)', t.focus, t.sheet, 3],
    ['gain marks on sheet (chart lines/bars, 3:1)', t.gain, t.sheet, 3],
    ['loss marks on sheet (chart bars, 3:1)', t.loss, t.sheet, 3],
    ['caution marks on sheet (drawdown line, 3:1)', t.caution, t.sheet, 3],
    ['graphite input border on sheet (UI, 3:1)', t.graphite, t.sheet, 3],
  ];
}

const wash = (t, isDark) => ({
  caution: mix(t.caution, isDark ? 0.12 : 0.14, t.sheet),
  loss: mix(t.loss, isDark ? 0.1 : 0.08, t.sheet),
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
