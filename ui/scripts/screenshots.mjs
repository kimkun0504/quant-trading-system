// Usage: node scripts/screenshots.mjs [baseUrl] [--only=name] [--out=dir]
// Captures every main page in light and dark at 1440 and 390 px (deviceScaleFactor 1).
import { mkdirSync } from 'node:fs';
import { chromium } from '@playwright/test';
import { chromiumPath } from './chromium.mjs';

const args = process.argv.slice(2);
const base = args.find((a) => !a.startsWith('--')) ?? 'http://localhost:5173';
const only = args.find((a) => a.startsWith('--only='))?.slice(7);
const out = args.find((a) => a.startsWith('--out='))?.slice(6) ?? 'docs/screenshots';
const widths = (args.find((a) => a.startsWith('--widths='))?.slice(9) ?? '1440,390')
  .split(',')
  .map(Number);
const themes = (args.find((a) => a.startsWith('--themes='))?.slice(9) ?? 'light,dark').split(',');
mkdirSync(out, { recursive: true });

const RUN = '20261104T175159Z_944dd5b4';
const pages = [
  ['runs', '/runs'],
  ['run-performance', `/runs/${RUN}`],
  ['run-risk', `/runs/${RUN}/risk`],
  ['run-alpha', `/runs/${RUN}/alpha`],
  ['run-costs', `/runs/${RUN}/costs`],
  ['run-execution', `/runs/${RUN}/execution`],
  ['run-validation', `/runs/${RUN}/validation`],
  ['run-suspicious', '/runs/20261105T102233Z_c07a55e2'],
  ['new-run', '/runs/new'],
  ['ledger', '/ledger'],
  ['models', '/models'],
].filter(([n]) => !only || only.split(',').includes(n));

const browser = await chromium.launch({ executablePath: chromiumPath() });
for (const theme of themes) {
  for (const width of widths) {
    const ctx = await browser.newContext({
      viewport: { width, height: 900 },
      deviceScaleFactor: 1,
      colorScheme: theme,
      reducedMotion: 'reduce',
    });
    const page = await ctx.newPage();
    page.on('pageerror', (e) => console.error(`[${theme} ${width}] pageerror`, e.message));
    page.on('console', (m) => {
      if (m.type() === 'error') console.error(`[${theme} ${width}] console`, m.text());
    });
    for (const [name, path] of pages) {
      await page.goto(base + path, { waitUntil: 'networkidle' });
      await page.waitForTimeout(500);
      const file = `${out}/${name}-${theme}-${width}.png`;
      await page.screenshot({ path: file, fullPage: true });
      console.log(file);
    }
    await ctx.close();
  }
}
await browser.close();
