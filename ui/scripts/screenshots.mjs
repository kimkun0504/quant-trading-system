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
const SEL = `sel=${RUN},20261106T083015Z_5e2c9f10,20261106T142207Z_9a7b3d44`;
const pages = [
  ['experiments', '/'],
  ['experiments-compare', `/?${SEL}`],
  ['compare', `/compare?${SEL}`],
  ['signal-lab', '/signal'],
  ['dossier', `/runs/${RUN}`],
  ['dossier-portfolio', '/runs/20261106T142207Z_9a7b3d44#portfolio'],
  ['dossier-config', `/runs/${RUN}#config`],
  ['dossier-suspicious', '/runs/20261105T102233Z_c07a55e2'],
  ['run-builder', '/builder'],
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
    // the selection is persisted; start each context clean
    await ctx.clearCookies();
    for (const [name, path] of pages) {
      await page.goto(base + path, { waitUntil: 'networkidle' });
      if (!path.includes('sel='))
        await page.evaluate(() => localStorage.removeItem('qts-selection'));
      if (!path.includes('sel=') && (await page.url()).includes('sel='))
        await page.goto(base + path, { waitUntil: 'networkidle' });
      await page.waitForTimeout(500);
      const file = `${out}/${name}-${theme}-${width}.png`;
      // a page opened on a section keeps that viewport; the rest are captured whole
      await page.screenshot({ path: file, fullPage: !path.includes('#') });
      console.log(file);
    }
    await ctx.close();
  }
}
await browser.close();
