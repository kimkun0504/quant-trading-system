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
const LONG = 'sel=20261101T061500Z_e4b17c2d';
const LAB3 = `sel=${RUN},20261106T083015Z_5e2c9f10,20261105T091402Z_3be1a7c0`;
const pages = [
  ['experiments', '/'],
  ['experiments-compare', `/?${SEL}`],
  ['compare', `/compare?${SEL}`],
  ['signal-lab', `/signal?${LONG}`],
  ['signal-lab-3', `/signal?${LAB3}`],
  ['signal-lab-picker', '/signal'],
  ['portfolio-lab', `/portfolio?${LONG}`],
  ['portfolio-lab-3', `/portfolio?${LAB3}`],
  ['costs-lab', `/costs?${LONG}`],
  ['costs-lab-3', `/costs?${LAB3}`],
  ['dossier', `/runs/${RUN}`],
  ['dossier-portfolio', '/runs/20261106T142207Z_9a7b3d44#portfolio'],
  ['dossier-config', `/runs/${RUN}#config`],
  ['dossier-suspicious', '/runs/20261105T102233Z_c07a55e2'],
  ['run-builder', '/builder'],
  // cloned from the baseline with two edits (deep link), the diff in the slip
  [
    'run-builder-diff',
    `/builder?from=${RUN}&set=alpha.sample_weight_halflife_years=8&set=portfolio.trade_aversion=0.5`,
  ],
  // two variants (half-life 0 vs 8) queued: one running, one waiting
  [
    'run-builder-queue',
    `/builder?from=${RUN}&set=alpha.sample_weight_halflife_years=8`,
    async (page) => {
      await page.getByRole('button', { name: 'Add variant' }).click();
      await page
        .getByLabel('Training recency half-life (years), 0 = equal weights', { exact: true })
        .fill('0');
      await page.getByRole('button', { name: 'Queue all 2 variants' }).click();
      await page.locator('[data-status="running"]').waitFor();
      await page.locator('[data-status="queued"]').waitFor();
      await page.waitForTimeout(10000); // into the backtest stage of the first job
      await page.locator('#queue').scrollIntoViewIfNeeded();
    },
    { viewport: true },
  ],
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
    // a mock run takes 24 s here, so the queue can be captured mid-run
    await ctx.addInitScript(() => localStorage.setItem('qts-mock-job-ms', '24000'));
    const page = await ctx.newPage();
    page.on('pageerror', (e) => console.error(`[${theme} ${width}] pageerror`, e.message));
    page.on('console', (m) => {
      if (m.type() === 'error') console.error(`[${theme} ${width}] console`, m.text());
    });
    // the selection is persisted; start each context clean
    await ctx.clearCookies();
    for (const [name, path, prepare, opts] of pages) {
      await page.goto(base + path, { waitUntil: 'networkidle' });
      if (!path.includes('sel='))
        await page.evaluate(() => localStorage.removeItem('qts-selection'));
      if (!path.includes('sel=') && (await page.url()).includes('sel='))
        await page.goto(base + path, { waitUntil: 'networkidle' });
      await page.waitForTimeout(500);
      if (prepare) await prepare(page);
      const file = `${out}/${name}-${theme}-${width}.png`;
      // a page opened on a section keeps that viewport; the rest are captured whole
      await page.screenshot({ path: file, fullPage: !path.includes('#') && !opts?.viewport });
      if (prepare) await page.reload({ waitUntil: 'networkidle' }); // drop the queued mock jobs
      console.log(file);
    }
    await ctx.close();
  }
}
await browser.close();
