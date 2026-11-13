import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';

const DEMO = '20261104T175159Z_944dd5b4';
const SEL = `sel=${DEMO},20261106T083015Z_5e2c9f10,20261106T142207Z_9a7b3d44`;
const LAB_ONE = 'sel=20261101T061500Z_e4b17c2d';
const LAB_THREE = `sel=${DEMO},20261106T083015Z_5e2c9f10,20261105T091402Z_3be1a7c0`;
const PAGES = [
  '/',
  `/?${SEL}`,
  `/compare?${SEL}`,
  '/signal',
  `/signal?${LAB_ONE}`,
  `/signal?${LAB_THREE}`,
  `/portfolio?${LAB_ONE}`,
  `/portfolio?${LAB_THREE}`,
  `/costs?${LAB_ONE}`,
  `/costs?${LAB_THREE}`,
  `/runs/${DEMO}`,
  `/runs/${DEMO}#config`,
  '/runs/20261106T142207Z_9a7b3d44#portfolio',
  '/runs/20261105T102233Z_c07a55e2',
  '/builder',
  `/builder?from=${DEMO}&set=alpha.sample_weight_halflife_years=8&set=alpha.horizon_days=900`,
  '/ledger',
  '/models',
];

for (const scheme of ['light', 'dark'] as const) {
  for (const path of PAGES) {
    test(`axe: ${path} (${scheme})`, async ({ page }) => {
      await page.emulateMedia({ colorScheme: scheme, reducedMotion: 'reduce' });
      await page.goto(path);
      await page.waitForLoadState('networkidle');
      await expect(page.locator('main h1, main h2').first()).toBeVisible();
      const results = await new AxeBuilder({ page })
        .withTags(['wcag2a', 'wcag2aa', 'wcag21aa', 'wcag22aa'])
        .analyze();
      const serious = results.violations.filter(
        (v) => v.impact === 'serious' || v.impact === 'critical',
      );
      const summary = serious.map(
        (v) =>
          `${v.id}: ${v.nodes.length} × ${v.nodes[0]?.target.join(' ') ?? ''} — ${v.nodes[0]?.failureSummary ?? ''}`,
      );
      expect(summary, summary.join('\n')).toEqual([]);
    });
  }
}
