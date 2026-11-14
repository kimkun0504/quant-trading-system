import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';
import type { Page } from '@playwright/test';

const DEMO = '20261104T175159Z_944dd5b4';
const HALF_LIFE = 'Training recency half-life (years), 0 = equal weights';

const slip = (page: Page) => page.getByRole('complementary', { name: 'Before launch' });
const queue = (page: Page) => page.getByRole('region', { name: 'Run queue' });

test('clone a run, change two fields, queue two variants, find them in Experiments', async ({
  page,
}) => {
  test.setTimeout(60_000);
  // from the dossier header
  await page.goto(`/runs/${DEMO}`);
  await page.getByRole('link', { name: 'Clone & edit' }).click();
  await expect(page).toHaveURL(new RegExp(`/builder\\?from=${DEMO}`));
  await expect(page.getByTestId('base-line')).toContainText('Composite baseline');

  // two edits; the slip lists exactly those keys
  await page.getByLabel(HALF_LIFE, { exact: true }).fill('8');
  await page.getByLabel('Trade aversion', { exact: true }).fill('0.5');
  const diff = slip(page).getByRole('table');
  await expect(
    diff.getByRole('row', { name: /alpha.sample_weight_halflife_years 0 8/ }),
  ).toBeVisible();
  await expect(diff.getByRole('row', { name: /portfolio.trade_aversion 1.0 0.5/ })).toBeVisible();
  await expect(diff.getByRole('row')).toHaveCount(3);
  await expect(page.getByTestId('slip-state')).toHaveText('Valid');
  await expect(page.getByTestId('config-hash')).toHaveText(/^[0-9a-f]{12}$/);
  await expect(page.getByTestId('trial-line')).toContainText(/This will be trial 3 in family/);
  const cli = slip(page).getByLabel('Equivalent CLI command');
  await expect(cli).toContainText(`--config runs/${DEMO}/config.yaml`);
  await expect(cli).toContainText('--set alpha.sample_weight_halflife_years=8');

  // the changed-only filter of the full editor shows the same two fields
  await page.getByRole('checkbox', { name: 'Changed only' }).check();
  await expect(page.getByText('2 of 33 settings')).toBeVisible();
  await page.getByRole('checkbox', { name: 'Changed only' }).uncheck();

  // a variant with half-life 0, then both to the queue
  await page.getByRole('button', { name: 'Add variant' }).click();
  await page.getByLabel(HALF_LIFE, { exact: true }).fill('0');
  await expect(diff.getByRole('columnheader', { name: 'Variant B' })).toBeVisible();
  await page.getByRole('button', { name: 'Queue all 2 variants' }).click();
  await expect(slip(page).getByText('Added 2 runs to the queue.')).toBeVisible();

  // one runs, one waits; the top bar says so on every page
  await expect(queue(page).locator('[data-status="running"]')).toHaveCount(1);
  await expect(queue(page).locator('[data-status="queued"]')).toHaveCount(1);
  await expect(queue(page).getByRole('progressbar')).toBeVisible();
  await expect(queue(page).getByRole('list', { name: 'Pipeline stages' })).toBeVisible();
  await expect(page.getByTestId('queue-indicator')).toHaveAccessibleName(/1 running, 1 queued/);
  await queue(page).getByRole('button', { name: 'Show log' }).first().click();
  await expect(queue(page).getByLabel(/^Log of Composite baseline/)).toContainText('qts_api.job');

  // both finish (4 s each in the mock) and link to their dossiers
  await expect(queue(page).locator('[data-status="done"]')).toHaveCount(2, { timeout: 20_000 });
  await expect(page.getByTestId('queue-indicator')).toHaveAccessibleName(/queue is empty/);
  await queue(page)
    .getByRole('button', {
      name: /^Add Composite baseline, sample weight halflife years 8.* to the tray$/,
    })
    .click();
  await expect(page.getByRole('region', { name: 'Selected runs' })).toContainText(
    '1 of 8 selected',
  );

  // the finished runs are in Experiments, in the base run's family
  await page.getByRole('link', { name: 'Experiments' }).click();
  const row = page.getByRole('row', {
    name: /sample weight halflife years 8, trade aversion 0\.5/,
  });
  await expect(row).toBeVisible();
  await expect(
    page.getByRole('row', { name: /Composite baseline, trade aversion 0\.5 / }),
  ).toBeVisible();
  await row.getByRole('link', { name: /^Composite baseline, sample weight/ }).click();
  await expect(page.getByRole('list', { name: 'Hurdle gates' })).toBeVisible();
  await page.goto(page.url().replace(/#.*$/, '') + '#config');
  await expect(page.getByRole('region', { name: 'Configuration' })).toContainText(
    'sample_weight_halflife_years',
  );
});

test('clone from an Experiments row; cancel and reorder queued runs', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('link', { name: 'Clone and edit Composite baseline, horizon 10' }).click();
  await expect(page.getByTestId('base-line')).toContainText('Composite baseline, horizon 10');
  await expect(page.getByLabel('Prediction horizon (days)', { exact: true })).toHaveValue('10');

  // three variants: horizon 5, 15, 30
  await page.getByLabel('Prediction horizon (days)', { exact: true }).fill('5');
  for (const h of ['15', '30']) {
    await page.getByRole('button', { name: 'Add variant' }).click();
    await page.getByLabel('Prediction horizon (days)', { exact: true }).fill(h);
  }
  await page.getByRole('button', { name: 'Queue all 3 variants' }).click();
  const waiting = queue(page).locator('[data-status="queued"]');
  await expect(waiting).toHaveCount(2);
  await expect(waiting.first()).toContainText('horizon days 15');
  await queue(page)
    .getByRole('button', { name: /^Move .*horizon days 30 up$/ })
    .click();
  await expect(waiting.first()).toContainText('horizon days 30');
  await queue(page)
    .getByRole('button', { name: /^Cancel .*horizon days 15$/ })
    .click();
  await expect(waiting).toHaveCount(1);
  await expect(queue(page).locator('[data-status="cancelled"]')).toContainText(
    'Cancelled before it started',
  );
});

test('inline validation blocks the launch and names the field', async ({ page }) => {
  await page.goto('/builder');
  const horizon = page.getByLabel('Prediction horizon (days)', { exact: true });
  await horizon.fill('900');
  await expect(horizon).toHaveAttribute('aria-invalid', 'true');
  await expect(page.getByText('Horizon days must be at most 252.')).toBeVisible();
  await expect(page.getByTestId('slip-state')).toHaveText('Needs a fix');
  await expect(page.getByRole('button', { name: 'Add to queue' })).toBeDisabled();
  await page.getByRole('button', { name: /Reset Prediction horizon/ }).click();
  await expect(page.getByTestId('slip-state')).toHaveText('Valid');
  await expect(page.getByRole('button', { name: 'Add to queue' })).toBeEnabled();
});

for (const scheme of ['light', 'dark'] as const) {
  test(`axe: builder with a running and a queued job (${scheme})`, async ({ page }) => {
    await page.emulateMedia({ colorScheme: scheme, reducedMotion: 'reduce' });
    await page.goto(`/builder?from=${DEMO}&set=alpha.sample_weight_halflife_years=8`);
    await page.getByRole('button', { name: 'Add variant' }).click();
    await page.getByLabel(HALF_LIFE, { exact: true }).fill('4');
    await page.getByRole('button', { name: 'Queue all 2 variants' }).click();
    await expect(queue(page).locator('[data-status="running"]')).toHaveCount(1);
    await queue(page).getByRole('button', { name: 'Show log' }).first().click();
    await page.getByRole('searchbox', { name: 'Search settings' }).fill('a');
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
