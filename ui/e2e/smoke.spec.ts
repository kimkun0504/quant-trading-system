import { expect, test } from '@playwright/test';

const DEMO = '20261104T175159Z_944dd5b4';

test('runs list → run detail → every tab renders', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));

  await page.goto('/runs');
  await expect(page.getByRole('heading', { level: 1, name: 'Runs' })).toBeVisible();
  await page.getByRole('link', { name: 'Composite baseline' }).click();

  await expect(page).toHaveURL(new RegExp(`/runs/${DEMO}$`));
  await expect(page.getByText(/not yet distinguishable from luck/)).toBeVisible();
  await expect(page.getByRole('list', { name: 'Hurdle gates' }).getByRole('listitem')).toHaveCount(
    9,
  );
  await expect(page.getByRole('img', { name: /Net asset value from/ })).toBeVisible();
  await expect(page.getByRole('table', { name: 'Key metrics' })).toBeVisible();

  const tabs = page.getByRole('navigation', { name: 'Run sections' });
  const expectations: [string, RegExp][] = [
    ['Risk', /Factor exposures over time/],
    ['Alpha', /Rank IC per formation date/],
    ['Costs', /Where the costs go/],
    ['Execution', /Routing/],
    ['Validation', /Hurdle gates/],
    ['Performance', /Net asset value and drawdown/],
  ];
  for (const [tab, heading] of expectations) {
    await tabs.getByRole('link', { name: tab }).click();
    await expect(page.getByRole('heading', { level: 2, name: heading })).toBeVisible();
    await expect(tabs.getByRole('link', { name: tab })).toHaveAttribute('aria-current', 'page');
  }

  // the chart's text alternative
  await page.getByRole('button', { name: 'Show data table' }).first().click();
  await expect(page.getByRole('table', { name: 'Month-end NAV and drawdown' })).toBeVisible();
  expect(errors).toEqual([]);
});

test('suspicious run is flagged before anything else', async ({ page }) => {
  await page.goto('/runs/20261105T102233Z_c07a55e2');
  await expect(page.getByRole('note', { name: 'Suspicious result' })).toBeVisible();
  await expect(page.getByText(/is implausibly high/)).toBeVisible();
});

test('new run form submits and shows progress through to the run', async ({ page }) => {
  await page.goto('/runs/new');
  await page.getByLabel('Seed').fill('7');
  await page.getByRole('button', { name: 'Launch run' }).click();
  await expect(page.getByRole('heading', { name: 'Running the pipeline' })).toBeVisible();
  await expect(page.getByRole('progressbar')).toBeVisible();
  await expect(page.getByRole('link', { name: 'Open the run' })).toBeVisible({ timeout: 10_000 });
  await page.getByRole('link', { name: 'Open the run' }).click();
  await expect(page.getByText(/Composite, seed 7/).first()).toBeVisible();
  await expect(page.getByRole('list', { name: 'Hurdle gates' })).toBeVisible();
});

test('keyboard: skip link reaches main content', async ({ page }) => {
  await page.goto('/runs');
  await expect(page.getByRole('heading', { level: 1, name: 'Runs' })).toBeVisible();
  await page.keyboard.press('Tab');
  const skip = page.getByRole('link', { name: 'Skip to content' });
  await expect(skip).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(page.locator('#main')).toBeFocused();
});

test('theme toggle switches to dark and persists', async ({ page }) => {
  await page.goto('/runs');
  await page.getByRole('radio', { name: 'Dark' }).check();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  await page.reload();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
});
