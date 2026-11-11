import { expect, test } from '@playwright/test';

const BASELINE = '20261104T175159Z_944dd5b4';
const TA0 = '20261106T083015Z_5e2c9f10';

test('sidebar reaches every workspace; an empty tray shows the labs’ run picker', async ({
  page,
}) => {
  await page.goto('/');
  await page.evaluate(() => localStorage.removeItem('qts-selection'));
  const nav = page.getByRole('navigation', { name: 'Workspaces' });
  for (const [label, heading] of [
    ['Signal lab', 'Signal lab'],
    ['Portfolio', 'Portfolio anatomy'],
    ['Costs & constraints', 'Cost & constraint lab'],
  ] as const) {
    await nav.getByRole('link', { name: label }).click();
    await expect(page.getByRole('heading', { level: 1, name: heading })).toBeVisible();
    await expect(page.getByRole('heading', { name: /Pick the runs to study/ })).toBeVisible();
  }
  await nav.getByRole('link', { name: 'Run builder' }).click();
  await expect(page.getByRole('heading', { level: 1, name: 'Run builder' })).toBeVisible();
  await nav.getByRole('link', { name: 'Ledger' }).click();
  await expect(page.getByRole('heading', { level: 1, name: 'Trial ledger' })).toBeVisible();
  await nav.getByRole('link', { name: 'Experiments' }).click();
  await expect(page.getByRole('heading', { level: 1, name: 'Experiments' })).toBeVisible();
  // the old addresses still land somewhere sensible
  await page.goto('/runs');
  await expect(page).toHaveURL(/\/$/);
  await page.goto('/runs/new');
  await expect(page).toHaveURL(/\/builder$/);
});

test('global search: "/" focuses it, typing filters, Enter opens the run', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByRole('heading', { level: 1, name: 'Experiments' })).toBeVisible();
  await page.keyboard.press('/');
  const search = page.getByRole('combobox', { name: 'Search runs' });
  await expect(search).toBeFocused();
  await search.fill('baseline aversion');
  const options = page.getByRole('listbox', { name: 'Runs' }).getByRole('option');
  await expect(options).toHaveCount(1);
  await expect(options.first()).toContainText('Composite baseline, trade aversion 0');
  await page.keyboard.press('Enter');
  await expect(page).toHaveURL(new RegExp(`/runs/${TA0}`));
});

test('tray: ticking runs fills it, persists across pages and in the URL, "c" opens compare', async ({
  page,
}) => {
  await page.goto('/');
  await page.getByLabel('Select Composite baseline for comparison').check();
  await page.getByLabel('Select Composite baseline, trade aversion 0 for comparison').check();
  const tray = page.getByRole('region', { name: 'Selected runs' });
  await expect(tray).toContainText('2 of 8 selected');
  await expect(page).toHaveURL(new RegExp(`sel=${BASELINE}(,|%2C)${TA0}`));

  // the inline side-by-side panel
  await expect(page.getByRole('heading', { level: 2, name: 'Comparing 2 runs' })).toBeVisible();
  await expect(page.getByRole('img', { name: /Net asset value of 2 runs/ })).toBeVisible();
  await expect(page.getByRole('img', { name: /Net Sharpe of 2 runs across/ })).toBeVisible();
  const diff = page.getByRole('region', { name: 'Config diff' });
  await expect(diff).toContainText('portfolio.trade_aversion');
  await page.getByRole('radio', { name: 'Log' }).check();
  await expect(page.getByRole('radio', { name: 'Log' })).toBeChecked();

  // persists across navigation and reload
  await page
    .getByRole('navigation', { name: 'Workspaces' })
    .getByRole('link', { name: 'Ledger' })
    .click();
  await expect(tray).toContainText('2 of 8 selected');
  await expect(page).toHaveURL(/sel=/);
  await page.reload();
  await expect(tray).toContainText('2 of 8 selected');

  // keyboard: c opens the compare page
  await page.keyboard.press('c');
  await expect(page).toHaveURL(/\/compare/);
  await expect(page.getByRole('heading', { level: 1, name: 'Comparing 2 runs' })).toBeVisible();
  await expect(page.getByRole('table', { name: 'Headline numbers' })).toBeVisible();

  // chips remove; the page explains what to do with one run left
  await tray.getByRole('button', { name: /Remove Composite baseline from the selection/ }).click();
  await expect(tray).toContainText('1 of 8 selected');
  await expect(page.getByRole('heading', { name: 'Select one more run' })).toBeVisible();
  await tray.getByRole('button', { name: 'Clear' }).click();
  await expect(tray).toContainText('Select runs to compare');
  await expect(page).not.toHaveURL(/sel=/);
});

test('a shared URL seeds the selection', async ({ page }) => {
  await page.goto(`/compare?sel=${BASELINE},${TA0}`);
  await expect(page.getByRole('heading', { level: 1, name: 'Comparing 2 runs' })).toBeVisible();
  await expect(page.getByRole('region', { name: 'Selected runs' })).toContainText(
    '2 of 8 selected',
  );
});
