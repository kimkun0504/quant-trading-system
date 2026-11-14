import { expect, test } from '@playwright/test';

/** Smoke: every research page and paper book renders on the mock API, with its key facts. */
test.describe('research tools and paper books (mock API)', () => {
  test('nav and the seal chip', async ({ page }) => {
    await page.goto('/');
    const nav = page.getByRole('navigation', { name: 'Workspaces' });
    for (const name of [
      'Holdout seal',
      'Signal tests',
      'Sweeps',
      'Capacity',
      'Risk calibration',
      'Mid-cap check',
      'Paper trading',
    ])
      await expect(nav.getByRole('link', { name })).toBeVisible();
    await expect(page.getByRole('banner').getByText(/Sealed from 9 Nov 2023/)).toBeVisible();
    await nav.getByRole('link', { name: 'Holdout seal' }).click();
    await expect(page.getByRole('heading', { level: 1, name: 'Holdout seal' })).toBeVisible();
  });

  test('signal test: launch preview and a result', async ({ page }) => {
    await page.goto('/signals');
    await expect(page.getByTestId('cli-command')).toContainText('qts signal test high_52w');
    await page.getByRole('button', { name: 'Run test…' }).click();
    await expect(page.getByRole('dialog').getByTestId('trial-cost')).toContainText(/2 → 3 tests/);
    await page.getByRole('button', { name: 'Cancel' }).click();
    await page.getByRole('link', { name: 'strev_21' }).click();
    await expect(page.getByTestId('testing-context')).toContainText('Signals tested in the family');
    await expect(
      page.getByRole('img', { name: /Primary t \+1\.01 against the hurdle/ }),
    ).toBeVisible();
  });

  test('sweep: preview, confirm, members and PBO', async ({ page }) => {
    await page.goto('/sweeps');
    await expect(page.getByTestId('trial-cost')).toContainText('Adds 3 trials');
    await page.getByRole('button', { name: 'Review and run…' }).click();
    await page.getByRole('button', { name: 'Run 3 members' }).click();
    await expect(page).toHaveURL(/\/sweeps\/sw_[0-9a-f]{12}$/);
    await expect(page.getByRole('region', { name: 'Sweep members' })).toBeVisible();
    await page.goto('/sweeps/sw_8503759dfeb8');
    await expect(page.getByTestId('pbo')).toContainText('0.73');
  });

  test('capacity, risk and the mid-cap check render', async ({ page }) => {
    await page.goto('/capacity/20261114T042605395445Z_feafa843');
    await expect(page.getByRole('img', { name: /Net Sharpe by NAV/ })).toBeVisible();
    await page.goto('/risk/20261114T042614680785Z_feafa843');
    await expect(page.getByTestId('recommendation')).toContainText('Recommended');
    await page.goto('/checks/midcap');
    await expect(page.getByTestId('midcap-blocked')).toContainText('once-only');
    await expect(page.getByRole('button', { name: 'Run a forced repeat…' })).toBeDisabled();
  });

  test('paper book: status, cones, books switch, ack form', async ({ page }) => {
    await page.goto('/paper/check');
    await expect(page.getByTestId('paper-status')).toContainText('OK');
    await expect(
      page.getByRole('img', { name: /Cumulative log return of the book/ }),
    ).toBeVisible();
    await page.getByLabel('Book', { exact: true }).selectOption('baseline');
    await expect(page).toHaveURL(/\/paper\/baseline$/);
    await expect(page.getByTestId('paper-status')).toContainText('STOP');
    await expect(page.getByTestId('ack-panel')).toBeVisible();
  });
});
