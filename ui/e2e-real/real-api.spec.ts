import { expect, test } from '@playwright/test';

test('real API: runs list → run detail → tabs render from the pipeline’s run directory', async ({
  page,
  request,
}) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));

  const runs = (await (await request.get('/api/runs')).json()) as {
    id: string;
    label: string;
  }[];
  const run = runs.at(0);
  expect(run).toBeDefined();
  if (run === undefined) return;

  await page.goto('/runs');
  await expect(page.getByRole('heading', { level: 1, name: 'Runs' })).toBeVisible();
  await page.getByRole('link', { name: run.label }).first().click();

  await expect(page).toHaveURL(new RegExp(`/runs/${run.id}$`));
  await expect(page.getByRole('heading', { level: 1, name: run.label })).toBeVisible();
  await expect(page.getByRole('list', { name: 'Hurdle gates' }).getByRole('listitem')).toHaveCount(
    9,
  );
  await expect(page.getByRole('img', { name: /Net asset value from/ })).toBeVisible();
  await expect(page.getByRole('table', { name: 'Key metrics' })).toBeVisible();

  const tabs = page.getByRole('navigation', { name: 'Run sections' });
  await tabs.getByRole('link', { name: 'Risk' }).click();
  await expect(
    page.getByRole('heading', { level: 2, name: /Factor exposures over time/ }),
  ).toBeVisible();
  await expect(page.getByText(/Post-trade book on/)).toBeVisible();
  await tabs.getByRole('link', { name: 'Execution' }).click();
  await expect(page.getByText(/routed to the/)).toBeVisible();
  await tabs.getByRole('link', { name: 'Validation' }).click();
  await expect(page.getByRole('heading', { level: 2, name: /Hurdle gates/ })).toBeVisible();

  // SPA fallback: a deep link loads straight from the server
  await page.goto(`/runs/${run.id}/costs`);
  await expect(page.getByRole('heading', { level: 2, name: /Where the costs go/ })).toBeVisible();
  expect(errors).toEqual([]);
});
