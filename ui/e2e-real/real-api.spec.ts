import { expect, test } from '@playwright/test';

test('real API: Experiments → dossier renders every section from the pipeline’s run directory', async ({
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

  await page.goto('/');
  await expect(page.getByRole('heading', { level: 1, name: 'Experiments' })).toBeVisible();
  await page.getByRole('link', { name: run.label }).first().click();

  await expect(page).toHaveURL(new RegExp(`/runs/${run.id}$`));
  await expect(page.getByRole('heading', { level: 1, name: run.label })).toBeVisible();
  await expect(
    page.getByRole('list', { name: 'Hurdle gates' }).getByRole('listitem'),
  ).not.toHaveCount(0);
  await expect(page.getByLabel('Headline numbers')).toContainText('Net Sharpe');
  await expect(page.getByRole('img', { name: /Net asset value/ })).toBeVisible();
  await expect(page.getByRole('table', { name: 'Key metrics' })).toBeVisible();

  // the sections that read the extra artifacts and the new endpoints
  const nav = page.getByRole('navigation', { name: 'Run sections' });
  await nav.getByRole('link', { name: /Portfolio/ }).click();
  await expect(page.getByRole('img', { name: /Gross and net exposure/ })).toBeVisible();
  await expect(page.getByText('Exposure snapshot')).toBeVisible();
  await nav.getByRole('link', { name: /Costs/ }).click();
  await expect(page.getByText(/routed to the/)).toBeVisible();
  await nav.getByRole('link', { name: /Validation/ }).click();
  await expect(page.getByRole('table', { name: /Gates in the order/ })).toBeVisible();
  await nav.getByRole('link', { name: /Artifacts/ }).click();
  await expect(page.getByText('weights.parquet')).toBeVisible();
  const href = await page.getByRole('link', { name: 'Download' }).first().getAttribute('href');
  expect(href).toBeTruthy();
  const file = await request.get(href ?? '');
  expect(file.ok()).toBe(true);

  // SPA fallback: an old tab deep link loads straight from the server and lands on its section
  await page.goto(`/runs/${run.id}/costs`);
  await expect(page).toHaveURL(new RegExp(`/runs/${run.id}#costs$`));
  await expect(page.getByText('Costs by month')).toBeVisible();
  expect(errors).toEqual([]);
});
