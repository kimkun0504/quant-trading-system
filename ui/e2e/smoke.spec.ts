import { expect, test } from '@playwright/test';

const DEMO = '20261104T175159Z_944dd5b4';

test('experiments → run dossier: header, every section, section nav, data table', async ({
  page,
}) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));

  await page.goto('/');
  await expect(page.getByRole('heading', { level: 1, name: 'Experiments' })).toBeVisible();
  await page.getByRole('link', { name: 'Composite baseline', exact: true }).click();

  await expect(page).toHaveURL(new RegExp(`/runs/${DEMO}$`));
  await expect(page.getByText(/^Passes the .* with 4 trials/)).toBeVisible();
  await expect(page.getByRole('list', { name: 'Hurdle gates' }).getByRole('listitem')).toHaveCount(
    11,
  );
  const facts = page.getByLabel('Headline numbers');
  await expect(facts).toContainText('Net Sharpe');
  await expect(facts).toContainText('4 trials');
  await expect(page.getByRole('img', { name: /Net asset value/ })).toBeVisible();
  await expect(page.getByRole('table', { name: 'Key metrics' })).toBeVisible();

  // every section is on the page, in order, with its lab link
  const sections = [
    'Performance',
    'Signal',
    'Portfolio',
    'Costs',
    'Validation',
    'Config',
    'Artifacts',
  ];
  for (const [i, name] of sections.entries()) {
    await expect(page.locator(`section#${name.toLowerCase()} > div > h2`)).toHaveText(
      `${i + 1}${name}`,
    );
  }
  await expect(page.getByRole('link', { name: /Open in Signal lab/ })).toHaveAttribute(
    'href',
    '/signal',
  );

  // the section nav follows the scroll and jumps on click; j / k step
  const nav = page.getByRole('navigation', { name: 'Run sections' });
  await expect(nav.getByRole('link', { name: /Performance/ })).toHaveAttribute(
    'aria-current',
    'location',
  );
  await nav.getByRole('link', { name: /Validation/ }).click();
  await expect(page).toHaveURL(/#validation$/);
  await expect(page.getByRole('heading', { level: 2, name: /^Validation$/ })).toBeFocused();
  await expect(nav.getByRole('link', { name: /Validation/ })).toHaveAttribute(
    'aria-current',
    'location',
  );
  await expect(page.getByRole('table', { name: /Gates in the order/ })).toBeInViewport();
  await page.keyboard.press('j');
  await expect(page.getByRole('heading', { level: 2, name: /^Config$/ })).toBeFocused();
  await expect(nav.getByRole('link', { name: /Config/ })).toHaveAttribute(
    'aria-current',
    'location',
  );
  await page.keyboard.press('k');
  await page.keyboard.press('k');
  await expect(page.getByRole('heading', { level: 2, name: /^Costs$/ })).toBeFocused();

  // the chart's text alternative
  await page.getByRole('button', { name: 'Show data table' }).first().click();
  await expect(page.getByRole('table', { name: 'Month-end NAV and drawdown' })).toBeVisible();
  expect(errors).toEqual([]);
});

test('dossier: old tab addresses redirect to the section; add to tray from the header', async ({
  page,
}) => {
  await page.goto(`/runs/${DEMO}/alpha`);
  await expect(page).toHaveURL(new RegExp(`/runs/${DEMO}#signal$`));
  await expect(page.getByRole('heading', { level: 2, name: /^Signal$/ })).toBeInViewport();
  await page.goto(`/runs/${DEMO}/risk`);
  await expect(page).toHaveURL(new RegExp(`/runs/${DEMO}#portfolio$`));
  await page.goto(`/runs/${DEMO}/validation`);
  await expect(page).toHaveURL(new RegExp(`/runs/${DEMO}#validation$`));
  await page.goto(`/runs/${DEMO}/bogus`);
  await expect(
    page.getByRole('heading', { level: 1, name: 'This page does not exist' }),
  ).toBeVisible();

  await page.goto(`/runs/${DEMO}`);
  const tray = page.getByRole('region', { name: 'Selected runs' });
  await page.getByRole('button', { name: 'Add to tray' }).click();
  await expect(tray).toContainText('1 of 8 selected');
  await expect(page).toHaveURL(new RegExp(`sel=${DEMO}`));
  await expect(page.getByRole('button', { name: 'Remove from tray' })).toBeVisible();
  // a second run from its own dossier, then Compare from the header
  await page.goto('/runs/20261106T083015Z_5e2c9f10');
  await page.getByRole('button', { name: 'Add to tray' }).click();
  await expect(tray).toContainText('2 of 8 selected');
  await page.getByRole('link', { name: 'Compare (2)' }).click();
  await expect(page.getByRole('heading', { level: 1, name: 'Comparing 2 runs' })).toBeVisible();
  await expect(page.getByRole('table', { name: 'Headline numbers' })).toBeVisible();

  // config: search and diff against another run
  await page.goto(`/runs/${DEMO}#config`);
  await page.getByRole('searchbox', { name: /Search configuration/ }).fill('horizon');
  await expect(page.getByText('1 of 38 keys')).toBeVisible();
  await expect(page.getByRole('region', { name: 'Configuration' })).not.toContainText(
    'trade_aversion',
  );
  await page.getByLabel('Compare with').selectOption('20261106T083015Z_5e2c9f10');
  await expect(page.getByRole('region', { name: 'Config diff' })).toContainText(
    'portfolio.trade_aversion',
  );

  // artifacts download link is served by the (mock) API
  const href = await page.getByRole('link', { name: 'Download' }).first().getAttribute('href');
  expect(href).toMatch(new RegExp(`/api/runs/${DEMO}/files/`));
});

test('suspicious run is flagged before anything else', async ({ page }) => {
  await page.goto('/runs/20261105T102233Z_c07a55e2');
  await expect(page.getByRole('note', { name: 'Suspicious result' })).toBeVisible();
  await expect(page.getByText(/is implausibly high/)).toBeVisible();
});

test('keyboard: skip link reaches main content', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByRole('heading', { level: 1, name: 'Experiments' })).toBeVisible();
  await page.keyboard.press('Tab');
  const skip = page.getByRole('link', { name: 'Skip to content' });
  await expect(skip).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(page.locator('#main')).toBeFocused();
});

test('theme toggle switches to dark, persists, and switches back', async ({ page }) => {
  await page.emulateMedia({ colorScheme: 'light' });
  await page.goto('/');
  await page.getByRole('button', { name: 'Switch to dark theme' }).click();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  await page.reload();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  await page.getByRole('button', { name: 'Switch to light theme' }).click();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
});
