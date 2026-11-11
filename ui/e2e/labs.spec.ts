import { expect, test } from '@playwright/test';

const BASE = '20261104T175159Z_944dd5b4';
const TA0 = '20261106T083015Z_5e2c9f10';
const ML = '20261105T091402Z_3be1a7c0';
const LONG = '20261101T061500Z_e4b17c2d';
const THREE = `sel=${BASE},${TA0},${ML}`;

test.describe('labs', () => {
  test('Signal lab: picker → deep view on one run → learned weights on the 24-year ML run', async ({
    page,
  }) => {
    const errors: string[] = [];
    page.on('pageerror', (e) => errors.push(e.message));
    await page.goto('/signal');
    await page.evaluate(() => localStorage.removeItem('qts-selection'));
    await page.goto('/signal');
    await expect(page.getByRole('heading', { level: 1, name: 'Signal lab' })).toBeVisible();
    await page.getByRole('checkbox', { name: 'Study Composite baseline', exact: true }).click();
    await expect(page).toHaveURL(/sel=20261104T175159Z_944dd5b4/);
    await expect(
      page.getByRole('img', { name: /Mean rank IC by sleeve and calendar year/ }),
    ).toBeVisible();
    await expect(page.getByRole('img', { name: /IC decay of 1 run/ })).toBeVisible();
    await expect(page.getByText('No ML sleeve in this run')).toBeVisible();
    await expect(page.getByRole('img', { name: /Theme weights of the composite/ })).toBeVisible();
    await expect(page.getByRole('table', { name: /processed features/ }).first()).toBeVisible();

    await page.goto(`/signal?sel=${LONG}`);
    const heat = page.getByRole('img', { name: /Feature importance of Ensemble/ });
    await expect(heat).toBeVisible();
    await page.getByRole('radio', { name: 'LightGBM' }).check();
    await expect(page.getByRole('img', { name: /Feature importance of LightGBM/ })).toBeVisible();
    // hover a cell: the readout names the refit window
    const box = await page
      .getByRole('img', { name: /Feature importance of LightGBM/ })
      .boundingBox();
    if (!box) throw new Error('heatmap not laid out');
    await page.mouse.move(box.x + box.width * 0.6, box.y + box.height * 0.5);
    await expect(page.getByText('Trained on').first()).toBeVisible();
    await page.getByRole('button', { name: 'Show data table' }).nth(3).click();
    await expect(
      page.getByRole('region', { name: /Importance share by feature and refit/ }),
    ).toBeVisible();
    expect(errors).toEqual([]);
  });

  test('Signal lab with three runs: lanes and overlays', async ({ page }) => {
    await page.goto(`/signal?${THREE}`);
    await expect(
      page.getByRole('img', { name: /Mean rank IC by sleeve and calendar year/ }),
    ).toHaveCount(3);
    await expect(page.getByRole('img', { name: /IC decay of 3 runs/ })).toBeVisible();
    await expect(
      page.getByRole('img', { name: /Trailing one-year mean rank IC of 3 runs/ }),
    ).toBeVisible();
    await expect(
      page.getByRole('list', { name: 'Runs in this lab' }).getByRole('listitem'),
    ).toHaveCount(3);
  });

  test('Portfolio anatomy: one run with the holdings scrubber, then three runs', async ({
    page,
  }) => {
    await page.goto(`/portfolio?sel=${TA0}`);
    await expect(page.getByRole('heading', { level: 1, name: 'Portfolio anatomy' })).toBeVisible();
    await expect(page.getByRole('img', { name: /Gross and net exposure of/ })).toBeVisible();
    await expect(
      page.getByRole('img', { name: /Long and short weight by industry/ }),
    ).toBeVisible();
    await expect(
      page.getByRole('img', { name: /Net exposure to the .* alpha themes/ }),
    ).toBeVisible();
    await expect(
      page.getByRole('img', { name: /Effective holding period over time of 1 run/ }),
    ).toBeVisible();
    await expect(page.getByRole('img', { name: /Realised 63-day volatility/ })).toBeVisible();
    const heading = page.getByRole('heading', { name: /^Holdings/ });
    const before = await heading.textContent();
    const slider = page.getByRole('slider', { name: 'Month end' });
    await slider.focus();
    await page.keyboard.press('Home');
    await expect(heading).not.toHaveText(before ?? '');
    await expect(page.getByRole('table', { name: /Every position on/ })).toBeVisible();

    await page.goto(`/portfolio?${THREE}`);
    await expect(page.getByRole('img', { name: /Gross and net exposure of/ })).toHaveCount(3);
    await expect(
      page.getByRole('img', { name: /Effective holding period over time of 3 runs/ }),
    ).toBeVisible();
    await expect(page.getByRole('slider', { name: 'Month end' })).toHaveCount(3);
  });

  test('Cost & constraint lab: one run, then three', async ({ page }) => {
    await page.goto(`/costs?sel=${BASE}`);
    await expect(
      page.getByRole('heading', { level: 1, name: 'Cost & constraint lab' }),
    ).toBeVisible();
    await expect(page.getByRole('img', { name: /Share of solves on which each of/ })).toBeVisible();
    await expect(
      page.getByRole('img', { name: /Transfer coefficient over time of 1 run/ }),
    ).toBeVisible();
    await expect(
      page.getByRole('img', { name: /ADV participation cap over time, 1 run/ }),
    ).toBeVisible();
    await expect(page.getByRole('img', { name: /Vol-target outcome shares/ })).toBeVisible();
    await expect(page.getByRole('img', { name: /Monthly costs of/ })).toBeVisible();
    await expect(
      page.getByRole('img', { name: /Distribution of daily cost per side/ }),
    ).toBeVisible();
    await expect(page.getByText(/the gate asks for 2×/)).toBeVisible();
    await page.getByRole('button', { name: 'Show data table' }).first().click();
    await expect(
      page.getByRole('region', { name: /Share of solves where each constraint bound/ }),
    ).toBeVisible();

    await page.goto(`/costs?${THREE}`);
    await expect(page.getByRole('img', { name: /Share of solves on which each of/ })).toHaveCount(
      3,
    );
    await expect(
      page.getByRole('img', { name: /Transfer coefficient over time of 3 runs/ }),
    ).toBeVisible();
    await expect(page.getByRole('img', { name: /Monthly costs of/ })).toHaveCount(3);
  });

  test('the dossier’s "Open in" links land in the labs with the run selected', async ({ page }) => {
    await page.goto(`/runs/${ML}`);
    await page.evaluate(() => localStorage.removeItem('qts-selection'));
    await page.getByRole('link', { name: /Open in Signal lab/ }).click();
    await expect(page).toHaveURL(/\/signal/);
    await expect(page.getByRole('heading', { level: 1, name: 'Signal lab' })).toBeVisible();
  });
});
