import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import { resetDb } from '@/mocks/db';
import { renderApp } from '@/test/render';

describe('New run page (integration, MSW)', () => {
  it('blocks invalid input with an error summary', async () => {
    renderApp('/runs/new');
    const user = userEvent.setup();
    const stocks = await screen.findByLabelText('Stocks');
    await user.clear(stocks);
    await user.type(stocks, '5');
    await user.click(screen.getByRole('button', { name: 'Launch run' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('One field needs a change');
    expect(stocks).toHaveAttribute('aria-invalid', 'true');
    expect(stocks).toHaveAccessibleDescription(/Use between 50 and 3,000/);
  });

  it('launches, polls the job and links to the run', async () => {
    resetDb({ jobDurationMs: 300 });
    renderApp('/runs/new');
    const user = userEvent.setup();
    await user.click(await screen.findByRole('button', { name: 'Launch run' }));
    expect(
      await screen.findByRole('heading', { name: 'Running the pipeline' }),
    ).toBeInTheDocument();
    expect(screen.getByRole('progressbar')).toBeInTheDocument();
    expect(
      await screen.findByRole('link', { name: 'Open the run' }, { timeout: 4000 }),
    ).toHaveAttribute('href', expect.stringMatching(/^\/runs\//));
  });
});
