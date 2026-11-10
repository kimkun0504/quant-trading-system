import { screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';
import { beforeEach, describe, expect, it } from 'vitest';
import { clearSelection } from '@/features/selection';
import { server } from '@/mocks/node';
import { renderApp } from '@/test/render';

describe('Experiments page (integration, MSW)', () => {
  beforeEach(() => {
    clearSelection();
    localStorage.clear();
  });

  it('groups runs into families with the differing keys and headline numbers', async () => {
    renderApp('/');
    const table = await screen.findByRole('table');
    const baseline = within(table)
      .getAllByRole('button', { name: /Composite baseline/ })
      .find((b) => b.getAttribute('aria-expanded') !== null);
    expect(baseline).toHaveTextContent('4 runs');
    expect(baseline).toHaveTextContent('varies: alpha.horizon_days');
    const rows = within(table).getAllByRole('row');
    const demo = rows.find((r) => within(r).queryByRole('link', { name: 'Composite baseline' }));
    expect(demo).toHaveTextContent('+0.74');
    expect(demo).toHaveTextContent('100%'); // invested days
    expect(screen.getByText('suspect')).toBeInTheDocument();
    expect(await screen.findByLabelText('Trials in the ledger')).toHaveTextContent('trials:');
  });

  it('collapses a family and sorts within families', async () => {
    renderApp('/');
    await screen.findByRole('table');
    const user = userEvent.setup();
    const toggle = screen
      .getAllByRole('button', { name: /Composite baseline/ })
      .find((b) => b.getAttribute('aria-expanded') === 'true');
    if (!toggle) throw new Error('family toggle missing');
    await user.click(toggle);
    expect(toggle).toHaveAttribute('aria-expanded', 'false');
    expect(screen.queryByRole('link', { name: 'Composite baseline' })).not.toBeInTheDocument();
    await user.click(toggle);
    await user.click(screen.getByRole('button', { name: /Net Sharpe/ }));
    expect(screen.getByRole('columnheader', { name: /Net Sharpe/ })).toHaveAttribute(
      'aria-sort',
      'descending',
    );
    const first = within(screen.getByRole('table')).getAllByRole('row')[2];
    expect(first).toHaveTextContent('Same-close execution');
  });

  it('selecting two runs fills the tray and shows the side-by-side panel with a config diff', async () => {
    renderApp('/');
    await screen.findByRole('table');
    const user = userEvent.setup();
    await user.click(screen.getByLabelText('Select Composite baseline for comparison'));
    await user.click(
      screen.getByLabelText('Select Composite baseline, trade aversion 0 for comparison'),
    );
    const tray = screen.getByRole('region', { name: 'Selected runs' });
    expect(tray).toHaveTextContent('2 of 8 selected');
    expect(screen.getByRole('heading', { name: 'Comparing 2 runs' })).toBeInTheDocument();
    const diff = await screen.findByRole('region', { name: 'Config diff' });
    expect(diff).toHaveTextContent('1 of 29 keys differ');
    expect(within(diff).getByRole('rowheader', { name: 'portfolio.trade_aversion' })).toBeVisible();
    await user.click(
      within(tray).getByRole('button', { name: /Remove Composite baseline from the selection/ }),
    );
    expect(tray).toHaveTextContent('1 of 8 selected');
    expect(screen.queryByRole('heading', { name: /Comparing/ })).not.toBeInTheDocument();
  });

  it('invites launching a run when the registry is empty', async () => {
    server.use(http.get('*/api/runs', () => HttpResponse.json([])));
    renderApp('/');
    expect(await screen.findByRole('heading', { name: 'No runs yet' })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Launch a run' })).toHaveAttribute('href', '/builder');
  });

  it('explains an unreachable API', async () => {
    server.use(
      http.get('*/api/runs', () =>
        HttpResponse.json({ detail: 'registry offline' }, { status: 503 }),
      ),
    );
    renderApp('/');
    expect(await screen.findByRole('alert')).toHaveTextContent('The API answered 503');
  });
});
