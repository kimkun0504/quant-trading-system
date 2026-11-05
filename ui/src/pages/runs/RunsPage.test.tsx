import { screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';
import { describe, expect, it } from 'vitest';
import { server } from '@/mocks/node';
import { renderApp } from '@/test/render';

describe('Runs page (integration, MSW)', () => {
  it('lists runs newest first with signed headline metrics and links', async () => {
    renderApp('/runs');
    const table = await screen.findByRole('table');
    const rows = within(table).getAllByRole('row').slice(1);
    expect(rows).toHaveLength(6);
    expect(rows[0]).toHaveTextContent('Composite, 600 names');
    expect(rows[0]).toHaveTextContent('Running');
    const demo = rows.find((r) => r.textContent.includes('Composite baseline'));
    expect(demo).toHaveTextContent('+0.74');
    expect(
      within(demo as HTMLElement).getByRole('link', { name: 'Composite baseline' }),
    ).toHaveAttribute('href', '/runs/20261104T175159Z_944dd5b4');
    expect(screen.getByText('suspect')).toBeInTheDocument();
  });

  it('sorts by a column and exposes aria-sort', async () => {
    renderApp('/runs');
    await screen.findByRole('table');
    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: /Net Sharpe/ }));
    const header = screen.getByRole('columnheader', { name: /Net Sharpe/ });
    expect(header).toHaveAttribute('aria-sort', 'descending');
    const first = within(screen.getByRole('table')).getAllByRole('row')[1];
    expect(first).toHaveTextContent('Same-close execution');
  });

  it('invites launching a run when the registry is empty', async () => {
    server.use(http.get('*/api/runs', () => HttpResponse.json([])));
    renderApp('/runs');
    expect(await screen.findByRole('heading', { name: 'No runs yet' })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Launch a run' })).toHaveAttribute('href', '/runs/new');
  });

  it('explains an unreachable API', async () => {
    server.use(
      http.get('*/api/runs', () =>
        HttpResponse.json({ detail: 'registry offline' }, { status: 503 }),
      ),
    );
    renderApp('/runs');
    expect(await screen.findByRole('alert')).toHaveTextContent('The API answered 503');
  });
});
