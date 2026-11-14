import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import { resetDb } from '@/mocks/db';
import { renderApp } from '@/test/render';

describe('Paper trading (integration, MSW)', () => {
  it('lists the books with their level and mode', async () => {
    renderApp('/paper');
    const table = await screen.findByRole('region', { name: /paper books/ });
    expect(within(table).getByRole('link', { name: 'baseline' })).toBeInTheDocument();
    expect(table).toHaveTextContent(/STOP/);
    expect(table).toHaveTextContent(/backfill/);
  });

  it('a STOP: reasons, the check table, the acknowledgement form', async () => {
    renderApp('/paper/baseline');
    const user = userEvent.setup();
    const status = await screen.findByTestId('paper-status');
    expect(status).toHaveTextContent(/STOP/);
    expect(status).toHaveTextContent(/trading halted/);
    expect(status).toHaveTextContent(/below the STOP boundary/);
    expect(screen.getByRole('region', { name: 'Status checks' })).toHaveTextContent(
      /Drawdown\s*STOP/,
    );
    const ack = screen.getByTestId('ack-panel');
    const button = within(ack).getByRole('button', { name: 'Acknowledge and resume' });
    expect(button).toBeDisabled();
    await user.type(
      within(ack).getByLabelText('Reason'),
      'checked the fills against the vendor; a data gap',
    );
    expect(within(ack).getByTestId('cli-command')).toHaveTextContent(
      "qts paper ack --name baseline --note 'checked the fills against the vendor; a data gap'",
    );
    await user.click(button);
    expect(
      await screen.findByText(/Acknowledged: the next run generates orders again/),
    ).toBeInTheDocument();
    await waitFor(() =>
      expect(screen.getByTestId('paper-status')).toHaveTextContent(/acknowledged/),
    );
  });

  it('"Run today" before the vendor publishes: stale data, exit code 3, explained', async () => {
    resetDb({ jobDurationMs: 1 });
    renderApp('/paper/check');
    const user = userEvent.setup();
    expect(await screen.findByTestId('paper-status')).toHaveTextContent(/OK/);
    expect(screen.getByText(/Backfill only/)).toBeInTheDocument();
    expect(screen.getAllByTestId('cli-command')[0]).toHaveTextContent(
      /^qts paper run --name check$/,
    );
    await user.click(screen.getByRole('button', { name: 'Run today…' }));
    const dialog = await screen.findByRole('dialog');
    expect(dialog).toHaveTextContent(/exit code 3/);
    await user.click(within(dialog).getByRole('button', { name: 'Run' }));
    expect(
      await screen.findByText(/Data not yet published \(exit code 3\)/, {}, { timeout: 8000 }),
    ).toBeInTheDocument();
  });

  it('orders with estimated costs, holdings and the schedule text', async () => {
    renderApp('/paper/check');
    expect(await screen.findByRole('region', { name: 'Orders' })).toHaveTextContent(/Est\. cost/);
    expect(
      await screen.findByRole('region', { name: 'launchd agent (com.qts.paper.check.plist)' }),
    ).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Copy the crontab line' })).toBeInTheDocument();
  });
});
