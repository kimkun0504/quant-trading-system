import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import { renderApp } from '@/test/render';

describe('Signal tests (integration, MSW)', () => {
  it('lists tests with the multiple-testing context and launches a counted test after confirmation', async () => {
    renderApp('/signals');
    const user = userEvent.setup();
    expect(await screen.findByRole('link', { name: 'strev_21' })).toBeInTheDocument();
    expect(screen.getByTestId('testing-context')).toHaveTextContent(/Signals tested\s*2/);
    // the form shows every parameter with its default and the exact command
    expect(screen.getByLabelText('Horizons (trading days)')).toHaveValue('5,20,60,120');
    await waitFor(() =>
      expect(screen.getByTestId('cli-command')).toHaveTextContent(
        /^qts signal test high_52w --store \S+ --start 2000-01-01 --max-rank 1500 --horizons 5,20,60,120$/,
      ),
    );
    await user.click(screen.getByRole('button', { name: 'Run test…' }));
    const dialog = await screen.findByRole('dialog', { name: /Run a signal test of high_52w/ });
    const cost = await within(dialog).findByTestId('trial-cost');
    expect(cost).toHaveTextContent(/Adds 1 test to signal:\w+: 2 → 3 tests/);
    expect(cost).toHaveTextContent(/Holm adjusts every test/);
    await user.click(within(dialog).getByRole('button', { name: 'Run and count the test' }));
    expect(await screen.findByText(/^Queued: /)).toBeInTheDocument();
  });

  it('an uncounted test says so before confirmation', async () => {
    renderApp('/signals');
    const user = userEvent.setup();
    await user.click(await screen.findByLabelText('Count this test (log it in the ledger)'));
    await waitFor(() => expect(screen.getByTestId('cli-command')).toHaveTextContent(/--no-log$/));
    await user.click(screen.getByRole('button', { name: 'Run test…' }));
    const dialog = await screen.findByRole('dialog');
    expect(
      await within(dialog).findByText(/The test is not logged: the family stays at 2 tests/),
    ).toBeInTheDocument();
  });

  it('a result leads with the family size, the hurdle and Holm', async () => {
    renderApp('/signals/20261114T042559180079Z_strev_21');
    const ctx = await screen.findByTestId('testing-context');
    expect(ctx).toHaveTextContent(/Primary t/);
    expect(ctx).toHaveTextContent(/Signals tested in the family\s*2/);
    expect(ctx).toHaveTextContent(/Below \|t\| ≥ 3/);
    expect(screen.getByTestId('signal-verdict')).toHaveTextContent(
      /2 signals tested in this family/,
    );
    expect(screen.getByRole('region', { name: 'Tests in the family' })).toHaveTextContent(
      /this test/,
    );
  });
});
