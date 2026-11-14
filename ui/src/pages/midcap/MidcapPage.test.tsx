import { screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import { renderApp } from '@/test/render';

describe('Mid-cap check (integration, MSW)', () => {
  it('reads the results from disk and refuses a second run without a typed force', async () => {
    renderApp('/checks/midcap');
    const user = userEvent.setup();
    expect(await screen.findByTestId('midcap-verdict')).toHaveTextContent('INCONCLUSIVE');
    expect(screen.getByRole('region', { name: 'Mid-cap results' })).toHaveTextContent(/matches/);
    expect(screen.getByTestId('midcap-blocked')).toHaveTextContent(
      /once-only, so a second run is no longer out-of-sample/,
    );
    const run = screen.getByRole('button', { name: 'Run a forced repeat…' });
    expect(run).toBeDisabled();
    await user.click(screen.getByLabelText('Force a repeat (flagged)'));
    const phrase = screen.getByLabelText('Type “not out-of-sample” to confirm');
    await user.type(phrase, 'not out of sample');
    expect(run).toBeDisabled();
    await user.clear(phrase);
    await user.type(phrase, 'not out-of-sample');
    expect(run).toBeEnabled();
    expect(screen.getAllByTestId('cli-command')[1]).toHaveTextContent(
      /--max-rank 2750 --cache --force$/,
    );
    await user.click(run);
    const dialog = await screen.findByRole('dialog', {
      name: 'Force a repeat of the mid-cap check?',
    });
    await user.click(within(dialog).getByRole('button', { name: 'Run the flagged repeat' }));
    expect(await screen.findByText(/^Queued: /)).toBeInTheDocument();
  });

  it('the dry run is always allowed and is not a trial', async () => {
    renderApp('/checks/midcap');
    const user = userEvent.setup();
    await user.click(await screen.findByRole('button', { name: 'Dry run: check coverage…' }));
    const dialog = await screen.findByRole('dialog', { name: 'Run the coverage dry run?' });
    expect(dialog).toHaveTextContent(/Not a trial/);
  });

  it('a result: validity first, the criteria, the side-by-side and the pre-registration hash', async () => {
    renderApp('/checks/midcap/20261114T042642805460Z_3ad20e22');
    expect(await screen.findByTestId('midcap-verdict')).toHaveTextContent(
      /unchanged since the run/,
    );
    expect(screen.getByRole('region', { name: 'Validity conditions' })).toHaveTextContent(
      /V1.*V2.*V3/,
    );
    expect(screen.getByRole('region', { name: 'Strategy criteria' })).toHaveTextContent(
      /not measurable/,
    );
    expect(screen.getByRole('region', { name: 'Side by side' })).toHaveTextContent(
      /net Sharpe\s*1\.83\s*-0\.36/,
    );
  });
});
