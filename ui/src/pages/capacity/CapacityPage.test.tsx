import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import { renderApp } from '@/test/render';

describe('Capacity and risk calibration (integration, MSW)', () => {
  it('capacity: default NAVs, the command, not a trial, queued after confirmation', async () => {
    renderApp('/capacity');
    const user = userEvent.setup();
    expect(await screen.findByLabelText('NAVs (USD)')).toHaveValue('5e6, 5e7, 2.5e8, 1e9');
    await waitFor(() =>
      expect(screen.getByTestId('cli-command')).toHaveTextContent(
        /--navs 5e\+06,5e\+07,2\.5e\+08,1e\+09$/,
      ),
    );
    expect(screen.getAllByText('Not a trial').length).toBeGreaterThan(0);
    await user.clear(screen.getByLabelText('NAVs (USD)'));
    await user.type(screen.getByLabelText('NAVs (USD)'), '5e6, -1');
    expect(screen.getByLabelText('NAVs (USD)')).toHaveAccessibleDescription(/above 0/);
    await user.clear(screen.getByLabelText('NAVs (USD)'));
    await user.type(screen.getByLabelText('NAVs (USD)'), '5e6, 1e9');
    await user.type(screen.getByLabelText('Max days'), '60');
    await user.click(screen.getByRole('button', { name: 'Run capacity curve…' }));
    const dialog = await screen.findByRole('dialog');
    expect(dialog).toHaveTextContent(/No family's trial count changes/);
    await user.click(within(dialog).getByRole('button', { name: 'Run the curve' }));
    expect(await screen.findByText(/^Queued: /)).toBeInTheDocument();
  });

  it('capacity curve: headline, cost split table and the half-Sharpe NAV', async () => {
    renderApp('/capacity/20261114T042605395445Z_feafa843');
    expect(await screen.findByRole('region', { name: 'Capacity points' })).toHaveTextContent(
      /\$1\.00B/,
    );
    expect(screen.getByLabelText('Capacity headline')).toHaveTextContent(
      /beyond the gridthe smallest NAV already loses money/,
    );
  });

  it('risk calibration: the recommendation is a report; adopting it opens the builder as one trial', async () => {
    renderApp('/risk/20261114T042614680785Z_feafa843');
    const rec = await screen.findByTestId('recommendation');
    expect(rec).toHaveTextContent(/Recommended ν\s*0\.00/);
    expect(rec).toHaveTextContent(/one logged trial/);
    expect(within(rec).getByRole('link', { name: /Open a run with ν = 0\.00/ })).toHaveAttribute(
      'href',
      '/builder?set=risk.alignment_nu_scale%3D0',
    );
    expect(screen.getByRole('region', { name: 'Calibration by ν' })).toHaveTextContent(
      /recommended/,
    );
  });

  it('risk calibration launch: the ν list and the command', async () => {
    renderApp('/risk');
    expect(await screen.findByDisplayValue('0, 0.05, 0.25, 1')).toBeInTheDocument();
    await waitFor(() =>
      expect(screen.getByTestId('cli-command')).toHaveTextContent(
        /qts risk calibrate .* --nu 0,0\.05,0\.25,1$/,
      ),
    );
  });
});
