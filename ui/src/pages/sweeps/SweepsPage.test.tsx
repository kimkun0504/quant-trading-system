import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import { renderApp } from '@/test/render';

describe('Sweeps (integration, MSW)', () => {
  it('previews members, trials and the DSR haircut, then creates the sweep and opens it', async () => {
    const { router } = renderApp('/sweeps');
    const user = userEvent.setup();
    expect(await screen.findByRole('link', { name: 'sw_8503759dfeb8' })).toBeInTheDocument();
    expect(await screen.findByTestId('member-count')).toHaveTextContent('3 members of at most 12');
    const slip = await screen.findByTestId('trial-cost', {}, { timeout: 3000 });
    expect(slip).toHaveTextContent(/Adds 3 trials to pipeline_phase2:/);
    expect(slip).toHaveTextContent(/DSR of the current best/);
    expect(screen.getByTestId('cli-command')).toHaveTextContent(
      /^qts sweep --config configs\/baseline\.yaml --param portfolio\.trade_aversion=0\.5,1,2 --store \S+ .* --id sw_[0-9a-f]{12}$/,
    );
    await user.click(screen.getByRole('button', { name: 'Review and run…' }));
    const dialog = await screen.findByRole('dialog');
    expect(within(dialog).getByRole('region', { name: 'Members' })).toHaveTextContent(
      /trade_aversion = 2/,
    );
    await user.click(within(dialog).getByRole('button', { name: 'Run 3 members' }));
    await waitFor(() =>
      expect(router.state.location.pathname).toMatch(/^\/sweeps\/sw_[0-9a-f]{12}$/),
    );
  });

  it('refuses more than 12 members and explains zip', async () => {
    renderApp('/sweeps');
    const user = userEvent.setup();
    await user.click(await screen.findByRole('button', { name: 'Add a setting' }));
    const keys = await screen.findAllByLabelText('Setting');
    await user.selectOptions(keys[1] as HTMLElement, 'alpha.horizon_days');
    const values = screen.getAllByLabelText('Values');
    await user.type(values[1] as HTMLElement, '5, 10, 20, 60, 120');
    expect(screen.getByTestId('member-count')).toHaveTextContent(
      /15 members .* a sweep has at most 12/,
    );
    expect(screen.getByRole('button', { name: 'Review and run…' })).toBeDisabled();
    await user.selectOptions(screen.getByLabelText('Combine'), 'zip');
    expect(screen.getByTestId('member-count')).toHaveTextContent(/same number of values/);
  });

  it('a sweep shows every member, PBO with its reading and the resume control', async () => {
    renderApp('/sweeps/sw_8503759dfeb8');
    expect(await screen.findByTestId('pbo')).toHaveTextContent(/0\.73\s*Winner likely overfit/);
    expect(screen.getByRole('region', { name: 'Sweep members' })).toHaveTextContent(
      /best net Sharpe/,
    );
    expect(screen.getByTestId('cli-command')).toHaveTextContent(/--id sw_8503759dfeb8$/);
  });
});
