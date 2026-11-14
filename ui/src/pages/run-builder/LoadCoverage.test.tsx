import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import { renderApp } from '@/test/render';

const RANK = 'Universe prefilter: max screened rank';
const slip = () => screen.getByRole('complementary', { name: 'Before launch' });

describe('Run builder: the store prefilter and the load-coverage guard', () => {
  it('a store run defaults to --max-rank 1500; an uncovered rank is refused with the guard’s advice', async () => {
    renderApp('/builder');
    const user = userEvent.setup();
    await user.click(await screen.findByRole('radio', { name: 'Data store' }));
    const rank = await screen.findByLabelText(RANK);
    expect(rank).toHaveValue('1500');
    await waitFor(() =>
      expect(within(slip()).getByLabelText('Equivalent CLI command')).toHaveTextContent(
        /--max-rank 1500/,
      ),
    );
    await user.clear(rank);
    await user.type(rank, '1000');
    await waitFor(() =>
      expect(rank).toHaveAccessibleDescription(/incomplete universe.*Use --max-rank 1500/),
    );
    expect(screen.getByTestId('slip-state')).toHaveTextContent('Needs a fix');
    // an explicit, flagged partial load
    await user.click(screen.getByRole('switch', { name: 'Load a partial universe' }));
    await waitFor(() => expect(screen.getByTestId('slip-state')).toHaveTextContent('Valid'));
    expect(
      within(slip()).getByText(/loading anyway \(flagged on the ledger record\)/),
    ).toBeInTheDocument();
    expect(within(slip()).getByLabelText('Equivalent CLI command')).toHaveTextContent(
      /--allow-partial-coverage/,
    );
  });

  it('runs whose load missed universe names carry a partial-load badge', async () => {
    renderApp('/');
    const badge = await screen.findByRole('button', {
      name: /^Seal: .*Partial load: the store prefilter/,
    });
    expect(badge).toHaveTextContent('Partial load');
  });
});
