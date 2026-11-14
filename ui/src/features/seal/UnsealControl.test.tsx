import { screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { db } from '@/mocks/db';
import { renderWithProviders } from '@/test/render';
import { UnsealControl } from './UnsealControl';

const usedHash = () =>
  [...db.runs.values()].find((r) => r.summary.seal.state === 'unsealed')?.summary.config_hash ?? '';

describe('UnsealControl', () => {
  it('asks for a typed reason and says it is once per configuration and cannot be undone', async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();
    renderWithProviders(
      <UnsealControl
        configHash="0123456789abcdef"
        phase={2}
        reason=""
        force={false}
        onChange={onChange}
      />,
    );
    await user.click(await screen.findByRole('button', { name: 'Unseal this configuration…' }));
    const dialog = screen.getByRole('dialog', {
      name: 'Unseal the holdout for this configuration?',
    });
    expect(dialog).toHaveTextContent(/allowed once per configuration and cannot be undone/);
    const confirm = within(dialog).getByRole('button', { name: 'Unseal on launch' });
    expect(confirm).toBeDisabled();
    await user.type(
      within(dialog).getByLabelText('Reason (required)'),
      'final check of candidate B',
    );
    expect(confirm).toBeEnabled();
    await user.click(confirm);
    expect(onChange).toHaveBeenCalledWith('final check of candidate B', false);
  });

  it('a second unseal of a used configuration needs the explicit force', async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();
    renderWithProviders(
      <UnsealControl
        configHash={usedHash()}
        phase={2}
        reason=""
        force={false}
        onChange={onChange}
      />,
    );
    expect(await screen.findByText(/holdout was already used/)).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Unseal this configuration…' }));
    const dialog = screen.getByRole('dialog');
    await user.type(within(dialog).getByLabelText('Reason (required)'), 'look again');
    const confirm = within(dialog).getByRole('button', { name: 'Unseal on launch' });
    expect(confirm).toBeDisabled();
    await user.click(
      within(dialog).getByLabelText(/Force a second unseal \(flagged: holdout already used\)/),
    );
    await user.click(confirm);
    expect(onChange).toHaveBeenCalledWith('look again', true);
  });

  it('shows the armed state and the force flag', () => {
    renderWithProviders(
      <UnsealControl configHash="x" phase={1} reason="why" force onChange={() => undefined} />,
    );
    return screen.findByText('Unseal, forced').then(() => {
      expect(
        screen.getByText(/Holdout already used: this configuration was unsealed before/),
      ).toBeInTheDocument();
    });
  });
});
