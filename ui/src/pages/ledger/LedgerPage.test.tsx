import { screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { renderApp } from '@/test/render';

describe('Trial ledger (integration, MSW)', () => {
  it('greys out records that are not counted and says why', async () => {
    renderApp('/ledger');
    const table = await screen.findByRole('table', { name: /trials and \d+ records not counted/ });
    const ref = within(table).getByRole('button', { name: /^Not counted: A reference re-run/ });
    expect(ref.closest('tr')?.className).toMatch(/notCounted/);
    expect(
      within(table).getByRole('button', { name: /^Not counted: An unseal event/ }),
    ).toBeInTheDocument();
    expect(
      within(table).getByRole('button', { name: /^Not counted: A capacity run/ }),
    ).toBeInTheDocument();
    expect(screen.getAllByText(/more logged, not counted/).length).toBeGreaterThan(0);
  });
});
