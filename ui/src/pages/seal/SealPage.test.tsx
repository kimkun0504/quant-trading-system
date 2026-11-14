import { screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { renderApp } from '@/test/render';

describe('Holdout seal page (integration, MSW)', () => {
  it('shows the seal, the inherited family, the unseals and the chip in the shell', async () => {
    renderApp('/seal');
    expect(
      await screen.findByRole('heading', { level: 1, name: 'Holdout seal' }),
    ).toBeInTheDocument();
    expect(await screen.findByText('9 Nov 2023')).toBeInTheDocument();
    expect(screen.getAllByText('pipeline_phase2:c1da477b63d2').length).toBeGreaterThan(0);
    expect(screen.getByText(/final check of the ML candidate/)).toBeInTheDocument();
    const banner = screen.getByRole('banner');
    expect(within(banner).getByText(/Sealed from 9 Nov 2023/)).toBeInTheDocument();
  });

  it('marks runs sealed / unsealed in the experiments list', async () => {
    renderApp('/');
    expect(
      (await screen.findAllByRole('button', { name: /^Seal: Sealed/ })).length,
    ).toBeGreaterThan(0);
    expect(screen.getAllByRole('button', { name: /^Seal: Unsealed/ }).length).toBeGreaterThan(0);
  });
});
