import { screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { renderApp } from '@/test/render';

const BASE = '20261104T175159Z_944dd5b4';

describe('Run builder: what a launch costs in trials', () => {
  it('shows the trial number, the family and the DSR haircut of the family best before launch', async () => {
    renderApp(`/builder?from=${BASE}&set=portfolio.trade_aversion=0.5`);
    const line = await screen.findByTestId('trial-line', {}, { timeout: 4000 });
    expect(line).toHaveTextContent(/This will be trial \d+ in family pipeline_phase\d:/);
    expect(await screen.findByTestId('dsr-haircut', {}, { timeout: 4000 })).toHaveTextContent(
      /The family's current best \(net Sharpe [\d.]+\) goes from DSR [\d.]+ to [\d.]+\./,
    );
  });
});
