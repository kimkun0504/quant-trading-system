import { screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { api } from '@/shared/api';
import { renderWithProviders } from '@/test/render';
import { RunHeader } from './RunHeader';

describe('RunHeader with fields an older run did not record', () => {
  it('omits the universe size instead of printing n/a', async () => {
    const run = await api.getRun('20261104T175159Z_944dd5b4');
    renderWithProviders(
      <RunHeader
        run={{
          ...run,
          config: { ...run.config, stocks: null, days: null },
          manifest: { ...run.manifest, git_dirty: null },
        }}
      />,
    );
    const meta = await screen.findByText(/Phase 1, composite, synthetic market/);
    expect(meta).toHaveTextContent('Phase 1, composite, synthetic market, seed 0, signal lag 1');
    expect(meta).not.toHaveTextContent('n/a');
  });
});
