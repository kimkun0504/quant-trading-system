import { screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { api } from '@/shared/api';
import { renderWithProviders } from '@/test/render';
import { GateStrip } from './GateStrip';
import { RunHeader } from './RunHeader';

const DEMO = '20261104T175159Z_944dd5b4';
const SUSPECT = '20261105T102233Z_c07a55e2';

describe('RunHeader (fixture via MSW)', () => {
  it('shows the verdict line and the manifest', async () => {
    const run = await api.getRun(DEMO);
    renderWithProviders(<RunHeader run={run} />);
    expect(
      await screen.findByRole('heading', { level: 1, name: 'Composite baseline' }),
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        /fails the decile t-stat gate and deflated Sharpe with 4 trials — not yet distinguishable from luck/,
      ),
    ).toBeInTheDocument();
    const manifest = screen.getByLabelText('Run manifest');
    expect(manifest).toHaveTextContent(/Period10 Dec 2019 to 17 Oct 2023, 1,006 days/);
    expect(screen.getByText(DEMO)).toBeInTheDocument();
    expect(screen.queryByRole('note', { name: 'Suspicious result' })).not.toBeInTheDocument();
  });

  it('flags a suspicious result before anything else', async () => {
    const run = await api.getRun(SUSPECT);
    const { container } = renderWithProviders(<RunHeader run={run} />);
    const banner = await screen.findByRole('note', { name: 'Suspicious result' });
    expect(banner).toHaveTextContent(/net Sharpe 2\.\d\d/);
    expect(banner).toHaveTextContent(/signal lag 0/);
    // the banner precedes the title in reading order
    const text = container.textContent;
    expect(text.indexOf('Suspicious result')).toBeLessThan(text.indexOf('Same-close execution'));
  });
});

describe('GateStrip', () => {
  it('lists every gate with its status in text, in pipeline order', async () => {
    const run = await api.getRun(DEMO);
    renderWithProviders(<GateStrip gates={[...run.hurdles.gates].reverse()} />);
    const list = await screen.findByRole('list', { name: 'Hurdle gates' });
    const items = within(list).getAllByRole('listitem');
    expect(items).toHaveLength(9);
    expect(items[0]).toHaveTextContent('Decile t-statfailed');
    expect(items[5]).toHaveTextContent('Deflated Sharpefailed (required)');
    expect(items[6]).toHaveTextContent('PBOskipped');
    expect(items[8]).toHaveTextContent('Plausibilitypassed (required)');
  });
});
