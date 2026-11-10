import { screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import { metricContext } from '@/entities/run';
import { api } from '@/shared/api';
import { renderWithProviders } from '@/test/render';
import { MetricTable } from './MetricTable';

describe('MetricTable (fixture via MSW)', () => {
  it('renders each metric with its value and a note specific to the run', async () => {
    const run = await api.getRun('20261104T175159Z_944dd5b4');
    renderWithProviders(
      <MetricTable
        label="Key metrics"
        ctx={metricContext(run)}
        rows={[
          { key: 'sharpe_net', value: run.summary.sharpe_net },
          { key: 'breakeven_cost_bps', value: run.summary.breakeven_cost_bps },
        ]}
      />,
    );
    const table = await screen.findByRole('table', { name: 'Key metrics' });
    const rows = within(table).getAllByRole('row');
    expect(rows).toHaveLength(3); // header + 2
    const sharpeRow = rows[1];
    if (!sharpeRow) throw new Error('missing row');
    expect(within(sharpeRow).getByRole('rowheader')).toHaveTextContent('Net Sharpe');
    const cells = within(sharpeRow).getAllByRole('cell');
    expect(cells[0]).toHaveTextContent('0.74');
    expect(cells[1]).toHaveTextContent(/clears the phase-1 bar of 0.30/);
    expect(within(cells[1] as HTMLElement).getByRole('link', { name: /Handbook/ })).toHaveAttribute(
      'target',
      '_blank',
    );
  });

  it('explains a term on focus and dismisses it with Escape', async () => {
    const run = await api.getRun('20261104T175159Z_944dd5b4');
    renderWithProviders(
      <MetricTable
        label="m"
        ctx={metricContext(run)}
        rows={[{ key: 'icir', value: run.ic_report.icir }]}
      />,
    );
    const term = await screen.findByRole('button', { name: 'ICIR' });
    expect(term).toHaveAccessibleDescription(/Sharpe ratio of the IC/);
    const user = userEvent.setup();
    await user.tab();
    expect(term).toHaveFocus();
    const tip = screen.getByRole('tooltip');
    expect(tip).toHaveAttribute('data-open', 'true');
    await user.keyboard('{Escape}');
    expect(tip).toHaveAttribute('data-open', 'false');
  });
});
