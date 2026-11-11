import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it } from 'vitest';
import { clearSelection, getSelection, setSelection } from '@/features/selection';
import { renderApp } from '@/test/render';

const BASE = '20261104T175159Z_944dd5b4';
const TA0 = '20261106T083015Z_5e2c9f10';
const ML = '20261105T091402Z_3be1a7c0';
const THREE = [BASE, TA0, ML];

/**
 * The three labs through the real route tree against the MSW fixtures: the picker with an
 * empty tray, the deep view with one run, lanes with three. jsdom has no layout, so SVG
 * charts measure 0 wide and render their frames, readouts and data tables only.
 */
describe('Labs (integration, MSW)', () => {
  beforeEach(() => {
    clearSelection();
    localStorage.clear();
  });

  it('Signal lab: the picker fills the tray and the lab opens on the run', async () => {
    renderApp('/signal');
    expect(
      await screen.findByRole('heading', { level: 1, name: 'Signal lab' }),
    ).toBeInTheDocument();
    const picker = await screen.findByRole('region', { name: 'Runs to pick from' });
    const box = await within(picker).findByRole('checkbox', { name: 'Study Composite baseline' });
    await userEvent.setup().click(box);
    expect(getSelection()).toEqual([BASE]);
    expect(await screen.findByText('IC by theme by year')).toBeInTheDocument();
    expect(await screen.findByText('Learned weights over time')).toBeInTheDocument();
    expect(screen.getByText('No ML sleeve in this run')).toBeInTheDocument();
    expect(await screen.findByText('Theme weights over time')).toBeInTheDocument();
    expect(await screen.findByText('Feature IC')).toBeInTheDocument();
    expect(screen.getByText('Rank IC with its trailing one-year mean')).toBeInTheDocument();
  });

  it('Signal lab with three runs: one lane per run, overlays once, a sleeve switch on the ML run', async () => {
    setSelection(THREE);
    renderApp('/signal');
    await waitFor(() => expect(screen.getAllByText('IC by theme by year')).toHaveLength(3));
    expect(await screen.findByText('Trailing one-year mean rank IC')).toBeInTheDocument();
    expect(screen.getAllByText('IC decay by horizon')).toHaveLength(1);
    const sleeve = await screen.findByRole('group', { name: 'Sleeve' });
    expect(within(sleeve).getByRole('radio', { name: 'Ensemble' })).toBeChecked();
    await userEvent.setup().click(within(sleeve).getByRole('radio', { name: 'Ridge' }));
    expect(within(sleeve).getByRole('radio', { name: 'Ridge' })).toBeChecked();
    expect(screen.getAllByText('No ML sleeve in this run')).toHaveLength(2);
    // the heatmap's text alternative lists every cell
    const user = userEvent.setup();
    const [firstTable] = screen.getAllByRole('button', { name: 'Show data table' });
    if (!firstTable) throw new Error('no data table button');
    await user.click(firstTable);
    const table = await screen.findByRole('region', { name: /Mean rank IC by sleeve and year/ });
    expect(within(table).getByRole('rowheader', { name: 'Value' })).toBeInTheDocument();
  });

  it('Portfolio anatomy with one run: book, industry, style, holding period, vol and a holdings scrubber', async () => {
    setSelection([TA0]);
    renderApp('/portfolio');
    expect(
      await screen.findByRole('heading', { level: 1, name: 'Portfolio anatomy' }),
    ).toBeInTheDocument();
    expect(await screen.findByText('Gross, net and turnover')).toBeInTheDocument();
    expect(
      await screen.findByText('Exposure by industry', {}, { timeout: 8000 }),
    ).toBeInTheDocument();
    expect(await screen.findByText('Exposure by style')).toBeInTheDocument();
    expect(screen.getByText('Effective holding period')).toBeInTheDocument();
    expect(screen.getByText('Realised against forecast risk')).toBeInTheDocument();
    const select = await screen.findByRole('combobox', { name: 'Holdings date' });
    const options = within(select).getAllByRole('option');
    expect(options.length).toBeGreaterThan(2);
    const before = screen.getByText(/Every position on/).textContent;
    const [firstOption] = options;
    if (!firstOption) throw new Error('no holdings dates');
    await userEvent.setup().selectOptions(select, firstOption);
    await waitFor(() => expect(screen.getByText(/Every position on/).textContent).not.toBe(before));
    expect(screen.getByRole('slider', { name: 'Month end' })).toHaveValue('0');
    expect(screen.getByText('Largest longs')).toBeInTheDocument();
    expect(screen.getByText('Largest shorts')).toBeInTheDocument();
  });

  it('Portfolio anatomy with three runs: three lanes and one holding-period overlay', async () => {
    setSelection(THREE);
    renderApp('/portfolio');
    await waitFor(() => expect(screen.getAllByText('Gross, net and turnover')).toHaveLength(3));
    await waitFor(() => expect(screen.getAllByText('Holdings')).toHaveLength(3), { timeout: 8000 });
    expect(screen.getAllByText('Effective holding period')).toHaveLength(1);
    expect(screen.getAllByText('Realised against forecast risk')).toHaveLength(3);
  });

  it('Cost & constraint lab with one run: binding grid, TC, participation, vol reason, costs, distribution, sensitivity', async () => {
    setSelection([BASE]);
    renderApp('/costs');
    expect(
      await screen.findByRole('heading', { level: 1, name: 'Cost & constraint lab' }),
    ).toBeInTheDocument();
    expect(await screen.findByText('Binding constraints over time')).toBeInTheDocument();
    expect(await screen.findByText('Transfer coefficient over time')).toBeInTheDocument();
    expect(screen.getByText('ADV participation cap usage')).toBeInTheDocument();
    expect(await screen.findByText('Why the vol target was or was not met')).toBeInTheDocument();
    expect(await screen.findByText('Costs by month and kind')).toBeInTheDocument();
    expect(await screen.findByText('Cost per side, day by day')).toBeInTheDocument();
    expect(await screen.findByText('Cost sensitivity and breakeven')).toBeInTheDocument();
    expect(screen.getByText(/the gate asks for 2×/)).toBeInTheDocument();
    // the binding share table carries every constraint
    const user = userEvent.setup();
    const [firstTable] = screen.getAllByRole('button', { name: 'Show data table' });
    if (!firstTable) throw new Error('no data table button');
    await user.click(firstTable);
    const table = await screen.findByRole('region', {
      name: /Share of solves where each constraint bound/,
    });
    expect(within(table).getByRole('rowheader', { name: 'ADV participation' })).toBeInTheDocument();
    expect(within(table).getByRole('rowheader', { name: 'Position cap' })).toBeInTheDocument();
  });

  it('Cost & constraint lab with three runs: three lanes, overlays once', async () => {
    setSelection(THREE);
    renderApp('/costs');
    await waitFor(() =>
      expect(screen.getAllByText('Binding constraints over time')).toHaveLength(3),
    );
    await waitFor(() =>
      expect(screen.getAllByText('Cost sensitivity and breakeven')).toHaveLength(3),
    );
    expect(screen.getAllByText('Transfer coefficient over time')).toHaveLength(1);
    expect(screen.getAllByText('ADV participation cap usage')).toHaveLength(1);
  });

  it('a lab whose tray holds only an unfinished run says so', async () => {
    setSelection(['20261105T113040Z_7d2e9b14']);
    renderApp('/signal');
    expect(
      await screen.findByText('None of the selected runs has results yet'),
    ).toBeInTheDocument();
  });
});
