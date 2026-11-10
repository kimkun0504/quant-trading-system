import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { clearSelection, getSelection } from '@/features/selection';
import { renderApp } from '@/test/render';

const DEMO = '20261104T175159Z_944dd5b4';
// eslint-disable-next-line @typescript-eslint/unbound-method -- re-applied with .call(this)
const originalRect = Element.prototype.getBoundingClientRect;
const scrollIntoView = vi.fn();
const TA0 = '20261106T083015Z_5e2c9f10';

describe('Run dossier (integration, MSW)', () => {
  afterEach(() => vi.restoreAllMocks());
  beforeEach(() => {
    clearSelection();
    localStorage.clear();
    Element.prototype.scrollIntoView = scrollIntoView;
    // jsdom has no layout: put every section below the fold so the scrollspy starts at the top
    vi.spyOn(Element.prototype, 'getBoundingClientRect').mockImplementation(function (
      this: Element,
    ) {
      const rect = originalRect.call(this);
      if (this.tagName !== 'SECTION') return rect;
      return new DOMRect(rect.x, 2000, rect.width, rect.height);
    });
  });

  it('renders the sticky header: name, verdict, headline numbers, gates, config hash', async () => {
    renderApp(`/runs/${DEMO}`);
    expect(
      await screen.findByRole('heading', { level: 1, name: 'Composite baseline' }),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/fails the decile t-stat gate and deflated Sharpe with 4 trials/),
    ).toBeInTheDocument();
    const facts = screen.getByLabelText('Headline numbers');
    const fact = (label: string) =>
      within(facts).getByRole('button', { name: label }).closest('div')?.textContent ?? '';
    expect(fact('Net Sharpe')).toMatch(/\+0\.74$/);
    expect(fact('DSR')).toMatch(/0\.665 4 trials$/);
    expect(fact('Invested days')).toMatch(/100%$/);
    expect(fact('Cost/side')).toMatch(/15\.5 bp$/);
    const gates = screen.getByRole('list', { name: 'Hurdle gates' });
    expect(within(gates).getAllByRole('listitem')).toHaveLength(10);
    expect(within(gates).getAllByRole('listitem')[0]).toHaveTextContent('Decile t-stat failed');
    expect(screen.getByText('Gates fail')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Copy config hash' })).toBeInTheDocument();
    expect(screen.queryByRole('note', { name: 'Suspicious result' })).not.toBeInTheDocument();
  });

  it('shows the seven sections in order, each with its lab link, and the section nav', async () => {
    renderApp(`/runs/${DEMO}`);
    await screen.findByRole('heading', { level: 1, name: 'Composite baseline' });
    const names = [
      'Performance',
      'Signal',
      'Portfolio',
      'Costs',
      'Validation',
      'Config',
      'Artifacts',
    ];
    const headings = screen
      .getAllByRole('heading', { level: 2 })
      .filter((h) => h.closest('section[id]')?.getAttribute('aria-labelledby') === h.id);
    expect(headings.map((h) => h.textContent)).toEqual(names.map((n, i) => `${i + 1}${n}`));
    expect(screen.getByRole('link', { name: /Open in Signal lab/ })).toHaveAttribute(
      'href',
      '/signal',
    );
    expect(screen.getByRole('link', { name: /Open in Cost & constraint lab/ })).toHaveAttribute(
      'href',
      '/costs',
    );
    const nav = screen.getByRole('navigation', { name: 'Run sections' });
    expect(within(nav).getAllByRole('link')).toHaveLength(7);
    await waitFor(() =>
      expect(within(nav).getByRole('link', { name: /Performance/ })).toHaveAttribute(
        'aria-current',
        'location',
      ),
    );
    // the content of each section arrives from its own endpoint
    // (SVG charts need a measured width, which jsdom does not give; their frames render)
    expect(await screen.findByRole('img', { name: /Net asset value/ })).toBeInTheDocument();
    expect(await screen.findByText('IC decay')).toBeInTheDocument();
    expect(await screen.findByText('Gross and net exposure')).toBeInTheDocument();
    expect(await screen.findByText('Costs by month')).toBeInTheDocument();
    expect(await screen.findByText('Exposure snapshot')).toBeInTheDocument();
    expect(screen.getByRole('table', { name: /Gates in the order/ })).toBeInTheDocument();
    expect(await screen.findByRole('region', { name: 'Configuration' })).toBeInTheDocument();
    expect(await screen.findByText('weights.parquet')).toBeInTheDocument();
  });

  it('j / k move between sections and focus the heading', async () => {
    renderApp(`/runs/${DEMO}`);
    await screen.findByRole('heading', { level: 1, name: 'Composite baseline' });
    const user = userEvent.setup();
    await user.keyboard('j');
    expect(screen.getByRole('heading', { level: 2, name: /Signal/ })).toHaveFocus();
    expect(scrollIntoView).toHaveBeenCalled();
    // the scrollspy has not moved (jsdom does not scroll), so k goes back to the first
    await user.keyboard('k');
    expect(screen.getByRole('heading', { level: 2, name: /Performance/ })).toHaveFocus();
  });

  it('adds the run to the tray and back out', async () => {
    renderApp(`/runs/${DEMO}`);
    await screen.findByRole('heading', { level: 1, name: 'Composite baseline' });
    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: 'Add to tray' }));
    expect(getSelection()).toEqual([DEMO]);
    expect(screen.getByRole('region', { name: 'Selected runs' })).toHaveTextContent(
      '1 of 8 selected',
    );
    await user.click(screen.getByRole('button', { name: 'Remove from tray' }));
    expect(getSelection()).toEqual([]);
  });

  it('config: search filters the keys; "diff against" reuses the compare config diff', async () => {
    renderApp(`/runs/${DEMO}`);
    await screen.findByRole('heading', { level: 1, name: 'Composite baseline' });
    const config = await screen.findByRole('region', { name: 'Configuration' });
    expect(within(config).getByText('trade_aversion')).toBeInTheDocument();
    const user = userEvent.setup();
    await user.type(screen.getByRole('searchbox', { name: /Search configuration/ }), 'horizon');
    await waitFor(() =>
      expect(
        within(screen.getByRole('region', { name: 'Configuration' })).getAllByRole('row'),
      ).toHaveLength(3),
    );
    expect(screen.getByText('1 of 29 keys')).toBeInTheDocument();

    await user.selectOptions(screen.getByLabelText('Compare with'), TA0);
    const diff = await screen.findByRole('region', { name: 'Config diff' });
    expect(diff).toHaveTextContent('portfolio.trade_aversion');
    expect(
      within(diff)
        .getAllByRole('columnheader')
        .map((h) => h.textContent),
    ).toEqual(['Key', 'Composite baseline', 'Composite baseline, trade aversion 0']);
  });

  it('artifacts: lists the run directory with download links served by the API', async () => {
    renderApp(`/runs/${DEMO}`);
    const row = (await screen.findByText('manifest.json')).closest('tr');
    expect(row).toHaveTextContent('Run manifest');
    expect(row).toHaveTextContent('JSON');
    expect(
      within(row as HTMLElement)
        .getByRole('link', { name: 'Download' })
        .getAttribute('href'),
    ).toMatch(new RegExp(`/api/runs/${DEMO}/files/manifest\\.json$`));
  });

  it('flags a suspicious run before the sections', async () => {
    renderApp('/runs/20261105T102233Z_c07a55e2');
    const banner = await screen.findByRole('note', { name: 'Suspicious result' });
    expect(banner).toHaveTextContent(/net Sharpe 2\.\d\d/);
    expect(banner).toHaveTextContent(/signal lag 0/);
  });

  it('redirects the old tab paths to the section anchors', async () => {
    const { router } = renderApp(`/runs/${DEMO}/alpha`);
    await screen.findByRole('heading', { level: 1, name: 'Composite baseline' });
    expect(router.state.location.pathname).toBe(`/runs/${DEMO}`);
    expect(router.state.location.hash).toBe('#signal');
    expect(screen.getByRole('heading', { level: 2, name: /Signal/ })).toHaveFocus();
  });

  it('says what went wrong for a run that does not exist', async () => {
    renderApp('/runs/nope');
    expect(await screen.findByText(/No run with id nope/)).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Back to Experiments' })).toBeInTheDocument();
  });
});
