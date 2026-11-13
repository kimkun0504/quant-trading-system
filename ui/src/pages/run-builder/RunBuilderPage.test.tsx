import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import { db, resetDb } from '@/mocks/db';
import { renderApp } from '@/test/render';

const BASE = '20261104T175159Z_944dd5b4';
const HALF_LIFE = 'Training recency half-life (years), 0 = equal weights';

const slip = () => screen.getByRole('complementary', { name: 'Before launch' });
const diff = () => within(slip()).getByRole('table');

describe('Run builder (integration, MSW)', () => {
  it('starts from the defaults: schema-driven editor, nothing changed, valid', async () => {
    renderApp('/builder');
    expect(await screen.findByRole('heading', { name: /^Common settings/ })).toBeInTheDocument();
    expect(screen.getByTestId('base-line')).toHaveTextContent('SystemConfig defaults');
    // curated labels on top, every schema section below
    expect(screen.getByLabelText(HALF_LIFE)).toHaveValue('0');
    expect(screen.getByLabelText('Prediction horizon (days)')).toHaveValue('20');
    for (const section of [
      'Universe',
      'Alpha model',
      'Alpha model: LightGBM',
      'Portfolio construction',
    ])
      expect(screen.getByText(section, { selector: 'summary span' })).toBeInTheDocument();
    expect(within(slip()).getByText(/Nothing differs from the base yet/)).toBeInTheDocument();
    await waitFor(() => expect(screen.getByTestId('slip-state')).toHaveTextContent('Valid'));
    expect(within(slip()).getByLabelText('Equivalent CLI command')).toHaveTextContent(
      /qts demo \\ --stocks 300 \\ --days 1512 \\ --seed 0/,
    );
    expect(screen.getByTestId('trial-line')).toHaveTextContent(
      /Every new configuration is a new trial/,
    );
    expect(await screen.findByText(/Nothing is queued/)).toBeInTheDocument();
  });

  it('shows inline validation and blocks the launch until it is fixed', async () => {
    renderApp('/builder');
    const user = userEvent.setup();
    const horizon = await screen.findByLabelText('Prediction horizon (days)');
    await user.clear(horizon);
    await user.type(horizon, '900');
    expect(horizon).toHaveAttribute('aria-invalid', 'true');
    expect(horizon).toHaveAccessibleDescription(/Horizon days must be at most 252/);
    expect(screen.getByTestId('slip-state')).toHaveTextContent('Needs a fix');
    expect(within(slip()).getByRole('button', { name: 'Add to queue' })).toBeDisabled();
    expect(within(diff()).getByText('not valid')).toBeInTheDocument();
    // per-field reset restores the base value
    await user.click(
      screen.getByRole('button', { name: /Reset Prediction horizon \(days\) to 20/ }),
    );
    expect(horizon).toHaveValue('20');
    await waitFor(() => expect(screen.getByTestId('slip-state')).toHaveTextContent('Valid'));
    // an error only the API knows (the warm-up floor) lands on its field
    const days = screen.getByLabelText('Days of market data');
    await user.clear(days);
    await user.type(days, '600');
    await waitFor(() => expect(days).toHaveAccessibleDescription(/at least 758/));
    expect(within(slip()).getByRole('button', { name: 'Add to queue' })).toBeDisabled();
  });

  it('searches the full editor, filters to changed fields and handles nullable fields', async () => {
    renderApp('/builder');
    const user = userEvent.setup();
    const search = await screen.findByRole('searchbox', { name: 'Search settings' });
    await user.type(search, 'leaves');
    expect(await screen.findByLabelText('Num leaves')).toHaveValue('15');
    expect(screen.getByText('1 of 30 settings')).toBeInTheDocument();
    await user.clear(screen.getByLabelText('Num leaves'));
    await user.type(screen.getByLabelText('Num leaves'), '31');
    await user.clear(search);
    await user.click(screen.getByRole('checkbox', { name: 'Changed only' }));
    expect(screen.getByText('1 of 30 settings')).toBeInTheDocument();
    expect(screen.getByLabelText('Num leaves')).toHaveValue('31');
    // the API warns that an ML-only key does nothing with the ML sleeve off
    expect(
      await within(slip()).findByText(/only affects the ML sleeve, which is off/),
    ).toBeInTheDocument();
    // a nullable field: "No value" sends null
    await user.click(
      screen.getByRole('checkbox', { name: 'Target volatility (annual): no value' }),
    );
    expect(screen.getAllByLabelText('Target volatility (annual)')[0]).toBeDisabled();
    expect(
      within(diff()).getByRole('row', { name: /portfolio.target_vol_annual 0.1 none/ }),
    ).toBeInTheDocument();
  });

  it('clones a run, diffs two edits, adds a variant and queues both', async () => {
    resetDb({ jobDurationMs: 600 });
    const { router } = renderApp(`/builder?from=${BASE}`);
    const user = userEvent.setup();
    expect(await screen.findByTestId('base-line')).toHaveTextContent('Composite baseline');
    const halfLife = screen.getByLabelText(HALF_LIFE);
    await user.clear(halfLife);
    await user.type(halfLife, '8');
    const aversion = screen.getByLabelText('Trade aversion');
    await user.clear(aversion);
    await user.type(aversion, '0.5');
    expect(
      within(diff()).getByRole('row', { name: /alpha.sample_weight_halflife_years 0 8/ }),
    ).toBeInTheDocument();
    expect(
      within(diff()).getByRole('row', { name: /portfolio.trade_aversion 1.0 0.5/ }),
    ).toBeInTheDocument();
    expect(within(diff()).getAllByRole('row')).toHaveLength(3);
    const cli = within(slip()).getByLabelText('Equivalent CLI command');
    expect(cli).toHaveTextContent(`--config runs/${BASE}/config.yaml`);
    expect(cli).toHaveTextContent('--set alpha.sample_weight_halflife_years=8');
    expect(cli).toHaveTextContent('--set portfolio.trade_aversion=0.5');
    await waitFor(() => expect(screen.getByTestId('slip-state')).toHaveTextContent('Valid'));
    expect(screen.getByTestId('config-hash')).toHaveTextContent(/^[0-9a-f]{12}$/);
    expect(screen.getByTestId('trial-line')).toHaveTextContent(
      /This will be trial 3 in family pipeline_phase1:[0-9a-f]{12}.*\(2 logged so far\)/,
    );

    // a variant: the same edit with half-life 0
    await user.click(screen.getByRole('button', { name: 'Add variant' }));
    expect(screen.getByRole('button', { name: /Variant B/ })).toHaveAttribute(
      'aria-pressed',
      'true',
    );
    await user.clear(screen.getByLabelText(HALF_LIFE));
    await user.type(screen.getByLabelText(HALF_LIFE), '0');
    const row = within(diff()).getByRole('row', { name: /alpha.sample_weight_halflife_years/ });
    expect(
      within(row)
        .getAllByRole('cell')
        .map((c) => c.textContent),
    ).toEqual(['0', '8', '=']);
    expect(screen.getByTestId('trial-line')).toHaveTextContent(/This will be trial 4 in family/);

    const queueAll = screen.getByRole('button', { name: 'Queue all 2 variants' });
    await waitFor(() => expect(queueAll).toBeEnabled());
    await user.click(queueAll);
    expect(await within(slip()).findByText('Added 2 runs to the queue.')).toBeInTheDocument();
    const queue = screen.getByRole('region', { name: 'Run queue' });
    await waitFor(() => {
      expect(within(queue).getAllByText('Running')).toHaveLength(1);
      expect(within(queue).getAllByText('Queued')).toHaveLength(1);
    });
    expect(within(queue).getByRole('progressbar')).toBeInTheDocument();
    expect(screen.getByTestId('queue-indicator')).toHaveAccessibleName(/1 running, 1 queued/);
    expect([...db.jobs.values()].map((r) => r.request.config_overrides)).toEqual([
      { 'alpha.sample_weight_halflife_years': 8, 'portfolio.trade_aversion': 0.5 },
      { 'portfolio.trade_aversion': 0.5 },
    ]);

    // both finish; each links to its dossier and can go to the tray
    const links = await within(queue).findAllByRole(
      'link',
      { name: /Composite baseline/ },
      { timeout: 6000 },
    );
    await waitFor(() => expect(within(queue).getAllByText('Finished')).toHaveLength(2), {
      timeout: 6000,
    });
    expect(links[0]).toHaveAttribute('href', expect.stringMatching(/^\/runs\//));
    await user.click(
      within(queue).getAllByRole('button', { name: /^Add .* to the tray$/ })[0] as HTMLElement,
    );
    expect(await screen.findByText('1 of 8 selected')).toBeInTheDocument();
    expect(router.state.location.search).toContain(`from=${BASE}`);
  }, 20_000);

  it('warns that more than three changed keys is a new strategy', async () => {
    renderApp(
      `/builder?from=${BASE}&set=alpha.horizon_days=10&set=portfolio.trade_aversion=2&set=risk.n_factors=8&set=portfolio.beta_neutral=false`,
    );
    expect(
      await within(await screen.findByRole('complementary', { name: 'Before launch' })).findByText(
        /4 keys differ: this is a new strategy, not a variant: comparisons with the base will not isolate one change/,
      ),
    ).toBeInTheDocument();
    expect(within(diff()).getAllByRole('row')).toHaveLength(5);
  });

  it('asks before a new base discards edits, and reports a base that cannot be used', async () => {
    renderApp('/builder?set=alpha.horizon_days=10');
    const user = userEvent.setup();
    await screen.findByRole('heading', { name: /^Common settings/ });
    await user.click(screen.getByRole('radio', { name: 'An existing run' }));
    await user.type(
      screen.getByRole('combobox', { name: 'Find a run to start from' }),
      'aversion 0',
    );
    await user.click(await screen.findByRole('option', { name: /trade aversion 0/ }));
    const ask = screen.getByRole('alertdialog', { name: 'Discard edits?' });
    expect(ask).toHaveTextContent(/discards the one edit/);
    await user.click(within(ask).getByRole('button', { name: 'Discard and switch' }));
    await waitFor(() =>
      expect(screen.getByTestId('base-line')).toHaveTextContent(
        'Composite baseline, trade aversion 0',
      ),
    );
    expect(screen.getByLabelText('Trade aversion')).toHaveValue('0');
    expect(screen.getByLabelText('Prediction horizon (days)')).toHaveValue('20');
  });

  it('explains a base run that is not finished', async () => {
    renderApp('/builder?from=20261105T113040Z_7d2e9b14');
    expect(await screen.findByText(/Only a finished run can be a base/)).toBeInTheDocument();
    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: 'Start from the defaults' }));
    expect(await screen.findByTestId('base-line')).toHaveTextContent('SystemConfig defaults');
  });
});
