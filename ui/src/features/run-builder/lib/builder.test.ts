import { describe, expect, it } from 'vitest';
import type { ConfigSchema, Ledger, RunDetail } from '@/shared/api';
import { CONFIG_SCHEMA } from '@/mocks/fixtures/configSchema';
import { generateRun } from '@/mocks/fixtures/generate';
import { SEED_SPECS } from '@/mocks/fixtures/runs';
import { cliCommand, cliLines, cliText, shellQuote } from './cli';
import { COMMON_SETTINGS, commonFields } from './common';
import {
  DEFAULT_RUN,
  autoLabel,
  baseFromRun,
  baseValue,
  defaultBase,
  diffRows,
  draftOverrides,
  fieldErrors,
  isChanged,
  parseRaw,
  rawValue,
  runErrors,
  strategyDiffCount,
  toRequest,
} from './draft';
import type { Draft } from './draft';
import { boundsText, fieldSections, matchesField, schemaFields } from './fields';
import type { Field } from './fields';
import { MAX_VARIANTS, builderReducer, initialState, variantLetter } from './reducer';
import { familyFor, forecastTrial } from './trials';

const fields = schemaFields(CONFIG_SCHEMA);
const field = (path: string): Field => {
  const f = fields.find((x) => x.path === path);
  if (!f) throw new Error(`no field ${path}`);
  return f;
};
const draft = (values: Draft['values'] = {}, run: Draft['run'] = {}): Draft => ({
  id: 'v1',
  values,
  run,
});
const baseline = generateRun(SEED_SPECS[0] as (typeof SEED_SPECS)[number]);
const runBase = () => baseFromRun(baseline.detail, baseline.summary);

describe('schema → form mapping', () => {
  it('flattens every scalar leaf with its type, default, bounds and description', () => {
    const f = field('portfolio.trade_aversion');
    expect(f).toMatchObject({
      name: 'trade_aversion',
      title: 'Trade aversion',
      type: 'number',
      section: 'portfolio',
      sectionTitle: 'Portfolio construction',
      default: 1,
      minimum: 0,
      maximum: 10,
      nullable: false,
    });
    expect(f.description).toMatch(/transaction-cost term/);
    expect(field('alpha.lgbm.num_leaves')).toMatchObject({
      type: 'integer',
      section: 'alpha.lgbm',
      sectionTitle: 'Alpha model: LightGBM',
    });
    expect(field('portfolio.book').enum).toEqual(['long_short', 'long_only', '130_30']);
    expect(field('portfolio.beta_neutral').type).toBe('boolean');
    expect(field('cost.impact_coefficient').exclusiveMinimum).toBe(0);
  });

  it('marks nullable fields, with no default when the default is null', () => {
    expect(field('portfolio.target_vol_annual')).toMatchObject({ nullable: true, default: 0.1 });
    const embargo = field('alpha.embargo_days');
    expect(embargo.nullable).toBe(true);
    expect(embargo.default).toBeUndefined();
  });

  it('leaves out what the builder sets itself and the operational cache section', () => {
    const paths = fields.map((f) => f.path);
    expect(paths).not.toContain('seed');
    expect(paths).not.toContain('execution.signal_lag_days');
    expect(paths.some((p) => p.startsWith('cache.'))).toBe(false);
  });

  it('picks up a field the schema gains, with no UI change', () => {
    const grown: ConfigSchema = structuredClone(CONFIG_SCHEMA);
    const alpha = grown.properties.alpha;
    if (alpha?.type !== 'object') throw new Error('alpha section');
    alpha.properties.brand_new_knob = {
      type: 'number',
      title: 'Brand new knob',
      description: 'Added to SystemConfig yesterday.',
      default: 3,
      minimum: 1,
    };
    grown.properties.trading_days_per_year = { type: 'integer', default: 252 };
    const f = schemaFields(grown).find((x) => x.path === 'alpha.brand_new_knob');
    expect(f).toMatchObject({ title: 'Brand new knob', default: 3, minimum: 1, section: 'alpha' });
    const sections = fieldSections(schemaFields(grown), grown);
    expect(sections.at(-1)).toMatchObject({ path: '', title: 'General' });
    expect(sections.at(-1)?.fields.map((x) => x.path)).toEqual(['trading_days_per_year']);
  });

  it('groups by section in schema order, with the section description', () => {
    const sections = fieldSections(fields, CONFIG_SCHEMA);
    expect(sections.map((s) => s.path)).toEqual([
      'universe',
      'alpha',
      'alpha.lgbm',
      'risk',
      'cost',
      'portfolio',
      'validation',
    ]);
    expect(sections.find((s) => s.path === 'alpha.lgbm')?.description).toMatch(/LightGBM/);
    expect(sections.reduce((a, s) => a + s.fields.length, 0)).toBe(fields.length);
  });

  it('describes bounds in words and searches name, path and description', () => {
    expect(boundsText(field('alpha.horizon_days'))).toBe('1 to 252');
    expect(boundsText(field('universe.min_price'))).toBe('at least 0');
    expect(boundsText(field('cost.impact_coefficient'))).toBe('greater than 0 and at most 5');
    expect(boundsText(field('risk.n_factors'))).toBe('');
    expect(matchesField(field('alpha.horizon_days'), 'HORIZON alpha')).toBe(true);
    expect(matchesField(field('alpha.horizon_days'), 'forward return')).toBe(true);
    expect(matchesField(field('alpha.horizon_days'), 'portfolio')).toBe(false);
  });

  it('curates the common settings and skips the ones the schema lacks', () => {
    const common = commonFields(fields);
    expect(common.map((c) => c.path)).toEqual(COMMON_SETTINGS.map((c) => c.path));
    expect(common.find((c) => c.path === 'alpha.sample_weight_halflife_years')?.label).toBe(
      'Training recency half-life (years), 0 = equal weights',
    );
    const older = fields.filter((f) => f.path !== 'alpha.sample_weight_halflife_years');
    expect(commonFields(older).map((c) => c.path)).not.toContain(
      'alpha.sample_weight_halflife_years',
    );
    expect(commonFields(older)).toHaveLength(COMMON_SETTINGS.length - 1);
  });
});

describe('validation states of one input', () => {
  it('accepts values of the right type inside the bounds', () => {
    expect(parseRaw(field('alpha.horizon_days'), ' 10 ')).toEqual({ ok: true, value: 10 });
    expect(parseRaw(field('portfolio.trade_aversion'), '0.5')).toEqual({ ok: true, value: 0.5 });
    expect(parseRaw(field('portfolio.beta_neutral'), false)).toEqual({ ok: true, value: false });
    expect(parseRaw(field('portfolio.book'), 'long_only')).toEqual({
      ok: true,
      value: 'long_only',
    });
    expect(parseRaw(field('portfolio.target_vol_annual'), null)).toEqual({ ok: true, value: null });
  });

  it.each([
    ['alpha.horizon_days', '', /Enter a number \(default 20\)/],
    ['alpha.horizon_days', 'ten', /must be a number/],
    ['alpha.horizon_days', '2.5', /whole number/],
    ['alpha.horizon_days', '0', /at least 1/],
    ['alpha.horizon_days', '900', /at most 252/],
    ['cost.impact_coefficient', '0', /greater than 0/],
    ['portfolio.book', 'short_only', /must be one of long_short, long_only, 130_30/],
    ['portfolio.target_vol_annual', '', /or choose “No value”/],
  ])('rejects %s = %j', (path, raw, message) => {
    const p = parseRaw(field(path), raw);
    expect(p.ok).toBe(false);
    if (!p.ok) expect(p.error).toMatch(message);
  });

  it('rejects null for a field that is not nullable', () => {
    const p = parseRaw(field('portfolio.trade_aversion'), null);
    expect(p).toEqual({ ok: false, error: 'Trade aversion needs a value.' });
  });

  it('validates the launch settings', () => {
    expect(runErrors(DEFAULT_RUN)).toEqual({});
    expect(runErrors({ ...DEFAULT_RUN, stocks: '5', seed: '-1', days: 'x' })).toMatchObject({
      stocks: expect.stringMatching(/between 50 and 3,000/) as string,
      seed: expect.stringMatching(/between 0/) as string,
      days: expect.stringMatching(/whole number/) as string,
    });
    expect(runErrors({ ...DEFAULT_RUN, days: '700' }, 758).days).toMatch(/at least 758/);
    const store = { ...DEFAULT_RUN, kind: 'store' as const, stocks: 'ignored' };
    expect(runErrors(store)).toEqual({});
    expect(
      runErrors({ ...store, start: '2018-01-01', end: '2010-01-01', maxNames: '3' }),
    ).toMatchObject({
      end: 'The end date is before the start date.',
      maxNames: expect.stringMatching(/between 10/) as string,
    });
    expect(runErrors({ ...store, start: '2010' }).start).toMatch(/YYYY-MM-DD/);
    expect(runErrors({ ...store, minMcapEver: '-2' }).minMcapEver).toMatch(/0 or more/);
  });
});

describe('base, values and diff', () => {
  it('takes the base value from the run, else the schema default', () => {
    const horizon10 = generateRun(SEED_SPECS[2] as (typeof SEED_SPECS)[number]);
    const base = baseFromRun(horizon10.detail, horizon10.summary);
    expect(baseValue(base, field('alpha.horizon_days'))).toBe(10);
    expect(baseValue(defaultBase(), field('alpha.horizon_days'))).toBe(20);
    expect(baseValue(defaultBase(), field('alpha.embargo_days'))).toBeNull();
    expect(rawValue(draft(), base, field('alpha.horizon_days'))).toBe('10');
    expect(rawValue(draft({ 'alpha.horizon_days': '5' }), base, field('alpha.horizon_days'))).toBe(
      '5',
    );
  });

  it('loads the data source and switches of the base run', () => {
    const base = runBase();
    expect(base).toMatchObject({
      runId: baseline.detail.id,
      configHash: baseline.detail.manifest.config_hash,
      sourceRecorded: true,
    });
    expect(base.run).toMatchObject({
      kind: 'synthetic',
      stocks: '300',
      days: '1512',
      seed: '0',
      ml: false,
      lag: '1',
    });
    const store = generateRun(SEED_SPECS[6] as (typeof SEED_SPECS)[number]);
    const sb = baseFromRun(store.detail, store.summary);
    expect(sb.run).toMatchObject({
      kind: 'store',
      start: '2001-01-02',
      maxNames: '1500',
      ml: true,
    });
    expect(sb.storePath).toBe('data/store');
    // an older run that did not record its source: the builder says so and keeps defaults
    const old: RunDetail = structuredClone(baseline.detail);
    old.config.source = { ...old.config.source, recorded: false, stocks: null, days: null };
    old.config.stocks = null;
    old.config.days = null;
    const ob = baseFromRun(old, undefined);
    expect(ob.sourceRecorded).toBe(false);
    expect(ob.run).toMatchObject({ stocks: '300', days: '1512' });
    expect(ob.leaves).toEqual({});
  });

  it('lists only the keys that differ from the base, config first', () => {
    const base = runBase();
    const d = draft(
      {
        'portfolio.trade_aversion': '0.5',
        'alpha.sample_weight_halflife_years': '8',
        'alpha.horizon_days': '20', // equal to the base: not a difference
        'portfolio.target_vol_annual': null,
      },
      { ml: true, seed: '3', days: '2000', cache: true, label: 'my label' },
    );
    const rows = diffRows(d, base, fields);
    expect(rows.map((r) => [r.key, r.group, r.from, r.to])).toEqual([
      ['alpha.sample_weight_halflife_years', 'config', 0, 8],
      ['portfolio.target_vol_annual', 'config', 0.1, null],
      ['portfolio.trade_aversion', 'config', 1, 0.5],
      ['ml', 'run', false, true],
      ['seed', 'run', '0', '3'],
      ['data.days', 'data', '1512', '2000'],
    ]);
    expect(strategyDiffCount(rows)).toBe(4); // three config keys and the ML switch
    expect(isChanged(d, base, field('alpha.horizon_days'))).toBe(false);
    expect(draftOverrides(d, base, fields)).toEqual({
      'alpha.sample_weight_halflife_years': 8,
      'portfolio.target_vol_annual': null,
      'portfolio.trade_aversion': 0.5,
    });
  });

  it('keeps an unparseable edit in the diff, flagged, and out of the request', () => {
    const d = draft({ 'alpha.horizon_days': 'soon' });
    const rows = diffRows(d, defaultBase(), fields);
    expect(rows).toEqual([
      {
        key: 'alpha.horizon_days',
        group: 'config',
        from: 20,
        to: 'soon',
        error: 'Horizon days must be a number.',
      },
    ]);
    expect(fieldErrors(d, fields)).toEqual({
      'alpha.horizon_days': 'Horizon days must be a number.',
    });
    expect(draftOverrides(d, defaultBase(), fields)).toEqual({});
  });

  it('ignores data settings of the other market kind', () => {
    const rows = diffRows(draft({}, { kind: 'store', start: '2010-01-01' }), defaultBase(), fields);
    expect(rows.map((r) => r.key)).toEqual(['data.kind', 'data.start']);
  });

  it('builds the launch request', () => {
    const base = runBase();
    const req = toRequest(
      draft({ 'alpha.sample_weight_halflife_years': '8' }, { maxDays: '250' }),
      base,
      fields,
    );
    expect(req).toEqual({
      kind: 'synthetic',
      ml: false,
      phase: 1,
      stocks: 300,
      days: 1512,
      seed: 0,
      signal_lag_days: 1,
      config_overrides: { 'alpha.sample_weight_halflife_years': 8 },
      label: 'Composite baseline, sample weight halflife years 8, max days 250',
      base_run_id: baseline.detail.id,
      max_days: 250,
      cache: false,
    });
    const store = toRequest(
      draft(
        {},
        {
          kind: 'store',
          start: '2010-01-01',
          maxNames: '500',
          minMcapEver: '2e9',
          ml: true,
          label: ' Store run ',
        },
      ),
      defaultBase(),
      fields,
    );
    expect(store).toMatchObject({
      kind: 'store',
      phase: 2,
      ml: true,
      start: '2010-01-01',
      end: null,
      max_names: 500,
      min_mcap_ever: 2e9,
      base_run_id: null,
      label: 'Store run',
    });
  });

  it('labels a variant by what changed', () => {
    const base = runBase();
    const rows = (v: Draft['values'], run: Draft['run'] = {}) =>
      diffRows(draft(v, run), base, fields);
    expect(autoLabel(base, rows({}))).toBe('Composite baseline, re-run');
    expect(autoLabel(base, rows({ 'portfolio.beta_neutral': false }))).toBe(
      'Composite baseline, beta neutral off',
    );
    expect(
      autoLabel(
        base,
        rows({ 'alpha.horizon_days': '5', 'portfolio.trade_aversion': '2', 'risk.n_factors': '8' }),
      ),
    ).toBe('Composite baseline, 3 changes');
    expect(autoLabel(defaultBase(), diffRows(draft({}, { ml: true }), defaultBase(), fields))).toBe(
      'Composite + ML, ml on',
    );
    expect(autoLabel(defaultBase(), [])).toBe('Composite');
  });
});

describe('CLI command builder', () => {
  const base = runBase();
  it('writes qts demo with --config of the base and one --set per changed key', () => {
    const req = toRequest(
      draft(
        {
          'alpha.sample_weight_halflife_years': '8',
          'portfolio.target_vol_annual': null,
          'portfolio.beta_neutral': false,
        },
        { lag: '0', cache: true, maxDays: '120' },
      ),
      base,
      fields,
    );
    expect(cliText(cliCommand(req, null))).toBe(
      `qts demo --stocks 300 --days 1512 --seed 0 --config runs/${baseline.detail.id}/config.yaml ` +
        '--signal-lag 0 --max-days 120 --cache --set alpha.sample_weight_halflife_years=8 ' +
        '--set portfolio.target_vol_annual=null --set portfolio.beta_neutral=false',
    );
  });

  it('writes qts run --store with the slice, the prefilter and the seed as a config key', () => {
    const req = toRequest(
      draft(
        { 'alpha.horizon_days': '10' },
        {
          kind: 'store',
          start: '2010-01-01',
          end: '2017-12-31',
          maxNames: '1500',
          minMcapEver: '2000000000',
          ml: true,
          seed: '3',
        },
      ),
      defaultBase(),
      fields,
    );
    expect(cliText(cliCommand(req, '/data/my store'))).toBe(
      "qts run --store '/data/my store' --start 2010-01-01 --end 2017-12-31 --max-names 1500 " +
        '--min-mcap-ever 2e+09 --ml --set seed=3 --set alpha.horizon_days=10',
    );
    expect(cliText(cliCommand({ ...req, seed: 0, config_overrides: {} }, null))).toBe(
      'qts run --store <store> --start 2010-01-01 --end 2017-12-31 --max-names 1500 --min-mcap-ever 2e+09 --ml'.replace(
        '<store>',
        "'<store>'",
      ),
    );
  });

  it('is plain qts demo for the untouched defaults', () => {
    const parts = cliCommand(toRequest(draft(), defaultBase(), fields), null);
    expect(cliText(parts)).toBe('qts demo --stocks 300 --days 1512 --seed 0');
    expect(cliLines(parts)).toEqual(['qts demo', '--stocks 300', '--days 1512', '--seed 0']);
  });

  it('quotes only what the shell needs quoted', () => {
    expect(shellQuote('alpha.horizon_days=10')).toBe('alpha.horizon_days=10');
    expect(shellQuote("it's here")).toBe(`'it'"'"'s here'`);
  });
});

describe('builder reducer', () => {
  const h = field('alpha.horizon_days');
  it('stores only what differs from the base', () => {
    let s = initialState(runBase());
    s = builderReducer(s, { type: 'setValue', field: h, raw: '10' });
    expect(s.variants[0]?.values).toEqual({ 'alpha.horizon_days': '10' });
    s = builderReducer(s, { type: 'setValue', field: h, raw: '20.0' }); // back to the base value
    expect(s.variants[0]?.values).toEqual({});
    s = builderReducer(s, { type: 'setValue', field: h, raw: 'x' }); // an invalid edit is kept
    expect(s.variants[0]?.values).toEqual({ 'alpha.horizon_days': 'x' });
    s = builderReducer(s, { type: 'resetField', path: h.path });
    expect(s.variants[0]?.values).toEqual({});
    s = builderReducer(s, { type: 'setRun', key: 'ml', value: true });
    expect(s.variants[0]?.run).toEqual({ ml: true });
    s = builderReducer(s, { type: 'setRun', key: 'ml', value: false });
    expect(s.variants[0]?.run).toEqual({});
    s = builderReducer(s, { type: 'setRun', key: 'seed', value: '4' });
    s = builderReducer(s, { type: 'resetRun', key: 'seed' });
    expect(s.variants[0]?.run).toEqual({});
  });

  it('adds a variant as a copy of the current edit and selects it', () => {
    let s = initialState();
    s = builderReducer(s, { type: 'setValue', field: h, raw: '10' });
    s = builderReducer(s, { type: 'setRun', key: 'label', value: 'First' });
    s = builderReducer(s, { type: 'addVariant' });
    expect(s.variants.map((d) => d.id)).toEqual(['v1', 'v2']);
    expect(s.active).toBe('v2');
    expect(s.variants[1]?.values).toEqual({ 'alpha.horizon_days': '10' });
    expect(s.variants[1]?.run).toEqual({}); // the copy gets its own label
    s = builderReducer(s, { type: 'setValue', field: h, raw: '5' });
    expect(s.variants.map((d) => d.values['alpha.horizon_days'])).toEqual(['10', '5']);
    s = builderReducer(s, { type: 'selectVariant', id: 'v1' });
    s = builderReducer(s, { type: 'addVariant' }); // inserted after the current one
    expect(s.variants.map((d) => d.id)).toEqual(['v1', 'v3', 'v2']);
    s = builderReducer(s, { type: 'selectVariant', id: 'nope' });
    expect(s.active).toBe('v3');
    expect([0, 1, 2].map(variantLetter)).toEqual(['A', 'B', 'C']);
  });

  it('removes variants but never the last, and caps their number', () => {
    let s = initialState();
    for (let i = 0; i < 20; i++) s = builderReducer(s, { type: 'addVariant' });
    expect(s.variants).toHaveLength(MAX_VARIANTS);
    s = builderReducer(s, { type: 'removeVariant', id: s.active });
    expect(s.variants).toHaveLength(MAX_VARIANTS - 1);
    expect(s.variants.some((d) => d.id === s.active)).toBe(true);
    while (s.variants.length > 1)
      s = builderReducer(s, { type: 'removeVariant', id: s.variants[0]?.id ?? '' });
    const only = s.variants[0]?.id ?? '';
    expect(builderReducer(s, { type: 'removeVariant', id: only }).variants).toHaveLength(1);
  });

  it('clears launched variants, keeping one to continue from, and resets on a new base', () => {
    let s = initialState();
    s = builderReducer(s, { type: 'setValue', field: h, raw: '10' });
    s = builderReducer(s, { type: 'addVariant' });
    s = builderReducer(s, { type: 'addVariant' });
    const some = builderReducer(s, { type: 'launched', ids: ['v1', 'v2'] });
    expect(some.variants.map((d) => d.id)).toEqual(['v3']);
    expect(some.active).toBe('v3');
    const all = builderReducer(s, { type: 'launched', ids: ['v1', 'v2', 'v3'] });
    expect(all.variants.map((d) => d.id)).toEqual(['v3']);
    expect(all.variants[0]?.values).toEqual({ 'alpha.horizon_days': '10' });
    const reset = builderReducer(s, { type: 'resetVariant' });
    expect(reset.variants.find((d) => d.id === reset.active)?.values).toEqual({});
    const rebased = builderReducer(s, { type: 'loadBase', base: runBase() });
    expect(rebased.variants).toEqual([{ id: 'v1', values: {}, run: {} }]);
    expect(rebased.base.runId).toBe(baseline.detail.id);
  });
});

describe('trial forecast', () => {
  const base = runBase();
  const dataHash = baseline.summary.data_hash;
  const family = `pipeline_phase1:${dataHash.slice(0, 12)}`;
  const ledger: Ledger = {
    trials: [],
    families: [{ family, n_trials: 5, effective_trials: 3, var_sr_annual: 0.2 }],
  };
  it('counts the next trial in the base run’s family', () => {
    expect(forecastTrial(base, [], false, ledger)).toEqual({
      family,
      nTrials: 5,
      trialNumber: 6,
      unknownReason: '',
    });
    expect(forecastTrial(base, [], false, ledger, 2).trialNumber).toBe(8); // third variant in line
    // the ML sleeve is phase 2: another family on the same data, not in the ledger yet
    expect(forecastTrial(base, [], true, ledger)).toMatchObject({
      family: `pipeline_phase2:${dataHash.slice(0, 12)}`,
      nTrials: 0,
      trialNumber: 1,
    });
  });

  it('follows the family the ledger recorded for the base run’s data', () => {
    const keyed: Ledger = {
      trials: [
        {
          id: 't1',
          ts: '',
          name: '',
          family: 'pipeline_phase1:marketabc123',
          config_hash: 'x',
          data_hash: dataHash,
          sharpe_annual: 1,
          n_obs: 1,
          mean_ic: null,
          n_reruns: 0,
        },
      ],
      families: [
        {
          family: 'pipeline_phase1:marketabc123',
          n_trials: 9,
          effective_trials: 4,
          var_sr_annual: 0.1,
        },
      ],
    };
    const older = { ...base, ledgerFamily: null }; // a manifest written before it recorded the family
    expect(familyFor(older, false, keyed)).toBe('pipeline_phase1:marketabc123');
    expect(familyFor(older, true, keyed)).toBe('pipeline_phase2:marketabc123');
    expect(forecastTrial(older, [], false, keyed).trialNumber).toBe(10);
    // the manifest's own family wins
    const recorded = { ...base, ledgerFamily: 'pipeline_phase1:fromManifest' };
    expect(familyFor(recorded, true, keyed)).toBe('pipeline_phase2:fromManifest');
  });

  it('does not guess when the data is unknown or changed', () => {
    expect(forecastTrial(defaultBase(), [], false, ledger)).toMatchObject({
      family: null,
      trialNumber: null,
      unknownReason: expect.stringMatching(/Start from a run/) as string,
    });
    const changed = diffRows(draft({}, { days: '2000' }), base, fields);
    expect(forecastTrial(base, changed, false, ledger)).toMatchObject({
      family: null,
      unknownReason: expect.stringMatching(/data differs/) as string,
    });
  });
});
