import { describe, expect, it } from 'vitest';
import {
  capacityCommand,
  g,
  midcapCommand,
  paperAckCommand,
  paperReconcileCommand,
  paperRunCommand,
  riskCommand,
  signalTestCommand,
  storeFlags,
  sweepCommand,
} from './cli';

/** The same inputs and strings as `api/tests/test_tasks.py::test_task_commands`. */
describe('research command lines (mirror qts_api.services.tasks.task_command)', () => {
  it('formats numbers as Python %g', () => {
    expect(g(5e6)).toBe('5e+06');
    expect(g(2.5e8)).toBe('2.5e+08');
    expect(g(3e8)).toBe('3e+08');
    expect(g(0.05)).toBe('0.05');
    expect(g(0)).toBe('0');
    expect(g(1)).toBe('1');
    expect(g(123456)).toBe('123456');
    expect(g(1234567)).toBe('1.23457e+06');
    expect(g(0.00001)).toBe('1e-05');
  });

  it('signal test', () => {
    expect(
      signalTestCommand({ feature: 'x', horizons: [60, 5], store: { start: '2000-01-01' } }, '/s'),
    ).toBe('qts signal test x --store /s --start 2000-01-01 --horizons 5,60');
    expect(
      signalTestCommand(
        {
          feature: 'x',
          horizons: [5],
          step: 10,
          cost_bps: 7.5,
          use_ml: false,
          cache: false,
          log_trial: false,
        },
        null,
      ),
    ).toBe(
      "qts signal test x --store '<store>' --horizons 5 --step 10 --cost-bps 7.5 --no-ml --no-cache --no-log",
    );
  });

  it('capacity', () => {
    expect(
      capacityCommand({ navs: [2.5e8, 5e6], store: { max_names: 3000, min_mcap_ever: 3e8 } }, '/s'),
    ).toBe('qts capacity --store /s --max-names 3000 --min-mcap-ever 3e+08 --navs 5e+06,2.5e+08');
  });

  it('risk calibration', () => {
    expect(riskCommand({ nu: [0.05, 0], store: {}, use_ml: false }, null)).toBe(
      "qts risk calibrate --store '<store>' --nu 0,0.05 --no-ml",
    );
  });

  it('sweep (a whole number stays whole, as the API receives it)', () => {
    expect(
      sweepCommand(
        {
          base_config: 'configs/baseline.yaml',
          params: [{ key: 'a.b', values: [0.5, 1, 2] }],
          mode: 'zip',
          store: {},
          nav: 5e6,
          max_days: 60,
          use_ml: true,
        },
        '/s',
        'sw_0123456789ab',
      ),
    ).toBe(
      'qts sweep --config configs/baseline.yaml --param a.b=0.5,1,2 --zip --store /s --max-days 60 --id sw_0123456789ab',
    );
  });

  it('mid-cap check: the default rank prefilter, the dry run and the force', () => {
    expect(midcapCommand({ store: { max_rank: 2750 }, cache: true, dry_run: true }, '/s')).toBe(
      'qts check midcap --store /s --max-rank 2750 --cache --dry-run',
    );
    expect(midcapCommand({ store: {}, force: true }, '/s')).toBe(
      'qts check midcap --store /s --max-rank 2750 --cache --force',
    );
    expect(midcapCommand({ store: { max_names: 4000 }, cache: false, nav: 1e7 }, '/s')).toBe(
      'qts check midcap --store /s --max-names 4000 --nav 1e+07',
    );
  });

  it('paper run, reconcile and ack (no --paper-root for the default root)', () => {
    expect(paperRunCommand('baseline', { until: '2026-11-09', no_sync: true }, null)).toBe(
      'qts paper run --name baseline --until 2026-11-09 --no-sync',
    );
    expect(
      paperRunCommand('b', { dry_run: true, redo_last: true, signal_mode: 'single' }, '/x/paper'),
    ).toBe(
      'qts paper run --name b --paper-root /x/paper --dry-run --redo-last --signal-mode single',
    );
    expect(paperReconcileCommand('b', null)).toBe('qts paper reconcile --name b');
    expect(paperAckCommand('b', "checked the fills, it's fine", null)).toBe(
      `qts paper ack --name b --note 'checked the fills, it'"'"'s fine'`,
    );
    expect(paperAckCommand('b', '', null)).toBe(
      "qts paper ack --name b --note '<why it is safe to continue>'",
    );
  });

  it('store flags', () => {
    expect(storeFlags({ start: '2000-01-01', end: '2020-01-01', max_rank: 2750 }, '/s')).toEqual([
      '--store', '/s', '--start', '2000-01-01', '--end', '2020-01-01', '--max-rank', '2750',
    ]); // prettier-ignore
  });
});
