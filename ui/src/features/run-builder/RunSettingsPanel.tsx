import { useId } from 'react';
import type { ReactNode } from 'react';
import { cx } from '@/shared/lib/cx';
import { IconButton, Panel, SegmentedControl } from '@/shared/ui';
import { Icons } from '@/shared/ui/icons';
import type { Base, RunSettingKey, RunSettings } from './lib/draft';
import fieldStyles from './FieldControl.module.css';
import styles from './RunSettingsPanel.module.css';

function Setting({
  label,
  hint,
  error,
  changed,
  baseText,
  onReset,
  children,
}: {
  label: string;
  hint?: ReactNode;
  error?: string | undefined;
  changed: boolean;
  baseText: string;
  onReset: () => void;
  children: (ids: { id: string; describedBy: string; invalid: boolean }) => ReactNode;
}) {
  const id = useId();
  const hintId = `${id}-hint`;
  const errorId = `${id}-error`;
  return (
    <div className={cx(fieldStyles.field, fieldStyles.stack, changed && fieldStyles.changed)}>
      <div className={fieldStyles.head}>
        <label htmlFor={id} className={fieldStyles.label}>
          {label}
        </label>
      </div>
      <div className={fieldStyles.inputRow}>
        {children({
          id,
          describedBy: [hintId, error ? errorId : ''].filter(Boolean).join(' '),
          invalid: Boolean(error),
        })}
        {changed ? (
          <IconButton size="sm" label={`Reset ${label} to ${baseText}`} onClick={onReset}>
            <Icons.Reset />
          </IconButton>
        ) : null}
      </div>
      {error ? (
        <p id={errorId} className={fieldStyles.error}>
          {error}
        </p>
      ) : null}
      <p id={hintId} className={fieldStyles.hint}>
        {changed ? (
          <span className={fieldStyles.was}>
            Base <code>{baseText}</code>.{' '}
          </span>
        ) : null}
        {hint}
      </p>
    </div>
  );
}

/**
 * What a run needs besides SystemConfig: the market it reads (a synthetic one, or a slice
 * of the data store with the universe prefilter), the ML sleeve, seed, signal lag and the
 * operational switches. These are the `qts demo` / `qts run` flags.
 */
export function RunSettingsPanel({
  run,
  base,
  errors,
  storePath,
  minDays,
  autoLabel,
  onChange,
  onReset,
}: {
  run: RunSettings;
  base: Base;
  errors: Partial<Record<RunSettingKey, string>>;
  /** The store the API reads (null: no store configured). */
  storePath: string | null | undefined;
  minDays: number | null | undefined;
  autoLabel: string;
  onChange: <K extends RunSettingKey>(key: K, value: RunSettings[K]) => void;
  onReset: (key: RunSettingKey) => void;
}) {
  const changed = (k: RunSettingKey) => run[k] !== base.run[k];
  const text = (
    key:
      | 'seed'
      | 'stocks'
      | 'days'
      | 'start'
      | 'end'
      | 'maxNames'
      | 'minMcapEver'
      | 'maxDays'
      | 'label',
    label: string,
    hint: ReactNode,
    opts: { placeholder?: string; mode?: 'numeric' | 'decimal' | 'text'; wide?: boolean } = {},
  ) => (
    <Setting
      label={label}
      hint={hint}
      error={errors[key]}
      changed={changed(key)}
      baseText={base.run[key] === '' ? 'none' : base.run[key]}
      onReset={() => onReset(key)}
    >
      {({ id, describedBy, invalid }) => (
        <input
          id={id}
          className={cx(
            fieldStyles.input,
            !opts.wide && fieldStyles.numeric,
            opts.wide && styles.wide,
          )}
          type="text"
          inputMode={opts.mode ?? 'numeric'}
          value={run[key]}
          placeholder={opts.placeholder}
          onChange={(e) => onChange(key, e.target.value)}
          aria-describedby={describedBy}
          aria-invalid={invalid}
          autoComplete="off"
          spellCheck={false}
        />
      )}
    </Setting>
  );
  const toggle = (key: 'ml' | 'cache', label: string, hint: ReactNode) => (
    <Setting
      label={label}
      hint={hint}
      error={errors[key]}
      changed={changed(key)}
      baseText={base.run[key] ? 'on' : 'off'}
      onReset={() => onReset(key)}
    >
      {({ id, describedBy }) => (
        <span className={fieldStyles.switch}>
          <input
            id={id}
            type="checkbox"
            role="switch"
            checked={run[key]}
            onChange={(e) => onChange(key, e.target.checked)}
            aria-describedby={describedBy}
          />
          <span className={fieldStyles.track} aria-hidden="true" />
          <span className={fieldStyles.switchWord}>{run[key] ? 'On' : 'Off'}</span>
        </span>
      )}
    </Setting>
  );

  const store = run.kind === 'store';
  return (
    <Panel
      title="Data and model"
      subtitle="the market the run reads and the switches that are not config keys"
    >
      <div className={styles.kind}>
        <SegmentedControl
          legend="Market data"
          value={run.kind}
          onChange={(v) => onChange('kind', v)}
          options={[
            { value: 'synthetic', label: 'Synthetic market' },
            { value: 'store', label: 'Data store' },
          ]}
        />
        <p className={styles.kindHint}>
          {store ? (
            storePath ? (
              <>
                Point-in-time store at <code>{storePath}</code>
                {base.storePath && base.storePath !== storePath ? (
                  <>
                    {' '}
                    (the base run read <code>{base.storePath}</code>)
                  </>
                ) : null}
                .
              </>
            ) : storePath === null ? (
              'The API has no data store configured (QTS_STORE_PATH); a store run will be refused.'
            ) : (
              'A slice of the point-in-time data store.'
            )
          ) : (
            'Seeded factor market with a planted alpha of known IC: it checks the plumbing, not the strategy.'
          )}
        </p>
      </div>
      <div className={styles.grid}>
        {store ? (
          <>
            {text(
              'start',
              'Start date',
              'First day loaded. Empty: from the first day in the store.',
              {
                placeholder: 'YYYY-MM-DD',
                mode: 'text',
              },
            )}
            {text('end', 'End date', 'Last day loaded. Empty: to the last day in the store.', {
              placeholder: 'YYYY-MM-DD',
              mode: 'text',
            })}
            {text(
              'maxNames',
              'Universe prefilter: max names',
              'Load only the N names with the largest peak market cap in the date range. Empty: all names.',
              { placeholder: 'all' },
            )}
            {text(
              'minMcapEver',
              'Universe prefilter: min peak market cap (USD)',
              'Load only names whose market cap ever reached this in the date range. Empty: no floor.',
              { placeholder: 'none', mode: 'decimal' },
            )}
          </>
        ) : (
          <>
            {text('stocks', 'Stocks', 'Size of the synthetic universe, 50 to 3,000.')}
            {text(
              'days',
              'Days of market data',
              minDays
                ? `At least ${minDays.toLocaleString('en-US')}: the warm-up of this configuration plus one year.`
                : 'The first days are warm-up for the risk and alpha models.',
            )}
          </>
        )}
        {toggle(
          'ml',
          'ML sleeve',
          'Adds the walk-forward LightGBM / MLP / ridge sleeve to the composite. Tested as phase 2, with a higher Sharpe bar, and much slower.',
        )}
        {text('seed', 'Seed', 'Same seed, same market (synthetic) and same models.')}
        <Setting
          label="Signal lag"
          hint="1 trades at the close after the signal. 0 trades at the close that produced it: optimistic, and flagged."
          changed={changed('lag')}
          baseText={`${base.run.lag} day${base.run.lag === '1' ? '' : 's'}`}
          onReset={() => onReset('lag')}
        >
          {({ id, describedBy }) => (
            <select
              id={id}
              className={cx(fieldStyles.input, fieldStyles.numeric)}
              value={run.lag}
              onChange={(e) => onChange('lag', e.target.value === '0' ? '0' : '1')}
              aria-describedby={describedBy}
            >
              <option value="1">1 day</option>
              <option value="0">0 days</option>
            </select>
          )}
        </Setting>
        {text(
          'maxDays',
          'Backtest days',
          'Backtest only the last N out-of-sample days, for a quick look. Empty: all of them.',
          { placeholder: 'all' },
        )}
        {toggle(
          'cache',
          'Stage cache',
          'Reuse fitted risk and alpha models from earlier runs. Changes no result and no config hash.',
        )}
        {text('label', 'Label', `Shown in Experiments. Empty: “${autoLabel}”.`, {
          placeholder: autoLabel,
          mode: 'text',
          wide: true,
        })}
      </div>
    </Panel>
  );
}
