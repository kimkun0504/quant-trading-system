import { useMemo, useReducer, useState } from 'react';
import type { ReactNode } from 'react';
import { useQueries, useQueryClient } from '@tanstack/react-query';
import { ApiError, api } from '@/shared/api';
import type { ConfigSchema, ConfigValidation, LaunchRunRequest, Ledger } from '@/shared/api';
import { queueKeys, runKeys, useHealth, useLedger } from '@/entities/run';
import { cx } from '@/shared/lib/cx';
import { useDebounced } from '@/shared/lib/useDebounced';
import { Button, IconButton } from '@/shared/ui';
import { Icons } from '@/shared/ui/icons';
import { CommonSettings, ConfigEditor } from './ConfigEditor';
import { LaunchSlip } from './LaunchSlip';
import type { SlipVariant } from './LaunchSlip';
import { RunSettingsPanel } from './RunSettingsPanel';
import { StartFrom } from './StartFrom';
import { cliCommand } from './lib/cli';
import { autoLabel, diffRows, fieldErrors, runErrors, runSettings, toRequest } from './lib/draft';
import type { Base, Draft, RunSettingKey } from './lib/draft';
import { schemaFields } from './lib/fields';
import type { Field } from './lib/fields';
import { MAX_VARIANTS, builderReducer, initialState, variantLetter } from './lib/reducer';
import type { BuilderState } from './lib/reducer';
import { forecastTrial } from './lib/trials';
import styles from './RunBuilder.module.css';

const RUN_FIELD_BY_LOC: Record<string, RunSettingKey> = {
  days: 'days',
  stocks: 'stocks',
  seed: 'seed',
  ml: 'ml',
  start: 'start',
  end: 'end',
  max_names: 'maxNames',
  min_mcap_ever: 'minMcapEver',
  max_days: 'maxDays',
  label: 'label',
  signal_lag_days: 'lag',
};

/** API issues of a validation, split into per-config-path and per-run-setting messages. */
function serverErrors(v: ConfigValidation | undefined) {
  const fields: Record<string, string> = {};
  const run: Partial<Record<RunSettingKey, string>> = {};
  for (const e of v?.errors ?? []) {
    const [, a, b] = e.loc;
    if (a === 'config_overrides' && typeof b === 'string') fields[b] = sentence(e.msg);
    else if (typeof a === 'string' && RUN_FIELD_BY_LOC[a])
      run[RUN_FIELD_BY_LOC[a]] = sentence(e.msg);
  }
  return { fields, run };
}

function sentence(msg: string): string {
  const s = msg.charAt(0).toUpperCase() + msg.slice(1);
  return /[.!?]$/.test(s) ? s : `${s}.`;
}

interface Derived {
  draft: Draft;
  request: LaunchRunRequest;
  rows: ReturnType<typeof diffRows>;
  clientFieldErrors: Record<string, string>;
  clientRunErrors: Partial<Record<RunSettingKey, string>>;
}

function derive(state: BuilderState, fields: readonly Field[]): Derived[] {
  return state.variants.map((draft) => ({
    draft,
    request: toRequest(draft, state.base, fields),
    rows: diffRows(draft, state.base, fields),
    clientFieldErrors: fieldErrors(draft, fields),
    clientRunErrors: runErrors(runSettings(draft, state.base)),
  }));
}

/** Initial edits from a deep link: `?set=alpha.horizon_days=10&set=…`. */
function seedEdits(state: BuilderState, fields: readonly Field[], seeds: readonly string[]) {
  let s = state;
  const byPath = new Map(fields.map((f) => [f.path, f]));
  for (const item of seeds) {
    const eq = item.indexOf('=');
    const field = eq > 0 ? byPath.get(item.slice(0, eq)) : undefined;
    if (!field) continue;
    const text = item.slice(eq + 1);
    const raw = text === 'null' ? null : field.type === 'boolean' ? text === 'true' : text;
    s = builderReducer(s, { type: 'setValue', field, raw });
  }
  return s;
}

/**
 * The builder's workbench for one base: variants, the editor, and the launch slip.
 * Mounted with `key = base`, so changing the base starts a clean bench.
 */
export function RunBuilder({
  base,
  schema,
  seeds = [],
  onBaseChange,
  aside,
}: {
  /** Shown under the launch slip in the side column (the queue). */
  aside?: ReactNode;
  base: Base;
  schema: ConfigSchema;
  /** `key=value` edits to start with (deep links). */
  seeds?: readonly string[];
  onBaseChange: (runId: string | null) => void;
}) {
  const fields = useMemo(() => schemaFields(schema), [schema]);
  const [state, dispatch] = useReducer(builderReducer, undefined, () =>
    seedEdits(initialState(base), fields, seeds),
  );
  const ledger = useLedger();
  const health = useHealth();
  const qc = useQueryClient();
  const [launching, setLaunching] = useState(false);
  const [status, setStatus] = useState('');
  const [launchErrors, setLaunchErrors] = useState<Record<string, string>>({});

  const derived = useMemo(() => derive(state, fields), [state, fields]);
  // validate what has stopped changing; a variant with client-side errors is not sent
  const settled = useDebounced(derived, 300);
  const validations = useQueries({
    queries: settled.map((d) => {
      const clean =
        Object.keys(d.clientFieldErrors).length === 0 &&
        Object.keys(d.clientRunErrors).length === 0;
      return {
        queryKey: ['config-validate', d.request] as const,
        queryFn: ({ signal }: { signal: AbortSignal }) => api.validateConfig(d.request, signal),
        enabled: clean,
        staleTime: 30_000,
        retry: false,
      };
    }),
  });
  const validationOf = (i: number): { data: ConfigValidation | undefined; pending: boolean } => {
    const d = derived[i];
    const s = settled[i];
    const q = validations[i];
    // a result belongs to the current edit only once the debounce has caught up
    const current = d !== undefined && s !== undefined && d.request === s.request;
    if (!current || !q) return { data: undefined, pending: true };
    if (q.isError && q.error instanceof ApiError)
      return {
        data: {
          valid: false,
          errors: q.error.errors.length
            ? [...q.error.errors]
            : [{ loc: ['body'], msg: q.error.detail, type: 'value_error' }],
          warnings: [],
          config_hash: null,
          base_config_hash: null,
          warmup_days: null,
          min_days: null,
          overrides_applied: 0,
        },
        pending: false,
      };
    return { data: q.data, pending: q.isFetching || (q.data === undefined && !q.isError) };
  };

  const activeIndex = Math.max(
    0,
    state.variants.findIndex((d) => d.id === state.active),
  );
  const active = derived[activeIndex];
  const activeValidation = validationOf(activeIndex).data;

  const slip: SlipVariant[] = derived.map((d, i) => {
    const v = validationOf(i);
    const r = runSettings(d.draft, state.base);
    const minDays = v.data?.min_days;
    const runErrs = runErrors(r, minDays);
    // variants ahead of this one that land in the same family push its trial number
    const ahead = derived
      .slice(0, i)
      .filter(
        (o) =>
          forecastFamily(state.base, o, ledger.data) === forecastFamily(state.base, d, ledger.data),
      ).length;
    return {
      id: d.draft.id,
      rows: d.rows,
      clientErrors: Object.keys(d.clientFieldErrors).length + Object.keys(runErrs).length,
      validation: v.data,
      validating: v.pending,
      cli: cliCommand(d.request, health.data?.store_path ?? state.base.storePath),
      trial: forecastTrial(state.base, d.rows, r.ml, ledger.data, ahead),
      label: d.request.label ?? '',
      launchError: launchErrors[d.draft.id],
    };
  });

  const launch = async (ids: readonly string[]) => {
    setLaunching(true);
    setStatus('');
    const done: string[] = [];
    const errors: Record<string, string> = {};
    for (const d of derived) {
      if (!ids.includes(d.draft.id)) continue;
      try {
        // one at a time, in order: the queue is FIFO and the order is the variants' order
        await api.launchRun(d.request);
        done.push(d.draft.id);
      } catch (e) {
        errors[d.draft.id] =
          e instanceof ApiError
            ? `The API refused this run: ${e.detail}`
            : 'The run could not be queued.';
      }
    }
    setLaunchErrors(errors);
    setLaunching(false);
    if (done.length) {
      dispatch({ type: 'launched', ids: done });
      setStatus(
        done.length === 1 ? 'Added 1 run to the queue.' : `Added ${done.length} runs to the queue.`,
      );
      void qc.invalidateQueries({ queryKey: queueKeys.all });
      void qc.invalidateQueries({ queryKey: runKeys.list() });
    }
  };

  if (!active) return null;
  const server = serverErrors(activeValidation);
  const fieldErrs = { ...server.fields, ...active.clientFieldErrors };
  const run = runSettings(active.draft, state.base);
  const runErrs = { ...server.run, ...runErrors(run, activeValidation?.min_days) };
  const edits = derived.reduce((a, d) => a + d.rows.length, 0);

  return (
    <div className={styles.builder}>
      <div className={styles.top}>
        <StartFrom base={state.base} edits={edits} onChange={onBaseChange} />
      </div>

      <div className={styles.variants} role="group" aria-label="Variants">
        {state.variants.map((d, i) => {
          const n = derived[i]?.rows.length ?? 0;
          const s = slip[i];
          const bad = s
            ? s.clientErrors > 0 || (s.validation ? !s.validation.valid : false)
            : false;
          return (
            <span
              key={d.id}
              className={cx(styles.variant, d.id === state.active && styles.current)}
            >
              <button
                type="button"
                className={styles.variantButton}
                aria-pressed={d.id === state.active}
                onClick={() => dispatch({ type: 'selectVariant', id: d.id })}
              >
                <span className={styles.letter}>Variant {variantLetter(i)}</span>
                <span className={styles.variantMeta}>
                  {n === 0 ? 'base' : `${n} ${n === 1 ? 'change' : 'changes'}`}
                  {bad ? ', needs a fix' : ''}
                </span>
              </button>
              {state.variants.length > 1 ? (
                <IconButton
                  size="sm"
                  label={`Remove variant ${variantLetter(i)}`}
                  onClick={() => dispatch({ type: 'removeVariant', id: d.id })}
                >
                  <Icons.Close />
                </IconButton>
              ) : null}
            </span>
          );
        })}
        <Button
          size="sm"
          onClick={() => dispatch({ type: 'addVariant' })}
          disabled={state.variants.length >= MAX_VARIANTS}
          title="Duplicate this variant, then change what should differ"
        >
          <Icons.Plus /> Add variant
        </Button>
        {active.rows.length > 0 ? (
          <Button size="sm" variant="quiet" onClick={() => dispatch({ type: 'resetVariant' })}>
            Reset variant {variantLetter(activeIndex)} to the base
          </Button>
        ) : null}
        <p className={styles.variantsHint}>
          A variant is a copy of the current edit. Line up the values to compare, for example a
          half-life of 0 and of 8, and queue them together.
        </p>
      </div>

      <div className={styles.columns}>
        <div className={styles.editor}>
          <CommonSettings
            fields={fields}
            draft={active.draft}
            base={state.base}
            errors={fieldErrs}
            onChange={(field, raw) => dispatch({ type: 'setValue', field, raw })}
            onReset={(path) => dispatch({ type: 'resetField', path })}
          />
          <RunSettingsPanel
            run={run}
            base={state.base}
            errors={runErrs}
            storePath={health.data ? health.data.store_path : undefined}
            minDays={activeValidation?.min_days}
            autoLabel={autoLabel(state.base, active.rows)}
            onChange={(key, value) => dispatch({ type: 'setRun', key, value })}
            onReset={(key) => dispatch({ type: 'resetRun', key })}
          />
          <ConfigEditor
            fields={fields}
            schemaSections={schema}
            draft={active.draft}
            base={state.base}
            errors={fieldErrs}
            onChange={(field, raw) => dispatch({ type: 'setValue', field, raw })}
            onReset={(path) => dispatch({ type: 'resetField', path })}
          />
        </div>
        <div className={styles.side}>
          <LaunchSlip
            base={state.base}
            variants={slip}
            active={state.active}
            launching={launching}
            status={status}
            onLaunch={(ids) => void launch(ids)}
          />
          {aside}
        </div>
      </div>
    </div>
  );
}

function forecastFamily(base: Base, d: Derived, ledger: Ledger | undefined): string | null {
  return forecastTrial(base, d.rows, d.request.ml, ledger).family;
}
