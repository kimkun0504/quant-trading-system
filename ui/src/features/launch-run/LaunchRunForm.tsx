import { useId, useMemo, useRef, useState } from 'react';
import type { ReactNode, SyntheticEvent } from 'react';
import type { Phase, RunKind } from '@/shared/api';
import { groupFields, overrideFields } from './schemaFields';
import type { OverrideField } from './schemaFields';
import { useConfigSchema, useLaunchRun } from '@/entities/run';
import { Button } from '@/shared/ui';
import { ErrorState } from '@/shared/ui/ErrorState';
import { DEFAULT_VALUES, toRequest, validateLaunch, WARMUP_DAYS } from './validation';
import type { FieldName, LaunchErrors, LaunchFormValues } from './validation';
import styles from './LaunchRunForm.module.css';

const FIELD_LABELS: Record<string, string> = {
  stocks: 'Stocks',
  days: 'Days of market data',
  seed: 'Seed',
  ml: 'ML sleeve',
};

export function LaunchRunForm({
  onLaunched,
}: {
  onLaunched: (jobId: string, runId: string) => void;
}) {
  const [values, setValues] = useState<LaunchFormValues>(DEFAULT_VALUES);
  const [errors, setErrors] = useState<LaunchErrors>({});
  const [submitted, setSubmitted] = useState(false);
  const summaryRef = useRef<HTMLDivElement | null>(null);
  const schema = useConfigSchema();
  const launch = useLaunchRun();
  const fields = useMemo(() => overrideFields(schema.data), [schema.data]);
  const groups = useMemo(() => groupFields(fields), [fields]);
  const idBase = useId();
  const fid = (name: string) => `${idBase}-${name.replace(/[^a-z0-9]/gi, '-')}`;

  const set = <K extends keyof LaunchFormValues>(k: K, v: LaunchFormValues[K]) => {
    const next = { ...values, [k]: v };
    setValues(next);
    if (submitted) setErrors(validateLaunch(next, fields));
  };
  const setOverride = (path: string, v: string | boolean) => {
    const next = { ...values, overrides: { ...values.overrides, [path]: v } };
    setValues(next);
    if (submitted) setErrors(validateLaunch(next, fields));
  };

  const onSubmit = (e: SyntheticEvent<HTMLFormElement>) => {
    e.preventDefault();
    setSubmitted(true);
    const errs = validateLaunch(values, fields);
    setErrors(errs);
    if (Object.keys(errs).length) {
      requestAnimationFrame(() => summaryRef.current?.focus());
      return;
    }
    launch.mutate(toRequest(values, fields), { onSuccess: (r) => onLaunched(r.job_id, r.run_id) });
  };

  const errorEntries = Object.entries(errors) as [FieldName, string][];
  const describedBy = (name: FieldName, hint?: string) =>
    [hint, errors[name] ? `${fid(name)}-error` : undefined].filter(Boolean).join(' ') || undefined;

  return (
    <form className={styles.form} onSubmit={onSubmit} noValidate aria-label="Launch a run">
      {errorEntries.length ? (
        <div className={styles.summary} role="alert" tabIndex={-1} ref={summaryRef}>
          <h2 className={styles.summaryTitle}>
            {errorEntries.length === 1
              ? 'One field needs a change'
              : `${errorEntries.length} fields need a change`}
          </h2>
          <ul>
            {errorEntries.map(([name, msg]) => (
              <li key={name}>
                <a href={`#${fid(name)}`}>
                  {FIELD_LABELS[name] ??
                    fields.find((f) => `override:${f.path}` === name)?.title ??
                    name}
                  : {msg}
                </a>
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      <Choice
        legend="Market data"
        name="kind"
        value={values.kind}
        onChange={(v) => set('kind', v as RunKind)}
        options={[
          {
            value: 'synthetic',
            label: 'Synthetic market',
            hint: 'Seeded factor structure with a planted alpha of known IC. Validates plumbing.',
          },
          {
            value: 'store',
            label: 'Data store',
            hint: 'Point-in-time Parquet store (Norgate / Sharadar).',
          },
        ]}
      />

      <Choice
        legend="Phase"
        name="phase"
        value={String(values.phase)}
        onChange={(v) => {
          const phase = Number(v) as Phase;
          const next = { ...values, phase, ml: phase === 2 ? true : values.ml };
          setValues(next);
          if (submitted) setErrors(validateLaunch(next, fields));
        }}
        options={[
          {
            value: '1',
            label: 'Phase 1: composite',
            hint: 'Six-theme rank composite. Net Sharpe bar 0.3.',
          },
          {
            value: '2',
            label: 'Phase 2: composite + ML',
            hint: 'Adds the walk-forward ML sleeve. Net Sharpe bar 0.5.',
          },
        ]}
      />

      <div className={styles.field}>
        <div className={styles.check}>
          <input
            id={fid('ml')}
            type="checkbox"
            checked={values.ml}
            onChange={(e) => set('ml', e.target.checked)}
            aria-invalid={Boolean(errors.ml)}
            aria-describedby={describedBy('ml', `${fid('ml')}-hint`)}
          />
          <label htmlFor={fid('ml')}>Train the ML sleeve (LightGBM / MLP / ridge)</label>
        </div>
        <p id={`${fid('ml')}-hint`} className={styles.hint}>
          Walk-forward with label embargo; adds several minutes per refit.
        </p>
        <FieldError id={`${fid('ml')}-error`} message={errors.ml} />
      </div>

      <div className={styles.row}>
        <NumberField
          id={fid('stocks')}
          label="Stocks"
          value={values.stocks}
          onChange={(v) => set('stocks', v)}
          hint="Universe size, 50 to 3,000."
          error={errors.stocks}
          describedBy={describedBy('stocks', `${fid('stocks')}-hint`)}
        />
        <NumberField
          id={fid('days')}
          label="Days of market data"
          value={values.days}
          onChange={(v) => set('days', v)}
          hint={`The first ${WARMUP_DAYS} are warm-up for the risk and alpha models.`}
          error={errors.days}
          describedBy={describedBy('days', `${fid('days')}-hint`)}
        />
        <NumberField
          id={fid('seed')}
          label="Seed"
          value={values.seed}
          onChange={(v) => set('seed', v)}
          hint="Same seed, same market and models."
          error={errors.seed}
          describedBy={describedBy('seed', `${fid('seed')}-hint`)}
        />
      </div>

      <Choice
        legend="Signal lag"
        name="lag"
        value={values.signal_lag_days}
        onChange={(v) => set('signal_lag_days', v === '0' ? '0' : '1')}
        options={[
          {
            value: '1',
            label: 'Trade at the next close (1 day)',
            hint: 'The realistic default.',
          },
          {
            value: '0',
            label: 'Trade at the same close (0 days)',
            hint: 'Uses the closing price that produced the signal: optimistic, and flagged on the run page.',
          },
        ]}
      />

      <details className={styles.details}>
        <summary>Config overrides</summary>
        <p className={styles.hint}>
          Leave a field empty to keep the SystemConfig default. Each change is a new trial in the
          ledger.
        </p>
        {schema.isError ? <ErrorState error={schema.error} /> : null}
        {groups.map(([group, groupFields]) => {
          const changed = groupFields.filter((f) => {
            const v = values.overrides[f.path];
            return v !== undefined && v !== '';
          }).length;
          return (
            <details key={group} className={styles.group}>
              <summary>
                {group}
                <span className={styles.groupMeta}>
                  {' '}
                  ({groupFields.length} settings{changed ? `, ${changed} changed` : ''})
                </span>
              </summary>
              <div className={styles.row}>
                {groupFields.map((f) => (
                  <OverrideInput
                    key={f.path}
                    field={f}
                    id={fid(`override:${f.path}`)}
                    value={values.overrides[f.path]}
                    error={errors[`override:${f.path}`]}
                    describedBy={describedBy(
                      `override:${f.path}`,
                      `${fid(`override:${f.path}`)}-hint`,
                    )}
                    onChange={(v) => setOverride(f.path, v)}
                  />
                ))}
              </div>
            </details>
          );
        })}
      </details>

      {launch.isError ? <ErrorState error={launch.error} /> : null}

      <div className={styles.actions}>
        <Button type="submit" variant="primary" disabled={launch.isPending}>
          {launch.isPending ? 'Launching…' : 'Launch run'}
        </Button>
      </div>
    </form>
  );
}

function OverrideInput({
  field: f,
  id,
  value,
  error,
  describedBy,
  onChange,
}: {
  field: OverrideField;
  id: string;
  value: string | boolean | undefined;
  error: string | undefined;
  describedBy: string | undefined;
  onChange: (v: string | boolean) => void;
}) {
  const hintId = `${id}-hint`;
  const leaf = f.leaf;
  if (leaf.type === 'boolean') {
    return (
      <div className={styles.field}>
        <div className={styles.check}>
          <input
            id={id}
            type="checkbox"
            checked={typeof value === 'boolean' ? value : (leaf.default ?? false)}
            onChange={(e) => onChange(e.target.checked)}
            aria-describedby={hintId}
          />
          <label htmlFor={id}>{f.title}</label>
        </div>
        <p id={hintId} className={styles.hint}>
          {leaf.description}
        </p>
      </div>
    );
  }
  const text = typeof value === 'string' ? value : '';
  if (leaf.type === 'string') {
    return (
      <div className={styles.field}>
        <label htmlFor={id} className={styles.label}>
          {f.title}
        </label>
        {leaf.enum ? (
          <select
            id={id}
            className={styles.input}
            value={text}
            onChange={(e) => onChange(e.target.value)}
            aria-describedby={describedBy}
          >
            <option value="">Default{leaf.default ? ` (${leaf.default})` : ''}</option>
            {leaf.enum.map((o) => (
              <option key={o} value={o}>
                {o}
              </option>
            ))}
          </select>
        ) : (
          <input
            id={id}
            className={styles.input}
            value={text}
            placeholder={leaf.default ?? ''}
            onChange={(e) => onChange(e.target.value)}
            aria-describedby={describedBy}
            autoComplete="off"
          />
        )}
        <p id={hintId} className={styles.hint}>
          {leaf.description ?? ''}
        </p>
        <FieldError id={`${id}-error`} message={error} />
      </div>
    );
  }
  return (
    <NumberField
      id={id}
      label={f.title}
      value={text}
      placeholder={leaf.default === undefined ? '' : String(leaf.default)}
      onChange={onChange}
      hint={leaf.description ?? ''}
      error={error}
      describedBy={describedBy}
      decimal={f.type === 'number'}
    />
  );
}

function FieldError({ id, message }: { id: string; message: string | undefined }) {
  return message ? (
    <p id={id} className={styles.error}>
      {message}
    </p>
  ) : null;
}

function NumberField({
  id,
  label,
  value,
  onChange,
  hint,
  error,
  describedBy,
  placeholder,
  decimal = false,
}: {
  id: string;
  label: string;
  value: string;
  onChange: (v: string) => void;
  hint: string;
  error: string | undefined;
  describedBy: string | undefined;
  placeholder?: string;
  decimal?: boolean;
}) {
  return (
    <div className={styles.field}>
      <label htmlFor={id} className={styles.label}>
        {label}
      </label>
      <input
        id={id}
        className={styles.input}
        inputMode={decimal ? 'decimal' : 'numeric'}
        value={value}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
        aria-invalid={Boolean(error)}
        aria-describedby={describedBy}
        autoComplete="off"
      />
      <p id={`${id}-hint`} className={styles.hint}>
        {hint}
      </p>
      <FieldError id={`${id}-error`} message={error} />
    </div>
  );
}

function Choice({
  legend,
  name,
  value,
  onChange,
  options,
}: {
  legend: string;
  name: string;
  value: string;
  onChange: (v: string) => void;
  options: readonly { value: string; label: string; hint: ReactNode }[];
}) {
  const base = useId();
  return (
    <fieldset className={styles.fieldset}>
      <legend className={styles.label}>{legend}</legend>
      <div className={styles.choices}>
        {options.map((o) => {
          const id = `${base}-${o.value}`;
          return (
            <div key={o.value} className={styles.choice}>
              <input
                id={id}
                type="radio"
                name={`${base}-${name}`}
                value={o.value}
                checked={value === o.value}
                onChange={() => onChange(o.value)}
                aria-describedby={`${id}-hint`}
              />
              <label htmlFor={id}>{o.label}</label>
              <p id={`${id}-hint`} className={styles.hint}>
                {o.hint}
              </p>
            </div>
          );
        })}
      </div>
    </fieldset>
  );
}
