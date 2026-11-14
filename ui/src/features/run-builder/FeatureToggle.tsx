import { useId } from 'react';
import type { FeatureInfo } from '@/shared/api';
import { useFeatures } from '@/entities/research';
import { Badge, Skeleton } from '@/shared/ui';
import { choicesFor, listItems, listText } from './lib/fields';
import type { Field } from './lib/fields';
import styles from './FeatureToggle.module.css';

const ROLE: Record<string, string> = {
  alpha_theme: 'theme feature',
  risk_only_proxy: 'risk-only proxy',
  model_input: 'model input',
};

/**
 * `alpha.disabled_features` as a checklist: one switch per feature of the selected
 * `alpha.feature_set` (on = the model uses it), grouped by the composite theme it belongs
 * to, with its one-line description. Turning features off never changes the label or the
 * sizing rule; a theme averages its remaining members, and a theme with none left is
 * omitted while the others' weights renormalise. Every setting is a logged trial.
 */
export function FeatureToggle({
  field,
  raw,
  featureSet,
  label,
  error,
  onChange,
}: {
  field: Field;
  raw: string;
  featureSet: string | null;
  label: string;
  error?: string | undefined;
  onChange: (raw: string) => void;
}) {
  const id = useId();
  const features = useFeatures(featureSet);
  const allowed = new Set(choicesFor(field, featureSet));
  const off = new Set(listItems(listText(raw)));
  const stale = [...off].filter((n) => !allowed.has(n));
  const set = (name: string, on: boolean) => {
    const next = new Set(off);
    if (on) next.delete(name);
    else next.add(name);
    onChange(listText([...next]));
  };
  if (!features.data) return <Skeleton height="6rem" />;
  const rows = features.data.features.filter((f) => allowed.has(f.name));
  const groups = new Map<string, FeatureInfo[]>();
  for (const f of rows) groups.set(f.theme, [...(groups.get(f.theme) ?? []), f]);
  const nOff = rows.filter((f) => off.has(f.name)).length;
  return (
    <fieldset className={styles.box} aria-describedby={`${id}-sum`} data-testid="feature-toggle">
      <legend className={styles.legend}>{label}</legend>
      <p id={`${id}-sum`} className={styles.summary}>
        Feature set <code>{featureSet ?? 'v1'}</code>: {rows.length - nOff} of {rows.length}{' '}
        features on
        {nOff ? `, ${nOff} off` : ''}. A theme averages the members left on; a theme with none left
        is dropped and the other themes&apos; weights renormalise. The label and the sizing rule
        never change.
      </p>
      {stale.length ? (
        <p className={styles.error} role="alert">
          {stale.join(', ')} {stale.length === 1 ? 'is' : 'are'} not in feature set {featureSet}:
          turn {stale.length === 1 ? 'it' : 'them'} back on or change the feature set.{' '}
          <button
            type="button"
            className={styles.link}
            onClick={() => onChange(listText([...off].filter((n) => allowed.has(n))))}
          >
            Remove {stale.length === 1 ? 'it' : 'them'}
          </button>
        </p>
      ) : null}
      {error ? <p className={styles.error}>{error}</p> : null}
      {[...groups.entries()].map(([theme, members]) => {
        const left = members.filter((m) => !off.has(m.name)).length;
        const composite = theme !== 'ml_only';
        return (
          <div key={theme} className={styles.group} data-theme={theme}>
            <div className={styles.groupHead}>
              <span className={styles.theme}>
                {composite ? `Theme ${theme.replace(/_/g, ' ')}` : 'ML only (no composite theme)'}
              </span>
              {composite && left === 0 ? (
                <Badge tone="caution">dropped: the other themes renormalise</Badge>
              ) : composite && left < members.length ? (
                <Badge tone="neutral">
                  {left} of {members.length} members: the theme averages what is left
                </Badge>
              ) : null}
            </div>
            <ul className={styles.list}>
              {members.map((f) => {
                const on = !off.has(f.name);
                return (
                  <li key={f.name} className={styles.item} data-feature={f.name} data-on={on}>
                    <label className={styles.switch}>
                      <input
                        type="checkbox"
                        role="switch"
                        checked={on}
                        onChange={(e) => set(f.name, e.target.checked)}
                        aria-describedby={`${id}-${f.name}`}
                      />
                      <code className={styles.name}>{f.name}</code>
                      <span className={styles.state}>{on ? 'On' : 'Off'}</span>
                    </label>
                    <span id={`${id}-${f.name}`} className={styles.desc}>
                      {f.description}{' '}
                      <span className={styles.role}>
                        ({ROLE[f.role] ?? f.role}, {f.feature_set})
                      </span>
                    </span>
                  </li>
                );
              })}
            </ul>
          </div>
        );
      })}
    </fieldset>
  );
}
