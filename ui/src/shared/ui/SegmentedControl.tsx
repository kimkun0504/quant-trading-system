import { useId } from 'react';
import styles from './SegmentedControl.module.css';

/** A radio group drawn as a segmented control. */
export function SegmentedControl<V extends string>({
  legend,
  options,
  value,
  onChange,
}: {
  legend: string;
  options: readonly { value: V; label: string }[];
  value: V;
  onChange: (v: V) => void;
}) {
  const name = useId();
  return (
    <fieldset className={styles.group}>
      <legend className={styles.legend}>{legend}</legend>
      <div className={styles.options}>
        {options.map((o) => (
          <label key={o.value} className={styles.option}>
            <input
              type="radio"
              name={name}
              value={o.value}
              checked={value === o.value}
              onChange={() => onChange(o.value)}
              className={styles.input}
            />
            <span className={styles.face}>{o.label}</span>
          </label>
        ))}
      </div>
    </fieldset>
  );
}
