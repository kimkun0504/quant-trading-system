import { Button } from '@/shared/ui';
import styles from './research.module.css';

export interface Preset<T> {
  label: string;
  /** What it sets, for the button's description. */
  title: string;
  value: T;
}

/** A row of presets that fill the form (each sets every field it names). */
export function Presets<T>({
  presets,
  onPick,
}: {
  presets: readonly Preset<T>[];
  onPick: (v: T) => void;
}) {
  return (
    <div className={styles.presets} role="group" aria-label="Presets">
      <span className={styles.presetsLabel}>Presets</span>
      {presets.map((p) => (
        <Button
          key={p.label}
          size="sm"
          variant="secondary"
          title={p.title}
          onClick={() => onPick(p.value)}
        >
          {p.label}
        </Button>
      ))}
    </div>
  );
}
