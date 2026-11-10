import styles from './Table.module.css';

/** Colour class for a signed number: gain / loss / none. The sign is always in the text too. */
export function signClass(v: number | null | undefined): string | undefined {
  if (v === null || v === undefined || !Number.isFinite(v)) return undefined;
  return v > 0 ? styles.gain : v < 0 ? styles.loss : undefined;
}
