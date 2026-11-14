import { Link } from 'react-router';
import { useSeal } from '@/entities/research';
import { date, int } from '@/shared/lib/format';
import { Tooltip } from '@/shared/ui';
import { Icons } from '@/shared/ui/icons';
import styles from './Seal.module.css';

/**
 * The holdout seal, always in view (top bar): research loads stop at the day before the
 * holdout. Hover for the inherited family and its trial count; click for the seal page.
 */
export function SealChip() {
  const seal = useSeal();
  if (!seal.data) return null;
  const s = seal.data.seal;
  if (!s)
    return (
      <Link to="/seal" className={styles.chip} data-state="none">
        <Icons.Unlock className={styles.glyph} />
        <span>No seal</span>
      </Link>
    );
  const fam = s.inherit_family[0];
  const n = fam ? seal.data.inherited_families[fam] : null;
  const tip = `Holdout from ${date(s.holdout_start)}: research data stop on ${date(seal.data.data_through)}. ${
    fam
      ? `Inherits ${fam}${n === null || n === undefined ? ' (not in this ledger)' : ` with ${int(n)} trials`}. `
      : ''
  }${seal.data.unseals.length} unseal${seal.data.unseals.length === 1 ? '' : 's'} so far.`;
  return (
    <Tooltip content={tip} align="end">
      <Link
        to="/seal"
        className={styles.chip}
        data-state="sealed"
        aria-label={`Holdout seal: ${tip}`}
      >
        <Icons.Lock className={styles.glyph} />
        <span>Sealed from {date(s.holdout_start)}</span>
        <span className={styles.chipMuted}>data through {date(seal.data.data_through)}</span>
        {fam && n !== null && n !== undefined ? (
          <span className={styles.chipMuted}>{int(n)} trials</span>
        ) : null}
      </Link>
    </Tooltip>
  );
}
