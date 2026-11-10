import type { CSSProperties } from 'react';
import { Link, useNavigate } from 'react-router';
import { useRuns } from '@/entities/run';
import { MAX_SELECTED, clearSelection, removeSelected, useSelection } from '@/features/selection';
import { Button, IconButton, Kbd } from '@/shared/ui';
import { Icons } from '@/shared/ui/icons';
import styles from './SelectionTray.module.css';

/**
 * The persistent bottom tray: the runs selected for comparison as chips (in series
 * colour order), remove buttons, and the Compare action. Always rendered so the page
 * never shifts when the first run is selected.
 */
export function SelectionTray() {
  const ids = useSelection();
  const runs = useRuns();
  const navigate = useNavigate();
  const labelOf = (id: string) => runs.data?.find((r) => r.id === id)?.label ?? id;
  const canCompare = ids.length >= 2;

  return (
    <div className={styles.tray} role="region" aria-label="Selected runs">
      {ids.length === 0 ? (
        <p className={styles.empty}>
          <Icons.Compare />
          Select runs to compare them side by side.
        </p>
      ) : (
        <>
          <span className={styles.count}>
            {ids.length} of {MAX_SELECTED} selected
          </span>
          <ul className={styles.chips} aria-label="Selected runs">
            {ids.map((id, i) => (
              <li key={id} className={styles.chip} style={{ '--i': i } as CSSProperties}>
                <span
                  className={styles.swatch}
                  style={{ background: `var(--color-series-${i + 1})` }}
                  aria-hidden="true"
                />
                <Link to={`/runs/${encodeURIComponent(id)}`} className={styles.chipLabel}>
                  {labelOf(id)}
                </Link>
                <IconButton
                  size="sm"
                  label={`Remove ${labelOf(id)} from the selection`}
                  onClick={() => removeSelected(id)}
                >
                  <Icons.Close />
                </IconButton>
              </li>
            ))}
          </ul>
        </>
      )}
      <div className={styles.actions}>
        {ids.length > 0 ? (
          <Button variant="quiet" size="sm" onClick={clearSelection}>
            Clear
          </Button>
        ) : null}
        <Button
          variant="primary"
          size="sm"
          disabled={!canCompare}
          onClick={() => void navigate('/compare')}
          aria-describedby="compare-hint"
        >
          Compare
          <span id="compare-hint" className={styles.hint}>
            <Kbd>c</Kbd>
          </span>
        </Button>
      </div>
    </div>
  );
}
