import type { SliceDraft } from './lib/store';
import { sliceErrors } from './lib/store';
import { TextField } from './Field';
import styles from './research.module.css';

/**
 * The data a research command loads: the store slice and its prefilter (the `qts ...
 * --store` flags; the store itself is the API server's QTS_STORE_PATH). The command loads
 * through the research seal: data stop the day before the holdout.
 */
export function StoreSliceFields({
  value,
  onChange,
  storePath,
  rank = false,
}: {
  value: SliceDraft;
  onChange: (v: SliceDraft) => void;
  storePath: string | null;
  /** The mid-cap check's form (its rank prefilter default is 2750). */
  rank?: boolean;
}) {
  const e = sliceErrors(value);
  const set = (k: keyof SliceDraft) => (v: string) => onChange({ ...value, [k]: v });
  return (
    <fieldset className={styles.fieldset}>
      <legend className={styles.legend}>Data</legend>
      <p className={styles.help}>
        Store <code>{storePath ?? 'not configured (QTS_STORE_PATH)'}</code>. Keep these flags the
        same as the sealed baseline run so its fits come from the stage cache. Every load is checked
        first: a prefilter that misses names the universe picks is refused.
      </p>
      <div className={styles.grid}>
        <TextField
          name="start"
          label="Start"
          value={value.start}
          onChange={set('start')}
          placeholder="whole store"
          error={e.start}
          help="First date loaded (YYYY-MM-DD). Default in the reference load: 2000-01-01."
        />
        <TextField
          name="end"
          label="End"
          value={value.end}
          onChange={set('end')}
          placeholder="the seal's last date"
          error={e.end}
          help="Leave empty: the seal cuts the load the day before the holdout."
        />
        <TextField
          name="max_rank"
          label="Max screened rank"
          numeric
          value={value.maxRank}
          onChange={set('maxRank')}
          placeholder="off"
          error={e.maxRank}
          help={
            rank
              ? 'Every id ranked ≤ K by market cap among names passing the price / ADV screens on some date. The mid-cap check uses 2750.'
              : 'Every id ranked ≤ K by market cap among names passing the price / ADV screens on some date: covers the 1,000-name universe from K = 1,250. Reference: 1500.'
          }
        />
        <TextField
          name="max_names"
          label="Max names"
          numeric
          value={value.maxNames}
          onChange={set('maxNames')}
          placeholder="off"
          error={e.maxNames}
          help="Load the N ids with the largest peak market cap. Can miss names the universe picks: refused unless the load covers the universe. Prefer the rank."
        />
        <TextField
          name="min_mcap_ever"
          label="Min market cap ever (USD)"
          numeric
          value={value.minMcapEver}
          onChange={set('minMcapEver')}
          placeholder="off"
          error={e.minMcapEver}
          help="Load only ids whose peak market cap reached this (3e8 = $300M). With max names; not with the rank."
        />
      </div>
    </fieldset>
  );
}
