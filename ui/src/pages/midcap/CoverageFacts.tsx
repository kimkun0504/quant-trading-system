import type { MidcapCoverage } from '@/shared/api';
import { Facts } from '@/features/research-launch';
import { int, num, pct } from '@/shared/lib/format';
import { Badge } from '@/shared/ui';

/** The V2 data-coverage summary of a check or its dry run. */
export function CoverageFacts({ c }: { c: MidcapCoverage }) {
  return (
    <Facts
      label="Coverage"
      facts={[
        {
          key: 'ok',
          label: 'Coverage (V2)',
          value: c.ok ? <Badge tone="pass">enough</Badge> : <Badge tone="fail">short</Badge>,
        },
        {
          key: 'share',
          label: `Dates with every rank ≤ ${int(c.rank)} loaded`,
          value: pct(c.share_dates_full, { digits: 1 }),
          note: `needs ${pct(c.min_share, { digits: 0 })}`,
        },
        {
          key: 'loaded',
          label: 'Names loaded per date',
          value: `${int(c.min_loaded)} min, ${num(c.median_loaded, { digits: 0 })} median`,
          note: `needs ${int(c.needed_loaded)}`,
        },
        {
          key: 'gb',
          label: 'Estimated load',
          value: `${num(c.est_load_gb, { digits: 1 })} GB`,
          note: c.prefilter,
        },
      ]}
    />
  );
}
