import { STATS } from '@/entities/research/definitions';
import type { StatKey } from '@/entities/research/definitions';
import { Definition } from '@/shared/ui';

/** A statistic's label that explains itself on hover / focus (what it is, a good value). */
export function Stat({ k, children }: { k: StatKey; children?: string }) {
  const d = STATS[k];
  return (
    <Definition definition={d.help} formula={'formula' in d ? d.formula : undefined}>
      {children ?? d.label}
    </Definition>
  );
}
