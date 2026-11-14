import type { RunSeal } from '@/shared/api';
import { Badge, Tooltip } from '@/shared/ui';
import { coverageExplanation, sealExplanation } from './explain';
import styles from './Seal.module.css';

/**
 * Badges for a run: sealed / unsealed / exempt, "holdout already used", and "reference"
 * (a re-run of a counted configuration: not a new trial). Nothing for a plain run made
 * before the seal existed.
 */
export function RunSealBadge({ seal, compact = false }: { seal: RunSeal; compact?: boolean }) {
  const parts = [];
  if (seal.holdout_already_used)
    parts.push(
      <Badge key="used" tone="fail" data-seal="holdout-used">
        Holdout already used
      </Badge>,
    );
  else if (seal.state === 'sealed')
    parts.push(
      <Badge key="sealed" tone="neutral" data-seal="sealed">
        Sealed
      </Badge>,
    );
  else if (seal.state === 'unsealed')
    parts.push(
      <Badge key="unsealed" tone="caution" data-seal="unsealed">
        Unsealed
      </Badge>,
    );
  else if (seal.state === 'exempt')
    parts.push(
      <Badge key="exempt" tone="accent" data-seal="exempt">
        Exempt
      </Badge>,
    );
  const partial = seal.load_coverage?.ok === false;
  if (partial)
    parts.push(
      <Badge key="coverage" tone="fail" data-seal="partial-load">
        {compact ? 'Partial load' : 'Partial load: look-ahead'}
      </Badge>,
    );
  if (seal.reference)
    parts.push(
      <Badge key="ref" tone="skipped" data-seal="reference">
        {compact ? 'Reference' : 'Reference, not counted'}
      </Badge>,
    );
  if (seal.record_kind !== 'trial')
    parts.push(
      <Badge key="kind" tone="skipped" data-seal="kind">
        {seal.record_kind}
      </Badge>,
    );
  if (!parts.length) return null;
  const tip = [
    sealExplanation(seal),
    seal.reference
      ? 'Reference: the same configuration as a counted trial; logged, not counted again.'
      : '',
    seal.record_kind !== 'trial'
      ? `A ${seal.record_kind} run: not an alpha trial, not counted in any DSR.`
      : '',
    partial ? coverageExplanation(seal) : '',
  ]
    .filter(Boolean)
    .join(' ');
  return (
    <Tooltip content={tip}>
      <button type="button" className={styles.badges} aria-label={`Seal: ${tip}`}>
        {parts}
      </button>
    </Tooltip>
  );
}
