import type { Ledger } from '@/shared/api';
import type { Base, DiffRow } from './draft';
import { dataChanged } from './draft';

export interface TrialForecast {
  /** The ledger family the run will log under; null when it cannot be known before launch. */
  family: string | null;
  /** Trials already in that family. */
  nTrials: number;
  /** This run's trial number in the family (nTrials + its place among the variants). */
  trialNumber: number | null;
  /** Why the family is unknown ('' when it is known). */
  unknownReason: string;
}

const PHASE_PREFIX = /^pipeline_phase[12]:/;

/**
 * The family of the base run with the phase of the new run: the family its manifest
 * records, else the one its ledger record was filed under (so this follows whatever the
 * pipeline keys families by: the market hash today), else `pipeline_phase{p}:{data_hash[:12]}`.
 */
export function familyFor(base: Base, ml: boolean, ledger: Ledger | undefined): string | null {
  const phase = `pipeline_phase${ml ? 2 : 1}:`;
  if (base.ledgerFamily && PHASE_PREFIX.test(base.ledgerFamily))
    return base.ledgerFamily.replace(PHASE_PREFIX, phase);
  if (!base.dataHash) return null;
  const own = ledger?.trials.find(
    (t) => t.data_hash === base.dataHash && PHASE_PREFIX.test(t.family),
  );
  if (own) return own.family.replace(PHASE_PREFIX, phase);
  return `${phase}${base.dataHash.slice(0, 12)}`;
}

/**
 * Which trial a launch will be. The family is one phase on one data set, so it is known
 * before launch only when the run reads the same data as its base run; a different
 * market or slice belongs to another family, counted from 1.
 * `ahead` = variants queued before this one in the same family.
 */
export function forecastTrial(
  base: Base,
  rows: readonly DiffRow[],
  ml: boolean,
  ledger: Ledger | undefined,
  ahead = 0,
): TrialForecast {
  const unknown = (unknownReason: string): TrialForecast => ({
    family: null,
    nTrials: 0,
    trialNumber: null,
    unknownReason,
  });
  if (!base.runId || (!base.dataHash && !base.ledgerFamily))
    return unknown(
      'The trial family is set by the data the run reads, which is known once the market is built. Start from a run to see the count before launch.',
    );
  if (dataChanged(rows))
    return unknown(
      'The data differs from the base run, so this run starts or joins another family; its trial count is known once the data is loaded.',
    );
  const family = familyFor(base, ml, ledger);
  if (!family) return unknown('The base run has no data hash, so its family is not known.');
  const nTrials = ledger?.families.find((f) => f.family === family)?.n_trials ?? 0;
  return { family, nTrials, trialNumber: nTrials + ahead + 1, unknownReason: '' };
}
