import { useId, useState } from 'react';
import { useSeal, useUnsealCheck } from '@/entities/research';
import { ConfirmDialog } from '@/features/research-launch';
import launch from '@/features/research-launch/research.module.css';
import { date, dateTime, shortHash } from '@/shared/lib/format';
import { Badge, Button } from '@/shared/ui';
import styles from './Unseal.module.css';

/**
 * Unseal the holdout for one store run (`qts run --unseal "<reason>" [--force]`). A typed
 * reason and a confirmation that says plainly what it does: once per configuration, logged
 * in the trial ledger before the run starts, and it cannot be undone. A configuration
 * whose holdout was already used needs an explicit force, and its result is flagged
 * "holdout already used". The choice is part of the draft; it takes effect on launch.
 */
export function UnsealControl({
  configHash,
  phase,
  reason,
  force,
  error,
  onChange,
}: {
  configHash: string | null;
  phase: 1 | 2;
  reason: string;
  force: boolean;
  error?: string | undefined;
  onChange: (reason: string, force: boolean) => void;
}) {
  const id = useId();
  const seal = useSeal();
  const check = useUnsealCheck(configHash, phase);
  const [open, setOpen] = useState(false);
  const [draft, setDraft] = useState('');
  const [forceDraft, setForceDraft] = useState(false);
  const s = seal.data?.seal;
  if (!seal.data) return null;
  if (!s)
    return (
      <p className={styles.note}>
        No holdout seal is in force: a store run loads the whole history.
      </p>
    );
  const prev = check.data?.previous ?? [];
  const used = check.data?.requires_force === true;
  const set = reason.trim() !== '';

  return (
    <div className={styles.box} data-testid="unseal-control" data-unseal={set ? 'on' : 'off'}>
      {set ? (
        <>
          <p className={styles.line}>
            <Badge tone={force ? 'fail' : 'caution'}>
              {force ? 'Unseal, forced' : 'Unseals on launch'}
            </Badge>
            <span>
              Loads the holdout from {date(s.holdout_start)}. Reason: <q>{reason}</q>
            </span>
          </p>
          {force ? (
            <p className={styles.danger}>
              Holdout already used: this configuration was unsealed before, so the result is flagged
              and is not out-of-sample.
            </p>
          ) : null}
          <Button size="sm" variant="secondary" onClick={() => onChange('', false)}>
            Keep the holdout sealed
          </Button>
        </>
      ) : (
        <>
          <p className={styles.line}>
            <Badge tone="neutral">Sealed</Badge>
            <span>
              This run loads data through {date(seal.data.data_through)}; the holdout from{' '}
              {date(s.holdout_start)} stays unseen.
            </span>
          </p>
          {used ? (
            <p className={styles.danger}>
              This configuration&apos;s holdout was already used (
              {prev.map((p) => dateTime(p.ts)).join(', ')}).
            </p>
          ) : null}
          <Button
            size="sm"
            variant="secondary"
            onClick={() => {
              setDraft('');
              setForceDraft(false);
              setOpen(true);
            }}
          >
            Unseal this configuration…
          </Button>
        </>
      )}
      {error ? <p className={styles.error}>{error}</p> : null}
      <ConfirmDialog
        open={open}
        title="Unseal the holdout for this configuration?"
        confirmLabel="Unseal on launch"
        tone="danger"
        confirmDisabled={draft.trim().length < 3 || (used && !forceDraft) || !configHash}
        onCancel={() => setOpen(false)}
        onConfirm={() => {
          onChange(draft.trim(), used && forceDraft);
          setOpen(false);
        }}
      >
        <p>
          The run will load the sealed holdout from <strong>{date(s.holdout_start)}</strong> for
          configuration <code>{configHash ? shortHash(configHash, 16) : 'not known yet'}</code>.
        </p>
        <p>
          <strong>This is allowed once per configuration and cannot be undone.</strong> The unseal
          is written to the trial ledger before the run starts, whether or not the run finishes.
          Unseal only the final candidate, and say which decision it serves.
        </p>
        {used ? (
          <div className={styles.danger}>
            <p>
              <strong>Holdout already used.</strong> This configuration was unsealed{' '}
              {prev.map((p) => `on ${dateTime(p.ts)} (“${p.unseal_reason ?? ''}”)`).join(', ')}. A
              second unseal is refused unless forced, and the result is flagged “holdout already
              used”: it is not out-of-sample.
            </p>
            <label className={launch.check} htmlFor={`${id}-force`}>
              <input
                id={`${id}-force`}
                type="checkbox"
                checked={forceDraft}
                onChange={(e) => setForceDraft(e.target.checked)}
              />
              <span>Force a second unseal (flagged: holdout already used)</span>
            </label>
          </div>
        ) : null}
        <div className={launch.field}>
          <label htmlFor={`${id}-reason`} className={launch.label}>
            Reason (required)
          </label>
          <textarea
            id={`${id}-reason`}
            className={launch.textarea}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder="final check of candidate X before paper trading"
            aria-describedby={`${id}-reason-help`}
          />
          <p id={`${id}-reason-help`} className={launch.help}>
            Recorded on the unseal event and the run. It is what you will read later when you ask
            why the holdout was opened.
          </p>
        </div>
      </ConfirmDialog>
    </div>
  );
}
