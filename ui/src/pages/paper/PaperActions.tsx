import { useId, useState } from 'react';
import type { PaperReconcile, PaperRunRequest, PaperStatus } from '@/shared/api';
import { ApiError } from '@/shared/api';
import { usePaperAck, usePaperReconcile, usePaperRun } from '@/entities/research';
import {
  CommandLine,
  ConfirmDialog,
  SelectField,
  SwitchField,
  TaskJobs,
  TextField,
  layout,
  paperAckCommand,
  paperReconcileCommand,
  paperRunCommand,
} from '@/features/research-launch';
import form from '@/features/research-launch/research.module.css';
import { date, dateTime, num, pct } from '@/shared/lib/format';
import { Badge, Button, Panel } from '@/shared/ui';

const ISO = /^\d{4}-\d{2}-\d{2}$/;

function ReconcileLine({ r }: { r: PaperReconcile }) {
  const [a, b, n] = r.dates;
  const span =
    typeof a === 'string' && typeof b === 'string' ? `${date(a)} – ${date(b)}` : 'its dates';
  return (
    <p className={form.help} data-testid="last-reconcile">
      Last reconcile over {span}
      {typeof n === 'number' ? ` (${n} days)` : ''}: daily-return correlation{' '}
      {num(r.daily_return_corr, { digits: 4 })}, tracking error{' '}
      {pct(r.tracking_error_ann, { digits: 2 })} a year, largest weight gap{' '}
      {pct(r.max_abs_weight_diff, { digits: 3 })}. {r.note}.
    </p>
  );
}

/** "Run today" and "Reconcile" as background jobs; the failures a daily job meets are explained by the job list. */
export function PaperRunPanel({ s, rootFlag }: { s: PaperStatus; rootFlag: string | null }) {
  const book = s.book;
  const [day, setDay] = useState('');
  const [until, setUntil] = useState('');
  const [noSync, setNoSync] = useState(false);
  const [dry, setDry] = useState(false);
  const [redo, setRedo] = useState(false);
  const [mode, setMode] = useState<'daily' | 'single'>('daily');
  const [confirm, setConfirm] = useState<'run' | 'reconcile' | null>(null);
  const [message, setMessage] = useState('');
  const run = usePaperRun(book);
  const reconcile = usePaperReconcile(book);
  const busy = s.jobs.some((j) => j.status === 'queued' || j.status === 'running');
  const dayErr = day && !ISO.test(day) ? 'YYYY-MM-DD, or empty for the latest session.' : undefined;
  const untilErr =
    until && !ISO.test(until)
      ? 'YYYY-MM-DD, or empty.'
      : day && until
        ? 'Use a day or a backfill end, not both.'
        : undefined;
  const body: PaperRunRequest = {
    date: day || null,
    until: until || null,
    no_sync: noSync,
    dry_run: dry,
    redo_last: redo,
    signal_mode: mode,
  };
  const halted = s.status.trading_halted;
  const go = async () => {
    try {
      const out =
        confirm === 'reconcile'
          ? await reconcile.mutateAsync(undefined)
          : await run.mutateAsync(body);
      setMessage(`Queued: ${out.label}.`);
    } catch (e) {
      setMessage(e instanceof ApiError ? e.detail : 'The job could not be queued.');
    }
    setConfirm(null);
  };
  return (
    <Panel title="Run the book" subtitle="background jobs; one at a time per book">
      <div className={form.form}>
        {halted ? (
          <div className={layout.notice} role="status">
            <span>
              <Badge tone="fail">Halted</Badge> A STOP is not acknowledged: a run records the day
              but generates no orders.
            </span>
          </div>
        ) : null}
        <div className={form.grid}>
          <TextField
            name="date"
            label="Day"
            value={day}
            onChange={setDay}
            placeholder="latest session"
            error={dayErr}
            help="Process through this day (--date). Empty: the latest published session."
          />
          <TextField
            name="until"
            label="Backfill through"
            value={until}
            onChange={setUntil}
            placeholder="none"
            error={untilErr}
            help="Catch up day by day to this date (--until); backfilled days are labelled as such."
          />
        </div>
        <SelectField
          name="signal_mode"
          label="Signal mode"
          value={mode}
          options={[
            { value: 'daily', label: 'daily: refit-free daily signal (default)' },
            { value: 'single', label: 'single: one pipeline pass for the whole span' },
          ]}
          onChange={setMode}
        />
        <SwitchField
          name="no_sync"
          label="Skip the data sync"
          checked={noSync}
          onChange={setNoSync}
          help="Use the store as it is (--no-sync)."
        />
        <SwitchField
          name="dry_run"
          label="Dry run"
          checked={dry}
          onChange={setDry}
          help="Compute orders but write nothing (--dry-run)."
        />
        <SwitchField
          name="redo_last"
          label="Redo the last day"
          checked={redo}
          onChange={setRedo}
          help="Replace the most recent record (--redo-last), e.g. after a late data correction."
        />
        <CommandLine command={paperRunCommand(book, body, rootFlag)} />
        <div className={form.actions}>
          <Button
            variant="primary"
            disabled={busy || Boolean(dayErr) || Boolean(untilErr)}
            onClick={() => setConfirm('run')}
          >
            {until ? 'Backfill…' : dry ? 'Dry run…' : 'Run today…'}
          </Button>
          <Button disabled={busy || s.n_days < 2} onClick={() => setConfirm('reconcile')}>
            Reconcile…
          </Button>
          {busy ? (
            <span className={form.help}>A job of this book is queued or running.</span>
          ) : null}
        </div>
        <CommandLine
          command={paperReconcileCommand(book, rootFlag)}
          note="Reconcile replays the book's dates in the batch simulator and compares weights, NAV and returns: a pipeline check, not evidence."
        />
        {s.last_reconcile ? <ReconcileLine r={s.last_reconcile} /> : null}
        {message ? (
          <p className={form.help} role="status">
            {message}
          </p>
        ) : null}
        <TaskJobs
          tasks={['paper_run', 'paper_reconcile']}
          subject={book}
          title="This book's jobs"
          limit={6}
        />
      </div>
      <ConfirmDialog
        open={confirm !== null}
        title={
          confirm === 'reconcile'
            ? `Reconcile ${book}?`
            : `Run ${book}${until ? ` through ${until}` : day ? ` for ${day}` : ''}?`
        }
        confirmLabel={confirm === 'reconcile' ? 'Reconcile' : 'Run'}
        busy={run.isPending || reconcile.isPending}
        onCancel={() => setConfirm(null)}
        onConfirm={() => void go()}
      >
        {confirm === 'reconcile' ? (
          <p>
            Replays every date of the book through the batch simulator (a minute or two). The book
            itself is not changed.
          </p>
        ) : (
          <>
            <p>
              Syncs the latest session{noSync ? ' (skipped)' : ''}, marks the book to the close,
              fills the pending orders and sizes the next ones. A paper book is not a trial: nothing
              is added to any family.
            </p>
            <p>
              If the vendor has not published the session yet the job stops with exit code 3 and
              changes nothing; run it again later. If another process holds the book&apos;s lock it
              stops with exit code 4.
            </p>
          </>
        )}
      </ConfirmDialog>
    </Panel>
  );
}

/** The acknowledgement a STOP needs before trading resumes (`qts paper ack`). */
export function AckPanel({ s, rootFlag }: { s: PaperStatus; rootFlag: string | null }) {
  const id = useId();
  const [note, setNote] = useState('');
  const [message, setMessage] = useState('');
  const ack = usePaperAck(s.book);
  const needs = s.status.level === 'STOP' && !s.status.acknowledged;
  const valid = note.trim().length >= 3;
  const submit = async () => {
    try {
      await ack.mutateAsync(note.trim());
      setMessage('Acknowledged: the next run generates orders again.');
      setNote('');
    } catch (e) {
      setMessage(e instanceof ApiError ? e.detail : 'The acknowledgement failed.');
    }
  };
  if (!needs && !s.acks.length) return null;
  return (
    <Panel
      title={needs ? 'Acknowledge the STOP' : 'Acknowledgements'}
      subtitle={needs ? 'required before trading resumes' : undefined}
      data-testid="ack-panel"
    >
      {needs ? (
        <div className={form.form}>
          <p className={form.help}>
            Write why it is safe to continue (what you checked). The note is kept with the book; the
            same rules will fire again if the condition persists.
          </p>
          <div className={form.field}>
            <label htmlFor={id} className={form.label}>
              Reason
            </label>
            <textarea
              id={id}
              className={form.textarea}
              rows={3}
              value={note}
              onChange={(e) => setNote(e.target.value)}
              maxLength={1000}
            />
          </div>
          <CommandLine command={paperAckCommand(s.book, note.trim(), rootFlag)} />
          <div className={form.actions}>
            <Button
              variant="primary"
              disabled={!valid || ack.isPending}
              onClick={() => void submit()}
            >
              Acknowledge and resume
            </Button>
          </div>
        </div>
      ) : null}
      {message ? (
        <p className={form.help} role="status">
          {message}
        </p>
      ) : null}
      {s.acks.length ? (
        <ul className={layout.list}>
          {s.acks.map((a) => (
            <li key={a.ts}>
              {dateTime(a.ts)} ({a.level} on {date(a.date)}): {a.note}
            </li>
          ))}
        </ul>
      ) : null}
    </Panel>
  );
}
