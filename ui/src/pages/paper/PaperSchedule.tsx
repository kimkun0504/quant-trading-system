import { usePaperSchedule } from '@/entities/research';
import { layout } from '@/features/research-launch';
import form from '@/features/research-launch/research.module.css';
import { CopyButton, ErrorState, Panel, Skeleton } from '@/shared/ui';
import styles from './Paper.module.css';

function Block({ title, text, label }: { title: string; text: string; label: string }) {
  return (
    <section className={styles.schedule} aria-label={title}>
      <div className={styles.scheduleHead}>
        <h3 className={form.slipTitle}>{title}</h3>
        <CopyButton text={text} label={label} />
      </div>
      {/* eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex -- a scrollable region must be keyboard reachable */}
      <pre className={styles.code} tabIndex={0}>
        {text}
      </pre>
    </section>
  );
}

/** The daily schedule (`qts paper schedule`): text to install by hand; nothing is installed from here. */
export function PaperSchedule({ book }: { book: string }) {
  const q = usePaperSchedule(book);
  return (
    <Panel
      title="Schedule the daily job"
      subtitle="install it yourself; the app does not touch your system"
    >
      {q.isPending ? (
        <Skeleton height="8rem" />
      ) : q.isError ? (
        <ErrorState error={q.error} onRetry={() => void q.refetch()} />
      ) : (
        <div className={layout.stack}>
          <p className={layout.lede}>
            The job runs on weekdays after the vendor publishes the previous session. On a Mac use
            the launchd agent; elsewhere the crontab line. It logs to <code>{q.data.log}</code>.
          </p>
          <Block title="Install (macOS)" text={q.data.install} label="Copy the install steps" />
          <Block
            title={`launchd agent (${q.data.label}.plist)`}
            text={q.data.plist}
            label="Copy the plist"
          />
          <Block title="crontab line" text={q.data.crontab} label="Copy the crontab line" />
        </div>
      )}
    </Panel>
  );
}
