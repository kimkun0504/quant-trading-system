import { useState } from 'react';
import { Outlet, useParams } from 'react-router';
import { useRun } from '@/entities/run';
import { claimEntrance, usePrefersReducedMotion } from '@/shared/lib/motion';
import { ErrorState, LinkButton, Page, Skeleton, SkeletonLines, TabNav } from '@/shared/ui';
import type { TabItem } from '@/shared/ui';
import { RunHeader } from '@/widgets/RunHeader';
import type { RunOutletContext } from './context';
import styles from './RunDetailPage.module.css';

const TABS: readonly TabItem[] = [
  { to: '', label: 'Performance', end: true },
  { to: 'risk', label: 'Risk' },
  { to: 'alpha', label: 'Alpha' },
  { to: 'costs', label: 'Costs' },
  { to: 'execution', label: 'Execution' },
  { to: 'validation', label: 'Validation' },
];

export function RunDetailPage() {
  const { runId = '' } = useParams();
  const run = useRun(runId);
  const reduced = usePrefersReducedMotion();
  const [animate] = useState(() => !reduced && claimEntrance(`run:${runId}`));

  if (run.isPending) {
    return (
      <Page docTitle="Loading run">
        <div aria-busy="true" aria-label="Loading run" className={styles.loading}>
          <Skeleton height="1rem" width="40%" />
          <Skeleton height="1.6rem" width="55%" />
          <SkeletonLines lines={2} />
          <Skeleton height="7rem" />
          <Skeleton height="22rem" />
        </div>
      </Page>
    );
  }
  if (run.isError) {
    return (
      <Page docTitle="Run" title="Run">
        <ErrorState error={run.error} onRetry={() => void run.refetch()} />
        <div>
          <LinkButton to="/runs" variant="secondary">
            Back to runs
          </LinkButton>
        </div>
      </Page>
    );
  }

  const ctx: RunOutletContext = { run: run.data, animate };
  return (
    <Page docTitle={run.data.label}>
      <RunHeader run={run.data} animate={animate} />
      <div className={styles.tabs}>
        <TabNav items={TABS} label="Run sections" />
      </div>
      <Outlet context={ctx} />
    </Page>
  );
}
