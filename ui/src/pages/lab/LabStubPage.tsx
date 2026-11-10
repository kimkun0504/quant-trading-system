import { Link } from 'react-router';
import { EmptyState, Page } from '@/shared/ui';
import type { LabStub } from './stubs';

/** A workspace that is routed and in the sidebar but not built yet. */
export function LabStubPage({ stub }: { stub: LabStub }) {
  return (
    <Page docTitle={stub.docTitle} title={stub.title}>
      <EmptyState title={`Coming in step ${stub.step} of the redesign`}>
        <p>{stub.what}</p>
        <p>
          Until then the same diagnostics live on each run’s page: open a run from{' '}
          <Link to="/">Experiments</Link>.
        </p>
      </EmptyState>
    </Page>
  );
}
