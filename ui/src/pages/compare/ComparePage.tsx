import { Link } from 'react-router';
import { ComparePanel } from '@/features/compare';
import { useSelection } from '@/features/selection';
import { EmptyState, LinkButton, Page } from '@/shared/ui';

/** The selection tray's Compare target: the side-by-side panel as a full page. */
export function ComparePage() {
  const ids = useSelection();
  return (
    <Page
      docTitle="Compare runs"
      title={ids.length >= 2 ? `Comparing ${ids.length} runs` : 'Compare runs'}
    >
      {ids.length >= 2 ? (
        <ComparePanel ids={ids} />
      ) : (
        <EmptyState
          title={ids.length === 1 ? 'Select one more run' : 'Nothing selected yet'}
          action={<LinkButton to="/">Go to Experiments</LinkButton>}
        >
          <p>
            Tick two or more runs in <Link to="/">Experiments</Link>; they collect in the tray at
            the bottom and this page lays them side by side.
          </p>
        </EmptyState>
      )}
    </Page>
  );
}
