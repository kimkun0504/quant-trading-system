import { Link } from 'react-router';
import { Page } from '@/shared/ui';

export function NotFoundPage() {
  return (
    <Page docTitle="Not found" title="This page does not exist">
      <p>
        The address does not match a page of the workbench. Go to the{' '}
        <Link to="/runs">runs list</Link>.
      </p>
    </Page>
  );
}
