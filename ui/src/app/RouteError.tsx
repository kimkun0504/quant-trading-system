import { isRouteErrorResponse, Link, useRouteError } from 'react-router';
import { Page } from '@/shared/ui';
import { ErrorState } from '@/shared/ui/ErrorState';

export function RouteError() {
  const error = useRouteError();
  if (isRouteErrorResponse(error) && error.status === 404) {
    return (
      <Page docTitle="Not found" title="This page does not exist">
        <p>
          Go back to the <Link to="/runs">runs list</Link>.
        </p>
      </Page>
    );
  }
  return (
    <Page docTitle="Error" title="This page failed to render">
      <ErrorState error={error} onRetry={() => window.location.reload()} />
    </Page>
  );
}
