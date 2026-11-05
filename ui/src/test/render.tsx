import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render } from '@testing-library/react';
import type { ReactElement } from 'react';
import { createMemoryRouter, RouterProvider } from 'react-router';
import { routes } from '@/app/routes';

export function testQueryClient(): QueryClient {
  return new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: Infinity } } });
}

/** Render a component inside the providers the app uses (query client + a router). */
export function renderWithProviders(ui: ReactElement, { path = '/' }: { path?: string } = {}) {
  const client = testQueryClient();
  const router = createMemoryRouter([{ path: '*', element: ui }], { initialEntries: [path] });
  return {
    client,
    ...render(
      <QueryClientProvider client={client}>
        <RouterProvider router={router} />
      </QueryClientProvider>,
    ),
  };
}

/** Render the real route tree at a path (integration). */
export function renderApp(path: string) {
  const client = testQueryClient();
  const router = createMemoryRouter(routes, { initialEntries: [path] });
  return {
    client,
    router,
    ...render(
      <QueryClientProvider client={client}>
        <RouterProvider router={router} />
      </QueryClientProvider>,
    ),
  };
}
