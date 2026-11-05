import { createBrowserRouter, createMemoryRouter, Navigate } from 'react-router';
import type { RouteObject } from 'react-router';
import { AppShell } from '@/widgets/AppShell';
import { RouteError } from './RouteError';

/** Every page and run tab is its own lazy chunk; uPlot loads only with the tabs that chart. */
export const routes: RouteObject[] = [
  {
    path: '/',
    element: <AppShell />,
    errorElement: <RouteError />,
    // Shown while the first lazy page chunk loads (no SSR, so this is a blank, busy page).
    hydrateFallbackElement: <div aria-busy="true" />,
    children: [
      {
        errorElement: <RouteError />,
        children: [
          { index: true, element: <Navigate to="/runs" replace /> },
          {
            path: 'runs',
            lazy: () =>
              import('@/pages/runs/RunsPage').then((m) => ({
                Component: m.RunsPage,
              })),
          },
          {
            path: 'runs/new',
            lazy: () =>
              import('@/pages/new-run/NewRunPage').then((m) => ({
                Component: m.NewRunPage,
              })),
          },
          {
            path: 'runs/:runId',
            lazy: () =>
              import('@/pages/run-detail/RunDetailPage').then((m) => ({
                Component: m.RunDetailPage,
              })),
            children: [
              {
                index: true,
                lazy: () =>
                  import('@/pages/run-detail/tabs/PerformanceTab').then((m) => ({
                    Component: m.PerformanceTab,
                  })),
              },
              {
                path: 'risk',
                lazy: () =>
                  import('@/pages/run-detail/tabs/RiskTab').then((m) => ({
                    Component: m.RiskTab,
                  })),
              },
              {
                path: 'alpha',
                lazy: () =>
                  import('@/pages/run-detail/tabs/AlphaTab').then((m) => ({
                    Component: m.AlphaTab,
                  })),
              },
              {
                path: 'costs',
                lazy: () =>
                  import('@/pages/run-detail/tabs/CostsTab').then((m) => ({
                    Component: m.CostsTab,
                  })),
              },
              {
                path: 'execution',
                lazy: () =>
                  import('@/pages/run-detail/tabs/ExecutionTab').then((m) => ({
                    Component: m.ExecutionTab,
                  })),
              },
              {
                path: 'validation',
                lazy: () =>
                  import('@/pages/run-detail/tabs/ValidationTab').then((m) => ({
                    Component: m.ValidationTab,
                  })),
              },
            ],
          },
          {
            path: 'models',
            lazy: () =>
              import('@/pages/models/ModelsPage').then((m) => ({
                Component: m.ModelsPage,
              })),
          },
          {
            path: 'ledger',
            lazy: () =>
              import('@/pages/ledger/LedgerPage').then((m) => ({
                Component: m.LedgerPage,
              })),
          },
          {
            path: '*',
            lazy: () =>
              import('@/pages/not-found/NotFoundPage').then((m) => ({
                Component: m.NotFoundPage,
              })),
          },
        ],
      },
    ],
  },
];

export function createAppRouter() {
  return createBrowserRouter(routes);
}

export function createTestRouter(initialPath: string) {
  return createMemoryRouter(routes, { initialEntries: [initialPath] });
}
