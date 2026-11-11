import { createBrowserRouter, createMemoryRouter, Navigate } from 'react-router';
import type { RouteObject } from 'react-router';
import { AppShell } from '@/widgets/AppShell';
import { RouteError } from './RouteError';

/**
 * Every page is its own lazy chunk; uPlot loads only with the pages that chart.
 * Workspaces: / (Experiments), /compare, /signal, /portfolio, /costs, /builder; a run is
 * one page, the dossier (/runs/:id, sections as #anchors; the old tab paths redirect);
 * /ledger and /models stay reachable.
 */
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
          {
            index: true,
            lazy: () =>
              import('@/pages/experiments/ExperimentsPage').then((m) => ({
                Component: m.ExperimentsPage,
              })),
          },
          { path: 'runs', element: <Navigate to="/" replace /> },
          { path: 'runs/new', element: <Navigate to="/builder" replace /> },
          {
            path: 'compare',
            lazy: () =>
              import('@/pages/compare/ComparePage').then((m) => ({
                Component: m.ComparePage,
              })),
          },
          {
            path: 'builder',
            lazy: () =>
              import('@/pages/new-run/NewRunPage').then((m) => ({
                Component: m.NewRunPage,
              })),
          },
          {
            path: 'signal',
            lazy: () =>
              import('@/pages/signal-lab/SignalLabPage').then((m) => ({
                Component: m.SignalLabPage,
              })),
          },
          {
            path: 'portfolio',
            lazy: () =>
              import('@/pages/portfolio-lab/PortfolioLabPage').then((m) => ({
                Component: m.PortfolioLabPage,
              })),
          },
          {
            path: 'costs',
            lazy: () =>
              import('@/pages/costs-lab/CostsLabPage').then((m) => ({
                Component: m.CostsLabPage,
              })),
          },
          {
            path: 'runs/:runId',
            lazy: () =>
              import('@/pages/run-dossier/RunDossierPage').then((m) => ({
                Component: m.RunDossierPage,
              })),
          },
          {
            // the six tabs of the first design: each one is a section of the dossier now
            path: 'runs/:runId/:tab',
            lazy: () =>
              import('@/pages/run-dossier/RunTabRedirect').then((m) => ({
                Component: m.RunTabRedirect,
              })),
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
