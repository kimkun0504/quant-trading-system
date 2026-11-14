import { createBrowserRouter, createMemoryRouter, Navigate } from 'react-router';
import type { RouteObject } from 'react-router';
import { AppShell } from '@/widgets/AppShell';
import { RouteError } from './RouteError';

/**
 * Every page is its own lazy chunk; uPlot loads only with the pages that chart.
 * Workspaces: / (Experiments), /compare, /signal, /portfolio, /costs, /builder; a run is
 * one page, the dossier (/runs/:id, sections as #anchors; the old tab paths redirect);
 * the research tools (/seal, /signals, /sweeps, /capacity, /risk, /checks/midcap) and the
 * paper books (/paper/:book); /ledger and /models stay reachable.
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
              import('@/pages/run-builder/RunBuilderPage').then((m) => ({
                Component: m.RunBuilderPage,
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
            path: 'seal',
            lazy: () =>
              import('@/pages/seal/SealPage').then((m) => ({
                Component: m.SealPage,
              })),
          },
          {
            path: 'signals',
            lazy: () =>
              import('@/pages/signal-tests/SignalTestsPage').then((m) => ({
                Component: m.SignalTestsPage,
              })),
          },
          {
            path: 'signals/:testId',
            lazy: () =>
              import('@/pages/signal-tests/SignalTestPage').then((m) => ({
                Component: m.SignalTestPage,
              })),
          },
          {
            path: 'sweeps',
            lazy: () =>
              import('@/pages/sweeps/SweepsPage').then((m) => ({
                Component: m.SweepsPage,
              })),
          },
          {
            path: 'sweeps/:sweepId',
            lazy: () =>
              import('@/pages/sweeps/SweepPage').then((m) => ({
                Component: m.SweepPage,
              })),
          },
          {
            path: 'capacity',
            lazy: () =>
              import('@/pages/capacity/CapacityPage').then((m) => ({
                Component: m.CapacityPage,
              })),
          },
          {
            path: 'capacity/:resultId',
            lazy: () =>
              import('@/pages/capacity/CapacityResultPage').then((m) => ({
                Component: m.CapacityResultPage,
              })),
          },
          {
            path: 'risk',
            lazy: () =>
              import('@/pages/risk-calibration/RiskPage').then((m) => ({
                Component: m.RiskPage,
              })),
          },
          {
            path: 'risk/:resultId',
            lazy: () =>
              import('@/pages/risk-calibration/RiskResultPage').then((m) => ({
                Component: m.RiskResultPage,
              })),
          },
          {
            path: 'checks/midcap',
            lazy: () =>
              import('@/pages/midcap/MidcapPage').then((m) => ({
                Component: m.MidcapPage,
              })),
          },
          {
            path: 'checks/midcap/:runId',
            lazy: () =>
              import('@/pages/midcap/MidcapResultPage').then((m) => ({
                Component: m.MidcapResultPage,
              })),
          },
          {
            path: 'paper',
            lazy: () =>
              import('@/pages/paper/PaperPage').then((m) => ({
                Component: m.PaperPage,
              })),
          },
          {
            path: 'paper/:book',
            lazy: () =>
              import('@/pages/paper/PaperBookPage').then((m) => ({
                Component: m.PaperBookPage,
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
