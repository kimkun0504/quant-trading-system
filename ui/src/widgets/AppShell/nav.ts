import type { Icons } from '@/shared/ui/icons';

type IconName = keyof typeof Icons;

export interface NavEntry {
  to: string;
  label: string;
  icon: IconName;
  end?: boolean;
}

/** The five workspaces, then the registry views. */
export const WORKSPACES: readonly NavEntry[] = [
  { to: '/', label: 'Experiments', icon: 'Flask', end: true },
  { to: '/signal', label: 'Signal lab', icon: 'Signal' },
  { to: '/portfolio', label: 'Portfolio', icon: 'Portfolio' },
  { to: '/costs', label: 'Costs & constraints', icon: 'Costs' },
  { to: '/builder', label: 'Run builder', icon: 'Builder' },
];

/** The research tools (trial-accounted) and the paper books. */
export const RESEARCH_NAV: readonly NavEntry[] = [
  { to: '/seal', label: 'Holdout seal', icon: 'Lock' },
  { to: '/signals', label: 'Signal tests', icon: 'Target' },
  { to: '/sweeps', label: 'Sweeps', icon: 'Grid' },
  { to: '/capacity', label: 'Capacity', icon: 'Gauge' },
  { to: '/risk', label: 'Risk calibration', icon: 'Portfolio' },
  { to: '/checks/midcap', label: 'Mid-cap check', icon: 'Check' },
  { to: '/paper', label: 'Paper trading', icon: 'Paper' },
];

export const REGISTRY_NAV: readonly NavEntry[] = [
  { to: '/ledger', label: 'Ledger', icon: 'Ledger' },
  { to: '/models', label: 'Models', icon: 'Models' },
];
