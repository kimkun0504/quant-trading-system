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

export const REGISTRY_NAV: readonly NavEntry[] = [
  { to: '/ledger', label: 'Ledger', icon: 'Ledger' },
  { to: '/models', label: 'Models', icon: 'Models' },
];
