/**
 * The dossier's sections in reading order. Each is a page anchor (`/runs/:id#signal`),
 * a row in the side navigation, and points at the lab that goes deeper (stubs until
 * steps 4–5 of the redesign; the link is there so the path is learnable now).
 */
export type SectionId =
  'performance' | 'signal' | 'portfolio' | 'costs' | 'validation' | 'config' | 'artifacts';

export interface SectionDef {
  id: SectionId;
  title: string;
  /** Where to go deeper: the lab's route and name; omitted where none exists. */
  lab?: { to: string; label: string };
}

export const SECTIONS: readonly SectionDef[] = [
  { id: 'performance', title: 'Performance', lab: { to: '/', label: 'Experiments' } },
  { id: 'signal', title: 'Signal', lab: { to: '/signal', label: 'Signal lab' } },
  { id: 'portfolio', title: 'Portfolio', lab: { to: '/portfolio', label: 'Portfolio anatomy' } },
  { id: 'costs', title: 'Costs', lab: { to: '/costs', label: 'Cost & constraint lab' } },
  { id: 'validation', title: 'Validation', lab: { to: '/ledger', label: 'Trial ledger' } },
  { id: 'config', title: 'Config', lab: { to: '/builder', label: 'Run builder' } },
  { id: 'artifacts', title: 'Artifacts' },
];

export const SECTION_IDS: readonly SectionId[] = SECTIONS.map((s) => s.id);

export function isSectionId(x: string): x is SectionId {
  return (SECTION_IDS as readonly string[]).includes(x);
}

/** The old run tabs (`/runs/:id/<tab>`) and the section each one became. */
export const TAB_TO_SECTION: Readonly<Record<string, SectionId>> = {
  performance: 'performance',
  alpha: 'signal',
  risk: 'portfolio',
  costs: 'costs',
  execution: 'costs',
  validation: 'validation',
  config: 'config',
};
