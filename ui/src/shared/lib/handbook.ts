/** Notion handbook pages. Notion has no stable heading anchors, so links name the section. */
export type HandbookPage =
  'validation' | 'backtesting' | 'tcost' | 'risk' | 'alpha' | 'portfolio' | 'glossary';

export const HANDBOOK: Record<HandbookPage, { title: string; url: string }> = {
  validation: {
    title: 'Validation & statistics',
    url: 'https://app.notion.com/p/40f8262e79e081a39129e966f8fa7c2b',
  },
  backtesting: {
    title: 'Backtesting',
    url: 'https://app.notion.com/p/40f8262e79e081b682e7f49e2520c2b7',
  },
  tcost: {
    title: 'Transaction costs',
    url: 'https://app.notion.com/p/40f8262e79e08124bee9d3f4d8366924',
  },
  risk: { title: 'Risk model', url: 'https://app.notion.com/p/40f8262e79e0817090b7ef82499a28b4' },
  alpha: {
    title: 'Alpha model',
    url: 'https://app.notion.com/p/40f8262e79e08106b0eeedbc2347444e',
  },
  portfolio: {
    title: 'Portfolio construction',
    url: 'https://app.notion.com/p/40f8262e79e0816ebfcfde09cf00838f',
  },
  glossary: {
    title: 'Glossary',
    url: 'https://app.notion.com/p/40f8262e79e081c1a4b7d4cf357f3d33',
  },
};

export interface HandbookRef {
  page: HandbookPage;
  /** Section heading inside the page, e.g. "Metrics › Sharpe ratio". */
  section: string;
}

export function handbookLabel(ref: HandbookRef): string {
  return `${HANDBOOK[ref.page].title} › ${ref.section}`;
}

export function handbookUrl(ref: HandbookRef): string {
  return HANDBOOK[ref.page].url;
}
