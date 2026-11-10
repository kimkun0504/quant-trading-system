export interface LabStub {
  title: string;
  docTitle: string;
  step: number;
  what: string;
}

export const LAB_STUBS = {
  signal: {
    title: 'Signal lab',
    docTitle: 'Signal lab',
    step: 4,
    what: 'IC by theme by year, IC decay per horizon, the feature IC table and learned weights over time per refit.',
  },
  portfolio: {
    title: 'Portfolio anatomy',
    docTitle: 'Portfolio',
    step: 5,
    what: 'Exposure over time by industry and style, gross and net, turnover, holding period, invested-day gaps and a holdings snapshot on a chosen date.',
  },
  costs: {
    title: 'Cost & constraint lab',
    docTitle: 'Costs & constraints',
    step: 5,
    what: 'Binding constraints over time and their transfer-coefficient cost, the cost breakdown, breakeven and cost sensitivity.',
  },
} satisfies Record<string, LabStub>;
