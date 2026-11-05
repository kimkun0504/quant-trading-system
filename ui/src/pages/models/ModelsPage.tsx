import { HANDBOOK } from '@/shared/lib/handbook';
import type { HandbookPage } from '@/shared/lib/handbook';
import { Annotated, HandbookLink, MarginNote, Page } from '@/shared/ui';
import styles from './ModelsPage.module.css';

const MODELS: { page: HandbookPage; name: string; does: string; diagnostics: string }[] = [
  {
    page: 'alpha',
    name: 'Alpha model',
    does: 'Point-in-time features, a six-theme rank composite with capped factor-momentum tilts, and a walk-forward LightGBM / MLP / ridge sleeve with label embargo. Scores become expected returns by Grinold scaling.',
    diagnostics: 'Rank IC, ICIR, decile spread, IC decay — on the Alpha tab of each run.',
  },
  {
    page: 'risk',
    name: 'Risk model',
    does: 'A Barra USE4-style fundamental model: cross-sectional regressions, EWMA and Newey–West factor covariance, eigenfactor and volatility-regime adjustments, shrunk specific risk.',
    diagnostics: 'Factor exposures and realised against forecast volatility — on the Risk tab.',
  },
  {
    page: 'tcost',
    name: 'Transaction cost model',
    does: 'The IBKR Pro and regulatory fee schedule, square-root market impact, auction against continuous venue costing, and borrow as a holding cost.',
    diagnostics: 'Cost by component, cost per side, breakeven and sensitivity — on the Costs tab.',
  },
  {
    page: 'portfolio',
    name: 'Portfolio construction',
    does: 'A single-period convex optimiser in factor form with position, liquidity, beta, style and industry constraints, turnover penalties and volatility targeting within hard caps.',
    diagnostics:
      'Turnover, effective holding and the transfer coefficient — in each run’s key metrics.',
  },
  {
    page: 'backtesting',
    name: 'Execution',
    does: 'Order generation and routing: closing-auction LOC/MOC orders, IBKR algorithms by size and urgency, VWAP / POV / Almgren–Chriss schedules and a fill simulator feeding the TCA ledger.',
    diagnostics:
      'Routing today; fills and implementation shortfall once recorded — on the Execution tab.',
  },
];

export function ModelsPage() {
  return (
    <Page docTitle="Models" title="The five models">
      <Annotated
        note={
          <MarginNote>
            Each model can be used on its own against the contracts in{' '}
            <code>qts/core/types.py</code>; the pipeline runs all five in order for every backtest
            day.
          </MarginNote>
        }
      >
        <p className={styles.lede}>
          A run is the five models working together. This page says what each one does and where its
          diagnostics appear on a run page.
        </p>
      </Annotated>
      {MODELS.map((m) => (
        <Annotated key={m.name} className={styles.model}>
          <h2 className={styles.name}>{m.name}</h2>
          <p className={styles.does}>{m.does}</p>
          <p className={styles.diag}>{m.diagnostics}</p>
          <HandbookLink refTo={{ page: m.page, section: 'Overview' }}>
            {HANDBOOK[m.page].title} in the handbook
          </HandbookLink>
        </Annotated>
      ))}
    </Page>
  );
}
