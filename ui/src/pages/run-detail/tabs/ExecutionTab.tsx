import { Annotated, Flow, MarginNote, Section } from '@/shared/ui';
import { bps } from '@/shared/lib/format';
import { useRunOutlet } from '../context';
import styles from './Tabs.module.css';

export function ExecutionTab() {
  const { run } = useRunOutlet();
  return (
    <Flow>
      <Section id="routing" title="Routing">
        <Annotated
          note={
            <MarginNote
              handbook={{
                page: 'tcost',
                section: 'Venue costing: auction vs continuous',
              }}
            >
              Until fills are recorded, every cost on the Costs tab is a model estimate:{' '}
              {bps(run.summary.avg_cost_bps_per_side)} per side from spread, square-root impact and
              the fee schedule.
            </MarginNote>
          }
        >
          <div className={styles.prose}>
            <p>
              This run’s orders were routed to the <strong>{run.execution_venue}</strong>. Signals
              form at the close of <em>t</em> and trade{' '}
              {run.config.signal_lag_days === 0 ? 'at that same close' : 'at the next close'}; the
              backtest books each trade at the closing price.
            </p>
            <p>
              Fills appear here once the execution simulator writes them to the TCA ledger. This tab
              will then show:
            </p>
          </div>
        </Annotated>
        <ul className={styles.list}>
          <li>
            Implementation shortfall per day, split into delay, impact, timing and opportunity cost.
          </li>
          <li>
            Realised against modelled cost per side, by venue and by order size as a share of ADV.
          </li>
          <li>Participation rates against the 5% ADV cap, with the orders that hit it.</li>
          <li>The monthly recalibration of the impact coefficient b from realised fills.</li>
        </ul>
      </Section>
    </Flow>
  );
}
