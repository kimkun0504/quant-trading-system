import { useParams } from 'react-router';
import type { CapacityResult } from '@/shared/api';
import { HANDBOOK_COSTS, useCapacity } from '@/entities/research';
import { Facts, Stat, layout } from '@/features/research-launch';
import {
  BoxSwatch,
  ChartFrame,
  LegendItem,
  LineSwatch,
  LogLineChart,
  StackedBarChart,
} from '@/shared/charts';
import type { StackKey } from '@/shared/charts';
import { bps, currency, date, num, pct, sharpe } from '@/shared/lib/format';
import { About, Badge, ErrorState, Page, Panel, Skeleton, Table, Td, Th } from '@/shared/ui';

type CostPart = 'spread' | 'impact' | 'fees';

/** Trading costs are one hue in decreasing strength, as in the run dossier. */
const COST_KEYS: readonly StackKey<CostPart>[] = [
  { key: 'spread', label: 'Spread', color: 'var(--color-gain)', opacity: 1 },
  { key: 'impact', label: 'Impact', color: 'var(--color-gain)', opacity: 0.62 },
  { key: 'fees', label: 'Fees', color: 'var(--color-gain)', opacity: 0.4 },
];

export function CapacityResultPage() {
  const { resultId = '' } = useParams();
  const q = useCapacity(resultId);
  return (
    <Page docTitle="Capacity curve" title="Capacity curve">
      {q.isPending ? (
        <Skeleton height="24rem" />
      ) : q.isError ? (
        <ErrorState error={q.error} onRetry={() => void q.refetch()} />
      ) : (
        <Body r={q.data} />
      )}
    </Page>
  );
}

function Body({ r }: { r: CapacityResult }) {
  const pts = [...r.points].sort((a, b) => a.nav - b.nav);
  const xs = pts.map((p) => p.nav);
  const fx = (v: number) => currency(v);
  const ref = r.reference_sharpe_net;
  const last = pts[pts.length - 1];
  return (
    <>
      <p className={layout.lede}>
        <code>{r.id}</code>: config <code>{r.config_hash.slice(0, 8)}</code>, data through{' '}
        {date(r.sealed_end)}. <Badge tone="skipped">Not a trial</Badge> logged as{' '}
        <code>{r.family}</code>.
      </p>
      <Facts
        label="Capacity headline"
        facts={[
          {
            key: 'ref',
            label: <Stat k="netSharpe">{`Net Sharpe at ${currency(pts[0]?.nav ?? null)}`}</Stat>,
            value: sharpe(ref),
          },
          {
            key: 'half',
            label: <Stat k="halfSharpeNav" />,
            value: r.half_sharpe_nav === null ? 'beyond the grid' : currency(r.half_sharpe_nav),
            note:
              r.half_sharpe_nav === null
                ? ref !== null && ref <= 0
                  ? 'the smallest NAV already loses money'
                  : `net Sharpe stays above half up to ${currency(last?.nav ?? null)}`
                : 'interpolated in log NAV',
          },
          {
            key: 'top',
            label: `Net Sharpe at ${currency(last?.nav ?? null)}`,
            value: sharpe(last?.sharpe_net ?? null),
          },
          {
            key: 'cap',
            label: <Stat k="advCap">At the ADV cap, largest NAV</Stat>,
            value: pct(last?.traded_at_adv_cap ?? null, { digits: 0 }),
          },
        ]}
      />
      <About label="How to read the curve" handbook={HANDBOOK_COSTS}>
        The gap between gross and net Sharpe is what trading costs take. It widens with NAV mostly
        through market impact, which grows with the size of each trade relative to the name&apos;s
        volume; spread and fees per dollar stay about the same. When many trades hit the
        participation cap the book can no longer trade to its targets, and the net Sharpe falls
        faster.
      </About>
      <div className={layout.row2}>
        <Panel title="Sharpe against NAV" subtitle="log scale of book size">
          <ChartFrame
            title="Gross and net Sharpe by NAV"
            legend={
              <>
                <LegendItem
                  swatch={<LineSwatch color="var(--color-series-1)" />}
                  label="Net Sharpe"
                />
                <LegendItem
                  swatch={<LineSwatch color="var(--color-series-2)" dashed />}
                  label="Gross Sharpe"
                />
              </>
            }
            table={() => ({
              caption: 'Sharpe by NAV',
              columns: ['NAV', 'Gross Sharpe', 'Net Sharpe', 'Net CAGR'],
              rows: pts.map((p) => [
                currency(p.nav),
                sharpe(p.sharpe_gross),
                sharpe(p.sharpe_net),
                pct(p.cagr_net),
              ]),
            })}
          >
            <LogLineChart
              ariaLabel={`Net Sharpe by NAV: ${pts.map((p) => `${currency(p.nav)} ${sharpe(p.sharpe_net)}`).join(', ')}`}
              xs={xs}
              series={[
                { label: 'Net Sharpe', paletteIndex: 0, values: pts.map((p) => p.sharpe_net) },
                {
                  label: 'Gross Sharpe',
                  paletteIndex: 1,
                  values: pts.map((p) => p.sharpe_gross),
                  dashed: true,
                },
              ]}
              format={(v) => sharpe(v, { signed: true })}
              formatX={fx}
              xLabel="NAV (USD)"
              refLines={
                ref !== null && ref > 0
                  ? [{ value: ref / 2, label: 'half' }]
                  : [{ value: 0, label: '0' }]
              }
              marker={
                r.half_sharpe_nav === null
                  ? null
                  : { x: r.half_sharpe_nav, label: `half-Sharpe ${currency(r.half_sharpe_nav)}` }
              }
            />
          </ChartFrame>
        </Panel>
        <Panel title="Cost per side by NAV" subtitle="basis points of each dollar traded">
          <ChartFrame
            title="Cost split by NAV"
            legend={COST_KEYS.map((k) => (
              <LegendItem
                key={k.key}
                swatch={<BoxSwatch color={k.color} opacity={k.opacity} />}
                label={k.label}
              />
            ))}
            table={() => ({
              caption: 'Cost per side by NAV',
              columns: ['NAV', 'Spread', 'Impact', 'Fees', 'Total', 'Break-even'],
              rows: pts.map((p) => [
                currency(p.nav),
                bps(p.spread_bps_per_side),
                bps(p.impact_bps_per_side),
                bps(p.fees_bps_per_side),
                bps(p.cost_bps_per_side),
                bps(p.breakeven_cost_bps),
              ]),
            })}
          >
            <StackedBarChart
              bars={pts.map((p) => ({
                label: String(p.nav),
                values: {
                  spread: p.spread_bps_per_side ?? 0,
                  impact: p.impact_bps_per_side ?? 0,
                  fees: p.fees_bps_per_side ?? 0,
                },
              }))}
              keys={COST_KEYS}
              format={(v) => bps(v)}
              labelFor={(l) => currency(Number(l))}
              tickFor={(l) => currency(Number(l))}
              ariaLabel={`Cost per side by NAV, stacked by spread, impact and fees: ${pts.map((p) => `${currency(p.nav)} ${bps(p.cost_bps_per_side)}`).join(', ')}`}
            />
          </ChartFrame>
        </Panel>
      </div>
      <Panel title="Every NAV">
        <Table
          label="Capacity points"
          caption="Break-even: the cost per side at which the net Sharpe would be zero."
        >
          <thead>
            <tr>
              <Th numeric>NAV</Th>
              <Th numeric>Gross Sharpe</Th>
              <Th numeric>
                <Stat k="netSharpe" />
              </Th>
              <Th numeric>
                <Stat k="costPerSide" />
              </Th>
              <Th numeric>Impact</Th>
              <Th numeric>Turnover / day</Th>
              <Th numeric>
                <Stat k="advCap" />
              </Th>
              <Th numeric>Solves capped</Th>
              <Th numeric>
                <Stat k="breakeven" />
              </Th>
            </tr>
          </thead>
          <tbody>
            {pts.map((p) => (
              <tr key={p.nav}>
                <Td numeric>{currency(p.nav)}</Td>
                <Td numeric>{sharpe(p.sharpe_gross, { signed: true })}</Td>
                <Td numeric>{sharpe(p.sharpe_net, { signed: true })}</Td>
                <Td numeric>{bps(p.cost_bps_per_side)}</Td>
                <Td numeric>{bps(p.impact_bps_per_side)}</Td>
                <Td numeric>{pct(p.turnover_per_day, { digits: 1 })}</Td>
                <Td numeric>{pct(p.traded_at_adv_cap, { digits: 0 })}</Td>
                <Td numeric>{pct(p.solves_with_adv_cap, { digits: 0 })}</Td>
                <Td numeric>{bps(p.breakeven_cost_bps, { signed: true })}</Td>
              </tr>
            ))}
          </tbody>
        </Table>
        {r.notes.length ? (
          <ul className={layout.list}>
            {r.notes.map((n) => (
              <li key={n}>{n}</li>
            ))}
          </ul>
        ) : null}
        <p className={layout.lede}>
          Stage cache:{' '}
          {Object.entries(r.cache)
            .map(([k, v]) => `${k} ${v}`)
            .join(', ') || 'not used'}
          .{' '}
          {num(
            pts.reduce((a, p) => a + (p.seconds ?? 0), 0),
            { digits: 0 },
          )}{' '}
          s of backtests.
        </p>
      </Panel>
    </>
  );
}
