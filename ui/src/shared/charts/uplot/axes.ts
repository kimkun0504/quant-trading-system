import type uPlot from 'uplot';
import type { ChartTokens } from '../tokens';
import { timeTickLabels } from './time';

export function timeAxis(t: ChartTokens): uPlot.Axis {
  return {
    stroke: t.graphite,
    font: t.font,
    grid: { show: false },
    ticks: { stroke: t.rule, width: 1, size: 4 },
    values: timeTickLabels,
    space: 70,
  };
}

export function valueAxis(t: ChartTokens, format: (v: number) => string, size = 56): uPlot.Axis {
  return {
    stroke: t.graphite,
    font: t.font,
    size,
    grid: { stroke: t.rule, width: 1 },
    ticks: { show: false },
    values: (_u, vals) => vals.map((v) => format(v)),
    space: 36,
  };
}

export const cursorBase = (syncKey?: string): uPlot.Cursor => ({
  y: false,
  points: { size: 8, width: 2 },
  drag: { x: false, y: false, setScale: false },
  ...(syncKey ? { sync: { key: syncKey } } : {}),
});
