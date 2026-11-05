import { useMemo } from 'react';
import { useResolvedTheme } from '@/shared/lib/theme';
import type { ResolvedTheme } from '@/shared/lib/theme';

/**
 * Canvas charts (uPlot) cannot use CSS variables directly, so they read the resolved
 * token values from :root. SVG charts use var(--color-*) directly.
 */
export interface ChartTokens {
  theme: ResolvedTheme;
  ink: string;
  graphite: string;
  rule: string;
  sheet: string;
  gain: string;
  loss: string;
  caution: string;
  drawdownFillOpacity: number;
  font: string;
}

function read(name: string, fallback: string): string {
  if (typeof document === 'undefined') return fallback;
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

export function readChartTokens(theme: ResolvedTheme): ChartTokens {
  return {
    theme,
    ink: read('--color-ink', '#17212b'),
    graphite: read('--color-graphite', '#5a6573'),
    rule: read('--color-rule', '#d3d9e0'),
    sheet: read('--color-sheet', '#ffffff'),
    gain: read('--color-gain', '#1f5f8b'),
    loss: read('--color-loss', '#9b2f52'),
    caution: read('--color-caution', '#b8860b'),
    drawdownFillOpacity: Number(read('--drawdown-fill-opacity', '0.18')),
    font: '13px "Source Sans 3", system-ui, sans-serif',
  };
}

export function useChartTokens(): ChartTokens {
  const theme = useResolvedTheme();
  return useMemo(() => readChartTokens(theme), [theme]);
}

/** Hex colour with alpha, for canvas fills. Accepts #rgb or #rrggbb. */
export function withAlpha(hex: string, alpha: number): string {
  const h = hex.replace('#', '');
  const full =
    h.length === 3
      ? h
          .split('')
          .map((c) => c + c)
          .join('')
      : h;
  const n = parseInt(full, 16);
  if (Number.isNaN(n)) return hex;
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
}
