import { useEffect, useRef } from 'react';
import uPlot from 'uplot';
import 'uplot/dist/uPlot.min.css';
import { useMeasure } from '@/shared/lib/useMeasure';
import styles from './UplotChart.module.css';

export interface UplotChartProps {
  data: uPlot.AlignedData;
  /** Builds options for a given width. Must be stable (useCallback) or the chart rebuilds. */
  makeOptions: (width: number) => uPlot.Options;
  height: number;
  ariaLabel: string;
  onCursor?: (idx: number | null) => void;
  /** Reveal the first series left to right once (≤ 600 ms). */
  animate?: boolean;
  animateMs?: number;
}

/** Thin React wrapper around uPlot: create on options/data change, resize on width change. */
export function UplotChart({
  data,
  makeOptions,
  height,
  ariaLabel,
  onCursor,
  animate = false,
  animateMs = 600,
}: UplotChartProps) {
  const [measureRef, width] = useMeasure();
  const host = useRef<HTMLDivElement | null>(null);
  const plot = useRef<uPlot | null>(null);
  const cursorCb = useRef(onCursor);
  useEffect(() => {
    cursorCb.current = onCursor;
  }, [onCursor]);

  const hasWidth = width > 0;

  useEffect(() => {
    const el = host.current;
    if (!el || !hasWidth) return;
    const w = el.getBoundingClientRect().width;
    const base = makeOptions(w);
    const opts: uPlot.Options = {
      ...base,
      width: w,
      height,
      hooks: {
        ...base.hooks,
        setCursor: [
          ...(base.hooks?.setCursor ?? []),
          (u: uPlot) => cursorCb.current?.(u.cursor.idx ?? null),
        ],
      },
    };
    let raf = 0;
    if (animate && data.length > 1) {
      const n = data[0].length;
      const start = performance.now();
      const full = data;
      const partial = (k: number): uPlot.AlignedData =>
        full.map((s, i) =>
          i === 0 ? s : Array.from(s, (v, j) => (j < k ? v : null)),
        ) as uPlot.AlignedData;
      const u = new uPlot(opts, partial(1), el);
      plot.current = u;
      const step = (t: number) => {
        const p = Math.min(1, (t - start) / animateMs);
        const eased = 1 - (1 - p) ** 3;
        u.setData(p >= 1 ? full : partial(Math.max(1, Math.round(eased * n))), false);
        u.redraw(false, false);
        if (p < 1) raf = requestAnimationFrame(step);
      };
      raf = requestAnimationFrame(step);
    } else {
      plot.current = new uPlot(opts, data, el);
    }
    return () => {
      cancelAnimationFrame(raf);
      plot.current?.destroy();
      plot.current = null;
    };
    // width changes are handled by setSize below; rebuild only on content changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, makeOptions, height, hasWidth]);

  useEffect(() => {
    if (plot.current && width > 0) plot.current.setSize({ width, height });
  }, [width, height]);

  return (
    <div
      ref={(el) => {
        host.current = el;
        measureRef(el);
      }}
      role="img"
      aria-label={ariaLabel}
      className={styles.host}
      style={{ minHeight: height }}
    />
  );
}
