import { useCallback, useEffect, useRef, useState } from 'react';

/** Width of an element, kept up to date with a ResizeObserver. */
export function useMeasure(): [(el: HTMLElement | null) => void, number] {
  const [width, setWidth] = useState(0);
  const observer = useRef<ResizeObserver | null>(null);

  const ref = useCallback((el: HTMLElement | null) => {
    observer.current?.disconnect();
    if (!el) return;
    setWidth(Math.floor(el.getBoundingClientRect().width));
    if (typeof ResizeObserver === 'undefined') return;
    observer.current = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect.width;
      if (w !== undefined) setWidth(Math.floor(w));
    });
    observer.current.observe(el);
  }, []);

  useEffect(() => () => observer.current?.disconnect(), []);
  return [ref, width];
}
