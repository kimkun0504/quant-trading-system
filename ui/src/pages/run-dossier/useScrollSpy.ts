import { useCallback, useEffect, useState } from 'react';

/**
 * Which of `ids` is being read: the last section whose top has passed `offset` pixels
 * from the top of the viewport (so a section counts as current once its heading is under
 * the sticky header), or the first section before any has. Recomputed on scroll and
 * resize, at most once per animation frame.
 */
export function useScrollSpy(ids: readonly string[], offset: number): string | null {
  const [active, setActive] = useState<string | null>(ids[0] ?? null);

  const compute = useCallback(() => {
    let current: string | null = ids[0] ?? null;
    const scrollHeight = document.documentElement.scrollHeight;
    const atBottom =
      scrollHeight > window.innerHeight && window.innerHeight + window.scrollY >= scrollHeight - 2;
    if (atBottom && ids.length) {
      // the last section is current when the page cannot scroll any further
      for (let i = ids.length - 1; i >= 0; i--) {
        if (document.getElementById(ids[i] ?? '')) return ids[i] ?? current;
      }
    }
    for (const id of ids) {
      const el = document.getElementById(id);
      if (!el) continue;
      if (el.getBoundingClientRect().top - offset <= 1) current = id;
      else break;
    }
    return current;
  }, [ids, offset]);

  useEffect(() => {
    let raf = 0;
    const schedule = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => setActive(compute()));
    };
    schedule();
    window.addEventListener('scroll', schedule, { passive: true });
    window.addEventListener('resize', schedule);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('scroll', schedule);
      window.removeEventListener('resize', schedule);
    };
  }, [compute]);

  return active;
}

/** Scroll a section under the sticky header and move focus to its heading. */
export function goToSection(id: string, behavior: ScrollBehavior = 'smooth'): void {
  const el = document.getElementById(id);
  if (!el) return;
  el.scrollIntoView({ behavior, block: 'start' });
  const heading = el.querySelector<HTMLElement>('h2');
  heading?.focus({ preventScroll: true });
}

/** The section `steps` away from `current` in `ids`, clamped to the ends. */
export function stepSection(
  ids: readonly string[],
  current: string | null,
  steps: 1 | -1,
): string | null {
  if (!ids.length) return null;
  const i = current ? ids.indexOf(current) : -1;
  const next = Math.max(0, Math.min(ids.length - 1, (i < 0 ? 0 : i) + steps));
  return ids[next] ?? null;
}
