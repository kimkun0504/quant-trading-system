import { useSyncExternalStore } from 'react';

const QUERY = '(prefers-reduced-motion: reduce)';

function subscribe(cb: () => void) {
  if (typeof window.matchMedia !== 'function') return () => undefined;
  const mq = window.matchMedia(QUERY);
  mq.addEventListener('change', cb);
  return () => mq.removeEventListener('change', cb);
}

export function usePrefersReducedMotion(): boolean {
  return useSyncExternalStore(
    subscribe,
    () => (typeof window.matchMedia === 'function' ? window.matchMedia(QUERY).matches : false),
    () => true,
  );
}

/**
 * The one orchestrated motion moment (DESIGN.md): it plays the first time a run is
 * opened in this page session, not on every tab switch or refetch.
 */
const played = new Map<string, number>();
const STRICT_MODE_WINDOW_MS = 250;

export function claimEntrance(key: string, now = Date.now()): boolean {
  const at = played.get(key);
  if (at === undefined) {
    played.set(key, now);
    return true;
  }
  // React StrictMode runs initialisers twice in development: same claim, same answer.
  return now - at < STRICT_MODE_WINDOW_MS;
}
