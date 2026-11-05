import { useSyncExternalStore } from 'react';

/** Theme preference store (light / dark / system), persisted to localStorage. */
export type ThemePref = 'light' | 'dark' | 'system';
export type ResolvedTheme = 'light' | 'dark';

const KEY = 'qts-theme';
const listeners = new Set<() => void>();

function readStored(): ThemePref {
  try {
    const v = localStorage.getItem(KEY);
    return v === 'light' || v === 'dark' ? v : 'system';
  } catch {
    return 'system';
  }
}

let pref: ThemePref = typeof window === 'undefined' ? 'system' : readStored();

function apply(p: ThemePref): void {
  const root = document.documentElement;
  if (p === 'system') delete root.dataset.theme;
  else root.dataset.theme = p;
}

export function setThemePref(p: ThemePref): void {
  pref = p;
  try {
    if (p === 'system') localStorage.removeItem(KEY);
    else localStorage.setItem(KEY, p);
  } catch {
    /* storage unavailable: the preference lasts for this page only */
  }
  apply(p);
  listeners.forEach((l) => l());
}

function subscribePref(cb: () => void) {
  listeners.add(cb);
  return () => listeners.delete(cb);
}

export function useThemePref(): ThemePref {
  return useSyncExternalStore(
    subscribePref,
    () => pref,
    () => 'system',
  );
}

const DARK_QUERY = '(prefers-color-scheme: dark)';

function subscribeSystem(cb: () => void) {
  const mq = window.matchMedia(DARK_QUERY);
  mq.addEventListener('change', cb);
  return () => mq.removeEventListener('change', cb);
}

function systemDark(): boolean {
  return typeof window !== 'undefined' && typeof window.matchMedia === 'function'
    ? window.matchMedia(DARK_QUERY).matches
    : false;
}

/** The theme actually in effect, resolving "system" through the media query. */
export function useResolvedTheme(): ResolvedTheme {
  const p = useThemePref();
  const dark = useSyncExternalStore(
    (cb) => (typeof window.matchMedia === 'function' ? subscribeSystem(cb) : () => undefined),
    systemDark,
    () => false,
  );
  if (p === 'system') return dark ? 'dark' : 'light';
  return p;
}
