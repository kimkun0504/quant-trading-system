import { useSyncExternalStore } from 'react';

/**
 * The selection tray's state: run ids chosen for comparison, in selection order.
 * Persisted to localStorage; `SelectionUrlSync` mirrors it into `?sel=` so a URL
 * carries the selection.
 */
export const SELECTION_KEY = 'qts-selection';
export const MAX_SELECTED = 8;
export const SELECTION_PARAM = 'sel';

const listeners = new Set<() => void>();

function readStored(): string[] {
  try {
    const raw = localStorage.getItem(SELECTION_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    return Array.isArray(parsed)
      ? parsed.filter((x): x is string => typeof x === 'string').slice(0, MAX_SELECTED)
      : [];
  } catch {
    return [];
  }
}

let ids: readonly string[] = typeof window === 'undefined' ? [] : readStored();
const EMPTY: readonly string[] = [];

function commit(next: readonly string[]): void {
  if (next.length === ids.length && next.every((x, i) => x === ids[i])) return;
  ids = next;
  try {
    if (next.length) localStorage.setItem(SELECTION_KEY, JSON.stringify(next));
    else localStorage.removeItem(SELECTION_KEY);
  } catch {
    /* storage unavailable: the selection lasts for this page only */
  }
  listeners.forEach((l) => l());
}

export function getSelection(): readonly string[] {
  return ids;
}

export function setSelection(next: readonly string[]): void {
  commit([...new Set(next)].slice(0, MAX_SELECTED));
}

/** Adds the id (if there is room) or removes it. Returns the new state. */
export function toggleSelected(id: string): readonly string[] {
  if (ids.includes(id)) commit(ids.filter((x) => x !== id));
  else if (ids.length < MAX_SELECTED) commit([...ids, id]);
  return ids;
}

export function removeSelected(id: string): void {
  commit(ids.filter((x) => x !== id));
}

export function clearSelection(): void {
  commit(EMPTY);
}

export function parseSelectionParam(value: string | null): string[] {
  if (!value) return [];
  return [
    ...new Set(
      value
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean),
    ),
  ].slice(0, MAX_SELECTED);
}

export function serializeSelection(sel: readonly string[]): string {
  return sel.join(',');
}

function subscribe(cb: () => void) {
  listeners.add(cb);
  return () => listeners.delete(cb);
}

export function useSelection(): readonly string[] {
  return useSyncExternalStore(subscribe, getSelection, () => EMPTY);
}
