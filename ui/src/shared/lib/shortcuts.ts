import { useEffect } from 'react';

/** True when a key press should not be treated as a shortcut (the user is typing). */
export function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  const tag = target.tagName;
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT';
}

export type ShortcutMap = Record<string, (e: KeyboardEvent) => void>;

/**
 * Single-key shortcuts (no modifiers) that are ignored while typing in a field.
 * Keys are `KeyboardEvent.key` values ("/", "c").
 */
export function useGlobalShortcuts(map: ShortcutMap): void {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey || e.defaultPrevented) return;
      if (isTypingTarget(e.target)) return;
      const handler = map[e.key];
      if (!handler) return;
      e.preventDefault();
      handler(e);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [map]);
}
