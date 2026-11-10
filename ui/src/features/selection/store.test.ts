import { beforeEach, describe, expect, it } from 'vitest';
import {
  MAX_SELECTED,
  SELECTION_KEY,
  clearSelection,
  getSelection,
  parseSelectionParam,
  removeSelected,
  serializeSelection,
  setSelection,
  toggleSelected,
} from './store';

describe('selection store', () => {
  beforeEach(() => {
    clearSelection();
    localStorage.clear();
  });

  it('toggles, removes and clears in selection order', () => {
    toggleSelected('a');
    toggleSelected('b');
    expect(getSelection()).toEqual(['a', 'b']);
    toggleSelected('a');
    expect(getSelection()).toEqual(['b']);
    toggleSelected('c');
    removeSelected('b');
    expect(getSelection()).toEqual(['c']);
    clearSelection();
    expect(getSelection()).toEqual([]);
  });

  it('caps the selection at the compare limit and dedupes', () => {
    setSelection(Array.from({ length: MAX_SELECTED + 3 }, (_, i) => `r${i}`));
    expect(getSelection()).toHaveLength(MAX_SELECTED);
    expect(toggleSelected('zzz')).toHaveLength(MAX_SELECTED);
    setSelection(['x', 'x', 'y']);
    expect(getSelection()).toEqual(['x', 'y']);
  });

  it('persists to localStorage', () => {
    setSelection(['a', 'b']);
    expect(JSON.parse(localStorage.getItem(SELECTION_KEY) ?? '[]')).toEqual(['a', 'b']);
    clearSelection();
    expect(localStorage.getItem(SELECTION_KEY)).toBeNull();
  });

  it('round-trips the URL parameter', () => {
    expect(parseSelectionParam(' a, b ,,a')).toEqual(['a', 'b']);
    expect(parseSelectionParam(null)).toEqual([]);
    expect(serializeSelection(['a', 'b'])).toBe('a,b');
  });
});
