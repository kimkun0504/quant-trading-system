import { useEffect, useRef } from 'react';
import { useLocation, useSearchParams } from 'react-router';
import {
  SELECTION_PARAM,
  parseSelectionParam,
  serializeSelection,
  setSelection,
  useSelection,
} from './store';

/**
 * Keeps `?sel=a,b` and the selection store in step. On first load a selection in the
 * URL wins (a shared link shows what it was meant to show); afterwards the store is
 * the source of truth and every navigation keeps the parameter.
 */
export function SelectionUrlSync() {
  const ids = useSelection();
  const { pathname } = useLocation();
  const [params, setParams] = useSearchParams();
  const seeded = useRef(false);

  useEffect(() => {
    if (seeded.current) return;
    seeded.current = true;
    const fromUrl = parseSelectionParam(params.get(SELECTION_PARAM));
    if (fromUrl.length) setSelection(fromUrl);
  }, [params]);

  useEffect(() => {
    if (!seeded.current) return;
    const want = serializeSelection(ids);
    const have = params.get(SELECTION_PARAM) ?? '';
    if (want === have) return;
    setParams(
      (prev) => {
        const next = new URLSearchParams(prev);
        if (want) next.set(SELECTION_PARAM, want);
        else next.delete(SELECTION_PARAM);
        return next;
      },
      { replace: true, preventScrollReset: true },
    );
  }, [ids, pathname, params, setParams]);

  return null;
}
