import { useOutletContext } from 'react-router';
import type { RunDetail } from '@/shared/api';

export interface RunOutletContext {
  run: RunDetail;
  /** True on the first open of this run in the session (the orchestrated entrance). */
  animate: boolean;
}

export function useRunOutlet(): RunOutletContext {
  return useOutletContext<RunOutletContext>();
}
