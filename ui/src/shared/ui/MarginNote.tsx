import type { ReactNode } from 'react';
import { About } from './About';
import type { HandbookRef } from '@/shared/lib/handbook';

export { HandbookLink } from './About';

/**
 * Compatibility wrapper from the first design: a margin note is now a collapsible
 * "About" line under the content it explains (no margin column).
 */
export function MarginNote({
  children,
  handbook,
  tone = 'plain',
  className,
  label,
}: {
  children: ReactNode;
  handbook?: HandbookRef;
  tone?: 'plain' | 'caution';
  className?: string;
  label?: string;
}) {
  return (
    <About handbook={handbook} tone={tone} className={className} label={label}>
      {children}
    </About>
  );
}
