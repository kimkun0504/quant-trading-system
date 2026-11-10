import { cloneElement, useId, useState } from 'react';
import type { KeyboardEvent, ReactElement, ReactNode } from 'react';
import { cx } from '@/shared/lib/cx';
import styles from './Tooltip.module.css';

interface TriggerProps {
  'aria-describedby'?: string;
  onMouseEnter?: (e: unknown) => void;
  onMouseLeave?: (e: unknown) => void;
  onFocus?: (e: unknown) => void;
  onBlur?: (e: unknown) => void;
  onKeyDown?: (e: KeyboardEvent) => void;
}

/**
 * Hover / focus tooltip (WCAG 1.4.13: dismissable with Escape, hoverable in the sense
 * that it does not cover the trigger, persistent while focused). The content is also
 * the trigger's description, so screen readers hear it without opening anything.
 * The child must be focusable (a button, a link, an input).
 */
export function Tooltip({
  content,
  formula,
  children,
  align = 'start',
}: {
  content: ReactNode;
  formula?: string;
  children: ReactElement<TriggerProps>;
  align?: 'start' | 'end';
}) {
  const id = useId();
  const [open, setOpen] = useState(false);
  const child = children.props;
  const trigger = cloneElement(children, {
    'aria-describedby': cx(child['aria-describedby'], id),
    onMouseEnter: (e: unknown) => {
      child.onMouseEnter?.(e);
      setOpen(true);
    },
    onMouseLeave: (e: unknown) => {
      child.onMouseLeave?.(e);
      setOpen(false);
    },
    onFocus: (e: unknown) => {
      child.onFocus?.(e);
      setOpen(true);
    },
    onBlur: (e: unknown) => {
      child.onBlur?.(e);
      setOpen(false);
    },
    onKeyDown: (e: KeyboardEvent) => {
      child.onKeyDown?.(e);
      if (e.key === 'Escape') setOpen(false);
    },
  });
  return (
    <span className={styles.anchor}>
      {trigger}
      <span
        role="tooltip"
        id={id}
        className={cx(styles.tip, align === 'end' && styles.end)}
        data-open={open}
      >
        {content}
        {formula ? <span className={styles.formula}>{formula}</span> : null}
      </span>
    </span>
  );
}
