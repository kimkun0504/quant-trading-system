import type { ButtonHTMLAttributes, ReactNode } from 'react';
import { cx } from '@/shared/lib/cx';
import styles from './Button.module.css';
import { Tooltip } from './Tooltip';

export interface IconButtonProps extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, 'children'> {
  /** The accessible name; shown as a tooltip on hover and focus. */
  label: string;
  variant?: 'secondary' | 'quiet';
  size?: 'md' | 'sm';
  children: ReactNode;
}

/** A square button whose only visible content is an icon; the label lives in a tooltip. */
export function IconButton({
  label,
  variant = 'quiet',
  size = 'md',
  className,
  type = 'button',
  children,
  ...rest
}: IconButtonProps) {
  return (
    <Tooltip content={label}>
      <button
        type={type}
        aria-label={label}
        className={cx(
          styles.button,
          styles.icon,
          styles[variant],
          size === 'sm' && styles.sm,
          className,
        )}
        {...rest}
      >
        {children}
      </button>
    </Tooltip>
  );
}
