import type { ButtonHTMLAttributes } from 'react';
import { cx } from '@/shared/lib/cx';
import styles from './Button.module.css';

export type ButtonVariant = 'primary' | 'secondary' | 'quiet';
export type ButtonSize = 'md' | 'sm';

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
}

export function Button({
  variant = 'secondary',
  size = 'md',
  className,
  type = 'button',
  ...rest
}: ButtonProps) {
  return (
    <button
      type={type}
      className={cx(styles.button, styles[variant], size === 'sm' && styles.sm, className)}
      {...rest}
    />
  );
}
