import { Link } from 'react-router';
import type { LinkProps } from 'react-router';
import { cx } from '@/shared/lib/cx';
import styles from './Button.module.css';

/** A router link that looks like a button (for navigation, not actions). */
export function LinkButton({
  variant = 'primary',
  className,
  ...rest
}: LinkProps & { variant?: 'primary' | 'secondary' | 'quiet' }) {
  return <Link className={cx(styles.button, styles[variant], styles.link, className)} {...rest} />;
}
