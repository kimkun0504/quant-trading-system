import { cx } from '@/shared/lib/cx';
import styles from './Skeleton.module.css';

/** Placeholder with the final element's footprint, so loading does not shift layout. */
export function Skeleton({
  height,
  width,
  className,
}: {
  height: string;
  width?: string;
  className?: string;
}) {
  return (
    <div aria-hidden="true" className={cx(styles.skeleton, className)} style={{ height, width }} />
  );
}

export function SkeletonLines({ lines = 3 }: { lines?: number }) {
  return (
    <div aria-hidden="true" className={styles.lines}>
      {Array.from({ length: lines }, (_, i) => (
        <div
          key={i}
          className={styles.skeleton}
          style={{ height: '1rem', width: i === lines - 1 ? '60%' : '100%' }}
        />
      ))}
    </div>
  );
}
