import { NavLink } from 'react-router';
import styles from './TabNav.module.css';

export interface TabItem {
  to: string;
  label: string;
  end?: boolean;
}

/**
 * Section navigation styled as tabs. These are links (each tab has a URL), so they use
 * nav + aria-current rather than the ARIA tab pattern.
 */
export function TabNav({ items, label }: { items: readonly TabItem[]; label: string }) {
  return (
    <nav aria-label={label} className={styles.nav}>
      <ul className={styles.list}>
        {items.map((t) => (
          <li key={t.to}>
            <NavLink
              to={t.to}
              end={t.end ?? false}
              className={({ isActive }) =>
                isActive ? `${styles.tab} ${styles.active}` : styles.tab
              }
              preventScrollReset
            >
              {t.label}
            </NavLink>
          </li>
        ))}
      </ul>
    </nav>
  );
}
