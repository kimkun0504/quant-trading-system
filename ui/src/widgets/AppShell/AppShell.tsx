import { useEffect, useRef } from 'react';
import { NavLink, Outlet, useLocation } from 'react-router';
import { ThemeToggle } from '@/features/theme-toggle';
import { HANDBOOK } from '@/shared/lib/handbook';
import { VisuallyHidden } from '@/shared/ui';
import styles from './AppShell.module.css';

const NAV = [
  { to: '/runs', label: 'Runs', end: true },
  { to: '/runs/new', label: 'New run', end: true },
  { to: '/models', label: 'Models', end: false },
  { to: '/ledger', label: 'Ledger', end: false },
] as const;

/** Navigation rail + main column. Moves focus to <main> after client-side navigation. */
export function AppShell() {
  const { pathname } = useLocation();
  const main = useRef<HTMLElement | null>(null);
  const section = pathname.split('/').slice(0, 3).join('/');
  const previous = useRef(section);

  // Move focus to <main> when the page (not a tab inside a run) changes. Comparing with
  // the previous section keeps this a no-op on first load, StrictMode re-runs included.
  useEffect(() => {
    if (previous.current === section) return;
    previous.current = section;
    main.current?.focus({ preventScroll: false });
  }, [section]);

  return (
    <div className={styles.shell}>
      <a href="#main" className={styles.skip}>
        Skip to content
      </a>
      <header className={styles.rail}>
        <NavLink to="/runs" className={styles.brand}>
          QTS Workbench
        </NavLink>
        <nav aria-label="Main" className={styles.nav}>
          <ul>
            {NAV.map((n) => (
              <li key={n.to}>
                <NavLink
                  to={n.to}
                  end={n.end}
                  className={({ isActive }) =>
                    isActive ? `${styles.link} ${styles.active}` : styles.link
                  }
                >
                  {n.label}
                </NavLink>
              </li>
            ))}
            <li>
              <a
                className={styles.link}
                href={HANDBOOK.glossary.url}
                target="_blank"
                rel="noreferrer"
              >
                Handbook
                <VisuallyHidden> (opens in a new tab)</VisuallyHidden>
              </a>
            </li>
          </ul>
        </nav>
        <div className={styles.theme}>
          <ThemeToggle />
        </div>
      </header>
      <main id="main" ref={main} tabIndex={-1} className={styles.main}>
        <Outlet />
      </main>
    </div>
  );
}
