import { useEffect, useMemo, useRef } from 'react';
import { NavLink, Outlet, useLocation, useNavigate } from 'react-router';
import { QueueIndicator } from '@/features/run-queue';
import { RunSearch } from '@/features/run-search';
import { SelectionUrlSync } from '@/features/selection';
import { ThemeToggle } from '@/features/theme-toggle';
import { HANDBOOK } from '@/shared/lib/handbook';
import { useGlobalShortcuts } from '@/shared/lib/shortcuts';
import { cx } from '@/shared/lib/cx';
import { VisuallyHidden } from '@/shared/ui';
import { Icons } from '@/shared/ui/icons';
import { SelectionTray } from '@/widgets/SelectionTray';
import { REGISTRY_NAV, WORKSPACES } from './nav';
import type { NavEntry } from './nav';
import styles from './AppShell.module.css';

function NavItem({ entry }: { entry: NavEntry }) {
  const Glyph = Icons[entry.icon];
  return (
    <li>
      <NavLink
        to={entry.to}
        end={entry.end ?? false}
        className={({ isActive }) => cx(styles.link, isActive && styles.active)}
      >
        <Glyph className={styles.glyph} />
        <span>{entry.label}</span>
      </NavLink>
    </li>
  );
}

/**
 * Sidebar (workspaces) + top bar (search, theme) + main + the selection tray.
 * Moves focus to <main> after a client-side page change; `/` focuses search,
 * `c` opens the compare view.
 */
export function AppShell() {
  const { pathname } = useLocation();
  const navigate = useNavigate();
  const main = useRef<HTMLElement | null>(null);
  const search = useRef<HTMLInputElement | null>(null);
  const section = pathname.split('/').slice(0, 3).join('/');
  const previous = useRef(section);

  useEffect(() => {
    if (previous.current === section) return;
    previous.current = section;
    main.current?.focus({ preventScroll: false });
  }, [section]);

  const shortcuts = useMemo(
    () => ({
      '/': () => search.current?.focus(),
      c: () => void navigate('/compare'),
    }),
    [navigate],
  );
  useGlobalShortcuts(shortcuts);

  return (
    <div className={styles.shell}>
      <SelectionUrlSync />
      <a href="#main" className={styles.skip}>
        Skip to content
      </a>
      <aside className={styles.sidebar}>
        <NavLink to="/" className={styles.brand} end>
          <span className={styles.mark} aria-hidden="true" />
          QTS research
        </NavLink>
        <nav aria-label="Workspaces" className={styles.nav}>
          <ul>
            {WORKSPACES.map((n) => (
              <NavItem key={n.to} entry={n} />
            ))}
          </ul>
          <div className={styles.navGroup}>Registry</div>
          <ul>
            {REGISTRY_NAV.map((n) => (
              <NavItem key={n.to} entry={n} />
            ))}
            <li>
              <a
                className={styles.link}
                href={HANDBOOK.glossary.url}
                target="_blank"
                rel="noreferrer"
              >
                <Icons.Book className={styles.glyph} />
                <span>Handbook</span>
                <VisuallyHidden> (opens in a new tab)</VisuallyHidden>
              </a>
            </li>
          </ul>
        </nav>
      </aside>
      <header className={styles.topbar}>
        <RunSearch ref={search} />
        <div className={styles.topbarEnd}>
          <QueueIndicator />
          <ThemeToggle />
        </div>
      </header>
      <main id="main" ref={main} tabIndex={-1} className={styles.main}>
        <Outlet />
      </main>
      <SelectionTray />
    </div>
  );
}
