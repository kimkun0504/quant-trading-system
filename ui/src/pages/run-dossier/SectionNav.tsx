import { Kbd } from '@/shared/ui';
import { cx } from '@/shared/lib/cx';
import { SECTIONS } from './sections';
import type { SectionId } from './sections';
import { goToSection } from './useScrollSpy';
import styles from './SectionNav.module.css';

/**
 * In-page navigation of the dossier: the sections in reading order, the one being
 * read marked (scrollspy). Sticky beside the content on wide screens; a row of links
 * above it on narrow ones.
 */
export function SectionNav({ active }: { active: SectionId | null }) {
  return (
    <nav aria-label="Run sections" className={styles.nav}>
      <ol className={styles.list}>
        {SECTIONS.map((s, i) => (
          <li key={s.id}>
            <a
              href={`#${s.id}`}
              className={cx(styles.link, active === s.id && styles.active)}
              aria-current={active === s.id ? 'location' : undefined}
              onClick={(e) => {
                e.preventDefault();
                history.replaceState(null, '', `#${s.id}`);
                goToSection(s.id);
              }}
            >
              <span className={styles.index}>{i + 1}</span>
              <span>{s.title}</span>
            </a>
          </li>
        ))}
      </ol>
      <p className={styles.hint}>
        <Kbd>j</Kbd> <Kbd>k</Kbd> next / previous section
      </p>
    </nav>
  );
}
