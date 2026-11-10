import type { ReactNode } from 'react';
import { Link } from 'react-router';
import { Icons } from '@/shared/ui/icons';
import { SECTIONS } from './sections';
import type { SectionId } from './sections';
import styles from './DossierSection.module.css';

/**
 * One numbered section of the dossier: heading row (number, title, the lab that goes
 * deeper) and a body of panels. The heading is focusable so `j` / `k` can land on it.
 */
export function DossierSection({
  id,
  lede,
  children,
}: {
  id: SectionId;
  lede?: ReactNode;
  children: ReactNode;
}) {
  const index = SECTIONS.findIndex((s) => s.id === id);
  const def = SECTIONS[index];
  if (!def) return null;
  return (
    <section id={id} aria-labelledby={`${id}-title`} className={styles.section}>
      <div className={styles.head}>
        <h2 id={`${id}-title`} tabIndex={-1} className={styles.title}>
          <span className={styles.index} aria-hidden="true">
            {index + 1}
          </span>
          {def.title}
        </h2>
        {lede ? <p className={styles.lede}>{lede}</p> : null}
        {def.lab ? (
          <Link to={def.lab.to} className={styles.lab}>
            Open in {def.lab.label}
            <Icons.External />
          </Link>
        ) : null}
      </div>
      <div className={styles.body}>{children}</div>
    </section>
  );
}
