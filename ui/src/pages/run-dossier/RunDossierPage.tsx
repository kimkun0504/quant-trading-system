import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import type { CSSProperties } from 'react';
import { useLocation, useParams } from 'react-router';
import { useRun } from '@/entities/run';
import { claimEntrance, usePrefersReducedMotion } from '@/shared/lib/motion';
import { useGlobalShortcuts } from '@/shared/lib/shortcuts';
import { ErrorState, LinkButton, Page, Skeleton, SkeletonLines } from '@/shared/ui';
import { DossierHeader } from './DossierHeader';
import { SectionNav } from './SectionNav';
import { SECTION_IDS, isSectionId } from './sections';
import type { SectionId } from './sections';
import { ArtifactsSection } from './sections/ArtifactsSection';
import { ConfigSection } from './sections/ConfigSection';
import { CostsSection } from './sections/CostsSection';
import { PerformanceSection } from './sections/PerformanceSection';
import { PortfolioSection } from './sections/PortfolioSection';
import { SignalSection } from './sections/SignalSection';
import { ValidationSection } from './sections/ValidationSection';
import { goToSection, stepSection, useScrollSpy } from './useScrollSpy';
import styles from './RunDossierPage.module.css';

/** Height of the sticky header, published as a CSS variable for scroll margins. */
function useHeaderHeight(): [(el: HTMLElement | null) => void, number | null] {
  const [height, setHeight] = useState<number | null>(null);
  const observer = useRef<ResizeObserver | null>(null);
  const ref = (el: HTMLElement | null) => {
    observer.current?.disconnect();
    if (!el) return;
    setHeight(Math.ceil(el.getBoundingClientRect().height));
    if (typeof ResizeObserver === 'undefined') return;
    observer.current = new ResizeObserver(([e]) => {
      if (e) setHeight(Math.ceil(e.target.getBoundingClientRect().height));
    });
    observer.current.observe(el);
  };
  useEffect(() => () => observer.current?.disconnect(), []);
  return [ref, height];
}

/**
 * The run dossier: one scrollable page with a sticky summary header, the seven sections
 * in reading order and a side navigation that follows the scroll. `j` / `k` step
 * between sections; `/runs/:id#section` opens on that section.
 */
export function RunDossierPage() {
  const { runId = '' } = useParams();
  const { hash } = useLocation();
  const run = useRun(runId);
  const reduced = usePrefersReducedMotion();
  const [animate] = useState(() => !reduced && claimEntrance(`run:${runId}`));
  const [headerRef, headerHeight] = useHeaderHeight();
  const topbar = useMemo(() => {
    if (typeof document === 'undefined') return 48;
    const v = getComputedStyle(document.documentElement).getPropertyValue('--topbar-height');
    return v.trim().endsWith('rem') ? parseFloat(v) * 16 : parseFloat(v) || 48;
  }, []);
  const active = useScrollSpy(SECTION_IDS, topbar + (headerHeight ?? 0) + 24);
  const activeId: SectionId | null = active && isSectionId(active) ? active : null;

  // open on the section named in the hash, once the run has rendered and the sticky
  // header has been measured (the scroll margin depends on its height)
  const ready = run.isSuccess && headerHeight !== null;
  const scrolledTo = useRef<string | null>(null);
  useLayoutEffect(() => {
    const id = hash.replace(/^#/, '');
    if (!ready || !isSectionId(id) || scrolledTo.current === hash) return;
    scrolledTo.current = hash;
    goToSection(id, 'auto');
  }, [ready, hash]);

  // Repeated presses step from the section last jumped to, not from wherever the smooth
  // scroll happens to be at that instant.
  const jumped = useRef<{ id: string; at: number } | null>(null);
  const shortcuts = useMemo(() => {
    const step = (dir: 1 | -1) => {
      const recent = jumped.current && Date.now() - jumped.current.at < 1500;
      const from = recent && jumped.current ? jumped.current.id : activeId;
      const next = stepSection(SECTION_IDS, from, dir);
      if (!next) return;
      jumped.current = { id: next, at: Date.now() };
      goToSection(next, reduced ? 'auto' : 'smooth');
    };
    return { j: () => step(1), k: () => step(-1) };
  }, [activeId, reduced]);
  useGlobalShortcuts(shortcuts);

  if (run.isPending) {
    return (
      <Page docTitle="Loading run">
        <div aria-busy="true" aria-label="Loading run" className={styles.loading}>
          <Skeleton height="1.4rem" width="45%" />
          <Skeleton height="1rem" width="70%" />
          <SkeletonLines lines={2} />
          <Skeleton height="2rem" />
          <Skeleton height="22rem" />
        </div>
      </Page>
    );
  }
  if (run.isError) {
    return (
      <Page docTitle="Run" title="Run">
        <ErrorState error={run.error} onRetry={() => void run.refetch()} />
        <div>
          <LinkButton to="/" variant="secondary">
            Back to Experiments
          </LinkButton>
        </div>
      </Page>
    );
  }

  const data = run.data;
  return (
    <Page docTitle={data.label}>
      <div
        className={styles.dossier}
        style={{ '--dossier-header-height': `${headerHeight ?? 0}px` } as CSSProperties}
      >
        <div ref={headerRef} className={styles.headerSlot}>
          <DossierHeader run={data} animate={animate} />
        </div>
        <div className={styles.layout}>
          <div className={styles.sections}>
            <PerformanceSection run={data} animate={animate} />
            <SignalSection run={data} />
            <PortfolioSection run={data} />
            <CostsSection run={data} />
            <ValidationSection run={data} />
            <ConfigSection run={data} />
            <ArtifactsSection run={data} />
          </div>
          <SectionNav active={activeId} />
        </div>
      </div>
    </Page>
  );
}
