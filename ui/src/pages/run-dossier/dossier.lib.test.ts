import { describe, expect, it } from 'vitest';
import { spansWhere } from '@/shared/charts/spans';
import { groupLeaves } from './lib/configGroups';
import { downloadHref, fileSize } from './lib/files';
import { SECTION_IDS, TAB_TO_SECTION, isSectionId } from './sections';
import { stepSection } from './useScrollSpy';

describe('dossier helpers', () => {
  it('steps between sections and clamps at the ends', () => {
    expect(stepSection(SECTION_IDS, 'performance', 1)).toBe('signal');
    expect(stepSection(SECTION_IDS, 'signal', -1)).toBe('performance');
    expect(stepSection(SECTION_IDS, 'performance', -1)).toBe('performance');
    expect(stepSection(SECTION_IDS, 'artifacts', 1)).toBe('artifacts');
    expect(stepSection(SECTION_IDS, null, 1)).toBe('signal');
    expect(stepSection([], null, 1)).toBeNull();
  });

  it('maps every old tab to a section', () => {
    for (const tab of ['performance', 'risk', 'alpha', 'costs', 'execution', 'validation']) {
      const section = TAB_TO_SECTION[tab];
      expect(section && isSectionId(section)).toBe(true);
    }
    expect(TAB_TO_SECTION.risk).toBe('portfolio');
    expect(TAB_TO_SECTION.alpha).toBe('signal');
    expect(isSectionId('nope')).toBe(false);
  });

  it('turns empty-book days into inclusive spans', () => {
    const gross = [2, 0, 0, 2, 2, 0, 2, 0];
    expect(spansWhere(gross.length, (i) => (gross[i] ?? 1) <= 1e-4)).toEqual([
      { from: 1, to: 2 },
      { from: 5, to: 5 },
      { from: 7, to: 7 },
    ]);
    expect(spansWhere(3, () => false)).toEqual([]);
  });

  it('groups config leaves by section and filters by key or value', () => {
    const leaves = {
      'portfolio.trade_aversion': 1,
      'alpha.horizon_days': 20,
      'alpha.themes': '["value","momentum"]',
      seed: 0,
      'cost.spread_bps': 2.5,
    };
    const all = groupLeaves(leaves, '');
    expect(all.map((g) => g.section)).toEqual(['alpha', 'cost', 'portfolio', 'top level']);
    expect(all[0]?.rows.map((r) => r.leaf)).toEqual(['horizon_days', 'themes']);
    expect(groupLeaves(leaves, 'aversion').flatMap((g) => g.rows.map((r) => r.key))).toEqual([
      'portfolio.trade_aversion',
    ]);
    expect(groupLeaves(leaves, 'momentum').flatMap((g) => g.rows.map((r) => r.key))).toEqual([
      'alpha.themes',
    ]);
    expect(groupLeaves(leaves, '2.5')[0]?.rows[0]?.value).toBe('2.5');
  });

  it('formats file sizes and builds download links under the API base', () => {
    expect(fileSize(512)).toBe('512 B');
    expect(fileSize(3_000)).toBe('2.9 kB');
    expect(fileSize(60_000)).toBe('59 kB');
    expect(fileSize(1_900_000)).toBe('1.8 MB');
    expect(
      downloadHref({
        name: 'nav.parquet',
        size_bytes: 1,
        modified_at: '',
        kind: 'parquet',
        description: '',
        download_url: '/api/runs/r1/files/nav.parquet',
      }),
    ).toMatch(/\/api\/runs\/r1\/files\/nav\.parquet$/);
  });
});
