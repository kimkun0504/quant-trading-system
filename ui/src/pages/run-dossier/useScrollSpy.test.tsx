import { act, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { SectionNav } from './SectionNav';
import { SECTION_IDS } from './sections';
import { useScrollSpy } from './useScrollSpy';

function Harness({ offset }: { offset: number }) {
  const active = useScrollSpy(SECTION_IDS, offset);
  return (
    <>
      <output data-testid="active">{active}</output>
      {SECTION_IDS.map((id) => (
        <section key={id} id={id} />
      ))}
    </>
  );
}

/** Place each section's top at `tops[i]` px from the viewport top. */
function layout(tops: readonly number[]) {
  SECTION_IDS.forEach((id, i) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.getBoundingClientRect = () => ({ top: tops[i] ?? 9999 }) as DOMRect;
  });
}

describe('useScrollSpy', () => {
  beforeEach(() => {
    vi.stubGlobal('requestAnimationFrame', (cb: FrameRequestCallback) => {
      cb(0);
      return 1;
    });
    vi.stubGlobal('cancelAnimationFrame', () => undefined);
    Object.defineProperty(document.documentElement, 'scrollHeight', {
      configurable: true,
      value: 10_000,
    });
  });
  afterEach(() => vi.unstubAllGlobals());

  it('marks the last section whose top has passed the offset, the first before any has', () => {
    render(<Harness offset={100} />);
    layout([400, 1200, 2000, 2800, 3600, 4400, 5200]);
    act(() => {
      window.dispatchEvent(new Event('scroll'));
    });
    expect(screen.getByTestId('active')).toHaveTextContent('performance');

    layout([-500, 60, 900, 1700, 2500, 3300, 4100]);
    act(() => {
      window.dispatchEvent(new Event('scroll'));
    });
    expect(screen.getByTestId('active')).toHaveTextContent('signal');

    layout([-3000, -2200, -1400, -600, 90, 980, 1800]);
    act(() => {
      window.dispatchEvent(new Event('scroll'));
    });
    expect(screen.getByTestId('active')).toHaveTextContent('validation');
  });

  it('counts the last section as current at the bottom of the page', () => {
    render(<Harness offset={100} />);
    layout([-5000, -4200, -3400, -2600, -1800, -1000, 300]);
    Object.defineProperty(window, 'scrollY', { configurable: true, value: 10_000 });
    act(() => {
      window.dispatchEvent(new Event('scroll'));
    });
    expect(screen.getByTestId('active')).toHaveTextContent('artifacts');
    Object.defineProperty(window, 'scrollY', { configurable: true, value: 0 });
  });
});

describe('SectionNav', () => {
  it('lists the sections in order and marks the active one', () => {
    render(<SectionNav active="costs" />);
    const nav = screen.getByRole('navigation', { name: 'Run sections' });
    const links = nav.querySelectorAll('a');
    expect([...links].map((a) => a.textContent)).toEqual([
      '1Performance',
      '2Signal',
      '3Portfolio',
      '4Costs',
      '5Validation',
      '6Config',
      '7Artifacts',
    ]);
    expect(screen.getByRole('link', { name: /Costs/ })).toHaveAttribute('aria-current', 'location');
    expect(screen.getByRole('link', { name: /Performance/ })).not.toHaveAttribute('aria-current');
  });
});
