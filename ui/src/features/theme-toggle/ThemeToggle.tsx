import { SegmentedControl } from '@/shared/ui';
import { setThemePref, useThemePref } from '@/shared/lib/theme';
import type { ThemePref } from '@/shared/lib/theme';

const OPTIONS: readonly { value: ThemePref; label: string }[] = [
  { value: 'light', label: 'Light' },
  { value: 'dark', label: 'Dark' },
  { value: 'system', label: 'System' },
];

export function ThemeToggle() {
  const pref = useThemePref();
  return <SegmentedControl legend="Theme" options={OPTIONS} value={pref} onChange={setThemePref} />;
}
