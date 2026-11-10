import { setThemePref, useResolvedTheme } from '@/shared/lib/theme';
import { IconButton } from '@/shared/ui';
import { Icons } from '@/shared/ui/icons';

/**
 * One button that flips between light and dark. Until it is pressed the app follows
 * the system; the first press stores an explicit preference.
 */
export function ThemeToggle() {
  const resolved = useResolvedTheme();
  const next = resolved === 'dark' ? 'light' : 'dark';
  return (
    <IconButton
      label={`Switch to ${next} theme`}
      onClick={() => setThemePref(next)}
      data-theme-toggle={resolved}
    >
      {resolved === 'dark' ? <Icons.Sun /> : <Icons.Moon />}
    </IconButton>
  );
}
