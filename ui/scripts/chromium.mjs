// Resolve a Chromium binary: PW_CHROMIUM_PATH, then Playwright's own, then the
// preinstalled browsers under PLAYWRIGHT_BROWSERS_PATH (/opt/pw-browsers here).
import { existsSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { chromium } from '@playwright/test';

export function chromiumPath() {
  if (process.env.PW_CHROMIUM_PATH) return process.env.PW_CHROMIUM_PATH;
  try {
    const own = chromium.executablePath();
    if (own && existsSync(own)) return undefined; // Playwright finds it itself
  } catch {
    /* fall through */
  }
  const root = process.env.PLAYWRIGHT_BROWSERS_PATH || '/opt/pw-browsers';
  if (!existsSync(root)) return undefined;
  const dirs = readdirSync(root)
    .filter((d) => /^chromium-\d+$/.test(d))
    .sort()
    .reverse();
  for (const d of dirs) {
    const p = join(root, d, 'chrome-linux64', 'chrome');
    if (existsSync(p)) return p;
  }
  return undefined;
}
