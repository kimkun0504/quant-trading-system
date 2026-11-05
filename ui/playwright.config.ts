import { defineConfig, devices } from '@playwright/test';
// @ts-expect-error -- plain ESM helper shared with the screenshot script
import { chromiumPath } from './scripts/chromium.mjs';

const PORT = Number(process.env.E2E_PORT ?? 5173);
const executablePath = (chromiumPath as () => string | undefined)();

export default defineConfig({
  testDir: 'e2e',
  timeout: 30_000,
  fullyParallel: true,
  reporter: [['list']],
  use: {
    baseURL: `http://localhost:${PORT}`,
    trace: 'retain-on-failure',
    ...devices['Desktop Chrome'],
    launchOptions: executablePath ? { executablePath } : {},
  },
  webServer: {
    // The dev server with the MSW mock layer (VITE_USE_MOCKS=true in .env.development).
    command: `npx vite --port ${PORT} --strictPort`,
    url: `http://localhost:${PORT}`,
    reuseExistingServer: true,
    env: { VITE_USE_MOCKS: 'true' },
  },
});
