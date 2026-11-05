import { defineConfig, devices } from '@playwright/test';
// @ts-expect-error -- plain ESM helper shared with the screenshot script
import { chromiumPath } from './scripts/chromium.mjs';

/**
 * End-to-end smoke against the REAL API: uvicorn serves the built UI (`npm run build`
 * first; `npm run test:e2e:real` does both) and the API over a runs root holding the
 * fixture run written by the pipeline (api/scripts/e2e_server.py).
 */
const PORT = Number(process.env.E2E_REAL_PORT ?? 8765);
const executablePath = (chromiumPath as () => string | undefined)();

export default defineConfig({
  testDir: 'e2e-real',
  timeout: 60_000,
  reporter: [['list']],
  use: {
    baseURL: `http://localhost:${PORT}`,
    trace: 'retain-on-failure',
    ...devices['Desktop Chrome'],
    launchOptions: executablePath ? { executablePath } : {},
  },
  webServer: {
    command: `python ../api/scripts/e2e_server.py --port ${PORT}`,
    url: `http://localhost:${PORT}/api/health`,
    reuseExistingServer: false,
    timeout: 240_000,
    stdout: 'pipe',
  },
});
