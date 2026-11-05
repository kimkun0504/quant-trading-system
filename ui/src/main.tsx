import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import '@fontsource/source-serif-4/latin-400.css';
import '@fontsource/source-serif-4/latin-400-italic.css';
import '@fontsource/source-serif-4/latin-600.css';
import '@fontsource/source-sans-3/latin-400.css';
import '@fontsource/source-sans-3/latin-600.css';
import '@/shared/styles/global.css';
import { App } from '@/app/App';

/** MSW runs only when VITE_USE_MOCKS=true; otherwise the chunk is never loaded. */
async function enableMocks(): Promise<void> {
  if (import.meta.env.VITE_USE_MOCKS !== 'true') return;
  const { worker } = await import('@/mocks/browser');
  await worker.start({ onUnhandledFrame: 'bypass', quiet: true });
}

void enableMocks().then(() => {
  const root = document.getElementById('root');
  if (!root) throw new Error('#root missing from index.html');
  createRoot(root).render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
});
