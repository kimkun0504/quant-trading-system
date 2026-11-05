/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Root URL of the qts API, `/api` included (default `/api`, same origin). */
  readonly VITE_API_URL?: string;
  /** "true" starts the MSW service worker and serves fixture data. */
  readonly VITE_USE_MOCKS?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
