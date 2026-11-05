/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Base URL of the qts API, e.g. http://localhost:8000. Empty = same origin. */
  readonly VITE_API_URL?: string;
  /** "true" starts the MSW service worker and serves fixture data. */
  readonly VITE_USE_MOCKS?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
