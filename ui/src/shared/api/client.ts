import type { ApiErrorBody } from './schema';

/**
 * Root URL of the API, `/api` included. Default `/api` (same origin: the Vite dev server
 * proxies it to the API, and the API serves the built UI itself with QTS_SERVE_UI=1).
 * Set VITE_API_URL=http://localhost:8000/api to call the API directly (CORS).
 */
export const API_BASE_URL: string = (import.meta.env.VITE_API_URL || '/api').replace(/\/$/, '');

export class ApiError extends Error {
  readonly status: number;
  readonly detail: string;

  constructor(status: number, detail: string) {
    super(`API ${status}: ${detail}`);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

type QueryValue = string | number | boolean | undefined | null;

export function buildUrl(path: string, query?: Record<string, QueryValue>): string {
  const url = `${API_BASE_URL}${path}`;
  if (!query) return url;
  const params = new URLSearchParams();
  for (const [k, v] of Object.entries(query)) {
    if (v !== undefined && v !== null && v !== '') params.set(k, String(v));
  }
  const qs = params.toString();
  return qs ? `${url}?${qs}` : url;
}

function isErrorBody(x: unknown): x is ApiErrorBody {
  return (
    typeof x === 'object' && x !== null && typeof (x as { detail?: unknown }).detail === 'string'
  );
}

async function parse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText || 'request failed';
    try {
      const body: unknown = await res.json();
      if (isErrorBody(body)) detail = body.detail;
    } catch {
      /* non-JSON error body: keep statusText */
    }
    throw new ApiError(res.status, detail);
  }
  const type = res.headers.get('content-type') ?? '';
  if (!type.includes('json')) {
    throw new ApiError(
      res.status,
      `Expected JSON from ${res.url || 'the API'} but got ${type || 'no content type'}. Check that VITE_API_URL points at the qts API, or set VITE_USE_MOCKS=true.`,
    );
  }
  return (await res.json()) as T;
}

export async function apiGet<T>(
  path: string,
  query?: Record<string, QueryValue>,
  signal?: AbortSignal,
): Promise<T> {
  const res = await fetch(buildUrl(path, query), {
    headers: { Accept: 'application/json' },
    signal: signal ?? null,
  });
  return parse<T>(res);
}

export async function apiDelete<T>(path: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(buildUrl(path), {
    method: 'DELETE',
    headers: { Accept: 'application/json' },
    signal: signal ?? null,
  });
  return parse<T>(res);
}

export async function apiPost<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  const res = await fetch(buildUrl(path), {
    method: 'POST',
    headers: { Accept: 'application/json', 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal: signal ?? null,
  });
  return parse<T>(res);
}
