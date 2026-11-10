import type { RunFile } from '@/shared/api';
import { API_BASE_URL } from '@/shared/api';

export function fileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(bytes < 10 * 1024 ? 1 : 0)} kB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

/** The API serves `/api/runs/{id}/files/{name}`; honour `VITE_API_URL` when it is set. */
export function downloadHref(f: RunFile): string {
  return `${API_BASE_URL}${f.download_url.replace(/^\/api/, '')}`;
}
