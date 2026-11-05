import { http, HttpResponse } from 'msw';
import { describe, expect, it } from 'vitest';
import { server } from '@/mocks/node';
import { api } from './endpoints';
import { ApiError, buildUrl } from './client';

describe('api client', () => {
  it('builds query strings and skips empty values', () => {
    expect(buildUrl('/api/runs/x/holdings', { date: undefined })).toMatch(
      /\/api\/runs\/x\/holdings$/,
    );
    expect(buildUrl('/api/runs/x/series', { names: 'nav,net' })).toMatch(/\?names=nav%2Cnet$/);
  });

  it('surfaces the API detail on errors', async () => {
    await expect(api.getRun('missing')).rejects.toMatchObject({
      status: 404,
      detail: 'No run with id missing',
    });
  });

  it('explains a non-JSON answer (wrong VITE_API_URL)', async () => {
    server.use(
      http.get(
        '*/api/runs',
        () => new HttpResponse('<!doctype html>', { headers: { 'content-type': 'text/html' } }),
      ),
    );
    const err = await api.listRuns().catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).detail).toMatch(/VITE_API_URL/);
  });

  it('returns the series that were asked for', async () => {
    const s = await api.getRunSeries('20261104T175159Z_944dd5b4', ['nav', 'drawdown']);
    expect(Object.keys(s.series)).toEqual(['nav', 'drawdown']);
    expect(s.dates).toHaveLength(1000);
    expect(s.series.nav).toHaveLength(1000);
  });
});
