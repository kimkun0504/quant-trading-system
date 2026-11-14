/**
 * The library's compact UTC stamps (`20261114T042559244087Z`, run ids and sweep
 * `created_at`) as ISO 8601, so the usual date formatting applies; anything else is
 * returned unchanged.
 */
export function stampIso(s: string | null | undefined): string | null {
  if (!s) return null;
  const m = /^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})/.exec(s);
  return m ? `${m[1]}-${m[2]}-${m[3]}T${m[4]}:${m[5]}:${m[6]}Z` : s;
}
