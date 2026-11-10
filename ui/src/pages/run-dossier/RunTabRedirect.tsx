import { Navigate, useParams } from 'react-router';
import { NotFoundPage } from '@/pages/not-found/NotFoundPage';
import { TAB_TO_SECTION } from './sections';

/** `/runs/:id/<old tab>` → `/runs/:id#<section>` (the dossier replaced the tabs). */
export function RunTabRedirect() {
  const { runId = '', tab = '' } = useParams();
  const section = TAB_TO_SECTION[tab];
  if (!section) return <NotFoundPage />;
  return <Navigate to={`/runs/${encodeURIComponent(runId)}#${section}`} replace />;
}
