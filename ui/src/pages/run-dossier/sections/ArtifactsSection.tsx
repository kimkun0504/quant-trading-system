import type { RunDetail, RunFile } from '@/shared/api';
import { useRunFiles } from '@/entities/run';
import { dateTime } from '@/shared/lib/format';
import { About, ErrorState, Panel, Skeleton, SortableTable } from '@/shared/ui';
import type { Column } from '@/shared/ui';
import { Icons } from '@/shared/ui/icons';
import { DossierSection } from '../DossierSection';
import { downloadHref, fileSize } from '../lib/files';
import styles from './Sections.module.css';

const KIND_LABEL: Record<RunFile['kind'], string> = {
  json: 'JSON',
  yaml: 'YAML',
  parquet: 'Parquet',
  text: 'Text',
  other: 'File',
};

const COLUMNS: Column<RunFile>[] = [
  {
    key: 'name',
    header: 'File',
    headerText: 'File',
    rowHeader: true,
    sortValue: (f) => f.name,
    render: (f) => <span className={styles.fileName}>{f.name}</span>,
  },
  {
    key: 'description',
    header: 'Contents',
    headerText: 'Contents',
    render: (f) => <span className={styles.reason}>{f.description || '—'}</span>,
  },
  {
    key: 'kind',
    header: 'Kind',
    headerText: 'Kind',
    sortValue: (f) => f.kind,
    render: (f) => KIND_LABEL[f.kind],
  },
  {
    key: 'size',
    header: 'Size',
    headerText: 'Size',
    align: 'end',
    sortValue: (f) => f.size_bytes,
    render: (f) => fileSize(f.size_bytes),
  },
  {
    key: 'modified',
    header: 'Written',
    headerText: 'Written',
    sortValue: (f) => f.modified_at,
    render: (f) => dateTime(f.modified_at),
  },
  {
    key: 'download',
    header: '',
    headerText: 'Download',
    render: (f) => (
      <a href={downloadHref(f)} download className={styles.download}>
        <Icons.Download />
        Download
      </a>
    ),
  },
];

export function ArtifactsSection({ run }: { run: RunDetail }) {
  const files = useRunFiles(run.id);
  const total = files.data?.files.reduce((a, f) => a + f.size_bytes, 0) ?? 0;
  return (
    <DossierSection
      id="artifacts"
      lede={
        files.data
          ? `${files.data.files.length} files, ${fileSize(total)}, in the run directory ${run.id}.`
          : `The run directory ${run.id}.`
      }
    >
      <Panel flush>
        {files.data ? (
          <SortableTable
            caption="What the pipeline wrote for this run; each file downloads as the API serves it"
            columns={COLUMNS}
            rows={files.data.files}
            rowKey={(f) => f.name}
          />
        ) : files.isError ? (
          <ErrorState error={files.error} onRetry={() => void files.refetch()} />
        ) : (
          <Skeleton height="16rem" />
        )}
        <About
          label="About these files"
          handbook={{ page: 'backtesting', section: 'Run directory' }}
          className={styles.inset}
        >
          manifest, summary, hurdles and config are what the report card and this dossier read; the
          Parquet panels are the backtest itself (NAV, returns, weights, trades, costs, IC) and the
          extra artifacts the pipeline records at save time. Load them with qts.pipeline.load_run.
        </About>
      </Panel>
    </DossierSection>
  );
}
