import { ApiError } from '@/shared/api';
import { Button } from './Button';
import styles from './States.module.css';

function describe(error: unknown): { title: string; body: string } {
  if (error instanceof ApiError) {
    if (error.status === 404) return { title: 'Not found', body: error.detail };
    if (error.status === 409) return { title: 'No results yet', body: error.detail };
    return { title: `The API answered ${error.status}`, body: error.detail };
  }
  if (error instanceof TypeError)
    return {
      title: 'The API is unreachable',
      body: 'Check that the qts API is running and that VITE_API_URL points at it.',
    };
  return {
    title: 'Something failed',
    body: error instanceof Error ? error.message : String(error),
  };
}

export function ErrorState({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  const { title, body } = describe(error);
  return (
    <div className={styles.state} role="alert">
      <h2 className={styles.title}>{title}</h2>
      <p className={styles.body}>{body}</p>
      {onRetry ? (
        <div className={styles.action}>
          <Button onClick={onRetry}>Try again</Button>
        </div>
      ) : null}
    </div>
  );
}
