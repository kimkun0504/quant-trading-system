import { useEffect, useState } from 'react';
import { Button, VisuallyHidden } from '@/shared/ui';
import { Icons } from '@/shared/ui/icons';
import styles from './research.module.css';

/**
 * The exact `qts` command equivalent to the form, for reproducibility: shown as literal
 * text (the one monospace place, DESIGN.md) with a "Copy as CLI command" button.
 */
export function CommandLine({ command, note }: { command: string; note?: string }) {
  const [copied, setCopied] = useState(false);
  useEffect(() => {
    if (!copied) return;
    const t = window.setTimeout(() => setCopied(false), 1500);
    return () => window.clearTimeout(t);
  }, [copied]);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(command);
      setCopied(true);
    } catch {
      /* clipboard unavailable: the text stays selectable */
    }
  };
  return (
    <div className={styles.command}>
      <div className={styles.commandHead}>
        <span className={styles.commandTitle}>Equivalent command</span>
        <Button size="sm" variant="secondary" onClick={() => void copy()}>
          {copied ? <Icons.Check /> : <Icons.Copy />}
          {copied ? 'Copied' : 'Copy as CLI command'}
        </Button>
      </div>
      <pre className={styles.commandText} data-testid="cli-command">
        {command}
      </pre>
      {note ? <p className={styles.help}>{note}</p> : null}
      <VisuallyHidden>
        <span role="status">{copied ? 'Command copied to the clipboard' : ''}</span>
      </VisuallyHidden>
    </div>
  );
}
