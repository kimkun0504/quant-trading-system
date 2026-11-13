import { useEffect, useState } from 'react';
import { IconButton } from './IconButton';
import { Icons } from './icons';
import { VisuallyHidden } from './VisuallyHidden';

/** Copies `text` to the clipboard; the label flips to "Copied" for a moment. */
export function CopyButton({ text, label }: { text: string; label: string }) {
  const [copied, setCopied] = useState(false);
  useEffect(() => {
    if (!copied) return;
    const t = window.setTimeout(() => setCopied(false), 1500);
    return () => window.clearTimeout(t);
  }, [copied]);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
    } catch {
      /* clipboard unavailable (insecure context): the text is still selectable */
    }
  };
  return (
    <>
      <IconButton size="sm" label={copied ? 'Copied' : label} onClick={() => void copy()}>
        {copied ? <Icons.Check /> : <Icons.Copy />}
      </IconButton>
      <VisuallyHidden>
        <span role="status">{copied ? 'Copied to the clipboard' : ''}</span>
      </VisuallyHidden>
    </>
  );
}
