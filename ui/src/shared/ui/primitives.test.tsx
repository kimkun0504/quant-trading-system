import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import { About } from './About';
import { Badge } from './Badge';
import { Button } from './Button';
import { IconButton } from './IconButton';
import { Kbd } from './Kbd';
import { Panel } from './Panel';
import { Table, Td, Th } from './Table';
import { Tooltip } from './Tooltip';

describe('primitives', () => {
  it('Button defaults to type=button and supports size and variant classes', () => {
    render(
      <Button variant="primary" size="sm">
        Go
      </Button>,
    );
    const b = screen.getByRole('button', { name: 'Go' });
    expect(b).toHaveAttribute('type', 'button');
    expect(b.className).toMatch(/primary/);
    expect(b.className).toMatch(/sm/);
  });

  it('IconButton names itself and shows the label as a tooltip on focus', async () => {
    render(
      <IconButton label="Close panel">
        <svg />
      </IconButton>,
    );
    const b = screen.getByRole('button', { name: 'Close panel' });
    const tip = screen.getByRole('tooltip');
    expect(tip).toHaveAttribute('data-open', 'false');
    await userEvent.setup().tab();
    expect(b).toHaveFocus();
    expect(tip).toHaveAttribute('data-open', 'true');
    expect(b).toHaveAccessibleDescription('Close panel');
  });

  it('Tooltip opens on hover and closes on Escape, keeping existing describedby', async () => {
    render(
      <Tooltip content="Net of costs" formula="mean / sd × √252">
        <button type="button" aria-describedby="other">
          Net Sharpe
        </button>
      </Tooltip>,
    );
    const user = userEvent.setup();
    const trigger = screen.getByRole('button');
    const tip = screen.getByRole('tooltip');
    expect(trigger.getAttribute('aria-describedby')).toContain('other');
    expect(trigger.getAttribute('aria-describedby')).toContain(tip.id);
    await user.hover(trigger);
    expect(tip).toHaveAttribute('data-open', 'true');
    expect(tip).toHaveTextContent('mean / sd × √252');
    trigger.focus();
    await user.keyboard('{Escape}');
    expect(tip).toHaveAttribute('data-open', 'false');
  });

  it('Badge carries its tone as a class and the word as text', () => {
    render(<Badge tone="fail">Fail</Badge>);
    const badge = screen.getByText('Fail');
    expect(badge.className).toMatch(/fail/);
  });

  it('Kbd renders a <kbd>', () => {
    render(<Kbd>c</Kbd>);
    expect(screen.getByText('c').tagName).toBe('KBD');
  });

  it('About is a closed disclosure with the handbook link', async () => {
    render(
      <About label="About this chart" handbook={{ page: 'validation', section: '9. DSR' }}>
        Explanation.
      </About>,
    );
    const details = screen.getByText('About this chart').closest('details');
    expect(details).not.toHaveAttribute('open');
    await userEvent.setup().click(screen.getByText('About this chart'));
    expect(details).toHaveAttribute('open');
    expect(screen.getByRole('link', { name: /Validation & statistics › 9. DSR/ })).toHaveAttribute(
      'target',
      '_blank',
    );
  });

  it('Panel renders a titled section and Table a labelled scroll region', () => {
    render(
      <Panel title="Gates" subtitle="per run" flush>
        <Table label="Gates by run">
          <thead>
            <tr>
              <Th scope="col">Gate</Th>
              <Th scope="col" numeric>
                Value
              </Th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <Th scope="row">DSR</Th>
              <Td numeric>0.67</Td>
            </tr>
          </tbody>
        </Table>
      </Panel>,
    );
    expect(screen.getByRole('heading', { name: /Gates.*per run/ })).toBeInTheDocument();
    expect(screen.getByRole('region', { name: 'Gates by run' })).toBeInTheDocument();
    expect(screen.getByRole('cell', { name: '0.67' }).className).toMatch(/end/);
  });
});
