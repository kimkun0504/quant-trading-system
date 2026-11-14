import { screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { describe, expect, it } from 'vitest';
import { renderWithProviders } from '@/test/render';
import { FeatureToggle } from './FeatureToggle';
import type { Field } from './lib/fields';

const FIELD: Field = {
  path: 'alpha.disabled_features',
  name: 'disabled_features',
  title: 'Disabled features',
  type: 'array',
  section: 'alpha',
  sectionTitle: 'Alpha model',
  nullable: false,
  default: '',
  choicesKey: 'alpha.feature_set',
  choices: {
    v1: [
      'value_bp',
      'value_ep',
      'quality_gpa',
      'mom_12_1',
      'mom_6_1',
      'industry_momentum',
      'strev_21',
    ],
    v2: ['value_bp', 'high_52w'],
  },
};

function Harness({ set = 'v1', initial = '' }: { set?: string; initial?: string }) {
  const [raw, setRaw] = useState(initial);
  return (
    <>
      <FeatureToggle field={FIELD} raw={raw} featureSet={set} label="Features" onChange={setRaw} />
      <output data-testid="raw">{raw}</output>
    </>
  );
}

describe('FeatureToggle', () => {
  it('turns features off by theme, renormalising a theme left empty, as a canonical list', async () => {
    renderWithProviders(<Harness />);
    const user = userEvent.setup();
    const box = await screen.findByTestId('feature-toggle');
    expect(box).toHaveTextContent(/7 of 7\s+features on/);
    const quality = box.querySelector('[data-theme="quality"]') as HTMLElement;
    await user.click(within(quality).getByRole('switch', { name: /quality_gpa/ }));
    expect(within(quality).getByText('dropped: the other themes renormalise')).toBeInTheDocument();
    const value = box.querySelector('[data-theme="value"]') as HTMLElement;
    await user.click(within(value).getByRole('switch', { name: /value_ep/ }));
    expect(
      within(value).getByText(/1 of 2 members: the theme averages what is left/),
    ).toBeInTheDocument();
    // sorted, comma separated: the --set syntax and the config's canonical form
    expect(screen.getByTestId('raw')).toHaveTextContent(/^quality_gpa,value_ep$/);
    expect(screen.getByRole('switch', { name: /value_ep/ })).toHaveAccessibleDescription(
      /earnings yield: .* \(theme feature, v1\)/,
    );
  });

  it('follows the feature set: names not in it are flagged and can be removed', async () => {
    renderWithProviders(<Harness set="v2" initial="mom_12_1,value_bp" />);
    const user = userEvent.setup();
    expect(await screen.findByRole('alert')).toHaveTextContent(/mom_12_1 is not in feature set v2/);
    await user.click(screen.getByRole('button', { name: 'Remove it' }));
    expect(screen.getByTestId('raw')).toHaveTextContent(/^value_bp$/);
  });
});
