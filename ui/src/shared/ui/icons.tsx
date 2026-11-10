/* eslint-disable react-refresh/only-export-components -- an icon set, not a page component */
import type { SVGProps } from 'react';

type IconProps = Omit<SVGProps<SVGSVGElement>, 'children'>;

function Icon({ children, ...rest }: SVGProps<SVGSVGElement>) {
  return (
    <svg
      viewBox="0 0 16 16"
      width="16"
      height="16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      {...rest}
    >
      {children}
    </svg>
  );
}

/* A small, consistent 16-px line icon set (1.5 px stroke). */
export const Icons = {
  Search: (p: IconProps) => (
    <Icon {...p}>
      <circle cx="7" cy="7" r="4.5" />
      <path d="M10.5 10.5 14 14" />
    </Icon>
  ),
  Sun: (p: IconProps) => (
    <Icon {...p}>
      <circle cx="8" cy="8" r="3" />
      <path d="M8 1.5v2M8 12.5v2M1.5 8h2M12.5 8h2M3.4 3.4l1.4 1.4M11.2 11.2l1.4 1.4M3.4 12.6l1.4-1.4M11.2 4.8l1.4-1.4" />
    </Icon>
  ),
  Moon: (p: IconProps) => (
    <Icon {...p}>
      <path d="M13.5 9.5A6 6 0 0 1 6.5 2.5a6 6 0 1 0 7 7Z" />
    </Icon>
  ),
  Close: (p: IconProps) => (
    <Icon {...p}>
      <path d="M4 4l8 8M12 4l-8 8" />
    </Icon>
  ),
  Chevron: (p: IconProps) => (
    <Icon {...p}>
      <path d="M6 3.5 10.5 8 6 12.5" />
    </Icon>
  ),
  Flask: (p: IconProps) => (
    <Icon {...p}>
      <path d="M6 1.5h4M6.5 1.5v4.5L2.5 13a1 1 0 0 0 .9 1.5h9.2a1 1 0 0 0 .9-1.5L9.5 6V1.5" />
      <path d="M4.5 10h7" />
    </Icon>
  ),
  Signal: (p: IconProps) => (
    <Icon {...p}>
      <path d="M1.5 11.5 5 7l3 2.5L11 4l3.5 5" />
      <path d="M1.5 14.5h13" />
    </Icon>
  ),
  Portfolio: (p: IconProps) => (
    <Icon {...p}>
      <circle cx="8" cy="8" r="6" />
      <path d="M8 2v6l4.5 3" />
    </Icon>
  ),
  Costs: (p: IconProps) => (
    <Icon {...p}>
      <path d="M8 2v12M4.5 5.5a3.5 2 0 0 1 7 0c0 1.5-1.5 2-3.5 2.5S4.5 9 4.5 10.5a3.5 2 0 0 0 7 0" />
    </Icon>
  ),
  Builder: (p: IconProps) => (
    <Icon {...p}>
      <path d="M9.5 2.5a3.5 3.5 0 0 0-4.3 4.3L1.5 10.5l2 2 3.7-3.7a3.5 3.5 0 0 0 4.3-4.3L9.5 6.5 7.5 4.5Z" />
      <path d="M10 11.5 13.5 15" />
    </Icon>
  ),
  Ledger: (p: IconProps) => (
    <Icon {...p}>
      <rect x="2.5" y="1.5" width="11" height="13" rx="1" />
      <path d="M5.5 5h5M5.5 8h5M5.5 11h3" />
    </Icon>
  ),
  Models: (p: IconProps) => (
    <Icon {...p}>
      <rect x="1.5" y="1.5" width="5" height="5" rx="1" />
      <rect x="9.5" y="9.5" width="5" height="5" rx="1" />
      <path d="M6.5 4h5a1 1 0 0 1 1 1v4.5M9.5 12h-5a1 1 0 0 1-1-1V6.5" />
    </Icon>
  ),
  Book: (p: IconProps) => (
    <Icon {...p}>
      <path d="M2 2.5h4.5a1.5 1.5 0 0 1 1.5 1.5v10a1 1 0 0 0-1-1H2ZM14 2.5H9.5A1.5 1.5 0 0 0 8 4v10a1 1 0 0 1 1-1h5Z" />
    </Icon>
  ),
  Compare: (p: IconProps) => (
    <Icon {...p}>
      <rect x="1.5" y="3.5" width="5.5" height="9" rx="1" />
      <rect x="9" y="3.5" width="5.5" height="9" rx="1" />
    </Icon>
  ),
  Check: (p: IconProps) => (
    <Icon {...p}>
      <path d="M3 8.5 6.5 12 13 4.5" />
    </Icon>
  ),
};
