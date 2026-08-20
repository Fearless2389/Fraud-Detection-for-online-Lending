/**
 * Display formatting.
 *
 * Centralised because inconsistent number formatting is the fastest way to
 * make a data-heavy interface feel unfinished, and because the rounding rules
 * here carry meaning: a fraud probability of 0.4% and one of 0.44% lead to the
 * same decision, but showing "0%" for either would be actively misleading.
 */

const INR = new Intl.NumberFormat('en-IN', {
  style: 'currency',
  currency: 'INR',
  maximumFractionDigits: 0,
});

const COUNT = new Intl.NumberFormat('en-IN');

export const formatRupees = (value: number): string => INR.format(value);

export const formatCount = (value: number): string => COUNT.format(Math.round(value));

/**
 * Fraud probabilities live overwhelmingly below 1%, so a fixed number of
 * decimal places either floors small values to "0%" or clutters large ones.
 * Precision scales with magnitude instead.
 */
export function formatProbability(value: number): string {
  const percent = value * 100;
  if (percent >= 10) return `${percent.toFixed(0)}%`;
  if (percent >= 1) return `${percent.toFixed(1)}%`;
  if (percent >= 0.01) return `${percent.toFixed(2)}%`;
  return '<0.01%';
}

export const formatPercent = (value: number, decimals = 1): string =>
  `${(value * 100).toFixed(decimals)}%`;

export const formatLatency = (ms: number): string =>
  ms >= 1000 ? `${(ms / 1000).toFixed(2)}s` : `${ms.toFixed(0)}ms`;

/** Turn a raw feature name into something a person can read. */
export const humanise = (feature: string): string =>
  feature.replace(/_/g, ' ').replace(/\b\w/, (c) => c.toUpperCase());

export const DECISION_META = {
  APPROVE: { label: 'APPROVE', glyph: '✓', color: 'var(--color-approve)' },
  REVIEW: { label: 'REVIEW', glyph: '◐', color: 'var(--color-review)' },
  BLOCK: { label: 'BLOCK', glyph: '✕', color: 'var(--color-block)' },
} as const;
