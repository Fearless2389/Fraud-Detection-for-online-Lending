/**
 * Shared display primitives.
 *
 * Kept in one place so the console has a single visual vocabulary: one pill,
 * one force bar, one stat block, reused everywhere. Divergent one-off styling
 * is what makes dense interfaces feel noisy.
 */

import type { ReactNode } from 'react';
import type { Decision, ReasonCode } from '../lib/api';
import { DECISION_META, formatProbability } from '../lib/format';

/* -------------------------------------------------------------------------
   Panel
   ------------------------------------------------------------------------- */

export function Panel({
  label,
  action,
  children,
  className = '',
}: {
  label: string;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`panel flex min-h-0 flex-col ${className}`}>
      <header className="flex shrink-0 items-center justify-between border-b border-line px-4 py-2.5">
        <h2 className="eyebrow">{label}</h2>
        {action}
      </header>
      <div className="min-h-0 flex-1 overflow-auto">{children}</div>
    </section>
  );
}

/* -------------------------------------------------------------------------
   Decision pill

   Colour plus glyph plus text. The redundancy is intentional: roughly 1 in 12
   men has a colour vision deficiency, and a fraud console that encodes
   approve/block in hue alone is unusable for them.
   ------------------------------------------------------------------------- */

export function DecisionPill({
  decision,
  size = 'md',
}: {
  decision: Decision;
  size?: 'sm' | 'md';
}) {
  const meta = DECISION_META[decision];
  return (
    <span
      className={`pill ${size === 'sm' ? 'text-[10px] px-1.5 py-0' : ''}`}
      style={{ color: meta.color }}
      aria-label={`Decision: ${meta.label}`}
    >
      <span aria-hidden="true">{meta.glyph}</span>
      {meta.label}
    </span>
  );
}

/* -------------------------------------------------------------------------
   Threshold meter

   Shows where an application's probability falls relative to the two live
   thresholds. The point is to make the decision look like a *consequence of
   position on a scale* rather than an opaque verdict.

   The scale is square-root rather than linear: the review threshold sits near
   0.045 and the block threshold near 0.217, so on a linear 0-1 axis every
   marker would pile up against the left edge and the chart would show nothing.
   ------------------------------------------------------------------------- */

const scale = (value: number) => Math.sqrt(Math.min(Math.max(value, 0), 1)) * 100;

export function ThresholdMeter({
  probability,
  thresholds,
}: {
  probability: number;
  thresholds: { review: number; block: number };
}) {
  return (
    <div className="space-y-2">
      <div className="relative h-2 w-full bg-surface-3">
        {/* bands */}
        <div
          className="absolute inset-y-0 left-0"
          style={{ width: `${scale(thresholds.review)}%`, background: 'color-mix(in oklab, var(--color-approve) 30%, transparent)' }}
        />
        <div
          className="absolute inset-y-0"
          style={{
            left: `${scale(thresholds.review)}%`,
            width: `${scale(thresholds.block) - scale(thresholds.review)}%`,
            background: 'color-mix(in oklab, var(--color-review) 30%, transparent)',
          }}
        />
        <div
          className="absolute inset-y-0 right-0"
          style={{
            left: `${scale(thresholds.block)}%`,
            background: 'color-mix(in oklab, var(--color-block) 30%, transparent)',
          }}
        />
        {/* the application's position */}
        <div
          className="absolute -top-1 h-4 w-0.5 bg-ink"
          style={{ left: `${scale(probability)}%` }}
          title={`Fraud probability ${formatProbability(probability)}`}
        />
      </div>
      <div className="flex justify-between text-[10px] text-ink-3">
        <span className="num">0</span>
        <span className="num" style={{ color: 'var(--color-review)' }}>
          review {formatProbability(thresholds.review)}
        </span>
        <span className="num" style={{ color: 'var(--color-block)' }}>
          block {formatProbability(thresholds.block)}
        </span>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------
   Force bar

   The signature element. Each factor's SHAP contribution is drawn as a bar
   from a central zero line: right pushes the application toward fraud, left
   pulls it back. Because SHAP values are additive, the bars genuinely sum to
   the decision - this is the explanation, not an illustration of one.
   ------------------------------------------------------------------------- */

export function ForceBar({
  reasons,
  max,
}: {
  reasons: ReasonCode[];
  max: number;
}) {
  if (reasons.length === 0) {
    return <p className="px-4 py-6 text-xs text-ink-3">No material contributions.</p>;
  }

  return (
    <ul className="divide-y divide-line">
      {reasons.map((reason) => {
        const magnitude = Math.abs(reason.contribution);
        const width = max > 0 ? (magnitude / max) * 50 : 0;
        const raises = reason.direction === 'increases_risk';
        const color = raises ? 'var(--color-block)' : 'var(--color-approve)';

        return (
          <li key={reason.feature} className="px-4 py-2.5 hover:bg-surface-2">
            <div className="flex items-baseline justify-between gap-3">
              <span className="text-xs text-ink">{reason.label}</span>
              <span className="num shrink-0 text-[11px] text-ink-2">{reason.value}</span>
            </div>

            <div className="mt-1.5 flex items-center gap-2">
              <div className="relative h-1.5 flex-1 bg-surface-3">
                {/* zero line */}
                <div className="absolute inset-y-0 left-1/2 w-px bg-line-bright" />
                <div
                  className="absolute inset-y-0"
                  style={{
                    background: color,
                    width: `${width}%`,
                    left: raises ? '50%' : `${50 - width}%`,
                  }}
                />
              </div>
              <span
                className="num w-16 shrink-0 text-right text-[10px]"
                style={{ color }}
              >
                {raises ? '+' : '−'}
                {magnitude.toFixed(3)}
              </span>
            </div>
          </li>
        );
      })}
    </ul>
  );
}

/* -------------------------------------------------------------------------
   Stat
   ------------------------------------------------------------------------- */

export function Stat({
  label,
  value,
  hint,
  tone = 'neutral',
}: {
  label: string;
  value: string;
  hint?: string;
  tone?: 'neutral' | 'good' | 'bad' | 'accent';
}) {
  const color = {
    neutral: 'var(--color-ink)',
    good: 'var(--color-approve)',
    bad: 'var(--color-block)',
    accent: 'var(--color-series-1)',
  }[tone];

  return (
    <div className="px-4 py-3">
      <div className="eyebrow">{label}</div>
      <div className="num mt-1 text-xl leading-none" style={{ color }}>
        {value}
      </div>
      {hint && <div className="mt-1 text-[10px] leading-tight text-ink-3">{hint}</div>}
    </div>
  );
}

/* -------------------------------------------------------------------------
   Empty / error states
   ------------------------------------------------------------------------- */

export function Notice({
  title,
  detail,
  tone = 'neutral',
}: {
  title: string;
  detail?: string;
  tone?: 'neutral' | 'error';
}) {
  return (
    <div className="px-4 py-8 text-center">
      <p
        className="text-xs font-medium"
        style={{ color: tone === 'error' ? 'var(--color-block)' : 'var(--color-ink-2)' }}
      >
        {title}
      </p>
      {detail && <p className="mx-auto mt-2 max-w-sm text-[11px] leading-relaxed text-ink-3">{detail}</p>}
    </div>
  );
}
