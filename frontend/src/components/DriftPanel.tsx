/**
 * Model health under drift.
 *
 * Two measures matter here - how much fraud the platform catches, and what each
 * decision costs - and they are drawn as two separate charts sharing an x-axis.
 * Not as one chart with two y-scales: a dual axis lets whoever draws it choose
 * scales that imply any correlation they like, and it is the single most
 * common way a chart misleads.
 *
 * Only out-of-sample months are plotted. Months 0-3 were trained on, so their
 * metrics are inflated by fitting; including them would present overfitting as
 * if it were drift, and the decay would look far more dramatic than it is.
 */

import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type { DriftAnalysis } from '../lib/api';
import { formatPercent, formatRupees } from '../lib/format';
import { Notice } from './primitives';

const AXIS = { stroke: '#6d6a64', fontSize: 10, fontFamily: 'IBM Plex Mono, monospace' };

function ChartTooltip({
  active,
  payload,
  label,
  format,
  title,
}: {
  active?: boolean;
  payload?: { value: number }[];
  label?: number;
  format: (value: number) => string;
  title: string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="border border-line-bright bg-surface-2 px-2.5 py-1.5 text-[11px]">
      <div className="text-ink-3">Month {label}</div>
      <div className="num mt-0.5 text-ink">
        {title} {format(payload[0].value)}
      </div>
    </div>
  );
}

export function DriftPanel({ analysis }: { analysis: DriftAnalysis | null }) {
  if (!analysis) {
    return <Notice title="Drift analysis unavailable" detail="Run ml/training/analyse.py to generate it." />;
  }

  const months = analysis.drift_by_month.filter((month) => month.split !== 'train');
  if (months.length === 0) {
    return <Notice title="No out-of-sample months to plot" />;
  }

  const first = months[0];
  const last = months[months.length - 1];
  const recallChange = last.recall / first.recall - 1;
  const costChange = last.cost_per_app / first.cost_per_app - 1;

  return (
    <div className="divide-y divide-line">
      <div className="grid grid-cols-2 divide-x divide-line">
        <div className="px-4 py-3">
          <div className="eyebrow">Recall, months {first.month}–{last.month}</div>
          <div className="num mt-1 text-lg leading-none" style={{ color: 'var(--color-block)' }}>
            {formatPercent(recallChange, 1)}
          </div>
          <div className="mt-1 text-[10px] text-ink-3">
            {formatPercent(first.recall, 0)} → {formatPercent(last.recall, 0)} at fixed thresholds
          </div>
        </div>
        <div className="px-4 py-3">
          <div className="eyebrow">Cost per application</div>
          <div className="num mt-1 text-lg leading-none" style={{ color: 'var(--color-block)' }}>
            +{formatPercent(costChange, 0)}
          </div>
          <div className="mt-1 text-[10px] text-ink-3">
            {formatRupees(first.cost_per_app)} → {formatRupees(last.cost_per_app)}
          </div>
        </div>
      </div>

      <div className="px-2 pb-2 pt-3">
        <div className="px-2">
          <span className="eyebrow">Fraud caught</span>
        </div>
        <ResponsiveContainer width="100%" height={120}>
          <LineChart data={months} margin={{ top: 12, right: 16, bottom: 4, left: -12 }}>
            <CartesianGrid stroke="#232323" vertical={false} />
            <XAxis dataKey="month" tick={AXIS} axisLine={false} tickLine={false} />
            <YAxis
              tick={AXIS}
              axisLine={false}
              tickLine={false}
              domain={[0.35, 0.75]}
              tickFormatter={(value: number) => `${(value * 100).toFixed(0)}%`}
            />
            <Tooltip
              content={<ChartTooltip format={(v) => formatPercent(v, 1)} title="recall" />}
              cursor={{ stroke: '#333333' }}
            />
            <Line
              type="monotone"
              dataKey="recall"
              stroke="var(--color-series-1)"
              strokeWidth={2}
              dot={{ r: 3, fill: 'var(--color-series-1)', strokeWidth: 0 }}
              activeDot={{ r: 5 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="px-2 pb-3 pt-3">
        <div className="px-2">
          <span className="eyebrow">Cost per decision</span>
        </div>
        <ResponsiveContainer width="100%" height={120}>
          <LineChart data={months} margin={{ top: 12, right: 16, bottom: 4, left: -12 }}>
            <CartesianGrid stroke="#232323" vertical={false} />
            <XAxis dataKey="month" tick={AXIS} axisLine={false} tickLine={false} />
            <YAxis
              tick={AXIS}
              axisLine={false}
              tickLine={false}
              domain={[180, 400]}
              tickFormatter={(value: number) => `${value}`}
            />
            <Tooltip
              content={<ChartTooltip format={formatRupees} title="cost/app" />}
              cursor={{ stroke: '#333333' }}
            />
            <Line
              type="monotone"
              dataKey="cost_per_app"
              stroke="var(--color-series-2)"
              strokeWidth={2}
              dot={{ r: 3, fill: 'var(--color-series-2)', strokeWidth: 0 }}
              activeDot={{ r: 5 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="px-4 py-3">
        <p className="text-[10px] leading-relaxed text-ink-3">
          Discrimination barely moves across this window — ROC-AUC falls about 2%. What
          decays is the <span className="text-ink-2">operating point</span>: fraud
          prevalence rises while the thresholds stay where they were set. The fix is
          re-deriving thresholds, not retraining the model.
        </p>
      </div>
    </div>
  );
}
