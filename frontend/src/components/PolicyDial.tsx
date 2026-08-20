/**
 * The risk-appetite control.
 *
 * This panel is the project's central argument made tangible. The
 * approve/review/block thresholds are not tuned constants that someone
 * searched for - they are arithmetic over four business costs:
 *
 *     tau_review = C_review / (catch_rate * C_fn)
 *     tau_block  = (C_fp - C_review) / (C_fn * (1 - catch_rate) + C_fp)
 *
 * Drag the cost of wrongly blocking a genuine customer upward and the block
 * threshold rises in front of you: the platform becomes measurably less willing
 * to decline people. No retraining, no redeploy - the model never changes,
 * because the model was never where the risk appetite lived.
 *
 * Recomputation happens server-side against the same function the scoring path
 * uses, so what this panel shows is the real derivation rather than a
 * re-implementation that could drift from it.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { RotateCcw } from 'lucide-react';
import { api, type PolicyState } from '../lib/api';
import { formatProbability, formatRupees } from '../lib/format';

interface Costs {
  cost_fp_inr: number;
  cost_fn_inr: number;
  cost_review_inr: number;
  analyst_catch_rate: number;
}

function Slider({
  label,
  hint,
  value,
  min,
  max,
  step,
  format,
  onChange,
}: {
  label: string;
  hint: string;
  value: number;
  min: number;
  max: number;
  step: number;
  format: (value: number) => string;
  onChange: (value: number) => void;
}) {
  return (
    <div className="px-4 py-2.5">
      <div className="flex items-baseline justify-between gap-2">
        <label className="text-[11px] text-ink-2">{label}</label>
        <span className="num text-xs text-ink">{format(value)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        className="mt-2 h-1 w-full cursor-pointer appearance-none bg-surface-3 accent-[var(--color-series-1)]"
        aria-label={label}
      />
      <p className="mt-1 text-[10px] leading-tight text-ink-3">{hint}</p>
    </div>
  );
}

export function PolicyDial({ policy }: { policy: PolicyState | null }) {
  const defaults: Costs = {
    cost_fp_inr: policy?.cost_model.cost_fp_inr ?? 1500,
    cost_fn_inr: policy?.cost_model.cost_fn_inr ?? 45000,
    cost_review_inr: policy?.cost_model.cost_review_inr ?? 200,
    analyst_catch_rate: policy?.cost_model.analyst_catch_rate ?? 0.9,
  };

  const [costs, setCosts] = useState<Costs>(defaults);
  const [thresholds, setThresholds] = useState<{ review: number; block: number } | null>(
    policy?.thresholds ?? null,
  );
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<number | null>(null);

  // Reset to the deployed cost model once it arrives from the server.
  useEffect(() => {
    if (policy) {
      setCosts({
        cost_fp_inr: policy.cost_model.cost_fp_inr,
        cost_fn_inr: policy.cost_model.cost_fn_inr,
        cost_review_inr: policy.cost_model.cost_review_inr,
        analyst_catch_rate: policy.cost_model.analyst_catch_rate,
      });
      setThresholds(policy.thresholds);
    }
  }, [policy]);

  // Debounced: dragging a slider fires continuously, and one request per pixel
  // would hammer the endpoint for frames nobody sees.
  const recompute = useCallback((next: Costs) => {
    if (timer.current) window.clearTimeout(timer.current);
    timer.current = window.setTimeout(async () => {
      try {
        const simulated = await api.simulatePolicy(next);
        setThresholds(simulated.thresholds);
        setError(null);
      } catch (caught) {
        // The cost model rejects economically incoherent combinations - most
        // often a review costing more than a wrongful block, which would make
        // reviewing never worthwhile. Surface that rather than swallow it.
        setError(caught instanceof Error ? caught.message : 'Could not derive thresholds');
      }
    }, 120);
  }, []);

  const update = (patch: Partial<Costs>) => {
    const next = { ...costs, ...patch };
    setCosts(next);
    recompute(next);
  };

  const reset = () => {
    setCosts(defaults);
    recompute(defaults);
  };

  const ratio = costs.cost_fn_inr / costs.cost_fp_inr;

  return (
    <div className="divide-y divide-line">
      <div className="px-4 py-3">
        <p className="text-[11px] leading-relaxed text-ink-2">
          Thresholds are <span className="text-ink">derived</span> from these costs, not
          tuned. Change the bank's risk appetite and the operating point moves — the model
          is untouched.
        </p>
      </div>

      <Slider
        label="Cost of blocking a genuine customer"
        hint="Lost customer, servicing cost, reputational drag. This is the false positive."
        value={costs.cost_fp_inr}
        min={200}
        max={20000}
        step={100}
        format={formatRupees}
        onChange={(value) => update({ cost_fp_inr: value })}
      />
      <Slider
        label="Cost of approving a fraudulent application"
        hint="Expected write-off on the extended credit."
        value={costs.cost_fn_inr}
        min={5000}
        max={200000}
        step={1000}
        format={formatRupees}
        onChange={(value) => update({ cost_fn_inr: value })}
      />
      <Slider
        label="Cost of one manual review"
        hint="Analyst time. Must stay below the cost of a wrongful block, or reviewing is never worth doing."
        value={costs.cost_review_inr}
        min={0}
        max={1400}
        step={25}
        format={formatRupees}
        onChange={(value) => update({ cost_review_inr: value })}
      />
      <Slider
        label="Analyst catch rate"
        hint="Probability an analyst correctly identifies fraud once a case reaches them."
        value={costs.analyst_catch_rate}
        min={0.5}
        max={1}
        step={0.01}
        format={(value) => `${(value * 100).toFixed(0)}%`}
        onChange={(value) => update({ analyst_catch_rate: value })}
      />

      {/* ---------- derived output ---------- */}
      <div className="bg-surface-2 px-4 py-3">
        <div className="flex items-center justify-between">
          <span className="eyebrow">Derived operating point</span>
          <button
            type="button"
            onClick={reset}
            className="flex items-center gap-1 text-[10px] text-ink-3 transition-colors hover:text-ink"
          >
            <RotateCcw size={10} aria-hidden="true" />
            reset
          </button>
        </div>

        {error ? (
          <p className="mt-2 text-[11px]" style={{ color: 'var(--color-block)' }}>
            {error}
          </p>
        ) : (
          <div className="mt-2 grid grid-cols-2 gap-3">
            <div>
              <div className="text-[10px] text-ink-3">send to review at</div>
              <div className="num text-lg leading-tight" style={{ color: 'var(--color-review)' }}>
                {thresholds ? formatProbability(thresholds.review) : '—'}
              </div>
            </div>
            <div>
              <div className="text-[10px] text-ink-3">block at</div>
              <div className="num text-lg leading-tight" style={{ color: 'var(--color-block)' }}>
                {thresholds ? formatProbability(thresholds.block) : '—'}
              </div>
            </div>
          </div>
        )}

        <p className="mt-3 text-[10px] leading-relaxed text-ink-3">
          Fraud is currently priced at{' '}
          <span className="num text-ink-2">{ratio.toFixed(0)}×</span> a wrongful block.
          Live thresholds are additionally capped by analyst capacity, which needs a
          score distribution and so is applied at serving time.
        </p>
      </div>
    </div>
  );
}
