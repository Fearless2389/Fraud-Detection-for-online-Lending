/**
 * The live decision ledger.
 *
 * Applications stream in and are scored one at a time, newest first. This is
 * the first thing anyone sees, so it has to make the system's behaviour legible
 * in seconds: what arrived, what the model thought, what the platform did about
 * it, and how long that took.
 *
 * The `actual_fraud` column is a demonstration affordance, not a production
 * one. Replaying held-out applications means the ground truth is known, so the
 * ledger can mark where the platform was right and where it was not - including
 * the misses. Hiding the misses would make the demo a sales pitch.
 */

import { useEffect, useRef } from 'react';
import type { Decision, DecisionResult } from '../lib/api';
import { formatLatency, formatProbability } from '../lib/format';
import { DecisionPill, Notice } from './primitives';

export interface LedgerRow {
  result: DecisionResult;
  actualFraud: boolean;
  /** The payload that produced this decision.
   *
   * Carried on the row so the feedback loop can re-send it verbatim when an
   * analyst confirms fraud. Re-deriving it from the stream offset would break
   * as soon as rows are trimmed, and would index the wrong application. */
  application: Record<string, unknown>;
}

/** Was the platform's action correct, given what we later learned? */
function outcomeOf(decision: Decision, actualFraud: boolean) {
  if (actualFraud) {
    if (decision === 'BLOCK') return { label: 'caught', tone: 'var(--color-approve)' };
    if (decision === 'REVIEW') return { label: 'to analyst', tone: 'var(--color-review)' };
    return { label: 'MISSED', tone: 'var(--color-block)' };
  }
  if (decision === 'BLOCK') return { label: 'false positive', tone: 'var(--color-block)' };
  if (decision === 'REVIEW') return { label: 'needless review', tone: 'var(--color-review)' };
  return { label: 'clean', tone: 'var(--color-ink-3)' };
}

export function LiveLedger({
  rows,
  selectedId,
  onSelect,
  isStreaming,
}: {
  rows: LedgerRow[];
  selectedId: string | null;
  onSelect: (row: LedgerRow) => void;
  isStreaming: boolean;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);

  // Newest rows enter at the top; keep the viewport pinned there so a running
  // demo never silently scrolls the newest decision out of view.
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: 0 });
  }, [rows.length]);

  if (rows.length === 0) {
    return (
      <Notice
        title={isStreaming ? 'Waiting for applications…' : 'Stream paused'}
        detail="Press Start to replay held-out applications through the live scoring endpoint."
      />
    );
  }

  return (
    <div ref={scrollRef} className="h-full overflow-auto">
      <table className="w-full border-collapse text-left">
        <thead className="sticky top-0 z-10 bg-surface-1">
          <tr className="border-b border-line text-[10px] uppercase tracking-wider text-ink-3">
            <th className="px-4 py-2 font-semibold">Application</th>
            <th className="px-2 py-2 text-right font-semibold">P(fraud)</th>
            <th className="px-2 py-2 font-semibold">Decision</th>
            <th className="px-2 py-2 font-semibold">Outcome</th>
            <th className="px-4 py-2 text-right font-semibold">Latency</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => {
            const { result, actualFraud } = row;
            const outcome = outcomeOf(result.decision, actualFraud);
            const selected = result.application_id === selectedId;

            return (
              <tr
                key={result.application_id}
                onClick={() => onSelect(row)}
                className={`${index === 0 ? 'row-enter' : ''} cursor-pointer border-b border-line/60 transition-colors ${
                  selected ? 'bg-surface-3' : 'hover:bg-surface-2'
                }`}
              >
                <td className="px-4 py-2">
                  <div className="flex items-center gap-2">
                    {/* A left rule in the decision colour: lets the eye group
                        the column without adding another badge. */}
                    <span
                      aria-hidden="true"
                      className="h-4 w-0.5 shrink-0"
                      style={{
                        background:
                          result.decision === 'BLOCK'
                            ? 'var(--color-block)'
                            : result.decision === 'REVIEW'
                              ? 'var(--color-review)'
                              : 'var(--color-approve)',
                      }}
                    />
                    <span className="num text-[11px] text-ink-2">{result.application_id}</span>
                  </div>
                </td>
                <td className="num px-2 py-2 text-right text-[11px] text-ink">
                  {formatProbability(result.fraud_probability)}
                </td>
                <td className="px-2 py-2">
                  <DecisionPill decision={result.decision} size="sm" />
                </td>
                <td className="px-2 py-2">
                  <span
                    className="num text-[10px] uppercase tracking-wide"
                    style={{ color: outcome.tone }}
                  >
                    {outcome.label}
                  </span>
                </td>
                <td className="num px-4 py-2 text-right text-[11px] text-ink-3">
                  {formatLatency(result.latency_ms)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
