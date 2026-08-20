/**
 * The append-only audit trail.
 *
 * Every decision this system makes and every verdict an analyst records is
 * written to Postgres, and this is the window onto it. It exists because an
 * automated lending decision that cannot be reconstructed afterwards is not a
 * decision anyone can defend - to a customer, an auditor or a regulator.
 *
 * Two things are deliberately visible here rather than implied:
 *
 * - the log is **append-only at the database level**, not by convention. A
 *   trigger rejects UPDATE and DELETE, so nothing - including a bug in this
 *   application - can rewrite a recorded decision.
 * - the log survives a restart. The counters below come from Postgres, so a
 *   number that persists across a restart is itself the evidence.
 *
 * When no database is configured the panel says so plainly rather than
 * rendering an empty list, because "nothing recorded yet" and "recording is
 * switched off" are very different states to be in.
 */

import { Database, DatabaseZap, ShieldCheck } from 'lucide-react';
import type { AuditEntry, StorageStatus } from '../lib/api';
import { formatCount } from '../lib/format';
import { Notice } from './primitives';

/** Colour and label per event type. Unknown events degrade to neutral ink. */
const EVENT_META: Record<string, { label: string; color: string }> = {
  'decision.recorded': { label: 'decision', color: 'var(--color-series-1)' },
  'analyst.verdict': { label: 'verdict', color: 'var(--color-review)' },
};

const DECISION_COLOR: Record<string, string> = {
  APPROVE: 'var(--color-approve)',
  REVIEW: 'var(--color-review)',
  BLOCK: 'var(--color-block)',
};

/** `14:22:07` — a wall-clock time is what an analyst correlates against. */
function clockTime(iso: string): string {
  const parsed = new Date(iso);
  if (Number.isNaN(parsed.getTime())) return '--:--:--';
  return parsed.toLocaleTimeString('en-GB', { hour12: false });
}

/**
 * One line of human-readable consequence per entry.
 *
 * The raw JSONB detail is the record of truth, but reading it in a panel is
 * unpleasant, so the fields that matter are pulled out by event type.
 */
function summarise(entry: AuditEntry): { text: string; color?: string } {
  const detail = entry.detail ?? {};

  if (entry.event === 'decision.recorded') {
    const decision = String(detail.decision ?? '');
    const escalated = detail.escalated === true;
    return {
      text: escalated ? `${decision} · escalated` : decision,
      color: DECISION_COLOR[decision],
    };
  }

  if (entry.event === 'analyst.verdict') {
    const outcome = String(detail.outcome ?? '');
    const indexed = detail.added_to_similarity_index === true;
    return {
      text:
        outcome === 'confirmed_fraud'
          ? `confirmed fraud${indexed ? ' · indexed' : ''}`
          : 'cleared genuine',
      color: outcome === 'confirmed_fraud' ? 'var(--color-block)' : 'var(--color-approve)',
    };
  }

  return { text: entry.event };
}

export function AuditTrail({
  entries,
  storage,
}: {
  entries: AuditEntry[];
  storage: StorageStatus | null;
}) {
  if (storage && !storage.persistence) {
    return (
      <Notice
        title="Running without persistence"
        detail="No DATABASE_URL is configured, so decisions are not being recorded and the similarity index will not survive a restart. Set DATABASE_URL in .env and restart the API."
      />
    );
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      {/* ---------- what is actually stored ---------- */}
      {storage && (
        <div className="shrink-0 border-b border-line">
          <div className="grid grid-cols-3 divide-x divide-line">
            <Counter label="Decisions" value={storage.tables.decisions ?? 0} />
            <Counter label="Audit rows" value={storage.tables.audit_log ?? 0} />
            <Counter
              label="Fraud vectors"
              value={storage.tables.fraud_vectors ?? 0}
              tone="var(--color-series-2)"
            />
          </div>

          <div className="flex items-center gap-1.5 border-t border-line px-4 py-2">
            <ShieldCheck
              size={11}
              style={{ color: 'var(--color-approve)' }}
              aria-hidden="true"
            />
            <span className="text-[10px] leading-tight text-ink-3">
              Append-only — Postgres rejects <span className="num text-ink-2">UPDATE</span>{' '}
              and <span className="num text-ink-2">DELETE</span> on this table
            </span>
          </div>
        </div>
      )}

      {/* ---------- the log itself ---------- */}
      <div className="min-h-0 flex-1 overflow-auto">
        {entries.length === 0 ? (
          <Notice
            title="No entries yet"
            detail="Start the stream, or record a verdict on a case, and entries appear here."
          />
        ) : (
          <ul className="divide-y divide-line">
            {entries.map((entry) => {
              const meta = EVENT_META[entry.event];
              const summary = summarise(entry);

              return (
                <li key={entry.id} className="px-4 py-2 hover:bg-surface-2">
                  <div className="flex items-baseline gap-2">
                    <span className="num shrink-0 text-[10px] text-ink-3">
                      {clockTime(entry.occurred_at)}
                    </span>
                    <span
                      className="shrink-0 text-[10px] tracking-wide"
                      style={{ color: meta?.color ?? 'var(--color-ink-2)' }}
                    >
                      {meta?.label ?? entry.event}
                    </span>
                    <span
                      className="num ml-auto shrink-0 text-[10px]"
                      style={{ color: summary.color ?? 'var(--color-ink-2)' }}
                    >
                      {summary.text}
                    </span>
                  </div>

                  <div className="mt-0.5 flex items-baseline gap-2">
                    <span className="num truncate text-[10px] text-ink-3">
                      {entry.application_id ?? '—'}
                    </span>
                    {entry.actor !== 'system' && (
                      <span className="ml-auto shrink-0 text-[10px] text-ink-3">
                        {entry.actor}
                      </span>
                    )}
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </div>
  );
}

function Counter({
  label,
  value,
  tone = 'var(--color-ink)',
}: {
  label: string;
  value: number;
  tone?: string;
}) {
  return (
    <div className="px-3 py-2.5">
      <div className="eyebrow">{label}</div>
      <div className="num mt-1 text-base leading-none" style={{ color: tone }}>
        {formatCount(value)}
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------
   Header indicator

   Small, but it answers a question the rest of the interface cannot: is what
   I am watching being written down anywhere? During a demonstration that is
   the difference between a claim and a demonstrated fact.
   ------------------------------------------------------------------------- */

export function PersistenceBadge({ storage }: { storage: StorageStatus | null }) {
  if (!storage) return null;

  const live = storage.persistence;
  const backlogged = storage.writer.queued > 100;
  const lossy = storage.writer.dropped > 0;

  // A saturated or lossy writer is reported rather than hidden: a green light
  // over a queue that is not draining would be worse than no light at all.
  const { color, text, Icon } = live
    ? lossy
      ? { color: 'var(--color-block)', text: `dropped ${storage.writer.dropped}`, Icon: DatabaseZap }
      : backlogged
        ? { color: 'var(--color-review)', text: `queue ${storage.writer.queued}`, Icon: DatabaseZap }
        : { color: 'var(--color-approve)', text: 'postgres', Icon: Database }
    : { color: 'var(--color-ink-3)', text: 'no database', Icon: Database };

  return (
    <span
      className="flex items-center gap-1.5"
      title={
        live
          ? `Recording to Postgres — ${formatCount(storage.tables.audit_log ?? 0)} audit rows, ${formatCount(storage.writer.written)} written this session`
          : 'No database configured; decisions are not being recorded'
      }
    >
      <Icon size={11} style={{ color }} aria-hidden="true" />
      <span className="num text-[10px] tracking-wide" style={{ color }}>
        {text}
      </span>
    </span>
  );
}
