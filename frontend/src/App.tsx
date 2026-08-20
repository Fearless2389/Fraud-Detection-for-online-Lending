/**
 * Aegis - fraud operations console.
 *
 * A single dense screen rather than several routes. An analyst working a queue
 * needs the alert, the reason for it, and the policy governing it visible at
 * once; making any of those a separate page turns one glance into three
 * navigations.
 *
 * Layout, left to right, follows the order the work happens in:
 *
 *     what arrived  ->  why it was decided  ->  the policy that decided it
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { Activity, Pause, Play, ShieldAlert } from 'lucide-react';
import {
  api,
  ApiError,
  type DriftAnalysis,
  type ModelMetrics,
  type PolicyState,
} from './lib/api';
import { formatCount, formatPercent, formatProbability } from './lib/format';
import { CaseDetail } from './components/CaseDetail';
import { DriftPanel } from './components/DriftPanel';
import { LiveLedger, type LedgerRow } from './components/LiveLedger';
import { PolicyDial } from './components/PolicyDial';
import { Notice, Panel, Stat } from './components/primitives';

/** Roughly one application per second: fast enough to feel live, slow enough to read. */
const STREAM_INTERVAL_MS = 900;
/** Bounded so a long-running demo cannot grow the DOM without limit. */
const MAX_ROWS = 60;

export default function App() {
  const [rows, setRows] = useState<LedgerRow[]>([]);
  const [selected, setSelected] = useState<LedgerRow | null>(null);
  const [streaming, setStreaming] = useState(false);
  const [fatalError, setFatalError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const [policy, setPolicy] = useState<PolicyState | null>(null);
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [drift, setDrift] = useState<DriftAnalysis | null>(null);
  const [trueFraudRate, setTrueFraudRate] = useState<number | null>(null);

  const offset = useRef(0);

  /* ---------------- initial load ---------------- */

  useEffect(() => {
    (async () => {
      try {
        const health = await api.health();
        if (!health.model_loaded) {
          setFatalError(
            'The API is running but no model is loaded. Run `python ml/training/train.py` in backend/, then restart the API.',
          );
          return;
        }
        const [policyState, modelMetrics, driftAnalysis] = await Promise.all([
          api.policy(),
          api.modelMetrics(),
          api.drift(),
        ]);
        setPolicy(policyState);
        setMetrics(modelMetrics);
        setDrift(driftAnalysis);
      } catch (caught) {
        setFatalError(
          caught instanceof ApiError && caught.status === 401
            ? 'API key rejected. Check VITE_API_KEY in frontend/.env matches API_KEY in the backend .env.'
            : `Cannot reach the API. Is it running on port 8000? (${
                caught instanceof Error ? caught.message : 'unknown error'
              })`,
        );
      }
    })();
  }, []);

  /* ---------------- the live stream ---------------- */

  const scoreNext = useCallback(async () => {
    try {
      const batch = await api.stream(1, offset.current);
      offset.current += 1;
      setTrueFraudRate(batch.true_fraud_rate);

      const streamed = batch.applications[0];
      if (!streamed) return;

      const result = await api.score(streamed.application);
      const row: LedgerRow = {
        result,
        actualFraud: streamed.actual_fraud,
        application: streamed.application,
      };

      setRows((previous) => [row, ...previous].slice(0, MAX_ROWS));
      // Auto-select the first arrival so the case pane is never empty, but
      // never steal the pane from an analyst who has chosen something.
      setSelected((current) => current ?? row);
    } catch (caught) {
      setStreaming(false);
      setFatalError(caught instanceof Error ? caught.message : 'Scoring failed');
    }
  }, []);

  useEffect(() => {
    if (!streaming) return;
    void scoreNext();
    const handle = window.setInterval(() => void scoreNext(), STREAM_INTERVAL_MS);
    return () => window.clearInterval(handle);
  }, [streaming, scoreNext]);

  /* ---------------- lazy narration ----------------
     Selecting a case fetches the language-model briefing for it. The stream
     itself never waits on the provider: a decision that takes three seconds to
     arrive is not a real-time decision. Because scoring is deterministic the
     re-scored result carries the same probability and the same outcome - only
     the prose is added, so nothing the analyst already saw can change. */

  const selectCase = useCallback(async (row: LedgerRow) => {
    setSelected(row);
    if (row.result.narrative_source === 'gemini') return;

    try {
      const enriched = await api.score(row.application, true);
      setSelected((current) =>
        // Discard if the analyst has since moved on to another case.
        current?.result.application_id === row.result.application_id
          ? { ...row, result: { ...enriched, application_id: row.result.application_id } }
          : current,
      );
    } catch {
      /* Keep the deterministic template narrative already on screen. */
    }
  }, []);

  /* ---------------- analyst feedback ---------------- */

  const recordVerdict = async (outcome: 'confirmed_fraud' | 'cleared_genuine') => {
    if (!selected) return;
    setBusy(true);
    try {
      await api.recordDecision(
        selected.result.application_id,
        outcome,
        selected.application,
      );
      // Confirming fraud grows the similarity index; reflect that immediately
      // so the header count visibly moves when an analyst acts.
      setPolicy(await api.policy());
    } catch {
      /* The feedback loop is best-effort in the console; the API logs failures. */
    } finally {
      setBusy(false);
    }
  };

  /* ---------------- derived counters ---------------- */

  const blocked = rows.filter((r) => r.result.decision === 'BLOCK').length;
  const reviewed = rows.filter((r) => r.result.decision === 'REVIEW').length;
  const missed = rows.filter((r) => r.actualFraud && r.result.decision === 'APPROVE').length;
  const caught = rows.filter((r) => r.actualFraud && r.result.decision !== 'APPROVE').length;
  const meanLatency = rows.length
    ? rows.reduce((total, r) => total + r.result.latency_ms, 0) / rows.length
    : 0;

  if (fatalError) {
    return (
      <div className="grid min-h-screen place-items-center bg-surface-0 p-8">
        <div className="panel max-w-lg">
          <div className="flex items-center gap-2 border-b border-line px-4 py-3">
            <ShieldAlert size={14} style={{ color: 'var(--color-block)' }} aria-hidden="true" />
            <span className="eyebrow">Console unavailable</span>
          </div>
          <Notice title="The console cannot start" detail={fatalError} tone="error" />
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-surface-0">
      {/* ================= header ================= */}
      <header className="flex shrink-0 items-center justify-between border-b border-line px-4 py-2.5">
        <div className="flex items-baseline gap-3">
          <h1 className="text-sm font-bold tracking-[0.2em] text-ink">AEGIS</h1>
          <span className="text-[10px] tracking-wide text-ink-3">
            adaptive application-fraud decisioning
          </span>
        </div>

        <div className="flex items-center gap-4">
          {policy && (
            <span className="num text-[10px] text-ink-3">
              {policy.model_version} · index {policy.similarity_index_size}
            </span>
          )}
          <div className="flex items-center gap-1.5">
            <span
              className={`h-1.5 w-1.5 rounded-full ${streaming ? 'live-dot' : ''}`}
              style={{ background: streaming ? 'var(--color-approve)' : 'var(--color-ink-3)' }}
              aria-hidden="true"
            />
            <span className="num text-[10px] uppercase tracking-wider text-ink-2">
              {streaming ? 'live' : 'paused'}
            </span>
          </div>
          <button
            type="button"
            onClick={() => setStreaming((on) => !on)}
            className="pill py-1 transition-colors hover:bg-surface-2"
            style={{ color: streaming ? 'var(--color-review)' : 'var(--color-approve)' }}
          >
            {streaming ? <Pause size={11} aria-hidden="true" /> : <Play size={11} aria-hidden="true" />}
            {streaming ? 'Pause' : 'Start'}
          </button>
        </div>
      </header>

      {/* ================= metrics strip ================= */}
      <div className="grid shrink-0 grid-cols-6 divide-x divide-line border-b border-line bg-surface-1">
        <Stat label="Scored" value={formatCount(rows.length)} hint="this session" />
        <Stat label="Blocked" value={formatCount(blocked)} tone="bad" hint={`${formatCount(reviewed)} to review`} />
        <Stat
          label="Fraud caught"
          value={caught + missed > 0 ? formatPercent(caught / (caught + missed), 0) : '—'}
          tone="good"
          hint={`${formatCount(missed)} missed`}
        />
        <Stat
          label="Mean latency"
          value={meanLatency ? `${meanLatency.toFixed(0)}ms` : '—'}
          tone="accent"
          hint="score + explain, server-side"
        />
        <Stat
          label="Behavioural signal"
          value={metrics ? formatPercent(metrics.behavioural_gain_share, 0) : '—'}
          hint="share of model gain"
        />
        <Stat
          label="Calibration error"
          value={metrics ? metrics.test_calibrated.ece.toFixed(4) : '—'}
          hint={
            metrics ? `from ${metrics.test_raw.ece.toFixed(3)} uncalibrated` : 'held-out months'
          }
        />
      </div>

      {/* ================= three columns ================= */}
      <main className="grid min-h-0 flex-1 grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)_minmax(0,0.85fr)] gap-px bg-line">
        <Panel
          label="Live decision ledger"
          action={
            trueFraudRate !== null && (
              <span className="text-[9px] text-ink-3">
                replayed held-out data · true fraud rate {formatPercent(trueFraudRate, 2)}
              </span>
            )
          }
        >
          <LiveLedger
            rows={rows}
            selectedId={selected?.result.application_id ?? null}
            onSelect={selectCase}
            isStreaming={streaming}
          />
        </Panel>

        <Panel
          label="Case detail"
          action={
            selected && (
              <span className="num text-[10px] text-ink-3">
                {formatProbability(selected.result.fraud_probability)}
              </span>
            )
          }
        >
          <CaseDetail
            result={selected?.result ?? null}
            actualFraud={selected?.actualFraud ?? null}
            onVerdict={recordVerdict}
            busy={busy}
          />
        </Panel>

        <div className="grid min-h-0 grid-rows-2 gap-px bg-line">
          <Panel label="Risk appetite">
            <PolicyDial policy={policy} />
          </Panel>
          <Panel
            label="Model health"
            action={<Activity size={11} className="text-ink-3" aria-hidden="true" />}
          >
            <DriftPanel analysis={drift} />
          </Panel>
        </div>
      </main>
    </div>
  );
}
