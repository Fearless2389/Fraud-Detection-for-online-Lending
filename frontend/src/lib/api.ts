/**
 * Typed client for the Aegis decisioning API.
 *
 * Configuration comes from Vite environment variables, never from literals in
 * the source. The dev server proxies `/api` to the backend (see vite.config.ts)
 * so the browser makes same-origin requests locally; setting VITE_API_BASE_URL
 * points the same build at a deployed backend without a code change.
 *
 * On the API key: in this prototype the browser holds it, which is acceptable
 * for a single-operator console but is NOT how a production deployment should
 * work - a shared secret shipped to a browser is readable by anyone with
 * devtools. The production path is a session-authenticated BFF that holds the
 * key server-side. This is stated in the README rather than hidden.
 */

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? '';
const API_KEY = import.meta.env.VITE_API_KEY ?? '';

export type Decision = 'APPROVE' | 'REVIEW' | 'BLOCK';
export type RiskBand = 'low' | 'elevated' | 'high' | 'severe';

export interface ReasonCode {
  feature: string;
  label: string;
  value: string;
  contribution: number;
  direction: 'increases_risk' | 'reduces_risk';
}

export interface SimilarCase {
  case_id: string;
  similarity: number;
  confirmed_fraud: boolean;
  /** Calibrated on held-out data: strong >= 0.60 (7.9x lift), moderate >= 0.55 (5.8x). */
  strength: 'strong' | 'moderate' | 'weak';
  matched_on: string[];
}

export interface DecisionResult {
  application_id: string;
  decision: Decision;
  fraud_probability: number;
  risk_band: RiskBand;
  thresholds: { review: number; block: number };
  top_risk_factors: ReasonCode[];
  top_protective_factors: ReasonCode[];
  adverse_action_reasons: string[];
  narrative: string;
  narrative_source: 'gemini' | 'template';
  similar_confirmed_cases: SimilarCase[];
  /** What the model and cost policy decided, before any escalation rule. */
  model_decision: Decision;
  escalated: boolean;
  escalation_reason: string | null;
  model_version: string;
  scored_at: string;
  latency_ms: number;
}

/** The dataset's true label. Present only in the replay feed, never at scoring. */
export interface StreamedApplication {
  application: Record<string, unknown>;
  actual_fraud: boolean;
}

export interface StreamResponse {
  applications: StreamedApplication[];
  offset: number;
  sampled_fraud_share: number;
  true_fraud_rate: number;
  disclosure: string;
}

export interface PolicyState {
  thresholds: { review: number; block: number };
  cost_model: {
    cost_fp_inr: number;
    cost_fn_inr: number;
    cost_review_inr: number;
    analyst_catch_rate: number;
  };
  model_version: string;
  similarity_index_size: number;
}

export interface SimulatedPolicy {
  thresholds: { review: number; block: number };
  explanation: Record<string, string>;
  capacity_cap: number;
  note: string;
}

export interface ModelMetrics {
  n_features: number;
  best_iteration: number;
  behavioural_gain_share: number;
  test_raw: Record<string, number>;
  test_calibrated: Record<string, number>;
  policy: Record<string, number>;
  test_outcome_derived: Record<string, number>;
  test_outcome_naive: Record<string, number>;
  top_features: { feature: string; gain: number; share: number }[];
}

export interface DriftMonth {
  month: number;
  split: string;
  applications: number;
  fraud_rate: number;
  pr_auc: number;
  roc_auc: number;
  recall: number;
  review_rate: number;
  cost_per_app: number;
}

export interface DriftAnalysis {
  equal_recall_comparison: {
    three_way: { fraud_caught: number; genuine_blocked: number; reviewed: number; blocked: number };
    binary_equivalent: { genuine_blocked: number; fraud_caught: number; block_rate: number };
    false_positives_avoided: number;
  };
  drift_by_month: DriftMonth[];
}

export class ApiError extends Error {
  // Declared explicitly rather than as a constructor parameter property:
  // this project builds with `erasableSyntaxOnly`, which rejects TypeScript
  // syntax that emits runtime code.
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': API_KEY,
      ...init?.headers,
    },
  });

  if (!response.ok) {
    // Surface the server's own message where it has one: the API returns
    // actionable detail (missing artifacts, invalid cost model) that a generic
    // "request failed" would throw away.
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body?.detail ?? detail;
      if (Array.isArray(detail)) detail = detail.map((d: any) => d.msg ?? d).join('; ');
    } catch {
      /* non-JSON error body: keep the status text */
    }
    throw new ApiError(response.status, String(detail));
  }

  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<{ status: string; model_loaded: boolean }>('/health'),

  /**
   * Score an application.
   *
   * `narrate` is off for the live stream and on only when an analyst opens a
   * case. The language model adds seconds of provider latency, and a fraud
   * decision cannot wait on it. Scoring is deterministic, so re-scoring with
   * narration returns the same decision with richer prose attached.
   */
  score: (application: Record<string, unknown>, narrate = false) =>
    request<DecisionResult>(
      `/api/v1/applications/score?narrate=${narrate}`,
      { method: 'POST', body: JSON.stringify(application) },
    ),

  stream: (count: number, offset: number) =>
    request<StreamResponse>(
      `/api/v1/stream/applications?count=${count}&offset=${offset}`,
    ),

  policy: () => request<PolicyState>('/api/v1/policy'),

  simulatePolicy: (costs: {
    cost_fp_inr: number;
    cost_fn_inr: number;
    cost_review_inr: number;
    analyst_catch_rate: number;
  }) =>
    request<SimulatedPolicy>('/api/v1/policy/simulate', {
      method: 'POST',
      body: JSON.stringify(costs),
    }),

  modelMetrics: () => request<ModelMetrics>('/api/v1/metrics/model'),

  drift: () => request<DriftAnalysis>('/api/v1/metrics/drift'),

  recordDecision: (
    applicationId: string,
    outcome: 'confirmed_fraud' | 'cleared_genuine',
    application: Record<string, unknown>,
    analystId = 'analyst-console',
  ) =>
    request<{ added_to_similarity_index: boolean }>(
      `/api/v1/applications/${applicationId}/decision`,
      {
        method: 'POST',
        body: JSON.stringify({
          verdict: { outcome, analyst_id: analystId, notes: '' },
          application,
        }),
      },
    ),
};
