# Aegis — Real-Time Fraud Detection & Prevention for Digital Lending

**Design specification**
Problem statement 1 — Synchrony Financial technology hackathon
Date: 2026-08-19 · Submission deadline: 2026-08-21, 12:00 IST

---

## 1. Problem restated

> Develop a real-time fraud detection platform that integrates machine learning and
> behavioral analytics to identify, prevent, and respond to fraudulent activities across
> digital lending channels, ensuring secure and seamless customer experiences.

The challenge names four requirements. This design maps a distinct, *measurable*
mechanism to each — no requirement is answered by architecture alone.

| Requirement (their words) | Mechanism | Evidence produced |
| --- | --- | --- |
| "real-time" | Synchronous scoring endpoint; no batch path exists | Measured p50/p95/p99 latency |
| "machine learning **and behavioral analytics**" | Velocity windows, device fingerprint, session behaviour features present natively in the dataset | Feature-importance share held by behavioural feature group |
| "reducing false positives" | Cost matrix (₹) → decision threshold derived from expected cost, not defaulted to 0.5 | Confusion matrix and total expected cost at derived vs naive threshold |
| "adapting to new fraud vectors" | Three-tier detection + analyst feedback writing back to a similarity index | Recall decay under temporal drift, and recovery after adaptation |

## 2. Key insight driving the design

Most fraud-detection prototypes are built on card-transaction datasets whose features
are anonymised principal components. Two consequences follow, and both are fatal for
this brief: the domain is payments rather than lending origination, and per-decision
explanation is impossible because no feature has a human meaning.

This design instead uses the **Bank Account Fraud (BAF) suite** (Feedzai, NeurIPS 2022) —
six ~1M-row datasets derived from a real, anonymised bank **account-opening** fraud
problem. Three properties make it the correct choice:

1. **Domain fidelity.** Account-opening fraud is structurally the same decision as
   digital lending origination fraud: a single application, assessed once, before any
   relationship exists. Card-transaction fraud is a different problem wearing similar clothes.
2. **Temporal dynamics.** The dataset is explicitly constructed to carry distribution
   shift over time. This converts "adapts to new fraud vectors" from an architectural
   claim into a *measurable experiment*.
3. **Semantic features and protected attributes.** Features have real names, so SHAP
   attributions are readable by an analyst; and demographic attributes are present, so
   the fairness analysis uses measurements rather than intentions.

Exact column names are confirmed against the downloaded data during EDA rather than
assumed here.

## 3. Architecture

```
┌──────────────────────────────┐        ┌──────────────────────────────────────┐
│  React + Vite + Tailwind     │  HTTPS │  FastAPI  (Python 3.13)              │
│                              │───────►│                                      │
│  1. Live Monitor             │        │  Decision Engine                     │
│  2. Case Detail              │        │   ├─ LightGBM        known fraud     │
│  3. Model Health             │        │   ├─ IsolationForest novel/unlabelled│
└──────────────────────────────┘        │   └─ pgvector k-NN   similar cases   │
                                        │                                      │
                                        │  Policy Layer                        │
                                        │   cost-derived thresholds →          │
                                        │   APPROVE / REVIEW / BLOCK           │
                                        │                                      │
                                        │  Explainability                      │
                                        │   SHAP → reason codes → Gemini prose │
                                        │                                      │
                                        │  Drift Monitor  PSI + rolling recall │
                                        └───────────────┬──────────────────────┘
                                                        │
                                        ┌───────────────▼──────────────────────┐
                                        │  Postgres + pgvector (Supabase)      │
                                        │  applications · decisions            │
                                        │  fraud_vectors · audit_log           │
                                        └──────────────────────────────────────┘
```

### 3.1 Why three detection tiers

A single supervised model can only find fraud that resembles its training labels — which
is precisely the weakness the challenge statement calls out. Each tier covers a failure
mode the others cannot:

- **LightGBM (supervised).** High precision on established fraud patterns. Chosen over
  XGBoost for native categorical handling and materially faster training on ~1M rows,
  which matters given the schedule.
- **Isolation Forest (unsupervised).** Scores how anomalous an application is without
  reference to any label. Catches patterns absent from training data. Crucially, "we have
  no labels for novel fraud" is a genuine property of the fraud domain, not an excuse —
  unsupervised scoring is standard practice in production fraud systems.
- **Vector similarity (pgvector k-NN).** Each application's feature signature is embedded
  and compared against an index of *confirmed* fraud cases. This is the fastest adaptation
  path in the system: confirming one fraud inserts one row, and every similar application
  is flagged from that moment on — **with no retraining**. It is also the only honest
  justification for a vector store in this stack.

### 3.2 Policy layer

The three tier scores are combined by an explicit, auditable rule set — not an averaged
blend — producing `APPROVE`, `REVIEW`, or `BLOCK`. Thresholds are derived by minimising
expected cost over the validation set given `COST_FP` and `COST_FN`, which live in
configuration because they are business inputs, not model hyperparameters.

This is the direct answer to "reducing false positives": a false positive is a blocked
genuine customer and has a rupee cost, so the operating point is *chosen*, not inherited
from a library default of 0.5.

### 3.3 The LLM boundary

Gemini receives the SHAP attributions and similar-case context for a decision **that has
already been made** and writes the analyst-facing narrative. It has no input to the
approve/block outcome. In a regulated lending context this separation is the difference
between an explainable system and an unaccountable one. Guardrails: temperature 0,
structured output schema, feature values supplied as facts, no PII in the prompt, and a
deterministic template fallback when `GEMINI_ENABLED=false`.

## 4. Interface (API-first)

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/v1/applications/score` | Score one application, return decision + explanation |
| `GET` | `/api/v1/applications/{id}` | Full case detail for the analyst view |
| `GET` | `/api/v1/applications/{id}/similar` | k-NN neighbours among confirmed fraud |
| `POST` | `/api/v1/applications/{id}/decision` | Analyst outcome; writes to feedback loop |
| `GET` | `/api/v1/metrics/model` | PR curve, threshold, confusion matrix |
| `GET` | `/api/v1/metrics/drift` | PSI per feature, rolling recall |
| `GET` | `/health` | Liveness |

OpenAPI documentation is generated by FastAPI at `/docs`.

## 5. Security

API-key authentication on every scoring route; Pydantic schema validation on all input;
rate limiting via slowapi; all secrets read from environment with a committed
`.env.example` and a gitignored `.env`; parameterised queries throughout; and an
append-only `audit_log` capturing every decision with model version, score, threshold,
and actor — a regulatory requirement for automated adverse decisions, not a nicety.

Dataset CSVs are never committed: BAF is CC BY-NC-ND licensed, and the repository ships
a download script instead.

## 6. Testing

Roughly 15–20 meaningful tests rather than a large vanity suite:

- feature pipeline determinism (same input → same vector)
- threshold policy correctness at known cost inputs
- decision banding boundaries (approve/review/block edges)
- API contract and auth rejection paths
- end-to-end scoring smoke test against a saved model artifact

## 7. Responsible AI

Fairness measured across the protected attributes present in BAF, comparing selection
rate and recall between groups at the chosen operating point. Any disparity found is
**reported rather than hidden** — an honest measurement is a stronger result than a
silent one. Explanations are surfaced for every decision, not only adverse ones. The
LLM boundary in §3.3 is itself a responsible-AI control.

## 8. Scope discipline

Declared in advance so that pressure does not drive the decision.

**Cut in this order if behind schedule:** secondary-dataset validation (IEEE-CIS) →
cloud deployment (local demo plus an honest architecture slide) → Gemini narration
(deterministic templates, interface unchanged) → multi-variant fairness comparison.

**Never cut:** the drift experiment, cost-derived thresholds, SHAP explanations, the
test suite, the README.

## 9. Known limitations

Stated openly in the deck rather than discovered by a reviewer.

- BAF is a privacy-preserving *synthetic* rendering (CTGAN + differential privacy) of a
  real dataset. It preserves realistic structure but is not live production data.
- Account-opening fraud is a close analogue of lending-origination fraud, not an
  identical problem; feature availability would differ in a real Synchrony deployment.
- The prototype runs locally. Cloud architecture is designed and documented but only
  partially exercised.
- The drift experiment uses the dataset's own temporal ordering; real concept drift
  includes adversarial adaptation that no static dataset can fully represent.

## 10. Delivery schedule

| When | Work | Definition of done |
| --- | --- | --- |
| 19 Aug, evening | Scaffold, data download, EDA, baseline LightGBM | Model artifact on disk with a real PR-AUC |
| 20 Aug, morning | Cost thresholds, Isolation Forest, SHAP, vector index, API | A JSON application returns a scored explanation |
| 20 Aug, afternoon | React — three screens | Clickable end-to-end demo |
| 20 Aug, evening | Drift experiment, feedback loop, Gemini narration, tests, README | Every slide claim is reproducible |
| **21 Aug, 00:00** | **Build stops** | — |
| 21 Aug, 00:00–06:00 | Deck, demo recording, rehearsal | — |
| 21 Aug, by 10:00 IST | Zip, verify roll-number naming, send | Two hours of buffer before the 12:00 deadline |
