"""Version 1 of the decisioning API.

Every route that touches an application or a model requires an API key. The
metrics routes are authenticated too: alert rates and thresholds tell an
attacker exactly what gets through.

FastAPI derives the OpenAPI document from these signatures and the schemas in
``app.schemas``, so the published contract cannot drift from the code that
serves it.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, status

from app.core.config import Settings, get_settings
from app.core.security import require_api_key
from app.db.repository import DecisionRepository
from app.schemas.application import (
    AnalystDecisionIn,
    AnalystDecisionOut,
    ApplicationIn,
    DecisionOut,
)
from app.services import demo_stream
from app.services.policy import (
    CostModel,
    constrain_review_capacity,
    derive_policy,
)
from app.services.scoring import ScoringService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", dependencies=[Depends(require_api_key)])


def get_scoring_service(request: Request) -> ScoringService:
    """Fetch the process-wide scoring service, or fail clearly."""
    service = getattr(request.app.state, "model_bundle", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts are not loaded. Run ml/training/train.py "
                   "and restart the service.",
        )
    return service


def get_repository(request: Request) -> DecisionRepository:
    """Fetch the decision recorder.

    Never raises. A missing or disabled repository is a valid state - the
    service scores without a database - and every write on it is a no-op in
    that case, so routes do not branch on persistence being configured.
    """
    repository = getattr(request.app.state, "repository", None)
    if repository is None:
        from app.db.session import Database
        return DecisionRepository(Database(None))
    return repository


def _read_artifact(settings: Settings, name: str) -> dict[str, Any]:
    path = Path(settings.artifact_dir) / name
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{name} not found. Run the training and analysis scripts first.",
        )
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


@router.post(
    "/applications/score",
    response_model=DecisionOut,
    summary="Score one application in real time",
    tags=["decisioning"],
)
async def score_application(
    application: ApplicationIn,
    service: Annotated[ScoringService, Depends(get_scoring_service)],
    repository: Annotated[DecisionRepository, Depends(get_repository)],
    application_id: Annotated[str | None, Query(max_length=64)] = None,
    narrate: Annotated[bool, Query(
        description="Request language-model prose for the explanation. Adds "
                    "seconds of provider latency and is off by default; the "
                    "decision and its reason codes are identical either way."
    )] = False,
) -> DecisionOut:
    """Return an auditable decision with per-case explanations.

    The response carries the thresholds in force and the model version, so the
    decision can be reconstructed later from the response alone.

    Latency note: with `narrate=false` this path is the model, the calibrator,
    the policy and SHAP - tens of milliseconds. Setting `narrate=true` adds a
    synchronous call to an external language model, which is why the console
    requests it only when an analyst opens a case rather than for every
    application in the stream.

    The decision is recorded to Postgres asynchronously. The write is queued,
    not awaited: the database is a network round trip away and the reported
    `latency_ms` would otherwise stop describing the decision and start
    describing the network.
    """
    decision = service.score(
        application, application_id=application_id, narrate=narrate
    )
    repository.record_decision(
        decision.application_id, application.model_dump(), decision
    )
    return decision


@router.post(
    "/applications/{application_id}/decision",
    response_model=AnalystDecisionOut,
    summary="Record an analyst's verdict on a reviewed case",
    tags=["decisioning"],
)
async def record_analyst_decision(
    application_id: str,
    # Two Pydantic body parameters, so FastAPI embeds each under its own key:
    # {"verdict": {...}, "application": {...}}. The application is resent
    # because indexing a confirmed fraud needs its full feature vector, and
    # this prototype holds no application store to look it up from.
    verdict: AnalystDecisionIn,
    application: ApplicationIn,
    service: Annotated[ScoringService, Depends(get_scoring_service)],
    repository: Annotated[DecisionRepository, Depends(get_repository)],
) -> AnalystDecisionOut:
    """Close the feedback loop.

    A confirmed fraud is indexed immediately, so applications resembling it are
    flagged from the next request onward - without retraining. This is the
    system's fastest adaptation path to a fraud pattern it has never seen.

    Unlike scoring, the index write here is synchronous. This is a human action
    taken a handful of times an hour, not a request on the real-time path, and
    an analyst must be told whether their confirmation was actually stored -
    `added_to_similarity_index` in the response is the answer, not a hopeful
    assumption. When persistence is configured the case is written to Postgres
    before it enters the in-process index, so a restart cannot lose it.
    """
    indexed = False
    if verdict.outcome == "confirmed_fraud":
        indexed = service.record_confirmed_fraud(application, application_id)

    repository.record_verdict(
        application_id, verdict.analyst_id, verdict.outcome, indexed=indexed
    )

    logger.info(
        "analyst %s marked %s as %s", verdict.analyst_id, application_id, verdict.outcome
    )
    return AnalystDecisionOut(
        application_id=application_id,
        outcome=verdict.outcome,
        recorded_at=datetime.now(timezone.utc),
        added_to_similarity_index=indexed,
    )


# ---------------------------------------------------------------------------
# policy
# ---------------------------------------------------------------------------


@router.get("/policy", summary="Thresholds currently in force", tags=["policy"])
async def current_policy(
    service: Annotated[ScoringService, Depends(get_scoring_service)],
) -> dict[str, Any]:
    costs = service.policy.cost_model
    return {
        "thresholds": {
            "review": service.policy.tau_review,
            "block": service.policy.tau_block,
        },
        "cost_model": {
            "cost_fp_inr": costs.cost_fp,
            "cost_fn_inr": costs.cost_fn,
            "cost_review_inr": costs.cost_review,
            "analyst_catch_rate": costs.analyst_catch_rate,
        },
        "model_version": service.model_version,
        "similarity_index_size": len(service.similarity_index),
    }


@router.post(
    "/policy/simulate",
    summary="Re-derive thresholds for a different risk appetite",
    tags=["policy"],
)
async def simulate_policy(
    service: Annotated[ScoringService, Depends(get_scoring_service)],
    settings: Annotated[Settings, Depends(get_settings)],
    cost_fp_inr: Annotated[float, Body(gt=0)] = 1_500.0,
    cost_fn_inr: Annotated[float, Body(gt=0)] = 45_000.0,
    cost_review_inr: Annotated[float, Body(ge=0)] = 200.0,
    analyst_catch_rate: Annotated[float, Body(gt=0, le=1)] = 0.90,
) -> dict[str, Any]:
    """Show what changing the bank's risk appetite does to the operating point.

    This is the interactive form of the project's central argument: the
    approve/review/block boundaries are not tuned constants, they are arithmetic
    over business costs. Raise the cost of wrongly blocking a genuine customer
    and the system becomes measurably less willing to block - no retraining
    involved, because the model never changes.
    """
    try:
        costs = CostModel(
            cost_fp=cost_fp_inr,
            cost_fn=cost_fn_inr,
            cost_review=cost_review_inr,
            analyst_catch_rate=analyst_catch_rate,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)
        ) from error

    unconstrained = derive_policy(costs)
    return {
        "thresholds": {
            "review": unconstrained.tau_review,
            "block": unconstrained.tau_block,
        },
        "explanation": {
            "tau_review": "cost_review / (analyst_catch_rate * cost_fn)",
            "tau_block": "(cost_fp - cost_review) / (cost_fn * (1 - analyst_catch_rate) + cost_fp)",
        },
        "capacity_cap": settings.max_review_rate,
        "note": "Derived thresholds, before the analyst-capacity cap is applied. "
                "The capacity cap needs a live score distribution.",
    }


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


@router.get("/metrics/model", summary="Held-out model performance", tags=["metrics"])
async def model_metrics(settings: Annotated[Settings, Depends(get_settings)]) -> dict[str, Any]:
    """Performance on the test months.

    Accuracy is deliberately absent: at 1.4% fraud prevalence, a model that
    never flags anything scores 98.6%, so the number carries no information.
    """
    return _read_artifact(settings, "metrics.json")


@router.get("/metrics/drift", summary="Month-by-month drift", tags=["metrics"])
async def drift_metrics(settings: Annotated[Settings, Depends(get_settings)]) -> dict[str, Any]:
    return _read_artifact(settings, "analysis.json")


@router.get("/metrics/adaptation", summary="Adaptation experiment", tags=["metrics"])
async def adaptation_metrics(settings: Annotated[Settings, Depends(get_settings)]) -> dict[str, Any]:
    return _read_artifact(settings, "adaptation.json")


# ---------------------------------------------------------------------------
# audit trail
# ---------------------------------------------------------------------------


@router.get(
    "/audit/recent",
    summary="Tail of the append-only audit log",
    tags=["audit"],
)
async def recent_audit(
    repository: Annotated[DecisionRepository, Depends(get_repository)],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> dict[str, Any]:
    """Every decision and every analyst verdict, newest first.

    Append-only is enforced by Postgres, not by this code: a trigger on the
    table rejects UPDATE and DELETE outright, so no code path - including a bug
    in this service - can rewrite a recorded decision. That is the difference
    between an audit log and a table that happens to be written once.

    Returns an empty list rather than an error when no database is configured,
    with `persistence` reporting which of the two situations you are in.
    """
    return {
        "persistence": repository.available,
        "entries": repository.recent_audit(limit),
    }


@router.get(
    "/storage",
    summary="Persistence status and row counts",
    tags=["audit"],
)
async def storage_status(
    repository: Annotated[DecisionRepository, Depends(get_repository)],
    service: Annotated[ScoringService, Depends(get_scoring_service)],
) -> dict[str, Any]:
    """What is actually stored, and whether the recorder is keeping up.

    `writer` exposes the queue depth alongside the written and dropped counts.
    A queue that grows without the written count moving means the database is
    unreachable and decisions are being recorded nowhere - a state that would
    otherwise be visible only in the logs.
    """
    index = service.similarity_index
    return {
        "persistence": repository.available,
        "tables": repository.counts(),
        "writer": repository.stats,
        "similarity_index": {
            "size": len(index),
            "durable": hasattr(index, "search_vector"),
        },
    }


# ---------------------------------------------------------------------------
# demo stream
# ---------------------------------------------------------------------------


@router.get(
    "/stream/applications",
    summary="Replay held-out applications for the dashboard",
    tags=["demo"],
)
async def stream_applications(
    count: Annotated[int, Query(ge=1, le=50)] = 5,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> dict[str, Any]:
    """Genuine held-out applications, for the live operations view.

    Fraud is over-sampled so a short demo shows meaningful volume; the true
    prevalence is returned alongside so the stratification is always visible
    rather than implied. `actual_fraud` exists only so the interface can show
    whether a decision was ultimately correct - the scoring path never sees it.
    """
    batch = demo_stream.take(count, offset)
    if not batch:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Demo data unavailable. Run scripts/download_data.py and "
                   "ml/eda.py to build the parquet cache.",
        )
    true_rate = demo_stream.true_fraud_rate()
    return {
        "applications": batch,
        "offset": offset,
        "sampled_fraud_share": demo_stream.FRAUD_SHARE,
        "true_fraud_rate": true_rate,
        "disclosure": (
            f"Demonstration ordering: genuine held-out applications, with fraud "
            f"over-sampled to {demo_stream.FRAUD_SHARE:.0%} and spaced every "
            f"{demo_stream.FRAUD_EVERY} positions so a short demo reliably shows "
            f"both outcomes. The true prevalence in these months is "
            f"{true_rate:.2%} and is reported alongside."
        ),
    }
