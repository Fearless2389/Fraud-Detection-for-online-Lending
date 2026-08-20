"""The scoring service - where a submitted application becomes a decision.

Everything the platform does to one application happens here, in a fixed order:

    validate -> features -> rank -> calibrate -> decide -> explain -> narrate

Two design commitments are visible in that sequence and both are deliberate.

**The decision is made before anything explains it.** The booster, the
calibrator and the cost-derived policy produce an outcome; SHAP then explains
that outcome and the language model narrates the explanation. Nothing
downstream of the policy can change what was decided. This is what makes the
system reproducible: replaying the same application against the same model
version yields the same decision, byte for byte, regardless of whether the
language model was reachable.

**The model is loaded once, not per request.** A fraud decision has a latency
budget in the tens of milliseconds. Deserialising a booster inside the request
path would spend the entire budget on I/O.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.core.config import Settings
from app.schemas.application import (
    ApplicationIn,
    DecisionOut,
    ReasonCodeOut,
    SimilarCaseOut,
)
from app.services.explain import (
    Explanation,
    GeminiNarrator,
    ShapExplainer,
    template_narrative,
)
from app.services.policy import Decision, DecisionPolicy
from app.services.similarity import (
    InMemorySimilarityIndex,
    SimilarityIndex,
    explain_match,
    leaf_assignment,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ScoringService:
    """Holds every long-lived object needed to score an application."""

    booster: Any
    calibrator: Any
    feature_spec: Any
    policy: DecisionPolicy
    explainer: ShapExplainer
    similarity_index: SimilarityIndex
    narrator: GeminiNarrator | None
    model_version: str

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    @classmethod
    def from_artifacts(
        cls,
        settings: Settings,
        *,
        bundle_name: str = "model_bundle.joblib",
    ) -> "ScoringService":
        """Load the trained bundle and assemble the service.

        Raises if the artifact is missing rather than starting in a degraded
        state: an API that accepts scoring requests without a model would
        return errors per request instead of failing once, visibly, at boot.
        """
        bundle_path = Path(settings.artifact_dir) / bundle_name
        if not bundle_path.exists():
            raise FileNotFoundError(
                f"No model artifact at {bundle_path}. "
                "Run `python ml/training/train.py` first."
            )

        bundle = joblib.load(bundle_path)
        booster = bundle["booster"]

        narrator = None
        if settings.gemini_available:
            try:
                narrator = GeminiNarrator(settings.gemini_api_key, settings.gemini_model)
            except Exception as error:   # noqa: BLE001
                # Narration is a convenience. Failing to construct it must not
                # prevent the service from scoring.
                logger.warning(
                    "Gemini narrator unavailable (%s); using deterministic templates",
                    type(error).__name__,
                )

        return cls(
            booster=booster,
            calibrator=bundle["calibrator"],
            feature_spec=bundle["feature_spec"],
            policy=bundle["policy"],
            explainer=ShapExplainer(booster, bundle["feature_spec"].feature_names),
            similarity_index=InMemorySimilarityIndex(),
            narrator=narrator,
            model_version=f"lgbm-{booster.best_iteration}-iter",
        )

    # ------------------------------------------------------------------
    # scoring
    # ------------------------------------------------------------------

    def _to_features(self, application: ApplicationIn) -> pd.DataFrame:
        """Apply the exact transformations used at training time.

        Imported from the training pipeline rather than reimplemented. Two
        implementations of the same feature logic is how training/serving skew
        gets into production, and it is invisible until scores are already wrong.
        """
        from ml.features.pipeline import prepare

        raw = pd.DataFrame([application.model_dump()])
        features, _ = prepare(raw, self.feature_spec)
        return features

    def score(
        self,
        application: ApplicationIn,
        *,
        application_id: str | None = None,
        narrate: bool = False,
    ) -> DecisionOut:
        """Score one application.

        Args:
            narrate: request language-model prose for the explanation. Off by
                default, and deliberately so. A call to Gemini costs roughly
                three seconds, which is longer than the entire rest of the
                pipeline by two orders of magnitude - putting it in the scoring
                path would make a "real-time" decision arrive after the
                applicant has given up. The narrative is an analyst convenience
                needed only when someone opens a case, so it is fetched then.

                Because scoring is deterministic, re-scoring the same
                application with narration produces an identical decision; the
                prose is additive and can never alter the outcome.
        """
        started = time.perf_counter()
        application_id = application_id or f"APP-{uuid.uuid4().hex[:10].upper()}"

        features = self._to_features(application)

        raw_score = float(
            self.booster.predict(features, num_iteration=self.booster.best_iteration)[0]
        )
        probability = float(np.clip(self.calibrator.predict([raw_score])[0], 0.0, 1.0))
        decision = self.policy.decide(probability)

        explanation = self.explainer.explain(features)
        narrative, narrative_source = self._narrate(explanation, decision, narrate)
        similar = self._find_similar(features, application)

        latency_ms = (time.perf_counter() - started) * 1_000.0

        return DecisionOut(
            application_id=application_id,
            decision=decision.value,
            fraud_probability=probability,
            risk_band=self._risk_band(probability),
            thresholds={
                "review": self.policy.tau_review,
                "block": self.policy.tau_block,
            },
            top_risk_factors=[_to_reason_out(r) for r in explanation.top_risk_factors],
            top_protective_factors=[
                _to_reason_out(r) for r in explanation.top_protective_factors
            ],
            adverse_action_reasons=explanation.adverse_action_reasons(),
            narrative=narrative,
            narrative_source=narrative_source,
            similar_confirmed_cases=similar,
            model_version=self.model_version,
            latency_ms=round(latency_ms, 2),
        )

    def _narrate(
        self, explanation: Explanation, decision: Decision, use_language_model: bool
    ) -> tuple[str, str]:
        """Deterministic prose unless language-model narration was requested.

        The template path is always available and always fast, so an explanation
        exists for every decision regardless of network conditions or provider
        availability. That matters beyond convenience: a decision that can only
        be explained when a third-party API is reachable is not auditable.
        """
        if not use_language_model or self.narrator is None:
            return template_narrative(explanation, decision.value), "template"
        return self.narrator.narrate(explanation, decision.value)

    def _find_similar(
        self, features: pd.DataFrame, application: ApplicationIn, k: int = 5
    ) -> list[SimilarCaseOut]:
        if len(self.similarity_index) == 0:
            return []

        leaves = leaf_assignment(self.booster, features)[0]
        matches = self.similarity_index.search(leaves, k=k)
        payload = application.model_dump()

        return [
            SimilarCaseOut(
                case_id=match.case_id,
                similarity=round(match.similarity, 4),
                confirmed_fraud=match.confirmed_fraud,
                matched_on=explain_match(payload, match.metadata),
            )
            for match in matches
            # Below roughly half the trees agreeing, "similar" overstates it.
            if match.similarity >= 0.45
        ]

    def _risk_band(self, probability: float) -> str:
        """A coarse label for the UI. The decision, not this, drives action."""
        if probability >= self.policy.tau_block:
            return "severe"
        if probability >= self.policy.tau_review:
            midpoint = (self.policy.tau_review + self.policy.tau_block) / 2
            return "high" if probability >= midpoint else "elevated"
        return "low"

    # ------------------------------------------------------------------
    # feedback loop
    # ------------------------------------------------------------------

    def record_confirmed_fraud(
        self, application: ApplicationIn, case_id: str
    ) -> bool:
        """Index a confirmed fraud so lookalikes are caught immediately.

        This is the fast adaptation path. It takes effect on the very next
        request - no retraining, no redeploy, no model risk sign-off.
        """
        features = self._to_features(application)
        leaves = leaf_assignment(self.booster, features)[0]
        self.similarity_index.add(
            case_id,
            leaves,
            confirmed_fraud=True,
            metadata=application.model_dump(),
        )
        logger.info("indexed confirmed fraud %s (index size=%d)",
                    case_id, len(self.similarity_index))
        return True


def _to_reason_out(reason) -> ReasonCodeOut:
    return ReasonCodeOut(
        feature=reason.feature,
        label=reason.label,
        value=reason._format_value(),
        contribution=round(reason.contribution, 6),
        direction=reason.direction,
    )
