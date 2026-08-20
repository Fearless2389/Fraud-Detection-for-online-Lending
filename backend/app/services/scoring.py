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
import pandas as pd  # noqa: TC002 - used at runtime for feature frames

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
    similarity_min_score: float = 0.60        # 'strong': 7.9x lift
    similarity_moderate_score: float = 0.55   # 'moderate': 5.8x lift
    similarity_seed_size: int = 400

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
            similarity_min_score=settings.similarity_min_score,
            similarity_moderate_score=settings.similarity_moderate_score,
            similarity_seed_size=settings.similarity_seed_size,
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

        model_decision = self.policy.decide(probability)
        similar = self._find_similar(features, application)
        decision, escalation_reason = self._apply_escalation(model_decision, similar)

        explanation = self.explainer.explain(features)
        narrative, narrative_source = self._narrate(explanation, decision, narrate)

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
            model_decision=model_decision.value,
            escalated=escalation_reason is not None,
            escalation_reason=escalation_reason,
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

        # Matches are qualified rather than filtered to a single cut-off.
        #
        # A hard threshold at the measured alert level (0.60) is correct for
        # *raising an alert* - it fires on 1.3% of genuine applications at 7.9x
        # lift - but it returns nothing at all on most cases, and an analyst
        # facing an empty panel cannot tell "no resemblance found" apart from
        # "this feature is broken".
        #
        # So the nearest cases are always returned with an honest strength
        # label. Only 'strong' warrants action; 'weak' explicitly tells the
        # analyst that nothing resembling this application is on file, which is
        # itself useful information.
        return [
            SimilarCaseOut(
                case_id=match.case_id,
                similarity=round(match.similarity, 4),
                confirmed_fraud=match.confirmed_fraud,
                strength=self._match_strength(match.similarity),
                matched_on=explain_match(payload, match.metadata),
            )
            for match in matches[:3]
        ]

    def _apply_escalation(
        self, model_decision: Decision, similar: list[SimilarCaseOut]
    ) -> tuple[Decision, str | None]:
        """Raise a decision when the application closely matches confirmed fraud.

        This is what makes the feedback loop adaptive rather than decorative.
        Without it, an analyst can confirm a fraud and the very next identical
        application still sails through on APPROVE - the match would be shown
        and ignored.

        The rule is deliberately narrow and one-directional:

        * it only ever raises a decision, never lowers one. A similarity match
          cannot clear an application the model considers risky.
        * it escalates APPROVE to REVIEW, never straight to BLOCK. A leaf-overlap
          match is evidence worth a human's attention, not grounds to decline a
          customer outright - and at 8.9% precision, auto-blocking on it would
          decline roughly eleven genuine applicants for every fraud stopped.
        * it fires only on a 'strong' match, whose threshold was measured rather
          than chosen (7.9x lift, firing on 1.3% of genuine applications).

        The model's own decision is preserved in the response, so an auditor can
        always see what was decided and what changed it.
        """
        strong = [match for match in similar if match.strength == "strong"]
        if not strong or model_decision is not Decision.APPROVE:
            return model_decision, None

        best = strong[0]
        return Decision.REVIEW, (
            f"Escalated from APPROVE: {best.similarity:.0%} match to confirmed "
            f"fraud case {best.case_id}"
            + (f" (shares {', '.join(best.matched_on)})" if best.matched_on else "")
        )

    def _match_strength(self, similarity: float) -> str:
        """Label a match against thresholds measured on held-out data."""
        if similarity >= self.similarity_min_score:
            return "strong"
        if similarity >= self.similarity_moderate_score:
            return "moderate"
        return "weak"

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

    def seed_similarity_index(self, limit: int | None = None) -> int:
        """Populate the index with fraud confirmed before the evaluation period.

        Without this the index starts empty and the lookalike-detection feature
        demonstrates nothing until an analyst happens to confirm a case by
        hand - which is not how a real deployment begins. A fraud team joining
        a new platform arrives with a history of confirmed cases, and that
        history is what makes similarity search useful on day one.

        Seeded strictly from the training and calibration months. Seeding from
        the evaluation months would leak the answers into the very population
        the reported metrics are measured on.

        Returns the number of cases indexed. Failure is logged and swallowed:
        an unavailable dataset should degrade the feature, not stop the API.
        """
        import os

        limit = self.similarity_seed_size if limit is None else limit
        if limit <= 0:
            return 0

        try:
            data_dir = Path(os.getenv("DATA_DIR", "C:/dev/data/baf"))
            parquet = data_dir / "base.parquet"
            if not parquet.exists():
                logger.warning(
                    "similarity index not seeded: %s not found. Run ml/eda.py "
                    "to build the parquet cache.", parquet
                )
                return 0

            from ml.features.pipeline import TARGET, TIME_COLUMN, prepare

            frame = pd.read_parquet(parquet)
            historical_fraud = frame[
                (frame[TARGET] == 1) & (frame[TIME_COLUMN] <= 5)
            ]
            if historical_fraud.empty:
                return 0

            sample = historical_fraud.sample(
                n=min(limit, len(historical_fraud)), random_state=20260820
            )
            features, _ = prepare(sample, self.feature_spec)

            # One batched forward pass for every case, rather than per-row.
            all_leaves = leaf_assignment(self.booster, features)

            for position, (_, row) in enumerate(sample.iterrows()):
                metadata = {
                    key: value for key, value in row.to_dict().items()
                    if key not in (TARGET, TIME_COLUMN)
                }
                self.similarity_index.add(
                    f"CONFIRMED-{position:04d}",
                    all_leaves[position],
                    confirmed_fraud=True,
                    metadata=metadata,
                )

            logger.info(
                "similarity index seeded with %d confirmed frauds from months 0-5",
                len(self.similarity_index),
            )
            return len(self.similarity_index)

        except Exception:   # noqa: BLE001 - seeding must never block startup
            logger.exception("similarity index seeding failed; continuing empty")
            return 0

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
