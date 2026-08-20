"""Per-decision explanations.

Every decision this system makes must be answerable to three different
audiences, and they need different things:

* **the analyst** working the case, who needs to know what to check first
* **the customer**, who is entitled to know why they were declined
* **the auditor**, who needs the same explanation to be reproducible months later

The mechanism is SHAP over the trained booster: an exact, additive attribution
of a single prediction to individual feature values. Not global feature
importance, which describes the model in general and says nothing about *this*
applicant.

A note on what is being explained
---------------------------------
Attributions are computed against the booster's raw output, not the calibrated
probability. Isotonic calibration is monotonic, so it cannot reorder anything:
the features pushing an application up the risk ranking are identical before
and after calibration. Explaining the raw score keeps the attributions exactly
additive, which is what makes them auditable.

On the language model
---------------------
Gemini writes prose *about attributions that already exist*. It receives the
computed SHAP values and turns them into a paragraph an analyst can read in
five seconds. It cannot alter a score, a threshold, or a decision, and if it
is unavailable the system falls back to deterministic templates and keeps
working. That boundary is a deliberate control: a lending decision that a
language model could influence would be neither reproducible nor defensible.
"""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    import lightgbm as lgb

logger = logging.getLogger(__name__)

# Plain-English labels for the BAF schema. Raw column names are fine for an
# engineer and useless to everyone else; an adverse-action reason reading
# "date_of_birth_distinct_emails_4w" helps no customer.
FEATURE_LABELS: dict[str, str] = {
    "income": "Declared income band",
    "name_email_similarity": "Match between applicant name and email address",
    "prev_address_months_count": "Time at previous address",
    "prev_address_months_count_missing": "Previous address not provided",
    "current_address_months_count": "Time at current address",
    "current_address_months_count_missing": "Current address history unavailable",
    "customer_age": "Applicant age band",
    "days_since_request": "Time since the application was started",
    "intended_balcon_amount": "Requested balance transfer amount",
    "has_balcon_intent": "Balance transfer requested",
    "payment_type": "Payment method selected",
    "zip_count_4w": "Applications from this postcode in the last 4 weeks",
    "velocity_6h": "Application rate in the last 6 hours",
    "velocity_24h": "Application rate in the last 24 hours",
    "velocity_4w": "Application rate over the last 4 weeks",
    "bank_branch_count_8w": "Applications via this branch in the last 8 weeks",
    "date_of_birth_distinct_emails_4w": "Distinct emails sharing this date of birth",
    "employment_status": "Employment status",
    "credit_risk_score": "Internal credit risk score",
    "email_is_free": "Free email provider used",
    "housing_status": "Housing status",
    "phone_home_valid": "Home phone verified",
    "phone_mobile_valid": "Mobile phone verified",
    "bank_months_count": "Age of existing banking relationship",
    "bank_months_count_missing": "No existing banking relationship on file",
    "has_other_cards": "Holds other cards with this institution",
    "proposed_credit_limit": "Requested credit limit",
    "foreign_request": "Application originated outside the home country",
    "source": "Application channel",
    "session_length_in_minutes": "Time spent completing the application",
    "session_length_in_minutes_missing": "Session duration not captured",
    "device_os": "Device operating system",
    "keep_alive_session": "Session kept alive during application",
    "device_distinct_emails_8w": "Distinct emails from this device in 8 weeks",
    "device_distinct_emails_8w_missing": "Device email history unavailable",
    "device_fraud_count": "Prior fraud linked to this device",
}


def humanise(feature: str) -> str:
    """Plain-English label, falling back to a tidied column name."""
    return FEATURE_LABELS.get(feature, feature.replace("_", " ").capitalize())


@dataclass(frozen=True, slots=True)
class ReasonCode:
    """One feature's contribution to a single decision."""

    feature: str
    label: str
    value: Any
    contribution: float          # SHAP value in log-odds; +ve increases risk
    direction: str               # "increases_risk" | "reduces_risk"

    def as_sentence(self) -> str:
        verb = "raised" if self.direction == "increases_risk" else "lowered"
        return f"{self.label} ({self._format_value()}) {verb} the risk assessment"

    def _format_value(self) -> str:
        if self.value is None or (isinstance(self.value, float) and np.isnan(self.value)):
            return "not provided"
        if isinstance(self.value, (bool, np.bool_)):
            return "yes" if self.value else "no"
        if isinstance(self.value, (int, np.integer)):
            return f"{int(self.value):,}"
        if isinstance(self.value, (float, np.floating)):
            return f"{float(self.value):,.2f}"
        return str(self.value)


@dataclass(frozen=True, slots=True)
class Explanation:
    """The complete explanation attached to one decision."""

    top_risk_factors: list[ReasonCode]
    top_protective_factors: list[ReasonCode]
    base_value: float
    raw_score: float
    narrative: str = ""
    narrative_source: str = "template"   # "gemini" | "template"

    def adverse_action_reasons(self, limit: int = 4) -> list[str]:
        """Principal reasons, in the form an adverse-action notice requires.

        Regulation B obliges a lender to state the specific principal reasons
        for a denial. Ordering by attribution magnitude is what makes these
        genuinely *principal* rather than a generic list.
        """
        return [reason.label for reason in self.top_risk_factors[:limit]]


class ShapExplainer:
    """Computes per-application attributions from the trained booster."""

    def __init__(self, booster: "lgb.Booster", feature_names: list[str]) -> None:
        import shap

        self._feature_names = feature_names
        # TreeExplainer is exact for tree ensembles and fast enough to run
        # inside the request path - which matters, because an explanation
        # produced later by a different process is not the same explanation.
        self._explainer = shap.TreeExplainer(booster)

    def explain(self, features: pd.DataFrame, top_n: int = 5) -> Explanation:
        """Explain a single application.

        Args:
            features: exactly one row, columns matching the training contract.
            top_n: how many factors to surface in each direction.
        """
        if len(features) != 1:
            raise ValueError(f"explain() handles one application at a time, got {len(features)}")

        shap_values = self._explainer.shap_values(features)
        contributions = np.asarray(shap_values).reshape(-1)
        base_value = float(np.ravel(self._explainer.expected_value)[0])

        row = features.iloc[0]
        reasons = [
            ReasonCode(
                feature=name,
                label=humanise(name),
                value=row[name],
                contribution=float(contribution),
                direction="increases_risk" if contribution > 0 else "reduces_risk",
            )
            for name, contribution in zip(features.columns, contributions, strict=True)
            # Attributions of essentially zero are noise, not explanation.
            if abs(contribution) > 1e-6
        ]

        risk_factors = sorted(
            (r for r in reasons if r.contribution > 0),
            key=lambda r: r.contribution,
            reverse=True,
        )
        protective_factors = sorted(
            (r for r in reasons if r.contribution < 0),
            key=lambda r: r.contribution,
        )

        return Explanation(
            top_risk_factors=risk_factors[:top_n],
            top_protective_factors=protective_factors[:top_n],
            base_value=base_value,
            raw_score=float(base_value + contributions.sum()),
        )


def template_narrative(explanation: Explanation, decision: str) -> str:
    """Deterministic prose, used when the language model is unavailable.

    Kept fully functional rather than as a stub: the system must degrade
    gracefully, and a demo that depends on an external API to render text is a
    demo that fails on a bad conference network.
    """
    if not explanation.top_risk_factors:
        return f"Decision: {decision}. No individual factor materially raised the risk assessment."

    leading = explanation.top_risk_factors[0]
    supporting = explanation.top_risk_factors[1:3]

    parts = [
        f"Decision: {decision}.",
        f"The strongest driver was {leading.label.lower()} "
        f"({leading._format_value()}).",
    ]
    if supporting:
        joined = "; ".join(f"{r.label.lower()} ({r._format_value()})" for r in supporting)
        parts.append(f"Also contributing: {joined}.")
    if explanation.top_protective_factors:
        best = explanation.top_protective_factors[0]
        parts.append(f"Offsetting this, {best.label.lower()} ({best._format_value()}) "
                     "reduced the assessed risk.")
    return " ".join(parts)


NARRATION_SYSTEM_PROMPT = """\
You are briefing a fraud analyst at a digital lending institution on one
application. The decision has ALREADY been made by a scoring system. Your job
is to make its reasoning readable, not to re-open it.

You receive the factors that drove the decision, each with the applicant's
actual value and a relative weight.

Write EXACTLY 2-3 sentences of continuous prose. Every sentence must carry
information. Do not spend a sentence restating the decision.

Structure:
1. Name the strongest one or two factors AND their specific values, and say
   plainly what they indicate about this application.
2. Note any listed factor pulling the other way.
3. Close with the single most useful next action: for REVIEW or BLOCK, the
   first thing to verify; for APPROVE, the residual risk worth monitoring, or
   state plainly that no factor warranted intervention.

Absolute rules:
- Use ONLY the listed factors. Never invent a factor or a value.
- Quote values concretely ("no previous address on file", "name-to-email match
  of 0.17"), never vaguely ("some indicators", "several signals").
- Never speculate about the applicant's identity, ethnicity, intent or
  character, and never imply guilt.
- Do not question or recompute the decision.
- Do not mention machine learning, models, SHAP, scores, weights or
  probabilities. The analyst wants the reasoning, not the mechanism.
- No bullet points, no headings, no preamble such as "Here is the briefing".

Too thin, never acceptable: "This application was approved."

Good: "Housing status BA and a name-to-email match of 0.17 are the strongest
concerns, the latter being a common synthetic-identity pattern. Holding other
cards with the institution offsets this somewhat. Verify the email address
against the stated name before releasing funds."
"""


class GeminiNarrator:
    """Turns computed attributions into an analyst-readable briefing.

    Strictly downstream of the decision. Every failure path returns the
    deterministic template rather than raising, because narration is a
    convenience and must never be able to take the scoring service down.
    """

    def __init__(self, api_key: str, model: str, cache_size: int = 512) -> None:
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model = model
        # Narration runs at temperature 0, so a given set of attributions always
        # produces the same briefing. Re-requesting one is therefore pure waste
        # - and on a metered or free-tier key it is the difference between a
        # demo that narrates and one that has exhausted its quota mid-recording.
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._cache_size = cache_size

    def _cache_key(self, explanation: Explanation, decision: str) -> str:
        signature = "|".join(
            f"{r.feature}:{r.contribution:.6f}"
            for r in (*explanation.top_risk_factors, *explanation.top_protective_factors)
        )
        return hashlib.sha256(f"{decision}|{signature}".encode()).hexdigest()

    def narrate(self, explanation: Explanation, decision: str) -> tuple[str, str]:
        """Return ``(narrative, source)`` where source is 'gemini' or 'template'."""
        from google.genai import types

        key = self._cache_key(explanation, decision)
        if cached := self._cache.get(key):
            self._cache.move_to_end(key)
            return cached, "gemini"

        facts = {
            "decision": decision,
            "factors_increasing_risk": [
                {"factor": r.label, "value": r._format_value(),
                 "relative_weight": round(abs(r.contribution), 4)}
                for r in explanation.top_risk_factors
            ],
            "factors_reducing_risk": [
                {"factor": r.label, "value": r._format_value(),
                 "relative_weight": round(abs(r.contribution), 4)}
                for r in explanation.top_protective_factors
            ],
        }

        # No max_output_tokens is set, deliberately.
        #
        # Current Gemini models spend tokens on internal reasoning before
        # emitting any prose, and that reasoning counts against
        # max_output_tokens. A cap sized for "three sentences" is therefore
        # exhausted before the first word is written, and the briefing arrives
        # truncated mid-sentence - which reads as a broken system and can cut
        # off the recommended action. Verified against this model: an
        # uncapped request returns finish_reason=STOP with complete sentences,
        # while an 800-token cap truncates. `thinking_budget` is rejected by
        # this model, so suppressing the reasoning is not an option either.
        #
        # The length constraint is carried by the prompt instead, which is the
        # right place for it: "exactly 2-3 sentences" is an instruction, not a
        # transport limit.
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=f"Case facts:\n{facts}",
                config=types.GenerateContentConfig(
                    system_instruction=NARRATION_SYSTEM_PROMPT,
                    # Zero temperature: the same case must produce the same
                    # briefing every time, or the audit trail is worthless.
                    temperature=0.0,
                ),
            )
            text = (response.text or "").strip()

            if not text:
                logger.warning("gemini returned an empty narration; using template")
            elif not text.endswith((".", "!", "?")):
                # Belt and braces: never show a half-finished briefing.
                logger.warning("gemini narration truncated; using template")
            else:
                self._cache[key] = text
                if len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
                return text, "gemini"

        except Exception as error:   # noqa: BLE001 - narration never breaks scoring
            # Log the provider's own message, not just the exception class.
            # "ClientError" alone is undiagnosable; the common causes - an
            # exhausted quota (429), a retired model name (404), a rejected
            # key (403) - all need different fixes and are distinguishable
            # only from the message.
            logger.warning(
                "gemini narration failed (%s: %s); using deterministic template",
                type(error).__name__,
                str(error).replace("\n", " ")[:200],
            )

        return template_narrative(explanation, decision), "template"
