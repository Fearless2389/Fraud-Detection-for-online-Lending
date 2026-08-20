"""API request and response contracts.

These models are the API's public surface. They exist for three reasons beyond
serialisation:

* **Input validation is a security control.** Every field is bounded. An
  unvalidated numeric field reaching a model is both a crash risk and a probing
  surface - an attacker who can send arbitrary values can map the decision
  boundary far faster than one constrained to plausible applications.
* **They document the contract.** FastAPI generates the OpenAPI schema from
  these definitions, so the docs cannot drift from the implementation.
* **They pin the training/serving boundary.** Field names match the trained
  feature contract exactly, so a mismatch surfaces as a 422 at the edge rather
  than as a silently wrong score.

Field semantics follow the BAF dataset. Values are bounded to the ranges
observed during EDA, widened where a legitimate real-world value could fall
outside the sample.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

# Categorical domains, taken from the dataset rather than invented. Codes are
# anonymised in BAF; in a real deployment these would carry business meaning.
PaymentType = Literal["AA", "AB", "AC", "AD", "AE"]
EmploymentStatus = Literal["CA", "CB", "CC", "CD", "CE", "CF", "CG"]
HousingStatus = Literal["BA", "BB", "BC", "BD", "BE", "BF", "BG"]
ApplicationSource = Literal["INTERNET", "TELEAPP"]
DeviceOS = Literal["windows", "linux", "macintosh", "x11", "other"]

Flag = Annotated[int, Field(ge=0, le=1, description="0 or 1")]


class ApplicationIn(BaseModel):
    """A digital lending application submitted for a real-time decision."""

    model_config = ConfigDict(
        extra="forbid",   # unknown fields are rejected, not silently ignored
        json_schema_extra={
            "example": {
                "income": 0.3,
                "name_email_similarity": 0.42,
                "prev_address_months_count": -1,
                "current_address_months_count": 24,
                "customer_age": 30,
                "days_since_request": 0.01,
                "intended_balcon_amount": -1.2,
                "payment_type": "AB",
                "zip_count_4w": 1300,
                "velocity_6h": 5200.0,
                "velocity_24h": 4800.0,
                "velocity_4w": 4500.0,
                "bank_branch_count_8w": 8,
                "date_of_birth_distinct_emails_4w": 3,
                "employment_status": "CA",
                "credit_risk_score": 130,
                "email_is_free": 1,
                "housing_status": "BC",
                "phone_home_valid": 0,
                "phone_mobile_valid": 1,
                "bank_months_count": 12,
                "has_other_cards": 0,
                "proposed_credit_limit": 1500.0,
                "foreign_request": 0,
                "source": "INTERNET",
                "session_length_in_minutes": 6.2,
                "device_os": "windows",
                "keep_alive_session": 1,
                "device_distinct_emails_8w": 1,
                "device_fraud_count": 0,
            }
        },
    )

    # --- applicant profile ---
    income: float = Field(ge=0.0, le=1.0, description="Income band, normalised 0.1-0.9")
    customer_age: int = Field(ge=10, le=110, description="Age band, in decades")
    employment_status: EmploymentStatus
    housing_status: HousingStatus
    credit_risk_score: int = Field(
        ge=-200, le=400,
        description="Internal risk score. Legitimately negative at the low end.",
    )

    # --- address history (-1 means not available) ---
    prev_address_months_count: int = Field(ge=-1, le=1_000)
    current_address_months_count: int = Field(ge=-1, le=1_000)

    # --- identity coherence ---
    name_email_similarity: float = Field(
        ge=0.0, le=1.0,
        description="Similarity between applicant name and email address. Low "
                    "values are a classic synthetic-identity signal.",
    )
    email_is_free: Flag
    date_of_birth_distinct_emails_4w: int = Field(ge=0, le=100)

    # --- contactability ---
    phone_home_valid: Flag
    phone_mobile_valid: Flag

    # --- existing relationship ---
    bank_months_count: int = Field(ge=-1, le=600)
    has_other_cards: Flag

    # --- the request itself ---
    payment_type: PaymentType
    proposed_credit_limit: float = Field(ge=0.0, le=10_000_000.0)
    intended_balcon_amount: float = Field(
        ge=-100.0, le=200.0,
        description="Negative values indicate no balance transfer intended.",
    )
    days_since_request: float = Field(ge=0.0, le=1_000.0)
    foreign_request: Flag
    source: ApplicationSource

    # --- velocity and network signals ---
    zip_count_4w: int = Field(ge=0, le=10_000)
    velocity_6h: float = Field(ge=-1_000.0, le=100_000.0)
    velocity_24h: float = Field(ge=0.0, le=100_000.0)
    velocity_4w: float = Field(ge=0.0, le=100_000.0)
    bank_branch_count_8w: int = Field(ge=0, le=10_000)

    # --- device and session behaviour ---
    device_os: DeviceOS
    session_length_in_minutes: float = Field(ge=-1.0, le=1_000.0)
    keep_alive_session: Flag
    device_distinct_emails_8w: int = Field(ge=-1, le=100)
    device_fraud_count: int = Field(ge=0, le=100)


class ReasonCodeOut(BaseModel):
    """One factor's contribution to the decision."""

    feature: str
    label: str = Field(description="Plain-English description of the factor")
    value: str = Field(description="The applicant's value, formatted for display")
    contribution: float = Field(description="Signed attribution; positive raises risk")
    direction: Literal["increases_risk", "reduces_risk"]


class SimilarCaseOut(BaseModel):
    """A previously confirmed fraud case resembling this application."""

    case_id: str
    similarity: float = Field(ge=0.0, le=1.0)
    confirmed_fraud: bool
    matched_on: list[str] = Field(
        default_factory=list,
        description="Factors this case has in common with the application",
    )


class DecisionOut(BaseModel):
    """The complete, auditable result of scoring one application."""

    application_id: str
    decision: Literal["APPROVE", "REVIEW", "BLOCK"]
    fraud_probability: float = Field(
        ge=0.0, le=1.0,
        description="Calibrated probability. A value of 0.03 means roughly "
                    "3 in 100 such applications are fraudulent.",
    )
    risk_band: Literal["low", "elevated", "high", "severe"]

    thresholds: dict[str, float] = Field(
        description="The review and block thresholds in force for this decision"
    )
    top_risk_factors: list[ReasonCodeOut]
    top_protective_factors: list[ReasonCodeOut]
    adverse_action_reasons: list[str] = Field(
        description="Principal reasons, ordered by contribution, in the form a "
                    "regulator expects for an adverse decision."
    )
    narrative: str
    narrative_source: Literal["gemini", "template"]
    similar_confirmed_cases: list[SimilarCaseOut] = Field(default_factory=list)

    model_version: str
    scored_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    latency_ms: float = Field(description="Server-side scoring latency")


class AnalystDecisionIn(BaseModel):
    """An analyst's verdict on a reviewed case - the feedback loop's input."""

    model_config = ConfigDict(extra="forbid")

    outcome: Literal["confirmed_fraud", "cleared_genuine"]
    analyst_id: str = Field(min_length=1, max_length=64)
    notes: str = Field(default="", max_length=2_000)


class AnalystDecisionOut(BaseModel):
    application_id: str
    outcome: str
    recorded_at: datetime
    added_to_similarity_index: bool = Field(
        description="Confirmed fraud is indexed immediately, so subsequent "
                    "lookalike applications are flagged without retraining."
    )
