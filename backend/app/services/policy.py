"""Cost-sensitive decision policy.

Why this module exists
----------------------
A fraud model outputs a probability. A *lending business* needs an action:
approve the application, send it to a human analyst, or block it. Choosing the
cut-off that turns one into the other is the single most consequential decision
in the whole system, and the usual approach — leaving it at 0.5, or tuning it
until the F1 score looks nice — has no defensible justification.

This module derives the cut-offs instead. Given what each outcome actually
costs the bank, the optimal action for an application with fraud probability
``p`` is simply the action with the lowest expected cost. Working that out
analytically gives two thresholds, and they fall out of arithmetic rather than
tuning:

    E[cost | APPROVE] = p * C_fn
    E[cost | REVIEW ] = C_rev + p * (1 - r) * C_fn
    E[cost | BLOCK  ] = (1 - p) * C_fp

where
    C_fn  cost of approving a fraudulent application (expected write-off)
    C_fp  cost of blocking a genuine applicant (lost customer, servicing,
          reputational drag) - this is the "false positive" the brief asks us
          to reduce, priced rather than hand-waved
    C_rev cost of a manual analyst review
    r     probability an analyst correctly identifies fraud once reviewing

Solving the pairwise indifference points:

    tau_review = C_rev / (r * C_fn)
    tau_block  = (C_fp - C_rev) / (C_fn * (1 - r) + C_fp)

Two things follow that are worth stating out loud in any review of this system:

1. Reducing false positives is not a modelling trick here, it is an economic
   statement. Raising C_fp moves tau_block up and fewer genuine customers are
   blocked. The risk owner controls that dial through configuration, without a
   retrain and without a deploy.

2. These thresholds are only meaningful if ``p`` is a *calibrated* probability.
   Raw gradient-boosting outputs are not calibrated - they rank well but are
   not true probabilities - so the training pipeline fits a calibration layer
   before this policy is applied. Applying cost-based thresholds to
   uncalibrated scores is a common and quietly invalidating mistake.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class Decision(StrEnum):
    """The three actions the platform can take on an application."""

    APPROVE = "APPROVE"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"


@dataclass(frozen=True, slots=True)
class CostModel:
    """Business cost inputs, in rupees, that determine the decision boundaries.

    These are deliberately *not* model hyperparameters. They describe the
    bank's risk appetite and belong to a risk owner, not to a data scientist.
    """

    cost_fp: float          # blocking a genuine applicant
    cost_fn: float          # approving a fraudulent application
    cost_review: float      # one manual analyst review
    analyst_catch_rate: float = 0.90    # r: analyst correctly flags fraud

    def __post_init__(self) -> None:
        if self.cost_fp <= 0 or self.cost_fn <= 0 or self.cost_review < 0:
            raise ValueError("costs must be positive (review cost may be zero)")
        if not 0.0 < self.analyst_catch_rate <= 1.0:
            raise ValueError("analyst_catch_rate must lie in (0, 1]")
        if self.cost_review >= self.cost_fp:
            raise ValueError(
                "cost_review must be below cost_fp, otherwise reviewing is never "
                "cheaper than blocking and the review band collapses"
            )


@dataclass(frozen=True, slots=True)
class DecisionPolicy:
    """Two thresholds and the rule that applies them."""

    tau_review: float
    tau_block: float
    cost_model: CostModel

    def decide(self, fraud_probability: float) -> Decision:
        """Map a calibrated fraud probability to an action."""
        if not 0.0 <= fraud_probability <= 1.0:
            raise ValueError(f"probability out of range: {fraud_probability}")
        if fraud_probability >= self.tau_block:
            return Decision.BLOCK
        if fraud_probability >= self.tau_review:
            return Decision.REVIEW
        return Decision.APPROVE

    def decide_many(self, probabilities: np.ndarray) -> np.ndarray:
        """Vectorised form of :meth:`decide`, for evaluation over a dataset."""
        probabilities = np.asarray(probabilities, dtype=float)
        actions = np.full(probabilities.shape, Decision.APPROVE.value, dtype=object)
        actions[probabilities >= self.tau_review] = Decision.REVIEW.value
        actions[probabilities >= self.tau_block] = Decision.BLOCK.value
        return actions


def derive_policy(cost_model: CostModel) -> DecisionPolicy:
    """Derive the Bayes-optimal thresholds from the cost model.

    No data is involved: given the costs, these boundaries are where the
    expected cost of one action overtakes another.
    """
    c_fp = cost_model.cost_fp
    c_fn = cost_model.cost_fn
    c_rev = cost_model.cost_review
    r = cost_model.analyst_catch_rate

    tau_review = c_rev / (r * c_fn)
    tau_block = (c_fp - c_rev) / (c_fn * (1.0 - r) + c_fp)

    # Degenerate case: when manual review is expensive relative to the loss it
    # prevents, the optimal policy has no review band at all and collapses to a
    # straight approve/block cut. Surfacing that rather than silently producing
    # an inverted band keeps the policy honest.
    if tau_review >= tau_block:
        midpoint = c_fp / (c_fp + c_fn)
        tau_review = tau_block = midpoint

    return DecisionPolicy(
        tau_review=float(np.clip(tau_review, 0.0, 1.0)),
        tau_block=float(np.clip(tau_block, 0.0, 1.0)),
        cost_model=cost_model,
    )


# --------------------------------------------------------------------------
# Evaluation helpers - used by the training pipeline and the /metrics routes
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PolicyOutcome:
    """What a policy actually does when applied to a labelled dataset."""

    n_total: int
    n_approved: int
    n_reviewed: int
    n_blocked: int
    fraud_approved: int      # fraud that slipped through - the expensive miss
    genuine_blocked: int     # false positives - the customer-experience cost
    fraud_blocked: int
    fraud_reviewed: int
    total_cost: float
    cost_per_application: float

    @property
    def review_rate(self) -> float:
        return self.n_reviewed / self.n_total if self.n_total else 0.0

    @property
    def false_positive_rate(self) -> float:
        """Share of genuine applicants that were blocked."""
        genuine = self.n_total - (self.fraud_approved + self.fraud_blocked + self.fraud_reviewed)
        return self.genuine_blocked / genuine if genuine else 0.0


def evaluate_policy(
    y_true: np.ndarray,
    fraud_probability: np.ndarray,
    policy: DecisionPolicy,
) -> PolicyOutcome:
    """Apply a policy to labelled data and price the result.

    Reported cost is what the bank would have spent on this population, which
    is the number a business reviewer can actually interpret - unlike AUC.
    """
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(fraud_probability, dtype=float)
    if y_true.shape != probabilities.shape:
        raise ValueError("y_true and fraud_probability must have the same shape")

    is_fraud = y_true == 1
    blocked = probabilities >= policy.tau_block
    reviewed = (probabilities >= policy.tau_review) & ~blocked
    approved = ~blocked & ~reviewed

    cm = policy.cost_model
    fraud_approved = int(np.sum(approved & is_fraud))
    genuine_blocked = int(np.sum(blocked & ~is_fraud))
    fraud_reviewed = int(np.sum(reviewed & is_fraud))

    # Fraud reaching review is caught with probability r; the remainder is
    # approved in error and costs a write-off.
    review_leakage = fraud_reviewed * (1.0 - cm.analyst_catch_rate)

    total_cost = (
        fraud_approved * cm.cost_fn
        + genuine_blocked * cm.cost_fp
        + int(np.sum(reviewed)) * cm.cost_review
        + review_leakage * cm.cost_fn
    )

    n_total = int(y_true.size)
    return PolicyOutcome(
        n_total=n_total,
        n_approved=int(np.sum(approved)),
        n_reviewed=int(np.sum(reviewed)),
        n_blocked=int(np.sum(blocked)),
        fraud_approved=fraud_approved,
        genuine_blocked=genuine_blocked,
        fraud_blocked=int(np.sum(blocked & is_fraud)),
        fraud_reviewed=fraud_reviewed,
        total_cost=float(total_cost),
        cost_per_application=float(total_cost / n_total) if n_total else 0.0,
    )


def constrain_review_capacity(
    policy: DecisionPolicy,
    reference_probabilities: np.ndarray,
    max_review_rate: float,
) -> DecisionPolicy:
    """Raise ``tau_review`` until the review queue fits the analyst team.

    The Bayes-optimal thresholds in :func:`derive_policy` assume manual review
    is available on demand at a fixed unit price. Real fraud operations have a
    fixed headcount, and a policy that routes 40% of applications to a team
    that can handle 5% is not a policy - it is a backlog.

    This raises the review threshold to whatever level fills the available
    capacity with the *highest-risk* reviewable applications, which is the
    correct triage when capacity binds: the analysts' time goes where expected
    loss is greatest. ``tau_block`` is untouched, because blocking capacity is
    not the constraint.

    Args:
        policy: the unconstrained, cost-derived policy.
        reference_probabilities: calibrated fraud probabilities from a
            representative sample (the validation split), used to locate the
            capacity quantile.
        max_review_rate: share of total volume the analyst team can absorb,
            e.g. 0.05 for 5%.
    """
    if not 0.0 <= max_review_rate <= 1.0:
        raise ValueError("max_review_rate must lie in [0, 1]")

    probabilities = np.asarray(reference_probabilities, dtype=float)
    if probabilities.size == 0:
        return policy

    reviewable = probabilities[
        (probabilities >= policy.tau_review) & (probabilities < policy.tau_block)
    ]
    capacity = int(np.floor(max_review_rate * probabilities.size))

    if capacity <= 0:
        # No review capability at all: degrade to a straight approve/block cut.
        return DecisionPolicy(
            tau_review=policy.tau_block,
            tau_block=policy.tau_block,
            cost_model=policy.cost_model,
        )

    if reviewable.size <= capacity:
        return policy   # already fits; nothing to constrain

    # Keep the `capacity` riskiest reviewable applications.
    cutoff = float(np.partition(reviewable, -capacity)[-capacity])
    return DecisionPolicy(
        tau_review=cutoff,
        tau_block=policy.tau_block,
        cost_model=policy.cost_model,
    )


def naive_baseline_policy(cost_model: CostModel, threshold: float = 0.5) -> DecisionPolicy:
    """The policy this project exists to beat.

    A single 0.5 cut-off with no review band - the library default that most
    fraud prototypes ship. Kept in the codebase so the comparison in the
    results is reproducible rather than rhetorical.
    """
    return DecisionPolicy(tau_review=threshold, tau_block=threshold, cost_model=cost_model)
