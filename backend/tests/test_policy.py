"""Tests for the cost-sensitive decision policy.

The thresholds in this system are derived arithmetic, not tuned constants, so
they can be verified exactly against hand-computed values. That is the whole
point of deriving them: the operating point of the fraud platform is provable
rather than empirical folklore.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.services.policy import (
    CostModel,
    Decision,
    DecisionPolicy,
    constrain_review_capacity,
    derive_policy,
    evaluate_policy,
    naive_baseline_policy,
)


def calibrated_population(
    n: int = 20_000, seed: int = 20260820
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a population whose scores are calibrated *by construction*.

    A fraud probability is drawn for each applicant, then the label is drawn
    from that probability. This guarantees that a score of 0.03 really does
    mean a 3% chance of fraud, which is the precondition for cost-derived
    thresholds to mean anything at all.

    The Beta(0.5, 45) shape gives a mean near 0.011, matching the roughly 1%
    fraud prevalence of the BAF dataset, with most applicants near zero risk
    and a thin high-risk tail.
    """
    rng = np.random.default_rng(seed)
    probabilities = rng.beta(0.5, 45.0, size=n)
    labels = (rng.random(n) < probabilities).astype(int)
    return labels, probabilities


@pytest.fixture
def costs() -> CostModel:
    """The project's default economics: see .env.example."""
    return CostModel(
        cost_fp=1_500.0,
        cost_fn=45_000.0,
        cost_review=200.0,
        analyst_catch_rate=0.90,
    )


class TestThresholdDerivation:
    def test_matches_hand_computed_values(self, costs: CostModel) -> None:
        policy = derive_policy(costs)

        # tau_review = C_rev / (r * C_fn) = 200 / (0.9 * 45000)
        assert policy.tau_review == pytest.approx(200 / (0.9 * 45_000))
        # tau_block = (C_fp - C_rev) / (C_fn * (1 - r) + C_fp)
        #           = (1500 - 200) / (45000 * 0.1 + 1500)
        assert policy.tau_block == pytest.approx(1_300 / (4_500 + 1_500))

    def test_review_band_is_ordered_and_non_empty(self, costs: CostModel) -> None:
        policy = derive_policy(costs)
        assert 0.0 < policy.tau_review < policy.tau_block < 1.0

    def test_raising_false_positive_cost_raises_block_threshold(self, costs: CostModel) -> None:
        """The economic claim behind 'reducing false positives'.

        Making a wrongly-blocked customer more expensive must make the system
        less willing to block. If this ever fails, the headline argument of the
        project is wrong.
        """
        cheap = derive_policy(costs)
        expensive = derive_policy(
            CostModel(
                cost_fp=costs.cost_fp * 4,
                cost_fn=costs.cost_fn,
                cost_review=costs.cost_review,
                analyst_catch_rate=costs.analyst_catch_rate,
            )
        )
        assert expensive.tau_block > cheap.tau_block

    def test_raising_fraud_loss_lowers_review_threshold(self, costs: CostModel) -> None:
        """Costlier fraud should pull more applications into human review."""
        base = derive_policy(costs)
        severe = derive_policy(
            CostModel(
                cost_fp=costs.cost_fp,
                cost_fn=costs.cost_fn * 4,
                cost_review=costs.cost_review,
                analyst_catch_rate=costs.analyst_catch_rate,
            )
        )
        assert severe.tau_review < base.tau_review

    def test_expensive_review_collapses_the_band(self) -> None:
        """When review costs nearly as much as a false block, review stops
        being worthwhile and the policy must degrade to a single cut rather
        than silently invert its thresholds."""
        policy = derive_policy(
            CostModel(
                cost_fp=1_000.0,
                cost_fn=1_200.0,
                cost_review=900.0,
                analyst_catch_rate=0.5,
            )
        )
        assert policy.tau_review == policy.tau_block


class TestCostModelValidation:
    def test_rejects_review_costlier_than_false_positive(self) -> None:
        with pytest.raises(ValueError, match="cost_review must be below cost_fp"):
            CostModel(cost_fp=100.0, cost_fn=5_000.0, cost_review=150.0)

    def test_rejects_non_positive_costs(self) -> None:
        with pytest.raises(ValueError, match="costs must be positive"):
            CostModel(cost_fp=0.0, cost_fn=5_000.0, cost_review=10.0)

    @pytest.mark.parametrize("rate", [0.0, -0.1, 1.5])
    def test_rejects_impossible_analyst_catch_rate(self, rate: float) -> None:
        with pytest.raises(ValueError, match="analyst_catch_rate"):
            CostModel(cost_fp=1_000.0, cost_fn=5_000.0, cost_review=10.0, analyst_catch_rate=rate)


class TestDecisionBoundaries:
    @pytest.fixture
    def policy(self) -> DecisionPolicy:
        return DecisionPolicy(
            tau_review=0.20,
            tau_block=0.60,
            cost_model=CostModel(cost_fp=1_000.0, cost_fn=20_000.0, cost_review=100.0),
        )

    @pytest.mark.parametrize(
        ("probability", "expected"),
        [
            (0.00, Decision.APPROVE),
            (0.199, Decision.APPROVE),
            (0.20, Decision.REVIEW),    # boundary is inclusive at the lower edge
            (0.599, Decision.REVIEW),
            (0.60, Decision.BLOCK),     # and at the upper edge
            (1.00, Decision.BLOCK),
        ],
    )
    def test_boundaries_are_inclusive_from_below(
        self, policy: DecisionPolicy, probability: float, expected: Decision
    ) -> None:
        assert policy.decide(probability) is expected

    @pytest.mark.parametrize("probability", [-0.01, 1.01, math.nan])
    def test_rejects_out_of_range_probabilities(
        self, policy: DecisionPolicy, probability: float
    ) -> None:
        with pytest.raises(ValueError):
            policy.decide(probability)

    def test_vectorised_form_agrees_with_scalar_form(self, policy: DecisionPolicy) -> None:
        probabilities = np.linspace(0.0, 1.0, 101)
        vectorised = policy.decide_many(probabilities)
        scalar = [policy.decide(p).value for p in probabilities]
        assert list(vectorised) == scalar


class TestPolicyEvaluation:
    def test_prices_a_hand_built_population(self) -> None:
        """Six applications with known outcomes, costed by hand.

        policy: review at 0.2, block at 0.6; r = 1.0 so review leaks nothing.
        """
        costs = CostModel(
            cost_fp=1_000.0,
            cost_fn=20_000.0,
            cost_review=100.0,
            analyst_catch_rate=1.0,
        )
        policy = DecisionPolicy(tau_review=0.2, tau_block=0.6, cost_model=costs)

        #                    genuine  fraud  genuine  genuine  fraud  fraud
        y_true = np.array([0, 1, 0, 0, 1, 1])
        scores = np.array([0.05, 0.10, 0.30, 0.90, 0.40, 0.95])
        #  action:        APPROVE APPROVE REVIEW BLOCK  REVIEW BLOCK

        outcome = evaluate_policy(y_true, scores, policy)

        assert outcome.n_approved == 2
        assert outcome.n_reviewed == 2
        assert outcome.n_blocked == 2
        assert outcome.fraud_approved == 1      # the 0.10-scored fraud slipped through
        assert outcome.genuine_blocked == 1     # the 0.90-scored genuine applicant

        # 1 missed fraud (20000) + 1 blocked genuine (1000) + 2 reviews (200)
        assert outcome.total_cost == pytest.approx(21_200.0)
        assert outcome.cost_per_application == pytest.approx(21_200.0 / 6)

    def test_review_leakage_is_charged_when_analysts_are_imperfect(self) -> None:
        """A 90% analyst lets 10% of reviewed fraud through, and that costs money."""
        costs = CostModel(
            cost_fp=1_000.0, cost_fn=20_000.0, cost_review=100.0, analyst_catch_rate=0.9
        )
        policy = DecisionPolicy(tau_review=0.2, tau_block=0.6, cost_model=costs)

        y_true = np.array([1])
        scores = np.array([0.40])       # one fraud, routed to review

        outcome = evaluate_policy(y_true, scores, policy)
        # one review (100) + 10% leakage of a 20000 write-off (2000)
        assert outcome.total_cost == pytest.approx(2_100.0)

    def test_derived_policy_beats_the_naive_half_threshold(self) -> None:
        """The project's central quantitative claim.

        Given calibrated probabilities, the cost-derived policy chooses the
        minimum-expected-cost action for every applicant, so it cannot be
        beaten by any fixed threshold - including the 0.5 default that most
        fraud prototypes ship. This is a property of the derivation, not an
        artefact of one dataset, which is why it is tested on a controlled
        population as well as measured on BAF.
        """
        y_true, probabilities = calibrated_population()
        costs = CostModel(
            cost_fp=1_500.0, cost_fn=45_000.0, cost_review=200.0, analyst_catch_rate=0.9
        )

        derived = evaluate_policy(y_true, probabilities, derive_policy(costs))
        naive = evaluate_policy(y_true, probabilities, naive_baseline_policy(costs))

        assert derived.total_cost < naive.total_cost

    def test_uncalibrated_scores_invalidate_the_cost_policy(self) -> None:
        """Regression guard for the mistake this design exists to avoid.

        Cost-derived thresholds are statements about *probabilities*. Feed them
        raw model scores that merely rank well - as gradient boosting produces
        by default - and the policy degrades badly: here it floods the review
        queue and loses to the naive baseline it should beat.

        This test asserts the failure deliberately. If a future change makes it
        pass, someone has silently altered the semantics of the score input,
        and the calibration step in the training pipeline is no longer load
        bearing. That would be worth knowing.
        """
        rng = np.random.default_rng(20260820)
        n = 20_000
        y_true = (rng.random(n) < 0.011).astype(int)

        # Ranks fraud above genuine, but the values are not probabilities:
        # the genuine class averages ~0.09 against a true 1.1% base rate.
        uncalibrated = np.where(
            y_true == 1,
            rng.beta(5.0, 5.0, size=n),
            rng.beta(1.2, 12.0, size=n),
        )

        costs = CostModel(
            cost_fp=1_500.0, cost_fn=45_000.0, cost_review=200.0, analyst_catch_rate=0.9
        )
        derived = evaluate_policy(y_true, uncalibrated, derive_policy(costs))
        naive = evaluate_policy(y_true, uncalibrated, naive_baseline_policy(costs))

        assert derived.total_cost > naive.total_cost
        assert derived.review_rate > 0.5      # the review queue floods


class TestReviewCapacity:
    """Bayes-optimal thresholds ignore headcount. Real fraud teams cannot."""

    @pytest.fixture
    def costs(self) -> CostModel:
        return CostModel(
            cost_fp=1_500.0, cost_fn=45_000.0, cost_review=200.0, analyst_catch_rate=0.9
        )

    def test_review_queue_is_capped_at_available_capacity(self, costs: CostModel) -> None:
        y_true, probabilities = calibrated_population()
        unconstrained = derive_policy(costs)
        constrained = constrain_review_capacity(unconstrained, probabilities, max_review_rate=0.05)

        outcome = evaluate_policy(y_true, probabilities, constrained)
        assert outcome.review_rate <= 0.05
        assert constrained.tau_review > unconstrained.tau_review
        assert constrained.tau_block == unconstrained.tau_block   # blocking is unconstrained

    def test_capacity_triage_keeps_the_riskiest_applications(self, costs: CostModel) -> None:
        """When capacity binds, analysts must see the highest-risk cases."""
        policy = DecisionPolicy(tau_review=0.01, tau_block=0.90, cost_model=costs)
        probabilities = np.array([0.02, 0.05, 0.30, 0.60, 0.80, 0.005, 0.001, 0.002, 0.003, 0.004])

        constrained = constrain_review_capacity(policy, probabilities, max_review_rate=0.2)

        # Two review slots for ten applications: the 0.60 and 0.80 cases.
        assert constrained.tau_review == pytest.approx(0.60)

    def test_zero_capacity_collapses_to_approve_or_block(self, costs: CostModel) -> None:
        _, probabilities = calibrated_population(n=1_000)
        policy = derive_policy(costs)
        constrained = constrain_review_capacity(policy, probabilities, max_review_rate=0.0)
        assert constrained.tau_review == constrained.tau_block

    def test_ample_capacity_leaves_the_policy_untouched(self, costs: CostModel) -> None:
        _, probabilities = calibrated_population(n=1_000)
        policy = derive_policy(costs)
        constrained = constrain_review_capacity(policy, probabilities, max_review_rate=1.0)
        assert constrained == policy

    def test_rejects_impossible_capacity(self, costs: CostModel) -> None:
        _, probabilities = calibrated_population(n=100)
        with pytest.raises(ValueError, match="max_review_rate"):
            constrain_review_capacity(derive_policy(costs), probabilities, max_review_rate=1.5)

    def test_rejects_mismatched_input_shapes(self) -> None:
        costs = CostModel(cost_fp=1_000.0, cost_fn=20_000.0, cost_review=100.0)
        policy = derive_policy(costs)
        with pytest.raises(ValueError, match="same shape"):
            evaluate_policy(np.array([0, 1]), np.array([0.5]), policy)
