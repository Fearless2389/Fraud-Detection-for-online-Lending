"""Tests for the training/serving feature contract.

The test that matters most here is `test_single_row_scoring_matches_batch`.
It guards a failure mode that produces no error at all: pandas categoricals are
identified by integer code, so building a category from a one-row request
assigns codes from that row alone and the model reads a different category than
the caller sent. Scores stay in range, nothing raises, and every prediction is
quietly wrong.

Batch-versus-single-row equivalence is the cheapest way to catch it, and it
catches the whole family of training/serving skew bugs, not just this one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.features.pipeline import (
    CATEGORICAL,
    SENTINEL_MISSING,
    build_feature_spec,
    clean_raw,
    prepare,
    temporal_split,
)

ARTIFACT = Path(__file__).resolve().parents[1] / "ml" / "artifacts" / "model_bundle.joblib"
DATA = Path("C:/dev/data/baf/base.parquet")

needs_artifacts = pytest.mark.skipif(
    not (ARTIFACT.exists() and DATA.exists()),
    reason="requires a trained model bundle and the BAF parquet cache",
)


def make_frame(**overrides) -> pd.DataFrame:
    """A minimal frame carrying every column the pipeline touches."""
    base = {
        "fraud_bool": 0,
        "month": 0,
        "income": 0.3,
        "name_email_similarity": 0.5,
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
    base.update(overrides)
    return pd.DataFrame([base])


class TestSentinelHandling:
    def test_sentinel_becomes_nan_with_an_indicator(self) -> None:
        cleaned = clean_raw(make_frame(prev_address_months_count=-1))
        assert np.isnan(cleaned["prev_address_months_count"].iloc[0])
        assert cleaned["prev_address_months_count_missing"].iloc[0] == 1

    def test_real_values_survive_untouched(self) -> None:
        cleaned = clean_raw(make_frame(prev_address_months_count=36))
        assert cleaned["prev_address_months_count"].iloc[0] == 36
        assert cleaned["prev_address_months_count_missing"].iloc[0] == 0

    @pytest.mark.parametrize("column", sorted(SENTINEL_MISSING))
    def test_every_sentinel_column_gets_an_indicator(self, column: str) -> None:
        cleaned = clean_raw(make_frame(**{column: SENTINEL_MISSING[column]}))
        assert f"{column}_missing" in cleaned.columns

    def test_impossible_negative_velocity_becomes_missing(self) -> None:
        """Differential-privacy noise pushes velocity below zero; it is not a
        real measurement and must not be treated as one."""
        cleaned = clean_raw(make_frame(velocity_6h=-170.6))
        assert np.isnan(cleaned["velocity_6h"].iloc[0])

    def test_balcon_amount_splits_into_flag_and_value(self) -> None:
        no_intent = clean_raw(make_frame(intended_balcon_amount=-15.5))
        assert no_intent["has_balcon_intent"].iloc[0] == 0
        assert np.isnan(no_intent["intended_balcon_amount"].iloc[0])

        with_intent = clean_raw(make_frame(intended_balcon_amount=42.0))
        assert with_intent["has_balcon_intent"].iloc[0] == 1
        assert with_intent["intended_balcon_amount"].iloc[0] == 42.0


class TestCategoricalContract:
    def test_levels_are_recorded_in_the_spec(self) -> None:
        frame = pd.concat([
            make_frame(device_os="windows"),
            make_frame(device_os="linux"),
            make_frame(device_os="macintosh"),
        ], ignore_index=True)
        _, spec = prepare(frame)
        assert set(spec.categorical_levels["device_os"]) == {"windows", "linux", "macintosh"}

    def test_serving_reuses_training_codes(self) -> None:
        """The core guard: a one-row request must encode as it did in training."""
        training = pd.concat(
            [make_frame(device_os=os_name) for os_name in
             ("linux", "macintosh", "other", "windows", "x11")],
            ignore_index=True,
        )
        _, spec = prepare(training)

        served, _ = prepare(make_frame(device_os="windows"), spec)
        expected_code = list(spec.categorical_levels["device_os"]).index("windows")
        assert served["device_os"].cat.codes.iloc[0] == expected_code

    def test_unseen_category_becomes_missing_not_a_wrong_code(self) -> None:
        training = pd.concat(
            [make_frame(device_os=name) for name in ("windows", "linux")],
            ignore_index=True,
        )
        _, spec = prepare(training)
        served, _ = prepare(make_frame(device_os="macintosh"), spec)
        # -1 is pandas' code for "not in categories" -> LightGBM reads missing.
        assert served["device_os"].cat.codes.iloc[0] == -1

    @pytest.mark.parametrize("column", CATEGORICAL)
    def test_all_categorical_columns_are_pinned(self, column: str) -> None:
        """Every categorical that survives into the model carries its levels.

        The frame must vary each column: a column with one distinct value is
        constant, carries no signal, and is dropped from the feature set by
        design - so it legitimately has no levels to pin.
        """
        varied = {
            "payment_type": ["AA", "AB"],
            "employment_status": ["CA", "CB"],
            "housing_status": ["BA", "BC"],
            "source": ["INTERNET", "TELEAPP"],
            "device_os": ["windows", "linux"],
        }
        frame = pd.concat(
            [
                make_frame(**{name: values[index] for name, values in varied.items()})
                for index in (0, 1)
            ],
            ignore_index=True,
        )
        _, spec = prepare(frame)
        assert column in spec.feature_names
        assert column in spec.categorical_levels
        assert len(spec.categorical_levels[column]) == 2


class TestTemporalSplit:
    def test_splits_are_disjoint(self) -> None:
        frame = pd.concat(
            [make_frame(month=month) for month in range(8)], ignore_index=True
        )
        split = temporal_split(frame)
        months = [
            set(part["month"]) for part in
            (split.train, split.validation, split.calibration, split.test)
        ]
        for index, first in enumerate(months):
            for second in months[index + 1:]:
                assert not (first & second)

    def test_overlapping_configuration_is_rejected(self) -> None:
        frame = pd.concat(
            [make_frame(month=month) for month in range(8)], ignore_index=True
        )
        with pytest.raises(ValueError, match="appears in both"):
            temporal_split(frame, train_months=(0, 1), validation_months=(1,))

    def test_test_months_are_the_latest(self) -> None:
        frame = pd.concat(
            [make_frame(month=month) for month in range(8)], ignore_index=True
        )
        split = temporal_split(frame)
        assert min(split.test["month"]) > max(split.train["month"])


class TestConstantColumnsDropped:
    def test_constant_column_is_excluded(self) -> None:
        frame = pd.concat([make_frame(income=0.3), make_frame(income=0.7)],
                          ignore_index=True)
        cleaned = clean_raw(frame)
        spec = build_feature_spec(cleaned)
        # device_fraud_count is 0 in every row here, so it carries no signal.
        assert "device_fraud_count" not in spec.feature_names
        assert "income" in spec.feature_names


@needs_artifacts
class TestBatchSingleRowEquivalence:
    """The regression test for silent training/serving skew."""

    def test_single_row_scoring_matches_batch(self) -> None:
        import joblib

        bundle = joblib.load(ARTIFACT)
        booster, spec = bundle["booster"], bundle["feature_spec"]

        frame = pd.read_parquet(DATA)
        sample = frame[frame["month"] == 7].sample(n=40, random_state=7)

        batch_features, _ = prepare(sample, spec)
        batch_scores = booster.predict(
            batch_features, num_iteration=booster.best_iteration
        )

        single_scores = []
        for position in range(len(sample)):
            row = sample.iloc[[position]]
            row_features, _ = prepare(row, spec)
            single_scores.append(
                booster.predict(row_features, num_iteration=booster.best_iteration)[0]
            )

        np.testing.assert_allclose(
            batch_scores, np.array(single_scores), rtol=1e-9, atol=1e-9,
            err_msg="single-row scoring diverges from batch scoring - "
                    "categorical codes are not pinned to the training levels",
        )

    def test_categorical_levels_survive_serialisation(self) -> None:
        import joblib

        spec = joblib.load(ARTIFACT)["feature_spec"]
        assert spec.categorical_levels, (
            "the persisted feature spec carries no categorical levels; "
            "serving cannot reproduce training codes"
        )
        for column in spec.categorical_features:
            assert len(spec.categorical_levels[column]) > 1
