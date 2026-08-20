"""Feature preparation for the BAF application-fraud dataset.

This module is the single definition of how raw application data becomes model
input. The API and the training pipeline both import it, which is what
guarantees that what is served matches what was trained - training/serving skew
is one of the most common and least visible production ML failures.

Three dataset-specific decisions are encoded here, each derived from the EDA in
``ml/eda.py`` rather than assumed:

1. **Sentinel missing values.** BAF has no NaNs. "Not available" is encoded as
   -1 in several columns, and one of them (``prev_address_months_count``) is
   71% sentinel. Left alone, a tree happily splits on "months at previous
   address < 0", learning a rule about *record-keeping* rather than about
   fraud. Each sentinel becomes a NaN plus an explicit missing indicator, so
   the model can use "this was unknown" as a signal - which it genuinely is -
   without treating -1 as a duration.

2. **Impossible negatives from differential privacy.** BAF was generated with
   DP noise added. That noise pushes a handful of values below zero on
   quantities that cannot be negative (velocity counters). These are cleaned to
   NaN rather than clipped to zero: a noisy unknown is not the same as a real
   zero, and pretending otherwise invents data.

3. **``intended_balcon_amount`` is not really continuous.** 74% of rows are
   negative across a continuous range, which reads as "no balance transfer
   intended", smeared by DP noise. It becomes a binary flag plus a cleaned
   amount, which is both more honest and more useful to the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TARGET = "fraud_bool"
TIME_COLUMN = "month"

# Columns where -1 means "unavailable" rather than a quantity.
SENTINEL_MISSING: dict[str, float] = {
    "prev_address_months_count": -1,
    "current_address_months_count": -1,
    "bank_months_count": -1,
    "device_distinct_emails_8w": -1,
    "session_length_in_minutes": -1.0,
}

# Quantities that are negative only because of differential-privacy noise.
IMPOSSIBLE_NEGATIVE: tuple[str, ...] = (
    "velocity_6h",
    "velocity_24h",
    "velocity_4w",
)

CATEGORICAL: tuple[str, ...] = (
    "payment_type",
    "employment_status",
    "housing_status",
    "source",
    "device_os",
)

# Behavioural and device signals, as distinct from static application details.
# Tracked as a group so the deck can state what share of the model's decisions
# rests on behaviour - which is what the brief actually asks about.
BEHAVIOURAL_FEATURES: tuple[str, ...] = (
    "velocity_6h",
    "velocity_24h",
    "velocity_4w",
    "session_length_in_minutes",
    "keep_alive_session",
    "device_os",
    "device_distinct_emails_8w",
    "device_fraud_count",
    "name_email_similarity",
    "email_is_free",
    "zip_count_4w",
    "bank_branch_count_8w",
    "date_of_birth_distinct_emails_4w",
    "days_since_request",
    "source",
    "foreign_request",
)

# Attributes used for fairness measurement. `customer_age` is both a strong
# predictor and a protected characteristic under fair-lending rules; that
# tension is measured and reported rather than quietly resolved.
PROTECTED_ATTRIBUTES: tuple[str, ...] = (
    "customer_age",
    "income",
    "employment_status",
    "housing_status",
)

# `device_fraud_count` is constant in BAF Base and carries no information;
# `month` is the time index, not a feature - including it would let the model
# learn "later months are riskier", which is exactly the drift we want to
# detect rather than memorise.
EXCLUDED_FROM_FEATURES: tuple[str, ...] = (TARGET, TIME_COLUMN)


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """The exact contract between training and serving."""

    feature_names: list[str]
    categorical_features: list[str]
    behavioural_features: list[str] = field(default_factory=list)
    # The category levels seen during training, in training order.
    #
    # This is load bearing, not bookkeeping. LightGBM identifies a pandas
    # categorical by its integer code, not its label. Calling
    # `.astype("category")` on a single-row scoring request derives the levels
    # from that one row, so `device_os="windows"` becomes code 0 - while in
    # training it may have been code 3. The model then reads a different
    # category than the one that was sent, produces a plausible-looking score,
    # and raises nothing. Pinning the levels here makes serving reproduce
    # training exactly.
    categorical_levels: dict[str, list] = field(default_factory=dict)

    def validate(self, frame: pd.DataFrame) -> None:
        """Fail loudly if a frame does not match the trained contract."""
        missing = [name for name in self.feature_names if name not in frame.columns]
        if missing:
            raise ValueError(f"frame is missing required features: {missing}")


def clean_raw(
    frame: pd.DataFrame, categorical_levels: dict[str, list] | None = None
) -> pd.DataFrame:
    """Apply the three dataset-specific corrections described in the module docstring.

    Args:
        frame: raw applications.
        categorical_levels: levels recorded at training time. Supplied when
            serving so category codes match training exactly; omitted during
            training, where the levels are inferred from the full dataset.
    """
    cleaned = frame.copy()

    # 1. Sentinel -1 -> NaN, with the fact of its absence preserved as a feature.
    for column, sentinel in SENTINEL_MISSING.items():
        if column not in cleaned.columns:
            continue
        is_missing = cleaned[column] == sentinel
        cleaned[f"{column}_missing"] = is_missing.astype("int8")
        cleaned.loc[is_missing, column] = np.nan

    # 2. DP noise producing impossible negatives -> NaN, not clipped to zero.
    for column in IMPOSSIBLE_NEGATIVE:
        if column not in cleaned.columns:
            continue
        cleaned.loc[cleaned[column] < 0, column] = np.nan

    # 3. intended_balcon_amount: a hidden binary wearing a continuous disguise.
    if "intended_balcon_amount" in cleaned.columns:
        has_intent = cleaned["intended_balcon_amount"] > 0
        cleaned["has_balcon_intent"] = has_intent.astype("int8")
        cleaned.loc[~has_intent, "intended_balcon_amount"] = np.nan

    # LightGBM consumes pandas categoricals natively - no one-hot explosion,
    # and split points stay interpretable in the SHAP output.
    for column in CATEGORICAL:
        if column not in cleaned.columns:
            continue
        if categorical_levels and column in categorical_levels:
            # Explicit levels: codes now match training regardless of which
            # values happen to appear in this batch. An unseen value becomes
            # NaN, which LightGBM handles as missing - the correct behaviour
            # for a category the model was never trained on.
            #
            # Unseen values are mapped to NaN *before* the Categorical is
            # built. Passing them straight to the constructor also yields NaN,
            # but pandas deprecates that path and will raise on it in a future
            # release; being explicit is both clearer and forward-compatible.
            levels = categorical_levels[column]
            values = cleaned[column]
            cleaned[column] = pd.Categorical(
                values.where(values.isin(levels)), categories=levels
            )
        else:
            cleaned[column] = cleaned[column].astype("category")

    return cleaned


def build_feature_spec(frame: pd.DataFrame) -> FeatureSpec:
    """Derive the feature contract from a cleaned frame.

    Constant columns are dropped: they cannot inform a split, and leaving them
    in produces misleading zero-importance rows in the explanation output.
    """
    candidates = [c for c in frame.columns if c not in EXCLUDED_FROM_FEATURES]

    informative = []
    for column in candidates:
        series = frame[column]
        distinct = series.nunique(dropna=True)
        if distinct <= 1:
            continue
        informative.append(column)

    categorical = [c for c in CATEGORICAL if c in informative]
    behavioural = [c for c in informative if c in BEHAVIOURAL_FEATURES]

    # Capture the levels in training order so serving can reproduce the codes.
    levels = {
        column: list(frame[column].cat.categories)
        for column in categorical
        if isinstance(frame[column].dtype, pd.CategoricalDtype)
    }

    return FeatureSpec(
        feature_names=informative,
        categorical_features=categorical,
        behavioural_features=behavioural,
        categorical_levels=levels,
    )


@dataclass(frozen=True, slots=True)
class TemporalSplit:
    """Four disjoint, time-ordered slices, each with exactly one job."""

    train: pd.DataFrame          # fit the booster
    validation: pd.DataFrame     # early stopping only
    calibration: pd.DataFrame    # fit the probability calibrator, set thresholds
    test: pd.DataFrame           # untouched until the final evaluation

    def summary(self) -> pd.DataFrame:
        rows = []
        for name in ("train", "validation", "calibration", "test"):
            part: pd.DataFrame = getattr(self, name)
            rows.append(
                {
                    "split": name,
                    "months": sorted(part[TIME_COLUMN].unique().tolist()),
                    "rows": len(part),
                    "fraud": int(part[TARGET].sum()),
                    "fraud_rate": float(part[TARGET].mean()),
                }
            )
        return pd.DataFrame(rows)


def temporal_split(
    frame: pd.DataFrame,
    train_months: tuple[int, ...] = (0, 1, 2, 3),
    validation_months: tuple[int, ...] = (4,),
    calibration_months: tuple[int, ...] = (5,),
    test_months: tuple[int, ...] = (6, 7),
) -> TemporalSplit:
    """Split by time, never randomly.

    A random split leaks the future into the past and inflates every metric,
    because fraud patterns evolve. Splitting on ``month`` measures what the
    system would actually have done: trained on what was known then, judged on
    what came next.

    Four slices rather than three, because two separate things need clean data
    and they must not be the same data:

    * **validation** stops training at the right number of trees. A model
      early-stopped on the calibration slice has effectively seen it, and the
      calibrator would then be fitted to scores tuned against itself.
    * **calibration** converts raw scores into probabilities that mean what
      they say, and sets the capacity-constrained thresholds. Fitting this on
      training data yields overconfident probabilities - and because the
      decision thresholds are derived from those probabilities, that error
      would propagate directly into the approve/block boundary.

    The test months are never touched until the final evaluation.
    """
    groups = {
        "train": train_months,
        "validation": validation_months,
        "calibration": calibration_months,
        "test": test_months,
    }
    seen: dict[int, str] = {}
    for name, months in groups.items():
        for month in months:
            if month in seen:
                raise ValueError(
                    f"month {month} appears in both '{seen[month]}' and '{name}' splits"
                )
            seen[month] = name

    return TemporalSplit(
        train=frame[frame[TIME_COLUMN].isin(train_months)].copy(),
        validation=frame[frame[TIME_COLUMN].isin(validation_months)].copy(),
        calibration=frame[frame[TIME_COLUMN].isin(calibration_months)].copy(),
        test=frame[frame[TIME_COLUMN].isin(test_months)].copy(),
    )


def prepare(frame: pd.DataFrame, spec: FeatureSpec | None = None) -> tuple[pd.DataFrame, FeatureSpec]:
    """Clean a raw frame and return model-ready features.

    At training time ``spec`` is None and the contract is derived. At serving
    time the trained spec is passed in, which both validates the incoming
    payload and guarantees identical column order.
    """
    if spec is None:
        cleaned = clean_raw(frame)
        spec = build_feature_spec(cleaned)
    else:
        cleaned = clean_raw(frame, categorical_levels=spec.categorical_levels)
        spec.validate(cleaned)
    return cleaned[spec.feature_names], spec
