"""Train, calibrate, and price the fraud detection model.

Produces every number quoted in the submission. Run it and the results
regenerate from scratch:

    python ml/training/train.py

The pipeline, in order, and why each step is there:

1. **Temporal split.** Train on the past, judge on the future (``pipeline.py``).
2. **LightGBM.** Ranks applications by fraud risk. Class imbalance is handled
   with ``scale_pos_weight`` rather than by resampling: resampling distorts the
   base rate, and a distorted base rate destroys calibration - which the whole
   decision policy depends on.
3. **Isotonic calibration.** Converts raw scores into probabilities that mean
   what they claim. Gradient boosting ranks well but is not calibrated; a raw
   score of 0.3 does not mean a 30% chance of fraud. Since the thresholds are
   derived from expected cost, uncalibrated inputs would silently invalidate
   the entire decision layer.
4. **Cost-derived thresholds**, capped by analyst capacity (``policy.py``).
5. **Evaluation on untouched test months**, reported in both statistical terms
   (PR-AUC, recall at fixed alert rate) and business terms (rupees).

Accuracy is deliberately never reported: at 1.1% prevalence a model that
predicts "never fraud" achieves 98.9%, so the metric conveys nothing.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)

# Allow `python ml/training/train.py` from the backend directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.services.policy import (  # noqa: E402
    CostModel,
    constrain_review_capacity,
    derive_policy,
    evaluate_policy,
    naive_baseline_policy,
)
from ml.features.pipeline import (  # noqa: E402
    TARGET,
    FeatureSpec,
    prepare,
    temporal_split,
)

DATA_DIR = Path(os.getenv("DATA_DIR", "C:/dev/data/baf"))
ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"

# Business economics. Mirrors .env defaults; see policy.py for the derivation.
COSTS = CostModel(
    cost_fp=1_500.0,
    cost_fn=45_000.0,
    cost_review=200.0,
    analyst_catch_rate=0.90,
)
MAX_REVIEW_RATE = 0.05

LGB_PARAMS: dict[str, object] = {
    "objective": "binary",
    "metric": "average_precision",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "min_child_samples": 200,      # 1.1% positives: guards against leaves fitting noise
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "seed": 20260820,
    "num_threads": 0,
}
NUM_BOOST_ROUND = 2_000
EARLY_STOPPING_ROUNDS = 100


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 20
) -> float:
    """Mean gap between predicted probability and observed frequency.

    Reported before and after calibration. This is the number that justifies
    the cost-derived thresholds: without a low ECE, a threshold of 0.217 is
    an arbitrary cut rather than an economic boundary.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.digitize(probabilities, bins[1:-1], right=False)
    total_error = 0.0
    for bin_index in range(n_bins):
        mask = indices == bin_index
        if not mask.any():
            continue
        weight = mask.mean()
        total_error += weight * abs(probabilities[mask].mean() - y_true[mask].mean())
    return float(total_error)


def recall_at_alert_rate(
    y_true: np.ndarray, scores: np.ndarray, alert_rate: float
) -> float:
    """Share of fraud caught when only the riskiest ``alert_rate`` are actioned.

    This is the metric a fraud operations manager actually cares about: given
    a team that can work 3% of applications, how much fraud do they see?
    """
    cutoff = np.quantile(scores, 1.0 - alert_rate)
    flagged = scores >= cutoff
    positives = y_true.sum()
    return float((flagged & (y_true == 1)).sum() / positives) if positives else 0.0


def score_report(y_true: np.ndarray, probabilities: np.ndarray, label: str) -> dict[str, float]:
    report = {
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "ece": expected_calibration_error(y_true, probabilities),
        "recall_at_1pct": recall_at_alert_rate(y_true, probabilities, 0.01),
        "recall_at_3pct": recall_at_alert_rate(y_true, probabilities, 0.03),
        "recall_at_5pct": recall_at_alert_rate(y_true, probabilities, 0.05),
        "prevalence": float(y_true.mean()),
    }
    print(f"\n  {label}")
    print(f"    PR-AUC (average precision) : {report['pr_auc']:.4f}")
    print(f"    ROC-AUC                    : {report['roc_auc']:.4f}")
    print(f"    Brier score                : {report['brier']:.6f}")
    print(f"    Expected calibration error : {report['ece']:.5f}")
    print(f"    Recall @ 1% alert rate     : {report['recall_at_1pct']:.3%}")
    print(f"    Recall @ 3% alert rate     : {report['recall_at_3pct']:.3%}")
    print(f"    Recall @ 5% alert rate     : {report['recall_at_5pct']:.3%}")
    print(f"    (baseline prevalence       : {report['prevalence']:.3%})")
    return report


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------


def load_data() -> pd.DataFrame:
    parquet = DATA_DIR / "base.parquet"
    if parquet.exists():
        return pd.read_parquet(parquet)
    frame = pd.read_csv(DATA_DIR / "Base.csv")
    frame.to_parquet(parquet, index=False)
    return frame


def train_booster(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_validation: pd.DataFrame,
    y_validation: np.ndarray,
    spec: FeatureSpec,
) -> lgb.Booster:
    """Fit LightGBM with early stopping on the validation months."""
    negatives, positives = int((y_train == 0).sum()), int((y_train == 1).sum())
    params = dict(LGB_PARAMS)
    # Reweight rather than resample: preserves the true base rate, which the
    # calibration step and every downstream cost calculation depend on.
    params["scale_pos_weight"] = negatives / positives

    train_set = lgb.Dataset(
        x_train, label=y_train, categorical_feature=spec.categorical_features
    )
    validation_set = lgb.Dataset(
        x_validation, label=y_validation, reference=train_set,
        categorical_feature=spec.categorical_features,
    )

    print(f"\ntraining LightGBM on {len(x_train):,} rows "
          f"({positives:,} fraud, scale_pos_weight={params['scale_pos_weight']:.1f})")
    started = time.perf_counter()
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[validation_set],
        valid_names=["validation"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=200),
        ],
    )
    print(f"  stopped at iteration {booster.best_iteration} "
          f"in {time.perf_counter() - started:.1f}s")
    return booster


def fit_calibrator(raw_scores: np.ndarray, y_true: np.ndarray) -> IsotonicRegression:
    """Map raw model scores onto honest probabilities.

    Isotonic rather than Platt scaling: it makes no assumption about the shape
    of the distortion, and with ~119k calibration rows there is ample data to
    support the extra flexibility.
    """
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(raw_scores, y_true)
    return calibrator


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("AEGIS - training run")
    print("=" * 78)

    frame = load_data()
    split = temporal_split(frame)
    print("\nsplit summary:")
    print(split.summary().to_string(index=False))

    x_train, spec = prepare(split.train)
    y_train = split.train[TARGET].to_numpy()
    x_validation, _ = prepare(split.validation, spec)
    y_validation = split.validation[TARGET].to_numpy()
    x_calibration, _ = prepare(split.calibration, spec)
    y_calibration = split.calibration[TARGET].to_numpy()
    x_test, _ = prepare(split.test, spec)
    y_test = split.test[TARGET].to_numpy()

    print(f"\nfeatures: {len(spec.feature_names)} "
          f"({len(spec.categorical_features)} categorical, "
          f"{len(spec.behavioural_features)} behavioural)")

    booster = train_booster(x_train, y_train, x_validation, y_validation, spec)

    # --- calibration -----------------------------------------------------
    raw_calibration = booster.predict(x_calibration, num_iteration=booster.best_iteration)
    raw_test = booster.predict(x_test, num_iteration=booster.best_iteration)

    print("\n" + "-" * 78)
    print("CALIBRATION")
    print("-" * 78)
    uncalibrated_report = score_report(y_test, raw_test, "TEST - raw model scores")

    calibrator = fit_calibrator(raw_calibration, y_calibration)
    calibrated_calibration = calibrator.predict(raw_calibration)
    calibrated_test = calibrator.predict(raw_test)
    calibrated_report = score_report(y_test, calibrated_test, "TEST - calibrated probabilities")

    print(f"\n  calibration error reduced "
          f"{uncalibrated_report['ece']:.5f} -> {calibrated_report['ece']:.5f} "
          f"({1 - calibrated_report['ece']/max(uncalibrated_report['ece'], 1e-12):.1%} lower)")

    # --- decision policy --------------------------------------------------
    print("\n" + "-" * 78)
    print("DECISION POLICY")
    print("-" * 78)
    unconstrained = derive_policy(COSTS)
    policy = constrain_review_capacity(
        unconstrained, calibrated_calibration, max_review_rate=MAX_REVIEW_RATE
    )
    print(f"  cost model: FP=Rs{COSTS.cost_fp:,.0f}  FN=Rs{COSTS.cost_fn:,.0f}  "
          f"review=Rs{COSTS.cost_review:,.0f}  analyst catch rate={COSTS.analyst_catch_rate:.0%}")
    print(f"  derived thresholds        : review>={unconstrained.tau_review:.5f}  "
          f"block>={unconstrained.tau_block:.5f}")
    print(f"  after {MAX_REVIEW_RATE:.0%} capacity cap : review>={policy.tau_review:.5f}  "
          f"block>={policy.tau_block:.5f}")

    outcome = evaluate_policy(y_test, calibrated_test, policy)
    naive = evaluate_policy(y_test, calibrated_test, naive_baseline_policy(COSTS))

    print("\n  TEST months, cost-derived policy:")
    print(f"    approved {outcome.n_approved:,}  reviewed {outcome.n_reviewed:,} "
          f"({outcome.review_rate:.2%})  blocked {outcome.n_blocked:,}")
    print(f"    fraud missed        : {outcome.fraud_approved:,}")
    print(f"    genuine blocked (FP): {outcome.genuine_blocked:,} "
          f"({outcome.false_positive_rate:.3%} of genuine applicants)")
    print(f"    total cost          : Rs{outcome.total_cost:,.0f} "
          f"(Rs{outcome.cost_per_application:.2f}/application)")

    print("\n  TEST months, naive 0.5 threshold (the usual prototype):")
    print(f"    fraud missed        : {naive.fraud_approved:,}")
    print(f"    genuine blocked (FP): {naive.genuine_blocked:,} "
          f"({naive.false_positive_rate:.3%} of genuine applicants)")
    print(f"    total cost          : Rs{naive.total_cost:,.0f} "
          f"(Rs{naive.cost_per_application:.2f}/application)")

    saving = naive.total_cost - outcome.total_cost
    print(f"\n  >> cost reduction: Rs{saving:,.0f} "
          f"({saving / naive.total_cost:.1%}) on {outcome.n_total:,} applications")
    fp_change = naive.genuine_blocked - outcome.genuine_blocked
    print(f"  >> false positives: {naive.genuine_blocked:,} -> {outcome.genuine_blocked:,} "
          f"({fp_change:+,})")

    # --- feature importance ----------------------------------------------
    importance = pd.DataFrame({
        "feature": booster.feature_name(),
        "gain": booster.feature_importance("gain"),
    }).sort_values("gain", ascending=False)
    importance["share"] = importance["gain"] / importance["gain"].sum()

    behavioural_share = importance.loc[
        importance["feature"].isin(spec.behavioural_features), "share"
    ].sum()

    print("\n" + "-" * 78)
    print("FEATURE IMPORTANCE (top 15 by gain)")
    print("-" * 78)
    for _, row in importance.head(15).iterrows():
        marker = " [behavioural]" if row["feature"] in spec.behavioural_features else ""
        print(f"    {row['feature']:<38} {row['share']:>7.2%}{marker}")
    print(f"\n  behavioural features account for {behavioural_share:.1%} of total gain")

    # --- persist ----------------------------------------------------------
    bundle = {
        "booster": booster,
        "calibrator": calibrator,
        "feature_spec": spec,
        "policy": policy,
        "cost_model": COSTS,
    }
    joblib.dump(bundle, ARTIFACT_DIR / "model_bundle.joblib")

    metrics = {
        "trained_at_month_range": split.summary().to_dict(orient="records"),
        "n_features": len(spec.feature_names),
        "best_iteration": booster.best_iteration,
        "test_raw": uncalibrated_report,
        "test_calibrated": calibrated_report,
        "behavioural_gain_share": float(behavioural_share),
        "policy": {
            "tau_review": policy.tau_review,
            "tau_block": policy.tau_block,
            "tau_review_unconstrained": unconstrained.tau_review,
            "tau_block_unconstrained": unconstrained.tau_block,
            "max_review_rate": MAX_REVIEW_RATE,
        },
        "test_outcome_derived": {
            "approved": outcome.n_approved,
            "reviewed": outcome.n_reviewed,
            "blocked": outcome.n_blocked,
            "fraud_missed": outcome.fraud_approved,
            "genuine_blocked": outcome.genuine_blocked,
            "false_positive_rate": outcome.false_positive_rate,
            "total_cost": outcome.total_cost,
        },
        "test_outcome_naive": {
            "fraud_missed": naive.fraud_approved,
            "genuine_blocked": naive.genuine_blocked,
            "false_positive_rate": naive.false_positive_rate,
            "total_cost": naive.total_cost,
        },
        "top_features": importance.head(25).to_dict(orient="records"),
    }
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    print(f"\nartifacts written to {ARTIFACT_DIR}")
    print("  model_bundle.joblib   booster + calibrator + spec + policy")
    print("  metrics.json          every number quoted in the submission\n")


if __name__ == "__main__":
    main()
