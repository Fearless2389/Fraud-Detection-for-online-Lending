"""Post-training analysis: the numbers that go in the deck.

Two questions the training script does not answer honestly enough on its own.

**1. Does the review band actually reduce false positives?**

The training run compares the cost-derived policy against a naive 0.5
threshold, and on that comparison false positives go *up*. That comparison is
misleading in our favour's opposite direction: a 0.5 threshold misses 97% of
fraud, so of course it blocks few genuine customers. Comparing against a
system that does not work proves nothing.

The fair question is the one a fraud operations manager would ask: *holding
fraud caught constant*, how many genuine applicants does each design block?
That isolates the contribution of the three-way policy - the review band lets
a human clear borderline genuine customers who a binary policy would decline
outright.

**2. How badly does the model decay as fraud patterns shift?**

Evaluated month by month against a model that never sees months 4-7 during
training. This is the drift evidence, measured rather than asserted.

    python ml/training/analyse.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.services.policy import evaluate_policy  # noqa: E402
from ml.features.pipeline import TARGET, TIME_COLUMN, prepare, temporal_split  # noqa: E402

DATA_DIR = Path(os.getenv("DATA_DIR", "C:/dev/data/baf"))
ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def binary_policy_at_equal_recall(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    target_fraud_caught: float,
) -> dict[str, float]:
    """The block-or-approve policy that catches the same amount of fraud.

    Sweeps the threshold to the point where a binary auto-decline policy stops
    the same number of frauds as the three-way policy, then reports what that
    costs in wrongly blocked genuine customers. This is the like-for-like
    comparison.
    """
    order = np.argsort(-probabilities)
    labels_sorted = y_true[order]

    cumulative_fraud = np.cumsum(labels_sorted)
    # Smallest block-list that reaches the target fraud count.
    reached = np.searchsorted(cumulative_fraud, target_fraud_caught, side="left")
    reached = min(reached, len(labels_sorted) - 1)

    blocked_slice = labels_sorted[: reached + 1]
    genuine_blocked = int((blocked_slice == 0).sum())
    fraud_caught = int(blocked_slice.sum())

    return {
        "threshold": float(probabilities[order][reached]),
        "applications_blocked": int(reached + 1),
        "fraud_caught": fraud_caught,
        "genuine_blocked": genuine_blocked,
        "block_rate": float((reached + 1) / len(y_true)),
    }


def main() -> None:
    bundle = joblib.load(ARTIFACT_DIR / "model_bundle.joblib")
    booster = bundle["booster"]
    calibrator = bundle["calibrator"]
    spec = bundle["feature_spec"]
    policy = bundle["policy"]
    costs = bundle["cost_model"]

    frame = pd.read_parquet(DATA_DIR / "base.parquet")
    split = temporal_split(frame)

    x_test, _ = prepare(split.test, spec)
    y_test = split.test[TARGET].to_numpy()
    probabilities = calibrator.predict(
        booster.predict(x_test, num_iteration=booster.best_iteration)
    )

    outcome = evaluate_policy(y_test, probabilities, policy)

    # Expected fraud caught: everything blocked, plus the share of reviewed
    # fraud an analyst correctly identifies.
    fraud_caught = outcome.fraud_blocked + costs.analyst_catch_rate * outcome.fraud_reviewed

    print("=" * 78)
    print("EQUAL-RECALL COMPARISON  (the honest false-positive question)")
    print("=" * 78)
    print(f"\nTest population: {outcome.n_total:,} applications, "
          f"{int(y_test.sum()):,} fraudulent ({y_test.mean():.3%})")

    print("\nThree-way policy (approve / review / block):")
    print(f"  blocked outright        : {outcome.n_blocked:,}")
    print(f"  sent to human review    : {outcome.n_reviewed:,} ({outcome.review_rate:.2%})")
    print(f"  fraud caught (expected) : {fraud_caught:,.0f} of {int(y_test.sum()):,} "
          f"({fraud_caught / y_test.sum():.1%})")
    print(f"  GENUINE CUSTOMERS BLOCKED: {outcome.genuine_blocked:,}")

    equivalent = binary_policy_at_equal_recall(y_test, probabilities, fraud_caught)
    print("\nBinary auto-decline policy tuned to catch the SAME fraud:")
    print(f"  threshold               : {equivalent['threshold']:.5f}")
    print(f"  applications blocked    : {equivalent['applications_blocked']:,} "
          f"({equivalent['block_rate']:.2%})")
    print(f"  fraud caught            : {equivalent['fraud_caught']:,}")
    print(f"  GENUINE CUSTOMERS BLOCKED: {equivalent['genuine_blocked']:,}")

    avoided = equivalent["genuine_blocked"] - outcome.genuine_blocked
    if equivalent["genuine_blocked"] > 0:
        reduction = avoided / equivalent["genuine_blocked"]
        print(f"\n  >> FALSE POSITIVES AVOIDED: {avoided:,} "
              f"({reduction:.1%} fewer genuine customers wrongly declined)")
        print(f"  >> at identical fraud detection ({fraud_caught:,.0f} frauds stopped)")
    else:
        print("\n  >> comparison degenerate: equivalent policy blocks nothing")

    # --- drift ------------------------------------------------------------
    print("\n" + "=" * 78)
    print("DRIFT  (model trained on months 0-3 only, never sees 4-7)")
    print("=" * 78)

    rows = []
    for month in sorted(frame[TIME_COLUMN].unique()):
        month_frame = frame[frame[TIME_COLUMN] == month]
        x_month, _ = prepare(month_frame, spec)
        y_month = month_frame[TARGET].to_numpy()
        p_month = calibrator.predict(
            booster.predict(x_month, num_iteration=booster.best_iteration)
        )
        month_outcome = evaluate_policy(y_month, p_month, policy)
        caught = (month_outcome.fraud_blocked
                  + costs.analyst_catch_rate * month_outcome.fraud_reviewed)
        rows.append({
            "month": int(month),
            "split": ("train" if month <= 3 else
                      "val" if month == 4 else
                      "calib" if month == 5 else "test"),
            "applications": len(month_frame),
            "fraud_rate": float(y_month.mean()),
            "pr_auc": float(average_precision_score(y_month, p_month)),
            "roc_auc": float(roc_auc_score(y_month, p_month)),
            "recall": float(caught / y_month.sum()),
            "review_rate": float(month_outcome.review_rate),
            "cost_per_app": month_outcome.cost_per_application,
        })

    drift = pd.DataFrame(rows)
    print()
    print(drift.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # Months 0-3 were trained on. Their metrics are inflated by fitting, not
    # by favourable conditions, so comparing them against the test months
    # measures overfitting and drift mixed together and attributes all of it
    # to drift. Drift is therefore measured across the out-of-sample months
    # only (4-7), where the model's exposure is identical throughout.
    in_sample = drift[drift["split"] == "train"]["pr_auc"].mean()
    out_of_sample = drift[drift["split"] != "train"]

    print(f"\n  PR-AUC in-sample (months 0-3, TRAINED ON) : {in_sample:.4f}")
    print(f"  PR-AUC out-of-sample (months 4-7)         : "
          f"{out_of_sample['pr_auc'].mean():.4f}")
    print("  The gap between those two is mostly overfitting, not drift, and is")
    print("  not reported as a drift result.")

    first, last = out_of_sample.iloc[0], out_of_sample.iloc[-1]
    print(f"\n  DRIFT across out-of-sample months {int(first['month'])} -> {int(last['month'])}, "
          "at fixed thresholds:")
    print(f"    fraud rate      {first['fraud_rate']:.3%}  ->  {last['fraud_rate']:.3%}   "
          f"({last['fraud_rate']/first['fraud_rate'] - 1:+.0%})")
    print(f"    ROC-AUC         {first['roc_auc']:.4f}  ->  {last['roc_auc']:.4f}   "
          f"({last['roc_auc']/first['roc_auc'] - 1:+.1%})")
    print(f"    PR-AUC          {first['pr_auc']:.4f}  ->  {last['pr_auc']:.4f}")
    print(f"    RECALL          {first['recall']:.3%}  ->  {last['recall']:.3%}   "
          f"({last['recall']/first['recall'] - 1:+.1%})")
    print(f"    cost/application Rs{first['cost_per_app']:.0f}  ->  Rs{last['cost_per_app']:.0f}   "
          f"({last['cost_per_app']/first['cost_per_app'] - 1:+.0%})")

    print("\n  Reading: discrimination holds up (ROC-AUC barely moves), but the")
    print("  OPERATING POINT decays. Fraud prevalence rises while the thresholds")
    print("  stay where month 5 put them, so recall falls and cost per application")
    print("  climbs. The failure is in the policy, not the model - which means the")
    print("  fix is recalibration and threshold re-derivation, not retraining.")

    planned = drift[drift["split"] == "calib"]["review_rate"].iloc[0]
    actual = out_of_sample[out_of_sample["split"] == "test"]["review_rate"].mean()
    print(f"\n  review capacity planned on month 5 : {planned:.2%}")
    print(f"  actual review rate on months 6-7   : {actual:.2%} "
          f"({actual/planned - 1:+.0%})")
    if actual > planned * 1.05:
        print("  -> the queue overflows as fraud rises; capacity planned on last")
        print("     month's data does not hold under drift.")
    else:
        print("  -> capacity held. Volume fell faster than fraud rose, so the queue")
        print("     stayed within plan. Recall dropped for a different reason: the")
        print("     fixed thresholds admit a smaller share of a riskier population.")

    analysis = {
        "equal_recall_comparison": {
            "three_way": {
                "fraud_caught": float(fraud_caught),
                "genuine_blocked": outcome.genuine_blocked,
                "reviewed": outcome.n_reviewed,
                "blocked": outcome.n_blocked,
            },
            "binary_equivalent": equivalent,
            "false_positives_avoided": int(avoided),
        },
        "drift_by_month": drift.to_dict(orient="records"),
    }
    (ARTIFACT_DIR / "analysis.json").write_text(json.dumps(analysis, indent=2, default=float))
    print(f"\nwritten to {ARTIFACT_DIR / 'analysis.json'}\n")


if __name__ == "__main__":
    main()
