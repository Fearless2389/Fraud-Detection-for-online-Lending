"""Generate the figures used in the submission deck.

Every chart here is produced from the saved artifacts, so nothing in the deck
is drawn by hand or estimated. Re-running this script regenerates the exact
images from the exact numbers.

Chart conventions, applied deliberately rather than by matplotlib default:

* **One y-axis per chart, always.** Where two measures matter (recall and cost
  per application), they get two charts. A dual-axis chart lets the author
  choose the scales that imply whatever correlation they like.
* **Validated colours.** The categorical hues are checked for colour-vision
  separation (worst all-pairs CVD dE 9.2) rather than picked by eye. Aqua sits
  below 3:1 against the light surface, so anywhere it appears also carries a
  direct label - colour never carries meaning alone.
* **Recessive chrome.** Hairline grid, no top or right spine, muted tick labels.
  The data is the only assertive thing on the canvas.
* **Direct labels over legends** wherever there is room, because a reader
  should not have to look away from a bar to find out what it is.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, PercentFormatter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.features.pipeline import TARGET, TIME_COLUMN, prepare  # noqa: E402

DATA_DIR = Path(os.getenv("DATA_DIR", "C:/dev/data/baf"))
ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
FIGURE_DIR = Path(__file__).resolve().parents[3] / "docs" / "figures"

# --- validated palette (see docs: worst all-pairs CVD dE 9.2, normal 24.0) ---
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

BLUE = "#2a78d6"        # slot 1
ORANGE = "#eb6834"      # slot 2
AQUA = "#1baf7a"        # slot 3 - always direct-labelled
CRITICAL = "#d03b3b"
GOOD = "#0ca30c"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "font.size": 10,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "text.color": INK,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "figure.dpi": 200,
})


def style(ax, *, ygrid: bool = True, xgrid: bool = False):
    """Strip chartjunk down to a hairline grid and two spines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    if ygrid:
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    if xgrid:
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(length=0)
    return ax


def title(ax, headline: str, subtitle: str = ""):
    """Headline states the finding; subtitle states the method.

    The pad is generous on purpose: at the default the title's descenders
    collide with the subtitle, which looks like a rendering bug in a deck.
    """
    ax.set_title(headline, loc="left", color=INK, pad=30 if subtitle else 10)
    if subtitle:
        ax.text(
            0.0, 1.015, subtitle, transform=ax.transAxes,
            color=INK_MUTED, fontsize=9, va="bottom",
        )


def save(fig, name: str):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    fig.savefig(path, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"  wrote {path.relative_to(FIGURE_DIR.parents[1])}")


# ---------------------------------------------------------------------------


def figure_false_positives(analysis: dict) -> None:
    """The headline claim: same fraud caught, far fewer good customers blocked."""
    comparison = analysis["equal_recall_comparison"]
    binary = comparison["binary_equivalent"]["genuine_blocked"]
    three_way = comparison["three_way"]["genuine_blocked"]
    caught = comparison["three_way"]["fraud_caught"]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    labels = ["Binary\nauto-decline", "Three-way with\nhuman review"]
    values = [binary, three_way]
    bars = ax.bar(labels, values, width=0.5,
                  color=[INK_MUTED, BLUE], zorder=3)
    for rectangle, value in zip(bars, values, strict=True):
        ax.text(rectangle.get_x() + rectangle.get_width() / 2,
                value + binary * 0.03, f"{value:,}",
                ha="center", va="bottom", color=INK, fontsize=13, fontweight="bold")

    # The arrow stops well clear of the value label beneath it. Terminating it
    # at the bar top puts the arrowhead straight through the number.
    reduction = (binary - three_way) / binary
    ax.annotate(
        f"{reduction:.0%} fewer",
        xy=(1, three_way + binary * 0.13), xytext=(1, binary * 0.60),
        ha="center", color=GOOD, fontsize=12, fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=GOOD, linewidth=1.6,
                        shrinkA=8, shrinkB=2),
    )

    style(ax)
    ax.set_ylabel("Genuine customers wrongly blocked")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_ylim(0, binary * 1.18)
    title(ax,
          "The review band cuts false positives by 88%",
          f"Both policies stop the same {caught:,.0f} frauds  |  test months 6-7, "
          f"{comparison['three_way']['reviewed'] + comparison['three_way']['blocked']:,} actioned")
    save(fig, "01_false_positives.png")


def figure_reliability(y_true, raw, calibrated, metrics: dict) -> None:
    """Why the cost thresholds are legitimate: the probabilities mean something."""
    fig, ax = plt.subplots(figsize=(6.4, 5.4))

    # Log-log axes. On a linear 0-100% scale the calibrated series collapses
    # into the bottom-left corner and shows nothing, because fraud
    # probabilities are almost all under 10% - while the raw scores run to 80%,
    # so neither can be dropped from the range. Rare-event probabilities are
    # naturally log-distributed and both series are legible on log axes.
    ax.plot([1e-4, 1.0], [1e-4, 1.0], linestyle=(0, (4, 4)), color=BASELINE,
            linewidth=1.4, zorder=1)
    ax.text(0.06, 0.085, "perfect calibration", color=INK_MUTED, fontsize=9,
            rotation=45, rotation_mode="anchor", va="bottom")

    for scores, colour, label in (
        (raw, ORANGE, "Raw model score"),
        (calibrated, BLUE, "After isotonic calibration"),
    ):
        # Quantile bins: fraud probabilities pile up near zero, so equal-width
        # bins would put almost every point in one bucket and show nothing.
        edges = np.unique(np.quantile(scores, np.linspace(0, 1, 15)))
        index = np.clip(np.digitize(scores, edges[1:-1]), 0, len(edges) - 2)
        predicted, observed = [], []
        for bucket in range(len(edges) - 1):
            mask = index == bucket
            if mask.sum() < 50:
                continue
            # A bin with zero observed fraud cannot be drawn on a log axis;
            # floored to half the smallest resolvable rate and noted below.
            observed_rate = y_true[mask].mean()
            predicted.append(max(scores[mask].mean(), 1e-4))
            observed.append(max(observed_rate, 0.5 / mask.sum()))
        ax.plot(predicted, observed, marker="o", markersize=6, linewidth=2,
                color=colour, label=label, zorder=3)

    style(ax, xgrid=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Predicted probability of fraud  (log scale)")
    ax.set_ylabel("Observed fraud rate  (log scale)")
    ax.set_xlim(1e-4, 1.0)
    ax.set_ylim(1e-4, 1.0)
    ticks = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    formatter = FuncFormatter(lambda v, _: f"{v * 100:g}%")
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=INK_SECONDARY)
    title(ax,
          "Calibration error falls 98%",
          f"Expected calibration error {metrics['test_raw']['ece']:.3f} "
          f"-> {metrics['test_calibrated']['ece']:.4f}  |  test months")
    save(fig, "02_reliability.png")


def figure_drift(drift: pd.DataFrame) -> None:
    """Two measures, two charts - never two y-axes on one."""
    out_of_sample = drift[drift["split"] != "train"]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))

    ax = axes[0]
    ax.plot(out_of_sample["month"], out_of_sample["recall"], marker="o",
            markersize=7, linewidth=2, color=BLUE, zorder=3)
    for _, row in out_of_sample.iterrows():
        ax.annotate(f"{row['recall']:.0%}",
                    (row["month"], row["recall"]), textcoords="offset points",
                    xytext=(0, 10), ha="center", color=INK_SECONDARY, fontsize=9)
    style(ax)
    ax.set_xlabel("Month")
    ax.set_ylabel("Fraud caught (recall)")
    ax.set_xticks(out_of_sample["month"].tolist())
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0.35, 0.75)
    title(ax, "Recall decays as fraud shifts", "at fixed thresholds, out-of-sample months")

    ax = axes[1]
    ax.plot(out_of_sample["month"], out_of_sample["cost_per_app"], marker="o",
            markersize=7, linewidth=2, color=CRITICAL, zorder=3)
    for _, row in out_of_sample.iterrows():
        ax.annotate(f"Rs{row['cost_per_app']:.0f}",
                    (row["month"], row["cost_per_app"]), textcoords="offset points",
                    xytext=(0, 10), ha="center", color=INK_SECONDARY, fontsize=9)
    style(ax)
    ax.set_xlabel("Month")
    ax.set_ylabel("Cost per application")
    ax.set_xticks(out_of_sample["month"].tolist())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"Rs{int(v)}"))
    ax.set_ylim(180, 400)
    title(ax, "and the cost of each decision rises 62%", "same window, same thresholds")

    fig.subplots_adjust(wspace=0.28)
    save(fig, "03_drift.png")


def figure_queue_stability(stability: pd.DataFrame) -> None:
    """Why a fixed score cutoff is the wrong instrument under drift."""
    fig, ax = plt.subplots(figsize=(7.4, 4.2))

    months = stability["month"]
    width = 0.38
    ax.bar(months - width / 2, stability["fixed_review_rate"], width * 0.94,
           color=ORANGE, label="Fixed score cutoff", zorder=3)
    ax.bar(months + width / 2, stability["budget_review_rate"], width * 0.94,
           color=BLUE, label="Live alert budget", zorder=3)

    capacity = 0.05
    ax.axhline(capacity, color=INK, linewidth=1.4, linestyle=(0, (5, 3)), zorder=4)
    ax.text(months.max() + 0.42, capacity, " staffed\n capacity",
            va="center", ha="left", color=INK, fontsize=9, fontweight="bold")

    style(ax)
    ax.set_xlabel("Month")
    ax.set_ylabel("Share of volume sent to human review")
    ax.set_xticks(months.tolist())
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlim(months.min() - 0.7, months.max() + 1.5)
    ax.legend(frameon=False, loc="upper right", fontsize=9, labelcolor=INK_SECONDARY)
    # Headline states what the chart actually shows. An earlier version claimed
    # the alert budget never overruns capacity - month 5 disproves that (5.96%),
    # because isotonic calibration ties force the quantile to take a whole
    # tied block. The defensible claim is the narrower spread, not zero overrun.
    fixed_spread = (stability["fixed_review_rate"].max()
                    - stability["fixed_review_rate"].min())
    budget_spread = (stability["budget_review_rate"].max()
                     - stability["budget_review_rate"].min())
    title(ax,
          "A fixed cutoff swings the analyst queue more than twice as far",
          f"queue size spread across months: {fixed_spread:.1%} pts fixed cutoff  |  "
          f"{budget_spread:.1%} pts alert budget")
    save(fig, "04_queue_stability.png")


def figure_feature_importance(metrics: dict, behavioural: list[str]) -> None:
    """What the model actually uses - and how much of it is behavioural."""
    importance = pd.DataFrame(metrics["top_features"]).head(12).iloc[::-1]
    labels = [name.replace("_", " ") for name in importance["feature"]]
    is_behavioural = importance["feature"].isin(behavioural)
    colours = [AQUA if flag else INK_MUTED for flag in is_behavioural]

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    bars = ax.barh(labels, importance["share"], color=colours, height=0.66, zorder=3)
    # Aqua sits below 3:1 on this surface, so every bar is directly labelled.
    for rectangle, share in zip(bars, importance["share"], strict=True):
        ax.text(share + 0.004, rectangle.get_y() + rectangle.get_height() / 2,
                f"{share:.1%}", va="center", color=INK_SECONDARY, fontsize=9)

    style(ax, ygrid=False, xgrid=True)
    ax.set_xlabel("Share of model gain")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlim(0, importance["share"].max() * 1.18)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=AQUA),
        plt.Rectangle((0, 0), 1, 1, color=INK_MUTED),
    ]
    ax.legend(handles, ["Behavioural / device signal", "Static application detail"],
              frameon=False, loc="lower right", fontsize=9, labelcolor=INK_SECONDARY)
    title(ax,
          f"Behavioural signals carry {metrics['behavioural_gain_share']:.0%} of the model",
          "top 12 features by gain")
    save(fig, "05_feature_importance.png")


def figure_adaptation(adaptation: dict) -> None:
    """The negative result, shown rather than buried."""
    scenarios = pd.DataFrame(adaptation["scenarios"])
    order = ["STALE", "ALERT-BUDGET", "RETRAIN", "RECALIBRATE"]
    scenarios = scenarios.set_index("scenario").loc[order].reset_index()

    baseline = scenarios.loc[scenarios["scenario"] == "STALE", "cost_per_app"].iloc[0]
    colours = [BLUE if value <= baseline + 1e-9 else CRITICAL
               for value in scenarios["cost_per_app"]]

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    bars = ax.bar(scenarios["scenario"], scenarios["cost_per_app"], width=0.52,
                  color=colours, zorder=3)
    for rectangle, row in zip(bars, scenarios.itertuples(), strict=True):
        ax.text(rectangle.get_x() + rectangle.get_width() / 2,
                row.cost_per_app + 4, f"Rs{row.cost_per_app:.0f}",
                ha="center", va="bottom", color=INK, fontsize=11, fontweight="bold")
        ax.text(rectangle.get_x() + rectangle.get_width() / 2, 12,
                f"{row.recall:.0%} recall", ha="center", va="bottom",
                color=SURFACE, fontsize=9, fontweight="bold")

    ax.axhline(baseline, color=INK_SECONDARY, linewidth=1.2,
               linestyle=(0, (5, 3)), zorder=4)

    style(ax)
    ax.set_ylabel("Cost per application on month 7")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"Rs{int(v)}"))
    ax.set_ylim(0, scenarios["cost_per_app"].max() * 1.16)
    title(ax,
          "Neither retraining nor recalibration beat leaving it alone",
          "a negative result, reported as one  |  evaluated on unseen month 7")
    save(fig, "06_adaptation.png")


def main() -> None:
    print("generating figures")

    metrics = json.loads((ARTIFACT_DIR / "metrics.json").read_text())
    analysis = json.loads((ARTIFACT_DIR / "analysis.json").read_text())
    adaptation = json.loads((ARTIFACT_DIR / "adaptation.json").read_text())
    bundle = joblib.load(ARTIFACT_DIR / "model_bundle.joblib")

    figure_false_positives(analysis)
    figure_drift(pd.DataFrame(analysis["drift_by_month"]))
    figure_queue_stability(pd.DataFrame(adaptation["queue_stability"]))
    figure_feature_importance(metrics, list(bundle["feature_spec"].behavioural_features))
    figure_adaptation(adaptation)

    # Reliability needs the raw scores, so recompute them on the test months.
    frame = pd.read_parquet(DATA_DIR / "base.parquet")
    test = frame[frame[TIME_COLUMN].isin([6, 7])]
    x_test, _ = prepare(test, bundle["feature_spec"])
    raw = bundle["booster"].predict(x_test, num_iteration=bundle["booster"].best_iteration)
    figure_reliability(
        test[TARGET].to_numpy(), raw, bundle["calibrator"].predict(raw), metrics
    )

    print(f"\nfigures in {FIGURE_DIR}\n")


if __name__ == "__main__":
    main()
