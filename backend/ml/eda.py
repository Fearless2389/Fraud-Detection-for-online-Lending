"""Exploratory analysis of the BAF Base dataset.

Run once before any modelling. Its job is to replace assumptions with facts:
the exact schema, the true fraud prevalence, how the temporal split behaves,
and how missing values are encoded. Every one of those decisions feeds the
feature pipeline, and getting them from a paper rather than from the file is
how silent modelling bugs start.

Also caches the CSV to Parquet. Re-reading a 213 MB CSV on every experiment is
a tax paid dozens of times over a build.

    python ml/eda.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(os.getenv("DATA_DIR", "C:/dev/data/baf"))
BASE_CSV = DATA_DIR / "Base.csv"
BASE_PARQUET = DATA_DIR / "base.parquet"

TARGET = "fraud_bool"
TIME_COLUMN = "month"


def load_base() -> pd.DataFrame:
    """Load Base.csv, caching a Parquet copy for fast subsequent reads."""
    if BASE_PARQUET.exists():
        print(f"reading cached parquet: {BASE_PARQUET}")
        return pd.read_parquet(BASE_PARQUET)

    print(f"reading csv: {BASE_CSV}")
    frame = pd.read_csv(BASE_CSV)
    frame.to_parquet(BASE_PARQUET, index=False)
    print(f"cached to {BASE_PARQUET}")
    return frame


def describe_schema(frame: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print(f"SHAPE: {frame.shape[0]:,} rows x {frame.shape[1]} columns")
    print("=" * 78)

    numeric = frame.select_dtypes(include=[np.number]).columns.tolist()
    categorical = frame.select_dtypes(exclude=[np.number]).columns.tolist()

    print(f"\nNUMERIC ({len(numeric)}):")
    for name in numeric:
        print(f"  {name}")
    print(f"\nCATEGORICAL / OBJECT ({len(categorical)}):")
    for name in categorical:
        values = frame[name].unique()
        shown = ", ".join(map(str, values[:6]))
        suffix = " ..." if len(values) > 6 else ""
        print(f"  {name:<34} {len(values):>3} distinct: {shown}{suffix}")


def describe_target(frame: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("TARGET")
    print("=" * 78)
    counts = frame[TARGET].value_counts().sort_index()
    positives = int(counts.get(1, 0))
    total = len(frame)
    print(f"  genuine  : {int(counts.get(0, 0)):>9,}")
    print(f"  fraud    : {positives:>9,}")
    print(f"  prevalence: {positives / total:.4%}  (1 in {total / max(positives,1):.0f})")
    print(f"  -> accuracy of a 'never fraud' model: {1 - positives/total:.4%}")
    print("     (which is why accuracy is never reported in this project)")


def describe_temporal_split(frame: pd.DataFrame) -> None:
    """The drift experiment depends entirely on this behaving as expected."""
    print("\n" + "=" * 78)
    print("TEMPORAL STRUCTURE  (basis of the drift experiment)")
    print("=" * 78)
    if TIME_COLUMN not in frame.columns:
        print(f"  !! no '{TIME_COLUMN}' column - drift experiment needs rethinking")
        return

    grouped = frame.groupby(TIME_COLUMN)[TARGET].agg(["count", "sum", "mean"])
    grouped.columns = ["applications", "fraud", "fraud_rate"]
    print(f"\n{'month':>6} {'applications':>14} {'fraud':>8} {'fraud_rate':>12}")
    for month, row in grouped.iterrows():
        print(
            f"{month:>6} {int(row.applications):>14,} {int(row.fraud):>8,} "
            f"{row.fraud_rate:>11.3%}"
        )

    rates = grouped["fraud_rate"]
    print(
        f"\n  fraud rate range: {rates.min():.3%} -> {rates.max():.3%} "
        f"({rates.max() / rates.min():.2f}x drift across the window)"
    )


def describe_missingness(frame: pd.DataFrame) -> None:
    """BAF encodes 'not available' as negative sentinels, not NaN.

    Treating -1 as a real numeric value silently poisons any model that splits
    on those columns, so the pipeline must handle them explicitly.
    """
    print("\n" + "=" * 78)
    print("MISSINGNESS")
    print("=" * 78)

    nan_counts = frame.isna().sum()
    nan_columns = nan_counts[nan_counts > 0]
    print(f"  true NaN columns: {len(nan_columns)}")
    for name, count in nan_columns.items():
        print(f"    {name:<34} {count:>9,} ({count/len(frame):.2%})")

    print("\n  negative-sentinel columns (likely encoded missing):")
    numeric = frame.select_dtypes(include=[np.number]).columns
    for name in numeric:
        if name == TARGET:
            continue
        negatives = int((frame[name] < 0).sum())
        if negatives:
            distinct_negatives = sorted(frame.loc[frame[name] < 0, name].unique())[:4]
            print(
                f"    {name:<34} {negatives:>9,} ({negatives/len(frame):.2%})"
                f"  values: {distinct_negatives}"
            )


def describe_protected_attributes(frame: pd.DataFrame) -> None:
    """Fairness analysis needs these; confirm they are actually present."""
    print("\n" + "=" * 78)
    print("CANDIDATE PROTECTED ATTRIBUTES  (for the fairness analysis)")
    print("=" * 78)
    candidates = [c for c in frame.columns if any(
        token in c.lower() for token in ("age", "income", "employment", "housing")
    )]
    for name in candidates:
        series = frame[name]
        if series.dtype == object or series.nunique() <= 12:
            grouped = frame.groupby(name)[TARGET].agg(["count", "mean"])
            print(f"\n  {name}:")
            for value, row in grouped.iterrows():
                print(
                    f"    {str(value):<16} n={int(row['count']):>8,}  "
                    f"fraud_rate={row['mean']:.3%}"
                )
        else:
            print(f"\n  {name}: continuous, range {series.min()} - {series.max()}")


def main() -> None:
    frame = load_base()
    describe_schema(frame)
    describe_target(frame)
    describe_temporal_split(frame)
    describe_missingness(frame)
    describe_protected_attributes(frame)
    print("\ndone.\n")


if __name__ == "__main__":
    main()
