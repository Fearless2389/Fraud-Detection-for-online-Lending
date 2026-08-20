"""A replayable stream of real applications, for the operations dashboard.

The dashboard needs a live feed of applications arriving to be scored. Rather
than invent synthetic traffic, this replays genuine held-out applications from
the test months - the same data the reported metrics were measured on, so what
a viewer sees on screen is consistent with what the results claim.

Two things about this module are demo-only and marked as such:

* it exposes the true fraud label, so the interface can show whether a decision
  was ultimately right. A production scoring path never has that at decision
  time, and nothing in the scoring service reads it;
* it samples with a fixed seed so a rehearsed demo and the recorded one show
  the same cases.

Sampling is stratified to over-represent fraud. At the natural 1.4% rate a
sixty-second demo would show roughly one fraudulent application, which
demonstrates nothing. The true rate is stated on screen so the stratification
is never mistaken for the real base rate.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.getenv("DATA_DIR", "C:/dev/data/baf"))
TEST_MONTHS = (6, 7)

# Enough variety that a demo never visibly loops, small enough to hold in memory.
SAMPLE_SIZE = 3_000
FRAUD_SHARE = 0.20
SEED = 20260820

# One fraudulent application every N positions. Random shuffling at a 20% rate
# still leaves an 11% chance that the first eleven applications contain no
# fraud at all, which would mean a recorded demo of a fraud platform in which
# nothing is ever caught. Deterministic spacing removes that risk: the cadence
# is fixed, the ordering is reproducible, and the true prevalence is disclosed
# alongside every response so the stratification is never mistaken for reality.
FRAUD_EVERY = 5


@lru_cache(maxsize=1)
def load_demo_pool() -> pd.DataFrame:
    """Load and cache a stratified sample of held-out applications."""
    parquet = DATA_DIR / "base.parquet"
    if not parquet.exists():
        logger.warning("demo pool unavailable: %s not found", parquet)
        return pd.DataFrame()

    frame = pd.read_parquet(parquet)
    held_out = frame[frame["month"].isin(TEST_MONTHS)]

    n_fraud = int(SAMPLE_SIZE * FRAUD_SHARE)
    n_genuine = SAMPLE_SIZE - n_fraud

    fraud = held_out[held_out["fraud_bool"] == 1].sample(
        n=min(n_fraud, int((held_out["fraud_bool"] == 1).sum())), random_state=SEED
    )
    genuine = held_out[held_out["fraud_bool"] == 0].sample(
        n=min(n_genuine, int((held_out["fraud_bool"] == 0).sum())), random_state=SEED
    )

    # Interleave rather than shuffle, so a fraudulent application lands on every
    # FRAUD_EVERY-th position. Both groups are themselves randomly sampled, so
    # which fraud appears is not cherry-picked - only the spacing is fixed.
    fraud_rows = [row for _, row in fraud.iterrows()]
    genuine_rows = [row for _, row in genuine.iterrows()]

    ordered: list[pd.Series] = []
    fraud_index = genuine_index = 0
    while fraud_index < len(fraud_rows) or genuine_index < len(genuine_rows):
        take_fraud = (len(ordered) % FRAUD_EVERY == FRAUD_EVERY - 1)
        if take_fraud and fraud_index < len(fraud_rows):
            ordered.append(fraud_rows[fraud_index])
            fraud_index += 1
        elif genuine_index < len(genuine_rows):
            ordered.append(genuine_rows[genuine_index])
            genuine_index += 1
        elif fraud_index < len(fraud_rows):
            ordered.append(fraud_rows[fraud_index])
            fraud_index += 1
        else:
            break

    pool = pd.DataFrame(ordered).reset_index(drop=True)

    logger.info(
        "demo pool ready: %d applications, %.1f%% fraud spaced every %d "
        "(true rate in these months: %.2f%%)",
        len(pool), 100 * pool["fraud_bool"].mean(), FRAUD_EVERY,
        100 * held_out["fraud_bool"].mean(),
    )
    return pool


def take(count: int, offset: int = 0) -> list[dict]:
    """Return applications from the pool, wrapping around at the end."""
    pool = load_demo_pool()
    if pool.empty:
        return []

    indices = [(offset + step) % len(pool) for step in range(count)]
    rows = pool.iloc[indices]

    payloads = []
    for _, row in rows.iterrows():
        record = row.to_dict()
        # `month` is the split index and `fraud_bool` is the answer; neither is
        # an input to scoring. The label is returned separately, clearly named.
        actual_fraud = bool(record.pop("fraud_bool"))
        record.pop("month", None)
        payloads.append({
            "application": {
                key: (int(value) if isinstance(value, bool) else value)
                for key, value in record.items()
            },
            "actual_fraud": actual_fraud,
        })
    return payloads


def true_fraud_rate() -> float:
    """The real prevalence in the held-out months, for on-screen disclosure."""
    parquet = DATA_DIR / "base.parquet"
    if not parquet.exists():
        return 0.0
    frame = pd.read_parquet(parquet, columns=["month", "fraud_bool"])
    return float(frame[frame["month"].isin(TEST_MONTHS)]["fraud_bool"].mean())
