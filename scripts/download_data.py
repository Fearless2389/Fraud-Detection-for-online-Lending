"""Download the Bank Account Fraud (BAF) dataset suite from Kaggle.

The BAF CSVs are never committed to this repository: they are large, and the
dataset is licensed CC BY-NC-ND 4.0 (academic / non-commercial, no derivatives).
Shipping this loader instead of the data is both licence-correct and keeps the
repository small.

Destination is read from the DATA_DIR environment variable so that the data can
live outside OneDrive-synced folders. Syncing a ~1 GB dataset causes file locks
and quota burn, and the data is reproducible from this script anyway.

Prerequisites
-------------
1. A Kaggle account.
2. An API token: kaggle.com -> Settings -> API -> "Create New Token".
   This downloads ``kaggle.json``. Place it at::

       %USERPROFILE%\\.kaggle\\kaggle.json

Usage
-----
    python scripts/download_data.py
    python scripts/download_data.py --dest D:/data/baf
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DATASET_SLUG = "sgpjesus/bank-account-fraud-dataset-neurips-2022"
DEFAULT_DEST = Path(os.getenv("DATA_DIR", "C:/dev/data/baf"))

# The suite ships one CSV per variant. "Base" is the primary training dataset;
# the numbered variants carry different injected biases and are used for the
# fairness comparison. Filenames are verified against the download rather than
# assumed, so this list is informational only.
EXPECTED_VARIANTS = ("Base", "Variant I", "Variant II", "Variant III", "Variant IV", "Variant V")


def _credentials_present() -> bool:
    """True if the Kaggle client will be able to authenticate."""
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        return True
    return (Path.home() / ".kaggle" / "kaggle.json").exists()


def _explain_missing_credentials() -> None:
    print(
        "\n".join(
            [
                "",
                "Kaggle credentials not found.",
                "",
                "  1. Sign in at https://www.kaggle.com",
                "  2. Go to Settings -> API -> 'Create New Token'",
                "  3. Save the downloaded kaggle.json to:",
                f"       {Path.home() / '.kaggle' / 'kaggle.json'}",
                "",
                "  You must also open the dataset page once and accept its terms:",
                f"       https://www.kaggle.com/datasets/{DATASET_SLUG}",
                "",
            ]
        ),
        file=sys.stderr,
    )


def download(dest: Path) -> int:
    if not _credentials_present():
        _explain_missing_credentials()
        return 1

    # Imported late: the kaggle package authenticates at import time and raises
    # if credentials are absent, which would bypass the friendly message above.
    from kaggle.api.kaggle_api_extended import KaggleApi

    dest.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    print(f"Downloading {DATASET_SLUG}")
    print(f"  destination: {dest}")
    print("  this is a multi-hundred-MB download; unzipping happens automatically.")
    api.dataset_download_files(DATASET_SLUG, path=str(dest), unzip=True, quiet=False)

    csvs = sorted(dest.glob("*.csv"))
    if not csvs:
        print("Download reported success but no CSV files were found.", file=sys.stderr)
        return 1

    print(f"\nDone. {len(csvs)} file(s) in {dest}:")
    for path in csvs:
        print(f"  {path.name:<24} {path.stat().st_size / 1e6:>8.1f} MB")

    missing = [v for v in EXPECTED_VARIANTS if not any(v in c.name for c in csvs)]
    if missing:
        print(f"\nNote: expected variants not matched by filename: {missing}")
        print("This is informational; EDA reads whatever files are present.")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"download directory (default: {DEFAULT_DEST}, from DATA_DIR)",
    )
    args = parser.parse_args()
    return download(args.dest)


if __name__ == "__main__":
    raise SystemExit(main())
