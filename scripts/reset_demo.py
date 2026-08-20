"""Return the database to a clean pre-demonstration state.

Rehearsing a demonstration leaves residue: scored applications, recorded
decisions, and - most consequentially - confirmed-fraud vectors created while
testing the feedback loop. That last one matters. Confirming a case is
permanent by design, so a rehearsal that confirms an application which was
actually *genuine* teaches the index a genuine profile is fraudulent, and every
later lookalike is escalated for nothing.

What this removes:

* rows in ``applications`` and ``decisions``;
* rows in ``analyst_verdicts``;
* vectors in ``fraud_vectors`` that were NOT part of the historical seed.

What it deliberately keeps:

* the 400 seeded historical frauds (``CONFIRMED-nnnn``), which represent a fraud
  team's existing case history and should be present from the first request;
* ``audit_log`` in its entirety - the table is append-only and the database
  rejects deletion, which is the point of it. It is left alone rather than
  worked around.

Dry run by default. Nothing is deleted without ``--apply``.

Usage:
    python scripts/reset_demo.py            # show what would be removed
    python scripts/reset_demo.py --apply    # actually remove it

Restart the API afterwards: the in-process similarity mirror is rebuilt from
Postgres at startup and will otherwise still hold the removed vectors.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Vectors written by the historical seed. Everything else in fraud_vectors was
# created by a person or a script confirming a case at runtime.
SEED_PREFIX = "CONFIRMED-"


def read_env(key: str) -> str:
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return ""
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true",
        help="perform the deletion (without this, only report what would go)",
    )
    parser.add_argument(
        "--vectors-only", action="store_true",
        help="remove only runtime-confirmed fraud vectors, keeping recorded "
             "applications and decisions. This is usually what you want: a "
             "wrongly-confirmed case changes future decisions, whereas a "
             "recorded decision is just history, and history left in place "
             "keeps the audit panel populated.",
    )
    arguments = parser.parse_args()

    database_url = read_env("DATABASE_URL")
    if not database_url:
        print("DATABASE_URL is not set in .env; nothing to reset.")
        return 1

    try:
        import psycopg
    except ImportError:
        print("psycopg is not installed in this interpreter.")
        return 2

    with psycopg.connect(database_url, connect_timeout=15) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT (SELECT count(*) FROM applications),
                       (SELECT count(*) FROM decisions),
                       (SELECT count(*) FROM analyst_verdicts),
                       (SELECT count(*) FROM audit_log),
                       (SELECT count(*) FROM fraud_vectors),
                       (SELECT count(*) FROM fraud_vectors WHERE case_id LIKE %s);
                """,
                (SEED_PREFIX + "%",),
            )
            apps, decisions, verdicts, audit, vectors, seeded = cursor.fetchone()
            runtime_vectors = vectors - seeded

            print("current state")
            print(f"  applications      {apps}")
            print(f"  decisions         {decisions}")
            print(f"  analyst_verdicts  {verdicts}")
            print(f"  audit_log         {audit}   (append-only, kept)")
            print(f"  fraud_vectors     {vectors}   "
                  f"({seeded} seeded, {runtime_vectors} added at runtime)")

            if runtime_vectors:
                cursor.execute(
                    """
                    SELECT case_id, indexed_at FROM fraud_vectors
                    WHERE case_id NOT LIKE %s ORDER BY indexed_at;
                    """,
                    (SEED_PREFIX + "%",),
                )
                print("\nruntime-confirmed vectors that would be removed:")
                for case_id, indexed_at in cursor.fetchall():
                    print(f"  {case_id}   {indexed_at:%Y-%m-%d %H:%M:%S}")

            if not arguments.apply:
                scope = ("runtime fraud vectors only" if arguments.vectors_only
                         else "vectors, applications, decisions and verdicts")
                print(f"\nDry run ({scope}). Re-run with --apply to delete.")
                return 0

            print("\napplying...")
            cursor.execute(
                "DELETE FROM fraud_vectors WHERE case_id NOT LIKE %s;",
                (SEED_PREFIX + "%",),
            )
            removed_vectors = cursor.rowcount
            removed_verdicts = removed_decisions = removed_apps = 0

            if not arguments.vectors_only:
                cursor.execute("DELETE FROM analyst_verdicts;")
                removed_verdicts = cursor.rowcount
                # decisions references applications, so it goes first.
                cursor.execute("DELETE FROM decisions;")
                removed_decisions = cursor.rowcount
                cursor.execute("DELETE FROM applications;")
                removed_apps = cursor.rowcount
        connection.commit()

    print(f"  removed {removed_vectors} runtime vectors")
    if not arguments.vectors_only:
        print(f"  removed {removed_verdicts} analyst verdicts")
        print(f"  removed {removed_decisions} decisions")
        print(f"  removed {removed_apps} applications")
    else:
        print(f"  kept    {apps} applications and {decisions} decisions")
    print(f"  kept    {seeded} seeded frauds and {audit} audit rows")
    print("\nRestart the API so the similarity mirror is rebuilt from Postgres.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
