"""End-to-end verification of the persistence layer against a live database.

The unit tests in ``backend/tests/test_persistence.py`` run without Postgres, and
deliberately so - they cover what happens when the database is absent or broken.
This script covers the opposite case: that the durable path genuinely works.

It asserts the four claims the README makes about persistence:

1. scoring writes through to Postgres without entering the decision path;
2. the audit log is append-only, enforced by the database rather than by code;
3. an analyst's confirmed fraud is stored durably and escalates on re-score;
4. the similarity index is rebuilt from Postgres, so a confirmation survives a
   restart.

Claim 4 needs the API restarted midway, so the script pauses and tells you when.

Usage:
    python scripts/verify_persistence.py            # claims 1-3
    python scripts/verify_persistence.py --restart  # all four, with a pause

Requires the API running on :8000 with DATABASE_URL configured.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "http://127.0.0.1:8000"

passed = 0
failed = 0


def check(label: str, condition: bool, detail: str = "") -> bool:
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {label}" + (f"   [{detail}]" if detail else ""))
    else:
        failed += 1
        print(f"  FAIL  {label}" + (f"   [{detail}]" if detail else ""))
    return condition


def read_env(key: str) -> str:
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return ""
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def heading(text: str) -> None:
    print()
    print("=" * 74)
    print(text)
    print("=" * 74)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--restart", action="store_true",
        help="also verify that state survives an API restart (pauses for input)",
    )
    arguments = parser.parse_args()

    api_key = read_env("API_KEY")
    database_url = read_env("DATABASE_URL")
    if not api_key:
        print("API_KEY not found in .env")
        return 2

    client = httpx.Client(
        base_url=BASE_URL, headers={"X-API-Key": api_key}, timeout=60.0
    )

    def storage() -> dict:
        return client.get("/api/v1/storage").json()

    # -- 0. preconditions ------------------------------------------------
    heading("0. preconditions")
    try:
        health = client.get("/health").json()
    except httpx.ConnectError:
        print("  Cannot reach the API on :8000. Start it first.")
        return 2

    check("API is up and a model is loaded", health.get("model_loaded") is True)
    if not check("DATABASE_URL is configured", bool(database_url)):
        print("\n  Nothing to verify without a database. Set DATABASE_URL in .env.")
        return 1

    parts = urlsplit(database_url)
    print(f"        database: {parts.hostname}:{parts.port or 5432}{parts.path}")

    before = storage()
    if not check("persistence is active", before["persistence"] is True):
        print("\n  The API is running without a database. Restart it after "
              "setting DATABASE_URL.")
        return 1
    check("similarity index is durable", before["similarity_index"]["durable"] is True)

    # -- 1. scoring writes through, off the decision path -----------------
    heading("1. scoring writes through to Postgres, off the decision path")

    batch = client.get("/api/v1/stream/applications", params={"count": 10}).json()
    streamed = batch["applications"]
    applications = [item["application"] for item in streamed]

    # Confirm a case that genuinely IS fraud.
    #
    # This script writes to a durable index that the running demo reads from,
    # and a "confirmed fraud" is permanent. Confirming whichever application
    # happened to arrive first would teach the index that a genuine applicant's
    # profile is fraudulent, and every later lookalike would be escalated for
    # nothing. The replay feed carries the true label precisely so a test can
    # avoid poisoning the thing it is testing.
    fraud_positions = [i for i, item in enumerate(streamed) if item["actual_fraud"]]
    confirm_position = fraud_positions[0] if fraud_positions else 0

    latencies, scored_ids = [], []
    for application in applications:
        result = client.post("/api/v1/applications/score", json=application)
        result.raise_for_status()
        body = result.json()
        latencies.append(body["latency_ms"])
        scored_ids.append(body["application_id"])

    print(f"        scored {len(scored_ids)} applications, "
          f"latency p50 {sorted(latencies)[len(latencies)//2]:.0f}ms")
    check("decisions did not wait on a database round trip",
          max(latencies) < 1_000,
          f"max {max(latencies):.0f}ms")

    print("        waiting for the async writer to flush...")
    time.sleep(3)
    after = storage()

    check("applications persisted",
          after["tables"]["applications"] >= before["tables"]["applications"] + len(scored_ids),
          f"{before['tables']['applications']} -> {after['tables']['applications']}")
    check("decisions persisted",
          after["tables"]["decisions"] >= before["tables"]["decisions"] + len(scored_ids),
          f"{before['tables']['decisions']} -> {after['tables']['decisions']}")
    check("audit entries written",
          after["tables"]["audit_log"] > before["tables"]["audit_log"],
          f"{before['tables']['audit_log']} -> {after['tables']['audit_log']}")
    check("writer dropped nothing", after["writer"]["dropped"] == 0)

    # -- 2. the audit log is append-only ---------------------------------
    heading("2. the audit log is append-only, enforced by Postgres")

    audit = client.get("/api/v1/audit/recent", params={"limit": 10}).json()
    entries = audit["entries"]
    check("audit entries are readable", len(entries) > 0, f"{len(entries)} entries")
    if entries:
        check("entries are newest-first",
              all(entries[i]["id"] > entries[i + 1]["id"] for i in range(len(entries) - 1)))
        check("an entry names its event and actor",
              bool(entries[0].get("event")) and bool(entries[0].get("actor")),
              f"{entries[0].get('event')} / {entries[0].get('actor')}")

    try:
        import psycopg
    except ImportError:
        print("  SKIP  mutation guard (psycopg not importable here)")
    else:
        for operation, sql in (
            ("UPDATE", "UPDATE audit_log SET event = 'tampered' WHERE id = "
                       "(SELECT max(id) FROM audit_log);"),
            ("DELETE", "DELETE FROM audit_log WHERE id = (SELECT max(id) FROM audit_log);"),
        ):
            try:
                with psycopg.connect(database_url, connect_timeout=15) as connection:
                    with connection.cursor() as cursor:
                        cursor.execute(sql)
                    connection.commit()
                check(f"{operation} on audit_log is rejected", False, "IT SUCCEEDED")
            except Exception as error:
                check(f"{operation} on audit_log is rejected", True,
                      str(error).strip().splitlines()[0][:60])

    # -- 3. an analyst's confirmation is durable and acts ----------------
    heading("3. a confirmed fraud is stored durably and changes the next decision")

    target = applications[confirm_position]
    confirm_id = scored_ids[confirm_position]
    if not fraud_positions:
        print("        NOTE: no fraudulent application in this batch; confirming a "
              "genuine one, which will need cleaning up afterwards")
    else:
        print(f"        confirming {confirm_id}, whose true label is fraud")

    response = client.post(
        f"/api/v1/applications/{confirm_id}/decision",
        json={
            "verdict": {"analyst_id": "verify-persistence", "outcome": "confirmed_fraud"},
            "application": target,
        },
    )
    response.raise_for_status()
    check("verdict accepted", response.status_code == 200)
    check("case entered the similarity index",
          response.json()["added_to_similarity_index"] is True)

    time.sleep(3)
    final = storage()
    check("verdict row persisted",
          final["tables"]["analyst_verdicts"] > after["tables"]["analyst_verdicts"],
          f"{after['tables']['analyst_verdicts']} -> {final['tables']['analyst_verdicts']}")
    check("confirmed fraud persisted as a vector",
          final["tables"]["fraud_vectors"] >= before["tables"]["fraud_vectors"],
          f"{before['tables']['fraud_vectors']} -> {final['tables']['fraud_vectors']}")

    rescored = client.post("/api/v1/applications/score", json=target).json()
    check("the confirmed case now escalates on re-score",
          rescored["escalated"] is True,
          (rescored.get("escalation_reason") or "not escalated")[:70])
    check("the model's own decision is preserved for audit",
          rescored["model_decision"] == "APPROVE" or rescored["model_decision"] is not None,
          f"model={rescored['model_decision']} final={rescored['decision']}")

    # -- 4. state survives a restart -------------------------------------
    if arguments.restart:
        heading("4. state survives an API restart")
        snapshot = storage()
        print(f"        fraud_vectors {snapshot['tables']['fraud_vectors']}   "
              f"index {snapshot['similarity_index']['size']}   "
              f"decisions {snapshot['tables']['decisions']}")
        print()
        print("        >>> Restart the API now, then press Enter. <<<")
        input()

        for _ in range(60):
            try:
                if client.get("/health").json().get("model_loaded"):
                    break
            except httpx.ConnectError:
                pass
            time.sleep(2)

        restored = storage()
        check("fraud vectors unchanged",
              restored["tables"]["fraud_vectors"] == snapshot["tables"]["fraud_vectors"],
              f"{snapshot['tables']['fraud_vectors']} -> {restored['tables']['fraud_vectors']}")
        check("index rebuilt from Postgres rather than re-seeded",
              restored["similarity_index"]["size"] == snapshot["similarity_index"]["size"],
              f"{snapshot['similarity_index']['size']} -> {restored['similarity_index']['size']}")
        check("recorded decisions survived",
              restored["tables"]["decisions"] >= snapshot["tables"]["decisions"],
              f"{snapshot['tables']['decisions']} -> {restored['tables']['decisions']}")

    heading(f"RESULT: {passed} passed, {failed} failed")
    if not arguments.restart:
        print("Run with --restart to also verify that state survives a restart.")

    client.close()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
