"""End-to-end check against a running API.

Verifies the whole path a real request takes - auth, validation, feature
transformation, scoring, calibration, policy, explanation - rather than any one
layer in isolation. Run it after starting the backend and before recording a
demo; a green run means the console will work.

    python scripts/smoke_test.py

Exits non-zero on the first failure, so it is usable as a CI gate.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = os.getenv("AEGIS_URL", "http://127.0.0.1:8000")

EXAMPLE_APPLICATION = {
    "income": 0.3,
    "name_email_similarity": 0.42,
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

PASS, FAIL = "  [PASS]", "  [FAIL]"
failures: list[str] = []


def check(condition: bool, description: str, detail: str = "") -> bool:
    print(f"{PASS if condition else FAIL} {description}{f'  {detail}' if detail else ''}")
    if not condition:
        failures.append(description)
    return condition


def read_api_key() -> str:
    if key := os.getenv("API_KEY"):
        return key
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("API_KEY="):
                return line.split("=", 1)[1].strip()
    print("No API_KEY found in environment or .env", file=sys.stderr)
    raise SystemExit(2)


def wait_for_ready(client: httpx.Client, timeout_s: float = 120.0) -> bool:
    """Poll /health until the model is loaded. Startup deserialises the booster
    and builds the SHAP explainer, which takes a few seconds."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            response = client.get("/health", timeout=5.0)
            if response.status_code == 200 and response.json().get("model_loaded"):
                return True
        except httpx.HTTPError:
            pass
        time.sleep(1.0)
    return False


def main() -> int:
    api_key = read_api_key()
    client = httpx.Client(base_url=BASE_URL, headers={"X-API-Key": api_key}, timeout=30.0)

    print(f"\nAegis smoke test -> {BASE_URL}\n" + "=" * 62)

    print("\nreadiness")
    if not check(wait_for_ready(client), "model loaded and API reachable"):
        print("\nIs the backend running?  uvicorn app.main:app --port 8000")
        return 1

    # --- security -------------------------------------------------------
    print("\nsecurity")
    unauthenticated = httpx.get(f"{BASE_URL}/api/v1/policy", timeout=10.0)
    check(unauthenticated.status_code == 401, "unauthenticated request rejected",
          f"got {unauthenticated.status_code}")

    wrong_key = httpx.get(
        f"{BASE_URL}/api/v1/policy", headers={"X-API-Key": "wrong"}, timeout=10.0
    )
    check(wrong_key.status_code == 401, "invalid API key rejected",
          f"got {wrong_key.status_code}")

    invalid = client.post("/api/v1/applications/score",
                          json={**EXAMPLE_APPLICATION, "customer_age": 9_999})
    check(invalid.status_code == 422, "out-of-range input rejected by validation",
          f"got {invalid.status_code}")

    unknown_field = client.post("/api/v1/applications/score",
                                json={**EXAMPLE_APPLICATION, "injected": "x"})
    check(unknown_field.status_code == 422, "unknown field rejected",
          f"got {unknown_field.status_code}")

    # --- scoring --------------------------------------------------------
    print("\nscoring")
    response = client.post("/api/v1/applications/score", json=EXAMPLE_APPLICATION)
    if not check(response.status_code == 200, "application scored",
                 f"got {response.status_code}: {response.text[:160]}"):
        return 1

    result = response.json()
    check(result["decision"] in {"APPROVE", "REVIEW", "BLOCK"},
          "decision is one of APPROVE/REVIEW/BLOCK", result["decision"])
    check(0.0 <= result["fraud_probability"] <= 1.0,
          "probability in range", f"{result['fraud_probability']:.5f}")
    check(len(result["top_risk_factors"]) > 0,
          "explanation returned", f"{len(result['top_risk_factors'])} risk factors")
    check(all(f["label"] != f["feature"] for f in result["top_risk_factors"][:1]),
          "reason codes are human-readable",
          result["top_risk_factors"][0]["label"])
    check(result["thresholds"]["review"] < result["thresholds"]["block"],
          "thresholds ordered",
          f"review {result['thresholds']['review']:.5f} < block {result['thresholds']['block']:.5f}")
    check(result["narrative"] != "", "narrative produced",
          f"source={result['narrative_source']}")

    # --- determinism ----------------------------------------------------
    # The same application must produce the same decision. If it does not, the
    # audit trail is worthless and the language model has leaked into scoring.
    print("\nreproducibility")
    again = client.post("/api/v1/applications/score", json=EXAMPLE_APPLICATION).json()
    check(again["fraud_probability"] == result["fraud_probability"],
          "identical input gives identical probability")
    check(again["decision"] == result["decision"], "identical input gives identical decision")

    # --- latency --------------------------------------------------------
    # Measured on the decision path, which is what "real-time" refers to.
    # Language-model narration is measured separately below and is explicitly
    # NOT part of this budget - it is an analyst-facing enrichment fetched when
    # a case is opened, not a step in reaching a decision.
    print("\nlatency - decision path (40 requests, no narration)")
    latencies, server_side = [], []
    for _ in range(40):
        started = time.perf_counter()
        response = client.post("/api/v1/applications/score", json=EXAMPLE_APPLICATION)
        latencies.append((time.perf_counter() - started) * 1000)
        server_side.append(response.json()["latency_ms"])

    latencies.sort()
    server_side.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95) - 1]
    print(f"         round-trip   p50 {p50:.1f}ms   p95 {p95:.1f}ms")
    print(f"         server-side  p50 {server_side[len(server_side)//2]:.1f}ms   "
          f"p95 {server_side[int(len(server_side)*0.95)-1]:.1f}ms")
    check(p95 < 250, "decision-path p95 under 250ms", f"{p95:.0f}ms")

    print("\nlatency - narration (language model, off the decision path)")
    started = time.perf_counter()
    narrated = client.post(
        "/api/v1/applications/score?narrate=true", json=EXAMPLE_APPLICATION
    ).json()
    narrated_ms = (time.perf_counter() - started) * 1000
    print(f"         narrated round-trip {narrated_ms:.0f}ms "
          f"(source={narrated['narrative_source']})")
    check(narrated["fraud_probability"] == result["fraud_probability"],
          "narration does not change the probability")
    check(narrated["decision"] == result["decision"],
          "narration does not change the decision")

    # --- policy and metrics ---------------------------------------------
    print("\npolicy and metrics")
    policy = client.get("/api/v1/policy")
    check(policy.status_code == 200, "policy endpoint")

    simulated = client.post("/api/v1/policy/simulate", json={
        "cost_fp_inr": 6000.0, "cost_fn_inr": 45000.0,
        "cost_review_inr": 200.0, "analyst_catch_rate": 0.9,
    })
    if check(simulated.status_code == 200, "policy simulation"):
        # The economic claim: a costlier false positive must make the system
        # less willing to block.
        raised = simulated.json()["thresholds"]["block"]
        current = policy.json()["thresholds"]["block"]
        check(raised > current,
              "raising the false-positive cost raises the block threshold",
              f"{current:.4f} -> {raised:.4f}")

    incoherent = client.post("/api/v1/policy/simulate", json={
        "cost_fp_inr": 100.0, "cost_fn_inr": 45000.0,
        "cost_review_inr": 500.0, "analyst_catch_rate": 0.9,
    })
    check(incoherent.status_code == 422,
          "incoherent cost model rejected (review dearer than a block)",
          f"got {incoherent.status_code}")

    check(client.get("/api/v1/metrics/model").status_code == 200, "model metrics")
    check(client.get("/api/v1/metrics/drift").status_code == 200, "drift metrics")

    stream = client.get("/api/v1/stream/applications?count=3&offset=0")
    if check(stream.status_code == 200, "demo stream"):
        body = stream.json()
        check(len(body["applications"]) == 3, "stream returns the requested count")
        check("true_fraud_rate" in body,
              "stream discloses the true fraud rate",
              f"{body['true_fraud_rate']:.4f}")

    # --- summary --------------------------------------------------------
    print("\n" + "=" * 62)
    if failures:
        print(f"FAILED: {len(failures)} check(s)")
        for description in failures:
            print(f"  - {description}")
        return 1

    print("All checks passed.")
    print(f"\nSample decision: {result['decision']} "
          f"at p={result['fraud_probability']:.4f}")
    print(f"Leading factor : {result['top_risk_factors'][0]['label']}")
    print(f"Narrative      : {result['narrative'][:150]}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
