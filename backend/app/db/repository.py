"""Reading and writing the decision record.

Why writes happen on a background thread
----------------------------------------
This service scores an application in roughly 80ms. The database is a managed
Postgres reached over the public internet, where a single round trip from here
costs 40-120ms depending on the link. Writing the audit record inside the
request would therefore *double or triple* the latency of a system whose central
claim is that it decides in real time - and it would make an outage in a
downstream store into an outage of the decisioning path, which is exactly
backwards. A decision must still be made when the recorder is unavailable.

So writes are enqueued and drained by a single background worker that batches
them. The decision path pays a queue append, measured in microseconds.

The honest cost of that choice: a process killed between the enqueue and the
flush loses at most one batch. That is the correct trade for a prototype talking
to a cloud database across the internet, and it is the wrong trade for a real
deployment - where the database is a millisecond away and the write belongs
inside the request's transaction, or ahead of it in a durable log. The structure
here does not change in that world; ``flush_interval`` goes to zero and the
enqueue becomes a direct call.

Nothing in this module is allowed to raise into a caller. Persistence is a
recording obligation, not a precondition for deciding.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from dataclasses import dataclass
from typing import Any

from app.db.session import Database

logger = logging.getLogger(__name__)

# Bounded so a database outage cannot grow memory without limit. When full,
# the oldest pending writes are dropped and the loss is counted and logged -
# silently discarding audit records would be far worse than saying so.
QUEUE_CAPACITY = 5_000

# How long the worker waits to accumulate a batch before writing.
FLUSH_INTERVAL_SECONDS = 0.5

# Upper bound on one INSERT ... executemany.
MAX_BATCH = 200


@dataclass(frozen=True, slots=True)
class _Write:
    """One pending statement: a kind, used to group, and its parameters."""

    kind: str
    params: tuple


class DecisionRepository:
    """Persists applications, decisions, verdicts and the audit trail.

    Construct it with an unavailable :class:`Database` and every write becomes a
    no-op, so callers never branch on whether persistence is configured.
    """

    def __init__(self, database: Database) -> None:
        self._database = database
        self._queue: queue.Queue[_Write | None] = queue.Queue(maxsize=QUEUE_CAPACITY)
        self._worker: threading.Thread | None = None
        self._stopping = threading.Event()
        self._dropped = 0
        self._written = 0

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        return self._database.available

    def start(self) -> None:
        """Begin draining the queue. No-op when persistence is disabled."""
        if not self.available or self._worker is not None:
            return
        self._worker = threading.Thread(
            target=self._drain_forever, name="aegis-writer", daemon=True
        )
        self._worker.start()
        logger.info("audit writer started (batching every %.1fs)", FLUSH_INTERVAL_SECONDS)

    def stop(self, timeout: float = 5.0) -> None:
        """Flush what is pending and stop the worker."""
        if self._worker is None:
            return
        self._stopping.set()
        try:
            self._queue.put_nowait(None)        # wake the worker immediately
        except queue.Full:
            pass
        self._worker.join(timeout=timeout)
        self._worker = None
        logger.info("audit writer stopped (%d written, %d dropped)",
                    self._written, self._dropped)

    # ------------------------------------------------------------------
    # writes
    # ------------------------------------------------------------------

    def record_decision(
        self, application_id: str, payload: dict[str, Any], decision: Any
    ) -> None:
        """Record the application, the decision it produced, and an audit entry.

        ``decision`` is a ``DecisionOut``; it is read here rather than imported
        so this module stays free of a dependency on the API schema layer.
        """
        if not self.available:
            return

        self._enqueue(_Write("application", (application_id, json.dumps(payload, default=str))))
        self._enqueue(_Write("decision", (
            application_id,
            decision.decision,
            decision.model_decision,
            float(decision.fraud_probability),
            decision.risk_band,
            float(decision.thresholds["review"]),
            float(decision.thresholds["block"]),
            bool(decision.escalated),
            decision.escalation_reason,
            json.dumps([
                {"feature": r.feature, "label": r.label,
                 "contribution": r.contribution, "direction": r.direction}
                for r in decision.top_risk_factors
            ], default=str),
            decision.model_version,
            float(decision.latency_ms),
        )))
        self._audit(
            "decision.recorded",
            application_id,
            actor="system",
            detail={
                "decision": decision.decision,
                "model_decision": decision.model_decision,
                "fraud_probability": round(float(decision.fraud_probability), 6),
                "escalated": bool(decision.escalated),
                "model_version": decision.model_version,
            },
        )

    def record_verdict(
        self, application_id: str, analyst_id: str, outcome: str, *, indexed: bool
    ) -> None:
        """Record a human's conclusion and whether it changed the index."""
        if not self.available:
            return

        self._enqueue(_Write("verdict", (application_id, analyst_id, outcome)))
        self._audit(
            "analyst.verdict",
            application_id,
            actor=analyst_id,
            detail={"outcome": outcome, "added_to_similarity_index": indexed},
        )

    def _audit(
        self, event: str, application_id: str | None, *, actor: str, detail: dict
    ) -> None:
        self._enqueue(_Write("audit", (
            event, application_id, actor, json.dumps(detail, default=str)
        )))

    def _enqueue(self, write: _Write) -> None:
        try:
            self._queue.put_nowait(write)
        except queue.Full:
            # Drop the oldest rather than the newest: recent decisions are the
            # ones an analyst is about to look at.
            self._dropped += 1
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(write)
            except (queue.Empty, queue.Full):
                pass
            if self._dropped % 100 == 1:
                logger.error(
                    "audit queue saturated; %d records dropped so far", self._dropped
                )

    # ------------------------------------------------------------------
    # the worker
    # ------------------------------------------------------------------

    _SQL: dict[str, str] = {
        "application": """
            INSERT INTO applications (application_id, payload)
            VALUES (%s, %s::jsonb)
            ON CONFLICT (application_id) DO NOTHING;
        """,
        "decision": """
            INSERT INTO decisions (
                application_id, decision, model_decision, fraud_probability,
                risk_band, tau_review, tau_block, escalated, escalation_reason,
                reason_codes, model_version, latency_ms
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s);
        """,
        "verdict": """
            INSERT INTO analyst_verdicts (application_id, analyst_id, outcome)
            VALUES (%s,%s,%s);
        """,
        "audit": """
            INSERT INTO audit_log (event, application_id, actor, detail)
            VALUES (%s,%s,%s,%s::jsonb);
        """,
    }

    def _drain_forever(self) -> None:
        while not (self._stopping.is_set() and self._queue.empty()):
            batch = self._collect_batch()
            if batch:
                self._write_batch(batch)

    def _collect_batch(self) -> list[_Write]:
        """Block for the first item, then take whatever else is ready."""
        batch: list[_Write] = []
        try:
            first = self._queue.get(timeout=FLUSH_INTERVAL_SECONDS)
        except queue.Empty:
            return batch
        if first is not None:
            batch.append(first)

        while len(batch) < MAX_BATCH:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is not None:
                batch.append(item)
        return batch

    def _write_batch(self, batch: list[_Write]) -> None:
        """Write one batch, preserving order within each statement kind.

        Applications must land before the decisions that reference them, so the
        kinds are written in dependency order rather than in arrival order.
        """
        grouped: dict[str, list[tuple]] = {}
        for write in batch:
            grouped.setdefault(write.kind, []).append(write.params)

        try:
            with self._database.cursor() as cursor:
                for kind in ("application", "decision", "verdict", "audit"):
                    rows = grouped.get(kind)
                    if rows:
                        cursor.executemany(self._SQL[kind], rows)
            self._written += len(batch)
        except Exception:               # noqa: BLE001 - a recorder must not crash
            logger.exception("audit batch of %d records failed", len(batch))

    # ------------------------------------------------------------------
    # reads
    # ------------------------------------------------------------------

    def recent_audit(self, limit: int = 50) -> list[dict[str, Any]]:
        """The tail of the audit log, newest first."""
        if not self.available:
            return []
        try:
            with self._database.cursor(commit=False) as cursor:
                cursor.execute(
                    """
                    SELECT id, event, application_id, actor, detail, occurred_at
                    FROM audit_log ORDER BY id DESC LIMIT %s;
                    """,
                    (limit,),
                )
                return [
                    {
                        "id": row[0], "event": row[1], "application_id": row[2],
                        "actor": row[3], "detail": row[4],
                        "occurred_at": row[5].isoformat(),
                    }
                    for row in cursor.fetchall()
                ]
        except Exception:               # noqa: BLE001
            logger.exception("audit read failed")
            return []

    def counts(self) -> dict[str, int]:
        """Row counts per table, for the operations view."""
        if not self.available:
            return {}
        try:
            with self._database.cursor(commit=False) as cursor:
                cursor.execute(
                    """
                    SELECT (SELECT count(*) FROM applications),
                           (SELECT count(*) FROM decisions),
                           (SELECT count(*) FROM analyst_verdicts),
                           (SELECT count(*) FROM audit_log),
                           (SELECT count(*) FROM fraud_vectors);
                    """
                )
                row = cursor.fetchone()
                return {
                    "applications": row[0], "decisions": row[1],
                    "analyst_verdicts": row[2], "audit_log": row[3],
                    "fraud_vectors": row[4],
                }
        except Exception:               # noqa: BLE001
            logger.exception("count query failed")
            return {}

    @property
    def stats(self) -> dict[str, int]:
        """Writer counters, for diagnosing a saturated queue."""
        return {
            "queued": self._queue.qsize(),
            "written": self._written,
            "dropped": self._dropped,
        }
