"""Database schema, applied idempotently at startup.

Kept as SQL in one place rather than spread across an ORM's model classes. The
schema is small, it is read by people evaluating this system, and the
constraints below carry the argument - they are easier to check as SQL than as
declarative Python.

Four tables and one extension:

``applications``      what was submitted, verbatim
``decisions``         what the system decided and why
``analyst_verdicts``  what a human concluded afterwards
``audit_log``         an append-only record of both, enforced by the database
``fraud_vectors``     pgvector embeddings of confirmed fraud, for lookalike search

Why the raw application is stored as JSONB rather than thirty typed columns: the
feature contract is owned by ``ml/features/pipeline.py`` and changes when the
model changes. Pinning it into DDL would mean a migration every time a feature is
added, and would let the table and the model disagree - which is the same
training/serving skew problem this project takes pains to avoid elsewhere. The
decision columns, by contrast, *are* typed: they are the regulated artefact and
their shape must not drift.
"""

from __future__ import annotations

import logging

from app.db.session import Database
from app.services.similarity import PROJECTION_DIMENSION

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The append-only guarantee
# ---------------------------------------------------------------------------
# The API documentation states that every decision is written to an append-only
# audit log. A comment promising that is worth nothing - anything holding a
# connection could rewrite history. This trigger makes the database refuse, so
# the guarantee holds against the application itself, not merely against
# well-behaved code paths.
APPEND_ONLY_GUARD = """
CREATE OR REPLACE FUNCTION aegis_reject_mutation() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'audit_log is append-only; % is not permitted', TG_OP;
END;
$$ LANGUAGE plpgsql;
"""

STATEMENTS: tuple[str, ...] = (
    # -- extension ---------------------------------------------------------
    "CREATE EXTENSION IF NOT EXISTS vector;",

    # -- applications ------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS applications (
        application_id  TEXT PRIMARY KEY,
        payload         JSONB       NOT NULL,
        received_at     TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    """,

    # -- decisions ---------------------------------------------------------
    # `decision` and `model_decision` are separate columns on purpose. When the
    # similarity layer escalates an application, an auditor must be able to see
    # both what the model concluded and what the system finally did; collapsing
    # them into one column would erase the intervention.
    """
    CREATE TABLE IF NOT EXISTS decisions (
        id                BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        application_id    TEXT        NOT NULL REFERENCES applications(application_id),
        decision          TEXT        NOT NULL CHECK (decision IN ('APPROVE','REVIEW','BLOCK')),
        model_decision    TEXT        NOT NULL CHECK (model_decision IN ('APPROVE','REVIEW','BLOCK')),
        fraud_probability DOUBLE PRECISION NOT NULL CHECK (fraud_probability BETWEEN 0 AND 1),
        risk_band         TEXT        NOT NULL,
        tau_review        DOUBLE PRECISION NOT NULL,
        tau_block         DOUBLE PRECISION NOT NULL,
        escalated         BOOLEAN     NOT NULL DEFAULT false,
        escalation_reason TEXT,
        reason_codes      JSONB       NOT NULL DEFAULT '[]'::jsonb,
        model_version     TEXT        NOT NULL,
        latency_ms        DOUBLE PRECISION NOT NULL,
        decided_at        TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    """,
    "CREATE INDEX IF NOT EXISTS decisions_decided_at_idx ON decisions (decided_at DESC);",
    "CREATE INDEX IF NOT EXISTS decisions_application_idx ON decisions (application_id);",

    # -- analyst verdicts --------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS analyst_verdicts (
        id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        application_id  TEXT        NOT NULL,
        analyst_id      TEXT        NOT NULL,
        outcome         TEXT        NOT NULL CHECK (outcome IN ('confirmed_fraud','cleared_genuine')),
        recorded_at     TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    """,
    "CREATE INDEX IF NOT EXISTS analyst_verdicts_application_idx ON analyst_verdicts (application_id);",

    # -- audit log ---------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS audit_log (
        id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        event           TEXT        NOT NULL,
        application_id  TEXT,
        actor           TEXT        NOT NULL DEFAULT 'system',
        detail          JSONB       NOT NULL DEFAULT '{}'::jsonb,
        occurred_at     TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    """,
    "CREATE INDEX IF NOT EXISTS audit_log_occurred_at_idx ON audit_log (occurred_at DESC);",

    # -- fraud vectors -----------------------------------------------------
    # Mirrors PgVectorSimilarityIndex.ensure_schema so both paths agree on shape.
    #
    # `leaves` holds the raw per-tree leaf assignment alongside the projected
    # embedding. The projection is lossy and one-way, so without this column the
    # process-local index could not be rebuilt exactly after a restart - it
    # would have to search approximate vectors and return neighbours that differ
    # from the ones the exact metric would give. Storing both keeps pgvector as
    # the system of record for k-NN while the hot path stays exact.
    f"""
    CREATE TABLE IF NOT EXISTS fraud_vectors (
        case_id          TEXT PRIMARY KEY,
        embedding        vector({PROJECTION_DIMENSION}) NOT NULL,
        leaves           INTEGER[]   NOT NULL DEFAULT '{{}}',
        confirmed_fraud  BOOLEAN     NOT NULL,
        metadata         JSONB       NOT NULL DEFAULT '{{}}'::jsonb,
        indexed_at       TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    """,
    # Added separately so an existing table from an earlier boot gains it.
    "ALTER TABLE fraud_vectors ADD COLUMN IF NOT EXISTS leaves INTEGER[] NOT NULL DEFAULT '{}';",
    # HNSW over cosine distance, matching the unit-length embeddings written by
    # PgVectorSimilarityIndex.embed. Built here rather than only in that class
    # so the index exists even before the first vector is written.
    """
    CREATE INDEX IF NOT EXISTS fraud_vectors_embedding_idx
    ON fraud_vectors USING hnsw (embedding vector_cosine_ops);
    """,
)

# Applied after the tables exist.
TRIGGERS: tuple[str, ...] = (
    APPEND_ONLY_GUARD,
    "DROP TRIGGER IF EXISTS audit_log_append_only ON audit_log;",
    """
    CREATE TRIGGER audit_log_append_only
    BEFORE UPDATE OR DELETE ON audit_log
    FOR EACH ROW EXECUTE FUNCTION aegis_reject_mutation();
    """,
)


def apply(database: Database) -> bool:
    """Create everything that is missing. Safe to run on every boot.

    Returns True when the schema is in place. Failure is logged and reported,
    never raised: the service falls back to its in-memory path.
    """
    if not database.available:
        return False

    try:
        with database.cursor() as cursor:
            for statement in STATEMENTS:
                cursor.execute(statement)
            for statement in TRIGGERS:
                cursor.execute(statement)
        logger.info("schema applied: applications, decisions, analyst_verdicts, "
                    "audit_log, fraud_vectors (audit_log is append-only)")
        return True
    except Exception:                   # noqa: BLE001
        logger.exception("schema creation failed; continuing without persistence")
        return False
