"""Similarity search over confirmed fraud cases.

The adaptation problem this solves
----------------------------------
A supervised model only recognises fraud that resembles its training labels.
When a new fraud pattern appears, the model is blind to it until someone
confirms enough cases, retrains, revalidates, and redeploys - weeks during
which the pattern runs unchecked.

Similarity search closes that gap. When an analyst confirms one fraud, its
signature is indexed immediately. Every subsequent application that resembles
it is flagged from that moment on, with **no retraining**. One confirmed case
becomes a detector.

How applications are compared
-----------------------------
Not by raw feature distance, which weights every column equally and treats
``proposed_credit_limit`` as though it mattered as much as
``name_email_similarity``. Instead, each application is represented by the
**leaves it occupies in the gradient-boosted ensemble**.

Every tree in the model partitions applications into leaves; two applications
landing in the same leaf were routed there by the same sequence of decisions.
Their similarity is the fraction of trees that place them together. This has
three properties that matter:

* it uses the model's own learned notion of what makes applications alike,
  including feature interactions, rather than a distance metric imposed on top;
* it is bounded in [0, 1] and directly interpretable - "these two agree in 82%
  of the model's decision paths";
* it is fast, needing one integer comparison per tree.

Storage backends
----------------
Two implementations behind one interface. The in-memory index is the default
and needs no infrastructure. The pgvector index persists across restarts and
scales past what fits in process memory; leaf assignments are projected to a
dense vector so Postgres can index them (Johnson-Lindenstrauss: a random
projection approximately preserves the distances that matter).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)

# Dimension of the dense projection used by the pgvector backend. 256 keeps
# JL distortion small for the leaf-overlap metric while staying well within
# pgvector's practical index limits.
PROJECTION_DIMENSION = 256


@dataclass(frozen=True, slots=True)
class SimilarCase:
    """A stored case and how closely it resembles the query."""

    case_id: str
    similarity: float
    confirmed_fraud: bool
    metadata: dict[str, object] = field(default_factory=dict)


@runtime_checkable
class SimilarityIndex(Protocol):
    """Storage-agnostic interface for the confirmed-fraud index."""

    def add(
        self,
        case_id: str,
        leaf_assignment: np.ndarray,
        *,
        confirmed_fraud: bool,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Index one case. Takes effect immediately for subsequent searches."""
        ...

    def search(self, leaf_assignment: np.ndarray, k: int = 5) -> list[SimilarCase]:
        """Return the ``k`` most similar indexed cases, most similar first."""
        ...

    def __len__(self) -> int:
        ...


class InMemorySimilarityIndex:
    """Leaf-overlap index held in process memory.

    Similarity is the fraction of trees assigning both applications to the same
    leaf. Exhaustive search is deliberate: at demo scale it is faster than any
    approximate structure, and it returns exact neighbours, so the explanation
    shown to an analyst is never an artefact of an index approximation.
    """

    def __init__(self) -> None:
        self._case_ids: list[str] = []
        self._leaves: list[np.ndarray] = []
        self._confirmed: list[bool] = []
        self._metadata: list[dict[str, object]] = []
        self._matrix: np.ndarray | None = None   # rebuilt lazily on write

    def add(
        self,
        case_id: str,
        leaf_assignment: np.ndarray,
        *,
        confirmed_fraud: bool,
        metadata: dict[str, object] | None = None,
    ) -> None:
        leaves = np.asarray(leaf_assignment, dtype=np.int32).ravel()
        if self._leaves and leaves.shape != self._leaves[0].shape:
            raise ValueError(
                f"leaf assignment has {leaves.shape[0]} trees, index holds "
                f"{self._leaves[0].shape[0]}; the model changed under the index"
            )
        self._case_ids.append(case_id)
        self._leaves.append(leaves)
        self._confirmed.append(confirmed_fraud)
        self._metadata.append(metadata or {})
        self._matrix = None

    def search(self, leaf_assignment: np.ndarray, k: int = 5) -> list[SimilarCase]:
        if not self._leaves:
            return []

        if self._matrix is None:
            self._matrix = np.vstack(self._leaves)

        query = np.asarray(leaf_assignment, dtype=np.int32).ravel()
        # Fraction of trees routing both applications to the same leaf.
        similarities = (self._matrix == query).mean(axis=1)

        k = min(k, len(similarities))
        top = np.argpartition(-similarities, k - 1)[:k]
        top = top[np.argsort(-similarities[top])]

        return [
            SimilarCase(
                case_id=self._case_ids[index],
                similarity=float(similarities[index]),
                confirmed_fraud=self._confirmed[index],
                metadata=self._metadata[index],
            )
            for index in top
        ]

    def __len__(self) -> int:
        return len(self._case_ids)


class PgVectorSimilarityIndex:
    """Postgres-backed index using the pgvector extension.

    Leaf assignments are integers with no meaningful magnitude - leaf 7 is not
    "greater than" leaf 3 - so they cannot be handed to a vector distance
    directly. Each assignment is one-hot encoded per tree and projected through
    a fixed random matrix to a dense vector, which approximately preserves
    leaf-overlap similarity as cosine similarity (Johnson-Lindenstrauss).

    The projection matrix is derived from a fixed seed rather than stored, so
    the same model always produces the same embedding and the index stays
    consistent across restarts and across processes.
    """

    def __init__(
        self,
        connection_factory,
        *,
        n_trees: int,
        max_leaves: int,
        table: str = "fraud_vectors",
        seed: int = 20260820,
    ) -> None:
        self._connect = connection_factory
        self._table = table
        self._n_trees = n_trees
        self._max_leaves = max_leaves
        generator = np.random.default_rng(seed)
        self._projection = generator.standard_normal(
            (n_trees * max_leaves, PROJECTION_DIMENSION)
        ).astype(np.float32) / np.sqrt(PROJECTION_DIMENSION)

    def embed(self, leaf_assignment: np.ndarray) -> np.ndarray:
        """Project a leaf assignment into a dense, unit-length vector."""
        leaves = np.asarray(leaf_assignment, dtype=np.int64).ravel()
        # Flat index of the active one-hot position for each tree.
        active = np.arange(self._n_trees) * self._max_leaves + np.clip(
            leaves, 0, self._max_leaves - 1
        )
        vector = self._projection[active].sum(axis=0)
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def ensure_schema(self) -> None:
        """Create the table and index if absent. Safe to call repeatedly."""
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    case_id          TEXT PRIMARY KEY,
                    embedding        vector({PROJECTION_DIMENSION}) NOT NULL,
                    confirmed_fraud  BOOLEAN NOT NULL,
                    metadata         JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    indexed_at       TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )
            # Cosine distance, matching the normalised embeddings above.
            cursor.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {self._table}_embedding_idx
                ON {self._table} USING hnsw (embedding vector_cosine_ops);
                """
            )
            connection.commit()

    def add(
        self,
        case_id: str,
        leaf_assignment: np.ndarray,
        *,
        confirmed_fraud: bool,
        metadata: dict[str, object] | None = None,
    ) -> None:
        import json

        leaves = np.asarray(leaf_assignment, dtype=np.int32).ravel()
        embedding = self.embed(leaves)
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO {self._table}
                       (case_id, embedding, leaves, confirmed_fraud, metadata)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (case_id) DO UPDATE
                    SET embedding = EXCLUDED.embedding,
                        leaves = EXCLUDED.leaves,
                        confirmed_fraud = EXCLUDED.confirmed_fraud,
                        metadata = EXCLUDED.metadata;
                """,
                (case_id, str(embedding.tolist()), [int(v) for v in leaves],
                 confirmed_fraud, json.dumps(metadata or {}, default=str)),
            )
            connection.commit()

    def add_many(self, records: list[tuple[str, np.ndarray, bool, dict]]) -> int:
        """Insert many cases in one round trip.

        Seeding writes several hundred cases at startup. One statement per case
        against a database reached over the internet would take minutes; batched
        it takes a second.
        """
        import json

        if not records:
            return 0

        rows = []
        for case_id, leaves_array, confirmed, metadata in records:
            leaves = np.asarray(leaves_array, dtype=np.int32).ravel()
            rows.append((
                case_id,
                str(self.embed(leaves).tolist()),
                [int(v) for v in leaves],
                confirmed,
                json.dumps(metadata or {}, default=str),
            ))

        with self._connect() as connection, connection.cursor() as cursor:
            cursor.executemany(
                f"""
                INSERT INTO {self._table}
                       (case_id, embedding, leaves, confirmed_fraud, metadata)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (case_id) DO NOTHING;
                """,
                rows,
            )
            connection.commit()
        return len(rows)

    def load_all(self) -> list[tuple[str, np.ndarray, bool, dict]]:
        """Every stored case, for rebuilding a process-local index at startup.

        This is what makes a confirmed fraud outlive the process that recorded
        it. Without it the index is repopulated from the training data on every
        boot and an analyst's work is silently discarded.
        """
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT case_id, leaves, confirmed_fraud, metadata
                FROM {self._table}
                WHERE array_length(leaves, 1) IS NOT NULL
                ORDER BY indexed_at;
                """
            )
            return [
                (row[0], np.asarray(row[1], dtype=np.int32), bool(row[2]), row[3] or {})
                for row in cursor.fetchall()
            ]

    def search(self, leaf_assignment: np.ndarray, k: int = 5) -> list[SimilarCase]:
        embedding = self.embed(leaf_assignment)
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT case_id,
                       1 - (embedding <=> %s::vector) AS similarity,
                       confirmed_fraud,
                       metadata
                FROM {self._table}
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (embedding.tolist(), embedding.tolist(), k),
            )
            return [
                SimilarCase(
                    case_id=row[0],
                    similarity=float(row[1]),
                    confirmed_fraud=bool(row[2]),
                    metadata=row[3] or {},
                )
                for row in cursor.fetchall()
            ]

    def __len__(self) -> int:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(f"SELECT count(*) FROM {self._table};")
            return int(cursor.fetchone()[0])


class DurableSimilarityIndex:
    """Postgres is the record; a process-local mirror serves the hot path.

    Why not query pgvector on every scoring request? Because the database is
    250ms away over the public internet and a decision is budgeted at 80ms.
    Putting the vector search inside the request would make the round trip to
    the index three times more expensive than everything else the system does
    combined, and would tie the availability of a real-time decision to the
    availability of a remote store.

    So this is cache-aside. Every write goes to Postgres first - if that fails,
    nothing is added anywhere, because an index entry that exists only in memory
    is a lie about durability. Reads on the scoring path come from the mirror,
    which is rebuilt from Postgres at startup and is therefore exact rather than
    approximate: it compares raw leaf assignments, not projected vectors.

    ``search_vector`` runs the genuine pgvector k-NN. It is used to verify that
    the stored embeddings and the HNSW index agree with the exact metric, and it
    is what a horizontally-scaled deployment would call directly once the index
    outgrew a single process's memory.
    """

    def __init__(
        self, store: "PgVectorSimilarityIndex", mirror: InMemorySimilarityIndex | None = None
    ) -> None:
        self._store = store
        # `mirror is None`, never `mirror or ...`: an index defines __len__, so
        # an empty one is falsy and `or` would silently discard the caller's
        # object and substitute a different one. Writes would then land in an
        # index nobody searches.
        self._mirror = InMemorySimilarityIndex() if mirror is None else mirror

    def restore(self) -> int:
        """Rebuild the mirror from Postgres. Returns the number of cases loaded."""
        records = self._store.load_all()
        for case_id, leaves, confirmed, metadata in records:
            self._mirror.add(
                case_id, leaves, confirmed_fraud=confirmed, metadata=metadata
            )
        logger.info("similarity mirror restored: %d cases from Postgres", len(records))
        return len(records)

    def add(
        self,
        case_id: str,
        leaf_assignment: np.ndarray,
        *,
        confirmed_fraud: bool,
        metadata: dict[str, object] | None = None,
    ) -> None:
        # Durable first. If this raises, the caller sees the failure and the
        # mirror is untouched, so the two never disagree.
        self._store.add(
            case_id, leaf_assignment,
            confirmed_fraud=confirmed_fraud, metadata=metadata,
        )
        self._mirror.add(
            case_id, leaf_assignment,
            confirmed_fraud=confirmed_fraud, metadata=metadata,
        )

    def add_many(self, records: list[tuple[str, np.ndarray, bool, dict]]) -> int:
        written = self._store.add_many(records)
        for case_id, leaves, confirmed, metadata in records:
            self._mirror.add(
                case_id, leaves, confirmed_fraud=confirmed, metadata=metadata
            )
        return written

    def search(self, leaf_assignment: np.ndarray, k: int = 5) -> list[SimilarCase]:
        return self._mirror.search(leaf_assignment, k=k)

    def search_vector(self, leaf_assignment: np.ndarray, k: int = 5) -> list[SimilarCase]:
        """k-NN executed by Postgres, through the HNSW index."""
        return self._store.search(leaf_assignment, k=k)

    def __len__(self) -> int:
        return len(self._mirror)


def leaf_assignment(booster, features) -> np.ndarray:
    """Which leaf of each tree this application falls into.

    ``pred_leaf=True`` asks LightGBM for leaf indices rather than a score, so
    this is a second forward pass over an already-loaded model - cheap enough
    to run inside the request path.
    """
    leaves = booster.predict(
        features, num_iteration=booster.best_iteration, pred_leaf=True
    )
    return np.asarray(leaves, dtype=np.int32).reshape(len(features), -1)


def explain_match(
    query_row, matched_metadata: dict[str, object], max_factors: int = 3
) -> list[str]:
    """Name the attributes a query and a matched case share.

    Leaf overlap says two applications are alike but not in what respect. An
    analyst needs the second part, so the shared categorical attributes are
    surfaced alongside the similarity score. Without this the match is an
    unexplained number, which is exactly what this system is meant to avoid.
    """
    interesting = (
        "device_os", "employment_status", "housing_status", "payment_type",
        "source", "email_is_free", "foreign_request", "phone_mobile_valid",
    )
    shared = [
        name for name in interesting
        if name in matched_metadata
        and name in query_row
        and matched_metadata[name] == query_row[name]
    ]
    return shared[:max_factors]
