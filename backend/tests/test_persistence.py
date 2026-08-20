"""Tests for the persistence layer.

These run without a database. That is deliberate: the behaviour that matters
most here is what happens when Postgres is *absent or broken*, because a fraud
decision must still be made and returned in that case. A test suite that only
passes with a live database would never exercise the path this system is most
likely to take during an incident.

The live-database path is verified separately, against the real Supabase
instance, by scripts/verify_persistence.py.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from app.db.repository import DecisionRepository, _Write
from app.db.session import Database, _summarise
from app.services.similarity import (
    DurableSimilarityIndex,
    InMemorySimilarityIndex,
)


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


class _FakeDecision:
    """The subset of DecisionOut that the repository reads."""

    def __init__(self, decision: str = "APPROVE") -> None:
        self.decision = decision
        self.model_decision = decision
        self.fraud_probability = 0.0123
        self.risk_band = "low"
        self.thresholds = {"review": 0.004, "block": 0.031}
        self.escalated = False
        self.escalation_reason = None
        self.top_risk_factors = []
        self.model_version = "lgbm-test"
        self.latency_ms = 42.0


class _ExplodingStore:
    """A durable store whose writes always fail."""

    def __init__(self) -> None:
        self.attempts = 0

    def add(self, *args, **kwargs) -> None:
        self.attempts += 1
        raise RuntimeError("database is on fire")

    def load_all(self):
        return []


class _RecordingStore:
    """A durable store that captures what it was asked to persist."""

    def __init__(self, existing=None) -> None:
        self.written: list[tuple] = []
        self._existing = existing or []

    def add(self, case_id, leaves, *, confirmed_fraud, metadata=None) -> None:
        self.written.append((case_id, np.asarray(leaves), confirmed_fraud))

    def add_many(self, records) -> int:
        self.written.extend((r[0], np.asarray(r[1]), r[2]) for r in records)
        return len(records)

    def load_all(self):
        return self._existing


# ---------------------------------------------------------------------------
# the service must work without a database
# ---------------------------------------------------------------------------


class TestDegradesWithoutDatabase:
    def test_connect_without_a_url_yields_a_disabled_handle(self) -> None:
        database = Database.connect("")
        assert database.available is False
        assert database.healthy() is False

    def test_unreachable_database_does_not_raise(self) -> None:
        """A bad URL must be reported, not thrown, or the process never boots."""
        database = Database.connect(
            "postgresql://nobody:nothing@127.0.0.1:1/does_not_exist"
        )
        assert database.available is False

    def test_disabled_cursor_fails_loudly_if_used_anyway(self) -> None:
        database = Database.connect("")
        with pytest.raises(RuntimeError, match="not configured"):
            with database.cursor():
                pass

    def test_every_repository_write_is_a_noop(self) -> None:
        repository = DecisionRepository(Database(None))
        assert repository.available is False

        # None of these may raise, and none may block.
        repository.record_decision("APP-1", {"income": 0.3}, _FakeDecision())
        repository.record_verdict("APP-1", "analyst", "confirmed_fraud", indexed=False)
        repository.start()
        repository.stop()

        assert repository.recent_audit() == []
        assert repository.counts() == {}
        assert repository.stats["queued"] == 0


# ---------------------------------------------------------------------------
# credentials must never reach a log
# ---------------------------------------------------------------------------


class TestCredentialSafety:
    def test_summary_drops_the_password(self) -> None:
        summary = _summarise(
            "postgresql://postgres.abcdef:sup3rs3cret@db.example.com:5432/postgres"
        )
        assert "sup3rs3cret" not in summary
        assert "postgres.abcdef" not in summary
        assert "db.example.com" in summary

    def test_summary_survives_an_unparseable_url(self) -> None:
        assert _summarise("not a url at all") is not None

    def test_disabled_database_reports_no_summary(self) -> None:
        assert Database(None).summary == ""


# ---------------------------------------------------------------------------
# the write queue
# ---------------------------------------------------------------------------


class TestWriteQueue:
    def _enabled_repository(self) -> DecisionRepository:
        """A repository that believes it has a database but has no worker.

        Nothing drains the queue, so this isolates exactly what the request
        thread does.
        """
        repository = DecisionRepository(Database(None))
        repository._database = Database(object())    # available, never used
        return repository

    def test_recording_a_decision_enqueues_application_decision_and_audit(self) -> None:
        repository = self._enabled_repository()
        repository.record_decision("APP-1", {"income": 0.3}, _FakeDecision())

        kinds = [repository._queue.get_nowait().kind for _ in range(3)]
        assert kinds == ["application", "decision", "audit"]

    def test_verdict_enqueues_the_verdict_and_an_audit_entry(self) -> None:
        repository = self._enabled_repository()
        repository.record_verdict("APP-1", "analyst-7", "confirmed_fraud", indexed=True)

        kinds = [repository._queue.get_nowait().kind for _ in range(2)]
        assert kinds == ["verdict", "audit"]

    def test_a_full_queue_drops_and_counts_rather_than_blocking(self) -> None:
        """Saturation must never stall the decision path."""
        repository = self._enabled_repository()
        for index in range(repository._queue.maxsize):
            repository._queue.put_nowait(_Write("audit", (index,)))

        started = time.perf_counter()
        repository.record_decision("APP-OVERFLOW", {}, _FakeDecision())
        elapsed = time.perf_counter() - started

        assert elapsed < 0.05, "a saturated queue blocked the caller"
        assert repository._dropped > 0
        assert repository.stats["dropped"] > 0

    def test_batches_are_written_in_dependency_order(self) -> None:
        """A decision references an application, so applications go first."""
        repository = self._enabled_repository()
        executed: list[str] = []

        class _Cursor:
            def executemany(self, sql, rows):
                executed.append(sql.strip().split()[2])   # the table name

        class _Database:
            available = True

            def cursor(self, *, commit=True):
                from contextlib import contextmanager

                @contextmanager
                def _ctx():
                    yield _Cursor()

                return _ctx()

        repository._database = _Database()
        # Deliberately queued in the wrong order.
        repository._write_batch([
            _Write("audit", ("e", None, "system", "{}")),
            _Write("decision", tuple(range(12))),
            _Write("application", ("APP-1", "{}")),
        ])

        assert executed.index("applications") < executed.index("decisions")

    def test_a_failing_batch_is_swallowed_not_raised(self) -> None:
        repository = self._enabled_repository()

        class _Database:
            available = True

            def cursor(self, *, commit=True):
                raise RuntimeError("connection reset")

        repository._database = _Database()
        repository._write_batch([_Write("audit", ("e", None, "system", "{}"))])
        assert repository.stats["written"] == 0


# ---------------------------------------------------------------------------
# the durable similarity index
# ---------------------------------------------------------------------------


class TestDurableSimilarityIndex:
    def test_a_failed_durable_write_leaves_the_mirror_untouched(self) -> None:
        """The two must never disagree about what is indexed.

        If the mirror accepted a case the database rejected, the service would
        report a confirmed fraud as indexed and then lose it on restart - the
        exact failure the persistence layer exists to prevent.
        """
        store = _ExplodingStore()
        mirror = InMemorySimilarityIndex()
        index = DurableSimilarityIndex(store, mirror)

        with pytest.raises(RuntimeError):
            index.add("CASE-1", np.array([1, 2, 3]), confirmed_fraud=True)

        assert store.attempts == 1
        assert len(mirror) == 0
        assert len(index) == 0

    def test_a_successful_write_reaches_both(self) -> None:
        store = _RecordingStore()
        mirror = InMemorySimilarityIndex()
        index = DurableSimilarityIndex(store, mirror)

        index.add("CASE-1", np.array([1, 2, 3]), confirmed_fraud=True, metadata={"a": 1})

        assert len(store.written) == 1
        assert len(mirror) == 1
        assert len(index) == 1

    def test_restore_rebuilds_the_mirror_from_storage(self) -> None:
        existing = [
            ("CASE-1", np.array([1, 2, 3]), True, {"device_os": "linux"}),
            ("CASE-2", np.array([1, 2, 9]), True, {"device_os": "windows"}),
        ]
        index = DurableSimilarityIndex(_RecordingStore(existing), InMemorySimilarityIndex())

        assert index.restore() == 2
        assert len(index) == 2

    def test_a_restored_case_is_findable(self) -> None:
        """Restoration must preserve the exact leaf assignment, not an approximation."""
        leaves = np.array([4, 8, 15, 16, 23, 42])
        index = DurableSimilarityIndex(
            _RecordingStore([("CASE-1", leaves, True, {})]), InMemorySimilarityIndex()
        )
        index.restore()

        matches = index.search(leaves, k=1)
        assert matches[0].case_id == "CASE-1"
        assert matches[0].similarity == pytest.approx(1.0)

    def test_search_does_not_touch_the_durable_store(self) -> None:
        """The hot path must stay off the network."""
        store = _ExplodingStore()          # any call to it raises
        mirror = InMemorySimilarityIndex()
        mirror.add("CASE-1", np.array([1, 2, 3]), confirmed_fraud=True)
        index = DurableSimilarityIndex(store, mirror)

        assert index.search(np.array([1, 2, 3]), k=1)[0].case_id == "CASE-1"
        assert store.attempts == 0
