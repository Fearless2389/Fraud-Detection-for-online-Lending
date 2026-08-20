"""Postgres connectivity.

One pool per process, created at startup, handed to repositories that need it.

Two properties of this module matter more than anything else in it:

**The database is optional.** ``Database.connect`` returns a disabled instance
when no ``DATABASE_URL`` is configured or when the server cannot be reached, and
every caller is written to cope with that. A fraud console that refuses to start
because a cloud database blinked is worse than one that runs with an in-memory
index and says so - especially during a live demonstration, where the failure is
unrecoverable and public.

**Connections are pooled.** This service talks to a managed Postgres over the
public internet; establishing a TLS connection costs a hundred milliseconds or
more, against a decision path budgeted in tens. Opening a connection per write
would make persistence the slowest thing the system does.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from collections.abc import Iterator
from typing import Any

logger = logging.getLogger(__name__)

# Supabase's session pooler tolerates far more, but a prototype has no reason to
# hold many idle connections open against a free-tier instance.
MIN_POOL_SIZE = 1
MAX_POOL_SIZE = 4

# Fail fast. A demo cannot wait thirty seconds to discover the network is down.
CONNECT_TIMEOUT_SECONDS = 10


class Database:
    """A pooled Postgres handle that degrades to a no-op when unavailable.

    Callers never branch on configuration. They check :attr:`available` if they
    have a fallback, or simply use :meth:`cursor`, which raises a clear error
    when the pool was never established.
    """

    def __init__(self, pool: Any | None, url_summary: str = "") -> None:
        self._pool = pool
        self._url_summary = url_summary

    # -- lifecycle ---------------------------------------------------------

    @classmethod
    def connect(cls, database_url: str) -> "Database":
        """Build a pool, or return a disabled handle with the reason logged.

        Never raises. Startup calls this, and a startup path that can throw on a
        network condition is a startup path that fails in the demonstration
        rather than in testing.
        """
        if not database_url:
            logger.info(
                "no DATABASE_URL configured; running without persistence "
                "(similarity index will be in-memory and will not survive restart)"
            )
            return cls(None)

        try:
            from psycopg_pool import ConnectionPool
        except ImportError:
            logger.warning("psycopg_pool is not installed; persistence disabled")
            return cls(None)

        summary = _summarise(database_url)
        try:
            pool = ConnectionPool(
                conninfo=database_url,
                min_size=MIN_POOL_SIZE,
                max_size=MAX_POOL_SIZE,
                kwargs={"connect_timeout": CONNECT_TIMEOUT_SECONDS},
                # Verify reachability here rather than on the first write, so a
                # misconfigured URL is reported at boot with a clear message.
                open=True,
                timeout=CONNECT_TIMEOUT_SECONDS,
                name="aegis",
            )
            pool.wait(timeout=CONNECT_TIMEOUT_SECONDS)
        except Exception as error:      # noqa: BLE001 - degrade, never crash
            logger.warning(
                "database unreachable at %s (%s: %s); continuing without "
                "persistence", summary, type(error).__name__, error,
            )
            return cls(None)

        logger.info("database pool ready: %s", summary)
        return cls(pool, summary)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    # -- access ------------------------------------------------------------

    @property
    def available(self) -> bool:
        return self._pool is not None

    @property
    def summary(self) -> str:
        """Host and database, never credentials. Safe to log or display."""
        return self._url_summary

    def connection(self):
        """Borrow a raw connection as a context manager.

        Exists for code that manages its own cursors and commits - notably
        :class:`~app.services.similarity.PgVectorSimilarityIndex`, which is
        written against a plain connection factory so it can be unit-tested
        against any Postgres without depending on this class.
        """
        if self._pool is None:
            raise RuntimeError("database is not configured")
        return self._pool.connection()

    @contextmanager
    def cursor(self, *, commit: bool = True) -> Iterator[Any]:
        """Borrow a connection and yield a cursor, committing on clean exit.

        The pool returns the connection automatically. On an exception the
        transaction is rolled back by psycopg's own context manager, so a failed
        write can never leave a half-applied audit record behind.
        """
        if self._pool is None:
            raise RuntimeError("database is not configured")

        with self._pool.connection() as connection:
            with connection.cursor() as cursor:
                yield cursor
            if commit:
                connection.commit()

    def healthy(self) -> bool:
        """Round-trip a trivial query. Used by the health endpoint."""
        if self._pool is None:
            return False
        try:
            with self.cursor(commit=False) as cursor:
                cursor.execute("SELECT 1;")
                return cursor.fetchone()[0] == 1
        except Exception:               # noqa: BLE001
            logger.warning("database health check failed", exc_info=True)
            return False


def _summarise(database_url: str) -> str:
    """``host:port/database`` - the parts of a URL that are safe to log.

    Credentials are dropped deliberately. Connection strings end up in log
    aggregators, crash reports and screen recordings; a password that is never
    formatted into a string is a password that cannot leak through any of them.
    """
    from urllib.parse import urlsplit

    try:
        parts = urlsplit(database_url)
        return f"{parts.hostname}:{parts.port or 5432}{parts.path}"
    except Exception:                   # noqa: BLE001
        return "<unparseable>"
