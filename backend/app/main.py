"""FastAPI application factory.

Assembled as a factory rather than a module-level singleton so that tests can
build an app with overridden settings, and so a future ASGI deployment can
construct it with different configuration without importing side effects.

Everything environment-specific - allowed origins, keys, database, model
artifacts - arrives through ``Settings``. There is no hostname or port
hardcoded anywhere in this file, which is what makes moving this off a laptop
a configuration change rather than a code change.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Rate limiting is keyed on client address. On a scoring endpoint this is a
# security control, not just hygiene: unlimited queries let an attacker map the
# model's decision boundary and learn precisely what profile slips through.
limiter = Limiter(key_func=get_remote_address, default_limits=["120/minute"])


def _build_durable_index(booster, database):
    """Assemble the Postgres-backed similarity index for a loaded model.

    The index geometry is derived from the booster rather than configured: the
    number of trees and the leaves per tree define the one-hot space that gets
    projected, so a model change must change the projection with it. Reading
    these from config instead would let a retrained model be indexed against a
    stale geometry - and the resulting neighbours would be quietly meaningless
    rather than obviously broken.
    """
    from app.services.similarity import (
        DurableSimilarityIndex,
        PgVectorSimilarityIndex,
    )

    params = getattr(booster, "params", None) or {}
    store = PgVectorSimilarityIndex(
        connection_factory=database.connection,
        n_trees=booster.best_iteration,
        max_leaves=int(params.get("num_leaves", 255)),
    )
    return DurableSimilarityIndex(store)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load expensive, long-lived resources once per process.

    Model artifacts are deserialised at startup rather than per request. A
    fraud decision has a latency budget in the tens of milliseconds; loading a
    booster from disk inside the request path would blow it entirely.

    The same reasoning applies to the database: one pooled connection is
    established here, not per write.

    Every step below degrades rather than aborts. Missing artifacts disable
    scoring but keep /health and /docs reachable; an unreachable database drops
    the service back to an in-memory index. A process that refuses to start is
    harder to diagnose than one that reports exactly what it is missing - and
    during a live demonstration it is unrecoverable.
    """
    settings: Settings = getattr(app.state, "settings", None) or get_settings()
    logger.info("starting %s in %s mode", settings.app_name, settings.app_env)

    from app.db import schema
    from app.db.repository import DecisionRepository
    from app.db.session import Database

    # -- persistence ----------------------------------------------------
    database = Database.connect(settings.database_url)
    if database.available and not schema.apply(database):
        database.close()
        database = Database(None)

    app.state.database = database
    app.state.repository = DecisionRepository(database)
    app.state.repository.start()

    # -- model ----------------------------------------------------------
    # Kept on app.state so request handlers reach it without global imports.
    app.state.model_bundle = None
    try:
        from app.services.scoring import ScoringService

        service = ScoringService.from_artifacts(settings)

        # The durable index needs the booster's geometry, so it is attached
        # after the model is loaded rather than injected at construction.
        if database.available:
            try:
                index = _build_durable_index(service.booster, database)
                restored = index.restore()
                service.similarity_index = index
                logger.info("similarity index restored from Postgres: %d cases", restored)
            except Exception:   # noqa: BLE001
                logger.exception(
                    "durable similarity index unavailable; using in-memory index "
                    "(confirmed frauds will not survive a restart)"
                )

        app.state.model_bundle = service
        logger.info(
            "model loaded: %s  (review>=%.5f, block>=%.5f)",
            service.model_version,
            service.policy.tau_review,
            service.policy.tau_block,
        )
        # A fraud team does not start with an empty case history. Seeding the
        # similarity index with previously confirmed fraud is what makes
        # lookalike detection useful from the first request rather than only
        # after an analyst has manually confirmed something. Skipped when the
        # index was restored from storage.
        seeded = service.seed_similarity_index()
        logger.info("similarity index: %d confirmed cases", seeded)
    except FileNotFoundError as error:
        # Start anyway so /health and /docs stay reachable; scoring routes
        # return 503 with an actionable message.
        logger.error("scoring disabled - %s", error)
    except Exception:   # noqa: BLE001
        logger.exception("scoring disabled - failed to load model artifacts")

    yield

    logger.info("shutting down %s", settings.app_name)
    app.state.repository.stop()
    database.close()


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application. Pass ``settings`` to override configuration in tests."""
    settings = settings or get_settings()

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        summary="Adaptive application-fraud decisioning for digital lending.",
        description=(
            "Scores digital lending applications in real time and returns an "
            "auditable decision with per-case explanations.\n\n"
            "Decision thresholds are derived from business costs rather than "
            "tuned. When a database is configured, every decision and every "
            "analyst verdict is written to an audit log that Postgres itself "
            "refuses to update or delete; `GET /api/v1/storage` reports "
            "whether that recording is currently active."
        ),
        lifespan=lifespan,
        # Interactive docs are the API-first contract; harmless locally, but
        # they should not be world-readable in a real deployment.
        docs_url="/docs" if settings.app_env != "production" else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.app_env != "production" else None,
    )

    # Stashed so the lifespan handler uses the same settings the app was built
    # with, rather than re-reading the environment and diverging in tests.
    app.state.settings = settings

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Explicit origins only - never a wildcard. `allow_credentials=True`
    # combined with `allow_origins=["*"]` is rejected by browsers anyway, and
    # a wildcard on an endpoint returning personal risk assessments would be
    # indefensible in review.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["X-API-Key", "Content-Type"],
    )

    from app.api.v1.routes import router as v1_router

    app.include_router(v1_router)

    @app.get("/health", tags=["operations"], summary="Liveness probe")
    async def health() -> dict[str, object]:
        """Unauthenticated so orchestrators can probe it.

        Deliberately reveals nothing beyond liveness and whether a model is
        loaded - no versions, no configuration, no paths.
        """
        # getattr with a default: a liveness probe must answer even if startup
        # has not finished populating state. A health endpoint that raises is
        # worse than useless - it makes an orchestrator kill a healthy process.
        #
        # Persistence status is deliberately NOT reported here. Whether this
        # service has a database behind it is infrastructure topology, and this
        # endpoint is unauthenticated; `GET /api/v1/storage` answers the same
        # question behind an API key, which is where that answer belongs.
        return {
            "status": "ok",
            "model_loaded": getattr(app.state, "model_bundle", None) is not None,
        }

    return app


app = create_app()
