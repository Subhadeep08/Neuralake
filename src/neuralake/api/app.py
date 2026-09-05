from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from neuralake.config.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    import logging

    from neuralake.storage.database import init_db, close_db
    from neuralake.workers.scheduler import start_scheduler, stop_scheduler

    settings = get_settings()
    if settings.environment != "development" and settings.auth.jwt_secret == "CHANGE-ME-IN-PRODUCTION":
        logging.getLogger("neuralake").critical(
            "JWT secret is still the default — set NEURALAKE_AUTH__JWT_SECRET before running in %s",
            settings.environment,
        )
        raise SystemExit(1)

    await init_db()
    await start_scheduler()
    yield
    await stop_scheduler()
    await close_db()


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="AI Knowledge Base with Memory Layer",
        lifespan=lifespan,
        debug=settings.debug,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from neuralake.api.middleware.logging import RequestLoggingMiddleware, configure_logging
    from neuralake.api.middleware.rate_limit import RateLimitMiddleware

    configure_logging(
        log_level=settings.log_level,
        json_format=settings.environment != "development",
    )

    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RateLimitMiddleware)

    from neuralake.api.routers import (
        analytics,
        collections,
        documents,
        graph,
        health,
        memories,
        search,
        tenants,
    )

    app.include_router(health.router, prefix="/api/v1")
    app.include_router(collections.router, prefix="/api/v1")
    app.include_router(documents.router, prefix="/api/v1")
    app.include_router(memories.router, prefix="/api/v1")
    app.include_router(search.router, prefix="/api/v1")
    app.include_router(graph.router, prefix="/api/v1")
    app.include_router(tenants.router, prefix="/api/v1")
    app.include_router(analytics.router, prefix="/api/v1")

    return app
