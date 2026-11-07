"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import agents, ai, fact_check, health, research, script
from app.core.config import Settings, get_settings
from app.core.logging import RequestLoggingMiddleware, configure_logging, get_logger
from app.dependencies import close_services
from app.exceptions.handlers import register_exception_handlers

logger = get_logger("main")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and configure the FastAPI application.

    A factory is used rather than a module-level instance so tests can
    create isolated applications with their own settings.
    """
    app_settings = settings or get_settings()

    configure_logging(
        level=app_settings.log_level,
        json_output=app_settings.log_json,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        """Log startup and release resources on shutdown."""
        logger.info(
            "application_started name=%s env=%s host=%s port=%s",
            app_settings.app_name,
            app_settings.app_env,
            app_settings.app_host,
            app_settings.app_port,
        )
        logger.info("ai_provider_configured model=%s", app_settings.ollama_model)
        logger.info(
            "research_configured search_provider=%s max_sources=%s fetch_timeout=%ss",
            app_settings.search_provider,
            app_settings.max_research_sources,
            app_settings.source_fetch_timeout,
        )

        yield

        logger.info("application_shutdown_started")
        # Closes the HTTP clients behind the search and fetch providers,
        # and the AI provider's connection.
        await close_services()

    app = FastAPI(
        title=app_settings.app_name,
        version="2.0.0",
        description="Creator AI backend API.",
        lifespan=lifespan,
    )

    # CORS origins come from configuration. A wildcard is never used:
    # it would allow any site to call the API from a browser.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.allowed_origins,
        allow_credentials=True,
        allow_methods=app_settings.allow_methods,
        allow_headers=app_settings.allow_headers,
    )

    app.add_middleware(RequestLoggingMiddleware)

    register_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(ai.router)
    app.include_router(agents.router)
    app.include_router(research.router)
    app.include_router(fact_check.router)
    app.include_router(script.router)

    logger.info(
        "application_configured cors_origins=%s",
        ",".join(app_settings.allowed_origins),
    )

    return app


# Module-level ASGI application instance.
#
# This exists so that standard ASGI tooling can import the app directly::
#
#     uvicorn app.main:app
#
# The factory above remains the preferred entry point for tests, which
# need isolated application instances.
app = create_app()