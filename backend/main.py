"""Development entry point.

The application itself lives in :mod:`app.main`. This file exists so the
server can be started with the simple command a developer expects::

    python main.py

For production, use an explicit ASGI server invocation instead::

    uvicorn app.main:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

from app.core.config import get_settings
from app.core.logging import get_logger
from app.main import app

logger = get_logger("entrypoint")

__all__ = ["app"]


def main() -> None:
    """Start the development server using configured host and port."""
    import uvicorn

    settings = get_settings()

    logger.info(
        "starting_development_server url=%s reload=%s",
        settings.server_url,
        settings.app_reload,
    )

    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_reload,
        log_config=None,  # logging is configured centrally in app.main
    )


if __name__ == "__main__":
    main()