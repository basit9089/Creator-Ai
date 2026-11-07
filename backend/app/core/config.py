"""Typed application configuration.

All environment-specific values are read from environment variables
(or a local ``.env`` file) exactly once, at import time, and exposed
through the immutable :class:`Settings` object.

Design notes
------------
``pydantic-settings`` is intentionally NOT used. The project only needs
plain environment-variable reading with type coercion, which can be
implemented with the standard library plus the already-installed
Pydantic. This keeps the dependency surface minimal.

Validation is strict and fails fast: an invalid value raises a clear
error at startup rather than causing confusing behaviour later.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    from app.services.ai.provider import GenerationOptions

# Environment name is intentionally a small closed set. An unexpected
# value is almost always a typo, and silently accepting it could weaken
# production behaviour (for example, disabling reload or debug logging).
EnvironmentType = Literal["development", "test", "production"]

#: Which search backend the Researcher retrieves candidates with. Kept a
#: closed set so a typo fails at startup instead of silently producing an
#: agent that returns nothing. ``"fallback"`` means "try the free web
#: search, then the free encyclopaedia", so one blocked provider does not
#: make research impossible.
SearchProviderName = Literal["duckduckgo", "wikipedia", "fallback"]

#: Directory containing the backend application (the repository's
#: ``backend`` folder). Used to locate the local ``.env`` file.
BACKEND_DIR: Final[Path] = Path(__file__).resolve().parent.parent.parent

#: Name of the local, git-ignored environment file.
ENV_FILE_NAME: Final[str] = ".env"

#: Defaults preserve the behaviour of the original working application.
DEFAULT_VALUES: Final[dict[str, str]] = {
    "APP_NAME": "Creator AI API",
    "APP_ENV": "development",
    "APP_HOST": "127.0.0.1",
    "APP_PORT": "8000",
    "APP_RELOAD": "true",
    "LOG_LEVEL": "INFO",
    "LOG_JSON": "false",
    "OLLAMA_HOST": "http://127.0.0.1:11434",
    "OLLAMA_MODEL": "qwen3:8b",
    "OLLAMA_TIMEOUT": "300",
    "FRONTEND_URL": "http://localhost:5173,http://127.0.0.1:5173",
    "CORS_ALLOW_METHODS": "GET,POST,PUT,PATCH,DELETE,OPTIONS",
    "CORS_ALLOW_HEADERS": "*",
    "MAX_PROMPT_LENGTH": "8000",
    "DEFAULT_AI_TEMPERATURE": "0.7",
    "DEFAULT_AI_TOP_P": "0.9",
    "DEFAULT_AI_THINKING": "false",
    "MAX_AI_OUTPUT_TOKENS": "2048",
    "AI_REQUEST_TIMEOUT": "600",
    # ---------- Research retrieval ----------
    # Bounded on purpose: research runs on a laptop alongside a local 8B
    # model, so the defaults must not be able to saturate the machine or
    # spend minutes waiting on the open web.
    "SEARCH_PROVIDER": "duckduckgo",
    "SEARCH_TIMEOUT": "15",
    "SEARCH_MAX_RESULTS": "30",
    "MAX_RESEARCH_SOURCES": "12",
    "MAX_CONCURRENT_SOURCE_FETCHES": "4",
    "SOURCE_FETCH_TIMEOUT": "20",
    "MAX_SOURCE_SIZE": "2000000",
    "MAX_EXTRACTED_TEXT": "20000",
    "MAX_SOURCE_REDIRECTS": "5",
    "MIN_USABLE_SOURCES": "2",
}


def parse_bool(raw: str) -> bool:
    """Interpret common textual boolean spellings.

    ``bool("false")`` would be ``True``, which is a classic source of
    silent misconfiguration, so values are compared explicitly.
    """
    normalised = raw.strip().lower()
    if normalised in {"1", "true", "yes", "on"}:
        return True
    if normalised in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Cannot interpret {raw!r} as a boolean value")


def split_csv(raw: str) -> list[str]:
    """Split a comma-separated value, dropping empty entries."""
    return [item.strip() for item in raw.split(",") if item.strip()]


def load_dotenv(path: Path = BACKEND_DIR / ENV_FILE_NAME) -> None:
    """Load ``KEY=VALUE`` pairs from ``path`` into ``os.environ``.

    Existing environment variables always win, so a real environment
    (CI, Docker, shell) overrides the file without special handling.

    Supports ``#`` comments, blank lines, optional ``export`` prefixes
    and quoted values. Malformed lines are ignored rather than fatal so
    that a stray character cannot stop the API from starting.
    """
    if not path.is_file():
        return

    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        # An unreadable .env should not crash startup; defaults apply.
        return

    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        key, _, value = stripped.partition("=")
        key = key.strip().removeprefix("export ").strip()
        if not key:
            continue

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]

        os.environ.setdefault(key, value)


class Settings(BaseModel):
    """Application settings resolved from the environment.

    Defaults mirror the original working configuration, so the service
    behaves identically even when no ``.env`` file exists.
    """

    # ---------- Application ----------
    app_name: str = DEFAULT_VALUES["APP_NAME"]
    app_env: EnvironmentType = DEFAULT_VALUES["APP_ENV"]  # type: ignore[assignment]
    app_host: str = DEFAULT_VALUES["APP_HOST"]
    app_port: int = Field(default=int(DEFAULT_VALUES["APP_PORT"]), ge=1, le=65535)
    app_reload: bool = Field(default=parse_bool(DEFAULT_VALUES["APP_RELOAD"]))

    # ---------- Logging ----------
    log_level: str = DEFAULT_VALUES["LOG_LEVEL"]
    log_json: bool = Field(default=parse_bool(DEFAULT_VALUES["LOG_JSON"]))

    # ---------- Ollama / AI provider ----------
    # Server-side only. These values are never returned to the browser.
    ollama_host: str = DEFAULT_VALUES["OLLAMA_HOST"]
    ollama_model: str = DEFAULT_VALUES["OLLAMA_MODEL"]
    ollama_timeout: float = Field(default=float(DEFAULT_VALUES["OLLAMA_TIMEOUT"]), gt=0)

    # ---------- CORS ----------
    frontend_url: str = DEFAULT_VALUES["FRONTEND_URL"]
    cors_allow_methods: str = DEFAULT_VALUES["CORS_ALLOW_METHODS"]
    cors_allow_headers: str = DEFAULT_VALUES["CORS_ALLOW_HEADERS"]

    # ---------- Limits ----------
    max_prompt_length: int = Field(default=int(DEFAULT_VALUES["MAX_PROMPT_LENGTH"]), ge=1)

    # ---------- Default AI generation settings ----------
    # These are the defaults applied to any generation that does not
    # specify its own options. Individual calls may always override them,
    # which is how an agent opts into (or out of) reasoning mode.
    default_ai_temperature: float = Field(
        default=float(DEFAULT_VALUES["DEFAULT_AI_TEMPERATURE"]), ge=0.0, le=2.0
    )
    default_ai_top_p: float = Field(
        default=float(DEFAULT_VALUES["DEFAULT_AI_TOP_P"]), gt=0.0, le=1.0
    )
    default_ai_thinking: bool = Field(
        default=parse_bool(DEFAULT_VALUES["DEFAULT_AI_THINKING"])
    )
    max_ai_output_tokens: int = Field(
        default=int(DEFAULT_VALUES["MAX_AI_OUTPUT_TOKENS"]), ge=1
    )

    #: Upper bound on a single agent or API generation, in seconds.
    #: Agents make longer chains of calls than a single chat request,
    #: so this is more generous than the provider connection timeout.
    ai_request_timeout: float = Field(default=float(DEFAULT_VALUES["AI_REQUEST_TIMEOUT"]), gt=0)

    # ---------- Research retrieval ----------
    # The Researcher retrieves web material itself. These bounds decide how
    # much of the web a single request may touch, which is the difference
    # between a research agent and a way to hang the laptop on a bad page.
    search_provider: SearchProviderName = DEFAULT_VALUES["SEARCH_PROVIDER"]  # type: ignore[assignment]
    search_timeout: float = Field(default=float(DEFAULT_VALUES["SEARCH_TIMEOUT"]), gt=0)
    search_max_results: int = Field(default=int(DEFAULT_VALUES["SEARCH_MAX_RESULTS"]), ge=1)
    max_research_sources: int = Field(default=int(DEFAULT_VALUES["MAX_RESEARCH_SOURCES"]), ge=1)
    max_concurrent_source_fetches: int = Field(
        default=int(DEFAULT_VALUES["MAX_CONCURRENT_SOURCE_FETCHES"]), ge=1
    )
    source_fetch_timeout: float = Field(default=float(DEFAULT_VALUES["SOURCE_FETCH_TIMEOUT"]), gt=0)
    max_source_size: int = Field(default=int(DEFAULT_VALUES["MAX_SOURCE_SIZE"]), ge=1)
    max_extracted_text: int = Field(default=int(DEFAULT_VALUES["MAX_EXTRACTED_TEXT"]), ge=1)
    max_source_redirects: int = Field(default=int(DEFAULT_VALUES["MAX_SOURCE_REDIRECTS"]), ge=0)
    min_usable_sources: int = Field(default=int(DEFAULT_VALUES["MIN_USABLE_SOURCES"]), ge=1)

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = value.strip().upper()
        if upper not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {sorted(allowed)}, got {value!r}")
        return upper

    @field_validator("ollama_host")
    @classmethod
    def _validate_ollama_host(cls, value: str) -> str:
        host = value.strip().rstrip("/")
        if not host.startswith(("http://", "https://")):
            raise ValueError(
                f"OLLAMA_HOST must start with http:// or https://, got {value!r}"
            )
        return host

    @field_validator("ollama_model", "app_name")
    @classmethod
    def _validate_not_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Value must not be empty")
        return stripped

    # ---------- Derived helpers ----------

    @property
    def allowed_origins(self) -> list[str]:
        """Browser origins permitted by CORS.

        Derived from configuration. The default supports the Vite dev
        server on both ``localhost`` and ``127.0.0.1``. A wildcard is
        never injected automatically.
        """
        return split_csv(self.frontend_url)

    @property
    def allow_methods(self) -> list[str]:
        return split_csv(self.cors_allow_methods)

    @property
    def allow_headers(self) -> list[str]:
        return split_csv(self.cors_allow_headers)

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"

    @property
    def server_url(self) -> str:
        """URL used when starting the development server."""
        return f"http://{self.app_host}:{self.app_port}"

    def default_generation_options(self) -> Any:
        """Build the default :class:`GenerationOptions` for this deployment.

        Imported lazily to avoid a circular import: ``provider`` depends on
        ``config`` for these values, and ``config`` must not depend on
        ``provider`` at module level.
        """
        from app.services.ai.provider import GenerationOptions

        return GenerationOptions(
            temperature=self.default_ai_temperature,
            top_p=self.default_ai_top_p,
            max_tokens=self.max_ai_output_tokens,
            think=self.default_ai_thinking,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings.

    ``.env`` is loaded before the first read so that file values are
    available, while real environment variables still take precedence.
    """
    load_dotenv()
    return Settings(
        app_name=os.getenv("APP_NAME", DEFAULT_VALUES["APP_NAME"]),
        app_env=os.getenv("APP_ENV", DEFAULT_VALUES["APP_ENV"]),
        app_host=os.getenv("APP_HOST", DEFAULT_VALUES["APP_HOST"]),
        app_port=int(os.getenv("APP_PORT", DEFAULT_VALUES["APP_PORT"])),
        app_reload=parse_bool(os.getenv("APP_RELOAD", DEFAULT_VALUES["APP_RELOAD"])),
        log_level=os.getenv("LOG_LEVEL", DEFAULT_VALUES["LOG_LEVEL"]),
        log_json=parse_bool(os.getenv("LOG_JSON", DEFAULT_VALUES["LOG_JSON"])),
        ollama_host=os.getenv("OLLAMA_HOST", DEFAULT_VALUES["OLLAMA_HOST"]),
        ollama_model=os.getenv("OLLAMA_MODEL", DEFAULT_VALUES["OLLAMA_MODEL"]),
        ollama_timeout=float(os.getenv("OLLAMA_TIMEOUT", DEFAULT_VALUES["OLLAMA_TIMEOUT"])),
        frontend_url=os.getenv("FRONTEND_URL", DEFAULT_VALUES["FRONTEND_URL"]),
        cors_allow_methods=os.getenv("CORS_ALLOW_METHODS", DEFAULT_VALUES["CORS_ALLOW_METHODS"]),
        cors_allow_headers=os.getenv("CORS_ALLOW_HEADERS", DEFAULT_VALUES["CORS_ALLOW_HEADERS"]),
        max_prompt_length=int(os.getenv("MAX_PROMPT_LENGTH", DEFAULT_VALUES["MAX_PROMPT_LENGTH"])),
        default_ai_temperature=float(
            os.getenv("DEFAULT_AI_TEMPERATURE", DEFAULT_VALUES["DEFAULT_AI_TEMPERATURE"])
        ),
        default_ai_top_p=float(
            os.getenv("DEFAULT_AI_TOP_P", DEFAULT_VALUES["DEFAULT_AI_TOP_P"])
        ),
        default_ai_thinking=parse_bool(
            os.getenv("DEFAULT_AI_THINKING", DEFAULT_VALUES["DEFAULT_AI_THINKING"])
        ),
        max_ai_output_tokens=int(
            os.getenv("MAX_AI_OUTPUT_TOKENS", DEFAULT_VALUES["MAX_AI_OUTPUT_TOKENS"])
        ),
        ai_request_timeout=float(
            os.getenv("AI_REQUEST_TIMEOUT", DEFAULT_VALUES["AI_REQUEST_TIMEOUT"])
        ),
        search_provider=os.getenv("SEARCH_PROVIDER", DEFAULT_VALUES["SEARCH_PROVIDER"]),
        search_timeout=float(os.getenv("SEARCH_TIMEOUT", DEFAULT_VALUES["SEARCH_TIMEOUT"])),
        search_max_results=int(
            os.getenv("SEARCH_MAX_RESULTS", DEFAULT_VALUES["SEARCH_MAX_RESULTS"])
        ),
        max_research_sources=int(
            os.getenv("MAX_RESEARCH_SOURCES", DEFAULT_VALUES["MAX_RESEARCH_SOURCES"])
        ),
        max_concurrent_source_fetches=int(
            os.getenv(
                "MAX_CONCURRENT_SOURCE_FETCHES",
                DEFAULT_VALUES["MAX_CONCURRENT_SOURCE_FETCHES"],
            )
        ),
        source_fetch_timeout=float(
            os.getenv("SOURCE_FETCH_TIMEOUT", DEFAULT_VALUES["SOURCE_FETCH_TIMEOUT"])
        ),
        max_source_size=int(os.getenv("MAX_SOURCE_SIZE", DEFAULT_VALUES["MAX_SOURCE_SIZE"])),
        max_extracted_text=int(
            os.getenv("MAX_EXTRACTED_TEXT", DEFAULT_VALUES["MAX_EXTRACTED_TEXT"])
        ),
        max_source_redirects=int(
            os.getenv("MAX_SOURCE_REDIRECTS", DEFAULT_VALUES["MAX_SOURCE_REDIRECTS"])
        ),
        min_usable_sources=int(
            os.getenv("MIN_USABLE_SOURCES", DEFAULT_VALUES["MIN_USABLE_SOURCES"])
        ),
    )


def reset_settings_cache() -> None:
    """Clear the cached settings.

    Used by tests that manipulate environment variables.
    """
    get_settings.cache_clear()