"""Tests for configuration loading and validation."""

from __future__ import annotations

import pytest

from app.core.config import (
    Settings,
    get_settings,
    parse_bool,
    reset_settings_cache,
    split_csv,
)


# ---------- Defaults preserve existing behaviour ----------


def test_defaults_match_original_configuration() -> None:
    """Defaults keep the original working values."""
    settings = Settings()

    assert settings.ollama_host == "http://127.0.0.1:11434"
    assert settings.ollama_model == "qwen3:8b"
    assert settings.app_host == "127.0.0.1"
    assert settings.app_port == 8000
    assert settings.app_env == "development"


def test_default_cors_origins_support_vite() -> None:
    """Both Vite dev-server origins are allowed by default."""
    settings = Settings()

    assert "http://localhost:5173" in settings.allowed_origins
    assert "http://127.0.0.1:5173" in settings.allowed_origins


def test_defaults_do_not_use_wildcard_origin() -> None:
    """A wildcard origin is never introduced by default."""
    settings = Settings()

    assert "*" not in settings.allowed_origins


# ---------- Environment overrides ----------


def test_environment_variables_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real environment variables take precedence over defaults."""
    monkeypatch.setenv("OLLAMA_MODEL", "llama3:8b")
    monkeypatch.setenv("APP_PORT", "9000")
    reset_settings_cache()

    settings = get_settings()

    assert settings.ollama_model == "llama3:8b"
    assert settings.app_port == 9000


def test_settings_are_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_settings returns the same instance until the cache is cleared."""
    reset_settings_cache()
    first = get_settings()
    assert get_settings() is first

    reset_settings_cache()
    assert get_settings() is not first


# ---------- Parsing helpers ----------


@pytest.mark.parametrize("raw", ["true", "True", "TRUE", "1", "yes", "on"])
def test_parse_bool_true_values(raw: str) -> None:
    """Truthy spellings are recognised."""
    assert parse_bool(raw) is True


@pytest.mark.parametrize("raw", ["false", "False", "0", "no", "off"])
def test_parse_bool_false_values(raw: str) -> None:
    """Falsy spellings are recognised.

    This matters because ``bool("false")`` is ``True`` in Python, which
    would silently invert a configuration flag.
    """
    assert parse_bool(raw) is False


def test_parse_bool_rejects_invalid_value() -> None:
    """An unrecognised boolean raises instead of defaulting to True."""
    with pytest.raises(ValueError):
        parse_bool("maybe")


def test_split_csv_trims_and_filters() -> None:
    """CSV parsing removes whitespace and empty entries."""
    assert split_csv(" a , b ,, c ") == ["a", "b", "c"]


# ---------- Validation ----------


def test_invalid_log_level_is_rejected() -> None:
    """An unknown log level fails fast."""
    with pytest.raises(ValueError, match="LOG_LEVEL"):
        Settings(log_level="VERBOSE")


def test_log_level_is_normalised_to_uppercase() -> None:
    """Log levels are case-insensitive."""
    assert Settings(log_level="debug").log_level == "DEBUG"


def test_ollama_host_requires_scheme() -> None:
    """A host without http:// or https:// is rejected."""
    with pytest.raises(ValueError, match="OLLAMA_HOST"):
        Settings(ollama_host="127.0.0.1:11434")


def test_ollama_host_trailing_slash_is_normalised() -> None:
    """A trailing slash is removed to avoid a double slash in URLs."""
    assert Settings(ollama_host="http://127.0.0.1:11434/").ollama_host == (
        "http://127.0.0.1:11434"
    )


def test_empty_model_is_rejected() -> None:
    """An empty model name is rejected."""
    with pytest.raises(ValueError):
        Settings(ollama_model="   ")


def test_invalid_port_is_rejected() -> None:
    """A port outside the valid range is rejected."""
    with pytest.raises(ValueError):
        Settings(app_port=70000)


def test_invalid_environment_name_is_rejected() -> None:
    """An unknown environment name is rejected."""
    with pytest.raises(ValueError):
        Settings(app_env="staging")


def test_invalid_app_env_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """A typo in APP_ENV surfaces at startup rather than later."""
    monkeypatch.setenv("APP_ENV", "dev")
    reset_settings_cache()

    with pytest.raises(ValueError):
        get_settings()


# ---------- Derived properties ----------


def test_cors_origins_come_from_configuration() -> None:
    """Origins are configurable rather than fixed in middleware."""
    settings = Settings(frontend_url="https://app.example.com")

    assert settings.allowed_origins == ["https://app.example.com"]


def test_environment_helpers() -> None:
    """The development and production flags reflect APP_ENV."""
    assert Settings(app_env="development").is_development is True
    assert Settings(app_env="development").is_production is False
    assert Settings(app_env="production").is_production is True


def test_server_url_is_derived() -> None:
    """The server URL is composed from host and port."""
    assert Settings(app_host="0.0.0.0", app_port=8080).server_url == (
        "http://0.0.0.0:8080"
    )


def test_multiple_cors_origins() -> None:
    """Several origins can be configured at once."""
    settings = Settings(frontend_url="http://localhost:5173, http://127.0.0.1:5173")

    assert settings.allowed_origins == [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]


# ---------- dotenv parsing ----------


def test_dotenv_parsing_handles_comments_and_quotes(tmp_path) -> None:
    """Comments, quotes and export prefixes are handled."""
    import os

    from app.core.config import load_dotenv

    # Save and restore these keys around the test. load_dotenv() writes to
    # os.environ, so without this the values persist and leak into
    # unrelated tests (for example the service name in a health check).
    saved = {key: os.environ.get(key) for key in ("OLLAMA_MODEL", "APP_NAME")}

    try:
        os.environ.pop("OLLAMA_MODEL", None)
        os.environ.pop("APP_NAME", None)

        env_file = tmp_path / ".env"
        env_file.write_text(
            '# a comment\n'
            'OLLAMA_MODEL="test-model"\n'
            "export APP_NAME='Quoted Name'\n"
            "\n"
            "MALFORMED_LINE_WITHOUT_EQUALS\n",
            encoding="utf-8",
        )

        load_dotenv(env_file)

        assert os.environ["OLLAMA_MODEL"] == "test-model"
        assert os.environ["APP_NAME"] == "Quoted Name"
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


    """A missing .env file does not raise."""
    from app.core.config import load_dotenv

    load_dotenv(tmp_path / "does-not-exist.env")


def test_existing_environment_wins_over_dotenv(tmp_path, monkeypatch) -> None:
    """setdefault semantics ensure the real environment takes priority."""
    import os

    from app.core.config import load_dotenv

    monkeypatch.setenv("OLLAMA_MODEL", "from-environment")

    env_file = tmp_path / ".env"
    env_file.write_text("OLLAMA_MODEL=from-file\n", encoding="utf-8")

    load_dotenv(env_file)

    assert os.environ["OLLAMA_MODEL"] == "from-environment"
