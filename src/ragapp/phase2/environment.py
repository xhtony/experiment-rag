"""Explicit configuration selection without falling back to Phase 1 secrets."""

import os
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values, load_dotenv
from sqlalchemy.engine import make_url

from ragapp.config import Settings, get_settings

ROOT = Path(__file__).resolve().parents[3]


def configure(env_file: str | Path = ".env.phase2") -> Settings:
    path = Path(env_file).resolve()
    if not path.is_file():
        raise ValueError(f"Configuration file not found: {path.name}")
    if path == (ROOT / ".env").resolve():
        raise ValueError("Phase 2 must not use the Phase 1 .env file")
    values = dotenv_values(path)
    if not values.get("RAGAPP_DATABASE_URL") or not values.get("RAGAPP_QDRANT_URL"):
        raise ValueError("Phase 2 configuration must explicitly set both database URLs")
    os.environ["RAGAPP_ENV_FILE"] = str(path)
    load_dotenv(path, override=True)
    get_settings.cache_clear()
    settings = get_settings()
    if not (0 <= settings.chunk_overlap < settings.chunk_size):
        raise ValueError("Require 0 <= chunk_overlap < chunk_size")
    if not (0 < settings.top_k <= settings.candidate_k):
        raise ValueError("Require 0 < top_k <= candidate_k")
    if settings.max_context_chars < 1 or settings.c3_max_retries < 0:
        raise ValueError("Invalid context or corrective retry budget")
    if settings.phase2_max_attempts < 1:
        raise ValueError("phase2_max_attempts must be positive")
    if settings.phase2_seed < 0:
        raise ValueError("phase2_seed must be nonnegative")
    assert_isolated(settings)
    return settings


def _db_location(url):
    parsed = make_url(url)
    host = "localhost" if parsed.host in {"127.0.0.1", "::1"} else parsed.host
    return (parsed.drivername.split("+")[0], host, parsed.port or 5432, parsed.database)


def _vector_location(url, collection):
    parsed = urlsplit(url)
    host = "localhost" if parsed.hostname in {"127.0.0.1", "::1"} else parsed.hostname
    return (host, parsed.port or (443 if parsed.scheme == "https" else 80), parsed.path, collection)


def assert_isolated(settings: Settings):
    old = dotenv_values(ROOT / ".env") if (ROOT / ".env").exists() else {}
    old_db = old.get("RAGAPP_DATABASE_URL") or Settings.model_fields["database_url"].default
    old_q = old.get("RAGAPP_QDRANT_URL") or Settings.model_fields["qdrant_url"].default
    old_collection = old.get("RAGAPP_QDRANT_COLLECTION") or Settings.model_fields["qdrant_collection"].default
    if settings.database_url.startswith("sqlite:"):
        return  # SQLite is used only for isolated offline tests.
    if _db_location(settings.database_url) == _db_location(old_db):
        raise ValueError("Phase 2 database points to the Phase 1 database; refusing writes")
    if _vector_location(settings.qdrant_url, settings.qdrant_collection) == _vector_location(old_q, old_collection):
        raise ValueError("Phase 2 vector index points to Phase 1; refusing writes")


def require_api_key():
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise ValueError("No active OPENAI_API_KEY. Set it in .env.phase2 or the process environment.")
    if any(os.environ.get(name, "").lower() in {"1", "true", "yes"} for name in (
        "RAGAPP_USE_FAKE_LLM", "RAGAPP_USE_FAKE_EMBEDDINGS"
    )):
        raise ValueError("Fake model flags cannot be used for recorded Phase 2 experiments")
