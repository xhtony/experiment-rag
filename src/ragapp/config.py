"""Central configuration. All experimental parameters live here so that every
run can be reproduced from a single logged config snapshot."""

from functools import lru_cache
import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAGAPP_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # LLM (identical across all conditions - controlled variable)
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    judge_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "university_policy_corpus"

    # PostgreSQL
    database_url: str = "postgresql+psycopg2://ragapp:ragapp@localhost:5432/ragapp"

    # Chunking (shared infrastructure)
    chunk_size: int = 800
    chunk_overlap: int = 120

    # Retrieval
    top_k: int = 5            # final context size handed to the generator
    candidate_k: int = 20     # candidate pool per retriever before fusion/rerank
    rrf_k: int = 60           # reciprocal-rank-fusion constant
    c3_max_retries: int = 1   # corrective re-retrieval attempts in C3
    max_context_chars: int = 12000  # Phase 2 character budget, not a token limit
    phase2_seed: int = 42
    phase2_max_attempts: int = 2
    llm_timeout: float = 120.0
    llm_max_retries: int = 2
    llm_reasoning_effort: str | None = None
    judge_reasoning_effort: str | None = None

    def snapshot(self) -> dict:
        """Serializable snapshot stored with every experiment run."""
        return self.model_dump(exclude={"database_url", "qdrant_url"})


@lru_cache
def get_settings() -> Settings:
    return Settings(_env_file=os.environ.get("RAGAPP_ENV_FILE", ".env"))
