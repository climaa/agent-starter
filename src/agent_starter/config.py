"""All runtime configuration, read from environment / .env. One place to look."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["bedrock", "anthropic", "openai", "fake"]
EmbeddingsProvider = Literal["bedrock", "openai", "local"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Models -----------------------------------------------------------
    llm_provider: LLMProvider = "bedrock"
    llm_model: str = ""  # empty = provider default (see providers.py)
    llm_max_tokens: int = 4096
    embeddings_provider: EmbeddingsProvider = "bedrock"
    embeddings_model: str = ""
    aws_region: str = "eu-west-1"

    # --- RAG --------------------------------------------------------------
    data_dir: Path = Path("data")
    index_dir: Path = Path(".index")
    chunk_size: int = 400
    chunk_overlap: int = 60
    top_k: int = 3

    # --- Agent ------------------------------------------------------------
    max_steps: int = 4  # max tool-calling rounds in the act loop
    max_input_chars: int = 4000

    # --- Tools ------------------------------------------------------------
    http_api_base_url: str = "https://api.github.com"
    http_timeout_s: float = 10.0
    http_retries: int = 2
    ticket_webhook_url: str = ""  # empty = dry-run (ticket is not posted)

    # --- API --------------------------------------------------------------
    n8n_webhook_token: str = ""  # if set, /webhook/n8n requires X-Webhook-Token

    # --- Observability ----------------------------------------------------
    log_level: str = "INFO"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"


@lru_cache
def get_settings() -> Settings:
    return Settings()
