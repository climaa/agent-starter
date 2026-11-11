"""Offline test setup: fake LLM, hashing embeddings, dry-run tools. No network."""

import os

os.environ.update(
    LLM_PROVIDER="fake",
    EMBEDDINGS_PROVIDER="local",
    TICKET_WEBHOOK_URL="",
    N8N_WEBHOOK_TOKEN="",
    LANGFUSE_PUBLIC_KEY="",
    LANGFUSE_SECRET_KEY="",
)

from collections.abc import Iterator

import pytest

from agent_starter.config import Settings, get_settings
from agent_starter.graph.build import Graph, build_graph
from agent_starter.providers import HashingEmbeddings, OfflineChatModel
from agent_starter.rag.store import Chunk, VectorStore
from agent_starter.tools import get_tools

DOCS = {
    "refunds.md": "Customers can request a refund within 30 days of delivery.",
    "shipping.md": "Express shipments are delivered next business day when ordered before 14:00.",
    "damaged.md": "Damaged parcels must be reported within 48 hours with photos.",
}


@pytest.fixture(autouse=True)
def _fresh_settings() -> Iterator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, llm_provider="fake", embeddings_provider="local")  # type: ignore[call-arg]


@pytest.fixture
def store() -> VectorStore:
    chunks = [Chunk(id=f"{name}#0", source=name, text=text) for name, text in DOCS.items()]
    return VectorStore.build(HashingEmbeddings(), chunks)


@pytest.fixture
def graph(store: VectorStore, settings: Settings) -> Graph:
    return build_graph(OfflineChatModel(), store, get_tools(), settings)
