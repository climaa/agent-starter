"""Ingest ./data (md, txt, pdf) → chunks → embeddings → local FAISS index.

Usage: uv run agent-ingest  (or `make ingest`)
"""

import logging
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from agent_starter.config import Settings, get_settings
from agent_starter.observability import setup_logging
from agent_starter.providers import get_embeddings
from agent_starter.rag.store import Chunk, VectorStore

log = logging.getLogger(__name__)
SUPPORTED = {".md", ".txt", ".pdf"}


def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return path.read_text(encoding="utf-8")


def load_chunks(data_dir: Path, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[Chunk] = []
    for path in sorted(p for p in data_dir.rglob("*") if p.suffix.lower() in SUPPORTED):
        rel = path.relative_to(data_dir).as_posix()
        for i, text in enumerate(splitter.split_text(read_file(path))):
            chunks.append(Chunk(id=f"{rel}#{i}", source=rel, text=text))
    return chunks


def ingest(settings: Settings) -> VectorStore:
    chunks = load_chunks(settings.data_dir, settings.chunk_size, settings.chunk_overlap)
    store = VectorStore.build(get_embeddings(settings), chunks)
    store.save(settings.index_dir)
    log.info("ingested", extra={"chunks": len(chunks), "index_dir": str(settings.index_dir)})
    return store


def main() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    ingest(settings)


if __name__ == "__main__":
    main()
