"""A tiny explicit vector store: FAISS inner-product index + chunks.json side file.

No pickle, no framework wrapper — ~60 lines you can read in one go.
"""

import json
from pathlib import Path

import faiss
import numpy as np
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel


class Chunk(BaseModel):
    id: str  # "<relative path>#<chunk index>"
    source: str  # relative path of the original file
    text: str


class Source(Chunk):
    """A retrieved chunk with its similarity score."""

    score: float


def _as_matrix(vectors: list[list[float]]) -> np.ndarray:
    m = np.asarray(vectors, dtype="float32")
    faiss.normalize_L2(m)  # cosine similarity via inner product
    return m


class VectorStore:
    def __init__(self, embeddings: Embeddings, chunks: list[Chunk], index: faiss.Index) -> None:
        self.embeddings = embeddings
        self.chunks = chunks
        self.index = index

    @classmethod
    def build(cls, embeddings: Embeddings, chunks: list[Chunk]) -> "VectorStore":
        if not chunks:
            raise ValueError("no chunks to index — is the data folder empty?")
        matrix = _as_matrix(embeddings.embed_documents([c.text for c in chunks]))
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        return cls(embeddings, chunks, index)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        (directory / "chunks.json").write_text(
            json.dumps([c.model_dump() for c in self.chunks], indent=1)
        )

    @classmethod
    def load(cls, directory: Path, embeddings: Embeddings) -> "VectorStore":
        if not (directory / "index.faiss").exists():
            raise FileNotFoundError(f"no index in {directory} — run `make ingest` first")
        index = faiss.read_index(str(directory / "index.faiss"))
        raw = json.loads((directory / "chunks.json").read_text())
        return cls(embeddings, [Chunk.model_validate(c) for c in raw], index)

    async def search(self, query: str, k: int = 4) -> list[Source]:
        vector = _as_matrix([await self.embeddings.aembed_query(query)])
        scores, ids = self.index.search(vector, min(k, len(self.chunks)))
        return [
            Source(**self.chunks[i].model_dump(), score=float(s))
            for s, i in zip(scores[0], ids[0], strict=True)
            if i != -1
        ]
