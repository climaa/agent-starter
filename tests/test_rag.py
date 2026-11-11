from pathlib import Path

from agent_starter.providers import HashingEmbeddings
from agent_starter.rag.ingest import load_chunks
from agent_starter.rag.store import VectorStore


def test_load_chunks_splits_and_tags_sources(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\n\n" + "alpha sentence. " * 40)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("beta text")
    (tmp_path / "ignored.csv").write_text("x,y")
    chunks = load_chunks(tmp_path, chunk_size=200, chunk_overlap=20)
    sources = {c.source for c in chunks}
    assert sources == {"a.md", "sub/b.txt"}
    assert len([c for c in chunks if c.source == "a.md"]) > 1
    assert all(len(c.text) <= 200 for c in chunks)


async def test_store_roundtrip_and_ranking(tmp_path: Path, store: VectorStore) -> None:
    store.save(tmp_path)
    loaded = VectorStore.load(tmp_path, HashingEmbeddings())
    hits = await loaded.search("how many days for a refund", k=2)
    assert hits[0].source == "refunds.md"
    assert hits[0].score >= hits[1].score
