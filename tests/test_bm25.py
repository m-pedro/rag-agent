from rag_agent.bm25 import BM25Index
from rag_agent.models import Chunk


def test_bm25_exact_identifier_ranking_and_metadata_filtering():
    chunks = [
        Chunk("c0", "d0", "The quick brown fox jumps over the lazy dog", 0, 43, {"lang": "en"}),
        Chunk("c1", "d1", "Error E0592 duplicate definitions with name test_hnsw_beam", 0, 58, {"lang": "rust"}),
        Chunk("c2", "d2", "Rust compiler borrow checker error E0382 use of moved value", 0, 59, {"lang": "rust"}),
    ]
    index = BM25Index()
    index.add_chunks(chunks)

    hits = index.search("rustc error E0592 duplicate definitions", top_k=2)
    assert len(hits) == 2
    assert hits[0].chunk.chunk_id == "c1"
    assert hits[0].score > hits[1].score
    assert hits[0].rank == 1

    # Verify metadata filter predicate
    filtered = index.search("error E0592", top_k=3, filter_fn=lambda c: c.metadata.get("lang") == "en")
    assert filtered == []
