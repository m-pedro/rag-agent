import pytest
from rag_agent.chunker import RecursiveChunker
from rag_agent.models import Document


def test_recursive_chunker_preserves_boundaries_and_overlap():
    md = (
        "# Systems Guide\n\n"
        "## Memory Management\n"
        "Rust uses compile-time ownership rules and lifetimes to free memory deterministically. "
        "Borrow checking eliminates use-after-free bugs without garbage collection pauses.\n\n"
        "## Vector Indexing\n"
        "HNSW builds a multi-layer navigable small-world graph for logarithmic nearest neighbor search."
    )
    doc = Document(doc_id="guide", text=md, metadata={"source": "handbook"})
    chunker = RecursiveChunker(chunk_size=120, chunk_overlap=30)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 3
    assert all(len(c.text) <= 130 for c in chunks)
    assert chunks[0].chunk_id == "guide#0"
    assert chunks[0].metadata["source"] == "handbook"

    # Check that key domain terms from every section survive chunking
    combined = " ".join(c.text for c in chunks)
    assert "ownership" in combined
    assert "HNSW" in combined

    with pytest.raises(ValueError, match="chunk_overlap"):
        RecursiveChunker(chunk_size=100, chunk_overlap=100)
