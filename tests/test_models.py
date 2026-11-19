import pytest
from rag_agent.models import Chunk, Document, ScoredChunk


def test_document_chunk_and_scored_chunk_invariants():
    doc = Document(doc_id="doc-1", text="Rust ownership prevents data races.", metadata={"topic": "rust"})
    chunk = Chunk(
        chunk_id="doc-1#0",
        doc_id=doc.doc_id,
        text=doc.text,
        start_char=0,
        end_char=len(doc.text),
        metadata=doc.metadata,
    )
    assert chunk.char_len == len(doc.text)

    scored = ScoredChunk(chunk=chunk, score=0.92, retriever="bm25", rank=1)
    assert scored.score == 0.92
    assert scored.chunk.doc_id == "doc-1"

    with pytest.raises(ValueError, match="doc_id"):
        Document(doc_id="   ", text="hello")

    with pytest.raises(ValueError, match="non-empty"):
        Document(doc_id="d2", text="   ")
