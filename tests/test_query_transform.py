from rag_agent.dense import DenseRetriever
from rag_agent.hybrid import HybridRetriever
from rag_agent.models import Chunk
from rag_agent.query_transform import (
    HyDERetriever,
    expand_multi_queries,
    multi_query_hybrid_search,
)


def test_multi_query_acronym_expansion_and_hyde_retrieval():
    queries = expand_multi_queries("How does WAL work?")
    assert len(queries) >= 2
    assert any("write ahead log" in q for q in queries)

    chunks = [
        Chunk(
            "c_wal",
            "d1",
            "A write ahead log appends sequential records with crc32 checksums for crash recovery.",
            0,
            85,
        ),
        Chunk("c_other", "d2", "CSS flexbox aligns items horizontally inside a container.", 0, 57),
    ]

    # Note: c_wal never uses the acronym "WAL", only "write ahead log"!
    # Both HyDE and Multi-Query expansion bridge the acronym gap and rank c_wal #1.
    dense = DenseRetriever()
    dense.add_chunks(chunks)
    hyde = HyDERetriever(dense)
    hyde_hits = hyde.search("WAL", top_k=2)
    assert hyde_hits[0].chunk.chunk_id == "c_wal"

    hybrid = HybridRetriever()
    hybrid.add_chunks(chunks)
    mq_hits = multi_query_hybrid_search(hybrid, "WAL", top_k=2)
    assert mq_hits[0].chunk.chunk_id == "c_wal"
