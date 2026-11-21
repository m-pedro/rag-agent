from rag_agent.hybrid import HybridRetriever, reciprocal_rank_fusion
from rag_agent.models import Chunk, ScoredChunk


def test_reciprocal_rank_fusion_promotes_consensus_candidates():
    # c_exact matches only the lexical identifier "err_wal_crc32_mismatch"
    c_exact = Chunk("c_exact", "d1", "Fatal code ERR_WAL_CRC32_MISMATCH during replay", 0, 47)
    # c_sem matches "OOM memory" ONLY via semantic concept cluster ("heap", "allocation", "ram")
    c_sem = Chunk("c_sem", "d2", "Heap allocation exceeded available RAM limits", 0, 45)
    # c_both bridges both lexical identifier + semantic memory cluster (longer text so BM25 ranks it #2)
    c_both = Chunk(
        "c_both",
        "d3",
        "ERR_WAL_CRC32_MISMATCH triggered during background journal checkpoint when heap allocation exceeds RAM",
        0,
        102,
    )

    # 1. Pure mathematical RRF check: #2 in both lists beats #1 in only one list
    list_bm25 = [
        ScoredChunk(c_exact, 8.5, "bm25", rank=1),
        ScoredChunk(c_both, 7.1, "bm25", rank=2),
    ]
    list_dense = [
        ScoredChunk(c_sem, 0.89, "dense", rank=1),
        ScoredChunk(c_both, 0.84, "dense", rank=2),
    ]

    fused = reciprocal_rank_fusion([list_bm25, list_dense], rrf_k=60, top_k=3)
    assert fused[0].chunk.chunk_id == "c_both"
    assert fused[0].rank == 1

    # 2. End-to-end HybridRetriever check: query has exact code + semantic synonym ("OOM memory")
    hybrid = HybridRetriever()
    hybrid.add_chunks([c_exact, c_sem, c_both])
    results = hybrid.search("ERR_WAL_CRC32_MISMATCH OOM memory", top_k=2)
    assert results[0].chunk.chunk_id == "c_both"
