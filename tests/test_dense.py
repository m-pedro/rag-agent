from rag_agent.dense import DenseRetriever, HashingEmbedder
from rag_agent.models import Chunk


def test_dense_semantic_synonym_retrieval_and_mmr_diversity():
    chunks = [
        Chunk("c0", "d0", "Server crashed due to heap allocation exceeding RAM limits", 0, 58),
        Chunk("c1", "d0", "Server crashed due to heap allocation exceeding RAM limits!", 0, 59),
        Chunk("c2", "d1", "Multi-threaded mutex lock contention slows parallel execution", 0, 61),
    ]
    retriever = DenseRetriever(HashingEmbedder(dim=256))
    retriever.add_chunks(chunks)

    # Query uses "OOM memory" which matches c0/c1 via semantic concept cluster even without exact word overlap
    hits = retriever.search("Diagnose OOM memory failure", top_k=2)
    assert hits[0].chunk.chunk_id in ("c0", "c1")

    # Standard top-2 returns the two near-duplicate RAM chunks (c0 and c1).
    # MMR with lambda_mult=0.45 penalizes the duplicate c1 and promotes the diverse c2 chunk into top-2!
    mmr_hits = retriever.search_mmr("Server heap allocation and thread mutex", top_k=2, lambda_mult=0.45)
    mmr_ids = {h.chunk.chunk_id for h in mmr_hits}
    assert "c2" in mmr_ids
