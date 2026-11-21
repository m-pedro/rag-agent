from rag_agent.models import Chunk, ScoredChunk
from rag_agent.reranker import CrossEncoderReranker


def test_cross_encoder_reranker_promotes_tight_phrase_proximity():
    # Scattered mentions of "raft", "leader", and "election" far apart
    c_scattered = Chunk(
        "c_scattered",
        "d1",
        "Raft is a protocol. Many unrelated storage metrics follow here across several sentences "
        "about disks and networks before mentioning a leader node or any election timeout.",
        0,
        175,
    )
    # Tight contiguous answer phrase "Raft leader election"
    c_tight = Chunk(
        "c_tight",
        "d2",
        "During a Raft leader election, candidates increment the current term and request votes.",
        0,
        87,
    )

    # Even if first-stage retrieval scored c_scattered slightly higher, reranker promotes c_tight to #1
    initial = [
        ScoredChunk(c_scattered, score=0.80, retriever="hybrid_rrf", rank=1),
        ScoredChunk(c_tight, score=0.75, retriever="hybrid_rrf", rank=2),
    ]
    reranker = CrossEncoderReranker()
    reranked = reranker.rerank("raft leader election", initial, top_k=2)

    assert reranked[0].chunk.chunk_id == "c_tight"
    assert reranked[0].score > reranked[1].score
    assert reranked[0].rank == 1
