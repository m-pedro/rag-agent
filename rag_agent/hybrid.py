from typing import Callable, Dict, List, Optional, Sequence
from rag_agent.bm25 import BM25Index
from rag_agent.dense import DenseRetriever
from rag_agent.models import Chunk, ScoredChunk


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[ScoredChunk]],
    rrf_k: int = 60,
    weights: Optional[Sequence[float]] = None,
    top_k: int = 5,
) -> List[ScoredChunk]:
    """Fuses multiple ranked retrieval lists using Reciprocal Rank Fusion (Cormack et al., 2009)."""
    if rrf_k <= 0:
        raise ValueError(f"rrf_k must be > 0, got {rrf_k}")
    if not ranked_lists or top_k <= 0:
        return []

    list_weights = (
        list(weights) if weights is not None else [1.0] * len(ranked_lists)
    )
    if len(list_weights) != len(ranked_lists):
        raise ValueError("weights length must match ranked_lists length")

    fused_scores: Dict[str, float] = {}
    chunk_lookup: Dict[str, Chunk] = {}

    for w, results in zip(list_weights, ranked_lists):
        for rank_idx, item in enumerate(results):
            cid = item.chunk.chunk_id
            chunk_lookup[cid] = item.chunk
            rank_1based = item.rank if item.rank > 0 else (rank_idx + 1)
            fused_scores[cid] = fused_scores.get(cid, 0.0) + (w / float(rrf_k + rank_1based))

    ordered = sorted(fused_scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
    return [
        ScoredChunk(
            chunk=chunk_lookup[cid],
            score=score,
            retriever="hybrid_rrf",
            rank=r + 1,
        )
        for r, (cid, score) in enumerate(ordered)
    ]


class HybridRetriever:
    """Production hybrid retriever combining Okapi BM25+ lexical search and Dense vector search."""

    def __init__(
        self,
        bm25: Optional[BM25Index] = None,
        dense: Optional[DenseRetriever] = None,
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0,
    ) -> None:
        self.bm25 = bm25 or BM25Index()
        self.dense = dense or DenseRetriever()
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

    def add_chunks(self, chunks: Sequence[Chunk]) -> None:
        self.bm25.add_chunks(chunks)
        self.dense.add_chunks(chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_pool: int = 20,
        filter_fn: Optional[Callable[[Chunk], bool]] = None,
    ) -> List[ScoredChunk]:
        pool = max(top_k * 2, candidate_pool)
        bm25_hits = self.bm25.search(query, top_k=pool, filter_fn=filter_fn)
        dense_hits = self.dense.search(query, top_k=pool, filter_fn=filter_fn)

        return reciprocal_rank_fusion(
            [bm25_hits, dense_hits],
            rrf_k=self.rrf_k,
            weights=[self.bm25_weight, self.dense_weight],
            top_k=top_k,
        )
