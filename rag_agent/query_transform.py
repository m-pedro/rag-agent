from typing import Callable, Dict, List, Optional
import numpy as np

from rag_agent.bm25 import tokenize_lexical
from rag_agent.dense import DenseRetriever
from rag_agent.hybrid import HybridRetriever, reciprocal_rank_fusion
from rag_agent.models import ScoredChunk

DOMAIN_EXPANSIONS: Dict[str, str] = {
    "wal": "write ahead log crash recovery crc32",
    "hnsw": "hierarchical navigable small world graph nearest neighbor",
    "lora": "low rank adaptation adapter freeze weights",
    "qlora": "4bit nf4 normalfloat quantized lora",
    "dpo": "direct preference optimization chosen rejected reward margin",
    "bm25": "lexical inverted index term frequency idf",
    "oom": "out of memory heap allocation ram limit",
}


def expand_multi_queries(query: str, max_queries: int = 3) -> List[str]:
    """Generates diverse lexical/sub-topic query reformulations for Multi-Query RRF."""
    clean = query.strip()
    if not clean:
        return []

    variants: List[str] = [clean]
    tokens = tokenize_lexical(clean)

    # Variant 2: Expand domain acronyms and technical shorthand
    expanded_terms: List[str] = []
    for tok in tokens:
        if tok in DOMAIN_EXPANSIONS:
            expanded_terms.append(DOMAIN_EXPANSIONS[tok])
    if expanded_terms:
        variants.append(f"{clean} {' '.join(expanded_terms)}")

    # Variant 3: High-signal keyword core query
    if len(tokens) >= 2:
        keyword_core = " ".join(tokens)
        if keyword_core.lower() != clean.lower() and keyword_core not in variants:
            variants.append(keyword_core)

    return variants[:max_queries]


def default_hyde_generator(query: str) -> str:
    """Synthesizes a hypothetical technical passage answering `query` (Gao et al., ACL 2023)."""
    tokens = tokenize_lexical(query)
    expansions = [DOMAIN_EXPANSIONS[t] for t in tokens if t in DOMAIN_EXPANSIONS]
    detail = " ".join(expansions) if expansions else " ".join(tokens)
    return (
        f"Technical documentation explaining {query}. "
        f"In systems architecture, {detail} ensures reliability, low latency, and memory efficiency."
    )


class HyDERetriever:
    """Hypothetical Document Embeddings (HyDE) wrapper over DenseRetriever."""

    def __init__(
        self,
        dense: DenseRetriever,
        generator_fn: Optional[Callable[[str], str]] = None,
        query_alpha: float = 0.4,
    ) -> None:
        self.dense = dense
        self.generator_fn = generator_fn or default_hyde_generator
        self.query_alpha = query_alpha

    def search(self, query: str, top_k: int = 5) -> List[ScoredChunk]:
        if not self.dense.chunks or top_k <= 0:
            return []

        hypo_doc = self.generator_fn(query)
        q_vec = self.dense.embedder.embed(query)
        h_vec = self.dense.embedder.embed(hypo_doc)

        blended = self.query_alpha * q_vec + (1.0 - self.query_alpha) * h_vec
        norm = float(np.linalg.norm(blended))
        if norm > 1e-8:
            blended /= norm

        sims = self.dense.matrix @ blended
        order = [int(i) for i in np.argsort(-sims)[:top_k]]
        return [
            ScoredChunk(
                chunk=self.dense.chunks[idx],
                score=float(sims[idx]),
                retriever="hyde_dense",
                rank=r + 1,
            )
            for r, idx in enumerate(order)
        ]


def multi_query_hybrid_search(
    retriever: HybridRetriever,
    query: str,
    top_k: int = 5,
) -> List[ScoredChunk]:
    """Executes Hybrid search across expanded sub-queries and fuses all lists with RRF."""
    sub_queries = expand_multi_queries(query)
    if not sub_queries:
        return []
    ranked_lists = [retriever.search(sq, top_k=max(top_k * 2, 10)) for sq in sub_queries]
    return reciprocal_rank_fusion(ranked_lists, rrf_k=60, top_k=top_k)
