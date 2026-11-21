from dataclasses import dataclass
import math
from typing import Callable, Dict, List, Mapping, Sequence, Set

from rag_agent.models import ScoredChunk


@dataclass(frozen=True)
class EvalQuery:
    query: str
    relevant_ids: Set[str]
    graded_relevance: Mapping[str, float] | None = None


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Set[str], k: int) -> float:
    if not relevant_ids or k <= 0:
        return 0.0
    top_slice = set(retrieved_ids[:k])
    return len(top_slice & relevant_ids) / float(len(relevant_ids))


def reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: Set[str]) -> float:
    for idx, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / float(idx)
    return 0.0


def ndcg_at_k(
    retrieved_ids: Sequence[str],
    graded_relevance: Mapping[str, float],
    k: int,
) -> float:
    if not graded_relevance or k <= 0:
        return 0.0

    def _dcg(ids: Sequence[str]) -> float:
        total = 0.0
        for idx, cid in enumerate(ids[:k], start=1):
            rel = float(graded_relevance.get(cid, 0.0))
            if rel > 0.0:
                total += (2.0**rel - 1.0) / math.log2(idx + 1.0)
        return total

    actual_dcg = _dcg(retrieved_ids)
    ideal_ids = [
        cid for cid, _ in sorted(graded_relevance.items(), key=lambda kv: -kv[1])
    ]
    ideal_dcg = _dcg(ideal_ids)
    return (actual_dcg / ideal_dcg) if ideal_dcg > 0.0 else 0.0


def context_precision_at_k(retrieved_ids: Sequence[str], relevant_ids: Set[str], k: int) -> float:
    """RAGAS-style Context Precision@k weighting earlier relevant chunks higher."""
    if not relevant_ids or k <= 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, cid in enumerate(retrieved_ids[:k], start=1):
        if cid in relevant_ids:
            hits += 1
            precision_sum += hits / float(rank)
    denom = min(k, len(relevant_ids))
    return (precision_sum / float(denom)) if hits > 0 else 0.0


def evaluate_retriever(
    search_fn: Callable[[str, int], Sequence[ScoredChunk]],
    queries: Sequence[EvalQuery],
    k: int = 3,
) -> Dict[str, float]:
    if not queries:
        return {"recall@k": 0.0, "mrr": 0.0, "ndcg@k": 0.0, "context_precision@k": 0.0}

    recalls: List[float] = []
    mrrs: List[float] = []
    ndcgs: List[float] = []
    cprecs: List[float] = []

    for eq in queries:
        hits = search_fn(eq.query, k)
        ids = [h.chunk.chunk_id for h in hits]
        graded = eq.graded_relevance or {cid: 1.0 for cid in eq.relevant_ids}

        recalls.append(recall_at_k(ids, eq.relevant_ids, k))
        mrrs.append(reciprocal_rank(ids, eq.relevant_ids))
        ndcgs.append(ndcg_at_k(ids, graded, k))
        cprecs.append(context_precision_at_k(ids, eq.relevant_ids, k))

    n = float(len(queries))
    return {
        "recall@k": sum(recalls) / n,
        "mrr": sum(mrrs) / n,
        "ndcg@k": sum(ndcgs) / n,
        "context_precision@k": sum(cprecs) / n,
    }
