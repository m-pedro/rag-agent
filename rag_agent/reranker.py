from typing import List, Sequence
from rag_agent.bm25 import tokenize_lexical
from rag_agent.models import ScoredChunk


class CrossEncoderReranker:
    """Second-stage interaction reranker scoring query-chunk term coverage, bigrams, and proximity."""

    def __init__(
        self,
        coverage_weight: float = 0.45,
        bigram_weight: float = 0.25,
        proximity_weight: float = 0.20,
        prior_weight: float = 0.10,
    ) -> None:
        self.coverage_weight = coverage_weight
        self.bigram_weight = bigram_weight
        self.proximity_weight = proximity_weight
        self.prior_weight = prior_weight

    def score_pair(self, query: str, text: str) -> float:
        q_tokens = tokenize_lexical(query)
        d_tokens = tokenize_lexical(text)
        if not q_tokens or not d_tokens:
            return 0.0

        d_set = set(d_tokens)
        unique_q = list(dict.fromkeys(q_tokens))

        # 1. Exact query term coverage in [0, 1]
        matched = [t for t in unique_q if t in d_set]
        coverage = len(matched) / float(len(unique_q))

        # 2. Ordered query bigram overlap in [0, 1]
        if len(q_tokens) >= 2 and len(d_tokens) >= 2:
            q_bigrams = set(zip(q_tokens, q_tokens[1:]))
            d_bigrams = set(zip(d_tokens, d_tokens[1:]))
            bigram_score = len(q_bigrams & d_bigrams) / float(len(q_bigrams))
        else:
            bigram_score = coverage

        # 3. Minimum window span proximity for matched query terms in [0, 1]
        proximity = self._compute_proximity(matched, d_tokens)

        total_w = self.coverage_weight + self.bigram_weight + self.proximity_weight
        return (
            self.coverage_weight * coverage
            + self.bigram_weight * bigram_score
            + self.proximity_weight * proximity
        ) / max(total_w, 1e-8)

    @staticmethod
    def _compute_proximity(matched_terms: List[str], d_tokens: List[str]) -> float:
        if len(matched_terms) < 2:
            return 1.0 if len(matched_terms) == 1 else 0.0

        positions = {t: [] for t in matched_terms}
        for idx, tok in enumerate(d_tokens):
            if tok in positions:
                positions[tok].append(idx)

        # Measure tightest pairwise distance between adjacent matched query terms
        inv_dists: List[float] = []
        for a, b in zip(matched_terms, matched_terms[1:]):
            min_dist = min(abs(pa - pb) for pa in positions[a] for pb in positions[b])
            inv_dists.append(1.0 / float(max(1, min_dist)))
        return sum(inv_dists) / float(len(inv_dists))

    def rerank(
        self,
        query: str,
        candidates: Sequence[ScoredChunk],
        top_k: int = 5,
    ) -> List[ScoredChunk]:
        if not candidates or top_k <= 0:
            return []

        max_prior = max(abs(c.score) for c in candidates) or 1.0
        rescored: List[tuple[int, float, ScoredChunk]] = []

        for idx, cand in enumerate(candidates):
            interaction = self.score_pair(query, cand.chunk.text)
            norm_prior = cand.score / max_prior
            combined = (1.0 - self.prior_weight) * interaction + self.prior_weight * norm_prior
            rescored.append((idx, combined, cand))

        rescored.sort(key=lambda item: (-item[1], item[0]))
        return [
            ScoredChunk(
                chunk=cand.chunk,
                score=float(score),
                retriever="reranked",
                rank=r + 1,
            )
            for r, (_, score, cand) in enumerate(rescored[:top_k])
        ]
