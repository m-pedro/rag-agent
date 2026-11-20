import math
import re
from typing import Callable, Dict, List, Optional, Sequence
from rag_agent.models import Chunk, ScoredChunk

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")

STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "in", "is", "it", "of", "on", "or", "that", "the", "to",
        "was", "were", "will", "with",
    }
)


def tokenize_lexical(text: str) -> List[str]:
    """Lowercases, extracts alphanumeric/identifier tokens, and filters common stopwords."""
    return [
        tok
        for tok in (m.group(0).lower() for m in TOKEN_PATTERN.finditer(text))
        if tok not in STOPWORDS
    ]


class BM25Index:
    """Okapi BM25+ inverted index for fast exact keyword and technical identifier retrieval."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 1.0,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.chunks: List[Chunk] = []
        self.doc_lens: List[int] = []
        self.avgdl: float = 0.0
        self.postings: Dict[str, Dict[int, int]] = {}
        self.idf: Dict[str, float] = {}

    def add_chunks(self, chunks: Sequence[Chunk]) -> None:
        for chunk in chunks:
            idx = len(self.chunks)
            self.chunks.append(chunk)
            tokens = tokenize_lexical(chunk.text)
            self.doc_lens.append(len(tokens))

            tf_map: Dict[str, int] = {}
            for tok in tokens:
                tf_map[tok] = tf_map.get(tok, 0) + 1

            for tok, tf in tf_map.items():
                if tok not in self.postings:
                    self.postings[tok] = {}
                self.postings[tok][idx] = tf

        self._recompute_statistics()

    def _recompute_statistics(self) -> None:
        n_docs = len(self.chunks)
        if n_docs == 0:
            self.avgdl = 0.0
            self.idf.clear()
            return

        self.avgdl = sum(self.doc_lens) / float(n_docs)
        self.idf = {}
        for term, doc_map in self.postings.items():
            df = len(doc_map)
            self.idf[term] = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_fn: Optional[Callable[[Chunk], bool]] = None,
    ) -> List[ScoredChunk]:
        if not self.chunks or top_k <= 0:
            return []

        q_terms = tokenize_lexical(query)
        if not q_terms:
            return []

        scores: Dict[int, float] = {}
        avgdl = max(self.avgdl, 1e-8)

        for term in q_terms:
            if term not in self.postings:
                continue
            term_idf = self.idf[term]
            for doc_idx, tf in self.postings[term].items():
                if filter_fn is not None and not filter_fn(self.chunks[doc_idx]):
                    continue
                dl = self.doc_lens[doc_idx]
                norm_tf = (tf * (self.k1 + 1.0)) / (
                    tf + self.k1 * (1.0 - self.b + self.b * (dl / avgdl))
                )
                scores[doc_idx] = scores.get(doc_idx, 0.0) + term_idf * (norm_tf + self.delta)

        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        return [
            ScoredChunk(
                chunk=self.chunks[doc_idx],
                score=float(score),
                retriever="bm25",
                rank=rank_idx + 1,
            )
            for rank_idx, (doc_idx, score) in enumerate(ranked)
        ]
