import hashlib
from typing import Callable, Dict, List, Optional, Sequence
import numpy as np

from rag_agent.bm25 import tokenize_lexical
from rag_agent.models import Chunk, ScoredChunk

# Semantic concept clusters so synonyms map to shared latent dimensions alongside n-gram features
SEMANTIC_CLUSTERS: Dict[str, str] = {
    "oom": "concept_memory_limit",
    "memory": "concept_memory_limit",
    "ram": "concept_memory_limit",
    "allocation": "concept_memory_limit",
    "heap": "concept_memory_limit",
    "latency": "concept_speed_perf",
    "fast": "concept_speed_perf",
    "throughput": "concept_speed_perf",
    "performance": "concept_speed_perf",
    "speedup": "concept_speed_perf",
    "concurrency": "concept_parallel_sync",
    "thread": "concept_parallel_sync",
    "mutex": "concept_parallel_sync",
    "lock": "concept_parallel_sync",
    "async": "concept_parallel_sync",
    "quantization": "concept_compression_quant",
    "nf4": "concept_compression_quant",
    "sq8": "concept_compression_quant",
    "4bit": "concept_compression_quant",
    "8bit": "concept_compression_quant",
    "compression": "concept_compression_quant",
}


class HashingEmbedder:
    """Deterministic L2-normalized dense embedder combining semantic clusters, words, and char n-grams."""

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be > 0, got {dim}")
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = tokenize_lexical(text)

        features: List[tuple[str, float]] = []
        for tok in tokens:
            features.append((f"w:{tok}", 2.0))
            if tok in SEMANTIC_CLUSTERS:
                features.append((f"c:{SEMANTIC_CLUSTERS[tok]}", 3.5))
            padded = f"<{tok}>"
            for n in (3, 4):
                for i in range(max(0, len(padded) - n + 1)):
                    features.append((f"g:{padded[i:i+n]}", 0.6))

        for feat, weight in features:
            digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            h = int.from_bytes(digest, "little")
            idx = h % self.dim
            sign = 1.0 if ((h >> 32) & 1) == 0 else -1.0
            vec[idx] += sign * weight

        norm = float(np.linalg.norm(vec))
        if norm > 1e-8:
            vec /= norm
        return vec

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self.embed(t) for t in texts])


class DenseRetriever:
    """In-memory dense vector index supporting Cosine top-k and Maximal Marginal Relevance (MMR)."""

    def __init__(self, embedder: Optional[HashingEmbedder] = None) -> None:
        self.embedder = embedder or HashingEmbedder(dim=256)
        self.chunks: List[Chunk] = []
        self.matrix: np.ndarray = np.zeros((0, self.embedder.dim), dtype=np.float32)

    def add_chunks(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            return
        new_vecs = self.embedder.embed_batch([c.text for c in chunks])
        self.chunks.extend(chunks)
        if self.matrix.shape[0] == 0:
            self.matrix = new_vecs
        else:
            self.matrix = np.vstack([self.matrix, new_vecs])

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_fn: Optional[Callable[[Chunk], bool]] = None,
    ) -> List[ScoredChunk]:
        if not self.chunks or top_k <= 0:
            return []
        q_vec = self.embedder.embed(query)
        sims = self.matrix @ q_vec

        candidates: List[tuple[int, float]] = []
        for idx, score in enumerate(sims):
            if filter_fn is not None and not filter_fn(self.chunks[idx]):
                continue
            candidates.append((idx, float(score)))

        candidates.sort(key=lambda kv: (-kv[1], kv[0]))
        return [
            ScoredChunk(
                chunk=self.chunks[idx],
                score=score,
                retriever="dense",
                rank=r + 1,
            )
            for r, (idx, score) in enumerate(candidates[:top_k])
        ]

    def search_mmr(
        self,
        query: str,
        top_k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.65,
    ) -> List[ScoredChunk]:
        """Maximal Marginal Relevance (MMR) balancing query relevance and inter-chunk diversity."""
        if not self.chunks or top_k <= 0:
            return []
        q_vec = self.embedder.embed(query)
        sims = self.matrix @ q_vec

        pool_size = min(max(fetch_k, top_k), len(self.chunks))
        cand_indices = [int(i) for i in np.argsort(-sims)[:pool_size]]

        selected: List[int] = []
        selected_scores: List[float] = []

        while cand_indices and len(selected) < top_k:
            best_idx = -1
            best_mmr = -1e9
            for idx in cand_indices:
                rel = float(sims[idx])
                if not selected:
                    red = 0.0
                else:
                    red = float(np.max(self.matrix[selected] @ self.matrix[idx]))
                mmr_val = lambda_mult * rel - (1.0 - lambda_mult) * red
                if mmr_val > best_mmr:
                    best_mmr = mmr_val
                    best_idx = idx

            selected.append(best_idx)
            selected_scores.append(best_mmr)
            cand_indices.remove(best_idx)

        return [
            ScoredChunk(
                chunk=self.chunks[idx],
                score=score,
                retriever="dense_mmr",
                rank=r + 1,
            )
            for r, (idx, score) in enumerate(zip(selected, selected_scores))
        ]
