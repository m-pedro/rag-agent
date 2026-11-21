from dataclasses import dataclass
import re
from typing import Dict, List, Sequence, Tuple

from rag_agent.bm25 import tokenize_lexical
from rag_agent.models import Chunk, ScoredChunk

CITATION_TAG_RE = re.compile(r"\[(\d+)\]")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class AttributionReport:
    total_sentences: int
    cited_sentences: int
    supported_sentences: int
    invalid_citation_refs: List[int]
    faithfulness_score: float


def build_cited_context(
    results: Sequence[ScoredChunk],
    max_chars: int = 2400,
) -> Tuple[str, Dict[int, Chunk]]:
    """Formats retrieved chunks as numbered [1], [2] evidence blocks within `max_chars`."""
    citation_map: Dict[int, Chunk] = {}
    blocks: List[str] = []
    used_chars = 0

    for idx, item in enumerate(results, start=1):
        entry = f"[{idx}] (doc={item.chunk.doc_id}, chunk={item.chunk.chunk_id}):\n{item.chunk.text}"
        if blocks and used_chars + len(entry) + 2 > max_chars:
            break
        blocks.append(entry)
        citation_map[idx] = item.chunk
        used_chars += len(entry) + 2

    return "\n\n".join(blocks), citation_map


def build_rag_prompt(question: str, results: Sequence[ScoredChunk], max_chars: int = 2400) -> str:
    context_block, _ = build_cited_context(results, max_chars=max_chars)
    return (
        "Answer the user question strictly using the numbered context sources below. "
        "Cite supporting sources inline at the end of every factual sentence like [1] or [2].\n\n"
        f"Context Sources:\n{context_block}\n\n"
        f"Question: {question}\nAnswer:"
    )


def verify_answer_citations(
    answer: str,
    citation_map: Dict[int, Chunk],
    min_term_overlap: float = 0.35,
) -> AttributionReport:
    """Verifies inline [k] citations exist in `citation_map` and ground sentence claims."""
    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(answer.strip()) if s.strip()]
    if not sentences:
        return AttributionReport(0, 0, 0, [], 1.0)

    cited_count = 0
    supported_count = 0
    invalid_refs: List[int] = []

    for sent in sentences:
        refs = [int(m.group(1)) for m in CITATION_TAG_RE.finditer(sent)]
        if not refs:
            continue
        cited_count += 1

        valid_chunks: List[Chunk] = []
        for r in refs:
            if r not in citation_map:
                invalid_refs.append(r)
            else:
                valid_chunks.append(citation_map[r])

        if not valid_chunks:
            continue

        # Strip [k] markers before checking lexical content grounding
        claim_text = CITATION_TAG_RE.sub("", sent)
        claim_tokens = set(tokenize_lexical(claim_text))
        if not claim_tokens:
            supported_count += 1
            continue

        evidence_tokens = set()
        for ch in valid_chunks:
            evidence_tokens.update(tokenize_lexical(ch.text))

        overlap = len(claim_tokens & evidence_tokens) / float(len(claim_tokens))
        if overlap >= min_term_overlap:
            supported_count += 1

    faithfulness = supported_count / float(len(sentences))
    return AttributionReport(
        total_sentences=len(sentences),
        cited_sentences=cited_count,
        supported_sentences=supported_count,
        invalid_citation_refs=invalid_refs,
        faithfulness_score=faithfulness,
    )
