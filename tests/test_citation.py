from rag_agent.citation import build_cited_context, build_rag_prompt, verify_answer_citations
from rag_agent.models import Chunk, ScoredChunk


def test_citation_context_builder_and_hallucination_attribution_verifier():
    c1 = Chunk("d1#0", "d1", "HNSW uses multi-layer navigable graphs for logarithmic search.", 0, 62)
    c2 = Chunk("d2#0", "d2", "SQ8 scalar quantization compresses fp32 vectors by 4x into uint8.", 0, 65)
    results = [ScoredChunk(c1, 0.9, "hybrid", 1), ScoredChunk(c2, 0.8, "hybrid", 2)]

    ctx, cmap = build_cited_context(results)
    assert "[1]" in ctx and "[2]" in ctx
    prompt = build_rag_prompt("How do HNSW and SQ8 work?", results)
    assert "Context Sources:" in prompt

    # 1. Grounded answer with valid [1] and [2] citations -> 100% faithfulness
    grounded = (
        "HNSW builds multi-layer graphs for logarithmic search [1]. "
        "SQ8 quantization compresses fp32 vectors by 4x [2]."
    )
    rep_ok = verify_answer_citations(grounded, cmap)
    assert rep_ok.faithfulness_score == 1.0
    assert rep_ok.invalid_citation_refs == []

    # 2. Hallucinated claim + nonexistent citation [9] -> flagged immediately
    hallucinated = (
        "HNSW builds multi-layer graphs [1]. "
        "Quantum teleportation replaces TCP sockets on Mars [9]."
    )
    rep_bad = verify_answer_citations(hallucinated, cmap)
    assert rep_bad.faithfulness_score == 0.5
    assert 9 in rep_bad.invalid_citation_refs
