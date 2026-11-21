import json
import sys
import time

from rag_agent.bm25 import BM25Index
from rag_agent.chunker import RecursiveChunker
from rag_agent.citation import build_cited_context, verify_answer_citations
from rag_agent.dense import DenseRetriever, HashingEmbedder
from rag_agent.eval import EvalQuery, evaluate_retriever
from rag_agent.hybrid import HybridRetriever
from rag_agent.models import Document
from rag_agent.react import ReActAgent
from rag_agent.reranker import CrossEncoderReranker
from rag_agent.tools import ToolRegistry


def _build_demo_index():
    docs = [
        Document("doc_wal", "Fatal storage panic ERR_WAL_7710 occurs when journal CRC32 header mismatches disk sector."),
        Document("doc_rust", "Compiler diagnostic E0382 is raised when a value is borrowed after move in Rust."),
        Document("doc_oom", "Process terminated because heap allocation exceeded physical RAM ceiling."),
        Document("doc_perf", "Tail p99 request latency spiked due to mutex lock contention across worker threads."),
        Document("doc_nf4", "NormalFloat 4bit weight compression shrinks 28 GB fp32 LLM weights by 4x down to 7 GB."),
        Document("doc_raft", "During a Raft leader election, candidates increment currentTerm and request cluster votes."),
        Document("doc_noise", "Raft overview note. Many unrelated disk rotation logs and UI styles appear before leader or election."),
    ]
    chunks = RecursiveChunker(chunk_size=240, chunk_overlap=32).chunk_documents(docs)

    bm25 = BM25Index()
    bm25.add_chunks(chunks)

    dense = DenseRetriever(HashingEmbedder(dim=256))
    dense.add_chunks(chunks)

    hybrid = HybridRetriever(bm25=bm25, dense=dense, rrf_k=60)
    reranker = CrossEncoderReranker(prior_weight=0.35)
    return bm25, dense, hybrid, reranker


def run_bench_ir() -> None:
    bm25, dense, hybrid, reranker = _build_demo_index()
    queries = [
        EvalQuery("ERR_WAL_7710", {"doc_wal#0"}),
        EvalQuery("diagnostic E0382", {"doc_rust#0"}),
        EvalQuery("OOM memory failure", {"doc_oom#0"}),
        EvalQuery("slow concurrency throughput", {"doc_perf#0"}),
        EvalQuery("NF4 quantization", {"doc_nf4#0"}),
        EvalQuery("Raft leader election", {"doc_raft#0"}, {"doc_raft#0": 2.0, "doc_noise#0": 0.5}),
    ]

    t0 = time.perf_counter()
    rows = [
        ("BM25+ Lexical", evaluate_retriever(lambda q, k: bm25.search(q, top_k=k), queries, k=2)),
        ("Dense Cosine", evaluate_retriever(lambda q, k: dense.search(q, top_k=k), queries, k=2)),
        ("Hybrid RRF", evaluate_retriever(lambda q, k: hybrid.search(q, top_k=k), queries, k=2)),
        (
            "Hybrid RRF + Rerank",
            evaluate_retriever(
                lambda q, k: reranker.rerank(q, hybrid.search(q, top_k=6), top_k=k),
                queries,
                k=2,
            ),
        ),
    ]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    print("=== rag-agent Information Retrieval Benchmark (k=2, 6 Multi-Type Queries) ===")
    print(f"{'Pipeline':<22} | {'Recall@2':>9} | {'MRR':>8} | {'NDCG@2':>8} | {'CtxPrec@2':>10}")
    print("-" * 69)
    for name, m in rows:
        print(
            f"{name:<22} | {m['recall@k']*100:>8.1f}% | {m['mrr']:>8.3f} | "
            f"{m['ndcg@k']:>8.3f} | {m['context_precision@k']:>10.3f}"
        )
    print(f"Evaluated 4 pipelines in {elapsed_ms:.2f} ms")


def run_demo_react() -> None:
    _, _, hybrid, reranker = _build_demo_index()
    tools = ToolRegistry()
    cited_chunks_holder = {}

    def search_knowledge_base(query: str) -> str:
        """Searches technical docs via Hybrid RRF + Cross-Encoder Reranking."""
        hits = reranker.rerank(query, hybrid.search(query, top_k=4), top_k=2)
        ctx, cmap = build_cited_context(hits)
        cited_chunks_holder.update(cmap)
        return ctx

    def compute_cluster_vram(model_gb: float, replicas: int) -> str:
        """Computes total cluster VRAM required across model replicas."""
        return json.dumps({"total_vram_gb": model_gb * replicas})

    tools.register(search_knowledge_base)
    tools.register(compute_cluster_vram)

    step_idx = {"n": 0}

    def demo_policy(_prompt: str, _schemas: list) -> str:
        step_idx["n"] += 1
        if step_idx["n"] == 1:
            return (
                "Thought: Search the knowledge base for NF4 quantization VRAM compression.\n"
                'Action: search_knowledge_base\nAction Input: {"query": "NF4 quantization"}'
            )
        if step_idx["n"] == 2:
            return (
                "Thought: Source [1] states NF4 shrinks 28 GB weights down to 7 GB per replica. "
                "Now compute total VRAM for 4 replicas.\n"
                'Action: compute_cluster_vram\nAction Input: {"model_gb": 7.0, "replicas": 4}'
            )
        return (
            "Thought: Combine grounded citation [1] with the calculator result (28.0 GB).\n"
            "Final Answer: NormalFloat 4bit weight compression shrinks 28 GB weights down to 7 GB [1]. "
            "Deploying 4 quantized replicas requires 28.0 GB of total cluster VRAM [1]."
        )

    agent = ReActAgent(tools=tools, policy_fn=demo_policy, max_steps=5)
    res = agent.run("How much cluster VRAM do 4 NF4-quantized replicas of a 28 GB model need?")
    attribution = verify_answer_citations(res.answer, cited_chunks_holder, min_term_overlap=0.25)

    print("=== rag-agent Autonomous ReAct + Citation Verification Trace ===")
    for s in res.steps:
        print(f"[Step {s.step_num}] Thought    : {s.thought}")
        if s.action:
            print(f"         Action     : {s.action}({s.action_input})")
            obs_one_line = (s.observation or "").replace("\n", " ")[:96]
            print(f"         Observation: {obs_one_line}...")
    print(f"\nFinal Answer     : {res.answer}")
    print(
        f"Faithfulness     : {attribution.faithfulness_score*100:.1f}% "
        f"({attribution.supported_sentences}/{attribution.total_sentences} sentences grounded, "
        f"invalid refs={attribution.invalid_citation_refs})"
    )


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "bench-ir":
        run_bench_ir()
    elif cmd == "demo-react":
        run_demo_react()
    else:
        print("Usage: python3 -m rag_agent.cli <bench-ir | demo-react>")


if __name__ == "__main__":
    main()
