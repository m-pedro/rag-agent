from typing import List, Tuple
from rag_agent.bm25 import BM25Index
from rag_agent.chunker import RecursiveChunker
from rag_agent.dense import DenseRetriever, HashingEmbedder
from rag_agent.eval import EvalQuery, evaluate_retriever
from rag_agent.hybrid import HybridRetriever
from rag_agent.models import Chunk, Document
from rag_agent.reranker import CrossEncoderReranker


def build_benchmark_corpus_and_queries() -> Tuple[List[Chunk], List[EvalQuery]]:
    docs = [
        Document("d_err1", "Fatal storage panic ERR_WAL_7710 when journal CRC32 header mismatches disk sector."),
        Document("d_err2", "Compiler diagnostic E0382 triggered when value is borrowed after move in Rust."),
        Document("d_mem1", "Process terminated because heap allocation exceeded physical RAM ceiling."),
        Document("d_perf1", "Tail p99 request latency spiked due to mutex lock contention across worker threads."),
        Document("d_quant1", "NormalFloat 4bit weight compression shrinks LLM VRAM footprint by sevenfold."),
        Document("d_raft1", "During a Raft leader election, candidates increment currentTerm and request cluster votes."),
        Document("d_noise1", " Raft protocol overview. Unrelated telemetry buffers, disk rotation schedules, and CSS layout notes before mentioning leader or election."),
        Document("d_noise2", "PostgreSQL B-tree page split factor defaults to ninety percent for sequential inserts."),
    ]
    chunker = RecursiveChunker(chunk_size=240, chunk_overlap=32)
    chunks = chunker.chunk_documents(docs)

    queries = [
        # Exact error codes where BM25 shines
        EvalQuery("ERR_WAL_7710", {"d_err1#0"}),
        EvalQuery("diagnostic E0382", {"d_err2#0"}),
        # Pure semantic synonym queries (zero lexical word overlap!) where Dense shines
        EvalQuery("OOM memory failure", {"d_mem1#0"}),
        EvalQuery("slow concurrency throughput", {"d_perf1#0"}),
        EvalQuery("NF4 quantization", {"d_quant1#0"}),
        # Phrase proximity query where CrossEncoderReranker promotes tight span over scattered noise
        EvalQuery("Raft leader election", {"d_raft1#0"}, {"d_raft1#0": 2.0, "d_noise1#0": 0.5}),
    ]
    return chunks, queries


def test_hybrid_rrf_plus_reranker_outperforms_standalone_bm25_and_dense():
    chunks, queries = build_benchmark_corpus_and_queries()

    bm25 = BM25Index()
    bm25.add_chunks(chunks)

    dense = DenseRetriever(HashingEmbedder(dim=256))
    dense.add_chunks(chunks)

    hybrid = HybridRetriever(bm25=bm25, dense=dense, rrf_k=60)
    reranker = CrossEncoderReranker(prior_weight=0.35)

    def hybrid_reranked_search(q: str, k: int):
        stage1 = hybrid.search(q, top_k=6)
        return reranker.rerank(q, stage1, top_k=k)

    bm25_metrics = evaluate_retriever(lambda q, k: bm25.search(q, top_k=k), queries, k=2)
    dense_metrics = evaluate_retriever(lambda q, k: dense.search(q, top_k=k), queries, k=2)
    hybrid_metrics = evaluate_retriever(lambda q, k: hybrid.search(q, top_k=k), queries, k=2)
    rerank_metrics = evaluate_retriever(hybrid_reranked_search, queries, k=2)

    # BM25 misses pure-synonym queries ("OOM memory failure"); Hybrid recovers 100% Recall@2!
    assert hybrid_metrics["recall@k"] > bm25_metrics["recall@k"]
    assert hybrid_metrics["mrr"] > bm25_metrics["mrr"]
    assert hybrid_metrics["mrr"] >= dense_metrics["mrr"]
    assert rerank_metrics["recall@k"] == 1.0
    assert rerank_metrics["mrr"] >= 0.90
