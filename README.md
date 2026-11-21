# rag-agent

A Hybrid Retrieval-Augmented Generation (`RAG`) and Autonomous `ReAct` Tool-Calling Agent framework built from scratch in Python and NumPy without `langchain` or `llama-index`. Implements recursive Markdown-aware chunking, **Okapi BM25+** lexical search, dense vector search with **Maximal Marginal Relevance (`MMR`)**, **Reciprocal Rank Fusion (`RRF`)**, cross-encoder token-proximity reranking, `HyDE` query transformation, citation faithfulness verification, and a multi-step `Thought -> Action -> Observation` agent loop.

---

## Repository Structure

```text
rag-agent/
├── LICENSE                         # MIT License
├── README.md                       # Architecture guide, IR math & retrieval benchmarks
├── pyproject.toml                  # Package metadata & pytest configuration
├── rag_agent/
│   ├── __init__.py                 # Public module exports
│   ├── models.py                   # Document, Chunk & ScoredChunk data models
│   ├── chunker.py                  # Recursive Markdown/semantic chunker with overlap
│   ├── bm25.py                     # Okapi BM25+ inverted index & lexical tokenizer
│   ├── dense.py                    # Signed n-gram/concept embedder, Cosine & MMR search
│   ├── hybrid.py                   # Reciprocal Rank Fusion (RRF) & HybridRetriever
│   ├── reranker.py                 # Cross-encoder interaction & span-proximity reranker
│   ├── query_transform.py          # Multi-query acronym expansion & HyDE retriever
│   ├── citation.py                 # Numbered [k] prompt builder & attribution verifier
│   ├── tools.py                    # Introspected JSON-Schema ToolRegistry & executor
│   ├── memory.py                   # Token-budgeted ConversationMemory & entity store
│   ├── react.py                    # ReAct (Thought-Action-Observation) loop & self-repair
│   ├── eval.py                     # Recall@k, MRR, NDCG@k & Context Precision@k
│   └── cli.py                      # CLI runner (bench-ir, demo-react)
└── tests/
    ├── test_models.py              # Document & Chunk invariant tests
    ├── test_chunker.py             # Hierarchical boundary & sliding overlap tests
    ├── test_bm25.py                # Exact error-code ranking & metadata filter tests
    ├── test_dense.py               # Synonym retrieval & MMR diversity deduplication
    ├── test_hybrid.py              # RRF consensus promotion & HybridRetriever tests
    ├── test_reranker.py            # Contiguous phrase proximity reranking tests
    ├── test_query_transform.py     # Multi-query & HyDE acronym bridging tests
    ├── test_citation.py            # Hallucination & invalid citation [k] detection
    ├── test_tools.py               # Signature-to-JSON-schema & error guard tests
    ├── test_memory.py              # Rolling summary eviction within token budget
    ├── test_react.py               # Multi-hop tool chaining & syntax self-repair
    ├── test_eval.py                # Exact IR metric calculation tests
    └── test_benchmark_hybrid.py    # End-to-end Hybrid RRF + Rerank superiority suite
```

---

## Hybrid RAG & Autonomous `ReAct` Architecture

```text
  ┌───────────────────────────────────────────────────────────────────┐
  │ 1. Query Transformation & Hybrid First-Stage Retrieval            │
  │                                                                   │
  │   User Query ─► [ Multi-Query / HyDE Expansion ]                  │
  │                        │                       │                  │
  │                        ▼                       ▼                  │
  │             [ Okapi BM25+ Index ]     [ Dense Cosine / MMR ]      │
  │               (Exact Identifiers)       (Semantic Concepts)       │
  │                        │                       │                  │
  │                        └───────► [ RRF ] ◄─────┘                  │
  │                         sum_m w_m / (60 + rank_m)                 │
  └─────────────────────────────────┬─────────────────────────────────┘
                                    │
                                    ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │ 2. Second-Stage Cross-Encoder Reranking & Citation Grounding      │
  │   • Scores joint term coverage, ordered bigrams & span proximity  │
  │   • Formats numbered [1], [2] evidence blocks within char budget  │
  └─────────────────────────────────┬─────────────────────────────────┘
                                    │
                                    ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │ 3. Autonomous ReAct Agent Loop & Attribution Verification         │
  │   • Thought -> Action (JSON Schema Tool) -> Observation -> Repeat │
  │   • Verifies inline [k] citations & flags unsupported sentences   │
  └───────────────────────────────────────────────────────────────────┘
```

---

## Core Algorithms & Academic References

| Component | Formula / Mechanism | Academic Reference |
| :--- | :--- | :--- |
| **`Okapi BM25+`** | `sum_t IDF(t) * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl)) + delta)` | Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond* (2009); Lv & Zhai (CIKM 2011) |
| **`MMR Diversity`** | `argmax_i [ lambda * sim(d_i, q) - (1 - lambda) * max_j sim(d_i, d_j) ]` | Carbonell & Goldstein, *The Use of MMR, Diversity-Based Reranking* (SIGIR 1998) |
| **`Hybrid RRF`** | `RRF(d) = sum_m w_m / (k_rrf + rank_m(d))` with `k_rrf = 60` | Cormack, Clarke & Buettcher, *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods* (SIGIR 2009) |
| **`HyDE`** | Embeds synthesized hypothetical answer passage blended with query vector | Gao et al., *Precise Zero-Shot Dense Retrieval without Relevance Labels* (ACL 2023) |
| **`ReAct Agent`** | Interleaved `Thought -> Action -> Observation` reasoning & tool execution | Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models* (ICLR 2023) |

---

## Information Retrieval Benchmark (`k = 2`, Multi-Type Technical Queries)

Run locally via `python3 -m rag_agent.cli bench-ir` across exact error-code queries (`ERR_WAL_7710`, `E0382`), zero-overlap semantic synonym queries (`OOM memory failure`, `slow concurrency throughput`), and phrase-proximity queries (`Raft leader election`):

| Retrieval Pipeline | Recall@2 | MRR | NDCG@2 | Context Precision@2 |
| :--- | ---: | ---: | ---: | ---: |
| **BM25+ Lexical Only** | 66.7% | 0.667 | 0.651 | 0.667 |
| **Dense Cosine Only** | 100.0% | 0.917 | 0.920 | 0.917 |
| **Hybrid RRF (`BM25+` + `Dense`)** | 100.0% | 0.917 | 0.920 | 0.917 |
| **Hybrid RRF + Cross-Encoder Rerank** | **100.0%** | **1.000** | **1.000** | **1.000** |

---

## Quickstart

### 1. Run All 13 Unit & Benchmark Test Suites

```bash
source .venv/bin/activate
pytest -v
```

### 2. Run the Side-by-Side IR Evaluation Benchmark

```bash
python3 -m rag_agent.cli bench-ir
```

### 3. Run the Autonomous `ReAct` + Citation Verification Demo

```bash
python3 -m rag_agent.cli demo-react
```

---

## 15-Day Incremental Engineering Roadmap (Month 5, Days 121–135)

- [x] **Day 121:** Initialize `rag_agent` package, `Document`/`Chunk` models, MIT license, and 15-day roadmap
- [x] **Day 122:** Implement recursive semantic and Markdown-aware document chunker (`rag_agent/chunker.py`)
- [x] **Day 123:** Implement Okapi `BM25+` lexical inverted index (`rag_agent/bm25.py`)
- [x] **Day 124:** Implement character n-gram hashing embedder, cosine index, and `MMR` diversity search (`rag_agent/dense.py`)
- [x] **Day 125:** Implement Reciprocal Rank Fusion (`RRF`) and hybrid score fusion (`rag_agent/hybrid.py`)
- [x] **Day 126:** Implement cross-encoder interaction reranker (`rag_agent/reranker.py`)
- [x] **Day 127:** Implement multi-query expansion and `HyDE` generator (`rag_agent/query_transform.py`)
- [x] **Day 128:** Implement citation-grounded context builder and attribution verifier (`rag_agent/citation.py`)
- [x] **Day 129:** Implement typed tool registry with JSON schema auto-generation (`rag_agent/tools.py`)
- [x] **Day 130:** Implement sliding-window conversation memory with entity store (`rag_agent/memory.py`)
- [x] **Day 131:** Implement `ReAct` (`Thought -> Action -> Observation`) agent loop (`rag_agent/react.py`)
- [x] **Day 132:** Implement IR & RAG evaluation metrics (`Recall@k`, `MRR`, `NDCG@k`, `Context Precision`) (`rag_agent/eval.py`)
- [x] **Day 133:** Add end-to-end benchmark proving Hybrid RRF outperforms standalone BM25 and Dense retrieval (`tests/test_benchmark_hybrid.py`)
- [x] **Day 134:** Build `rag-agent` CLI (`bench-ir`, `demo-react`) (`rag_agent/cli.py`)
- [x] **Day 135:** Complete Hybrid RAG & ReAct Agent architecture guide, math reference table, and IR benchmarks
