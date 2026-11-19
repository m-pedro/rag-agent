# rag-agent

A Hybrid Retrieval-Augmented Generation (`RAG`) and Autonomous `ReAct` Tool-Calling Agent engine built from scratch in Python and NumPy. Implements recursive Markdown-aware chunking, **Okapi BM25+** lexical search, dense vector search with **Maximal Marginal Relevance (`MMR`)**, **Reciprocal Rank Fusion (`RRF`)**, cross-encoder reranking, citation verification, and a multi-step `Thought -> Action -> Observation` agent loop.

## 15-Day Incremental Engineering Roadmap (Month 5, Days 121–135)

- [x] **Day 121:** Initialize `rag_agent` package, `Document`/`Chunk` models, MIT license, and 15-day roadmap
- [ ] **Day 122:** Implement recursive semantic and Markdown-aware document chunker (`rag_agent/chunker.py`)
- [ ] **Day 123:** Implement Okapi `BM25+` lexical inverted index (`rag_agent/bm25.py`)
- [ ] **Day 124:** Implement character n-gram hashing embedder, cosine index, and `MMR` diversity search (`rag_agent/dense.py`)
- [ ] **Day 125:** Implement Reciprocal Rank Fusion (`RRF`) and hybrid score fusion (`rag_agent/hybrid.py`)
- [ ] **Day 126:** Implement cross-encoder interaction reranker (`rag_agent/reranker.py`)
- [ ] **Day 127:** Implement multi-query expansion and `HyDE` generator (`rag_agent/query_transform.py`)
- [ ] **Day 128:** Implement citation-grounded context builder and attribution verifier (`rag_agent/citation.py`)
- [ ] **Day 129:** Implement typed tool registry with JSON schema auto-generation (`rag_agent/tools.py`)
- [ ] **Day 130:** Implement sliding-window conversation memory with entity store (`rag_agent/memory.py`)
- [ ] **Day 131:** Implement `ReAct` (`Thought -> Action -> Observation`) agent loop (`rag_agent/react.py`)
- [ ] **Day 132:** Implement IR & RAG evaluation metrics (`Recall@k`, `MRR`, `NDCG@k`, `Faithfulness`) (`rag_agent/eval.py`)
- [ ] **Day 133:** Add end-to-end benchmark proving Hybrid RRF outperforms standalone BM25 and Dense retrieval
- [ ] **Day 134:** Build `rag-agent` CLI (`bench-hybrid`, `demo-react`, `eval-suite`)
- [ ] **Day 135:** Complete Hybrid RAG & ReAct Agent architecture guide, math reference table, and IR benchmarks
