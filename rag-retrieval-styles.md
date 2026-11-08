# RAG Retrieval Styles — Candidate Library

**Purpose:** Document retrieval styles not currently in the controlled experiment so they can be considered later.  
**Status:** Reference only. No code changes implied.  
**Related:** `rag-exploration.md` (main experiment log), `rag_experiment.py` (controlled script).

---

## Covered in current experiment (levels 0–6)

| Level | Style | What it isolates |
|-------|-------|------------------|
| 0 | Dense (vector) RAG | Baseline: cosine similarity on child chunks |
| 1 | Sparse + Dense hybrid (BM25 + vector, RRF fused) | Does keyword search add value to dense? |
| 2 | Parent-child retrieval | Does returning larger parent context improve answers over raw child chunks? |
| 3 | Cross-encoder reranking | Does re-ranking top candidates improve answer quality? |
| 4 | Query expansion (multi-query via LLM) | Does rewriting the question into variants improve retrieval? |
| 5 | Full-context baseline (constrained to MAX_CONTEXT_CHARS) | Does retrieval help at all vs just pasting the whole doc? |
| 6 | All advanced combined | Combined effect of hybrid + parent-child + rerank + query expansion |

Constants enforced across all levels: model, document, embedding, chunking, prompt, eval questions, context cap (3000 chars), thinking level ("low"), OpenRouter backend.

---

## Deferred: GraphRAG

**Status:** Explicitly deferred to a separate experiment, not a level in this script.

**What it is:** Index-time LLM extracts entities and relationships from the document, builds a knowledge graph, detects communities, generates community summaries. Query time retrieves relevant subgraphs/communities as structured context.

**Why deferred:**
- Replaces the whole retrieval paradigm (chunk → graph), not a single feature.
- Requires index-time LLM entity extraction, which breaks the "document processing is constant" principle.
- Needs new dependencies (graph construction, community detection, graph traversal).
- High impact expected for multi-document cross-reasoning; modest for single document.

**Future path:** Build as a separate script if dense/sparse hybrid plateaus or corpus expands beyond one document.

---

## Not covered — candidates for later consideration

### A. Sparse-only (BM25 without vector)

**What it is:** Pure keyword retrieval using BM25/TF-IDF, no embeddings.

**Why not covered:** Current script only tests dense-only (level 0) and hybrid (level 1). There is no standalone sparse-only level.

**Complexity to add:** Trivial — call `BM25Index.search()` directly without vector fusion.

**Variable isolated:** Does keyword search alone beat vector search alone? (Useful complement to level 1 which tests "does adding keyword to vector help?")

**Fit with constants:** Clean. Same chunking, same TOP_K, same context cap. Only the retrieval method changes.

**Verdict:** Easy add if you want a dense-vs-sparse head-to-head before deciding on hybrid.

---

### B. Late interaction (ColBERT / ColPali)

**What it is:** Token-level embeddings stored per chunk; query tokens interact with document tokens at retrieval time via MaxSim similarity. Better than single-vector dense for long documents and precise term matching.

**Examples:** ColBERTv2, ColPali (for PDFs with layout).

**Why not covered:**
- Requires specialized models and indexes (not plain cosine similarity over one vector per chunk).
- Breaks the `VectorStore` abstraction — retrieval unit is token-level, not chunk-level.
- New hard dependency on late-interaction model and index format.

**Complexity to add:** High. Would need a separate retrieval path or library (e.g., ColBERT implementation).

**Fit with constants:** Poor. Different model family, different index structure, different similarity computation.

**Verdict:** Separate experiment later. Not a clean single-variable add to this script.

---

### C. Multi-vector / MaxSim per chunk

**What it is:** Store multiple embeddings per chunk (e.g., one per sentence, or one per semantic aspect) and retrieve using maximum similarity across them.

**Why not covered:**
- Changes vector store schema from `(n_chunks, dim)` to `(n_chunks, n_vectors_per_chunk, dim)`.
- `VectorStore.search()` assumes one embedding per chunk.
- New hyperparameter: how many vectors per chunk, how to pool results.

**Complexity to add:** Medium. Schema change to embedding store and search logic.

**Fit with constants:** Moderate. The "vectors per chunk" setting must be held constant across levels.

**Verdict:** Could be a level later if dense retrieval plateaus. Not a priority for single-document testing.

---

### D. Query routing

**What it is:** A classifier (LLM or smaller model) decides at query time whether to use dense, sparse, hybrid, or no retrieval based on question type.

**Why not covered:**
- Meta-layer on top of retrieval, not a retrieval style itself.
- Introduces a classifier as a new variable not constant across levels.
- Retrieval method varies per question, not per level — violates the "assess only one variable" principle.

**Complexity to add:** Medium-high. Needs a router model/prompt and per-question retrieval dispatch.

**Fit with constants:** Poor. Breaks the level-based comparison structure.

**Verdict:** Exclude from this controlled experiment. Production concern, not an isolation variable.

---

### E. Multi-hop / iterative retrieval (Self-RAG, recursive retrieval)

**What it is:** Agentic loop: LLM generates a query, retrieves, decides if more info is needed, rewrites the query, retrieves again — potentially multiple rounds.

**Why not covered:**
- Requires an agentic loop where the LLM decides when to stop retrieving.
- Variable number of retrieval steps per question breaks the "same context budget" constant unless strictly capped.
- Latency and token usage become unpredictable.

**Complexity to add:** High. Agentic framework + stop condition + budget cap.

**Fit with constants:** Poor. Variable retrieval depth per question.

**Verdict:** Separate agentic RAG project later. Not appropriate for this controlled single-document experiment.

---

### F. Metadata filtering

**What it is:** Restrict retrieval to chunks matching metadata (date, source, document type, permissions) before vector search.

**Why not covered:** Single document only — metadata filtering is moot with one document.

**Complexity to add:** Low if metadata exists; but no metadata schema defined in current script.

**Fit with constants:** N/A for single document.

**Verdict:** Relevant only for multi-document corpora. Defer until corpus expands.

---

### G. Context compression / summarization before context window

**What it is:** Instead of raw chunks, summarize or extract only relevant sentences from retrieved chunks before stuffing into the prompt (e.g., LLMLingua, sentence selection, extractive summarization).

**Why not covered:** Current `fit_context()` truncates to 3000 chars but does not semantically compress or summarize.

**Complexity to add:** Low-medium. Needs a summarization/extraction model or prompt as an additional constant.

**Fit with constants:** Moderate. The compression method/prompt must be held constant across levels.

**Verdict:** Could be a level if you want to test "does compressing retrieved context beat raw truncation?" within the same 3000-char budget.

---

## Summary table: add vs defer

| Style | Add now? | Complexity | Breaks constants? | Best for |
|-------|----------|------------|-------------------|----------|
| Sparse-only | Yes (easy) | Trivial | No | Dense vs sparse head-to-head |
| Late interaction (ColBERT) | No | High | Yes | Later, separate experiment |
| Multi-vector / MaxSim | No | Medium | Moderate | Later, if dense plateaus |
| Query routing | No | Medium | Yes | Production, not this experiment |
| Multi-hop / iterative | No | High | Yes | Agentic RAG project |
| Metadata filtering | No | Low | N/A | Multi-document corpus |
| Context compression | Maybe | Low-medium | Moderate | Test compression vs truncation |
| GraphRAG | Deferred | High | Yes | Separate experiment, multi-doc |

---

## Suggested next steps (your choice)

1. **Add sparse-only** as a level between 0 and 1 if you want dense-vs-sparse isolation before hybrid.
2. **Run the current 0–6 progression** first to see if any level plateaus.
3. **If a level plateaus**, revisit this file to pick the next style (multi-vector, context compression, etc.).
4. **GraphRAG and multi-hop** should remain separate projects, not levels in this script.

No code changes are implied by this document. Edit `rag_experiment.py` only when you decide which style to add next.