# RAG Exploration Log

**Started:** 2026-07-16  
**Purpose:** Record why personal RAG attempts underperform compared to commercial/web chatbot RAG, and document the controlled experiment used to isolate which RAG features actually close the gap.

---

## Query

> Why do my own attempts to do RAG seem miserable, but when ChatGPT, Gemini in web chatbot, or agent harnesses like pi.dev use it when I feed a document and ask a query, the results are really good?

---

## Findings / Summary

**Short answer:** Personal "naive RAG" pipelines usually underperform because commercial experiences combine **advanced RAG**, **very large context windows**, and **stronger underlying models** — often all at once.

### 1. What most people build first: naive RAG

Typical pipeline:
1. Split text into fixed-size chunks (e.g., 512 chars).
2. Embed each chunk with a generic embedding model.
3. Store in a vector database.
4. Query: embed question → cosine similarity search → top-k chunks → stuff into prompt.

Common failure modes:
- **Bad chunking:** sentences/tables/code split across chunk boundaries; chunks lack standalone context.
- **Generic embeddings:** not domain-tuned; semantic mismatch between query and stored chunks.
- **Pure vector search:** no BM25/keyword fallback; exact terms (product names, error codes) get buried.
- **No reranking:** top-5 by cosine similarity are not necessarily the 5 most useful chunks.
- **No query transformation:** user question used as-is (no expansion, synonyms, or hypothetical document embedding).
- **Stuffing:** raw chunks dumped into the prompt with no weighting, dedup, or token-limit awareness.

Result: model receives poor context and either hallucinates, misses the answer, or refuses.

### 2. What good systems actually do (advanced RAG)

- **Smart chunking:** recursive/semantic/structure-aware splitting; keep headings with sections, don't split tables, use parent-child chunks.
- **Hybrid search:** vector search + BM25/keyword, fused (e.g., RRF).
- **Reranking:** cross-encoder (BGE-reranker, Cohere rerank) re-scores top candidates and drops noise.
- **Query expansion/rewriting:** HyDE, multi-query, or LLM rewrites question into search-friendly variants before retrieval.
- **Metadata filtering:** date, document type, permissions applied before vector search.
- **Context compression:** only relevant sentences kept, or chunks summarized before generation.
- **Evaluation loop:** RAGAS or similar metrics; system tuned based on observed failures.

### 3. The hidden factor: it may not be RAG at all

Web chatbots (ChatGPT, Gemini, Claude) often use **very large context windows** (128k–1M+ tokens), not retrieval. The whole document is parsed (OCR, tables, images) and stuffed into the prompt. For documents under ~100 pages, long-context generation usually beats homemade RAG because nothing is lost to a bad retrieval step.

### 4. The model matters enormously

A 7B/13B local model over retrieved chunks is not comparable to GPT-4o, Gemini 2.5, or Claude Sonnet 4. Frontier models are better at ignoring irrelevant context, combining facts across chunks, and saying "I don't know" instead of hallucinating.

### 5. What to try next (in order of effort)

1. Check if retrieval is even needed — if the document fits in the model's context window, paste the whole document in.
2. Improve chunking (paragraph/section/sentence splits with overlap, not fixed character count).
3. Add hybrid search (vector + BM25).
4. Add a reranker (cross-encoder) over top vector results.
5. Rewrite the query (multi-query / HyDE) before retrieval.
6. Use a stronger embedding model (BGE, E5, Cohere).

### Diagnostic questions

- Am I splitting documents in ways that break meaning across chunk boundaries?
- Am I retrieving only by vector similarity, with no keyword fallback?
- Am I feeding top chunks raw, or reranking/compressing them first?
- Am I comparing my pipeline to a system that pastes the whole document into a 200k-token context window?

---

## Experimental Design

To isolate which RAG features actually matter, we run a controlled experiment on a **single document** where everything is held constant except the feature under test. The implementation lives in `rag_experiment.py` (same directory).

### Feature levels

| Level | Configuration |
|-------|---------------|
| 0 | Naive RAG: vector search on child chunks, return child chunks |
| 1 | + Hybrid: vector + BM25 keyword search (RRF fused), return child chunks |
| 2 | + Parent-child: vector search on child chunks, return **parent chunks** as context |
| 3 | + Cross-encoder reranking over top candidates, return child chunks |
| 4 | + Query expansion (multi-query via LLM), return child chunks |
| 5 | Full-context baseline: no retrieval, whole document in prompt (constrained to same window) |
| 6 | All advanced: hybrid + parent-child + rerank + query expansion |

### Output

Each run writes to `results/level_<N>_<timestamp>.jsonl` — one JSON line per eval question — containing the answer, retrieved chunk IDs/scores, context character count, truncation flag, latency, and the full config dict.

---

## Constants Enforced Across All Levels

**Principle:** every constant in the script must be actively enforced in code, not merely defined. If a constant is not threaded through the pipeline and recorded in output, it is not a constant — it is a comment.

The following are fixed for every level and recorded in every `RunResult.config`:

| Constant | Value | Purpose |
|----------|-------|---------|
| `DOCUMENT_PATH` | `document.txt` | Single document under test |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `LLM_BACKEND` | `openrouter` | LLM provider backend |
| `LLM_MODEL_NAME` | `openai/gpt-4o-mini` | Model identifier on OpenRouter |
| `CHUNK_SIZE` | 512 characters | Child chunk size (retrieval unit) |
| `CHUNK_OVERLAP` | 50 characters | Child chunk overlap |
| `PARENT_CHUNK_SIZE` | 1536 characters | Parent chunk size (context unit at parent-child levels) |
| `PARENT_CHUNK_OVERLAP` | 200 characters | Parent chunk overlap |
| `TOP_K` | 5 | Final chunks passed to LLM |
| `HYBRID_TOP_K` | 50 | Candidate pool before rerank/final cut |
| `MAX_CONTEXT_CHARS` | 3000 | Hard cap on context section passed to LLM |
| `THINKING_LEVEL` | `"low"` | Reasoning effort passed to LLM on every call |
| `PROMPT_TEMPLATE` | See script | Prompt template |
| `EVAL_QUESTIONS` | See script | Fixed set of questions |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter API endpoint |
| `OPENROUTER_HEADERS` | `HTTP-Referer`, `X-Title` | OpenRouter attribution headers |

### Parent-child retrieval (level 2 and level 6)

**What it is:** Instead of returning the raw child chunks that match the query, parent-child retrieval returns the larger parent chunk that contains each matching child. The child chunks (512 chars) are the retrieval unit; the parent chunks (1536 chars) are the context unit.

**Why it helps:** A child chunk may contain a keyword match but lack enough surrounding context to make sense on its own. The parent chunk gives the model the full section around the match without dumping the entire document into the prompt.

**How it works in this script:**
1. The document is chunked twice: into children (512 chars, with overlap) and parents (1536 chars, with overlap).
2. Each child is mapped to the parent(s) that contain it.
3. Vector search runs on children (same retrieval unit as level 0).
4. The top-K children are mapped to their unique parents.
5. Context is built from those parent chunks, then capped at `MAX_CONTEXT_CHARS` like every other level.

**Progressive position:** Placed at level 2 (after hybrid, before reranking) because it isolates "does structural context enrichment improve answers?" independently of ranking quality or query rewriting. Level 6 combines it with hybrid, reranking, and query expansion.

### Context window constraint

`MAX_CONTEXT_CHARS = 3000` caps the `{context}` section passed to the LLM at **every** level, including level 5 (full-context baseline). This removes the "large context window" advantage of commercial web chatbots so we assess only RAG retrieval/ranking features, not context capacity.

- Level 5 gets the first 3000 chars of the whole document.
- RAG levels (0–4, 6) get the first 3000 chars of retrieved chunks (child or parent depending on level).
- `fit_context()` truncates at a newline or sentence boundary where possible.
- Each `RunResult` records `context_chars` and `truncated` so we can see when the cap bit.

### Thinking level constraint

`THINKING_LEVEL = "low"` is enforced on every LLM call, not just logged. Reasoning effort is a hidden variable that can change answer quality independently of RAG features, so it must be held constant like the model, document, chunking, and context cap.

- `LLM.generate()` accepts `thinking_level` (default `THINKING_LEVEL`).
- `OpenAILLM.generate()` sends `reasoning={"effort": thinking_level}` to OpenRouter on every call (OpenRouter maps this to the underlying model's native setting: `reasoning_effort` for OpenAI models, `thinking` for Anthropic, etc.).
- `StubLLM.generate()` echoes the thinking level in placeholder output.
- `run_question()` passes `thinking_level=self.thinking_level` to `llm.generate()`.

### OpenRouter backend

All LLM calls route through OpenRouter (`https://openrouter.ai/api/v1`) using the OpenAI Python client with a custom `base_url`. OpenRouter provides a standardized, provider-agnostic API; `reasoning={"effort": "low"}` is its standard way to set reasoning/thinking level across models.

- `OpenAILLM.__init__` constructs the OpenAI client with `base_url=OPENROUTER_BASE_URL`, `api_key=OPENROUTER_API_KEY` (from env), `default_headers=OPENROUTER_HEADERS`.
- Prerequisite: `export OPENROUTER_API_KEY="sk-or-..."` before running.
- `RunResult.config` records `openrouter_base_url`, `openrouter_api_key_set`, `openrouter_headers`.

---

## Running the Experiment

```bash
cd /home/psg/.pi/rag-experiment

# 1. Place your document at document.txt
# 2. Fill in EVAL_QUESTIONS in the script with 3–5 questions about that document
# 3. Install dependencies
pip install sentence-transformers rank_bm25 numpy openai

# 4. Set your OpenRouter key
export OPENROUTER_API_KEY="sk-or-..."

# 5. Compare naive RAG vs full-context baseline (identical model, context cap, thinking level)
python rag_experiment.py --levels 0 5

# 6. Add progressive features one at a time
python rag_experiment.py --level 1   # + hybrid (vector + BM25)
python rag_experiment.py --level 2   # + parent-child (return parent chunks)
python rag_experiment.py --level 3   # + cross-encoder reranking
python rag_experiment.py --level 4   # + query expansion (multi-query)
python rag_experiment.py --level 6   # all advanced combined
```

Without an OpenRouter key, temporarily set `LLM_BACKEND = "stub"` to test the pipeline without API calls.


## Deferred: GraphRAG

**Status:** Deferred. Not part of the current progressive experiment.

**What it is:** A retrieval paradigm that replaces unstructured chunk retrieval with a knowledge graph. At index time, an LLM extracts entities and relationships from the document, builds a graph (nodes = entities, edges = relationships), detects communities, and generates community summaries. At query time, the system retrieves relevant communities/subgraphs and uses the structured relationships + summaries as context.

**How it differs from dense/sparse/hybrid:**
- Dense/sparse retrieve *text chunks*. GraphRAG retrieves *structured relationships and pre-computed community summaries*.
- Changes the unit of retrieval from "chunk" to "graph subgraph / community."
- Requires an LLM pass over the document at index time for entity extraction (expensive and slow).
- For a **single document**, GraphRAG is often overkill; dense/sparse hybrid usually performs comparably with far less complexity.

**Why deferred from this experiment:**
| Factor | Assessment |
|--------|-----------|
| **Variable isolated** | Replaces the whole retrieval paradigm, not a single feature. Does not isolate cleanly. |
| **Implementation complexity** | High. Needs entity extraction, graph construction, community detection (Leiden/Louvain), graph traversal, new prompt template for graph context. |
| **Fits current script?** | Poorly. Script is built around chunk-based retrieval (`chunk_text`, `VectorStore`, `BM25Index`). GraphRAG needs a parallel pipeline or rewrite. |
| **Breaks constants?** | Yes. Index-time LLM extraction means document processing is no longer constant across levels. Would need extraction prompt and model as additional fixed constants. |
| **Expected impact** | High for multi-document, cross-document reasoning. Modest for a single document. |

**Future path:** If dense/sparse hybrid plateaus or you move to a multi-document corpus, build GraphRAG as a **separate experiment script**, not a level in this one, so it doesn't corrupt the controlled constants here.

---

## Architecture Diagrams and Pseudo Code

The following diagrams and pseudo code describe the experiment architecture implemented in `rag_experiment.py`. They live here (rather than in the Python file) so they can evolve independently of code edits without risking doc drift.

### Diagram 1: Level pipeline — how a question flows through each level

```
┌─────────────────────────────────────────────────────────────────────┐
│  QUESTION                                                           │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  RETRIEVAL STRATEGY (level-dependent)                               │
├─────────────────┬─────────────────┬─────────────────┬──────────────┤
│  Level 0        │  Level 1        │  Level 2        │  Level 3     │
│  dense only     │  hybrid (V+BM)  │  hybrid +       │  hybrid +    │
│  → child chunks │  → child chunks │  parent context │  rerank →    │
│                 │                 │  (children→parents)│ child chunks│
└─────────────────┴─────────────────┴─────────────────┴──────────────┘
┌─────────────────┬─────────────────┬─────────────────┬──────────────┐
│  Level 4        │  Level 5        │  Level 6        │              │
│  hybrid +       │  full-context   │  hybrid +       │              │
│  query expansion│  (no retrieval) │  parent-child + │              │
│  → child chunks │  → whole doc    │  rerank +       │              │
│                 │  (capped)       │  expansion →    │              │
│                 │                 │  parent chunks  │              │
└─────────────────┴─────────────────┴─────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  CONTEXT BUILD: join hits with [Chunk i (score s)] separators       │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  fit_context(): cap at MAX_CONTEXT_CHARS (3000), cut at newline     │
│  or sentence boundary. All levels get the same budget.              │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PROMPT TEMPLATE: {context} + {question}                            │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LLM GENERATE: reasoning={"effort": "low"} via OpenRouter           │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  ANSWER + RunResult (config, latency, context_chars, truncated)     │
└─────────────────────────────────────────────────────────────────────┘
```

### Diagram 2: Parent-child mapping (levels 2 and 6)

```
DOCUMENT TEXT
──────────────────────────────────────────────────────────────
│  [parent 0: 0..1536]                                         │
│   ┌─────────────────────────────────────────────────────┐   │
│   │ [child 0: 0..512] │ [child 1: 462..974] │ ...     │   │
│   └─────────────────────────────────────────────────────┘   │
│  [parent 1: 1336..2872]                                      │
│   ┌─────────────────────────────────────────────────────┐   │
│   │ ... │ [child N: 1400..1912] │ ...                 │   │
│   └─────────────────────────────────────────────────────┘   │
│  [parent 2: 2672..4208]                                      │
└──────────────────────────────────────────────────────────────

child_to_parent map (index parallel to child_offsets):
  child_to_parent[0]  →  [0]        (child 0 is inside parent 0)
  child_to_parent[1]  →  [0]        (child 1 is inside parent 0)
  child_to_parent[N]  →  [1]        (child N is inside parent 1)
  ...
  If a child straddles two parents, fallback = parent with max overlap.

At level 2/6:
  retrieve children → map to unique parents → return parent chunks
```

### Diagram 3: Experiment loop

```
┌─────────────┐      ┌──────────────────┐      ┌─────────────────┐
│    main()   │─────►│ Experiment(level)│─────►│   run_all()     │
│  (CLI)      │      │  (init: chunks,  │      │  for each       │
│             │      │   embedder, vs,  │      │   question:     │
│             │      │   bm25, reranker,│      │   run_question()│
│             │      │   LLM)           │      │                 │
└─────────────┘      └──────────────────┘      └────────┬────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │ run_question()  │
                                               │                 │
                                               │ 1. retrieve()     │
                                               │ 2. build_context│
                                               │ 3. fit_context()  │
                                               │ 4. LLM generate   │
                                               │ 5. RunResult      │
                                               └────────┬────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │ save_results()  │
                                               │  → results/     │
                                               │    level_N_*.jsonl│
                                               └─────────────────┘
```

### Pseudo Code: `retrieve()` — the level branching logic

```
FUNCTION retrieve(question):
    IF level == 5:
        RETURN [], False          # full-context baseline: no retrieval

    use_parents = (level IN {2, 6})

    IF level == 0:
        child_hits = dense_search(question, top_k=TOP_K)
    ELSE IF level == 1:
        child_hits = hybrid_search(question, top_k=TOP_K, candidate_k=HYBRID_TOP_K)
    ELSE IF level IN {2, 3, 4, 6}:
        IF level == 4:
            # query expansion: generate variants, search each, fuse by RRF
            variants = [question] + LLM_generate_variants(question)
            all_hits = []
            FOR q IN variants:
                all_hits.append(hybrid_search(q, top_k=HYBRID_TOP_K))
            child_hits = rrf_fuse(all_hits)[:HYBRID_TOP_K]
        ELSE:
            child_hits = hybrid_search(question, top_k=HYBRID_TOP_K, candidate_k=HYBRID_TOP_K)

        IF reranker EXISTS:           # levels 3 and 6
            child_hits = cross_encoder_rerank(question, child_hits, top_k=TOP_K)
        ELSE:
            child_hits = child_hits[:TOP_K]
    ELSE:
        child_hits = dense_search(question, top_k=TOP_K)

    IF use_parents:
        parent_hits = []
        seen = {}
        FOR (child_idx, score) IN child_hits:
            FOR parent_idx IN child_to_parent[child_idx]:
                IF parent_idx NOT IN seen:
                    seen.add(parent_idx)
                    parent_hits.append((parent_idx, score))
        RETURN parent_hits, True

    RETURN child_hits, False
```

### Pseudo Code: `run_question()` — end-to-end flow per question

```
FUNCTION run_question(question):
    start = timer()

    IF level == 5:
        context = full_document            # no retrieval
        retrieved_chunks = []
    ELSE:
        hits, use_parents = retrieve(question)
        chunk_source = parent_chunks IF use_parents ELSE child_chunks
        context = build_context(chunk_source, hits)
        retrieved_chunks = format_hits(hits, chunk_source)

    context, truncated = fit_context(context, MAX_CONTEXT_CHARS=3000)
    context_chars = length(context)

    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    answer = LLM.generate(prompt, thinking_level="low")
    latency = timer() - start

    config = {
        level, document_path, embedding_model_name, llm_backend,
        llm_model_name, chunk_size, chunk_overlap, parent_chunk_size,
        parent_chunk_overlap, top_k, hybrid_top_k, max_context_chars,
        thinking_level, prompt_template,
        openrouter_base_url, openrouter_api_key_set, openrouter_headers
    }

    RETURN RunResult(level, config, question, answer, retrieved_chunks,
                     context_chars, truncated, latency, timestamp)
```

### Pseudo Code: `build_child_to_parent_map()` — containment logic

```
FUNCTION build_child_to_parent_map(child_offsets, parent_offsets):
    mapping = []
    FOR (c_start, c_end) IN child_offsets:
        # Find parents that fully contain this child
        containing = [p_idx FOR p_idx, (p_start, p_end) IN parent_offsets
                      IF p_start <= c_start AND c_end <= p_end]
        IF containing NOT EMPTY:
            mapping.append(containing)
            CONTINUE

        # Fallback: parent with maximum character overlap
        best_idx = 0
        best_overlap = -1
        FOR p_idx, (p_start, p_end) IN parent_offsets:
            overlap = min(c_end, p_end) - max(c_start, p_start)
            IF overlap > best_overlap:
                best_overlap = overlap
                best_idx = p_idx
        mapping.append([best_idx])
    RETURN mapping
```

### Pseudo Code: `fit_context()` — context window cap

```
FUNCTION fit_context(context, max_chars=3000):
    IF length(context) <= max_chars:
        RETURN context, False

    cut = context[0:max_chars]

    # Prefer newline boundary within last 200 chars
    newline_idx = cut.rfind("\n", start=max(0, max_chars-200))
    IF newline_idx != -1:
        RETURN cut[0:newline_idx], True

    # Fallback: sentence end
    sentence_idx = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    IF sentence_idx != -1:
        RETURN cut[0:sentence_idx+1], True

    RETURN cut, True
```

---

## Incremental Updates

*(Future entries go here. The design decisions above were previously logged as separate dated entries; they are now incorporated into the main text so the document reads as a coherent whole.)*

### 2026-07-16 — Initial session
- Recorded initial query and high-level findings in "Findings / Summary".
- No code changes made yet.
- Next: decide whether to audit an existing RAG pipeline or sketch an improved architecture.

---