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

## Key Terms & Glossary

The following terms are used throughout the experiment script and logs. They are defined here so the levels table and constants table are readable without outside knowledge.

### Bi-encoder (Embedder)
A model that encodes the **query** and each **document chunk independently** into separate vectors. At search time, the query vector is compared against all pre-computed chunk vectors using cosine similarity.  
**Pros:** Fast — chunks are embedded once at index time; search is just vector math.  
**Cons:** The query and chunk never "see" each other during encoding, so subtle relationships (negation, specific entity binding, exact phrases) can be missed.  
**In this script:** `Embedder` + `VectorStore`.

### Cross-encoder (Reranker)
A model that takes the **query and a candidate chunk together as a single input pair** and outputs a single relevance score. Because both texts are processed jointly (self-attention across both), it is more accurate than a bi-encoder but much slower — it cannot pre-compute scores, so it is applied only to a small candidate pool.  
**In this script:** `Reranker`, applied at levels 3 and 6 over the top 50 hybrid candidates.

### Dense retrieval (vector search)
Search by converting text into high-dimensional vectors and finding chunks whose vectors are closest to the query vector. Good for semantic meaning (e.g., "how do I restart the cluster?" matches "cluster reboot procedure" despite different words). Weak at exact terms (error codes, product names, acronyms).

### Sparse retrieval (BM25)
Search by matching exact words/terms using statistical frequency (Okapi BM25). Good for exact terms ("ERR-003", "worker-gpu", "novactl"). Weak at semantic paraphrase.

### Hybrid search (vector + BM25, fused with RRF)
Runs dense and sparse search in parallel, producing two ranked lists, then fuses them into one. This captures both semantic matches and exact-term matches. In this script, fusion uses Reciprocal Rank Fusion.

### RRF (Reciprocal Rank Fusion)
A formula for fusing ranked lists without needing to compare raw scores (which are on different scales for vector similarity vs. BM25). For each document, score = sum of `1 / (k + rank)` across all lists where it appears, where `k` is a smoothing constant (60 in this script). Documents that rank highly in multiple lists rise to the top.

### Parent-child retrieval
The document is chunked twice: small **children** (512 chars) for retrieval, larger **parents** (1,536 chars) for context. At search time, the best children are found, then the **parent chunks containing those children** are returned to the LLM. A child may contain a keyword match but lack surrounding context; the parent gives the model the full section around the match.  
**In this script:** Levels 2 and 6. Child-to-parent mapping is built via character-offset containment (`build_child_to_parent_map`).

### Query expansion / multi-query
The LLM rewrites the user's original question into 2–3 alternative phrasings. Each variant is searched separately, and results are fused. Helps when the user's wording differs from the document's wording.  
**In this script:** Level 4.

### Full-context baseline (level 5)
No retrieval at all. The entire document (capped at `MAX_CONTEXT_CHARS`) is stuffed into the prompt as context. This answers: "Would the model do just as well reading everything without any retrieval?"

### Context window cap (`MAX_CONTEXT_CHARS`)
A hard limit on how much context text is passed to the LLM. In this script it is **derived** from retrieval parameters, with a 10% buffer for headroom:

```
CHILD_CONTEXT_CHARS   = TOP_K * (CHUNK_SIZE + overhead)
PARENT_CONTEXT_CHARS  = max(3, TOP_K // 2) * (PARENT_CHUNK_SIZE + overhead)
RAW_CAP               = max(CHILD_CONTEXT_CHARS, PARENT_CONTEXT_CHARS)
MAX_CONTEXT_CHARS     = int(RAW_CAP * 1.10)   # 10% buffer
```

With `TOP_K=5`, `CHUNK_SIZE=512`, `PARENT_CHUNK_SIZE=1536`, `overhead=120`:
- Child budget: 5 × (512 + 120) = 3,160
- Parent budget: 3 × (1536 + 120) = 4,968
- Raw cap: 4,968
- With 10% buffer: **5,464 characters**

This is enough for 5 child chunks or up to 3 whole parent chunks plus formatting overhead. The cap is identical across all levels so context capacity never confounds the comparison.

At parent-child levels (2 and 6), whole parents are selected greedily by score until the next parent would exceed the cap. Parents are never truncated mid-text — if only one to three parents fit, only those are returned.

### Thinking level / reasoning effort (`THINKING_LEVEL`)
A fixed parameter sent to the LLM (via OpenRouter's `reasoning={"effort": "low"}`) controlling how much reasoning the model performs before answering. Held constant across all levels so model reasoning quality does not confound the retrieval comparison.

### Chunking (child vs. parent)
Splitting a document into pieces for search and context. **Child chunks** (512 chars) are the search unit — small enough for precise vector matching. **Parent chunks** (1,536 chars) are the context unit — large enough to preserve surrounding text around a match.

### Top-k (`TOP_K`)
The final number of chunks passed to the LLM as context after all retrieval, fusion, and reranking is done.

### Candidate-k (`HYBRID_TOP_K`)
The number of candidates retrieved from each index (vector and BM25) **before** fusion and reranking. The final `TOP_K` is selected from this larger pool.

### Interrelated constants (validation)
The script enforces these relationships at startup:
- `CHUNK_OVERLAP < CHUNK_SIZE`
- `PARENT_CHUNK_OVERLAP < PARENT_CHUNK_SIZE`
- `PARENT_CHUNK_SIZE > CHUNK_SIZE`
- `HYBRID_TOP_K >= TOP_K`
- `TOP_K > 0`

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
| `CHUNK_CONTEXT_OVERHEAD` | 120 chars | Per-chunk formatting overhead (header + separator) |
| `CONTEXT_BUFFER_RATIO` | 1.10 | 10% headroom multiplier on the derived cap |
| `CHILD_CONTEXT_CHARS` | `TOP_K * (CHUNK_SIZE + overhead)` ≈ 3,160 | Derived context budget for child-chunk levels |
| `PARENT_CONTEXT_CHARS` | `max(3, TOP_K // 2) * (PARENT_CHUNK_SIZE + overhead)` ≈ 4,968 | Derived context budget for parent-chunk levels |
| `MAX_CONTEXT_CHARS` | `int(max(CHILD_CONTEXT_CHARS, PARENT_CONTEXT_CHARS) * 1.10)` ≈ 5,464 | Hard cap on context section passed to LLM (derived, with 10% buffer) |
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
4. The top-K children are mapped to their parent chunks. Parents are deduplicated (keeping the best child score per parent).
5. Whole parents are selected greedily by descending score until the next parent would exceed `MAX_CONTEXT_CHARS`. Parents are never truncated mid-text.

**Progressive position:** Placed at level 2 (after hybrid, before reranking) because it isolates "does structural context enrichment improve answers?" independently of ranking quality or query rewriting. Level 6 combines it with hybrid, reranking, and query expansion.

### Context window constraint

`MAX_CONTEXT_CHARS` caps the `{context}` section passed to the LLM at **every** level, including level 5 (full-context baseline). The cap is derived from retrieval parameters with a 10% buffer and is currently approximately **5,464 characters** — enough for 5 child chunks or up to 3 whole parent chunks. This removes the "large context window" advantage of commercial web chatbots so we assess only RAG retrieval/ranking features, not context capacity.

- Level 5 gets the first ~5,464 chars of the whole document.
- RAG levels (0–4, 6) get up to ~5,464 chars of retrieved chunks (child or parent depending on level).
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
- Prerequisite: Put `OPENROUTER_API_KEY="sk-or-..."` in `.env` (auto-loaded) or export it.
- `RunResult.config` records `openrouter_base_url`, `openrouter_api_key_set`, `openrouter_headers`.

---

## Running the Experiment

```bash
cd /home/psg/.pi/rag-experiment

# 1. Place your document at document.txt
# 2. Create questions.json with your evaluation questions (see questions.json.example)
#    Format: [{"question": "Q1", "reference_answer": "A1"}, ...] or ["Q1", "Q2"]
#    or edit EVAL_QUESTIONS in the script as fallback
# 3. Install dependencies
pip install sentence-transformers rank_bm25 numpy openai

# 4. Set your OpenRouter key (in .env or export)
export OPENROUTER_API_KEY="sk-or-..."

# 5. Compare naive RAG vs full-context baseline (identical model, context cap, thinking level)
python rag_experiment.py --levels 0 5 --questions-file questions.json

# 6. Add progressive features one at a time
python rag_experiment.py --level 1   # + hybrid (vector + BM25)
python rag_experiment.py --level 2   # + parent-child (return parent chunks)
python rag_experiment.py --level 3   # + cross-encoder reranking
python rag_experiment.py --level 4   # + query expansion (multi-query)
python rag_experiment.py --level 6   # all advanced combined

Without an OpenRouter key, temporarily set `LLM_BACKEND = "stub"` to test the pipeline without API calls.

**Questions:** Questions are held constant across all levels for fair A/B comparison. Load them from a JSON file via `--questions-file`, or use the `EVAL_QUESTIONS` constant in the script as fallback. Each question can include an optional `reference_answer` field; if provided, it is saved in every `RunResult` (output JSONL) for later scoring/manual comparison.


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

### Diagram 1: Experiment loop — end-to-end flow from CLI to results

```mermaid
flowchart TD
    subgraph Constants["Constants fixed across all levels"]
    C1[Model: gpt-4o-mini via OpenRouter]
    C2[Embedding: all-MiniLM-L6-v2]
    C3[Chunk: 512 child / 1536 parent]
    C4[Context cap: MAX_CONTEXT_CHARS]
    C5[Thinking: low]
    end

    M[main] --> E[Experiment level 0-6]
    E --> RQ[run_all]
    RQ --> RQQ[run_question]
    RQQ --> RET[retrieve]
    RET --> BC[build_context]
    BC --> FC[fit_context cap MAX_CONTEXT_CHARS]
    FC --> LLM[LLM generate reasoning=low]
    LLM --> RR[RunResult]
    RR --> SR[save_results JSONL]

    classDef const fill:#e1f5e1,stroke:#333;
    classDef proc fill:#e3f2fd,stroke:#333;
    class C1,C2,C3,C4,C5 const;
    class M,E,RQ,RQQ,RET,BC,FC,LLM,RR,SR proc;
```

### Diagram 2: Retrieval level branching — inside `retrieve()`

```mermaid
flowchart LR
    Q[Question]
    Q --> L0[Level 0: vector only]
    Q --> L1[Level 1: hybrid vector+BM25]
    Q --> L2[Level 2: hybrid → parent chunks]
    Q --> L3[Level 3: hybrid + rerank]
    Q --> L4[Level 4: hybrid + query expansion]
    Q --> L5[Level 5: full doc no retrieval]
    Q --> L6[Level 6: all advanced]

    L0 --> CH[Child chunks]
    L1 --> CH
    L2 --> PH[Parent chunks]
    L3 --> CH
    L4 --> CH
    L5 --> FD[Full document capped]
    L6 --> PH

    CH --> FCX[fit_context ≤ MAX_CONTEXT_CHARS]
    PH --> FCX
    FD --> FCX
```

### Diagram 3: ASCII level pipeline — how a question flows through each level

```
┌─────────────────────────────────────────────────────────────────────┐
│  QUESTION                                                           │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────────┐
│  RETRIEVAL STRATEGY (level-dependent)                                │
├─────────────────┬─────────────────┬─────────────────┬────────────────┤
│  Level 0        │  Level 1        │  Level 2        │  Level 3       │
│  dense only     │  hybrid (V+BM)  │  hybrid +       │  hybrid +      │
│  → child chunks │  → child chunks │  parent context │  rerank →      │
│                 │                 │  (children→parents)│ child chunks│
└─────────────────┴─────────────────┴─────────────────┴────────────────┘
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
│  fit_context(): cap at MAX_CONTEXT_CHARS, cut at newline            │
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
│   ┌─────────────────────────────────────────────────────┐    │
│   │ [child 0: 0..512] │ [child 1: 462..974] │ ...       │    │
│   └─────────────────────────────────────────────────────┘    │
│  [parent 1: 1336..2872]                                      │
│   ┌─────────────────────────────────────────────────────┐    │
│   │ ... │ [child N: 1400..1912] │ ...                   │    │
│   └─────────────────────────────────────────────────────┘    │
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
                                               ┌───────────────────┐
                                               │ run_question()    │
                                               │                   │
                                               │ 1. retrieve()     │
                                               │ 2. build_context  │
                                               │ 3. fit_context()  │
                                               │ 4. LLM generate   │
                                               │ 5. RunResult      │
                                               └────────┬──────────┘
                                                        │
                                                        ▼
                                               ┌───────────────────┐
                                               │ save_results()    │
                                               │  → results/       │
                                               │    level_N_*.jsonl│
                                               └───────────────────┘
```

### Pseudo Code: `retrieve()` — the level branching logic

```
FUNCTION retrieve(question):
    IF level == 5:                          # level 5: full-context baseline
        RETURN [], False                    # no retrieval

    use_parents = (level IN {2, 6})         # levels 2, 6: parent-child context

    IF level == 0:                          # level 0: naive RAG
        child_hits = dense_search(question, top_k=TOP_K)
    ELSE IF level == 1:                     # level 1: hybrid RAG
        child_hits = hybrid_search(question, top_k=TOP_K, candidate_k=HYBRID_TOP_K)
    ELSE IF level IN {2, 3, 4, 6}:          # levels 2, 3, 4, 6: hybrid pool
        IF level == 4:                      # level 4: query expansion
            # query expansion: generate variants, search each, fuse by RRF
            variants = [question] + LLM_generate_variants(question)
            all_hits = []
            FOR q IN variants:
                all_hits.append(hybrid_search(q, top_k=HYBRID_TOP_K))
            child_hits = rrf_fuse(all_hits)[:HYBRID_TOP_K]
        ELSE:                               # levels 2, 3, 6: hybrid only
            child_hits = hybrid_search(question, top_k=HYBRID_TOP_K, candidate_k=HYBRID_TOP_K)

        IF reranker EXISTS:                 # levels 3, 6: reranker active
            child_hits = cross_encoder_rerank(question, child_hits, top_k=TOP_K)
        ELSE:                               # level 2: no reranker
            child_hits = child_hits[:TOP_K]
    ELSE:                                   # unreachable safety fallback
        child_hits = dense_search(question, top_k=TOP_K)

    IF use_parents:                         # levels 2, 6: map to parents
        parent_hits = []
        seen = {}
        FOR (child_idx, score) IN child_hits:
            FOR parent_idx IN child_to_parent[child_idx]:
                IF parent_idx NOT IN seen:  # deduplicate parents
                    seen.add(parent_idx)
                    parent_hits.append((parent_idx, score))
        RETURN parent_hits, True

    RETURN child_hits, False                # levels 0, 1, 3, 4: return children
```

### Pseudo Code: `run_question()` — end-to-end flow per question

```
FUNCTION run_question(question):
    start = timer()

    IF level == 5:                          # level 5: full-context baseline
        context = full_document            # no retrieval
        retrieved_chunks = []
    ELSE:                                   # levels 0, 1, 2, 3, 4, 6: all retrieval levels
        hits, use_parents = retrieve(question)
        chunk_source = parent_chunks IF use_parents ELSE child_chunks
        context = build_context(chunk_source, hits)
        retrieved_chunks = format_hits(hits, chunk_source)

    context, truncated = fit_context(context, MAX_CONTEXT_CHARS)
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
        IF containing NOT EMPTY:           # exact containment found
            mapping.append(containing)
            CONTINUE                       # skip fallback path

        # Fallback: parent with maximum character overlap
        best_idx = 0
        best_overlap = -1
        FOR p_idx, (p_start, p_end) IN parent_offsets:
            overlap = min(c_end, p_end) - max(c_start, p_start)
            IF overlap > best_overlap:     # track max overlap parent
                best_overlap = overlap
                best_idx = p_idx
        mapping.append([best_idx])         # use best-overlap parent as fallback
    RETURN mapping
```

### Pseudo Code: `fit_context()` — context window cap

```
FUNCTION fit_context(context, max_chars=MAX_CONTEXT_CHARS):
    IF length(context) <= max_chars:        # no truncation needed
        RETURN context, False

    cut = context[0:max_chars]

    # Prefer newline boundary within last 200 chars
    newline_idx = cut.rfind("\n", start=max(0, max_chars-200))
    IF newline_idx != -1:                   # cut at newline boundary
        RETURN cut[0:newline_idx], True

    # Fallback: sentence end
    sentence_idx = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    IF sentence_idx != -1:                  # cut at sentence end
        RETURN cut[0:sentence_idx+1], True

    RETURN cut, True                        # hard cut at max_chars (no clean boundary found)
```

---

## Incremental Updates

*(Future entries go here. The design decisions above were previously logged as separate dated entries; they are now incorporated into the main text so the document reads as a coherent whole.)*

### 2026-07-16 — Initial session
- Recorded initial query and high-level findings in "Findings / Summary".
- No code changes made yet.
- Next: decide whether to audit an existing RAG pipeline or sketch an improved architecture.

---