# Progressive RAG Experiment

Controlled A/B testing of RAG retrieval features on a **single document**, where everything is held constant except the feature under test.

## Purpose

Most "my first RAG" pipelines use naive vector search and underperform compared to commercial web chatbots. This experiment isolates which RAG features actually close the gap by holding the model, document, embedding model, chunking, prompt, context window size, and thinking level constant across all levels.

## Repository structure

| File | Purpose |
|------|---------|
| `rag_experiment.py` | The experiment script (8 levels, a–h) |
| `rag_experiment_graphrag.py` | Separate GraphRAG experiment: index-time LLM entity extraction → knowledge graph → community summaries → graph-based retrieval |
| `analyze_results.py` | Analysis script: reads results JSONL, classifies questions by type, finds best level per type, compares predicted vs actual level |
| `rag-exploration.md` | Main experiment log: findings, design, constants, architecture diagrams, pseudo code |
| `rag-retrieval-styles.md` | Candidate retrieval styles for future consideration (ColBERT, multi-vector, routing, multi-hop, etc.) |
| `results/` | Generated at runtime — one JSONL per level per run (gitignored) |
| `document.txt` | **You provide this** — the single document under test (gitignored if present) |

## Prerequisites

```bash
# Python dependencies
pip install sentence-transformers rank_bm25 numpy openai

# OpenRouter API key (required for real LLM calls)
# Put in .env (auto-loaded) or export:
export OPENROUTER_API_KEY="sk-or-..."

# Or dry-run without an API key:
#   Set LLM_BACKEND = "stub" in rag_experiment.py
```

## Quick start

```bash
# 1. Place your document at document.txt
# 2. Create questions.json with your evaluation questions (see questions.json.example)
#    Format: [{"question": "Q1", "reference_answer": "A1"}, ...] or ["Q1", "Q2"]
# 3. Run baseline comparison: naive RAG vs full-context baseline
python rag_experiment.py --levels a f --questions-file questions.json

# 4. Add progressive features one at a time
python rag_experiment.py --level b   # hybrid (vector + BM25)
python rag_experiment.py --level c   # parent-child retrieval
python rag_experiment.py --level d   # cross-encoder reranking
python rag_experiment.py --level e   # query expansion (multi-query)
python rag_experiment.py --level g   # all advanced combined
python rag_experiment.py --level h   # sparse-only (BM25 keyword search only)
```

## Feature levels

| Level | Configuration |
|-------|---------------|
| a | Naive RAG: vector search on child chunks, return child chunks |
| b | + Hybrid: vector + BM25 keyword search (RRF fused), return child chunks |
| c | + Parent-child: vector search on child chunks, return parent chunks as context |
| d | + Cross-encoder reranking over top candidates, return child chunks |
| e | + Query expansion (multi-query via LLM), return child chunks |
| f | Full-context baseline: no retrieval, whole document capped at MAX_CONTEXT_CHARS |
| g | All advanced: hybrid + parent-child + rerank + query expansion |
| h | Sparse-only RAG: BM25 keyword search only, return child chunks |

## Constants enforced across all levels

- Model: `openai/gpt-4o-mini` via OpenRouter
- Embedding: `sentence-transformers/all-MiniLM-L6-v2`
- Chunk size: 512 chars (child), 1536 chars (parent)
- Top-K: 5 chunks
- Context cap: ~5,464 chars with 10% buffer (all levels, including baseline)
- Thinking level: `low` (via OpenRouter `reasoning={"effort": "low"}`)
- Prompt template: fixed
- Eval questions: fixed across all levels (load from `--questions-file` JSON or `EVAL_QUESTIONS` constant)
- Confidence-based toggles (all off by default):
  - `CONFIDENCE_THRESHOLD`: `None` — abstain from LLM if top chunk score < value
  - `EXPANSION_THRESHOLD`: `None` — skip query expansion at level e if top score >= value
  - `WEIGHT_CONTEXT_BY_SCORE`: `False` — sort context chunks by score descending
  - `USE_WEIGHTED_RRF`: `False` — weight RRF fusion by chunk similarity scores

See `rag-exploration.md` for the full constants table and enforcement principle.

## Output

Each run writes to `results/level_<X>_<timestamp>.jsonl` — one JSON line per eval question — containing the answer, reference_answer (if provided in questions.json), retrieved chunk IDs/scores, context character count, truncation flag, latency, and the full config dict.

## Analysis

After running all levels, use `analyze_results.py` to determine which level works best for which question type:

```bash
# Analyze all results in results/ directory
python analyze_results.py results/

# Save full report to file
python analyze_results.py results/ --output analysis_report.json
```

The analysis script:
1. Classifies each question by type (fact_lookup, entity_extraction, synthesis, conceptual, other)
2. Scores each level's answer against `reference_answer`
3. Determines best level per question type (majority vote)
4. Recommends `CONFIDENCE_THRESHOLD` and `EXPANSION_THRESHOLD` values based on observed data
5. Identifies cases where query expansion helped or hurt

## GraphRAG Experiment (separate script)

`rag_experiment_graphrag.py` is a **separate experiment**, not a level in `rag_experiment.py`. GraphRAG replaces the chunk-based retrieval paradigm entirely, so it cannot be added as another level (i, j, etc.) without breaking the constant-enforcement principle of the controlled experiment.

### What it does

1. **Index-time:** LLM extracts entities + relationships from document chunks → builds a knowledge graph (nodes = entities, edges = relationships) → detects communities → generates community summaries via LLM.
2. **Query-time:** LLM identifies relevant entities from the question → traverses the graph → ranks communities by relevance → returns community summaries + entity descriptions as context → LLM generates the answer.

### Constants analysis

| Category | Constants | Status |
|----------|-----------|--------|
| **Apply** (unchanged) | `DOCUMENT_PATH`, `EMBEDDING_MODEL_NAME`, `LLM_BACKEND`, `LLM_MODEL_NAME`, `THINKING_LEVEL`, `TOP_K`, `MAX_CONTEXT_CHARS`, `OPENROUTER_*`, `EVAL_QUESTIONS`, `RESULTS_DIR`, `CONFIDENCE_THRESHOLD` | Kept identical for fair A/B comparison |
| **Don't apply** (removed) | `CHUNK_SIZE`, `CHUNK_OVERLAP`, `PARENT_CHUNK_SIZE`, `PARENT_CHUNK_OVERLAP`, `HYBRID_TOP_K`, `CHUNK_CONTEXT_OVERHEAD`, `CHILD_CONTEXT_CHARS`, `PARENT_CONTEXT_CHARS`, `WEIGHT_CONTEXT_BY_SCORE`, `USE_WEIGHTED_RRF` | Removed — GraphRAG doesn't use fixed-size chunks or RRF fusion |
| **Change** (GraphRAG-specific) | `ENTITY_EXTRACTION_MODEL`, `ENTITY_EXTRACTION_THINKING`, `COMMUNITY_DETECTION_ALGORITHM`, `MAX_ENTITIES_PER_COMMUNITY`, `GRAPH_EDGE_WEIGHT_THRESHOLD`, `MAX_COMMUNITIES_RETURNED` | New constants for index-time entity extraction and community detection |

Removed constants are recorded in every `RunResult.config` as `note_*` fields (e.g., `"note_chunk_size": "N/A — GraphRAG uses entity extraction, not fixed chunks"`), keeping the output self-describing.

### Output

Results are written to `results/graphrag_<timestamp>.jsonl` (separate from `level_*.jsonl` files).

### Why separate

GraphRAG requires index-time LLM entity extraction, which means document processing is **not** constant across levels — it's a different retrieval paradigm entirely. This is why `rag-retrieval-styles.md` defers GraphRAG to a separate experiment. Running it alongside `rag_experiment.py` lets you compare chunk-based RAG (levels a–h) against graph-based RAG on the same document and questions.

## License

Unspecified. Add your choice when you create the GitHub repository.