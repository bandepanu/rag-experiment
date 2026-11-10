# Progressive RAG Experiment

Controlled A/B testing of RAG retrieval features on a **single document**, where everything is held constant except the feature under test.

## Purpose

Most "my first RAG" pipelines use naive vector search and underperform compared to commercial web chatbots. This experiment isolates which RAG features actually close the gap by holding the model, document, embedding model, chunking, prompt, context window size, and thinking level constant across all levels.

## Repository structure

| File | Purpose |
|------|---------|
| `rag_experiment.py` | The experiment script (7 levels, 0–6) |
| `rag-exploration.md` | Main experiment log: findings, design, constants, architecture diagrams, pseudo code |
| `rag-retrieval-styles.md` | Candidate retrieval styles for future consideration (sparse-only, ColBERT, multi-vector, GraphRAG, etc.) |
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
python rag_experiment.py --levels 0 5 --questions-file questions.json

# 4. Add progressive features one at a time
python rag_experiment.py --level 1   # hybrid (vector + BM25)
python rag_experiment.py --level 2   # parent-child retrieval
python rag_experiment.py --level 3   # cross-encoder reranking
python rag_experiment.py --level 4   # query expansion (multi-query)
python rag_experiment.py --level 6   # all advanced combined
```

## Feature levels

| Level | Configuration |
|-------|---------------|
| 0 | Naive RAG: vector search on child chunks, return child chunks |
| 1 | + Hybrid: vector + BM25 keyword search (RRF fused), return child chunks |
| 2 | + Parent-child: vector search on children, return parent chunks as context |
| 3 | + Cross-encoder reranking over top candidates, return child chunks |
| 4 | + Query expansion (multi-query via LLM), return child chunks |
| 5 | Full-context baseline: no retrieval, whole document capped at MAX_CONTEXT_CHARS |
| 6 | All advanced: hybrid + parent-child + rerank + query expansion |

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
  - `EXPANSION_THRESHOLD`: `None` — skip query expansion at level 4 if top score >= value
  - `WEIGHT_CONTEXT_BY_SCORE`: `False` — sort context chunks by score descending
  - `USE_WEIGHTED_RRF`: `False` — weight RRF fusion by chunk similarity scores

See `rag-exploration.md` for the full constants table and enforcement principle.

## Output

Each run writes to `results/level_<N>_<timestamp>.jsonl` — one JSON line per eval question — containing the answer, reference_answer (if provided in questions.json), retrieved chunk IDs/scores, context character count, truncation flag, latency, and the full config dict.

## License

Unspecified. Add your choice when you create the GitHub repository.