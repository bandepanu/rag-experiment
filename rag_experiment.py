#!/usr/bin/env python3
"""
Progressive RAG experimentation on a single document.

Design principle: hold everything constant except the feature under test.
The model, document, embedding model, chunking, prompt, eval questions,
context window size (MAX_CONTEXT_CHARS), thinking level (THINKING_LEVEL),
and OpenRouter backend are fixed across all levels. Only the retrieval /
ranking feature set changes per level.

Feature levels:
  0 = Naive RAG: vector search on child chunks, return child chunks
  1 = Hybrid RAG: vector + BM25 keyword search (RRF fused), return child chunks
  2 = Parent-child retrieval: vector search on child chunks, return parent chunks as context
  3 = Hybrid + Cross-encoder reranking over top candidates, return child chunks
  4 = Hybrid + Query expansion (multi-query via LLM), return child chunks
  5 = Full-context baseline: no retrieval, whole document capped at MAX_CONTEXT_CHARS
  6 = All advanced: hybrid + parent-child + rerank + query expansion

Usage:
  python rag_experiment.py --level 0
  python rag_experiment.py --level 6
  python rag_experiment.py --levels 0 1 2 3 4 5 6

Outputs:
  results/level_<N>_<timestamp>.jsonl  (one JSON line per eval question)

Prerequisite:
  Put OPENROUTER_API_KEY="sk-or-..." in .env (auto-loaded) or export it.
  (or set LLM_BACKEND="stub" to dry-run)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Iterable

import numpy as np


# ---------------------------------------------------------------------------
# .env loader (no external dependency)
# ---------------------------------------------------------------------------

def _load_env() -> None:
    """Load KEY=VALUE pairs from a .env file in the current working directory.

    Existing environment variables take precedence over .env values.
    No external dependency (e.g., python-dotenv) is required.
    """
    env_path = Path(".env")
    if not env_path.exists():
        return
    try:
        with env_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        pass


_load_env()

# ---------------------------------------------------------------------------
# CONSTANTS — edit these to fix the experimental conditions
#
# Every constant below is actively enforced in code and recorded in every
# RunResult.config. If a constant is not threaded through the pipeline and
# logged, it is not a constant — it is a comment.
# ---------------------------------------------------------------------------

# Path to the single document under test. All levels read the same file.
DOCUMENT_PATH = Path("document.txt")

# Embedding model used for both document chunks and query embedding.
# Fixed across all levels so retrieval differences come from ranking, not embeddings.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# LLM provider backend. "openrouter" routes through OpenRouter's standardized API;
# "stub" returns a placeholder for dry-runs without an API key.
LLM_BACKEND = "openrouter"

# Model identifier on OpenRouter. Fixed across all levels so generation quality
# is constant; only retrieval context changes between levels.
LLM_MODEL_NAME = "openai/gpt-4o-mini"

# ---------------------------------------------------------------------------
# Core retrieval sizing (interrelated parameters)
# ---------------------------------------------------------------------------

# Child chunk size in characters. Child chunks are the retrieval unit at every level.
CHUNK_SIZE = 512

# Child chunk overlap in characters. Must be smaller than CHUNK_SIZE to avoid
# degenerate or duplicated chunks.
CHUNK_OVERLAP = 50

# Parent chunk size in characters. Parent chunks are the context unit at
# parent-child levels (2 and 6). Parent should be larger than child so it
# provides surrounding context around a matching child.
PARENT_CHUNK_SIZE = 1536

# Parent chunk overlap in characters. Must be smaller than PARENT_CHUNK_SIZE.
PARENT_CHUNK_OVERLAP = 200

# Final chunks passed to the LLM as context. Fixed across all levels.
TOP_K = 5

# Candidate pool retrieved from each index (vector and BM25) before fusion
# and reranking. Must be >= TOP_K so there are enough candidates to select from.
HYBRID_TOP_K = 50

# ---------------------------------------------------------------------------
# Derived context window cap
# ---------------------------------------------------------------------------

# Approximate per-chunk overhead for context formatting:
# "[Chunk i (score s)]\n" header + "\n\n---\n\n" separator per chunk.
CHUNK_CONTEXT_OVERHEAD = 120

# Extra headroom multiplier applied to the derived cap so the context window
# is never exactly full and fit_context() has a little slack before truncating.
CONTEXT_BUFFER_RATIO = 1.10

# For child-chunk levels (0, 1, 3, 4): context = TOP_K * (CHUNK_SIZE + overhead)
# For parent-chunk levels (2, 6): we target fitting at least 3 parent chunks
# plus overhead. Each parent is roughly 3x a child, so we size the cap to hold
# multiple whole parents rather than truncating them mid-text.
CHILD_CONTEXT_CHARS = TOP_K * (CHUNK_SIZE + CHUNK_CONTEXT_OVERHEAD)
PARENT_CONTEXT_CHARS = max(3, TOP_K // 2) * (PARENT_CHUNK_SIZE + CHUNK_CONTEXT_OVERHEAD)

# Hard cap on context section passed to the LLM.
# Derived from retrieval parameters so it scales with chunk size and top-k,
# then multiplied by CONTEXT_BUFFER_RATIO for headroom.
# Kept identical across ALL levels so context capacity never confounds results.
MAX_CONTEXT_CHARS = int(max(CHILD_CONTEXT_CHARS, PARENT_CONTEXT_CHARS) * CONTEXT_BUFFER_RATIO)

# ---------------------------------------------------------------------------
# Validation of interrelated constants
# ---------------------------------------------------------------------------

assert CHUNK_OVERLAP < CHUNK_SIZE, "CHUNK_OVERLAP must be less than CHUNK_SIZE"
assert PARENT_CHUNK_OVERLAP < PARENT_CHUNK_SIZE, "PARENT_CHUNK_OVERLAP must be less than PARENT_CHUNK_SIZE"
assert PARENT_CHUNK_SIZE > CHUNK_SIZE, "PARENT_CHUNK_SIZE should be larger than CHUNK_SIZE"
assert HYBRID_TOP_K >= TOP_K, "HYBRID_TOP_K must be >= TOP_K"
assert TOP_K > 0, "TOP_K must be positive"

# ---------------------------------------------------------------------------
# LLM and backend settings
# ---------------------------------------------------------------------------

# Fixed reasoning effort passed to the LLM on every generate() call.
# OpenRouter maps this to the model's native setting (reasoning_effort for
# OpenAI models, thinking for Anthropic, etc.).
THINKING_LEVEL = "low"

# OpenRouter API endpoint. OpenAI-compatible base URL for the OpenAI Python client.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# OpenRouter API key, read from the environment. Required when LLM_BACKEND="openrouter".
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Attribution headers OpenRouter recommends for usage tracking.
OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/psg/rag-experiment",
    "X-Title": "RAG Experiment",
}

# Prompt template used at every level. {context} and {question} are filled in
# by run_question(). Kept constant so prompt wording never confounds results.
PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question using ONLY the provided context.
If the answer is not in the context, say "I don't know based on the provided document."

Context:
{context}

Question:
{question}

Answer:
"""

# Fixed evaluation questions. Same questions are asked at every level so answers
# are directly comparable. Fill in with questions about your document.
# These are used as the fallback unless --questions-file is provided.
EVAL_QUESTIONS: List[str] = [
    "What is this document about?",
    "Can you summarize the key points?",
]

# Path to a JSON file containing a list of question strings OR objects.
# If set (or passed via --questions-file), questions are loaded from this file
# instead of EVAL_QUESTIONS.
# Format: ["Q1", "Q2"] or [{"question": "Q1", "reference_answer": "A1"}, ...]
QUESTIONS_FILE: Optional[Path] = None

# Directory for experiment output. One JSONL file per level per run.
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Lightweight LLM interface (swap backend here; everything else stays constant)
# ---------------------------------------------------------------------------

class LLM:
    """Abstract interface for language model generation.

    Substituting a different backend here is the only change needed to swap
    model providers; every other part of the experiment stays identical.
    """

    def generate(self, prompt: str, thinking_level: str = THINKING_LEVEL) -> str:
        """Generate a completion for the given prompt.

        Args:
            prompt: The full prompt text (system + user content combined).
            thinking_level: Reasoning effort passed to the model backend.
                Defaults to the module-level THINKING_LEVEL constant.

        Returns:
            The model's text response.
        """
        raise NotImplementedError


class OpenAILLM(LLM):
    """OpenAI-compatible LLM client pointed at OpenRouter.

    Uses the OpenAI Python SDK with a custom base_url so all calls route through
    OpenRouter's standardized API. The `reasoning={"effort": thinking_level}`
    parameter is OpenRouter's provider-agnostic way to set reasoning/thinking
    effort; OpenRouter maps it to the underlying model's native setting.
    """

    def __init__(self, model_name: str):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")
        self.client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
            default_headers=OPENROUTER_HEADERS,
        )
        self.model_name = model_name

    def generate(self, prompt: str, thinking_level: str = THINKING_LEVEL) -> str:
        """Generate a completion via OpenRouter.

        Args:
            prompt: The full prompt text.
            thinking_level: Reasoning effort (e.g., "low", "medium", "high").

        Returns:
            The model's text response, or empty string if no content.
        """
        kwargs = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }
        # OpenRouter's provider-agnostic "reasoning" parameter. Enforced on every
        # call so thinking level is constant across all experiment levels.
        if thinking_level:
            kwargs["reasoning"] = {"effort": thinking_level}
        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""


class StubLLM(LLM):
    """Placeholder LLM for dry-runs without an API key.

    Echoes the thinking level in its output so the constant is visible
    even when no real model is called.
    """

    def generate(self, prompt: str, thinking_level: str = THINKING_LEVEL) -> str:
        return f"[STUB LLM thinking={thinking_level}] Would answer based on {len(prompt)} chars of prompt."


def make_llm() -> LLM:
    """Factory that builds the LLM backend selected by LLM_BACKEND.

    Returns:
        An OpenAILLM (OpenRouter) or StubLLM instance.

    Raises:
        ValueError: If LLM_BACKEND is not "openrouter" or "stub".
    """
    if LLM_BACKEND == "openrouter":
        return OpenAILLM(LLM_MODEL_NAME)
    if LLM_BACKEND == "stub":
        return StubLLM()
    raise ValueError(f"Unknown LLM_BACKEND: {LLM_BACKEND}")


# ---------------------------------------------------------------------------
# Document loading & chunking (kept constant)
# ---------------------------------------------------------------------------

def load_document(path: Path) -> str:
    """Load the full text of the document under test.

    Args:
        path: Path to the document file.

    Returns:
        The document text as a single string.
    """
    return path.read_text(encoding="utf-8")


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into fixed-size character chunks with overlap.

    Simplest chunking strategy: no sentence or paragraph awareness. Kept
    constant across all experiment levels so chunking never confounds results.

    Args:
        text: The text to chunk.
        size: Target chunk size in characters.
        overlap: Number of overlapping characters between adjacent chunks.

    Returns:
        List of text chunks.
    """
    if size <= 0:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def chunk_with_offsets(
    text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Split text into chunks and return both chunks and their character offsets.

    The offsets are used by build_child_to_parent_map() to determine which
    parent chunk contains each child chunk.

    Args:
        text: The text to chunk.
        size: Target chunk size in characters.
        overlap: Number of overlapping characters between adjacent chunks.

    Returns:
        A tuple of (chunks, offsets) where offsets is a list of (start, end)
        character index pairs parallel to chunks.
    """
    if size <= 0:
        return [text], [(0, len(text))]
    chunks = []
    offsets = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        offsets.append((start, end))
        if end >= len(text):
            break
        start = end - overlap
    return chunks, offsets


def build_child_to_parent_map(
    child_offsets: List[Tuple[int, int]],
    parent_offsets: List[Tuple[int, int]],
) -> List[List[int]]:
    """Map each child chunk to the parent chunk(s) that fully contain it.

    Used by parent-child retrieval (levels 2 and 6): after retrieving child
    chunks by vector search, we map them back to their containing parents so
    the parent's larger context can be passed to the LLM.

    Args:
        child_offsets: List of (start, end) offsets for child chunks.
        parent_offsets: List of (start, end) offsets for parent chunks.

    Returns:
        A list parallel to child_offsets. Each element is a list of parent
        chunk indices that contain that child. Falls back to the single parent
        with maximum character overlap if no exact containment is found.
    """
    mapping: List[List[int]] = []
    for c_start, c_end in child_offsets:
        containing = [
            p_idx
            for p_idx, (p_start, p_end) in enumerate(parent_offsets)
            if p_start <= c_start and c_end <= p_end
        ]
        if containing:
            mapping.append(containing)
            continue
        # Fallback: pick parent with maximum character overlap
        best_idx = 0
        best_overlap = -1
        for p_idx, (p_start, p_end) in enumerate(parent_offsets):
            overlap_len = min(c_end, p_end) - max(c_start, p_start)
            if overlap_len > best_overlap:
                best_overlap = overlap_len
                best_idx = p_idx
        mapping.append([best_idx])
    return mapping


# ---------------------------------------------------------------------------
# Embedding + vector store (kept constant; numpy cosine similarity)
# ---------------------------------------------------------------------------

class Embedder:
    """Embeds text using a sentence-transformers model.

    The same Embedder instance is used for both document chunks (at index time)
    and queries (at retrieval time), ensuring query and document vectors live
    in the same space. Fixed across all experiment levels.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts with L2 normalization.

        Args:
            texts: List of strings to embed.

        Returns:
            NumPy array of shape (len(texts), dim) with unit-length rows,
            so dot product equals cosine similarity.
        """
        return self.model.encode(texts, normalize_embeddings=True)


class VectorStore:
    """In-memory vector store over document chunks using cosine similarity.

    Built once on child chunks and shared by all levels. Retrieval returns
    (chunk_index, similarity_score) pairs sorted by descending similarity.
    """

    def __init__(self, embedder: Embedder, chunks: List[str]):
        self.chunks = chunks
        # Pre-compute embeddings for all chunks. Embeddings are L2-normalized
        # by the Embedder, so the dot product below is cosine similarity.
        self.embeddings = embedder.encode(chunks)  # shape: (n_chunks, dim)

    def search(self, query_embedding: np.ndarray, top_k: int = TOP_K) -> List[Tuple[int, float]]:
        """Search for the top_k chunks most similar to the query embedding.

        Args:
            query_embedding: L2-normalized query vector of shape (dim,).
            top_k: Number of results to return.

        Returns:
            List of (chunk_index, cosine_similarity) pairs, sorted by
            descending similarity.
        """
        sims = self.embeddings @ query_embedding
        top_idx = np.argsort(-sims)[:top_k]
        return [(int(i), float(sims[i])) for i in top_idx]


# ---------------------------------------------------------------------------
# BM25 keyword index (for hybrid search)
# ---------------------------------------------------------------------------

class BM25Index:
    """Keyword-based index using BM25 (Okapi BM25) for sparse retrieval.

    Complements the dense VectorStore: exact term matches (product names,
    error codes, acronyms) that dense embeddings may miss are captured here.
    Hybrid search fuses BM25 results with vector results via RRF.
    """

    def __init__(self, chunks: List[str]):
        from rank_bm25 import BM25Okapi
        self.chunks = chunks
        tokenized = [self._tokenize(c) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def _tokenize(self, text: str) -> List[str]:
        """Lowercase and split text into word tokens for BM25."""
        return re.findall(r"\w+", text.lower())

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> List[Tuple[int, float]]:
        """Search for the top_k chunks matching the query by BM25 score.

        Args:
            query: The user question text.
            top_k: Number of results to return.

        Returns:
            List of (chunk_index, bm25_score) pairs, sorted by descending score.
        """
        tokenized = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized)
        top_idx = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in top_idx]


# ---------------------------------------------------------------------------
# Reranker (cross-encoder)
# ---------------------------------------------------------------------------

class Reranker:
    """Cross-encoder reranker that re-scores retrieval candidates.

    Takes the top candidate pool from hybrid/vector search and applies a
    more expressive cross-encoder model to re-order them. Used at levels 3
    and 6. Unlike the bi-encoder (Embedder), the cross-encoder attends to
    the query and candidate jointly, producing more accurate relevance scores.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[Tuple[int, str]], top_k: int = TOP_K) -> List[Tuple[int, float]]:
        """Re-rank candidate (chunk_index, text) pairs by cross-encoder score.

        Args:
            query: The user question.
            candidates: List of (chunk_index, chunk_text) pairs to re-score.
            top_k: Number of top candidates to return after re-ranking.

        Returns:
            List of (chunk_index, cross_encoder_score) pairs, sorted by
            descending score, truncated to top_k.
        """
        if not candidates:
            return []
        pairs = [[query, text] for _, text in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(zip([c[0] for c in candidates], scores), key=lambda x: -x[1])
        return [(int(i), float(s)) for i, s in ranked[:top_k]]


# ---------------------------------------------------------------------------
# Query expansion (multi-query via LLM)
# ---------------------------------------------------------------------------

def expand_query(llm: LLM, question: str, n_variants: int = 3) -> List[str]:
    """Generate alternative query formulations using the LLM.

    Used at level 4 (query expansion). The original question plus LLM-generated
    variants are each used for retrieval, and results are fused. This helps when
    the user's wording differs from the document's wording.

    Args:
        llm: LLM backend for generating variants.
        question: The user's original question.
        n_variants: Maximum number of variant queries to generate.

    Returns:
        List of queries starting with the original, followed by up to n_variants
        LLM-generated variants.
    """
    prompt = f"""Given the user question, generate {n_variants} different search queries that would help retrieve relevant information from a document. Return one query per line, no numbering.

User question: {question}

Queries:"""
    raw = llm.generate(prompt)
    variants = [line.strip() for line in raw.splitlines() if line.strip()]
    return [question] + variants[:n_variants]


# ---------------------------------------------------------------------------
# Retrieval strategies
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    lists: List[List[Tuple[int, float]]], k: int = 60
) -> List[Tuple[int, float]]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion (RRF).

    RRF assigns each document a score of sum(1 / (k + rank)) across all lists,
    where rank is its position in that list. Documents that appear highly in
    multiple lists rise to the top. The constant k=60 dampens the impact of
    rank position, following the standard RRF formulation.

    Args:
        lists: A list of ranked lists, each being a list of (doc_id, score).
        k: RRF smoothing constant (default 60).

    Returns:
        A single ranked list of (doc_id, rrf_score) pairs sorted by descending
        RRF score.
    """
    scores: Dict[int, float] = {}
    for ranked_list in lists:
        for rank, (doc_id, _) in enumerate(ranked_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def retrieve_naive(
    question: str, embedder: Embedder, vs: VectorStore, top_k: int = TOP_K
) -> List[Tuple[int, float]]:
    """Dense-only retrieval: embed the question and search the vector store.

    Used at level 0 (naive RAG). No keyword search, no reranking, no expansion.

    Args:
        question: The user question.
        embedder: Embedder instance for encoding the query.
        vs: VectorStore built over document chunks.
        top_k: Number of chunks to return.

    Returns:
        List of (chunk_index, cosine_similarity) pairs.
    """
    q_emb = embedder.encode([question])[0]
    return vs.search(q_emb, top_k=top_k)


def retrieve_hybrid(
    question: str,
    embedder: Embedder,
    vs: VectorStore,
    bm25: BM25Index,
    top_k: int = TOP_K,
    candidate_k: int = HYBRID_TOP_K,
) -> List[Tuple[int, float]]:
    """Hybrid retrieval: dense vector search + BM25 keyword search, fused by RRF.

    Used at levels 1, 2, 3, 4, 6 as the base retrieval step. Retrieves
    candidate_k results from each index, fuses them with RRF, and returns
    the fused ranking (later narrowed to top_k by the caller).

    Args:
        question: The user question.
        embedder: Embedder instance for encoding the query.
        vs: VectorStore built over document chunks.
        bm25: BM25Index built over document chunks.
        top_k: Number of results to return after fusion.
        candidate_k: Number of candidates to retrieve from each index before fusion.

    Returns:
        List of (chunk_index, rrf_score) pairs, sorted by descending score,
        truncated to top_k.
    """
    q_emb = embedder.encode([question])[0]
    vector_hits = vs.search(q_emb, top_k=candidate_k)
    bm25_hits = bm25.search(question, top_k=candidate_k)
    fused = reciprocal_rank_fusion([vector_hits, bm25_hits])
    return fused[:top_k]


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def build_context(chunks: List[str], hits: List[Tuple[int, float]]) -> str:
    """Assemble retrieved chunks into a single context string for the prompt.

    Each chunk is prefixed with its index and score so the model can see
    provenance. Chunks are joined with a separator.

    Args:
        chunks: List of chunk text strings (either child or parent chunks).
        hits: List of (chunk_index, score) pairs from retrieval.

    Returns:
        A formatted context string with one chunk per section.
    """
    parts = []
    for doc_id, score in hits:
        parts.append(f"[Chunk {doc_id} (score {score:.3f})]\n{chunks[doc_id]}")
    return "\n\n---\n\n".join(parts)


def fit_context(context: str, max_chars: int = MAX_CONTEXT_CHARS) -> Tuple[str, bool]:
    """Truncate context to max_chars, cutting at a clean boundary when possible.

    Enforced at ALL levels so the LLM always sees the same maximum context
    budget, preventing the full-context baseline (level 5) from winning simply
    by stuffing more text into the prompt.

    Cutting preference:
      1. Newline boundary within the last 200 chars of the cap (keeps chunks
         or sections intact where possible).
      2. Sentence end (". ", "! ", "? ") as a fallback.
      3. Hard cut at max_chars if no boundary is found.

    Args:
        context: The full context string to potentially truncate.
        max_chars: Maximum allowed characters in the returned context.

    Returns:
        A tuple of (truncated_context, was_truncated).
    """
    if len(context) <= max_chars:
        return context, False

    cut = context[:max_chars]
    # Prefer cutting at a newline boundary within the last 200 chars
    newline_idx = cut.rfind("\n", max(0, max_chars - 200))
    if newline_idx != -1:
        return cut[:newline_idx], True

    # Fallback: cut at a sentence end
    sentence_idx = max(
        cut.rfind(". "),
        cut.rfind("! "),
        cut.rfind("? "),
    )
    if sentence_idx != -1:
        return cut[: sentence_idx + 1], True

    return cut, True


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    """Result of a single evaluation question at a single experiment level.

    Every field is JSON-serializable and written to the output JSONL file.
    The config sub-dict captures all held-fixed constants so the output is
    self-describing: you can reproduce the experiment from the file alone.
    """

    level: int
    config: dict
    question: str
    answer: str
    reference_answer: Optional[str]
    retrieved_chunks: List[dict]
    context_chars: int
    truncated: bool
    latency_seconds: float
    timestamp: str


class Experiment:
    """Runs one level of the progressive RAG experiment on a single document.

    Each Experiment instance is bound to a single level (0-6). All constants
    (document, model, embeddings, chunking, context cap, thinking level, etc.)
    are captured as instance attributes and recorded in every RunResult so the
    experimental conditions are fully reproducible from the output file alone.

    Attributes:
        level: The experiment level (0-6), determining which retrieval/ranking
            features are active.
        document_path, embedding_model_name, llm_backend, llm_model_name,
        chunk_size, chunk_overlap, top_k, hybrid_top_k, max_context_chars,
        thinking_level, prompt_template, parent_chunk_size, parent_chunk_overlap:
            Experimental constants, mirrored from module-level values.
        openrouter_base_url, openrouter_api_key_set, openrouter_headers:
            OpenRouter connection metadata, recorded for reproducibility.
    """

    def __init__(self, level: int, questions: Optional[List[str]] = None, reference_answers: Optional[Dict[str, str]] = None):
        self.level = level

        # Record all constants as instance attributes so they appear verbatim
        # in every RunResult.config — not just defined at module level.
        # Questions are held constant across all levels for fair A/B comparison.
        self.questions = questions if questions is not None else EVAL_QUESTIONS
        # Reference answers (optional, from questions.json objects) for later scoring/manual comparison.
        self.reference_answers = reference_answers if reference_answers is not None else {}
        self.document_path = str(DOCUMENT_PATH)
        self.embedding_model_name = EMBEDDING_MODEL_NAME
        self.llm_backend = LLM_BACKEND
        self.llm_model_name = LLM_MODEL_NAME
        self.chunk_size = CHUNK_SIZE
        self.chunk_overlap = CHUNK_OVERLAP
        self.top_k = TOP_K
        self.hybrid_top_k = HYBRID_TOP_K
        self.max_context_chars = MAX_CONTEXT_CHARS
        self.thinking_level = THINKING_LEVEL
        self.prompt_template = PROMPT_TEMPLATE
        self.openrouter_base_url = OPENROUTER_BASE_URL
        self.openrouter_api_key_set = bool(OPENROUTER_API_KEY)
        self.openrouter_headers = OPENROUTER_HEADERS
        self.parent_chunk_size = PARENT_CHUNK_SIZE
        self.parent_chunk_overlap = PARENT_CHUNK_OVERLAP

        self.document = load_document(DOCUMENT_PATH)

        # Chunk the document twice:
        #   - child chunks (512 chars): retrieval unit at ALL levels
        #   - parent chunks (1536 chars): context unit at parent-child levels (2, 6)
        self.child_chunks, self.child_offsets = chunk_with_offsets(
            self.document, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP
        )
        self.parent_chunks, self.parent_offsets = chunk_with_offsets(
            self.document, size=PARENT_CHUNK_SIZE, overlap=PARENT_CHUNK_OVERLAP
        )

        # Map each child chunk to the parent chunk(s) that contain it.
        # Enables parent-child retrieval: retrieve children, return parents.
        self.child_to_parent = build_child_to_parent_map(
            self.child_offsets, self.parent_offsets
        )

        self.embedder = Embedder()
        # Vector store and BM25 are built on child chunks only. All levels share
        # the same retrieval unit; parent-child changes only which chunks are
        # returned as context, not which chunks are searched.
        self.vs = VectorStore(self.embedder, self.child_chunks)
        self.bm25 = BM25Index(self.child_chunks)
        # Cross-encoder reranker is only instantiated at levels 3 and 6.
        self.reranker = Reranker() if level in (3, 6) else None
        self.llm = make_llm()

    def retrieve(
        self, question: str
    ) -> Tuple[List[Tuple[int, float]], bool]:
        """Retrieve candidate chunks for a question at this experiment level.

        Returns:
            A tuple (hits, use_parents) where:
              - hits: list of (chunk_index, score) pairs. chunk_index indexes
                into self.child_chunks unless use_parents is True, in which
                case it indexes into self.parent_chunks.
              - use_parents: True at levels 2 and 6 (parent-child retrieval),
                False otherwise.

        Level behavior:
          - Level 0: dense (vector) search only, return top_k child chunks.
          - Level 1: hybrid (vector + BM25 via RRF), return top_k child chunks.
          - Level 2: hybrid search on children, map to whole parent chunks as context.
          - Level 3: hybrid + cross-encoder reranking, return top_k child chunks.
          - Level 4: hybrid + query expansion (multi-query), return top_k child chunks.
          - Level 5: full-context baseline, no retrieval (returns empty).
          - Level 6: all advanced (hybrid + parent-child + rerank + expansion).

        Parent-child selection (levels 2 and 6):
          Retrieved children are mapped to their containing parent chunks.
          Multiple children may map to the same parent, so parents are
          deduplicated (keeping the best child score). Whole parents are then
          selected greedily by descending score until the next parent would
          exceed MAX_CONTEXT_CHARS. Parents are never truncated mid-text — if
          only one parent fits, only one is returned.
        """
        # Level 5 is the full-context baseline: no retrieval at all.
        if self.level == 5:
            return [], False

        # Levels 2 and 6 return parent chunks as context instead of raw children.
        use_parents = self.level in (2, 6)

        if self.level == 0:
            # Naive RAG: vector search only.
            child_hits = retrieve_naive(
                question, self.embedder, self.vs, top_k=TOP_K
            )
        elif self.level == 1:
            # Hybrid: vector + BM25 fused by RRF.
            child_hits = retrieve_hybrid(
                question, self.embedder, self.vs, self.bm25,
                top_k=TOP_K, candidate_k=HYBRID_TOP_K,
            )
        elif self.level in (2, 3, 4, 6):
            # Levels 2, 3, 4, 6 all start from a hybrid candidate pool over children.
            if self.level == 4:
                # Query expansion: retrieve for the original question plus
                # LLM-generated variants, then fuse all result lists by RRF.
                variants = expand_query(self.llm, question)
                all_hits = []
                for q in variants:
                    h = retrieve_hybrid(
                        q, self.embedder, self.vs, self.bm25,
                        top_k=HYBRID_TOP_K, candidate_k=HYBRID_TOP_K,
                    )
                    all_hits.append(h)
                fused = reciprocal_rank_fusion(all_hits)
                child_hits = fused[:HYBRID_TOP_K]
            else:
                child_hits = retrieve_hybrid(
                    question, self.embedder, self.vs, self.bm25,
                    top_k=HYBRID_TOP_K, candidate_k=HYBRID_TOP_K,
                )

            # Apply cross-encoder reranking only when the reranker exists
            # (levels 3 and 6). Otherwise keep the top_k hybrid results.
            if self.reranker is not None:
                candidate_texts = [(i, self.child_chunks[i]) for i, _ in child_hits]
                child_hits = self.reranker.rerank(
                    question, candidate_texts, top_k=TOP_K
                )
            else:
                child_hits = child_hits[:TOP_K]
        else:
            # Fallback: dense-only retrieval (should not be reached with
            # current level definitions, but kept for safety).
            child_hits = retrieve_naive(
                question, self.embedder, self.vs, top_k=TOP_K
            )

        if use_parents:
            # Map retrieved children to their containing parent chunks.
            # Deduplicate parents: keep the best score per parent.
            parent_best_score: Dict[int, float] = {}
            for child_idx, score in child_hits:
                for parent_idx in self.child_to_parent[child_idx]:
                    if parent_idx not in parent_best_score or score > parent_best_score[parent_idx]:
                        parent_best_score[parent_idx] = score

            # Sort parents by score descending.
            sorted_parents = sorted(parent_best_score.items(), key=lambda x: -x[1])

            # Greedily select whole parents that fit within MAX_CONTEXT_CHARS.
            # Account for per-parent formatting overhead ("[Chunk i (score s)]\n"
            # header + separator). Parents are never truncated mid-text.
            parent_hits: List[Tuple[int, float]] = []
            budget = MAX_CONTEXT_CHARS
            for parent_idx, score in sorted_parents:
                parent_text_len = len(self.parent_chunks[parent_idx]) + CHUNK_CONTEXT_OVERHEAD
                if parent_text_len <= budget:
                    parent_hits.append((parent_idx, score))
                    budget -= parent_text_len
                else:
                    # Parent too large for remaining budget; skip it and try
                    # the next smaller-scored parent (which may be shorter).
                    continue

            return parent_hits, True

        return child_hits, False

    def run_question(self, question: str) -> RunResult:
        """Run a single evaluation question through this experiment level.

        The full flow per question:
          1. Build context (retrieve chunks, or use whole document at level 5).
          2. Cap context at MAX_CONTEXT_CHARS via fit_context().
          3. Format the prompt with PROMPT_TEMPLATE.
          4. Generate an answer via the LLM with THINKING_LEVEL.
          5. Record latency, context length, truncation flag, and full config.

        Args:
            question: The evaluation question to answer.

        Returns:
            A RunResult dataclass containing the answer, retrieved chunks,
            context metadata, and the complete experiment configuration.
        """
        start = time.time()

        if self.level == 5:
            # Full-context baseline: entire document as context (then capped).
            # No retrieval; the model must find the answer in the whole document.
            context = self.document
            retrieved_chunks = []
        else:
            hits, use_parents = self.retrieve(question)
            context_chunks = self.parent_chunks if use_parents else self.child_chunks
            context = build_context(context_chunks, hits)
            retrieved_chunks = [
                {
                    "chunk_id": i,
                    "score": s,
                    "text": context_chunks[i][:200] + "...",
                    "parent": use_parents,
                }
                for i, s in hits
            ]

        # Enforce constant context window across ALL levels.
        context, truncated = fit_context(context, MAX_CONTEXT_CHARS)
        context_chars = len(context)

        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        answer = self.llm.generate(prompt, thinking_level=self.thinking_level)
        latency = time.time() - start

        # Manual config dict records every held-fixed constant for
        # reproducibility. asdict(self) would fail because Experiment is not
        # a dataclass and would include non-serializable objects
        # (embedder, vector store, LLM client, etc.).
        config = {
            "level": self.level,
            "document_path": self.document_path,
            "embedding_model_name": self.embedding_model_name,
            "llm_backend": self.llm_backend,
            "llm_model_name": self.llm_model_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "parent_chunk_size": self.parent_chunk_size,
            "parent_chunk_overlap": self.parent_chunk_overlap,
            "top_k": self.top_k,
            "hybrid_top_k": self.hybrid_top_k,
            "max_context_chars": self.max_context_chars,
            "thinking_level": self.thinking_level,
            "prompt_template": self.prompt_template,
            "openrouter_base_url": self.openrouter_base_url,
            "openrouter_api_key_set": self.openrouter_api_key_set,
            "openrouter_headers": self.openrouter_headers,
        }

        return RunResult(
            level=self.level,
            config=config,
            question=question,
            answer=answer,
            reference_answer=self.reference_answers.get(question),
            retrieved_chunks=retrieved_chunks,
            context_chars=context_chars,
            truncated=truncated,
            latency_seconds=round(latency, 3),
            timestamp=datetime.utcnow().isoformat() + "Z",
        )

    def run_all(self) -> List[RunResult]:
        """Run all evaluation questions at this experiment level.

        Returns:
            List of RunResult, one per question in self.questions.
        """
        return [self.run_question(q) for q in self.questions]


def save_results(results: List[RunResult], level: int) -> Path:
    """Save experiment results to a JSONL file.

    Each RunResult is serialized as one JSON line. The file name includes
    the level and a UTC timestamp so multiple runs of the same level don't
    overwrite each other.

    Args:
        results: List of RunResult objects from a single level run.
        level: The experiment level (0-6) these results belong to.

    Returns:
        Path to the written JSONL file.
    """
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"level_{level}_{ts}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# Human-readable description of each experiment level, printed by main().
# Updated whenever level behavior changes.
LEVEL_DESCRIPTIONS = {
    0: "Naive RAG (vector search only, return child chunks)",
    1: "Hybrid RAG (vector + BM25, RRF fused, return child chunks)",
    2: "Parent-child retrieval (vector search children, return parent chunks)",
    3: "Hybrid + Cross-encoder reranking (return child chunks)",
    4: "Hybrid + Query expansion (multi-query, return child chunks)",
    5: "Full-context baseline (no retrieval, whole document capped)",
    6: "All advanced (hybrid + parent-child + rerank + query expansion)",
}


def main():
    """Command-line entry point for the progressive RAG experiment.

    Parses CLI arguments, optionally overrides the document path, then runs
    the requested level(s) sequentially. For each level it:
      1. Constructs an Experiment instance (loads document, builds chunks,
         initializes embedder, vector store, BM25, optional reranker, LLM).
      2. Calls run_all() to answer every EVAL_QUESTIONS at that level.
      3. Saves results to results/level_<N>_<timestamp>.jsonl.
      4. Prints a summary of each answer to stdout.

    CLI flags:
        --level N          Run a single level (0-6).
        --levels N N       Run multiple levels.
        --document         Path to the document under test (overrides DOCUMENT_PATH).
        --questions-file   Path to JSON file with list of question strings (overrides EVAL_QUESTIONS).

    All other experimental conditions are fixed by module-level constants.
    """
    # Allow CLI override of document path and questions (keeps everything else constant)
    global DOCUMENT_PATH, QUESTIONS_FILE
    parser = argparse.ArgumentParser(description="Progressive RAG experiment on one document")
    parser.add_argument("--level", type=int, choices=list(LEVEL_DESCRIPTIONS.keys()),
                        help="Run a single level")
    parser.add_argument("--levels", nargs="+", type=int, choices=list(LEVEL_DESCRIPTIONS.keys()),
                        help="Run multiple levels")
    parser.add_argument("--document", type=Path, default=DOCUMENT_PATH,
                        help="Path to the single document (overrides CONSTANTS)")
    parser.add_argument("--questions-file", type=Path, default=QUESTIONS_FILE,
                        help="JSON file with list of question strings (overrides EVAL_QUESTIONS)")
    args = parser.parse_args()

    if args.document != DOCUMENT_PATH:
        DOCUMENT_PATH = args.document

    if args.questions_file is not None:
        QUESTIONS_FILE = args.questions_file

    # Load questions from JSON file if provided, otherwise use EVAL_QUESTIONS constant
    # Supports both flat strings ["Q1", "Q2"] and objects [{"question": "Q1", "reference_answer": "A1"}]
    reference_answers: Dict[str, str] = {}
    if QUESTIONS_FILE is not None:
        with QUESTIONS_FILE.open("r", encoding="utf-8") as f:
            raw_questions = json.load(f)
        if not isinstance(raw_questions, list):
            raise ValueError(f"{QUESTIONS_FILE} must contain a JSON list")
        questions = []
        for item in raw_questions:
            if isinstance(item, str):
                questions.append(item)
            elif isinstance(item, dict) and "question" in item:
                questions.append(item["question"])
                if "reference_answer" in item and isinstance(item["reference_answer"], str):
                    reference_answers[item["question"]] = item["reference_answer"]
            else:
                raise ValueError(f"Each item in {QUESTIONS_FILE} must be a string or an object with a 'question' key")
    else:
        questions = EVAL_QUESTIONS

    levels = args.levels or ([args.level] if args.level is not None else list(LEVEL_DESCRIPTIONS.keys()))

    for level in levels:
        print(f"\n=== Level {level}: {LEVEL_DESCRIPTIONS[level]} ===")
        exp = Experiment(level, questions=questions, reference_answers=reference_answers)
        results = exp.run_all()
        path = save_results(results, level)
        print(f"Saved {len(results)} results to {path}")
        for r in results:
            print(f"\nQ: {r.question}")
            print(f"A: {r.answer[:300]}{'...' if len(r.answer) > 300 else ''}")
            print(f"(latency: {r.latency_seconds}s, chunks: {len(r.retrieved_chunks)})")


if __name__ == "__main__":
    main()