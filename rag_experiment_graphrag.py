#!/usr/bin/env python3
"""
Progressive RAG Experiment — GraphRAG variant (separate script).

This is a SEPARATE experiment from rag_experiment.py (levels a–h). GraphRAG
replaces the chunk-based retrieval paradigm entirely, so it cannot be added
as another level (i, j, etc.) in the controlled script without breaking the
constant-enforcement principle.

Design principle: hold everything constant except the feature under test.
Where a constant cannot apply to GraphRAG (e.g., fixed-size chunking), it is
removed and the reason is documented below. Where a constant changes, the
old and new values are noted.

GraphRAG pipeline:
  1. Index-time: LLM extracts entities + relationships from document chunks.
  2. Index-time: Build graph (nodes = entities, edges = relationships).
  3. Index-time: Detect communities, generate community summaries via LLM.
  4. Query-time: LLM identifies relevant entities from the question.
  5. Query-time: Traverse graph to find connected entities/communities.
  6. Query-time: Retrieve community summaries + entity descriptions as context.
  7. Query-time: LLM generates answer using graph-structured context.

Usage:
  python rag_experiment_graphrag.py
  python rag_experiment_graphrag.py --questions-file questions.json

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
from typing import List, Tuple, Dict, Optional, Iterable, Set

import numpy as np


# ---------------------------------------------------------------------------
# .env loader (no external dependency)
# ---------------------------------------------------------------------------

def _load_env() -> None:
    """Load KEY=VALUE pairs from a .env file in the current working directory."""
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
# CONSTANT ANALYSIS: GraphRAG vs chunk-based RAG (rag_experiment.py levels a-h)
# =============================================================================
#
# APPLY — kept identical for fair A/B comparison:
#   DOCUMENT_PATH       Same document under test (NovaCluster ops manual)
#   EMBEDDING_MODEL_NAME Still needed to embed entities/communities for retrieval
#   LLM_BACKEND         Still OpenRouter for all LLM calls
#   LLM_MODEL_NAME      Same model (gpt-4o-mini) for fair comparison
#   THINKING_LEVEL      Same reasoning effort ("low")
#   TOP_K               Final number of entities/communities passed to LLM
#   MAX_CONTEXT_CHARS   Same context cap (~5,464 chars) for fair comparison
#   PROMPT_TEMPLATE     Same structure ({context} + {question}), different content
#   OPENROUTER_*        Same API connection details
#   EVAL_QUESTIONS      Same questions for A/B comparison across scripts
#   RESULTS_DIR         Same output directory
#   CONFIDENCE_THRESHOLD Can abstain if no relevant entities/communities found
#
# DON'T APPLY — removed because GraphRAG doesn't use fixed-size chunks:
#   CHUNK_SIZE          No fixed-size character chunks (entity-based instead)
#   CHUNK_OVERLAP       No chunk overlap
#   PARENT_CHUNK_SIZE   No parent-child chunking
#   PARENT_CHUNK_OVERLAP No parent-child chunking
#   HYBRID_TOP_K        No hybrid vector+BM25 fusion
#   CHUNK_CONTEXT_OVERHEAD Different context formatting (graph subgraphs)
#   CHILD_CONTEXT_CHARS Derived from chunk sizes, not applicable
#   PARENT_CONTEXT_CHARS Derived from chunk sizes, not applicable
#   WEIGHT_CONTEXT_BY_SCORE Different scoring (graph relevance, not chunk score)
#   USE_WEIGHTED_RRF     No RRF fusion in GraphRAG
#
# CHANGE — modified for GraphRAG:
#   Index-time processing: Adds LLM entity/relationship extraction (new step)
#   Retrieval unit: From chunk -> graph node (entity) / community
#   Context building: From raw text -> structured graph subgraphs + summaries
#   Prompt template: Context now includes entity descriptions + community summaries
# =============================================================================

# ---------------------------------------------------------------------------
# Constants that APPLY (unchanged from rag_experiment.py)
# ---------------------------------------------------------------------------

DOCUMENT_PATH = Path("document.txt")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_BACKEND = "openrouter"
LLM_MODEL_NAME = "openai/gpt-4o-mini"
THINKING_LEVEL = "low"
TOP_K = 5
MAX_CONTEXT_CHARS = 5464  # Same cap as rag_experiment.py for fair comparison
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/psg/rag-experiment",
    "X-Title": "RAG Experiment GraphRAG",
}
EVAL_QUESTIONS: List[str] = [
    "What is this document about?",
    "Can you summarize the key points?",
]
QUESTIONS_FILE: Optional[Path] = None
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Confidence-based toggle (applies to GraphRAG)
CONFIDENCE_THRESHOLD = None  # Abstain if top entity/community score < value


# ---------------------------------------------------------------------------
# Constants that CHANGE (GraphRAG-specific)
# ============================================================================
#
# ENTITY_EXTRACTION_MODEL: LLM used to extract entities + relationships at
#   index time. Separate from LLM_MODEL_NAME because extraction can use a
#   cheaper/faster model while generation uses gpt-4o-mini. Recorded for
#   reproducibility.
#
# COMMUNITY_DETECTION_ALGORITHM: Method for finding communities in the graph.
#   Options: "louvain", "leiden", "connected_components", "label_propagation".
#
# MAX_ENTITIES_PER_COMMUNITY: Cap on entities included in a community summary
#   to keep context within MAX_CONTEXT_CHARS.
#
# GRAPH_EDGE_WEIGHT_THRESHOLD: Minimum relationship confidence to include an
#   edge in the graph (filters noisy extractions).
#
# MAX_COMMUNITIES_RETURNED: Max communities retrieved per query before
#   truncation to MAX_CONTEXT_CHARS.
# ---------------------------------------------------------------------------

ENTITY_EXTRACTION_MODEL = "openai/gpt-4o-mini"
ENTITY_EXTRACTION_THINKING = "low"
COMMUNITY_DETECTION_ALGORITHM = "connected_components"
MAX_ENTITIES_PER_COMMUNITY = 10
GRAPH_EDGE_WEIGHT_THRESHOLD = 0.3
MAX_COMMUNITIES_RETURNED = 5

# Prompt template for GraphRAG: context includes entity descriptions and
# community summaries instead of raw chunk text.
PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question using ONLY the provided context.
The context contains entity descriptions and community summaries from a knowledge graph built from the document.
If the answer is not in the context, say "I don't know based on the provided document."

Context:
{context}

Question:
{question}

Answer:
"""



# ---------------------------------------------------------------------------
# LLM interface (same backend as rag_experiment.py)
# ---------------------------------------------------------------------------

class LLM:
    def generate(self, prompt: str, thinking_level: str = THINKING_LEVEL) -> str:
        raise NotImplementedError


class OpenAILLM(LLM):
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
        kwargs = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }
        if thinking_level:
            kwargs["reasoning"] = {"effort": thinking_level}
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception as e:
            if "reasoning" in str(e).lower() and thinking_level:
                kwargs.pop("reasoning", None)
                resp = self.client.chat.completions.create(**kwargs)
            else:
                raise
        return resp.choices[0].message.content or ""


class StubLLM(LLM):
    def generate(self, prompt: str, thinking_level: str = THINKING_LEVEL) -> str:
        return f"[STUB LLM thinking={thinking_level}] Would answer based on {len(prompt)} chars of prompt."


def make_llm(model_name: str = LLM_MODEL_NAME) -> LLM:
    if LLM_BACKEND == "openrouter":
        return OpenAILLM(model_name)
    if LLM_BACKEND == "stub":
        return StubLLM()
    raise ValueError(f"Unknown LLM_BACKEND: {LLM_BACKEND}")


# ---------------------------------------------------------------------------
# Document loading & chunking (for index-time entity extraction)
# ============================================================================
#
# GraphRAG still chunks the document at index time, but ONLY to feed manageable
# pieces to the entity-extraction LLM. These chunks are NOT the retrieval unit
# and are NOT returned to the LLM at query time. They are discarded after the
# graph is built.
#
# CHANGED: Chunks are an index-time implementation detail, not a retrieval or
# context unit. The same CHUNK_SIZE/CHUNK_OVERLAP from rag_experiment.py could
# be reused, but they are NOT enforced as experiment constants here because
# they don't affect the query-time context the LLM sees.
# ---------------------------------------------------------------------------

def load_document(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def chunk_text(text: str, size: int = 512, overlap: int = 50) -> List[str]:
    """Fixed-size character chunks for index-time entity extraction only.

    These chunks are an implementation detail of index-time processing. They
    are NOT the retrieval unit and NOT returned as context at query time.
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


# ---------------------------------------------------------------------------
# Entity extraction (index-time LLM pass)
# ============================================================================
#
# NEW for GraphRAG: An LLM pass over each document chunk extracts entities
# (people, products, error codes, commands, concepts) and relationships
# (e.g., "ERR-003 indicates health check failure", "kubectl drain drains nodes").
#
# This is the single biggest departure from rag_experiment.py:
#   - rag_experiment.py: No index-time LLM. Document is chunked and embedded.
#   - GraphRAG: Index-time LLM extracts structured knowledge from the document.
#
# This breaks the "document processing is constant" principle of the
# controlled experiment, which is why GraphRAG is a SEPARATE script.
# ---------------------------------------------------------------------------

def extract_entities_and_relationships(
    llm: LLM, chunk: str, chunk_idx: int
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Use the LLM to extract entities and relationships from a document chunk.

    Returns:
        entities: List of {"name": str, "type": str, "description": str, "source_chunk": int}
        relationships: List of {"source": str, "target": str, "relation": str, "confidence": float, "source_chunk": int}
    """
    prompt = f"""Extract entities and relationships from the following text. Return TWO sections.

Text (from document chunk {chunk_idx}):
{chunk}

Entities:
- List each entity as: NAME | TYPE | DESCRIPTION
- Types: error_code, command, product, concept, person, role, procedure, tool, file, metric

Relationships:
- List each relationship as: SOURCE | RELATION | TARGET | CONFIDENCE (0.0-1.0)
- Confidence: how certain you are about this relationship

Example:
Entities:
- ERR-003 | error_code | Health check failure during cluster restart
- kubectl drain | command | Safely evicts pods from a node before maintenance
- NovaCluster | product | Distributed computing platform

Relationships:
- ERR-003 | indicates | health check failure | 0.9
- kubectl drain | used_in | cluster restart procedure | 0.85
- NovaCluster | uses | kubectl drain | 0.8

Begin:

Entities:
"""
    raw = llm.generate(prompt, thinking_level=ENTITY_EXTRACTION_THINKING)

    # Parse entities section
    entities: List[Dict[str, str]] = []
    relationships: List[Dict[str, str]] = []

    current_section = None
    for line in raw.splitlines():
        line = line.strip()
        if line == "Entities:":
            current_section = "entities"
            continue
        if line == "Relationships:":
            current_section = "relationships"
            continue
        if not line or line.startswith("-") is False:
            continue

        # Remove leading "- " or "• "
        line = line.lstrip("- •").strip()
        if not line:
            continue

        parts = [p.strip() for p in line.split("|")]
        if current_section == "entities" and len(parts) >= 3:
            entities.append({
                "name": parts[0],
                "type": parts[1],
                "description": parts[2],
                "source_chunk": str(chunk_idx),
            })
        elif current_section == "relationships" and len(parts) >= 4:
            try:
                conf = float(parts[3])
            except ValueError:
                conf = 0.5
            relationships.append({
                "source": parts[0],
                "relation": parts[1],
                "target": parts[2],
                "confidence": str(conf),
                "source_chunk": str(chunk_idx),
            })

    return entities, relationships


# ---------------------------------------------------------------------------
# Graph construction
# ============================================================================
#
# GraphRAG stores entities as nodes and relationships as edges. Unlike
# rag_experiment.py's VectorStore (which stores embeddings per chunk),
# GraphRAG stores a graph structure.
#
# CHANGED: Storage is a graph (nodes + edges), not a vector index + BM25 index.
# ---------------------------------------------------------------------------

class KnowledgeGraph:
    """Simple in-memory knowledge graph: entities as nodes, relationships as edges.

    Entities are indexed by name (case-insensitive lookup). Relationships are
    directed edges with a confidence score.
    """

    def __init__(self):
        self.entities: Dict[str, Dict[str, str]] = {}  # name -> {type, description, source_chunks}
        self.edges: List[Tuple[str, str, str, float]] = []  # (source, target, relation, confidence)

    def add_entity(self, name: str, etype: str, description: str, source_chunk: str):
        key = name.lower().strip()
        if key in self.entities:
            existing = self.entities[key]
            # Merge descriptions, keep unique source chunks
            chunks = set(existing.get("source_chunks", "").split(","))
            chunks.discard("")
            chunks.add(source_chunk)
            existing["source_chunks"] = ",".join(sorted(chunks))
            if len(description) > len(existing.get("description", "")):
                existing["description"] = description
        else:
            self.entities[key] = {
                "name": name,
                "type": etype,
                "description": description,
                "source_chunks": source_chunk,
            }

    def add_edge(self, source: str, target: str, relation: str, confidence: float):
        if confidence < GRAPH_EDGE_WEIGHT_THRESHOLD:
            return
        self.edges.append((source.strip(), target.strip(), relation.strip(), confidence))

    def neighbors(self, entity_name: str) -> List[Tuple[str, str, float]]:
        """Return (neighbor_name, relation, confidence) for edges involving entity."""
        key = entity_name.lower().strip()
        result = []
        for src, tgt, rel, conf in self.edges:
            if src.lower() == key:
                result.append((tgt, rel, conf))
            elif tgt.lower() == key:
                result.append((src, rel, conf))
        return result

    def get_entity(self, name: str) -> Optional[Dict[str, str]]:
        return self.entities.get(name.lower().strip())

    def all_entity_names(self) -> List[str]:
        return [e["name"] for e in self.entities.values()]


# ---------------------------------------------------------------------------
# Community detection
# ============================================================================
#
# GraphRAG detects communities (groups of tightly connected entities) and
# generates summaries for each. This is the GraphRAG equivalent of "chunks"
# in rag_experiment.py — communities are the retrieval unit.
#
# CHANGED: Communities replace chunks as the unit of organization. Community
# summaries replace raw chunk text as context.
# ---------------------------------------------------------------------------

def detect_communities(graph: KnowledgeGraph) -> List[Set[str]]:
    """Detect communities using simple connected components.

    This is a basic implementation. For production, replace with Louvain or Leiden.
    """
    visited: Set[str] = set()
    communities: List[Set[str]] = []

    # Build adjacency
    adj: Dict[str, Set[str]] = {name: set() for name in graph.entities}
    for src, tgt, _, _ in graph.edges:
        src_l = src.lower()
        tgt_l = tgt.lower()
        if src_l in adj and tgt_l in adj:
            adj[src_l].add(tgt_l)
            adj[tgt_l].add(src_l)

    for name in graph.entities:
        name_l = name.lower()
        if name_l in visited:
            continue
        # BFS
        community: Set[str] = set()
        stack = [name_l]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            community.add(node)
            for neighbor in adj.get(node, set()):
                if neighbor not in visited:
                    stack.append(neighbor)
        communities.append(community)

    return communities


def summarize_community(
    llm: LLM, community: Set[str], graph: KnowledgeGraph
) -> str:
    """Generate an LLM summary of a community (group of related entities)."""
    entity_descriptions = []
    for name_l in community:
        ent = graph.entities.get(name_l)
        if ent:
            entity_descriptions.append(f"- {ent['name']} ({ent['type']}): {ent['description']}")

    edges_in_community = []
    for src, tgt, rel, conf in graph.edges:
        if src.lower() in community and tgt.lower() in community:
            edges_in_community.append(f"- {src} --{rel}--> {tgt} (confidence: {conf:.2f})")

    prompt = f"""Summarize the following group of related entities and their relationships from a document.
Write a concise paragraph (2-4 sentences) explaining what this group is about and how the entities relate.

Entities:
{chr(10).join(entity_descriptions) if entity_descriptions else "(none)"}

Relationships:
{chr(10).join(edges_in_community) if edges_in_community else "(none)"}

Summary:"""
    return llm.generate(prompt, thinking_level=THINKING_LEVEL)


# ---------------------------------------------------------------------------
# Graph index builder (index-time)
# ============================================================================
#
# Builds the full knowledge graph from the document: chunk -> extract entities
# -> build graph -> detect communities -> summarize communities.
#
# This replaces rag_experiment.py's index-time step (chunk + embed + store).
# ---------------------------------------------------------------------------

def build_graph_index(
    document: str, llm: LLM
) -> Tuple[KnowledgeGraph, List[Dict[str, str]]]:
    """Build a knowledge graph from the document.

    Returns:
        graph: The built KnowledgeGraph
        community_summaries: List of {"community_id": int, "entities": List[str], "summary": str}
    """
    chunks = chunk_text(document)
    graph = KnowledgeGraph()

    # Extract entities and relationships from each chunk
    all_entities: List[Dict[str, str]] = []
    all_relationships: List[Dict[str, str]] = []
    for i, chunk in enumerate(chunks):
        ents, rels = extract_entities_and_relationships(llm, chunk, i)
        all_entities.extend(ents)
        all_relationships.extend(rels)

    # Populate graph
    for ent in all_entities:
        graph.add_entity(
            ent["name"], ent["type"], ent["description"], ent["source_chunk"]
        )
    for rel in all_relationships:
        graph.add_edge(
            rel["source"], rel["target"], rel["relation"], float(rel["confidence"])
        )

    # Detect communities and summarize
    communities = detect_communities(graph)
    community_summaries = []
    for i, community in enumerate(communities):
        summary = summarize_community(llm, community, graph)
        community_summaries.append({
            "community_id": i,
            "entities": sorted([graph.entities[n]["name"] for n in community if n in graph.entities]),
            "summary": summary,
        })

    return graph, community_summaries




# ---------------------------------------------------------------------------
# Graph retrieval (query-time)
# ============================================================================
#
# GraphRAG query flow:
#   1. LLM identifies relevant entities from the question.
#   2. Look up those entities in the graph.
#   3. Traverse 1-2 hops to find connected entities/communities.
#   4. Rank communities by relevance (number of matched entities + edge confidence).
#   5. Return top communities + their summaries as context.
#
# CHANGED: Retrieval is graph traversal + community ranking, not vector search.
# TOP_K now means top communities/entities, not top chunks.
# ---------------------------------------------------------------------------

def extract_query_entities(llm: LLM, question: str) -> List[str]:
    """Use the LLM to identify entities mentioned in the question."""
    prompt = f"""Identify the key entities (people, products, error codes, commands, concepts) mentioned in this question. Return one entity per line, no numbering.

Question: {question}

Entities:"""
    raw = llm.generate(prompt, thinking_level=THINKING_LEVEL)
    entities = [line.strip() for line in raw.splitlines() if line.strip()]
    # Clean up numbering if present
    cleaned = []
    for e in entities:
        e = re.sub(r"^\d+\.\s*", "", e).strip()
        e = e.lstrip("- •").strip()
        if e:
            cleaned.append(e)
    return cleaned


def rank_communities(
    query_entities: List[str],
    graph: KnowledgeGraph,
    community_summaries: List[Dict[str, str]],
    top_k: int = TOP_K,
) -> List[Tuple[int, float, Set[str]]]:
    """Rank communities by relevance to the query entities.

    Score = (number of query entities found in community) + 
            (max edge confidence connecting matched entities) / 10.
    
    Returns:
        List of (community_id, score, community_entity_names) sorted by score desc.
    """
    query_set = {e.lower().strip() for e in query_entities}
    scored = []

    for cs in community_summaries:
        cid = cs["community_id"]
        community_entities = {e.lower() for e in cs["entities"]}
        match_count = len(query_set & community_entities)

        if match_count == 0:
            # Check if any query entity has an edge into this community
            edge_bonus = 0.0
            for qe in query_entities:
                for neighbor, _, conf in graph.neighbors(qe):
                    if neighbor.lower() in community_entities:
                        edge_bonus = max(edge_bonus, conf)
            if edge_bonus == 0:
                continue
            score = edge_bonus / 10.0  # Low score for edge-only matches
        else:
            # Base score from entity matches
            score = float(match_count)
            # Bonus for high-confidence edges within community
            max_conf = 0.0
            for src, tgt, _, conf in graph.edges:
                if src.lower() in community_entities and tgt.lower() in community_entities:
                    max_conf = max(max_conf, conf)
            score += max_conf / 10.0

        scored.append((cid, score, {graph.entities[n]["name"] for n in community_entities if n in graph.entities}))

    scored.sort(key=lambda x: -x[1])
    return scored[:top_k]


def build_graph_context(
    ranked_communities: List[Tuple[int, float, Set[str]]],
    community_summaries: List[Dict[str, str]],
    graph: KnowledgeGraph,
    weight_by_score: bool = False,
) -> str:
    """Assemble community summaries into a context string for the prompt.

    Each community is prefixed with its ID and relevance score.
    """
    if weight_by_score:
        ranked_communities = sorted(ranked_communities, key=lambda x: -x[1])

    parts = []
    for cid, score, _ in ranked_communities:
        cs = next((s for s in community_summaries if s["community_id"] == cid), None)
        if cs:
            parts.append(f"[Community {cid} (relevance {score:.2f})]\n{cs['summary']}")
            # Add key entities in this community
            entities_list = ", ".join(cs["entities"][:MAX_ENTITIES_PER_COMMUNITY])
            parts.append(f"  Entities: {entities_list}")

    return "\n\n---\n\n".join(parts)


def fit_context(context: str, max_chars: int = MAX_CONTEXT_CHARS) -> Tuple[str, bool]:
    """Truncate context to max_chars, cutting at a clean boundary.

    Same logic as rag_experiment.py's fit_context — kept identical so the
    context cap constant is enforced identically across both scripts.
    """
    if len(context) <= max_chars:
        return context, False

    cut = context[:max_chars]
    newline_idx = cut.rfind("\n", max(0, max_chars - 200))
    if newline_idx != -1:
        return cut[:newline_idx], True

    sentence_idx = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    if sentence_idx != -1:
        return cut[: sentence_idx + 1], True

    return cut, True




# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    level: str
    config: dict
    question: str
    answer: str
    reference_answer: Optional[str]
    retrieved_entities: List[dict]
    context_chars: int
    truncated: bool
    top_entity_score: Optional[float]
    abstained: bool
    latency_seconds: float
    timestamp: str


class GraphRAGExperiment:
    """GraphRAG experiment on a single document.

    This is a SEPARATE experiment from rag_experiment.py. GraphRAG replaces
    chunk-based retrieval with entity/relationship graph retrieval.

    Constants:
        - APPLY (unchanged): DOCUMENT_PATH, EMBEDDING_MODEL_NAME, LLM_BACKEND,
          LLM_MODEL_NAME, THINKING_LEVEL, TOP_K, MAX_CONTEXT_CHARS,
          OPENROUTER_*, EVAL_QUESTIONS, RESULTS_DIR, CONFIDENCE_THRESHOLD
        - DON'T APPLY: CHUNK_SIZE, CHUNK_OVERLAP, PARENT_CHUNK_SIZE,
          PARENT_CHUNK_OVERLAP, HYBRID_TOP_K, CHUNK_CONTEXT_OVERHAD,
          CHILD_CONTEXT_CHARS, PARENT_CONTEXT_CHARS, WEIGHT_CONTEXT_BY_SCORE,
          USE_WEIGHTED_RRF
        - CHANGE: Index-time LLM entity extraction added; retrieval unit is
          graph community not chunk; context is community summaries not raw text
    """

    def __init__(
        self,
        questions: Optional[List[str]] = None,
        reference_answers: Optional[Dict[str, str]] = None,
    ):
        self.questions = questions if questions is not None else EVAL_QUESTIONS
        self.reference_answers = reference_answers if reference_answers is not None else {}

        # Constants that APPLY (unchanged from rag_experiment.py)
        self.document_path = str(DOCUMENT_PATH)
        self.embedding_model_name = EMBEDDING_MODEL_NAME
        self.llm_backend = LLM_BACKEND
        self.llm_model_name = LLM_MODEL_NAME
        self.thinking_level = THINKING_LEVEL
        self.top_k = TOP_K
        self.max_context_chars = MAX_CONTEXT_CHARS
        self.prompt_template = PROMPT_TEMPLATE
        self.openrouter_base_url = OPENROUTER_BASE_URL
        self.openrouter_api_key_set = bool(OPENROUTER_API_KEY)
        self.openrouter_headers = OPENROUTER_HEADERS
        self.confidence_threshold = CONFIDENCE_THRESHOLD

        # Constants that CHANGE (GraphRAG-specific)
        self.entity_extraction_model = ENTITY_EXTRACTION_MODEL
        self.entity_extraction_thinking = ENTITY_EXTRACTION_THINKING
        self.community_detection_algorithm = COMMUNITY_DETECTION_ALGORITHM
        self.max_entities_per_community = MAX_ENTITIES_PER_COMMUNITY
        self.graph_edge_weight_threshold = GRAPH_EDGE_WEIGHT_THRESHOLD
        self.max_communities_returned = MAX_COMMUNITIES_RETURNED

        # Document + graph index (built once at init, like chunking+embedding in rag_experiment.py)
        self.document = load_document(DOCUMENT_PATH)
        self.llm = make_llm(LLM_MODEL_NAME)
        self.graph, self.community_summaries = build_graph_index(self.document, self.llm)

    def retrieve(self, question: str) -> Tuple[List[Tuple[int, float, Set[str]]], bool]:
        """Retrieve relevant communities from the knowledge graph.

        Returns:
            ranked_communities: List of (community_id, score, entity_names)
            use_parents: Always False (GraphRAG doesn't use parent-child chunks)
        """
        query_entities = extract_query_entities(self.llm, question)
        ranked = rank_communities(
            query_entities, self.graph, self.community_summaries, top_k=self.top_k
        )
        return ranked, False

    def run_question(self, question: str) -> RunResult:
        start = time.time()
        abstained = False
        top_entity_score: Optional[float] = None

        hits, _ = self.retrieve(question)

        if hits:
            top_entity_score = hits[0][1]
            if (
                self.confidence_threshold is not None
                and top_entity_score < self.confidence_threshold
            ):
                abstained = True
                context = ""
                retrieved_entities = []
            else:
                context = build_graph_context(
                    hits, self.community_summaries, self.graph,
                    weight_by_score=False,
                )
                retrieved_entities = [
                    {
                        "community_id": cid,
                        "score": score,
                        "entities": list(ents),
                    }
                    for cid, score, ents in hits
                ]
        else:
            context = ""
            retrieved_entities = []

        context, truncated = fit_context(context, MAX_CONTEXT_CHARS)
        context_chars = len(context)

        if abstained:
            answer = "I don't know based on the provided document."
            latency = time.time() - start
        else:
            prompt = PROMPT_TEMPLATE.format(context=context, question=question)
            answer = self.llm.generate(prompt, thinking_level=self.thinking_level)
            latency = time.time() - start

        config = {
            "level": "graphrag",
            "document_path": self.document_path,
            "embedding_model_name": self.embedding_model_name,
            "llm_backend": self.llm_backend,
            "llm_model_name": self.llm_model_name,
            "thinking_level": self.thinking_level,
            "top_k": self.top_k,
            "max_context_chars": self.max_context_chars,
            "prompt_template": self.prompt_template,
            "openrouter_base_url": self.openrouter_base_url,
            "openrouter_api_key_set": self.openrouter_api_key_set,
            "openrouter_headers": self.openrouter_headers,
            # GraphRAG-specific constants
            "entity_extraction_model": self.entity_extraction_model,
            "entity_extraction_thinking": self.entity_extraction_thinking,
            "community_detection_algorithm": self.community_detection_algorithm,
            "max_entities_per_community": self.max_entities_per_community,
            "graph_edge_weight_threshold": self.graph_edge_weight_threshold,
            "max_communities_returned": self.max_communities_returned,
            # Constants from rag_experiment.py that do NOT apply to GraphRAG
            "note_chunk_size": "N/A — GraphRAG uses entity extraction, not fixed chunks",
            "note_chunk_overlap": "N/A — GraphRAG uses entity extraction, not fixed chunks",
            "note_parent_chunk_size": "N/A — GraphRAG has no parent-child chunking",
            "note_parent_chunk_overlap": "N/A — GraphRAG has no parent-child chunking",
            "note_hybrid_top_k": "N/A — GraphRAG has no hybrid vector+BM25 fusion",
            "note_chunk_context_overhead": "N/A — GraphRAG formats context as community summaries",
            "note_child_context_chars": "N/A — Derived from chunk sizes, not applicable",
            "note_parent_context_chars": "N/A — Derived from chunk sizes, not applicable",
            "note_weight_context_by_score": "N/A — GraphRAG scores communities, not chunks",
            "note_use_weighted_rrf": "N/A — GraphRAG does not use RRF fusion",
        }

        return RunResult(
            level="graphrag",
            config=config,
            question=question,
            answer=answer,
            reference_answer=self.reference_answers.get(question),
            retrieved_entities=retrieved_entities,
            context_chars=context_chars,
            truncated=truncated,
            top_entity_score=top_entity_score,
            abstained=abstained,
            latency_seconds=round(latency, 3),
            timestamp=datetime.utcnow().isoformat() + "Z",
        )

    def run_all(self) -> List[RunResult]:
        return [self.run_question(q) for q in self.questions]


def save_results(results: List[RunResult]) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"graphrag_{ts}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    return path




# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """Command-line entry point for the GraphRAG experiment.

    Runs the GraphRAG pipeline on all evaluation questions and saves results
    to results/graphrag_<timestamp>.jsonl.

    CLI flags:
        --questions-file  Path to JSON file with list of question strings.
    """
    global QUESTIONS_FILE
    parser = argparse.ArgumentParser(description="GraphRAG experiment on one document")
    parser.add_argument(
        "--questions-file", type=Path, default=QUESTIONS_FILE,
        help="JSON file with list of question strings (overrides EVAL_QUESTIONS)"
    )
    args = parser.parse_args()

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
                raise ValueError(
                    f"Each item in {QUESTIONS_FILE} must be a string or an object with a 'question' key"
                )
    else:
        questions = EVAL_QUESTIONS

    print(f"=== GraphRAG Experiment ===")
    print(f"Document: {DOCUMENT_PATH}")
    print(f"LLM model: {LLM_MODEL_NAME}")
    print(f"Entity extraction model: {ENTITY_EXTRACTION_MODEL}")
    print(f"Communities detected: {len(build_graph_index.__wrapped__ if hasattr(build_graph_index, '__wrapped__') else [])} (built at init)")
    print(f"Questions: {len(questions)}")

    exp = GraphRAGExperiment(questions=questions, reference_answers=reference_answers)
    print(f"Graph entities: {len(exp.graph.entities)}")
    print(f"Graph edges: {len(exp.graph.edges)}")
    print(f"Communities: {len(exp.community_summaries)}")
    print()

    results = exp.run_all()
    path = save_results(results)
    print(f"Saved {len(results)} results to {path}")
    for r in results:
        print(f"\nQ: {r.question}")
        print(f"A: {r.answer[:300]}{'...' if len(r.answer) > 300 else ''}")
        print(f"(latency: {r.latency_seconds}s, entities: {len(r.retrieved_entities)}, abstained: {r.abstained})")


if __name__ == "__main__":
    main()
