#!/usr/bin/env python3
"""
Analyze RAG experiment results across all levels.

Reads results/level_<N>_<timestamp>.jsonl files, classifies questions by type,
groups results by question_type + question, and determines which RAG level
produced the best answer for each question type.

Also evaluates confidence-based thresholds (CONFIDENCE_THRESHOLD, EXPANSION_THRESHOLD)
to recommend fixed values for future runs.

Usage:
    python analyze_results.py [results_dir]
    # Defaults to ./results/
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Question type classification (heuristic, no LLM required)
QUESTION_TYPE_KEYWORDS = {
    "fact_lookup": [
        "what is", "what are", "who is", "who are", "when", "where",
        "how many", "how much", "which", "error code", "error code",
    ],
    "entity_extraction": [
        "name", "code", "number", "date", "time", "amount", "percentage",
        "percent", "error", "warning", "alert",
    ],
    "synthesis": [
        "why", "how does", "how can", "what caused", "what led to",
        "relationship between", "difference between", "compare",
    ],
    "conceptual": [
        "what is the", "what are the", "explain", "describe",
        "overview", "summary", "key points", "main idea",
    ],
}

# Classification-to-level mapping: our hypothesis about which RAG level
# works best for each question type. This is compared against actual
# best levels from the experiment results.
QUESTION_TYPE_TO_LEVEL = {
    "fact_lookup": 1,           # hybrid: keyword + vector for exact entities
    "entity_extraction": 2,     # parent-child: parent context around entities
    "synthesis": 3,             # cross-encoder reranker for multi-chunk reasoning
    "conceptual": 4,            # query expansion for broad conceptual queries
    "other": 0,                 # naive RAG fallback
}


def classify_question(question: str) -> str:
    """Classify a question into a type using heuristic keyword matching.

    Returns one of: fact_lookup, entity_extraction, synthesis, conceptual, other.
    """
    q = question.lower().strip()

    # Synthesis: why/how questions asking for causality or mechanism
    if any(k in q for k in ["why ", "how does ", "how can ", "what caused",
                            "what led to", "relationship between",
                            "difference between", "compare"]):
        return "synthesis"

    # Entity extraction: specific named entities, codes, numbers
    if any(k in q for k in ["error code", "which ", "who ", "when ",
                            "how many", "how much", "name", "code",
                            "number", "date", "percentage", "percent"]):
        return "entity_extraction"

    # Conceptual: broad "what is" questions about themes/ideas
    if any(k in q for k in ["what is the ", "what are the ", "explain ",
                            "describe ", "overview", "summary",
                            "key points", "main idea"]):
        return "conceptual"

    # Fact lookup: short "what is" questions
    if q.startswith("what is ") or q.startswith("what are "):
        return "fact_lookup"

    return "other"


def load_results(results_dir: Path) -> List[Dict[str, Any]]:
    """Load all JSONL result files from results_dir.

    Returns a list of RunResult dicts merged from all files.
    """
    results: List[Dict[str, Any]] = []
    jsonl_files = sorted(results_dir.glob("level_*.jsonl"))

    if not jsonl_files:
        print(f"Warning: No JSONL files found in {results_dir}")
        return results

    for filepath in jsonl_files:
        with filepath.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    record["_source_file"] = filepath.name
                    results.append(record)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping malformed line in {filepath}: {e}")

    print(f"Loaded {len(results)} results from {len(jsonl_files)} files")
    return results


def score_answer(answer: str, reference_answer: Optional[str]) -> float:
    """Score an answer against a reference answer.

    Returns a score from 0.0 to 1.0:
    - 1.0: exact match (case-insensitive)
    - 0.5-0.9: partial match based on token overlap
    - 0.0: no match or no reference answer
    """
    if not reference_answer or not answer:
        return 0.0

    answer_clean = answer.strip().lower()
    ref_clean = reference_answer.strip().lower()

    # Exact match
    if answer_clean == ref_clean:
        return 1.0

    # Check if reference answer is contained in answer or vice versa
    if ref_clean in answer_clean or answer_clean in ref_clean:
        return 0.9

    # Token overlap scoring
    ref_tokens = set(re.findall(r"\w+", ref_clean))
    answer_tokens = set(re.findall(r"\w+", answer_clean))

    if not ref_tokens:
        return 0.0

    overlap = len(ref_tokens & answer_tokens)
    overlap_ratio = overlap / len(ref_tokens)

    # Also check for key named entities (capitalized words in reference)
    ref_entities = set(w for w in re.findall(r"\b[A-Z][a-z]+\b", reference_answer))
    answer_entities = set(w for w in re.findall(r"\b[A-Z][a-z]+\b", answer))
    entity_overlap = len(ref_entities & answer_entities)
    entity_ratio = (entity_overlap / len(ref_entities)) if ref_entities else 0.0

    # Combined score: token overlap weighted more, entities as bonus
    score = 0.7 * overlap_ratio + 0.3 * entity_ratio
    return min(score, 0.85)  # Cap below 1.0 for non-exact matches


def group_results(results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group results by question text.

    Returns dict mapping question string -> list of RunResult dicts (one per level).
    """
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        q = r.get("question", "")
        grouped[q].append(r)
    return dict(grouped)


def analyze_question_group(
    question: str,
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Analyze results for a single question across all levels.

    Returns a dict with per-level scores, best level, and confidence analysis.
    """
    question_type = classify_question(question)
    reference_answer = results[0].get("reference_answer") if results else None

    # Score each level's answer
    level_scores: Dict[int, float] = {}
    level_answers: Dict[int, str] = {}
    level_abstained: Dict[int, bool] = {}
    level_scores_raw: Dict[int, Optional[float]] = {}

    for r in results:
        level = r.get("level", -1)
        answer = r.get("answer", "")
        score = score_answer(answer, reference_answer)
        level_scores[level] = score
        level_answers[level] = answer
        level_abstained[level] = r.get("abstained", False)
        level_scores_raw[level] = r.get("top_chunk_score")

    # Find best level
    if level_scores:
        best_level = max(level_scores, key=level_scores.get)
        best_score = level_scores[best_level]
    else:
        best_level = -1
        best_score = 0.0

    # Confidence analysis: at what threshold would abstention have helped?
    # If a level abstained but reference_answer exists, a lower threshold would
    # have allowed an answer (possibly correct). If a level didn't abstain but
    # answer is wrong (score=0), a higher threshold would have prevented hallucination.
    abstention_analysis: Dict[str, Any] = {
        "abstained_levels": [l for l, a in level_abstained.items() if a],
        "non_abstained_wrong_levels": [
            l for l, s in level_scores.items()
            if not level_abstained[l] and s < 0.1
        ],
        "recommended_threshold": None,
    }

    # If any non-abstained level got score 0, recommend threshold above
    # the max score among non-abstained wrong levels
    wrong_scores = [
        level_scores_raw[l]
        for l in level_abstained
        if not level_abstained[l]
        and level_scores[l] < 0.1
        and level_scores_raw[l] is not None
    ]
    if wrong_scores:
        abstention_analysis["recommended_threshold"] = round(max(wrong_scores) + 0.05, 2)

    # Expansion analysis: level 4 vs level 3
    expansion_analysis: Dict[str, Any] = {
        "level_3_score": level_scores.get(3, 0.0),
        "level_4_score": level_scores.get(4, 0.0),
        "expansion_helped": False,
        "expansion_hurt": False,
        "expansion_neutral": False,
    }
    if 3 in level_scores and 4 in level_scores:
        if level_scores[4] > level_scores[3] + 0.05:
            expansion_analysis["expansion_helped"] = True
        elif level_scores[4] < level_scores[3] - 0.05:
            expansion_analysis["expansion_hurt"] = True
        else:
            expansion_analysis["expansion_neutral"] = True

    return {
        "question": question,
        "question_type": question_type,
        "predicted_level": QUESTION_TYPE_TO_LEVEL.get(question_type, 0),
        "reference_answer": reference_answer,
        "level_scores": level_scores,
        "best_level": best_level,
        "best_score": best_score,
        "abstention_analysis": abstention_analysis,
        "expansion_analysis": expansion_analysis,
    }


def analyze_by_question_type(
    question_analyses: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate question-level analyses by question_type.

    Returns stats per question_type including best level (majority vote).
    """
    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for qa in question_analyses:
        by_type[qa["question_type"]].append(qa)

    type_summary: Dict[str, Any] = {}
    for qtype, analyses in by_type.items():
        # Count best levels
        level_votes: Dict[int, int] = defaultdict(int)
        total_scores: Dict[int, List[float]] = defaultdict(list)
        expansion_helped_count = 0
        expansion_hurt_count = 0
        total_with_expansion = 0

        match_count = 0
        mismatch_count = 0

        for a in analyses:
            level_votes[a["best_level"]] += 1
            for lvl, score in a["level_scores"].items():
                total_scores[lvl].append(score)
            ea = a["expansion_analysis"]
            if ea["expansion_helped"]:
                expansion_helped_count += 1
            if ea["expansion_hurt"]:
                expansion_hurt_count += 1
            total_with_expansion += 1

            # Check if predicted level matches actual best level
            if a["predicted_level"] == a["best_level"]:
                match_count += 1
            else:
                mismatch_count += 1

        best_level_by_vote = max(level_votes, key=level_votes.get) if level_votes else -1

        # Average score per level across questions of this type
        avg_scores: Dict[int, float] = {}
        for lvl, scores in total_scores.items():
            avg_scores[lvl] = round(sum(scores) / len(scores), 3) if scores else 0.0

        type_summary[qtype] = {
            "question_count": len(analyses),
            "predicted_level": QUESTION_TYPE_TO_LEVEL.get(qtype, 0),
            "best_level_by_vote": best_level_by_vote,
            "match_count": match_count,
            "mismatch_count": mismatch_count,
            "match_rate": round(match_count / len(analyses), 2) if analyses else 0.0,
            "level_votes": dict(level_votes),
            "avg_score_per_level": avg_scores,
            "expansion_helped_count": expansion_helped_count,
            "expansion_hurt_count": expansion_hurt_count,
            "total_with_expansion": total_with_expansion,
        }

    return type_summary


def recommend_confidence_threshold(
    question_analyses: List[Dict[str, Any]],
) -> Optional[float]:
    """Recommend a CONFIDENCE_THRESHOLD value based on analysis.

    Looks at non-abstained wrong answers (score < 0.1) and their top_chunk_score.
    Recommends threshold = max(top_chunk_score of wrong answers) + 0.05.
    """
    wrong_scores: List[float] = []
    for qa in question_analyses:
        for r in qa.get("_raw_results", []):
            if not r.get("abstained", False) and qa["best_score"] < 0.1:
                score = r.get("top_chunk_score")
                if score is not None:
                    wrong_scores.append(score)

    if not wrong_scores:
        return None
    return round(max(wrong_scores) + 0.05, 2)


def recommend_expansion_threshold(
    question_analyses: List[Dict[str, Any]],
) -> Optional[float]:
    """Recommend an EXPANSION_THRESHOLD value.

    Looks at questions where expansion hurt (level 4 score < level 3 score)
    and their top_chunk_score from the base hybrid search.
    Recommends threshold = min(top_chunk_score where expansion hurt) - 0.05,
    so expansion skips when retrieval is decent.
    """
    hurt_scores: List[float] = []
    for qa in question_analyses:
        ea = qa["expansion_analysis"]
        if ea["expansion_hurt"]:
            # Find the base hybrid score (level 3 or level 1 without expansion)
            # We look at the question's level 3 score context
            # Use the question's own top_chunk_score from non-expansion level
            for r in qa.get("_raw_results", []):
                if r.get("level") == 1 and r.get("top_chunk_score") is not None:
                    hurt_scores.append(r["top_chunk_score"])

    if not hurt_scores:
        return None
    return round(min(hurt_scores) - 0.05, 2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze RAG experiment results across all levels"
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        default="results",
        help="Directory containing level_*.jsonl files (default: results/)",
    )
    parser.add_argument(
        "--output",
        nargs="?",
        default=None,
        help="Output file for analysis report (default: print to stdout)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory '{results_dir}' does not exist")
        sys.exit(1)

    results = load_results(results_dir)
    if not results:
        print("No results to analyze")
        sys.exit(1)

    # Group by question
    grouped = group_results(results)

    # Attach raw results to question analyses for threshold recommendations
    question_analyses: List[Dict[str, Any]] = []
    for question, q_results in grouped.items():
        qa = analyze_question_group(question, q_results)
        # Attach raw results for threshold recommendation
        qa["_raw_results"] = q_results
        question_analyses.append(qa)

    # Aggregate by question type
    type_summary = analyze_by_question_type(question_analyses)

    # Recommend thresholds
    recommended_confidence = recommend_confidence_threshold(question_analyses)
    recommended_expansion = recommend_expansion_threshold(question_analyses)

    # Build report
    report: Dict[str, Any] = {
        "summary": {
            "total_questions": len(question_analyses),
            "total_levels_run": len(set(r.get("level") for r in results)),
            "results_files": len(list(results_dir.glob("level_*.jsonl"))),
        },
        "question_type_summary": type_summary,
        "recommended_confidence_threshold": recommended_confidence,
        "recommended_expansion_threshold": recommended_expansion,
        "question_details": [
            {
                "question": qa["question"],
                "question_type": qa["question_type"],
                "predicted_level": qa["predicted_level"],
                "best_level": qa["best_level"],
                "best_score": qa["best_score"],
                "level_match": qa["predicted_level"] == qa["best_level"],
                "level_scores": qa["level_scores"],
                "abstention_analysis": qa["abstention_analysis"],
                "expansion_analysis": qa["expansion_analysis"],
            }
            for qa in question_analyses
        ],
    }

    # Print summary
    print("\n" + "=" * 70)
    print("RAG EXPERIMENT ANALYSIS REPORT")
    print("=" * 70)
    print(f"\nTotal questions: {report['summary']['total_questions']}")
    print(f"Total levels run: {report['summary']['total_levels_run']}")
    print(f"Results files: {report['summary']['results_files']}")

    print("\n--- Question Type Summary ---")
    for qtype, stats in type_summary.items():
        match_str = (
            f"{stats['match_count']}/{stats['question_count']} "
            f"({stats['match_rate']*100:.0f}%)"
        )
        print(f"\n  {qtype} ({stats['question_count']} questions)")
        print(f"    Predicted level: {stats['predicted_level']}")
        print(f"    Best level by vote: {stats['best_level_by_vote']}")
        print(f"    Predicted vs actual match: {match_str}")
        print(f"    Level votes: {stats['level_votes']}")
        print(f"    Avg score per level: {stats['avg_score_per_level']}")
        print(f"    Expansion helped: {stats['expansion_helped_count']}, "
              f"hurt: {stats['expansion_hurt_count']}, "
              f"total tested: {stats['total_with_expansion']}")

    print("\n--- Recommended Thresholds ---")
    print(f"  CONFIDENCE_THRESHOLD: {recommended_confidence}")
    print(f"  EXPANSION_THRESHOLD:  {recommended_expansion}")

    print("\n--- Question Details ---")
    for qd in question_analyses:
        match_mark = "✓" if qd["predicted_level"] == qd["best_level"] else "✗"
        print(f"\n  Q: {qd['question'][:80]}...")
        print(f"    Type: {qd['question_type']}")
        print(f"    Predicted level: {qd['predicted_level']} | Actual best: {qd['best_level']} "
              f"(score: {qd['best_score']}) [{match_mark}]")
        print(f"    Level scores: {qd['level_scores']}")
        ea = qd["expansion_analysis"]
        if ea["expansion_helped"]:
            print(f"    Expansion: HELPED (L3={ea['level_3_score']}, L4={ea['level_4_score']})")
        elif ea["expansion_hurt"]:
            print(f"    Expansion: HURT (L3={ea['level_3_score']}, L4={ea['level_4_score']})")

    # Write to file if requested
    if args.output:
        output_path = Path(args.output)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nReport saved to {output_path}")


if __name__ == "__main__":
    main()