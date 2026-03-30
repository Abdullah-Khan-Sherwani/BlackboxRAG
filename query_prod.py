#!/usr/bin/env python3
"""
Production query tool v2: PRIORITY-FIRST retrieval
- No reliance on cross-encoder reranking (which is domain-unaware)
- Keyword matching in text
- Section-aware filtering
- Simple ranking by relevance

Usage: python query_prod.py "Your question here"
"""

import sys
import os
import json
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval.query import load_model, init_pinecone, retrieve
from src.retrieval.report_mapper import detect_report_from_query
from src.retrieval.hybrid import build_bm25_index, enrich_with_neighbors
from src.generation.generate import generate_answer

# Section priorities for different question types
SECTION_PRIORITIES = {
    "arff|rescue|firefight|foam|obscur|smoke|emergency": [
        "1.15", "1.15.4", "1.15.3", "1.15.2", "1.10", "1.1"
    ],
    "crash|impact|accident|collision|sequence": [
        "1.1", "1.12", "1.11", "Executive Summary"
    ],
    "mechanical|engine|hydraulic|electrical|fuel": [
        "1.8", "2.3", "2.4", "1.2", "History"
    ],
    "crew|pilot|human error|training|flight hour|instructor|experience|captain|first officer": [
        "1.5.2", "1.5.1", "1.17.2.5", "1.17.2", "2.1", "1.8", "Executive"
    ],
    "weather|icing|wind|visibility|thunderstorm": [
        "1.3", "1.6", "1.9", "Safety"
    ]
}

TOP_K_FINAL = 15  # Increased from 10 to support enhanced retrieval with neighbor enrichment


def detect_question_type(query):
    """Determine question category to prioritize relevant sections.
    
    Check crew-related keywords FIRST before general ones to avoid misclassification.
    """
    query_lower = query.lower()
    
    # ============================================================================
    # CHECK SPECIFIC KEYWORDS FIRST (highest specificity)
    # ============================================================================
    
    # Crew/Pilot FIRST (to block from being caught by 'accident')
    if any(kw in query_lower for kw in ["crew", "pilot", "training", "flight hour", "instructor", "experience", "captain", "first officer", "pm"]):
        return ["1.5.2", "1.5.1", "1.17.2.5", "1.17.2", "2.1", "1.8", "Executive"]
    
    # ARFF/Emergency
    if any(kw in query_lower for kw in ["arff", "rescue", "firefight", "foam", "obscur", "smoke", "emergency"]):
        return ["1.15", "1.15.4", "1.15.3", "1.15.2", "1.10", "1.1"]
    
    # Weather
    if any(kw in query_lower for kw in ["weather", "icing", "wind", "visibility", "thunderstorm"]):
        return ["1.3", "1.6", "1.9", "Safety"]
    
    # Mechanical
    if any(kw in query_lower for kw in ["mechanical", "engine", "hydraulic", "electrical", "fuel"]):
        return ["1.8", "2.3", "2.4", "1.2", "History"]
    
    # Crash/Impact (check LAST)
    if any(kw in query_lower for kw in ["crash", "impact", "accident", "collision", "sequence"]):
        return ["1.1", "1.12", "1.11", "Executive Summary"]
    
    # Default: general sections
    return ["1.15", "Executive Summary", "1.1", "Safety", "1.13"]


def keyword_score(text, query):
    """Score text by keyword density and query term coverage."""
    if not text:
        return 0
    
    text_lower = text.lower()
    query_terms = query.lower().split()
    
    # Count exact phrase matches
    phrase_matches = 0
    if query.lower() in text_lower:
        phrase_matches += 10
    
    # Count individual term matches
    term_matches = sum(1 for term in query_terms if len(term) > 2 and term in text_lower)
    
    # Keyword density (avoid false positives from stopwords)
    important_terms = [t for t in query_terms if len(t) > 3]
    matched_terms = sum(1 for t in important_terms if t in text_lower)
    keyword_density = matched_terms / len(important_terms) if important_terms else 0
    
    return phrase_matches + (term_matches * 2) + (keyword_density * 5)


def section_priority_score(chunk, priority_sections):
    """Score chunk based on section priority (high = most relevant)."""
    section_title = chunk.get("section_title", "").upper()
    
    # HIGHEST PRIORITY: Sections specifically listed for this question type
    for i, priority_section in enumerate(priority_sections):
        if priority_section.upper() in section_title:
            return 100 - (i * 5)  # Top priority sections get 100, then 95, 90, etc.
    
    # Anything else gets penalized
    return 10


def smart_retrieve(query, top_k=TOP_K_FINAL):
    """Smart retrieval: text matching + section priority."""
    print(f"[RETRIEVAL] Query: {query[:70]}...")
    
    # 1. DETECT REPORT
    detected_report = detect_report_from_query(query)
    print(f"[RETRIEVAL] Detected report: {detected_report}")
    
    # 2. DETECT QUESTION TYPE → Section priorities
    priority_sections = detect_question_type(query)
    print(f"[RETRIEVAL] Priority sections: {priority_sections[:4]}")
    
    # 3. LOAD CHUNKS - Use chunks_md_section which has full text + metadata
    print(f"[RETRIEVAL] Loading chunks...")
    bm25, all_chunks = build_bm25_index("section")
    print(f"[RETRIEVAL]   - Loaded {len(all_chunks)} total chunks")
    
    # 4. FILTER TO DETECTED REPORT (if available)
    if detected_report:
        chunks = [c for c in all_chunks if detected_report in c.get("ntsb_no", "")]
        print(f"[RETRIEVAL]   - Filtered to {detected_report}: {len(chunks)} chunks")
    else:
        chunks = all_chunks
    
    # 5. SCORE ALL CHUNKS
    print(f"[RETRIEVAL] Scoring chunks...")
    for chunk in chunks:
        # Component 1: Section priority (0-100)
        section_score = section_priority_score(chunk, priority_sections)
        
        # Component 2: Keyword match (0-20)
        text_score = keyword_score(chunk.get("text", ""), query)
        text_score = min(20, text_score)  # Cap at 20
        
        # Component 3: BM25 score (0-10, normalized)
        bm25_score = chunk.get("score", 0)
        bm25_normalized = min(10, bm25_score / 2) if bm25_score > 0 else 0
        
        # COMPOSITE SCORE: Section priority (60%) + Text keywords (30%) + BM25 (10%)
        composite = (section_score * 0.6) + (text_score * 0.3) + (bm25_normalized * 0.1)
        chunk["composite_score"] = composite
    
    # 6. SORT AND SELECT TOP-K
    ranked = sorted(chunks, key=lambda x: x.get("composite_score", 0), reverse=True)[:top_k]
    
    # 7. ENRICH WITH NEIGHBORS (add surrounding chunks for context)
    ranked = enrich_with_neighbors(ranked, all_chunks, window=2)
    
    print(f"[RETRIEVAL] Final: {len(ranked)} chunks selected (with neighbor enrichment)\n")
    
    # Debug output
    for i, chunk in enumerate(ranked[:top_k], 1):
        section = chunk.get("section_title", "Unknown")[:55]
        score = chunk.get("composite_score", 0)
        keywords_found = sum(1 for term in query.lower().split() if len(term) > 3 and term in chunk.get("text", "").lower())
        print(f"  [{i:2d}] {section:<55} score={score:6.2f} kw={keywords_found}")
    
    return ranked


def main():
    if len(sys.argv) < 2:
        print("Usage: python query_prod.py \"Your question here\"")
        sys.exit(1)
    
    query = " ".join(sys.argv[1:])
    
    print("\n" + "=" * 80)
    print("PRODUCTION RAG QUERY TOOL (v2 - Priority-First)")
    print("=" * 80 + "\n")
    
    # Retrieve
    matches = smart_retrieve(query, top_k=TOP_K_FINAL)
    
    if not matches:
        print("\n[ERROR] No matches found")
        return
    
    # Generate answer
    print("[GENERATION] Generating answer...\n")
    answer = generate_answer(query, matches)
    
    print("\n" + "=" * 80)
    print("ANSWER:")
    print("=" * 80)
    print(answer)
    print("=" * 80)


if __name__ == "__main__":
    main()
