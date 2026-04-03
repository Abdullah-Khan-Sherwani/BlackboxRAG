#!/usr/bin/env python3
"""
Query-to-report mapper: Intelligent detection of which accident report a query targets.
Uses multiple fallback strategies to handle diverse query formats:
1. Explicit NTSB number extraction (AAR-18/01, AAR-99-08, etc.)
2. Fuzzy matching against known report patterns
3. Semantic similarity to loaded report metadata
4. Falls back to no filtering for discovery/cross-report queries
"""

import os
import json
import re
from pathlib import Path
from difflib import SequenceMatcher

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load all available reports from chunks to auto-detect
def _load_available_reports():
    """Dynamically load all unique NTSB numbers from indexed chunks."""
    chunk_file = BASE_DIR / "data" / "processed" / "chunks_md_md_recursive.json"
    if not chunk_file.exists():
        chunk_file = BASE_DIR / "data" / "processed" / "chunks_md_parent_child.json"
    if not chunk_file.exists():
        return {}
    
    try:
        with open(chunk_file) as f:
            chunks = json.load(f)
        
        reports = {}
        for chunk in chunks:
            ntsb_no = chunk.get("ntsb_no", "")
            if ntsb_no:
                if ntsb_no not in reports:
                    # Extract searchable keywords from text (airline, flight#, etc.)
                    text = chunk.get("text", "").lower()
                    keywords = []
                    
                    # Extract airline name if mentioned (e.g., Asiana, United, American)
                    airlines = ["asiana", "united", "american", "southwest", "delta", "alaska"]
                    for airline in airlines:
                        if airline in text:
                            keywords.append(airline)
                    
                    # Extract flight number pattern (Flight NNN)
                    import re
                    flight_match = re.search(r"flight\s+(\d+)", text)
                    if flight_match:
                        keywords.append(f"flight {flight_match.group(1)}")
                    
                    reports[ntsb_no] = {
                        "aircraft": chunk.get("make", "") + " " + chunk.get("model", ""),
                        "date": chunk.get("event_date", ""),
                        "location": chunk.get("state", ""),
                        "keywords": " ".join(keywords),  # Searchable text from chunks
                    }
        return reports
    except:
        return {}

AVAILABLE_REPORTS = _load_available_reports()


def detect_report_from_query(query: str) -> str:
    """
    Detect which NTSB report a query targets using intelligent fallback strategies.
    
    Returns NTSB number (e.g., "NTSB/AAR-99-08") or empty string if ambiguous.
    
    Strategy priority:
    1. Explicit NTSB number extraction
    2. Combined keyword matching (airline + flight number)
    3. Aircraft + date + location fuzzy matching
    4. Empty (allows cross-report discovery)
    """
    if not query:
        return ""
    
    q = query.lower().strip()
    
    # ============================================================================
    # STRATEGY 1: EXPLICIT NTSB NUMBER EXTRACTION
    # ============================================================================
    # Try to find NTSB format directly: AAR-18/01, AAR-99-08, AAR_18_01, etc.
    ntsb_match = re.search(r"aar[_\-/\s]*(\d{2,4})[_\-/\s]*(\d{2})", q)
    if ntsb_match:
        year, num = ntsb_match.groups()
        # Search available reports for match
        for report_id in AVAILABLE_REPORTS.keys():
            if year in report_id and num in report_id:
                return report_id
    
    # Also try DCA format (e.g., DCA16MA261)
    dca_match = re.search(r"dca\d+[a-z]{2}\d+", q)
    if dca_match:
        dca_str = dca_match.group(0)
        for report_id in AVAILABLE_REPORTS.keys():
            if dca_str.upper() in report_id or "dca" in report_id.lower():
                return report_id
    
    # ============================================================================
    # STRATEGY 2: COMBINED KEYWORD MATCHING (HIGHEST SPECIFICITY)
    # ============================================================================
    # Extract airline and flight number from query
    airlines = ["asiana", "united", "american", "southwest", "delta", "alaska", "korean"]
    airline_in_query = None
    for airline in airlines:
        if airline in q:
            airline_in_query = airline
            break
    
    flight_match = re.search(r"(?:flight\s+)?(\d{2,4})", q)
    flight_in_query = None
    if flight_match:
        flight_in_query = f"flight {flight_match.group(1)}"
    
    # If we have both airline and flight number from query, find exact match
    if airline_in_query and flight_in_query:
        for report_id, metadata in AVAILABLE_REPORTS.items():
            keywords = metadata.get("keywords", "").lower()
            if airline_in_query in keywords and flight_in_query in keywords:
                return report_id  # Exact match!
    
    # ============================================================================
    # STRATEGY 3: FUZZY MATCHING AGAINST LOADED METADATA & KEYWORDS
    # ============================================================================
    best_match = None
    best_score = 0.60
    
    for report_id, metadata in AVAILABLE_REPORTS.items():
        # Build a searchable profile from metadata
        profile = " ".join([
            metadata.get("aircraft", ""),
            metadata.get("date", ""),
            metadata.get("location", ""),
            metadata.get("keywords", ""),
        ]).lower()
        
        score = SequenceMatcher(None, q, profile).ratio()
        if score > best_score:
            best_score = score
            best_match = report_id
    
    if best_match:
        return best_match
    
    # ============================================================================
    # STRATEGY 4: FALLBACK - NO SPECIFIC REPORT DETECTED (DISCOVERY MODE)
    # ============================================================================
    return ""


def get_pinecone_filter(query: str, strategy: str = "md_recursive") -> dict:
    """
    Build a Pinecone metadata filter for a query.

    Always filters by strategy so the selected chunking approach is used
    consistently for both semantic and BM25 retrieval paths.
    Additionally filters by NTSB report number when the query unambiguously
    targets a specific report.

    Returns: {"strategy": {"$eq": strategy}} or
             {"$and": [{"strategy": ...}, {"ntsb_no": ...}]}
    """
    ntsb_no = detect_report_from_query(query)

    if ntsb_no:
        return {"$and": [{"strategy": {"$eq": strategy}}, {"ntsb_no": {"$eq": ntsb_no}}]}

    return {"strategy": {"$eq": strategy}}


def resolve_report_number(regex_result: str, llm_result: str, llm_confidence: str) -> str:
    """Reconcile regex-detected and LLM-detected NTSB report numbers.

    Priority:
    1. Both agree → use it
    2. Regex found one, LLM didn't → use regex (user explicitly mentioned it)
    3. LLM found one (high/medium confidence), regex didn't → use LLM
    4. Both disagree → prefer regex (explicit mention trumps LLM guess)
    5. Neither found one → empty string (discovery mode)
    """
    if regex_result and llm_result:
        if regex_result == llm_result:
            return regex_result
        # Conflict: prefer regex (explicit user mention)
        print(f"[ReportMapper] Conflict: regex='{regex_result}', LLM='{llm_result}' — using regex")
        return regex_result

    if regex_result:
        return regex_result

    if llm_result and llm_confidence in ("high", "medium"):
        return llm_result

    return ""


if __name__ == "__main__":
    # Test the detection with diverse query formats
    test_queries = [
        # Explicit NTSB numbers
        "What is in NTSB/AAR-18/01?",
        "Tell me about AAR-99-08",
        
        # Flight numbers + location
        "What were the pilot hours on the Little Rock MD-82 crash?",
        "Flight 383 engine failure chronology",
        
        # Aircraft + date patterns
        "Boeing 767 Chicago October 2016",
        "Q400 Buffalo 2009 crash",
        
        # Generic discovery queries
        "Compare different accidents",
        "What are common causes of engine failure?",
    ]
    
    print(f"Loaded {len(AVAILABLE_REPORTS)} reports from chunks")
    for ntsb_no, metadata in list(AVAILABLE_REPORTS.items())[:3]:
        print(f"  {ntsb_no}: {metadata}")
    print()
    
    for q in test_queries:
        detected = detect_report_from_query(q)
        filter_obj = get_pinecone_filter(q)
        print(f"Query: {q[:60]}...")
        print(f"  → Detected: {detected if detected else '(discovery mode)'}")
        print(f"  → Filter: {filter_obj}")
        print()

