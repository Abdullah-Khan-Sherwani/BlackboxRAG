#!/usr/bin/env python3
"""
Accuracy Validation Tool for NTSB RAG System

Workflow:
1. User provides question
2. System retrieves golden chunks from RAG
3. User provides their LLM response
4. Tool validates accuracy by checking:
   - Numerical claims (are numbers in source chunks?)
   - NTSB citations (are they correct and present?)
   - Missing/incorrect sections
   - 2-3 key points extraction
"""

import sys
import os
import json
import re
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.retrieval.query import load_model, init_pinecone
from src.retrieval.hybrid import build_bm25_index, expand_query_variants, bm25_retrieve, rrf_fuse_lists, enrich_with_neighbors
from src.retrieval.query import retrieve


# ============================================================================
# NUMBER EXTRACTION & VALIDATION
# ============================================================================

def extract_numbers_with_context(text: str) -> List[Tuple[str, str]]:
    """Extract numbers with surrounding context (e.g., '12,307 flight hours')"""
    
    # Find all numbers (including formatted ones like 12,307)
    pattern = r'\d+(?:,\d{3})*\s+[a-zA-Z\s]*?(?:hour|flight|total|experience|time|day|week|year)'
    
    matches = re.finditer(pattern, text, re.IGNORECASE)
    result = []
    
    for match in matches:
        phrase = match.group(0).strip()
        # Extract the number part
        num_match = re.search(r'\d+(?:,\d{3})*', phrase)
        if num_match:
            num = num_match.group(0)
            result.append((num, phrase))
    
    # Also find standalone numbers
    standalone = re.findall(r'\b(\d+(?:,\d{3})*)\b', text)
    for num in standalone:
        if (num, None) not in result:  # Avoid duplicates
            result.append((num, f"{num} (standalone)"))
    
    return result


def find_number_in_chunks(number: str, chunks_text: str, context_window: int = 50) -> List[str]:
    """Find where a number appears in chunks with surrounding context"""
    
    # Normalize number for searching
    normalized = number.replace(",", "")
    
    # Look for exact number
    pattern_exact = re.escape(number)
    # Also look for number without formatting
    pattern_normalized = re.escape(normalized)
    
    contexts = []
    for pattern in [pattern_exact, pattern_normalized]:
        for match in re.finditer(pattern, chunks_text):
            start = max(0, match.start() - context_window)
            end = min(len(chunks_text), match.end() + context_window)
            context = chunks_text[start:end].strip()
            if context and context not in contexts:
                contexts.append(context)
    
    return contexts


def extract_ntsb_citations(text: str) -> List[str]:
    """Extract NTSB accident report numbers (AAR-XX/XX, DCA-XX-XX, etc.)"""
    
    patterns = [
        r'(?:NTSB[:\s/]*)?AAR[:\s/\-]*(\d{2})[:\s/\-]*(\d{2})',  # AAR-14/01
        r'(?:NTSB[:\s/]*)?DCA[:\s/\-]*([0-9]{2}[A-Z]*[0-9]+)',   # DCA16MA261
        r'\[NTSB[:\s/]*([A-Z]{1,3}[:\s/\-]*\d+[A-Z]?[:\s/\-]*\d+)\]',  # [NTSB: AAR-14/01]
    ]
    
    citations = []
    
    # AAR format (AAR-14/01 or AAR 14/01 or NTSB/AAR-14/01)
    aar_matches = re.finditer(r'(?:NTSB[:\s/]*)?AAR[:\s/\-]*(\d{2})[:\s/\-]*(\d{2})', text, re.IGNORECASE)
    for match in aar_matches:
        year = match.group(1)
        num = match.group(2)
        citation = f"AAR-{year}/{num}"
        if citation not in citations:
            citations.append(citation)
    
    # DCA format (DCA-16-MA-261 or similar)
    dca_matches = re.finditer(r'(?:NTSB[:\s/]*)?DCA[:\s/\-]*([0-9]{2}[A-Z]*[0-9]+)', text, re.IGNORECASE)
    for match in dca_matches:
        citation = f"DCA-{match.group(1)}"
        if citation not in citations:
            citations.append(citation)
    
    # Bracketed format [NTSB: AAR-14/01]
    bracket_matches = re.finditer(r'\[NTSB[:\s]*([A-Z]{1,3}[:\s/\-]*[0-9\-/]+)\]', text, re.IGNORECASE)
    for match in bracket_matches:
        citation = match.group(1).strip()
        # Normalize
        citation = re.sub(r'[:\s]+', '-', citation)
        if citation not in citations:
            citations.append(citation)
    
    # Also search for just numbers after AAR keywords
    if 'AAR' in text or 'NTSB' in text or 'DCA' in text:
        # Find patterns like "AAR-14-01" or standalone "14/01"
        loose = re.findall(r'(?:AAR\D)?(\d{2})[/\-](\d{2})', text)
        for year, num in loose:
            citation = f"AAR-{year}/{num}"
            if citation not in citations:
                citations.append(citation)
    
    return citations


# ============================================================================
# RETRIEVAL & VALIDATION
# ============================================================================

def retrieve_golden_chunks(query: str, top_k: int = 10) -> List[Dict]:
    """Retrieve golden chunks from NTSB reports for given query"""
    
    print("\n" + "="*80)
    print("RETRIEVING GOLDEN CHUNKS FROM NTSB REPORTS")
    print("="*80)
    
    # Load resources
    print("[1] Loading models and resources...")
    try:
        jina_model = load_model()
        pinecone_index = init_pinecone()
        bm25, chunks = build_bm25_index("section")
        print(f"✓ Loaded all resources ({len(chunks)} total chunks)")
    except Exception as e:
        print(f"✗ Error loading resources: {e}")
        return []
    
    # Expand queries
    print("\n[2] Expanding queries...")
    queries = expand_query_variants(query)
    print(f"✓ Generated {len(queries)} query variants")
    
    # Retrieve
    print("\n[3] Performing hybrid retrieval...")
    ranked_lists = []
    
    for q in queries:
        # Semantic
        semantic_results = retrieve(q, "section", top_k=60, model=jina_model, index=pinecone_index)
        ranked_lists.append(semantic_results)
        
        # BM25
        bm25_results = bm25_retrieve(q, bm25, chunks, top_k=60)
        ranked_lists.append(bm25_results)
    
    print(f"✓ Got {len(ranked_lists)} result lists")
    
    # Fuse
    print("\n[4] RRF Fusion...")
    fused = rrf_fuse_lists(ranked_lists)
    print(f"✓ Fused to {len(fused)} candidates")
    
    # Select top-k
    matches = sorted(fused, key=lambda x: x.get("score", 0), reverse=True)[:top_k]
    
    # Enrich with neighbors
    print("\n[5] Enriching with neighbors...")
    matches = enrich_with_neighbors(matches, chunks, window=2)
    print(f"✓ Enriched {len(matches)} results")
    
    print("\n" + "="*80)
    print(f"RETRIEVED {len(matches)} GOLDEN CHUNKS:")
    print("="*80)
    
    for i, m in enumerate(matches, 1):
        ntsb_no = m.get("ntsb_no", "N/A")
        section = m.get("section_title", "N/A")[:50]
        score = m.get("score", 0)
        print(f"[{i}] {ntsb_no} | {section:<50} | Score: {score:.4f}")
    
    return matches


def validate_llm_response(
    query: str,
    llm_response: str,
    golden_chunks: List[Dict],
) -> Dict:
    """
    Validate LLM response against golden chunks.
    
    Checks:
    1. Numerical accuracy (numbers appear in source)
    2. NTSB citations (correct format and present)
    3. Section coverage (are right sections mentioned?)
    4. Information completeness (2-3 key points present?)
    """
    
    print("\n" + "="*80)
    print("VALIDATING LLM RESPONSE")
    print("="*80)
    
    report = {
        "query": query,
        "llm_response": llm_response,
        "validation_sections": {},
        "numerical_accuracy": {},
        "ntsb_citations": {},
        "overall_score": 0.0,
        "issues": [],
        "recommendations": []
    }
    
    # ===== SECTION 1: NUMERICAL ACCURACY =====
    print("\n[1] NUMERICAL ACCURACY CHECK")
    print("-" * 40)
    
    response_numbers = extract_numbers_with_context(llm_response)
    golden_text = "\n".join(m.get("text", "") for m in golden_chunks)
    
    numerical_report = {
        "found_in_response": [],
        "verified_in_source": [],
        "NOT_VERIFIED": [],
        "accuracy_rate": 0.0
    }
    
    for number, phrase in response_numbers:
        numerical_report["found_in_response"].append({
            "number": number,
            "phrase": phrase
        })
        
        # Search in golden chunks
        contexts = find_number_in_chunks(number, golden_text)
        
        if contexts:
            numerical_report["verified_in_source"].append({
                "number": number,
                "phrase": phrase,
                "contexts": contexts[:2]  # Limit to first 2 contexts
            })
            print(f"✓ VERIFIED: {phrase}")
        else:
            numerical_report["NOT_VERIFIED"].append({
                "number": number,
                "phrase": phrase
            })
            print(f"✗ NOT FOUND: {phrase}")
            report["issues"].append(f"Number '{number}' from '{phrase}' not found in NTSB source")
    
    if response_numbers:
        verified = len(numerical_report["verified_in_source"])
        total = len(response_numbers)
        numerical_report["accuracy_rate"] = (verified / total) * 100
        print(f"\nNumerical Accuracy: {verified}/{total} ({numerical_report['accuracy_rate']:.1f}%)")
    else:
        print("No numerical claims found in response")
        numerical_report["accuracy_rate"] = 100.0  # No claims = perfect
    
    report["numerical_accuracy"] = numerical_report
    
    # ===== SECTION 2: NTSB CITATIONS =====
    print("\n[2] NTSB CITATION VERIFICATION")
    print("-" * 40)
    
    response_citations = extract_ntsb_citations(llm_response)
    golden_ntsb_nos = set(m.get("ntsb_no", "") for m in golden_chunks if m.get("ntsb_no"))
    
    citations_report = {
        "found_in_response": response_citations,
        "verified_in_source": [],
        "UNVERIFIED": []
    }
    
    for citation in response_citations:
        # Normalize formats
        cite_normalized = citation.replace("-", "/").replace(":", "/")
        
        # Check if any golden chunk matches
        found = False
        for ntsb_no in golden_ntsb_nos:
            if cite_normalized in ntsb_no or ntsb_no in cite_normalized:
                citations_report["verified_in_source"].append(citation)
                print(f"✓ VERIFIED: {citation} → {ntsb_no}")
                found = True
                break
        
        if not found:
            citations_report["UNVERIFIED"].append(citation)
            print(f"✗ NOT VERIFIED: {citation}")
    
    if response_citations:
        verified = len(citations_report["verified_in_source"])
        citations_report["verification_rate"] = (verified / len(response_citations)) * 100
        print(f"\nCitation Verification: {verified}/{len(response_citations)} ({citations_report['verification_rate']:.1f}%)")
    else:
        print("⚠ No NTSB citations found - adding to recommendations")
        report["recommendations"].append("Add NTSB citations [AAR-XX/XX] format to substantiate claims")
        citations_report["verification_rate"] = 0.0
    
    report["ntsb_citations"] = citations_report
    
    # ===== SECTION 3: SECTION COVERAGE =====
    print("\n[3] SECTION COVERAGE CHECK")
    print("-" * 40)
    
    golden_sections = {}
    for m in golden_chunks:
        section = m.get("section_title", "N/A")
        if section:
            golden_sections[section] = golden_sections.get(section, 0) + 1
    
    response_lower = llm_response.lower()
    coverage_report = {
        "golden_sections": list(golden_sections.keys())[:5],
        "mentioned_sections": [],
        "NOT_mentioned": []
    }
    
    for section in golden_sections.keys():
        section_lower = section.lower()
        if section_lower in response_lower or any(word in response_lower for word in section_lower.split()[:2]):
            coverage_report["mentioned_sections"].append(section)
            print(f"✓ MENTIONED: {section}")
        else:
            coverage_report["NOT_mentioned"].append(section)
            print(f"✗ NOT MENTIONED: {section}")
    
    coverage_rate = 100
    if golden_sections:
        coverage_rate = (len(coverage_report["mentioned_sections"]) / len(golden_sections)) * 100
    coverage_report["coverage_rate"] = coverage_rate
    
    report["validation_sections"] = coverage_report
    
    # ===== SECTION 4: OVERALL SCORING =====
    print("\n[4] OVERALL SCORE CALCULATION")
    print("-" * 40)
    
    scores = {
        "numerical_accuracy": numerical_report.get("accuracy_rate", 0),
        "citation_verification": citations_report.get("verification_rate", 0) if response_citations else 50,
        "section_coverage": coverage_report.get("coverage_rate", 0)
    }
    
    # Citations weighted more heavily
    overall = (
        scores["numerical_accuracy"] * 0.35 +
        scores["citation_verification"] * 0.40 +
        scores["section_coverage"] * 0.25
    )
    
    report["overall_score"] = overall
    
    print(f"Numerical Accuracy: {scores['numerical_accuracy']:.1f}%")
    print(f"Citation Verification: {scores['citation_verification']:.1f}%")
    print(f"Section Coverage: {scores['section_coverage']:.1f}%")
    print(f"\n>>> OVERALL ACCURACY: {overall:.1f}%")
    
    # ===== RECOMMENDATIONS =====
    if overall < 80:
        print("\n[5] RECOMMENDATIONS TO IMPROVE")
        print("-" * 40)
        
        if scores["numerical_accuracy"] < 80:
            report["recommendations"].append(
                f"Numerical accuracy low ({scores['numerical_accuracy']:.1f}%). "
                f"Verify all flight hours, dates, and aircraft specs against source."
            )
        
        if scores["citation_verification"] < 80:
            report["recommendations"].append(
                f"Citations incomplete ({scores['citation_verification']:.1f}%). "
                f"Add [NTSB/AAR-XX/XX] citations to all key claims."
            )
        
        if scores["section_coverage"] < 80:
            report["recommendations"].append(
                f"Section coverage weak ({scores['section_coverage']:.1f}%). "
                f"Include evidence from more report sections for completeness."
            )
        
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"{i}. {rec}")
    else:
        print("\n✓ EXCELLENT! Response is well-sourced and accurate.")
    
    print("\n" + "="*80 + "\n")
    
    return report


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    """Main accuracy validation workflow"""
    
    print("\n" + "="*80)
    print("NTSB RAG ACCURACY VALIDATION TOOL")
    print("="*80)
    print("\nThis tool validates your LLM response against NTSB source documents.")
    print("Workflow:")
    print("  1. Enter a question about an NTSB accident")
    print("  2. System retrieves golden chunks from NTSB reports")
    print("  3. Enter your LLM's response")
    print("  4. Tool validates accuracy (numbers, citations, sections)")
    print("="*80)
    
    # Get query
    print("\nStep 1: Enter your question about NTSB accidents")
    print("-" * 40)
    
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"Query: {query}")
    else:
        query = input("Question: ").strip()
        if not query:
            print("✗ No query provided")
            sys.exit(1)
    
    # Retrieve golden chunks
    golden_chunks = retrieve_golden_chunks(query, top_k=10)
    
    if not golden_chunks:
        print("✗ Could not retrieve golden chunks")
        sys.exit(1)
    
    # Get LLM response
    print("\nStep 2: Enter your LLM's response to validate")
    print("-" * 40)
    print("(Paste the response, then press Enter twice to finish)")
    
    lines = []
    empty_count = 0
    while True:
        try:
            line = input()
            if line.strip():
                lines.append(line)
                empty_count = 0
            else:
                empty_count += 1
                if empty_count >= 2:
                    break
        except EOFError:
            break
    
    llm_response = "\n".join(lines).strip()
    
    if not llm_response:
        print("✗ No response provided")
        sys.exit(1)
    
    print(f"\nValidating response ({len(llm_response)} chars)...\n")
    
    # Validate
    validation_report = validate_llm_response(query, llm_response, golden_chunks)
    
    # Save report
    report_file = f"accuracy_report_{validation_report['overall_score']:.0f}pct.json"
    with open(report_file, "w") as f:
        json.dump(validation_report, f, indent=2)
    
    print(f"✓ Report saved to {report_file}")
    
    # Return exit code based on accuracy
    if validation_report["overall_score"] >= 90:
        sys.exit(0)
    elif validation_report["overall_score"] >= 70:
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()
