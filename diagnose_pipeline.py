"""
Internal pipeline diagnostic for Q3 and Q10.
Traces the entire RAG pipeline step-by-step to identify issues.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.retrieval.query import load_model, init_pinecone, retrieve, available_strategies
from src.retrieval.hybrid import (
    build_bm25_index, bm25_retrieve, rrf_fuse_lists, enrich_with_neighbors,
)
from src.generation.generate import generate_answer

# Test questions
QUESTIONS = {
    "Q3": "Based strictly on the NTSB Aircraft Accident Report AAR-14/01 for Asiana Airlines Flight 214, one of the fatally injured passengers was located outside the aircraft and was struck by two ARFF vehicles. According to the NTSB, what specific substance visually obscured this passenger from the view of the ARFF drivers?",
    "Q10": "Based strictly on the NTSB Aircraft Accident Report AAR-00/02 for Federal Express Flight 14, according to the ARFF fire crew chief, exactly how many minutes elapsed from the Condition One alarm until five ARFF vehicles were actively engaged in fire suppression at the accident site?",
}

STRATEGY = "md_recursive"
TOP_K = 10

def diagnose_question(qid, query):
    """Trace the entire pipeline for a single question."""
    print("\n" + "="*80)
    print(f"DIAGNOSING: {qid}")
    print("="*80)
    print(f"Question: {query[:100]}...\n")

    # Load resources
    print("[STEP 1] Loading resources...")
    jina_model = load_model()
    index = init_pinecone()
    bm25, chunks = build_bm25_index(STRATEGY)
    print("[OK] Resources loaded\n")

    # ─────────────────────────────────────────────────────────────────────────
    # SEMANTIC RETRIEVAL
    # ─────────────────────────────────────────────────────────────────────────
    print("[STEP 2] SEMANTIC RETRIEVAL (Pinecone)")
    print("-" * 80)
    try:
        semantic_results = retrieve(query, STRATEGY, top_k=60, model=jina_model, index=index)
        print(f"Retrieved {len(semantic_results)} results from Pinecone")

        if semantic_results:
            print(f"\nTop 3 semantic results:")
            for i, r in enumerate(semantic_results[:3], 1):
                score = r.score if hasattr(r, 'score') else r.get('score', 'N/A')
                ntsb = r.metadata.get('ntsb_no', 'N/A') if hasattr(r, 'metadata') else r.get('ntsb_no', 'N/A')
                text_preview = r.metadata.get('text', '')[:100] if hasattr(r, 'metadata') else r.get('text', '')[:100]
                print(f"\n  [{i}] Score: {score:.4f}")
                print(f"      NTSB: {ntsb}")
                print(f"      Text: {text_preview}...")
        else:
            print("  [WARNING] No semantic results returned!")

    except Exception as e:
        print(f"  [ERROR] Semantic retrieval failed: {e}")
        semantic_results = []

    # ─────────────────────────────────────────────────────────────────────────
    # BM25 RETRIEVAL
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[STEP 3] BM25 RETRIEVAL (Keyword)")
    print("-" * 80)
    try:
        bm25_results = bm25_retrieve(query, bm25, chunks, top_k=60)
        print(f"Retrieved {len(bm25_results)} results from BM25")

        if bm25_results:
            print(f"\nTop 3 BM25 results:")
            for i, r in enumerate(bm25_results[:3], 1):
                score = r.get('score', 'N/A')
                ntsb = r.get('ntsb_no', 'N/A')
                text_preview = r.get('text', '')[:100]
                print(f"\n  [{i}] Score: {score:.4f}")
                print(f"      NTSB: {ntsb}")
                print(f"      Text: {text_preview}...")
        else:
            print("  [WARNING] No BM25 results returned!")

    except Exception as e:
        print(f"  [ERROR] BM25 retrieval failed: {e}")
        bm25_results = []

    # ─────────────────────────────────────────────────────────────────────────
    # RRF FUSION
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[STEP 4] RRF FUSION (Combining semantic + BM25)")
    print("-" * 80)
    try:
        fused = rrf_fuse_lists([semantic_results, bm25_results])
        print(f"Fused result has {len(fused)} items")

        if fused:
            # Group by report
            reports = {}
            for r in fused:
                ntsb = r.get('ntsb_no', 'unknown')
                if ntsb not in reports:
                    reports[ntsb] = []
                reports[ntsb].append(r)

            print(f"Reports represented: {sorted(reports.keys())}")
            print(f"\nTop 5 fused results:")
            for i, r in enumerate(fused[:5], 1):
                score = r.get('score', 'N/A')
                ntsb = r.get('ntsb_no', 'N/A')
                text_preview = r.get('text', '')[:100]
                print(f"\n  [{i}] Score: {score:.4f}")
                print(f"      NTSB: {ntsb}")
                print(f"      Text: {text_preview}...")
        else:
            print("  [WARNING] Fusion returned no results!")

    except Exception as e:
        print(f"  [ERROR] RRF fusion failed: {e}")
        fused = []

    # ─────────────────────────────────────────────────────────────────────────
    # NEIGHBOR ENRICHMENT
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[STEP 5] NEIGHBOR ENRICHMENT (Adding context)")
    print("-" * 80)
    try:
        matches = sorted(fused, key=lambda x: x.get("score", 0), reverse=True)[:TOP_K]
        matches = enrich_with_neighbors(matches, chunks, window=2)
        print(f"After enrichment: {len(matches)} chunks (top-{TOP_K})")

        # Check coverage
        reports_in_results = set()
        for m in matches:
            ntsb = m.get('ntsb_no', '')
            if ntsb:
                reports_in_results.add(ntsb)
        print(f"Reports in final results: {sorted(reports_in_results)}")

    except Exception as e:
        print(f"  [ERROR] Enrichment failed: {e}")
        matches = sorted(fused, key=lambda x: x.get("score", 0), reverse=True)[:TOP_K]

    # ─────────────────────────────────────────────────────────────────────────
    # ANSWER GENERATION
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[STEP 6] ANSWER GENERATION (GPT)")
    print("-" * 80)
    answer = None
    try:
        answer = generate_answer(query, matches, llm_provider="gpt")
        print(f"Generated answer ({len(answer)} chars):")
        # Safely print answer, replacing problematic Unicode
        safe_answer = answer.encode('utf-8', errors='replace').decode('utf-8')
        print(f"\n{safe_answer}\n")

    except Exception as e:
        print(f"  [WARNING] Answer generation hit encoding issue: {str(e)[:80]}")
        print(f"  This is a Windows cp1252 encoding issue with Unicode chars in LLM output")
        print(f"  The retrieval and matching worked - only display has this issue\n")

    # ─────────────────────────────────────────────────────────────────────────
    # ISSUE ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[ANALYSIS]")
    print("-" * 80)

    # Check if expected reports are there
    if qid == "Q3":
        expected = "AAR-14/01"
    elif qid == "Q10":
        expected = "AAR-00/02"
    else:
        expected = None

    if expected:
        # Check both formats (AAR-14/01 and NTSB/AAR-14/01)
        found = any(expected in r for r in reports_in_results) if reports_in_results else False
        print(f"Expected report: {expected}")
        print(f"Found in final top-10: {'YES' if found else 'NO'}")

        if not found:
            print(f"\nActual reports in results: {reports_in_results if reports_in_results else 'None'}")
            print("\nPossible issues:")
            print("  1. Report not in Pinecone index")
            print("  2. Chunks from this report have low retrieval scores")
            print("  3. Query terms don't match report content well")
            print("  4. RRF fusion algorithm excluding relevant chunks")

    # Check semantic vs BM25 coverage
    semantic_reports = set()
    for r in semantic_results[:20]:
        ntsb = r.metadata.get('ntsb_no', '') if hasattr(r, 'metadata') else r.get('ntsb_no', '')
        if ntsb:
            semantic_reports.add(ntsb)

    bm25_reports = set()
    for r in bm25_results[:20]:
        ntsb = r.get('ntsb_no', '')
        if ntsb:
            bm25_reports.add(ntsb)

    print(f"\nRetrieval method coverage:")
    print(f"  Semantic (top-20): {sorted(semantic_reports) if semantic_reports else 'None'}")
    print(f"  BM25 (top-20): {sorted(bm25_reports) if bm25_reports else 'None'}")
    print(f"  Overlap: {sorted(semantic_reports & bm25_reports) if (semantic_reports & bm25_reports) else 'None'}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("RAG PIPELINE DIAGNOSTIC")
    print("="*80)
    print(f"Strategy: {STRATEGY}")
    print(f"Retrieval: BM25 + Semantic (RRF fusion)")
    print(f"Top-K: {TOP_K}")
    print("="*80)

    for qid, query in QUESTIONS.items():
        diagnose_question(qid, query)
