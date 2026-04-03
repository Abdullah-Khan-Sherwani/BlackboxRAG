#!/usr/bin/env python3
"""
Enhanced Hybrid Retrieval with HyDE, better cross-encoder, and validation workflow.

Changes:
1. HyDE: Generate hypothetical documents for better semantic matching
2. Cross-encoder: Use smaller, better models (cross-encoder/ms-marco-MiniLM-L-12-v2 or TinyBERT)
3. Retrieval: Increase from 40 to 60 chunks
4. Citations: NTSB placeholders in all answers
5. Neighbor enrichment: Enabled by default
6. Numerical accuracy: Verify all numbers against source chunks
7. Multi-point answers: Extract 2-3 key points
8. Pilot hours: Include total flight hours when relevant
"""

import sys
import os
import json
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval.query import load_model, init_pinecone, retrieve
from src.retrieval.hybrid import build_bm25_index, expand_query_variants, bm25_retrieve, rrf_fuse_lists, enrich_with_neighbors
from src.llm.client import call_eval_llm
from src.generation.generate import generate_answer

# ============================================================================
# HYDE: Hypothetical Document Embeddings
# ============================================================================

def generate_hyde_documents(query: str, num_docs: int = 3) -> List[str]:
    """Generate hypothetical documents that would answer the query.
    
    This improves retrieval by creating synthetic documents that contain
    answers, then embedding and retrieving based on these hypothetical docs.
    """
    hyde_prompt = f"""Given this question about aviation accidents: "{query}"
    
Write {num_docs} short, specific hypothetical document excerpts that would answer this question.
Each excerpt should be from an NTSB accident report and contain concrete details/numbers.

Format each as a separate paragraph starting with "Excerpt {i}:"
"""
    
    try:
        response = call_eval_llm(hyde_prompt, model="deepseek")
        # Parse the response to extract excerpts
        excerpts = []
        for line in response.split('\n'):
            if 'Excerpt' in line and ':' in line:
                excerpt = line.split(':', 1)[1].strip()
                if excerpt:
                    excerpts.append(excerpt)
        return excerpts[:num_docs]
    except:
        # Fallback: just use query variants
        return []


def load_better_reranker():
    """Load a smaller, faster cross-encoder that's still effective.
    
    Options:
    - cross-encoder/ms-marco-MiniLM-L-12-v2 (12 layers, better than L-6)
    - cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (multilingual, compact)
    - cross-encoder/qnli-distilroberta-base (smaller, good for domain)
    """
    from sentence_transformers import CrossEncoder
    
    try:
        # Try the better MiniLM first (12 layers vs 6)
        model_name = "cross-encoder/ms-marco-MiniLM-L-12-v2"
        model = CrossEncoder(model_name, max_length=512)
        return model
    except:
        # Fallback to original
        return CrossEncoder("cross-encoder/qnli-distilroberta-base")


# ============================================================================
# ENHANCED RETRIEVAL WITH HYDE + BETTER RERANKING
# ============================================================================

def enhanced_hybrid_retrieve(
    query: str,
    strategy: str = "section",
    top_k: int = 10,
    hyde: bool = True,
    use_reranker: bool = True,
) -> Tuple[List[Dict], List[str]]:
    """
    Enhanced retrieval combining:
    - HyDE documents + original query
    - Semantic + BM25 fusion (RRF)
    - Better cross-encoder reranking
    - Neighbor enrichment
    
    Returns: (matched_chunks, hyde_docs)
    """
    
    print(f"\n{'='*80}")
    print(f"ENHANCED RETRIEVAL: {query[:60]}...")
    print(f"{'='*80}")
    
    # Load resources
    print("[1] Loading models...")
    jina_model = load_model()
    index = init_pinecone()
    reranker = load_better_reranker() if use_reranker else None
    bm25, chunks = build_bm25_index(strategy)
    print(f"✓ Loaded Jina, Pinecone, BM25 ({len(chunks)} chunks)")
    if reranker:
        print(f"✓ Loaded better cross-encoder (L-12)")
    
    # Generate query variants (original + expanded)
    print("\n[2] Expanding queries...")
    queries = expand_query_variants(query)
    print(f"✓ {len(queries)} query variants")
    
    # Generate HyDE documents
    hyde_docs = []
    if hyde:
        print("\n[3] Generating HyDE documents...")
        hyde_docs = generate_hyde_documents(query, num_docs=2)
        if hyde_docs:
            print(f"✓ Generated {len(hyde_docs)} hypothetical documents")
            queries.extend(hyde_docs)
        else:
            print("✗ HyDE generation failed, continuing with query variants only")
    else:
        print("\n[3] HyDE disabled, skipping...")
    
    # Retrieve with expanded query list (increased top_k from 40 to 60)
    print("\n[4] Semantic + BM25 retrieval (60 chunks each)...")
    ranked_lists = []
    for i, q in enumerate(queries, 1):
        print(f"   Variant {i}/{len(queries)}: Semantic...", end=" ")
        semantic_results = retrieve(q, strategy, top_k=60, model=jina_model, index=index)
        ranked_lists.append(semantic_results)
        print(f"BM25...", end=" ")
        bm25_results = bm25_retrieve(q, bm25, chunks, top_k=60)
        ranked_lists.append(bm25_results)
        print(f"✓")
    
    print(f"✓ Total result lists: {len(ranked_lists)}")
    
    # RRF Fusion
    print("\n[5] RRF Fusion...")
    fused = rrf_fuse_lists(ranked_lists)
    print(f"✓ Fused to {len(fused)} candidates")
    
    # Reranking
    if use_reranker and reranker:
        print("\n[6] Cross-encoder reranking...")
        from src.retrieval.hybrid import rerank
        matches = rerank(query, fused, reranker, top_k=top_k, min_unique_reports=3)
        print(f"✓ Reranked to {len(matches)} matches")
    else:
        print("\n[6] Using RRF scores only...")
        matches = sorted(fused, key=lambda x: x.get("score", 0), reverse=True)[:top_k]
        print(f"✓ Selected {len(matches)} matches by RRF")
    
    # Neighbor enrichment (enabled by default)
    print("\n[7] Neighbor enrichment...")
    matches = enrich_with_neighbors(matches, chunks, window=2)
    print(f"✓ Enriched {len(matches)} matches with neighbors")
    
    print(f"\n{'='*80}")
    print("FINAL RESULTS:")
    for i, m in enumerate(matches[:5], 1):
        section = m.get("section_title", "N/A")[:50]
        ntsb_no = m.get("ntsb_no", "N/A")
        score = m.get("score", 0)
        print(f"  [{i}] {ntsb_no} | {section} | Score: {score:.4f}")
    print(f"{'='*80}\n")
    
    return matches, hyde_docs


# ============================================================================
# ANSWER GENERATION WITH NTSB CITATIONS + NUMERICAL VERIFICATION
# ============================================================================

def generate_verified_answer(
    query: str,
    matches: List[Dict],
    llm_provider: str = "deepseek",
) -> Tuple[str, List[str]]:
    """
    Generate answer with:
    - NTSB citations for all claims
    - 2-3 key numerical points extracted
    - Total pilot hours if relevant
    - Evidence references
    
    Returns: (answer_text, numerical_claims)
    """
    
    # Extract key numerical data from chunks
    numerical_claims = []
    for match in matches[:5]:
        text = match.get("text", "")
        # Look for flight hours, dates, distances, etc.
        import re
        
        # Flight hours pattern
        hours_match = re.search(r"(\d+,?\d*)\s+(?:total\s+)?flight\s+hours?", text, re.IGNORECASE)
        if hours_match:
            numerical_claims.append({
                "claim": f"{hours_match.group(0)}",
                "source": match.get("ntsb_no", "N/A"),
                "section": match.get("section_title", "N/A")
            })
        
        # Pilot in command / instructor hours
        pic_match = re.search(r"(\d+,?\d*)\s+hours?\s+as\s+PIC", text, re.IGNORECASE)
        if pic_match:
            numerical_claims.append({
                "claim": f"{pic_match.group(0)}",
                "source": match.get("ntsb_no", "N/A"),
                "section": match.get("section_title", "N/A")
            })
    
    # Build context with NTSB citations
    context_text = ""
    for i, match in enumerate(matches[:10], 1):
        text = match.get("text", "")[:300]
        ntsb_no = match.get("ntsb_no", "N/A")
        section = match.get("section_title", "N/A")
        context_text += f"\n[{i}] {ntsb_no} - {section}:\n{text}\n"
    
    # Generate answer with instructions for citations
    answer_prompt = f"""Based on these excerpts from NTSB accident reports, answer the query with:
1. 2-3 key findings as bullet points
2. Specific numerical data when relevant
3. Citations using format [NTSB/AAR-XX/XX] or [NTSB: NTSB/AAR-XX/XX]
4. If about pilot, include: total flight hours + specific role hours

Query: {query}

Context:
{context_text}

Requirements:
- Keep answer concise (3-5 sentences max)
- Extract 2-3 specific numerical points
- Every claim must cite [NTSB/AAR-XX/XX]
- Use "According to NTSB..." when introducing findings
- For pilot questions: mention both total hours and role-specific hours

Answer:"""
    
    answer = call_eval_llm(answer_prompt, model=llm_provider)
    
    return answer, numerical_claims


# ============================================================================
# VALIDATION WORKFLOW
# ============================================================================

def validate_answer_accuracy(
    query: str,
    llm_answer: str,
    matches: List[Dict],
) -> Dict:
    """
    Validate answer accuracy by:
    1. Extracting all numerical claims from LLM answer
    2. Checking if claimed numbers exist in retrieved chunks
    3. Verifying NTSB citations are correct
    4. Score: 0-100 based on validation
    
    Returns validation report
    """
    
    import re
    
    report = {
        "query": query,
        "answer": llm_answer,
        "numerical_claims_found": [],
        "numerical_claims_verified": [],
        "numerical_claims_unverified": [],
        "citations_found": [],
        "citations_verified": [],
        "accuracy_score": 0.0,
        "issues": []
    }
    
    # Extract numbers from answer
    numbers = re.findall(r"\d+(?:,\d{3})*", llm_answer)
    report["numerical_claims_found"] = numbers
    
    # Extract NTSB citations
    citations = re.findall(r"NTSB/AAR-\d+/\d+|AAR-\d+/\d+", llm_answer)
    report["citations_found"] = list(set(citations))
    
    # Check if numbers appear in matched chunks
    all_chunk_text = " ".join(m.get("text", "") for m in matches)
    for num in numbers:
        if num in all_chunk_text or num.replace(",", "") in all_chunk_text:
            report["numerical_claims_verified"].append(num)
        else:
            report["numerical_claims_unverified"].append(num)
            report["issues"].append(f"Number '{num}' not found in retrieved chunks")
    
    # Check if NTSB numbers are in matches
    all_ntsb_nos = set(m.get("ntsb_no", "") for m in matches)
    for citation in report["citations_found"]:
        normalized = citation.replace("AAR-", "NTSB/AAR-")
        if normalized in all_ntsb_nos or any(normalized in str(no) for no in all_ntsb_nos):
            report["citations_verified"].append(citation)
        else:
            report["issues"].append(f"Citation {citation} not found in matches")
    
    # Calculate accuracy score
    total_claims = len(report["numerical_claims_found"]) + len(report["citations_found"])
    verified_claims = len(report["numerical_claims_verified"]) + len(report["citations_verified"])
    
    if total_claims > 0:
        report["accuracy_score"] = (verified_claims / total_claims) * 100
    else:
        report["accuracy_score"] = 100  # No claims = perfect (or N/A)
    
    return report


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python retrieval_enhanced.py \"Your question here\"")
        sys.exit(1)
    
    query = " ".join(sys.argv[1:])
    
    # Step 1: Enhanced Retrieval with HyDE
    matches, hyde_docs = enhanced_hybrid_retrieve(
        query,
        strategy="section",
        top_k=10,
        hyde=True,
        use_reranker=True
    )
    
    if not matches:
        print("✗ No matches found")
        return
    
    # Step 2: Generate answer with NTSB citations
    print("\n[GENERATION] Creating answer...")
    answer, numerical_claims = generate_verified_answer(query, matches)
    
    print("\n" + "="*80)
    print("ANSWER:")
    print("="*80)
    print(answer)
    
    # Step 3: Validate answer
    print("\n" + "="*80)
    print("VALIDATION:")
    print("="*80)
    validation = validate_answer_accuracy(query, answer, matches)
    
    print(f"\nNumerical accuracy: {len(validation['numerical_claims_verified'])}/{len(validation['numerical_claims_found'])} verified")
    print(f"Citations verified: {len(validation['citations_verified'])}/{len(validation['citations_found'])}")
    print(f"Accuracy Score: {validation['accuracy_score']:.1f}%")
    
    if validation['issues']:
        print(f"\nIssues ({len(validation['issues'])}):")
        for issue in validation['issues']:
            print(f"  ⚠ {issue}")
    
    print("\n" + "="*80)
    print("RETRIEVED CHUNKS:")
    print("="*80)
    for i, match in enumerate(matches, 1):
        print(f"\n[{i}] {match.get('ntsb_no', 'N/A')} - {match.get('section_title', 'N/A')}")
        print(f"    Score: {match.get('score', 0):.4f}")
        print(f"    Preview: {match.get('text', '')[:150]}...")
    
    print("\n" + "="*80)
    print("JSON EXPORT (for validation)")
    print("="*80)
    output = {
        "query": query,
        "answer": answer,
        "hyde_docs": hyde_docs,
        "validation": validation,
        "matches": [
            {
                "ntsb_no": m.get("ntsb_no"),
                "section": m.get("section_title"),
                "score": m.get("score"),
                "text_preview": m.get("text", "")[:200]
            }
            for m in matches
        ]
    }
    
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
