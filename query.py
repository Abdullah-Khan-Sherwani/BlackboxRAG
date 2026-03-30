#!/usr/bin/env python3
"""
PRODUCTION RAG QUERY TOOL
Just run: python query.py "Your question here"
No Streamlit UI needed. Hybrid retrieval by default.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval.query import load_model, init_pinecone, retrieve
from src.retrieval.hybrid import (
    build_bm25_index, load_reranker, expand_query_variants,
    bm25_retrieve, rrf_fuse_lists, rerank, enrich_with_neighbors
)
from src.generation.generate import generate_answer

def query_rag(question: str, strategy: str = "section", top_k: int = 10):
    """
    Query the NTSB RAG system using hybrid retrieval.
    Returns the answer string.
    """
    
    print(f"\n{'='*90}")
    print(f"Query: {question}")
    print(f"{'='*90}\n")
    
    try:
        # Load models
        print("[Loading...] Initializing models and indexes...")
        jina_model = load_model()
        index = init_pinecone()
        reranker = load_reranker()
        bm25, chunks = build_bm25_index(strategy)
        
        # Query expansion
        print("[Retrieving...] Expanding query variants...")
        queries = expand_query_variants(question)
        
        # Multi-retrieval (semantic + BM25)
        ranked_lists = []
        for q in queries:
            ranked_lists.append(retrieve(q, strategy, top_k=20, model=jina_model, index=index))
            ranked_lists.append(bm25_retrieve(q, bm25, chunks, top_k=20))
        
        # RRF Fusion + Reranking
        print("[Processing...] Fusing and reranking results...")
        fused = rrf_fuse_lists(ranked_lists)
        matches = rerank(question, fused, reranker, top_k=top_k, min_unique_reports=3)
        matches = enrich_with_neighbors(matches, chunks, window=1)
        
        # Generate answer
        print("[Generating...] Creating response from context...\n")
        answer = generate_answer(question, matches)
        
        # Display
        print(f"{'─'*90}")
        print(answer)
        print(f"{'─'*90}\n")
        
        return answer
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python query.py \"Your question about NTSB accidents\"")
        print("\nExamples:")
        print('  python query.py "What happened to Asiana Flight 214?"')
        print('  python query.py "What substance obscured the ARFF passenger?"')
        sys.exit(1)
    
    question = " ".join(sys.argv[1:])
    query_rag(question)
