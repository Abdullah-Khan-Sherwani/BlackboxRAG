"""
RAG Answer Generation using Google Gemini.
Takes retrieved chunks + user query → calls Gemini → produces an answer.
Supports both Pinecone matches and dict-based hybrid results.
"""
import os
import sys

from dotenv import load_dotenv
import google.generativeai as genai

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.retrieval.query import load_model, init_pinecone, retrieve
from src.retrieval.hybrid import (
    build_bm25_index, load_reranker, hybrid_retrieve,
)

GEMINI_MODEL = "gemini-2.0-flash"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SAMPLE_QUERIES = [
    "What are common causes of engine failure during takeoff?",
    "Cessna accidents in instrument meteorological conditions",
    "How does pilot experience affect landing accident outcomes?",
]


def init_gemini():
    """Load environment variables and configure the Gemini API."""
    load_dotenv(os.path.join(BASE_DIR, ".env"))
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not found in environment. Check your .env file.")
    genai.configure(api_key=api_key)


def _get_meta(chunk, key, default="N/A"):
    """Get metadata from a Pinecone match or a plain dict."""
    if hasattr(chunk, "metadata"):
        return chunk.metadata.get(key, default)
    return chunk.get(key, default)


def build_prompt(query, retrieved_chunks):
    """Build a prompt with system instruction, context chunks, and the user query.

    Works with both Pinecone match objects and plain dicts.
    """
    system = (
        "You are an NTSB aviation safety expert. Answer based only on the provided context. "
        "If the context is insufficient, say so."
    )

    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        header = (
            f"[Context {i}] NTSB No: {_get_meta(chunk, 'ntsb_no')} | "
            f"Date: {_get_meta(chunk, 'event_date')} | "
            f"Aircraft: {_get_meta(chunk, 'make')} {_get_meta(chunk, 'model')}"
        )
        text = _get_meta(chunk, "text", "")
        context_blocks.append(f"{header}\n{text}")

    context_str = "\n\n".join(context_blocks)
    return f"{system}\n\n--- Context ---\n{context_str}\n\n--- Question ---\n{query}"


def generate_answer(query, retrieved_chunks):
    """Generate an answer using Gemini given the query and retrieved chunks."""
    prompt = build_prompt(query, retrieved_chunks)
    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt)
    return response.text


def rag_pipeline(query, strategy, top_k=5, model=None, index=None):
    """End-to-end semantic-only RAG: retrieve chunks then generate an answer."""
    matches = retrieve(query, strategy, top_k=top_k, model=model, index=index)
    answer = generate_answer(query, matches)
    source_ids = [m.id for m in matches]
    return {
        "query": query,
        "strategy": strategy,
        "answer": answer,
        "sources": source_ids,
        "num_chunks": len(matches),
    }


def rag_pipeline_hybrid(query, strategy, top_k=5,
                        model=None, index=None,
                        bm25=None, chunks=None, reranker=None):
    """End-to-end hybrid RAG: BM25 + semantic → RRF → rerank → generate."""
    results = hybrid_retrieve(
        query, strategy, top_k=top_k,
        model=model, index=index,
        bm25=bm25, chunks=chunks, reranker=reranker,
    )
    answer = generate_answer(query, results)
    source_ids = [r.get("chunk_id", r.get("ntsb_no", "unknown")) for r in results]
    return {
        "query": query,
        "strategy": strategy,
        "answer": answer,
        "sources": source_ids,
        "num_chunks": len(results),
        "contexts": results,
    }


def main():
    jina_model = load_model()
    index = init_pinecone()
    init_gemini()
    reranker = load_reranker()

    strategies = ["fixed", "recursive", "semantic"]

    for query in SAMPLE_QUERIES:
        for strategy in strategies:
            print(f"\n{'='*80}")
            print(f"Query: {query}")
            print(f"Strategy: {strategy}")

            # Semantic-only
            result = rag_pipeline(query, strategy, top_k=5, model=jina_model, index=index)
            print(f"\n[Semantic] Answer:\n{result['answer'][:300]}...")

            # Hybrid
            bm25, chunks = build_bm25_index(strategy)
            result_h = rag_pipeline_hybrid(
                query, strategy, top_k=5,
                model=jina_model, index=index,
                bm25=bm25, chunks=chunks, reranker=reranker,
            )
            print(f"\n[Hybrid] Answer:\n{result_h['answer'][:300]}...")


if __name__ == "__main__":
    main()
