"""
RAG Answer Generation using DeepSeek V3.2 via NVIDIA API.
Takes retrieved chunks + user query → calls LLM → produces an answer.
Supports both Pinecone matches and dict-based hybrid results.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.llm.client import call_llm
from src.llm.ollama_client import call_ollama
from src.retrieval.query import load_model, init_pinecone, retrieve
from src.retrieval.hybrid import (
    build_bm25_index, load_reranker, hybrid_retrieve,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SYSTEM_PROMPT = """\
You are an expert in aircraft accident analysis with deep knowledge of NTSB investigation reports.
Your task is to answer the user's question based ONLY on the provided context excerpts from NTSB accident reports.

===Response Requirements===
1. Analyze the question from up to four aspects where relevant: human factors, aircraft/mechanical factors, environmental factors, and organizational factors.
2. All conclusions must be derived from the provided context. No external assumptions or interpretations are permitted.
3. Cite specific NTSB report numbers (e.g., ERA19FA249) when referencing findings.
4. If the context is insufficient to answer the question, state so clearly rather than speculating.
5. Keep the response concise and well-structured.
===End of Response Requirements==="""

SAMPLE_QUERIES = [
    "What are common causes of engine failure during takeoff?",
    "Cessna accidents in instrument meteorological conditions",
    "How does pilot experience affect landing accident outcomes?",
]


def _get_meta(chunk, key, default="N/A"):
    """Get metadata from a Pinecone match or a plain dict."""
    if hasattr(chunk, "metadata"):
        return chunk.metadata.get(key, default)
    return chunk.get(key, default)


def build_prompt(query, retrieved_chunks):
    """Build system instruction and user prompt from query and context chunks.

    Works with both Pinecone match objects and plain dicts.

    Returns:
        (system_message, user_prompt) tuple.
    """
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
    user_prompt = f"--- Context ---\n{context_str}\n\n--- Question ---\n{query}"
    return SYSTEM_PROMPT, user_prompt


def generate_answer(query, retrieved_chunks, llm_provider="deepseek", ollama_model="qwen2.5:32b"):
    """Generate an answer given the query and retrieved chunks."""
    system, user_prompt = build_prompt(query, retrieved_chunks)
    provider = (llm_provider or "deepseek").lower()
    if provider == "ollama":
        return call_ollama(
            user_prompt,
            system_prompt=system,
            model=ollama_model,
            temperature=0.1,
            max_tokens=2048,
            timeout=240,
        )
    return call_llm(user_prompt, system=system)


def rag_pipeline(query, strategy, top_k=5, model=None, index=None,
                 llm_provider="deepseek", ollama_model="qwen2.5:32b"):
    """End-to-end semantic-only RAG: retrieve chunks then generate an answer."""
    matches = retrieve(query, strategy, top_k=top_k, model=model, index=index)
    answer = generate_answer(
        query,
        matches,
        llm_provider=llm_provider,
        ollama_model=ollama_model,
    )
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
                        bm25=None, chunks=None, reranker=None,
                        llm_provider="deepseek", ollama_model="qwen2.5:32b"):
    """End-to-end hybrid RAG: BM25 + semantic → RRF → rerank → generate."""
    results = hybrid_retrieve(
        query, strategy, top_k=top_k,
        model=model, index=index,
        bm25=bm25, chunks=chunks, reranker=reranker,
    )
    answer = generate_answer(
        query,
        results,
        llm_provider=llm_provider,
        ollama_model=ollama_model,
    )
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
