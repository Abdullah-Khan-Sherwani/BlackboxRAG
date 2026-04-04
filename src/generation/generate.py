"""
RAG Answer Generation using DeepSeek V3.2 via NVIDIA API.
Takes retrieved chunks + user query → calls LLM → produces an answer.
Supports both Pinecone matches and dict-based hybrid results.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.llm.client import call_llm, MODEL_GPT
from src.llm.ollama_client import call_ollama
from src.retrieval.query import load_model, init_pinecone, retrieve, available_strategies
from src.retrieval.hybrid import (
    build_bm25_index, load_reranker, hybrid_retrieve,
)
from src.retrieval.report_mapper import get_exec_summary

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SYSTEM_PROMPT = """\
You are an expert in aircraft accident analysis with deep knowledge of NTSB investigation reports.
Your task is to answer the user's question based ONLY on the provided context excerpts from NTSB accident reports.

===Response Requirements===
1. Analyze the question from up to four aspects where relevant: human factors, aircraft/mechanical factors, environmental factors, and organizational factors.
2. All conclusions must be derived from the provided context. No external assumptions or interpretations are permitted.
3. Every factual claim MUST include at least one citation in this exact format: [NTSB: <report_number>].
4. If a claim cannot be supported by the provided context, explicitly say "Insufficient context" and do not guess.
5. Keep the response concise and well-structured.
6. Use this structure:
     - Evidence:
         - bullet claims with citations
     - Answer:
         - synthesized conclusion based only on evidence above
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


def _is_comparison_query(query):
    q = (query or "").lower()
    patterns = [r"\bcompare\b", r"\bversus\b", r"\bvs\.?\b", r"\bdifference between\b", r"\bacross\b"]
    return any(re.search(p, q) for p in patterns)


def _report_ids(chunks):
    ids = []
    for c in chunks:
        rid = _get_meta(c, "ntsb_no", "") or _get_meta(c, "report_id", "")
        rid = str(rid).strip()
        if rid:
            ids.append(rid)
    return ids


def _dominant_report_id(chunks):
    counts = {}
    for rid in _report_ids(chunks):
        counts[rid] = counts.get(rid, 0) + 1
    if not counts:
        return ""
    return max(counts, key=counts.get)


def build_prompt(query, retrieved_chunks):
    """Build system instruction and user prompt from query and context chunks.

    Works with both Pinecone match objects and plain dicts.

    Returns:
        (system_message, user_prompt) tuple.
    """
    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        report_id = (
            _get_meta(chunk, "report_id", "")
            or _get_meta(chunk, "ntsb_no", "")
            or _get_meta(chunk, "entity_id", "")
            or "Unknown"
        )
        role = _get_meta(chunk, "role", "Unknown")
        section = _get_meta(chunk, "section_title", "")
        context_summary = _get_meta(chunk, "context_summary", "")
        numerics = _get_meta(chunk, "numerics", "None")
        aircraft_components = _get_meta(chunk, "aircraft_components", "None")
        event_date = _get_meta(chunk, "event_date", "")
        make = _get_meta(chunk, "make", "")
        model = _get_meta(chunk, "model", "")
        state = _get_meta(chunk, "state", "")
        aircraft = f"{make} {model}".strip() or "N/A"
        header = (
            f"[Report: {report_id} | Date: {event_date or 'N/A'} | "
            f"Aircraft: {aircraft} | Location: {state or 'N/A'} | Section: {section or 'N/A'}]"
        )
        contextualized = _get_meta(chunk, "contextualized_text", "")
        text = contextualized if contextualized else _get_meta(chunk, "text", "")
        context_blocks.append(
            f"{header}\n"
            f"[Chunk: {i} | Role: {role}]\n"
            f"[Key Numerics: {numerics} | Components: {aircraft_components}]\n"
            f"[Context: {context_summary or 'N/A'}]\n"
            f"Data: {text}"
        )

    context_str = "\n\n".join(context_blocks)

    # Prepend executive summaries for each unique report found in the chunks.
    # This gives the LLM a high-level crash overview before reading detailed chunks.
    seen_rids: set = set()
    overview_blocks = []
    for chunk in retrieved_chunks:
        rid = (
            _get_meta(chunk, "report_id", "")
            or _get_meta(chunk, "ntsb_no", "")
            or _get_meta(chunk, "entity_id", "")
        ).strip()
        if rid and rid not in seen_rids:
            seen_rids.add(rid)
            summary = get_exec_summary(rid)
            if summary:
                # Cap at 3000 chars to avoid overwhelming the context window.
                # For short reports this includes the full overview + probable cause;
                # for long reports it covers the crash scenario (probable cause
                # comes from the retrieved detail chunks instead).
                overview_blocks.append(f"[REPORT OVERVIEW: {rid}]\n{summary[:3000]}")

    overview_str = ""
    if overview_blocks:
        overview_str = (
            "--- Executive Summaries (high-level crash context) ---\n"
            + "\n\n".join(overview_blocks)
            + "\n\n"
        )

    # Anti-merge guard: for single-event questions, avoid combining facts across
    # different report ids when retrieval is mixed.
    report_ids = set(_report_ids(retrieved_chunks))
    guard = ""
    if len(report_ids) > 1 and not _is_comparison_query(query):
        dominant = _dominant_report_id(retrieved_chunks)
        guard = (
            "\n\n--- Safety Rule ---\n"
            f"The retrieved context spans multiple reports ({', '.join(sorted(report_ids)[:6])}). "
            f"Treat this as a single-event question and prioritize report {dominant or 'the dominant report'}. "
            "Do NOT combine crew numbers, dates, or quantitative facts across different report IDs. "
            "If key facts conflict across reports, state the ambiguity and ask for a specific report/flight identifier."
        )

    user_prompt = f"--- Context ---\n{overview_str}{context_str}{guard}\n\n--- Question ---\n{query}"
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
    if provider == "gpt":
        return call_llm(user_prompt, system=system, model=MODEL_GPT)
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


def rag_pipeline_hybrid(query, strategy, top_k=10,
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

    strategies = available_strategies()

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
