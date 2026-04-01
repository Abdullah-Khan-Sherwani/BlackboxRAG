"""
Hybrid retrieval: BM25 + semantic search with RRF fusion and cross-encoder reranking.
Enhanced with HyDE support, better lightweight cross-encoder, and increased retrieval.
"""
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from src.llm.client import call_eval_llm
from src.retrieval.query import load_model, init_pinecone, retrieve, available_strategies

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STRATEGIES = ["section", "fixed", "recursive", "semantic"]
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CHUNK_FILE_BY_STRATEGY = {
    "fixed": "chunks_fixed.json",
    "recursive": "chunks_recursive.json",
    "semantic": "chunks_semantic.json",
    "section": "chunks_md_section.json",
}


def build_bm25_index(strategy):
    """Load chunks for a strategy and build a BM25 index.

    Returns (BM25Okapi, list[dict]) — the index and the original chunk dicts.
    """
    filename = CHUNK_FILE_BY_STRATEGY.get(strategy, f"chunks_{strategy}.json")
    path = os.path.join(BASE_DIR, "data", "processed", filename)
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    tokenized = [chunk["text"].lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized)
    return bm25, chunks


def bm25_retrieve(query, bm25, chunks, top_k=20):
    """Return top-k chunks by BM25 score."""
    scores = bm25.get_scores(query.lower().split())
    top_indices = scores.argsort()[::-1][:top_k]
    results = []
    for idx in top_indices:
        chunk = dict(chunks[idx])
        chunk["score"] = float(scores[idx])
        chunk["retrieval_strategy"] = "bm25"
        results.append(chunk)
    return results


def rrf_fuse_lists(result_lists, k=60):
    """Reciprocal Rank Fusion over multiple ranked lists.

    Returns fused list sorted by RRF score (descending).
    """
    scores = {}
    docs = {}

    for ranked_list in result_lists:
        for rank, item in enumerate(ranked_list):
            doc_id = _get_id(item)
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
            item_strategy = _strategy_set(item)
            if doc_id not in docs:
                docs[doc_id] = _to_dict(item)
                docs[doc_id]["retrieval_strategies"] = sorted(item_strategy)
            else:
                existing = set(docs[doc_id].get("retrieval_strategies", []))
                docs[doc_id]["retrieval_strategies"] = sorted(existing | item_strategy)

    fused = []
    for doc_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        doc = docs[doc_id]
        doc["score"] = score
        strategies = doc.get("retrieval_strategies", [])
        if len(strategies) == 1:
            doc["retrieval_strategy"] = strategies[0]
        elif len(strategies) > 1:
            doc["retrieval_strategy"] = "+".join(strategies)
        else:
            doc["retrieval_strategy"] = "unknown"
        fused.append(doc)

    return fused


def load_reranker():
    """Load the cross-encoder reranking model."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return CrossEncoder(RERANKER_MODEL, device=device)


def _is_comparison_query(query):
    q = (query or "").lower()
    patterns = [
        r"\bcompare\b",
        r"\bversus\b",
        r"\bvs\.?\b",
        r"\bdifference between\b",
        r"\bacross\b",
    ]
    return any(re.search(p, q) for p in patterns)


def _is_single_event_query(query):
    """Heuristic: queries not phrased as comparisons are treated as single-event intent."""
    return not _is_comparison_query(query)


def _soft_report_budget(ranked, top_k, dominant_report, keep_ratio=0.8):
    """Keep mostly dominant-report chunks while preserving minority evidence.

    This is a soft budget (not a hard filter): a small portion of non-dominant
    chunks is retained to avoid overfitting and preserve context breadth.
    """
    if not ranked or not dominant_report:
        return ranked[:top_k]

    dom = [r for r in ranked if r.get("ntsb_no", "") == dominant_report]
    other = [r for r in ranked if r.get("ntsb_no", "") != dominant_report]

    dom_budget = max(1, min(top_k, int(round(top_k * keep_ratio))))
    other_budget = max(0, top_k - dom_budget)

    selected = dom[:dom_budget] + other[:other_budget]

    # Fill if one side is sparse.
    if len(selected) < top_k:
        already = {_get_id(s) for s in selected}
        for item in ranked:
            if _get_id(item) in already:
                continue
            selected.append(item)
            if len(selected) >= top_k:
                break

    return selected[:top_k]


def _entity_id(item: dict) -> str:
    """Return canonical entity identifier used for dynamic weighting."""
    return str(item.get("entity_id") or item.get("ntsb_no") or item.get("report_id") or "").strip()


def _dominant_entity_stats(candidates: list[dict], top_n: int = 10) -> tuple[str, float]:
    """Compute dominant entity and share in top-n candidates."""
    if not candidates:
        return "", 0.0

    ranked = sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)
    probe = ranked[:top_n]
    counts: dict[str, int] = {}
    for item in probe:
        eid = _entity_id(item)
        if not eid:
            continue
        counts[eid] = counts.get(eid, 0) + 1

    if not counts:
        return "", 0.0

    dominant = max(counts, key=counts.get)
    ratio = counts[dominant] / max(len(probe), 1)
    return dominant, ratio


def apply_dynamic_entity_weighting(
    candidates: list[dict],
    top_n: int = 10,
    threshold: float = 0.8,
    boost: float = 0.18,
) -> tuple[list[dict], dict]:
    """Apply explicit 80/20 weighting: if top-n is 80% one entity, boost it in remaining pool."""
    dominant_entity, ratio = _dominant_entity_stats(candidates, top_n=top_n)
    debug = {
        "dominant_entity": dominant_entity,
        "dominant_ratio": ratio,
        "weighting_applied": False,
    }
    if not dominant_entity or ratio < threshold:
        return candidates, debug

    ranked = sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)
    head = ranked[:top_n]
    tail = ranked[top_n:]
    for item in tail:
        if _entity_id(item) == dominant_entity:
            item["score"] = float(item.get("score", 0.0)) + boost

    boosted = sorted(head + tail, key=lambda x: x.get("score", 0.0), reverse=True)
    debug["weighting_applied"] = True
    return boosted, debug


def rerank(query, candidates, reranker, top_k=10, min_unique_reports=3):
    """Rerank candidates using a cross-encoder and return top-k.

    Adds a mild diversity constraint so one report does not monopolize results.
    """
    if not candidates:
        return []
    pairs = [[query, c["text"]] for c in candidates]
    ce_scores = reranker.predict(pairs)
    for i, score in enumerate(ce_scores):
        candidates[i]["score"] = float(score)

    # Soft anti-mixing bias for single-event queries:
    # prefer the dominant report but keep a small minority slice.
    if _is_single_event_query(query):
        report_strength = {}
        for c in candidates:
            rid = c.get("ntsb_no", "")
            if not rid:
                continue
            report_strength[rid] = report_strength.get(rid, 0.0) + max(0.0, c["score"])

        dominant_report = max(report_strength, key=report_strength.get) if report_strength else ""
        for c in candidates:
            rid = c.get("ntsb_no", "")
            if not rid or not dominant_report:
                continue
            if rid == dominant_report:
                c["score"] += 0.12
            else:
                c["score"] -= 0.06

    ranked = sorted(candidates, key=lambda x: x["score"], reverse=True)

    selected = []
    per_report = {}
    max_per_report = 2
    for item in ranked:
        ntsb_no = item.get("ntsb_no", "")
        if ntsb_no:
            if per_report.get(ntsb_no, 0) >= max_per_report:
                continue
            per_report[ntsb_no] = per_report.get(ntsb_no, 0) + 1
        selected.append(item)
        if len(selected) >= top_k:
            break

    # Fill if diversity filtering was too strict on sparse results.
    if len(selected) < top_k:
        already = {_get_id(s) for s in selected}
        for item in ranked:
            if _get_id(item) in already:
                continue
            selected.append(item)
            if len(selected) >= top_k:
                break

    # Ensure we cover at least `min_unique_reports` if possible.
    unique_ntsb = {s.get("ntsb_no", "") for s in selected if s.get("ntsb_no", "")}
    if len(unique_ntsb) < min_unique_reports:
        selected_ids = {_get_id(s) for s in selected}
        selected_ntsb = set(unique_ntsb)

        for candidate in ranked:
            cid = _get_id(candidate)
            ntsb = candidate.get("ntsb_no", "")
            if cid in selected_ids:
                continue
            if not ntsb or ntsb in selected_ntsb:
                continue

            # Replace the lowest-scored over-represented report item.
            replace_idx = None
            for i in range(len(selected) - 1, -1, -1):
                s_ntsb = selected[i].get("ntsb_no", "")
                if s_ntsb and list(x.get("ntsb_no", "") for x in selected).count(s_ntsb) > 1:
                    replace_idx = i
                    break

            if replace_idx is None:
                continue

            selected[replace_idx] = candidate
            selected_ids = {_get_id(s) for s in selected}
            selected_ntsb = {s.get("ntsb_no", "") for s in selected if s.get("ntsb_no", "")}
            if len(selected_ntsb) >= min_unique_reports:
                break

    if _is_single_event_query(query):
        # Identify dominant report after rerank and apply soft 80/20 budget.
        counts = {}
        for item in ranked[: max(top_k * 3, top_k)]:
            rid = item.get("ntsb_no", "")
            if rid:
                counts[rid] = counts.get(rid, 0) + 1
        dominant_report = max(counts, key=counts.get) if counts else ""
        selected = _soft_report_budget(ranked, top_k=top_k, dominant_report=dominant_report, keep_ratio=0.8)

    return selected


def expand_query_variants(query):
    """Generate lightweight query variants for fusion retrieval.

    This is deterministic and local (no extra LLM call), so it improves recall
    without adding API latency.
    """
    variants = [query.strip()]
    q = query.strip()

    # Decompose simple comparison queries.
    compare_match = re.search(r"compare\s+(.+?)\s+(?:vs\.?|versus|and)\s+(.+)", q, flags=re.IGNORECASE)
    if compare_match:
        left = compare_match.group(1).strip(" ?.!")
        right = compare_match.group(2).strip(" ?.!")
        variants.append(f"{left} aviation accident factors")
        variants.append(f"{right} aviation accident factors")

    # Step-back style broader framing can improve recall.
    variants.append(f"common contributing factors for {q}")

    deduped = []
    seen = set()
    for v in variants:
        key = v.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(v)
    return deduped[:3]


def generate_multi_queries(query, model=None):
    """Generate alternative query phrasings for multi-query retrieval.

    Returns a list of question strings. Each rephrasing targets a different
    angle from which NTSB reports describe the same information (probable cause,
    crew qualifications, ATC communications, aircraft systems, weather, CVR/FDR
    data, safety recommendations, etc.).
    """
    prompt = f"""You are an expert in NTSB (National Transportation Safety Board) aviation accident reports.

Your task is to generate five alternative phrasings of the question below.
Each rephrasing should approach the same information need from a different angle,
using vocabulary and framing consistent with NTSB accident reports.

NTSB reports are organized around these areas — use them to diversify your phrasings:
- Probable cause and contributing factors
- Flight crew qualifications, experience, and decision-making
- Aircraft systems, maintenance records, and airworthiness
- Air traffic control communications and instructions
- Cockpit Voice Recorder (CVR) and Flight Data Recorder (FDR) findings
- Meteorological conditions and weather data
- Safety recommendations and corrective actions

Rules:
- Return exactly five questions, one per line, no numbering or bullet points.
- Do NOT invent specific numbers, names, or dates.
- Do NOT answer the question — only rephrase it.
- Keep each question concise (one sentence).
- CRITICAL: Preserve every specific entity from the original question exactly as written —
  this includes directional terms (left/right), component names, crew roles (captain/first officer/PF/PM),
  aircraft identifiers, flight numbers, and any other named element. Never substitute, swap, or generalize them.

Original question: {query}
"""
    try:
        from src.llm.client import call_llm, MODEL_GPT
        raw = call_llm(prompt, model=MODEL_GPT if model == "gpt" else None)
        if not raw:
            return []
        variants = [line.strip() for line in raw.splitlines() if line.strip()]
        return variants[:5]
    except Exception:
        return []


def generate_hyde_documents(query: str, num_docs: int = 3, llm_provider: str = "nvidia", ollama_model: str = "qwen2.5:32b") -> list[str]:
    """Generate hypothetical answer-like excerpts for HyDE retrieval expansion."""
    hyde_prompt = f"""Given this question about aviation accidents: "{query}"

Write {num_docs} short, specific hypothetical document excerpts that would answer this question.
Each excerpt should be from an NTSB accident report and contain concrete details/numbers.

Format each as a separate paragraph starting with "Excerpt {{i}}:"
"""

    try:
        if llm_provider == "ollama":
            from src.llm.ollama_client import call_ollama
            response = call_ollama(hyde_prompt, model=ollama_model)
        elif llm_provider == "gpt":
            from src.llm.client import call_llm, MODEL_GPT
            response = call_llm(hyde_prompt, model=MODEL_GPT)
        else:
            response = call_eval_llm(hyde_prompt)

        excerpts = []
        for line in response.split("\n"):
            if "Excerpt" in line and ":" in line:
                excerpt = line.split(":", 1)[1].strip()
                if excerpt:
                    excerpts.append(excerpt)
        return excerpts[:num_docs]
    except Exception as e:
        print(f"[HyDE] generation failed: {e}")
        return []


def generate_knowledge_doc(query: str, llm_provider: str = "nvidia", ollama_model: str = "qwen2.5:32b") -> str:
    """Generate a direct answer using the LLM's own aviation knowledge.

    The answer is used only for semantic retrieval — not BM25 — to pull the
    search into answer-space and surface chunks that match the answer rather
    than the question phrasing.
    """
    prompt = f"""You are an aviation accident investigator writing a factual narrative based on your knowledge of NTSB investigations from 1996 to 2025.

Answer the following question by writing a concise investigative narrative in the style of an NTSB report finding.
Use precise technical language, third-person past tense, and include specific details such as flight hours, system states,
crew actions, meteorological conditions, or causal factors as appropriate. Do not hedge or qualify — write as a definitive finding.
Only draw on accidents and events from NTSB reports between 1996 and 2025.

Question: {query}
"""
    try:
        if llm_provider == "ollama":
            from src.llm.ollama_client import call_ollama
            response = call_ollama(prompt, model=ollama_model)
        elif llm_provider == "gpt":
            from src.llm.client import call_llm, MODEL_GPT
            response = call_llm(prompt, model=MODEL_GPT)
        else:
            response = call_eval_llm(prompt)
        return response.strip()
    except Exception as e:
        print(f"[KnowledgeDoc] generation failed: {e}")
        return ""


def _chunk_maps(chunks):
    """Build quick lookup maps for neighbor context enrichment."""
    by_doc = {}
    for c in chunks:
        key = c.get("ntsb_no") or c.get("report_id", "")
        if key:
            by_doc.setdefault(key, []).append(c)

    for key, doc_chunks in by_doc.items():
        doc_chunks.sort(key=lambda x: x.get("chunk_id", ""))
        by_doc[key] = doc_chunks

    return by_doc


def enrich_with_neighbors(results, chunks, window=1):
    """Attach neighboring chunk text to each result for better grounding."""
    if not results or window <= 0:
        return results

    by_doc = _chunk_maps(chunks)
    enriched = []

    for r in results:
        cid = r.get("chunk_id", "")
        ntsb = r.get("ntsb_no") or r.get("report_id", "")
        if not cid or not ntsb or ntsb not in by_doc:
            enriched.append(r)
            continue

        doc_chunks = by_doc[ntsb]
        idx = next((i for i, c in enumerate(doc_chunks) if c.get("chunk_id") == cid), None)
        if idx is None:
            enriched.append(r)
            continue

        start = max(0, idx - window)
        end = min(len(doc_chunks), idx + window + 1)
        segment = []
        for c in doc_chunks[start:end]:
            t = c.get("text", "")
            if t:
                segment.append(t)

        out = dict(r)
        if segment:
            out["text"] = "\n\n".join(segment)
        enriched.append(out)

    return enriched


def hybrid_retrieve(query, strategy, top_k=10, model=None, index=None,
                    bm25=None, chunks=None, reranker=None,
                    semantic_top_k=60, bm25_top_k=60,
                    enable_query_expansion=True,
                    neighbor_window=2,
                    use_multi_query=True,
                    use_hyde=False,
                    min_unique_reports=3,
                    return_debug=False):
    """Full hybrid retrieval pipeline: semantic + BM25 → RRF → rerank.

    Args:
        query: Search query string
        strategy: Chunking strategy (section, fixed, recursive, etc.)
        top_k: Final number of results to return
        model: Semantic model instance
        index: Pinecone index instance
        bm25: BM25 index instance
        chunks: All chunks for BM25
        reranker: Cross-encoder instance
        semantic_top_k: Top-k for semantic retrieval (increased from 40→60)
        bm25_top_k: Top-k for BM25 retrieval (increased from 40→60)
        enable_query_expansion: Whether to expand queries
        neighbor_window: Context window for enrichment (increased from 1→2)
        use_multi_query: Whether to use multi-query expansion (replaces use_hyde)
        use_hyde: Whether to add HyDE hypothetical excerpts as retrieval queries
        min_unique_reports: Minimum unique NTSB reports to include
        return_debug: Return debug info with results

    Returns list[dict] with keys: text, ntsb_no, event_date, make, model, score, etc.
    If return_debug=True, returns (results, debug_info).
    """
    debug = {
        "multi_queries": None,
        "hyde_docs": [],
        "query_variants": [],
        "dynamic_weighting": {
            "dominant_entity": "",
            "dominant_ratio": 0.0,
            "weighting_applied": False,
        },
    }

    # Query set: prioritize LLM multi-query (3-5 variants) and backfill deterministically.
    queries = [query]
    if use_multi_query:
        multi_queries = generate_multi_queries(query)
        if multi_queries:
            debug["multi_queries"] = multi_queries
            queries.extend(multi_queries)

    if use_hyde:
        hyde_docs = generate_hyde_documents(query, num_docs=2)
        if hyde_docs:
            debug["hyde_docs"] = hyde_docs
            print("[HyDE] Generated hypothetical snippets:")
            for i, doc in enumerate(hyde_docs, 1):
                print(f"  [{i}] {doc}")
            queries.extend(hyde_docs)

    if enable_query_expansion and len(queries) < 3:
        queries.extend(expand_query_variants(query))

    deduped = []
    seen = set()
    for q in queries:
        key = (q or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(q.strip())
    queries = deduped[:5]
    debug["query_variants"] = list(queries)

    ranked_lists = []
    for q in queries:
        # Execute lexical and semantic retrieval in parallel per query variant.
        with ThreadPoolExecutor(max_workers=2) as executor:
            fut_sem = executor.submit(retrieve, q, strategy, semantic_top_k, model, index)
            fut_bm25 = executor.submit(bm25_retrieve, q, bm25, chunks, bm25_top_k)
            semantic_results = fut_sem.result()
            bm25_results = fut_bm25.result()
        ranked_lists.append(semantic_results)
        ranked_lists.append(bm25_results)

    # Fuse all ranked lists (semantic + lexical over expanded queries)
    fused = rrf_fuse_lists(ranked_lists)
    fused, dyn_dbg = apply_dynamic_entity_weighting(fused, top_n=10, threshold=0.8, boost=0.18)
    debug["dynamic_weighting"] = dyn_dbg

    # Rerank
    results = rerank(query, fused, reranker, top_k=top_k, min_unique_reports=min_unique_reports)
    results = enrich_with_neighbors(results, chunks, window=neighbor_window)

    if return_debug:
        return results, debug
    return results


# --- helpers ---

def _get_id(item):
    """Extract a unique ID from either a Pinecone match or a dict."""
    if hasattr(item, "id"):
        return item.id
    return item.get("chunk_id", id(item))


def _to_dict(item):
    """Normalize a Pinecone match or plain dict into a standard dict."""
    if hasattr(item, "metadata"):
        d = dict(item.metadata)
        d["score"] = float(item.score)
        return d
    return dict(item)


def _strategy_set(item):
    """Extract retrieval source(s) from a Pinecone match or dict item."""
    if hasattr(item, "metadata"):
        value = item.metadata.get("retrieval_strategy", "semantic")
    else:
        value = item.get("retrieval_strategy", "bm25")

    if isinstance(value, str):
        parts = [p.strip() for p in value.split("+") if p.strip()]
        if parts:
            return set(parts)
    if isinstance(value, list):
        return {str(v).strip() for v in value if str(v).strip()}
    return set()


def main():
    """Demo: run hybrid retrieval for a sample query."""
    jina_model = load_model()
    pinecone_index = init_pinecone()
    reranker = load_reranker()

    strategies = available_strategies()
    strategy = "recursive" if "recursive" in strategies else strategies[0]
    bm25, chunks = build_bm25_index(strategy)

    query = "What are common causes of engine failure during takeoff?"
    results = hybrid_retrieve(
        query, strategy, top_k=5,
        model=jina_model, index=pinecone_index,
        bm25=bm25, chunks=chunks, reranker=reranker,
    )

    print(f"Query: {query}\n")
    for i, r in enumerate(results, 1):
        print(f"[{i}] Score: {r['score']:.4f} | NTSB: {r.get('ntsb_no', 'N/A')}")
        print(f"    {r['text'][:150]}...\n")


if __name__ == "__main__":
    main()
