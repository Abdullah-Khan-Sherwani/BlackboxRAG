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
STRATEGIES = ["section", "md_recursive", "parent_child", "fixed", "recursive", "semantic"]
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _candidate_chunk_paths(strategy: str):
    """Return candidate chunk paths for strategy across legacy/new layouts."""
    base = os.path.join(BASE_DIR, "data", "processed")
    alt = os.path.join(base, "chunks_md_recursive")

    by_strategy = {
        "section": ["chunks_md_section_enriched.json", "chunks_md_section.json"],
        "md_recursive": ["chunks_md_md_recursive.json", "chunks_md_recursive.json"],
        "parent_child": ["chunks_md_parent_child.json", "chunks_parent_child.json"],
        "parent": ["chunks_md_parent_child.json", "chunks_parent.json"],
        "fixed": ["chunks_baseline_fixed.json", "chunks_fixed.json"],
        "recursive": ["chunks_baseline_recursive.json", "chunks_recursive.json"],
        "semantic": ["chunks_baseline_semantic.json", "chunks_semantic.json"],
    }

    names = by_strategy.get(strategy, [f"chunks_{strategy}.json"])
    paths = []
    for name in names:
        paths.append(os.path.join(base, name))
        paths.append(os.path.join(alt, name))
    return paths


def build_bm25_index(strategy):
    """Load chunks for a strategy and build a BM25 index.

    Returns (BM25Okapi, list[dict]) — the index and the original chunk dicts.
    """
    path = next((p for p in _candidate_chunk_paths(strategy) if os.path.exists(p)), None)
    if not path:
        tried = "\n  - " + "\n  - ".join(_candidate_chunk_paths(strategy))
        raise FileNotFoundError(
            f"No chunk file found for strategy '{strategy}'. Tried:{tried}"
        )
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


def rrf_fuse_lists(result_lists, k=15, max_per_report=8):
    """Reciprocal Rank Fusion over multiple ranked lists.

    k=15 is used instead of the web-corpus default (60) because our corpus is
    ~100 documents — smaller k makes rank differences more discriminative.

    max_per_report caps how many chunks from any single ntsb_no enter the
    fused pool, preventing high-volume reports from flooding the reranker input.

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
    per_report_counts: dict[str, int] = {}
    for doc_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        doc = docs[doc_id]
        ntsb_no = doc.get("ntsb_no", "")
        if ntsb_no:
            if per_report_counts.get(ntsb_no, 0) >= max_per_report:
                continue
            per_report_counts[ntsb_no] = per_report_counts.get(ntsb_no, 0) + 1
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


_MULTI_EVENT_PATTERNS = [
    r"\bcompare\b",
    r"\bversus\b",
    r"\bvs\.?\b",
    r"\bdifference between\b",
    r"\bacross\b",
    r"\bcommon\b",
    r"\bhow often\b",
    r"\blist\b",
    r"\bgenerally\b",
    r"\btrend\b",
]


def _is_single_event_query(query):
    """Heuristic: queries with comparison/frequency/trend signals are multi-event."""
    q = (query or "").lower()
    return not any(re.search(p, q) for p in _MULTI_EVENT_PATTERNS)


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




def rerank(query, candidates, reranker, top_k=10, min_unique_reports=3):
    """Rerank candidates using a cross-encoder and return top-k.

    For single-event queries the CE reranker is the sole authority on relevance:
    - dominant report is detected via MAX CE score (not sum) so one high-scoring
      chunk beats many mediocre ones from a larger report.
    - min_unique_reports is reduced to 1 so diversity logic never injects
      irrelevant reports when the CE already found the right one.
    - _soft_report_budget is NOT applied after selection; the CE ranking stands.
    """
    if not candidates:
        return []
    pairs = [[query, c["text"]] for c in candidates]
    ce_scores = reranker.predict(pairs)
    for i, score in enumerate(ce_scores):
        candidates[i]["score"] = float(score)

    # Use MAX CE score per report so a single highly-relevant chunk dominates,
    # not a report that happens to have many mediocre chunks in the pool.
    if _is_single_event_query(query):
        report_max_score: dict[str, float] = {}
        for c in candidates:
            rid = c.get("ntsb_no", "")
            if not rid:
                continue
            report_max_score[rid] = max(report_max_score.get(rid, float("-inf")), c["score"])

        dominant_report = max(report_max_score, key=report_max_score.get) if report_max_score else ""
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
    per_report: dict[str, int] = {}
    max_per_report = 2
    for item in ranked:
        # Use a sentinel key for chunks without ntsb_no so they are also capped.
        ntsb_no = item.get("ntsb_no", "") or "__unknown__"
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

    # For single-event queries the CE already identified the right report —
    # do not force diversity by injecting other reports.
    effective_min = 1 if _is_single_event_query(query) else min_unique_reports
    unique_ntsb = {s.get("ntsb_no", "") for s in selected if s.get("ntsb_no", "")}
    if len(unique_ntsb) < effective_min:
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
            if len(selected_ntsb) >= effective_min:
                break

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
Keep the response to 3–4 sentences. Do not invent specific report numbers, tail numbers, or airline names — only include them if you are certain they are accurate.

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
