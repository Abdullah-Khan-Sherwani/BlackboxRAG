"""
Hybrid retrieval: BM25 + semantic search with RRF fusion and cross-encoder reranking.
"""
import json
import os
import re

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from src.llm.client import call_eval_llm
from src.retrieval.query import load_model, init_pinecone, retrieve, available_strategies

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STRATEGIES = ["fixed", "recursive", "semantic", "parent"]
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def build_bm25_index(strategy):
    """Load chunks for a strategy and build a BM25 index.

    Returns (BM25Okapi, list[dict]) — the index and the original chunk dicts.
    """
    path = os.path.join(BASE_DIR, "data", "processed", f"chunks_{strategy}.json")
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
            if doc_id not in docs:
                docs[doc_id] = _to_dict(item)

    fused = []
    for doc_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        doc = docs[doc_id]
        doc["score"] = score
        fused.append(doc)

    return fused


def load_reranker():
    """Load the cross-encoder reranking model."""
    return CrossEncoder(RERANKER_MODEL)


def rerank(query, candidates, reranker, top_k=6, min_unique_reports=3):
    """Rerank candidates using a cross-encoder and return top-k.

    Adds a mild diversity constraint so one report does not monopolize results.
    """
    if not candidates:
        return []
    pairs = [[query, c["text"]] for c in candidates]
    ce_scores = reranker.predict(pairs)
    for i, score in enumerate(ce_scores):
        candidates[i]["score"] = float(score)
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


def generate_hyde_document(query):
    """Generate a short hypothetical answer-like document for HyDE retrieval."""
    prompt = f"""Write a concise technical paragraph that would likely answer this aviation safety question.
Do not include meta commentary.

Question:
{query}
"""
    try:
        return call_eval_llm(prompt)
    except Exception:
        return ""


def _chunk_maps(chunks):
    """Build quick lookup maps for neighbor context enrichment."""
    by_doc = {}
    for c in chunks:
        ntsb = c.get("ntsb_no", "")
        if ntsb:
            by_doc.setdefault(ntsb, []).append(c)

    for ntsb, doc_chunks in by_doc.items():
        doc_chunks.sort(key=lambda x: x.get("chunk_id", ""))
        by_doc[ntsb] = doc_chunks

    return by_doc


def enrich_with_neighbors(results, chunks, window=1):
    """Attach neighboring chunk text to each result for better grounding."""
    if not results or window <= 0:
        return results

    by_doc = _chunk_maps(chunks)
    enriched = []

    for r in results:
        cid = r.get("chunk_id", "")
        ntsb = r.get("ntsb_no", "")
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


def hybrid_retrieve(query, strategy, top_k=6, model=None, index=None,
                    bm25=None, chunks=None, reranker=None,
                    semantic_top_k=40, bm25_top_k=40,
                    enable_query_expansion=True,
                    neighbor_window=1,
                    use_hyde=False,
                    min_unique_reports=3):
    """Full hybrid retrieval pipeline: semantic + BM25 → RRF → rerank.

    Returns list[dict] with keys: text, ntsb_no, event_date, make, model, score, etc.
    """
    # Query expansion + fusion retrieval
    queries = expand_query_variants(query) if enable_query_expansion else [query]
    if use_hyde:
        hyde_doc = generate_hyde_document(query)
        if hyde_doc:
            queries.append(hyde_doc)

    ranked_lists = []
    for q in queries:
        semantic_results = retrieve(q, strategy, top_k=semantic_top_k, model=model, index=index)
        bm25_results = bm25_retrieve(q, bm25, chunks, top_k=bm25_top_k)
        ranked_lists.append(semantic_results)
        ranked_lists.append(bm25_results)

    # Fuse all ranked lists (semantic + lexical over expanded queries)
    fused = rrf_fuse_lists(ranked_lists)

    # Rerank
    results = rerank(query, fused, reranker, top_k=top_k, min_unique_reports=min_unique_reports)
    results = enrich_with_neighbors(results, chunks, window=neighbor_window)
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
