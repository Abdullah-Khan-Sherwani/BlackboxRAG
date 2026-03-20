"""
Hybrid retrieval: BM25 + semantic search with RRF fusion and cross-encoder reranking.
"""
import json
import os

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from src.retrieval.query import load_model, init_pinecone, retrieve

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STRATEGIES = ["fixed", "recursive", "semantic"]
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


def rrf_fuse(semantic_results, bm25_results, k=60):
    """Reciprocal Rank Fusion of two ranked lists.

    Returns fused list sorted by RRF score (descending).
    """
    scores = {}
    docs = {}

    for rank, item in enumerate(semantic_results):
        doc_id = _get_id(item)
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
        docs[doc_id] = _to_dict(item)

    for rank, item in enumerate(bm25_results):
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


def rerank(query, candidates, reranker, top_k=5):
    """Rerank candidates using a cross-encoder and return top-k."""
    if not candidates:
        return []
    pairs = [[query, c["text"]] for c in candidates]
    ce_scores = reranker.predict(pairs)
    for i, score in enumerate(ce_scores):
        candidates[i]["score"] = float(score)
    ranked = sorted(candidates, key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]


def hybrid_retrieve(query, strategy, top_k=5, model=None, index=None,
                    bm25=None, chunks=None, reranker=None,
                    semantic_top_k=20, bm25_top_k=20):
    """Full hybrid retrieval pipeline: semantic + BM25 → RRF → rerank.

    Returns list[dict] with keys: text, ntsb_no, event_date, make, model, score, etc.
    """
    # Semantic retrieval from Pinecone
    semantic_results = retrieve(query, strategy, top_k=semantic_top_k, model=model, index=index)

    # BM25 retrieval
    bm25_results = bm25_retrieve(query, bm25, chunks, top_k=bm25_top_k)

    # Fuse
    fused = rrf_fuse(semantic_results, bm25_results)

    # Rerank
    results = rerank(query, fused, reranker, top_k=top_k)
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

    strategy = "recursive"
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
