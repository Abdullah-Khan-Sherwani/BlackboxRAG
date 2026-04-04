"""
Fast param sweep: tests stacking multi_query with hyde/llm_knowledge,
and varies TOP_K. No answer generation — retrieval scores only.

Combos tested:
  - baseline (no augmentation)
  - multi_query
  - hyde
  - llm_knowledge
  - multi_query + hyde
  - multi_query + llm_knowledge
  - multi_query + hyde + llm_knowledge

TOP_K sweep (on best stacking combo): [5, 10, 15, 20]
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.retrieval.query import load_model, init_pinecone, retrieve
from src.retrieval.hybrid import (
    build_bm25_index, bm25_retrieve, rrf_fuse_lists,
    enrich_with_neighbors, generate_knowledge_doc,
    generate_hyde_documents, generate_multi_queries,
)

# ── Questions (in-corpus only for meaningful score comparison) ────────────────
QUESTIONS = [
    {
        "id": "Q2",
        "question": "Based strictly on the NTSB Aircraft Accident Report AAR-14/01 for Asiana Airlines Flight 214, detail the survival and evacuation factors: When the tail section of the aircraft struck the seawall and separated from the fuselage, two flight attendants seated in the aft cabin were ejected onto the runway and survived. What were the exact jumpseat designators for these two flight attendants?",
        "target_report": "NTSB/AAR-14/01",
    },
    {
        "id": "Q10",
        "question": "Based strictly on the NTSB Aircraft Accident Report AAR-00/02 for Federal Express Flight 14, according to the ARFF fire crew chief, exactly how many minutes elapsed from the Condition One alarm until five ARFF vehicles were actively engaged in fire suppression at the accident site?",
        "target_report": "NTSB/AAR-00/02",
    },
]

STRATEGY   = "md_recursive"
LLM_MODEL  = "gpt"
SEM_TOP_K  = 60
BM25_TOP_K = 60

# Stacking combos to test
COMBOS = [
    {"name": "baseline",               "mq": False, "hyde": False, "llm_k": False},
    {"name": "multi_query",            "mq": True,  "hyde": False, "llm_k": False},
    {"name": "hyde",                   "mq": False, "hyde": True,  "llm_k": False},
    {"name": "llm_knowledge",          "mq": False, "hyde": False, "llm_k": True},
    {"name": "mq+hyde",                "mq": True,  "hyde": True,  "llm_k": False},
    {"name": "mq+llm_k",               "mq": True,  "hyde": False, "llm_k": True},
    {"name": "mq+hyde+llm_k",          "mq": True,  "hyde": True,  "llm_k": True},
]

TOP_K_VALUES = [5, 10, 15, 20]


def rrf_retrieve(query, jina_model, index, bm25, chunks, extra_queries, top_k):
    all_queries = [query] + extra_queries
    ranked_lists = []
    for q in all_queries:
        ranked_lists.append(retrieve(q, STRATEGY, top_k=SEM_TOP_K, model=jina_model, index=index))
        ranked_lists.append(bm25_retrieve(q, bm25, chunks, top_k=BM25_TOP_K))
    fused = rrf_fuse_lists(ranked_lists)
    matches = sorted(fused, key=lambda x: x.get("score", 0), reverse=True)[:top_k]
    return enrich_with_neighbors(matches, chunks, window=2)


def avg_score(matches):
    if not matches:
        return 0.0
    return round(sum(m.get("score", 0) for m in matches) / len(matches), 5)


def target_in_top(matches, target_report):
    return any(target_report in m.get("ntsb_no", "") for m in matches)


def target_rank(matches, target_report):
    """Return 1-based rank of first chunk from target report, or None."""
    for i, m in enumerate(matches, 1):
        if target_report in m.get("ntsb_no", ""):
            return i
    return None


def main():
    print("=" * 70)
    print("PARAM SWEEP: Stacking + TOP_K")
    print("=" * 70)
    print(f"Strategy: {STRATEGY}  |  SEM_TOP_K={SEM_TOP_K}  BM25_TOP_K={BM25_TOP_K}")
    print()

    print("[1/3] Loading embedding model...")
    jina_model = load_model()
    print("[2/3] Connecting to Pinecone...")
    index = init_pinecone()
    print("[3/3] Building BM25 index...")
    bm25, chunks = build_bm25_index(STRATEGY)
    print("Ready.\n")

    # ── Phase 1: Pre-generate augmentation docs for each question ─────────────
    # Do LLM calls once, reuse across TOP_K sweep
    print("Pre-generating augmentation docs for all questions...")
    aug_cache = {}  # qid → {mq: [...], hyde: [...], llm_k: KnowledgeResult}
    for qdata in QUESTIONS:
        qid   = qdata["id"]
        query = qdata["question"]
        print(f"  [{qid}] multi_query...", end="", flush=True)
        mq_variants = generate_multi_queries(query, model=LLM_MODEL)
        print(f" {len(mq_variants)} variants", end=" | ", flush=True)

        print("hyde...", end="", flush=True)
        hyde_docs = generate_hyde_documents(query, num_docs=2, llm_provider=LLM_MODEL)
        print(f" {len(hyde_docs)} docs", end=" | ", flush=True)

        print("llm_knowledge...", end="", flush=True)
        llm_k_result = generate_knowledge_doc(query, llm_provider=LLM_MODEL)
        llm_k_queries = [llm_k_result.narrative] if llm_k_result.narrative else []
        print(f" conf={llm_k_result.confidence}")

        aug_cache[qid] = {
            "mq":    mq_variants,
            "hyde":  hyde_docs,
            "llm_k": llm_k_queries,
        }
    print()

    # ── Phase 2: Stacking combos at TOP_K=10 ─────────────────────────────────
    print("=" * 70)
    print("PHASE 1 — Stacking combos (TOP_K=10)")
    print("=" * 70)
    rows_combo = []
    TOP_K_FIXED = 10

    for combo in COMBOS:
        scores = []
        hits   = []
        ranks  = []
        for qdata in QUESTIONS:
            qid   = qdata["id"]
            query = qdata["question"]
            cache = aug_cache[qid]

            extra = []
            if combo["mq"]:
                extra += cache["mq"]
            if combo["hyde"]:
                extra += cache["hyde"]
            if combo["llm_k"]:
                extra += cache["llm_k"]

            matches = rrf_retrieve(query, jina_model, index, bm25, chunks, extra, TOP_K_FIXED)
            scores.append(avg_score(matches))
            hits.append(target_in_top(matches, qdata["target_report"]))
            r = target_rank(matches, qdata["target_report"])
            ranks.append(r if r else 99)

        avg_s = round(sum(scores) / len(scores), 5)
        hit_rate = sum(hits) / len(hits)
        avg_rank = round(sum(ranks) / len(ranks), 1)
        print(f"  {combo['name']:<25}  avg_score={avg_s:.5f}  hit_rate={hit_rate:.0%}  avg_rank={avg_rank}")
        rows_combo.append({
            "combo": combo["name"], "top_k": TOP_K_FIXED,
            "avg_score": avg_s, "hit_rate": hit_rate, "avg_rank": avg_rank,
        })

    # Find best combo
    best_combo_name = max(rows_combo, key=lambda r: r["avg_score"])["combo"]
    best_combo = next(c for c in COMBOS if c["name"] == best_combo_name)
    print(f"\n  Best combo: {best_combo_name}")

    # ── Phase 3: TOP_K sweep on best combo ────────────────────────────────────
    print()
    print("=" * 70)
    print(f"PHASE 2 — TOP_K sweep using best combo: {best_combo_name}")
    print("=" * 70)
    rows_topk = []

    for top_k in TOP_K_VALUES:
        scores = []
        hits   = []
        ranks  = []
        for qdata in QUESTIONS:
            qid   = qdata["id"]
            query = qdata["question"]
            cache = aug_cache[qid]

            extra = []
            if best_combo["mq"]:
                extra += cache["mq"]
            if best_combo["hyde"]:
                extra += cache["hyde"]
            if best_combo["llm_k"]:
                extra += cache["llm_k"]

            matches = rrf_retrieve(query, jina_model, index, bm25, chunks, extra, top_k)
            scores.append(avg_score(matches))
            hits.append(target_in_top(matches, qdata["target_report"]))
            r = target_rank(matches, qdata["target_report"])
            ranks.append(r if r else 99)

        avg_s = round(sum(scores) / len(scores), 5)
        hit_rate = sum(hits) / len(hits)
        avg_rank = round(sum(ranks) / len(ranks), 1)
        print(f"  top_k={top_k:<4}  avg_score={avg_s:.5f}  hit_rate={hit_rate:.0%}  avg_rank={avg_rank}")
        rows_topk.append({
            "combo": best_combo_name, "top_k": top_k,
            "avg_score": avg_s, "hit_rate": hit_rate, "avg_rank": avg_rank,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    best_topk_row = max(rows_topk, key=lambda r: r["avg_score"])
    print(f"  Best combo : {best_combo_name}")
    print(f"  Best TOP_K : {best_topk_row['top_k']}")
    print(f"  Avg score  : {best_topk_row['avg_score']}")
    print(f"  Hit rate   : {best_topk_row['hit_rate']:.0%}")

    # Save
    all_rows = rows_combo + [r for r in rows_topk if r["top_k"] != TOP_K_FIXED]
    df = pd.DataFrame(all_rows)
    out = os.path.join(os.path.dirname(__file__), "data", "param_sweep_results.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
