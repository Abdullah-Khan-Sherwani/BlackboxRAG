"""
Fast retrieval method comparison test.
Compares 4 retrieval augmentation modes on 3 questions:
  - Baseline      : BM25 + semantic RRF (no augmentation)
  - + LLM Know.   : + NVIDIA Nemotron knowledge doc
  - + HyDE        : + hypothetical document expansion
  - + Multi-Query : + LLM-generated query variants

Strategy: md_recursive only. No cross-encoder. No faithfulness/relevancy (speed).
Shows answers so quality can be judged manually.

NOTE: All recent commit changes (exec summaries, qnli reranker) are active
because this script imports from src modules directly.
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
from src.generation.generate import generate_answer

QUESTIONS = [
    {
        "id": "Q2",
        "question": "Based strictly on the NTSB Aircraft Accident Report AAR-14/01 for Asiana Airlines Flight 214, detail the survival and evacuation factors: When the tail section of the aircraft struck the seawall and separated from the fuselage, two flight attendants seated in the aft cabin were ejected onto the runway and survived. What were the exact jumpseat designators for these two flight attendants?",
        "expected": "4L and 4R",
    },
    {
        "id": "Q10",
        "question": "Based strictly on the NTSB Aircraft Accident Report AAR-00/02 for Federal Express Flight 14, according to the ARFF fire crew chief, exactly how many minutes elapsed from the Condition One alarm until five ARFF vehicles were actively engaged in fire suppression at the accident site?",
        "expected": "4 minutes",
    },
    {
        "id": "Q_AFG",
        "question": "What were the total flight hours of the first officer in the Afghanistan crash?",
        "expected": "Unknown / not in corpus",
    },
]

STRATEGY   = "md_recursive"
LLM_MODEL  = "gpt"   # used for answer generation, HyDE, multi-query
TOP_K      = 10
SEM_TOP_K  = 60
BM25_TOP_K = 60

MODES = [
    "baseline",
    "llm_knowledge",
    "hyde",
    "multi_query",
]


def rrf_retrieve(query, jina_model, index, bm25, chunks, extra_queries=None):
    """Run BM25+semantic RRF with optional extra query variants."""
    all_queries = [query] + (extra_queries or [])
    ranked_lists = []
    for q in all_queries:
        ranked_lists.append(retrieve(q, STRATEGY, top_k=SEM_TOP_K, model=jina_model, index=index))
        ranked_lists.append(bm25_retrieve(q, bm25, chunks, top_k=BM25_TOP_K))
    fused = rrf_fuse_lists(ranked_lists)
    matches = sorted(fused, key=lambda x: x.get("score", 0), reverse=True)[:TOP_K]
    return enrich_with_neighbors(matches, chunks, window=2)


def reports_in(matches):
    return sorted({m.get("ntsb_no", "") for m in matches if m.get("ntsb_no")})


def avg_score(matches):
    if not matches:
        return 0.0
    return round(sum(m.get("score", 0) for m in matches) / len(matches), 5)


def run():
    print("\n" + "="*80)
    print("RETRIEVAL METHOD COMPARISON TEST")
    print("="*80)
    print(f"Strategy : {STRATEGY}")
    print(f"Retrieval: BM25 + Semantic RRF (NO cross-encoder)")
    print(f"Modes    : {', '.join(MODES)}")
    print(f"Questions: {len(QUESTIONS)}")
    print("="*80)

    print("\n[1/3] Loading embedding model...")
    jina_model = load_model()
    print("[2/3] Connecting to Pinecone...")
    index = init_pinecone()
    print("[3/3] Building BM25 index...")
    bm25, chunks = build_bm25_index(STRATEGY)
    print("Ready.\n")

    rows = []

    for qi, qdata in enumerate(QUESTIONS, 1):
        qid      = qdata["id"]
        query    = qdata["question"]
        expected = qdata["expected"]

        print(f"\n{'='*80}")
        print(f"[Q {qi}/3] {qid}")
        print(f"Expected: {expected}")
        print(f"Question: {query[:90]}...")
        print("="*80)

        for mode in MODES:
            print(f"\n  -- MODE: {mode} --")
            extra_queries = []
            augmentation_note = ""

            if mode == "llm_knowledge":
                print("  Generating LLM knowledge doc...", end="", flush=True)
                result = generate_knowledge_doc(query, llm_provider=LLM_MODEL)
                if result.narrative:
                    extra_queries = [result.narrative]
                    augmentation_note = f"report={result.ntsb_number or 'none'} conf={result.confidence}"
                print(f" done  [{augmentation_note}]")

            elif mode == "hyde":
                print("  Generating HyDE docs...", end="", flush=True)
                hyde_docs = generate_hyde_documents(query, num_docs=2, llm_provider=LLM_MODEL)
                extra_queries = hyde_docs or []
                augmentation_note = f"{len(extra_queries)} hypothetical docs"
                print(f" done  [{augmentation_note}]")

            elif mode == "multi_query":
                print("  Generating multi-query variants...", end="", flush=True)
                variants = generate_multi_queries(query, model=LLM_MODEL)
                extra_queries = variants or []
                augmentation_note = f"{len(extra_queries)} variants"
                print(f" done  [{augmentation_note}]")

            # Retrieve
            matches = rrf_retrieve(query, jina_model, index, bm25, chunks, extra_queries)
            retrieved_reports = reports_in(matches)
            score = avg_score(matches)

            # Generate answer
            print("  Generating answer...", end="", flush=True)
            try:
                answer = generate_answer(query, matches, llm_provider=LLM_MODEL)
            except Exception as e:
                answer = f"[Generation error: {e}]"
            print(" done")

            # Print result
            print(f"\n  Reports in top-{TOP_K}: {retrieved_reports}")
            print(f"  Avg score  : {score}")
            print(f"  Answer     :\n")
            safe = answer.encode("ascii", errors="replace").decode("ascii")
            for line in safe.splitlines():
                print(f"    {line}")

            rows.append({
                "question_id"    : qid,
                "mode"           : mode,
                "avg_score"      : score,
                "reports"        : str(retrieved_reports),
                "augmentation"   : augmentation_note,
                "answer_preview" : answer[:300],
            })

    # Summary table
    print("\n\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(
        index="question_id",
        columns="mode",
        values="avg_score",
        aggfunc="first"
    ).round(5)
    print("\nAvg Retrieval Score by Question × Mode:")
    print(pivot.to_string())

    print("\nAvg Score by Mode (across all questions):")
    print(df.groupby("mode")["avg_score"].mean().round(5).sort_values(ascending=False).to_string())

    out = os.path.join(os.path.dirname(__file__), "data", "method_comparison_results.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    run()
