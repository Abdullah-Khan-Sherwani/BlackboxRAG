"""
Full evaluation: all MANUAL_COMPARE_QA questions + 3 general questions.
Tests 4 combos x 2 strategies = 8 configurations per question.
- Aug docs pre-generated in parallel (all questions at once)
- Answer generation parallelised in batches to avoid rate limits
- Results saved to data/full_eval_results.csv for manual review
"""
import os
import sys
import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.retrieval.query import load_model, init_pinecone, retrieve
from src.retrieval.hybrid import (
    build_bm25_index, bm25_retrieve, rrf_fuse_lists,
    enrich_with_neighbors, generate_hyde_documents, generate_multi_queries,
)
from src.generation.generate import generate_answer
from src.evaluation.evaluate import MANUAL_COMPARE_QA

# ── Question set ──────────────────────────────────────────────────────────────
GENERAL_QUESTIONS = [
    {
        "id": "G1",
        "question": "What were the probable causes of the crash of Korean Air Flight 801 near Guam?",
        "reference_answer": None,
    },
    {
        "id": "G2",
        "question": "What role did crew resource management play in the crash of American Airlines Flight 1420?",
        "reference_answer": None,
    },
    {
        "id": "G3",
        "question": "What were the main findings regarding the TWA Flight 800 fuel tank explosion?",
        "reference_answer": None,
    },
]

ALL_QUESTIONS = [
    {"id": q["id"], "question": q["question"], "reference_answer": q["reference_answer"]}
    for q in MANUAL_COMPARE_QA
] + GENERAL_QUESTIONS

STRATEGIES = ["md_recursive", "section"]
LLM_MODEL  = "gpt"
TOP_K      = 10
SEM_TOP_K  = 60
BM25_TOP_K = 60
GEN_WORKERS = 6   # parallel answer generation (stay under rate limit)

COMBOS = [
    {"name": "baseline",    "mq": False, "hyde": False},
    {"name": "multi_query", "mq": True,  "hyde": False},
    {"name": "hyde",        "mq": False, "hyde": True},
    {"name": "mq+hyde",     "mq": True,  "hyde": True},
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def rrf_retrieve(query, strategy, jina_model, index, bm25, chunks, extra_queries):
    all_q = [query] + extra_queries
    lists = []
    for q in all_q:
        lists.append(retrieve(q, strategy, top_k=SEM_TOP_K, model=jina_model, index=index))
        lists.append(bm25_retrieve(q, bm25, chunks, top_k=BM25_TOP_K))
    fused = rrf_fuse_lists(lists)
    matches = sorted(fused, key=lambda x: x.get("score", 0), reverse=True)[:TOP_K]
    return enrich_with_neighbors(matches, chunks, window=2)


def safe_answer(text):
    return text.encode("ascii", errors="replace").decode("ascii")


# ── Parallel aug pre-generation ───────────────────────────────────────────────

def _gen_mq(qdata):
    return qdata["id"], "mq", generate_multi_queries(qdata["question"], model=LLM_MODEL)

def _gen_hyde(qdata):
    docs = generate_hyde_documents(qdata["question"], num_docs=2, llm_provider=LLM_MODEL)
    return qdata["id"], "hyde", docs or []

def pregenerate(questions):
    cache = {q["id"]: {"mq": [], "hyde": []} for q in questions}
    tasks = [(fn, q) for q in questions for fn in (_gen_mq, _gen_hyde)]
    print(f"  Firing {len(tasks)} aug API calls in parallel...")
    with ThreadPoolExecutor(max_workers=len(tasks)) as ex:
        futures = {ex.submit(fn, q): q["id"] for fn, q in tasks}
        for fut in as_completed(futures):
            try:
                qid, aug_type, data = fut.result()
                cache[qid][aug_type] = data
                print(f"  [{qid}] {aug_type} done ({len(data)} items)")
            except Exception as e:
                qid = futures[fut]
                print(f"  [{qid}] aug FAILED: {e}")
    return cache


# ── Single generation job (used in thread pool) ───────────────────────────────

def _run_job(job):
    """job = dict with all inputs needed to retrieve + generate one answer."""
    query    = job["question"]
    strategy = job["strategy"]
    combo    = job["combo"]
    jina_model = job["jina_model"]
    index      = job["index"]
    bm25, chunks = job["bm25_chunks"]
    aug_cache  = job["aug_cache"]

    extra = []
    if combo["mq"]:
        extra += aug_cache[job["id"]]["mq"]
    if combo["hyde"]:
        extra += aug_cache[job["id"]]["hyde"]

    matches = rrf_retrieve(query, strategy, jina_model, index, bm25, chunks, extra)
    reports = sorted({m.get("ntsb_no", m.get("report_id", "")) for m in matches if m.get("ntsb_no") or m.get("report_id")})

    try:
        answer = generate_answer(query, matches, llm_provider=LLM_MODEL)
    except Exception as e:
        answer = f"[ERROR: {e}]"

    return {
        "question_id":  job["id"],
        "strategy":     strategy,
        "combo":        combo["name"],
        "reports":      str(reports),
        "answer":       answer,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("FULL EVALUATION  (manual review)")
    print(f"Questions: {len(ALL_QUESTIONS)}  |  Combos: {len(COMBOS)}  |  Strategies: {len(STRATEGIES)}")
    total = len(ALL_QUESTIONS) * len(COMBOS) * len(STRATEGIES)
    print(f"Total answer generations: {total}")
    print("=" * 72)

    print("\n[1/4] Loading embedding model...")
    jina_model = load_model()
    print("[2/4] Connecting to Pinecone...")
    index = init_pinecone()

    print("\n[3/4] Pre-generating aug docs (parallel)...")
    aug_cache = pregenerate(ALL_QUESTIONS)

    print("\n[4/4] Building BM25 indexes...")
    bm25_indexes = {}
    for s in STRATEGIES:
        bm25_indexes[s] = build_bm25_index(s)
        print(f"  {s}: {len(bm25_indexes[s][1])} chunks")

    # ── Build job list ────────────────────────────────────────────────────────
    jobs = []
    for qdata in ALL_QUESTIONS:
        for strategy in STRATEGIES:
            for combo in COMBOS:
                jobs.append({
                    "id":         qdata["id"],
                    "question":   qdata["question"],
                    "strategy":   strategy,
                    "combo":      combo,
                    "jina_model": jina_model,
                    "index":      index,
                    "bm25_chunks": bm25_indexes[strategy],
                    "aug_cache":  aug_cache,
                })

    # ── Run generation in parallel batches ───────────────────────────────────
    results = []
    done = 0
    print(f"\nGenerating {len(jobs)} answers ({GEN_WORKERS} in parallel)...")

    out_path = os.path.join(os.path.dirname(__file__), "data", "full_eval_results.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fieldnames = ["question_id", "strategy", "combo", "reports", "answer"]
    with open(out_path, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()

        with ThreadPoolExecutor(max_workers=GEN_WORKERS) as ex:
            futures = {ex.submit(_run_job, job): job for job in jobs}
            for fut in as_completed(futures):
                done += 1
                job = futures[fut]
                try:
                    row = fut.result()
                    writer.writerow(row)
                    csvf.flush()
                    ans_preview = safe_answer(row["answer"])[:80].replace("\n", " ")
                    print(f"  [{done:>3}/{len(jobs)}] [{row['question_id']}] [{row['strategy']}] [{row['combo']}]  {ans_preview}...")
                except Exception as e:
                    print(f"  [{done:>3}/{len(jobs)}] [{job['id']}] [{job['strategy']}] [{job['combo']['name']}]  ERROR: {e}")

    print(f"\nAll done. Results saved to {out_path}")
    print("Run review_eval.py (or ask Claude) to assess the answers.")


if __name__ == "__main__":
    main()
