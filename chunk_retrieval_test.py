"""
Chunk retrieval test — no answer generation.
Shows retrieved chunks so you can manually verify if the answer is present.

- Both strategies: md_recursive, section
- All stacking combos: baseline, mq, hyde, llm_k, mq+hyde, mq+llm_k, mq+hyde+llm_k
- Augmentation docs pre-generated in PARALLEL (concurrent API calls)
- Known answer chunks flagged with [ANSWER]
"""
import os
import sys
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.retrieval.query import load_model, init_pinecone, retrieve
from src.retrieval.hybrid import (
    build_bm25_index, bm25_retrieve, rrf_fuse_lists,
    enrich_with_neighbors, generate_knowledge_doc,
    generate_hyde_documents, generate_multi_queries,
)

# ── Questions + known answer chunk IDs ───────────────────────────────────────
QUESTIONS = [
    {
        "id": "Q2",
        "question": (
            "Based strictly on the NTSB Aircraft Accident Report AAR-14/01 for Asiana Airlines "
            "Flight 214, detail the survival and evacuation factors: When the tail section of the "
            "aircraft struck the seawall and separated from the fuselage, two flight attendants "
            "seated in the aft cabin were ejected onto the runway and survived. What were the exact "
            "jumpseat designators for these two flight attendants?"
        ),
        "target_report": "AAR-14/01",
        # Chunks confirmed to contain the answer
        "answer_chunks": {
            "AAR1401_mdrec_107_000",  # R4/L4 ejected with galley structure
            "AAR1401_mdrec_107_001",  # blood stains, FA R4/L4 remained seated
            "AAR1401_mdrec_107_004",  # M4A/M4B found on runway
            "AAR1401_mdrec_044_004",  # all four designators listed
        },
        "answer_keywords": ["L4", "R4", "M4A", "M4B", "jumpseat", "ejected"],
    },
    {
        "id": "Q10",
        "question": (
            "Based strictly on the NTSB Aircraft Accident Report AAR-00/02 for Federal Express "
            "Flight 14, according to the ARFF fire crew chief, exactly how many minutes elapsed "
            "from the Condition One alarm until five ARFF vehicles were actively engaged in fire "
            "suppression at the accident site?"
        ),
        "target_report": "AAR-00/02",
        "answer_chunks": set(),  # will flag by keyword instead
        "answer_keywords": ["3 minutes", "five ARFF", "Condition One", "fire suppression", "crew chief"],
    },
]

STRATEGIES  = ["md_recursive", "section"]
LLM_MODEL   = "gpt"
TOP_K       = 10
SEM_TOP_K   = 60
BM25_TOP_K  = 60

COMBOS = [
    {"name": "baseline",      "mq": False, "hyde": False, "llm_k": False},
    {"name": "multi_query",   "mq": True,  "hyde": False, "llm_k": False},
    {"name": "hyde",          "mq": False, "hyde": True,  "llm_k": False},
    {"name": "llm_knowledge", "mq": False, "hyde": False, "llm_k": True},
    {"name": "mq+hyde",       "mq": True,  "hyde": True,  "llm_k": False},
    {"name": "mq+llm_k",      "mq": True,  "hyde": False, "llm_k": True},
    {"name": "mq+hyde+llm_k", "mq": True,  "hyde": True,  "llm_k": True},
]


# ── Parallel aug generation ───────────────────────────────────────────────────

def _gen_mq(qdata):
    variants = generate_multi_queries(qdata["question"], model=LLM_MODEL)
    return qdata["id"], "mq", variants

def _gen_hyde(qdata):
    docs = generate_hyde_documents(qdata["question"], num_docs=2, llm_provider=LLM_MODEL)
    return qdata["id"], "hyde", docs or []

def _gen_llmk(qdata):
    result = generate_knowledge_doc(qdata["question"], llm_provider=LLM_MODEL)
    queries = [result.narrative] if result.narrative else []
    return qdata["id"], "llm_k", queries, result.confidence

def pregenerate_parallel(questions):
    """Fire all aug generation calls concurrently. Returns dict[qid][aug_type]."""
    cache = {q["id"]: {"mq": [], "hyde": [], "llm_k": []} for q in questions}
    tasks = []
    for q in questions:
        tasks.append((_gen_mq,   q))
        tasks.append((_gen_hyde, q))
        tasks.append((_gen_llmk, q))

    print(f"  Firing {len(tasks)} API calls in parallel...")
    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        futures = {executor.submit(fn, arg): (fn.__name__, arg["id"]) for fn, arg in tasks}
        for future in as_completed(futures):
            fn_name, qid = futures[future]
            try:
                result = future.result()
                aug_type = result[1]
                data     = result[2]
                conf     = result[3] if len(result) > 3 else ""
                cache[qid][aug_type] = data
                note = f"conf={conf}" if conf else f"{len(data)} items"
                print(f"  [{qid}] {aug_type} done ({note})")
            except Exception as e:
                print(f"  [{qid}] {fn_name} FAILED: {e}")
    return cache


# ── Retrieval helpers ─────────────────────────────────────────────────────────

def rrf_retrieve(query, jina_model, index, bm25, chunks, extra_queries):
    all_queries = [query] + extra_queries
    ranked_lists = []
    for q in all_queries:
        ranked_lists.append(retrieve(q, "md_recursive", top_k=SEM_TOP_K, model=jina_model, index=index))
        ranked_lists.append(bm25_retrieve(q, bm25, chunks, top_k=BM25_TOP_K))
    fused = rrf_fuse_lists(ranked_lists)
    matches = sorted(fused, key=lambda x: x.get("score", 0), reverse=True)[:TOP_K]
    return enrich_with_neighbors(matches, chunks, window=2)


def rrf_retrieve_strategy(query, strategy, jina_model, index, bm25, chunks, extra_queries):
    all_queries = [query] + extra_queries
    ranked_lists = []
    for q in all_queries:
        ranked_lists.append(retrieve(q, strategy, top_k=SEM_TOP_K, model=jina_model, index=index))
        ranked_lists.append(bm25_retrieve(q, bm25, chunks, top_k=BM25_TOP_K))
    fused = rrf_fuse_lists(ranked_lists)
    matches = sorted(fused, key=lambda x: x.get("score", 0), reverse=True)[:TOP_K]
    return enrich_with_neighbors(matches, chunks, window=2)


def chunk_has_answer(chunk, answer_chunks, answer_keywords):
    cid  = chunk.get("chunk_id", "")
    text = chunk.get("text", "")
    if cid in answer_chunks:
        return True
    return any(kw.lower() in text.lower() for kw in answer_keywords)


def print_chunks(matches, answer_chunks, answer_keywords):
    for i, m in enumerate(matches, 1):
        cid     = m.get("chunk_id", m.get("id", "?"))
        ntsb    = m.get("ntsb_no", m.get("report_id", "?"))
        section = m.get("section_title", "")
        score   = round(m.get("score", 0), 5)
        text    = (m.get("contextualized_text") or m.get("text", ""))[:220].replace("\n", " ")
        flag    = " [ANSWER]" if chunk_has_answer(m, answer_chunks, answer_keywords) else ""
        print(f"    {i:>2}. [{ntsb}] {cid} (score={score}){flag}")
        if section:
            print(f"        section: {section}")
        print(f"        {text}...")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("CHUNK RETRIEVAL TEST  (no answer generation)")
    print("=" * 72)

    print("\n[1/4] Loading embedding model...")
    jina_model = load_model()
    print("[2/4] Connecting to Pinecone...")
    index = init_pinecone()

    print("[3/4] Pre-generating augmentation docs (parallel)...")
    aug_cache = pregenerate_parallel(QUESTIONS)

    print("\n[4/4] Building BM25 indexes...")
    bm25_indexes = {}
    for strategy in STRATEGIES:
        bm25_indexes[strategy] = build_bm25_index(strategy)
        print(f"  {strategy}: {len(bm25_indexes[strategy][1])} chunks")

    # ── Run retrieval ─────────────────────────────────────────────────────────
    for qdata in QUESTIONS:
        qid      = qdata["id"]
        query    = qdata["question"]
        cache    = aug_cache[qid]

        print(f"\n\n{'=' * 72}")
        print(f"QUESTION: {qid}")
        print(f"Target report: {qdata['target_report']}")
        print(f"{'=' * 72}")

        for strategy in STRATEGIES:
            bm25, chunks = bm25_indexes[strategy]

            print(f"\n  -- Strategy: {strategy} --")

            for combo in COMBOS:
                extra = []
                if combo["mq"]:
                    extra += cache["mq"]
                if combo["hyde"]:
                    extra += cache["hyde"]
                if combo["llm_k"]:
                    extra += cache["llm_k"]

                matches = rrf_retrieve_strategy(
                    query, strategy, jina_model, index, bm25, chunks, extra
                )
                answer_hits = sum(
                    1 for m in matches
                    if chunk_has_answer(m, qdata["answer_chunks"], qdata["answer_keywords"])
                )

                print(f"\n  [{combo['name']}]  answer chunks in top-{TOP_K}: {answer_hits}/{TOP_K}")
                print_chunks(matches, qdata["answer_chunks"], qdata["answer_keywords"])

    print("\n\nDone.")


if __name__ == "__main__":
    main()
