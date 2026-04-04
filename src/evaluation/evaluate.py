"""
Evaluation module with Faithfulness and Relevancy metrics.

Faithfulness: claim extraction → verification → % supported by context.
Relevancy: alternate query generation → cosine similarity with original query.

Uses GPT (openai/gpt-oss-120b via NVIDIA API) for both generation and LLM-as-judge.
Runs in parallel batches of 10 jobs (10 × 3 LLM calls = 30 RPM, 62s window).
"""
import json
import os
import re
import sys
import hashlib
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.llm.client import call_llm, call_eval_llm, MODEL_GPT
from src.retrieval.query import load_model, init_pinecone, retrieve, available_strategies
from src.retrieval.hybrid import build_bm25_index, load_reranker, hybrid_retrieve, bm25_retrieve
from src.generation.generate import generate_answer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_PATH = os.path.join(BASE_DIR, "data", "processed", "eval_llm_cache.json")

# LLM used for both answer generation and LLM-as-judge evaluation
EVAL_MODEL = MODEL_GPT

# Batch processing constants (30 RPM rate limit → 10 jobs × 3 LLM calls = 30/batch)
EVAL_BATCH_SIZE = 10
EVAL_WORKERS = 10
BATCH_WINDOW_S = 62  # seconds; sleep out the remainder after each batch

# Thread-safety locks
_cache_lock = threading.Lock()
_csv_lock = threading.Lock()

# Fixed test set: Q2, Q3, Q4, Q6, Q9 from MANUAL_COMPARE_QA (specific NTSB queries
# with known reference answers) + 7 general aviation queries = 12 total (within 10-20)
EVAL_QUERIES = [
    # Specific NTSB queries (Q2, Q3, Q4, Q6, Q9)
    "Based strictly on AAR-14/01 for Asiana Airlines Flight 214, what were the exact jumpseat designators for the two flight attendants ejected onto the runway?",
    "Based strictly on AAR-14/01 for Asiana Airlines Flight 214, what substance visually obscured the fatally injured passenger from ARFF drivers?",
    "Based strictly on AAR-14/01 for Asiana Airlines Flight 214, how many flight hours of experience did the PM have as Instructor Pilot in the Boeing 777 prior to the accident?",
    "Based strictly on AAR-00/03 for TWA Flight 800, which debris field was the smallest and what were the exact fuselage station markers for the wreckage it contained?",
    "Based strictly on AAR-00/01 for Korean Air Flight 801, what was the exact diameter of the severed fuel oil pipeline and approximately how many gallons of oil were spilled?",
    # General aviation queries (G1-G7)
    "How do crew coordination failures contribute to commercial aviation accidents?",
    "What role does controlled flight into terrain play in fatal airline accidents?",
    "How did instrument approach errors contribute to accidents in the NTSB reports?",
    "What maintenance-related factors led to in-flight emergencies in these reports?",
    "How did weather and environmental conditions affect accident outcomes?",
    "What evacuation and survival factors are documented across these accident reports?",
    "How does pilot situational awareness failure manifest in approach and landing accidents?",
]


MANUAL_COMPARE_QA = [
    {
        "id": "Q1",
        "question": "Based strictly on the NTSB Aviation Accident Report AAR-18/01 for American Airlines Flight 383, provide the exact chronological telemetry from the Flight Data Recorder (FDR): Exactly how many seconds elapsed between the start of the right engine failure and the time the spar valve was finally closed to cut off the fuel supply?",
        "reference_answer": "Answer: Exactly 164 seconds elapsed between the start of the right engine failure and the closure of the spar valve.",
    },
    {
        "id": "Q2",
        "question": "Based strictly on the NTSB Aircraft Accident Report AAR-14/01 for Asiana Airlines Flight 214, detail the survival and evacuation factors: When the tail section of the aircraft struck the seawall and separated from the fuselage, two flight attendants seated in the aft cabin were ejected onto the runway and survived. What were the exact jumpseat designators for these two flight attendants?",
        "reference_answer": "The two ejected flight attendants were seated in jumpseats 4L and 4R.",
        
    },
    {
        "id": "Q3",
        "question": "Based strictly on the NTSB Aircraft Accident Report AAR-14/01 for Asiana Airlines Flight 214, one of the fatally injured passengers was located outside the aircraft and was struck by two ARFF vehicles. According to the NTSB, what specific substance visually obscured this passenger from the view of the ARFF drivers?",
        "reference_answer": "The passenger was visually obscured by Aqueous Film-Forming Foam (AFFF), also known as firefighting foam.",
    },
    {
        "id": "Q4",
        "question": "Based strictly on the NTSB Aircraft Accident Report AAR-14/01 for Asiana Airlines Flight 214, exactly how many flight hours of experience did the PM have acting specifically as an Instructor Pilot in the Boeing 777 prior to this accident flight?",
        "reference_answer": "The PM had 0 hours of experience acting specifically as an Instructor Pilot in the Boeing 777 prior to the accident.",
    },
    {
        "id": "Q5",
        "question": "Based strictly on the NTSB Aircraft Accident Report AAR-14/01 for Asiana Airlines Flight 214, exactly how many total flight hours did the PF have in the Boeing 777 prior to the accident flight?",
        "reference_answer": "The PF had a total of 43 hours in the Boeing 777 prior to the accident flight.",
    },
    {
        "id": "Q6",
        "question": "Based strictly on the NTSB Aircraft Accident Report AAR-00/03 for Trans World Airlines (TWA) Flight 800, wreckage was recovered from three debris fields (red, yellow, and green zones). Which field was the smallest, was contained within the red zone on its northeastern side, and what were the exact fuselage station markers for the wreckage it contained?",
        "reference_answer": "The Yellow Zone was the smallest, and it contained wreckage from STA 840 to STA 1000.",
    },
    {
        "id": "Q7",
        "question": "Based strictly on the NTSB Aircraft Accident Report AAR-01/01 for United Airlines Flight 585, the captain had 9,902 total flight hours with United. Exactly how many flight hours and minutes of PIC experience did he have specifically in the Boeing 737-200 prior to the accident flight?",
        "reference_answer": "The captain had 167 hours and 17 minutes of PIC experience specifically in the Boeing 737-200.",
    },
    {
        "id": "Q8",
        "question": "Based strictly on the NTSB Aircraft Accident Report AAR-01/02 for American Airlines Flight 1420, exactly how many feet to the left of the centerline were the main landing gear tire marks located at the end of runway 4R?",
        "reference_answer": "The tire marks were located 14 feet to the left of the runway centerline at the end of the runway surface.",
    },
    {
        "id": "Q9",
        "question": "Based strictly on the NTSB Aircraft Accident Report AAR-00/01 for Korean Air Flight 801, what was the exact diameter of the severed fuel oil pipeline, and approximately how many gallons of oil were spilled?",
        "reference_answer": "The pipeline was 12 inches in diameter, and approximately 1,500 gallons of oil were spilled.",
    },
    {
        "id": "Q10",
        "question": "Based strictly on the NTSB Aircraft Accident Report AAR-00/02 for Federal Express Flight 14, according to the ARFF fire crew chief, exactly how many minutes elapsed from the Condition One alarm until five ARFF vehicles were actively engaged in fire suppression at the accident site?",
        "reference_answer": "Exactly 4 minutes elapsed from the time of the alarm until five ARFF vehicles were actively engaged in fire suppression.",
    },
]


def _parse_json(text):
    """Parse JSON from an LLM response, handling markdown code blocks."""
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(text)


def _normalize_verified_claims(raw):
    """Normalize judge output into list[dict(claim,supported,reasoning)].

    Handles common malformed shapes from LLM outputs:
    - JSON string that itself contains JSON
    - single dict
    - list of strings (claims only)
    - mixed lists
    """
    if raw is None:
        return []

    # If model returned JSON as a string value, parse one more time.
    if isinstance(raw, str):
        try:
            raw = _parse_json(raw)
        except Exception:
            return []

    if isinstance(raw, dict):
        raw = [raw]

    if not isinstance(raw, list):
        return []

    out = []
    for item in raw:
        if isinstance(item, dict):
            claim = str(item.get("claim", "")).strip()
            reasoning = str(item.get("reasoning", "")).strip()
            supported = item.get("supported", False)
            supported = bool(supported) if isinstance(supported, (bool, int)) else str(supported).strip().lower() in {"true", "yes", "1"}
            if claim:
                out.append({"claim": claim, "supported": supported, "reasoning": reasoning})
        elif isinstance(item, str):
            claim = item.strip()
            if claim:
                out.append({"claim": claim, "supported": False, "reasoning": "string_only_claim"})

    return out


def _normalize_alternates(raw, n=3):
    """Normalize alternate-query outputs into a clean list of strings."""
    if raw is None:
        return []

    if isinstance(raw, str):
        try:
            raw = _parse_json(raw)
        except Exception:
            return []

    if isinstance(raw, dict):
        raw = raw.get("alternates", [])

    if not isinstance(raw, list):
        return []

    out = []
    for item in raw:
        if isinstance(item, str):
            s = item.strip()
        elif isinstance(item, dict):
            s = str(item.get("query", item.get("text", ""))).strip()
        else:
            s = str(item).strip()
        if s:
            out.append(s)

    # De-duplicate while preserving order.
    deduped = []
    seen = set()
    for s in out:
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        deduped.append(s)
    return deduped[:n]


def _load_cache(cache_path=CACHE_PATH):
    """Load persistent evaluation cache from disk."""
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}


def _save_cache(cache, cache_path=CACHE_PATH):
    """Persist evaluation cache to disk."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f)


def _cached_llm(prompt, system=None, cache=None):
    """Call LLM (GPT) with prompt-level memoization. Thread-safe via _cache_lock."""
    if cache is None:
        return call_llm(prompt, system=system, model=EVAL_MODEL)

    key_src = json.dumps({"system": system or "", "prompt": prompt}, ensure_ascii=False, sort_keys=True)
    key = hashlib.sha256(key_src.encode("utf-8")).hexdigest()

    with _cache_lock:
        if key in cache:
            return cache[key]

    out = call_llm(prompt, system=system, model=EVAL_MODEL)

    with _cache_lock:
        cache[key] = out
    return out


# ── Faithfulness ──────────────────────────────────────────────────────────────

def extract_and_verify_claims(answer, context_texts, cache=None):
    """Single-call faithfulness judge: extract claims and verify support.

    Returns list of dicts with keys: claim, supported (bool), reasoning.
    """
    context_str = "\n\n".join(context_texts)

    prompt = f"""You are an impartial fact-checker for aviation accident analysis.
Your task is to extract factual claims from the answer and verify each claim against the context.

===Context===
{context_str}

===Answer===
{answer}

===Return Requirements===
1. Return a JSON array only.
2. Each element must have exactly keys: "claim", "supported", "reasoning".
3. "claim": one atomic factual claim extracted from the answer.
4. "supported": true only if directly supported by context, else false.
5. "reasoning": one short sentence citing context support or mismatch.
6. Include all meaningful factual claims from the answer.
7. No text outside JSON.

Example:
[{{"claim": "...", "supported": true, "reasoning": "..."}}]"""

    try:
        raw = _parse_json(_cached_llm(prompt, cache=cache))
    except (json.JSONDecodeError, ValueError):
        return []
    return _normalize_verified_claims(raw)


def compute_faithfulness(answer, context_texts, cache=None):
    """Compute faithfulness score: fraction of claims supported by context.

    Returns (score, details) where score is 0.0-1.0 and details is the
    list of verified claims.
    """
    verified = extract_and_verify_claims(answer, context_texts, cache=cache)
    if not verified:
        return 1.0, []

    supported = sum(1 for v in verified if v.get("supported", False))
    score = supported / len(verified) if verified else 0.0
    return score, verified


# ── Relevancy ─────────────────────────────────────────────────────────────────

def generate_alternate_queries(query, n=3):
    """Generate n alternate phrasings of the query using the LLM (GPT)."""
    prompt = f"""You are an aviation safety research assistant.
Your task is to generate {n} alternative phrasings of the following question about aviation accidents.

===Question===
{query}

===Return Requirements===
1. Each alternative must capture the same information need but use different wording.
2. Maintain aviation domain terminology where appropriate.
3. The output must be a properly formatted JSON array of exactly {n} strings.
4. Do NOT include any additional explanation, commentary, or text outside of the JSON array.
5. The output must include only the JSON array and no additional text before or after it.

Example: ["alt 1", "alt 2", "alt 3"]"""
    try:
        return _parse_json(call_llm(prompt, model=EVAL_MODEL))
    except (json.JSONDecodeError, ValueError):
        return []
def _extract_final_answer(answer):
    """Extract only the Answer section from a structured LLM response.

    The generation prompt enforces an 'Answer:' section at the end.
    Falls back to the full answer if the section is not found.
    """
    match = re.search(r"(?i)answer\s*[:\-]?\s*(.*)", answer, re.DOTALL)
    if match:
        return match.group(1).strip()
    return answer


def compute_relevancy(query, answer, jina_model, cache=None):
    """Compute relevancy by generating 3 questions from the final answer, then
    computing cosine similarity between those questions and the original query.

    Returns (score, alternates) where score is mean cosine similarity (0-1).
    """
    final_answer = _extract_final_answer(answer)

    prompt = f"""You are an aviation safety research assistant.
Given the answer below, generate exactly 3 questions that this answer directly addresses.

===Answer===
{final_answer}

===Return Requirements===
1. Output a JSON array of exactly 3 strings.
2. Each string must be a question that the answer above can answer.
3. Use aviation domain terminology where appropriate.
4. No text outside the JSON array.

Example: ["question 1?", "question 2?", "question 3?"]"""
    try:
        raw_alternates = _parse_json(_cached_llm(prompt, cache=cache))
    except (json.JSONDecodeError, ValueError):
        raw_alternates = []

    alternates = _normalize_alternates(raw_alternates, n=3)

    if not alternates:
        return 0.0, []

    all_texts = [query] + alternates
    embeddings = jina_model.encode(texts=all_texts, task="text-matching")
    embeddings = np.array(embeddings)

    query_emb = embeddings[0]
    alt_embs = embeddings[1:]

    similarities = []
    for alt_emb in alt_embs:
        cos_sim = np.dot(query_emb, alt_emb) / (
            np.linalg.norm(query_emb) * np.linalg.norm(alt_emb) + 1e-10
        )
        similarities.append(float(cos_sim))

    return float(np.mean(similarities)), alternates


# ── Retrieval metrics ─────────────────────────────────────────────────────────

def eval_retrieval(results):
    """Compute retrieval quality metrics from results (Pinecone matches or dicts)."""
    if not results:
        return {"avg_score": 0, "min_score": 0, "max_score": 0, "num_unique_reports": 0}

    scores = []
    ntsb_nos = set()
    for r in results:
        if hasattr(r, "score"):
            scores.append(r.score)
        else:
            scores.append(r.get("score", 0))
        ntsb = r.metadata.get("ntsb_no", "") if hasattr(r, "metadata") else r.get("ntsb_no", "")
        if ntsb:
            ntsb_nos.add(ntsb)

    return {
        "avg_score": sum(scores) / len(scores),
        "min_score": min(scores),
        "max_score": max(scores),
        "num_unique_reports": len(ntsb_nos),
    }


# ── Main evaluation loop ─────────────────────────────────────────────────────

def _load_completed(output_path):
    """Load already-completed (query, strategy, mode) tuples from a CSV."""
    completed = set()
    if os.path.exists(output_path):
        df = pd.read_csv(output_path)
        for _, row in df.iterrows():
            completed.add((row["query"], row["strategy"], row["mode"]))
    return completed


def _append_result(result, output_path):
    """Append a single result dict to the CSV. Thread-safe via _csv_lock."""
    row = {k: v for k, v in result.items() if k not in ("faith_details", "rel_alternates")}
    df = pd.DataFrame([row])
    with _csv_lock:
        write_header = not os.path.exists(output_path)
        df.to_csv(output_path, mode="a", header=write_header, index=False)


def _eval_single_job(args):
    """Execute one (query, strategy, mode) evaluation job.

    Runs retrieve → generate (GPT) → faithfulness → relevancy and records
    retrieval_time, generation_time, and total_time for §3D report metrics.
    Thread-safe: uses module-level _cache_lock for shared cache access.
    """
    query        = args["query"]
    strategy     = args["strategy"]
    mode         = args["mode"]
    jina_model   = args["jina_model"]
    index        = args["index"]
    bm25_cache   = args["bm25_cache"]
    reranker     = args["reranker"]
    top_k        = args["top_k"]
    compute_faith = args["compute_faith"]
    compute_rel  = args["compute_rel"]
    cache        = args["cache"]
    use_hyde     = args["use_hyde"]
    allow_bm25_fallback = args["allow_bm25_fallback"]

    t_total = time.perf_counter()

    # ── Retrieve ────────────────────────────────────────────────────────────────
    t_ret = time.perf_counter()
    try:
        if mode == "hybrid" and bm25_cache and strategy in bm25_cache and reranker:
            bm25, chunks = bm25_cache[strategy]
            matches = hybrid_retrieve(
                query, strategy, top_k=top_k,
                model=jina_model, index=index,
                bm25=bm25, chunks=chunks, reranker=reranker,
                use_hyde=use_hyde,
            )
        else:
            matches = retrieve(query, strategy, top_k=top_k, model=jina_model, index=index)
    except Exception as e:
        if allow_bm25_fallback and bm25_cache and strategy in bm25_cache:
            print(f"  Warning: Pinecone failed [{mode}/{strategy}] -> BM25 fallback. {e}")
            bm25, chunks = bm25_cache[strategy]
            matches = bm25_retrieve(query, bm25, chunks, top_k=top_k)
        else:
            raise
    retrieval_time = round(time.perf_counter() - t_ret, 3)

    context_texts = [
        m.metadata.get("text", "") if hasattr(m, "metadata") else m.get("text", "")
        for m in matches
    ]

    # ── Generate (GPT) ──────────────────────────────────────────────────────────
    t_gen = time.perf_counter()
    try:
        answer = generate_answer(query, matches, llm_provider="gpt")
    except Exception as e:
        print(f"  Warning: generation failed [{mode}/{strategy}] -> placeholder. {e}")
        answer = "Generation unavailable due to LLM connection error."
    generation_time = round(time.perf_counter() - t_gen, 3)
    total_time = round(time.perf_counter() - t_total, 3)

    ret_metrics = eval_retrieval(matches)

    # ── Faithfulness (GPT-as-judge) ──────────────────────────────────────────────
    if compute_faith:
        try:
            faith_score, faith_details = compute_faithfulness(answer, context_texts, cache=cache)
        except Exception as e:
            print(f"  Warning: faithfulness failed [{mode}/{strategy}] -> 0.0. {e}")
            faith_score, faith_details = 0.0, []
    else:
        faith_score, faith_details = 0.0, []

    # ── Relevancy (GPT-as-judge + Jina embeddings) ───────────────────────────────
    if compute_rel:
        try:
            rel_score, rel_alternates = compute_relevancy(query, answer, jina_model, cache=cache)
        except Exception as e:
            print(f"  Warning: relevancy failed [{mode}/{strategy}] -> 0.0. {e}")
            rel_score, rel_alternates = 0.0, []
    else:
        rel_score, rel_alternates = 0.0, []

    return {
        "query":           query,
        "strategy":        strategy,
        "mode":            mode,
        "answer":          answer,
        "num_chunks":      len(matches),
        **ret_metrics,
        "retrieval_time":  retrieval_time,
        "generation_time": generation_time,
        "total_time":      total_time,
        "faithfulness":    round(faith_score, 3),
        "relevancy":       round(rel_score, 3),
        "faith_details":   faith_details,
        "rel_alternates":  rel_alternates,
    }


def run_evaluation(queries, strategies, jina_model, index, mode="semantic",
                   bm25_cache=None, reranker=None, output_path=None,
                   top_k=15, compute_faith=True, compute_rel=True,
                   cache=None, use_hyde=False, allow_bm25_fallback=False,
                   workers=EVAL_WORKERS, batch_size=EVAL_BATCH_SIZE):
    """Run evaluation across all queries and strategies in parallel batches.

    Batches of `batch_size` jobs run with `workers` threads. After each batch
    (except the last) we sleep out the remainder of a 62-second window to stay
    within the 30 RPM rate limit (10 jobs × 3 LLM calls = 30 calls/batch).

    mode: "semantic" or "hybrid"
    output_path: incremental CSV save with resume support.
    Returns list of result dicts.
    """
    completed = _load_completed(output_path) if output_path else set()

    jobs = []
    for query in queries:
        for strategy in strategies:
            if (query, strategy, mode) in completed:
                print(f"  Skipping (done): [{mode}/{strategy}] {query[:50]}...")
                continue
            jobs.append({
                "query": query, "strategy": strategy, "mode": mode,
                "jina_model": jina_model, "index": index,
                "bm25_cache": bm25_cache, "reranker": reranker,
                "top_k": top_k, "compute_faith": compute_faith,
                "compute_rel": compute_rel, "cache": cache,
                "use_hyde": use_hyde, "allow_bm25_fallback": allow_bm25_fallback,
            })

    if not jobs:
        print("  All entries already completed.")
        return []

    total = len(jobs)
    results = []
    done = 0
    n_batches = (total + batch_size - 1) // batch_size

    for batch_idx, batch_start in enumerate(range(0, total, batch_size), 1):
        batch = jobs[batch_start:batch_start + batch_size]
        is_last = batch_idx == n_batches
        t_batch = time.perf_counter()

        print(f"\n  Batch {batch_idx}/{n_batches}: "
              f"jobs {batch_start+1}–{batch_start+len(batch)} of {total}")

        with ThreadPoolExecutor(max_workers=min(workers, len(batch))) as ex:
            future_to_job = {ex.submit(_eval_single_job, job): job for job in batch}
            for fut in as_completed(future_to_job):
                done += 1
                job = future_to_job[fut]
                try:
                    result = fut.result()
                    results.append(result)
                    if output_path:
                        _append_result(result, output_path)
                    print(
                        f"  [{done:>3}/{total}] [{mode}/{result['strategy']}] "
                        f"faith={result['faithfulness']:.2f} rel={result['relevancy']:.2f} "
                        f"ret={result['retrieval_time']:.1f}s gen={result['generation_time']:.1f}s"
                    )
                except Exception as e:
                    print(f"  [{done:>3}/{total}] [{mode}/{job['strategy']}] ERROR: {e}")

        # Persist cache after each batch (not per-result to avoid lock contention)
        if cache is not None:
            _save_cache(cache)

        if not is_last:
            elapsed = time.perf_counter() - t_batch
            wait = max(0.0, BATCH_WINDOW_S - elapsed)
            if wait > 0:
                print(f"  [rate-limit] sleeping {wait:.1f}s before next batch...")
                time.sleep(wait)

    return results


def print_detailed_examples(results, n=3):
    """Print detailed output for n example evaluations."""
    print(f"\n{'='*80}")
    print(f"DETAILED EXAMPLES (first {n})")
    print("=" * 80)

    for r in results[:n]:
        print(f"\nQuery: {r['query']}")
        print(f"Strategy: {r['strategy']} | Mode: {r['mode']}")
        print(f"Answer: {r['answer'][:300]}...")
        print(f"Faithfulness: {r['faithfulness']:.3f}")
        if r["faith_details"]:
            for fd in r["faith_details"][:5]:
                status = "+" if fd.get("supported") else "-"
                print(f"  {status} {fd.get('claim', '')[:100]}")
        print(f"Relevancy: {r['relevancy']:.3f}")
        if r["rel_alternates"]:
            for alt in r["rel_alternates"]:
                print(f"  ~ {alt}")
        print("-" * 80)


def summarize(results):
    """Group results by strategy+mode and compute mean scores including timing (§3D)."""
    df = pd.DataFrame(results)
    cols = [
        "avg_score", "num_unique_reports",
        "faithfulness", "relevancy",
        "retrieval_time", "generation_time", "total_time",
    ]
    existing = [c for c in cols if c in df.columns]
    summary = df.groupby(["mode", "strategy"])[existing].mean().round(3)
    return summary


def run_manual_compare_questions(
    qa_items,
    strategies,
    jina_model,
    index,
    bm25_cache,
    reranker,
    top_k=10,
    modes=("semantic", "hybrid"),
    output_path=None,
    use_hyde=False,
):
    """Run fixed manual QA set and print RAG answer vs reference answer."""
    rows = []

    for mode in modes:
        for strategy in strategies:
            if mode == "hybrid" and strategy not in bm25_cache:
                print(f"Skipping manual compare for [hybrid/{strategy}] because BM25 index is unavailable.")
                continue

            print(f"\n{'='*80}")
            print(f"MANUAL COMPARE | mode={mode} | strategy={strategy}")
            print("=" * 80)

            for item in qa_items:
                qid = item["id"]
                query = item["question"]
                reference_answer = item["reference_answer"]

                try:
                    if mode == "hybrid":
                        bm25, chunks = bm25_cache[strategy]
                        matches = hybrid_retrieve(
                            query,
                            strategy,
                            top_k=top_k,
                            model=jina_model,
                            index=index,
                            bm25=bm25,
                            chunks=chunks,
                            reranker=reranker,
                            use_hyde=use_hyde,
                        )
                    else:
                        matches = retrieve(query, strategy, top_k=top_k, model=jina_model, index=index)
                except Exception as e:
                    matches = []
                    print(f"{qid} retrieval failed [{mode}/{strategy}]: {e}")

                try:
                    rag_answer = generate_answer(query, matches)
                except Exception as e:
                    rag_answer = f"Generation failed: {e}"

                print(f"\n{qid}: {query}")
                print("RAG_ANSWER:")
                print(rag_answer)
                print("REFERENCE_ANSWER:")
                print(reference_answer)
                print("-" * 80)

                rows.append(
                    {
                        "question_id": qid,
                        "question": query,
                        "mode": mode,
                        "strategy": strategy,
                        "rag_answer": rag_answer,
                        "reference_answer": reference_answer,
                        "num_chunks": len(matches),
                    }
                )

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pd.DataFrame(rows).to_csv(output_path, index=False)
        print(f"\nManual comparison results saved to {output_path}")

    return rows


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete existing results and start from scratch")
    parser.add_argument("--max-queries", type=int, default=0,
                        help="Limit number of evaluation queries (0 = all)")
    parser.add_argument("--top-k", type=int, default=15,
                        help="Number of chunks to retrieve per query")
    parser.add_argument("--skip-faithfulness", action="store_true",
                        help="Skip faithfulness scoring for faster iteration")
    parser.add_argument("--skip-relevancy", action="store_true",
                        help="Skip relevancy scoring for faster iteration")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable prompt cache for evaluation LLM calls")
    parser.add_argument("--fast", action="store_true",
                        help="Quick iteration mode: fewer queries/top-k and cache enabled")
    parser.add_argument("--eval-llm-provider", choices=["nvidia", "hf"], default=None,
                        help="Override evaluation LLM provider")
    parser.add_argument("--eval-hf-model", type=str, default=None,
                        help="HF model ID for eval judge when provider=hf")
    parser.add_argument("--use-hyde", action="store_true",
                        help="Enable HyDE in hybrid retrieval (better recall, slower)")
    parser.add_argument("--allow-bm25-fallback", action="store_true",
                        help="If Pinecone retrieval fails, fall back to BM25 so evaluation can continue")
    parser.add_argument("--manual-qa", action="store_true",
                        help="Run fixed 10-question manual comparison set")
    parser.add_argument("--manual-qa-only", action="store_true",
                        help="Run only manual QA compare and skip standard evaluation")
    parser.add_argument("--manual-qa-modes", choices=["semantic", "hybrid", "both"], default="both",
                        help="Retrieval modes for manual QA compare")
    parser.add_argument("--manual-qa-top-k", type=int, default=10,
                        help="Top-k for manual QA compare")
    parser.add_argument("--manual-qa-output", type=str,
                        default=os.path.join(BASE_DIR, "data", "manual_compare_results.csv"),
                        help="CSV path for manual QA compare output")
    args = parser.parse_args()

    if args.eval_llm_provider:
        os.environ["EVAL_LLM_PROVIDER"] = args.eval_llm_provider
    if args.eval_hf_model:
        os.environ["EVAL_HF_MODEL"] = args.eval_hf_model

    output_path = os.path.join(BASE_DIR, "data", "evaluation_results.csv")

    if args.fresh and os.path.exists(output_path):
        os.remove(output_path)
        print("Removed existing results. Starting fresh.\n")
    elif os.path.exists(output_path):
        print(f"Resuming from {output_path} (use --fresh to start over)\n")

    jina_model = load_model()
    index = init_pinecone()
    reranker = load_reranker()

    eval_queries = EVAL_QUERIES
    top_k = args.top_k
    if args.fast:
        eval_queries = EVAL_QUERIES[:5]
        top_k = min(top_k, 3)

    if args.max_queries and args.max_queries > 0:
        eval_queries = eval_queries[:args.max_queries]

    cache = None if args.no_cache else _load_cache()

    strategies = available_strategies()

    # Pre-build BM25 indices
    bm25_cache = {}
    for s in strategies:
        print(f"Building BM25 index for {s}...")
        try:
            bm25_cache[s] = build_bm25_index(s)
        except FileNotFoundError as e:
            print(f"  Skipping BM25 for {s}: {e}")

    semantic_strategies = list(strategies)
    hybrid_strategies = [s for s in strategies if s in bm25_cache]
    if not hybrid_strategies:
        print("Warning: No strategies have BM25 artifacts; hybrid evaluation will be skipped.")

    if args.manual_qa:
        manual_modes = ["semantic", "hybrid"] if args.manual_qa_modes == "both" else [args.manual_qa_modes]
        run_manual_compare_questions(
            qa_items=MANUAL_COMPARE_QA,
            strategies=strategies,
            jina_model=jina_model,
            index=index,
            bm25_cache=bm25_cache,
            reranker=reranker,
            top_k=args.manual_qa_top_k,
            modes=manual_modes,
            output_path=args.manual_qa_output,
            use_hyde=args.use_hyde,
        )
        if args.manual_qa_only:
            print("\nManual QA compare complete.")
            return

    print("\n--- Semantic-only evaluation ---")
    sem_results = run_evaluation(
        eval_queries, semantic_strategies, jina_model, index, mode="semantic",
        bm25_cache=bm25_cache,
        output_path=output_path,
        top_k=top_k,
        compute_faith=not args.skip_faithfulness,
        compute_rel=not args.skip_relevancy,
        cache=cache,
        allow_bm25_fallback=args.allow_bm25_fallback or args.fast,
    )

    print("\n--- Hybrid evaluation ---")
    if hybrid_strategies:
        hyb_results = run_evaluation(
            eval_queries, hybrid_strategies, jina_model, index, mode="hybrid",
            bm25_cache=bm25_cache, reranker=reranker,
            output_path=output_path,
            top_k=top_k,
            compute_faith=not args.skip_faithfulness,
            compute_rel=not args.skip_relevancy,
            cache=cache,
            use_hyde=args.use_hyde,
            allow_bm25_fallback=args.allow_bm25_fallback or args.fast,
        )
    else:
        hyb_results = []

    # Print summary from the full CSV (includes resumed + new results)
    if os.path.exists(output_path):
        df = pd.read_csv(output_path)
        summary = summarize(df.to_dict("records"))
        print(f"\n{'='*80}")
        print("EVALUATION SUMMARY")
        print("=" * 80)
        print(summary.to_string())

    # Detailed examples from this run
    all_results = sem_results + hyb_results
    if all_results:
        print_detailed_examples(all_results, n=3)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
