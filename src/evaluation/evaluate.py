"""
Evaluation module with Faithfulness and Relevancy metrics.

Faithfulness: claim extraction → verification → % supported by context.
Relevancy: alternate query generation → cosine similarity with original query.

Uses DeepSeek V3.2 as LLM-as-judge for both metrics.
"""
import json
import os
import sys
import hashlib

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.llm.client import call_llm, call_eval_llm
from src.retrieval.query import load_model, init_pinecone, retrieve, available_strategies
from src.retrieval.hybrid import build_bm25_index, load_reranker, hybrid_retrieve, bm25_retrieve
from src.generation.generate import generate_answer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_PATH = os.path.join(BASE_DIR, "data", "processed", "eval_llm_cache.json")

EVAL_QUERIES = [
    "What are common causes of engine failure during takeoff?",
    "Cessna accidents in instrument meteorological conditions",
    "How does pilot experience affect landing accident outcomes?",
    "What role does weather play in fatal general aviation accidents?",
    "Describe common factors in runway excursion incidents",
    "What maintenance issues lead to in-flight structural failures?",
    "How do fuel management errors contribute to aviation accidents?",
    "What are the most frequent causes of controlled flight into terrain?",
    "What factors contribute to loss of control during landing?",
    "How does night flying increase accident risk for private pilots?",
    "What are common causes of mid-air collisions in uncontrolled airspace?",
    "Describe the role of fatigue in pilot error accidents",
    "What types of mechanical failures cause forced landings?",
    "How do icing conditions contribute to general aviation crashes?",
    "What are the leading causes of helicopter accidents?",
    "How does inadequate pre-flight inspection contribute to accidents?",
    "What patterns exist in accidents involving student pilots?",
    "How do crosswind conditions affect landing accident rates?",
    "What role does air traffic control play in preventing mid-air collisions?",
    "Describe common factors in accidents during instrument approaches",
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
    """Call LLM with prompt-level memoization for faster repeated evaluations."""
    if cache is None:
        return call_eval_llm(prompt, system=system)

    key_src = json.dumps({"system": system or "", "prompt": prompt}, ensure_ascii=False, sort_keys=True)
    key = hashlib.sha256(key_src.encode("utf-8")).hexdigest()
    if key in cache:
        return cache[key]

    out = call_eval_llm(prompt, system=system)
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
    """Generate n alternate phrasings of the query using the LLM."""
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
        return _parse_json(call_llm(prompt))
    except (json.JSONDecodeError, ValueError):
        return []
def compute_relevancy(query, answer, jina_model, cache=None):
    """Compute relevancy via cosine similarity between original query and
    alternate queries generated from the answer.

    Returns (score, alternates) where score is mean cosine similarity (0-1).
    """
    prompt = f"""You are an aviation safety research assistant.
Generate {3} alternative phrasings of the ORIGINAL question, informed by the answer content.

===Original Question===
{query}

===Answer===
{answer}

===Return Requirements===
1. Output JSON array of exactly 3 strings.
2. Preserve the same information need as the original question.
3. Do not add details not implied by the original question.
4. No text outside JSON.

Example: ["alt 1", "alt 2", "alt 3"]"""
    try:
        raw_alternates = _parse_json(_cached_llm(prompt, cache=cache))
    except (json.JSONDecodeError, ValueError):
        raw_alternates = []

    alternates = _normalize_alternates(raw_alternates, n=3)

    if not alternates:
        return 0.0, []

    all_texts = [query] + alternates
    embeddings = jina_model.encode(texts=all_texts, task="retrieval", prompt_name="query")
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
    """Append a single result dict to the CSV (create header if new file)."""
    row = {k: v for k, v in result.items() if k not in ("faith_details", "rel_alternates")}
    df = pd.DataFrame([row])
    write_header = not os.path.exists(output_path)
    df.to_csv(output_path, mode="a", header=write_header, index=False)


def run_evaluation(queries, strategies, jina_model, index, mode="semantic",
                   bm25_cache=None, reranker=None, output_path=None,
                   top_k=6, compute_faith=True, compute_rel=True,
                   cache=None, use_hyde=False,
                   allow_bm25_fallback=False):
    """Run evaluation across all queries and strategies.

    mode: "semantic" or "hybrid"
    output_path: if provided, results are saved incrementally and
                 already-completed entries are skipped (resume support).
    Returns list of result dicts.
    """
    completed = _load_completed(output_path) if output_path else set()
    results = []

    pinecone_available = True

    for qi, query in enumerate(queries):
        for strategy in strategies:
            # Skip already-completed entries
            if (query, strategy, mode) in completed:
                print(f"  Skipping (already done): [{mode}/{strategy}] {query[:50]}...")
                continue

            label = f"[{mode}/{strategy}] ({qi+1}/{len(queries)}) {query[:50]}..."
            print(f"  Evaluating: {label}")

            # Retrieve
            used_fallback = False
            if pinecone_available:
                try:
                    if mode == "hybrid" and bm25_cache and reranker:
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
                    can_fallback = allow_bm25_fallback and bm25_cache and strategy in bm25_cache
                    if not can_fallback:
                        raise
                    used_fallback = True
                    pinecone_available = False
                    print(f"  Warning: Pinecone retrieval failed for [{mode}/{strategy}] -> using BM25 fallback for the rest of this run. Error: {e}")

            if (not pinecone_available) or used_fallback:
                can_fallback = bm25_cache and strategy in bm25_cache
                if not can_fallback:
                    raise RuntimeError(f"BM25 fallback requested but cache missing for strategy '{strategy}'")
                bm25, chunks = bm25_cache[strategy]
                matches = bm25_retrieve(query, bm25, chunks, top_k=top_k)

            # Extract context texts
            context_texts = []
            for m in matches:
                if hasattr(m, "metadata"):
                    context_texts.append(m.metadata.get("text", ""))
                else:
                    context_texts.append(m.get("text", ""))

            # Generate answer
            try:
                answer = generate_answer(query, matches)
            except Exception as e:
                print(f"  Warning: generation failed for [{mode}/{strategy}] -> storing placeholder answer. Error: {e}")
                answer = "Generation unavailable due to LLM connection error."

            # Retrieval metrics
            ret_metrics = eval_retrieval(matches)

            # Faithfulness
            if compute_faith:
                try:
                    faith_score, faith_details = compute_faithfulness(answer, context_texts, cache=cache)
                except Exception as e:
                    print(f"  Warning: faithfulness eval failed for [{mode}/{strategy}] -> defaulting to 0.0. Error: {e}")
                    faith_score, faith_details = 0.0, []
            else:
                faith_score, faith_details = 0.0, []

            # Relevancy
            if compute_rel:
                try:
                    rel_score, rel_alternates = compute_relevancy(query, answer, jina_model, cache=cache)
                except Exception as e:
                    print(f"  Warning: relevancy eval failed for [{mode}/{strategy}] -> defaulting to 0.0. Error: {e}")
                    rel_score, rel_alternates = 0.0, []
            else:
                rel_score, rel_alternates = 0.0, []

            result = {
                "query": query,
                "strategy": strategy,
                "mode": mode,
                "answer": answer,
                "num_chunks": len(matches),
                **ret_metrics,
                "faithfulness": round(faith_score, 3),
                "relevancy": round(rel_score, 3),
                "faith_details": faith_details,
                "rel_alternates": rel_alternates,
            }
            results.append(result)

            # Save incrementally
            if output_path:
                _append_result(result, output_path)

            # Persist cache incrementally so interrupted runs still benefit.
            if cache is not None:
                _save_cache(cache)

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
    """Group results by strategy+mode and compute mean scores."""
    df = pd.DataFrame(results)
    cols = ["avg_score", "num_unique_reports", "faithfulness", "relevancy"]
    existing = [c for c in cols if c in df.columns]
    summary = df.groupby(["mode", "strategy"])[existing].mean().round(3)
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete existing results and start from scratch")
    parser.add_argument("--max-queries", type=int, default=0,
                        help="Limit number of evaluation queries (0 = all)")
    parser.add_argument("--top-k", type=int, default=6,
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
        bm25_cache[s] = build_bm25_index(s)

    print("\n--- Semantic-only evaluation ---")
    sem_results = run_evaluation(
        eval_queries, strategies, jina_model, index, mode="semantic",
        bm25_cache=bm25_cache,
        output_path=output_path,
        top_k=top_k,
        compute_faith=not args.skip_faithfulness,
        compute_rel=not args.skip_relevancy,
        cache=cache,
        allow_bm25_fallback=args.allow_bm25_fallback or args.fast,
    )

    print("\n--- Hybrid evaluation ---")
    hyb_results = run_evaluation(
        eval_queries, strategies, jina_model, index, mode="hybrid",
        bm25_cache=bm25_cache, reranker=reranker,
        output_path=output_path,
        top_k=top_k,
        compute_faith=not args.skip_faithfulness,
        compute_rel=not args.skip_relevancy,
        cache=cache,
        use_hyde=args.use_hyde,
        allow_bm25_fallback=args.allow_bm25_fallback or args.fast,
    )

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
