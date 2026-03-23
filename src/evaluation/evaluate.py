"""
Evaluation module with Faithfulness and Relevancy metrics.

Faithfulness: claim extraction → verification → % supported by context.
Relevancy: alternate query generation → cosine similarity with original query.

Uses DeepSeek V3.2 as LLM-as-judge for both metrics.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.llm.client import call_llm
from src.retrieval.query import load_model, init_pinecone, retrieve
from src.retrieval.hybrid import build_bm25_index, load_reranker, hybrid_retrieve
from src.generation.generate import generate_answer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


# ── Faithfulness ──────────────────────────────────────────────────────────────

def extract_claims(answer):
    """Extract factual claims from an answer using the LLM."""
    prompt = f"""You are an expert fact-checker for aviation accident analysis.
Your task is to extract all distinct factual claims from the provided answer.

===Answer===
{answer}

===Return Requirements===
1. The output must be a properly formatted JSON array of strings.
2. Each string must be a single atomic factual claim (one fact per item).
3. Do NOT include opinions, hedging language, or meta-commentary.
4. Do NOT include any additional explanation, commentary, or text outside of the JSON array.
5. The output must include only the JSON array and no additional text before or after it.

Example: ["claim 1", "claim 2", "claim 3"]"""
    try:
        return _parse_json(call_llm(prompt))
    except (json.JSONDecodeError, ValueError):
        return []


def verify_claims(claims, context_texts):
    """Verify each claim against the provided context using the LLM.

    Returns list of dicts with keys: claim, supported (bool), reasoning.
    """
    if not claims:
        return []

    context_str = "\n\n".join(context_texts)
    claims_str = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))

    prompt = f"""You are an impartial fact-checker for aviation accident analysis.
Your task is to verify each claim against the provided context and determine if it is supported.

===Context===
{context_str}

===Claims===
{claims_str}

===Return Requirements===
1. The output must be a properly formatted JSON array.
2. Each element must have exactly three keys: "claim", "supported", "reasoning".
3. "claim": the original claim text.
4. "supported": boolean true if the claim is directly supported by the context, false otherwise.
5. "reasoning": a brief one-sentence explanation citing the relevant context.
6. All judgments must be derived from the provided context only. No external knowledge is permitted.
7. Do NOT include any additional explanation, commentary, or text outside of the JSON array.
8. The output must include only the JSON array and no additional text before or after it.

Example: [{{"claim": "...", "supported": true, "reasoning": "..."}}]"""

    try:
        return _parse_json(call_llm(prompt))
    except (json.JSONDecodeError, ValueError):
        return [{"claim": c, "supported": False, "reasoning": "parse_error"} for c in claims]


def compute_faithfulness(answer, context_texts):
    """Compute faithfulness score: fraction of claims supported by context.

    Returns (score, details) where score is 0.0-1.0 and details is the
    list of verified claims.
    """
    claims = extract_claims(answer)
    if not claims:
        return 1.0, []

    verified = verify_claims(claims, context_texts)
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


def compute_relevancy(query, answer, jina_model):
    """Compute relevancy via cosine similarity between original query and
    alternate queries generated from the answer.

    Returns (score, alternates) where score is mean cosine similarity (0-1).
    """
    alternates = generate_alternate_queries(query, n=3)
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
                   bm25_cache=None, reranker=None, output_path=None):
    """Run evaluation across all queries and strategies.

    mode: "semantic" or "hybrid"
    output_path: if provided, results are saved incrementally and
                 already-completed entries are skipped (resume support).
    Returns list of result dicts.
    """
    completed = _load_completed(output_path) if output_path else set()
    results = []

    for qi, query in enumerate(queries):
        for strategy in strategies:
            # Skip already-completed entries
            if (query, strategy, mode) in completed:
                print(f"  Skipping (already done): [{mode}/{strategy}] {query[:50]}...")
                continue

            label = f"[{mode}/{strategy}] ({qi+1}/{len(queries)}) {query[:50]}..."
            print(f"  Evaluating: {label}")

            # Retrieve
            if mode == "hybrid" and bm25_cache and reranker:
                bm25, chunks = bm25_cache[strategy]
                matches = hybrid_retrieve(
                    query, strategy, top_k=5,
                    model=jina_model, index=index,
                    bm25=bm25, chunks=chunks, reranker=reranker,
                )
            else:
                matches = retrieve(query, strategy, top_k=5, model=jina_model, index=index)

            # Extract context texts
            context_texts = []
            for m in matches:
                if hasattr(m, "metadata"):
                    context_texts.append(m.metadata.get("text", ""))
                else:
                    context_texts.append(m.get("text", ""))

            # Generate answer
            answer = generate_answer(query, matches)

            # Retrieval metrics
            ret_metrics = eval_retrieval(matches)

            # Faithfulness
            faith_score, faith_details = compute_faithfulness(answer, context_texts)

            # Relevancy
            rel_score, rel_alternates = compute_relevancy(query, answer, jina_model)

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
    args = parser.parse_args()

    output_path = os.path.join(BASE_DIR, "data", "evaluation_results.csv")

    if args.fresh and os.path.exists(output_path):
        os.remove(output_path)
        print("Removed existing results. Starting fresh.\n")
    elif os.path.exists(output_path):
        print(f"Resuming from {output_path} (use --fresh to start over)\n")

    jina_model = load_model()
    index = init_pinecone()
    reranker = load_reranker()

    strategies = ["fixed", "recursive", "semantic"]

    # Pre-build BM25 indices
    bm25_cache = {}
    for s in strategies:
        print(f"Building BM25 index for {s}...")
        bm25_cache[s] = build_bm25_index(s)

    print("\n--- Semantic-only evaluation ---")
    sem_results = run_evaluation(
        EVAL_QUERIES, strategies, jina_model, index, mode="semantic",
        output_path=output_path,
    )

    print("\n--- Hybrid evaluation ---")
    hyb_results = run_evaluation(
        EVAL_QUERIES, strategies, jina_model, index, mode="hybrid",
        bm25_cache=bm25_cache, reranker=reranker,
        output_path=output_path,
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
