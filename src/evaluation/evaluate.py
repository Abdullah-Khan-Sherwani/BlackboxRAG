"""
Evaluation module: compare fixed vs recursive vs semantic chunking strategies.
Uses retrieval quality metrics and LLM-as-judge answer quality scoring.
"""
import json
import os
import sys

import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.retrieval.query import load_model, init_pinecone, retrieve
from src.generation.generate import init_gemini, generate_answer, GEMINI_MODEL

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
]


def eval_retrieval(matches):
    """Compute retrieval quality metrics from Pinecone matches.

    Returns dict with avg_score, min_score, max_score, num_unique_reports.
    """
    if not matches:
        return {"avg_score": 0, "min_score": 0, "max_score": 0, "num_unique_reports": 0}

    scores = [m.score for m in matches]
    ntsb_nos = {m.metadata.get("ntsb_no", "") for m in matches}
    ntsb_nos.discard("")

    return {
        "avg_score": sum(scores) / len(scores),
        "min_score": min(scores),
        "max_score": max(scores),
        "num_unique_reports": len(ntsb_nos),
    }


def eval_answer_quality(query, answer, context_texts):
    """Use Gemini as a judge to rate answer relevance and faithfulness.

    Returns dict with relevance (1-5) and faithfulness (1-5).
    """
    judge_prompt = f"""You are an impartial evaluator. Given a question, an answer, and the context used to produce the answer, rate the answer on two dimensions:

1. **Relevance** (1-5): How well does the answer address the question?
2. **Faithfulness** (1-5): Is the answer supported by the provided context without hallucination?

Return ONLY a JSON object like: {{"relevance": 4, "faithfulness": 5}}

--- Context ---
{chr(10).join(context_texts)}

--- Question ---
{query}

--- Answer ---
{answer}
"""
    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(judge_prompt)

    try:
        text = response.text.strip()
        # Handle markdown code blocks in response
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        scores = json.loads(text)
        return {
            "relevance": int(scores.get("relevance", 0)),
            "faithfulness": int(scores.get("faithfulness", 0)),
        }
    except (json.JSONDecodeError, ValueError, AttributeError):
        print(f"  Warning: Could not parse judge response: {response.text[:200]}")
        return {"relevance": 0, "faithfulness": 0}


def run_evaluation(queries, strategies, model, index):
    """Run full evaluation across all queries and strategies.

    Returns a list of result dicts (one per query-strategy pair).
    """
    results = []

    for query in queries:
        for strategy in strategies:
            print(f"  Evaluating: [{strategy}] {query[:60]}...")

            # Retrieve
            matches = retrieve(query, strategy, top_k=5, model=model, index=index)

            # Generate
            answer = generate_answer(query, matches)

            # Eval retrieval
            ret_metrics = eval_retrieval(matches)

            # Eval answer quality
            context_texts = [m.metadata.get("text", "") for m in matches]
            ans_metrics = eval_answer_quality(query, answer, context_texts)

            results.append({
                "query": query,
                "strategy": strategy,
                "answer": answer,
                "num_chunks": len(matches),
                **ret_metrics,
                **ans_metrics,
            })

    return results


def summarize(results):
    """Group results by strategy and compute mean scores. Returns a DataFrame."""
    df = pd.DataFrame(results)
    summary = df.groupby("strategy")[
        ["avg_score", "min_score", "max_score", "num_unique_reports", "relevance", "faithfulness"]
    ].mean().round(3)
    return summary


def main():
    # Initialize resources
    jina_model = load_model()
    index = init_pinecone()
    init_gemini()

    strategies = ["fixed", "recursive", "semantic"]

    print("Running evaluation across all queries and strategies...\n")
    results = run_evaluation(EVAL_QUERIES, strategies, jina_model, index)

    # Save detailed results
    output_path = os.path.join(BASE_DIR, "data", "evaluation_results.csv")
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to {output_path}")

    # Print summary
    summary = summarize(results)
    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON SUMMARY")
    print("=" * 80)
    print(summary.to_string())
    print()


if __name__ == "__main__":
    main()
