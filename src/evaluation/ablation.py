"""
Ablation study: 3 chunking strategies x 2 retrieval modes = 6 configurations.

Runs evaluation for each config and outputs data/ablation_summary.csv.
Supports incremental saving and resume on failure.
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.retrieval.query import load_model, init_pinecone, available_strategies
from src.retrieval.hybrid import build_bm25_index, load_reranker
from src.evaluation.evaluate import (
    EVAL_QUERIES, run_evaluation, summarize,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODES = ["semantic", "hybrid"]


def run_ablation(fresh=False, max_queries=0, top_k=6,
                 skip_faithfulness=False, skip_relevancy=False,
                 fast=False, use_hyde=False,
                 allow_bm25_fallback=False):
    """Run the full ablation study and save results."""
    strategies = available_strategies()

    detail_path = os.path.join(BASE_DIR, "data", "ablation_detailed.csv")
    summary_path = os.path.join(BASE_DIR, "data", "ablation_summary.csv")

    if fresh and os.path.exists(detail_path):
        os.remove(detail_path)
        print("Removed existing results. Starting fresh.\n")
    elif os.path.exists(detail_path):
        print(f"Resuming from {detail_path} (use --fresh to start over)\n")

    print("Loading models...")
    jina_model = load_model()
    index = init_pinecone()
    reranker = load_reranker()

    # Pre-build BM25 indices
    bm25_cache = {}
    for s in strategies:
        print(f"Building BM25 index for {s}...")
        bm25_cache[s] = build_bm25_index(s)

    eval_queries = EVAL_QUERIES
    if fast:
        eval_queries = EVAL_QUERIES[:5]
        top_k = min(top_k, 3)
    if max_queries and max_queries > 0:
        eval_queries = eval_queries[:max_queries]

    all_results = []

    for mode in MODES:
        print(f"\n{'='*60}")
        print(f"Running ablation: mode={mode}")
        print("=" * 60)

        results = run_evaluation(
            eval_queries, strategies, jina_model, index,
            mode=mode,
            bm25_cache=bm25_cache,
            reranker=reranker if mode == "hybrid" else None,
            output_path=detail_path,
            top_k=top_k,
            compute_faith=not skip_faithfulness,
            compute_rel=not skip_relevancy,
            use_hyde=use_hyde,
            allow_bm25_fallback=allow_bm25_fallback or fast,
        )
        all_results.extend(results)

    # Build summary from the full CSV (includes resumed + new results)
    if os.path.exists(detail_path):
        df = pd.read_csv(detail_path)
        summary = summarize(df.to_dict("records"))
        summary.to_csv(summary_path)
        print(f"\nAblation summary saved to {summary_path}")

        print(f"\n{'='*60}")
        print("ABLATION SUMMARY")
        print("=" * 60)
        print(summary.to_string())
        print()

    return all_results, summary if os.path.exists(detail_path) else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ablation study")
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
    parser.add_argument("--fast", action="store_true",
                        help="Quick iteration mode: fewer queries/top-k")
    parser.add_argument("--use-hyde", action="store_true",
                        help="Enable HyDE in hybrid retrieval (better recall, slower)")
    parser.add_argument("--allow-bm25-fallback", action="store_true",
                        help="If Pinecone retrieval fails, fall back to BM25 so ablation can continue")
    args = parser.parse_args()
    run_ablation(
        fresh=args.fresh,
        max_queries=args.max_queries,
        top_k=args.top_k,
        skip_faithfulness=args.skip_faithfulness,
        skip_relevancy=args.skip_relevancy,
        fast=args.fast,
        use_hyde=args.use_hyde,
        allow_bm25_fallback=args.allow_bm25_fallback,
    )
