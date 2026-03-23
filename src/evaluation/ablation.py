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
from src.retrieval.query import load_model, init_pinecone
from src.retrieval.hybrid import build_bm25_index, load_reranker
from src.evaluation.evaluate import (
    EVAL_QUERIES, run_evaluation, summarize,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STRATEGIES = ["fixed", "recursive", "semantic"]
MODES = ["semantic", "hybrid"]


def run_ablation(fresh=False):
    """Run the full ablation study and save results."""
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
    for s in STRATEGIES:
        print(f"Building BM25 index for {s}...")
        bm25_cache[s] = build_bm25_index(s)

    all_results = []

    for mode in MODES:
        print(f"\n{'='*60}")
        print(f"Running ablation: mode={mode}")
        print("=" * 60)

        results = run_evaluation(
            EVAL_QUERIES, STRATEGIES, jina_model, index,
            mode=mode,
            bm25_cache=bm25_cache if mode == "hybrid" else None,
            reranker=reranker if mode == "hybrid" else None,
            output_path=detail_path,
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
    args = parser.parse_args()
    run_ablation(fresh=args.fresh)
