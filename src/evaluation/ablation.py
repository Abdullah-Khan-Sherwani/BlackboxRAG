"""
Ablation study: 3 chunking strategies x 2 retrieval modes = 6 configurations.

Runs evaluation for each config and outputs data/ablation_summary.csv.
"""
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


def run_ablation():
    """Run the full ablation study and save results."""
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
        )
        all_results.extend(results)

    # Build detailed results DataFrame
    df = pd.DataFrame(all_results)
    detail_path = os.path.join(BASE_DIR, "data", "ablation_detailed.csv")
    df_save = df.drop(columns=["faith_details", "rel_alternates"], errors="ignore")
    df_save.to_csv(detail_path, index=False)
    print(f"\nDetailed ablation results saved to {detail_path}")

    # Build summary
    summary = summarize(all_results)
    summary_path = os.path.join(BASE_DIR, "data", "ablation_summary.csv")
    summary.to_csv(summary_path)
    print(f"Ablation summary saved to {summary_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print("ABLATION SUMMARY")
    print("=" * 60)
    print(summary.to_string())
    print()

    return all_results, summary


if __name__ == "__main__":
    run_ablation()
