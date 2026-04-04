"""
Ablation study: 3 chunking strategies × 2 retrieval modes = 6 configurations.

Strategies: md_recursive, section, semantic (placeholder until chunks are ready).
Modes:      semantic-only, hybrid + cross-encoder rerank.
Metrics:    faithfulness, relevancy, retrieval_time, generation_time, total_time.

Runs in parallel batches (10 jobs × 3 LLM calls = 30 RPM, 62s window).
Supports incremental saving and resume on failure.
"""
import argparse
import os
import sys
import warnings

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.retrieval.query import load_model, init_pinecone, available_strategies
from src.retrieval.hybrid import build_bm25_index, load_reranker
from src.evaluation.evaluate import (
    EVAL_QUERIES, run_evaluation, summarize,
    EVAL_BATCH_SIZE, EVAL_WORKERS, BATCH_WINDOW_S,
    _load_cache, _save_cache, CACHE_PATH,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Locked ablation strategies — semantic is a placeholder until chunks are available
ABLATION_STRATEGIES = ["md_recursive", "section", "semantic"]
MODES = ["semantic", "hybrid"]


def _inject_placeholder(detail_path, strategy, mode):
    """Insert NA placeholder rows for a missing chunking strategy.

    Guards against duplicates on resume: skips insertion if rows for
    (strategy, mode) already exist in the CSV.
    """
    if os.path.exists(detail_path):
        existing = pd.read_csv(detail_path)
        already = existing[
            (existing["strategy"] == strategy) & (existing["mode"] == mode)
        ]
        if not already.empty:
            print(f"  Placeholder rows for [{strategy}/{mode}] already present — skipping.")
            return

    rows = []
    for query in EVAL_QUERIES:
        rows.append({
            "query":              query,
            "strategy":           strategy,
            "mode":               mode,
            "answer":             f"PLACEHOLDER — {strategy} chunking not yet available",
            "num_chunks":         0,
            "avg_score":          float("nan"),
            "min_score":          float("nan"),
            "max_score":          float("nan"),
            "num_unique_reports": 0,
            "retrieval_time":     float("nan"),
            "generation_time":    float("nan"),
            "total_time":         float("nan"),
            "faithfulness":       float("nan"),
            "relevancy":          float("nan"),
        })
    df = pd.DataFrame(rows)
    write_header = not os.path.exists(detail_path)
    df.to_csv(detail_path, mode="a", header=write_header, index=False)
    print(f"  Inserted {len(rows)} placeholder rows for [{strategy}/{mode}].")


def run_ablation(fresh=False, max_queries=0, top_k=15,
                 skip_faithfulness=False, skip_relevancy=False,
                 fast=False, use_hyde=False,
                 allow_bm25_fallback=False):
    """Run the full ablation study and save results."""
    detail_path = os.path.join(BASE_DIR, "data", "ablation_detailed.csv")
    summary_path = os.path.join(BASE_DIR, "data", "ablation_summary.csv")

    if fresh and os.path.exists(detail_path):
        os.remove(detail_path)
        print("Removed existing results. Starting fresh.\n")
    elif os.path.exists(detail_path):
        print(f"Resuming from {detail_path} (use --fresh to start over)\n")

    # Determine which strategies actually have chunk files
    actual_available = available_strategies()
    ready_strategies = [s for s in ABLATION_STRATEGIES if s in actual_available]
    missing_strategies = [s for s in ABLATION_STRATEGIES if s not in actual_available]

    if missing_strategies:
        warnings.warn(
            f"\n{'!'*60}\n"
            f"  WARNING: The following chunking strategies are NOT available\n"
            f"  (chunk files missing): {missing_strategies}\n"
            f"  Placeholder NA rows will be inserted in the ablation table.\n"
            f"  Once the chunk files are ready, re-run with --fresh to replace them.\n"
            f"{'!'*60}\n",
            stacklevel=2,
        )

    print("Loading models...")
    jina_model = load_model()
    index = init_pinecone()
    reranker = load_reranker()

    # Load LLM response cache for eval judge (avoids repeat API calls on resume)
    cache = _load_cache(CACHE_PATH)

    # Build BM25 only for strategies that have chunk files
    bm25_cache = {}
    for s in ready_strategies:
        print(f"Building BM25 index for {s}...")
        bm25_cache[s] = build_bm25_index(s)

    eval_queries = EVAL_QUERIES
    if fast:
        eval_queries = EVAL_QUERIES[:5]
        top_k = min(top_k, 3)
    if max_queries and max_queries > 0:
        eval_queries = eval_queries[:max_queries]

    for mode in MODES:
        print(f"\n{'='*60}")
        print(f"Running ablation: mode={mode}")
        print("=" * 60)

        # Insert placeholder rows for missing strategies before real evaluation
        # so the CSV is complete even if the run is interrupted partway through.
        for s in missing_strategies:
            _inject_placeholder(detail_path, s, mode)

        if not ready_strategies:
            print(f"  No ready strategies for mode={mode}. Skipping.")
            continue

        run_evaluation(
            eval_queries, ready_strategies, jina_model, index,
            mode=mode,
            bm25_cache=bm25_cache,
            reranker=reranker if mode == "hybrid" else None,
            output_path=detail_path,
            top_k=top_k,
            compute_faith=not skip_faithfulness,
            compute_rel=not skip_relevancy,
            cache=cache,
            use_hyde=use_hyde,
            allow_bm25_fallback=allow_bm25_fallback or fast,
        )

    # Build summary from full CSV (includes resumed + new + placeholder rows)
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
        return None, summary

    return None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ablation study")
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
    parser.add_argument("--fast", action="store_true",
                        help="Quick iteration mode: 5 queries, top-k=3")
    parser.add_argument("--use-hyde", action="store_true",
                        help="Enable HyDE in hybrid retrieval (better recall, slower)")
    parser.add_argument("--allow-bm25-fallback", action="store_true",
                        help="Fall back to BM25 if Pinecone retrieval fails")
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
