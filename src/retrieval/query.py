"""
Query Pinecone index and retrieve relevant chunks.
Supports filtering by chunking strategy.
Text is stored locally (not in Pinecone) and looked up by chunk_id after retrieval.
"""
import json
import os
import warnings

from dotenv import load_dotenv
from pinecone import Pinecone

from src.retrieval.report_mapper import get_pinecone_filter

# Section titles that are structural noise and should never surface as evidence.
BLOCKED_SECTION_TITLES: frozenset = frozenset({"Contents", "CONTENT"})

INDEX_NAME = "ntsb-rag"
MODEL_NAME = "jinaai/jina-embeddings-v5-text-nano"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ALL_STRATEGIES = ["section", "md_recursive", "parent_child", "fixed", "recursive", "semantic"]

SAMPLE_QUERIES = [
    "What are common causes of engine failure during takeoff?",
    "Cessna accidents in instrument meteorological conditions",
    "How does pilot experience affect landing accident outcomes?",
]

# Cache for local chunk lookups: {strategy: {chunk_id: chunk_dict}}
_chunks_cache = {}


def _canonical_strategy(strategy: str) -> str:
    """Map legacy names to the canonical markdown strategies (for local files)."""
    if strategy in {"md_recursive", "recursive_3/4/26"}:
        return "md_recursive"
    if strategy in {"parent", "parent_child"}:
        return "parent_child"
    return strategy


def _pinecone_strategy(strategy: str) -> str:
    """Map strategy names to the Pinecone metadata filter value."""
    if strategy == "md_recursive":
        return "recursive_3/4/26"
    if strategy in {"parent", "parent_child"}:
        return "parent_child"
    return strategy


def _chunks_file_for_strategy(strategy: str) -> str:
    """Resolve local chunk artifact name across advanced and baseline modes."""
    s = _canonical_strategy(strategy)
    if s == "section":
        return "chunks_md_section_enriched.json"
    if s == "md_recursive":
        return "chunks_md_md_recursive.json"
    if s == "parent_child":
        return "chunks_md_parent_child.json"
    if s in {"fixed", "recursive", "semantic"}:
        baseline_name = f"chunks_baseline_{s}.json"
        baseline_path = os.path.join(BASE_DIR, "data", "processed", baseline_name)
        if os.path.exists(baseline_path):
            return baseline_name

        # Legacy fallback from earlier tabular experiments.
        legacy = {
            "fixed": "chunks_fixed.json",
            "recursive": "chunks_recursive.json",
            "semantic": "chunks_semantic.json",
        }
        return legacy[s]
    return f"chunks_md_{s}.json"


def _resolve_chunks_path(filename: str) -> str:
    """Resolve chunk path across current and nested legacy folder layouts."""
    base = os.path.join(BASE_DIR, "data", "processed")
    candidates = [
        os.path.join(base, filename),
        os.path.join(base, "chunks_md_recursive", filename),
    ]

    # Fallback aliases when naming changed across experiments.
    alias_map = {
        "chunks_md_md_recursive.json": ["chunks_md_recursive.json"],
        "chunks_md_parent_child.json": ["chunks_parent_child.json", "chunks_parent.json"],
    }
    for alias in alias_map.get(filename, []):
        candidates.append(os.path.join(base, alias))
        candidates.append(os.path.join(base, "chunks_md_recursive", alias))

    for path in candidates:
        if os.path.exists(path):
            return path
    return ""


def load_chunks(strategy):
    """Load and cache the local chunks JSON for a strategy."""
    strategy = _canonical_strategy(strategy)
    if strategy not in _chunks_cache:
        filename = _chunks_file_for_strategy(strategy)
        path = _resolve_chunks_path(filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Chunks file not found for strategy '{strategy}' (requested '{filename}')"
            )
        with open(path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        by_id = {c["chunk_id"]: c for c in chunks}
        parent_lookup = {}
        for c in chunks:
            pid = c.get("parent_id", "")
            ptext = c.get("parent_text", "")
            if pid and ptext and pid not in parent_lookup:
                parent_lookup[pid] = ptext
        _chunks_cache[strategy] = {"by_id": by_id, "parent_lookup": parent_lookup}
    return _chunks_cache[strategy]


def available_strategies():
    """Return chunking strategies that have local chunk files available."""
    out = []
    for s in ALL_STRATEGIES:
        filename = _chunks_file_for_strategy(s)
        path = _resolve_chunks_path(filename)
        if os.path.exists(path):
            out.append(s)
    return out


def load_model():
    """Load the Jina embedding model."""
    import torch
    import sys
    from io import StringIO
    from transformers import AutoModel
    
    # Suppress tqdm/stderr to avoid broken pipe errors in Streamlit
    old_stderr = sys.stderr
    old_stdout = sys.stdout

    # Silence noisy upstream deprecation warnings from optional image modules.
    warnings.filterwarnings(
        "ignore",
        message=r"Accessing `__path__` from `.*`\. Returning `__path__` instead\..*",
    )
    
    try:
        # Redirect stderr and stdout to avoid broken pipe on tqdm
        sys.stderr = StringIO()
        sys.stdout = StringIO()
        
        model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
        runtime_device = "cuda" if torch.cuda.is_available() else "cpu"
        if hasattr(model, "to"):
            model = model.to(runtime_device)
        setattr(model, "_runtime_device", runtime_device)
        return model
    finally:
        # Restore stderr and stdout
        sys.stderr = old_stderr
        sys.stdout = old_stdout
        print(f"Model loaded: {MODEL_NAME}")


def init_pinecone():
    """Initialize Pinecone client and return the index."""
    load_dotenv(os.path.join(BASE_DIR, ".env"))
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY not found in environment. Check your .env file.")

    pc = Pinecone(api_key=api_key)
    return pc.Index(INDEX_NAME)


def retrieve(query, strategy, top_k=5, model=None, index=None, ntsb_override=None):
    """Encode a query and retrieve top-k matching chunks from Pinecone.

    Uses query-to-report mapping to filter results for single-event queries.
    Enriches each match's metadata with the full text from local JSON.

    Args:
        ntsb_override: If provided, use this NTSB number for filtering instead
            of running detect_report_from_query() on the query text.
    """
    canonical_strategy = _canonical_strategy(strategy)
    pinecone_strategy = _pinecone_strategy(strategy)

    runtime_device = getattr(model, "_runtime_device", "cpu") if model is not None else "cpu"
    try:
        query_embedding = model.encode(
            texts=[query],
            task="retrieval",
            prompt_name="query",
            device=runtime_device,
        )
    except TypeError:
        try:
            query_embedding = model.encode(texts=[query], task="retrieval", prompt_name="query")
        except ValueError:
            query_embedding = model.encode(texts=[query], task="retrieval")
    except ValueError:
        try:
            query_embedding = model.encode(texts=[query], task="retrieval", device=runtime_device)
        except TypeError:
            query_embedding = model.encode(texts=[query], task="retrieval")

    # Build filter: includes strategy and optional NTSB number filter
    if ntsb_override:
        if isinstance(ntsb_override, list):
            filter_dict = {"$and": [
                {"strategy": {"$eq": pinecone_strategy}},
                {"ntsb_no": {"$in": ntsb_override}},
            ]}
        else:
            filter_dict = {"$and": [
                {"strategy": {"$eq": pinecone_strategy}},
                {"ntsb_no": {"$eq": ntsb_override}},
            ]}
    else:
        filter_dict = get_pinecone_filter(query, pinecone_strategy)
    
    import time as _time
    _max_retries = 3
    for _attempt in range(1, _max_retries + 1):
        try:
            results = index.query(
                vector=query_embedding[0].tolist(),
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict,
            )
            break
        except Exception as e:
            if _attempt == _max_retries:
                raise RuntimeError(
                    "Pinecone query failed after retries (network/DNS or service issue). "
                    "Check internet/VPN and Pinecone availability, or run evaluation with BM25 fallback."
                ) from e
            _time.sleep(2 ** _attempt)

    # Attach full text from local storage.
    # Each Pinecone vector carries a 'strategy' metadata field. Route each match
    # to the correct local chunk store so text lookup is correct regardless of
    # whether Pinecone returns a single strategy or a mix.
    _local_stores: dict[str, dict] = {}

    def _get_local_store(strat: str) -> dict:
        """Lazy-load and cache local chunk store per strategy."""
        if strat not in _local_stores:
            try:
                _local_stores[strat] = load_chunks(strat)
            except FileNotFoundError:
                _local_stores[strat] = {"by_id": {}, "parent_lookup": {}}
        return _local_stores[strat]

    # Pre-load the requested strategy so the common path is always warm.
    _get_local_store(canonical_strategy)

    for match in results.matches:
        match_strategy = match.metadata.get("strategy") or canonical_strategy
        store = _get_local_store(match_strategy)
        # Strip legacy strategy prefixes added during old upsert runs (e.g. "r3426_")
        local_id = match.id
        if not store["by_id"].get(local_id):
            underscore = local_id.find("_")
            if underscore != -1:
                local_id = local_id[underscore + 1:]
        local = store["by_id"].get(local_id) or {}

        # Provenance marker for semantic retrieval path.
        match.metadata["retrieval_strategy"] = "semantic"

        # Parent-child strategy retrieves child vectors but sends parent context to LLM.
        if match_strategy == "parent_child":
            parent_id = local.get("parent_id", "")
            parent_text = store["parent_lookup"].get(parent_id, "") if parent_id else ""
            match.metadata["text"] = parent_text or local.get("text", "")
        else:
            match.metadata["text"] = local.get("text", "")

        # Enrich Pinecone metadata with any fields only present in the local store.
        for field in (
            "entity_id", "source_filename", "context_summary", "role",
            "section_title", "report_id", "ntsb_no", "event_date",
            "make", "model", "entities", "aircraft_components",
            "numerics", "parent_id", "contextualized_text",
        ):
            if field in local and not match.metadata.get(field):
                match.metadata[field] = local[field]

    return [
        m for m in results.matches
        if m.metadata.get("section_title", "") not in BLOCKED_SECTION_TITLES
    ]


def print_results(results, query):
    """Pretty-print retrieval results."""
    print(f"\nQuery: {query}")
    print("-" * 80)
    for rank, match in enumerate(results, 1):
        text_snippet = match.metadata.get("text", "")[:200]
        print(f"  [{rank}] Score: {match.score:.4f} | ID: {match.id}")
        print(f"       {text_snippet}...")
        print()


def main():
    model = load_model()
    index = init_pinecone()

    strategies = available_strategies()

    for query in SAMPLE_QUERIES:
        for strategy in strategies:
            print(f"\n{'='*80}")
            print(f"Strategy: {strategy}")
            results = retrieve(query, strategy, top_k=5, model=model, index=index)
            print_results(results, query)


if __name__ == "__main__":
    main()
