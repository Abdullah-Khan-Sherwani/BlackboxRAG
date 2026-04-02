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

INDEX_NAME = "ntsb-rag"
MODEL_NAME = "jinaai/jina-embeddings-v5-text-nano"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ALL_STRATEGIES = ["section", "md_recursive", "parent_child", "fixed", "recursive", "semantic", "parent"]

SAMPLE_QUERIES = [
    "What are common causes of engine failure during takeoff?",
    "Cessna accidents in instrument meteorological conditions",
    "How does pilot experience affect landing accident outcomes?",
]

# Cache for local chunk lookups: {strategy: {chunk_id: chunk_dict}}
_chunks_cache = {}


def _canonical_strategy(strategy: str) -> str:
    """Map legacy names to the canonical markdown strategies."""
    if strategy == "md_recursive":
        return "md_recursive"
    if strategy in {"parent", "parent_child"}:
        return "parent_child"
    return strategy


def _chunks_file_for_strategy(strategy: str) -> str:
    """Resolve local chunk artifact name across advanced and baseline modes."""
    s = _canonical_strategy(strategy)
    if s == "section":
        return "chunks_md_section.json"
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


def retrieve(query, strategy, top_k=5, model=None, index=None):
    """Encode a query and retrieve top-k matching chunks from Pinecone.

    Uses query-to-report mapping to filter results for single-event queries.
    Enriches each match's metadata with the full text from local JSON.
    """
    canonical_strategy = _canonical_strategy(strategy)

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
    filter_dict = get_pinecone_filter(query, canonical_strategy)
    
    try:
        results = index.query(
            vector=query_embedding[0].tolist(),
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict,
        )
    except Exception as e:
        raise RuntimeError(
            "Pinecone query failed (network/DNS or service issue). "
            "Check internet/VPN and Pinecone availability, or run evaluation with BM25 fallback."
        ) from e

    # Attach full text from local storage.
    # The Pinecone index contains section-aware vectors, so always look up
    # chunk text from the section store.  Fall back to the requested strategy
    # in case the ID happens to exist there instead.
    section_store = load_chunks("section")
    section_dict = section_store["by_id"]
    parent_lookup = section_store["parent_lookup"]

    # If the user picked a different strategy, load it as a secondary lookup.
    if canonical_strategy != "section":
        try:
            alt_store = load_chunks(canonical_strategy)
            alt_dict = alt_store["by_id"]
        except FileNotFoundError:
            alt_dict = {}
    else:
        alt_dict = {}

    for match in results.matches:
        local = section_dict.get(match.id) or alt_dict.get(match.id) or {}
        # Provenance marker for semantic retrieval path.
        match.metadata["retrieval_strategy"] = "semantic"
        # Parent-child strategy retrieves child vectors but sends parent context to LLM.
        if canonical_strategy == "parent_child":
            parent_id = local.get("parent_id", "")
            parent_text = parent_lookup.get(parent_id, "") if parent_id else ""
            match.metadata["text"] = parent_text or local.get("text", "")
        else:
            match.metadata["text"] = local.get("text", "")
        # Ensure provenance fields are available even when Pinecone metadata is sparse.
        if "entity_id" in local and not match.metadata.get("entity_id"):
            match.metadata["entity_id"] = local.get("entity_id", "")
        if "source_filename" in local and not match.metadata.get("source_filename"):
            match.metadata["source_filename"] = local.get("source_filename", "")
        if "context_summary" in local and not match.metadata.get("context_summary"):
            match.metadata["context_summary"] = local.get("context_summary", "")
        if "role" in local and not match.metadata.get("role"):
            match.metadata["role"] = local.get("role", "Unknown")
        if "section_title" in local and not match.metadata.get("section_title"):
            match.metadata["section_title"] = local.get("section_title", "")
        if "report_id" in local and not match.metadata.get("report_id"):
            match.metadata["report_id"] = local.get("report_id", "")
        if "ntsb_no" in local and not match.metadata.get("ntsb_no"):
            match.metadata["ntsb_no"] = local.get("ntsb_no", "")
        if "event_date" in local and not match.metadata.get("event_date"):
            match.metadata["event_date"] = local.get("event_date", "")
        if "make" in local and not match.metadata.get("make"):
            match.metadata["make"] = local.get("make", "")
        if "model" in local and not match.metadata.get("model"):
            match.metadata["model"] = local.get("model", "")
        if "entities" in local and not match.metadata.get("entities"):
            match.metadata["entities"] = local.get("entities", "")
        if "aircraft_components" in local and not match.metadata.get("aircraft_components"):
            match.metadata["aircraft_components"] = local.get("aircraft_components", "")
        if "numerics" in local and not match.metadata.get("numerics"):
            match.metadata["numerics"] = local.get("numerics", "")
        if "parent_id" in local:
            match.metadata["parent_id"] = local.get("parent_id", "")

    return results.matches


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
