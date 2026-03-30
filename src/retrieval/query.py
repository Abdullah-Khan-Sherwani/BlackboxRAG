"""
Query Pinecone index and retrieve relevant chunks.
Supports filtering by chunking strategy.
Text is stored locally (not in Pinecone) and looked up by chunk_id after retrieval.
"""
import json
import os

from dotenv import load_dotenv
from pinecone import Pinecone

from src.retrieval.report_mapper import get_pinecone_filter

INDEX_NAME = "ntsb-rag"
MODEL_NAME = "jinaai/jina-embeddings-v5-text-nano"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ALL_STRATEGIES = ["section", "fixed", "recursive", "semantic", "parent"]

SAMPLE_QUERIES = [
    "What are common causes of engine failure during takeoff?",
    "Cessna accidents in instrument meteorological conditions",
    "How does pilot experience affect landing accident outcomes?",
]

# Cache for local chunk lookups: {strategy: {chunk_id: chunk_dict}}
_chunks_cache = {}

CHUNK_FILE_BY_STRATEGY = {
    "fixed": "chunks_fixed.json",
    "recursive": "chunks_recursive.json",
    "semantic": "chunks_semantic.json",
    "section": "chunks_md_section.json",
    "parent": "chunks_parent.json",
}


def load_chunks(strategy):
    """Load and cache the local chunks JSON for a strategy."""
    if strategy not in _chunks_cache:
        filename = CHUNK_FILE_BY_STRATEGY.get(strategy, f"chunks_{strategy}.json")
        path = os.path.join(BASE_DIR, "data", "processed", filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Chunks file not found for strategy '{strategy}': {path}"
            )
        with open(path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        _chunks_cache[strategy] = {c["chunk_id"]: c for c in chunks}
    return _chunks_cache[strategy]


def available_strategies():
    """Return chunking strategies that have local chunk files available."""
    out = []
    for s in ALL_STRATEGIES:
        filename = CHUNK_FILE_BY_STRATEGY.get(s, f"chunks_{s}.json")
        path = os.path.join(BASE_DIR, "data", "processed", filename)
        if os.path.exists(path):
            out.append(s)
    return out


def load_model():
    """Load the Jina embedding model."""
    import os as os_module
    import sys
    from io import StringIO
    from transformers import AutoModel
    
    # Suppress tqdm/stderr to avoid broken pipe errors in Streamlit
    old_stderr = sys.stderr
    old_stdout = sys.stdout
    
    try:
        # Redirect stderr and stdout to avoid broken pipe on tqdm
        sys.stderr = StringIO()
        sys.stdout = StringIO()
        
        model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
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
    try:
        query_embedding = model.encode(texts=[query], task="retrieval", prompt_name="query")
    except ValueError:
        query_embedding = model.encode(texts=[query], task="retrieval")
    
    # Build filter: includes strategy and optional NTSB number filter
    filter_dict = get_pinecone_filter(query, strategy)
    
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

    # Attach full text from local storage
    chunks_dict = load_chunks(strategy)
    for match in results.matches:
        local = chunks_dict.get(match.id, {})
        # Parent strategy uses child retrieval with parent-level context for generation.
        match.metadata["text"] = local.get("parent_text") or local.get("text", "")
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
