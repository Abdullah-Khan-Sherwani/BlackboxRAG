"""
Query Pinecone index and retrieve relevant chunks.
Supports filtering by chunking strategy.
Text is stored locally (not in Pinecone) and looked up by chunk_id after retrieval.
"""
import json
import os

from dotenv import load_dotenv
from pinecone import Pinecone

INDEX_NAME = "ntsb-rag"
MODEL_NAME = "jinaai/jina-embeddings-v5-text-nano"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def load_model():
    """Load the Jina embedding model."""
    from transformers import AutoModel
    print(f"Loading model: {MODEL_NAME}")
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print("Model loaded.")
    return model


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

    Enriches each match's metadata with the full text from local JSON.
    """
    try:
        query_embedding = model.encode(texts=[query], task="retrieval", prompt_name="query")
    except ValueError:
        query_embedding = model.encode(texts=[query], task="retrieval")
    results = index.query(
        vector=query_embedding[0].tolist(),
        top_k=top_k,
        include_metadata=True,
        filter={"strategy": {"$eq": strategy}},
    )

    # Attach full text from local storage
    chunks_dict = load_chunks(strategy)
    for match in results.matches:
        local = chunks_dict.get(match.id, {})
        match.metadata["text"] = local.get("text", "")

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

    strategies = ["section", "fixed", "recursive", "semantic"]

    for query in SAMPLE_QUERIES:
        for strategy in strategies:
            print(f"\n{'='*80}")
            print(f"Strategy: {strategy}")
            results = retrieve(query, strategy, top_k=5, model=model, index=index)
            print_results(results, query)


if __name__ == "__main__":
    main()
