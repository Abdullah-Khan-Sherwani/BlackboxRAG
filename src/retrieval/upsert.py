"""
Upsert chunk embeddings to Pinecone.
All 3 chunking strategies go into one index, distinguished by a 'strategy' metadata field.
"""
import json
import os
import numpy as np
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

INDEX_NAME = "ntsb-rag"
DIMENSION = 768
BATCH_SIZE = 100
STRATEGIES = ["fixed", "recursive", "semantic", "parent"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHUNK_FILES = {
    s: os.path.join(BASE_DIR, "data", "processed", f"chunks_{s}.json")
    for s in STRATEGIES
}
EMBEDDING_FILES = {
    s: os.path.join(BASE_DIR, "data", "processed", f"embeddings_{s}.npz")
    for s in STRATEGIES
}


def init_pinecone():
    """Initialize Pinecone client and ensure the index exists."""
    load_dotenv(os.path.join(BASE_DIR, ".env"))
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY not found in environment. Check your .env file.")

    pc = Pinecone(api_key=api_key)

    existing = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in existing:
        print(f"Creating index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print("Index created.")
    else:
        print(f"Index '{INDEX_NAME}' already exists.")

    return pc.Index(INDEX_NAME)


def load_data(strategy):
    """Load embeddings (.npz) and chunk metadata (.json) for a strategy."""
    npz = np.load(EMBEDDING_FILES[strategy])
    chunk_ids = npz["chunk_ids"].tolist()
    embeddings = npz["embeddings"]

    with open(CHUNK_FILES[strategy], "r", encoding="utf-8") as f:
        chunks = json.load(f)
    chunks_dict = {c["chunk_id"]: c for c in chunks}

    return chunk_ids, embeddings, chunks_dict


def build_vectors(chunk_ids, embeddings, chunks_dict, strategy):
    """Build Pinecone vector dicts with metadata."""
    vectors = []
    for cid, emb in zip(chunk_ids, embeddings):
        chunk = chunks_dict[cid]
        entity_id = chunk.get("entity_id") or chunk.get("ntsb_no") or chunk.get("report_id", "")
        vectors.append({
            "id": cid,
            "values": emb.tolist(),
            "metadata": {
                "ntsb_no": chunk.get("ntsb_no", ""),
                "report_id": chunk.get("report_id", ""),
                "entity_id": entity_id,
                "event_date": chunk.get("event_date", ""),
                "state": chunk.get("state", ""),
                "make": chunk.get("make", ""),
                "model": chunk.get("model", ""),
                "phase_of_flight": chunk.get("phase_of_flight", ""),
                "weather": chunk.get("weather", ""),
                "role": chunk.get("role", "Unknown"),
                "source_filename": chunk.get("source_filename", ""),
                "context_summary": chunk.get("context_summary", ""),
                "section_title": chunk.get("section_title", ""),
                "strategy": strategy,
            },
        })
    return vectors


def upsert_vectors(index, vectors):
    """Batch upsert vectors to Pinecone."""
    for i in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[i : i + BATCH_SIZE]
        index.upsert(vectors=batch)
        print(f"  Upserted {min(i + BATCH_SIZE, len(vectors))}/{len(vectors)}")


def main():
    index = init_pinecone()

    for strategy in STRATEGIES:
        if not os.path.exists(CHUNK_FILES[strategy]) or not os.path.exists(EMBEDDING_FILES[strategy]):
            print(f"\n--- {strategy} ---")
            print("  Skipping: chunk or embedding artifact not found")
            continue

        print(f"\n--- {strategy} ---")
        chunk_ids, embeddings, chunks_dict = load_data(strategy)
        print(f"  Loaded {len(chunk_ids)} chunks, embeddings shape {embeddings.shape}")
        vectors = build_vectors(chunk_ids, embeddings, chunks_dict, strategy)
        upsert_vectors(index, vectors)

    print("\n--- Index Stats ---")
    print(index.describe_index_stats())


if __name__ == "__main__":
    main()
