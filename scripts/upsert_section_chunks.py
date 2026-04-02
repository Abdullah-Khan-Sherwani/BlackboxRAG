"""
Upsert enriched section chunks to Pinecone using pre-built embeddings.

Run enrich_section_chunks.py first to produce chunks_md_section_enriched.json.

Reads  : data/processed/chunks_md_section_enriched.json
         data/processed/embeddings_md_section.npz
Upserts: Pinecone index 'ntsb-rag' with strategy='section'
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from pinecone import Pinecone

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

INDEX_NAME     = "ntsb-rag"
CHUNK_FILE     = BASE_DIR / "data" / "processed" / "chunks_md_section_enriched.json"
EMBEDDING_FILE = BASE_DIR / "data" / "processed" / "embeddings_md_section.npz"
BATCH_SIZE     = 100


def init_pinecone() -> "pinecone.Index":
    load_dotenv(BASE_DIR / "data" / "processed" / "env")
    load_dotenv(BASE_DIR / ".env")
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY not found. Check .env or data/processed/env.")
    pc = Pinecone(api_key=api_key)
    return pc.Index(INDEX_NAME)


def load_artifacts() -> tuple[dict[str, dict], np.ndarray, list[str]]:
    if not CHUNK_FILE.exists():
        raise FileNotFoundError(
            f"{CHUNK_FILE.name} not found. Run enrich_section_chunks.py first."
        )
    if not EMBEDDING_FILE.exists():
        raise FileNotFoundError(f"{EMBEDDING_FILE.name} not found.")

    print(f"Loading enriched chunks from {CHUNK_FILE.name}...")
    with open(CHUNK_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    by_id = {c["chunk_id"]: c for c in chunks}
    print(f"  {len(by_id)} chunks loaded")

    print(f"Loading embeddings from {EMBEDDING_FILE.name}...")
    npz = np.load(EMBEDDING_FILE)
    chunk_ids = npz["chunk_ids"].tolist()
    embeddings = npz["embeddings"]
    print(f"  {len(chunk_ids)} embeddings, shape={embeddings.shape}")

    return by_id, embeddings, chunk_ids


def build_vectors(by_id: dict, embeddings: np.ndarray, chunk_ids: list[str]) -> list[dict]:
    vectors = []
    skipped = 0
    for cid, emb in zip(chunk_ids, embeddings):
        chunk = by_id.get(cid)
        if chunk is None:
            skipped += 1
            continue
        vectors.append({
            "id": cid,
            "values": emb.tolist(),
            "metadata": {
                "ntsb_no":        chunk.get("ntsb_no", ""),
                "report_id":      chunk.get("report_id", ""),
                "entity_id":      chunk.get("entity_id", chunk.get("report_id", "")),
                "event_date":     chunk.get("event_date", ""),
                "state":          chunk.get("state", ""),
                "make":           chunk.get("make", ""),
                "model":          chunk.get("model", ""),
                "phase_of_flight": chunk.get("phase_of_flight", ""),
                "weather":        chunk.get("weather", ""),
                "section_title":  chunk.get("section_title", ""),
                "source_filename": chunk.get("source_filename", ""),
                "context_summary": chunk.get("context_summary", ""),
                "strategy":       "section",
            },
        })
    if skipped:
        print(f"  WARNING: {skipped} chunk IDs in embeddings had no match in chunks file")
    return vectors


def upsert_vectors(index, vectors: list[dict]) -> None:
    total = len(vectors)
    print(f"Upserting {total} vectors in batches of {BATCH_SIZE}...")
    for i in range(0, total, BATCH_SIZE):
        batch = vectors[i : i + BATCH_SIZE]
        index.upsert(vectors=batch)
        done = min(i + BATCH_SIZE, total)
        print(f"  {done}/{total} ({done/total*100:.1f}%)", flush=True)
    print("Upsert complete.")


def main():
    print("=== Upsert Section Chunks to Pinecone ===\n")

    index = init_pinecone()
    print(f"Connected to index '{INDEX_NAME}'")

    by_id, embeddings, chunk_ids = load_artifacts()
    vectors = build_vectors(by_id, embeddings, chunk_ids)
    print(f"Built {len(vectors)} vectors with enriched metadata\n")

    upsert_vectors(index, vectors)

    stats = index.describe_index_stats()
    print(f"\nIndex total vectors: {stats.total_vector_count}")


if __name__ == "__main__":
    main()
