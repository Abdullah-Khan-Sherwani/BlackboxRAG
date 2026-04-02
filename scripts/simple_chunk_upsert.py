"""
Simple chunk & upsert script for baseline comparison.
Uses all-MiniLM-L6-v2 (fast, 384-dim) with minimal metadata.
Strategies: recursive, semantic
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.data_prep.chunking import (
    chunk_markdown_baseline_recursive,
    chunk_markdown_baseline_semantic,
)

MD_DIR = BASE_DIR / "dataset-pipeline" / "data" / "extracted" / "extracted"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

INDEX_NAME = "ntsb-rag"
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 100


def parse_args():
    parser = argparse.ArgumentParser(description="Simple chunk & upsert for baseline comparison.")
    parser.add_argument("--strategy", choices=["recursive", "semantic"], required=True)
    parser.add_argument("--reset-index", action="store_true", help="Delete and recreate the Pinecone index.")
    return parser.parse_args()


def load_and_chunk(strategy: str) -> list[dict]:
    md_files = sorted(MD_DIR.glob("*.md"))
    if not md_files:
        raise RuntimeError(f"No markdown files in {MD_DIR}")

    print(f"Found {len(md_files)} markdown files")
    chunk_fn = chunk_markdown_baseline_recursive if strategy == "recursive" else chunk_markdown_baseline_semantic

    all_chunks = []
    for idx, md_file in enumerate(md_files, 1):
        file_chunks = chunk_fn(str(md_file))
        all_chunks.extend(file_chunks)
        if idx % 25 == 0 or idx == len(md_files):
            print(f"  Chunked {idx}/{len(md_files)} files -> {len(all_chunks)} chunks", flush=True)

    return all_chunks


def embed(chunks: list[dict], batch_size: int = 256) -> np.ndarray:
    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    texts = [c["text"] for c in chunks]
    total = len(texts)
    print(f"Encoding {total} chunks (batch_size={batch_size})...")

    all_vecs = []
    start = time.time()
    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        vecs = model.encode(batch, show_progress_bar=False)
        all_vecs.append(np.asarray(vecs, dtype=np.float32))

        done = min(i + batch_size, total)
        elapsed = max(time.time() - start, 1e-6)
        rate = done / elapsed
        eta = int((total - done) / max(rate, 1e-6))
        print(f"  Encoded {done}/{total} | {elapsed:.0f}s | {rate:.0f}/s | ETA: {eta}s", flush=True)

    embeddings = np.concatenate(all_vecs, axis=0)
    print(f"Embeddings shape: {embeddings.shape} | total: {time.time() - start:.0f}s")
    return embeddings


def save_artifacts(chunks: list[dict], embeddings: np.ndarray, strategy: str):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    chunks_path = PROCESSED_DIR / f"chunks_baseline_{strategy}.json"
    emb_path = PROCESSED_DIR / f"embeddings_baseline_{strategy}.npz"

    with chunks_path.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    np.savez_compressed(emb_path, chunk_ids=np.array([c["chunk_id"] for c in chunks]), embeddings=embeddings)
    print(f"Saved: {chunks_path.name}, {emb_path.name}")


def init_index(dimension: int, reset: bool):
    load_dotenv(BASE_DIR / ".env")
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY not set")

    pc = Pinecone(api_key=api_key)
    existing = {idx.name for idx in pc.list_indexes()}

    if reset and INDEX_NAME in existing:
        print(f"Deleting index '{INDEX_NAME}'...")
        pc.delete_index(INDEX_NAME)
        existing.discard(INDEX_NAME)
        print("Waiting 30s for index deletion to propagate...")
        time.sleep(30)

    if INDEX_NAME not in existing:
        print(f"Creating index '{INDEX_NAME}' (dim={dimension})...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    return pc.Index(INDEX_NAME)


def upsert_vectors(index, chunks: list[dict], embeddings: np.ndarray, strategy: str):
    vectors = []
    for chunk, emb in zip(chunks, embeddings):
        vectors.append({
            "id": chunk["chunk_id"],
            "values": emb.tolist(),
            "metadata": {
                "report_id": chunk.get("report_id", ""),
                "ntsb_no": chunk.get("ntsb_no", ""),
                "event_date": chunk.get("event_date", "unknown"),
                "make": chunk.get("make", "unknown"),
                "model": chunk.get("model", "unknown"),
                "strategy": strategy,
                "text": chunk["text"][:1000],  # store truncated text for retrieval display
            },
        })

    total = len(vectors)
    print(f"Upserting {total} vectors (batch={BATCH_SIZE})...")
    for i in range(0, total, BATCH_SIZE):
        batch = vectors[i : i + BATCH_SIZE]
        index.upsert(vectors=batch)
        done = min(i + BATCH_SIZE, total)
        print(f"  Upserted {done}/{total}", flush=True)


def main():
    args = parse_args()
    strategy = args.strategy

    print(f"\n{'='*60}")
    print(f"  Strategy: {strategy} | Model: {MODEL_NAME}")
    print(f"{'='*60}\n")

    # 1. Chunk
    chunks = load_and_chunk(strategy)

    # 2. Embed
    embeddings = embed(chunks)

    # 3. Save locally
    save_artifacts(chunks, embeddings, strategy)

    # 4. Upsert to Pinecone
    index = init_index(dimension=int(embeddings.shape[1]), reset=args.reset_index)
    upsert_vectors(index, chunks, embeddings, strategy)

    print("\nIndex stats:")
    print(index.describe_index_stats())
    print("\nDone!")


if __name__ == "__main__":
    main()
