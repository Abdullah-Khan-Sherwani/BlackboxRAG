"""
Generate embeddings for chunk files.
"""
import json
import os
import re

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MODEL_NAME = "all-MiniLM-L6-v2"
DIMENSION = 384
BATCH_SIZE = 32

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHUNK_FILES = {
    "fixed": os.path.join(BASE_DIR, "data", "processed", "chunks_fixed.json"),
    "recursive": os.path.join(BASE_DIR, "data", "processed", "chunks_recursive.json"),
    "semantic": os.path.join(BASE_DIR, "data", "processed", "chunks_semantic.json"),
    "parent": os.path.join(BASE_DIR, "data", "processed", "chunks_parent.json"),
}

OUTPUT_FILES = {
    "fixed": os.path.join(BASE_DIR, "data", "processed", "embeddings_fixed.npz"),
    "recursive": os.path.join(BASE_DIR, "data", "processed", "embeddings_recursive.npz"),
    "semantic": os.path.join(BASE_DIR, "data", "processed", "embeddings_semantic.npz"),
    "parent": os.path.join(BASE_DIR, "data", "processed", "embeddings_parent.npz"),
}


def load_model():
    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")
    model.to(device)

    return model


def embed_chunks(chunks: list[dict], model=None) -> tuple[list[dict], np.ndarray]:
    """Take a list of chunks and return generated embeddings."""
    if model is None:
        model = load_model()

    for chunk in chunks:
        if not chunk.get("contextualized_text"):
            source = chunk.get("source_filename", "local_artifact")
            entity_id = chunk.get("entity_id") or chunk.get("ntsb_no") or chunk.get("report_id", "")
            context = chunk.get("context_summary") or _density_context(chunk.get("text", ""))
            chunk["context_summary"] = context
            chunk["contextualized_text"] = (
                f"[Source: {source}] [Entity_ID: {entity_id}] [Context: {context}]\n"
                f"{chunk.get('text', '')}"
            )

    texts = [c.get("contextualized_text", c.get("text", "")) for c in chunks]

    all_embeddings = []
    print(f"Starting embedding generation for {len(texts)} chunks...")
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding"):
        batch = texts[i : i + BATCH_SIZE]
        emb = model.encode(batch, convert_to_numpy=True)
        all_embeddings.append(emb)

    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    return chunks, embeddings


def save_embeddings(chunks: list[dict], embeddings: np.ndarray, out_path: str):
    chunk_ids = np.array([c["chunk_id"] for c in chunks])
    np.savez_compressed(out_path, chunk_ids=chunk_ids, embeddings=embeddings)


def _density_context(text: str) -> str:
    """Fallback density summary preserving identifiers and numbers."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return "No context available"

    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    id_re = re.compile(r"\d|[A-Z]{2,}\d+|\b[A-Z]{2,}[\-/]\d+")
    selected = [p.strip() for p in parts if id_re.search(p)]
    if not selected:
        selected = [p.strip() for p in parts[:2] if p.strip()]

    merged = " ".join(selected)
    density = len(id_re.findall(cleaned))
    max_len = 240 if density < 6 else 420 if density < 14 else 620
    return merged[:max_len].strip()


def main():
    model = load_model()

    for strategy, chunk_path in CHUNK_FILES.items():
        output_path = OUTPUT_FILES[strategy]

        if not os.path.exists(chunk_path):
            print(f"Skipping {strategy} - {os.path.basename(chunk_path)} not found.")
            continue

        print(f"\nLoading chunks for strategy: {strategy}")
        with open(chunk_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        print(f"Loaded {len(chunks)} chunks.")

        chunks, embeddings = embed_chunks(chunks, model=model)
        print(f"Embedding shape: {embeddings.shape}")
        if embeddings.shape[1] != DIMENSION:
            print(
                f"Warning: embedding dimension {embeddings.shape[1]} does not match "
                f"expected {DIMENSION}."
            )

        save_embeddings(chunks, embeddings, output_path)
        print(f"Saved embeddings to {output_path}")


if __name__ == "__main__":
    main()
