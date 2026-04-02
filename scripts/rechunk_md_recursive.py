"""
Rechunk → Embed → Upsert pipeline for the md_recursive strategy.

Changes from previous run:
  - chunk_size: 512 → 2048 (4x)
  - chunk_overlap: 50 → 200
  - hard token cap: 512 tokens per chunk (via _rebalance_to_token_bounds)

Usage:
    python scripts/rechunk_md_recursive.py
    python scripts/rechunk_md_recursive.py --dry-run      # chunk + embed only, no upsert
    python scripts/rechunk_md_recursive.py --skip-embed   # re-use existing embeddings file
"""

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_prep.chunking import chunk_markdown_md_recursive

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = project_root / "dataset-pipeline" / "data" / "extracted" / "extracted"
PROCESSED_DIR = project_root / "data" / "processed"
CHUNKS_FILE = PROCESSED_DIR / "chunks_md_md_recursive.json"
EMBEDDINGS_FILE = PROCESSED_DIR / "embeddings_md_recursive.npz"

# ── Pinecone / model config ─────────────────────────────────────────────────────
INDEX_NAME = "ntsb-rag"
STRATEGY_LABEL = "md_recursive"
MODEL_NAME = "jinaai/jina-embeddings-v5-text-nano"
EMBED_DIM = 768
BATCH_SIZE_EMBED = 16   # sweet spot: avoids OOM fragmentation for 512-token Jina on 3090
BATCH_SIZE_UPSERT = 100


# ── Step 1: Chunking ───────────────────────────────────────────────────────────

def run_chunking() -> list[dict]:
    md_files = sorted(DATA_DIR.glob("*.md"))
    if not md_files:
        logger.error(f"No .md files found in {DATA_DIR}")
        sys.exit(1)

    logger.info(f"Found {len(md_files)} markdown files — chunking with md_recursive (2048/200, cap=512 tokens)...")
    all_chunks = []
    for md_file in tqdm(md_files, desc="Chunking"):
        try:
            chunks = chunk_markdown_md_recursive(str(md_file))
            all_chunks.extend(chunks)
        except Exception as e:
            logger.warning(f"Failed to chunk {md_file.name}: {e}")

    logger.info(f"Total chunks generated: {len(all_chunks)}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    logger.info(f"Chunks saved to {CHUNKS_FILE}")

    return all_chunks


# ── Step 2: Embedding ──────────────────────────────────────────────────────────

def _load_jina_model():
    import os
    import torch
    from transformers import AutoModel

    # Prevent CUDA memory fragmentation across sequential large batches
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    warnings.filterwarnings("ignore")
    logger.info(f"Loading embedding model: {MODEL_NAME}")
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    setattr(model, "_device", device)
    logger.info(f"Model on device: {device}")
    return model


def _embed_batch(model, texts: list[str]) -> np.ndarray:
    device = getattr(model, "_device", "cpu")
    try:
        emb = model.encode(texts=texts, task="retrieval", prompt_name="passage", device=device)
    except TypeError:
        try:
            emb = model.encode(texts=texts, task="retrieval", prompt_name="passage")
        except Exception:
            emb = model.encode(texts=texts, task="retrieval")
    except Exception:
        emb = model.encode(texts=texts, task="retrieval")

    # CUDA tensors must be moved to CPU before numpy conversion
    import torch
    if isinstance(emb, torch.Tensor):
        emb = emb.cpu().detach()
    return np.array(emb, dtype=np.float32)


def run_embedding(chunks: list[dict]) -> np.ndarray:
    model = _load_jina_model()
    texts = [c.get("text", "") for c in chunks]
    all_embeddings = []

    logger.info(f"Embedding {len(texts)} chunks in batches of {BATCH_SIZE_EMBED}...")
    import torch
    for i in tqdm(range(0, len(texts), BATCH_SIZE_EMBED), desc="Embedding"):
        batch = texts[i : i + BATCH_SIZE_EMBED]
        emb = _embed_batch(model, batch)
        all_embeddings.append(emb)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # prevent VRAM fragmentation accumulation

    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    logger.info(f"Embeddings shape: {embeddings.shape}")

    if embeddings.shape[1] != EMBED_DIM:
        logger.warning(f"Unexpected embedding dimension {embeddings.shape[1]} (expected {EMBED_DIM})")

    chunk_ids = np.array([c["chunk_id"] for c in chunks])
    np.savez_compressed(str(EMBEDDINGS_FILE), chunk_ids=chunk_ids, embeddings=embeddings)
    logger.info(f"Embeddings saved to {EMBEDDINGS_FILE}")

    return embeddings


# ── Step 3: Upsert to Pinecone ─────────────────────────────────────────────────

def run_upsert(chunks: list[dict], embeddings: np.ndarray):
    load_dotenv(project_root / ".env")
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        logger.error("PINECONE_API_KEY not found in .env — cannot upsert.")
        sys.exit(1)

    from pinecone import Pinecone
    pc = Pinecone(api_key=api_key)
    index = pc.Index(INDEX_NAME)
    logger.info(f"Connected to Pinecone index '{INDEX_NAME}'")

    # Build vector dicts
    vectors = []
    for chunk, emb in zip(chunks, embeddings):
        entity_id = chunk.get("ntsb_no") or chunk.get("report_id", "")
        vectors.append({
            "id": chunk["chunk_id"],
            "values": emb.tolist(),
            "metadata": {
                "ntsb_no": chunk.get("ntsb_no", ""),
                "report_id": chunk.get("report_id", ""),
                "entity_id": entity_id,
                "event_date": chunk.get("event_date", ""),
                "state": chunk.get("state", ""),
                "make": chunk.get("make", ""),
                "model": chunk.get("model", ""),
                "section_title": chunk.get("section_title", ""),
                "strategy": STRATEGY_LABEL,
            },
        })

    logger.info(f"Upserting {len(vectors)} vectors to Pinecone (strategy={STRATEGY_LABEL})...")
    for i in tqdm(range(0, len(vectors), BATCH_SIZE_UPSERT), desc="Upserting"):
        batch = vectors[i : i + BATCH_SIZE_UPSERT]
        index.upsert(vectors=batch)

    logger.info("Upsert complete.")
    stats = index.describe_index_stats()
    logger.info(f"Index stats: {stats}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Rechunk md_recursive (4x) → embed → upsert")
    parser.add_argument("--dry-run", action="store_true", help="Chunk + embed only, skip Pinecone upsert")
    parser.add_argument("--skip-embed", action="store_true", help="Load existing embeddings file instead of re-embedding")
    parser.add_argument("--skip-chunk", action="store_true", help="Load existing chunks file instead of re-chunking")
    args = parser.parse_args()

    # 1. Chunk
    if args.skip_chunk:
        logger.info(f"--skip-chunk: loading chunks from {CHUNKS_FILE}")
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        logger.info(f"Loaded {len(chunks)} chunks")
    else:
        chunks = run_chunking()

    # 2. Embed
    if args.skip_embed:
        logger.info(f"--skip-embed: loading embeddings from {EMBEDDINGS_FILE}")
        npz = np.load(str(EMBEDDINGS_FILE))
        embeddings = npz["embeddings"]
        logger.info(f"Loaded embeddings shape: {embeddings.shape}")
    else:
        embeddings = run_embedding(chunks)

    # 3. Upsert
    if args.dry_run:
        logger.info("--dry-run: skipping Pinecone upsert")
    else:
        run_upsert(chunks, embeddings)

    logger.info("Done.")


if __name__ == "__main__":
    main()
