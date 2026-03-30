#!/usr/bin/env python3
"""
Generate embeddings for section-aware chunks and upsert to Pinecone.
This script handles the new metadata-enriched section chunks.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.retrieval.query import load_model

INDEX_NAME = "ntsb-rag"
CHUNK_FILE = BASE_DIR / "data" / "processed" / "chunks_md_section.json"
EMBEDDING_FILE = BASE_DIR / "data" / "processed" / "embeddings_section.npz"
DIMENSION = 768
BATCH_SIZE = 100


def init_pinecone():
    """Initialize and return Pinecone index."""
    load_dotenv(BASE_DIR / ".env")
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        print("ERROR: PINECONE_API_KEY not found in .env")
        return None
    
    pc = Pinecone(api_key=api_key)
    
    try:
        existing = [idx.name for idx in pc.list_indexes()]
        if INDEX_NAME not in existing:
            print(f"Creating Pinecone index '{INDEX_NAME}'...")
            pc.create_index(
                name=INDEX_NAME,
                dimension=DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            print("✓ Index created")
        else:
            print(f"✓ Index '{INDEX_NAME}' exists")
        
        return pc.Index(INDEX_NAME)
    except Exception as e:
        print(f"ERROR connecting to Pinecone: {e}")
        return None


def load_chunks():
    """Load section chunks from JSON file."""
    if not CHUNK_FILE.exists():
        print(f"ERROR: Chunk file not found: {CHUNK_FILE}")
        return None
    
    with open(CHUNK_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    print(f"✓ Loaded {len(chunks)} section chunks")
    return chunks


def generate_embeddings(model, chunks):
    """Generate embeddings for chunks using Jina model (batch processing)."""
    print(f"\nGenerating embeddings for {len(chunks)} chunks (batch processing)...")
    
    # Extract text and IDs
    texts = [c.get("text", "")[:512] for c in chunks]  # Limit text length
    chunk_ids = [c["chunk_id"] for c in chunks]
    
    # Process in batches to avoid OOM
    embedding_batch_size = 500
    all_embeddings = []
    
    try:
        for i in range(0, len(texts), embedding_batch_size):
            batch_texts = texts[i:i + embedding_batch_size]
            batch_ids = chunk_ids[i:i + embedding_batch_size]
            
            print(f"  Processing batch {i//embedding_batch_size + 1}...", end=" ")
            
            # Encode batch
            batch_embeddings = model.encode(
                texts=batch_texts,
                task="retrieval",
                prompt_name="query"
            )
            all_embeddings.append(batch_embeddings)
            print(f"✓ ({len(batch_texts)} chunks)")
        
        # Combine all embeddings
        embeddings = np.vstack(all_embeddings)
        print(f"✓ Generated embeddings shape: {embeddings.shape}")
        
        # Save embeddings
        np.savez_compressed(
            EMBEDDING_FILE,
            embeddings=embeddings,
            chunk_ids=np.array(chunk_ids, dtype=object)
        )
        print(f"✓ Saved embeddings to {EMBEDDING_FILE}")
        
        return embeddings, chunk_ids
    except Exception as e:
        print(f"ERROR generating embeddings: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def build_vectors(chunks, embeddings, chunk_ids):
    """Build Pinecone vector dicts with metadata."""
    print(f"\nBuilding Pinecone vectors...")
    
    vectors = []
    for cid, emb in zip(chunk_ids, embeddings):
        # Find matching chunk
        chunk = next((c for c in chunks if c["chunk_id"] == cid), None)
        if not chunk:
            continue
        
        # Build metadata from chunk
        vectors.append({
            "id": cid,
            "values": emb.tolist(),
            "metadata": {
                "ntsb_no": chunk.get("ntsb_no", ""),
                "event_date": chunk.get("event_date", ""),
                "state": chunk.get("state", ""),
                "make": chunk.get("make", ""),
                "model": chunk.get("model", ""),
                "phase_of_flight": chunk.get("phase_of_flight", ""),
                "weather": chunk.get("weather", ""),
                "section_title": chunk.get("section_title", ""),
                "strategy": "section",  # Important: mark as section strategy
            },
        })
    
    print(f"✓ Built {len(vectors)} vectors with metadata")
    return vectors


def upsert_vectors(index, vectors):
    """Batch upsert vectors to Pinecone."""
    print(f"\nUpserting {len(vectors)} vectors to Pinecone...")
    
    try:
        for i in range(0, len(vectors), BATCH_SIZE):
            batch = vectors[i : i + BATCH_SIZE]
            index.upsert(vectors=batch)
            pct = min(i + BATCH_SIZE, len(vectors)) / len(vectors) * 100
            print(f"  [{pct:.1f}%] Upserted {min(i + BATCH_SIZE, len(vectors))}/{len(vectors)}")
        
        print(f"✓ All vectors upserted successfully!")
        return True
    except Exception as e:
        print(f"ERROR upserting vectors: {e}")
        return False


def main():
    print("=" * 80)
    print("SECTION-AWARE CHUNKS: EMBED & UPSERT TO PINECONE")
    print("=" * 80)
    print()
    
    # Initialize
    index = init_pinecone()
    if not index:
        return False
    
    chunks = load_chunks()
    if not chunks:
        return False
    
    # Load embedding model
    print("\nLoading embedding model...")
    model = load_model()
    if not model:
        print("ERROR: Failed to load model")
        return False
    
    # Generate embeddings
    embeddings, chunk_ids = generate_embeddings(model, chunks)
    if embeddings is None:
        return False
    
    # Build vectors
    vectors = build_vectors(chunks, embeddings, chunk_ids)
    if not vectors:
        return False
    
    # Upsert to Pinecone
    success = upsert_vectors(index, vectors)
    
    print()
    print("=" * 80)
    if success:
        print("✅ SECTION CHUNKS SUCCESSFULLY INDEXED IN PINECONE")
        print(f"   Total chunks: {len(vectors)}")
        print(f"   Strategy: section (markdown-aware with metadata)")
        print(f"   Metadata fields: ntsb_no, event_date, make, model, state")
        print()
        print("Next steps:")
        print("  • Test retrieval with crew/pilot hours query")
        print("  • Verify that crew section chunks are now retrievable")
        print("  • Compare answers before and after indexing")
    else:
        print("❌ FAILED TO UPSERT SECTION CHUNKS")
    print("=" * 80)
    print()
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
