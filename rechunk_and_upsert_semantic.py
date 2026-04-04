"""
Re-chunk existing baseline_semantic chunks to proper sizes (~1000-1200 chars),
embed with Jina v5 (768-dim), and upsert to Pinecone as strategy='semantic'.
"""
import json
import os
import sys
import numpy as np
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

CHUNKS_IN = os.path.join(BASE_DIR, "data", "processed", "chunks_baseline_semantic.json")
CHUNKS_OUT = os.path.join(BASE_DIR, "data", "processed", "chunks_semantic_rechunked.json")
EMB_OUT = os.path.join(BASE_DIR, "data", "processed", "embeddings_semantic_rechunked.npz")
INDEX_NAME = "ntsb-rag"
STRATEGY = "semantic"
BATCH_SIZE = 100

# --- Step 1: Re-chunk ---
print("=== Step 1: Re-chunking baseline_semantic chunks ===")
with open(CHUNKS_IN) as f:
    orig_chunks = json.load(f)
print(f"  Loaded {len(orig_chunks)} original chunks")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " "],
)

new_chunks = []
for orig in orig_chunks:
    text = orig.get("text", "")
    if len(text) <= 1400:
        # Already small enough, keep as-is
        new_chunks.append(orig)
        continue

    sub_texts = splitter.split_text(text)
    for j, sub_text in enumerate(sub_texts):
        new_chunk = {
            "chunk_id": f"{orig['chunk_id']}_{j:03d}",
            "section_title": orig.get("section_title", "Document"),
            "text": sub_text,
            "report_id": orig.get("report_id", ""),
            "ntsb_no": orig.get("ntsb_no", ""),
            "event_date": orig.get("event_date", ""),
            "make": orig.get("make", ""),
            "model": orig.get("model", ""),
        }
        new_chunks.append(new_chunk)

print(f"  Re-chunked to {len(new_chunks)} chunks")
lengths = [len(c["text"]) for c in new_chunks]
print(f"  Avg size: {sum(lengths)/len(lengths):.0f} chars, min: {min(lengths)}, max: {max(lengths)}")

# Save rechunked
with open(CHUNKS_OUT, "w", encoding="utf-8") as f:
    json.dump(new_chunks, f, indent=2)
print(f"  Saved to {CHUNKS_OUT}")

# --- Step 2: Embed with Jina v5 (768-dim) ---
print("\n=== Step 2: Embedding with Jina v5 ===")
from src.retrieval.query import load_model
model = load_model()

texts = [c["text"] for c in new_chunks]
all_embs = []
for i in tqdm(range(0, len(texts), 64), desc="Embedding"):
    batch = texts[i:i+64]
    emb = model.encode(texts=batch, task="retrieval")
    if not isinstance(emb, np.ndarray):
        emb = np.array(emb)
    all_embs.append(emb)

embeddings = np.concatenate(all_embs, axis=0).astype(np.float32)
print(f"  Embedding shape: {embeddings.shape}")

chunk_ids = np.array([c["chunk_id"] for c in new_chunks])
np.savez_compressed(EMB_OUT, chunk_ids=chunk_ids, embeddings=embeddings)
print(f"  Saved to {EMB_OUT}")

# --- Step 3: Upsert to Pinecone ---
print("\n=== Step 3: Upserting to Pinecone ===")
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)

vectors = []
for c, emb in zip(new_chunks, embeddings):
    entity_id = c.get("entity_id") or c.get("ntsb_no") or c.get("report_id", "")
    vectors.append({
        "id": c["chunk_id"],
        "values": emb.tolist(),
        "metadata": {
            "ntsb_no": c.get("ntsb_no", ""),
            "report_id": c.get("report_id", ""),
            "entity_id": entity_id,
            "event_date": c.get("event_date", ""),
            "state": c.get("state", ""),
            "make": c.get("make", ""),
            "model": c.get("model", ""),
            "phase_of_flight": c.get("phase_of_flight", ""),
            "weather": c.get("weather", ""),
            "section_title": c.get("section_title", ""),
            "source_filename": c.get("source_filename", ""),
            "context_summary": c.get("context_summary", ""),
            "strategy": STRATEGY,
        },
    })

print(f"  Upserting {len(vectors)} vectors...")
for i in tqdm(range(0, len(vectors), BATCH_SIZE), desc="Upserting"):
    batch = vectors[i:i+BATCH_SIZE]
    index.upsert(vectors=batch)

print("\n=== Final Index Stats ===")
stats = index.describe_index_stats()
print(stats)
print("Done!")
