"""Re-embed semantic chunks with Jina v5 (768d) and upsert to Pinecone."""
import json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

BASE_DIR = os.path.dirname(__file__)
CHUNKS_PATH = os.path.join(BASE_DIR, "data", "processed", "chunks_baseline_semantic.json")
EMB_OUT = os.path.join(BASE_DIR, "data", "processed", "embeddings_baseline_semantic_jina.npz")
STRATEGY = "semantic"
BATCH_SIZE = 100

# --- Load chunks ---
print("Loading chunks...")
with open(CHUNKS_PATH) as f:
    chunks = json.load(f)
print(f"  {len(chunks)} chunks loaded")

# --- Embed with Jina ---
print("Loading Jina model...")
from src.retrieval.query import load_model
model = load_model()

texts = []
for c in chunks:
    texts.append(c.get("contextualized_text") or c.get("text", ""))

print(f"Embedding {len(texts)} texts with Jina...")
all_embs = []
for i in range(0, len(texts), 64):
    batch = texts[i:i+64]
    try:
        emb = model.encode(texts=batch, task="retrieval", prompt_name="passage")
    except TypeError:
        emb = model.encode(texts=batch, task="retrieval")
    all_embs.append(emb)
    if (i // 64) % 10 == 0:
        print(f"  Embedded {min(i+64, len(texts))}/{len(texts)}")

embeddings = np.concatenate(all_embs, axis=0).astype(np.float32)
print(f"  Embedding shape: {embeddings.shape}")

# Save embeddings
chunk_ids = np.array([c["chunk_id"] for c in chunks])
np.savez_compressed(EMB_OUT, chunk_ids=chunk_ids, embeddings=embeddings)
print(f"  Saved to {EMB_OUT}")

# --- Upsert to Pinecone ---
print("Connecting to Pinecone...")
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index("ntsb-rag")

print("Building vectors...")
vectors = []
for c, emb in zip(chunks, embeddings):
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
            "role": c.get("role", "Unknown"),
            "source_filename": c.get("source_filename", ""),
            "context_summary": c.get("context_summary", ""),
            "section_title": c.get("section_title", ""),
            "strategy": STRATEGY,
        },
    })

print(f"Upserting {len(vectors)} vectors...")
for i in range(0, len(vectors), BATCH_SIZE):
    batch = vectors[i:i+BATCH_SIZE]
    index.upsert(vectors=batch)
    if (i // BATCH_SIZE) % 10 == 0:
        print(f"  Upserted {min(i+BATCH_SIZE, len(vectors))}/{len(vectors)}")

print("\n=== Final Index Stats ===")
print(index.describe_index_stats())
print("Done!")
