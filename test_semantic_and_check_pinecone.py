"""Quick sanity check: semantic strategy + Pinecone strategy name audit."""
import os, json, sys
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index("ntsb-rag")

# --- 1. Index stats ---
stats = index.describe_index_stats()
print("=== INDEX STATS ===")
print(stats)

# --- 2. Check what strategy values exist by sampling vectors ---
print("\n=== STRATEGY NAME CHECK ===")
# Query with a zero vector to sample from each known strategy
dim = stats.dimension
zero_vec = [0.0] * dim

for strat in ["semantic", "md_recursive", "section", "parent_child", "recursive", "fixed",
              "baseline_semantic", "baseline_recursive", "md_section"]:
    try:
        res = index.query(
            vector=zero_vec,
            top_k=2,
            include_metadata=True,
            filter={"strategy": {"$eq": strat}},
        )
        count = len(res.matches)
        if count > 0:
            sample_id = res.matches[0].id
            print(f"  strategy='{strat}' -> {count} sample hits, e.g. {sample_id}")
        else:
            print(f"  strategy='{strat}' -> NO matches")
    except Exception as e:
        print(f"  strategy='{strat}' -> ERROR: {e}")

# --- 3. Test semantic strategy locally ---
print("\n=== SEMANTIC STRATEGY LOCAL CHECK ===")
chunks_path = os.path.join(os.path.dirname(__file__), "data", "processed", "chunks_baseline_semantic.json")
emb_path = os.path.join(os.path.dirname(__file__), "data", "processed", "embeddings_baseline_semantic.npz")

if os.path.exists(chunks_path):
    with open(chunks_path) as f:
        chunks = json.load(f)
    print(f"  chunks_baseline_semantic.json: {len(chunks)} chunks")
    print(f"  Sample chunk_id: {chunks[0]['chunk_id']}")
    print(f"  Sample text[:150]: {chunks[0].get('text','')[:150]}")
else:
    print(f"  MISSING: {chunks_path}")

if os.path.exists(emb_path):
    import numpy as np
    data = np.load(emb_path)
    print(f"  embeddings_baseline_semantic.npz: {data['embeddings'].shape}")
else:
    print(f"  MISSING: {emb_path}")

# --- 4. Full retrieval test with semantic strategy ---
print("\n=== SEMANTIC RETRIEVAL TEST ===")
try:
    from src.retrieval.query import load_model, retrieve, init_pinecone as q_init
    model = load_model()
    idx = q_init()
    results = retrieve("What causes engine failure during takeoff?", "semantic", top_k=3, model=model, index=idx)
    print(f"  Got {len(results)} results")
    for i, m in enumerate(results):
        print(f"  [{i+1}] score={m.score:.4f} id={m.id} text={m.metadata.get('text','')[:100]}...")
except Exception as e:
    print(f"  RETRIEVAL ERROR: {e}")

print("\nDone.")
