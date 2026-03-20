"""
Generate Jina embeddings for all chunk files.
Outputs .npz files containing chunk_ids and embedding vectors.
"""
import json
import os
import numpy as np
import torch
from tqdm import tqdm

MODEL_NAME = "jinaai/jina-embeddings-v5-text-nano"
BATCH_SIZE = 4  # Reduced for better visibility on CPU

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHUNK_FILES = {
    'fixed': os.path.join(BASE_DIR, 'data', 'processed', 'chunks_fixed.json'),
    'recursive': os.path.join(BASE_DIR, 'data', 'processed', 'chunks_recursive.json'),
    'semantic': os.path.join(BASE_DIR, 'data', 'processed', 'chunks_semantic.json'),
}

OUTPUT_FILES = {
    'fixed': os.path.join(BASE_DIR, 'data', 'processed', 'embeddings_fixed.npz'),
    'recursive': os.path.join(BASE_DIR, 'data', 'processed', 'embeddings_recursive.npz'),
    'semantic': os.path.join(BASE_DIR, 'data', 'processed', 'embeddings_semantic.npz'),
}


def load_model():
    from transformers import AutoModel
    print(f"Loading model: {MODEL_NAME}")
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print("Model loaded.")
    return model


def sanity_test(model):
    """Run sanity checks before full embedding generation."""
    print("\n--- Sanity Test ---")

    query = "What caused the engine failure during takeoff?"
    related_passage = "The pilot reported a loss of engine power shortly after takeoff. Inspection revealed a fractured crankshaft."
    unrelated_passage = "The restaurant menu featured a variety of pasta dishes and seasonal salads."

    query_emb = model.encode(texts=[query], task="retrieval", prompt_name="query")
    assert query_emb.shape == (1, 768), f"Query shape mismatch: {query_emb.shape}"

    passage_emb = model.encode(texts=[related_passage], task="retrieval", prompt_name="document")
    assert passage_emb.shape == (1, 768), f"Passage shape mismatch: {passage_emb.shape}"

    unrelated_emb = model.encode(texts=[unrelated_passage], task="retrieval", prompt_name="document")
    assert unrelated_emb.shape == (1, 768), f"Unrelated shape mismatch: {unrelated_emb.shape}"

    from scipy.spatial.distance import cosine

    # Scipy's cosine function calculates *distance*, so we subtract it from 1 to get *similarity*
    sim_related = 1 - cosine(query_emb[0], passage_emb[0])
    sim_unrelated = 1 - cosine(query_emb[0], unrelated_emb[0])


    print(f"  Related similarity:   {sim_related:.4f}")
    print(f"  Unrelated similarity: {sim_unrelated:.4f}")
    assert sim_related > sim_unrelated, (
        f"Sanity test FAILED: related ({sim_related:.4f}) should be > unrelated ({sim_unrelated:.4f})"
    )
    print("Sanity test passed!\n")


def generate_embeddings(model, chunk_path, output_path):
    """Load chunks, batch-encode, and save as .npz."""
    with open(chunk_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    chunk_ids = [c['chunk_id'] for c in chunks]
    texts = [c['text'] for c in chunks]

    all_embeddings = []
    print(f"Starting embedding generation for {len(texts)} chunks...")
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=os.path.basename(chunk_path)):
        batch = texts[i:i + BATCH_SIZE]
        # Log every batch since semantic chunks are large
        emb = model.encode(texts=batch, task="retrieval", prompt_name="document")
        all_embeddings.append(emb)

    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    print(f"  -> {os.path.basename(output_path)}: {embeddings.shape}")

    np.savez_compressed(
        output_path,
        chunk_ids=np.array(chunk_ids, dtype=str),
        embeddings=embeddings,
    )


def main():
    model = load_model()
    
    # Speed up: Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")
    model.to(device)
    
    sanity_test(model)

    for strategy, chunk_path in CHUNK_FILES.items():
        output_path = OUTPUT_FILES[strategy]
        
        # Skip if file already exists
        if os.path.exists(output_path):
            print(f"Skipping {strategy} - {os.path.basename(output_path)} already exists.")
            continue
            
        print(f"\nGenerating embeddings for {strategy} chunks...")
        generate_embeddings(model, chunk_path, output_path)

    print("\nDone! All embeddings saved.")


if __name__ == "__main__":
    main()
